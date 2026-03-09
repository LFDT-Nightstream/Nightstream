use std::collections::{HashMap, HashSet};

use p3_goldilocks::Goldilocks as F;

use crate::cpu::bus_layout::{
    build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes, BusLayout, ShoutInstanceShape,
};
use crate::cpu::constraints::{
    CpuConstraint, CpuConstraintBuilder, ShoutCpuBinding, TwistCpuBinding, CPU_BUS_COL_DISABLED,
};
use crate::cpu::r1cs_adapter::SharedCpuBusConfig;
use crate::plain::PlainMemLayout;
use crate::riscv::lookups::{PROG_ID, RAM_ID, REG_ID};
use crate::riscv::trace::{
    rv32_decode_lookup_table_id_for_col, rv32_decode_lookup_val_slot_for_col, rv32_is_decode_lookup_table_id,
    rv32_is_width_lookup_table_id, rv32_trace_lookup_addr_group_for_table_id,
    rv32_trace_lookup_selector_group_for_table_id, rv64_is_width_lookup_table_id, Rv32DecodeSidecarLayout,
};

use super::super::constants::{
    ADD_TABLE_ID, AND_TABLE_ID, DIVU_TABLE_ID, DIV_TABLE_ID, EQ_TABLE_ID, MULHSU_TABLE_ID, MULHU_TABLE_ID,
    MULH_TABLE_ID, MUL_TABLE_ID, NEQ_TABLE_ID, OR_TABLE_ID, REMU_TABLE_ID, REM_TABLE_ID, RV32_XLEN, SLL_TABLE_ID,
    SLTU_TABLE_ID, SLT_TABLE_ID, SRA_TABLE_ID, SRL_TABLE_ID, SUB_TABLE_ID, XOR_TABLE_ID,
};
use super::super::trace::Rv32TraceCcsLayout;
use super::shared::{
    audit_bus_tail_constraint_coverage, derive_trace_shout_shapes as derive_trace_shout_shapes_shared,
    TraceShoutBusSpec, TraceShoutShape,
};

#[inline]
fn trace_cpu_col(layout: &Rv32TraceCcsLayout, trace_col: usize) -> usize {
    layout.cell(trace_col, 0)
}

#[inline]
fn trace_shout_binding(layout: &Rv32TraceCcsLayout, table_id: u32) -> Option<ShoutCpuBinding> {
    if rv32_is_decode_lookup_table_id(table_id) {
        Some(ShoutCpuBinding {
            has_lookup: CPU_BUS_COL_DISABLED,
            addr: Some(trace_cpu_col(layout, layout.trace.pc_before)),
            val: CPU_BUS_COL_DISABLED,
        })
    } else if rv32_is_width_lookup_table_id(table_id) || rv64_is_width_lookup_table_id(table_id) {
        Some(ShoutCpuBinding {
            has_lookup: CPU_BUS_COL_DISABLED,
            addr: Some(trace_cpu_col(layout, layout.trace.cycle)),
            val: CPU_BUS_COL_DISABLED,
        })
    } else {
        None
    }
}

#[inline]
fn validate_trace_shout_table_id(table_id: u32) -> Result<(), String> {
    match table_id {
        AND_TABLE_ID | XOR_TABLE_ID | OR_TABLE_ID | ADD_TABLE_ID | SUB_TABLE_ID | SLT_TABLE_ID | SLTU_TABLE_ID
        | SLL_TABLE_ID | SRL_TABLE_ID | SRA_TABLE_ID | EQ_TABLE_ID | NEQ_TABLE_ID | MUL_TABLE_ID | MULH_TABLE_ID
        | MULHU_TABLE_ID | MULHSU_TABLE_ID | DIV_TABLE_ID | DIVU_TABLE_ID | REM_TABLE_ID | REMU_TABLE_ID => Ok(()),
        _ => Err(format!("RV32 trace shared bus: unsupported shout table_id={table_id}")),
    }
}

#[inline]
fn trace_lookup_addr_group_for_table_id(table_id: u32, ell_addr: usize) -> Option<u32> {
    if table_id <= REMU_TABLE_ID && ell_addr != 2 * RV32_XLEN {
        None
    } else {
        rv32_trace_lookup_addr_group_for_table_id(table_id)
    }
}

#[inline]
fn trace_lookup_selector_group_for_table_id(table_id: u32) -> Option<u32> {
    rv32_trace_lookup_selector_group_for_table_id(table_id)
}

fn derive_trace_shout_shapes(
    shout_table_ids: &[u32],
    extra_shout_specs: &[TraceShoutBusSpec],
) -> Result<Vec<TraceShoutShape>, String> {
    derive_trace_shout_shapes_shared(
        shout_table_ids,
        extra_shout_specs,
        |table_id, extra_specs| {
            if let Err(err) = validate_trace_shout_table_id(table_id) {
                if extra_specs.iter().any(|spec| spec.table_id == table_id) {
                    Ok(None)
                } else {
                    Err(err)
                }
            } else {
                Ok(Some((2 * RV32_XLEN, 1usize)))
            }
        },
        trace_lookup_addr_group_for_table_id,
        trace_lookup_selector_group_for_table_id,
        "RV32 trace shared bus",
    )
}

#[inline]
fn trace_disabled_twist_binding(_layout: &Rv32TraceCcsLayout) -> TwistCpuBinding {
    TwistCpuBinding {
        has_read: CPU_BUS_COL_DISABLED,
        has_write: CPU_BUS_COL_DISABLED,
        read_addr: CPU_BUS_COL_DISABLED,
        write_addr: CPU_BUS_COL_DISABLED,
        rv: CPU_BUS_COL_DISABLED,
        wv: CPU_BUS_COL_DISABLED,
        inc: None,
    }
}

#[derive(Clone, Copy, Debug)]
struct TraceDecodeSelectorCols {
    ram_has_read: usize,
    ram_has_write: usize,
}

fn resolve_trace_decode_selector_cols(
    layout: &Rv32TraceCcsLayout,
    shout_shapes: &[TraceShoutShape],
    mem_layouts: &HashMap<u32, PlainMemLayout>,
) -> Result<TraceDecodeSelectorCols, String> {
    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();
    let mut twist_shapes = Vec::with_capacity(mem_ids.len());
    for mem_id in &mem_ids {
        let mem_layout = mem_layouts
            .get(mem_id)
            .ok_or_else(|| format!("RV32 trace shared bus: missing mem layout for mem_id={mem_id}"))?;
        if mem_layout.n_side == 0 || !mem_layout.n_side.is_power_of_two() {
            return Err(format!(
                "RV32 trace shared bus: mem_id={mem_id} n_side={} must be power-of-two",
                mem_layout.n_side
            ));
        }
        let ell = mem_layout.n_side.trailing_zeros() as usize;
        let ell_addr = mem_layout.d * ell;
        twist_shapes.push((ell_addr, mem_layout.lanes.max(1)));
    }

    let bus = build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
        layout.m,
        layout.m_in,
        layout.t,
        shout_shapes.iter().map(|shape| ShoutInstanceShape {
            ell_addr: shape.ell_addr,
            lanes: 1usize,
            n_vals: shape.n_vals.max(1),
            addr_group: shape.addr_group.map(|v| v as u64),
            selector_group: shape.selector_group.map(|v| v as u64),
        }),
        twist_shapes.iter().copied(),
    )?;

    trace_decode_selector_cols_from_bus(&bus, shout_shapes)
}

fn trace_decode_selector_cols_from_bus(
    bus: &crate::cpu::bus_layout::BusLayout,
    shout_shapes: &[TraceShoutShape],
) -> Result<TraceDecodeSelectorCols, String> {
    let decode_layout = Rv32DecodeSidecarLayout::new();
    let ram_has_read_table_id = rv32_decode_lookup_table_id_for_col(decode_layout.ram_has_read);
    let ram_has_write_table_id = rv32_decode_lookup_table_id_for_col(decode_layout.ram_has_write);
    let ram_has_read_slot = rv32_decode_lookup_val_slot_for_col(decode_layout.ram_has_read)
        .ok_or_else(|| "RV32 trace shared bus: missing decode value slot for ram_has_read".to_string())?;
    let ram_has_write_slot = rv32_decode_lookup_val_slot_for_col(decode_layout.ram_has_write)
        .ok_or_else(|| "RV32 trace shared bus: missing decode value slot for ram_has_write".to_string())?;
    let table_val_col = |table_id: u32, val_slot: usize| -> Result<usize, String> {
        let shout_idx = shout_shapes
            .iter()
            .position(|shape| shape.table_id == table_id)
            .ok_or_else(|| {
                format!(
                    "RV32 trace shared bus: missing decode lookup table_id={table_id} required for Twist selector binding"
                )
            })?;
        let inst_cols = bus.shout_cols.get(shout_idx).ok_or_else(|| {
            format!("RV32 trace shared bus: missing shout cols for decode lookup table_id={table_id}")
        })?;
        let lane0 = inst_cols.lanes.first().ok_or_else(|| {
            format!("RV32 trace shared bus: expected one shout lane for decode lookup table_id={table_id}")
        })?;
        let val_col = lane0.vals.get(val_slot).copied().ok_or_else(|| {
            format!(
                "RV32 trace shared bus: decode val_slot={val_slot} out of range for table_id={table_id} (n_vals={})",
                lane0.vals.len()
            )
        })?;
        bus.bus_base
            .checked_add(val_col * bus.chunk_size)
            .ok_or_else(|| "RV32 trace shared bus: decode selector column overflow".to_string())
    };
    Ok(TraceDecodeSelectorCols {
        ram_has_read: table_val_col(ram_has_read_table_id, ram_has_read_slot)?,
        ram_has_write: table_val_col(ram_has_write_table_id, ram_has_write_slot)?,
    })
}

#[inline]
fn trace_twist_primary_binding(
    layout: &Rv32TraceCcsLayout,
    mem_id: u32,
    decode_selectors: TraceDecodeSelectorCols,
) -> TwistCpuBinding {
    let active = trace_cpu_col(layout, layout.trace.active);
    if mem_id == RAM_ID.0 {
        TwistCpuBinding {
            has_read: decode_selectors.ram_has_read,
            has_write: decode_selectors.ram_has_write,
            read_addr: trace_cpu_col(layout, layout.trace.ram_addr),
            write_addr: trace_cpu_col(layout, layout.trace.ram_addr),
            rv: trace_cpu_col(layout, layout.trace.ram_rv),
            wv: trace_cpu_col(layout, layout.trace.ram_wv),
            inc: None,
        }
    } else if mem_id == PROG_ID.0 {
        TwistCpuBinding {
            has_read: active,
            has_write: CPU_BUS_COL_DISABLED,
            read_addr: trace_cpu_col(layout, layout.trace.pc_before),
            write_addr: CPU_BUS_COL_DISABLED,
            rv: trace_cpu_col(layout, layout.trace.instr_word),
            wv: CPU_BUS_COL_DISABLED,
            inc: None,
        }
    } else if mem_id == REG_ID.0 {
        TwistCpuBinding {
            has_read: active,
            has_write: trace_cpu_col(layout, layout.trace.rd_has_write),
            read_addr: trace_cpu_col(layout, layout.trace.rs1_addr),
            write_addr: trace_cpu_col(layout, layout.trace.rd_addr),
            rv: trace_cpu_col(layout, layout.trace.rs1_val),
            wv: trace_cpu_col(layout, layout.trace.rd_val),
            inc: None,
        }
    } else {
        trace_disabled_twist_binding(layout)
    }
}

pub fn rv32_trace_shared_cpu_bus_config(
    layout: &Rv32TraceCcsLayout,
    shout_table_ids: &[u32],
    mem_layouts: HashMap<u32, PlainMemLayout>,
    initial_mem: HashMap<(u32, u64), F>,
) -> Result<SharedCpuBusConfig<F>, String> {
    rv32_trace_shared_cpu_bus_config_with_specs(layout, shout_table_ids, &[], mem_layouts, initial_mem)
}

pub fn rv32_trace_shared_cpu_bus_config_with_specs(
    layout: &Rv32TraceCcsLayout,
    shout_table_ids: &[u32],
    extra_shout_specs: &[TraceShoutBusSpec],
    mem_layouts: HashMap<u32, PlainMemLayout>,
    initial_mem: HashMap<(u32, u64), F>,
) -> Result<SharedCpuBusConfig<F>, String> {
    let shout_shapes = derive_trace_shout_shapes(shout_table_ids, extra_shout_specs)?;
    let decode_selectors = resolve_trace_decode_selector_cols(layout, &shout_shapes, &mem_layouts)?;

    let mut shout_cpu = HashMap::new();
    for shape in &shout_shapes {
        let binding = trace_shout_binding(layout, shape.table_id);
        shout_cpu.insert(shape.table_id, binding.into_iter().collect());
    }

    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();
    let mut twist_cpu = HashMap::new();
    for mem_id in mem_ids {
        let lanes = mem_layouts
            .get(&mem_id)
            .map(|l| l.lanes.max(1))
            .ok_or_else(|| format!("RV32 trace shared bus: missing mem layout for mem_id={mem_id}"))?;

        if mem_id == REG_ID.0 {
            if lanes < 2 {
                return Err(format!(
                    "RV32 trace shared bus: REG_ID requires lanes>=2 (got lanes={lanes})"
                ));
            }
            let mut bindings = Vec::with_capacity(lanes);
            bindings.push(trace_twist_primary_binding(layout, mem_id, decode_selectors));
            bindings.push(TwistCpuBinding {
                has_read: trace_cpu_col(layout, layout.trace.active),
                has_write: CPU_BUS_COL_DISABLED,
                read_addr: trace_cpu_col(layout, layout.trace.rs2_addr),
                write_addr: CPU_BUS_COL_DISABLED,
                rv: trace_cpu_col(layout, layout.trace.rs2_val),
                wv: CPU_BUS_COL_DISABLED,
                inc: None,
            });
            let disabled = trace_disabled_twist_binding(layout);
            for _ in 2..lanes {
                bindings.push(disabled.clone());
            }
            twist_cpu.insert(mem_id, bindings);
        } else {
            let primary = trace_twist_primary_binding(layout, mem_id, decode_selectors);
            let disabled = trace_disabled_twist_binding(layout);
            let mut bindings = Vec::with_capacity(lanes);
            bindings.push(primary);
            for _ in 1..lanes {
                bindings.push(disabled.clone());
            }
            twist_cpu.insert(mem_id, bindings);
        }
    }

    let mut shout_addr_groups = HashMap::new();
    let mut shout_selector_groups = HashMap::new();
    for shape in &shout_shapes {
        if let Some(g) = shape.addr_group {
            shout_addr_groups.insert(shape.table_id, g as u64);
        }
        if let Some(g) = shape.selector_group {
            shout_selector_groups.insert(shape.table_id, g as u64);
        }
    }

    Ok(SharedCpuBusConfig {
        mem_layouts,
        initial_mem,
        const_one_col: layout.const_one,
        shout_cpu,
        twist_cpu,
        shout_addr_groups,
        shout_selector_groups,
    })
}

pub fn rv32_trace_shared_bus_requirements(
    layout: &Rv32TraceCcsLayout,
    shout_table_ids: &[u32],
    mem_layouts: &HashMap<u32, PlainMemLayout>,
) -> Result<(usize, usize), String> {
    rv32_trace_shared_bus_requirements_with_specs(layout, shout_table_ids, &[], mem_layouts)
}

pub fn rv32_trace_shared_bus_requirements_with_specs(
    layout: &Rv32TraceCcsLayout,
    shout_table_ids: &[u32],
    extra_shout_specs: &[TraceShoutBusSpec],
    mem_layouts: &HashMap<u32, PlainMemLayout>,
) -> Result<(usize, usize), String> {
    let snapshot =
        rv32_trace_shared_bus_extraction_with_specs(layout, shout_table_ids, extra_shout_specs, mem_layouts)?;
    Ok((snapshot.bus.bus_region_len(), snapshot.constraints.len()))
}

#[derive(Clone, Debug)]
pub struct TraceSharedBusExtraction {
    pub bus: BusLayout,
    pub constraints: Vec<CpuConstraint<F>>,
}

pub fn rv32_trace_shared_bus_extraction(
    layout: &Rv32TraceCcsLayout,
    shout_table_ids: &[u32],
    mem_layouts: &HashMap<u32, PlainMemLayout>,
) -> Result<TraceSharedBusExtraction, String> {
    rv32_trace_shared_bus_extraction_with_specs(layout, shout_table_ids, &[], mem_layouts)
}

pub fn rv32_trace_shared_bus_extraction_with_specs(
    layout: &Rv32TraceCcsLayout,
    shout_table_ids: &[u32],
    extra_shout_specs: &[TraceShoutBusSpec],
    mem_layouts: &HashMap<u32, PlainMemLayout>,
) -> Result<TraceSharedBusExtraction, String> {
    let shout_shapes = derive_trace_shout_shapes(shout_table_ids, extra_shout_specs)?;

    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();

    let mut shout_cols = 0usize;
    let mut seen_addr_groups = HashMap::<u32, usize>::new();
    let mut seen_selector_groups = HashSet::<u32>::new();
    for shape in &shout_shapes {
        if let Some(group) = shape.addr_group {
            if let Some(prev_ell) = seen_addr_groups.insert(group, shape.ell_addr) {
                if prev_ell != shape.ell_addr {
                    return Err(format!(
                        "RV32 trace shared bus: addr_group={} has conflicting ell_addr ({} vs {})",
                        group, prev_ell, shape.ell_addr
                    ));
                }
            } else {
                shout_cols = shout_cols
                    .checked_add(shape.ell_addr)
                    .ok_or_else(|| "RV32 trace shared bus: shout shared-addr width overflow".to_string())?;
            }
        } else {
            shout_cols = shout_cols
                .checked_add(shape.ell_addr)
                .ok_or_else(|| "RV32 trace shared bus: shout lane width overflow".to_string())?;
        }
        if let Some(selector_group) = shape.selector_group {
            if seen_selector_groups.insert(selector_group) {
                shout_cols = shout_cols
                    .checked_add(1)
                    .ok_or_else(|| "RV32 trace shared bus: shout selector width overflow".to_string())?;
            }
        } else {
            shout_cols = shout_cols
                .checked_add(1)
                .ok_or_else(|| "RV32 trace shared bus: shout selector width overflow".to_string())?;
        }
        shout_cols = shout_cols
            .checked_add(shape.n_vals)
            .ok_or_else(|| "RV32 trace shared bus: shout value width overflow".to_string())?;
    }

    let mut twist_cols = 0usize;
    let mut twist_shapes = Vec::with_capacity(mem_ids.len());
    for mem_id in &mem_ids {
        let mem_layout = mem_layouts
            .get(mem_id)
            .ok_or_else(|| format!("RV32 trace shared bus: missing mem layout for mem_id={mem_id}"))?;
        if mem_layout.n_side == 0 || !mem_layout.n_side.is_power_of_two() {
            return Err(format!(
                "RV32 trace shared bus: mem_id={mem_id} n_side={} must be power-of-two",
                mem_layout.n_side
            ));
        }
        let ell = mem_layout.n_side.trailing_zeros() as usize;
        let ell_addr = mem_layout.d * ell;
        let lanes = mem_layout.lanes.max(1);
        if *mem_id == REG_ID.0 && lanes < 2 {
            return Err(format!(
                "RV32 trace shared bus: REG_ID requires lanes>=2 (got lanes={lanes})"
            ));
        }
        twist_cols = twist_cols
            .checked_add((2 * ell_addr + 5) * lanes)
            .ok_or_else(|| "RV32 trace shared bus: twist bus column overflow".to_string())?;
        twist_shapes.push((ell_addr, lanes));
    }

    let bus_cols = shout_cols
        .checked_add(twist_cols)
        .ok_or_else(|| "RV32 trace shared bus: bus column overflow".to_string())?;
    let bus_region_len = bus_cols
        .checked_mul(layout.t)
        .ok_or_else(|| "RV32 trace shared bus: bus region overflow".to_string())?;
    let m_total = layout
        .m
        .checked_add(bus_region_len)
        .ok_or_else(|| "RV32 trace shared bus: total m overflow".to_string())?;

    let bus = build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
        m_total,
        layout.m_in,
        layout.t,
        shout_shapes.iter().map(|shape| ShoutInstanceShape {
            ell_addr: shape.ell_addr,
            lanes: 1usize,
            n_vals: shape.n_vals.max(1),
            addr_group: shape.addr_group.map(|v| v as u64),
            selector_group: shape.selector_group.map(|v| v as u64),
        }),
        twist_shapes.iter().copied(),
    )?;
    let decode_selectors = trace_decode_selector_cols_from_bus(&bus, &shout_shapes)?;

    let mut builder = CpuConstraintBuilder::<F>::new(m_total, m_total, layout.const_one);

    let mut addr_range_counts = HashMap::<(usize, usize), usize>::new();
    for inst_cols in &bus.shout_cols {
        for lane_cols in &inst_cols.lanes {
            let key = (lane_cols.addr_bits.start, lane_cols.addr_bits.end);
            *addr_range_counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut addr_range_bitness_added = HashSet::<(usize, usize)>::new();
    let mut selector_bitness_added = HashSet::<usize>::new();
    let mut shout_key_binding_added = HashSet::<(bool, usize, usize, usize, usize)>::new();
    for (i, shape) in shout_shapes.iter().enumerate() {
        let lane0 = &bus.shout_cols[i].lanes[0];
        let is_packed_opcode_lane = shape.table_id <= REMU_TABLE_ID && shape.ell_addr != 2 * RV32_XLEN;
        if let Some(binding) = trace_shout_binding(layout, shape.table_id) {
            let mut dedup_binding = binding.clone();
            if let Some(addr_base) = dedup_binding.addr {
                let (is_bus_gate, gate_base) = if dedup_binding.has_lookup == CPU_BUS_COL_DISABLED {
                    (true, lane0.has_lookup)
                } else {
                    (false, dedup_binding.has_lookup)
                };
                let key_sig = (
                    is_bus_gate,
                    gate_base,
                    addr_base,
                    lane0.addr_bits.start,
                    lane0.addr_bits.end,
                );
                if !shout_key_binding_added.insert(key_sig) {
                    dedup_binding.addr = None;
                }
            }
            builder.add_shout_instance_linkage_bound(&bus, lane0, &dedup_binding);
        }

        let key = (lane0.addr_bits.start, lane0.addr_bits.end);
        let shared_addr_group = addr_range_counts.get(&key).copied().unwrap_or(0) > 1;
        let selector_first = selector_bitness_added.insert(lane0.has_lookup);
        if is_packed_opcode_lane {
            if selector_first {
                builder.add_shout_instance_padding_value_only(&bus, lane0);
            } else {
                builder.add_shout_instance_value_padding_only(&bus, lane0);
            }
            builder.add_shout_instance_addr_padding_only(&bus, lane0);
        } else if shared_addr_group {
            if selector_first {
                builder.add_shout_instance_padding_value_only(&bus, lane0);
            } else {
                builder.add_shout_instance_value_padding_only(&bus, lane0);
            }
            if addr_range_bitness_added.insert(key) {
                builder.add_shout_instance_addr_bit_bitness(&bus, lane0);
            }
        } else if selector_first {
            builder.add_shout_instance_padding(&bus, lane0);
        } else {
            builder.add_shout_instance_padding_without_selector_bitness(&bus, lane0);
        }
    }

    for (i, &mem_id) in mem_ids.iter().enumerate() {
        let inst = &bus.twist_cols[i];
        if inst.lanes.is_empty() {
            continue;
        }

        if mem_id == REG_ID.0 {
            let is_virtual_col_base = trace_cpu_col(layout, layout.trace.is_virtual);
            let lane0 = trace_twist_primary_binding(layout, mem_id, decode_selectors);
            builder.add_twist_instance_bound(&bus, &inst.lanes[0], &lane0);
            builder.add_twist_read_addr_domain_split_nonvirtual(&bus, &inst.lanes[0], is_virtual_col_base, 5);
            builder.add_twist_write_addr_domain_split(&bus, &inst.lanes[0], is_virtual_col_base, 5);
            let lane1 = TwistCpuBinding {
                has_read: trace_cpu_col(layout, layout.trace.active),
                has_write: CPU_BUS_COL_DISABLED,
                read_addr: trace_cpu_col(layout, layout.trace.rs2_addr),
                write_addr: CPU_BUS_COL_DISABLED,
                rv: trace_cpu_col(layout, layout.trace.rs2_val),
                wv: CPU_BUS_COL_DISABLED,
                inc: None,
            };
            if inst.lanes.len() >= 2 {
                builder.add_twist_instance_bound(&bus, &inst.lanes[1], &lane1);
                builder.add_twist_read_addr_domain_split_nonvirtual(&bus, &inst.lanes[1], is_virtual_col_base, 5);
            }
            if inst.lanes.len() > 2 {
                let disabled = trace_disabled_twist_binding(layout);
                for lane_cols in &inst.lanes[2..] {
                    builder.add_twist_instance_bound(&bus, lane_cols, &disabled);
                }
            }
        } else {
            let lane0 = trace_twist_primary_binding(layout, mem_id, decode_selectors);
            builder.add_twist_instance_bound(&bus, &inst.lanes[0], &lane0);
            if inst.lanes.len() > 1 {
                let disabled = trace_disabled_twist_binding(layout);
                for lane_cols in &inst.lanes[1..] {
                    builder.add_twist_instance_bound(&bus, lane_cols, &disabled);
                }
            }
        }
    }

    audit_bus_tail_constraint_coverage(&builder, &bus, "RV32 trace shared bus")?;

    Ok(TraceSharedBusExtraction {
        bus,
        constraints: builder.constraints().to_vec(),
    })
}
