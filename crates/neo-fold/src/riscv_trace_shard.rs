//! Legacy convenience runner for RV32 trace-wiring CCS (time-in-rows).
//!
//! This is an ergonomic wrapper around the existing trace wiring artifacts:
//! - `neo_memory::riscv::trace` for execution-table extraction, and
//! - `neo_memory::riscv::ccs::trace` for fixed-width trace wiring CCS.
//!
//! It remains useful for RV32-specific internal regression/reference coverage,
//! but it is not a supported product-facing path. The maintained note repros go
//! through `crate::rv64_trace_shard::Rv64TraceWiring::from_elf` and target the
//! RV64IM-only product contract.
//!
//! The runner intentionally targets the retained Tier 2.1 scope:
//! - fixed-width trace-wiring CCS steps with PROG/REG/RAM sidecar instances,
//! - no decode/semantics sidecar proofs in this wrapper yet.

#![allow(non_snake_case)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use crate::output_binding::OutputBindingConfig;
use crate::pi_ccs::FoldingMode;
use crate::session::FoldingSession;
use crate::shard::{ShardProof, StepLinkingConfig};
use crate::PiCcsError;
use neo_ajtai::AjtaiSModule;
use neo_ccs::CcsStructure;
use neo_math::{D, F, K};
use neo_memory::cpu::bus_layout::{
    build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes, ShoutInstanceShape,
};
use neo_memory::output_check::ProgramIO;
use neo_memory::plain::{LutTable, PlainMemLayout};
use neo_memory::riscv::ccs::{
    build_rv32_trace_wiring_ccs, rv32_trace_ccs_witness_from_exec_table, Rv32TraceCcsLayout, TraceShoutBusSpec,
};
use neo_memory::riscv::exec_table::{Rv32ExecRow, Rv32ExecTable};
use neo_memory::riscv::lookups::{
    decode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID, REG_ID,
};
use neo_memory::riscv::packed::{rv32_packed_d, rv32_packed_rollout_opcode};
use neo_memory::riscv::rom_init::prog_rom_layout_and_init_words;
use neo_memory::riscv::trace::{
    rv32_decode_lookup_backed_row_from_instr_word, rv32_decode_lookup_table_id_for_col,
    rv32_decode_lookup_transport_cols, rv32_trace_lookup_addr_group_for_table_id,
    rv32_trace_lookup_n_vals_for_table_id, rv32_trace_lookup_selector_group_for_table_id,
    rv32_width_lookup_backed_cols, rv32_width_lookup_table_id_for_col, rv32_width_sidecar_witness_from_exec_table,
    Rv32DecodeSidecarLayout, Rv32WidthSidecarLayout, RV32_TRACE_OPCODE_ADDR_GROUP,
    RV32_TRACE_OPCODE_COMBINED_ADDR_GROUP,
};
use neo_memory::{LutTableSpec, MemInit, R1csCpu};
use neo_params::NeoParams;
use neo_vm_trace::{ShoutEvent, ShoutId, StepTrace, Twist as _, TwistOpKind, VmTrace};
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField64;
use rand_chacha::rand_core::SeedableRng;

#[cfg(target_arch = "wasm32")]
use js_sys::Date;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
type TimePoint = f64;
#[cfg(not(target_arch = "wasm32"))]
type TimePoint = Instant;

#[inline]
fn time_now() -> TimePoint {
    #[cfg(target_arch = "wasm32")]
    {
        Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Instant::now()
    }
}

#[inline]
fn elapsed_duration(start: TimePoint) -> Duration {
    #[cfg(target_arch = "wasm32")]
    {
        let elapsed_ms = Date::now() - start;
        Duration::from_secs_f64(elapsed_ms / 1_000.0)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        start.elapsed()
    }
}

/// Hard instruction cap for trace-wiring mode (Option C).
const DEFAULT_RV32_TRACE_MAX_STEPS: usize = 1 << 20;
/// Conservative upper bound on emitted trace rows per architectural RV32 instruction
/// when runtime decomposition is enabled.
const RV32_RUNTIME_DECOMP_MAX_ROWS_PER_INSTR: usize = 32;
/// Hard VM-step cap after expansion through runtime decomposition.
const DEFAULT_RV32_TRACE_MAX_VM_STEPS: usize = DEFAULT_RV32_TRACE_MAX_STEPS * RV32_RUNTIME_DECOMP_MAX_ROWS_PER_INSTR;

/// Default per-step trace rows for trace-mode IVC.
///
/// The full trace is split into fixed-size chunks of this row count (except when the whole
/// trace is smaller), and those chunks are folded with step-linking.
const DEFAULT_RV32_TRACE_CHUNK_ROWS: usize = 1 << 16;

fn max_ram_addr_from_exec(exec: &Rv32ExecTable) -> Option<u64> {
    exec.rows
        .iter()
        .filter(|r| r.active)
        .flat_map(|r| r.ram_events.iter().map(|e| e.addr))
        .max()
}

fn max_consecutive_pc_run(exec: &Rv32ExecTable) -> usize {
    let mut best = 1usize;
    let mut cur = 0usize;
    let mut prev_pc: Option<u64> = None;
    for row in exec.rows.iter().filter(|r| r.active) {
        if prev_pc == Some(row.pc_before) {
            cur += 1;
        } else {
            cur = 1;
            prev_pc = Some(row.pc_before);
        }
        best = best.max(cur);
    }
    best
}

fn required_bits_for_max_addr(max_addr: u64) -> usize {
    if max_addr == 0 {
        1
    } else {
        (u64::BITS - max_addr.leading_zeros()) as usize
    }
}

fn final_reg_state_dense(exec: &Rv32ExecTable, reg_init: &HashMap<u64, u64>, k: usize) -> Result<Vec<F>, PiCcsError> {
    let mut regs = vec![0u64; k];
    for (&reg, &value) in reg_init {
        if reg >= 32 {
            return Err(PiCcsError::InvalidInput(format!(
                "reg_init_u32: register index out of range: reg={reg} (expected 0..32)"
            )));
        }
        if reg == 0 && value != 0 {
            return Err(PiCcsError::InvalidInput(
                "reg_init_u32: x0 must be 0 (non-zero init is forbidden)".into(),
            ));
        }
        regs[reg as usize] = value as u32 as u64;
    }
    regs[0] = 0;

    for r in exec.rows.iter().filter(|r| r.active) {
        if let Some(w) = &r.reg_write_lane0 {
            let addr_usize = usize::try_from(w.addr).map_err(|_| {
                PiCcsError::InvalidInput(format!(
                    "trace register write addr does not fit usize at cycle {}: addr={}",
                    r.cycle, w.addr
                ))
            })?;
            if addr_usize >= k {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace register write addr out of range at cycle {}: addr={}",
                    r.cycle, w.addr
                )));
            }
            if w.addr == 0 {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace writes x0 at cycle {} which is invalid",
                    r.cycle
                )));
            }
            regs[addr_usize] = w.value as u32 as u64;
            regs[0] = 0;
        }
    }

    Ok(regs.into_iter().map(F::from_u64).collect())
}

fn final_ram_state_dense(exec: &Rv32ExecTable, ram_init: &HashMap<u64, u64>, k: usize) -> Result<Vec<F>, PiCcsError> {
    let mut out = vec![F::ZERO; k];
    for (&addr, &value) in ram_init {
        let addr_usize = usize::try_from(addr)
            .map_err(|_| PiCcsError::InvalidInput(format!("ram_init_u32: addr does not fit usize: addr={addr}")))?;
        if addr_usize >= k {
            return Err(PiCcsError::InvalidInput(format!(
                "ram_init_u32: addr out of range for output binding domain: addr={addr} >= k={k}"
            )));
        }
        out[addr_usize] = F::from_u64(value as u32 as u64);
    }

    for r in exec.rows.iter().filter(|r| r.active) {
        for e in &r.ram_events {
            if e.kind != TwistOpKind::Write {
                continue;
            }
            let addr_usize = usize::try_from(e.addr).map_err(|_| {
                PiCcsError::InvalidInput(format!(
                    "trace RAM write addr does not fit usize at cycle {}: addr={}",
                    r.cycle, e.addr
                ))
            })?;
            if addr_usize >= k {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace RAM write addr out of range for output binding domain at cycle {}: addr={} >= k={k}",
                    r.cycle, e.addr
                )));
            }
            out[addr_usize] = F::from_u64(e.value as u32 as u64);
        }
    }

    Ok(out)
}

fn split_exec_into_fixed_chunks(exec: &Rv32ExecTable, chunk_rows: usize) -> Result<Vec<Rv32ExecTable>, PiCcsError> {
    if chunk_rows == 0 {
        return Err(PiCcsError::InvalidInput("trace chunk_rows must be non-zero".into()));
    }
    if exec.rows.is_empty() {
        return Err(PiCcsError::InvalidInput("trace execution table is empty".into()));
    }

    if exec.rows.len() <= chunk_rows {
        return Ok(vec![exec.clone()]);
    }

    let mut out = Vec::<Rv32ExecTable>::new();
    let total = exec.rows.len();
    let mut start = 0usize;
    while start < total {
        let end = (start + chunk_rows).min(total);
        let mut rows = exec.rows[start..end].to_vec();
        if rows.len() < chunk_rows {
            let last = rows
                .last()
                .ok_or_else(|| PiCcsError::InvalidInput("trace chunk unexpectedly empty".into()))?
                .clone();
            let mut cycle = last.cycle;
            let pad_pc = last.pc_after;
            let pad_halted = last.halted;
            while rows.len() < chunk_rows {
                cycle = cycle
                    .checked_add(1)
                    .ok_or_else(|| PiCcsError::InvalidInput("cycle overflow while chunk-padding trace".into()))?;
                rows.push(neo_memory::riscv::exec_table::Rv32ExecRow::inactive(
                    cycle, pad_pc, pad_halted,
                ));
            }
        }
        out.push(Rv32ExecTable { rows });
        start = end;
    }

    Ok(out)
}

fn boundary_splits_virtual_sequence(exec: &Rv32ExecTable, chunk_rows: usize) -> bool {
    if chunk_rows == 0 {
        return false;
    }
    let total = exec.rows.len();
    if total <= chunk_rows {
        return false;
    }

    let mut boundary = chunk_rows;
    while boundary < total {
        let prev = &exec.rows[boundary - 1];
        // Splitting right after a virtual row drops transition constraints
        // (virtual countdown/commit-link) across chunks.
        if prev.active && prev.is_virtual {
            return true;
        }
        boundary = match boundary.checked_add(chunk_rows) {
            Some(next) => next,
            None => break,
        };
    }
    false
}

fn rv32_trace_chunk_to_witness(
    layout: Rv32TraceCcsLayout,
) -> Box<dyn Fn(&[StepTrace<u64, u64, u128>]) -> Vec<F> + Send + Sync> {
    Box::new(move |chunk: &[StepTrace<u64, u64, u128>]| {
        rv32_trace_chunk_to_witness_checked(&layout, chunk)
            .unwrap_or_else(|e| panic!("rv32_trace_chunk_to_witness failed for chunk_len={}: {e}", chunk.len()))
    })
}

fn rv32_trace_chunk_to_witness_checked(
    layout: &Rv32TraceCcsLayout,
    chunk: &[StepTrace<u64, u64, u128>],
) -> Result<Vec<F>, String> {
    if chunk.is_empty() {
        return Err("trace chunk witness: chunk must contain at least one step".into());
    }
    if chunk.len() > layout.t {
        return Err(format!(
            "trace chunk witness: chunk.len()={} exceeds layout.t={}",
            chunk.len(),
            layout.t
        ));
    }

    let mut rows = Vec::with_capacity(layout.t);
    for step in chunk {
        rows.push(Rv32ExecRow::from_step(step)?);
    }

    let mut cycle = rows
        .last()
        .ok_or_else(|| "trace chunk witness: empty rows after conversion".to_string())?
        .cycle;
    let pad_pc = rows.last().expect("rows non-empty").pc_after;
    let pad_halted = rows.last().expect("rows non-empty").halted;
    while rows.len() < layout.t {
        cycle = cycle
            .checked_add(1)
            .ok_or_else(|| "trace chunk witness: cycle overflow while padding".to_string())?;
        rows.push(Rv32ExecRow::inactive(cycle, pad_pc, pad_halted));
    }

    let exec = Rv32ExecTable { rows };
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(layout, &exec)?;
    Ok(x.into_iter().chain(w).collect())
}

fn infer_required_trace_shout_opcodes(program: &[RiscvInstruction]) -> HashSet<RiscvOpcode> {
    let mut ops = HashSet::new();
    // Required for shared wiring (address/PC arithmetic).
    ops.insert(RiscvOpcode::Add);
    for instr in program {
        match instr {
            RiscvInstruction::RAlu { op, .. } => {
                ops.insert(*op);
            }
            RiscvInstruction::IAlu { op, .. } => {
                ops.insert(*op);
            }
            RiscvInstruction::Branch { cond, .. } => {
                ops.insert(cond.to_shout_opcode());
            }
            // Address arithmetic in these classes uses ADD shout semantics.
            RiscvInstruction::Load { .. }
            | RiscvInstruction::Store { .. }
            | RiscvInstruction::Jalr { .. }
            | RiscvInstruction::Auipc { .. } => {
                ops.insert(RiscvOpcode::Add);
            }
            _ => {}
        }
    }
    ops
}

fn rv32_canonical_shout_opcode_families() -> &'static [RiscvOpcode] {
    &[
        RiscvOpcode::And,
        RiscvOpcode::Xor,
        RiscvOpcode::Or,
        RiscvOpcode::Add,
        RiscvOpcode::Sub,
        RiscvOpcode::Slt,
        RiscvOpcode::Sltu,
        RiscvOpcode::Sll,
        RiscvOpcode::Srl,
        RiscvOpcode::Sra,
        RiscvOpcode::Eq,
        RiscvOpcode::Neq,
    ]
}

fn validate_trace_opcode_lookup_one_hot(trace: &VmTrace<u64, u64, u128>, xlen: usize) -> Result<(), PiCcsError> {
    let shout = RiscvShoutTables::new(xlen);
    for (step_idx, step) in trace.steps.iter().enumerate() {
        let mut seen_table_id: Option<u32> = None;
        for ev in step.shout_events.iter() {
            if shout.id_to_opcode(ev.shout_id).is_none() {
                continue;
            }
            if let Some(prev) = seen_table_id {
                return Err(PiCcsError::ProtocolError(format!(
                    "instruction-table lookup flags are not one-hot at step {step_idx}: multiple opcode table_ids ({prev}, {})",
                    ev.shout_id.0
                )));
            }
            seen_table_id = Some(ev.shout_id.0);
        }
    }
    Ok(())
}

fn program_requires_ram_sidecar(program: &[RiscvInstruction]) -> bool {
    program.iter().any(|instr| {
        matches!(
            instr,
            RiscvInstruction::Load { .. }
                | RiscvInstruction::Store { .. }
                | RiscvInstruction::LoadReserved { .. }
                | RiscvInstruction::StoreConditional { .. }
                | RiscvInstruction::Amo { .. }
        )
    })
}

fn program_requires_width_lookup(program: &[RiscvInstruction]) -> bool {
    program
        .iter()
        .any(|instr| matches!(instr, RiscvInstruction::Load { .. } | RiscvInstruction::Store { .. }))
}

#[inline]
fn trace_lookup_addr_group_for_table_shape(table_id: u32, ell_addr: usize) -> Option<u64> {
    let group = rv32_trace_lookup_addr_group_for_table_id(table_id);
    let is_opcode_group =
        group == Some(RV32_TRACE_OPCODE_ADDR_GROUP) || group == Some(RV32_TRACE_OPCODE_COMBINED_ADDR_GROUP);
    if is_opcode_group && ell_addr != 64 {
        // Packed opcode lanes use opcode-local widths and must not share the
        // canonical RV32 opcode (ell_addr=64) address group.
        None
    } else {
        group.map(|v| v as u64)
    }
}

fn table_ell_addr_for_shared_bus(
    table_id: u32,
    table_specs: &HashMap<u32, LutTableSpec>,
    lut_tables: &HashMap<u32, LutTable<F>>,
) -> Result<usize, PiCcsError> {
    let (d, n_side) = if let Some(spec) = table_specs.get(&table_id) {
        match spec {
            LutTableSpec::RiscvOpcode { xlen, .. } => (
                xlen.checked_mul(2)
                    .ok_or_else(|| PiCcsError::InvalidInput(format!("2*xlen overflow for table_id={table_id}")))?,
                2usize,
            ),
            LutTableSpec::RiscvOpcodePacked { opcode, xlen } => {
                if *xlen != 32 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "RiscvOpcodePacked requires xlen=32 in RV32 trace mode (table_id={table_id}, xlen={xlen})"
                    )));
                }
                (rv32_packed_d(*opcode)?, 2usize)
            }
            LutTableSpec::RiscvOpcodeEventTablePacked { .. } => {
                return Err(PiCcsError::InvalidInput(
                    "RiscvOpcodeEventTablePacked is not supported in RV32 trace shared-bus mode".into(),
                ));
            }
            LutTableSpec::IdentityU32 => (32usize, 2usize),
        }
    } else {
        let table = lut_tables
            .get(&table_id)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing LutTable metadata for table_id={table_id}")))?;
        (table.d, table.n_side)
    };
    if n_side == 0 || !n_side.is_power_of_two() {
        return Err(PiCcsError::InvalidInput(format!(
            "table_id={table_id} has non power-of-two n_side={n_side}"
        )));
    }
    let ell = n_side.trailing_zeros() as usize;
    d.checked_mul(ell)
        .ok_or_else(|| PiCcsError::InvalidInput("ell_addr overflow".into()))
}

fn estimate_route_a_bus_cols(
    chunk_size: usize,
    table_specs: &HashMap<u32, LutTableSpec>,
    lut_tables: &HashMap<u32, LutTable<F>>,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    lut_lanes: &HashMap<u32, usize>,
) -> Result<usize, PiCcsError> {
    if chunk_size == 0 {
        return Err(PiCcsError::InvalidInput(
            "route-a bus estimator requires chunk_size > 0".into(),
        ));
    }
    let mut table_ids: Vec<u32> = table_specs
        .keys()
        .copied()
        .chain(lut_tables.keys().copied())
        .collect();
    table_ids.sort_unstable();
    table_ids.dedup();

    let mut shout_shapes = Vec::<ShoutInstanceShape>::with_capacity(table_ids.len());
    let mut shout_upper_cols = 0usize;
    for table_id in table_ids {
        let ell_addr = table_ell_addr_for_shared_bus(table_id, table_specs, lut_tables)?;
        let lanes = lut_lanes.get(&table_id).copied().unwrap_or(1).max(1);
        let shout_lane_cols = ell_addr
            .checked_add(2)
            .ok_or_else(|| PiCcsError::InvalidInput("shout lane width overflow".into()))?;
        shout_upper_cols = shout_upper_cols
            .checked_add(
                shout_lane_cols
                    .checked_mul(lanes)
                    .ok_or_else(|| PiCcsError::InvalidInput("shout width overflow".into()))?,
            )
            .ok_or_else(|| PiCcsError::InvalidInput("bus width overflow".into()))?;
        shout_shapes.push(ShoutInstanceShape {
            ell_addr,
            lanes,
            n_vals: rv32_trace_lookup_n_vals_for_table_id(table_id).max(1),
            addr_group: trace_lookup_addr_group_for_table_shape(table_id, ell_addr),
            selector_group: rv32_trace_lookup_selector_group_for_table_id(table_id).map(|v| v as u64),
        });
    }

    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();
    let mut twist_shapes = Vec::<(usize, usize)>::with_capacity(mem_ids.len());
    let mut twist_upper_cols = 0usize;
    for mem_id in mem_ids {
        let layout = mem_layouts
            .get(&mem_id)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing mem layout for mem_id={mem_id}")))?;
        if layout.n_side == 0 || !layout.n_side.is_power_of_two() {
            return Err(PiCcsError::InvalidInput(format!(
                "mem_id={mem_id} has non power-of-two n_side={}",
                layout.n_side
            )));
        }
        let ell = layout.n_side.trailing_zeros() as usize;
        let ell_addr = layout
            .d
            .checked_mul(ell)
            .ok_or_else(|| PiCcsError::InvalidInput("twist ell_addr overflow".into()))?;
        let twist_lane_cols = ell_addr
            .checked_mul(2)
            .and_then(|v| v.checked_add(5))
            .ok_or_else(|| PiCcsError::InvalidInput("twist lane width overflow".into()))?;
        let lanes = layout.lanes.max(1);
        twist_upper_cols = twist_upper_cols
            .checked_add(
                twist_lane_cols
                    .checked_mul(lanes)
                    .ok_or_else(|| PiCcsError::InvalidInput("twist width overflow".into()))?,
            )
            .ok_or_else(|| PiCcsError::InvalidInput("bus width overflow".into()))?;
        twist_shapes.push((ell_addr, lanes));
    }

    let upper_bus_cols = shout_upper_cols
        .checked_add(twist_upper_cols)
        .ok_or_else(|| PiCcsError::InvalidInput("bus width overflow".into()))?;
    let m_probe = upper_bus_cols
        .checked_mul(chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("route-a bus estimator m_probe overflow".into()))?;
    let layout = build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
        m_probe,
        0usize,
        chunk_size,
        shout_shapes,
        twist_shapes,
    )
    .map_err(PiCcsError::InvalidInput)?;
    Ok(layout.bus_cols)
}

fn program_requires_poseidon_stage(program: &[RiscvInstruction]) -> bool {
    program.iter().any(|instr| {
        matches!(
            instr,
            RiscvInstruction::Poseidon2AbsorbElem { .. }
                | RiscvInstruction::Poseidon2Finalize
                | RiscvInstruction::Poseidon2SqueezeWord { .. }
        )
    })
}

fn rv32_trace_table_specs(shout_ops: &HashSet<RiscvOpcode>) -> Result<HashMap<u32, LutTableSpec>, PiCcsError> {
    let shout = RiscvShoutTables::new(32);
    let mut table_specs = HashMap::new();
    // Keep opcode lookup families fixed (Jolt-style) in canonical non-packed form.
    for &op in rv32_canonical_shout_opcode_families() {
        let table_id = shout.opcode_to_id(op).0;
        table_specs.insert(table_id, LutTableSpec::RiscvOpcode { opcode: op, xlen: 32 });
    }
    // Add any extra opcodes surfaced by trace/runtime decomposition that are not part of
    // the canonical fixed family set.
    for &op in shout_ops {
        let table_id = shout.opcode_to_id(op).0;
        if table_specs.contains_key(&table_id) {
            continue;
        }
        if rv32_packed_rollout_opcode(op)
            && !neo_memory::riscv::trace::rv32_trace_uses_combined_operand_key_table_id(table_id)
        {
            table_specs.insert(table_id, LutTableSpec::RiscvOpcodePacked { opcode: op, xlen: 32 });
        } else {
            table_specs.insert(table_id, LutTableSpec::RiscvOpcode { opcode: op, xlen: 32 });
        }
    }
    Ok(table_specs)
}

fn build_rv32_decode_lookup_tables(
    prog_layout: &PlainMemLayout,
    prog_init_words: &HashMap<(u32, u64), F>,
) -> HashMap<u32, LutTable<F>> {
    let decode_layout = Rv32DecodeSidecarLayout::new();
    let decode_cols = rv32_decode_lookup_transport_cols(&decode_layout);
    if decode_cols.is_empty() {
        return HashMap::new();
    }
    let mut unique_table_ids: Vec<u32> = decode_cols
        .iter()
        .map(|&col_id| rv32_decode_lookup_table_id_for_col(col_id))
        .collect();
    unique_table_ids.sort_unstable();
    unique_table_ids.dedup();
    let mut out = HashMap::new();
    if unique_table_ids.len() == 1 {
        let table_id = unique_table_ids[0];
        let n_vals = decode_cols.len().max(1);
        let content_len = prog_layout
            .k
            .checked_mul(n_vals)
            .expect("decode lookup content length overflow");
        let mut content = vec![F::ZERO; content_len];
        for addr in 0..prog_layout.k {
            let instr_word = prog_init_words
                .get(&(PROG_ID.0, addr as u64))
                .copied()
                .unwrap_or(F::ZERO)
                .as_canonical_u64() as u32;
            let row = rv32_decode_lookup_backed_row_from_instr_word(&decode_layout, instr_word, /*active=*/ true);
            let base = addr * n_vals;
            for (val_slot, &col_id) in decode_cols.iter().enumerate() {
                content[base + val_slot] = row[col_id];
            }
        }
        out.insert(
            table_id,
            LutTable {
                table_id,
                k: prog_layout.k,
                d: prog_layout.d,
                n_side: prog_layout.n_side,
                content,
            },
        );
        return out;
    }
    for &col_id in decode_cols.iter() {
        let table_id = rv32_decode_lookup_table_id_for_col(col_id);
        let mut content = vec![F::ZERO; prog_layout.k];
        for addr in 0..prog_layout.k {
            let instr_word = prog_init_words
                .get(&(PROG_ID.0, addr as u64))
                .copied()
                .unwrap_or(F::ZERO)
                .as_canonical_u64() as u32;
            let row = rv32_decode_lookup_backed_row_from_instr_word(&decode_layout, instr_word, /*active=*/ true);
            content[addr] = row[col_id];
        }
        out.insert(
            table_id,
            LutTable {
                table_id,
                k: prog_layout.k,
                d: prog_layout.d,
                n_side: prog_layout.n_side,
                content,
            },
        );
    }
    out
}

fn inject_rv32_decode_lookup_events_into_trace(
    trace: &mut VmTrace<u64, u64, u128>,
    prog_layout: &PlainMemLayout,
    prog_init_words: &HashMap<(u32, u64), F>,
) -> Result<(), PiCcsError> {
    let decode_layout = Rv32DecodeSidecarLayout::new();
    let decode_cols = rv32_decode_lookup_transport_cols(&decode_layout);
    for (step_idx, step) in trace.steps.iter_mut().enumerate() {
        let prog_read = step
            .twist_events
            .iter()
            .find(|e| e.twist_id == PROG_ID && e.kind == TwistOpKind::Read)
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "missing PROG read event while injecting decode lookup events at step {step_idx}"
                ))
            })?;
        let addr = prog_read.addr;
        if (addr as usize) >= prog_layout.k {
            return Err(PiCcsError::ProtocolError(format!(
                "decode lookup event addr out of range at step {step_idx}: addr={addr}, k={}",
                prog_layout.k
            )));
        }
        let instr_word = prog_init_words
            .get(&(PROG_ID.0, addr))
            .copied()
            .unwrap_or_else(|| F::from_u64(prog_read.value))
            .as_canonical_u64() as u32;
        let row = rv32_decode_lookup_backed_row_from_instr_word(&decode_layout, instr_word, /*active=*/ true);
        for &col_id in decode_cols.iter() {
            step.shout_events.push(ShoutEvent {
                shout_id: ShoutId(rv32_decode_lookup_table_id_for_col(col_id)),
                key: u128::from(addr),
                value: row[col_id].as_canonical_u64(),
            });
        }
    }
    Ok(())
}

fn build_rv32_width_lookup_tables(
    width_layout: &Rv32WidthSidecarLayout,
    exec: &Rv32ExecTable,
    trace_steps: usize,
) -> Result<(HashMap<u32, LutTable<F>>, usize), PiCcsError> {
    // Width lookup tables here are execution-indexed helper transport tables.
    // They are not a standalone trust root: Route-A width residual claims bind
    // every opened helper value back to committed trace columns (`ram_rv`,
    // `rs2_val`), and WB/WP enforce the associated bitness/quiescence properties.
    let max_cycle = exec
        .rows
        .iter()
        .take(trace_steps)
        .map(|r| r.cycle)
        .max()
        .unwrap_or(0);
    let cycle_d = required_bits_for_max_addr(max_cycle).max(2);
    let cycle_k = 1usize
        .checked_shl(cycle_d as u32)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("width lookup cycle width too large: d={cycle_d}")))?;

    let wit = rv32_width_sidecar_witness_from_exec_table(width_layout, exec);
    let width_cols = rv32_width_lookup_backed_cols(width_layout);
    if width_cols.is_empty() {
        return Ok((HashMap::new(), cycle_d));
    }
    let mut unique_table_ids: Vec<u32> = width_cols
        .iter()
        .map(|&col_id| rv32_width_lookup_table_id_for_col(col_id))
        .collect();
    unique_table_ids.sort_unstable();
    unique_table_ids.dedup();
    let mut out = HashMap::new();
    if unique_table_ids.len() == 1 {
        let table_id = unique_table_ids[0];
        let n_vals = width_cols.len().max(1);
        let content_len = cycle_k
            .checked_mul(n_vals)
            .ok_or_else(|| PiCcsError::InvalidInput("width lookup content length overflow".into()))?;
        let mut content = vec![F::ZERO; content_len];
        for (i, row) in exec.rows.iter().enumerate().take(trace_steps) {
            let cycle = row.cycle as usize;
            if cycle >= cycle_k {
                return Err(PiCcsError::ProtocolError(format!(
                    "width lookup cycle out of range at row {i}: cycle={}, k={cycle_k}",
                    row.cycle
                )));
            }
            let base = cycle * n_vals;
            for (val_slot, &col_id) in width_cols.iter().enumerate() {
                content[base + val_slot] = wit.cols[col_id][i];
            }
        }
        out.insert(
            table_id,
            LutTable {
                table_id,
                k: cycle_k,
                d: cycle_d,
                n_side: 2,
                content,
            },
        );
        return Ok((out, cycle_d));
    }
    for &col_id in width_cols.iter() {
        let table_id = rv32_width_lookup_table_id_for_col(col_id);
        let mut content = vec![F::ZERO; cycle_k];
        for (i, row) in exec.rows.iter().enumerate().take(trace_steps) {
            let cycle = row.cycle as usize;
            if cycle >= cycle_k {
                return Err(PiCcsError::ProtocolError(format!(
                    "width lookup cycle out of range at row {i}: cycle={}, k={cycle_k}",
                    row.cycle
                )));
            }
            content[cycle] = wit.cols[col_id][i];
        }
        out.insert(
            table_id,
            LutTable {
                table_id,
                k: cycle_k,
                d: cycle_d,
                n_side: 2,
                content,
            },
        );
    }
    Ok((out, cycle_d))
}

fn inject_rv32_width_lookup_events_into_trace(
    trace: &mut VmTrace<u64, u64, u128>,
    exec: &Rv32ExecTable,
    width_layout: &Rv32WidthSidecarLayout,
) -> Result<(), PiCcsError> {
    if trace.steps.len() > exec.rows.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "width lookup injection drift: trace steps {} > exec rows {}",
            trace.steps.len(),
            exec.rows.len()
        )));
    }
    let wit = rv32_width_sidecar_witness_from_exec_table(width_layout, exec);
    let width_cols = rv32_width_lookup_backed_cols(width_layout);
    for (i, step) in trace.steps.iter_mut().enumerate() {
        let cycle = exec
            .rows
            .get(i)
            .ok_or_else(|| PiCcsError::ProtocolError("missing exec row while injecting width lookups".into()))?
            .cycle;
        for &col_id in width_cols.iter() {
            step.shout_events.push(ShoutEvent {
                shout_id: ShoutId(rv32_width_lookup_table_id_for_col(col_id)),
                key: u128::from(cycle),
                value: wit.cols[col_id][i].as_canonical_u64(),
            });
        }
    }
    Ok(())
}

/// High-level builder for proving/verifying the RV32 trace wiring CCS.
///
/// This path is intentionally narrow:
/// - builds a padded execution table,
/// - proves one or more trace-wiring CCS steps (IVC),
/// - verifies the resulting shard proof.
#[derive(Clone, Copy, Debug, Default)]
enum OutputTarget {
    #[default]
    Ram,
    Reg,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32TraceProvePhaseDurations {
    pub setup: Duration,
    pub chunk_build_commit: Duration,
    pub fold_and_prove: Duration,
}

#[derive(Clone, Debug)]
pub struct Rv32TraceWiring {
    program_base: u64,
    program_bytes: Vec<u8>,
    xlen: usize,
    max_steps: Option<usize>,
    min_trace_len: usize,
    chunk_rows: Option<usize>,
    mode: FoldingMode,
    ram_init: HashMap<u64, u64>,
    reg_init: HashMap<u64, u64>,
    output_claims: ProgramIO<F>,
    output_target: OutputTarget,
    shout_ops: Option<HashSet<RiscvOpcode>>,
    extra_lut_table_specs: HashMap<u32, LutTableSpec>,
    extra_shout_bus_specs: Vec<TraceShoutBusSpec>,
}

impl Rv32TraceWiring {
    /// Create a trace runner from ROM bytes.
    pub fn from_rom(program_base: u64, program_bytes: &[u8]) -> Self {
        Self {
            program_base,
            program_bytes: program_bytes.to_vec(),
            xlen: 32,
            max_steps: None,
            min_trace_len: 4,
            chunk_rows: None,
            mode: FoldingMode::Optimized,
            ram_init: HashMap::new(),
            reg_init: HashMap::new(),
            output_claims: ProgramIO::new(),
            output_target: OutputTarget::Ram,
            shout_ops: None,
            extra_lut_table_specs: HashMap::new(),
            extra_shout_bus_specs: Vec::new(),
        }
    }

    pub fn xlen(mut self, xlen: usize) -> Self {
        self.xlen = xlen;
        self
    }

    /// Lower-bound for execution-table length.
    ///
    /// Final `t` is `max(trace_len, min_trace_len)`.
    pub fn min_trace_len(mut self, min_trace_len: usize) -> Self {
        self.min_trace_len = min_trace_len.max(1);
        self
    }

    /// Fixed rows per trace step for IVC folding.
    ///
    /// The trace is split into fixed-size chunks, each chunk is proven with the same step CCS,
    /// and step-linking enforces `pc_final -> pc0`.
    pub fn chunk_rows(mut self, chunk_rows: usize) -> Self {
        self.chunk_rows = Some(chunk_rows);
        self
    }

    /// Toggle shared-CPU-bus trace proving mode.
    ///
    /// Only `true` is supported. Passing `false` is rejected by [`Rv32TraceWiring::prove`].
    pub fn shared_cpu_bus(self, enabled: bool) -> Self {
        assert!(enabled, "legacy no-shared CPU bus mode was removed");
        self
    }

    /// Bound executed instruction count.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    pub fn mode(mut self, mode: FoldingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Initialize RAM byte-addressed word cell to a u32 value.
    pub fn ram_init_u32(mut self, addr: u64, value: u32) -> Self {
        self.ram_init.insert(addr, value as u64);
        self
    }

    /// Initialize register `reg` (x0..x31) to a u32 value.
    pub fn reg_init_u32(mut self, reg: u64, value: u32) -> Self {
        self.reg_init.insert(reg, value as u64);
        self
    }

    pub fn output(mut self, output_addr: u64, expected_output: F) -> Self {
        self.output_claims = ProgramIO::new().with_output(output_addr, expected_output);
        self.output_target = OutputTarget::Ram;
        self
    }

    pub fn output_claim(mut self, addr: u64, value: F) -> Self {
        if !matches!(self.output_target, OutputTarget::Ram) {
            self.output_target = OutputTarget::Ram;
            self.output_claims = ProgramIO::new();
        }
        self.output_claims = self.output_claims.with_output(addr, value);
        self
    }

    pub fn reg_output(mut self, reg: u64, expected: F) -> Self {
        self.output_claims = ProgramIO::new().with_output(reg, expected);
        self.output_target = OutputTarget::Reg;
        self
    }

    pub fn reg_output_claim(mut self, reg: u64, expected: F) -> Self {
        if !matches!(self.output_target, OutputTarget::Reg) {
            self.output_target = OutputTarget::Reg;
            self.output_claims = ProgramIO::new();
        }
        self.output_claims = self.output_claims.with_output(reg, expected);
        self
    }

    /// Use the default program-inferred minimal shout set.
    pub fn shout_auto_minimal(mut self) -> Self {
        self.shout_ops = None;
        self
    }

    /// Optional override for shout tables.
    ///
    /// The override must be a superset of the program-inferred required shout set.
    pub fn shout_ops(mut self, ops: impl IntoIterator<Item = RiscvOpcode>) -> Self {
        self.shout_ops = Some(ops.into_iter().collect());
        self
    }

    /// Add an extra implicit lookup-table spec by `table_id`.
    ///
    /// The id must not collide with inferred opcode-table ids.
    pub fn extra_lut_table_spec(mut self, table_id: u32, spec: LutTableSpec) -> Self {
        self.extra_lut_table_specs.insert(table_id, spec);
        self
    }

    /// Optional extra Shout family geometry for trace shared-bus mode.
    ///
    /// Each spec adds/overrides a `table_id -> ell_addr` mapping used to size shout lanes.
    pub fn extra_shout_bus_specs(mut self, specs: impl IntoIterator<Item = TraceShoutBusSpec>) -> Self {
        self.extra_shout_bus_specs = specs.into_iter().collect();
        self
    }

    pub fn prove(self) -> Result<Rv32TraceWiringRun, PiCcsError> {
        if self.xlen != 32 {
            return Err(PiCcsError::InvalidInput(format!(
                "RV32 trace wiring runner requires xlen == 32 (got {})",
                self.xlen
            )));
        }
        if self.program_base != 0 {
            return Err(PiCcsError::InvalidInput(
                "RV32 trace wiring runner requires program_base == 0".into(),
            ));
        }
        if self.program_bytes.is_empty() {
            return Err(PiCcsError::InvalidInput("program_bytes must be non-empty".into()));
        }
        if self.min_trace_len > DEFAULT_RV32_TRACE_MAX_STEPS {
            return Err(PiCcsError::InvalidInput(format!(
                "min_trace_len={} exceeds trace-mode hard cap {}. Increase chunk_rows and prove in chunks for longer executions.",
                self.min_trace_len, DEFAULT_RV32_TRACE_MAX_STEPS
            )));
        }
        if self.program_bytes.len() % 4 != 0 {
            return Err(PiCcsError::InvalidInput(
                "program_bytes must be 4-byte aligned (RVC is not supported)".into(),
            ));
        }
        for (i, chunk) in self.program_bytes.chunks_exact(4).enumerate() {
            let first_half = u16::from_le_bytes([chunk[0], chunk[1]]);
            if (first_half & 0b11) != 0b11 {
                return Err(PiCcsError::InvalidInput(format!(
                    "compressed instruction encoding (RVC) is not supported at word index {i}"
                )));
            }
        }

        let program = decode_program(&self.program_bytes)
            .map_err(|e| PiCcsError::InvalidInput(format!("decode_program failed: {e}")))?;
        let requires_poseidon_stage = program_requires_poseidon_stage(&program);
        if requires_poseidon_stage && !cfg!(feature = "poseidon-precompile") {
            return Err(PiCcsError::InvalidInput(
                "program uses Poseidon2 precompile instructions, but feature `poseidon-precompile` is disabled".into(),
            ));
        }
        let using_default_max_steps = self.max_steps.is_none();
        let max_steps = match self.max_steps {
            Some(n) => {
                if n == 0 {
                    return Err(PiCcsError::InvalidInput("max_steps must be non-zero".into()));
                }
                if n > DEFAULT_RV32_TRACE_MAX_STEPS {
                    return Err(PiCcsError::InvalidInput(format!(
                        "max_steps={} exceeds trace-mode hard cap {}. Increase chunk_rows and prove in chunks for longer executions.",
                        n, DEFAULT_RV32_TRACE_MAX_STEPS
                    )));
                }
                n
            }
            None => DEFAULT_RV32_TRACE_MAX_STEPS,
        };
        let vm_max_steps = max_steps
            .saturating_mul(RV32_RUNTIME_DECOMP_MAX_ROWS_PER_INSTR)
            .min(DEFAULT_RV32_TRACE_MAX_VM_STEPS);
        let ram_init_map = self.ram_init.clone();
        let reg_init_map = self.reg_init.clone();
        let output_claims = self.output_claims.clone();
        let output_target = self.output_target;
        let (prog_layout, prog_init_words) =
            prog_rom_layout_and_init_words::<F>(PROG_ID, /*base_addr=*/ 0, &self.program_bytes)
                .map_err(|e| PiCcsError::InvalidInput(format!("prog_rom_layout_and_init_words failed: {e}")))?;

        let mut vm = RiscvCpu::new(self.xlen);
        vm.load_program(/*base=*/ 0, program.clone());
        vm.set_runtime_decomposition_enabled(true);

        let mut twist =
            RiscvMemory::with_program_in_twist(self.xlen, PROG_ID, /*base_addr=*/ 0, &self.program_bytes);
        for (&addr, &value) in &ram_init_map {
            twist.store(RAM_ID, addr, value as u32 as u64);
        }
        for (&reg, &value) in &reg_init_map {
            if reg >= 32 {
                return Err(PiCcsError::InvalidInput(format!(
                    "reg_init_u32: register index out of range: reg={reg} (expected 0..32)"
                )));
            }
            if reg == 0 && value != 0 {
                return Err(PiCcsError::InvalidInput(
                    "reg_init_u32: x0 must be 0 (non-zero init is forbidden)".into(),
                ));
            }
            twist.store(REG_ID, reg, value as u32 as u64);
        }
        let shout = RiscvShoutTables::new(self.xlen);

        let mut trace = neo_vm_trace::trace_program(vm, twist, shout, vm_max_steps)
            .map_err(|e| PiCcsError::InvalidInput(format!("trace_program failed: {e}")))?;
        validate_trace_opcode_lookup_one_hot(&trace, self.xlen)?;

        if using_default_max_steps && !trace.did_halt() {
            return Err(PiCcsError::InvalidInput(format!(
                "RV32 execution did not halt within max_steps={max_steps} (vm_max_steps={vm_max_steps} after decomposition expansion); call .max_steps(...) to raise the limit or ensure the guest halts"
            )));
        }

        let target_len = trace.steps.len().max(self.min_trace_len);
        if target_len > DEFAULT_RV32_TRACE_MAX_VM_STEPS {
            return Err(PiCcsError::InvalidInput(format!(
                "trace length {} exceeds expanded trace-row hard cap {}. Lower max_steps/min_trace_len or reduce decomposition density.",
                target_len, DEFAULT_RV32_TRACE_MAX_VM_STEPS
            )));
        }
        inject_rv32_decode_lookup_events_into_trace(&mut trace, &prog_layout, &prog_init_words)?;
        let exec = Rv32ExecTable::from_trace_padded(&trace, target_len)
            .map_err(|e| PiCcsError::InvalidInput(format!("Rv32ExecTable::from_trace_padded failed: {e}")))?;
        exec.validate_cycle_chain()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_cycle_chain failed: {e}")))?;
        exec.validate_pc_chain()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_pc_chain failed: {e}")))?;
        exec.validate_halted_tail()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_halted_tail failed: {e}")))?;
        exec.validate_inactive_rows_are_empty()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_inactive_rows_are_empty failed: {e}")))?;
        let width_layout = Rv32WidthSidecarLayout::new();
        let include_width_lookup = program_requires_width_lookup(&program);
        let (width_lookup_tables, width_lookup_addr_d) = if include_width_lookup {
            let (tables, addr_d) = build_rv32_width_lookup_tables(&width_layout, &exec, trace.steps.len())?;
            inject_rv32_width_lookup_events_into_trace(&mut trace, &exec, &width_layout)?;
            (tables, addr_d)
        } else {
            (HashMap::new(), 0usize)
        };

        // Default policy:
        // - Poseidon-precompile programs should try a single-fold geometry first.
        // - Other programs keep the historical default chunk size.
        let requested_chunk_rows_arch = match self.chunk_rows {
            Some(rows) => rows,
            None if requires_poseidon_stage => exec.rows.len().max(1),
            None => DEFAULT_RV32_TRACE_CHUNK_ROWS,
        };
        if requested_chunk_rows_arch == 0 {
            return Err(PiCcsError::InvalidInput("trace chunk_rows must be non-zero".into()));
        }
        let requested_chunk_rows = requested_chunk_rows_arch.max(max_consecutive_pc_run(&exec));
        let base_step_rows = requested_chunk_rows.min(exec.rows.len().max(1));
        let mut step_rows = base_step_rows;
        while step_rows < exec.rows.len() && boundary_splits_virtual_sequence(&exec, step_rows) {
            step_rows = step_rows
                .checked_add(base_step_rows)
                .ok_or_else(|| PiCcsError::InvalidInput(format!("trace chunk_rows overflow: {step_rows}")))?;
        }
        step_rows = step_rows.min(exec.rows.len().max(1));
        let exec_chunks = split_exec_into_fixed_chunks(&exec, step_rows)?;

        let mut layout = Rv32TraceCcsLayout::new_uniform(step_rows)
            .map_err(|e| PiCcsError::InvalidInput(format!("Rv32TraceCcsLayout::new_uniform failed: {e}")))?;

        let prove_start = time_now();
        let setup_start = prove_start;

        let mut max_ram_addr = max_ram_addr_from_exec(&exec).unwrap_or(0);
        if let Some(max_init_addr) = ram_init_map.keys().copied().max() {
            max_ram_addr = max_ram_addr.max(max_init_addr);
        }
        let mut max_reg_addr = trace
            .steps
            .iter()
            .flat_map(|step| step.twist_events.iter())
            .filter(|event| event.twist_id == REG_ID)
            .map(|event| event.addr)
            .max()
            .unwrap_or(31);
        if let Some(max_init_reg_addr) = reg_init_map.keys().copied().max() {
            max_reg_addr = max_reg_addr.max(max_init_reg_addr);
        }
        let wants_ram_output = matches!(output_target, OutputTarget::Ram) && !output_claims.is_empty();
        if matches!(output_target, OutputTarget::Ram) {
            if let Some(max_claim_addr) = output_claims.claimed_addresses().max() {
                max_ram_addr = max_ram_addr.max(max_claim_addr);
            }
        }
        let reg_d = required_bits_for_max_addr(max_reg_addr).max(5);
        let reg_k = 1usize
            .checked_shl(reg_d as u32)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("REG address width too large: d={reg_d}")))?;
        let ram_d = required_bits_for_max_addr(max_ram_addr).max(2);
        let ram_k = 1usize
            .checked_shl(ram_d as u32)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("RAM address width too large: d={ram_d}")))?;
        // Track A used-set derivation must be deterministic from public inputs/config.
        // Do not derive RAM inclusion from runtime witness/events.
        let include_ram_sidecar =
            program_requires_ram_sidecar(&program) || !ram_init_map.is_empty() || wants_ram_output;

        let mut mem_layouts: HashMap<u32, PlainMemLayout> = HashMap::from([
            (
                REG_ID.0,
                PlainMemLayout {
                    k: reg_k,
                    d: reg_d,
                    n_side: 2,
                    lanes: 2,
                },
            ),
            (PROG_ID.0, prog_layout.clone()),
        ]);
        if include_ram_sidecar {
            mem_layouts.insert(
                RAM_ID.0,
                PlainMemLayout {
                    k: ram_k,
                    d: ram_d,
                    n_side: 2,
                    lanes: 1,
                },
            );
        }

        let inferred_shout_ops = infer_required_trace_shout_opcodes(&program);
        let shout_ops = match &self.shout_ops {
            Some(override_ops) => {
                let missing: HashSet<RiscvOpcode> = inferred_shout_ops
                    .difference(override_ops)
                    .copied()
                    .collect();
                if !missing.is_empty() {
                    let mut missing_names: Vec<String> = missing.into_iter().map(|op| format!("{op:?}")).collect();
                    missing_names.sort_unstable();
                    return Err(PiCcsError::InvalidInput(format!(
                        "trace shout_ops override must be a superset of required opcodes; missing [{}]",
                        missing_names.join(", ")
                    )));
                }
                override_ops.clone()
            }
            None => inferred_shout_ops,
        };
        let decode_layout = Rv32DecodeSidecarLayout::new();
        let decode_lookup_tables = build_rv32_decode_lookup_tables(&prog_layout, &prog_init_words);
        let decode_lookup_bus_specs: Vec<TraceShoutBusSpec> = {
            let decode_lookup_cols = rv32_decode_lookup_transport_cols(&decode_layout);
            let mut decode_table_ids: Vec<u32> = decode_lookup_cols
                .iter()
                .map(|&col_id| rv32_decode_lookup_table_id_for_col(col_id))
                .collect();
            decode_table_ids.sort_unstable();
            decode_table_ids.dedup();
            if decode_table_ids.len() == 1 {
                vec![TraceShoutBusSpec {
                    table_id: decode_table_ids[0],
                    ell_addr: prog_layout.d,
                    n_vals: decode_lookup_cols.len().max(1),
                }]
            } else {
                decode_lookup_cols
                    .iter()
                    .copied()
                    .map(|col_id| TraceShoutBusSpec {
                        table_id: rv32_decode_lookup_table_id_for_col(col_id),
                        ell_addr: prog_layout.d,
                        n_vals: 1usize,
                    })
                    .collect()
            }
        };
        let width_lookup_bus_specs: Vec<TraceShoutBusSpec> = if include_width_lookup {
            let width_lookup_cols = rv32_width_lookup_backed_cols(&width_layout);
            let mut width_table_ids: Vec<u32> = width_lookup_cols
                .iter()
                .map(|&col_id| rv32_width_lookup_table_id_for_col(col_id))
                .collect();
            width_table_ids.sort_unstable();
            width_table_ids.dedup();
            if width_table_ids.len() == 1 {
                vec![TraceShoutBusSpec {
                    table_id: width_table_ids[0],
                    ell_addr: width_lookup_addr_d,
                    n_vals: width_lookup_cols.len().max(1),
                }]
            } else {
                width_lookup_cols
                    .iter()
                    .copied()
                    .map(|col_id| TraceShoutBusSpec {
                        table_id: rv32_width_lookup_table_id_for_col(col_id),
                        ell_addr: width_lookup_addr_d,
                        n_vals: 1usize,
                    })
                    .collect()
            }
        } else {
            Vec::new()
        };
        let mut table_specs = rv32_trace_table_specs(&shout_ops)?;
        // Runtime decomposition can introduce helper Shout opcodes that are not directly
        // visible in the architectural program stream. Ensure those table_ids are present
        // in table_specs, defaulting to canonical non-packed opcode specs.
        let shout_tables = RiscvShoutTables::new(self.xlen);
        let trace_shout_table_ids: HashSet<u32> = trace
            .steps
            .iter()
            .flat_map(|step| step.shout_events.iter())
            .map(|event| event.shout_id.0)
            .collect();
        for table_id in trace_shout_table_ids {
            if table_specs.contains_key(&table_id) {
                continue;
            }
            let Some(opcode) = shout_tables.id_to_opcode(ShoutId(table_id)) else {
                continue;
            };
            let spec = if rv32_packed_rollout_opcode(opcode)
                && !neo_memory::riscv::trace::rv32_trace_uses_combined_operand_key_table_id(table_id)
            {
                LutTableSpec::RiscvOpcodePacked {
                    opcode,
                    xlen: self.xlen,
                }
            } else {
                LutTableSpec::RiscvOpcode {
                    opcode,
                    xlen: self.xlen,
                }
            };
            table_specs.insert(table_id, spec);
        }
        for (&table_id, spec) in &self.extra_lut_table_specs {
            if table_specs.contains_key(&table_id) {
                return Err(PiCcsError::InvalidInput(format!(
                    "extra_lut_table_spec collides with existing table_id={table_id}"
                )));
            }
            table_specs.insert(table_id, spec.clone());
        }
        let mut base_shout_table_ids: Vec<u32> = table_specs
            .iter()
            .filter_map(|(&table_id, spec)| match spec {
                LutTableSpec::RiscvOpcodePacked { .. } | LutTableSpec::RiscvOpcodeEventTablePacked { .. } => None,
                _ => Some(table_id),
            })
            .collect();
        base_shout_table_ids.sort_unstable();
        let mut all_extra_shout_specs = self.extra_shout_bus_specs.clone();
        for (&table_id, spec) in &table_specs {
            match spec {
                LutTableSpec::RiscvOpcodePacked { opcode, xlen } => {
                    if *xlen != 32 {
                        return Err(PiCcsError::InvalidInput(format!(
                            "RiscvOpcodePacked requires xlen=32 in RV32 trace mode (table_id={table_id}, xlen={xlen})"
                        )));
                    }
                    all_extra_shout_specs.push(TraceShoutBusSpec {
                        table_id,
                        ell_addr: rv32_packed_d(*opcode)?,
                        n_vals: 1usize,
                    });
                }
                LutTableSpec::RiscvOpcodeEventTablePacked { .. } => {
                    return Err(PiCcsError::InvalidInput(
                        "RiscvOpcodeEventTablePacked is not supported in RV32 trace shared-bus mode".into(),
                    ));
                }
                _ => {}
            }
        }
        all_extra_shout_specs.extend(decode_lookup_bus_specs.clone());
        all_extra_shout_specs.extend(width_lookup_bus_specs.clone());
        for spec in &all_extra_shout_specs {
            if !table_specs.contains_key(&spec.table_id)
                && !decode_lookup_tables.contains_key(&spec.table_id)
                && !width_lookup_tables.contains_key(&spec.table_id)
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "extra_shout_bus_specs includes table_id={} without a table spec/table content",
                    spec.table_id
                )));
            }
        }

        let mut lut_tables = decode_lookup_tables.clone();
        lut_tables.extend(width_lookup_tables.clone());
        let lut_lanes: HashMap<u32, usize> = HashMap::new();

        // Route-A in-place uniform kernel: physical CCS width is per-step columns
        // (m_in + cpu_cols + mem_cols), not row-flattened col*t spans.
        let bus_cols = estimate_route_a_bus_cols(layout.t, &table_specs, &lut_tables, &mem_layouts, &lut_lanes)?;
        let uniform_m = layout
            .m_in
            .checked_add(layout.trace.cols)
            .and_then(|v| v.checked_add(bus_cols))
            .ok_or_else(|| PiCcsError::InvalidInput("uniform m overflow".into()))?;
        layout.m = uniform_m;

        let ccs = build_rv32_trace_wiring_ccs(&layout)
            .map_err(|e| PiCcsError::InvalidInput(format!("build_rv32_trace_wiring_ccs failed: {e}")))?;

        // Keep params as selected by NeoParams presets during debugging.
        let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n.max(ccs.m))
            .map_err(|e| PiCcsError::InvalidInput(format!("NeoParams::goldilocks_auto_r1cs_ccs failed: {e}")))?;
        let m_commit = neo_memory::ajtai::commit_cols_for_ccs_m(ccs.m);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let pp = neo_ajtai::setup_par(&mut rng, D, params.kappa as usize, m_commit)
            .map_err(|e| PiCcsError::InvalidInput(format!("Ajtai setup failed: {e}")))?;
        let mut session = FoldingSession::new(self.mode.clone(), params, AjtaiSModule::new(Arc::new(pp)));
        session.set_step_linking(StepLinkingConfig::new(vec![(layout.pc_final, layout.pc0)]));

        let mut prog_init_pairs: Vec<(u64, F)> = prog_init_words
            .into_iter()
            .filter_map(|((mem_id, addr), value)| (mem_id == PROG_ID.0 && value != F::ZERO).then_some((addr, value)))
            .collect();
        prog_init_pairs.sort_by_key(|(addr, _)| *addr);
        let prog_mem_init = if prog_init_pairs.is_empty() {
            MemInit::Zero
        } else {
            MemInit::Sparse(prog_init_pairs)
        };
        let mut initial_mem: HashMap<(u32, u64), F> = HashMap::new();
        if let MemInit::Sparse(pairs) = &prog_mem_init {
            for &(addr, value) in pairs {
                if value != F::ZERO {
                    initial_mem.insert((PROG_ID.0, addr), value);
                }
            }
        }
        for (&reg, &value) in &reg_init_map {
            let v = F::from_u64(value as u32 as u64);
            if v != F::ZERO {
                initial_mem.insert((REG_ID.0, reg), v);
            }
        }
        for (&addr, &value) in &ram_init_map {
            let v = F::from_u64(value as u32 as u64);
            if v != F::ZERO {
                initial_mem.insert((RAM_ID.0, addr), v);
            }
        }

        let setup_duration = elapsed_duration(setup_start);
        let mut chunk_build_commit_duration = Duration::ZERO;
        let chunk_start = time_now();

        let mut lookup_addr_groups = HashMap::<u32, u64>::new();
        let mut lookup_selector_groups = HashMap::<u32, u64>::new();
        let all_table_ids: HashSet<u32> = table_specs
            .keys()
            .copied()
            .chain(lut_tables.keys().copied())
            .collect();
        for table_id in all_table_ids {
            let ell_addr = table_ell_addr_for_shared_bus(table_id, &table_specs, &lut_tables)?;
            if let Some(g) = trace_lookup_addr_group_for_table_shape(table_id, ell_addr) {
                lookup_addr_groups.insert(table_id, g);
            }
            if let Some(g) = rv32_trace_lookup_selector_group_for_table_id(table_id) {
                lookup_selector_groups.insert(table_id, g as u64);
            }
        }
        chunk_build_commit_duration += elapsed_duration(chunk_start);

        let cpu = R1csCpu::new(
            ccs.clone(),
            session.params().clone(),
            session.committer().clone(),
            layout.m_in,
            &lut_tables,
            &table_specs,
            rv32_trace_chunk_to_witness(layout.clone()),
        )
        .map_err(|e| PiCcsError::InvalidInput(format!("R1csCpu::new failed: {e}")))?
        .with_lookup_sharing_groups(lookup_addr_groups, lookup_selector_groups);

        session.execute_shard_shared_cpu_bus_from_trace(
            &trace,
            vm_max_steps,
            layout.t,
            &mem_layouts,
            &lut_tables,
            &table_specs,
            &lut_lanes,
            &initial_mem,
            &cpu,
        )?;

        if session.steps_witness().len() != exec_chunks.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "shared trace build drift: step bundle count {} != exec chunk count {}",
                session.steps_witness().len(),
                exec_chunks.len()
            )));
        }
        chunk_build_commit_duration += elapsed_duration(chunk_start);

        let mem_order = session
            .steps_public()
            .first()
            .map(|s| {
                s.mem_insts
                    .iter()
                    .map(|inst| inst.mem_id)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let ram_ob_mem_idx = if wants_ram_output {
            Some(
                mem_order
                    .iter()
                    .position(|&id| id == RAM_ID.0)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing RAM mem instance for output binding".into()))?,
            )
        } else {
            None
        };
        let reg_ob_mem_idx = mem_order
            .iter()
            .position(|&id| id == REG_ID.0)
            .ok_or_else(|| PiCcsError::ProtocolError("missing REG mem instance for output binding".into()))?;

        let fold_start = time_now();
        let (proof, output_binding_cfg) = if output_claims.is_empty() {
            (session.fold_and_prove(&ccs)?, None)
        } else {
            let (ob_mem_idx, ob_num_bits, ob_max_addr) = match output_target {
                OutputTarget::Ram => (
                    ram_ob_mem_idx.ok_or_else(|| {
                        PiCcsError::ProtocolError("missing RAM mem instance for output binding".into())
                    })?,
                    ram_d,
                    max_ram_addr,
                ),
                OutputTarget::Reg => (reg_ob_mem_idx, reg_d, max_reg_addr),
            };
            let use_dense_output_sumcheck = ob_num_bits <= neo_memory::output_check::OUTPUT_SUMCHECK_MAX_NUM_BITS;
            if use_dense_output_sumcheck {
                output_claims.validate(ob_num_bits).map_err(|e| {
                    let target = match output_target {
                        OutputTarget::Ram => "RAM",
                        OutputTarget::Reg => "REG",
                    };
                    PiCcsError::InvalidInput(format!(
                        "output binding invalid for {target} target (num_bits={ob_num_bits}, max_addr={ob_max_addr}): {e}"
                    ))
                })?;
            } else {
                neo_memory::output_check::validate_output_binding_domain(&output_claims, ob_num_bits).map_err(|e| {
                    let target = match output_target {
                        OutputTarget::Ram => "RAM",
                        OutputTarget::Reg => "REG",
                    };
                    PiCcsError::InvalidInput(format!(
                        "output binding invalid for {target} target (num_bits={ob_num_bits}, max_addr={ob_max_addr}) in sparse point-check mode: {e}"
                    ))
                })?;
            }
            let final_memory_state = if use_dense_output_sumcheck {
                match output_target {
                    OutputTarget::Ram => final_ram_state_dense(&exec, &ram_init_map, ram_k)?,
                    OutputTarget::Reg => final_reg_state_dense(&exec, &reg_init_map, reg_k)?,
                }
            } else {
                Vec::new()
            };
            let ob_cfg = OutputBindingConfig::new(ob_num_bits, output_claims).with_mem_idx(ob_mem_idx);
            let proof = session.fold_and_prove_with_output_binding_simple(&ccs, &ob_cfg, &final_memory_state)?;
            (proof, Some(ob_cfg))
        };
        let fold_and_prove_duration = elapsed_duration(fold_start);
        let prove_duration = elapsed_duration(prove_start);
        let prove_phase_durations = Rv32TraceProvePhaseDurations {
            setup: setup_duration,
            chunk_build_commit: chunk_build_commit_duration,
            fold_and_prove: fold_and_prove_duration,
        };

        let mut used_mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
        used_mem_ids.sort_unstable();
        let mut used_shout_table_ids = base_shout_table_ids.clone();
        for spec in &all_extra_shout_specs {
            if !used_shout_table_ids.contains(&spec.table_id) {
                used_shout_table_ids.push(spec.table_id);
            }
        }
        used_shout_table_ids.sort_unstable();

        Ok(Rv32TraceWiringRun {
            session,
            ccs,
            layout,
            exec,
            proof,
            used_mem_ids,
            used_shout_table_ids,
            output_binding_cfg,
            requires_poseidon_stage,
            prove_duration,
            prove_phase_durations,
            verify_duration: None,
        })
    }
}

/// Completed trace-wiring proof run.
pub struct Rv32TraceWiringRun {
    session: FoldingSession<AjtaiSModule>,
    ccs: CcsStructure<F>,
    layout: Rv32TraceCcsLayout,
    exec: Rv32ExecTable,
    proof: ShardProof,
    used_mem_ids: Vec<u32>,
    used_shout_table_ids: Vec<u32>,
    output_binding_cfg: Option<OutputBindingConfig>,
    requires_poseidon_stage: bool,
    prove_duration: Duration,
    prove_phase_durations: Rv32TraceProvePhaseDurations,
    verify_duration: Option<Duration>,
}

impl Rv32TraceWiringRun {
    pub fn params(&self) -> &NeoParams {
        self.session.params()
    }

    pub fn committer(&self) -> &AjtaiSModule {
        self.session.committer()
    }

    pub fn ccs(&self) -> &CcsStructure<F> {
        &self.ccs
    }

    pub fn layout(&self) -> &Rv32TraceCcsLayout {
        &self.layout
    }

    pub fn exec_table(&self) -> &Rv32ExecTable {
        &self.exec
    }

    pub fn proof(&self) -> &ShardProof {
        &self.proof
    }

    /// Auto-derived memory sidecar IDs used by this run (`S_memory`).
    pub fn used_memory_ids(&self) -> &[u32] {
        &self.used_mem_ids
    }

    /// Auto-derived shout lookup table IDs used by this run (`S_lookup`).
    pub fn used_shout_table_ids(&self) -> &[u32] {
        &self.used_shout_table_ids
    }

    /// Whether static program scan requires Poseidon sidecar proof lanes.
    pub fn requires_poseidon_stage(&self) -> bool {
        self.requires_poseidon_stage
    }

    pub fn verify_proof(&self, proof: &ShardProof) -> Result<(), PiCcsError> {
        if !self.requires_poseidon_stage {
            for (step_idx, step) in proof.steps.iter().enumerate() {
                if step.poseidon_local_time.is_some()
                    || !step.mem.poseidon_cycle_me_claims.is_empty()
                    || !step.mem.poseidon_local_me_claims.is_empty()
                    || !step.poseidon_cycle_fold.is_empty()
                    || !step.poseidon_local_fold.is_empty()
                {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: poseidon proof artifacts present but stage is not required",
                        step_idx
                    )));
                }
            }
        }
        let ok = match &self.output_binding_cfg {
            None => self.session.verify_collected(&self.ccs, proof)?,
            Some(cfg) => self
                .session
                .verify_with_output_binding_collected_simple(&self.ccs, proof, cfg)?,
        };
        if !ok {
            return Err(PiCcsError::ProtocolError("verification failed".into()));
        }
        Ok(())
    }

    pub fn verify(&mut self) -> Result<(), PiCcsError> {
        let verify_start = time_now();
        self.verify_proof(&self.proof)?;
        self.verify_duration = Some(elapsed_duration(verify_start));
        Ok(())
    }

    pub fn ccs_num_constraints(&self) -> usize {
        self.ccs.n
    }

    pub fn ccs_num_variables(&self) -> usize {
        self.ccs.m
    }

    /// Uniform per-step CCS width proxy: public inputs + named CPU/memory columns.
    ///
    /// This reflects the in-place migration target where width is column-based and
    /// independent of shard chunk rows.
    pub fn uniform_ccs_num_variables(&self) -> usize {
        let steps = self.steps_public();
        let Some(step0) = steps.first() else {
            return self.ccs.m;
        };
        if step0.time_columns.t == 0 {
            return self.ccs.m;
        }
        if step0.time_columns.cpu_cols.is_empty() && step0.time_columns.mem_cols.is_empty() {
            return self.ccs.m;
        }
        let stable = steps.iter().all(|step| {
            step.mcs_inst.m_in == step0.mcs_inst.m_in
                && step.time_columns.t == step0.time_columns.t
                && step.time_columns.cpu_cols.len() == step0.time_columns.cpu_cols.len()
                && step.time_columns.mem_cols.len() == step0.time_columns.mem_cols.len()
        });
        if !stable {
            return self.ccs.m;
        }
        step0
            .mcs_inst
            .m_in
            .checked_add(step0.time_columns.cpu_cols.len())
            .and_then(|v| v.checked_add(step0.time_columns.mem_cols.len()))
            .unwrap_or(self.ccs.m)
    }

    /// Number of real (active) rows in the unpadded trace.
    pub fn trace_len(&self) -> usize {
        self.exec.rows.iter().filter(|r| r.active).count()
    }

    /// Number of collected folding steps.
    pub fn fold_count(&self) -> usize {
        if let Some(meta) = self.proof.segment_meta.as_deref() {
            let total_public_steps: usize = meta.iter().map(|entry| entry.public_steps).sum();
            if total_public_steps != 0 {
                return total_public_steps;
            }
        }
        self.proof
            .steps
            .iter()
            .map(|step| {
                step.compressed_substeps
                    .as_ref()
                    .map_or(1, |sub| sub.len() + 1)
            })
            .sum()
    }

    pub fn prove_duration(&self) -> Duration {
        self.prove_duration
    }

    pub fn prove_phase_durations(&self) -> Rv32TraceProvePhaseDurations {
        self.prove_phase_durations
    }

    pub fn verify_duration(&self) -> Option<Duration> {
        self.verify_duration
    }

    pub fn steps_public(&self) -> Vec<neo_memory::witness::StepInstanceBundle<neo_ajtai::Commitment, F, K>> {
        self.session.steps_public()
    }
}
