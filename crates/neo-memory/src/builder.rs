use crate::addr::for_each_addr_bit_dim_major_le_u128;
use crate::mem_init::mem_init_from_state_map;
use crate::plain::{LutTable, PlainMemLayout};
use crate::public_digest::memory_public_digest_fields;
use crate::riscv::exec_table::{RiscvExecRow, RiscvExecTable};
use crate::riscv::lookups::uninterleave_bits;
use crate::witness::{LutInstance, LutTableSpec, LutWitness, MemInstance, MemWitness, StepWitnessBundle, TimeColumns};
use crate::{
    cpu::{build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes, ShoutInstanceShape},
    riscv::trace::{
        infer_riscv_trace_machine_xlen, riscv_trace_lookup_n_vals_for_table_id, rv32_trace_cpu_cols,
        rv64_trace_cpu_cols, Rv32TraceLayout, Rv32TraceWitness, Rv64TraceLayout, Rv64TraceWitness,
    },
    AffineWordAddressRemap,
};
use neo_vm_trace::{StepTrace, TwistOpKind, VmTrace};

use neo_ccs::relations::{CcsClaim, CcsWitness};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Range;

// Placeholder for CPU arithmetization interface
pub trait CpuArithmetization<F, Cmt, Key = u64> {
    type Error: std::fmt::Debug + std::fmt::Display;

    fn build_ccs_chunks(
        &self,
        trace: &VmTrace<u64, u64, Key>,
        chunk_size: usize,
    ) -> Result<Vec<(CcsClaim<Cmt, F>, CcsWitness<F>)>, Self::Error>;

    fn build_ccs_steps(
        &self,
        trace: &VmTrace<u64, u64, Key>,
    ) -> Result<Vec<(CcsClaim<Cmt, F>, CcsWitness<F>)>, Self::Error> {
        self.build_ccs_chunks(trace, 1)
    }

    /// Per-table address-sharing group ids for bus layout column sharing.
    ///
    /// Tables with the same group id share `addr_bits` columns in the bus layout.
    /// Default: empty (no sharing). Override in trace mode for column efficiency.
    fn shout_addr_groups(&self) -> &HashMap<u32, u64> {
        static EMPTY: std::sync::LazyLock<HashMap<u32, u64>> = std::sync::LazyLock::new(HashMap::new);
        &EMPTY
    }

    /// Per-table selector-sharing group ids for bus layout column sharing.
    ///
    /// Tables with the same group id share `has_lookup` columns in the bus layout.
    /// Default: empty (no sharing). Override in trace mode for column efficiency.
    fn shout_selector_groups(&self) -> &HashMap<u32, u64> {
        static EMPTY: std::sync::LazyLock<HashMap<u32, u64>> = std::sync::LazyLock::new(HashMap::new);
        &EMPTY
    }
}

#[derive(Debug)]
pub enum ShardBuildError {
    VmError(String),
    CcsError(String),
    InvalidChunkSize(String),
    InvalidInit(String),
    MissingLayout(String),
    MissingTable(String),
}

/// Auxiliary outputs from `build_shard_witness_shared_cpu_bus_with_aux`.
#[derive(Clone, Debug)]
pub struct ShardWitnessAux {
    /// Original (unpadded) VM trace length.
    pub original_len: usize,
    /// Whether the VM halted before reaching `max_steps`.
    pub did_halt: bool,
    pub max_steps: usize,
    pub chunk_size: usize,
    /// Deterministic ordering of Twist instances used by the builder (and by the shared CPU bus).
    pub mem_ids: Vec<u32>,
    /// Final sparse memory states at the end of the shard: mem_id -> (addr -> value), with zero cells omitted.
    pub final_mem_states: HashMap<u32, HashMap<u64, Goldilocks>>,
}

fn ell_from_pow2_n_side(n_side: usize) -> Result<usize, ShardBuildError> {
    if n_side == 0 || !n_side.is_power_of_two() {
        return Err(ShardBuildError::InvalidInit(format!(
            "n_side must be a power of two under bit addressing, got {n_side}"
        )));
    }
    Ok(n_side.trailing_zeros() as usize)
}

fn validate_chunk_size(chunk_size: usize) -> Result<(), ShardBuildError> {
    if chunk_size == 0 {
        return Err(ShardBuildError::InvalidChunkSize("chunk_size must be >= 1".into()));
    }
    Ok(())
}

#[inline]
fn canonical_chunk_size_pow2(chunk_size: usize) -> Result<usize, ShardBuildError> {
    validate_chunk_size(chunk_size)?;
    Ok(chunk_size)
}

fn bundles_only<Cmt, K>(
    out: Result<(Vec<StepWitnessBundle<Cmt, Goldilocks, K>>, ShardWitnessAux), ShardBuildError>,
) -> Result<Vec<StepWitnessBundle<Cmt, Goldilocks, K>>, ShardBuildError> {
    out.map(|(bundles, _aux)| bundles)
}

fn write_addr_bits_into_mem_cols(
    mem_cols: &mut [Vec<Goldilocks>],
    bit_cols: Range<usize>,
    j: usize,
    addr: u128,
    d: usize,
    n_side: usize,
    ell: usize,
) -> Result<(), ShardBuildError> {
    if bit_cols.end.saturating_sub(bit_cols.start) != d * ell {
        return Err(ShardBuildError::CcsError(format!(
            "shared-bus addr-bit width mismatch: range_len={} vs d*ell={}",
            bit_cols.end.saturating_sub(bit_cols.start),
            d * ell
        )));
    }
    for_each_addr_bit_dim_major_le_u128(addr, d, n_side, ell, |idx, is_one| {
        let col_id = bit_cols.start + idx;
        mem_cols[col_id][j] = if is_one { Goldilocks::ONE } else { Goldilocks::ZERO };
    });
    Ok(())
}

#[inline]
fn shout_n_vals_for_table_id(table_id: u32) -> usize {
    riscv_trace_lookup_n_vals_for_table_id(table_id).max(1)
}

fn build_shared_bus_layout_for_time_columns<Cmt>(
    chunk_size: usize,
    lut_instances: &[(LutInstance<Cmt, Goldilocks>, LutWitness<Goldilocks>)],
    mem_instances: &[(MemInstance<Cmt, Goldilocks>, MemWitness<Goldilocks>)],
) -> Result<crate::cpu::BusLayout, ShardBuildError> {
    let shout_shapes: Vec<ShoutInstanceShape> = lut_instances
        .iter()
        .map(|(inst, _)| ShoutInstanceShape {
            ell_addr: inst.d * inst.ell,
            lanes: inst.lanes.max(1),
            n_vals: shout_n_vals_for_table_id(inst.table_id),
            addr_group: inst.addr_group,
            selector_group: inst.selector_group,
        })
        .collect();
    let twist_shapes: Vec<(usize, usize)> = mem_instances
        .iter()
        .map(|(inst, _)| (inst.d * inst.ell, inst.lanes.max(1)))
        .collect();

    let shout_upper_cols = shout_shapes.iter().try_fold(0usize, |acc, shape| {
        let per_lane = shape
            .ell_addr
            .checked_add(1usize)
            .and_then(|v| v.checked_add(shape.n_vals.max(1)))
            .ok_or_else(|| {
                ShardBuildError::CcsError("shared-bus layout overflow while sizing shout lane columns".into())
            })?;
        let lane_cols = shape.lanes.max(1).checked_mul(per_lane).ok_or_else(|| {
            ShardBuildError::CcsError("shared-bus layout overflow while sizing shout instance columns".into())
        })?;
        acc.checked_add(lane_cols).ok_or_else(|| {
            ShardBuildError::CcsError("shared-bus layout overflow while accumulating shout columns".into())
        })
    })?;
    let twist_upper_cols = twist_shapes
        .iter()
        .try_fold(0usize, |acc, (ell_addr, lanes)| {
            let per_lane = ell_addr
                .checked_mul(2usize)
                .and_then(|v| v.checked_add(5usize))
                .ok_or_else(|| {
                    ShardBuildError::CcsError("shared-bus layout overflow while sizing twist lane columns".into())
                })?;
            let lane_cols = (*lanes).max(1).checked_mul(per_lane).ok_or_else(|| {
                ShardBuildError::CcsError("shared-bus layout overflow while sizing twist instance columns".into())
            })?;
            acc.checked_add(lane_cols).ok_or_else(|| {
                ShardBuildError::CcsError("shared-bus layout overflow while accumulating twist columns".into())
            })
        })?;
    let upper_bus_cols = shout_upper_cols
        .checked_add(twist_upper_cols)
        .ok_or_else(|| ShardBuildError::CcsError("shared-bus layout overflow while sizing total bus columns".into()))?;
    let m_probe = chunk_size.checked_mul(upper_bus_cols).ok_or_else(|| {
        ShardBuildError::CcsError("shared-bus layout overflow while sizing time-column witness width".into())
    })?;

    build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
        m_probe,
        0usize,
        chunk_size,
        shout_shapes.iter().copied(),
        twist_shapes.iter().copied(),
    )
    .map_err(ShardBuildError::CcsError)
}

fn build_mem_time_columns_for_step<Cmt, Key>(
    chunk_steps: &[StepTrace<u64, u64, Key>],
    chunk_size: usize,
    lut_instances: &[(LutInstance<Cmt, Goldilocks>, LutWitness<Goldilocks>)],
    mem_instances: &[(MemInstance<Cmt, Goldilocks>, MemWitness<Goldilocks>)],
    mem_addr_remaps: &HashMap<u32, AffineWordAddressRemap>,
) -> Result<Vec<Vec<Goldilocks>>, ShardBuildError>
where
    Key: Copy + TryInto<u128> + Eq,
    <Key as TryInto<u128>>::Error: std::fmt::Debug,
{
    if lut_instances.is_empty() && mem_instances.is_empty() {
        return Ok(Vec::new());
    }
    if chunk_steps.len() > chunk_size {
        return Err(ShardBuildError::InvalidChunkSize(format!(
            "chunk_steps.len()={} > chunk_size={chunk_size}",
            chunk_steps.len()
        )));
    }

    let bus = build_shared_bus_layout_for_time_columns(chunk_size, lut_instances, mem_instances)?;
    let mut mem_cols = vec![vec![Goldilocks::ZERO; chunk_size]; bus.bus_cols];

    let mut shout_idx_by_id = HashMap::<u32, usize>::new();
    for (idx, (inst, _)) in lut_instances.iter().enumerate() {
        if shout_idx_by_id.insert(inst.table_id, idx).is_some() {
            return Err(ShardBuildError::InvalidInit(format!(
                "duplicate shout table_id={} in step metadata",
                inst.table_id
            )));
        }
    }
    let mut mem_idx_by_id = HashMap::<u32, usize>::new();
    for (idx, (inst, _)) in mem_instances.iter().enumerate() {
        if mem_idx_by_id.insert(inst.mem_id, idx).is_some() {
            return Err(ShardBuildError::InvalidInit(format!(
                "duplicate mem_id={} in step metadata",
                inst.mem_id
            )));
        }
    }

    let mut mem_states: Vec<HashMap<u64, Goldilocks>> = Vec::with_capacity(mem_instances.len());
    for (inst, _) in mem_instances.iter() {
        let mut st = HashMap::new();
        match &inst.init {
            crate::mem_init::MemInit::Zero => {}
            crate::mem_init::MemInit::Sparse(pairs) => {
                for &(addr, val) in pairs {
                    if val != Goldilocks::ZERO {
                        st.insert(addr, val);
                    }
                }
            }
        }
        mem_states.push(st);
    }

    #[derive(Clone)]
    struct ShoutLaneEvent {
        key: u128,
        vals: Vec<Option<(u64, Goldilocks)>>,
    }

    let mut shout_events: Vec<Vec<Option<ShoutLaneEvent>>> = bus
        .shout_cols
        .iter()
        .map(|inst| vec![None; inst.lanes.len()])
        .collect();
    let mut used_shout: Vec<usize> = vec![0; shout_events.len()];
    let mut twist_reads: Vec<Vec<Option<(u64, Goldilocks)>>> = bus
        .twist_cols
        .iter()
        .map(|inst| vec![None; inst.lanes.len()])
        .collect();
    let mut twist_writes: Vec<Vec<Option<(u64, Goldilocks)>>> = bus
        .twist_cols
        .iter()
        .map(|inst| vec![None; inst.lanes.len()])
        .collect();

    for (j, step) in chunk_steps.iter().enumerate() {
        used_shout.fill(0);
        for lanes in shout_events.iter_mut() {
            lanes.fill(None);
        }
        for lanes in twist_reads.iter_mut() {
            lanes.fill(None);
        }
        for lanes in twist_writes.iter_mut() {
            lanes.fill(None);
        }

        for ev in &step.shout_events {
            let table_id = ev.shout_id.0;
            let key = ev
                .key
                .try_into()
                .expect("shared-bus shout lane keys must fit in u128");
            let idx = shout_idx_by_id.get(&table_id).copied().ok_or_else(|| {
                ShardBuildError::MissingTable(format!("trace shout event references unknown table_id={table_id}"))
            })?;
            let lanes = shout_events[idx].len();
            let n_vals = bus
                .shout_cols
                .get(idx)
                .and_then(|inst_cols| inst_cols.lanes.first())
                .map(|lane| lane.vals.len())
                .unwrap_or(1)
                .max(1);
            let slot_idx = used_shout[idx];
            let total_slots = lanes.checked_mul(n_vals).ok_or_else(|| {
                ShardBuildError::InvalidChunkSize(format!(
                    "shout slot overflow for table_id={table_id}: lanes={lanes}, n_vals={n_vals}"
                ))
            })?;
            if slot_idx >= total_slots {
                return Err(ShardBuildError::InvalidChunkSize(format!(
                    "too many shout events for table_id={table_id} at chunk row j={j}: lanes={lanes}, n_vals={n_vals}, total_slots={total_slots}"
                )));
            }
            let lane_idx = slot_idx / n_vals;
            let val_idx = slot_idx % n_vals;
            let lane_entry = shout_events[idx][lane_idx].get_or_insert_with(|| ShoutLaneEvent {
                key,
                vals: vec![None; n_vals],
            });
            if lane_entry.key != key {
                return Err(ShardBuildError::InvalidChunkSize(format!(
                    "mixed shout keys for table_id={table_id} at chunk row j={j}, lane={lane_idx}: first_key={}, new_key={}",
                    lane_entry.key, key
                )));
            }
            if lane_entry.vals[val_idx].is_some() {
                return Err(ShardBuildError::InvalidChunkSize(format!(
                    "duplicate shout value slot for table_id={table_id} at chunk row j={j}, lane={lane_idx}, val_slot={val_idx}"
                )));
            }
            lane_entry.vals[val_idx] = Some((ev.value, Goldilocks::from_u64(ev.value)));
            used_shout[idx] += 1;
        }

        for ev in &step.twist_events {
            let mem_id = ev.twist_id.0;
            let idx = mem_idx_by_id.get(&mem_id).copied().ok_or_else(|| {
                ShardBuildError::MissingLayout(format!("trace twist event references unknown mem_id={mem_id}"))
            })?;
            let mapped_addr = remap_twist_addr(mem_id, ev.addr, mem_addr_remaps)?;
            match ev.kind {
                TwistOpKind::Read => {
                    let lanes = twist_reads.get_mut(idx).ok_or_else(|| {
                        ShardBuildError::MissingLayout(format!("missing read lanes for mem_id={mem_id}"))
                    })?;
                    let lane_idx = if let Some(lane) = ev.lane {
                        let lane_idx = usize::try_from(lane).map_err(|_| {
                            ShardBuildError::InvalidChunkSize(format!(
                                "invalid twist read lane for mem_id={mem_id}: lane={lane}"
                            ))
                        })?;
                        if lane_idx >= lanes.len() {
                            return Err(ShardBuildError::InvalidChunkSize(format!(
                                "twist read lane out of range for mem_id={mem_id}: lane={lane_idx}, lanes={}",
                                lanes.len()
                            )));
                        }
                        if lanes[lane_idx].is_some() {
                            return Err(ShardBuildError::InvalidChunkSize(format!(
                                "multiple twist reads assigned to same lane for mem_id={mem_id}: lane={lane_idx}"
                            )));
                        }
                        lane_idx
                    } else {
                        lanes.iter().position(|x| x.is_none()).ok_or_else(|| {
                            ShardBuildError::InvalidChunkSize(format!(
                                "too many twist reads for mem_id={mem_id} at j={j}: lanes={}",
                                lanes.len()
                            ))
                        })?
                    };
                    lanes[lane_idx] = Some((mapped_addr, Goldilocks::from_u64(ev.value)));
                }
                TwistOpKind::Write => {
                    let lanes = twist_writes.get_mut(idx).ok_or_else(|| {
                        ShardBuildError::MissingLayout(format!("missing write lanes for mem_id={mem_id}"))
                    })?;
                    if lanes.iter().flatten().any(|(addr, _)| *addr == mapped_addr) {
                        return Err(ShardBuildError::InvalidChunkSize(format!(
                            "duplicate twist write addr in one step for mem_id={mem_id}: addr={mapped_addr}",
                        )));
                    }
                    let lane_idx = if let Some(lane) = ev.lane {
                        let lane_idx = usize::try_from(lane).map_err(|_| {
                            ShardBuildError::InvalidChunkSize(format!(
                                "invalid twist write lane for mem_id={mem_id}: lane={lane}"
                            ))
                        })?;
                        if lane_idx >= lanes.len() {
                            return Err(ShardBuildError::InvalidChunkSize(format!(
                                "twist write lane out of range for mem_id={mem_id}: lane={lane_idx}, lanes={}",
                                lanes.len()
                            )));
                        }
                        if lanes[lane_idx].is_some() {
                            return Err(ShardBuildError::InvalidChunkSize(format!(
                                "multiple twist writes assigned to same lane for mem_id={mem_id}: lane={lane_idx}"
                            )));
                        }
                        lane_idx
                    } else {
                        lanes.iter().position(|x| x.is_none()).ok_or_else(|| {
                            ShardBuildError::InvalidChunkSize(format!(
                                "too many twist writes for mem_id={mem_id} at j={j}: lanes={}",
                                lanes.len()
                            ))
                        })?
                    };
                    lanes[lane_idx] = Some((mapped_addr, Goldilocks::from_u64(ev.value)));
                }
            }
        }

        for (inst_idx, (inst, _)) in lut_instances.iter().enumerate() {
            let inst_cols = bus.shout_cols.get(inst_idx).ok_or_else(|| {
                ShardBuildError::CcsError(format!("missing bus shout columns for inst_idx={inst_idx}"))
            })?;
            let d = inst.d;
            let n_side = inst.n_side;
            let ell = inst.ell;

            for (lane_idx, shout_cols) in inst_cols.lanes.iter().enumerate() {
                if let Some(lane_event) = shout_events[inst_idx][lane_idx].as_ref() {
                    let key = lane_event.key;
                    if lane_event.vals.len() != shout_cols.vals.len() {
                        return Err(ShardBuildError::CcsError(format!(
                            "shared-bus shout value slot mismatch for table_id={}: event_slots={} bus_slots={}",
                            inst.table_id,
                            lane_event.vals.len(),
                            shout_cols.vals.len()
                        )));
                    }
                    let (primary_val_u64, _primary_val_field) =
                        lane_event.vals.first().and_then(|v| *v).ok_or_else(|| {
                            ShardBuildError::InvalidChunkSize(format!(
                                "missing primary shout value for table_id={} at chunk row j={j}, lane={lane_idx}",
                                inst.table_id
                            ))
                        })?;
                    match &inst.table_spec {
                        Some(LutTableSpec::RiscvOpcodePacked { opcode, xlen }) => {
                            let (lhs_raw, rhs_raw) = uninterleave_bits(key as u128);
                            let packed_cols = crate::riscv::packed::build_rv_packed_cols::<Goldilocks>(
                                *opcode,
                                lhs_raw,
                                rhs_raw,
                                primary_val_u64,
                                *xlen,
                            )
                            .map_err(|e| ShardBuildError::CcsError(e.to_string()))?;
                            if packed_cols.len() != (shout_cols.addr_bits.end - shout_cols.addr_bits.start) {
                                return Err(ShardBuildError::CcsError(format!(
                                    "packed shout width mismatch for table_id={}: packed_cols={} bus_cols={}",
                                    inst.table_id,
                                    packed_cols.len(),
                                    shout_cols.addr_bits.end - shout_cols.addr_bits.start
                                )));
                            }
                            for (packed_idx, col_id) in shout_cols.addr_bits.clone().enumerate() {
                                mem_cols[col_id][j] = packed_cols[packed_idx];
                            }
                        }
                        Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. }) => {
                            return Err(ShardBuildError::InvalidInit(format!(
                                "RiscvOpcodeEventTablePacked is not supported in shared-bus step builder (table_id={})",
                                inst.table_id
                            )));
                        }
                        _ => {
                            write_addr_bits_into_mem_cols(
                                &mut mem_cols,
                                shout_cols.addr_bits.clone(),
                                j,
                                key,
                                d,
                                n_side,
                                ell,
                            )?;
                        }
                    }
                    mem_cols[shout_cols.has_lookup][j] = Goldilocks::ONE;
                    for (val_slot, &col_id) in shout_cols.vals.iter().enumerate() {
                        let (_slot_val_u64, slot_val_field) = lane_event.vals[val_slot].ok_or_else(|| {
                            ShardBuildError::InvalidChunkSize(format!(
                                "missing shout value for table_id={} at chunk row j={j}, lane={lane_idx}, val_slot={val_slot}",
                                inst.table_id
                            ))
                        })?;
                        mem_cols[col_id][j] = slot_val_field;
                    }
                }
            }
        }

        for (inst_idx, (inst, _)) in mem_instances.iter().enumerate() {
            let inst_cols = bus.twist_cols.get(inst_idx).ok_or_else(|| {
                ShardBuildError::CcsError(format!("missing bus twist columns for inst_idx={inst_idx}"))
            })?;
            let state = mem_states
                .get_mut(inst_idx)
                .ok_or_else(|| ShardBuildError::MissingLayout(format!("missing state for mem_idx={inst_idx}")))?;
            let k_u64 = u64::try_from(inst.k)
                .map_err(|_| ShardBuildError::InvalidInit(format!("mem k overflows u64 for mem_id={}", inst.mem_id)))?;

            let mut writes_to_apply: Vec<(u64, Goldilocks)> = Vec::new();
            for (lane_idx, twist_cols) in inst_cols.lanes.iter().enumerate() {
                let (has_read, ra, rv) = if let Some((addr, val)) = twist_reads[inst_idx][lane_idx] {
                    (Goldilocks::ONE, addr, val)
                } else {
                    (Goldilocks::ZERO, 0u64, Goldilocks::ZERO)
                };
                if has_read == Goldilocks::ONE {
                    if ra >= k_u64 {
                        return Err(ShardBuildError::InvalidChunkSize(format!(
                            "twist read addr out of range for mem_id={}: addr={} >= k={}",
                            inst.mem_id, ra, inst.k
                        )));
                    }
                    write_addr_bits_into_mem_cols(
                        &mut mem_cols,
                        twist_cols.ra_bits.clone(),
                        j,
                        u128::from(ra),
                        inst.d,
                        inst.n_side,
                        inst.ell,
                    )?;
                    mem_cols[twist_cols.rv][j] = rv;
                }

                let (has_write, wa, wv) = if let Some((addr, val)) = twist_writes[inst_idx][lane_idx] {
                    (Goldilocks::ONE, addr, val)
                } else {
                    (Goldilocks::ZERO, 0u64, Goldilocks::ZERO)
                };
                let mut inc = Goldilocks::ZERO;
                if has_write == Goldilocks::ONE {
                    if wa >= k_u64 {
                        return Err(ShardBuildError::InvalidChunkSize(format!(
                            "twist write addr out of range for mem_id={}: addr={} >= k={}",
                            inst.mem_id, wa, inst.k
                        )));
                    }
                    write_addr_bits_into_mem_cols(
                        &mut mem_cols,
                        twist_cols.wa_bits.clone(),
                        j,
                        u128::from(wa),
                        inst.d,
                        inst.n_side,
                        inst.ell,
                    )?;
                    mem_cols[twist_cols.wv][j] = wv;
                    let old = state.get(&wa).copied().unwrap_or(Goldilocks::ZERO);
                    inc = wv - old;
                    writes_to_apply.push((wa, wv));
                }

                mem_cols[twist_cols.has_read][j] = has_read;
                mem_cols[twist_cols.has_write][j] = has_write;
                mem_cols[twist_cols.inc][j] = inc;
            }

            for (wa, wv) in writes_to_apply {
                if wv == Goldilocks::ZERO {
                    state.remove(&wa);
                } else {
                    state.insert(wa, wv);
                }
            }
        }
    }

    Ok(mem_cols)
}

fn build_time_columns_for_step<Cmt, Key>(
    _mcs: &(CcsClaim<Cmt, Goldilocks>, CcsWitness<Goldilocks>),
    chunk_size: usize,
    chunk_len: usize,
    chunk_steps: &[StepTrace<u64, u64, Key>],
    lut_instances: &[(LutInstance<Cmt, Goldilocks>, LutWitness<Goldilocks>)],
    mem_instances: &[(MemInstance<Cmt, Goldilocks>, MemWitness<Goldilocks>)],
    mem_addr_remaps: &HashMap<u32, AffineWordAddressRemap>,
) -> Result<TimeColumns<Goldilocks>, ShardBuildError>
where
    Key: Copy + TryInto<u128> + Eq,
    <Key as TryInto<u128>>::Error: std::fmt::Debug,
{
    fn cpu_cols_from_chunk_steps<Key>(
        chunk_steps: &[StepTrace<u64, u64, Key>],
        chunk_size: usize,
        expected_cpu_cols: usize,
        strict_rv32: bool,
    ) -> Result<Vec<Vec<Goldilocks>>, ShardBuildError>
    where
        Key: Copy + TryInto<u128>,
        <Key as TryInto<u128>>::Error: std::fmt::Debug,
    {
        if chunk_steps.len() > chunk_size {
            return Err(ShardBuildError::InvalidChunkSize(format!(
                "chunk_steps.len()={} > chunk_size={chunk_size}",
                chunk_steps.len()
            )));
        }
        if expected_cpu_cols == 0 {
            return Ok(Vec::new());
        }
        let machine_xlen = infer_riscv_trace_machine_xlen(expected_cpu_cols).unwrap_or(32);

        let mut rows = Vec::with_capacity(chunk_size);
        for step in chunk_steps {
            match RiscvExecRow::from_step_with_xlen(step, machine_xlen) {
                Ok(row) => rows.push(row),
                Err(_e) if !strict_rv32 => {
                    // Non-RV32 harness callers may use opaque instruction words.
                    // For those callers, we only need a shape-compatible time-column payload.
                    return Ok(vec![vec![Goldilocks::ZERO; chunk_size]; expected_cpu_cols]);
                }
                Err(e) => return Err(ShardBuildError::CcsError(format!("step->exec conversion failed: {e}"))),
            }
        }
        if chunk_steps.is_empty() {
            return Ok(vec![vec![Goldilocks::ZERO; chunk_size]; expected_cpu_cols]);
        }
        let mut cycle = rows
            .last()
            .ok_or_else(|| ShardBuildError::CcsError("empty rows after step conversion".into()))?
            .cycle;
        let pad_pc = rows.last().expect("rows non-empty").pc_after;
        let pad_halted = rows.last().expect("rows non-empty").halted;
        while rows.len() < chunk_size {
            cycle = cycle
                .checked_add(1)
                .ok_or_else(|| ShardBuildError::CcsError("cycle overflow while padding chunk rows".into()))?;
            rows.push(RiscvExecRow::inactive(cycle, pad_pc, pad_halted));
        }
        let exec = RiscvExecTable { rows };
        if expected_cpu_cols == rv32_trace_cpu_cols() {
            let trace_layout = Rv32TraceLayout::new();
            let wit = Rv32TraceWitness::from_exec_table(&trace_layout, &exec)
                .map_err(|e| ShardBuildError::CcsError(format!("RV32 time trace witness build failed: {e}")))?;
            Ok(wit.cols)
        } else if expected_cpu_cols == rv64_trace_cpu_cols() {
            let trace_layout = Rv64TraceLayout::new();
            let wit = Rv64TraceWitness::from_exec_table(&trace_layout, &exec)
                .map_err(|e| ShardBuildError::CcsError(format!("RV64 time trace witness build failed: {e}")))?;
            Ok(wit.cols)
        } else {
            Err(ShardBuildError::CcsError(format!(
                "unsupported CPU time-column profile width {expected_cpu_cols} (expected {} or {})",
                rv32_trace_cpu_cols(),
                rv64_trace_cpu_cols()
            )))
        }
    }
    let mem_cols =
        build_mem_time_columns_for_step(chunk_steps, chunk_size, lut_instances, mem_instances, mem_addr_remaps)?;
    let logical_m = _mcs
        .0
        .m_in
        .checked_add(_mcs.1.w.len())
        .ok_or_else(|| ShardBuildError::CcsError("time-column builder logical m overflow".into()))?;
    let payload_cols = logical_m.checked_sub(_mcs.0.m_in).ok_or_else(|| {
        ShardBuildError::CcsError(format!(
            "time-column builder logical width underflow: logical_m={}, m_in={}",
            logical_m, _mcs.0.m_in
        ))
    })?;
    let expected_cpu_cols = if payload_cols == 0 {
        0
    } else {
        payload_cols.checked_sub(mem_cols.len()).ok_or_else(|| {
            ShardBuildError::CcsError(format!(
                "time-column builder width underflow: logical_m={}, m_in={}, mem_cols={}",
                logical_m,
                _mcs.0.m_in,
                mem_cols.len()
            ))
        })?
    };
    let cpu_cols = cpu_cols_from_chunk_steps(
        chunk_steps,
        chunk_size,
        expected_cpu_cols,
        /*strict_rv32=*/ _mcs.0.m_in == 5,
    )?;

    if chunk_len > chunk_size {
        return Err(ShardBuildError::InvalidChunkSize(format!(
            "chunk_len={} exceeds chunk_size={chunk_size}",
            chunk_len
        )));
    }
    let mut active_col = vec![Goldilocks::ZERO; chunk_size];
    for v in active_col.iter_mut().take(chunk_len) {
        *v = Goldilocks::ONE;
    }

    let mut out = TimeColumns {
        t: chunk_size,
        cpu_cols,
        mem_cols,
        // This is the chunk pad mask consumed by time-domain gating checks.
        active_col,
        col_ids: Vec::new(),
    };

    out.col_ids = (0..(out.cpu_cols.len() + out.mem_cols.len())).collect();
    Ok(out)
}

/// Build shard witness bundles for **shared CPU bus** mode.
///
/// In this mode Twist/Shout access-row columns are expected to live in the CPU witness `z`
/// committed by `mcs_inst.c`, and the memory sidecar will consume openings derived from the CPU
/// commitment (no independent mem/lut commitments).
///
/// This builder therefore emits:
/// - `MemInstance/LutInstance` **metadata only** (`comms = []`)
/// - empty `MemWitness/LutWitness` (`mats = []`)
pub fn build_shard_witness_shared_cpu_bus<V, Cmt, K, A, Tw, Sh>(
    vm: V,
    twist: Tw,
    shout: Sh,
    max_steps: usize,
    chunk_size: usize,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    lut_tables: &HashMap<u32, LutTable<Goldilocks>>,
    lut_table_specs: &HashMap<u32, LutTableSpec>,
    lut_lanes: &HashMap<u32, usize>,
    initial_mem: &HashMap<(u32, u64), Goldilocks>,
    cpu_arith: &A,
) -> Result<Vec<StepWitnessBundle<Cmt, Goldilocks, K>>, ShardBuildError>
where
    V: neo_vm_trace::VmCpu<u64, u64, u128>,
    Tw: neo_vm_trace::Twist<u64, u64>,
    Sh: neo_vm_trace::Shout<u128, u64>,
    A: CpuArithmetization<Goldilocks, Cmt, u128>,
{
    bundles_only(build_shard_witness_shared_cpu_bus_with_aux(
        vm,
        twist,
        shout,
        max_steps,
        chunk_size,
        mem_layouts,
        lut_tables,
        lut_table_specs,
        lut_lanes,
        initial_mem,
        cpu_arith,
    ))
}

/// Build shard witness bundles for **shared CPU bus** mode from an already-executed VM trace.
///
/// This is equivalent to `build_shard_witness_shared_cpu_bus(...)`, but avoids re-running the VM
/// when the caller already has a `VmTrace` available.
pub fn build_shard_witness_shared_cpu_bus_from_trace<Cmt, K, A, Key>(
    trace: &VmTrace<u64, u64, Key>,
    max_steps: usize,
    chunk_size: usize,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    lut_tables: &HashMap<u32, LutTable<Goldilocks>>,
    lut_table_specs: &HashMap<u32, LutTableSpec>,
    lut_lanes: &HashMap<u32, usize>,
    initial_mem: &HashMap<(u32, u64), Goldilocks>,
    cpu_arith: &A,
) -> Result<Vec<StepWitnessBundle<Cmt, Goldilocks, K>>, ShardBuildError>
where
    A: CpuArithmetization<Goldilocks, Cmt, Key>,
    Key: Copy + TryInto<u128> + Eq,
    <Key as TryInto<u128>>::Error: std::fmt::Debug,
{
    bundles_only(build_shard_witness_shared_cpu_bus_from_trace_with_aux(
        trace,
        max_steps,
        chunk_size,
        mem_layouts,
        lut_tables,
        lut_table_specs,
        lut_lanes,
        initial_mem,
        cpu_arith,
    ))
}

fn remap_twist_addr(
    mem_id: u32,
    addr: u64,
    mem_addr_remaps: &HashMap<u32, AffineWordAddressRemap>,
) -> Result<u64, ShardBuildError> {
    let Some(remap) = mem_addr_remaps.get(&mem_id) else {
        return Ok(addr);
    };
    remap.remap_guest_addr(addr).map_err(|e| {
        ShardBuildError::InvalidInit(format!(
            "twist address remap failed for mem_id={mem_id}, guest_addr={addr:#x}: {e}"
        ))
    })
}

pub fn build_shard_witness_shared_cpu_bus_from_trace_with_aux_and_mem_remaps<Cmt, K, A, Key>(
    trace: &VmTrace<u64, u64, Key>,
    max_steps: usize,
    chunk_size: usize,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    lut_tables: &HashMap<u32, LutTable<Goldilocks>>,
    lut_table_specs: &HashMap<u32, LutTableSpec>,
    lut_lanes: &HashMap<u32, usize>,
    initial_mem: &HashMap<(u32, u64), Goldilocks>,
    mem_addr_remaps: &HashMap<u32, AffineWordAddressRemap>,
    cpu_arith: &A,
) -> Result<(Vec<StepWitnessBundle<Cmt, Goldilocks, K>>, ShardWitnessAux), ShardBuildError>
where
    A: CpuArithmetization<Goldilocks, Cmt, Key>,
    Key: Copy + TryInto<u128> + Eq,
    <Key as TryInto<u128>>::Error: std::fmt::Debug,
{
    build_shard_witness_shared_cpu_bus_from_trace_with_aux_impl(
        trace,
        max_steps,
        chunk_size,
        mem_layouts,
        lut_tables,
        lut_table_specs,
        lut_lanes,
        initial_mem,
        mem_addr_remaps,
        cpu_arith,
    )
}

/// Like `build_shard_witness_shared_cpu_bus_from_trace`, but also returns auxiliary outputs useful
/// for higher-level APIs (e.g. output binding that needs terminal Twist memory states).
pub fn build_shard_witness_shared_cpu_bus_from_trace_with_aux<Cmt, K, A, Key>(
    trace: &VmTrace<u64, u64, Key>,
    max_steps: usize,
    chunk_size: usize,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    lut_tables: &HashMap<u32, LutTable<Goldilocks>>,
    lut_table_specs: &HashMap<u32, LutTableSpec>,
    lut_lanes: &HashMap<u32, usize>,
    initial_mem: &HashMap<(u32, u64), Goldilocks>,
    cpu_arith: &A,
) -> Result<(Vec<StepWitnessBundle<Cmt, Goldilocks, K>>, ShardWitnessAux), ShardBuildError>
where
    A: CpuArithmetization<Goldilocks, Cmt, Key>,
    Key: Copy + TryInto<u128> + Eq,
    <Key as TryInto<u128>>::Error: std::fmt::Debug,
{
    build_shard_witness_shared_cpu_bus_from_trace_with_aux_impl(
        trace,
        max_steps,
        chunk_size,
        mem_layouts,
        lut_tables,
        lut_table_specs,
        lut_lanes,
        initial_mem,
        &HashMap::new(),
        cpu_arith,
    )
}

fn build_shard_witness_shared_cpu_bus_from_trace_with_aux_impl<Cmt, K, A, Key>(
    trace: &VmTrace<u64, u64, Key>,
    max_steps: usize,
    chunk_size: usize,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    lut_tables: &HashMap<u32, LutTable<Goldilocks>>,
    lut_table_specs: &HashMap<u32, LutTableSpec>,
    lut_lanes: &HashMap<u32, usize>,
    initial_mem: &HashMap<(u32, u64), Goldilocks>,
    mem_addr_remaps: &HashMap<u32, AffineWordAddressRemap>,
    cpu_arith: &A,
) -> Result<(Vec<StepWitnessBundle<Cmt, Goldilocks, K>>, ShardWitnessAux), ShardBuildError>
where
    A: CpuArithmetization<Goldilocks, Cmt, Key>,
    Key: Copy + TryInto<u128> + Eq,
    <Key as TryInto<u128>>::Error: std::fmt::Debug,
{
    let chunk_size = canonical_chunk_size_pow2(chunk_size)?;
    if trace.steps.len() > max_steps {
        return Err(ShardBuildError::InvalidChunkSize(format!(
            "trace length {} exceeds max_steps {}",
            trace.steps.len(),
            max_steps
        )));
    }

    // Shared-bus mode does not support "silent dropping" of trace events: if the trace contains
    // Twist/Shout events, the corresponding instance metadata must be provided so the prover
    // actually proves those semantics.
    for (j, step) in trace.steps.iter().enumerate() {
        for ev in &step.twist_events {
            let mem_id = ev.twist_id.0;
            if !mem_layouts.contains_key(&mem_id) {
                return Err(ShardBuildError::MissingLayout(format!(
                    "trace contains twist events for twist_id={mem_id} at step {j}, but mem_layouts has no entry"
                )));
            }
        }
        for ev in &step.shout_events {
            let table_id = ev.shout_id.0;
            if !lut_tables.contains_key(&table_id) && !lut_table_specs.contains_key(&table_id) {
                return Err(ShardBuildError::MissingTable(format!(
                    "trace contains shout events for table_id={table_id} at step {j}, but neither lut_tables nor lut_table_specs has an entry"
                )));
            }
        }
    }
    for ((mem_id, _addr), _val) in initial_mem.iter() {
        if !mem_layouts.contains_key(mem_id) {
            return Err(ShardBuildError::MissingLayout(format!(
                "initial_mem contains entries for twist_id={mem_id}, but mem_layouts has no entry"
            )));
        }
    }

    let original_len = trace.steps.len();
    let did_halt = trace.did_halt();
    let steps_len = trace.steps.len();
    let chunks_len = steps_len.div_ceil(chunk_size);

    // Deterministic ordering (required for the shared-bus column schema).
    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();
    let mut table_ids: Vec<u32> = lut_tables
        .keys()
        .copied()
        .chain(lut_table_specs.keys().copied())
        .collect();
    table_ids.sort_unstable();
    table_ids.dedup();
    let precomputed_shout_table_digests: HashMap<u32, [u8; 32]> = table_ids
        .iter()
        .copied()
        .map(|table_id| {
            let digest = lut_tables
                .get(&table_id)
                .map(|table| memory_public_digest_fields(b"shout/table", &table.content))
                .unwrap_or_else(|| memory_public_digest_fields(b"shout/table", &[]));
            (table_id, digest)
        })
        .collect();
    let zero_mem_init_digest = memory_public_digest_fields(b"twist/init/zero", &[]);

    // 3) CPU arithmetization chunks.
    let mcss = cpu_arith
        .build_ccs_chunks(trace, chunk_size)
        .map_err(|e| ShardBuildError::CcsError(e.to_string()))?;
    if mcss.len() != chunks_len {
        return Err(ShardBuildError::CcsError(format!(
            "cpu arithmetization returned {} chunks, expected {} (steps={}, chunk_size={})",
            mcss.len(),
            chunks_len,
            steps_len,
            chunk_size
        )));
    }

    // 4) Track sparse memory state across chunks to compute per-chunk MemInit (rollover).
    let mut mem_states: HashMap<u32, HashMap<u64, Goldilocks>> = HashMap::new();
    for mem_id in mem_ids.iter().copied() {
        let layout = mem_layouts
            .get(&mem_id)
            .ok_or_else(|| ShardBuildError::MissingLayout(format!("missing PlainMemLayout for twist_id {}", mem_id)))?;
        let mut state = HashMap::<u64, Goldilocks>::new();
        for ((init_mem_id, addr), &val) in initial_mem.iter() {
            if *init_mem_id != mem_id || val == Goldilocks::ZERO {
                continue;
            }
            let addr_usize = usize::try_from(*addr).map_err(|_| {
                ShardBuildError::InvalidInit(format!(
                    "initial_mem address doesn't fit usize for twist_id {}: addr={addr}",
                    mem_id
                ))
            })?;
            if addr_usize >= layout.k {
                return Err(ShardBuildError::InvalidInit(format!(
                    "initial_mem address out of range for twist_id {}: addr={} >= k={}",
                    mem_id, addr, layout.k
                )));
            }
            if state.insert(*addr, val).is_some() {
                return Err(ShardBuildError::InvalidInit(format!(
                    "initial_mem contains duplicate address {} for twist_id {}",
                    addr, mem_id
                )));
            }
        }
        mem_states.insert(mem_id, state);
    }

    let mut step_bundles = Vec::with_capacity(chunks_len);
    let mut chunk_start = 0usize;
    for (chunk_idx, mcs) in mcss.into_iter().enumerate() {
        let chunk_end = (chunk_start + chunk_size).min(steps_len);
        let chunk_len = chunk_end.saturating_sub(chunk_start);
        if chunk_len == 0 {
            return Err(ShardBuildError::CcsError(format!(
                "internal error: empty chunk at chunk_idx {} (start={}, end={}, steps={})",
                chunk_idx, chunk_start, chunk_end, steps_len
            )));
        }

        // Memory instances (metadata-only).
        let mut mem_instances = Vec::new();
        for mem_id in mem_ids.iter().copied() {
            let layout = mem_layouts.get(&mem_id).ok_or_else(|| {
                ShardBuildError::MissingLayout(format!("missing PlainMemLayout for twist_id {}", mem_id))
            })?;
            let state = mem_states
                .get_mut(&mem_id)
                .ok_or_else(|| ShardBuildError::MissingLayout(format!("missing state for twist_id {}", mem_id)))?;
            let init = mem_init_from_state_map(mem_id, layout.k, state)
                .map_err(|e| ShardBuildError::InvalidInit(e.to_string()))?;
            let init_digest = match &init {
                crate::mem_init::MemInit::Zero => Some(zero_mem_init_digest),
                crate::mem_init::MemInit::Sparse(pairs) => {
                    let mut fs = Vec::with_capacity(2 * pairs.len());
                    for (addr, val) in pairs.iter() {
                        fs.push(Goldilocks::from_u64(*addr));
                        fs.push(*val);
                    }
                    Some(memory_public_digest_fields(b"twist/init/sparse", &fs))
                }
            };
            let ell = ell_from_pow2_n_side(layout.n_side)?;

            let inst = MemInstance::<Cmt, Goldilocks> {
                mem_id,
                comms: Vec::new(),
                k: layout.k,
                d: layout.d,
                n_side: layout.n_side,
                steps: chunk_size,
                lanes: layout.lanes.max(1),
                ell,
                init,
                init_digest,
                guest_addr_remap: mem_addr_remaps.get(&mem_id).cloned(),
            };
            let wit = MemWitness { mats: Vec::new() };
            mem_instances.push((inst, wit));

            // Advance state across this chunk for the next chunk's init.
            for t in chunk_start..chunk_end {
                let step = trace
                    .steps
                    .get(t)
                    .ok_or_else(|| ShardBuildError::VmError(format!("missing trace step t={t}")))?;
                for ev in &step.twist_events {
                    if ev.twist_id.0 != mem_id || ev.kind != TwistOpKind::Write {
                        continue;
                    }
                    let addr = remap_twist_addr(mem_id, ev.addr, mem_addr_remaps)?;
                    let Ok(addr_usize) = usize::try_from(addr) else {
                        continue;
                    };
                    if addr_usize >= layout.k {
                        continue;
                    }
                    let new_val = Goldilocks::from_u64(ev.value);
                    if new_val == Goldilocks::ZERO {
                        state.remove(&addr);
                    } else {
                        state.insert(addr, new_val);
                    }
                }
            }
        }

        // Lookup instances (metadata-only).
        let mut lut_instances = Vec::new();
        for table_id in table_ids.iter().copied() {
            if lut_tables.contains_key(&table_id) && lut_table_specs.contains_key(&table_id) {
                return Err(ShardBuildError::InvalidInit(format!(
                    "shout table_id={table_id} appears in both lut_tables (explicit) and lut_table_specs (implicit); pick exactly one to avoid schema ambiguity"
                )));
            }
            let table_spec = lut_table_specs.get(&table_id).cloned();

            let (k, d, n_side, ell, table) = if let Some(spec) = &table_spec {
                // Derive addressing parameters from the implicit table spec.
                match spec {
                    LutTableSpec::RiscvOpcode { xlen, .. } => {
                        let d = xlen.checked_mul(2).ok_or_else(|| {
                            ShardBuildError::InvalidInit("2*xlen overflow for RISC-V shout table".into())
                        })?;
                        let n_side = 2usize;
                        let ell = 1usize;
                        (0usize, d, n_side, ell, Vec::new())
                    }
                    LutTableSpec::RiscvOpcodePacked { opcode, xlen } => {
                        let d = crate::riscv::packed::rv_packed_d(*opcode, *xlen)
                            .map_err(|e| ShardBuildError::InvalidInit(format!("invalid packed opcode spec: {e}")))?;
                        let n_side = 2usize;
                        let ell = 1usize;
                        (0usize, d, n_side, ell, Vec::new())
                    }
                    LutTableSpec::RiscvOpcodeEventTablePacked { .. } => {
                        return Err(ShardBuildError::InvalidInit(
                            "RiscvOpcodeEventTablePacked is not supported in the chunked builder path".into(),
                        ));
                    }
                    LutTableSpec::IdentityU32 => (0usize, 32usize, 2usize, 1usize, Vec::new()),
                }
            } else {
                let table = lut_tables.get(&table_id).ok_or_else(|| {
                    ShardBuildError::MissingTable(format!("missing LutTable for shout_id {}", table_id))
                })?;
                let ell = ell_from_pow2_n_side(table.n_side)?;
                (table.k, table.d, table.n_side, ell, table.content.clone())
            };

            let lanes = lut_lanes.get(&table_id).copied().unwrap_or(1).max(1);
            let inst = LutInstance::<Cmt, Goldilocks> {
                table_id,
                comms: Vec::new(),
                k,
                d,
                n_side,
                steps: chunk_size,
                lanes,
                ell,
                table_spec,
                table,
                table_digest: precomputed_shout_table_digests.get(&table_id).copied(),
                addr_group: cpu_arith.shout_addr_groups().get(&table_id).copied(),
                selector_group: cpu_arith.shout_selector_groups().get(&table_id).copied(),
            };
            let wit = LutWitness { mats: Vec::new() };
            lut_instances.push((inst, wit));
        }

        let chunk_steps = &trace.steps[chunk_start..chunk_end];
        let time_columns = build_time_columns_for_step(
            &mcs,
            chunk_size,
            chunk_len,
            chunk_steps,
            &lut_instances,
            &mem_instances,
            mem_addr_remaps,
        )?;
        step_bundles.push(StepWitnessBundle {
            mcs,
            lut_instances,
            mem_instances,
            time_columns,
            _phantom: PhantomData,
        });

        chunk_start = chunk_end;
    }

    let aux = ShardWitnessAux {
        original_len,
        did_halt,
        max_steps,
        chunk_size,
        mem_ids,
        final_mem_states: mem_states,
    };
    Ok((step_bundles, aux))
}

/// Like `build_shard_witness_shared_cpu_bus`, but also returns auxiliary outputs useful for
/// higher-level APIs (e.g. output binding that needs the terminal Twist memory state).
pub fn build_shard_witness_shared_cpu_bus_with_aux<V, Cmt, K, A, Tw, Sh>(
    vm: V,
    twist: Tw,
    shout: Sh,
    max_steps: usize,
    chunk_size: usize,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    lut_tables: &HashMap<u32, LutTable<Goldilocks>>,
    lut_table_specs: &HashMap<u32, LutTableSpec>,
    lut_lanes: &HashMap<u32, usize>,
    initial_mem: &HashMap<(u32, u64), Goldilocks>,
    cpu_arith: &A,
) -> Result<(Vec<StepWitnessBundle<Cmt, Goldilocks, K>>, ShardWitnessAux), ShardBuildError>
where
    V: neo_vm_trace::VmCpu<u64, u64, u128>,
    Tw: neo_vm_trace::Twist<u64, u64>,
    Sh: neo_vm_trace::Shout<u128, u64>,
    A: CpuArithmetization<Goldilocks, Cmt, u128>,
{
    let chunk_size = canonical_chunk_size_pow2(chunk_size)?;

    // 1) Run VM and collect the executed trace for this shard (up to `max_steps`).
    //
    // NOTE: We intentionally do **not** pad out to `max_steps` here. Padding is handled at the
    // per-chunk level by the CPU arithmetization via `is_active`, so the last chunk may be partial.
    //
    // This keeps the proof size proportional to the executed trace length instead of the caller's
    // safety bound.
    let trace = neo_vm_trace::trace_program(vm, twist, shout, max_steps)
        .map_err(|e| ShardBuildError::VmError(e.to_string()))?;
    build_shard_witness_shared_cpu_bus_from_trace_with_aux(
        &trace,
        max_steps,
        chunk_size,
        mem_layouts,
        lut_tables,
        lut_table_specs,
        lut_lanes,
        initial_mem,
        cpu_arith,
    )
}
