use crate::mem_init::mem_init_from_state_map;
use crate::plain::{build_plain_mem_traces, LutTable, PlainMemLayout};
use crate::witness::{LutInstance, LutTableSpec, LutWitness, MemInstance, MemWitness, StepWitnessBundle};
use neo_vm_trace::VmTrace;
use neo_vm_trace::StepTrace;

use neo_ccs::relations::{McsInstance, McsWitness};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use std::collections::HashMap;
use std::marker::PhantomData;

// Placeholder for CPU arithmetization interface
pub trait CpuArithmetization<F, Cmt> {
    type Error: std::fmt::Debug + std::fmt::Display;

    fn build_ccs_chunks(
        &self,
        trace: &VmTrace<u64, u64>,
        chunk_size: usize,
    ) -> Result<Vec<(McsInstance<Cmt, F>, McsWitness<F>)>, Self::Error>;

    fn build_ccs_steps(
        &self,
        trace: &VmTrace<u64, u64>,
    ) -> Result<Vec<(McsInstance<Cmt, F>, McsWitness<F>)>, Self::Error> {
        self.build_ccs_chunks(trace, 1)
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

fn ell_from_pow2_n_side(n_side: usize) -> Result<usize, ShardBuildError> {
    if n_side == 0 || !n_side.is_power_of_two() {
        return Err(ShardBuildError::InvalidInit(format!(
            "n_side must be a power of two under bit addressing, got {n_side}"
        )));
    }
    Ok(n_side.trailing_zeros() as usize)
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
    initial_mem: &HashMap<(u32, u64), Goldilocks>,
    cpu_arith: &A,
) -> Result<Vec<StepWitnessBundle<Cmt, Goldilocks, K>>, ShardBuildError>
where
    V: neo_vm_trace::VmCpu<u64, u64>,
    Tw: neo_vm_trace::Twist<u64, u64>,
    Sh: neo_vm_trace::Shout<u64>,
    A: CpuArithmetization<Goldilocks, Cmt>,
{
    if chunk_size == 0 {
        return Err(ShardBuildError::InvalidChunkSize("chunk_size must be >= 1".into()));
    }

    // 1) Run VM and collect full trace for this shard (then pad to fixed length).
    let mut trace = neo_vm_trace::trace_program(vm, twist, shout, max_steps)
        .map_err(|e| ShardBuildError::VmError(e.to_string()))?;
    let original_len = trace.steps.len();
    debug_assert!(
        original_len <= max_steps,
        "trace_program must not exceed max_steps (got {}, max_steps={})",
        original_len,
        max_steps
    );

    // L1-style fixed-row execution: pad to exactly `max_steps` by repeating the final architectural
    // state with no Twist/Shout events. (Per-row padding is expressed via `is_active` in the CPU CCS.)
    if original_len < max_steps {
        if let Some(last) = trace.steps.last() {
            let pc = last.pc_after;
            let regs = last.regs_after.clone();
            let start = original_len;
            trace.steps.reserve(max_steps - start);
            for t in start..max_steps {
                trace.steps.push(StepTrace {
                    cycle: t as u64,
                    pc_before: pc,
                    pc_after: pc,
                    opcode: 0,
                    regs_before: regs.clone(),
                    regs_after: regs.clone(),
                    twist_events: Vec::new(),
                    shout_events: Vec::new(),
                    halted: true,
                });
            }
        }
    }
    if original_len > 0 && original_len < max_steps {
        debug_assert_eq!(
            trace.steps.len(),
            max_steps,
            "internal error: expected builder padding to reach max_steps"
        );
        for (t, step) in trace.steps.iter().enumerate().skip(original_len) {
            debug_assert!(
                step.twist_events.is_empty(),
                "padded step {t} must have no twist events"
            );
            debug_assert!(
                step.shout_events.is_empty(),
                "padded step {t} must have no shout events"
            );
            debug_assert!(step.halted, "padded step {t} must be halted");
            debug_assert_eq!(
                step.pc_before, step.pc_after,
                "padded step {t} must keep pc constant"
            );
            debug_assert_eq!(
                step.regs_before, step.regs_after,
                "padded step {t} must keep regs constant"
            );
        }
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

    let steps_len = trace.steps.len();
    let chunks_len = steps_len.div_ceil(chunk_size);

    // 2) Build plain traces over [0..T).
    let plain_mem = build_plain_mem_traces::<Goldilocks>(&trace, mem_layouts, initial_mem);

    // Deterministic ordering (required for the shared-bus column schema).
    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();
    let mut table_ids: Vec<u32> = lut_tables.keys().copied().chain(lut_table_specs.keys().copied()).collect();
    table_ids.sort_unstable();
    table_ids.dedup();

    // 3) CPU arithmetization chunks.
    let mcss = cpu_arith
        .build_ccs_chunks(&trace, chunk_size)
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
            let ell = ell_from_pow2_n_side(layout.n_side)?;

            let inst = MemInstance::<Cmt, Goldilocks> {
                comms: Vec::new(),
                k: layout.k,
                d: layout.d,
                n_side: layout.n_side,
                steps: chunk_size,
                ell,
                init,
                _phantom: PhantomData,
            };
            let wit = MemWitness { mats: Vec::new() };
            mem_instances.push((inst, wit));

            // Advance state across this chunk for the next chunk's init.
            let plain = plain_mem.get(&mem_id).ok_or_else(|| {
                ShardBuildError::MissingLayout(format!("missing PlainMemTrace for twist_id {}", mem_id))
            })?;
            for t in chunk_start..chunk_end {
                if plain.has_write[t] != Goldilocks::ONE {
                    continue;
                }
                let addr = plain.write_addr[t];
                let Ok(addr_usize) = usize::try_from(addr) else {
                    continue;
                };
                if addr_usize >= layout.k {
                    continue;
                }
                let new_val = plain.write_val[t];
                if new_val == Goldilocks::ZERO {
                    state.remove(&addr);
                } else {
                    state.insert(addr, new_val);
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
                        let d = xlen
                            .checked_mul(2)
                            .ok_or_else(|| ShardBuildError::InvalidInit("2*xlen overflow for RISC-V shout table".into()))?;
                        let n_side = 2usize;
                        let ell = 1usize;
                        (0usize, d, n_side, ell, Vec::new())
                    }
                }
            } else {
                let table = lut_tables
                    .get(&table_id)
                    .ok_or_else(|| ShardBuildError::MissingTable(format!("missing LutTable for shout_id {}", table_id)))?;
                let ell = ell_from_pow2_n_side(table.n_side)?;
                (table.k, table.d, table.n_side, ell, table.content.clone())
            };

            let inst = LutInstance::<Cmt, Goldilocks> {
                comms: Vec::new(),
                k,
                d,
                n_side,
                steps: chunk_size,
                ell,
                table_spec,
                table,
                _phantom: PhantomData,
            };
            let wit = LutWitness { mats: Vec::new() };
            lut_instances.push((inst, wit));
        }

        step_bundles.push(StepWitnessBundle {
            mcs,
            lut_instances,
            mem_instances,
            _phantom: PhantomData,
        });

        chunk_start = chunk_end;
    }

    Ok(step_bundles)
}
