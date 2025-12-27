use crate::mem_init::MemInit;
use crate::plain::{build_plain_mem_traces, LutTable, PlainMemLayout};
use crate::witness::{LutInstance, LutWitness, MemInstance, MemWitness, StepWitnessBundle};
use neo_vm_trace::VmTrace;

use neo_ccs::relations::{McsInstance, McsWitness};
use p3_goldilocks::Goldilocks;
use p3_field::PrimeCharacteristicRing;
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
    if chunk_size != 1 {
        return Err(ShardBuildError::InvalidChunkSize(format!(
            "shared_cpu_bus currently supports chunk_size=1 (got {chunk_size})"
        )));
    }

    // 1) Run VM and collect full trace for this shard.
    let trace = neo_vm_trace::trace_program(vm, twist, shout, max_steps)
        .map_err(|e| ShardBuildError::VmError(e.to_string()))?;

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
            if !lut_tables.contains_key(&table_id) {
                return Err(ShardBuildError::MissingTable(format!(
                    "trace contains shout events for table_id={table_id} at step {j}, but lut_tables has no entry"
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
    let mut table_ids: Vec<u32> = lut_tables.keys().copied().collect();
    table_ids.sort_unstable();

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
        let layout = mem_layouts.get(&mem_id).ok_or_else(|| {
            ShardBuildError::MissingLayout(format!("missing PlainMemLayout for twist_id {}", mem_id))
        })?;
        let mut state = HashMap::<u64, Goldilocks>::new();
        for ((init_mem_id, addr), &val) in initial_mem.iter() {
            if *init_mem_id != mem_id || val == Goldilocks::ZERO {
                continue;
            }
            if (*addr as usize) >= layout.k {
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

    // Local helper: MemInit from sparse state.
    fn mem_init_from_state(
        mem_id: u32,
        k: usize,
        state: &HashMap<u64, Goldilocks>,
    ) -> Result<MemInit<Goldilocks>, ShardBuildError> {
        if state.is_empty() {
            return Ok(MemInit::Zero);
        }
        let mut pairs: Vec<(u64, Goldilocks)> = state
            .iter()
            .filter_map(|(&addr, &val)| (val != Goldilocks::ZERO).then_some((addr, val)))
            .collect();
        pairs.sort_by_key(|(addr, _)| *addr);
        if let Some((addr, _)) = pairs.last() {
            if (*addr as usize) >= k {
                return Err(ShardBuildError::InvalidInit(format!(
                    "internal error: state address out of range for twist_id {}: addr={} >= k={}",
                    mem_id, addr, k
                )));
            }
        }
        Ok(MemInit::Sparse(pairs))
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
            let init = mem_init_from_state(mem_id, layout.k, state)?;
            let ell = ell_from_pow2_n_side(layout.n_side)?;

            let inst = MemInstance::<Cmt, Goldilocks> {
                comms: Vec::new(),
                k: layout.k,
                d: layout.d,
                n_side: layout.n_side,
                steps: chunk_len,
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
                if (addr as usize) >= layout.k {
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
            let table = lut_tables
                .get(&table_id)
                .ok_or_else(|| ShardBuildError::MissingTable(format!("missing LutTable for shout_id {}", table_id)))?;
            let ell = ell_from_pow2_n_side(table.n_side)?;

            let inst = LutInstance::<Cmt, Goldilocks> {
                comms: Vec::new(),
                k: table.k,
                d: table.d,
                n_side: table.n_side,
                steps: chunk_len,
                ell,
                table_spec: None,
                table: table.content.clone(),
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
