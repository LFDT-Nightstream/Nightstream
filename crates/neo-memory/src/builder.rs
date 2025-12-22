use crate::encode::{encode_lut_for_shout, encode_mem_for_twist};
use crate::mem_init::MemInit;
use crate::plain::{
    build_plain_lut_traces, build_plain_mem_traces, LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace,
};
use crate::witness::StepWitnessBundle;
use neo_vm_trace::VmTrace;

use neo_ccs::matrix::Mat;
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_params::NeoParams;
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

/// Build shard witness with optional CCS width alignment.
///
/// # Parameters
/// * `ccs_m` - The CCS witness width (`s.m`). If `None`, uses legacy mode (NOT RECOMMENDED).
/// * `m_in` - The number of public input columns for CCS-aligned encoding.
/// * `chunk_size` - Number of VM steps per folding chunk (`StepWitnessBundle`). Must be >= 1.
///
/// When `ccs_m` is provided, all memory/LUT witnesses are encoded at exactly `ccs_m` columns
/// with data embedded at offset `m_in`, ensuring proper alignment with Neo's ME relation.
///
/// Note: `cpu_arith` must implement `build_ccs_chunks` for the chosen `chunk_size`. The default
/// `build_ccs_steps` helper uses `chunk_size = 1`.
pub fn build_shard_witness<V, Cmt, L, K, A, Tw, Sh>(
    vm: V,
    twist: Tw,
    shout: Sh,
    max_steps: usize,
    chunk_size: usize,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    lut_tables: &HashMap<u32, LutTable<Goldilocks>>,
    // table_sizes: &HashMap<u32, (usize, usize)>, // Unused, derived from lut_tables
    initial_mem: &HashMap<(u32, u64), Goldilocks>,
    params: &NeoParams,
    commit: &L,
    cpu_arith: &A,
    ccs_m: Option<usize>,
    m_in: usize,
) -> Result<Vec<StepWitnessBundle<Cmt, Goldilocks, K>>, ShardBuildError>
where
    V: neo_vm_trace::VmCpu<u64, u64>,
    Tw: neo_vm_trace::Twist<u64, u64>,
    Sh: neo_vm_trace::Shout<u64>,
    L: Fn(&Mat<Goldilocks>) -> Cmt,
    A: CpuArithmetization<Goldilocks, Cmt>,
{
    if chunk_size == 0 {
        return Err(ShardBuildError::InvalidChunkSize("chunk_size must be >= 1".into()));
    }

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
            .filter_map(|(&addr, &val)| {
                if val == Goldilocks::ZERO {
                    return None;
                }
                Some((addr, val))
            })
            .collect();
        pairs.sort_by_key(|(addr, _)| *addr);

        if pairs.len() > 1 {
            for w in pairs.windows(2) {
                if w[0].0 == w[1].0 {
                    return Err(ShardBuildError::InvalidInit(format!(
                        "internal error: duplicate address {} in state for twist_id {}",
                        w[0].0, mem_id
                    )));
                }
            }
        }
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

    // 1) Run VM and collect full trace for this shard
    // We use trace_program now. It returns Result<VmTrace, V::Error>
    let trace = neo_vm_trace::trace_program(vm, twist, shout, max_steps)
        .map_err(|e| ShardBuildError::VmError(e.to_string()))?;

    let steps_len = trace.steps.len();
    let chunks_len = steps_len.div_ceil(chunk_size);

    // 3) Build plain twist/shout traces over [0..T)
    // We iterate the lut_tables to get the table_sizes for plain trace construction
    // This replaces the unused `table_sizes` parameter
    let mut table_sizes = HashMap::new();
    for (id, table) in lut_tables {
        table_sizes.insert(*id, (table.k, table.d));
    }

    let plain_mem = build_plain_mem_traces::<Goldilocks>(&trace, mem_layouts, initial_mem);
    let plain_lut = build_plain_lut_traces::<Goldilocks>(&trace, &table_sizes);

    // 4) Turn trace into per-chunk CCS instances/witnesses (CPU arithmetization)
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

    // Maintain a running sparse memory state per instance so each chunk can start from
    // the correct initial state (rollover across chunks).
    let mut mem_states: HashMap<u32, HashMap<u64, Goldilocks>> = HashMap::new();
    for (mem_id, layout) in mem_layouts {
        let mut state = HashMap::<u64, Goldilocks>::new();
        for ((init_mem_id, addr), &val) in initial_mem.iter() {
            if init_mem_id != mem_id || val == Goldilocks::ZERO {
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
        mem_states.insert(*mem_id, state);
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
        if let Some(ccs_m_val) = ccs_m {
            if m_in + chunk_len > ccs_m_val {
                return Err(ShardBuildError::InvalidChunkSize(format!(
                    "chunk_len={} does not fit in ccs_m={} with m_in={} (need m_in+chunk_len <= ccs_m)",
                    chunk_len, ccs_m_val, m_in
                )));
            }
        }

        // Build per-chunk memory witnesses
        let mut mem_instances = Vec::new();
        for (mem_id, plain) in &plain_mem {
            let layout = mem_layouts.get(mem_id).ok_or_else(|| {
                ShardBuildError::MissingLayout(format!("missing PlainMemLayout for twist_id {}", mem_id))
            })?;

            let state = mem_states
                .get_mut(mem_id)
                .ok_or_else(|| ShardBuildError::MissingLayout(format!("missing state for twist_id {}", mem_id)))?;
            let init = mem_init_from_state(*mem_id, layout.k, state)?;
            let chunk_plain = PlainMemTrace {
                steps: chunk_len,
                has_read: plain.has_read[chunk_start..chunk_end].to_vec(),
                has_write: plain.has_write[chunk_start..chunk_end].to_vec(),
                read_addr: plain.read_addr[chunk_start..chunk_end].to_vec(),
                write_addr: plain.write_addr[chunk_start..chunk_end].to_vec(),
                read_val: plain.read_val[chunk_start..chunk_end].to_vec(),
                write_val: plain.write_val[chunk_start..chunk_end].to_vec(),
                inc_at_write_addr: plain.inc_at_write_addr[chunk_start..chunk_end].to_vec(),
            };

            let (inst, wit) = encode_mem_for_twist(params, layout, &init, &chunk_plain, commit, ccs_m, m_in);
            mem_instances.push((inst, wit));

            // Advance state across this chunk for the next chunk's init.
            for t in 0..chunk_len {
                if chunk_plain.has_write[t] != Goldilocks::ONE {
                    continue;
                }
                let addr = chunk_plain.write_addr[t];
                if (addr as usize) >= layout.k {
                    continue;
                }
                let new_val = chunk_plain.write_val[t];
                if new_val == Goldilocks::ZERO {
                    state.remove(&addr);
                } else {
                    state.insert(addr, new_val);
                }
            }
        }

        // Build per-chunk lookup witnesses
        let mut lut_instances = Vec::new();
        for (table_id, plain) in &plain_lut {
            let table = lut_tables
                .get(table_id)
                .ok_or_else(|| ShardBuildError::MissingTable(format!("missing LutTable for shout_id {}", table_id)))?;

            let chunk_plain = PlainLutTrace {
                has_lookup: plain.has_lookup[chunk_start..chunk_end].to_vec(),
                addr: plain.addr[chunk_start..chunk_end].to_vec(),
                val: plain.val[chunk_start..chunk_end].to_vec(),
            };

            let (inst, wit) = encode_lut_for_shout(params, table, &chunk_plain, commit, ccs_m, m_in);
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
