use crate::encode::{encode_lut_for_shout, encode_mem_for_twist};
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
    fn build_ccs_steps(
        &self,
        trace: &VmTrace<u64, u64>,
    ) -> Result<Vec<(McsInstance<Cmt, F>, McsWitness<F>)>, Self::Error>;
}

#[derive(Debug)]
pub enum ShardBuildError {
    VmError(String),
    CcsError(String),
    MissingLayout(String),
    MissingTable(String),
}

pub trait Program {}

/// Build shard witness with optional CCS width alignment.
///
/// # Parameters
/// * `ccs_m` - The CCS witness width (`s.m`). If `None`, uses legacy mode (NOT RECOMMENDED).
/// * `m_in` - The number of public input columns for CCS-aligned encoding.
///
/// When `ccs_m` is provided, all memory/LUT witnesses are encoded at exactly `ccs_m` columns
/// with data embedded at offset `m_in`, ensuring proper alignment with Neo's ME relation.
pub fn build_shard_witness<V, Cmt, L, K, A, Tw, Sh>(
    vm: V,
    twist: Tw,
    shout: Sh,
    max_steps: usize,
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
    // 1) Run VM and collect full trace for this shard
    // We use trace_program now. It returns Result<VmTrace, V::Error>
    let trace = neo_vm_trace::trace_program(vm, twist, shout, max_steps)
        .map_err(|e| ShardBuildError::VmError(e.to_string()))?;

    // 2) Turn trace into per-step CCS instances/witnesses (CPU arithmetization)
    let mcss = cpu_arith
        .build_ccs_steps(&trace)
        .map_err(|e| ShardBuildError::CcsError(e.to_string()))?;

    // 3) Build plain twist/shout traces over [0..T)
    // We iterate the lut_tables to get the table_sizes for plain trace construction
    // This replaces the unused `table_sizes` parameter
    let mut table_sizes = HashMap::new();
    for (id, table) in lut_tables {
        table_sizes.insert(*id, (table.k, table.d));
    }

    let plain_mem = build_plain_mem_traces::<Goldilocks>(&trace, mem_layouts, initial_mem);
    let plain_lut = build_plain_lut_traces::<Goldilocks>(&trace, &table_sizes);

    // 4) Encode per-step Twist/Shout instances
    let steps_len = trace.steps.len();
    if mcss.len() != steps_len {
        return Err(ShardBuildError::CcsError(format!(
            "cpu arithmetization returned {} steps, trace has {}",
            mcss.len(),
            steps_len
        )));
    }

    // Track rolling memory state so each per-step witness has the correct init_vals
    let mut mem_states: HashMap<u32, Vec<Goldilocks>> = HashMap::new();
    for (mem_id, plain) in &plain_mem {
        mem_states.insert(*mem_id, plain.init_vals.clone());
    }

    let mut step_bundles = Vec::with_capacity(steps_len);

    for (step_idx, mcs) in mcss.into_iter().enumerate() {
        // Build per-step memory witnesses
        let mut mem_instances = Vec::new();
        for (mem_id, plain) in &plain_mem {
            let layout = mem_layouts.get(mem_id).ok_or_else(|| {
                ShardBuildError::MissingLayout(format!("missing PlainMemLayout for twist_id {}", mem_id))
            })?;
            let mut state = mem_states
                .get(mem_id)
                .cloned()
                .ok_or_else(|| ShardBuildError::MissingLayout(format!("missing mem state for twist_id {}", mem_id)))?;

            let single_plain = PlainMemTrace {
                init_vals: state.clone(),
                steps: 1,
                has_read: vec![plain.has_read[step_idx]],
                has_write: vec![plain.has_write[step_idx]],
                read_addr: vec![plain.read_addr[step_idx]],
                write_addr: vec![plain.write_addr[step_idx]],
                read_val: vec![plain.read_val[step_idx]],
                write_val: vec![plain.write_val[step_idx]],
                inc: plain.inc.iter().map(|row| vec![row[step_idx]]).collect(),
            };

            let (inst, wit) = encode_mem_for_twist(params, layout, &single_plain, commit, ccs_m, m_in);
            mem_instances.push((inst, wit));

            // Advance memory state for the next step
            if plain.has_write[step_idx] == Goldilocks::ONE {
                let addr = plain.write_addr[step_idx] as usize;
                if addr < state.len() {
                    state[addr] = plain.write_val[step_idx];
                    mem_states.insert(*mem_id, state);
                }
            }
        }

        // Build per-step lookup witnesses
        let mut lut_instances = Vec::new();
        for (table_id, plain) in &plain_lut {
            let table = lut_tables
                .get(table_id)
                .ok_or_else(|| ShardBuildError::MissingTable(format!("missing LutTable for shout_id {}", table_id)))?;

            let single_plain = PlainLutTrace {
                has_lookup: vec![plain.has_lookup[step_idx]],
                addr: vec![plain.addr[step_idx]],
                val: vec![plain.val[step_idx]],
            };

            let (inst, wit) = encode_lut_for_shout(params, table, &single_plain, commit, ccs_m, m_in);
            lut_instances.push((inst, wit));
        }

        step_bundles.push(StepWitnessBundle {
            mcs,
            lut_instances,
            mem_instances,
            _phantom: PhantomData,
        });
    }

    Ok(step_bundles)
}
