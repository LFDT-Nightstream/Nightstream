use crate::encode::{encode_lut_for_shout, encode_mem_for_twist};
use crate::plain::{build_plain_lut_traces, build_plain_mem_traces, LutTable, PlainMemLayout};
use crate::witness::ShardWitnessBundle;
use neo_vm_trace::VmTrace;

use neo_ccs::matrix::Mat;
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_params::NeoParams;
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
) -> Result<ShardWitnessBundle<Cmt, Goldilocks, K>, ShardBuildError>
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

    // 4) Encode Twist: one MemInstance per mem_id
    let mut mem_shard_instances = Vec::new();
    for (mem_id, plain) in plain_mem.iter() {
        let layout = mem_layouts
            .get(mem_id)
            .ok_or_else(|| ShardBuildError::MissingLayout(format!("missing PlainMemLayout for twist_id {}", mem_id)))?;
        let (inst, wit) = encode_mem_for_twist(params, layout, plain, commit);
        mem_shard_instances.push((inst, wit));
    }

    // 5) Encode Shout: one LutInstance per table
    let mut lut_shard_instances = Vec::new();
    for (table_id, plain) in plain_lut.iter() {
        let table = lut_tables
            .get(table_id)
            .ok_or_else(|| ShardBuildError::MissingTable(format!("missing LutTable for shout_id {}", table_id)))?;
        let (inst, wit) = encode_lut_for_shout(params, table, plain, commit);
        lut_shard_instances.push((inst, wit));
    }

    Ok(ShardWitnessBundle {
        mcss,
        lut_shard_instances,
        mem_shard_instances,
        _phantom: PhantomData,
    })
}
