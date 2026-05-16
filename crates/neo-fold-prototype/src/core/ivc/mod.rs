//! Owns the generic native SuperNeo IVC/NIFS carrier.
//!
//! This module threads the algebraic SuperNeo accumulator through chunks:
//! `CE(b)^k + CCS^K -> CE(b)^k`. It intentionally does not own HyperNova
//! Construction-2 hash images, application step semantics, or Spartan
//! compression circuits.

mod relation;
mod state;
mod support;
mod transcript;
mod types;

use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat};
use neo_math::F;
use neo_params::NeoParams;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;

use crate::proof::{partition_step_inputs, Carry, FoldSchedule, StepInput};
use crate::prover::CommitmentMixers;

use support::elapsed_ms;

pub use types::{SuperNeoIvcBuild, SuperNeoIvcState, SuperNeoIvcStepRelation, SuperNeoIvcTranscriptSnapshot};

pub fn build_superneo_ivc_relations_with_perf<L, MR, MB>(
    schedule: FoldSchedule,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    steps: impl IntoIterator<Item = StepInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<SuperNeoIvcBuild, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    build_superneo_ivc_relations_with_initial_carry_perf(
        schedule,
        params,
        structure,
        steps,
        Carry::default(),
        log,
        mixers,
    )
}

pub fn build_superneo_ivc_relations_with_initial_carry_perf<L, MR, MB>(
    schedule: FoldSchedule,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    steps: impl IntoIterator<Item = StepInput>,
    initial_carry: Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<SuperNeoIvcBuild, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = Instant::now();
    let cache_started = Instant::now();
    let optimized_cache = OptimizedStructureCache::build(structure)?;
    let cache_build_ms = elapsed_ms(cache_started);

    let mut state = SuperNeoIvcState::seed_with_carry(initial_carry);
    let mut relations = Vec::new();
    for chunk in partition_step_inputs(schedule, steps.into_iter().collect())? {
        let (next_state, relation) =
            state.append_chunk_with_perf(params, structure, chunk, log, mixers, &optimized_cache)?;
        relation.verify(params, structure, log, mixers, &optimized_cache)?;
        state = next_state;
        relations.push(relation);
    }

    Ok(SuperNeoIvcBuild {
        relations,
        final_state: state,
        cache_build_ms,
        total_ms: elapsed_ms(total_started),
    })
}

pub fn build_superneo_ivc_relations_with_initial_carry_accumulator_handle_perf<L, MR, MB>(
    schedule: FoldSchedule,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    steps: impl IntoIterator<Item = StepInput>,
    initial_carry: Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<SuperNeoIvcBuild, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = Instant::now();
    let cache_started = Instant::now();
    let optimized_cache = OptimizedStructureCache::build(structure)?;
    let cache_build_ms = elapsed_ms(cache_started);

    let mut state = SuperNeoIvcState::seed_with_carry(initial_carry);
    let mut relations = Vec::new();
    for chunk in partition_step_inputs(schedule, steps.into_iter().collect())? {
        let (next_state, relation) = state.append_chunk_with_perf_and_accumulator_handle(
            params,
            structure,
            chunk,
            log,
            mixers,
            &optimized_cache,
        )?;
        relation.verify_with_accumulator_handle(params, structure, log, mixers, &optimized_cache)?;
        state = next_state;
        relations.push(relation);
    }

    Ok(SuperNeoIvcBuild {
        relations,
        final_state: state,
        cache_build_ms,
        total_ms: elapsed_ms(total_started),
    })
}
