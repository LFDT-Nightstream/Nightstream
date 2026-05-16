use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;

use crate::finalize::{
    package_session_proof, verify_finalized_session, verify_finalized_session_with_detailed_perf_and_cache,
    verify_finalized_session_with_perf, verify_finalized_session_with_perf_and_cache, PackagedVerifyPerf,
};
use crate::proof::{
    partition_prover_step_inputs, Carry, FoldSchedule, PackagedProof, RunProvePerf, RunVerifyPerf, StepInput,
};
use crate::prover::CommitmentMixers;

use super::prove::{prove_prepared_chunks_with_final_carry_perf, prove_prepared_chunks_with_perf};

pub fn prove_and_package<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    steps: impl IntoIterator<Item = StepInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<PackagedProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    Ok(prove_and_package_with_perf(mode, schedule, params, s, steps, log, mixers)?.0)
}

pub fn prove_and_package_with_perf<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    steps: impl IntoIterator<Item = StepInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(PackagedProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let steps_vec: Vec<StepInput> = steps.into_iter().collect();
    let input_chunks = partition_prover_step_inputs(schedule, steps_vec)?;
    let public_chunks = input_chunks
        .iter()
        .map(|chunk| chunk.public_chunk.clone())
        .collect::<Vec<_>>();
    let (session, perf) = prove_prepared_chunks_with_perf(mode, schedule, params, s, input_chunks, log, mixers)?;
    let packaged = package_session_proof(public_chunks, session)?;
    Ok((packaged, perf))
}

pub fn prove_and_package_with_final_carry_perf<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    steps: impl IntoIterator<Item = StepInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(PackagedProof, RunProvePerf, Carry), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let steps_vec: Vec<StepInput> = steps.into_iter().collect();
    let input_chunks = partition_prover_step_inputs(schedule, steps_vec)?;
    let public_chunks = input_chunks
        .iter()
        .map(|chunk| chunk.public_chunk.clone())
        .collect::<Vec<_>>();
    let (session, perf, final_carry) =
        prove_prepared_chunks_with_final_carry_perf(mode, schedule, params, s, input_chunks, log, mixers)?;
    let packaged = package_session_proof(public_chunks, session)?;
    Ok((packaged, perf, final_carry))
}

pub fn verify_packaged<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    proof: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<Vec<CeClaim<Commitment, F, K>>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_finalized_session(mode, params, s, proof, mixers)
}

pub fn verify_packaged_with_perf<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    proof: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, RunVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_finalized_session_with_perf(mode, params, s, proof, mixers)
}

pub fn verify_packaged_with_perf_and_cache<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    proof: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, RunVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_finalized_session_with_perf_and_cache(mode, params, s, proof, mixers, provided_cache)
}

pub(crate) fn verify_packaged_with_detailed_perf_and_cache<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    proof: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, PackagedVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_finalized_session_with_detailed_perf_and_cache(mode, params, s, proof, mixers, provided_cache)
}
