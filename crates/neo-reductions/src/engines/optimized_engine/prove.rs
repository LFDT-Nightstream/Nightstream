//! Optimized prover entrypoints for the selected one-joint PiCCS protocol.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;

use crate::error::PiCcsError;

use super::{OptimizedStructureCache, PiCcsProof, PiCcsProvePerf, PiDecProverPrecompute};

pub fn optimized_prove(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    let cache = OptimizedStructureCache::build(structure)?;
    let (outputs, proof, _) = optimized_prove_with_cache_and_perf(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        &cache,
    )?;
    Ok((outputs, proof))
}

#[allow(clippy::too_many_arguments)]
pub fn optimized_prove_with_cache_and_perf(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    cache: &OptimizedStructureCache,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, PiCcsProvePerf), PiCcsError> {
    super::paper_joint::prove(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        cache,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn optimized_prove_with_cache_and_precompute_and_perf(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    cache: &OptimizedStructureCache,
) -> Result<
    (
        Vec<CeClaim<Cmt, F, K>>,
        PiCcsProof,
        PiCcsProvePerf,
        PiDecProverPrecompute,
    ),
    PiCcsError,
> {
    let (outputs, proof, perf) = super::paper_joint::prove(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        cache,
    )?;
    let precompute = PiDecProverPrecompute {
        row_chals: outputs
            .first()
            .ok_or_else(|| PiCcsError::ProtocolError("Pi_CCS produced no output claims".into()))?
            .r
            .clone(),
    };
    Ok((outputs, proof, perf, precompute))
}
