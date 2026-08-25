//! Optimized prover entrypoints for the selected one-joint PiCCS protocol.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;

use crate::engines::pi_ccs_joint_protocol::TranscriptBinding;
use crate::error::PiCcsError;

use super::{OptimizedStructureCache, PaperJointOracleBackend, PiCcsProof, PiCcsProvePerf, PiDecProverPrecompute};

#[allow(clippy::too_many_arguments)]
pub fn optimized_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    let cache = OptimizedStructureCache::build(structure)?;
    optimized_prove_with_cache(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        commitment,
        &cache,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn optimized_prove_with_cache<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
    cache: &OptimizedStructureCache,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    let (outputs, proof, _) = optimized_prove_with_cache_and_perf(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        commitment,
        cache,
    )?;
    Ok((outputs, proof))
}

#[allow(clippy::too_many_arguments)]
pub fn optimized_prove_with_cache_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
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
        commitment,
        cache,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn optimized_prove_with_cache_and_precompute_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
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
    let (outputs, proof, perf) = super::paper_joint::prove_with_binding(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        commitment,
        cache,
        TranscriptBinding::digest_only(),
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

#[allow(clippy::too_many_arguments)]
pub fn optimized_prove_with_cache_and_precompute_and_backend_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
    cache: &OptimizedStructureCache,
    backend: &mut dyn PaperJointOracleBackend,
) -> Result<
    (
        Vec<CeClaim<Cmt, F, K>>,
        PiCcsProof,
        PiCcsProvePerf,
        PiDecProverPrecompute,
    ),
    PiCcsError,
> {
    let (outputs, proof, perf) = super::paper_joint::prove_with_binding_and_backend(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        commitment,
        cache,
        TranscriptBinding::digest_only(),
        backend,
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

pub fn optimized_prove_simple<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    commitment: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    optimized_prove(
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        &[],
        &[],
        commitment,
    )
}
