//! Owns RV64IM build/prove adapters for the direct main-relation Spartan path.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::rv64im::decider_relation::Rv64imDeciderRelation;
use crate::rv64im::final_relation::{
    prove_rv64im_final_statement_from_accepted_with_output_and_perf_and_source, Rv64imFinalBuildOutput,
    Rv64imFinalProof, Rv64imFinalProofComponentDigests, Rv64imFinalStatement,
};
use crate::rv64im::kernel::{
    accepted_proof_artifact_from_prover_materials, build_rv64im_accepted_proof_artifact,
    build_rv64im_kernel_export_source_from_accepted_artifact, prove_rv64im_public_proof_prover_seam_with_perf,
    Rv64imAcceptedProofArtifact, Rv64imKernelExportRelationResult, Rv64imKernelExportSource, Rv64imProof,
    Rv64imProofInput, Rv64imProofProvePerf, Rv64imPublicProofOptions,
};
use crate::rv64im::main_relation_spartan::{
    prove_rv64im_spartan2_decider as prove_main_relation_spartan,
    setup_rv64im_spartan2_decider as setup_main_relation_spartan,
    setup_rv64im_spartan2_decider_cached as setup_main_relation_spartan_cached,
    verify_rv64im_spartan2_decider as verify_main_relation_spartan, Rv64imSpartan2DeciderKeyPair,
    Rv64imSpartan2DeciderProof, Rv64imSpartan2DeciderProverKey, Rv64imSpartan2DeciderVerifierKey,
};
use crate::rv64im::SimpleKernelError;

static RV64IM_SPARTAN2_DECIDER_PROOF_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Arc<Rv64imSpartan2DeciderProof>>>> =
    OnceLock::new();

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv64imPublishedProofSeamBuildPerf {
    pub accepted_artifact_ms: f64,
    pub kernel_export_source_ms: f64,
    pub final_statement_ms: f64,
    pub final_statement_kernel_export_ms: f64,
    pub final_statement_recursive_proof_ms: f64,
    pub final_statement_recursive_prepare_inputs_ms: f64,
    pub final_statement_recursive_ccs_bind_ms: f64,
    pub final_statement_recursive_ccs_sample_challenges_ms: f64,
    pub final_statement_recursive_ccs_fe_sumcheck_ms: f64,
    pub final_statement_recursive_ccs_nc_sumcheck_ms: f64,
    pub final_statement_recursive_ccs_output_materialize_ms: f64,
    pub final_statement_recursive_ccs_ms: f64,
    pub final_statement_recursive_dims_ms: f64,
    pub final_statement_recursive_rlc_prepare_ms: f64,
    pub final_statement_recursive_rlc_ms: f64,
    pub final_statement_recursive_dec_split_ms: f64,
    pub final_statement_recursive_dec_commit_ms: f64,
    pub final_statement_recursive_dec_ms: f64,
    pub final_statement_folded_digest_ms: f64,
    pub final_statement_final_proof_ms: f64,
    pub final_statement_statement_digest_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug)]
pub struct Rv64imPublishedProofSeam {
    pub accepted_artifact: Rv64imAcceptedProofArtifact,
    pub final_statement: Rv64imFinalStatement,
    pub final_proof: Rv64imFinalProof,
    pub(crate) final_component_digests: Rv64imFinalProofComponentDigests,
    pub(crate) verified_kernel: Rv64imKernelExportRelationResult,
}

#[derive(Clone, Debug, Default)]
pub struct Rv64imPublicProofAndSeamBuildPerf {
    pub proof: Rv64imProofProvePerf,
    pub seam: Rv64imPublishedProofSeamBuildPerf,
}

impl Rv64imPublishedProofSeam {
    pub fn kernel_export_source(&self) -> &Rv64imKernelExportSource {
        &self.final_proof.kernel_export.source
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn rv64im_spartan2_decider_cache_key(statement: &Rv64imFinalStatement, proof: &Rv64imFinalProof) -> [u8; 32] {
    let mut digest = [0u8; 32];
    for ((dst, lhs), rhs) in digest
        .iter_mut()
        .zip(statement.digest.iter())
        .zip(proof.proof_digest.iter())
    {
        *dst = *lhs ^ *rhs;
    }
    digest
}

pub fn build_rv64im_published_proof_seam(proof: &Rv64imProof) -> Result<Rv64imPublishedProofSeam, SimpleKernelError> {
    let (built, _) = build_rv64im_published_proof_seam_with_perf(proof)?;
    Ok(built)
}

pub fn build_rv64im_published_proof_seam_with_perf(
    proof: &Rv64imProof,
) -> Result<(Rv64imPublishedProofSeam, Rv64imPublishedProofSeamBuildPerf), SimpleKernelError> {
    let total_started = Instant::now();

    let started = Instant::now();
    let accepted_artifact = build_rv64im_accepted_proof_artifact(proof)?;
    let accepted_artifact_ms = elapsed_ms(started);

    let started = Instant::now();
    let kernel_export_source = build_rv64im_kernel_export_source_from_accepted_artifact(&accepted_artifact)?;
    let kernel_export_source_ms = elapsed_ms(started);

    let started = Instant::now();
    let (
        Rv64imFinalBuildOutput {
            statement: final_statement,
            proof: final_proof,
            component_digests: final_component_digests,
            verified_kernel,
        },
        final_perf,
    ) = prove_rv64im_final_statement_from_accepted_with_output_and_perf_and_source(
        &accepted_artifact,
        Some(kernel_export_source),
        None,
    )?;
    let final_statement_ms = elapsed_ms(started);

    Ok((
        Rv64imPublishedProofSeam {
            accepted_artifact,
            final_statement,
            final_proof,
            final_component_digests,
            verified_kernel,
        },
        Rv64imPublishedProofSeamBuildPerf {
            accepted_artifact_ms,
            kernel_export_source_ms,
            final_statement_ms,
            final_statement_kernel_export_ms: final_perf.folded.kernel_export_ms,
            final_statement_recursive_proof_ms: final_perf.folded.recursive.total_ms,
            final_statement_recursive_prepare_inputs_ms: final_perf.folded.recursive.prepare_inputs_ms,
            final_statement_recursive_ccs_bind_ms: final_perf.folded.recursive.ccs_bind_ms,
            final_statement_recursive_ccs_sample_challenges_ms: final_perf.folded.recursive.ccs_sample_challenges_ms,
            final_statement_recursive_ccs_fe_sumcheck_ms: final_perf.folded.recursive.ccs_fe_sumcheck_ms,
            final_statement_recursive_ccs_nc_sumcheck_ms: final_perf.folded.recursive.ccs_nc_sumcheck_ms,
            final_statement_recursive_ccs_output_materialize_ms: final_perf.folded.recursive.ccs_output_materialize_ms,
            final_statement_recursive_ccs_ms: final_perf.folded.recursive.ccs_ms,
            final_statement_recursive_dims_ms: final_perf.folded.recursive.dims_ms,
            final_statement_recursive_rlc_prepare_ms: final_perf.folded.recursive.rlc_prepare_ms,
            final_statement_recursive_rlc_ms: final_perf.folded.recursive.rlc_ms,
            final_statement_recursive_dec_split_ms: final_perf.folded.recursive.dec_split_ms,
            final_statement_recursive_dec_commit_ms: final_perf.folded.recursive.dec_commit_ms,
            final_statement_recursive_dec_ms: final_perf.folded.recursive.dec_ms,
            final_statement_folded_digest_ms: final_perf.folded.folded_digest_ms,
            final_statement_final_proof_ms: final_perf.final_proof_ms,
            final_statement_statement_digest_ms: final_perf.statement_digest_ms,
            total_ms: elapsed_ms(total_started),
        },
    ))
}

pub fn prove_rv64im_public_proof_and_published_seam_with_perf(
    input: &Rv64imProofInput,
) -> Result<
    (
        (Rv64imProof, Rv64imPublishedProofSeam),
        Rv64imPublicProofAndSeamBuildPerf,
    ),
    SimpleKernelError,
> {
    prove_rv64im_public_proof_and_published_seam_with_options_and_perf(input, Rv64imPublicProofOptions::default())
}

pub fn prove_rv64im_public_proof_and_published_seam_with_options_and_perf(
    input: &Rv64imProofInput,
    options: Rv64imPublicProofOptions,
) -> Result<
    (
        (Rv64imProof, Rv64imPublishedProofSeam),
        Rv64imPublicProofAndSeamBuildPerf,
    ),
    SimpleKernelError,
> {
    let (built, proof_perf) = prove_rv64im_public_proof_prover_seam_with_perf(input, options)?;

    let total_started = Instant::now();

    let started = Instant::now();
    let accepted_artifact = accepted_proof_artifact_from_prover_materials(
        &built.proof.claim,
        &built.proof.statement,
        &built.kernel,
        &built.sidecar,
        &built.proof.kernel.main_lane,
        &built.proof.kernel.stage_claims,
        &built.proof.kernel.stage_packages,
        &built.proof.kernel.kernel_opening,
        &built.proof.kernel.kernel_claims,
        &built.proof.kernel.root_lane_columns,
        &built.proof.kernel.root_lane_commitment,
    )?;
    let accepted_artifact_ms = elapsed_ms(started);

    let started = Instant::now();
    let kernel_export_source = build_rv64im_kernel_export_source_from_accepted_artifact(&accepted_artifact)?;
    let kernel_export_source_ms = elapsed_ms(started);

    let started = Instant::now();
    let (
        Rv64imFinalBuildOutput {
            statement: final_statement,
            proof: final_proof,
            component_digests: final_component_digests,
            verified_kernel,
        },
        final_perf,
    ) = prove_rv64im_final_statement_from_accepted_with_output_and_perf_and_source(
        &accepted_artifact,
        Some(kernel_export_source),
        Some(built.main_lane_inputs),
    )?;
    let final_statement_ms = elapsed_ms(started);

    let seam = Rv64imPublishedProofSeam {
        accepted_artifact,
        final_statement,
        final_proof,
        final_component_digests,
        verified_kernel,
    };
    let seam_perf = Rv64imPublishedProofSeamBuildPerf {
        accepted_artifact_ms,
        kernel_export_source_ms,
        final_statement_ms,
        final_statement_kernel_export_ms: final_perf.folded.kernel_export_ms,
        final_statement_recursive_proof_ms: final_perf.folded.recursive.total_ms,
        final_statement_recursive_prepare_inputs_ms: final_perf.folded.recursive.prepare_inputs_ms,
        final_statement_recursive_ccs_bind_ms: final_perf.folded.recursive.ccs_bind_ms,
        final_statement_recursive_ccs_sample_challenges_ms: final_perf.folded.recursive.ccs_sample_challenges_ms,
        final_statement_recursive_ccs_fe_sumcheck_ms: final_perf.folded.recursive.ccs_fe_sumcheck_ms,
        final_statement_recursive_ccs_nc_sumcheck_ms: final_perf.folded.recursive.ccs_nc_sumcheck_ms,
        final_statement_recursive_ccs_output_materialize_ms: final_perf.folded.recursive.ccs_output_materialize_ms,
        final_statement_recursive_ccs_ms: final_perf.folded.recursive.ccs_ms,
        final_statement_recursive_dims_ms: final_perf.folded.recursive.dims_ms,
        final_statement_recursive_rlc_prepare_ms: final_perf.folded.recursive.rlc_prepare_ms,
        final_statement_recursive_rlc_ms: final_perf.folded.recursive.rlc_ms,
        final_statement_recursive_dec_split_ms: final_perf.folded.recursive.dec_split_ms,
        final_statement_recursive_dec_commit_ms: final_perf.folded.recursive.dec_commit_ms,
        final_statement_recursive_dec_ms: final_perf.folded.recursive.dec_ms,
        final_statement_folded_digest_ms: final_perf.folded.folded_digest_ms,
        final_statement_final_proof_ms: final_perf.final_proof_ms,
        final_statement_statement_digest_ms: final_perf.statement_digest_ms,
        total_ms: elapsed_ms(total_started),
    };

    Ok((
        (built.proof, seam),
        Rv64imPublicProofAndSeamBuildPerf {
            proof: proof_perf,
            seam: seam_perf,
        },
    ))
}

pub fn setup_rv64im_spartan2_decider(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<(Rv64imSpartan2DeciderProverKey, Rv64imSpartan2DeciderVerifierKey), SimpleKernelError> {
    setup_main_relation_spartan(statement, proof)
}

pub fn setup_rv64im_spartan2_decider_cached(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imSpartan2DeciderKeyPair, SimpleKernelError> {
    setup_main_relation_spartan_cached(statement, proof)
}

pub fn prove_rv64im_spartan2_decider(
    pk: &Rv64imSpartan2DeciderProverKey,
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imSpartan2DeciderProof, SimpleKernelError> {
    prove_main_relation_spartan(pk, statement, proof)
}

pub fn prove_rv64im_spartan2_decider_cached(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imSpartan2DeciderProof, SimpleKernelError> {
    let cache_key = rv64im_spartan2_decider_cache_key(statement, proof);
    let cache = RV64IM_SPARTAN2_DECIDER_PROOF_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(proof) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main decider proof cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok((*proof).clone());
    }

    let keys = setup_main_relation_spartan_cached(statement, proof)?;
    let proof = Arc::new(prove_main_relation_spartan(&keys.as_ref().0, statement, proof)?);
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main decider proof cache poisoned".into()))?
        .insert(cache_key, proof.clone());
    Ok((*proof).clone())
}

pub fn verify_rv64im_spartan2_decider(
    vk: &Rv64imSpartan2DeciderVerifierKey,
    public_statement_digest: [u8; 32],
    relation: &Rv64imDeciderRelation,
    decider_proof: &Rv64imSpartan2DeciderProof,
) -> Result<(), SimpleKernelError> {
    verify_main_relation_spartan(vk, public_statement_digest, relation, decider_proof)
}

pub fn setup_rv64im_spartan2_decider_from_public_proof(
    proof: &Rv64imProof,
) -> Result<(Rv64imSpartan2DeciderProverKey, Rv64imSpartan2DeciderVerifierKey), SimpleKernelError> {
    let built = build_rv64im_published_proof_seam(proof)?;
    setup_main_relation_spartan(&built.final_statement, &built.final_proof)
}

pub fn setup_rv64im_spartan2_decider_from_public_proof_cached(
    proof: &Rv64imProof,
) -> Result<Rv64imSpartan2DeciderKeyPair, SimpleKernelError> {
    let built = build_rv64im_published_proof_seam(proof)?;
    setup_main_relation_spartan_cached(&built.final_statement, &built.final_proof)
}

pub fn prove_rv64im_spartan2_decider_from_public_proof(
    pk: &Rv64imSpartan2DeciderProverKey,
    proof: &Rv64imProof,
) -> Result<Rv64imSpartan2DeciderProof, SimpleKernelError> {
    let built = build_rv64im_published_proof_seam(proof)?;
    prove_main_relation_spartan(pk, &built.final_statement, &built.final_proof)
}

pub fn prove_rv64im_spartan2_decider_from_public_proof_cached(
    proof: &Rv64imProof,
) -> Result<Rv64imSpartan2DeciderProof, SimpleKernelError> {
    let built = build_rv64im_published_proof_seam(proof)?;
    prove_rv64im_spartan2_decider_cached(&built.final_statement, &built.final_proof)
}
