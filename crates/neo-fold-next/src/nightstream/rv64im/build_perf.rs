//! Owns compact build-time diagnostics for the RV64IM Nightstream boundary.

use std::time::Instant;

use crate::rv64im::kernel::build_rv64im_eval_claim_witnesses_from_accepted_artifact_with_perf;

use super::side_eval_claim_relation::rebind_phase0_claim_witnesses_to_side_bundle;
use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv64imNightstreamVerifiedSeamsBuildPerf {
    pub final_surface_guard_ms: f64,
    pub decider_relation_ms: f64,
    pub linkage_claims_ms: f64,
    pub main_decider_proof_ms: f64,
    pub main_residual_proof_ms: f64,
    pub linkage_artifact_ms: f64,
    pub linkage_root_ms: f64,
    pub statement_ms: f64,
    pub bind_side_bundle_ms: f64,
    pub opening_phase0_artifact_ms: f64,
    pub opening_phase0_claim_witnesses_ms: f64,
    pub opening_phase0_relation_artifact_ms: f64,
    pub opening_phase0_packed_columns_ms: f64,
    pub opening_phase0_commitment_vector_ms: f64,
    pub opening_phase0_commitment_params_ms: f64,
    pub opening_phase0_commitment_committer_ms: f64,
    pub opening_phase0_commitment_mats_ms: f64,
    pub opening_phase0_commitment_commit_many_ms: f64,
    pub opening_phase0_commitment_root_ms: f64,
    pub opening_phase0_opened_object_id_ms: f64,
    pub opening_phase0_opened_object_total_ms: f64,
    pub opening_phase0_binding_digest_ms: f64,
    pub opening_phase0_point_derivation_ms: f64,
    pub opening_phase0_payload_eval_ms: f64,
    pub opening_phase0_claim_build_ms: f64,
    pub opening_phase0_slot_claims_total_ms: f64,
    pub opening_artifact_ms: f64,
    pub opening_convergence_total_ms: f64,
    pub opening_convergence_phase1_ms: f64,
    pub opening_convergence_phase2_ms: f64,
    pub opening_convergence_final_openings_ms: f64,
    pub opening_convergence_final_openings_witness_map_ms: f64,
    pub opening_convergence_final_openings_representative_ms: f64,
    pub opening_convergence_final_openings_commitment_validate_ms: f64,
    pub opening_convergence_final_openings_opened_commitment_digest_ms: f64,
    pub opening_convergence_final_openings_opening_proof_digest_ms: f64,
    pub opening_convergence_final_openings_target_build_ms: f64,
    pub opening_convergence_digest_ms: f64,
    pub opening_artifact_wrap_ms: f64,
    pub hybrid_side_bridge_prepare_ms: f64,
    pub hybrid_side_bridge_backend_shell_setup_ms: f64,
    pub hybrid_side_bridge_backend_proof_ms: f64,
    pub hybrid_side_bridge_artifact_ms: f64,
    pub proof_binding_root_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv64imNightstreamBuildPerf {
    pub accepted_artifact_ms: f64,
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
    pub side_proof_bundle_ms: f64,
    pub verified_seams: Rv64imNightstreamVerifiedSeamsBuildPerf,
    pub total_ms: f64,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn build_rv64im_nightstream_linkage_claims_from_relation_unchecked(
    relation: &crate::rv64im::decider_relation::Rv64imDeciderRelation,
    bridge_handoff_digests: &[[u8; 32]],
) -> Result<Rv64imLinkageClaims, SimpleKernelError> {
    if relation.chunk_summaries.len() != bridge_handoff_digests.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream linkage handoff count does not match the carried decider relation".into(),
        ));
    }
    let mut claims = Rv64imLinkageClaims {
        public_chunk_digests: relation
            .chunk_summaries
            .iter()
            .map(|summary| summary.public_chunk_digest)
            .collect(),
        bridge_handoff_digests: bridge_handoff_digests.to_vec(),
        digest: [0; 32],
    };
    claims.digest = claims.expected_digest();
    Ok(claims)
}

fn guard_locally_built_final_surface(
    accepted_artifact: &Rv64imAcceptedProofArtifact,
    final_statement: &Rv64imFinalStatement,
    final_proof: &Rv64imFinalProof,
    final_component_digests: &crate::rv64im::final_relation::Rv64imFinalProofComponentDigests,
) -> Result<(), SimpleKernelError> {
    if final_statement.public_statement_digest != accepted_artifact.statement.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream local final statement does not match the carried accepted-artifact statement".into(),
        ));
    }
    if final_component_digests.kernel_export_proof_digest != final_proof.kernel_export.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream local final proof component digests do not match the carried kernel export proof"
                .into(),
        ));
    }
    if final_statement.folded.chunk_count as usize != final_proof.chunk_summaries.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream local final statement chunk count does not match the carried final proof chunk summaries"
                .into(),
        ));
    }
    if final_statement.folded.chunk_count as usize != final_component_digests.chunk_transition_digests.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream local final statement chunk count does not match the carried replay witness digests"
                .into(),
        ));
    }
    Ok(())
}

pub(super) fn build_rv64im_nightstream_from_verified_seams_with_perf(
    accepted_artifact: &Rv64imAcceptedProofArtifact,
    verifier_context_digest: [u8; 32],
    final_statement: &Rv64imFinalStatement,
    final_proof: &Rv64imFinalProof,
    final_component_digests: &crate::rv64im::final_relation::Rv64imFinalProofComponentDigests,
    verified_kernel: &crate::rv64im::kernel::Rv64imKernelExportRelationResult,
    side_proof_bundle: Rv64imSideProofBundle,
) -> Result<
    (
        (NightstreamStatement, Rv64imNightstreamProof),
        Rv64imNightstreamVerifiedSeamsBuildPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();

    let started = Instant::now();
    guard_locally_built_final_surface(accepted_artifact, final_statement, final_proof, final_component_digests)?;
    let final_surface_guard_ms = elapsed_ms(started);

    let started = Instant::now();
    let decider_relation =
        crate::rv64im::decider_relation::build_rv64im_decider_relation_from_verified_final_with_component_digests(
            final_statement,
            final_proof,
            verified_kernel,
            final_component_digests,
        )?;
    let decider_relation_ms = elapsed_ms(started);

    let bridge_handoff_digests = verified_kernel
        .chunk_handoffs
        .iter()
        .map(|handoff| handoff.bridge_handoff.digest)
        .collect::<Vec<_>>();

    let started = Instant::now();
    let main_decider_proof = build_rv64im_main_decider_proof_from_relation(&decider_relation)?;
    let main_decider_proof_ms = elapsed_ms(started);

    let started = Instant::now();
    let main_residual_proof = Rv64imMainResidualProof {
        public_statement_digest: final_statement.public_statement_digest,
        decider_relation,
        bridge_handoff_digests: bridge_handoff_digests.clone(),
    };
    let main_residual_proof_ms = elapsed_ms(started);

    let started = Instant::now();
    let linkage_claims = build_rv64im_nightstream_linkage_claims_from_relation_unchecked(
        &main_residual_proof.decider_relation,
        &bridge_handoff_digests,
    )?;
    let linkage_claims_ms = elapsed_ms(started);

    let started = Instant::now();
    let linkage_artifact = build_rv64im_linkage_artifact_from_claims(&linkage_claims)?;
    let linkage_artifact_ms = elapsed_ms(started);

    let started = Instant::now();
    let linkage_root = rv64im_nightstream_linkage_root(
        rv64im_kernel_export_anchor_digest_from_relation(&main_residual_proof.decider_relation)?,
        &linkage_claims,
    );
    let linkage_root_ms = elapsed_ms(started);

    let started = Instant::now();
    let mut statement = build_rv64im_nightstream_statement_from_relation(
        final_statement.public_statement_digest,
        verifier_context_digest,
        &main_residual_proof.decider_relation,
        linkage_root,
        [0; 32],
    )?;
    let statement_ms = elapsed_ms(started);

    let started = Instant::now();
    let side_proof_bundle =
        bind_rv64im_side_proof_bundle_to_statement_core(&side_proof_bundle, statement.core_digest())?;
    let bind_side_bundle_ms = elapsed_ms(started);

    let started = Instant::now();
    let (accepted_claim_witnesses, claim_witness_perf) =
        build_rv64im_eval_claim_witnesses_from_accepted_artifact_with_perf(accepted_artifact)?;
    let claim_witnesses = rebind_phase0_claim_witnesses_to_side_bundle(&side_proof_bundle, &accepted_claim_witnesses)?;
    let opening_phase0_claim_witnesses_ms = elapsed_ms(started);

    let started = Instant::now();
    let opening_phase0_artifact = super::side_eval_claim_relation::
        build_rv64im_side_eval_claim_artifact_from_claim_witnesses_and_trusted_side_bundle(
            &accepted_artifact.statement,
            &side_proof_bundle,
            &claim_witnesses,
        )?;
    let opening_phase0_relation_artifact_ms = elapsed_ms(started);
    let opening_phase0_artifact_ms = opening_phase0_claim_witnesses_ms + opening_phase0_relation_artifact_ms;

    let started = Instant::now();
    let phase0_binding_surface =
        super::side_eval_claim_relation::build_rv64im_phase0_binding_surface_from_side_bundle(&side_proof_bundle);
    let opening_phase0_binding_surface_ms = elapsed_ms(started);

    let started = Instant::now();
    let (convergence_artifact, convergence_perf) =
        build_rv64im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local_with_perf(
            &phase0_binding_surface,
            &opening_phase0_artifact.eval_claim_bundle,
            &claim_witnesses,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM Nightstream opening convergence artifact build failed: {err}"
            ))
        })?;
    let opening_artifact_convergence_ms = opening_phase0_binding_surface_ms + elapsed_ms(started);

    let started = Instant::now();
    super::opening_artifact::build_rv64im_opening_artifact_from_trusted_local_phase0_and_convergence_artifacts(
        &opening_phase0_artifact,
        &convergence_artifact,
    )?;
    let opening_artifact_wrap_ms = elapsed_ms(started);
    let opening_artifact_ms = opening_artifact_convergence_ms + opening_artifact_wrap_ms;

    let started = Instant::now();
    let (bridge_artifact, hybrid_side_bridge_relation) =
        super::hybrid_side_bridge_decider::build_rv64im_hybrid_side_bridge_material_from_accepted_artifact(
            &statement,
            &main_residual_proof.bridge_handoff_digests,
            &accepted_artifact.statement,
            &side_proof_bundle,
            accepted_artifact,
        )?;
    let hybrid_side_bridge_prepare_ms = elapsed_ms(started);

    let started = Instant::now();
    super::hybrid_side_bridge_decider::prewarm_rv64im_hybrid_side_bridge_backend_shell_cache_for_relation(
        &hybrid_side_bridge_relation,
    )?;
    let hybrid_side_bridge_backend_shell_setup_ms = elapsed_ms(started);

    let started = Instant::now();
    let hybrid_side_bridge_backend_proof =
        super::hybrid_side_bridge_decider::prove_rv64im_hybrid_side_bridge_backend_proof_from_decider_relation(
            &hybrid_side_bridge_relation,
        )?;
    let hybrid_side_bridge_backend_proof_ms = elapsed_ms(started);

    let started = Instant::now();
    let hybrid_side_bridge_artifact = super::hybrid_side_bridge_decider::assemble_rv64im_hybrid_side_bridge_artifact(
        bridge_artifact,
        hybrid_side_bridge_backend_proof,
    );
    let hybrid_side_bridge_finalize_ms = elapsed_ms(started);
    let hybrid_side_bridge_artifact_ms = hybrid_side_bridge_prepare_ms
        + hybrid_side_bridge_backend_shell_setup_ms
        + hybrid_side_bridge_backend_proof_ms
        + hybrid_side_bridge_finalize_ms;

    let started = Instant::now();
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_decider_proof_digest: main_decider_proof.expected_digest(),
        main_residual_proof_digest: main_residual_proof.expected_digest(),
        side_bridge_artifact_digest: hybrid_side_bridge_artifact.digest,
        linkage_artifact_digest: linkage_artifact.digest,
    };
    statement.proof_binding_root = nightstream_proof_binding_root(statement.core_digest(), &proof_binding_inputs);
    let proof_binding_root_ms = elapsed_ms(started);

    let perf = Rv64imNightstreamVerifiedSeamsBuildPerf {
        final_surface_guard_ms,
        decider_relation_ms,
        linkage_claims_ms,
        main_decider_proof_ms,
        main_residual_proof_ms,
        linkage_artifact_ms,
        linkage_root_ms,
        statement_ms,
        bind_side_bundle_ms,
        opening_phase0_artifact_ms,
        opening_phase0_claim_witnesses_ms,
        opening_phase0_relation_artifact_ms,
        opening_phase0_packed_columns_ms: claim_witness_perf.packed_columns_ms,
        opening_phase0_commitment_vector_ms: claim_witness_perf.commitment_vector_ms,
        opening_phase0_commitment_params_ms: claim_witness_perf.commitment_params_ms,
        opening_phase0_commitment_committer_ms: claim_witness_perf.commitment_committer_ms,
        opening_phase0_commitment_mats_ms: claim_witness_perf.commitment_mats_ms,
        opening_phase0_commitment_commit_many_ms: claim_witness_perf.commitment_commit_many_ms,
        opening_phase0_commitment_root_ms: claim_witness_perf.commitment_root_ms,
        opening_phase0_opened_object_id_ms: claim_witness_perf.opened_object_id_ms,
        opening_phase0_opened_object_total_ms: claim_witness_perf.opened_object_total_ms,
        opening_phase0_binding_digest_ms: claim_witness_perf.binding_digest_ms,
        opening_phase0_point_derivation_ms: claim_witness_perf.point_derivation_ms,
        opening_phase0_payload_eval_ms: claim_witness_perf.payload_eval_ms,
        opening_phase0_claim_build_ms: claim_witness_perf.claim_build_ms,
        opening_phase0_slot_claims_total_ms: claim_witness_perf.slot_claims_total_ms,
        opening_artifact_ms,
        opening_convergence_total_ms: convergence_perf.total_ms,
        opening_convergence_phase1_ms: convergence_perf.phase1_results_ms,
        opening_convergence_phase2_ms: convergence_perf.phase2_ms,
        opening_convergence_final_openings_ms: convergence_perf.final_openings_ms,
        opening_convergence_final_openings_witness_map_ms: convergence_perf.final_openings_witness_map_ms,
        opening_convergence_final_openings_representative_ms: convergence_perf.final_openings_representative_ms,
        opening_convergence_final_openings_commitment_validate_ms: convergence_perf
            .final_openings_commitment_validate_ms,
        opening_convergence_final_openings_opened_commitment_digest_ms: convergence_perf
            .final_openings_opened_commitment_digest_ms,
        opening_convergence_final_openings_opening_proof_digest_ms: convergence_perf
            .final_openings_opening_proof_digest_ms,
        opening_convergence_final_openings_target_build_ms: convergence_perf.final_openings_target_build_ms,
        opening_convergence_digest_ms: convergence_perf.digest_ms,
        opening_artifact_wrap_ms,
        hybrid_side_bridge_prepare_ms,
        hybrid_side_bridge_backend_shell_setup_ms,
        hybrid_side_bridge_backend_proof_ms,
        hybrid_side_bridge_artifact_ms,
        proof_binding_root_ms,
        total_ms: elapsed_ms(total_started),
    };
    Ok((
        (
            statement,
            Rv64imNightstreamProof {
                main_decider_proof,
                main_residual_proof,
                hybrid_side_bridge_artifact,
                linkage_artifact,
            },
        ),
        perf,
    ))
}

pub fn build_rv64im_nightstream_from_public_proof_with_perf(
    proof: &Rv64imProof,
) -> Result<
    (
        (NightstreamStatement, Rv64imNightstreamProof),
        Rv64imNightstreamBuildPerf,
    ),
    SimpleKernelError,
> {
    let (published_seam, seam_perf) = crate::rv64im::decider::build_rv64im_published_proof_seam_with_perf(proof)?;
    build_rv64im_nightstream_from_published_proof_seam_with_perf(&published_seam, &seam_perf)
}

pub fn build_rv64im_nightstream_from_published_proof_seam_with_perf(
    published_seam: &crate::rv64im::decider::Rv64imPublishedProofSeam,
    seam_perf: &crate::rv64im::decider::Rv64imPublishedProofSeamBuildPerf,
) -> Result<
    (
        (NightstreamStatement, Rv64imNightstreamProof),
        Rv64imNightstreamBuildPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();
    let crate::rv64im::decider::Rv64imPublishedProofSeam {
        accepted_artifact: artifact,
        final_statement,
        final_proof,
        final_component_digests,
        verified_kernel,
        ..
    } = published_seam;

    let started = Instant::now();
    let side_proof_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact_and_kernel_export(artifact, &final_proof.kernel_export)?;
    let side_proof_bundle_ms = elapsed_ms(started);

    let ((statement, nightstream_proof), verified_seams) = build_rv64im_nightstream_from_verified_seams_with_perf(
        artifact,
        rv64im_verifier_context_digest(artifact.statement.root_params_id),
        final_statement,
        final_proof,
        final_component_digests,
        verified_kernel,
        side_proof_bundle,
    )?;

    Ok((
        (statement, nightstream_proof),
        Rv64imNightstreamBuildPerf {
            accepted_artifact_ms: seam_perf.accepted_artifact_ms,
            final_statement_ms: seam_perf.final_statement_ms,
            final_statement_kernel_export_ms: seam_perf.final_statement_kernel_export_ms,
            final_statement_recursive_proof_ms: seam_perf.final_statement_recursive_proof_ms,
            final_statement_recursive_prepare_inputs_ms: seam_perf.final_statement_recursive_prepare_inputs_ms,
            final_statement_recursive_ccs_bind_ms: seam_perf.final_statement_recursive_ccs_bind_ms,
            final_statement_recursive_ccs_sample_challenges_ms: seam_perf
                .final_statement_recursive_ccs_sample_challenges_ms,
            final_statement_recursive_ccs_fe_sumcheck_ms: seam_perf.final_statement_recursive_ccs_fe_sumcheck_ms,
            final_statement_recursive_ccs_nc_sumcheck_ms: seam_perf.final_statement_recursive_ccs_nc_sumcheck_ms,
            final_statement_recursive_ccs_output_materialize_ms: seam_perf
                .final_statement_recursive_ccs_output_materialize_ms,
            final_statement_recursive_ccs_ms: seam_perf.final_statement_recursive_ccs_ms,
            final_statement_recursive_dims_ms: seam_perf.final_statement_recursive_dims_ms,
            final_statement_recursive_rlc_prepare_ms: seam_perf.final_statement_recursive_rlc_prepare_ms,
            final_statement_recursive_rlc_ms: seam_perf.final_statement_recursive_rlc_ms,
            final_statement_recursive_dec_split_ms: seam_perf.final_statement_recursive_dec_split_ms,
            final_statement_recursive_dec_commit_ms: seam_perf.final_statement_recursive_dec_commit_ms,
            final_statement_recursive_dec_ms: seam_perf.final_statement_recursive_dec_ms,
            final_statement_folded_digest_ms: seam_perf.final_statement_folded_digest_ms,
            final_statement_final_proof_ms: seam_perf.final_statement_final_proof_ms,
            final_statement_statement_digest_ms: seam_perf.final_statement_statement_digest_ms,
            side_proof_bundle_ms,
            verified_seams,
            total_ms: elapsed_ms(total_started),
        },
    ))
}
