//! Owns compact build-time diagnostics for the RV64IM Nightstream boundary.

use std::time::Instant;

use crate::rv64im::kernel::{
    build_rv64im_eval_claim_witnesses_from_accepted_artifact_with_perf,
    verify_rv64im_kernel_export_proof_with_relation_output,
};
use crate::rv64im::{setup_rv64im_ivc_snark_from_final_cached, Rv64imCompressedMainProof, Rv64imPublishedProofSeam};

use super::side_eval_claim_relation::rebind_phase0_claim_witnesses_to_side_bundle;
use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv64imNightstreamSeamBuildPerf {
    pub final_surface_guard_ms: f64,
    pub main_proof_ms: f64,
    pub statement_ms: f64,
    pub bind_side_statement_core_ms: f64,
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
    pub opening_support_bundle_ms: f64,
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
    pub opening_support_wrap_ms: f64,
    pub side_binding_prepare_ms: f64,
    pub side_binding_setup_ms: f64,
    pub side_binding_prove_ms: f64,
    pub side_binding_ms: f64,
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
    pub side_support_bundle_ms: f64,
    pub seam_build: Rv64imNightstreamSeamBuildPerf,
    pub total_ms: f64,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn guard_locally_built_compact_main_proof(
    accepted_artifact: &Rv64imAcceptedProofArtifact,
    main_proof: &Rv64imCompressedMainProof,
    final_statement: &crate::rv64im::final_relation::Rv64imFinalStatement,
    final_proof: &crate::rv64im::final_relation::Rv64imFinalBuildProof,
) -> Result<(), SimpleKernelError> {
    if final_statement.public_statement_digest != accepted_artifact.statement.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream local final statement does not match the carried accepted-artifact statement".into(),
        ));
    }
    let (_, verified_kernel) = verify_rv64im_kernel_export_proof_with_relation_output(&final_proof.kernel_export)?;
    let expected_main_proof =
        Rv64imCompressedMainProof::from_final_artifacts(final_statement, final_proof, verified_kernel.final_pc)?;
    if main_proof != &expected_main_proof {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream compact main proof does not match the rebuilt local final seam".into(),
        ));
    }
    Ok(())
}

pub(super) fn build_rv64im_nightstream_from_published_seam_with_perf(
    root_params_id: [u8; 32],
    published_seam: &Rv64imPublishedProofSeam,
    main_proof_ms: f64,
    side_proof_bundle: Rv64imSideProofBundle,
) -> Result<
    (
        (NightstreamStatement, Rv64imNightstreamProof),
        Rv64imNightstreamSeamBuildPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();
    let accepted_artifact = &published_seam.accepted_artifact;

    let started = Instant::now();
    let final_statement = published_seam.rebuild_final_statement()?;
    let final_proof = published_seam.final_proof()?;
    guard_locally_built_compact_main_proof(
        accepted_artifact,
        &published_seam.main_proof,
        &final_statement,
        &final_proof,
    )?;
    let final_surface_guard_ms = elapsed_ms(started);

    let compressed_main_proof = published_seam.main_proof.clone();

    let started = Instant::now();
    let ivc_recursion_snark_keys = setup_rv64im_ivc_snark_from_final_cached(&final_statement, &final_proof)?;
    let verifier_context_digest = rv64im_verifier_context_digest(
        root_params_id,
        compressed_main_proof.published_statement(),
        &ivc_recursion_snark_keys.as_ref().1,
    )?;
    let mut statement = build_rv64im_nightstream_statement_from_published_statement(
        verifier_context_digest,
        compressed_main_proof.published_statement(),
        [0; 32],
    )?;
    let statement_ms = elapsed_ms(started);

    let started = Instant::now();
    let side_proof_bundle =
        bind_rv64im_side_proof_bundle_to_statement_core(&side_proof_bundle, statement.core_digest())?;
    let bind_side_statement_core_ms = elapsed_ms(started);

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
    let _opening_artifact =
        super::opening_artifact::build_rv64im_opening_artifact_from_trusted_local_phase0_and_convergence_artifacts(
            &opening_phase0_artifact,
            &convergence_artifact,
        )?;
    let opening_support_wrap_ms = elapsed_ms(started);
    let opening_support_bundle_ms = opening_artifact_convergence_ms + opening_support_wrap_ms;

    let started = Instant::now();
    let side_opening =
        side_eval_claim_relation::build_rv64im_side_eval_claim_artifact_from_claim_witnesses_and_side_bundle(
            &accepted_artifact.statement,
            &side_proof_bundle,
            &claim_witnesses,
        )?;
    let side_opening_statement = super::side_opening_relation::build_rv64im_side_opening_relation_statement(
        &accepted_artifact.statement,
        &side_proof_bundle,
    )?;
    let side_opening_witness =
        super::side_opening_relation::build_rv64im_side_opening_relation_witness_from_accepted_artifact(
            accepted_artifact,
        );
    let side_public = build_rv64im_side_opening_public(&side_proof_bundle, &side_opening)?;
    let side_statement = build_rv64im_side_binding_statement(&statement, &accepted_artifact.statement, &side_public)?;
    let side_binding_prepare_ms = elapsed_ms(started);

    let started = Instant::now();
    let side_opening_keys = super::side_opening_spartan::setup_rv64im_side_opening_spartan_cached(
        &side_opening_statement,
        &side_opening_witness,
    )?;
    let side_keys = setup_rv64im_side_binding_cached(&side_statement, &side_public)?;
    let side_binding_setup_ms = elapsed_ms(started);

    let started = Instant::now();
    let side_opening_proof = super::side_opening_spartan::prove_rv64im_side_opening_spartan(
        &side_opening_keys.as_ref().0,
        &side_opening_statement,
        &side_opening_witness,
    )?;
    let side_binding_proof =
        prove_rv64im_side_binding(&side_keys.as_ref().0, &side_statement, &side_public, &claim_witnesses)?;
    let side_binding_prove_ms = elapsed_ms(started);
    let side_proof = Rv64imSideProof {
        opening_public: side_public,
        opening_statement: side_opening_statement,
        opening: side_opening_proof,
        binding: side_binding_proof,
    };
    let side_binding_ms = side_binding_prepare_ms + side_binding_setup_ms + side_binding_prove_ms;

    let started = Instant::now();
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_proof_digest: rv64im_main_nightstream_proof_digest(&compressed_main_proof),
        side_proof_digest: side_proof.expected_digest(),
        public_statement_digest: accepted_artifact.statement.digest,
    };
    statement.proof_binding_root = nightstream_proof_binding_root(statement.core_digest(), &proof_binding_inputs);
    let proof_binding_root_ms = elapsed_ms(started);

    let perf = Rv64imNightstreamSeamBuildPerf {
        final_surface_guard_ms,
        main_proof_ms,
        statement_ms,
        bind_side_statement_core_ms,
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
        opening_support_bundle_ms,
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
        opening_support_wrap_ms,
        side_binding_prepare_ms,
        side_binding_setup_ms,
        side_binding_prove_ms,
        side_binding_ms,
        proof_binding_root_ms,
        total_ms: elapsed_ms(total_started),
    };
    Ok((
        (
            statement,
            Rv64imNightstreamProof {
                main_proof: compressed_main_proof,
                side_proof,
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
    let (published_seam, seam_perf) = crate::rv64im::audit::build_rv64im_published_proof_seam_with_perf(proof)?;
    build_rv64im_nightstream_from_published_proof_seam_with_perf(&published_seam, &seam_perf)
}

pub fn build_rv64im_nightstream_from_published_proof_seam_with_perf(
    published_seam: &crate::rv64im::audit::Rv64imPublishedProofSeam,
    seam_perf: &crate::rv64im::audit::Rv64imPublishedProofSeamBuildPerf,
) -> Result<
    (
        (NightstreamStatement, Rv64imNightstreamProof),
        Rv64imNightstreamBuildPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();
    let artifact = &published_seam.accepted_artifact;
    let main_proof_ms = seam_perf.main_proof_ms;

    let started = Instant::now();
    let side_proof_bundle = build_rv64im_side_proof_bundle_from_accepted_artifact_and_kernel_export(
        artifact,
        published_seam.kernel_export(),
    )?;
    let side_support_bundle_ms = elapsed_ms(started);

    let ((statement, nightstream_proof), seam_build) = build_rv64im_nightstream_from_published_seam_with_perf(
        artifact.statement.root_params_id,
        published_seam,
        main_proof_ms,
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
            side_support_bundle_ms,
            seam_build,
            total_ms: elapsed_ms(total_started),
        },
    ))
}
