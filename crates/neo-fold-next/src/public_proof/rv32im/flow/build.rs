//! Owns the prover-side flow for building an RV32IM published proof.

use std::time::Instant;

use crate::public_proof::{nightstream_proof_binding_root, NightstreamProofBindingInputs, NightstreamStatement};
use crate::rv32im::kernel::{
    build_rv32im_eval_claim_witnesses_from_accepted_artifact_with_perf,
    build_rv32im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local_with_perf,
    verify_rv32im_kernel_export_proof_with_relation_output, Rv32imAcceptedProofArtifact, Rv32imProof,
    SimpleKernelError,
};
use crate::rv32im::{setup_rv32im_ivc_snark_from_final_cached, Rv32imCompressedMainProof, Rv32imPublishedProofSeam};

use super::perf::{elapsed_ms, Rv32imNightstreamBuildPerf, Rv32imNightstreamSeamBuildPerf};
use crate::public_proof::rv32im::authoritative_side::{
    build_rv32im_side_binding_statement, build_rv32im_side_opening_public,
};
use crate::public_proof::rv32im::opening_artifact;
use crate::public_proof::rv32im::proof::{
    rv32im_main_nightstream_proof_digest, Rv32imNightstreamProof, Rv32imSideProof,
};
use crate::public_proof::rv32im::side_bridges::Rv32imSideProofBundle;
use crate::public_proof::rv32im::side_bundle::{
    bind_rv32im_side_proof_bundle_to_statement_core,
    build_rv32im_side_proof_bundle_from_accepted_artifact_and_kernel_export,
};
use crate::public_proof::rv32im::side_eval_claim_relation::{self, rebind_phase0_claim_witnesses_to_side_bundle};
use crate::public_proof::rv32im::side_opening_relation::{
    build_rv32im_side_opening_relation_statement, build_rv32im_side_opening_relation_witness_from_accepted_artifact,
};
use crate::public_proof::rv32im::side_opening_spartan::{
    prove_rv32im_side_opening_spartan, setup_rv32im_side_opening_spartan_cached,
};
use crate::public_proof::rv32im::side_relation_spartan::{prove_rv32im_side_binding, setup_rv32im_side_binding_cached};
use crate::public_proof::rv32im::statement::{
    build_rv32im_nightstream_statement_from_published_statement, rv32im_verifier_context_digest,
};

fn guard_locally_built_compact_main_proof(
    accepted_artifact: &Rv32imAcceptedProofArtifact,
    main_proof: &Rv32imCompressedMainProof,
    final_statement: &crate::rv32im::final_relation::Rv32imFinalStatement,
    final_proof: &crate::rv32im::final_relation::Rv32imFinalBuildProof,
) -> Result<(), SimpleKernelError> {
    if final_statement.public_statement_digest != accepted_artifact.statement.digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream local final statement does not match the carried accepted-artifact statement".into(),
        ));
    }
    let (_, verified_kernel) = verify_rv32im_kernel_export_proof_with_relation_output(&final_proof.kernel_export)?;
    let expected_main_proof =
        Rv32imCompressedMainProof::from_final_artifacts(final_statement, final_proof, verified_kernel.final_pc)?;
    if main_proof != &expected_main_proof {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream compact main proof does not match the rebuilt local final seam".into(),
        ));
    }
    Ok(())
}

pub(super) fn build_rv32im_nightstream_from_published_seam_with_perf(
    root_params_id: [u8; 32],
    published_seam: &Rv32imPublishedProofSeam,
    main_proof_ms: f64,
    side_proof_bundle: Rv32imSideProofBundle,
) -> Result<
    (
        (NightstreamStatement, Rv32imNightstreamProof),
        Rv32imNightstreamSeamBuildPerf,
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
    let ivc_recursion_snark_keys = setup_rv32im_ivc_snark_from_final_cached(&final_statement, &final_proof)?;
    let verifier_context_digest = rv32im_verifier_context_digest(
        root_params_id,
        compressed_main_proof.published_statement(),
        &ivc_recursion_snark_keys.as_ref().1,
    )?;
    let mut statement = build_rv32im_nightstream_statement_from_published_statement(
        verifier_context_digest,
        compressed_main_proof.published_statement(),
        [0; 32],
    )?;
    let statement_ms = elapsed_ms(started);

    let started = Instant::now();
    let side_proof_bundle =
        bind_rv32im_side_proof_bundle_to_statement_core(&side_proof_bundle, statement.core_digest())?;
    let bind_side_statement_core_ms = elapsed_ms(started);

    let started = Instant::now();
    let (accepted_claim_witnesses, claim_witness_perf) =
        build_rv32im_eval_claim_witnesses_from_accepted_artifact_with_perf(accepted_artifact)?;
    let claim_witnesses = rebind_phase0_claim_witnesses_to_side_bundle(&side_proof_bundle, &accepted_claim_witnesses)?;
    let opening_phase0_claim_witnesses_ms = elapsed_ms(started);

    let started = Instant::now();
    let opening_phase0_artifact =
        side_eval_claim_relation::build_rv32im_side_eval_claim_artifact_from_claim_witnesses_and_trusted_side_bundle(
            &accepted_artifact.statement,
            &side_proof_bundle,
            &claim_witnesses,
        )?;
    let opening_phase0_relation_artifact_ms = elapsed_ms(started);
    let opening_phase0_artifact_ms = opening_phase0_claim_witnesses_ms + opening_phase0_relation_artifact_ms;

    let started = Instant::now();
    let phase0_binding_surface =
        side_eval_claim_relation::build_rv32im_phase0_binding_surface_from_side_bundle(&side_proof_bundle);
    let opening_phase0_binding_surface_ms = elapsed_ms(started);

    let started = Instant::now();
    let (convergence_artifact, convergence_perf) =
        build_rv32im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local_with_perf(
            &phase0_binding_surface,
            &opening_phase0_artifact.eval_claim_bundle,
            &claim_witnesses,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV32IM Nightstream opening convergence artifact build failed: {err}"
            ))
        })?;
    let opening_artifact_convergence_ms = opening_phase0_binding_surface_ms + elapsed_ms(started);

    let started = Instant::now();
    let _opening_artifact =
        opening_artifact::build_rv32im_opening_artifact_from_trusted_local_phase0_and_convergence_artifacts(
            &opening_phase0_artifact,
            &convergence_artifact,
        )?;
    let opening_support_wrap_ms = elapsed_ms(started);
    let opening_support_bundle_ms = opening_artifact_convergence_ms + opening_support_wrap_ms;

    let started = Instant::now();
    let side_opening =
        side_eval_claim_relation::build_rv32im_side_eval_claim_artifact_from_claim_witnesses_and_side_bundle(
            &accepted_artifact.statement,
            &side_proof_bundle,
            &claim_witnesses,
        )?;
    let side_opening_statement =
        build_rv32im_side_opening_relation_statement(&accepted_artifact.statement, &side_proof_bundle)?;
    let side_opening_witness = build_rv32im_side_opening_relation_witness_from_accepted_artifact(accepted_artifact);
    let side_public = build_rv32im_side_opening_public(&side_proof_bundle, &side_opening)?;
    let side_statement = build_rv32im_side_binding_statement(&statement, &accepted_artifact.statement, &side_public)?;
    let side_binding_prepare_ms = elapsed_ms(started);

    let started = Instant::now();
    let side_opening_keys = setup_rv32im_side_opening_spartan_cached(&side_opening_statement, &side_opening_witness)?;
    let side_keys = setup_rv32im_side_binding_cached(&side_statement, &side_public)?;
    let side_binding_setup_ms = elapsed_ms(started);

    let started = Instant::now();
    let side_opening_proof = prove_rv32im_side_opening_spartan(
        &side_opening_keys.as_ref().0,
        &side_opening_statement,
        &side_opening_witness,
    )?;
    let side_binding_proof =
        prove_rv32im_side_binding(&side_keys.as_ref().0, &side_statement, &side_public, &claim_witnesses)?;
    let side_binding_prove_ms = elapsed_ms(started);
    let side_proof = Rv32imSideProof::from_parts(
        side_public,
        side_opening_statement,
        side_opening_proof,
        side_binding_proof,
    );
    let side_binding_ms = side_binding_prepare_ms + side_binding_setup_ms + side_binding_prove_ms;

    let started = Instant::now();
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_proof_digest: rv32im_main_nightstream_proof_digest(&compressed_main_proof),
        side_proof_digest: side_proof.expected_digest(),
        public_statement_digest: accepted_artifact.statement.digest,
    };
    statement.proof_binding_root = nightstream_proof_binding_root(statement.core_digest(), &proof_binding_inputs);
    let proof_binding_root_ms = elapsed_ms(started);

    let perf = Rv32imNightstreamSeamBuildPerf {
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
            Rv32imNightstreamProof::from_parts(compressed_main_proof, side_proof),
        ),
        perf,
    ))
}

pub fn build_rv32im_nightstream_from_public_proof_with_perf(
    proof: &Rv32imProof,
) -> Result<
    (
        (NightstreamStatement, Rv32imNightstreamProof),
        Rv32imNightstreamBuildPerf,
    ),
    SimpleKernelError,
> {
    let (published_seam, seam_perf) = crate::rv32im::audit::build_rv32im_published_proof_seam_with_perf(proof)?;
    build_rv32im_nightstream_from_published_proof_seam_with_perf(&published_seam, &seam_perf)
}

pub fn build_rv32im_nightstream_from_published_proof_seam_with_perf(
    published_seam: &crate::rv32im::audit::Rv32imPublishedProofSeam,
    seam_perf: &crate::rv32im::audit::Rv32imPublishedProofSeamBuildPerf,
) -> Result<
    (
        (NightstreamStatement, Rv32imNightstreamProof),
        Rv32imNightstreamBuildPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();
    let artifact = &published_seam.accepted_artifact;
    let main_proof_ms = seam_perf.main_proof_ms;

    let started = Instant::now();
    let side_proof_bundle = build_rv32im_side_proof_bundle_from_accepted_artifact_and_kernel_export(
        artifact,
        published_seam.kernel_export(),
    )?;
    let side_support_bundle_ms = elapsed_ms(started);

    let ((statement, nightstream_proof), seam_build) = build_rv32im_nightstream_from_published_seam_with_perf(
        artifact.statement.root_params_id,
        published_seam,
        main_proof_ms,
        side_proof_bundle,
    )?;

    Ok((
        (statement, nightstream_proof),
        Rv32imNightstreamBuildPerf {
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
