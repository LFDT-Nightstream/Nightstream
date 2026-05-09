//! Owns timing data for the RV32IM published-proof boundary.

use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32imNightstreamSeamBuildPerf {
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
pub struct Rv32imNightstreamBuildPerf {
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
    pub seam_build: Rv32imNightstreamSeamBuildPerf,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rv32imNightstreamVerifyPerf {
    pub carried_boundary_ms: f64,
    pub statement_binding_ms: f64,
    pub side_proof_ms: f64,
    pub remaining_side_surfaces_ms: f64,
    pub main_proof_ms: f64,
    pub total_ms: f64,
}

impl Rv32imNightstreamVerifyPerf {
    pub fn before_main_proof_ms(&self) -> f64 {
        self.total_ms - self.main_proof_ms
    }
}

pub(super) fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}
