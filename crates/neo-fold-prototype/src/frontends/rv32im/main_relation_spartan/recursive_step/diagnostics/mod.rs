use std::io::{self, Write};
use std::time::Instant;

use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};
use neo_math::{KExtensions, K};
use neo_reductions::engines::utils::me_digest_poseidon_into;
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices_with_sparse_cache};
use neo_reductions::engines::utils::{
    PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG, PI_CCS_SUMCHECK_INITIAL_RAW_TAG, PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG,
};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;

use super::*;
use crate::rv32im::f_prime::Rv32imMainRecursionFPrimeAdvice;
use crate::rv32im::final_relation::RV32IM_CHUNK_DONE_RAW_TAG;
use crate::rv32im::kernel::{rv32im_cached_root_main_lane_context, rv32im_cached_root_main_lane_optimized_cache};
use crate::rv32im::main_relation_spartan::fingerprint_cs::FingerprintCS;
use crate::rv32im::main_relation_spartan::recursive_cover::{
    alloc_recursive_carried_projection_claims, alloc_recursive_cover_claims, alloc_recursive_cover_state,
    debug_measure_recursive_accumulator_instance_digest_circuit_from_claims_aux,
    recursive_accumulator_instance_digest_circuit_from_claims, Rv32imRecursiveCoverClaimVar,
};
use crate::rv32im::main_relation_spartan::Rv32imMainRecursionFPrimePayload;
use crate::rv32im::main_relation_spartan::{rv32im_main_relation_delta, Rv32imClaimBundle};
use crate::spartan_backend::{Rv32imDeciderEngine, ShapeCS, SpartanCircuit, SpartanF, SpartanShape, SplitR1CSShape};
use crate::superneo_circuit::claim::{enforce_claim_projection_eq_native, me_digest_poseidon};
use crate::superneo_circuit::transcript::Poseidon2TranscriptCircuit;

fn stage_err(stage: &str, err: impl ToString) -> Rv32imMainRecursionStepSpartanError {
    Rv32imMainRecursionStepSpartanError::Prepare(format!("{stage}: {}", err.to_string()))
}

fn emit_trace(trace_prefix: &str, label: &str, elapsed_ms: f64) {
    eprintln!("{trace_prefix}.{label}={elapsed_ms:.2}ms");
    let _ = io::stderr().flush();
}

fn push_constraint_delta(
    stages: &mut Vec<Rv32imNamedConstraintDelta>,
    previous: &mut usize,
    current: usize,
    name: impl Into<String>,
) {
    let delta = current.saturating_sub(*previous);
    *previous = current;
    stages.push(Rv32imNamedConstraintDelta {
        name: name.into(),
        delta,
    });
}

fn ensure_stage_satisfied(
    cs: &TestConstraintSystem<SpartanF>,
    stage: &str,
) -> Result<(), Rv32imMainRecursionStepSpartanError> {
    if cs.is_satisfied() {
        Ok(())
    } else {
        Err(stage_err(
            stage,
            cs.which_is_unsatisfied().unwrap_or("unknown constraint"),
        ))
    }
}

fn alloc_live_state_in_projection_claims<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    _witness: &Rv32imMainRecursionFPrimeAdvice,
    payload: &Rv32imMainRecursionFPrimePayload,
    label: &str,
) -> Result<Vec<Rv32imRecursiveCoverClaimVar>, SynthesisError> {
    alloc_recursive_carried_projection_claims(cs, &payload.state_in_claims, label)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rv32imMainRecursionStepSpartanShapeSynthesisMetrics {
    pub shared_ms: f64,
    pub precommitted_ms: f64,
    pub synthesize_ms: f64,
    pub num_inputs: usize,
    pub num_aux: usize,
    pub num_constraints: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imMainRecursionStepChunkReplayFingerprint {
    pub after_state_cover: String,
    pub after_chunk_meta: String,
    pub after_pi_ccs: String,
    pub after_synthetic_relation_io: String,
    pub after_pi_rlc_parent_claim: String,
    pub after_pi_rlc_rhos: String,
    pub after_pi_rlc_rho_mats: String,
    pub after_pi_rlc_public: String,
    pub after_pi_rlc: String,
    pub after_chunk_body: String,
    pub after_chunk_replay: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imMainRecursionStepStageAuxCounts {
    pub after_private_witness_inputs: usize,
    pub after_alloc_cover_states: usize,
    pub after_bind_state_and_pc: usize,
    pub after_chunk_replay: usize,
    pub after_inactive_side_lane_and_x_out: usize,
    pub after_public_output_eq: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imMainRecursionStepChunkReplayAuxCounts {
    pub after_state_cover: usize,
    pub after_chunk_meta: usize,
    pub after_pi_ccs: usize,
    pub after_synthetic_relation_io: usize,
    pub after_pi_rlc_parent_claim: usize,
    pub after_pi_rlc_rhos: usize,
    pub after_pi_rlc_rho_mats: usize,
    pub after_pi_rlc_public: usize,
    pub after_pi_rlc: usize,
    pub after_chunk_body: usize,
    pub after_chunk_replay: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imMainRecursionStepChunkReplayTailAuxCounts {
    pub after_state_out_projection_eq: usize,
    pub after_expected_digest: usize,
    pub after_chunk_done: usize,
    pub after_transcript_state_eq: usize,
    pub after_transcript_absorbed_eq: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiCcsStageAuxCounts {
    pub after_bind_header: usize,
    pub after_bind_me_inputs: usize,
    pub after_sample_challenges: usize,
    pub after_alloc_fresh_claims: usize,
    pub after_fe_sumcheck: usize,
    pub after_nc_sumcheck: usize,
    pub after_fold_digest: usize,
    pub after_alloc_outputs: usize,
    pub after_output_binding: usize,
    pub after_terminal_fe: usize,
    pub after_terminal_nc: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiCcsStageConstraintCounts {
    pub after_bind_header: usize,
    pub after_bind_me_inputs: usize,
    pub after_sample_challenges: usize,
    pub after_alloc_fresh_claims: usize,
    pub after_fe_sumcheck: usize,
    pub after_nc_sumcheck: usize,
    pub after_fold_digest: usize,
    pub after_alloc_outputs: usize,
    pub after_output_binding: usize,
    pub after_terminal_fe: usize,
    pub after_terminal_nc: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiCcsBindMeInputsAuxBreakdown {
    pub after_bind_header: usize,
    pub after_claim_digests: Vec<usize>,
    pub after_bind_digests: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiCcsSumcheckConstraintBreakdown {
    pub fe_cover_round_lengths: Vec<u64>,
    pub fe_effective_round_lengths: Vec<usize>,
    pub fe_stages: Vec<Rv32imNamedConstraintDelta>,
    pub nc_cover_round_lengths: Vec<u64>,
    pub nc_effective_round_lengths: Vec<usize>,
    pub nc_stages: Vec<Rv32imNamedConstraintDelta>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiCcsStageFingerprint {
    pub after_bind_header: String,
    pub after_bind_me_inputs: String,
    pub after_sample_challenges: String,
    pub after_alloc_fresh_claims: String,
    pub after_fe_sumcheck: String,
    pub after_nc_sumcheck: String,
    pub after_fold_digest: String,
    pub after_alloc_outputs: String,
    pub after_output_binding: String,
    pub after_terminal_fe: String,
    pub after_terminal_nc: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiRlcPublicConstraintBreakdown {
    pub shared_point_constraints: usize,
    pub x_constraints: usize,
    pub c_constraints: usize,
    pub y_ring_constraints: usize,
    pub y_zcol_constraints: usize,
    pub aux_constraints: usize,
    pub total_constraints: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imNamedConstraintDelta {
    pub name: String,
    pub delta: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiRlcPublicStageBreakdown {
    pub stages: Vec<Rv32imNamedConstraintDelta>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imMainRecursionStepChunkReplayTailDigestAuxBreakdown {
    pub after_header: usize,
    pub claim_after_digests: Vec<usize>,
    pub after_outer_hash: usize,
}

mod chunk_replay_fingerprint;
mod pi_ccs_aux;
mod pi_ccs_breakdowns;
mod pi_ccs_constraints;
mod shape_checks;
mod stages_and_pi_rlc;
mod tail_and_fingerprints;

pub use chunk_replay_fingerprint::debug_measure_rv32im_main_recursion_step_chunk_replay_fingerprint;
pub use pi_ccs_aux::debug_measure_rv32im_main_recursion_step_pi_ccs_aux_counts;
pub use pi_ccs_breakdowns::{
    debug_measure_rv32im_main_recursion_step_pi_ccs_bind_me_inputs_aux_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_ccs_sumcheck_constraint_breakdown,
};
pub use pi_ccs_constraints::debug_measure_rv32im_main_recursion_step_pi_ccs_constraint_counts;
pub use shape_checks::{
    debug_check_rv32im_main_recursion_step_spartan_fresh_output_accumulator_digest_parity,
    debug_check_rv32im_main_recursion_step_spartan_live_claim_me_digest_parity,
    debug_measure_rv32im_main_recursion_step_shape_only_circuit_shape,
    debug_measure_rv32im_main_recursion_step_spartan_shape_synthesis,
    debug_profile_rv32im_main_recursion_step_chunk_replay_stages,
    debug_trace_rv32im_main_recursion_step_fingerprint_synthesize,
    debug_trace_rv32im_main_recursion_step_shape_only_circuit_shape_measurement,
    debug_trace_rv32im_main_recursion_step_shape_only_fingerprint_synthesize,
    debug_trace_rv32im_main_recursion_step_spartan_circuit_shape_measurement,
    debug_trace_rv32im_main_recursion_step_spartan_shape_synthesis,
};
pub use stages_and_pi_rlc::{
    debug_measure_rv32im_main_recursion_step_chunk_replay_aux_counts,
    debug_measure_rv32im_main_recursion_step_pi_rlc_public_constraint_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_rlc_public_stage_breakdown,
    debug_measure_rv32im_main_recursion_step_spartan_commitment_key,
    debug_measure_rv32im_main_recursion_step_stage_aux_counts,
};
pub use tail_and_fingerprints::{
    debug_measure_rv32im_main_recursion_step_chunk_replay_tail_aux_counts,
    debug_measure_rv32im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint,
};
