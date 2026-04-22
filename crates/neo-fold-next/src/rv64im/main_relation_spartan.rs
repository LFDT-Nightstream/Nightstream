//! Owns the shared RV64IM Spartan substrates used by chunk-step IVC, recursive
//! F' replay, and the remaining fixed-shape diagnostics.
//!
//! The live Goal 3 path no longer routes through a standalone full-trace
//! `R_main^SN` circuit here. Terminal compression is owned in `ivc_snark.rs`
//! and reuses the one-step chunk-step IVC substrate directly.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{KExtensions, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::{
    build_dims_and_policy, digest_ccs_matrices_with_sparse_cache, Dims, PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG,
    PI_CCS_SUMCHECK_INITIAL_RAW_TAG, PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use crate::finalize::digest32_as_fields;
use crate::rv64im::chunk_relation::RV64IM_CHUNK_RELATION_DIGEST_RAW_TAG;
use crate::rv64im::final_relation::{
    rv64im_chunk_fold_initial_transcript_snapshot, Rv64imChunkFoldTranscriptSnapshot, RV64IM_CHUNK_DONE_RAW_TAG,
};
use crate::rv64im::ivc_snark::SpartanF;
use crate::rv64im::kernel::{rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache};
use crate::rv64im::main_relation_circuit::claim::{
    alloc_ce_claim, alloc_ce_claim_dec_surface, alloc_ce_claim_dec_surface_with_shared_r,
    alloc_ce_claim_public_surface_with_shared_point, CeClaimVar,
};
use crate::rv64im::main_relation_circuit::initial_sum::claimed_initial_sum_from_me_inputs;
use crate::rv64im::main_relation_circuit::k_field::{alloc_constant_k, alloc_k, KNum, KNumVar};
use crate::rv64im::main_relation_circuit::output_binding::enforce_me_outputs_against_inputs;
use crate::rv64im::main_relation_circuit::pi_ccs::{
    bind_header_and_instance_digest, bind_me_inputs, sample_challenges,
};
use crate::rv64im::main_relation_circuit::pi_dec::enforce_dec_public;
use crate::rv64im::main_relation_circuit::pi_rlc::enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk;
use crate::rv64im::main_relation_circuit::rho_sampling::{
    materialize_goldilocks_rot_matrices, sample_goldilocks_rot_rhos,
};
use crate::rv64im::main_relation_circuit::sumcheck_replay::verify_sumcheck_rounds;
use crate::rv64im::main_relation_circuit::terminal_identity::{
    enforce_terminal_identity_fe, enforce_terminal_identity_nc,
};
use crate::rv64im::main_relation_circuit::transcript::{
    hash_field_linear_combinations_raw, Poseidon2TranscriptCircuit,
};
use crate::rv64im::main_relation_trace::{
    Rv64imMainCircuitCeClaimShape, Rv64imMainCircuitChunkCover, Rv64imMainCircuitChunkReplaySurface,
    Rv64imMainCircuitHandoff, CHUNK_META_RAW_TAG,
};
mod chunk_diagnostics;
mod chunk_stage_ranges;
mod chunk_step_ivc;
mod chunk_step_recursive;
mod fingerprint_cs;
mod fixed_transcript;
mod nifs_v_stages;
mod recursive_cover;
mod recursive_step;
mod step_statement;
mod transcript_k;

const RV64IM_MAIN_RELATION_DELTA: u64 = 7;

#[allow(unused_imports)]
pub use chunk_diagnostics::debug_measure_rv64im_main_relation_state_in_prefix_fingerprints;
pub(crate) use chunk_diagnostics::{
    debug_locate_rv64im_main_relation_chunk_stage, debug_profile_rv64im_main_relation_chunk_stage_progress,
};
pub(crate) use chunk_stage_ranges::{
    debug_check_rv64im_rlc_public_x_native_values, debug_compare_rv64im_pi_ccs_transcript_state,
    debug_compare_rv64im_pi_rlc_rho_mats, debug_locate_rv64im_pi_ccs_late_transcript_stage,
    debug_measure_rv64im_main_relation_chunk_stage_ranges, debug_measure_rv64im_pi_rlc_stage_ranges,
    debug_measure_rv64im_rlc_public_stage_ranges,
};

pub(crate) use chunk_step_ivc::prepare_rv64im_chunk_step_ivc_circuit_inputs;
pub use chunk_step_ivc::{
    build_rv64im_chunk_step_ivc_recursive_step_cover_shape, build_rv64im_chunk_step_ivc_recursive_step_padding,
    build_rv64im_chunk_step_ivc_recursive_step_padding_from_shape, build_rv64im_chunk_step_ivc_shape,
    Rv64imChunkStepIvcRecursiveStepPadding, Rv64imChunkStepIvcShape, Rv64imChunkStepIvcSpartanError,
};
pub use chunk_step_recursive::{
    build_rv64im_main_recursion_f_prime_backend_relations,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices_and_perf,
    build_rv64im_main_recursion_f_prime_claim_cover, build_rv64im_main_recursion_f_prime_payload,
    build_rv64im_main_recursion_f_prime_payloads, build_rv64im_main_recursion_f_prime_payloads_with_spartan_shape,
    build_rv64im_main_recursion_step_spartan_shape,
    debug_check_rv64im_chunk_step_recursive_effective_chunk_trace_matches_native,
    debug_check_rv64im_main_recursion_f_prime_backend_relation_semantics,
    debug_trace_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices, Rv64imCcsClaimShape,
    Rv64imCcsWitnessShape, Rv64imCeClaimDigestShape, Rv64imMainRecursionFPrimeBackendRelation,
    Rv64imMainRecursionFPrimeBackendRelationBuildPerf, Rv64imMainRecursionFPrimeClaimCover,
    Rv64imMainRecursionFPrimePayload, Rv64imMainRecursionStepSpartanShape,
};
use nifs_v_stages::{
    enforce_outer_chunk_relation_public_io, enforce_synthetic_outer_chunk_relation_public_io, synthesize_pi_ccs_stage,
    synthesize_pi_dec_stage, synthesize_rv64im_chunk_nifs_verifier_body_with_synthetic_chunk_relation_io,
    Rv64imChunkNifsVerifierCtx, Rv64imPiRlcStageOutput,
};
#[allow(unused_imports)]
pub use recursive_step::Rv64imMainRecursionStepChunkReplayFingerprint;
pub use recursive_step::{
    build_rv64im_main_recursion_step_authoritative_chunk_surface,
    build_rv64im_main_recursion_step_spartan_published_target,
    debug_check_rv64im_main_recursion_step_authoritative_chunk_surface_matches_native,
    debug_check_rv64im_main_recursion_step_spartan_chunk_replay_surface,
    debug_check_rv64im_main_recursion_step_spartan_circuit,
    debug_check_rv64im_main_recursion_step_spartan_embedded_body,
    debug_check_rv64im_main_recursion_step_spartan_fresh_output_accumulator_digest_parity,
    debug_check_rv64im_main_recursion_step_spartan_inactive_side_lane_constraints,
    debug_check_rv64im_main_recursion_step_spartan_live_claim_me_digest_parity,
    debug_check_rv64im_main_recursion_step_spartan_pi_ccs_replay_lengths,
    debug_check_rv64im_main_recursion_x_out_gadget_parity,
    debug_measure_rv64im_main_recursion_step_chunk_replay_aux_counts,
    debug_measure_rv64im_main_recursion_step_chunk_replay_fingerprint,
    debug_measure_rv64im_main_recursion_step_chunk_replay_tail_aux_counts,
    debug_measure_rv64im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown,
    debug_measure_rv64im_main_recursion_step_pi_ccs_aux_counts,
    debug_measure_rv64im_main_recursion_step_pi_ccs_bind_me_inputs_aux_breakdown,
    debug_measure_rv64im_main_recursion_step_pi_ccs_constraint_counts,
    debug_measure_rv64im_main_recursion_step_pi_ccs_fingerprint,
    debug_measure_rv64im_main_recursion_step_pi_ccs_sumcheck_constraint_breakdown,
    debug_measure_rv64im_main_recursion_step_pi_rlc_public_constraint_breakdown,
    debug_measure_rv64im_main_recursion_step_pi_rlc_public_stage_breakdown,
    debug_measure_rv64im_main_recursion_step_shape_only_circuit_shape,
    debug_measure_rv64im_main_recursion_step_spartan_circuit_shape,
    debug_measure_rv64im_main_recursion_step_spartan_commitment_key,
    debug_measure_rv64im_main_recursion_step_spartan_shape_synthesis,
    debug_measure_rv64im_main_recursion_step_stage_aux_counts,
    debug_profile_rv64im_main_recursion_step_chunk_replay_stages,
    debug_trace_rv64im_main_recursion_step_fingerprint_synthesize,
    debug_trace_rv64im_main_recursion_step_shape_only_circuit_shape_measurement,
    debug_trace_rv64im_main_recursion_step_shape_only_fingerprint_synthesize,
    debug_trace_rv64im_main_recursion_step_spartan_circuit_shape_measurement,
    debug_trace_rv64im_main_recursion_step_spartan_shape_synthesis, Rv64imMainRecursionStepAuthoritativeChunkSurface,
    Rv64imMainRecursionStepChunkReplayAuxCounts, Rv64imMainRecursionStepChunkReplayTailAuxCounts,
    Rv64imMainRecursionStepChunkReplayTailDigestAuxBreakdown, Rv64imMainRecursionStepSpartanCircuitShape,
    Rv64imMainRecursionStepSpartanError, Rv64imMainRecursionStepSpartanPublishedTarget,
    Rv64imMainRecursionStepStageAuxCounts, Rv64imNamedConstraintDelta, Rv64imPiCcsBindMeInputsAuxBreakdown,
    Rv64imPiCcsStageAuxCounts, Rv64imPiCcsStageConstraintCounts, Rv64imPiCcsStageFingerprint,
    Rv64imPiCcsSumcheckConstraintBreakdown, Rv64imPiRlcPublicConstraintBreakdown, Rv64imPiRlcPublicStageBreakdown,
};
pub use step_statement::Rv64imMainRecursionStepSpartanStatement;
use transcript_k::append_k_to_transcript;

#[derive(Clone)]
pub(crate) struct Rv64imClaimBundle {
    claims: Vec<CeClaimVar>,
    effective_count: usize,
}

fn enforce_allocated_num_eq_constant<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: &AllocatedNum<SpartanF>,
    expected: SpartanF,
    label: &str,
) {
    cs.enforce(
        || label,
        |lc| lc + value.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (expected, CS::one()),
    );
}

fn import_chunk_fold_transcript_in<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    state_in_var: &recursive_cover::Rv64imRecursiveCoverStateVar,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    initial_transcript_in: bool,
    label: &str,
) -> Result<Poseidon2TranscriptCircuit, SynthesisError> {
    let transcript_in_values = transcript_in
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    if !initial_transcript_in {
        return Poseidon2TranscriptCircuit::from_state(
            state_in_var.transcript_state.clone(),
            transcript_in_values,
            transcript_in.absorbed,
        );
    }

    let expected = rv64im_chunk_fold_initial_transcript_snapshot();
    if transcript_in != &expected {
        return Err(SynthesisError::Unsatisfiable);
    }
    let expected_state = expected
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    for (idx, (allocated, value)) in state_in_var
        .transcript_state
        .iter()
        .zip(expected_state.iter())
        .enumerate()
    {
        enforce_allocated_num_eq_constant(cs, allocated, *value, &format!("{label}_state_{idx}"));
    }
    enforce_allocated_num_eq_constant(
        cs,
        &state_in_var.transcript_absorbed,
        SpartanF::from_canonical_u64(expected.absorbed as u64),
        &format!("{label}_absorbed"),
    );
    Poseidon2TranscriptCircuit::from_constant_state(expected_state, expected.absorbed)
}

impl Rv64imClaimBundle {
    pub(crate) fn from_effective_claims(claims: Vec<CeClaimVar>) -> Self {
        let effective_count = claims.len();
        Self {
            claims,
            effective_count,
        }
    }

    pub(crate) fn from_padded_claims(claims: Vec<CeClaimVar>, effective_count: usize) -> Self {
        debug_assert!(effective_count <= claims.len());
        Self {
            claims,
            effective_count,
        }
    }

    pub(crate) fn effective_claims(&self) -> &[CeClaimVar] {
        &self.claims[..self.effective_count]
    }

    pub(crate) fn effective_count(&self) -> usize {
        self.effective_count
    }

    pub(crate) fn into_effective_claims(self) -> Vec<CeClaimVar> {
        self.claims.into_iter().take(self.effective_count).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rv64imChunkBoundaryMode {
    Interior,
    TerminalPreserveIncoming,
    TerminalCarryChildren,
}

impl Rv64imChunkBoundaryMode {
    pub(crate) fn from_terminal_flags(is_terminal_chunk: bool, carry_terminal_children: bool) -> Self {
        match (is_terminal_chunk, carry_terminal_children) {
            (false, _) => Self::Interior,
            (true, false) => Self::TerminalPreserveIncoming,
            (true, true) => Self::TerminalCarryChildren,
        }
    }

    fn is_terminal(self) -> bool {
        !matches!(self, Self::Interior)
    }

    fn preserves_incoming_carry(self) -> bool {
        matches!(self, Self::TerminalPreserveIncoming)
    }

    fn uses_last_chunk_rlc_dec_shortcut(
        self,
        effective_fresh_claim_count: usize,
        effective_output_count: usize,
    ) -> bool {
        self.preserves_incoming_carry() && effective_fresh_claim_count == effective_output_count
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rv64imChunkChildClaimSource {
    ReplayedChildren,
    TerminalFinalClaims,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rv64imChunkNextCarryMode {
    ReplaceWithEffectiveChildren,
    PreserveIncoming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rv64imChunkRlcMode {
    Standard { constant_child_prefix: usize },
    TerminalLastChunkShortcut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rv64imChunkBoundaryPlan {
    pub(crate) child_claim_source: Rv64imChunkChildClaimSource,
    pub(crate) next_carry_mode: Rv64imChunkNextCarryMode,
    pub(crate) rlc_mode: Rv64imChunkRlcMode,
}

impl Rv64imChunkBoundaryPlan {
    pub(crate) fn from_boundary_mode(
        boundary_mode: Rv64imChunkBoundaryMode,
        effective_fresh_claim_count: usize,
        effective_output_count: usize,
    ) -> Self {
        let child_claim_source = if boundary_mode.is_terminal() {
            Rv64imChunkChildClaimSource::TerminalFinalClaims
        } else {
            Rv64imChunkChildClaimSource::ReplayedChildren
        };
        let next_carry_mode = if boundary_mode.preserves_incoming_carry() {
            Rv64imChunkNextCarryMode::PreserveIncoming
        } else {
            Rv64imChunkNextCarryMode::ReplaceWithEffectiveChildren
        };
        let rlc_mode =
            if boundary_mode.uses_last_chunk_rlc_dec_shortcut(effective_fresh_claim_count, effective_output_count) {
                Rv64imChunkRlcMode::TerminalLastChunkShortcut
            } else {
                Rv64imChunkRlcMode::Standard {
                    // Every effective Π_CCS output already has authoritative `c` / `x`
                    // binding before Π_RLC runs, so the standard RLC gadget can fold that
                    // prefix from native values instead of re-paying the full child-var path.
                    constant_child_prefix: effective_output_count,
                }
            };
        Self {
            child_claim_source,
            next_carry_mode,
            rlc_mode,
        }
    }
}

pub(crate) fn rv64im_main_relation_delta() -> SpartanF {
    SpartanF::from_canonical_u64(RV64IM_MAIN_RELATION_DELTA)
}

pub(crate) fn synthesize_rv64im_main_relation_chunk<CS: ConstraintSystem<SpartanF>>(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    terminal_final_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    cs: &mut CS,
    chunk_index: usize,
    cover_chunk: &Rv64imMainCircuitChunkCover,
    chunk: &Rv64imMainCircuitChunkReplaySurface,
    public_inputs: &[AllocatedNum<SpartanF>],
    public_cursor: &mut usize,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: Rv64imClaimBundle,
    boundary_plan: Rv64imChunkBoundaryPlan,
    enforce_chunk_relation_public_io: bool,
    append_chunk_done: bool,
) -> Result<Rv64imClaimBundle, SynthesisError> {
    let body_output = nifs_v_stages::synthesize_rv64im_chunk_nifs_verifier_body_with_outer_relation_mode(
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        cs,
        chunk_index,
        cover_chunk,
        chunk,
        transcript,
        carried_claims,
        None,
        None,
        boundary_plan,
        0,
        None,
        false,
        None,
    )?;
    let next_carried_claims = body_output.next_claims;
    let ctx = Rv64imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        chunk_index,
        cover_chunk,
        chunk,
        logical_me_input_claims: None,
        logical_me_input_digests: None,
        boundary_plan,
        rlc_zero_commit_suffix_len: 0,
        exact_initial_chunk_step_count: None,
    };
    if enforce_chunk_relation_public_io {
        // The standalone chunk theorem binds the relation digest as public IO.
        // Recursive F' replay uses only the inner verifier body and must skip
        // this outer theorem wrapper.
        enforce_outer_chunk_relation_public_io(
            &ctx,
            cs,
            &body_output.pi_ccs_fold_digest,
            public_inputs,
            public_cursor,
        )?;
    }
    if append_chunk_done {
        transcript.append_const_fields_raw(
            cs.namespace(|| format!("chunk_done_{chunk_index}")),
            &[
                SpartanF::from_canonical_u64(RV64IM_CHUNK_DONE_RAW_TAG),
                SpartanF::from_canonical_u64(1),
            ],
        )?;
    }
    Ok(next_carried_claims)
}

fn cover_ce_claim_with_shared_point(
    shape: &Rv64imMainCircuitCeClaimShape,
    effective: Option<&CeClaim<neo_ajtai::Commitment, F, K>>,
    shared_r_values: &[K],
    shared_s_col_values: &[K],
) -> Result<CeClaim<neo_ajtai::Commitment, F, K>, SynthesisError> {
    let mut claim = if let Some(claim) = effective {
        pad_ce_claim_to_cover_shape(shape, claim)?
    } else {
        shape.zero_claim()
    };
    claim.r = shared_r_values.to_vec();
    claim.s_col = shared_s_col_values.to_vec();
    Ok(claim)
}

fn cover_ce_claim(
    shape: &Rv64imMainCircuitCeClaimShape,
    effective: Option<&CeClaim<neo_ajtai::Commitment, F, K>>,
) -> Result<CeClaim<neo_ajtai::Commitment, F, K>, SynthesisError> {
    if let Some(claim) = effective {
        return pad_ce_claim_to_cover_shape(shape, claim);
    }
    Ok(shape.zero_claim())
}

fn pad_f_matrix_to_shape(matrix: &Mat<F>, rows: usize, cols: usize) -> Result<Mat<F>, SynthesisError> {
    if matrix.rows() > rows || matrix.cols() > cols {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut out = Mat::zero(rows, cols, F::ZERO);
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            out[(row, col)] = matrix[(row, col)];
        }
    }
    Ok(out)
}

fn pad_k_row_to_len(row: &[K], target_len: usize) -> Result<Vec<K>, SynthesisError> {
    if row.len() > target_len {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut out = row.to_vec();
    out.resize(target_len, K::ZERO);
    Ok(out)
}

fn pad_ce_claim_to_cover_shape(
    shape: &Rv64imMainCircuitCeClaimShape,
    claim: &CeClaim<neo_ajtai::Commitment, F, K>,
) -> Result<CeClaim<neo_ajtai::Commitment, F, K>, SynthesisError> {
    if !shape.covers_claim(claim) {
        return Err(SynthesisError::Unsatisfiable);
    }
    let y_ring_row_count = shape.y_ring_row_count as usize;
    if y_ring_row_count < shape.ct_len as usize {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut y_ring = Vec::with_capacity(y_ring_row_count);
    for row_idx in 0..y_ring_row_count {
        let mut target_len = shape.y_ring_row_lens.get(row_idx).copied().unwrap_or(0) as usize;
        if row_idx < shape.ct_len as usize {
            target_len = target_len.max(1);
        }
        let row = claim.y_ring.get(row_idx).map(Vec::as_slice).unwrap_or(&[]);
        y_ring.push(pad_k_row_to_len(row, target_len)?);
    }
    let mut c_step_coords = claim.c_step_coords.clone();
    c_step_coords.resize(shape.c_step_coords_len as usize, F::ZERO);
    Ok(CeClaim {
        c: claim.c.clone(),
        X: pad_f_matrix_to_shape(&claim.X, shape.x_rows as usize, shape.x_cols as usize)?,
        r: pad_k_row_to_len(&claim.r, shape.r_len as usize)?,
        s_col: pad_k_row_to_len(&claim.s_col, shape.s_col_len as usize)?,
        y_ring,
        ct: pad_k_row_to_len(&claim.ct, shape.ct_len as usize)?,
        aux_openings: pad_k_row_to_len(&claim.aux_openings, shape.aux_openings_len as usize)?,
        y_zcol: pad_k_row_to_len(&claim.y_zcol, shape.y_zcol_len as usize)?,
        m_in: claim.m_in,
        fold_digest: claim.fold_digest,
        c_step_coords,
        u_offset: claim.u_offset,
        u_len: claim.u_len,
    })
}

fn cover_ccs_claim(
    shape: &crate::rv64im::main_relation_trace::Rv64imMainCircuitCcsClaimShape,
    effective: Option<&neo_ccs::CcsClaim<neo_ajtai::Commitment, F>>,
) -> Result<neo_ccs::CcsClaim<neo_ajtai::Commitment, F>, SynthesisError> {
    if let Some(claim) = effective {
        if !shape.covers_claim(claim) {
            return Err(SynthesisError::Unsatisfiable);
        }
        let mut out = claim.clone();
        out.x.resize(shape.x_len as usize, F::ZERO);
        out.m_in = shape.x_len as usize;
        return Ok(out);
    }
    Ok(shape.zero_claim())
}

fn alloc_rounds<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    cover_round_lengths: &[u64],
    effective_rounds: &[Vec<K>],
    label: &str,
) -> Result<Vec<Vec<KNumVar>>, SynthesisError> {
    if cover_round_lengths.len() < effective_rounds.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    cover_round_lengths
        .iter()
        .enumerate()
        .map(|(round_idx, cover_len)| {
            let effective = effective_rounds
                .get(round_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if effective.len() > *cover_len as usize {
                return Err(SynthesisError::Unsatisfiable);
            }
            (0..(*cover_len as usize))
                .map(|coeff_idx| {
                    let coeff = effective.get(coeff_idx).copied().unwrap_or(K::ZERO);
                    alloc_k(
                        cs,
                        Some(KNum::from_neo_k(coeff)),
                        &format!("{label}_{round_idx}_{coeff_idx}"),
                    )
                })
                .collect()
        })
        .collect()
}

fn pad_round_values(cover_round_lengths: &[u64], effective_rounds: &[Vec<K>]) -> Result<Vec<Vec<K>>, SynthesisError> {
    if cover_round_lengths.len() < effective_rounds.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    cover_round_lengths
        .iter()
        .enumerate()
        .map(|(round_idx, cover_len)| {
            let effective = effective_rounds
                .get(round_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if effective.len() > *cover_len as usize {
                return Err(SynthesisError::Unsatisfiable);
            }
            let mut out = effective.to_vec();
            out.resize(*cover_len as usize, K::ZERO);
            Ok(out)
        })
        .collect()
}

fn max_degree_from_cover_round_lengths(round_lengths: &[u64]) -> usize {
    round_lengths
        .iter()
        .copied()
        .map(|len| len.saturating_sub(1) as usize)
        .max()
        .unwrap_or(0)
}

fn chunk_sumcheck_challenges(prefix: &[K], suffix: &[K]) -> Vec<K> {
    let mut out = Vec::with_capacity(prefix.len() + suffix.len());
    out.extend_from_slice(prefix);
    out.extend_from_slice(suffix);
    out
}

pub(crate) fn append_chunk_meta<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    transcript: &mut Poseidon2TranscriptCircuit,
    handoff: &Rv64imMainCircuitHandoff,
) -> Result<(), SynthesisError> {
    append_chunk_meta_with_exact_initial_constants(cs, transcript, handoff, None)
}

pub(crate) fn append_chunk_meta_with_exact_initial_constants<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    transcript: &mut Poseidon2TranscriptCircuit,
    handoff: &Rv64imMainCircuitHandoff,
    exact_initial_step_count: Option<usize>,
) -> Result<(), SynthesisError> {
    if let Some(step_count) = exact_initial_step_count {
        if handoff.public_chunk.start_index != 0 || handoff.public_chunk.steps.len() != step_count {
            return Err(SynthesisError::Unsatisfiable);
        }
        transcript.append_const_fields_raw(
            cs.namespace(|| "chunk_meta"),
            &[
                SpartanF::from_canonical_u64(CHUNK_META_RAW_TAG),
                SpartanF::ZERO,
                SpartanF::from_canonical_u64(step_count as u64),
            ],
        )?;
        return Ok(());
    }

    let chunk_meta_values = [
        SpartanF::from_canonical_u64(handoff.public_chunk.start_index as u64),
        SpartanF::from_canonical_u64(handoff.public_chunk.steps.len() as u64),
    ];
    let chunk_meta_vars = alloc_private_field_values(cs, &chunk_meta_values, "chunk_meta")?;
    transcript.append_field_linear_combinations_raw(
        cs.namespace(|| "chunk_meta"),
        &[
            Vec::new(),
            vec![(chunk_meta_vars[0].get_variable(), SpartanF::ONE)],
            vec![(chunk_meta_vars[1].get_variable(), SpartanF::ONE)],
        ],
        &[
            SpartanF::from_canonical_u64(CHUNK_META_RAW_TAG),
            SpartanF::ZERO,
            SpartanF::ZERO,
        ],
        &[
            SpartanF::from_canonical_u64(CHUNK_META_RAW_TAG),
            chunk_meta_values[0],
            chunk_meta_values[1],
        ],
    )?;
    Ok(())
}

pub(crate) fn next_public_digest(
    public_inputs: &[AllocatedNum<SpartanF>],
    cursor: &mut usize,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    if *cursor + 4 > public_inputs.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let out = core::array::from_fn(|idx| public_inputs[*cursor + idx].clone());
    *cursor += 4;
    let _ = label;
    Ok(out)
}

pub(crate) fn enforce_digest_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>; 4],
    expected: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<(), SynthesisError> {
    for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
        cs.enforce(
            || format!("{label}_{idx}"),
            |lc| lc + lhs.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + rhs.get_variable(),
        );
    }
    Ok(())
}

fn split_vec<T: Clone>(values: &[T], prefix_len: usize) -> Result<(Vec<T>, Vec<T>), SynthesisError> {
    if prefix_len > values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok((values[..prefix_len].to_vec(), values[prefix_len..].to_vec()))
}

fn chunk_relation_digest_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    public_chunk_digest: [u8; 32],
    main_relation_digest: &[AllocatedNum<SpartanF>; 4],
    bridge_handoff_digest: [u8; 32],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let public_chunk_digest_fields = digest32_as_spartan_fields(public_chunk_digest);
    let bridge_handoff_digest_fields = digest32_as_spartan_fields(bridge_handoff_digest);
    let mut field_terms = Vec::with_capacity(13);
    let mut field_constants = Vec::with_capacity(13);
    let mut field_values = Vec::with_capacity(13);

    field_terms.push(Vec::new());
    field_constants.push(SpartanF::from_canonical_u64(RV64IM_CHUNK_RELATION_DIGEST_RAW_TAG));
    field_values.push(SpartanF::from_canonical_u64(RV64IM_CHUNK_RELATION_DIGEST_RAW_TAG));

    for value in public_chunk_digest_fields {
        field_terms.push(Vec::new());
        field_constants.push(value);
        field_values.push(value);
    }

    for lane in main_relation_digest {
        field_terms.push(vec![(lane.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_values.push(lane.get_value().unwrap_or(SpartanF::ZERO));
    }

    for value in bridge_handoff_digest_fields {
        field_terms.push(Vec::new());
        field_constants.push(value);
        field_values.push(value);
    }

    hash_field_linear_combinations_raw(
        cs.namespace(|| "chunk_relation_digest_hash"),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

pub(crate) fn alloc_const_field_values<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    values: &[SpartanF],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let out = AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(*value))?;
            cs.enforce(
                || format!("{label}_{idx}_const"),
                |lc| lc + out.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (*value, CS::one()),
            );
            Ok(out)
        })
        .collect()
}

pub(crate) fn digest_const_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest: [u8; 32],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    alloc_const_field_values(cs, &digest32_as_spartan_fields(digest), label)?
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)
}

pub(crate) fn alloc_private_field_values<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    values: &[SpartanF],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(*value)))
        .collect()
}

pub(crate) fn digest32_as_spartan_fields(digest: [u8; 32]) -> [SpartanF; 4] {
    digest32_as_fields(digest).map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
}
