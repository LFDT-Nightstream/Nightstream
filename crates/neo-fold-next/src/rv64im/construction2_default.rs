//! Owns default-pair derivation for the native RV64IM Construction-2 surface.
//!
//! HyperNova Def. 12 requires the canonical `u_perp` default instance to be a
//! pure function of `(pp, s)`. The production path in this module therefore
//! derives the native F' witness width from canonical protocol structure only:
//! the recursion verifier key, the root CCS context, and the phi-side shape.
//!
//! Relation-derived cover builders remain available as audit helpers so tests
//! can compare honest traces against the structural builder.

use std::io::{self, Write};
use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::build_dims_and_policy;
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use crate::proof::Carry;
use crate::rv64im::ccs::{RV64IM_ROOT_PUBLIC_INPUTS, RV64IM_ROOT_ROW_WIDTH};
use crate::rv64im::chunk_fold_step::adapt_rv64im_chunk_to_fresh_ccs;
use crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcRelation;
use crate::rv64im::construction2::{
    build_rv64im_main_recursion_construction2_fresh_instance_from_full_vector,
    build_rv64im_main_recursion_construction2_pi_fold_from_relation,
    build_rv64im_main_recursion_construction2_verified_step_statement_from_relation,
    debug_trace_build_rv64im_main_recursion_construction2_fresh_instance_from_full_vector,
    Rv64imMainRecursionConstruction2Commitment, Rv64imMainRecursionConstruction2FPrimeCcsShape,
    Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage, Rv64imMainRecursionConstruction2FreshInstance,
};
use crate::rv64im::f_prime::{
    Rv64imMainRecursionPhiSide, Rv64imVerifierKeyFs, RV64IM_ENC_INST_BITS, RV64IM_ENC_INST_RING_DEGREE,
    RV64IM_ENC_INST_RING_SLOTS,
};
use crate::rv64im::final_relation::Rv64imChunkFoldState;
use crate::rv64im::main_relation_spartan::{
    Rv64imCcsClaimShape, Rv64imCcsWitnessShape, Rv64imCeClaimDigestShape, Rv64imChunkStepIvcShape,
    Rv64imMainRecursionFPrimeClaimCover,
};
use crate::rv64im::main_relation_trace::build_rv64im_main_circuit_chunk_trace_from_authoritative_parts;
use crate::rv64im::recursion_shape::build_rv64im_recursion_shape;
use crate::rv64im::SimpleKernelError;
use crate::witness_layout::commit_cols_for_full_width;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionConstruction2DefaultPair {
    u_perp: Rv64imMainRecursionConstruction2FreshInstance,
    w_perp: Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage,
}

impl Rv64imMainRecursionConstruction2DefaultPair {
    pub fn u_perp(&self) -> &Rv64imMainRecursionConstruction2FreshInstance {
        &self.u_perp
    }

    pub fn w_perp(&self) -> &Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage {
        &self.w_perp
    }
}

fn count_u64_field() -> usize {
    1
}

fn count_digest_fields() -> usize {
    4
}

fn count_k_coeffs() -> usize {
    K::ZERO.as_coeffs().len()
}

fn count_commitment_shape_fields(c_data_len: u64) -> usize {
    3 + c_data_len as usize
}

fn count_commitment_fields_for_full_width(full_width: usize) -> Result<usize, SimpleKernelError> {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Construction-2 default-pair params failed for full width {full_width}: {err}"
        ))
    })?;
    Ok(3 + D * params.kappa as usize)
}

fn count_f_slice(len: usize) -> usize {
    1 + len
}

fn count_f_matrix_rows_cols(rows: usize, cols: usize) -> usize {
    2 + rows * cols
}

fn count_k_slice(len: usize) -> usize {
    1 + len * count_k_coeffs()
}

fn count_k_rows_from_lens(row_lens: &[u64]) -> usize {
    1 + row_lens
        .iter()
        .map(|len| count_k_slice(*len as usize))
        .sum::<usize>()
}

fn count_ce_claim_shape_fields(shape: &Rv64imCeClaimDigestShape) -> usize {
    count_commitment_shape_fields(shape.c_data_len)
        + count_u64_field()
        + count_f_matrix_rows_cols(shape.x_rows as usize, shape.x_cols as usize)
        + count_k_slice(shape.r_len as usize)
        + count_k_slice(shape.s_col_len as usize)
        + count_k_rows_from_lens(&shape.y_ring_row_lens)
        + count_k_slice(shape.ct_len as usize)
        + count_k_slice(shape.aux_openings_len as usize)
        + count_k_slice(shape.y_zcol_len as usize)
}

fn count_pi_ccs_output_payload_shape_fields(shape: &Rv64imCeClaimDigestShape) -> usize {
    count_k_rows_from_lens(&shape.y_ring_row_lens) + count_k_slice(shape.y_zcol_len as usize)
}

fn count_pi_dec_child_payload_shape_fields(shape: &Rv64imCeClaimDigestShape) -> usize {
    count_commitment_shape_fields(shape.c_data_len) + count_k_rows_from_lens(&shape.y_ring_row_lens)
}

fn count_step_input_shape_fields(claim_shape: &Rv64imCcsClaimShape, witness_shape: &Rv64imCcsWitnessShape) -> usize {
    count_commitment_shape_fields(claim_shape.c_data_len)
        + count_u64_field()
        + count_f_slice(claim_shape.x_len as usize)
        + count_f_slice(witness_shape.w_len as usize)
        + count_f_matrix_rows_cols(witness_shape.z_rows as usize, witness_shape.z_cols as usize)
}

fn count_phi_side_fields_from_shape(shape: &Rv64imMainRecursionConstruction2FPrimeCcsShape) -> usize {
    count_u64_field()
        + shape
            .phi_side_commitment_word_lens
            .iter()
            .map(|len| count_u64_field() + *len as usize)
            .sum::<usize>()
}

fn canonical_phi_side_commitment_word_lens(phi_side: &Rv64imMainRecursionPhiSide) -> Vec<u64> {
    phi_side
        .commitment_words()
        .iter()
        .map(|words| words.len() as u64)
        .collect()
}

fn build_root_ce_claim_shape() -> Result<Rv64imCeClaimDigestShape, SimpleKernelError> {
    let (params, _, structure) = crate::rv64im::kernel::rv64im_cached_root_main_lane_context()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Build(format!("RV64IM canonical CE claim shape dims failed: {err}")))?;
    let d_pad = 1usize
        .checked_shl(dims.ell_d as u32)
        .ok_or_else(|| SimpleKernelError::Build("RV64IM canonical CE claim d_pad overflow".into()))?;
    Ok(Rv64imCeClaimDigestShape {
        commitment_d: D as u64,
        commitment_kappa: params.kappa as u64,
        c_data_len: (D * params.kappa as usize) as u64,
        x_rows: D as u64,
        x_cols: RV64IM_ROOT_PUBLIC_INPUTS as u64,
        r_len: dims.ell_n as u64,
        s_col_len: dims.ell_m as u64,
        y_ring_row_count: structure.t() as u64,
        y_ring_row_lens: vec![d_pad as u64; structure.t()],
        ct_len: structure.t() as u64,
        aux_openings_len: 0,
        y_zcol_len: d_pad as u64,
        c_step_coords_len: 0,
    })
}

fn build_root_ccs_claim_shape() -> Result<Rv64imCcsClaimShape, SimpleKernelError> {
    let (params, _, _) = crate::rv64im::kernel::rv64im_cached_root_main_lane_context()?;
    Ok(Rv64imCcsClaimShape {
        commitment_d: D as u64,
        commitment_kappa: params.kappa as u64,
        c_data_len: (D * params.kappa as usize) as u64,
        x_len: RV64IM_ROOT_PUBLIC_INPUTS as u64,
    })
}

fn build_root_ccs_witness_shape() -> Rv64imCcsWitnessShape {
    Rv64imCcsWitnessShape {
        w_len: (RV64IM_ROOT_ROW_WIDTH - RV64IM_ROOT_PUBLIC_INPUTS) as u64,
        z_rows: D as u64,
        z_cols: commit_cols_for_full_width(RV64IM_ROOT_ROW_WIDTH) as u64,
    }
}

pub(crate) fn build_rv64im_main_recursion_canonical_zero_carry() -> Result<Carry, SimpleKernelError> {
    let (params, _, _) = crate::rv64im::kernel::rv64im_cached_root_main_lane_context()?;
    let claim_shape = build_root_ce_claim_shape()?;
    let witness_shape = build_root_ccs_witness_shape();
    let claim_count = params.k_rho as usize;
    let zero_claim = claim_shape.zero_claim();
    let zero_witness = witness_shape.zero_witness().Z;
    Ok(Carry {
        claims: vec![zero_claim; claim_count],
        witnesses: vec![zero_witness; claim_count],
    })
}

pub fn build_rv64im_main_recursion_construction2_canonical_shape(
    vk_fs: &Rv64imVerifierKeyFs,
    phi_side: &Rv64imMainRecursionPhiSide,
) -> Result<Rv64imMainRecursionConstruction2FPrimeCcsShape, SimpleKernelError> {
    let recursion_shape = build_rv64im_recursion_shape()?;
    let recursion_shape_digest = recursion_shape.canonical_digest();
    if vk_fs.main_lane_shape_digest != recursion_shape_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 canonical shape builder requires the canonical recursion verifier-key shape".into(),
        ));
    }

    let (params, _, structure) = crate::rv64im::kernel::rv64im_cached_root_main_lane_context()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Build(format!("RV64IM canonical F' shape dims failed: {err}")))?;
    let ce_claim_shape = build_root_ce_claim_shape()?;
    let ccs_claim_shape = build_root_ccs_claim_shape()?;
    let ccs_witness_shape = build_root_ccs_witness_shape();
    let carried_claim_count = params.k_rho as usize;
    let ccs_output_count = carried_claim_count
        .checked_add(1)
        .ok_or_else(|| SimpleKernelError::Build("RV64IM canonical F' ccs_output count overflow".into()))?;
    let round_len = (dims.d_sc + 1) as u64;

    Ok(Rv64imMainRecursionConstruction2FPrimeCcsShape {
        verifier_key_fs_digest: vk_fs.expected_digest(),
        recursion_shape_digest,
        x_i_bit_len: RV64IM_ENC_INST_BITS as u64,
        x_i_ring_slot_count: RV64IM_ENC_INST_RING_SLOTS as u64,
        x_i_ring_degree: RV64IM_ENC_INST_RING_DEGREE as u64,
        phi_side_commitment_word_lens: canonical_phi_side_commitment_word_lens(phi_side),
        step_cover_shape: Rv64imChunkStepIvcShape {
            terminal_step: false,
            state_in_claim_count: carried_claim_count as u64,
            state_out_claim_count: carried_claim_count as u64,
            fresh_claim_count: 1,
            fresh_witness_count: 1,
            ccs_output_count: ccs_output_count as u64,
            child_count: carried_claim_count as u64,
            transcript_in_absorbed: 0,
            transcript_out_absorbed: 0,
            fe_round_lengths: vec![round_len; dims.ell_n + dims.ell_d],
            nc_round_lengths: vec![round_len; dims.ell_m + dims.ell_d],
        },
        claim_cover: Rv64imMainRecursionFPrimeClaimCover {
            state_in_claim_shapes: vec![ce_claim_shape.clone(); carried_claim_count],
            state_out_claim_shapes: vec![ce_claim_shape.clone(); carried_claim_count],
            fresh_claim_shapes: vec![ccs_claim_shape],
            fresh_witness_shapes: vec![ccs_witness_shape],
            parent_claim_shape: ce_claim_shape.clone(),
            ccs_output_shapes: vec![ce_claim_shape.clone(); ccs_output_count],
            child_claim_shapes: vec![ce_claim_shape; carried_claim_count],
        },
    })
}

pub fn build_rv64im_main_recursion_construction2_canonical_full_width(
    vk_fs: &Rv64imVerifierKeyFs,
    phi_side: &Rv64imMainRecursionPhiSide,
) -> Result<usize, SimpleKernelError> {
    build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape(
        &build_rv64im_main_recursion_construction2_canonical_shape(vk_fs, phi_side)?,
    )
}

pub fn build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape(
    shape: &Rv64imMainRecursionConstruction2FPrimeCcsShape,
) -> Result<usize, SimpleKernelError> {
    if shape.claim_cover.state_in_claim_shapes.len() != shape.step_cover_shape.state_in_claim_count as usize
        || shape.claim_cover.fresh_claim_shapes.len() != 1
        || shape.claim_cover.fresh_witness_shapes.len() != 1
        || shape.claim_cover.ccs_output_shapes.len() != shape.step_cover_shape.ccs_output_count as usize
        || shape.claim_cover.child_claim_shapes.len() != shape.step_cover_shape.child_count as usize
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 default-pair width requires a canonical fixed native F' shape cover".into(),
        ));
    }

    let count_default_witness_logical_fields = |full_width: usize| -> Result<usize, SimpleKernelError> {
        let fresh_instance_fields = count_commitment_fields_for_full_width(full_width)? + RV64IM_ENC_INST_BITS;
        Ok(count_u64_field()
            + count_digest_fields()
            + count_digest_fields()
            + count_u64_field()
            + count_phi_side_fields_from_shape(shape)
            + count_u64_field()
            + shape
                .claim_cover
                .state_in_claim_shapes
                .iter()
                .map(count_ce_claim_shape_fields)
                .sum::<usize>()
            + fresh_instance_fields
            + count_u64_field()
            + count_u64_field()
            + count_step_input_shape_fields(
                &shape.claim_cover.fresh_claim_shapes[0],
                &shape.claim_cover.fresh_witness_shapes[0],
            )
            + count_u64_field()
            + shape
                .claim_cover
                .ccs_output_shapes
                .iter()
                .map(count_pi_ccs_output_payload_shape_fields)
                .sum::<usize>()
            + count_u64_field()
            + shape
                .step_cover_shape
                .fe_round_lengths
                .iter()
                .map(|round_len| count_k_slice(*round_len as usize))
                .sum::<usize>()
            + count_u64_field()
            + shape
                .step_cover_shape
                .nc_round_lengths
                .iter()
                .map(|round_len| count_k_slice(*round_len as usize))
                .sum::<usize>()
            + count_u64_field()
            + shape
                .claim_cover
                .child_claim_shapes
                .iter()
                .map(count_pi_dec_child_payload_shape_fields)
                .sum::<usize>())
    };

    let mut full_width = RV64IM_ENC_INST_BITS;
    for _ in 0..8 {
        let next = RV64IM_ENC_INST_BITS + count_default_witness_logical_fields(full_width)? * 64;
        if next == full_width {
            return Ok(full_width);
        }
        full_width = next;
    }
    Err(SimpleKernelError::Bridge(
        "RV64IM Construction-2 default-pair width did not converge from the fixed native witness layout".into(),
    ))
}

/// Derives the canonical F' CCS shape cover from an honest relation chain.
///
/// F' is a fixed-shape recursive step at the *circuit* level (HyperNova
/// Construction-2 §6.3), but the per-chunk native witnesses legitimately
/// differ — the accumulator grows claims as chunks fold, sumcheck rounds
/// can vary by chunk index, etc. The protocol padded circuit shape is the
/// fixed point that covers every chunk; this helper builds that cover via
/// MAX-merge across `state_in/out`, `fresh_claim/witness`, `ccs_output`,
/// `child`, and sumcheck round-length slots.
///
/// This helper is intentionally relation-owned and is used only for audit /
/// conformance comparisons against the canonical builder.
pub fn build_rv64im_main_recursion_construction2_default_shape_cover_from_relations(
    vk_fs: &Rv64imVerifierKeyFs,
    accumulator_in: &Rv64imChunkFoldState,
    relations: &[Rv64imChunkStepIvcRelation],
    phi_side: &Rv64imMainRecursionPhiSide,
) -> Result<Rv64imMainRecursionConstruction2FPrimeCcsShape, SimpleKernelError> {
    if relations.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 default width cover requires at least one relation".into(),
        ));
    }
    let recursion_shape_digest = build_rv64im_recursion_shape()?.canonical_digest();
    if vk_fs.main_lane_shape_digest != recursion_shape_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 default width cover requires the canonical recursion verifier-key shape".into(),
        ));
    }

    let mut step_cover_shape = Rv64imChunkStepIvcShape::recursive_step_cover_seed();
    let mut phi_side_commitment_word_lens = Vec::new();
    let mut state_in_claim_shapes = Vec::new();
    let mut state_out_claim_shapes = Vec::new();
    let mut fresh_claim_shapes = Vec::new();
    let mut fresh_witness_shapes = Vec::new();
    let mut parent_claim_shape: Option<Rv64imCeClaimDigestShape> = None;
    let mut ccs_output_shapes = Vec::new();
    let mut child_claim_shapes = Vec::new();

    merge_phi_side_commitment_word_cover(&mut phi_side_commitment_word_lens, phi_side.commitment_words());
    // Thread the carried native state across relations. Later chunks do not
    // replay from the seed accumulator/transcript.
    let mut running_state = accumulator_in.clone();

    for relation in relations {
        if relation.witness.state_in.carry.terminal_handle != running_state.carry.terminal_handle
            || relation.witness.state_in.carry.main.claims != running_state.carry.main.claims
            || relation.witness.state_in.carry.main.witnesses != running_state.carry.main.witnesses
            || relation.witness.state_in.transcript != running_state.transcript
        {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Construction-2 default width cover requires a contiguous relation-owned carried state chain"
                    .into(),
            ));
        }
        let native_verified_step_statement =
            build_rv64im_main_recursion_construction2_verified_step_statement_from_relation(relation)?;
        let main_circuit_chunk_summary = native_verified_step_statement.fixed_shape_chunk_summary()?;
        let main_circuit_chunk_trace = build_rv64im_main_circuit_chunk_trace_from_authoritative_parts(
            relation.witness.handoff.bridge_handoff.chunk_index as usize,
            &relation.witness.handoff,
            &main_circuit_chunk_summary,
            &running_state.carry,
            &relation.witness.state_out.carry,
            &running_state.transcript,
            &relation.witness.state_out.transcript,
            &relation.witness.replay_witness,
        )?;
        let construction2_pi_fold = build_rv64im_main_recursion_construction2_pi_fold_from_relation(relation)?;
        let fresh = adapt_rv64im_chunk_to_fresh_ccs(&relation.witness.handoff);
        merge_claim_shape_cover(&mut state_in_claim_shapes, &running_state.carry.main.claims);

        step_cover_shape = step_cover_shape.recursive_step_cover_merge(&Rv64imChunkStepIvcShape {
            terminal_step: false,
            state_in_claim_count: running_state.carry.main.claims.len() as u64,
            state_out_claim_count: relation.witness.state_out.carry.main.claims.len() as u64,
            fresh_claim_count: fresh.fresh_claims.len() as u64,
            fresh_witness_count: fresh.fresh_witnesses.len() as u64,
            ccs_output_count: construction2_pi_fold.ccs_output_payloads.len() as u64,
            child_count: construction2_pi_fold.dec_child_payloads.len() as u64,
            transcript_in_absorbed: running_state.transcript.absorbed as u64,
            transcript_out_absorbed: relation.witness.state_out.transcript.absorbed as u64,
            fe_round_lengths: construction2_pi_fold
                .ccs_replay_payload
                .sumcheck_rounds
                .iter()
                .map(|round| round.len() as u64)
                .collect(),
            nc_round_lengths: construction2_pi_fold
                .ccs_replay_payload
                .sumcheck_rounds_nc
                .iter()
                .map(|round| round.len() as u64)
                .collect(),
        });
        merge_claim_shape_cover(
            &mut state_out_claim_shapes,
            &relation.witness.state_out.carry.main.claims,
        );
        merge_ccs_claim_shape_cover(&mut fresh_claim_shapes, &fresh.fresh_claims);
        merge_ccs_witness_shape_cover(&mut fresh_witness_shapes, &fresh.fresh_witnesses);
        let trace_parent_shape = Rv64imCeClaimDigestShape::from_claim(&main_circuit_chunk_trace.ccs_trace.parent);
        parent_claim_shape = Some(match parent_claim_shape {
            Some(existing) => existing.merge(&trace_parent_shape),
            None => trace_parent_shape,
        });
        merge_claim_shape_cover(&mut ccs_output_shapes, &main_circuit_chunk_trace.ccs_trace.ccs_outputs);
        merge_claim_shape_cover(&mut child_claim_shapes, &main_circuit_chunk_trace.ccs_trace.children);
        running_state = relation.witness.state_out.clone();
    }

    Ok(Rv64imMainRecursionConstruction2FPrimeCcsShape {
        verifier_key_fs_digest: vk_fs.expected_digest(),
        recursion_shape_digest,
        x_i_bit_len: RV64IM_ENC_INST_BITS as u64,
        x_i_ring_slot_count: RV64IM_ENC_INST_RING_SLOTS as u64,
        x_i_ring_degree: RV64IM_ENC_INST_RING_DEGREE as u64,
        phi_side_commitment_word_lens,
        step_cover_shape,
        claim_cover: Rv64imMainRecursionFPrimeClaimCover {
            state_in_claim_shapes,
            state_out_claim_shapes,
            fresh_claim_shapes,
            fresh_witness_shapes,
            parent_claim_shape: parent_claim_shape.ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV64IM Construction-2 default width cover requires at least one parent CE claim shape".into(),
                )
            })?,
            ccs_output_shapes,
            child_claim_shapes,
        },
    })
}

fn merge_phi_side_commitment_word_cover(slots: &mut Vec<u64>, commitment_words: &[Vec<u64>]) {
    for (idx, words) in commitment_words.iter().enumerate() {
        if let Some(existing) = slots.get_mut(idx) {
            *existing = (*existing).max(words.len() as u64);
        } else {
            slots.push(words.len() as u64);
        }
    }
}

fn merge_claim_shape_cover(slots: &mut Vec<Rv64imCeClaimDigestShape>, claims: &[CeClaim<Commitment, F, K>]) {
    for (idx, claim) in claims.iter().enumerate() {
        let shape = Rv64imCeClaimDigestShape::from_claim(claim);
        if let Some(existing) = slots.get_mut(idx) {
            *existing = existing.merge(&shape);
        } else {
            slots.push(shape);
        }
    }
}

fn merge_ccs_claim_shape_cover(slots: &mut Vec<Rv64imCcsClaimShape>, claims: &[CcsClaim<Commitment, F>]) {
    for (idx, claim) in claims.iter().enumerate() {
        let shape = Rv64imCcsClaimShape::from_claim(claim);
        if let Some(existing) = slots.get_mut(idx) {
            *existing = existing.merge(&shape);
        } else {
            slots.push(shape);
        }
    }
}

fn merge_ccs_witness_shape_cover(slots: &mut Vec<Rv64imCcsWitnessShape>, witnesses: &[CcsWitness<F>]) {
    for (idx, witness) in witnesses.iter().enumerate() {
        let shape = Rv64imCcsWitnessShape::from_witness(witness);
        if let Some(existing) = slots.get_mut(idx) {
            *existing = existing.merge(&shape);
        } else {
            slots.push(shape);
        }
    }
}

pub fn build_rv64im_main_recursion_construction2_default_full_width_from_relations(
    vk_fs: &Rv64imVerifierKeyFs,
    accumulator_in: &Rv64imChunkFoldState,
    relations: &[Rv64imChunkStepIvcRelation],
    phi_side: &Rv64imMainRecursionPhiSide,
) -> Result<usize, SimpleKernelError> {
    build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape(
        &build_rv64im_main_recursion_construction2_default_shape_cover_from_relations(
            vk_fs,
            accumulator_in,
            relations,
            phi_side,
        )?,
    )
}

pub fn build_rv64im_main_recursion_construction2_default_pair_for_full_width(
    vk_fs: &Rv64imVerifierKeyFs,
    full_width: usize,
) -> Result<Rv64imMainRecursionConstruction2DefaultPair, SimpleKernelError> {
    let expected_vk_fs = crate::rv64im::f_prime::build_rv64im_main_recursion_verifier_key_fs()?;
    if vk_fs != &expected_vk_fs {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 default pair requires the canonical recursion verifier-key context".into(),
        ));
    }
    if full_width < RV64IM_ENC_INST_BITS {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM native Construction-2 default pair full width {full_width} is smaller than the 256-bit x image"
        )));
    }
    let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Construction-2 default pair params failed for full width {full_width}: {err}"
        ))
    })?;
    let w_perp = Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage {
        binary_values: vec![F::ZERO; full_width - RV64IM_ENC_INST_BITS],
    };
    let x_i = crate::rv64im::f_prime::Rv64imEncodedPublicInput::from_digest_bytes([0; 32]);
    let u_perp = Rv64imMainRecursionConstruction2FreshInstance::from_parts(
        Rv64imMainRecursionConstruction2Commitment::from_commitment(Commitment::zeros(D, params.kappa as usize)),
        x_i,
    );
    Ok(Rv64imMainRecursionConstruction2DefaultPair { u_perp, w_perp })
}

pub(crate) fn debug_trace_build_rv64im_main_recursion_construction2_default_pair_for_full_width(
    vk_fs: &Rv64imVerifierKeyFs,
    full_width: usize,
    trace_prefix: &str,
) -> Result<Rv64imMainRecursionConstruction2DefaultPair, SimpleKernelError> {
    build_rv64im_main_recursion_construction2_default_pair_for_full_width_impl(vk_fs, full_width, Some(trace_prefix))
}

fn emit_debug_timing(trace_prefix: Option<&str>, label: &str, elapsed_ms: f64) {
    if let Some(prefix) = trace_prefix {
        eprintln!("{prefix}.{label}={elapsed_ms:.2}ms");
        let _ = io::stderr().flush();
    }
}

fn build_rv64im_main_recursion_construction2_default_pair_for_full_width_impl(
    vk_fs: &Rv64imVerifierKeyFs,
    full_width: usize,
    trace_prefix: Option<&str>,
) -> Result<Rv64imMainRecursionConstruction2DefaultPair, SimpleKernelError> {
    let expected_vk_fs = crate::rv64im::f_prime::build_rv64im_main_recursion_verifier_key_fs()?;
    if vk_fs != &expected_vk_fs {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 default pair requires the canonical recursion verifier-key context".into(),
        ));
    }
    if full_width < RV64IM_ENC_INST_BITS {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM native Construction-2 default pair full width {full_width} is smaller than the 256-bit x image"
        )));
    }
    let started = Instant::now();
    let w_perp = Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage {
        binary_values: vec![F::ZERO; full_width - RV64IM_ENC_INST_BITS],
    };
    emit_debug_timing(
        trace_prefix,
        "w_perp_allocate",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    let started = Instant::now();
    let x_i = crate::rv64im::f_prime::Rv64imEncodedPublicInput::from_digest_bytes([0; 32]);
    let mut full_vector = Vec::with_capacity(full_width);
    full_vector.extend(x_i.field_image());
    full_vector.extend_from_slice(w_perp.binary_values());
    emit_debug_timing(
        trace_prefix,
        "full_vector_materialize",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    let u_perp = if let Some(prefix) = trace_prefix {
        debug_trace_build_rv64im_main_recursion_construction2_fresh_instance_from_full_vector(
            0,
            x_i,
            &full_vector,
            &format!("{prefix}.u_perp"),
        )?
    } else {
        build_rv64im_main_recursion_construction2_fresh_instance_from_full_vector(0, x_i, &full_vector)?
    };
    Ok(Rv64imMainRecursionConstruction2DefaultPair { u_perp, w_perp })
}
