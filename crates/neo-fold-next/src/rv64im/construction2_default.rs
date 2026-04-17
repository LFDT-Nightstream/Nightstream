//! Owns default-pair derivation for the native RV64IM Construction-2 surface.

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use crate::rv64im::chunk_fold_step::adapt_rv64im_chunk_to_fresh_ccs;
use crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcRelation;
use crate::rv64im::construction2::{
    build_rv64im_main_recursion_construction2_fresh_instance_from_full_vector,
    build_rv64im_main_recursion_construction2_pi_fold_from_relation,
    build_rv64im_main_recursion_construction2_verified_step_statement_from_relation,
    Rv64imMainRecursionConstruction2FPrimeCcsShape, Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage,
    Rv64imMainRecursionConstruction2FreshInstance,
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
    merge_claim_shape_cover(&mut state_in_claim_shapes, &accumulator_in.carry.main.claims);

    for relation in relations {
        let native_verified_step_statement =
            build_rv64im_main_recursion_construction2_verified_step_statement_from_relation(relation)?;
        let main_circuit_chunk_summary = native_verified_step_statement.fixed_shape_chunk_summary()?;
        let main_circuit_chunk_trace = build_rv64im_main_circuit_chunk_trace_from_authoritative_parts(
            relation.witness.handoff.bridge_handoff.chunk_index as usize,
            &relation.witness.handoff,
            &main_circuit_chunk_summary,
            &accumulator_in.carry,
            &relation.witness.state_out.carry,
            &accumulator_in.transcript,
            &relation.witness.state_out.transcript,
            &relation.witness.replay_witness,
        )?;
        let construction2_pi_fold = build_rv64im_main_recursion_construction2_pi_fold_from_relation(relation)?;
        let fresh = adapt_rv64im_chunk_to_fresh_ccs(&relation.witness.handoff);

        step_cover_shape = step_cover_shape.recursive_step_cover_merge(&Rv64imChunkStepIvcShape {
            terminal_step: false,
            state_in_claim_count: accumulator_in.carry.main.claims.len() as u64,
            state_out_claim_count: relation.witness.state_out.carry.main.claims.len() as u64,
            fresh_claim_count: fresh.fresh_claims.len() as u64,
            fresh_witness_count: fresh.fresh_witnesses.len() as u64,
            ccs_output_count: construction2_pi_fold.ccs_output_payloads.len() as u64,
            child_count: construction2_pi_fold.dec_child_payloads.len() as u64,
            transcript_in_absorbed: accumulator_in.transcript.absorbed as u64,
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
    let w_perp = Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage {
        binary_values: vec![F::ZERO; full_width - RV64IM_ENC_INST_BITS],
    };
    let x_i = crate::rv64im::f_prime::Rv64imEncodedPublicInput::from_digest_bytes([0; 32]);
    let mut full_vector = Vec::with_capacity(full_width);
    full_vector.extend(x_i.field_image());
    full_vector.extend_from_slice(w_perp.binary_values());
    let u_perp = build_rv64im_main_recursion_construction2_fresh_instance_from_full_vector(0, x_i, &full_vector)?;
    Ok(Rv64imMainRecursionConstruction2DefaultPair { u_perp, w_perp })
}
