//! Owns default-pair derivation for the native RV32IM Construction-2 surface.
//!
//! HyperNova Def. 12 requires the canonical `u_perp` default instance to be a
//! pure function of `(pp, s)`. The default instance carries only zero `C` and
//! zero `x_i`; the terminal SuperNeo R2 proof owns the committed F' witness.
//!
//! Relation-derived cover builders remain available as audit helpers so tests
//! can compare honest traces against the structural builder.

use std::io::{self, Write};
use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::build_dims_and_policy;
use serde::{Deserialize, Serialize};

use crate::proof::Carry;
use crate::rv32im::ccs::{RV32IM_ROOT_PUBLIC_INPUTS, RV32IM_ROOT_ROW_WIDTH};
use crate::rv32im::chunk_fold_step::adapt_rv32im_chunk_to_fresh_ccs;
use crate::rv32im::chunk_step_ivc::Rv32imChunkStepIvcRelation;
use crate::rv32im::construction2::{
    build_rv32im_main_recursion_construction2_pi_fold_from_relation,
    build_rv32im_main_recursion_construction2_verified_step_statement_from_relation,
    Rv32imMainRecursionConstruction2Commitment, Rv32imMainRecursionConstruction2FPrimeCcsShape,
    Rv32imMainRecursionConstruction2FreshInstance,
};
use crate::rv32im::f_prime::{
    Rv32imMainRecursionPhiSide, Rv32imVerifierKeyFs, RV32IM_ENC_INST_BITS, RV32IM_ENC_INST_RING_DEGREE,
    RV32IM_ENC_INST_RING_SLOTS,
};
use crate::rv32im::final_relation::Rv32imChunkFoldState;
use crate::rv32im::main_relation_spartan::{
    Rv32imCcsClaimShape, Rv32imCcsWitnessShape, Rv32imCeClaimDigestShape, Rv32imChunkStepIvcShape,
    Rv32imMainRecursionFPrimeClaimCover,
};
use crate::rv32im::main_relation_trace::build_rv32im_main_circuit_chunk_trace_from_authoritative_parts;
use crate::rv32im::recursion_shape::build_rv32im_recursion_shape_for_step_cap;
use crate::rv32im::SimpleKernelError;
use crate::witness_layout::commit_cols_for_full_width;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imMainRecursionConstruction2DefaultPair {
    u_perp: Rv32imMainRecursionConstruction2FreshInstance,
}

impl Rv32imMainRecursionConstruction2DefaultPair {
    pub fn u_perp(&self) -> &Rv32imMainRecursionConstruction2FreshInstance {
        &self.u_perp
    }
}

fn canonical_phi_side_commitment_word_lens(phi_side: &Rv32imMainRecursionPhiSide) -> Vec<u64> {
    phi_side
        .commitment_words()
        .iter()
        .map(|words| words.len() as u64)
        .collect()
}

fn build_root_ce_claim_shape_for_step_cap(step_cap: usize) -> Result<Rv32imCeClaimDigestShape, SimpleKernelError> {
    let (params, _, structure) = crate::rv32im::kernel::rv32im_root_main_lane_context_for_step_cap(step_cap)?;
    build_root_ce_claim_shape_from_params(&params, structure)
}

fn build_root_ce_claim_shape_for_claim_count(
    claim_count: usize,
) -> Result<Rv32imCeClaimDigestShape, SimpleKernelError> {
    let (params, _, structure) = crate::rv32im::kernel::rv32im_root_main_lane_context_for_claim_count(claim_count)?;
    build_root_ce_claim_shape_from_params(&params, structure)
}

fn build_root_ce_claim_shape_from_params(
    params: &NeoParams,
    structure: &neo_ccs::CcsStructure<F>,
) -> Result<Rv32imCeClaimDigestShape, SimpleKernelError> {
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Build(format!("RV32IM canonical CE claim shape dims failed: {err}")))?;
    let d_pad = 1usize
        .checked_shl(dims.ell_d as u32)
        .ok_or_else(|| SimpleKernelError::Build("RV32IM canonical CE claim d_pad overflow".into()))?;
    Ok(Rv32imCeClaimDigestShape {
        commitment_d: D as u64,
        commitment_kappa: params.kappa as u64,
        c_data_len: (D * params.kappa as usize) as u64,
        x_rows: D as u64,
        x_cols: RV32IM_ROOT_PUBLIC_INPUTS as u64,
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

fn build_root_ccs_claim_shape_for_step_cap(step_cap: usize) -> Result<Rv32imCcsClaimShape, SimpleKernelError> {
    let params = crate::rv32im::kernel::rv32im_simple_root_params_for_step_cap(step_cap);
    Ok(Rv32imCcsClaimShape {
        commitment_d: D as u64,
        commitment_kappa: params.kappa as u64,
        c_data_len: (D * params.kappa as usize) as u64,
        x_len: RV32IM_ROOT_PUBLIC_INPUTS as u64,
    })
}

fn build_root_ccs_witness_shape() -> Rv32imCcsWitnessShape {
    Rv32imCcsWitnessShape {
        w_len: (RV32IM_ROOT_ROW_WIDTH - RV32IM_ROOT_PUBLIC_INPUTS) as u64,
        z_rows: D as u64,
        z_cols: commit_cols_for_full_width(RV32IM_ROOT_ROW_WIDTH) as u64,
    }
}

pub(crate) fn build_rv32im_main_recursion_canonical_zero_carry_for_claim_count(
    claim_count: usize,
) -> Result<Carry, SimpleKernelError> {
    if claim_count == 0 {
        return Err(SimpleKernelError::Bridge(
            "RV32IM canonical zero carry requires at least one carried claim".into(),
        ));
    }
    let claim_shape = build_root_ce_claim_shape_for_claim_count(claim_count)?;
    let witness_shape = build_root_ccs_witness_shape();
    let zero_claim = claim_shape.zero_claim();
    let zero_witness = witness_shape.zero_witness().Z;
    Ok(Carry {
        claims: vec![zero_claim; claim_count],
        witnesses: vec![zero_witness; claim_count],
    })
}

pub fn build_rv32im_main_recursion_construction2_canonical_shape(
    vk_fs: &Rv32imVerifierKeyFs,
    phi_side: &Rv32imMainRecursionPhiSide,
) -> Result<Rv32imMainRecursionConstruction2FPrimeCcsShape, SimpleKernelError> {
    let step_cap = vk_fs.step_cap()?;
    let recursion_shape = build_rv32im_recursion_shape_for_step_cap(step_cap)?;
    let recursion_shape_digest = recursion_shape.canonical_digest();
    if vk_fs.main_lane_shape_digest != recursion_shape_digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Construction-2 canonical shape builder requires the canonical recursion verifier-key shape".into(),
        ));
    }

    let (params, _, structure) = crate::rv32im::kernel::rv32im_root_main_lane_context_for_step_cap(step_cap)?;
    let dims = build_dims_and_policy(&params, structure)
        .map_err(|err| SimpleKernelError::Build(format!("RV32IM canonical F' shape dims failed: {err}")))?;
    let ce_claim_shape = build_root_ce_claim_shape_for_step_cap(step_cap)?;
    let ccs_claim_shape = build_root_ccs_claim_shape_for_step_cap(step_cap)?;
    let ccs_witness_shape = build_root_ccs_witness_shape();
    let carried_claim_count = params.k_rho as usize;
    let ccs_output_count = carried_claim_count
        .checked_add(step_cap)
        .ok_or_else(|| SimpleKernelError::Build("RV32IM canonical F' ccs_output count overflow".into()))?;
    let round_len = (dims.d_sc + 1) as u64;

    Ok(Rv32imMainRecursionConstruction2FPrimeCcsShape {
        verifier_key_fs_digest: vk_fs.expected_digest(),
        recursion_shape_digest,
        x_i_bit_len: RV32IM_ENC_INST_BITS as u64,
        x_i_ring_slot_count: RV32IM_ENC_INST_RING_SLOTS as u64,
        x_i_ring_degree: RV32IM_ENC_INST_RING_DEGREE as u64,
        phi_side_commitment_word_lens: canonical_phi_side_commitment_word_lens(phi_side),
        step_cover_shape: Rv32imChunkStepIvcShape {
            terminal_step: false,
            state_in_claim_count: carried_claim_count as u64,
            state_out_claim_count: carried_claim_count as u64,
            fresh_claim_count: step_cap as u64,
            fresh_witness_count: step_cap as u64,
            ccs_output_count: ccs_output_count as u64,
            child_count: carried_claim_count as u64,
            transcript_in_absorbed: 0,
            transcript_out_absorbed: 0,
            fe_round_lengths: vec![round_len; dims.ell_n + dims.ell_d],
            nc_round_lengths: vec![round_len; dims.ell_m + dims.ell_d],
        },
        claim_cover: Rv32imMainRecursionFPrimeClaimCover {
            state_in_claim_shapes: vec![ce_claim_shape.clone(); carried_claim_count],
            state_out_claim_shapes: vec![ce_claim_shape.clone(); carried_claim_count],
            fresh_claim_shapes: vec![ccs_claim_shape; step_cap],
            fresh_witness_shapes: vec![ccs_witness_shape; step_cap],
            parent_claim_shape: ce_claim_shape.clone(),
            ccs_output_shapes: vec![ce_claim_shape.clone(); ccs_output_count],
            child_claim_shapes: vec![ce_claim_shape; carried_claim_count],
        },
    })
}

pub fn build_rv32im_main_recursion_construction2_canonical_full_width(
    vk_fs: &Rv32imVerifierKeyFs,
    phi_side: &Rv32imMainRecursionPhiSide,
) -> Result<usize, SimpleKernelError> {
    build_rv32im_main_recursion_construction2_default_full_width_from_ccs_shape(
        &build_rv32im_main_recursion_construction2_canonical_shape(vk_fs, phi_side)?,
    )
}

pub fn build_rv32im_main_recursion_construction2_default_full_width_from_ccs_shape(
    shape: &Rv32imMainRecursionConstruction2FPrimeCcsShape,
) -> Result<usize, SimpleKernelError> {
    if shape.claim_cover.state_in_claim_shapes.len() != shape.step_cover_shape.state_in_claim_count as usize
        || shape.claim_cover.fresh_claim_shapes.len() != shape.step_cover_shape.fresh_claim_count as usize
        || shape.claim_cover.fresh_witness_shapes.len() != shape.step_cover_shape.fresh_witness_count as usize
        || shape.claim_cover.ccs_output_shapes.len() != shape.step_cover_shape.ccs_output_count as usize
        || shape.claim_cover.child_claim_shapes.len() != shape.step_cover_shape.child_count as usize
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Construction-2 default-pair width requires a canonical fixed native F' shape cover".into(),
        ));
    }
    Ok(RV32IM_ENC_INST_BITS)
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
pub fn build_rv32im_main_recursion_construction2_default_shape_cover_from_relations(
    vk_fs: &Rv32imVerifierKeyFs,
    accumulator_in: &Rv32imChunkFoldState,
    relations: &[Rv32imChunkStepIvcRelation],
    phi_side: &Rv32imMainRecursionPhiSide,
) -> Result<Rv32imMainRecursionConstruction2FPrimeCcsShape, SimpleKernelError> {
    if relations.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Construction-2 default width cover requires at least one relation".into(),
        ));
    }
    let recursion_shape_digest = build_rv32im_recursion_shape_for_step_cap(vk_fs.step_cap()?)?.canonical_digest();
    if vk_fs.main_lane_shape_digest != recursion_shape_digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Construction-2 default width cover requires the canonical recursion verifier-key shape".into(),
        ));
    }

    let mut step_cover_shape = Rv32imChunkStepIvcShape::recursive_step_cover_seed();
    let mut phi_side_commitment_word_lens = Vec::new();
    let mut state_in_claim_shapes = Vec::new();
    let mut state_out_claim_shapes = Vec::new();
    let mut fresh_claim_shapes = Vec::new();
    let mut fresh_witness_shapes = Vec::new();
    let mut parent_claim_shape: Option<Rv32imCeClaimDigestShape> = None;
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
                "RV32IM Construction-2 default width cover requires a contiguous relation-owned carried state chain"
                    .into(),
            ));
        }
        let native_verified_step_statement =
            build_rv32im_main_recursion_construction2_verified_step_statement_from_relation(relation)?;
        let main_circuit_chunk_summary = native_verified_step_statement.fixed_shape_chunk_summary()?;
        let main_circuit_chunk_trace = build_rv32im_main_circuit_chunk_trace_from_authoritative_parts(
            relation.witness.handoff.bridge_handoff.chunk_index as usize,
            &relation.witness.handoff,
            &main_circuit_chunk_summary,
            &running_state.carry,
            &relation.witness.state_out.carry,
            &running_state.transcript,
            &relation.witness.state_out.transcript,
            &relation.witness.replay_witness,
        )?;
        let construction2_pi_fold = build_rv32im_main_recursion_construction2_pi_fold_from_relation(relation)?;
        let fresh = adapt_rv32im_chunk_to_fresh_ccs(&relation.witness.handoff);
        merge_claim_shape_cover(&mut state_in_claim_shapes, &running_state.carry.main.claims);

        step_cover_shape = step_cover_shape.recursive_step_cover_merge(&Rv32imChunkStepIvcShape {
            terminal_step: false,
            state_in_claim_count: running_state.carry.main.claims.len() as u64,
            state_out_claim_count: relation.witness.state_out.carry.main.claims.len() as u64,
            fresh_claim_count: fresh.fresh_claims.len() as u64,
            fresh_witness_count: fresh.fresh_witnesses.len() as u64,
            ccs_output_count: (running_state.carry.main.claims.len() + fresh.fresh_claims.len()) as u64,
            child_count: relation.witness.state_out.carry.main.claims.len() as u64,
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
        let trace_parent_shape = Rv32imCeClaimDigestShape::from_claim(&main_circuit_chunk_trace.ccs_trace.parent);
        parent_claim_shape = Some(match parent_claim_shape {
            Some(existing) => existing.merge(&trace_parent_shape),
            None => trace_parent_shape,
        });
        merge_claim_shape_cover(&mut ccs_output_shapes, &main_circuit_chunk_trace.ccs_trace.ccs_outputs);
        merge_claim_shape_cover(&mut child_claim_shapes, &main_circuit_chunk_trace.ccs_trace.children);
        running_state = relation.witness.state_out.clone();
    }

    Ok(Rv32imMainRecursionConstruction2FPrimeCcsShape {
        verifier_key_fs_digest: vk_fs.expected_digest(),
        recursion_shape_digest,
        x_i_bit_len: RV32IM_ENC_INST_BITS as u64,
        x_i_ring_slot_count: RV32IM_ENC_INST_RING_SLOTS as u64,
        x_i_ring_degree: RV32IM_ENC_INST_RING_DEGREE as u64,
        phi_side_commitment_word_lens,
        step_cover_shape,
        claim_cover: Rv32imMainRecursionFPrimeClaimCover {
            state_in_claim_shapes,
            state_out_claim_shapes,
            fresh_claim_shapes,
            fresh_witness_shapes,
            parent_claim_shape: parent_claim_shape.ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV32IM Construction-2 default width cover requires at least one parent CE claim shape".into(),
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

fn merge_claim_shape_cover(slots: &mut Vec<Rv32imCeClaimDigestShape>, claims: &[CeClaim<Commitment, F, K>]) {
    for (idx, claim) in claims.iter().enumerate() {
        let shape = Rv32imCeClaimDigestShape::from_claim(claim);
        if let Some(existing) = slots.get_mut(idx) {
            *existing = existing.merge(&shape);
        } else {
            slots.push(shape);
        }
    }
}

fn merge_ccs_claim_shape_cover(slots: &mut Vec<Rv32imCcsClaimShape>, claims: &[CcsClaim<Commitment, F>]) {
    for (idx, claim) in claims.iter().enumerate() {
        let shape = Rv32imCcsClaimShape::from_claim(claim);
        if let Some(existing) = slots.get_mut(idx) {
            *existing = existing.merge(&shape);
        } else {
            slots.push(shape);
        }
    }
}

fn merge_ccs_witness_shape_cover(slots: &mut Vec<Rv32imCcsWitnessShape>, witnesses: &[CcsWitness<F>]) {
    for (idx, witness) in witnesses.iter().enumerate() {
        let shape = Rv32imCcsWitnessShape::from_witness(witness);
        if let Some(existing) = slots.get_mut(idx) {
            *existing = existing.merge(&shape);
        } else {
            slots.push(shape);
        }
    }
}

pub fn build_rv32im_main_recursion_construction2_default_full_width_from_relations(
    vk_fs: &Rv32imVerifierKeyFs,
    accumulator_in: &Rv32imChunkFoldState,
    relations: &[Rv32imChunkStepIvcRelation],
    phi_side: &Rv32imMainRecursionPhiSide,
) -> Result<usize, SimpleKernelError> {
    build_rv32im_main_recursion_construction2_default_full_width_from_ccs_shape(
        &build_rv32im_main_recursion_construction2_default_shape_cover_from_relations(
            vk_fs,
            accumulator_in,
            relations,
            phi_side,
        )?,
    )
}

pub fn build_rv32im_main_recursion_construction2_default_pair_for_full_width(
    vk_fs: &Rv32imVerifierKeyFs,
    full_width: usize,
) -> Result<Rv32imMainRecursionConstruction2DefaultPair, SimpleKernelError> {
    let expected_vk_fs =
        crate::rv32im::f_prime::build_rv32im_main_recursion_verifier_key_fs_for_step_cap(vk_fs.step_cap()?)?;
    if vk_fs != &expected_vk_fs {
        return Err(SimpleKernelError::Bridge(
            "RV32IM native Construction-2 default pair requires the canonical recursion verifier-key context".into(),
        ));
    }
    if full_width < RV32IM_ENC_INST_BITS {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM native Construction-2 default pair full width {full_width} is smaller than the 256-bit x image"
        )));
    }
    let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV32IM Construction-2 default pair params failed for full width {full_width}: {err}"
        ))
    })?;
    let x_i = crate::rv32im::f_prime::Rv32imEncodedPublicInput::from_digest_bytes([0; 32]);
    let u_perp = Rv32imMainRecursionConstruction2FreshInstance::from_parts(
        Rv32imMainRecursionConstruction2Commitment::from_commitment(Commitment::zeros(D, params.kappa as usize)),
        x_i,
    );
    Ok(Rv32imMainRecursionConstruction2DefaultPair { u_perp })
}

pub(crate) fn debug_trace_build_rv32im_main_recursion_construction2_default_pair_for_full_width(
    vk_fs: &Rv32imVerifierKeyFs,
    full_width: usize,
    trace_prefix: &str,
) -> Result<Rv32imMainRecursionConstruction2DefaultPair, SimpleKernelError> {
    build_rv32im_main_recursion_construction2_default_pair_for_full_width_impl(vk_fs, full_width, Some(trace_prefix))
}

fn emit_debug_timing(trace_prefix: Option<&str>, label: &str, elapsed_ms: f64) {
    if let Some(prefix) = trace_prefix {
        eprintln!("{prefix}.{label}={elapsed_ms:.2}ms");
        let _ = io::stderr().flush();
    }
}

fn build_rv32im_main_recursion_construction2_default_pair_for_full_width_impl(
    vk_fs: &Rv32imVerifierKeyFs,
    full_width: usize,
    trace_prefix: Option<&str>,
) -> Result<Rv32imMainRecursionConstruction2DefaultPair, SimpleKernelError> {
    let expected_vk_fs =
        crate::rv32im::f_prime::build_rv32im_main_recursion_verifier_key_fs_for_step_cap(vk_fs.step_cap()?)?;
    if vk_fs != &expected_vk_fs {
        return Err(SimpleKernelError::Bridge(
            "RV32IM native Construction-2 default pair requires the canonical recursion verifier-key context".into(),
        ));
    }
    if full_width < RV32IM_ENC_INST_BITS {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM native Construction-2 default pair full width {full_width} is smaller than the 256-bit x image"
        )));
    }
    let started = Instant::now();
    let x_i = crate::rv32im::f_prime::Rv32imEncodedPublicInput::from_digest_bytes([0; 32]);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV32IM Construction-2 default pair params failed for full width {full_width}: {err}"
        ))
    })?;
    emit_debug_timing(trace_prefix, "u_perp_params", started.elapsed().as_secs_f64() * 1_000.0);
    let u_perp = Rv32imMainRecursionConstruction2FreshInstance::from_parts(
        Rv32imMainRecursionConstruction2Commitment::from_commitment(Commitment::zeros(D, params.kappa as usize)),
        x_i,
    );
    Ok(Rv32imMainRecursionConstruction2DefaultPair { u_perp })
}
