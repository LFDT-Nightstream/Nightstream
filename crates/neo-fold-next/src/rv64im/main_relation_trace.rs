//! Owns the native RV64IM main-circuit trace builder one chunk at a time.
//!
//! This module bridges the verified chunk-fold step chain into the concrete
//! replay artifacts consumed by the current Spartan circuit. It is not
//! theorem-facing and does not own circuit synthesis.

#![allow(dead_code)]

use neo_ajtai::{AjtaiSModule, Commitment};
use neo_ccs::{build_superneo_ring_forms, CcsClaim, CcsStructure, CcsWitness, Mat, SModuleHomomorphism};
use neo_math::{balanced::to_balanced_i128, KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{rlc_public, sample_rot_rhos_n_typed, RotRing};
use neo_reductions::common::{
    compute_y_zcol_from_witness, compute_y_zcol_from_witness_digits, decode_superneo_coeffs_from_witness_mat,
    project_x_from_witness_mat,
};
use neo_reductions::engines::utils::{build_dims_and_policy, Dims};
use neo_reductions::optimized_engine::{
    optimized_replay_terminal_state_with_cache_and_instance_digest_and_perf, Challenges, OptimizedStructureCache,
    PiCcsReplayProofWitness, PiCcsReplayTerminalState,
};
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};

use crate::chunk_relation::ChunkReplayWitness;
use crate::finalize::FixedShapeChunkSummary;
use crate::proof::PublicChunk;
use crate::rv64im::chunk_fold_step::Rv64imChunkFoldCarry;
use crate::rv64im::chunk_relation::{trace_rv64im_chunk_relation_with_replay, Rv64imChunkRelationTrace};
use crate::rv64im::final_relation::{rv64im_chunk_fold_carried_transcript_snapshot, Rv64imChunkFoldTranscriptSnapshot};
use crate::rv64im::kernel::{
    rv64im_ajtai_mixers, rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache,
    rv64im_root_main_lane_context_for_claim_count, Rv64imVerifiedKernelChunkHandoff, SimpleKernelError,
};
use crate::rv64im::main_relation_circuit::structure::pad_ccs_structure_to_block_width;

pub(crate) const CHUNK_META_RAW_TAG: u64 = 14;

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainCircuitHandoff {
    pub(crate) public_chunk: PublicChunk,
    pub(crate) public_chunk_instance_digest: [F; 4],
    pub(crate) public_chunk_digest: [u8; 32],
    pub(crate) bridge_handoff_digest: [u8; 32],
    pub(crate) chunk_relation_digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainCircuitChunkTrace {
    pub(crate) handoff: Rv64imMainCircuitHandoff,
    pub(crate) fresh_claims: Vec<CcsClaim<Commitment, F>>,
    pub(crate) fresh_witnesses: Vec<CcsWitness<F>>,
    pub(crate) ccs_trace: Rv64imChunkRelationTrace,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainCircuitPiCcsReplaySurface {
    pub(crate) ccs_outputs: Vec<neo_ccs::CeClaim<Commitment, F, K>>,
    pub(crate) replay_proof: PiCcsReplayProofWitness,
    pub(crate) public_challenges: Challenges,
    pub(crate) row_chals: Vec<K>,
    pub(crate) alpha_prime: Vec<K>,
    pub(crate) s_col: Vec<K>,
    pub(crate) alpha_prime_nc: Vec<K>,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainCircuitPiRlcReplaySurface {
    pub(crate) parent: neo_ccs::CeClaim<Commitment, F, K>,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainCircuitPiDecReplaySurface {
    pub(crate) children: Vec<neo_ccs::CeClaim<Commitment, F, K>>,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainCircuitChunkReplaySurface {
    pub(crate) handoff: Rv64imMainCircuitHandoff,
    pub(crate) fresh_claims: Vec<CcsClaim<Commitment, F>>,
    pub(crate) fresh_witnesses: Vec<CcsWitness<F>>,
    pub(crate) pi_ccs: Rv64imMainCircuitPiCcsReplaySurface,
    pub(crate) pi_rlc: Rv64imMainCircuitPiRlcReplaySurface,
    pub(crate) pi_dec: Rv64imMainCircuitPiDecReplaySurface,
}

impl Rv64imMainCircuitChunkTrace {
    pub(crate) fn step_lo(&self) -> u64 {
        self.handoff.public_chunk.start_index as u64
    }

    pub(crate) fn step_hi(&self) -> u64 {
        self.step_lo() + self.handoff.public_chunk.steps.len() as u64
    }

    pub(crate) fn replay_surface(&self) -> Result<Rv64imMainCircuitChunkReplaySurface, SimpleKernelError> {
        build_rv64im_main_circuit_chunk_replay_surface(
            &self.handoff,
            &self.fresh_claims,
            &self.fresh_witnesses,
            build_rv64im_main_circuit_pi_ccs_replay_surface(
                self.ccs_trace.ccs_outputs.clone(),
                self.ccs_trace.ccs_replay_proof.clone(),
                self.ccs_trace.terminal_state.challenges_public.clone(),
                self.ccs_trace.terminal_state.row_chals.clone(),
                self.ccs_trace.terminal_state.alpha_prime.clone(),
                self.ccs_trace.terminal_state.s_col.clone(),
                self.ccs_trace.terminal_state.alpha_prime_nc.clone(),
            ),
            self.ccs_trace.parent.clone(),
            self.ccs_trace.children.clone(),
        )
    }
}

pub(crate) fn build_rv64im_main_circuit_chunk_replay_surface(
    handoff: &Rv64imMainCircuitHandoff,
    fresh_claims: &[CcsClaim<Commitment, F>],
    fresh_witnesses: &[CcsWitness<F>],
    pi_ccs: Rv64imMainCircuitPiCcsReplaySurface,
    parent: neo_ccs::CeClaim<Commitment, F, K>,
    children: Vec<neo_ccs::CeClaim<Commitment, F, K>>,
) -> Result<Rv64imMainCircuitChunkReplaySurface, SimpleKernelError> {
    Ok(Rv64imMainCircuitChunkReplaySurface {
        handoff: handoff.clone(),
        fresh_claims: fresh_claims.to_vec(),
        fresh_witnesses: fresh_witnesses.to_vec(),
        pi_ccs,
        pi_rlc: Rv64imMainCircuitPiRlcReplaySurface { parent },
        pi_dec: Rv64imMainCircuitPiDecReplaySurface { children },
    })
}

pub(crate) fn build_rv64im_main_circuit_pi_ccs_replay_surface(
    ccs_outputs: Vec<neo_ccs::CeClaim<Commitment, F, K>>,
    replay_proof: PiCcsReplayProofWitness,
    public_challenges: Challenges,
    row_chals: Vec<K>,
    alpha_prime: Vec<K>,
    s_col: Vec<K>,
    alpha_prime_nc: Vec<K>,
) -> Rv64imMainCircuitPiCcsReplaySurface {
    Rv64imMainCircuitPiCcsReplaySurface {
        ccs_outputs,
        replay_proof,
        public_challenges,
        row_chals,
        alpha_prime,
        s_col,
        alpha_prime_nc,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Rv64imMainCircuitCeClaimShape {
    pub(crate) commitment_d: u64,
    pub(crate) commitment_kappa: u64,
    pub(crate) c_data_len: u64,
    pub(crate) x_rows: u64,
    pub(crate) x_cols: u64,
    pub(crate) r_len: u64,
    pub(crate) s_col_len: u64,
    pub(crate) y_ring_row_count: u64,
    pub(crate) y_ring_row_lens: Vec<u64>,
    pub(crate) ct_len: u64,
    pub(crate) aux_openings_len: u64,
    pub(crate) y_zcol_len: u64,
    pub(crate) c_step_coords_len: u64,
}

impl Rv64imMainCircuitCeClaimShape {
    pub(crate) fn from_claim(claim: &neo_ccs::CeClaim<Commitment, F, K>) -> Self {
        Self {
            commitment_d: claim.c.d as u64,
            commitment_kappa: claim.c.kappa as u64,
            c_data_len: claim.c.data.len() as u64,
            x_rows: claim.X.rows() as u64,
            x_cols: claim.X.cols() as u64,
            r_len: claim.r.len() as u64,
            s_col_len: claim.s_col.len() as u64,
            y_ring_row_count: claim.y_ring.len() as u64,
            y_ring_row_lens: claim.y_ring.iter().map(|row| row.len() as u64).collect(),
            ct_len: claim.ct.len() as u64,
            aux_openings_len: claim.aux_openings.len() as u64,
            y_zcol_len: claim.y_zcol.len() as u64,
            c_step_coords_len: claim.c_step_coords.len() as u64,
        }
    }

    pub(crate) fn covers_claim(&self, claim: &neo_ccs::CeClaim<Commitment, F, K>) -> bool {
        self.commitment_d == claim.c.d as u64
            && self.commitment_kappa == claim.c.kappa as u64
            && self.c_data_len == claim.c.data.len() as u64
            && self.x_rows >= claim.X.rows() as u64
            && self.x_cols >= claim.X.cols() as u64
            && self.r_len >= claim.r.len() as u64
            && self.s_col_len >= claim.s_col.len() as u64
            && self.y_ring_row_count >= claim.y_ring.len() as u64
            && claim
                .y_ring
                .iter()
                .enumerate()
                .all(|(idx, row)| self.y_ring_row_lens.get(idx).copied().unwrap_or(0) >= row.len() as u64)
            && self.ct_len >= claim.ct.len() as u64
            && self.aux_openings_len >= claim.aux_openings.len() as u64
            && self.y_zcol_len >= claim.y_zcol.len() as u64
            && self.c_step_coords_len >= claim.c_step_coords.len() as u64
    }

    pub(crate) fn zero_claim(&self) -> neo_ccs::CeClaim<Commitment, F, K> {
        neo_ccs::CeClaim {
            c: Commitment::zeros(self.commitment_d as usize, self.commitment_kappa as usize),
            X: Mat::zero(self.x_rows as usize, self.x_cols as usize, F::ZERO),
            r: vec![K::ZERO; self.r_len as usize],
            s_col: vec![K::ZERO; self.s_col_len as usize],
            y_ring: self
                .y_ring_row_lens
                .iter()
                .map(|len| vec![K::ZERO; *len as usize])
                .collect(),
            ct: vec![K::ZERO; self.ct_len as usize],
            aux_openings: vec![K::ZERO; self.aux_openings_len as usize],
            y_zcol: vec![K::ZERO; self.y_zcol_len as usize],
            m_in: self.x_cols as usize,
            fold_digest: [0; 32],
            c_step_coords: vec![F::ZERO; self.c_step_coords_len as usize],
            u_offset: 0,
            u_len: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Rv64imMainCircuitCcsClaimShape {
    pub(crate) commitment_d: u64,
    pub(crate) commitment_kappa: u64,
    pub(crate) c_data_len: u64,
    pub(crate) x_len: u64,
}

impl Rv64imMainCircuitCcsClaimShape {
    pub(crate) fn from_claim(claim: &CcsClaim<Commitment, F>) -> Self {
        Self {
            commitment_d: claim.c.d as u64,
            commitment_kappa: claim.c.kappa as u64,
            c_data_len: claim.c.data.len() as u64,
            x_len: claim.x.len() as u64,
        }
    }

    pub(crate) fn covers_claim(&self, claim: &CcsClaim<Commitment, F>) -> bool {
        self.commitment_d == claim.c.d as u64
            && self.commitment_kappa == claim.c.kappa as u64
            && self.c_data_len == claim.c.data.len() as u64
            && self.x_len >= claim.x.len() as u64
    }

    pub(crate) fn zero_claim(&self) -> CcsClaim<Commitment, F> {
        CcsClaim {
            c: Commitment::zeros(self.commitment_d as usize, self.commitment_kappa as usize),
            x: vec![F::ZERO; self.x_len as usize],
            m_in: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Rv64imMainCircuitCcsWitnessShape {
    pub(crate) w_len: u64,
    pub(crate) z_rows: u64,
    pub(crate) z_cols: u64,
}

impl Rv64imMainCircuitCcsWitnessShape {
    pub(crate) fn from_witness(witness: &CcsWitness<F>) -> Self {
        Self {
            w_len: witness.w.len() as u64,
            z_rows: witness.Z.rows() as u64,
            z_cols: witness.Z.cols() as u64,
        }
    }

    pub(crate) fn covers_witness(&self, witness: &CcsWitness<F>) -> bool {
        self.w_len >= witness.w.len() as u64
            && self.z_rows >= witness.Z.rows() as u64
            && self.z_cols >= witness.Z.cols() as u64
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Rv64imMainCircuitChunkCover {
    pub(crate) fresh_claim_count: u64,
    pub(crate) fresh_witness_count: u64,
    pub(crate) fresh_claim_shapes: Vec<Rv64imMainCircuitCcsClaimShape>,
    pub(crate) fresh_witness_shapes: Vec<Rv64imMainCircuitCcsWitnessShape>,
    pub(crate) ccs_output_count: u64,
    pub(crate) child_count: u64,
    pub(crate) parent_claim_shape: Rv64imMainCircuitCeClaimShape,
    pub(crate) ccs_output_shapes: Vec<Rv64imMainCircuitCeClaimShape>,
    pub(crate) child_claim_shapes: Vec<Rv64imMainCircuitCeClaimShape>,
    pub(crate) fe_round_lengths: Vec<u64>,
    pub(crate) nc_round_lengths: Vec<u64>,
}

impl Rv64imMainCircuitChunkCover {
    pub(crate) fn from_trace(trace: &Rv64imMainCircuitChunkTrace) -> Self {
        Self {
            fresh_claim_count: trace.fresh_claims.len() as u64,
            fresh_witness_count: trace.fresh_witnesses.len() as u64,
            fresh_claim_shapes: trace
                .fresh_claims
                .iter()
                .map(Rv64imMainCircuitCcsClaimShape::from_claim)
                .collect(),
            fresh_witness_shapes: trace
                .fresh_witnesses
                .iter()
                .map(Rv64imMainCircuitCcsWitnessShape::from_witness)
                .collect(),
            ccs_output_count: trace.ccs_trace.ccs_outputs.len() as u64,
            child_count: trace.ccs_trace.children.len() as u64,
            parent_claim_shape: Rv64imMainCircuitCeClaimShape::from_claim(&trace.ccs_trace.parent),
            ccs_output_shapes: trace
                .ccs_trace
                .ccs_outputs
                .iter()
                .map(Rv64imMainCircuitCeClaimShape::from_claim)
                .collect(),
            child_claim_shapes: trace
                .ccs_trace
                .children
                .iter()
                .map(Rv64imMainCircuitCeClaimShape::from_claim)
                .collect(),
            fe_round_lengths: trace
                .ccs_trace
                .ccs_replay_proof
                .sumcheck_rounds
                .iter()
                .map(|round| round.len() as u64)
                .collect(),
            nc_round_lengths: trace
                .ccs_trace
                .ccs_replay_proof
                .sumcheck_rounds_nc
                .iter()
                .map(|round| round.len() as u64)
                .collect(),
        }
    }

    pub(crate) fn covers_replay_surface(&self, surface: &Rv64imMainCircuitChunkReplaySurface) -> bool {
        self.fresh_claim_count >= surface.fresh_claims.len() as u64
            && surface.fresh_claims.iter().enumerate().all(|(idx, claim)| {
                self.fresh_claim_shapes
                    .get(idx)
                    .is_some_and(|shape| shape.covers_claim(claim))
            })
            && self.ccs_output_count >= surface.pi_ccs.ccs_outputs.len() as u64
            && self.child_count >= surface.pi_dec.children.len() as u64
            && self.parent_claim_shape.covers_claim(&surface.pi_rlc.parent)
            && surface
                .pi_ccs
                .ccs_outputs
                .iter()
                .enumerate()
                .all(|(idx, claim)| {
                    self.ccs_output_shapes
                        .get(idx)
                        .is_some_and(|shape| shape.covers_claim(claim))
                })
            && surface
                .pi_dec
                .children
                .iter()
                .enumerate()
                .all(|(idx, claim)| {
                    self.child_claim_shapes
                        .get(idx)
                        .is_some_and(|shape| shape.covers_claim(claim))
                })
            && self.fe_round_lengths.len() >= surface.pi_ccs.replay_proof.sumcheck_rounds.len()
            && self
                .fe_round_lengths
                .iter()
                .zip(surface.pi_ccs.replay_proof.sumcheck_rounds.iter())
                .all(|(cover_len, round)| *cover_len >= round.len() as u64)
            && self.nc_round_lengths.len() >= surface.pi_ccs.replay_proof.sumcheck_rounds_nc.len()
            && self
                .nc_round_lengths
                .iter()
                .zip(surface.pi_ccs.replay_proof.sumcheck_rounds_nc.iter())
                .all(|(cover_len, round)| *cover_len >= round.len() as u64)
    }
}

struct Rv64imMainCircuitTraceBuildContext<'a> {
    params: &'a NeoParams,
    log: &'a AjtaiSModule,
    structure: &'a CcsStructure<F>,
    ce_structure: &'a CcsStructure<F>,
    dims: Dims,
    optimized_cache: &'a OptimizedStructureCache,
}

pub(crate) fn build_rv64im_main_circuit_chunk_trace_from_authoritative_parts(
    chunk_index: usize,
    handoff: &Rv64imVerifiedKernelChunkHandoff,
    chunk_summary: &FixedShapeChunkSummary,
    carry_in: &Rv64imChunkFoldCarry,
    carry_out: &Rv64imChunkFoldCarry,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    transcript_out: &Rv64imChunkFoldTranscriptSnapshot,
    replay_witness: &ChunkReplayWitness,
) -> Result<Rv64imMainCircuitChunkTrace, SimpleKernelError> {
    let (params, log, structure) = rv64im_root_main_lane_context_for_claim_count(carry_in.main.claims.len())?;
    let ce_structure = pad_ccs_structure_to_block_width(structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM padded CE structure failed: {err}")))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let dims = build_dims_and_policy(&params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation dims failed: {err}")))?;
    let ctx = Rv64imMainCircuitTraceBuildContext {
        params: &params,
        log,
        structure,
        ce_structure: &ce_structure,
        dims,
        optimized_cache: &optimized_cache,
    };
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(transcript_in.state, transcript_in.absorbed);
    let chunk_trace = build_rv64im_main_circuit_chunk_trace_from_parts(
        &ctx,
        chunk_index,
        handoff,
        chunk_summary,
        carry_in,
        carry_out,
        transcript_in,
        replay_witness,
        &mut transcript,
    )?;
    let replayed_transcript_out = rv64im_chunk_fold_carried_transcript_snapshot(&Rv64imChunkFoldTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    });
    if &replayed_transcript_out != transcript_out {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation circuit trace transcript_out does not match the carried private transcript state"
                .into(),
        ));
    }
    Ok(chunk_trace)
}

fn build_rv64im_main_circuit_chunk_trace_from_parts(
    ctx: &Rv64imMainCircuitTraceBuildContext<'_>,
    chunk_index: usize,
    handoff: &Rv64imVerifiedKernelChunkHandoff,
    chunk_summary: &FixedShapeChunkSummary,
    carry_in: &Rv64imChunkFoldCarry,
    carry_out: &Rv64imChunkFoldCarry,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    replay_witness: &ChunkReplayWitness,
    transcript: &mut Poseidon2Transcript,
) -> Result<Rv64imMainCircuitChunkTrace, SimpleKernelError> {
    let fresh = crate::rv64im::chunk_fold_step::adapt_rv64im_chunk_to_fresh_ccs(handoff);
    let trace = trace_rv64im_chunk_relation_with_replay(
        chunk_index,
        handoff,
        &carry_in.main,
        replay_witness,
        transcript,
        ctx.params,
        ctx.structure,
        ctx.log,
        ctx.optimized_cache,
    )?;
    let (replayed_terminal_state, mut replay_transcript) = replay_main_relation_pi_ccs_terminal_state(
        ctx,
        transcript_in,
        &handoff.public_chunk,
        handoff.public_chunk_instance_digest,
        &fresh.fresh_claims,
        &fresh.fresh_witnesses,
        &carry_in.main.claims,
        &carry_in.main.witnesses,
    )?;
    check_pi_ccs_terminal_state_native(chunk_index, &trace.terminal_state, &replayed_terminal_state)?;
    check_claim_fold_digest_native(
        &trace.ccs_outputs,
        &trace.parent,
        &trace.children,
        &trace.terminal_state.fold_digest,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM main relation chunk {chunk_index} fold-digest binding failed: {err}"
        ))
    })?;
    check_output_binding_native(
        ctx.structure,
        &fresh.fresh_claims,
        &fresh.fresh_witnesses,
        &carry_in.main.claims,
        &trace.ccs_outputs,
        &trace.terminal_state.row_chals,
        &trace.terminal_state.s_col,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM main relation chunk {chunk_index} output binding failed: {err}"
        ))
    })?;
    let expected_rhos = sample_main_relation_pi_rlc_rhos(&mut replay_transcript, ctx.params, trace.ccs_outputs.len())?;
    let mixers = rv64im_ajtai_mixers();
    let expected_parent = rlc_public(
        ctx.structure,
        ctx.params,
        &expected_rhos,
        &trace.ccs_outputs,
        mixers.mix_rhos_commits,
        ctx.dims.ell_d,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM main relation chunk {chunk_index} Pi_RLC public recompute failed: {err}"
        ))
    })?;
    if let Some(mismatch) = describe_ce_claim_mismatch(&expected_parent, &trace.parent) {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main relation chunk {chunk_index} Pi_RLC parent claim does not match the independently recomputed fold: {mismatch}"
        )));
    }
    let mut ccs_output_zs = fresh
        .fresh_witnesses
        .iter()
        .map(|witness| witness.Z.clone())
        .collect::<Vec<_>>();
    ccs_output_zs.extend(carry_in.main.witnesses.iter().cloned());
    if trace.ccs_outputs.len() != ccs_output_zs.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main relation chunk {chunk_index} output/witness arity mismatch"
        )));
    }
    for (output_index, (claim, z_matrix)) in trace
        .ccs_outputs
        .iter()
        .zip(ccs_output_zs.iter())
        .enumerate()
    {
        check_output_claim_consistency(ctx.params, ctx.structure, ctx.ce_structure, claim, z_matrix).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM main relation chunk {chunk_index} backend consistency failed for ccs_output {output_index}: {err}"
            ))
        })?;
    }
    if chunk_summary.public_chunk_digest != handoff.public_chunk_digest {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main relation chunk {chunk_index} verified public chunk digest mismatch"
        )));
    }
    if carry_out.main.claims != trace.children || carry_out.main.witnesses != trace.z_split {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main relation chunk {chunk_index} trace/verify next-main mismatch"
        )));
    }
    if chunk_summary.chunk_relation_digest != trace.chunk_relation_digest {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main relation chunk {chunk_index} trace chunk_relation_digest does not match the carried summary"
        )));
    }
    for (child_index, (claim, z_matrix)) in trace.children.iter().zip(trace.z_split.iter()).enumerate() {
        check_dec_child_claim_consistency(ctx.params, ctx.structure, ctx.ce_structure, ctx.log, claim, z_matrix)
            .map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM main relation chunk {chunk_index} child {child_index} backend consistency failed: {err}"
                ))
            })?;
    }

    let mut bridge_handoff = handoff.bridge_handoff.clone();
    for binding in &mut bridge_handoff.step_bindings {
        binding.digest = binding.expected_digest();
    }
    bridge_handoff.digest = bridge_handoff.expected_digest();

    Ok(Rv64imMainCircuitChunkTrace {
        handoff: Rv64imMainCircuitHandoff {
            public_chunk: fresh.public_chunk.clone(),
            public_chunk_instance_digest: fresh.public_chunk_instance_digest,
            public_chunk_digest: fresh.public_chunk_digest,
            bridge_handoff_digest: bridge_handoff.digest,
            chunk_relation_digest: trace.chunk_relation_digest,
        },
        fresh_claims: fresh.fresh_claims,
        fresh_witnesses: fresh.fresh_witnesses,
        ccs_trace: trace,
    })
}

fn append_chunk_meta_native(transcript: &mut Poseidon2Transcript, public_chunk: &PublicChunk) {
    transcript.append_fields_raw(&[
        F::from_u64(CHUNK_META_RAW_TAG),
        F::from_u64(public_chunk.start_index as u64),
        F::from_u64(public_chunk.steps.len() as u64),
    ]);
}

fn check_output_binding_native(
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Commitment, F>],
    fresh_witnesses: &[CcsWitness<F>],
    me_inputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    me_outputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    r_prime: &[K],
    s_col_prime: &[K],
) -> Result<(), String> {
    if me_outputs.len() != fresh_claims.len() + me_inputs.len() {
        return Err("output arity mismatch".into());
    }
    if fresh_witnesses.len() != fresh_claims.len() {
        return Err("fresh witness arity mismatch".into());
    }

    for (index, output) in me_outputs.iter().enumerate() {
        if output.r != r_prime {
            return Err(format!("output {index} r mismatch"));
        }
        if output.s_col != s_col_prime {
            return Err(format!("output {index} s_col mismatch"));
        }
        for matrix_index in 0..structure.t() {
            if output.ct.get(matrix_index).copied() != output.y_ring[matrix_index].first().copied() {
                return Err(format!("output {index} ct[{matrix_index}] mismatch"));
            }
        }

        if index < fresh_claims.len() {
            let fresh = &fresh_claims[index];
            let fresh_witness = &fresh_witnesses[index];
            if output.c.data != fresh.c.data {
                return Err(format!("fresh output {index} commitment mismatch"));
            }
            if output.m_in != fresh.m_in {
                return Err(format!("fresh output {index} m_in mismatch"));
            }
            let expected_x = project_x_from_witness_mat(&fresh_witness.Z, structure.m, fresh.m_in)
                .map_err(|err| format!("fresh output {index} X projection failed: {err}"))?;
            if output.X != expected_x {
                return Err(format!("fresh output {index} X mismatch"));
            }
        } else {
            let me_index = index - fresh_claims.len();
            let input = &me_inputs[me_index];
            if output.c.data != input.c.data {
                return Err(format!("me_input output {me_index} commitment mismatch"));
            }
            if output.X != input.X {
                return Err(format!("me_input output {me_index} X mismatch"));
            }
        }
    }

    Ok(())
}

fn check_claim_fold_digest_native(
    outputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    parent: &neo_ccs::CeClaim<Commitment, F, K>,
    children: &[neo_ccs::CeClaim<Commitment, F, K>],
    terminal_fold_digest: &[u8; 32],
) -> Result<(), String> {
    for (index, claim) in outputs.iter().enumerate() {
        if &claim.fold_digest != terminal_fold_digest {
            return Err(format!("ccs_output {index} fold digest mismatch"));
        }
    }
    if &parent.fold_digest != terminal_fold_digest {
        return Err("parent fold digest mismatch".into());
    }
    for (index, claim) in children.iter().enumerate() {
        if &claim.fold_digest != terminal_fold_digest {
            return Err(format!("child {index} fold digest mismatch"));
        }
    }
    Ok(())
}

fn check_output_claim_consistency(
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    claim: &neo_ccs::CeClaim<Commitment, F, K>,
    z_matrix: &neo_ccs::Mat<F>,
) -> Result<(), String> {
    if !(claim.s_col.is_empty() && claim.y_zcol.is_empty()) {
        let chi_s = neo_ccs::tensor_point::<K>(&claim.s_col);
        let y_zcol = compute_y_zcol_from_witness_digits(params, z_matrix, base_structure.m, &chi_s, claim.y_zcol.len())
            .map_err(|err| err.to_string())?;
        if y_zcol != claim.y_zcol {
            return Err("y_zcol != Z_digits · χ_{s_col}".into());
        }
    }

    let z_coeffs =
        decode_superneo_coeffs_from_witness_mat(z_matrix, base_structure.m).map_err(|err| err.to_string())?;
    let ring_forms = build_superneo_ring_forms(ring_structure, &claim.r).map_err(|err| err.to_string())?;
    for (matrix_index, forms) in ring_forms.iter().enumerate() {
        let mut row = vec![K::ZERO; claim.y_ring[matrix_index].len()];
        for logical_col in 0..forms.len() {
            for rho in 0..D {
                row[rho] += forms[logical_col][rho] * z_coeffs[logical_col];
            }
        }
        if row != claim.y_ring[matrix_index] {
            return Err(format!("y_ring[{matrix_index}] mismatch"));
        }
        if claim.ct.get(matrix_index).copied() != row.first().copied() {
            return Err(format!("ct[{matrix_index}] mismatch"));
        }
    }

    Ok(())
}

fn check_dec_child_claim_consistency(
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    log: &AjtaiSModule,
    claim: &neo_ccs::CeClaim<Commitment, F, K>,
    z_matrix: &neo_ccs::Mat<F>,
) -> Result<(), String> {
    if log.commit(z_matrix) != claim.c {
        return Err("c != L(Z)".into());
    }

    let z_coeffs =
        decode_superneo_coeffs_from_witness_mat(z_matrix, base_structure.m).map_err(|err| err.to_string())?;
    let max_digit = i128::from(params.b) - 1;
    for (logical_col, coeff) in z_coeffs.iter().enumerate() {
        let coeffs = coeff.as_coeffs();
        if coeffs[1] != F::ZERO {
            return Err(format!("child logical_col={logical_col} has non-base coefficient"));
        }
        let centered = to_balanced_i128(coeffs[0]);
        if centered.abs() > max_digit {
            return Err(format!(
                "child logical_col={logical_col} is outside the balanced digit alphabet"
            ));
        }
    }

    if !(claim.s_col.is_empty() && claim.y_zcol.is_empty()) {
        let chi_s = neo_ccs::tensor_point::<K>(&claim.s_col);
        let y_zcol = compute_y_zcol_from_witness(params, z_matrix, base_structure.m, &chi_s, claim.y_zcol.len())
            .map_err(|err| err.to_string())?;
        if y_zcol != claim.y_zcol {
            return Err("y_zcol != Z · χ_{s_col}".into());
        }
    }

    let ring_forms = build_superneo_ring_forms(ring_structure, &claim.r).map_err(|err| err.to_string())?;
    for (matrix_index, forms) in ring_forms.iter().enumerate() {
        let mut row = vec![K::ZERO; claim.y_ring[matrix_index].len()];
        for logical_col in 0..forms.len() {
            for rho in 0..D {
                row[rho] += forms[logical_col][rho] * z_coeffs[logical_col];
            }
        }
        if row != claim.y_ring[matrix_index] {
            return Err(format!("y_ring[{matrix_index}] mismatch"));
        }
        if claim.ct.get(matrix_index).copied() != row.first().copied() {
            return Err(format!("ct[{matrix_index}] mismatch"));
        }
    }

    Ok(())
}

fn replay_main_relation_pi_ccs_terminal_state(
    ctx: &Rv64imMainCircuitTraceBuildContext<'_>,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    public_chunk: &PublicChunk,
    public_chunk_instance_digest: [F; 4],
    fresh_claims: &[CcsClaim<Commitment, F>],
    fresh_witnesses: &[CcsWitness<F>],
    me_inputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    me_witnesses: &[Mat<F>],
) -> Result<(PiCcsReplayTerminalState, Poseidon2Transcript), SimpleKernelError> {
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(transcript_in.state, transcript_in.absorbed);
    append_chunk_meta_native(&mut transcript, public_chunk);
    let terminal_state = optimized_replay_terminal_state_with_cache_and_instance_digest_and_perf(
        &mut transcript,
        ctx.params,
        ctx.structure,
        fresh_claims,
        fresh_witnesses,
        me_inputs,
        me_witnesses,
        public_chunk_instance_digest,
        ctx.log,
        ctx.optimized_cache,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation Pi_CCS transcript replay failed: {err}")))?;
    Ok((terminal_state, transcript))
}

pub(crate) fn debug_replay_rv64im_main_relation_pi_ccs_transcript_state(
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    handoff: &Rv64imMainCircuitHandoff,
    fresh_claims: &[CcsClaim<Commitment, F>],
    fresh_witnesses: &[CcsWitness<F>],
    me_inputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    me_witnesses: &[Mat<F>],
) -> Result<Rv64imChunkFoldTranscriptSnapshot, SimpleKernelError> {
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let ce_structure = pad_ccs_structure_to_block_width(structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM padded CE structure failed: {err}")))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation dims failed: {err}")))?;
    let ctx = Rv64imMainCircuitTraceBuildContext {
        params,
        log,
        structure,
        ce_structure: &ce_structure,
        dims,
        optimized_cache: &optimized_cache,
    };
    let (_, transcript) = replay_main_relation_pi_ccs_terminal_state(
        &ctx,
        transcript_in,
        &handoff.public_chunk,
        handoff.public_chunk_instance_digest,
        fresh_claims,
        fresh_witnesses,
        me_inputs,
        me_witnesses,
    )?;
    Ok(Rv64imChunkFoldTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    })
}

pub(crate) fn debug_describe_rv64im_main_relation_pi_ccs_terminal_state_mismatch(
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    handoff: &Rv64imMainCircuitHandoff,
    fresh_claims: &[CcsClaim<Commitment, F>],
    fresh_witnesses: &[CcsWitness<F>],
    me_inputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    me_witnesses: &[Mat<F>],
    live: &PiCcsReplayTerminalState,
) -> Result<String, SimpleKernelError> {
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let ce_structure = pad_ccs_structure_to_block_width(structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM padded CE structure failed: {err}")))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation dims failed: {err}")))?;
    let ctx = Rv64imMainCircuitTraceBuildContext {
        params,
        log,
        structure,
        ce_structure: &ce_structure,
        dims,
        optimized_cache: &optimized_cache,
    };
    let (replayed, _) = replay_main_relation_pi_ccs_terminal_state(
        &ctx,
        transcript_in,
        &handoff.public_chunk,
        handoff.public_chunk_instance_digest,
        fresh_claims,
        fresh_witnesses,
        me_inputs,
        me_witnesses,
    )?;
    Ok(
        describe_pi_ccs_terminal_state_mismatch(live, &replayed)
            .unwrap_or_else(|| "pi_ccs_terminal_state_match".into()),
    )
}

pub(crate) fn debug_describe_rv64im_main_relation_pi_rlc_parent_mismatch(
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    handoff: &Rv64imMainCircuitHandoff,
    fresh_claims: &[CcsClaim<Commitment, F>],
    fresh_witnesses: &[CcsWitness<F>],
    me_inputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    me_witnesses: &[Mat<F>],
    ccs_outputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    live_parent: &neo_ccs::CeClaim<Commitment, F, K>,
) -> Result<String, SimpleKernelError> {
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let ce_structure = pad_ccs_structure_to_block_width(structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM padded CE structure failed: {err}")))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation dims failed: {err}")))?;
    let ctx = Rv64imMainCircuitTraceBuildContext {
        params,
        log,
        structure,
        ce_structure: &ce_structure,
        dims,
        optimized_cache: &optimized_cache,
    };
    let (_, mut replay_transcript) = replay_main_relation_pi_ccs_terminal_state(
        &ctx,
        transcript_in,
        &handoff.public_chunk,
        handoff.public_chunk_instance_digest,
        fresh_claims,
        fresh_witnesses,
        me_inputs,
        me_witnesses,
    )?;
    let expected_rhos = sample_main_relation_pi_rlc_rhos(&mut replay_transcript, ctx.params, ccs_outputs.len())?;
    let mixers = rv64im_ajtai_mixers();
    let expected_parent = rlc_public(
        ctx.structure,
        ctx.params,
        &expected_rhos,
        ccs_outputs,
        mixers.mix_rhos_commits,
        ctx.dims.ell_d,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation Pi_RLC public recompute failed: {err}")))?;
    Ok(describe_ce_claim_mismatch(&expected_parent, live_parent).unwrap_or_else(|| "pi_rlc_parent_match".into()))
}

pub(crate) fn debug_describe_rv64im_main_relation_pi_rlc_x_flat_mismatch(
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    handoff: &Rv64imMainCircuitHandoff,
    fresh_claims: &[CcsClaim<Commitment, F>],
    fresh_witnesses: &[CcsWitness<F>],
    me_inputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    me_witnesses: &[Mat<F>],
    ccs_outputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    live_parent: &neo_ccs::CeClaim<Commitment, F, K>,
) -> Result<String, SimpleKernelError> {
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let ce_structure = pad_ccs_structure_to_block_width(structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM padded CE structure failed: {err}")))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation dims failed: {err}")))?;
    let ctx = Rv64imMainCircuitTraceBuildContext {
        params,
        log,
        structure,
        ce_structure: &ce_structure,
        dims,
        optimized_cache: &optimized_cache,
    };
    let (_, mut replay_transcript) = replay_main_relation_pi_ccs_terminal_state(
        &ctx,
        transcript_in,
        &handoff.public_chunk,
        handoff.public_chunk_instance_digest,
        fresh_claims,
        fresh_witnesses,
        me_inputs,
        me_witnesses,
    )?;
    let expected_rhos = sample_main_relation_pi_rlc_rhos(&mut replay_transcript, ctx.params, ccs_outputs.len())?;
    let cols = live_parent.X.cols();
    let observed = live_parent.X.as_slice();
    if observed.len() != D * cols {
        return Ok("pi_rlc_x_flat_parent_len_mismatch".into());
    }
    for child in ccs_outputs {
        let child_values = child.X.as_slice();
        if child_values.len() != D * cols {
            return Ok("pi_rlc_x_flat_child_len_mismatch".into());
        }
    }
    for row in 0..D {
        for col in 0..cols {
            let mut expected = F::ZERO;
            for (rho, child) in expected_rhos.iter().zip(ccs_outputs.iter()) {
                let native_mat = rho.as_mat();
                let child_values = child.X.as_slice();
                for k in 0..D {
                    let child_idx_flat = k * cols + col;
                    expected += native_mat[(row, k)] * child_values[child_idx_flat];
                }
            }
            let parent_idx = row * cols + col;
            if expected != observed[parent_idx] {
                return Ok(format!(
                    "pi_rlc_x_flat_mismatch[row={row},col={col},expected={},observed={}]",
                    expected.as_canonical_u64(),
                    observed[parent_idx].as_canonical_u64()
                ));
            }
        }
    }
    Ok("pi_rlc_x_flat_match".into())
}

fn check_pi_ccs_terminal_state_native(
    chunk_index: usize,
    live: &PiCcsReplayTerminalState,
    replayed: &PiCcsReplayTerminalState,
) -> Result<(), SimpleKernelError> {
    if live.me_outputs != replayed.me_outputs
        || live.challenges_public.alpha != replayed.challenges_public.alpha
        || live.challenges_public.beta_a != replayed.challenges_public.beta_a
        || live.challenges_public.beta_r != replayed.challenges_public.beta_r
        || live.challenges_public.beta_m != replayed.challenges_public.beta_m
        || live.challenges_public.gamma != replayed.challenges_public.gamma
        || live.row_chals != replayed.row_chals
        || live.alpha_prime != replayed.alpha_prime
        || live.s_col != replayed.s_col
        || live.alpha_prime_nc != replayed.alpha_prime_nc
        || live.sumcheck_final != replayed.sumcheck_final
        || live.sumcheck_final_nc != replayed.sumcheck_final_nc
        || live.fold_digest != replayed.fold_digest
    {
        let detail =
            describe_pi_ccs_terminal_state_mismatch(live, replayed).unwrap_or_else(|| "unknown mismatch".into());
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main relation chunk {chunk_index} Pi_CCS terminal replay drifted from the verified chunk trace: {detail}"
        )));
    }
    Ok(())
}

fn describe_pi_ccs_terminal_state_mismatch(
    live: &PiCcsReplayTerminalState,
    replayed: &PiCcsReplayTerminalState,
) -> Option<String> {
    if live.me_outputs.len() != replayed.me_outputs.len() {
        return Some(format!(
            "me_outputs len mismatch (live {}, replayed {})",
            live.me_outputs.len(),
            replayed.me_outputs.len()
        ));
    }
    for (idx, (live_claim, replayed_claim)) in live
        .me_outputs
        .iter()
        .zip(replayed.me_outputs.iter())
        .enumerate()
    {
        if let Some(mismatch) = describe_ce_claim_mismatch(live_claim, replayed_claim) {
            return Some(format!("me_outputs[{idx}] {mismatch}"));
        }
    }
    if live.challenges_public.alpha != replayed.challenges_public.alpha {
        let idx = live
            .challenges_public
            .alpha
            .iter()
            .zip(replayed.challenges_public.alpha.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("public.alpha[{idx}] mismatch"));
    }
    if live.challenges_public.beta_a != replayed.challenges_public.beta_a {
        let idx = live
            .challenges_public
            .beta_a
            .iter()
            .zip(replayed.challenges_public.beta_a.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("public.beta_a[{idx}] mismatch"));
    }
    if live.challenges_public.beta_r != replayed.challenges_public.beta_r {
        let idx = live
            .challenges_public
            .beta_r
            .iter()
            .zip(replayed.challenges_public.beta_r.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("public.beta_r[{idx}] mismatch"));
    }
    if live.challenges_public.beta_m != replayed.challenges_public.beta_m {
        let idx = live
            .challenges_public
            .beta_m
            .iter()
            .zip(replayed.challenges_public.beta_m.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("public.beta_m[{idx}] mismatch"));
    }
    if live.challenges_public.gamma != replayed.challenges_public.gamma {
        return Some("public.gamma mismatch".into());
    }
    if live.row_chals != replayed.row_chals {
        let idx = live
            .row_chals
            .iter()
            .zip(replayed.row_chals.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("row_chals[{idx}] mismatch"));
    }
    if live.alpha_prime != replayed.alpha_prime {
        let idx = live
            .alpha_prime
            .iter()
            .zip(replayed.alpha_prime.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("alpha_prime[{idx}] mismatch"));
    }
    if live.s_col != replayed.s_col {
        let idx = live
            .s_col
            .iter()
            .zip(replayed.s_col.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("s_col[{idx}] mismatch"));
    }
    if live.alpha_prime_nc != replayed.alpha_prime_nc {
        let idx = live
            .alpha_prime_nc
            .iter()
            .zip(replayed.alpha_prime_nc.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("alpha_prime_nc[{idx}] mismatch"));
    }
    if live.sumcheck_final != replayed.sumcheck_final {
        return Some("sumcheck_final mismatch".into());
    }
    if live.sumcheck_final_nc != replayed.sumcheck_final_nc {
        return Some("sumcheck_final_nc mismatch".into());
    }
    if live.fold_digest != replayed.fold_digest {
        return Some("fold_digest mismatch".into());
    }
    None
}

fn describe_ce_claim_mismatch(
    expected: &neo_ccs::CeClaim<Commitment, F, K>,
    observed: &neo_ccs::CeClaim<Commitment, F, K>,
) -> Option<String> {
    if expected.c != observed.c {
        return Some("c mismatch".into());
    }
    if expected.X.rows() != observed.X.rows() || expected.X.cols() != observed.X.cols() {
        return Some(format!(
            "X shape mismatch (expected {}x{}, observed {}x{})",
            expected.X.rows(),
            expected.X.cols(),
            observed.X.rows(),
            observed.X.cols()
        ));
    }
    for row in 0..expected.X.rows() {
        for col in 0..expected.X.cols() {
            if expected.X[(row, col)] != observed.X[(row, col)] {
                return Some(format!("X[{row},{col}] mismatch"));
            }
        }
    }
    if expected.r != observed.r {
        let idx = expected
            .r
            .iter()
            .zip(observed.r.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("r[{idx}] mismatch"));
    }
    if expected.s_col != observed.s_col {
        let idx = expected
            .s_col
            .iter()
            .zip(observed.s_col.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("s_col[{idx}] mismatch"));
    }
    if expected.y_ring.len() != observed.y_ring.len() {
        return Some(format!(
            "y_ring row-count mismatch (expected {}, observed {})",
            expected.y_ring.len(),
            observed.y_ring.len()
        ));
    }
    for (row_idx, (expected_row, observed_row)) in expected
        .y_ring
        .iter()
        .zip(observed.y_ring.iter())
        .enumerate()
    {
        if expected_row.len() != observed_row.len() {
            return Some(format!(
                "y_ring[{row_idx}] len mismatch (expected {}, observed {})",
                expected_row.len(),
                observed_row.len()
            ));
        }
        if let Some(col_idx) = expected_row
            .iter()
            .zip(observed_row.iter())
            .position(|(lhs, rhs)| lhs != rhs)
        {
            return Some(format!("y_ring[{row_idx}][{col_idx}] mismatch"));
        }
    }
    if expected.ct != observed.ct {
        let idx = expected
            .ct
            .iter()
            .zip(observed.ct.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("ct[{idx}] mismatch"));
    }
    if expected.aux_openings != observed.aux_openings {
        let idx = expected
            .aux_openings
            .iter()
            .zip(observed.aux_openings.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("aux_openings[{idx}] mismatch"));
    }
    if expected.y_zcol != observed.y_zcol {
        let idx = expected
            .y_zcol
            .iter()
            .zip(observed.y_zcol.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Some(format!("y_zcol[{idx}] mismatch"));
    }
    if expected.c_step_coords != observed.c_step_coords {
        return Some("c_step_coords mismatch".into());
    }
    if expected.u_offset != observed.u_offset {
        return Some(format!(
            "u_offset mismatch (expected {}, observed {})",
            expected.u_offset, observed.u_offset
        ));
    }
    if expected.u_len != observed.u_len {
        return Some(format!(
            "u_len mismatch (expected {}, observed {})",
            expected.u_len, observed.u_len
        ));
    }
    if expected.m_in != observed.m_in {
        return Some(format!(
            "m_in mismatch (expected {}, observed {})",
            expected.m_in, observed.m_in
        ));
    }
    if expected.fold_digest != observed.fold_digest {
        return Some("fold_digest mismatch".into());
    }
    None
}

fn sample_main_relation_pi_rlc_rhos(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    claim_count: usize,
) -> Result<Vec<neo_reductions::api::RotRho>, SimpleKernelError> {
    let ring = RotRing::goldilocks();
    sample_rot_rhos_n_typed(transcript, params, &ring, claim_count)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation Pi_RLC rho sampling failed: {err}")))
}
