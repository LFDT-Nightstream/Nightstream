//! Owns the native RV64IM main-circuit trace builder one chunk at a time.
//!
//! This module bridges the verified chunk-fold step chain into the concrete
//! replay artifacts consumed by the current Spartan circuit. It is not
//! theorem-facing and does not own circuit synthesis.

use neo_ajtai::{AjtaiSModule, Commitment};
use neo_ccs::{build_superneo_ring_forms, CcsClaim, CcsStructure, CcsWitness, Mat, SModuleHomomorphism};
use neo_math::{balanced::to_balanced_i128, KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::common::{
    compute_y_zcol_from_witness, compute_y_zcol_from_witness_digits, decode_superneo_coeffs_from_witness_mat,
};
use neo_reductions::engines::utils::{
    bind_header_and_instance_digest_with_digest, build_dims_and_policy, digest_ccs_matrices_with_sparse_cache, Dims,
    PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG, PI_CCS_SUMCHECK_INITIAL_RAW_TAG, PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG,
};
use neo_reductions::optimized_engine::{
    Challenges, OptimizedStructureCache, PiCcsProvePerf, PiCcsReplayProofWitness, PiCcsReplayTerminalState,
};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Serialize};

use crate::chunk_relation::{build_inert_chunk_replay_proof_witness, ChunkReplayWitness};
use crate::finalize::{digest_fields_as_digest32, public_chunk_digest, FixedShapeChunkSummary};
use crate::proof::{PublicChunk, PublicStep};
use crate::rv64im::chunk_fold_step::{Rv64imAccumulatorHandle, Rv64imChunkFoldCarry};
use crate::rv64im::chunk_relation::rv64im_chunk_relation_digest_from_fold_digest;
use crate::rv64im::chunk_relation::{trace_rv64im_chunk_relation_with_replay, Rv64imChunkRelationTrace};
use crate::rv64im::final_relation::{
    build_rv64im_chunk_fold_step_traces_from_components, rv64im_chunk_fold_carried_transcript_snapshot,
    Rv64imChunkFoldTranscriptSnapshot, Rv64imChunkTransitionWitness, Rv64imFinalProofComponentDigests,
    Rv64imFinalStatement, Rv64imFoldedStatement, Rv64imRecursiveAccumulator, RV64IM_CHUNK_DONE_RAW_TAG,
};
use crate::rv64im::kernel::{
    rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache, Rv64imKernelExportProof,
    Rv64imVerifiedKernelChunkHandoff, SimpleKernelError,
};
use crate::rv64im::main_relation_circuit::structure::pad_ccs_structure_to_block_width;

pub(crate) const CHUNK_META_RAW_TAG: u64 = 14;
pub(crate) const STEP_INDEX_RAW_TAG: u64 = 15;

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
    pub(crate) transcript_in: Rv64imChunkFoldTranscriptSnapshot,
    pub(crate) state_in_claims: Vec<neo_ccs::CeClaim<Commitment, F, K>>,
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
            &self.transcript_in,
            &self.handoff,
            &self.fresh_claims,
            &self.state_in_claims,
            self.ccs_trace.ccs_outputs.clone(),
            self.ccs_trace.ccs_replay_proof.clone(),
            self.ccs_trace.parent.clone(),
            self.ccs_trace.children.clone(),
        )
    }
}

pub(crate) fn build_rv64im_main_circuit_chunk_replay_surface(
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    handoff: &Rv64imMainCircuitHandoff,
    fresh_claims: &[CcsClaim<Commitment, F>],
    state_in_claims: &[neo_ccs::CeClaim<Commitment, F, K>],
    ccs_outputs: Vec<neo_ccs::CeClaim<Commitment, F, K>>,
    replay_proof: PiCcsReplayProofWitness,
    parent: neo_ccs::CeClaim<Commitment, F, K>,
    children: Vec<neo_ccs::CeClaim<Commitment, F, K>>,
) -> Result<Rv64imMainCircuitChunkReplaySurface, SimpleKernelError> {
    Ok(Rv64imMainCircuitChunkReplaySurface {
        handoff: handoff.clone(),
        fresh_claims: fresh_claims.to_vec(),
        pi_ccs: derive_rv64im_main_circuit_pi_ccs_replay_surface(
            transcript_in,
            handoff,
            fresh_claims,
            state_in_claims,
            ccs_outputs,
            replay_proof,
        )?,
        pi_rlc: Rv64imMainCircuitPiRlcReplaySurface { parent },
        pi_dec: Rv64imMainCircuitPiDecReplaySurface { children },
    })
}

pub(crate) fn derive_rv64im_main_circuit_pi_ccs_replay_surface(
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    handoff: &Rv64imMainCircuitHandoff,
    fresh_claims: &[CcsClaim<Commitment, F>],
    me_inputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    ccs_outputs: Vec<neo_ccs::CeClaim<Commitment, F, K>>,
    replay_proof: PiCcsReplayProofWitness,
) -> Result<Rv64imMainCircuitPiCcsReplaySurface, SimpleKernelError> {
    let (params, _, structure) = rv64im_cached_root_main_lane_context()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation dims failed: {err}")))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mat_digest_vec = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()));
    let mat_digest: [Goldilocks; 4] = mat_digest_vec
        .try_into()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main relation matrix digest length mismatch".into()))?;
    let mut replay_transcript =
        Poseidon2Transcript::from_state_and_absorbed(transcript_in.state, transcript_in.absorbed);
    append_chunk_meta_native(&mut replay_transcript, &handoff.public_chunk);
    let challenges = derive_replay_challenges_from_rounds(
        &mut replay_transcript,
        params,
        structure,
        dims,
        &mat_digest,
        fresh_claims,
        me_inputs,
        &replay_proof,
        handoff.public_chunk_instance_digest,
    )?;
    Ok(Rv64imMainCircuitPiCcsReplaySurface {
        ccs_outputs,
        replay_proof,
        public_challenges: challenges.public_challenges,
        row_chals: challenges.row_chals,
        alpha_prime: challenges.alpha_prime,
        s_col: challenges.s_col,
        alpha_prime_nc: challenges.alpha_prime_nc,
    })
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

    pub(crate) fn zero_witness(&self) -> CcsWitness<F> {
        CcsWitness {
            w: vec![F::ZERO; self.w_len as usize],
            Z: Mat::zero(self.z_rows as usize, self.z_cols as usize, F::ZERO),
        }
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Rv64imMainRelationSetupChunkShape {
    public_step_count: u64,
    state_in_claim_shapes: Vec<Rv64imMainCircuitCeClaimShape>,
    cover: Rv64imMainCircuitChunkCover,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRelationSetupShape {
    terminal_final_claim_shapes: Vec<Rv64imMainCircuitCeClaimShape>,
    chunks: Vec<Rv64imMainRelationSetupChunkShape>,
}

pub(crate) fn rv64im_main_relation_setup_shape_from_trace(
    trace: &Rv64imMainCircuitTrace,
) -> Rv64imMainRelationSetupShape {
    Rv64imMainRelationSetupShape {
        terminal_final_claim_shapes: trace
            .statement
            .folded
            .final_accumulator
            .final_main_claims
            .iter()
            .map(Rv64imMainCircuitCeClaimShape::from_claim)
            .collect(),
        chunks: trace
            .chunk_traces
            .iter()
            .map(|chunk| Rv64imMainRelationSetupChunkShape {
                public_step_count: chunk.handoff.public_chunk.steps.len() as u64,
                state_in_claim_shapes: chunk
                    .state_in_claims
                    .iter()
                    .map(Rv64imMainCircuitCeClaimShape::from_claim)
                    .collect(),
                cover: Rv64imMainCircuitChunkCover::from_trace(chunk),
            })
            .collect(),
    }
}

pub(crate) fn build_rv64im_main_relation_setup_shape_from_step_components(
    statement: &Rv64imFinalStatement,
    proof_digest: [u8; 32],
    kernel_export: &Rv64imKernelExportProof,
    chunk_summaries: &[FixedShapeChunkSummary],
    steps: &[Rv64imChunkTransitionWitness],
    component_digests: &Rv64imFinalProofComponentDigests,
) -> Result<Rv64imMainRelationSetupShape, SimpleKernelError> {
    let trace = build_rv64im_main_circuit_trace_from_step_components(
        statement,
        proof_digest,
        kernel_export,
        chunk_summaries,
        steps,
        component_digests,
    )?;
    Ok(rv64im_main_relation_setup_shape_from_trace(&trace))
}

pub(crate) fn build_rv64im_main_circuit_trace_from_setup_shape(
    shape: &Rv64imMainRelationSetupShape,
) -> Result<Rv64imMainCircuitTrace, SimpleKernelError> {
    let mut transcript = crate::rv64im::final_relation::rv64im_chunk_fold_initial_transcript();
    let mut chunk_traces = Vec::with_capacity(shape.chunks.len());
    let mut start_index = 0usize;
    for (chunk_index, chunk_shape) in shape.chunks.iter().enumerate() {
        let transcript_in = Rv64imChunkFoldTranscriptSnapshot {
            state: transcript.state(),
            absorbed: transcript.absorbed(),
        };
        let dummy_chunk = build_dummy_main_relation_chunk_trace(chunk_index, start_index, &transcript_in, chunk_shape)?;
        chunk_traces.push(dummy_chunk);
        transcript.append_fields_raw(&[F::from_u64(RV64IM_CHUNK_DONE_RAW_TAG), F::ONE]);
        start_index = start_index.saturating_add(chunk_shape.public_step_count as usize);
    }
    Ok(Rv64imMainCircuitTrace {
        statement: build_dummy_main_relation_statement(shape, start_index as u64),
        chunk_traces,
    })
}

fn build_dummy_main_relation_statement(
    shape: &Rv64imMainRelationSetupShape,
    semantic_step_count: u64,
) -> Rv64imFinalStatement {
    let final_main_claims = shape
        .terminal_final_claim_shapes
        .iter()
        .map(Rv64imMainCircuitCeClaimShape::zero_claim)
        .collect();
    Rv64imFinalStatement {
        public_statement_digest: [0; 32],
        folded: Rv64imFoldedStatement {
            fold_schedule: crate::proof::FoldSchedule::WholeTrace,
            chunk_count: shape.chunks.len() as u64,
            semantic_step_count,
            kernel_relation_digest: [0; 32],
            final_accumulator: Rv64imRecursiveAccumulator {
                final_main_claims,
                terminal_handle: Rv64imAccumulatorHandle([0; 32]),
            },
            digest: [0; 32],
        },
        digest: [0; 32],
    }
}

fn build_dummy_main_relation_chunk_trace(
    _chunk_index: usize,
    start_index: usize,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    chunk_shape: &Rv64imMainRelationSetupChunkShape,
) -> Result<Rv64imMainCircuitChunkTrace, SimpleKernelError> {
    let public_chunk = build_dummy_public_chunk(chunk_shape, start_index);
    let public_chunk_instance_digest = public_chunk_digest(&public_chunk);
    let public_chunk_digest = digest_fields_as_digest32(public_chunk_instance_digest);
    let replay_proof = build_inert_chunk_replay_proof_witness(
        &chunk_shape.cover.fe_round_lengths,
        &chunk_shape.cover.nc_round_lengths,
    );
    let handoff = Rv64imMainCircuitHandoff {
        public_chunk: public_chunk.clone(),
        public_chunk_instance_digest,
        public_chunk_digest,
        bridge_handoff_digest: [0; 32],
        chunk_relation_digest: rv64im_chunk_relation_digest_from_fold_digest(public_chunk_digest, [0; 32], [0; 32]),
    };
    let state_in_claims = chunk_shape
        .state_in_claim_shapes
        .iter()
        .map(Rv64imMainCircuitCeClaimShape::zero_claim)
        .collect::<Vec<_>>();
    let fresh_claims = chunk_shape
        .cover
        .fresh_claim_shapes
        .iter()
        .map(Rv64imMainCircuitCcsClaimShape::zero_claim)
        .collect::<Vec<_>>();
    let ccs_outputs = chunk_shape
        .cover
        .ccs_output_shapes
        .iter()
        .map(Rv64imMainCircuitCeClaimShape::zero_claim)
        .collect::<Vec<_>>();
    let parent = chunk_shape.cover.parent_claim_shape.zero_claim();
    let children = chunk_shape
        .cover
        .child_claim_shapes
        .iter()
        .map(Rv64imMainCircuitCeClaimShape::zero_claim)
        .collect::<Vec<_>>();
    let replay_surface = build_rv64im_main_circuit_chunk_replay_surface(
        transcript_in,
        &handoff,
        &fresh_claims,
        &state_in_claims,
        ccs_outputs.clone(),
        replay_proof.clone(),
        parent.clone(),
        children.clone(),
    )?;
    let chunk_relation_digest = handoff.chunk_relation_digest;
    Ok(Rv64imMainCircuitChunkTrace {
        handoff,
        transcript_in: transcript_in.clone(),
        state_in_claims,
        fresh_claims,
        fresh_witnesses: chunk_shape
            .cover
            .fresh_witness_shapes
            .iter()
            .map(Rv64imMainCircuitCcsWitnessShape::zero_witness)
            .collect(),
        ccs_trace: Rv64imChunkRelationTrace {
            chunk_relation_digest,
            ccs_outputs,
            ccs_replay_proof: replay_proof.clone(),
            terminal_state: PiCcsReplayTerminalState {
                me_outputs: replay_surface.pi_ccs.ccs_outputs.clone(),
                challenges_public: replay_surface.pi_ccs.public_challenges.clone(),
                row_chals: replay_surface.pi_ccs.row_chals.clone(),
                alpha_prime: replay_surface.pi_ccs.alpha_prime.clone(),
                s_col: replay_surface.pi_ccs.s_col.clone(),
                alpha_prime_nc: replay_surface.pi_ccs.alpha_prime_nc.clone(),
                sumcheck_final: K::ZERO,
                sumcheck_final_nc: K::ZERO,
                fold_digest: [0; 32],
                perf: PiCcsProvePerf::default(),
            },
            parent,
            children,
            z_split: Vec::new(),
        },
    })
}

fn build_dummy_public_chunk(chunk_shape: &Rv64imMainRelationSetupChunkShape, start_index: usize) -> PublicChunk {
    let public_step = PublicStep {
        label: "dummy".to_string(),
        mcs: chunk_shape
            .cover
            .fresh_claim_shapes
            .first()
            .map(Rv64imMainCircuitCcsClaimShape::zero_claim)
            .unwrap_or_else(|| CcsClaim {
                c: Commitment::zeros(0, 0),
                x: Vec::new(),
                m_in: 0,
            }),
    };
    PublicChunk {
        start_index,
        steps: vec![public_step; chunk_shape.public_step_count.max(1) as usize],
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainCircuitTrace {
    pub(crate) statement: Rv64imFinalStatement,
    pub(crate) chunk_traces: Vec<Rv64imMainCircuitChunkTrace>,
}

struct Rv64imMainCircuitTraceBuildContext<'a> {
    params: &'a NeoParams,
    log: &'a AjtaiSModule,
    structure: &'a CcsStructure<F>,
    ce_structure: &'a CcsStructure<F>,
    dims: Dims,
    mat_digest: [Goldilocks; 4],
    optimized_cache: &'a OptimizedStructureCache,
}

pub(crate) fn build_rv64im_main_circuit_trace_from_step_components(
    statement: &Rv64imFinalStatement,
    proof_digest: [u8; 32],
    kernel_export: &Rv64imKernelExportProof,
    chunk_summaries: &[FixedShapeChunkSummary],
    steps: &[Rv64imChunkTransitionWitness],
    component_digests: &Rv64imFinalProofComponentDigests,
) -> Result<Rv64imMainCircuitTrace, SimpleKernelError> {
    if chunk_summaries.len() != statement.folded.chunk_count as usize {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation chunk summary count does not match the folded statement chunk count".into(),
        ));
    }

    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let ce_structure = pad_ccs_structure_to_block_width(structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM padded CE structure failed: {err}")))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation dims failed: {err}")))?;
    let mat_digest_vec = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()));
    let mat_digest: [Goldilocks; 4] = mat_digest_vec
        .try_into()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main relation matrix digest length mismatch".into()))?;
    let ctx = Rv64imMainCircuitTraceBuildContext {
        params,
        log,
        structure,
        ce_structure: &ce_structure,
        dims,
        mat_digest,
        optimized_cache: &optimized_cache,
    };
    let step_traces = build_rv64im_chunk_fold_step_traces_from_components(
        statement,
        proof_digest,
        kernel_export,
        chunk_summaries,
        steps,
        component_digests,
    )?;
    let mut transcript = crate::rv64im::final_relation::rv64im_chunk_fold_initial_transcript();
    let mut chunk_traces = Vec::with_capacity(step_traces.len());
    for (chunk_index, step_trace) in step_traces.iter().enumerate() {
        chunk_traces.push(build_rv64im_main_circuit_chunk_trace_from_parts(
            &ctx,
            chunk_index,
            &step_trace.handoff,
            &step_trace.chunk_summary,
            &step_trace.carry_in,
            &step_trace.carry_out,
            &step_trace.transcript_in,
            &step_trace.replay_witness,
            &mut transcript,
        )?);
        transcript.append_fields_raw(&[F::from_u64(RV64IM_CHUNK_DONE_RAW_TAG), F::ONE]);
    }
    Ok(Rv64imMainCircuitTrace {
        statement: statement.clone(),
        chunk_traces,
    })
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
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let ce_structure = pad_ccs_structure_to_block_width(structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM padded CE structure failed: {err}")))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM main relation dims failed: {err}")))?;
    let mat_digest_vec = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()));
    let mat_digest: [Goldilocks; 4] = mat_digest_vec
        .try_into()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM main relation matrix digest length mismatch".into()))?;
    let ctx = Rv64imMainCircuitTraceBuildContext {
        params,
        log,
        structure,
        ce_structure: &ce_structure,
        dims,
        mat_digest,
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
    let mut replay_transcript = transcript.clone();
    append_chunk_meta_native(&mut replay_transcript, &handoff.public_chunk);
    let replay_challenges = derive_replay_challenges_from_rounds(
        &mut replay_transcript,
        ctx.params,
        ctx.structure,
        ctx.dims,
        &ctx.mat_digest,
        &fresh.fresh_claims,
        &carry_in.main.claims,
        &replay_witness.ccs_replay_proof,
        handoff.public_chunk_instance_digest,
    )?;
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
    if trace.ccs_outputs != trace.terminal_state.me_outputs {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main relation chunk {chunk_index} replay outputs do not match the terminal state outputs"
        )));
    }
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
        &carry_in.main.claims,
        &trace.ccs_outputs,
        &replay_challenges.row_chals,
        &replay_challenges.s_col,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM main relation chunk {chunk_index} output binding failed: {err}"
        ))
    })?;
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

    Ok(Rv64imMainCircuitChunkTrace {
        handoff: Rv64imMainCircuitHandoff {
            public_chunk: fresh.public_chunk.clone(),
            public_chunk_instance_digest: fresh.public_chunk_instance_digest,
            public_chunk_digest: fresh.public_chunk_digest,
            bridge_handoff_digest: fresh.bridge_handoff_digest,
            chunk_relation_digest: trace.chunk_relation_digest,
        },
        transcript_in: transcript_in.clone(),
        state_in_claims: carry_in.main.claims.clone(),
        fresh_claims: fresh.fresh_claims,
        fresh_witnesses: fresh.fresh_witnesses,
        ccs_trace: trace,
    })
}

fn append_chunk_meta_native(transcript: &mut Poseidon2Transcript, public_chunk: &PublicChunk) {
    if public_chunk.steps.len() == 1 {
        transcript.append_fields_raw(&[
            F::from_u64(STEP_INDEX_RAW_TAG),
            F::from_u64(public_chunk.start_index as u64),
        ]);
    } else {
        transcript.append_fields_raw(&[
            F::from_u64(CHUNK_META_RAW_TAG),
            F::from_u64(public_chunk.start_index as u64),
            F::from_u64(public_chunk.steps.len() as u64),
        ]);
    }
}

fn check_output_binding_native(
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Commitment, F>],
    me_inputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    me_outputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    r_prime: &[K],
    s_col_prime: &[K],
) -> Result<(), String> {
    if me_outputs.len() != fresh_claims.len() + me_inputs.len() {
        return Err("output arity mismatch".into());
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
            if output.c.data != fresh.c.data {
                return Err(format!("fresh output {index} commitment mismatch"));
            }
            if output.m_in != fresh.m_in {
                return Err(format!("fresh output {index} m_in mismatch"));
            }
            let expected_x = project_x_from_f_slice(&fresh.x, fresh.m_in)
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

fn project_x_from_f_slice(values: &[F], m_in: usize) -> Result<Mat<F>, String> {
    if values.len() != m_in {
        return Err("x length mismatch".into());
    }
    let mut projected = Mat::zero(D, m_in, F::ZERO);
    for (column, value) in values.iter().copied().enumerate() {
        projected[(column % D, column)] = value;
    }
    Ok(projected)
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

struct DerivedReplayChallenges {
    public_challenges: Challenges,
    row_chals: Vec<K>,
    alpha_prime: Vec<K>,
    s_col: Vec<K>,
    alpha_prime_nc: Vec<K>,
}

#[allow(clippy::too_many_arguments)]
fn derive_replay_challenges_from_rounds(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    fresh_claims: &[CcsClaim<Commitment, F>],
    me_inputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    replay_proof: &PiCcsReplayProofWitness,
    public_instance_digest: [F; 4],
) -> Result<DerivedReplayChallenges, SimpleKernelError> {
    bind_header_and_instance_digest_with_digest(
        transcript,
        params,
        structure,
        dims,
        mat_digest,
        &public_instance_digest,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM replay challenge header binding failed: {err}")))?;
    neo_reductions::engines::utils::bind_me_inputs(transcript, me_inputs)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM replay challenge ME binding failed: {err}")))?;
    let mut public_challenges = neo_reductions::engines::utils::sample_challenges(transcript, dims.ell_d, dims.ell)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM replay challenge public sampling failed: {err}")))?;
    public_challenges.beta_m = neo_reductions::engines::utils::sample_beta_m(transcript, dims.ell_m)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM replay challenge beta_m sampling failed: {err}")))?;

    transcript.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)]);
    let initial_sum = neo_reductions::optimized_engine::claimed_initial_sum_from_inputs_with_k_mcs(
        structure,
        &public_challenges,
        fresh_claims.len(),
        me_inputs,
    );
    transcript.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    transcript.append_fields_raw(&initial_sum.as_coeffs());
    transcript.append_fields_raw(&[F::from_u64(
        neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
    )]);
    let (fe_all, _, fe_ok) = neo_reductions::sumcheck::verify_sumcheck_rounds_poseidon_v3(
        transcript,
        dims.d_sc,
        initial_sum,
        &replay_proof.sumcheck_rounds,
    );
    if !fe_ok {
        return Err(SimpleKernelError::Bridge(
            "RV64IM replay challenge derivation failed: FE rounds invalid".into(),
        ));
    }
    let (row_chals, alpha_prime) = fe_all.split_at(dims.ell_n);

    transcript.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)]);
    let initial_sum_nc = K::ZERO;
    transcript.append_fields_raw(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    transcript.append_fields_raw(&initial_sum_nc.as_coeffs());
    transcript.append_fields_raw(&[F::from_u64(
        neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
    )]);
    let (nc_all, _, nc_ok) = neo_reductions::sumcheck::verify_sumcheck_rounds_poseidon_v3(
        transcript,
        dims.d_sc,
        initial_sum_nc,
        &replay_proof.sumcheck_rounds_nc,
    );
    if !nc_ok {
        return Err(SimpleKernelError::Bridge(
            "RV64IM replay challenge derivation failed: NC rounds invalid".into(),
        ));
    }
    let (s_col, alpha_prime_nc) = nc_all.split_at(dims.ell_m);

    Ok(DerivedReplayChallenges {
        public_challenges,
        row_chals: row_chals.to_vec(),
        alpha_prime: alpha_prime.to_vec(),
        s_col: s_col.to_vec(),
        alpha_prime_nc: alpha_prime_nc.to_vec(),
    })
}
