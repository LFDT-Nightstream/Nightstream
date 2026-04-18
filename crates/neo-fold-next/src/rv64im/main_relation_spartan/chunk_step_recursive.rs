//! Owns canonical padded payloads for the future fixed RV64IM recursive step backend.

use std::io::{self, Write};
use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_reductions::engines::utils::me_digest_poseidon_into;
use neo_reductions::optimized_engine::PiCcsReplayProofWitness;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use super::chunk_step_ivc::{
    build_rv64im_chunk_step_ivc_recursive_step_cover_shape,
    build_rv64im_chunk_step_ivc_recursive_step_padding_from_shape, Rv64imChunkStepIvcRecursiveStepPadding,
    Rv64imChunkStepIvcShape, Rv64imChunkStepIvcSpartanError,
};
use super::fixed_transcript::derive_rv64im_fixed_transcript_out_from_chunk_body;
use super::{
    Rv64imChunkBoundaryPlan, Rv64imChunkChildClaimSource, Rv64imChunkNextCarryMode, Rv64imChunkRlcMode,
    Rv64imMainRecursionStepSpartanStatement,
};
use crate::finalize::{digest32_as_fields, digest_fields_as_digest32};
use crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcRelation;
use crate::rv64im::construction2::build_rv64im_main_recursion_construction2_verified_step_statement_from_relation;
use crate::rv64im::final_relation::{rv64im_chunk_fold_transcript_snapshot_digest, Rv64imChunkFoldTranscriptSnapshot};
use crate::rv64im::main_recursion::{
    build_rv64im_main_recursion_backend_statement_from_advice, build_rv64im_main_recursion_f_prime_advices,
    Rv64imMainRecursionFPrimeAdvice,
};
use crate::rv64im::main_relation_trace::{
    build_rv64im_main_circuit_chunk_replay_surface, build_rv64im_main_circuit_chunk_trace_from_authoritative_parts,
    Rv64imMainCircuitCcsClaimShape, Rv64imMainCircuitCcsWitnessShape, Rv64imMainCircuitCeClaimShape,
    Rv64imMainCircuitChunkCover, Rv64imMainCircuitChunkReplaySurface, Rv64imMainCircuitChunkTrace,
    Rv64imMainCircuitHandoff,
};
use crate::rv64im::SimpleKernelError;

#[derive(Clone, Debug)]
pub struct Rv64imMainRecursionFPrimePayload {
    pub(crate) boundary_plan: Rv64imChunkBoundaryPlan,
    pub step_shape: Rv64imChunkStepIvcShape,
    pub cover_shape: Rv64imChunkStepIvcShape,
    pub padding: Rv64imChunkStepIvcRecursiveStepPadding,
    pub(crate) z_0: [u8; 32],
    pub(crate) z_i: [u8; 32],
    pub(crate) z_next: [u8; 32],
    pub(crate) pc_i: u64,
    pub(crate) pc_next: u64,
    pub(crate) phi_side_commitment_words: Vec<Vec<u64>>,
    pub(crate) handoff: Rv64imMainCircuitHandoff,
    pub(crate) fixed_transcript_out: Rv64imChunkFoldTranscriptSnapshot,
    pub state_in_claims: Vec<CeClaim<Commitment, F, K>>,
    pub state_out_claims: Vec<CeClaim<Commitment, F, K>>,
    pub fresh_claims: Vec<CcsClaim<Commitment, F>>,
    pub fresh_witnesses: Vec<CcsWitness<F>>,
    pub pi_ccs: Rv64imMainRecursionFPrimePiCcsPayload,
    pub pi_rlc: Rv64imMainRecursionFPrimePiRlcPayload,
    pub pi_dec: Rv64imMainRecursionFPrimePiDecPayload,
    pub(crate) chunk_cover: Rv64imMainCircuitChunkCover,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imMainRecursionFPrimePiCcsPayload {
    pub ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    pub replay: PiCcsReplayProofWitness,
}

impl PartialEq for Rv64imMainRecursionFPrimePiCcsPayload {
    fn eq(&self, other: &Self) -> bool {
        self.ccs_outputs == other.ccs_outputs && self.replay == other.replay
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionFPrimePiRlcPayload {
    pub parent: CeClaim<Commitment, F, K>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionFPrimePiDecPayload {
    pub children: Vec<CeClaim<Commitment, F, K>>,
}

#[derive(Clone, Debug)]
pub struct Rv64imMainRecursionFPrimeBackendRelation {
    pub f_prime_advice: Rv64imMainRecursionFPrimeAdvice,
    pub spartan_statement: Rv64imMainRecursionStepSpartanStatement,
    pub payload: Rv64imMainRecursionFPrimePayload,
}

#[derive(Clone, Debug, Default)]
pub struct Rv64imMainRecursionFPrimeBackendRelationBuildPerf {
    pub spartan_shape_ms: f64,
    pub payloads_ms: f64,
    pub statement_build_ms: f64,
    pub semantics_check_ms: f64,
    pub total_ms: f64,
    pub relation_count: usize,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn emit_debug_timing(trace_prefix: Option<&str>, label: &str, elapsed_ms: f64) {
    if let Some(prefix) = trace_prefix {
        eprintln!("{prefix}.{label}={elapsed_ms:.2}ms");
        let _ = io::stderr().flush();
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imCeClaimDigestShape {
    pub commitment_d: u64,
    pub commitment_kappa: u64,
    pub c_data_len: u64,
    pub x_rows: u64,
    pub x_cols: u64,
    pub r_len: u64,
    pub s_col_len: u64,
    pub y_ring_row_count: u64,
    pub y_ring_row_lens: Vec<u64>,
    pub ct_len: u64,
    pub aux_openings_len: u64,
    pub y_zcol_len: u64,
    pub c_step_coords_len: u64,
}

impl Rv64imCeClaimDigestShape {
    pub fn from_claim(claim: &CeClaim<Commitment, F, K>) -> Self {
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

    pub fn merge(&self, other: &Self) -> Self {
        let y_ring_len = self.y_ring_row_lens.len().max(other.y_ring_row_lens.len());
        let y_ring_row_lens = (0..y_ring_len)
            .map(|idx| {
                self.y_ring_row_lens
                    .get(idx)
                    .copied()
                    .unwrap_or(0)
                    .max(other.y_ring_row_lens.get(idx).copied().unwrap_or(0))
            })
            .collect();
        Self {
            commitment_d: self.commitment_d.max(other.commitment_d),
            commitment_kappa: self.commitment_kappa.max(other.commitment_kappa),
            c_data_len: self.c_data_len.max(other.c_data_len),
            x_rows: self.x_rows.max(other.x_rows),
            x_cols: self.x_cols.max(other.x_cols),
            r_len: self.r_len.max(other.r_len),
            s_col_len: self.s_col_len.max(other.s_col_len),
            y_ring_row_count: self.y_ring_row_count.max(other.y_ring_row_count),
            y_ring_row_lens,
            ct_len: self.ct_len.max(other.ct_len),
            aux_openings_len: self.aux_openings_len.max(other.aux_openings_len),
            y_zcol_len: self.y_zcol_len.max(other.y_zcol_len),
            c_step_coords_len: self.c_step_coords_len.max(other.c_step_coords_len),
        }
    }

    pub fn covers_claim(&self, claim: &CeClaim<Commitment, F, K>) -> bool {
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

    pub fn zero_claim(&self) -> CeClaim<Commitment, F, K> {
        CeClaim {
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
pub struct Rv64imCcsClaimShape {
    pub commitment_d: u64,
    pub commitment_kappa: u64,
    pub c_data_len: u64,
    pub x_len: u64,
}

impl Rv64imCcsClaimShape {
    pub fn from_claim(claim: &CcsClaim<Commitment, F>) -> Self {
        Self {
            commitment_d: claim.c.d as u64,
            commitment_kappa: claim.c.kappa as u64,
            c_data_len: claim.c.data.len() as u64,
            x_len: claim.x.len() as u64,
        }
    }

    pub fn merge(&self, other: &Self) -> Self {
        Self {
            commitment_d: self.commitment_d.max(other.commitment_d),
            commitment_kappa: self.commitment_kappa.max(other.commitment_kappa),
            c_data_len: self.c_data_len.max(other.c_data_len),
            x_len: self.x_len.max(other.x_len),
        }
    }

    pub fn covers_claim(&self, claim: &CcsClaim<Commitment, F>) -> bool {
        self.commitment_d == claim.c.d as u64
            && self.commitment_kappa == claim.c.kappa as u64
            && self.c_data_len == claim.c.data.len() as u64
            && self.x_len >= claim.x.len() as u64
    }

    pub fn zero_claim(&self) -> CcsClaim<Commitment, F> {
        CcsClaim {
            c: Commitment::zeros(self.commitment_d as usize, self.commitment_kappa as usize),
            x: vec![F::ZERO; self.x_len as usize],
            m_in: self.x_len as usize,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imCcsWitnessShape {
    pub w_len: u64,
    pub z_rows: u64,
    pub z_cols: u64,
}

impl Rv64imCcsWitnessShape {
    pub fn from_witness(witness: &CcsWitness<F>) -> Self {
        Self {
            w_len: witness.w.len() as u64,
            z_rows: witness.Z.rows() as u64,
            z_cols: witness.Z.cols() as u64,
        }
    }

    pub fn merge(&self, other: &Self) -> Self {
        Self {
            w_len: self.w_len.max(other.w_len),
            z_rows: self.z_rows.max(other.z_rows),
            z_cols: self.z_cols.max(other.z_cols),
        }
    }

    pub fn covers_witness(&self, witness: &CcsWitness<F>) -> bool {
        self.w_len >= witness.w.len() as u64
            && self.z_rows >= witness.Z.rows() as u64
            && self.z_cols >= witness.Z.cols() as u64
    }

    pub fn zero_witness(&self) -> CcsWitness<F> {
        CcsWitness {
            w: vec![F::ZERO; self.w_len as usize],
            Z: Mat::zero(self.z_rows as usize, self.z_cols as usize, F::ZERO),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionFPrimeClaimCover {
    pub state_in_claim_shapes: Vec<Rv64imCeClaimDigestShape>,
    pub state_out_claim_shapes: Vec<Rv64imCeClaimDigestShape>,
    pub fresh_claim_shapes: Vec<Rv64imCcsClaimShape>,
    pub fresh_witness_shapes: Vec<Rv64imCcsWitnessShape>,
    pub parent_claim_shape: Rv64imCeClaimDigestShape,
    pub ccs_output_shapes: Vec<Rv64imCeClaimDigestShape>,
    pub child_claim_shapes: Vec<Rv64imCeClaimDigestShape>,
}

impl Rv64imMainRecursionFPrimeClaimCover {
    pub fn expected_digest(&self) -> [u8; 32] {
        let encoded = bincode::serialize(self).expect("rv64im recursive-step claim cover encodes");
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_f_prime_claim_cover");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_f_prime_claim_cover/version",
            b"v1",
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_f_prime_claim_cover/encoded",
            &encoded,
        );
        tr.digest32()
    }

    pub fn covers_relation(&self, relation: &Rv64imChunkStepIvcRelation) -> bool {
        let native_verified_step_statement =
            match build_rv64im_main_recursion_construction2_verified_step_statement_from_relation(relation) {
                Ok(statement) => statement,
                Err(_) => return false,
            };
        let native_chunk_summary = match native_verified_step_statement.fixed_shape_chunk_summary() {
            Ok(summary) => summary,
            Err(_) => return false,
        };
        let chunk_trace = match build_rv64im_main_circuit_chunk_trace_from_authoritative_parts(
            native_verified_step_statement.chunk_index as usize,
            &relation.witness.handoff,
            &native_chunk_summary,
            &relation.witness.state_in.carry,
            &relation.witness.state_out.carry,
            &relation.witness.state_in.transcript,
            &relation.witness.state_out.transcript,
            &relation.witness.replay_witness,
        ) {
            Ok(trace) => trace,
            Err(_) => return false,
        };
        self.state_in_claim_shapes.len() >= relation.witness.state_in.carry.main.claims.len()
            && self.state_out_claim_shapes.len() >= relation.witness.state_out.carry.main.claims.len()
            && self.fresh_claim_shapes.len() >= chunk_trace.fresh_claims.len()
            && self.fresh_witness_shapes.len() >= chunk_trace.fresh_witnesses.len()
            && self.ccs_output_shapes.len() >= chunk_trace.ccs_trace.ccs_outputs.len()
            && self.child_claim_shapes.len() >= chunk_trace.ccs_trace.children.len()
            && relation
                .witness
                .state_in
                .carry
                .main
                .claims
                .iter()
                .enumerate()
                .all(|(idx, claim)| self.state_in_claim_shapes[idx].covers_claim(claim))
            && relation
                .witness
                .state_out
                .carry
                .main
                .claims
                .iter()
                .enumerate()
                .all(|(idx, claim)| self.state_out_claim_shapes[idx].covers_claim(claim))
            && chunk_trace
                .fresh_claims
                .iter()
                .enumerate()
                .all(|(idx, claim)| self.fresh_claim_shapes[idx].covers_claim(claim))
            && chunk_trace
                .fresh_witnesses
                .iter()
                .enumerate()
                .all(|(idx, witness)| self.fresh_witness_shapes[idx].covers_witness(witness))
            && self
                .parent_claim_shape
                .covers_claim(&chunk_trace.ccs_trace.parent)
            && chunk_trace
                .ccs_trace
                .ccs_outputs
                .iter()
                .enumerate()
                .all(|(idx, claim)| self.ccs_output_shapes[idx].covers_claim(claim))
            && chunk_trace
                .ccs_trace
                .children
                .iter()
                .enumerate()
                .all(|(idx, claim)| self.child_claim_shapes[idx].covers_claim(claim))
    }

    pub fn matches_payload(&self, payload: &Rv64imMainRecursionFPrimePayload) -> bool {
        self.state_in_claim_shapes.len() == payload.state_in_claims.len()
            && self.state_out_claim_shapes.len() == payload.state_out_claims.len()
            && self.fresh_claim_shapes.len() == payload.fresh_claims.len()
            && self.fresh_witness_shapes.len() == payload.fresh_witnesses.len()
            && self.ccs_output_shapes.len() == payload.pi_ccs.ccs_outputs.len()
            && self.child_claim_shapes.len() == payload.pi_dec.children.len()
            && payload
                .state_in_claims
                .iter()
                .enumerate()
                .all(|(idx, claim)| Rv64imCeClaimDigestShape::from_claim(claim) == self.state_in_claim_shapes[idx])
            && payload
                .state_out_claims
                .iter()
                .enumerate()
                .all(|(idx, claim)| Rv64imCeClaimDigestShape::from_claim(claim) == self.state_out_claim_shapes[idx])
            && payload
                .fresh_claims
                .iter()
                .enumerate()
                .all(|(idx, claim)| Rv64imCcsClaimShape::from_claim(claim) == self.fresh_claim_shapes[idx])
            && payload
                .fresh_witnesses
                .iter()
                .enumerate()
                .all(|(idx, witness)| Rv64imCcsWitnessShape::from_witness(witness) == self.fresh_witness_shapes[idx])
            && Rv64imCeClaimDigestShape::from_claim(&payload.pi_rlc.parent) == self.parent_claim_shape
            && payload
                .pi_ccs
                .ccs_outputs
                .iter()
                .enumerate()
                .all(|(idx, claim)| Rv64imCeClaimDigestShape::from_claim(claim) == self.ccs_output_shapes[idx])
            && payload
                .pi_dec
                .children
                .iter()
                .enumerate()
                .all(|(idx, claim)| Rv64imCeClaimDigestShape::from_claim(claim) == self.child_claim_shapes[idx])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionStepSpartanShape {
    pub cover_shape: Rv64imChunkStepIvcShape,
    pub claim_cover: Rv64imMainRecursionFPrimeClaimCover,
}

impl Rv64imMainRecursionStepSpartanShape {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_step_spartan_shape");
        tr.append_message(b"neo.fold.next/rv64im/main_recursion_step_spartan_shape/version", b"v1");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_spartan_shape/cover_shape",
            &self.cover_shape.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_spartan_shape/claim_cover",
            &self.claim_cover.expected_digest(),
        );
        tr.digest32()
    }

    pub fn matches_payload(&self, payload: &Rv64imMainRecursionFPrimePayload) -> bool {
        payload.cover_shape == self.cover_shape
            && payload.matches_cover_shape()
            && self.claim_cover.matches_payload(payload)
    }
}

fn build_rv64im_main_recursion_f_prime_step_shape(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    chunk_trace: &Rv64imMainCircuitChunkTrace,
    fixed_transcript_out_absorbed: u64,
) -> Rv64imChunkStepIvcShape {
    Rv64imChunkStepIvcShape {
        terminal_step: advice.bridge_handoff_halted_out(),
        state_in_claim_count: advice.running_state().carry.main.claims.len() as u64,
        state_out_claim_count: advice.fresh_state_out().carry.main.claims.len() as u64,
        fresh_claim_count: chunk_trace.fresh_claims.len() as u64,
        fresh_witness_count: chunk_trace.fresh_witnesses.len() as u64,
        ccs_output_count: chunk_trace.ccs_trace.ccs_outputs.len() as u64,
        child_count: chunk_trace.ccs_trace.children.len() as u64,
        transcript_in_absorbed: advice.running_state().transcript.absorbed as u64,
        transcript_out_absorbed: fixed_transcript_out_absorbed,
        fe_round_lengths: chunk_trace
            .ccs_trace
            .ccs_replay_proof
            .sumcheck_rounds
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
        nc_round_lengths: chunk_trace
            .ccs_trace
            .ccs_replay_proof
            .sumcheck_rounds_nc
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
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

pub fn build_rv64im_main_recursion_f_prime_claim_cover(
    advices: &[Rv64imMainRecursionFPrimeAdvice],
) -> Result<Rv64imMainRecursionFPrimeClaimCover, Rv64imChunkStepIvcSpartanError> {
    if advices.is_empty() {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step claim cover requires at least one main recursion F' advice".into(),
        ));
    }
    let mut state_in_claim_shapes = Vec::new();
    let mut state_out_claim_shapes = Vec::new();
    let mut fresh_claim_shapes = Vec::new();
    let mut fresh_witness_shapes = Vec::new();
    let mut parent_claim_shape: Option<Rv64imCeClaimDigestShape> = None;
    let mut ccs_output_shapes = Vec::new();
    let mut child_claim_shapes = Vec::new();
    for advice in advices {
        merge_claim_shape_cover(&mut state_in_claim_shapes, &advice.running_state().carry.main.claims);
        merge_claim_shape_cover(&mut state_out_claim_shapes, &advice.fresh_state_out().carry.main.claims);
        let chunk_trace = advice.main_circuit_chunk_trace();
        merge_ccs_claim_shape_cover(&mut fresh_claim_shapes, &chunk_trace.fresh_claims);
        merge_ccs_witness_shape_cover(&mut fresh_witness_shapes, &chunk_trace.fresh_witnesses);
        let trace_parent_shape = Rv64imCeClaimDigestShape::from_claim(&chunk_trace.ccs_trace.parent);
        parent_claim_shape = Some(match parent_claim_shape {
            Some(existing) => existing.merge(&trace_parent_shape),
            None => trace_parent_shape,
        });
        merge_claim_shape_cover(&mut ccs_output_shapes, &chunk_trace.ccs_trace.ccs_outputs);
        merge_claim_shape_cover(&mut child_claim_shapes, &chunk_trace.ccs_trace.children);
    }
    Ok(Rv64imMainRecursionFPrimeClaimCover {
        state_in_claim_shapes,
        state_out_claim_shapes,
        fresh_claim_shapes,
        fresh_witness_shapes,
        parent_claim_shape: parent_claim_shape.ok_or_else(|| {
            Rv64imChunkStepIvcSpartanError::Prepare(
                "rv64im recursive-step claim cover requires at least one parent claim shape".into(),
            )
        })?,
        ccs_output_shapes,
        child_claim_shapes,
    })
}

#[inline]
fn extend_packed_bytes_as_fields(dst: &mut Vec<F>, bytes: &[u8]) {
    const BYTES_PER_LIMB: usize = 7;
    dst.push(F::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        dst.push(F::from_u64(u64::from_le_bytes(limb)));
    }
}

pub(crate) fn rv64im_chunk_step_recursive_carry_state_digest(
    claims: &[CeClaim<Commitment, F, K>],
    transcript: &crate::rv64im::final_relation::Rv64imChunkFoldTranscriptSnapshot,
    terminal_handle_digest: [u8; 32],
) -> [u8; 32] {
    let mut scratch = Vec::<F>::with_capacity(2048);
    let mut canonical_claims = claims.to_vec();
    if let Some((first, rest)) = canonical_claims.split_first_mut() {
        let shared_r = first.r.clone();
        let shared_s_col = first.s_col.clone();
        for claim in rest {
            claim.r = shared_r.clone();
            claim.s_col = shared_s_col.clone();
        }
    }
    let claim_digests = canonical_claims
        .iter()
        .map(|claim| me_digest_poseidon_into(&mut scratch, claim))
        .collect::<Vec<_>>();

    let mut preimage = Vec::with_capacity(32 + claim_digests.len() * 4);
    extend_packed_bytes_as_fields(
        &mut preimage,
        b"neo.fold.next/rv64im/main_recursion_fixed_step_accumulator_instance/v1",
    );
    preimage.push(F::from_u64(claim_digests.len() as u64));
    preimage.extend(
        claim_digests
            .iter()
            .flat_map(|digest| digest.iter().copied()),
    );
    preimage.extend(digest32_as_fields(rv64im_chunk_fold_transcript_snapshot_digest(
        transcript,
    )));
    preimage.extend(digest32_as_fields(terminal_handle_digest));
    digest_fields_as_digest32(poseidon2_hash(&preimage))
}

pub(crate) fn build_rv64im_main_recursion_step_spartan_statement(
    f_prime_advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<Rv64imMainRecursionStepSpartanStatement, SimpleKernelError> {
    Ok(build_rv64im_main_recursion_backend_statement_from_advice(f_prime_advice)?.native_statement())
}

pub fn debug_check_rv64im_main_recursion_f_prime_backend_relation_semantics(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), SimpleKernelError> {
    if !backend_relation
        .payload
        .matches_explicit_semantics(&backend_relation.f_prime_advice)
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM recursive-step backend relation payload explicit z/pc semantics drifted from the native F' advice"
                .into(),
        ));
    }
    let canonical_statement = build_rv64im_main_recursion_step_spartan_statement(&backend_relation.f_prime_advice)?;
    if backend_relation.spartan_statement != canonical_statement {
        return Err(SimpleKernelError::Bridge(
            "RV64IM recursive-step backend relation requires the canonical per-step Spartan statement derived from native F'"
                .into(),
        ));
    }
    Ok(())
}

impl Rv64imMainRecursionFPrimePayload {
    pub fn phi_side_commitment_words(&self) -> &[Vec<u64>] {
        &self.phi_side_commitment_words
    }

    pub fn z_0(&self) -> &[u8; 32] {
        &self.z_0
    }

    pub fn z_i(&self) -> &[u8; 32] {
        &self.z_i
    }

    pub fn z_next(&self) -> &[u8; 32] {
        &self.z_next
    }

    pub fn pc_i(&self) -> u64 {
        self.pc_i
    }

    pub fn pc_next(&self) -> u64 {
        self.pc_next
    }

    pub fn fixed_transcript_out(&self) -> &Rv64imChunkFoldTranscriptSnapshot {
        &self.fixed_transcript_out
    }

    pub fn padded_fresh_claim_count(&self) -> usize {
        self.chunk_cover.fresh_claim_count as usize
    }

    pub fn effective_fresh_claim_count(&self) -> usize {
        self.step_shape.fresh_claim_count as usize
    }

    pub(crate) fn effective_chunk_replay_surface(
        &self,
        transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
        live_state_in_claims: &[CeClaim<Commitment, F, K>],
    ) -> Result<Rv64imMainCircuitChunkReplaySurface, SimpleKernelError> {
        let mut replay = self.pi_ccs.replay.clone();
        replay
            .sumcheck_rounds
            .truncate(self.step_shape.fe_round_lengths.len());
        for (round, live_len) in replay
            .sumcheck_rounds
            .iter_mut()
            .zip(self.step_shape.fe_round_lengths.iter())
        {
            if round.len() < *live_len as usize {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM recursive-step payload cannot truncate a padded FE round to the live coefficient count"
                        .into(),
                ));
            }
            round.truncate(*live_len as usize);
        }
        replay
            .sumcheck_rounds_nc
            .truncate(self.step_shape.nc_round_lengths.len());
        for (round, live_len) in replay
            .sumcheck_rounds_nc
            .iter_mut()
            .zip(self.step_shape.nc_round_lengths.iter())
        {
            if round.len() < *live_len as usize {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM recursive-step payload cannot truncate a padded NC round to the live coefficient count"
                        .into(),
                ));
            }
            round.truncate(*live_len as usize);
        }
        build_rv64im_main_circuit_chunk_replay_surface(
            transcript_in,
            &self.handoff,
            &self.fresh_claims[..self.step_shape.fresh_claim_count as usize],
            live_state_in_claims,
            self.pi_ccs.ccs_outputs[..self.step_shape.ccs_output_count as usize].to_vec(),
            replay,
            self.pi_rlc.parent.clone(),
            self.pi_dec.children[..self.step_shape.child_count as usize].to_vec(),
        )
    }

    pub fn matches_cover_shape(&self) -> bool {
        self.state_in_claims.len() == self.cover_shape.state_in_claim_count as usize
            && self.state_out_claims.len() == self.cover_shape.state_out_claim_count as usize
            && self.chunk_cover.fresh_claim_count == self.cover_shape.fresh_claim_count
            && self.chunk_cover.fresh_witness_count == self.cover_shape.fresh_witness_count
            && self.chunk_cover.fresh_claim_shapes.len() == self.cover_shape.fresh_claim_count as usize
            && self.chunk_cover.fresh_witness_shapes.len() == self.cover_shape.fresh_witness_count as usize
            && self
                .fresh_claims
                .iter()
                .enumerate()
                .all(|(idx, claim)| self.chunk_cover.fresh_claim_shapes[idx].covers_claim(claim))
            && self.fresh_witnesses.len() == self.cover_shape.fresh_witness_count as usize
            && self
                .fresh_witnesses
                .iter()
                .enumerate()
                .all(|(idx, witness)| self.chunk_cover.fresh_witness_shapes[idx].covers_witness(witness))
            && self
                .chunk_cover
                .parent_claim_shape
                .covers_claim(&self.pi_rlc.parent)
            && self.chunk_cover.ccs_output_count == self.cover_shape.ccs_output_count
            && self.chunk_cover.child_count == self.cover_shape.child_count
            && self.chunk_cover.ccs_output_shapes.len() == self.cover_shape.ccs_output_count as usize
            && self.chunk_cover.child_claim_shapes.len() == self.cover_shape.child_count as usize
            && self.pi_ccs.ccs_outputs.len() == self.cover_shape.ccs_output_count as usize
            && self
                .pi_ccs
                .ccs_outputs
                .iter()
                .enumerate()
                .all(|(idx, claim)| self.chunk_cover.ccs_output_shapes[idx].covers_claim(claim))
            && self.pi_dec.children.len() == self.cover_shape.child_count as usize
            && self
                .pi_dec
                .children
                .iter()
                .enumerate()
                .all(|(idx, claim)| self.chunk_cover.child_claim_shapes[idx].covers_claim(claim))
            && self.chunk_cover.fe_round_lengths == self.cover_shape.fe_round_lengths
            && self.chunk_cover.nc_round_lengths == self.cover_shape.nc_round_lengths
            && self.pi_ccs.replay.sumcheck_rounds.len() == self.cover_shape.fe_round_lengths.len()
            && self.pi_ccs.replay.sumcheck_rounds_nc.len() == self.cover_shape.nc_round_lengths.len()
            && self
                .pi_ccs
                .replay
                .sumcheck_rounds
                .iter()
                .zip(self.cover_shape.fe_round_lengths.iter())
                .all(|(round, live_len)| round.len() == *live_len as usize)
            && self
                .pi_ccs
                .replay
                .sumcheck_rounds_nc
                .iter()
                .zip(self.cover_shape.nc_round_lengths.iter())
                .all(|(round, live_len)| round.len() == *live_len as usize)
    }

    pub fn matches_explicit_semantics(&self, advice: &Rv64imMainRecursionFPrimeAdvice) -> bool {
        self.z_0 == *advice.z_0()
            && self.z_i == *advice.z_i()
            && self.z_next == advice.fresh_state_out().carry.terminal_handle.0
            && self.pc_i == advice.pc_i()
            && self.pc_next == crate::rv64im::main_recursion::RV64IM_MAIN_RECURSION_TRIVIAL_PC
            && self.phi_side_commitment_words == advice.phi_side().commitment_words()
    }
}

fn pad_matrix_to_shape(matrix: &Mat<F>, rows: usize, cols: usize) -> Result<Mat<F>, Rv64imChunkStepIvcSpartanError> {
    if matrix.rows() > rows || matrix.cols() > cols {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload cannot shrink a CE claim matrix into the canonical claim cover".into(),
        ));
    }
    let mut out = Mat::zero(rows, cols, F::ZERO);
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            out[(row, col)] = matrix[(row, col)];
        }
    }
    Ok(out)
}

fn pad_row_to_len(row: &[K], target_len: usize) -> Result<Vec<K>, Rv64imChunkStepIvcSpartanError> {
    if row.len() > target_len {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload cannot shrink a CE claim row into the canonical claim cover".into(),
        ));
    }
    let mut out = row.to_vec();
    out.resize(target_len, K::ZERO);
    Ok(out)
}

fn pad_ce_claim_to_digest_shape(
    claim: &CeClaim<Commitment, F, K>,
    shape: &Rv64imCeClaimDigestShape,
) -> Result<CeClaim<Commitment, F, K>, Rv64imChunkStepIvcSpartanError> {
    if claim.c.data.len() as u64 != shape.c_data_len {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload cannot change Ajtai commitment width in the canonical claim cover".into(),
        ));
    }
    let mut r = claim.r.clone();
    if r.len() > shape.r_len as usize {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload cannot shrink a CE claim r vector into the canonical claim cover".into(),
        ));
    }
    r.resize(shape.r_len as usize, K::ZERO);
    let mut s_col = claim.s_col.clone();
    if s_col.len() > shape.s_col_len as usize {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload cannot shrink a CE claim s_col vector into the canonical claim cover".into(),
        ));
    }
    s_col.resize(shape.s_col_len as usize, K::ZERO);

    let y_ring_row_count = shape.y_ring_row_count as usize;
    if y_ring_row_count < shape.ct_len as usize {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload cannot preserve CT/Y-ring aliasing when the canonical claim cover has fewer Y rows than CT entries".into(),
        ));
    }
    let mut y_ring = Vec::with_capacity(y_ring_row_count);
    for row_idx in 0..y_ring_row_count {
        let mut target_len = shape.y_ring_row_lens.get(row_idx).copied().unwrap_or(0) as usize;
        if row_idx < shape.ct_len as usize {
            target_len = target_len.max(1);
        }
        let row = claim.y_ring.get(row_idx).map(Vec::as_slice).unwrap_or(&[]);
        y_ring.push(pad_row_to_len(row, target_len)?);
    }

    Ok(CeClaim {
        c: claim.c.clone(),
        X: pad_matrix_to_shape(&claim.X, shape.x_rows as usize, shape.x_cols as usize)?,
        r,
        s_col,
        y_ring,
        ct: pad_row_to_len(&claim.ct, shape.ct_len as usize)?,
        aux_openings: pad_row_to_len(&claim.aux_openings, shape.aux_openings_len as usize)?,
        y_zcol: pad_row_to_len(&claim.y_zcol, shape.y_zcol_len as usize)?,
        m_in: claim.m_in,
        fold_digest: claim.fold_digest,
        c_step_coords: {
            let mut out = claim.c_step_coords.clone();
            if out.len() > shape.c_step_coords_len as usize {
                return Err(Rv64imChunkStepIvcSpartanError::Prepare(
                    "rv64im recursive-step payload cannot shrink c_step_coords into the canonical claim cover".into(),
                ));
            }
            out.resize(shape.c_step_coords_len as usize, F::ZERO);
            out
        },
        u_offset: claim.u_offset,
        u_len: claim.u_len,
    })
}

fn build_padded_state_claims(
    claims: &[CeClaim<Commitment, F, K>],
    cover_shapes: &[Rv64imCeClaimDigestShape],
) -> Result<Vec<CeClaim<Commitment, F, K>>, Rv64imChunkStepIvcSpartanError> {
    if cover_shapes.is_empty() {
        return Ok(Vec::new());
    }
    let first_source = claims
        .first()
        .cloned()
        .unwrap_or_else(|| cover_shapes[0].zero_claim());
    let first = pad_ce_claim_to_digest_shape(&first_source, &cover_shapes[0])?;
    let shared_r = first.r.clone();
    let shared_s_col = first.s_col.clone();
    let mut out = Vec::with_capacity(cover_shapes.len());
    out.push(first);
    for (idx, shape) in cover_shapes.iter().enumerate().skip(1) {
        let source = claims
            .get(idx)
            .cloned()
            .unwrap_or_else(|| shape.zero_claim());
        let mut claim = pad_ce_claim_to_digest_shape(&source, shape)?;
        claim.r = shared_r.clone();
        claim.s_col = shared_s_col.clone();
        out.push(claim);
    }
    Ok(out)
}

fn build_padded_ce_claims(
    claims: &[CeClaim<Commitment, F, K>],
    cover_shapes: &[Rv64imCeClaimDigestShape],
) -> Result<Vec<CeClaim<Commitment, F, K>>, Rv64imChunkStepIvcSpartanError> {
    if claims.len() > cover_shapes.len() {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload cannot shrink a CE-claim vector into the canonical claim cover".into(),
        ));
    }
    cover_shapes
        .iter()
        .enumerate()
        .map(|(idx, shape)| {
            let source = claims
                .get(idx)
                .cloned()
                .unwrap_or_else(|| shape.zero_claim());
            pad_ce_claim_to_digest_shape(&source, shape)
        })
        .collect()
}

fn build_padded_ccs_claims(
    claims: &[CcsClaim<Commitment, F>],
    cover_shapes: &[Rv64imCcsClaimShape],
) -> Result<Vec<CcsClaim<Commitment, F>>, Rv64imChunkStepIvcSpartanError> {
    if claims.len() > cover_shapes.len() {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload cannot shrink a CCS-claim vector into the canonical claim cover".into(),
        ));
    }
    cover_shapes
        .iter()
        .enumerate()
        .map(|(idx, shape)| {
            let source = claims
                .get(idx)
                .cloned()
                .unwrap_or_else(|| shape.zero_claim());
            if !shape.covers_claim(&source) {
                return Err(Rv64imChunkStepIvcSpartanError::Prepare(
                    "rv64im recursive-step payload cannot fit a CCS claim into the canonical claim cover".into(),
                ));
            }
            let mut out = source;
            out.x.resize(shape.x_len as usize, F::ZERO);
            Ok(out)
        })
        .collect()
}

fn build_padded_ccs_witnesses(
    witnesses: &[CcsWitness<F>],
    cover_shapes: &[Rv64imCcsWitnessShape],
) -> Result<Vec<CcsWitness<F>>, Rv64imChunkStepIvcSpartanError> {
    if witnesses.len() > cover_shapes.len() {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload cannot shrink a CCS-witness vector into the canonical witness cover".into(),
        ));
    }
    cover_shapes
        .iter()
        .enumerate()
        .map(|(idx, shape)| {
            let source = witnesses
                .get(idx)
                .cloned()
                .unwrap_or_else(|| shape.zero_witness());
            if !shape.covers_witness(&source) {
                return Err(Rv64imChunkStepIvcSpartanError::Prepare(
                    "rv64im recursive-step payload cannot fit a CCS witness into the canonical witness cover".into(),
                ));
            }
            let mut out = source;
            out.w.resize(shape.w_len as usize, F::ZERO);
            out.Z = pad_matrix_to_shape(&out.Z, shape.z_rows as usize, shape.z_cols as usize)?;
            Ok(out)
        })
        .collect()
}

fn pad_rounds(rounds: &[Vec<K>], cover_round_lengths: &[u64]) -> Result<Vec<Vec<K>>, Rv64imChunkStepIvcSpartanError> {
    if rounds.len() > cover_round_lengths.len() {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload cannot shrink sumcheck round count into the canonical cover shape".into(),
        ));
    }
    let mut out = Vec::with_capacity(cover_round_lengths.len());
    for (idx, cover_len) in cover_round_lengths.iter().enumerate() {
        let cover_len = *cover_len as usize;
        let round = rounds.get(idx).cloned().unwrap_or_default();
        if round.len() > cover_len {
            return Err(Rv64imChunkStepIvcSpartanError::Prepare(
                "rv64im recursive-step payload cannot shrink sumcheck round width into the canonical cover shape"
                    .into(),
            ));
        }
        let mut padded = round;
        padded.resize(cover_len, K::ZERO);
        out.push(padded);
    }
    Ok(out)
}

pub fn build_rv64im_main_recursion_f_prime_payload(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    cover_shape: &Rv64imChunkStepIvcShape,
    claim_cover: &Rv64imMainRecursionFPrimeClaimCover,
) -> Result<Rv64imMainRecursionFPrimePayload, Rv64imChunkStepIvcSpartanError> {
    build_rv64im_main_recursion_f_prime_payload_with_trace(advice, cover_shape, claim_cover, None)
}

fn build_rv64im_main_recursion_f_prime_payload_with_trace(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    cover_shape: &Rv64imChunkStepIvcShape,
    claim_cover: &Rv64imMainRecursionFPrimeClaimCover,
    trace_prefix: Option<&str>,
) -> Result<Rv64imMainRecursionFPrimePayload, Rv64imChunkStepIvcSpartanError> {
    let total_started = Instant::now();
    let chunk_trace = advice.main_circuit_chunk_trace();
    let started = Instant::now();
    if claim_cover.state_in_claim_shapes.len() != cover_shape.state_in_claim_count as usize {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step claim cover does not match the canonical state-in claim count".into(),
        ));
    }
    if claim_cover.state_out_claim_shapes.len() != cover_shape.state_out_claim_count as usize {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step claim cover does not match the canonical state-out claim count".into(),
        ));
    }
    if claim_cover.fresh_claim_shapes.len() != cover_shape.fresh_claim_count as usize {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step claim cover does not match the canonical fresh-claim count".into(),
        ));
    }
    if claim_cover.fresh_witness_shapes.len() != cover_shape.fresh_witness_count as usize {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step claim cover does not match the canonical fresh-witness count".into(),
        ));
    }
    if claim_cover.ccs_output_shapes.len() != cover_shape.ccs_output_count as usize {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step claim cover does not match the canonical CCS-output claim count".into(),
        ));
    }
    if claim_cover.child_claim_shapes.len() != cover_shape.child_count as usize {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step claim cover does not match the canonical child-claim count".into(),
        ));
    }
    emit_debug_timing(trace_prefix, "prechecks", elapsed_ms(started));

    let started = Instant::now();
    let state_in_claims = build_padded_state_claims(
        &advice.running_state().carry.main.claims,
        &claim_cover.state_in_claim_shapes,
    )?;
    emit_debug_timing(trace_prefix, "state_in_claims", elapsed_ms(started));
    let started = Instant::now();
    let state_out_claims = build_padded_state_claims(
        &advice.fresh_state_out().carry.main.claims,
        &claim_cover.state_out_claim_shapes,
    )?;
    emit_debug_timing(trace_prefix, "state_out_claims", elapsed_ms(started));

    let started = Instant::now();
    let fresh_claims = if chunk_trace.fresh_claims.is_empty() && cover_shape.fresh_claim_count == 0 {
        Vec::new()
    } else {
        build_padded_ccs_claims(&chunk_trace.fresh_claims, &claim_cover.fresh_claim_shapes)?
    };
    emit_debug_timing(trace_prefix, "fresh_claims", elapsed_ms(started));
    let started = Instant::now();
    let fresh_witnesses = if chunk_trace.fresh_witnesses.is_empty() && cover_shape.fresh_witness_count == 0 {
        Vec::new()
    } else {
        build_padded_ccs_witnesses(&chunk_trace.fresh_witnesses, &claim_cover.fresh_witness_shapes)?
    };
    emit_debug_timing(trace_prefix, "fresh_witnesses", elapsed_ms(started));
    let started = Instant::now();
    let ccs_outputs = if chunk_trace.ccs_trace.ccs_outputs.is_empty() && cover_shape.ccs_output_count == 0 {
        Vec::new()
    } else {
        build_padded_state_claims(&chunk_trace.ccs_trace.ccs_outputs, &claim_cover.ccs_output_shapes)?
    };
    emit_debug_timing(trace_prefix, "ccs_outputs", elapsed_ms(started));
    let started = Instant::now();
    let parent = pad_ce_claim_to_digest_shape(&chunk_trace.ccs_trace.parent, &claim_cover.parent_claim_shape)?;
    emit_debug_timing(trace_prefix, "parent", elapsed_ms(started));
    let started = Instant::now();
    let children = if chunk_trace.ccs_trace.children.is_empty() && cover_shape.child_count == 0 {
        Vec::new()
    } else {
        build_padded_ce_claims(&chunk_trace.ccs_trace.children, &claim_cover.child_claim_shapes)?
    };
    emit_debug_timing(trace_prefix, "children", elapsed_ms(started));

    let started = Instant::now();
    let fe_rounds = pad_rounds(
        &chunk_trace.ccs_trace.ccs_replay_proof.sumcheck_rounds,
        &cover_shape.fe_round_lengths,
    )?;
    let nc_rounds = pad_rounds(
        &chunk_trace.ccs_trace.ccs_replay_proof.sumcheck_rounds_nc,
        &cover_shape.nc_round_lengths,
    )?;
    emit_debug_timing(trace_prefix, "replay_rounds", elapsed_ms(started));

    let started = Instant::now();
    let mut replay = chunk_trace.ccs_trace.ccs_replay_proof.clone();
    replay.sumcheck_rounds = fe_rounds;
    replay.sumcheck_rounds_nc = nc_rounds;
    let pi_ccs = Rv64imMainRecursionFPrimePiCcsPayload { ccs_outputs, replay };
    let pi_rlc = Rv64imMainRecursionFPrimePiRlcPayload { parent };
    let pi_dec = Rv64imMainRecursionFPrimePiDecPayload { children };

    let chunk_cover = Rv64imMainCircuitChunkCover {
        fresh_claim_count: fresh_claims.len() as u64,
        fresh_witness_count: fresh_witnesses.len() as u64,
        fresh_claim_shapes: fresh_claims
            .iter()
            .map(Rv64imMainCircuitCcsClaimShape::from_claim)
            .collect(),
        fresh_witness_shapes: fresh_witnesses
            .iter()
            .map(Rv64imMainCircuitCcsWitnessShape::from_witness)
            .collect(),
        ccs_output_count: pi_ccs.ccs_outputs.len() as u64,
        child_count: pi_dec.children.len() as u64,
        parent_claim_shape: Rv64imMainCircuitCeClaimShape::from_claim(&pi_rlc.parent),
        ccs_output_shapes: pi_ccs
            .ccs_outputs
            .iter()
            .map(Rv64imMainCircuitCeClaimShape::from_claim)
            .collect(),
        child_claim_shapes: pi_dec
            .children
            .iter()
            .map(Rv64imMainCircuitCeClaimShape::from_claim)
            .collect(),
        fe_round_lengths: pi_ccs
            .replay
            .sumcheck_rounds
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
        nc_round_lengths: pi_ccs
            .replay
            .sumcheck_rounds_nc
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
    };
    emit_debug_timing(trace_prefix, "chunk_cover", elapsed_ms(started));
    // Recursive-step F' owns one uniform carried-state transition:
    // replay Π_CCS -> Π_RLC -> Π_DEC and carry the replayed children forward,
    // even when the underlying chunk is terminal in the outer folded proof.
    // Letting terminal-only boundary modes leak into this payload changes the
    // compiled verifier body across steps, which violates Construction 2's
    // fixed-shape F' discipline.
    let boundary_plan = Rv64imChunkBoundaryPlan {
        child_claim_source: Rv64imChunkChildClaimSource::ReplayedChildren,
        next_carry_mode: Rv64imChunkNextCarryMode::ReplaceWithEffectiveChildren,
        rlc_mode: Rv64imChunkRlcMode::Standard {
            constant_child_prefix: 0,
        },
    };
    let provisional_step_shape = build_rv64im_main_recursion_f_prime_step_shape(
        advice,
        &chunk_trace,
        advice.fresh_state_out().transcript.absorbed as u64,
    );
    let started = Instant::now();
    let mut payload = Rv64imMainRecursionFPrimePayload {
        boundary_plan,
        step_shape: provisional_step_shape.clone(),
        cover_shape: cover_shape.clone(),
        padding: build_rv64im_chunk_step_ivc_recursive_step_padding_from_shape(&provisional_step_shape, cover_shape)?,
        z_0: *advice.z_0(),
        z_i: *advice.z_i(),
        z_next: advice.fresh_state_out().carry.terminal_handle.0,
        pc_i: advice.pc_i(),
        pc_next: crate::rv64im::main_recursion::RV64IM_MAIN_RECURSION_TRIVIAL_PC,
        phi_side_commitment_words: advice.phi_side().commitment_words().to_vec(),
        handoff: chunk_trace.handoff.clone(),
        fixed_transcript_out: advice.fresh_state_out().transcript.clone(),
        state_in_claims,
        state_out_claims,
        fresh_claims,
        fresh_witnesses,
        pi_ccs,
        pi_rlc,
        pi_dec,
        chunk_cover,
    };
    emit_debug_timing(trace_prefix, "materialize_payload", elapsed_ms(started));
    let started = Instant::now();
    let replay_chunk = payload
        .effective_chunk_replay_surface(
            &advice.running_state().transcript,
            &advice.running_state().carry.main.claims,
        )
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Prepare(err.to_string()))?;
    emit_debug_timing(trace_prefix, "effective_chunk_replay_surface", elapsed_ms(started));
    if trace_prefix.is_some() {
        let started = Instant::now();
        let fixed_transcript_trace_prefix = trace_prefix.map(|prefix| format!("{prefix}.fixed_transcript"));
        payload.fixed_transcript_out = derive_rv64im_fixed_transcript_out_from_chunk_body(
            &payload,
            &advice.running_state().transcript,
            &replay_chunk,
            &advice.running_state().carry.main.claims,
            &advice.fresh_state_out().carry.main.claims,
            advice.running_state().carry.terminal_handle.0,
            fixed_transcript_trace_prefix.as_deref(),
        )
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Prepare(err.to_string()))?;
        emit_debug_timing(trace_prefix, "fixed_transcript_out", elapsed_ms(started));
        if payload.fixed_transcript_out != advice.fresh_state_out().transcript {
            return Err(Rv64imChunkStepIvcSpartanError::Prepare(
                format!(
                    "rv64im recursive-step fixed transcript out does not match the carried native state_out transcript for chunk {} (halted_out={}, in_absorbed={}, derived_out_absorbed={}, native_out_absorbed={})",
                    advice.chunk_index(),
                    advice.bridge_handoff_halted_out(),
                    advice.running_state().transcript.absorbed,
                    payload.fixed_transcript_out.absorbed,
                    advice.fresh_state_out().transcript.absorbed,
                ),
            ));
        }
    } else {
        // `main_circuit_chunk_trace` was already built from authoritative replay inputs and
        // checked that the native replayed transcript_out equals the carried state_out
        // transcript. Replaying the circuit body again here is redundant on the non-debug path.
        payload.fixed_transcript_out = advice.fresh_state_out().transcript.clone();
    }
    let started = Instant::now();
    payload.step_shape = build_rv64im_main_recursion_f_prime_step_shape(
        advice,
        &chunk_trace,
        payload.fixed_transcript_out.absorbed as u64,
    );
    payload.padding = build_rv64im_chunk_step_ivc_recursive_step_padding_from_shape(&payload.step_shape, cover_shape)?;
    emit_debug_timing(trace_prefix, "final_shape_and_padding", elapsed_ms(started));
    let started = Instant::now();
    if !payload.matches_cover_shape() {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload does not match the canonical cover shape".into(),
        ));
    }
    if !payload.matches_explicit_semantics(advice) {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im recursive-step payload explicit z/pc semantics drifted from the native F' advice".into(),
        ));
    }
    emit_debug_timing(trace_prefix, "final_checks", elapsed_ms(started));
    emit_debug_timing(trace_prefix, "total", elapsed_ms(total_started));
    Ok(payload)
}

pub fn build_rv64im_main_recursion_f_prime_payloads(
    advices: &[Rv64imMainRecursionFPrimeAdvice],
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
) -> Result<Vec<Rv64imMainRecursionFPrimePayload>, Rv64imChunkStepIvcSpartanError> {
    advices
        .iter()
        .map(|advice| {
            build_rv64im_main_recursion_f_prime_payload(advice, &spartan_shape.cover_shape, &spartan_shape.claim_cover)
        })
        .collect()
}

pub fn build_rv64im_main_recursion_step_spartan_shape(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<Rv64imMainRecursionStepSpartanShape, Rv64imChunkStepIvcSpartanError> {
    let advices = build_rv64im_main_recursion_f_prime_advices(relations)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Prepare(err.to_string()))?;
    build_rv64im_main_recursion_step_spartan_shape_from_advices(relations, &advices)
}

pub fn build_rv64im_main_recursion_step_spartan_shape_from_advices(
    relations: &[Rv64imChunkStepIvcRelation],
    advices: &[Rv64imMainRecursionFPrimeAdvice],
) -> Result<Rv64imMainRecursionStepSpartanShape, Rv64imChunkStepIvcSpartanError> {
    let cover_shape = build_rv64im_chunk_step_ivc_recursive_step_cover_shape(relations)?;
    let claim_cover = build_rv64im_main_recursion_f_prime_claim_cover(&advices)?;
    Ok(Rv64imMainRecursionStepSpartanShape {
        cover_shape,
        claim_cover,
    })
}

pub fn build_rv64im_main_recursion_f_prime_payloads_with_spartan_shape(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<
    (
        Rv64imMainRecursionStepSpartanShape,
        Vec<Rv64imMainRecursionFPrimePayload>,
    ),
    Rv64imChunkStepIvcSpartanError,
> {
    let advices = build_rv64im_main_recursion_f_prime_advices(relations)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Prepare(err.to_string()))?;
    let spartan_shape = build_rv64im_main_recursion_step_spartan_shape_from_advices(relations, &advices)?;
    let payloads = build_rv64im_main_recursion_f_prime_payloads(&advices, &spartan_shape)?;
    Ok((spartan_shape, payloads))
}

pub fn build_rv64im_main_recursion_f_prime_backend_relations(
    relations: &[Rv64imChunkStepIvcRelation],
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
) -> Result<Vec<Rv64imMainRecursionFPrimeBackendRelation>, SimpleKernelError> {
    let f_prime_advices = build_rv64im_main_recursion_f_prime_advices(relations)?;
    let payloads = build_rv64im_main_recursion_f_prime_payloads(&f_prime_advices, spartan_shape)
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    if f_prime_advices.len() != payloads.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM recursive-step backend builder produced mismatched step and payload counts".into(),
        ));
    }
    f_prime_advices
        .into_iter()
        .zip(payloads)
        .map(|(f_prime_advice, payload)| {
            let spartan_statement = build_rv64im_main_recursion_step_spartan_statement(&f_prime_advice)?;
            let backend_relation = Rv64imMainRecursionFPrimeBackendRelation {
                f_prime_advice,
                spartan_statement,
                payload,
            };
            debug_check_rv64im_main_recursion_f_prime_backend_relation_semantics(&backend_relation)?;
            Ok(backend_relation)
        })
        .collect()
}

pub fn build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<
    (
        Rv64imMainRecursionStepSpartanShape,
        Vec<Rv64imMainRecursionFPrimeBackendRelation>,
    ),
    SimpleKernelError,
> {
    let advices = build_rv64im_main_recursion_f_prime_advices(relations)?;
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(relations, &advices)
}

pub fn build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
    relations: &[Rv64imChunkStepIvcRelation],
    advices: &[Rv64imMainRecursionFPrimeAdvice],
) -> Result<
    (
        Rv64imMainRecursionStepSpartanShape,
        Vec<Rv64imMainRecursionFPrimeBackendRelation>,
    ),
    SimpleKernelError,
> {
    Ok(
        build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices_and_perf(
            relations, advices, None,
        )?
        .0,
    )
}

pub fn build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices_and_perf(
    relations: &[Rv64imChunkStepIvcRelation],
    advices: &[Rv64imMainRecursionFPrimeAdvice],
    trace_prefix: Option<&str>,
) -> Result<
    (
        (
            Rv64imMainRecursionStepSpartanShape,
            Vec<Rv64imMainRecursionFPrimeBackendRelation>,
        ),
        Rv64imMainRecursionFPrimeBackendRelationBuildPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();
    let started = Instant::now();
    let spartan_shape = build_rv64im_main_recursion_step_spartan_shape_from_advices(relations, advices)
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let spartan_shape_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "spartan_shape", spartan_shape_ms);
    let started = Instant::now();
    let mut payloads = Vec::with_capacity(advices.len());
    for (step_index, advice) in advices.iter().enumerate() {
        let payload_trace_prefix = trace_prefix.map(|prefix| format!("{prefix}.step_{step_index}_payload"));
        let payload_started = Instant::now();
        let payload = build_rv64im_main_recursion_f_prime_payload_with_trace(
            advice,
            &spartan_shape.cover_shape,
            &spartan_shape.claim_cover,
            payload_trace_prefix.as_deref(),
        )
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
        emit_debug_timing(
            trace_prefix,
            &format!("step_{step_index}_payload_total"),
            elapsed_ms(payload_started),
        );
        payloads.push(payload);
    }
    let payloads_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "payloads", payloads_ms);
    if advices.len() != payloads.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM recursive-step backend builder produced mismatched step and payload counts".into(),
        ));
    }
    let mut statement_build_ms = 0.0;
    let mut semantics_check_ms = 0.0;
    let backend_relations = advices
        .iter()
        .cloned()
        .zip(payloads)
        .enumerate()
        .map(|(step_index, (f_prime_advice, payload))| {
            let started = Instant::now();
            let spartan_statement = build_rv64im_main_recursion_step_spartan_statement(&f_prime_advice)?;
            let statement_ms = elapsed_ms(started);
            statement_build_ms += statement_ms;
            emit_debug_timing(
                trace_prefix,
                &format!("step_{step_index}_statement_build"),
                statement_ms,
            );
            let backend_relation = Rv64imMainRecursionFPrimeBackendRelation {
                f_prime_advice,
                spartan_statement,
                payload,
            };
            let started = Instant::now();
            debug_check_rv64im_main_recursion_f_prime_backend_relation_semantics(&backend_relation)?;
            let semantics_ms = elapsed_ms(started);
            semantics_check_ms += semantics_ms;
            emit_debug_timing(
                trace_prefix,
                &format!("step_{step_index}_semantics_check"),
                semantics_ms,
            );
            Ok(backend_relation)
        })
        .collect::<Result<Vec<_>, SimpleKernelError>>()?;
    let perf = Rv64imMainRecursionFPrimeBackendRelationBuildPerf {
        spartan_shape_ms,
        payloads_ms,
        statement_build_ms,
        semantics_check_ms,
        total_ms: elapsed_ms(total_started),
        relation_count: advices.len(),
    };
    emit_debug_timing(trace_prefix, "statement_build_total", statement_build_ms);
    emit_debug_timing(trace_prefix, "semantics_check_total", semantics_check_ms);
    emit_debug_timing(trace_prefix, "total", perf.total_ms);
    Ok(((spartan_shape, backend_relations), perf))
}

pub fn debug_trace_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
    relations: &[Rv64imChunkStepIvcRelation],
    advices: &[Rv64imMainRecursionFPrimeAdvice],
    trace_prefix: &str,
) -> Result<
    (
        (
            Rv64imMainRecursionStepSpartanShape,
            Vec<Rv64imMainRecursionFPrimeBackendRelation>,
        ),
        Rv64imMainRecursionFPrimeBackendRelationBuildPerf,
    ),
    SimpleKernelError,
> {
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices_and_perf(
        relations,
        advices,
        Some(trace_prefix),
    )
}

fn ccs_claim_matches(left: &CcsClaim<Commitment, F>, right: &CcsClaim<Commitment, F>) -> bool {
    left.c == right.c && left.x == right.x && left.m_in == right.m_in
}

fn ccs_witness_matches(left: &CcsWitness<F>, right: &CcsWitness<F>) -> bool {
    left.w == right.w
        && left.Z.rows() == right.Z.rows()
        && left.Z.cols() == right.Z.cols()
        && left.Z.as_slice() == right.Z.as_slice()
}

pub fn debug_check_rv64im_chunk_step_recursive_effective_chunk_trace_matches_native(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), SimpleKernelError> {
    let native_trace = backend_relation.f_prime_advice.main_circuit_chunk_trace();
    let effective_replay_surface = backend_relation.payload.effective_chunk_replay_surface(
        &backend_relation.f_prime_advice.running_state().transcript,
        &backend_relation
            .f_prime_advice
            .running_state()
            .carry
            .main
            .claims,
    )?;
    let native_replay_surface = native_trace.replay_surface()?;

    if effective_replay_surface.handoff.public_chunk.start_index
        != native_replay_surface.handoff.public_chunk.start_index
        || effective_replay_surface.handoff.public_chunk.steps.len()
            != native_replay_surface.handoff.public_chunk.steps.len()
        || effective_replay_surface
            .handoff
            .public_chunk_instance_digest
            != native_replay_surface.handoff.public_chunk_instance_digest
        || effective_replay_surface.handoff.public_chunk_digest != native_replay_surface.handoff.public_chunk_digest
        || effective_replay_surface.handoff.bridge_handoff_digest != native_replay_surface.handoff.bridge_handoff_digest
        || effective_replay_surface.handoff.chunk_relation_digest != native_replay_surface.handoff.chunk_relation_digest
        || effective_replay_surface.fresh_claims.len() != native_replay_surface.fresh_claims.len()
        || effective_replay_surface.pi_ccs.ccs_outputs != native_replay_surface.pi_ccs.ccs_outputs
        || effective_replay_surface.pi_ccs.replay_proof != native_replay_surface.pi_ccs.replay_proof
        || effective_replay_surface.pi_rlc.parent != native_replay_surface.pi_rlc.parent
        || effective_replay_surface.pi_dec.children != native_replay_surface.pi_dec.children
        || effective_replay_surface.pi_ccs.public_challenges.alpha
            != native_replay_surface.pi_ccs.public_challenges.alpha
        || effective_replay_surface.pi_ccs.public_challenges.beta_a
            != native_replay_surface.pi_ccs.public_challenges.beta_a
        || effective_replay_surface.pi_ccs.public_challenges.beta_r
            != native_replay_surface.pi_ccs.public_challenges.beta_r
        || effective_replay_surface.pi_ccs.public_challenges.beta_m
            != native_replay_surface.pi_ccs.public_challenges.beta_m
        || effective_replay_surface.pi_ccs.public_challenges.gamma
            != native_replay_surface.pi_ccs.public_challenges.gamma
        || effective_replay_surface.pi_ccs.row_chals != native_replay_surface.pi_ccs.row_chals
        || effective_replay_surface.pi_ccs.alpha_prime != native_replay_surface.pi_ccs.alpha_prime
        || effective_replay_surface.pi_ccs.s_col != native_replay_surface.pi_ccs.s_col
        || effective_replay_surface.pi_ccs.alpha_prime_nc != native_replay_surface.pi_ccs.alpha_prime_nc
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM effective chunk replay surface recovered from the recursive payload does not match the native trace"
                .into(),
        ));
    }

    for (effective, native) in effective_replay_surface
        .fresh_claims
        .iter()
        .zip(native_replay_surface.fresh_claims.iter())
    {
        if !ccs_claim_matches(effective, native) {
            return Err(SimpleKernelError::Bridge(
                "RV64IM effective fresh claim recovered from the recursive payload does not match the native trace"
                    .into(),
            ));
        }
    }

    for (effective, native) in backend_relation
        .payload
        .fresh_witnesses
        .iter()
        .take(backend_relation.payload.step_shape.fresh_witness_count as usize)
        .zip(native_trace.fresh_witnesses.iter())
    {
        if !ccs_witness_matches(effective, native) {
            return Err(SimpleKernelError::Bridge(
                "RV64IM effective fresh witness recovered from the recursive payload does not match the native trace"
                    .into(),
            ));
        }
    }

    Ok(())
}
