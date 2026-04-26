//! Owns native fixed-shape chunk-step shape/padding helpers.
//!
//! These helpers feed the main-recursion F' backend builder; they are not a
//! standalone compressed-proof authority.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::rv64im::chunk_step_ivc::{
    build_rv64im_chunk_step_ivc_published_target, build_rv64im_chunk_step_ivc_statement_from_authoritative_parts,
    validate_rv64im_chunk_step_ivc_surface, Rv64imChunkStepIvcPublishedTarget, Rv64imChunkStepIvcRelation,
    Rv64imChunkStepIvcStatement, Rv64imChunkStepIvcWitness,
};
use crate::rv64im::main_relation_trace::{
    build_rv64im_main_circuit_chunk_trace_from_authoritative_parts, Rv64imMainCircuitChunkTrace,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imChunkStepIvcShape {
    pub terminal_step: bool,
    pub state_in_claim_count: u64,
    pub state_out_claim_count: u64,
    pub fresh_claim_count: u64,
    pub fresh_witness_count: u64,
    pub ccs_output_count: u64,
    pub child_count: u64,
    pub transcript_in_absorbed: u64,
    pub transcript_out_absorbed: u64,
    pub fe_round_lengths: Vec<u64>,
    pub nc_round_lengths: Vec<u64>,
}

impl Rv64imChunkStepIvcShape {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/chunk_step_ivc/shape");
        tr.append_u64s(
            b"neo.fold.next/rv64im/chunk_step_ivc/shape/meta",
            &[
                u64::from(self.terminal_step),
                self.state_in_claim_count,
                self.state_out_claim_count,
                self.fresh_claim_count,
                self.fresh_witness_count,
                self.ccs_output_count,
                self.child_count,
                self.transcript_in_absorbed,
                self.transcript_out_absorbed,
                self.fe_round_lengths.len() as u64,
                self.nc_round_lengths.len() as u64,
            ],
        );
        for len in &self.fe_round_lengths {
            tr.append_u64s(b"neo.fold.next/rv64im/chunk_step_ivc/shape/fe_round", &[*len]);
        }
        for len in &self.nc_round_lengths {
            tr.append_u64s(b"neo.fold.next/rv64im/chunk_step_ivc/shape/nc_round", &[*len]);
        }
        tr.digest32()
    }

    pub fn recursive_step_cover_seed() -> Self {
        Self {
            // A future fixed recursive step should treat terminality as a selector,
            // not as a separate circuit shape family.
            terminal_step: false,
            state_in_claim_count: 0,
            state_out_claim_count: 0,
            fresh_claim_count: 0,
            fresh_witness_count: 0,
            ccs_output_count: 0,
            child_count: 0,
            transcript_in_absorbed: 0,
            transcript_out_absorbed: 0,
            fe_round_lengths: Vec::new(),
            nc_round_lengths: Vec::new(),
        }
    }

    pub fn recursive_step_cover_merge(&self, other: &Self) -> Self {
        fn merge_round_lengths(left: &[u64], right: &[u64]) -> Vec<u64> {
            let len = left.len().max(right.len());
            (0..len)
                .map(|idx| {
                    left.get(idx)
                        .copied()
                        .unwrap_or(0)
                        .max(right.get(idx).copied().unwrap_or(0))
                })
                .collect()
        }

        Self {
            terminal_step: false,
            state_in_claim_count: self.state_in_claim_count.max(other.state_in_claim_count),
            state_out_claim_count: self.state_out_claim_count.max(other.state_out_claim_count),
            fresh_claim_count: self.fresh_claim_count.max(other.fresh_claim_count),
            fresh_witness_count: self.fresh_witness_count.max(other.fresh_witness_count),
            ccs_output_count: self.ccs_output_count.max(other.ccs_output_count),
            child_count: self.child_count.max(other.child_count),
            transcript_in_absorbed: self
                .transcript_in_absorbed
                .max(other.transcript_in_absorbed),
            transcript_out_absorbed: self
                .transcript_out_absorbed
                .max(other.transcript_out_absorbed),
            fe_round_lengths: merge_round_lengths(&self.fe_round_lengths, &other.fe_round_lengths),
            nc_round_lengths: merge_round_lengths(&self.nc_round_lengths, &other.nc_round_lengths),
        }
    }

    pub fn covers_recursive_step_shape(&self, other: &Self) -> bool {
        fn covers_round_lengths(cover: &[u64], other: &[u64]) -> bool {
            if cover.len() < other.len() {
                return false;
            }
            other
                .iter()
                .enumerate()
                .all(|(idx, value)| cover[idx] >= *value)
        }

        self.state_in_claim_count >= other.state_in_claim_count
            && self.state_out_claim_count >= other.state_out_claim_count
            && self.fresh_claim_count >= other.fresh_claim_count
            && self.fresh_witness_count >= other.fresh_witness_count
            && self.ccs_output_count >= other.ccs_output_count
            && self.child_count >= other.child_count
            && self.transcript_in_absorbed >= other.transcript_in_absorbed
            && self.transcript_out_absorbed >= other.transcript_out_absorbed
            && covers_round_lengths(&self.fe_round_lengths, &other.fe_round_lengths)
            && covers_round_lengths(&self.nc_round_lengths, &other.nc_round_lengths)
    }

    /// Strict structural-equality check across two recursive-step shapes.
    ///
    /// Compares every field that participates in the fixed F' circuit
    /// shape (claim counts, sumcheck round lengths, output counts), but
    /// intentionally ignores the transcript absorb counters
    /// `transcript_in_absorbed` / `transcript_out_absorbed` which
    /// accumulate monotonically per chunk and are therefore *not* shape
    /// fields. Also ignores `terminal_step`, which is a selector, not a
    /// shape family.
    pub fn canonical_recursive_step_shape_equal(&self, other: &Self) -> bool {
        self.state_in_claim_count == other.state_in_claim_count
            && self.state_out_claim_count == other.state_out_claim_count
            && self.fresh_claim_count == other.fresh_claim_count
            && self.fresh_witness_count == other.fresh_witness_count
            && self.ccs_output_count == other.ccs_output_count
            && self.child_count == other.child_count
            && self.fe_round_lengths == other.fe_round_lengths
            && self.nc_round_lengths == other.nc_round_lengths
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imChunkStepIvcRecursiveStepPadding {
    pub terminal_step: bool,
    pub state_in_claim_pad: u64,
    pub state_out_claim_pad: u64,
    pub fresh_claim_pad: u64,
    pub fresh_witness_pad: u64,
    pub ccs_output_pad: u64,
    pub child_pad: u64,
    pub fe_round_count_pad: u64,
    pub fe_round_coeff_pad: Vec<u64>,
    pub nc_round_count_pad: u64,
    pub nc_round_coeff_pad: Vec<u64>,
}

impl Rv64imChunkStepIvcRecursiveStepPadding {
    pub fn is_noop(&self) -> bool {
        self.state_in_claim_pad == 0
            && self.state_out_claim_pad == 0
            && self.fresh_claim_pad == 0
            && self.fresh_witness_pad == 0
            && self.ccs_output_pad == 0
            && self.child_pad == 0
            && self.fe_round_count_pad == 0
            && self.fe_round_coeff_pad.iter().all(|pad| *pad == 0)
            && self.nc_round_count_pad == 0
            && self.nc_round_coeff_pad.iter().all(|pad| *pad == 0)
    }
}

#[derive(Debug, Error)]
pub enum Rv64imChunkStepIvcSpartanError {
    #[error("rv64im chunk-step ivc prepare failed: {0}")]
    Prepare(String),
    #[error("rv64im chunk-step ivc verify failed: {0}")]
    Verify(String),
}

pub(crate) fn prepare_rv64im_chunk_step_ivc_circuit_inputs(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<(Rv64imChunkStepIvcPublishedTarget, Rv64imMainCircuitChunkTrace), Rv64imChunkStepIvcSpartanError> {
    validate_rv64im_chunk_step_ivc_surface(statement, witness)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    let published_target = build_rv64im_chunk_step_ivc_published_target(statement)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    let published_summary = published_target.chunk_summary();
    let chunk_trace = build_rv64im_main_circuit_chunk_trace_from_authoritative_parts(
        witness.handoff.bridge_handoff.chunk_index as usize,
        &witness.handoff,
        &published_summary,
        &witness.state_in.carry,
        &witness.state_out.carry,
        &witness.state_in.transcript,
        &witness.state_out.transcript,
        &witness.replay_witness,
    )
    .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    let canonical_statement = build_rv64im_chunk_step_ivc_statement_from_authoritative_parts(
        published_target.program_digest,
        witness,
        chunk_trace.handoff.chunk_relation_digest,
    );
    if canonical_statement != *statement {
        return Err(Rv64imChunkStepIvcSpartanError::Verify(
            "rv64im chunk-step ivc statement shell does not match the authoritative published step statement".into(),
        ));
    }
    Ok((published_target, chunk_trace))
}

pub fn build_rv64im_chunk_step_ivc_shape(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<Rv64imChunkStepIvcShape, Rv64imChunkStepIvcSpartanError> {
    let (_, effective_chunk) = prepare_rv64im_chunk_step_ivc_circuit_inputs(statement, witness)?;
    Ok(rv64im_chunk_step_ivc_shape_from_trace(witness, &effective_chunk))
}

pub fn build_rv64im_chunk_step_ivc_recursive_step_cover_shape(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<Rv64imChunkStepIvcShape, Rv64imChunkStepIvcSpartanError> {
    let mut cover = Rv64imChunkStepIvcShape::recursive_step_cover_seed();
    for relation in relations {
        let step_shape = build_rv64im_chunk_step_ivc_shape(&relation.statement, &relation.witness)?;
        cover = cover.recursive_step_cover_merge(&step_shape);
    }
    Ok(cover)
}

pub fn build_rv64im_chunk_step_ivc_recursive_step_padding(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
    cover_shape: &Rv64imChunkStepIvcShape,
) -> Result<Rv64imChunkStepIvcRecursiveStepPadding, Rv64imChunkStepIvcSpartanError> {
    let step_shape = build_rv64im_chunk_step_ivc_shape(statement, witness)?;
    build_rv64im_chunk_step_ivc_recursive_step_padding_from_shape(&step_shape, cover_shape)
}

pub fn build_rv64im_chunk_step_ivc_recursive_step_padding_from_shape(
    step_shape: &Rv64imChunkStepIvcShape,
    cover_shape: &Rv64imChunkStepIvcShape,
) -> Result<Rv64imChunkStepIvcRecursiveStepPadding, Rv64imChunkStepIvcSpartanError> {
    if !cover_shape.covers_recursive_step_shape(step_shape) {
        return Err(Rv64imChunkStepIvcSpartanError::Prepare(
            "rv64im chunk-step recursive cover shape does not dominate the per-step shape".into(),
        ));
    }

    fn build_round_coeff_pad(step_rounds: &[u64], cover_rounds: &[u64]) -> Vec<u64> {
        (0..cover_rounds.len())
            .map(|idx| cover_rounds[idx] - step_rounds.get(idx).copied().unwrap_or(0))
            .collect()
    }

    Ok(Rv64imChunkStepIvcRecursiveStepPadding {
        terminal_step: step_shape.terminal_step,
        state_in_claim_pad: cover_shape.state_in_claim_count - step_shape.state_in_claim_count,
        state_out_claim_pad: cover_shape.state_out_claim_count - step_shape.state_out_claim_count,
        fresh_claim_pad: cover_shape.fresh_claim_count - step_shape.fresh_claim_count,
        fresh_witness_pad: cover_shape.fresh_witness_count - step_shape.fresh_witness_count,
        ccs_output_pad: cover_shape.ccs_output_count - step_shape.ccs_output_count,
        child_pad: cover_shape.child_count - step_shape.child_count,
        fe_round_count_pad: cover_shape.fe_round_lengths.len() as u64 - step_shape.fe_round_lengths.len() as u64,
        fe_round_coeff_pad: build_round_coeff_pad(&step_shape.fe_round_lengths, &cover_shape.fe_round_lengths),
        nc_round_count_pad: cover_shape.nc_round_lengths.len() as u64 - step_shape.nc_round_lengths.len() as u64,
        nc_round_coeff_pad: build_round_coeff_pad(&step_shape.nc_round_lengths, &cover_shape.nc_round_lengths),
    })
}

fn rv64im_chunk_step_ivc_shape_from_trace(
    witness: &Rv64imChunkStepIvcWitness,
    effective_chunk: &Rv64imMainCircuitChunkTrace,
) -> Rv64imChunkStepIvcShape {
    Rv64imChunkStepIvcShape {
        terminal_step: witness.terminal_step,
        state_in_claim_count: witness.state_in.carry.main.claims.len() as u64,
        state_out_claim_count: witness.state_out.carry.main.claims.len() as u64,
        fresh_claim_count: effective_chunk.fresh_claims.len() as u64,
        fresh_witness_count: effective_chunk.fresh_witnesses.len() as u64,
        ccs_output_count: effective_chunk.ccs_trace.ccs_outputs.len() as u64,
        child_count: effective_chunk.ccs_trace.children.len() as u64,
        transcript_in_absorbed: witness.state_in.transcript.absorbed as u64,
        transcript_out_absorbed: witness.state_out.transcript.absorbed as u64,
        fe_round_lengths: effective_chunk
            .ccs_trace
            .ccs_replay_proof
            .sumcheck_rounds
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
        nc_round_lengths: effective_chunk
            .ccs_trace
            .ccs_replay_proof
            .sumcheck_rounds_nc
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
    }
}
