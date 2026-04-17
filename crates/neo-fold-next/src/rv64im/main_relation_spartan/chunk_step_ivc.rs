//! Owns a fixed-shape Spartan surface for one RV64IM chunk-step IVC relation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ccs::CcsStructure;
use neo_math::F;
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use serde::{Deserialize, Serialize};
use spartan2::{
    bellpepper::poseidon2::hash_packed_goldilocks_fields,
    traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use thiserror::Error;

use super::*;
use crate::rv64im::chunk_relation::rv64im_chunk_replay_witness_digest;
use crate::rv64im::chunk_step_ivc::{
    build_rv64im_chunk_step_ivc_published_target, build_rv64im_chunk_step_ivc_statement_from_authoritative_parts,
    rv64im_bridge_handoff_chain_digest_from_digests, rv64im_chunk_step_ivc_initial_state,
    rv64im_step_statement_chain_digest_from_digests, validate_rv64im_chunk_step_ivc_surface,
    verify_rv64im_chunk_step_ivc_chain, Rv64imChunkStepIvcPublishedTarget, Rv64imChunkStepIvcRelation,
    Rv64imChunkStepIvcStatement, Rv64imChunkStepIvcWitness,
};
use crate::rv64im::final_relation::rv64im_chunk_fold_state_instance_digest;
use crate::rv64im::main_relation_circuit::claim::{
    alloc_ce_claim, alloc_ce_claim_with_shared_point, enforce_claim_eq_native, me_digest_poseidon,
    packed_bytes_field_values, CeClaimVar,
};
use crate::rv64im::main_relation_trace::{
    build_rv64im_main_circuit_chunk_trace_from_authoritative_parts, Rv64imMainCircuitChunkCover,
};

pub type Rv64imChunkStepIvcSpartanProverKey = Rv64imSpartan2DeciderProverKey;
pub type Rv64imChunkStepIvcSpartanVerifierKey = Rv64imSpartan2DeciderVerifierKey;
pub type Rv64imChunkStepIvcSpartanKeyPair =
    Arc<(Rv64imChunkStepIvcSpartanProverKey, Rv64imChunkStepIvcSpartanVerifierKey)>;

static RV64IM_CHUNK_STEP_IVC_SETUP_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Rv64imChunkStepIvcSpartanKeyPair>>> =
    OnceLock::new();
#[allow(dead_code)]
static RV64IM_CHUNK_STEP_IVC_COMPRESSED_CHAIN_SETUP_CACHE: OnceLock<
    Mutex<HashMap<[u8; 32], Rv64imChunkStepIvcSpartanKeyPair>>,
> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imChunkStepIvcSpartanProof {
    pub snark_data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imChunkStepIvcSpartanChainProof {
    pub step_proofs: Vec<Rv64imChunkStepIvcSpartanProof>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imChunkStepIvcSpartanCompressedChainProof {
    pub snark_data: Vec<u8>,
}

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
    #[error("rv64im chunk-step ivc setup failed: {0}")]
    Setup(String),
    #[error("rv64im chunk-step ivc prepare failed: {0}")]
    Prepare(String),
    #[error("rv64im chunk-step ivc prove failed: {0}")]
    Prove(String),
    #[error("rv64im chunk-step ivc verify failed: {0}")]
    Verify(String),
    #[error("rv64im chunk-step ivc proof encoding failed: {0}")]
    Encode(String),
    #[error("rv64im chunk-step ivc proof decoding failed: {0}")]
    Decode(String),
    #[error("rv64im chunk-step ivc public IO mismatch")]
    PublicIoMismatch,
    #[error("rv64im chunk-step ivc chain length mismatch")]
    ChainLengthMismatch,
}

#[derive(Clone)]
struct Rv64imChunkStepIvcCircuit {
    params: NeoParams,
    structure: CcsStructure<F>,
    dims: Dims,
    mat_digest: [Goldilocks; 4],
    published_target: Rv64imChunkStepIvcPublishedTarget,
    witness: Rv64imChunkStepIvcWitness,
    cover_chunk: Rv64imMainCircuitChunkCover,
    effective_chunk: Rv64imMainCircuitChunkTrace,
}

#[derive(Clone)]
#[allow(dead_code)]
struct Rv64imChunkStepIvcCompressedChainCircuit {
    step_circuits: Vec<Rv64imChunkStepIvcCircuit>,
}

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct Rv64imChunkStepIvcStateVar {
    pub(crate) claims: Vec<CeClaimVar>,
    pub(crate) transcript_state: [AllocatedNum<SpartanF>; neo_params::poseidon2_goldilocks::WIDTH],
    pub(crate) transcript_absorbed: usize,
    pub(crate) terminal_handle: [AllocatedNum<SpartanF>; 4],
}

#[allow(dead_code)]
struct Rv64imChunkStepIvcBoundaryVar {
    state_in: Rv64imChunkStepIvcStateVar,
    state_out: Rv64imChunkStepIvcStateVar,
}

impl Rv64imChunkStepIvcCircuit {
    fn expected_public_values(&self) -> Vec<SpartanF> {
        chunk_step_ivc_spartan_public_values(&self.published_target)
    }
}

impl SpartanCircuit<Rv64imSpartan2DeciderEngine> for Rv64imChunkStepIvcCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.expected_public_values())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let public_inputs = self
            .expected_public_values()
            .into_iter()
            .enumerate()
            .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut public_cursor = 0usize;
        let _ = synthesize_chunk_step_ivc_relation_body(
            self,
            &mut cs.namespace(|| "chunk_step_ivc"),
            &public_inputs,
            &mut public_cursor,
        )?;
        if public_cursor != public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(())
    }
}

#[allow(dead_code)]
impl Rv64imChunkStepIvcCompressedChainCircuit {
    fn expected_public_values(&self) -> Vec<SpartanF> {
        let step_statement_digests = self
            .step_circuits
            .iter()
            .map(|circuit| circuit.published_target.expected_digest())
            .collect::<Vec<_>>();
        let bridge_handoff_digests = self
            .step_circuits
            .iter()
            .map(|circuit| circuit.witness.handoff.bridge_handoff.digest)
            .collect::<Vec<_>>();
        let final_state = self
            .step_circuits
            .last()
            .map(|circuit| circuit.witness.state_out.clone())
            .unwrap_or_else(rv64im_chunk_step_ivc_initial_state);
        let mut out = Vec::with_capacity(17);
        out.push(SpartanF::from_canonical_u64(self.step_circuits.len() as u64));
        out.extend(digest32_as_spartan_fields(rv64im_chunk_fold_state_instance_digest(
            &final_state,
        )));
        out.extend(digest32_as_spartan_fields(
            rv64im_step_statement_chain_digest_from_digests(&step_statement_digests),
        ));
        out.extend(digest32_as_spartan_fields(
            rv64im_bridge_handoff_chain_digest_from_digests(&bridge_handoff_digests),
        ));
        out.extend(digest32_as_spartan_fields(final_state.carry.terminal_handle.0));
        out
    }
}

impl SpartanCircuit<Rv64imSpartan2DeciderEngine> for Rv64imChunkStepIvcCompressedChainCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.expected_public_values())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let public_inputs = self
            .expected_public_values()
            .into_iter()
            .enumerate()
            .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut public_cursor = 0usize;
        let chain_len_input = next_public_u64(&public_inputs, &mut public_cursor)?;
        let accumulator_instance_input =
            next_public_digest(&public_inputs, &mut public_cursor, "accumulator_instance_digest")?;
        let step_statement_chain_input =
            next_public_digest(&public_inputs, &mut public_cursor, "step_statement_chain_digest")?;
        let bridge_handoff_chain_input =
            next_public_digest(&public_inputs, &mut public_cursor, "bridge_handoff_chain_digest")?;
        let terminal_handle_input = next_public_digest(&public_inputs, &mut public_cursor, "terminal_handle_digest")?;
        enforce_u64_input_eq(
            &mut cs.namespace(|| "chain_len_eq"),
            &chain_len_input,
            self.step_circuits.len() as u64,
            "chain_len_eq",
        )?;

        let initial_state = alloc_native_chunk_step_state(
            &mut cs.namespace(|| "initial_state"),
            &rv64im_chunk_step_ivc_initial_state(),
            "initial_state",
        )?;
        let mut step_statement_chain = chunk_step_ivc_digest_chain_seed_circuit(
            &mut cs.namespace(|| "step_statement_chain_seed"),
            0x7276_3634_7374_6d74,
            "step_statement_chain_seed",
        )?;
        let mut bridge_handoff_chain = chunk_step_ivc_digest_chain_seed_circuit(
            &mut cs.namespace(|| "bridge_handoff_chain_seed"),
            0x7276_3634_62686467,
            "bridge_handoff_chain_seed",
        )?;
        let mut previous_state_out: Option<Rv64imChunkStepIvcStateVar> = None;
        for (relation_index, circuit) in self.step_circuits.iter().enumerate() {
            let relation_public_inputs = alloc_const_field_values(
                &mut cs.namespace(|| format!("relation_{relation_index}_public_inputs")),
                &chunk_step_ivc_spartan_public_values(&circuit.published_target),
                &format!("relation_{relation_index}_public_inputs"),
            )?;
            let mut relation_public_cursor = 0usize;
            let boundary = synthesize_chunk_step_ivc_relation_body(
                circuit,
                &mut cs.namespace(|| format!("relation_{relation_index}")),
                &relation_public_inputs,
                &mut relation_public_cursor,
            )?;
            if relation_public_cursor != relation_public_inputs.len() {
                return Err(SynthesisError::Unsatisfiable);
            }
            let bridge_handoff_const = digest_const_inputs(
                &mut cs.namespace(|| format!("relation_{relation_index}_bridge_handoff_const")),
                circuit.witness.handoff.bridge_handoff.digest,
                &format!("relation_{relation_index}_bridge_handoff_const"),
            )?;
            let statement_digest_const = digest_const_inputs(
                &mut cs.namespace(|| format!("relation_{relation_index}_statement_digest_const")),
                circuit.published_target.expected_digest(),
                &format!("relation_{relation_index}_statement_digest_const"),
            )?;
            step_statement_chain = chunk_step_ivc_digest_chain_fold_circuit(
                &mut cs.namespace(|| format!("relation_{relation_index}_step_statement_chain")),
                &step_statement_chain,
                &statement_digest_const,
                0x7276_3634_7374_6d74,
                &format!("relation_{relation_index}_step_statement_chain"),
            )?;
            bridge_handoff_chain = chunk_step_ivc_digest_chain_fold_circuit(
                &mut cs.namespace(|| format!("relation_{relation_index}_bridge_handoff_chain")),
                &bridge_handoff_chain,
                &bridge_handoff_const,
                0x7276_3634_62686467,
                &format!("relation_{relation_index}_bridge_handoff_chain"),
            )?;
            if let Some(previous_state) = previous_state_out.as_ref() {
                enforce_chunk_step_state_eq(
                    &mut cs.namespace(|| format!("relation_{relation_index}_state_chain")),
                    previous_state,
                    &boundary.state_in,
                    &format!("relation_{relation_index}_state_chain"),
                )?;
            } else {
                enforce_chunk_step_state_eq(
                    &mut cs.namespace(|| "initial_state_chain"),
                    &initial_state,
                    &boundary.state_in,
                    "initial_state_chain",
                )?;
            }
            previous_state_out = Some(boundary.state_out);
        }
        let final_accumulator_state = previous_state_out.as_ref().unwrap_or(&initial_state);
        let final_accumulator_digest = chunk_step_ivc_state_instance_digest_circuit(
            &mut cs.namespace(|| "final_accumulator_digest"),
            final_accumulator_state,
            "final_accumulator_digest",
        )?;
        let final_terminal_handle = final_accumulator_state.terminal_handle.clone();
        enforce_digest_eq(
            &mut cs.namespace(|| "accumulator_instance_output_eq"),
            &accumulator_instance_input,
            &final_accumulator_digest,
            "accumulator_instance_output_eq",
        )?;
        enforce_digest_eq(
            &mut cs.namespace(|| "step_statement_chain_output_eq"),
            &step_statement_chain_input,
            &step_statement_chain,
            "step_statement_chain_output_eq",
        )?;
        enforce_digest_eq(
            &mut cs.namespace(|| "bridge_handoff_chain_output_eq"),
            &bridge_handoff_chain_input,
            &bridge_handoff_chain,
            "bridge_handoff_chain_output_eq",
        )?;
        enforce_digest_eq(
            &mut cs.namespace(|| "terminal_handle_output_eq"),
            &terminal_handle_input,
            &final_terminal_handle,
            "terminal_handle_output_eq",
        )?;
        if public_cursor != public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(())
    }
}

pub fn setup_rv64im_chunk_step_ivc_spartan(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<(Rv64imChunkStepIvcSpartanProverKey, Rv64imChunkStepIvcSpartanVerifierKey), Rv64imChunkStepIvcSpartanError>
{
    let circuit = build_rv64im_chunk_step_ivc_circuit(statement, witness)?;
    Rv64imSpartan2DeciderSnark::setup(circuit).map_err(|err| Rv64imChunkStepIvcSpartanError::Setup(err.to_string()))
}

pub fn build_rv64im_chunk_step_ivc_shape(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<Rv64imChunkStepIvcShape, Rv64imChunkStepIvcSpartanError> {
    let circuit = build_rv64im_chunk_step_ivc_circuit(statement, witness)?;
    Ok(rv64im_chunk_step_ivc_shape(&circuit))
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

pub fn setup_rv64im_chunk_step_ivc_spartan_cached(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<Rv64imChunkStepIvcSpartanKeyPair, Rv64imChunkStepIvcSpartanError> {
    let circuit = build_rv64im_chunk_step_ivc_circuit(statement, witness)?;
    let cache_key = rv64im_chunk_step_ivc_cache_key(&circuit)?;
    let cache = RV64IM_CHUNK_STEP_IVC_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| Rv64imChunkStepIvcSpartanError::Setup("rv64im chunk-step ivc setup cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(keys);
    }
    let keys = Arc::new(
        Rv64imSpartan2DeciderSnark::setup(circuit)
            .map_err(|err| Rv64imChunkStepIvcSpartanError::Setup(err.to_string()))?,
    );
    cache
        .lock()
        .map_err(|_| Rv64imChunkStepIvcSpartanError::Setup("rv64im chunk-step ivc setup cache poisoned".into()))?
        .insert(cache_key, keys.clone());
    Ok(keys)
}

pub fn prove_rv64im_chunk_step_ivc_spartan(
    pk: &Rv64imChunkStepIvcSpartanProverKey,
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<Rv64imChunkStepIvcSpartanProof, Rv64imChunkStepIvcSpartanError> {
    let circuit = build_rv64im_chunk_step_ivc_circuit(statement, witness)?;
    let prep = Rv64imSpartan2DeciderSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Prepare(err.to_string()))?;
    let proof = Rv64imSpartan2DeciderSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Prove(err.to_string()))?;
    let snark_data =
        bincode::serialize(&proof).map_err(|err| Rv64imChunkStepIvcSpartanError::Encode(err.to_string()))?;
    Ok(Rv64imChunkStepIvcSpartanProof { snark_data })
}

pub fn verify_rv64im_chunk_step_ivc_spartan(
    vk: &Rv64imChunkStepIvcSpartanVerifierKey,
    statement: &Rv64imChunkStepIvcStatement,
    proof: &Rv64imChunkStepIvcSpartanProof,
) -> Result<(), Rv64imChunkStepIvcSpartanError> {
    let published_target = build_rv64im_chunk_step_ivc_published_target(statement)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    let proof: Rv64imSpartan2DeciderSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    if public_values != chunk_step_ivc_spartan_public_values(&published_target) {
        return Err(Rv64imChunkStepIvcSpartanError::PublicIoMismatch);
    }
    Ok(())
}

pub fn prove_rv64im_chunk_step_ivc_spartan_chain(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<Rv64imChunkStepIvcSpartanChainProof, Rv64imChunkStepIvcSpartanError> {
    let mut step_proofs = Vec::with_capacity(relations.len());
    for relation in relations {
        let keys = setup_rv64im_chunk_step_ivc_spartan_cached(&relation.statement, &relation.witness)?;
        let (pk, _) = &*keys;
        step_proofs.push(prove_rv64im_chunk_step_ivc_spartan(
            pk,
            &relation.statement,
            &relation.witness,
        )?);
    }
    Ok(Rv64imChunkStepIvcSpartanChainProof { step_proofs })
}

pub fn verify_rv64im_chunk_step_ivc_spartan_chain(
    relations: &[Rv64imChunkStepIvcRelation],
    proof: &Rv64imChunkStepIvcSpartanChainProof,
) -> Result<(), Rv64imChunkStepIvcSpartanError> {
    if relations.len() != proof.step_proofs.len() {
        return Err(Rv64imChunkStepIvcSpartanError::ChainLengthMismatch);
    }
    verify_rv64im_chunk_step_ivc_chain(relations)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    for (relation, step_proof) in relations.iter().zip(proof.step_proofs.iter()) {
        let keys = setup_rv64im_chunk_step_ivc_spartan_cached(&relation.statement, &relation.witness)?;
        let (_, vk) = &*keys;
        verify_rv64im_chunk_step_ivc_spartan(vk, &relation.statement, step_proof)?;
    }
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn prove_rv64im_chunk_step_ivc_spartan_compressed_chain(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<Rv64imChunkStepIvcSpartanCompressedChainProof, Rv64imChunkStepIvcSpartanError> {
    let circuit = build_rv64im_chunk_step_ivc_compressed_chain_circuit(relations)?;
    let keys = setup_rv64im_chunk_step_ivc_spartan_compressed_chain_cached(&circuit)?;
    let (pk, _) = &*keys;
    let prep = Rv64imSpartan2DeciderSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Prepare(err.to_string()))?;
    let proof = Rv64imSpartan2DeciderSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Prove(err.to_string()))?;
    let snark_data =
        bincode::serialize(&proof).map_err(|err| Rv64imChunkStepIvcSpartanError::Encode(err.to_string()))?;
    Ok(Rv64imChunkStepIvcSpartanCompressedChainProof { snark_data })
}

#[allow(dead_code)]
pub(crate) fn verify_rv64im_chunk_step_ivc_spartan_compressed_chain(
    relations: &[Rv64imChunkStepIvcRelation],
    proof: &Rv64imChunkStepIvcSpartanCompressedChainProof,
) -> Result<(), Rv64imChunkStepIvcSpartanError> {
    verify_rv64im_chunk_step_ivc_chain(relations)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    let circuit = build_rv64im_chunk_step_ivc_compressed_chain_circuit(relations)?;
    let keys = setup_rv64im_chunk_step_ivc_spartan_compressed_chain_cached(&circuit)?;
    let (_, vk) = &*keys;
    let proof: Rv64imSpartan2DeciderSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    if public_values != circuit.expected_public_values() {
        return Err(Rv64imChunkStepIvcSpartanError::PublicIoMismatch);
    }
    Ok(())
}

fn build_rv64im_chunk_step_ivc_circuit(
    statement: &Rv64imChunkStepIvcStatement,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<Rv64imChunkStepIvcCircuit, Rv64imChunkStepIvcSpartanError> {
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
    let (params, _, structure) = rv64im_cached_root_main_lane_context()
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Verify(err.to_string()))?;
    let mat_digest_vec = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()));
    let mat_digest = mat_digest_vec
        .try_into()
        .map_err(|_| Rv64imChunkStepIvcSpartanError::Verify("matrix digest length mismatch".into()))?;
    Ok(Rv64imChunkStepIvcCircuit {
        params: params.clone(),
        structure: structure.clone(),
        dims,
        mat_digest,
        published_target,
        witness: witness.clone(),
        cover_chunk: Rv64imMainCircuitChunkCover::from_trace(&chunk_trace),
        effective_chunk: chunk_trace,
    })
}

#[allow(dead_code)]
fn build_rv64im_chunk_step_ivc_compressed_chain_circuit(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<Rv64imChunkStepIvcCompressedChainCircuit, Rv64imChunkStepIvcSpartanError> {
    let step_circuits = relations
        .iter()
        .map(|relation| build_rv64im_chunk_step_ivc_circuit(&relation.statement, &relation.witness))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Rv64imChunkStepIvcCompressedChainCircuit { step_circuits })
}

fn rv64im_chunk_step_ivc_shape(circuit: &Rv64imChunkStepIvcCircuit) -> Rv64imChunkStepIvcShape {
    Rv64imChunkStepIvcShape {
        terminal_step: circuit.witness.terminal_step,
        state_in_claim_count: circuit.witness.state_in.carry.main.claims.len() as u64,
        state_out_claim_count: circuit.witness.state_out.carry.main.claims.len() as u64,
        fresh_claim_count: circuit.effective_chunk.fresh_claims.len() as u64,
        fresh_witness_count: circuit.effective_chunk.fresh_witnesses.len() as u64,
        ccs_output_count: circuit.effective_chunk.ccs_trace.ccs_outputs.len() as u64,
        child_count: circuit.effective_chunk.ccs_trace.children.len() as u64,
        transcript_in_absorbed: circuit.witness.state_in.transcript.absorbed as u64,
        transcript_out_absorbed: circuit.witness.state_out.transcript.absorbed as u64,
        fe_round_lengths: circuit
            .effective_chunk
            .ccs_trace
            .ccs_replay_proof
            .sumcheck_rounds
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
        nc_round_lengths: circuit
            .effective_chunk
            .ccs_trace
            .ccs_replay_proof
            .sumcheck_rounds_nc
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
    }
}

#[allow(dead_code)]
fn rv64im_chunk_step_ivc_compressed_chain_shape_digest(circuit: &Rv64imChunkStepIvcCompressedChainCircuit) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/chunk_step_ivc/compressed_chain_shape");
    tr.append_u64s(
        b"neo.fold.next/rv64im/chunk_step_ivc/compressed_chain_shape/meta",
        &[circuit.step_circuits.len() as u64],
    );
    for step_circuit in &circuit.step_circuits {
        tr.append_message(
            b"neo.fold.next/rv64im/chunk_step_ivc/compressed_chain_shape/step_shape",
            &rv64im_chunk_step_ivc_shape(step_circuit).expected_digest(),
        );
    }
    tr.digest32()
}

#[allow(dead_code)]
fn setup_rv64im_chunk_step_ivc_spartan_compressed_chain_cached(
    circuit: &Rv64imChunkStepIvcCompressedChainCircuit,
) -> Result<Rv64imChunkStepIvcSpartanKeyPair, Rv64imChunkStepIvcSpartanError> {
    let cache_key = rv64im_chunk_step_ivc_compressed_chain_shape_digest(circuit);
    let cache = RV64IM_CHUNK_STEP_IVC_COMPRESSED_CHAIN_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| {
            Rv64imChunkStepIvcSpartanError::Setup("rv64im chunk-step ivc compressed-chain setup cache poisoned".into())
        })?
        .get(&cache_key)
        .cloned()
    {
        return Ok(keys);
    }
    let keys = Arc::new(
        Rv64imSpartan2DeciderSnark::setup(circuit.clone())
            .map_err(|err| Rv64imChunkStepIvcSpartanError::Setup(err.to_string()))?,
    );
    cache
        .lock()
        .map_err(|_| {
            Rv64imChunkStepIvcSpartanError::Setup("rv64im chunk-step ivc compressed-chain setup cache poisoned".into())
        })?
        .insert(cache_key, keys.clone());
    Ok(keys)
}

fn rv64im_chunk_step_ivc_cache_key(
    circuit: &Rv64imChunkStepIvcCircuit,
) -> Result<[u8; 32], Rv64imChunkStepIvcSpartanError> {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key");
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/shape",
        &rv64im_chunk_step_ivc_shape(circuit).expected_digest(),
    );
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/published_target_digest",
        &circuit.published_target.expected_digest(),
    );
    let state_in_bytes = bincode::serialize(&circuit.witness.state_in)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Setup(err.to_string()))?;
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/state_in",
        &state_in_bytes,
    );
    let state_out_bytes = bincode::serialize(&circuit.witness.state_out)
        .map_err(|err| Rv64imChunkStepIvcSpartanError::Setup(err.to_string()))?;
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/state_out",
        &state_out_bytes,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/public_chunk_digest",
        &circuit.witness.handoff.public_chunk_digest,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/bridge_handoff_digest",
        &circuit.witness.handoff.bridge_handoff.digest,
    );
    for digest in &circuit.witness.handoff.prepared_step_digests {
        tr.append_message(
            b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/prepared_step_digest",
            digest,
        );
    }
    tr.append_message(
        b"neo.fold.next/rv64im/chunk_step_ivc/setup_cache_key/replay_witness_digest",
        &rv64im_chunk_replay_witness_digest(&circuit.witness.replay_witness),
    );
    Ok(tr.digest32())
}

fn chunk_step_ivc_spartan_public_values(target: &Rv64imChunkStepIvcPublishedTarget) -> Vec<SpartanF> {
    target
        .public_values()
        .into_iter()
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
        .collect()
}

fn synthesize_chunk_step_ivc_relation_body<CS: ConstraintSystem<SpartanF>>(
    circuit: &Rv64imChunkStepIvcCircuit,
    cs: &mut CS,
    public_inputs: &[AllocatedNum<SpartanF>],
    public_cursor: &mut usize,
) -> Result<Rv64imChunkStepIvcBoundaryVar, SynthesisError> {
    let program_digest_input = next_public_digest(public_inputs, public_cursor, "program_digest")?;
    let chunk_index_input = next_public_u64(public_inputs, public_cursor)?;
    let step_lo_input = next_public_u64(public_inputs, public_cursor)?;
    let step_hi_input = next_public_u64(public_inputs, public_cursor)?;
    let halted_out_input = next_public_u64(public_inputs, public_cursor)?;
    let state_in_input = next_public_digest(public_inputs, public_cursor, "state_in")?;
    let state_out_input = next_public_digest(public_inputs, public_cursor, "state_out")?;
    let summary_start_input = next_public_u64(public_inputs, public_cursor)?;
    let summary_step_count_input = next_public_u64(public_inputs, public_cursor)?;
    let public_chunk_digest_input = next_public_digest(public_inputs, public_cursor, "public_chunk_digest")?;

    let program_digest_const = digest_const_inputs(
        &mut cs.namespace(|| "program_digest_const"),
        circuit.published_target.program_digest,
        "program_digest_const",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "program_digest_eq"),
        &program_digest_input,
        &program_digest_const,
        "program_digest_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "chunk_index_eq"),
        &chunk_index_input,
        circuit.published_target.chunk_index,
        "chunk_index_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "step_lo_eq"),
        &step_lo_input,
        circuit.published_target.step_lo,
        "step_lo_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "step_hi_eq"),
        &step_hi_input,
        circuit.published_target.step_hi,
        "step_hi_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "halted_out_eq"),
        &halted_out_input,
        u64::from(circuit.published_target.halted_out),
        "halted_out_eq",
    )?;
    let state_in_const = digest_const_inputs(
        &mut cs.namespace(|| "state_in_const"),
        circuit.published_target.state_in,
        "state_in_const",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "state_in_eq"),
        &state_in_input,
        &state_in_const,
        "state_in_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "summary_start_eq"),
        &summary_start_input,
        circuit.published_target.summary_start,
        "summary_start_eq",
    )?;
    enforce_u64_input_eq(
        &mut cs.namespace(|| "summary_step_count_eq"),
        &summary_step_count_input,
        circuit.published_target.summary_step_count,
        "summary_step_count_eq",
    )?;
    let public_chunk_digest_const = digest_const_inputs(
        &mut cs.namespace(|| "public_chunk_digest_const"),
        circuit.published_target.public_chunk_digest,
        "public_chunk_digest_const",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "public_chunk_digest_eq"),
        &public_chunk_digest_input,
        &public_chunk_digest_const,
        "public_chunk_digest_eq",
    )?;

    let transcript_in_fields = alloc_private_transcript_state(&mut cs.namespace(|| "transcript_in"), &circuit.witness)?;
    let transcript_in_values = circuit
        .witness
        .state_in
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut transcript = Poseidon2TranscriptCircuit::from_state(
        transcript_in_fields.clone(),
        transcript_in_values,
        circuit.witness.state_in.transcript.absorbed,
    )?;
    let carried_claims = alloc_state_in_claims(
        &mut cs.namespace(|| "state_in_claims"),
        &circuit.witness.state_in.carry.main.claims,
    )?;
    let state_in = Rv64imChunkStepIvcStateVar {
        claims: carried_claims.clone(),
        transcript_state: transcript_in_fields,
        transcript_absorbed: circuit.witness.state_in.transcript.absorbed,
        terminal_handle: state_in_input.clone(),
    };

    let replay_chunk = circuit
        .effective_chunk
        .replay_surface()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let next_claims = synthesize_rv64im_main_relation_chunk(
        &circuit.params,
        &circuit.structure,
        circuit.dims,
        &circuit.mat_digest,
        &circuit.witness.state_out.carry.main.claims,
        &mut cs.namespace(|| "chunk_step"),
        circuit.witness.handoff.bridge_handoff.chunk_index as usize,
        &circuit.cover_chunk,
        &replay_chunk,
        public_inputs,
        public_cursor,
        &mut transcript,
        Rv64imClaimBundle::from_effective_claims(carried_claims),
        Rv64imChunkBoundaryPlan::from_boundary_mode(
            Rv64imChunkBoundaryMode::from_terminal_flags(circuit.witness.terminal_step, false),
            circuit.effective_chunk.fresh_claims.len(),
            circuit.effective_chunk.ccs_trace.ccs_outputs.len(),
        ),
        true,
        false,
    )?;

    if next_claims.effective_count() != circuit.witness.state_out.carry.main.claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (claim_index, (actual, expected)) in next_claims
        .effective_claims()
        .iter()
        .zip(circuit.witness.state_out.carry.main.claims.iter())
        .enumerate()
    {
        enforce_claim_eq_native(
            &mut cs.namespace(|| format!("state_out_claim_{claim_index}")),
            actual,
            expected,
            &format!("state_out_claim_{claim_index}"),
        )?;
    }

    transcript.append_const_fields_raw(
        cs.namespace(|| "chunk_done"),
        &[
            SpartanF::from_canonical_u64(RV64IM_CHUNK_DONE_RAW_TAG),
            SpartanF::from_canonical_u64(1),
        ],
    )?;
    let transcript_out = transcript.state_fields(cs.namespace(|| "carried_transcript_out"))?;
    for (lane_index, (actual, expected)) in transcript_out
        .iter()
        .zip(circuit.witness.state_out.transcript.state.iter())
        .enumerate()
    {
        let expected = SpartanF::from_canonical_u64(expected.as_canonical_u64());
        cs.enforce(
            || format!("transcript_out_lane_{lane_index}"),
            |lc| lc + actual.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (expected, CS::one()),
        );
    }
    if transcript.absorbed() != circuit.witness.state_out.transcript.absorbed {
        return Err(SynthesisError::Unsatisfiable);
    }

    let state_out_handle = chunk_step_handle_circuit(
        &mut cs.namespace(|| "state_out_handle"),
        &state_in_input,
        circuit.witness.handoff.bridge_handoff.chunk_index as u64,
        circuit.effective_chunk.handoff.public_chunk.start_index as u64,
        circuit.effective_chunk.handoff.public_chunk.steps.len() as u64,
        circuit.effective_chunk.handoff.chunk_relation_digest,
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "state_out_eq"),
        &state_out_handle,
        &state_out_input,
        "state_out_eq",
    )?;

    Ok(Rv64imChunkStepIvcBoundaryVar {
        state_in,
        state_out: Rv64imChunkStepIvcStateVar {
            claims: next_claims.into_effective_claims(),
            transcript_state: transcript_out,
            transcript_absorbed: transcript.absorbed(),
            terminal_handle: state_out_handle,
        },
    })
}

#[allow(dead_code)]
pub(crate) fn alloc_native_chunk_step_state<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    state: &crate::rv64im::final_relation::Rv64imChunkFoldState,
    label: &str,
) -> Result<Rv64imChunkStepIvcStateVar, SynthesisError> {
    let claims = alloc_state_in_claims(
        &mut cs.namespace(|| format!("{label}_claims")),
        &state.carry.main.claims,
    )?;
    let transcript_state = alloc_const_field_values(
        &mut cs.namespace(|| format!("{label}_transcript_state")),
        &state
            .transcript
            .state
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
        &format!("{label}_transcript_state"),
    )?
    .try_into()
    .map_err(|_| SynthesisError::Unsatisfiable)?;
    let terminal_handle = digest_const_inputs(
        &mut cs.namespace(|| format!("{label}_terminal_handle")),
        state.carry.terminal_handle.0,
        label,
    )?;
    Ok(Rv64imChunkStepIvcStateVar {
        claims,
        transcript_state,
        transcript_absorbed: state.transcript.absorbed,
        terminal_handle,
    })
}

#[allow(dead_code)]
pub(crate) fn chunk_step_ivc_transcript_snapshot_digest_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    state: &[AllocatedNum<SpartanF>; neo_params::poseidon2_goldilocks::WIDTH],
    absorbed: usize,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = alloc_const_field_values(
        &mut cs.namespace(|| format!("{label}_domain")),
        &packed_bytes_field_values(b"neo.fold.next/rv64im/main_recursion_transcript_snapshot/v2"),
        &format!("{label}_domain"),
    )?;
    preimage.extend(alloc_const_field_values(
        &mut cs.namespace(|| format!("{label}_absorbed")),
        &[SpartanF::from_canonical_u64(absorbed as u64)],
        &format!("{label}_absorbed"),
    )?);
    preimage.extend(state.iter().cloned());
    hash_packed_goldilocks_fields(cs.namespace(|| format!("{label}_hash")), &preimage)
}

#[allow(dead_code)]
pub(crate) fn chunk_step_ivc_state_instance_digest_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    state: &Rv64imChunkStepIvcStateVar,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = alloc_const_field_values(
        &mut cs.namespace(|| format!("{label}_domain")),
        &packed_bytes_field_values(b"neo.fold.next/rv64im/main_recursion_accumulator_instance/v2"),
        &format!("{label}_domain"),
    )?;
    preimage.extend(alloc_const_field_values(
        &mut cs.namespace(|| format!("{label}_claim_count")),
        &[SpartanF::from_canonical_u64(state.claims.len() as u64)],
        &format!("{label}_claim_count"),
    )?);
    for (claim_index, claim) in state.claims.iter().enumerate() {
        let claim_digest = me_digest_poseidon(
            &mut cs.namespace(|| format!("{label}_claim_digest_{claim_index}")),
            claim,
            &format!("{label}_claim_digest_{claim_index}"),
        )?;
        preimage.extend(claim_digest.iter().cloned());
    }
    let transcript_digest = chunk_step_ivc_transcript_snapshot_digest_circuit(
        &mut cs.namespace(|| format!("{label}_transcript_digest")),
        &state.transcript_state,
        state.transcript_absorbed,
        &format!("{label}_transcript_digest"),
    )?;
    preimage.extend(transcript_digest.iter().cloned());
    preimage.extend(state.terminal_handle.iter().cloned());
    hash_packed_goldilocks_fields(cs.namespace(|| format!("{label}_hash")), &preimage)
}

#[allow(dead_code)]
fn enforce_chunk_step_state_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    expected: &Rv64imChunkStepIvcStateVar,
    actual: &Rv64imChunkStepIvcStateVar,
    label: &str,
) -> Result<(), SynthesisError> {
    if expected.claims.len() != actual.claims.len() || expected.transcript_absorbed != actual.transcript_absorbed {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (claim_index, (expected_claim, actual_claim)) in expected.claims.iter().zip(actual.claims.iter()).enumerate() {
        let expected_digest = me_digest_poseidon(
            &mut cs.namespace(|| format!("{label}_expected_claim_digest_{claim_index}")),
            expected_claim,
            &format!("{label}_expected_claim_digest_{claim_index}"),
        )?;
        let actual_digest = me_digest_poseidon(
            &mut cs.namespace(|| format!("{label}_actual_claim_digest_{claim_index}")),
            actual_claim,
            &format!("{label}_actual_claim_digest_{claim_index}"),
        )?;
        enforce_digest_eq(
            &mut cs.namespace(|| format!("{label}_claim_eq_{claim_index}")),
            &actual_digest,
            &expected_digest,
            &format!("{label}_claim_eq_{claim_index}"),
        )?;
    }
    for (lane_index, (expected_lane, actual_lane)) in expected
        .transcript_state
        .iter()
        .zip(actual.transcript_state.iter())
        .enumerate()
    {
        cs.enforce(
            || format!("{label}_transcript_lane_eq_{lane_index}"),
            |lc| lc + actual_lane.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + expected_lane.get_variable(),
        );
    }
    enforce_digest_eq(
        &mut cs.namespace(|| format!("{label}_terminal_handle_eq")),
        &actual.terminal_handle,
        &expected.terminal_handle,
        &format!("{label}_terminal_handle_eq"),
    )?;
    Ok(())
}

fn alloc_private_transcript_state<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &Rv64imChunkStepIvcWitness,
) -> Result<[AllocatedNum<SpartanF>; neo_params::poseidon2_goldilocks::WIDTH], SynthesisError> {
    let mut out = Vec::with_capacity(neo_params::poseidon2_goldilocks::WIDTH);
    for (lane_index, lane) in witness.state_in.transcript.state.iter().enumerate() {
        out.push(AllocatedNum::alloc(
            cs.namespace(|| format!("transcript_lane_{lane_index}")),
            || Ok(SpartanF::from_canonical_u64(lane.as_canonical_u64())),
        )?);
    }
    out.try_into().map_err(|_| SynthesisError::Unsatisfiable)
}

pub(crate) fn alloc_state_in_claims<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, neo_math::K>],
) -> Result<Vec<CeClaimVar>, SynthesisError> {
    let Some((first, rest)) = claims.split_first() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(claims.len());
    let first_var = alloc_ce_claim(&mut cs.namespace(|| "claim_0"), first, "claim_0")?;
    let shared_r = first_var.r.clone();
    let shared_r_values = first_var.r_values.clone();
    let shared_s_col = first_var.s_col.clone();
    let shared_s_col_values = first_var.s_col_values.clone();
    out.push(first_var);
    for (idx, claim) in rest.iter().enumerate() {
        out.push(alloc_ce_claim_with_shared_point(
            &mut cs.namespace(|| format!("claim_{}", idx + 1)),
            claim,
            &shared_r,
            &shared_r_values,
            &shared_s_col,
            &shared_s_col_values,
            &format!("claim_{}", idx + 1),
        )?);
    }
    Ok(out)
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

pub(crate) fn enforce_u64_input_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &AllocatedNum<SpartanF>,
    expected: u64,
    label: &str,
) -> Result<(), SynthesisError> {
    let expected = SpartanF::from_canonical_u64(expected);
    cs.enforce(
        || label,
        |lc| lc + actual.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (expected, CS::one()),
    );
    Ok(())
}

pub(crate) fn next_public_u64(
    public_inputs: &[AllocatedNum<SpartanF>],
    cursor: &mut usize,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    if *cursor >= public_inputs.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let out = public_inputs[*cursor].clone();
    *cursor += 1;
    Ok(out)
}

#[allow(dead_code)]
fn chunk_step_ivc_digest_chain_seed_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    raw_tag: u64,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let preimage = alloc_const_field_values(
        &mut cs.namespace(|| format!("{label}_tag")),
        &[SpartanF::from_canonical_u64(raw_tag)],
        &format!("{label}_tag"),
    )?;
    hash_packed_goldilocks_fields(cs.namespace(|| format!("{label}_hash")), &preimage)
}

pub(crate) fn chunk_step_ivc_digest_chain_fold_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    current: &[AllocatedNum<SpartanF>; 4],
    item: &[AllocatedNum<SpartanF>; 4],
    raw_tag: u64,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = alloc_const_field_values(
        &mut cs.namespace(|| format!("{label}_tag")),
        &[SpartanF::from_canonical_u64(raw_tag)],
        &format!("{label}_tag"),
    )?;
    preimage.extend(current.iter().cloned());
    preimage.extend(item.iter().cloned());
    hash_packed_goldilocks_fields(cs.namespace(|| format!("{label}_hash")), &preimage)
}

pub(crate) fn chunk_step_handle_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    previous_handle_digest: &[AllocatedNum<SpartanF>; 4],
    chunk_index: u64,
    chunk_start_index: u64,
    public_step_count: u64,
    chunk_relation_digest: [u8; 32],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = Vec::new();
    preimage.extend(previous_handle_digest.iter().cloned());
    preimage.extend(alloc_const_field_values(
        &mut cs.namespace(|| "chunk_step_meta"),
        &[
            SpartanF::from_canonical_u64(chunk_index),
            SpartanF::from_canonical_u64(chunk_start_index),
            SpartanF::from_canonical_u64(public_step_count),
        ],
        "chunk_step_meta",
    )?);
    preimage.extend(alloc_const_field_values(
        &mut cs.namespace(|| "chunk_step_digest"),
        &packed_bytes_field_values(&chunk_relation_digest),
        "chunk_step_digest",
    )?);
    hash_packed_goldilocks_fields(cs.namespace(|| "chunk_step_handle_hash"), &preimage)
}
