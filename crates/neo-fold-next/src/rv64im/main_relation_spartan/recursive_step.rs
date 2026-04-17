//! Owns the first fixed-step Spartan backend for RV64IM main recursion.
//!
//! This circuit proves one carried `U_i -> U_{i+1}` public transition with fixed
//! public IO over the canonical padded carried-state cover. It is not yet the
//! full recursive owner because it does not consume the fresh padded chunk
//! payload inside the constraint body.

mod authoritative_surface;
mod chunk_replay;
mod compressed_chain;
mod diagnostics;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use bellpepper_core::{
    num::AllocatedNum, test_cs::TestConstraintSystem, Comparable, ConstraintSystem, Delta, SynthesisError,
};
use neo_math::F;
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices_with_sparse_cache};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Serialize};
use spartan2::{
    bellpepper::{r1cs::SpartanShape, shape_cs::ShapeCS},
    provider::goldi::F as SpartanF,
    traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use thiserror::Error;

use super::chunk_step_ivc::{chunk_step_ivc_digest_chain_fold_circuit, digest_const_inputs, next_public_u64};
use super::chunk_step_recursive::{
    build_rv64im_main_recursion_step_spartan_statement as build_rv64im_main_recursion_step_spartan_statement_from_payload,
    rv64im_chunk_step_recursive_carry_state_digest, Rv64imMainRecursionFPrimeBackendRelation,
    Rv64imMainRecursionStepSpartanShape,
};
use super::recursive_cover::alloc_recursive_cover_state;
use super::{
    alloc_const_field_values, alloc_private_field_values, digest32_as_spartan_fields, enforce_digest_eq,
    next_public_digest, Rv64imMainRecursionStepSpartanStatement, Rv64imSpartan2DeciderEngine,
    Rv64imSpartan2DeciderKeyPair, Rv64imSpartan2DeciderProverKey, Rv64imSpartan2DeciderSnark,
    Rv64imSpartan2DeciderVerifierKey,
};
use crate::finalize::{digest_fields_as_digest32, FixedShapeChunkSummary};
use crate::proof::{Carry, ChunkInput, StepInput};
use crate::rv64im::chunk_fold_step::{Rv64imAccumulatorHandle, Rv64imChunkFoldCarry};
use crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcRelation;
use crate::rv64im::construction2::build_rv64im_main_recursion_construction2_verified_step_statement_from_relation;
use crate::rv64im::final_relation::{Rv64imChunkFoldState, Rv64imChunkFoldTranscriptSnapshot};
use crate::rv64im::kernel::{rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache};
use crate::rv64im::kernel::{
    Rv64imChunkBridgeHandoff, Rv64imPreparedStepBridgeBinding, Rv64imVerifiedKernelChunkHandoff,
};
use crate::rv64im::main_recursion::{
    build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs, build_rv64im_main_recursion_verifier_key_fs,
    Rv64imEncodedPublicInput,
};
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::rv64im::main_relation_spartan::chunk_step_ivc::Rv64imChunkStepIvcShape;
use crate::rv64im::main_relation_spartan::chunk_step_recursive::build_rv64im_main_recursion_f_prime_payload;
use chunk_replay::synthesize_rv64im_main_recursion_step_chunk_replay;

pub type Rv64imMainRecursionStepSpartanProverKey = Rv64imSpartan2DeciderProverKey;
pub type Rv64imMainRecursionStepSpartanVerifierKey = Rv64imSpartan2DeciderVerifierKey;
pub type Rv64imMainRecursionStepSpartanKeyPair = Rv64imSpartan2DeciderKeyPair;

pub use authoritative_surface::{
    build_rv64im_main_recursion_step_authoritative_chunk_surface,
    debug_check_rv64im_main_recursion_step_authoritative_chunk_surface_matches_native,
    Rv64imMainRecursionStepAuthoritativeChunkSurface,
};
pub use compressed_chain::{
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_circuit,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_parity,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_public_io,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_shape_only_circuit,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_shape_only_setup,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_statement_binding,
    debug_check_rv64im_main_recursion_step_spartan_shape_only_chain_parity,
    debug_measure_rv64im_main_recursion_step_spartan_compressed_chain_circuit_shape,
    debug_profile_rv64im_main_recursion_step_spartan_compressed_chain_prove_stages,
    prove_rv64im_main_recursion_step_spartan_compressed_chain,
    verify_rv64im_main_recursion_step_spartan_compressed_chain,
    Rv64imMainRecursionStepSpartanCompressedChainProveMetrics,
};
pub use diagnostics::{
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_wrapper_only,
    debug_check_rv64im_main_recursion_step_spartan_live_claim_me_digest_parity,
    debug_measure_rv64im_main_recursion_step_spartan_commitment_key,
    debug_measure_rv64im_main_recursion_step_spartan_shape_synthesis,
    debug_profile_rv64im_main_recursion_step_chunk_replay_stages,
};

static RV64IM_MAIN_RECURSION_STEP_SETUP_CACHE: OnceLock<
    Mutex<HashMap<[u8; 32], Rv64imMainRecursionStepSpartanKeyPair>>,
> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionStepSpartanProof {
    pub snark_data: Vec<u8>,
}

pub type Rv64imMainRecursionStepSpartanChainProof = Vec<Rv64imMainRecursionStepSpartanProof>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionStepSpartanCompressedChainProof {
    pub snark_data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionStepSpartanCompressedChainShape {
    pub spartan_shape: Rv64imMainRecursionStepSpartanShape,
    pub step_shapes: Vec<Rv64imChunkStepIvcShape>,
}

impl Rv64imMainRecursionStepSpartanCompressedChainShape {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr =
            Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_step_spartan/compressed_chain_shape");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_spartan/compressed_chain_shape/version",
            b"v1",
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_spartan/compressed_chain_shape/spartan_shape",
            &self.spartan_shape.expected_digest(),
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_step_spartan/compressed_chain_shape/step_count",
            &[self.step_shapes.len() as u64],
        );
        for step_shape in &self.step_shapes {
            tr.append_message(
                b"neo.fold.next/rv64im/main_recursion_step_spartan/compressed_chain_shape/step_shape",
                &step_shape.expected_digest(),
            );
        }
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionStepSpartanCircuitShape {
    pub num_inputs: usize,
    pub num_aux: usize,
    pub num_constraints: usize,
    pub constraint_fingerprint: String,
}

#[derive(Debug, Error)]
pub enum Rv64imMainRecursionStepSpartanError {
    #[error("rv64im main recursion step setup failed: {0}")]
    Setup(String),
    #[error("rv64im main recursion step prepare failed: {0}")]
    Prepare(String),
    #[error("rv64im main recursion step prove failed: {0}")]
    Prove(String),
    #[error("rv64im main recursion step verify failed: {0}")]
    Verify(String),
    #[error("rv64im main recursion step proof encoding failed: {0}")]
    Encode(String),
    #[error("rv64im main recursion step proof decoding failed: {0}")]
    Decode(String),
    #[error("rv64im main recursion step public IO mismatch")]
    PublicIoMismatch,
}

#[derive(Clone)]
struct Rv64imMainRecursionStepCircuit {
    spartan_shape: Rv64imMainRecursionStepSpartanShape,
    backend_relation: Rv64imMainRecursionFPrimeBackendRelation,
}

#[derive(Clone)]
struct Rv64imMainRecursionStepPublicVar {
    chunk_index: AllocatedNum<SpartanF>,
    carry_state_in_digest: [AllocatedNum<SpartanF>; 4],
    step_statement_chain_digest_in: [AllocatedNum<SpartanF>; 4],
    step_statement_chain_digest_out: [AllocatedNum<SpartanF>; 4],
    bridge_handoff_chain_digest_in: [AllocatedNum<SpartanF>; 4],
    bridge_handoff_chain_digest_out: [AllocatedNum<SpartanF>; 4],
    carry_state_out_digest: [AllocatedNum<SpartanF>; 4],
    folded_accumulator_out_digest: [AllocatedNum<SpartanF>; 4],
    terminal_handle_digest_out: [AllocatedNum<SpartanF>; 4],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionStepSpartanPublishedTarget {
    pub chunk_index: u64,
    pub carry_state_in_digest: [u8; 32],
    pub folded_accumulator_in_digest: [u8; 32],
    pub step_statement_chain_digest_in: [u8; 32],
    pub bridge_handoff_chain_digest_in: [u8; 32],
    pub carry_state_out_digest: [u8; 32],
    pub x_out: Rv64imEncodedPublicInput,
    pub folded_accumulator_out_digest: [u8; 32],
    pub step_statement_chain_digest: [u8; 32],
    pub bridge_handoff_chain_digest: [u8; 32],
    pub terminal_handle_digest: [u8; 32],
}

impl Rv64imMainRecursionStepSpartanPublishedTarget {
    const PUBLIC_VALUE_ARITY: usize = 41;

    pub fn public_values(&self) -> Vec<SpartanF> {
        let mut values = Vec::with_capacity(Self::PUBLIC_VALUE_ARITY);
        values.push(SpartanF::from_canonical_u64(self.chunk_index));
        values.extend(digest32_as_spartan_fields(self.carry_state_in_digest));
        values.extend(digest32_as_spartan_fields(self.folded_accumulator_in_digest));
        values.extend(digest32_as_spartan_fields(self.step_statement_chain_digest_in));
        values.extend(digest32_as_spartan_fields(self.bridge_handoff_chain_digest_in));
        values.extend(digest32_as_spartan_fields(self.carry_state_out_digest));
        values.extend(digest32_as_spartan_fields(self.x_out.bytes()));
        values.extend(digest32_as_spartan_fields(self.folded_accumulator_out_digest));
        values.extend(digest32_as_spartan_fields(self.step_statement_chain_digest));
        values.extend(digest32_as_spartan_fields(self.bridge_handoff_chain_digest));
        values.extend(digest32_as_spartan_fields(self.terminal_handle_digest));
        values
    }

    pub fn from_public_values(public_values: &[SpartanF]) -> Result<Self, Rv64imMainRecursionStepSpartanError> {
        if public_values.len() != Self::PUBLIC_VALUE_ARITY {
            return Err(Rv64imMainRecursionStepSpartanError::Verify(format!(
                "rv64im main recursion step proof public IO arity mismatch: expected {}, got {}",
                Self::PUBLIC_VALUE_ARITY,
                public_values.len()
            )));
        }

        let mut cursor = 1usize;
        let mut next_digest = || -> Result<[u8; 32], Rv64imMainRecursionStepSpartanError> {
            let digest = spartan_public_digest32(&public_values[cursor..cursor + 4])?;
            cursor += 4;
            Ok(digest)
        };

        Ok(Self {
            chunk_index: public_values[0].to_canonical_u64(),
            carry_state_in_digest: next_digest()?,
            folded_accumulator_in_digest: next_digest()?,
            step_statement_chain_digest_in: next_digest()?,
            bridge_handoff_chain_digest_in: next_digest()?,
            carry_state_out_digest: next_digest()?,
            x_out: Rv64imEncodedPublicInput::from_digest_bytes(next_digest()?),
            folded_accumulator_out_digest: next_digest()?,
            step_statement_chain_digest: next_digest()?,
            bridge_handoff_chain_digest: next_digest()?,
            terminal_handle_digest: next_digest()?,
        })
    }

    pub fn output_statement(&self) -> Rv64imMainRecursionStepSpartanStatement {
        Rv64imMainRecursionStepSpartanStatement {
            x_out: self.x_out.clone(),
            folded_accumulator_digest: self.folded_accumulator_out_digest,
            step_statement_chain_digest: self.step_statement_chain_digest,
            bridge_handoff_chain_digest: self.bridge_handoff_chain_digest,
            terminal_handle_digest: self.terminal_handle_digest,
        }
    }
}

fn spartan_public_digest32(public_values: &[SpartanF]) -> Result<[u8; 32], Rv64imMainRecursionStepSpartanError> {
    if public_values.len() != 4 {
        return Err(Rv64imMainRecursionStepSpartanError::Verify(format!(
            "rv64im main recursion step digest decode arity mismatch: expected 4, got {}",
            public_values.len()
        )));
    }
    Ok(digest_fields_as_digest32(core::array::from_fn(|idx| {
        F::from_u64(public_values[idx].to_canonical_u64())
    })))
}

fn main_recursion_step_public_values(statement: &Rv64imMainRecursionStepSpartanStatement) -> Vec<SpartanF> {
    let mut values = Vec::with_capacity(20);
    values.extend(digest32_as_spartan_fields(statement.x_out.bytes()));
    values.extend(digest32_as_spartan_fields(statement.folded_accumulator_digest));
    values.extend(digest32_as_spartan_fields(statement.step_statement_chain_digest));
    values.extend(digest32_as_spartan_fields(statement.bridge_handoff_chain_digest));
    values.extend(digest32_as_spartan_fields(statement.terminal_handle_digest));
    values
}

fn mark_unsatisfied<CS: ConstraintSystem<SpartanF>>(cs: &mut CS, label: &str) -> Result<(), SynthesisError> {
    cs.enforce(|| label, |lc| lc + CS::one(), |lc| lc + CS::one(), |lc| lc);
    Ok(())
}

fn rv64im_main_recursion_step_setup_cache_key(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
) -> Result<[u8; 32], Rv64imMainRecursionStepSpartanError> {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_step_spartan/setup_cache_key");
    tr.append_message(
        b"neo.fold.next/rv64im/main_recursion_step_spartan/setup_cache_key/version",
        b"v4",
    );
    tr.append_message(
        b"neo.fold.next/rv64im/main_recursion_step_spartan/setup_cache_key/spartan_shape",
        &spartan_shape.expected_digest(),
    );
    Ok(tr.digest32())
}

impl Rv64imMainRecursionStepCircuit {
    fn expected_public_values(&self) -> Vec<SpartanF> {
        build_rv64im_main_recursion_step_spartan_published_target(&self.backend_relation)
            .expect("recursive-step circuit must be built from a canonical backend relation")
            .public_values()
    }
}

fn initial_main_recursion_step_spartan_statement(
) -> Result<Rv64imMainRecursionStepSpartanStatement, Rv64imMainRecursionStepSpartanError> {
    let initial_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state();
    let folded_accumulator_digest =
        crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(&initial_state.carry);
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs()
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    Ok(build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
        &vk_fs,
        0,
        folded_accumulator_digest,
        crate::rv64im::chunk_step_ivc::rv64im_step_statement_chain_digest_init(),
        crate::rv64im::chunk_step_ivc::rv64im_bridge_handoff_chain_digest_init(),
        initial_state.carry.terminal_handle.0,
    ))
}

fn build_rv64im_main_recursion_step_spartan_statement(
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<Rv64imMainRecursionStepSpartanStatement, Rv64imMainRecursionStepSpartanError> {
    backend_relations
        .last()
        .map(|relation| Ok(relation.spartan_statement.clone()))
        .unwrap_or_else(initial_main_recursion_step_spartan_statement)
}

fn canonical_main_recursion_step_spartan_statement(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepSpartanStatement, Rv64imMainRecursionStepSpartanError> {
    build_rv64im_main_recursion_step_spartan_statement_from_payload(&backend_relation.f_prime_advice)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))
}

pub fn build_rv64im_main_recursion_step_spartan_published_target(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepSpartanPublishedTarget, Rv64imMainRecursionStepSpartanError> {
    let canonical_statement = canonical_main_recursion_step_spartan_statement(backend_relation)?;
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    Ok(Rv64imMainRecursionStepSpartanPublishedTarget {
        chunk_index: witness.chunk_index(),
        carry_state_in_digest: rv64im_chunk_step_recursive_carry_state_digest(
            &payload.state_in_claims,
            &witness.running_state().transcript,
            witness.running_state().carry.terminal_handle.0,
        ),
        folded_accumulator_in_digest:
            crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(
                &witness.running_state().carry,
            ),
        step_statement_chain_digest_in: witness.step_statement_chain_digest_in(),
        bridge_handoff_chain_digest_in: witness.bridge_handoff_chain_digest_in(),
        carry_state_out_digest: rv64im_chunk_step_recursive_carry_state_digest(
            &payload.state_out_claims,
            payload.fixed_transcript_out(),
            witness.fresh_state_out().carry.terminal_handle.0,
        ),
        x_out: canonical_statement.x_out,
        folded_accumulator_out_digest: canonical_statement.folded_accumulator_digest,
        step_statement_chain_digest: canonical_statement.step_statement_chain_digest,
        bridge_handoff_chain_digest: canonical_statement.bridge_handoff_chain_digest,
        terminal_handle_digest: canonical_statement.terminal_handle_digest,
    })
}

fn ensure_main_recursion_step_spartan_statement_binding(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let canonical_statement = canonical_main_recursion_step_spartan_statement(backend_relation)?;
    if backend_relation.spartan_statement != canonical_statement {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step circuit requires the canonical per-step Spartan statement derived from native F'"
                .into(),
        ));
    }
    Ok(())
}

pub fn build_rv64im_main_recursion_step_spartan_compressed_chain_shape(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<Rv64imMainRecursionStepSpartanCompressedChainShape, Rv64imMainRecursionStepSpartanError> {
    let step_shapes = collect_main_recursion_step_chain_shapes(spartan_shape, backend_relations)?;
    Ok(Rv64imMainRecursionStepSpartanCompressedChainShape {
        spartan_shape: spartan_shape.clone(),
        step_shapes,
    })
}

pub fn validate_rv64im_main_recursion_step_spartan_chain_shape(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    ensure_main_recursion_step_chain_matches_shape(spartan_shape, backend_relations)?;
    Ok(())
}

fn collect_main_recursion_step_chain_shapes(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<Vec<Rv64imChunkStepIvcShape>, Rv64imMainRecursionStepSpartanError> {
    let mut step_shapes = Vec::with_capacity(backend_relations.len());
    for relation in backend_relations {
        if !spartan_shape.matches_payload(&relation.payload) {
            return Err(Rv64imMainRecursionStepSpartanError::Prepare(
                "rv64im main recursion step chain shape requires payloads matching the explicit Spartan shape".into(),
            ));
        }
        step_shapes.push(relation.payload.step_shape.clone());
    }
    Ok(step_shapes)
}

fn ensure_main_recursion_step_chain_matches_shape(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    for relation in backend_relations {
        if !spartan_shape.matches_payload(&relation.payload) {
            return Err(Rv64imMainRecursionStepSpartanError::Prepare(
                "rv64im main recursion step chain shape requires payloads matching the explicit Spartan shape".into(),
            ));
        }
    }
    Ok(())
}

fn zero_step_inputs(
    fresh_claim_shapes: &[crate::rv64im::main_relation_spartan::chunk_step_recursive::Rv64imCcsClaimShape],
    fresh_witness_shapes: &[crate::rv64im::main_relation_spartan::chunk_step_recursive::Rv64imCcsWitnessShape],
    live_len: usize,
) -> Vec<StepInput> {
    (0..live_len)
        .map(|idx| StepInput {
            label: format!("dummy-step-{idx}"),
            mcs: fresh_claim_shapes[idx].zero_claim(),
            witness: fresh_witness_shapes[idx].zero_witness(),
        })
        .collect()
}

fn dummy_backend_relation_from_chain_step(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    step_shape: &Rv64imChunkStepIvcShape,
    chunk_count_in: u64,
    step_statement_chain_digest_in: [u8; 32],
    bridge_handoff_chain_digest_in: [u8; 32],
    running_state: &Rv64imChunkFoldState,
) -> Result<Rv64imMainRecursionFPrimeBackendRelation, Rv64imMainRecursionStepSpartanError> {
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs()
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let public_chunk_input = ChunkInput {
        start_index: 0,
        steps: zero_step_inputs(
            &spartan_shape.claim_cover.fresh_claim_shapes,
            &spartan_shape.claim_cover.fresh_witness_shapes,
            step_shape.fresh_claim_count as usize,
        ),
    };
    let public_chunk = public_chunk_input.public();
    let prepared_step_digests = crate::rv64im::kernel::prepared_step_digests(&public_chunk_input.steps);
    let mut handoff = Rv64imVerifiedKernelChunkHandoff {
        chunk_input: public_chunk_input,
        public_chunk,
        public_chunk_digest: [0; 32],
        public_chunk_instance_digest: [F::ZERO; 4],
        prepared_step_digests: prepared_step_digests.clone(),
        bridge_handoff: Rv64imChunkBridgeHandoff {
            chunk_index: chunk_count_in,
            chunk_start_index: 0,
            public_step_count: step_shape.fresh_claim_count,
            step_bindings: prepared_step_digests
                .iter()
                .enumerate()
                .map(|(idx, prepared_step_digest)| {
                    let binding = Rv64imPreparedStepBridgeBinding {
                        logical_index: idx as u64,
                        trace_index: idx as u64,
                        row_binding_digest: [0; 32],
                        prepared_step_digest: *prepared_step_digest,
                        digest: [0; 32],
                    };
                    Rv64imPreparedStepBridgeBinding {
                        digest: binding.expected_digest(),
                        ..binding
                    }
                })
                .collect(),
            digest: [0; 32],
        },
    };
    handoff.public_chunk_instance_digest = crate::finalize::public_chunk_digest(&handoff.public_chunk);
    handoff.public_chunk_digest = crate::rv64im::kernel::rv64im_public_chunk_digest(&handoff.public_chunk);
    handoff.bridge_handoff.digest = handoff.bridge_handoff.expected_digest();
    let (params, log, structure) = rv64im_cached_root_main_lane_context()
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let mut prove_transcript =
        Poseidon2Transcript::from_state_and_absorbed(running_state.transcript.state, running_state.transcript.absorbed);
    let ((replay_witness, _next_main, public_chunk_digest, chunk_relation_digest), _) =
        crate::rv64im::chunk_relation::prove_rv64im_chunk_transition_with_perf(
            chunk_count_in as usize,
            &handoff,
            &running_state.carry.main,
            &mut prove_transcript,
            params,
            structure,
            log,
            optimized_cache,
        )
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let mut trace_transcript =
        Poseidon2Transcript::from_state_and_absorbed(running_state.transcript.state, running_state.transcript.absorbed);
    let trace = crate::rv64im::chunk_relation::trace_rv64im_chunk_relation_with_replay(
        chunk_count_in as usize,
        &handoff,
        &running_state.carry.main,
        &replay_witness,
        &mut trace_transcript,
        params,
        structure,
        log,
        optimized_cache,
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let next_carry = Rv64imChunkFoldCarry {
        main: Carry {
            claims: trace.children.clone(),
            witnesses: trace.z_split.clone(),
        },
        terminal_handle: Rv64imAccumulatorHandle(crate::rv64im::chunk_relation::rv64im_step_handle(
            running_state.carry.terminal_handle.0,
            chunk_count_in as usize,
            handoff.public_chunk.start_index,
            handoff.public_chunk.steps.len(),
            chunk_relation_digest,
        )),
    };
    let transcript_out = crate::rv64im::final_relation::rv64im_chunk_fold_carried_transcript_snapshot(
        &Rv64imChunkFoldTranscriptSnapshot {
            state: trace_transcript.state(),
            absorbed: trace_transcript.absorbed(),
        },
    );
    let fresh = crate::rv64im::chunk_fold_step::adapt_rv64im_chunk_to_fresh_ccs(&handoff);
    let native_step_statement = crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcStatement {
        step_public: crate::rv64im::chunk_fold_step::build_rv64im_chunk_step_public(
            [0; 32],
            chunk_count_in as usize,
            &fresh,
            &running_state.carry,
            &next_carry,
            step_shape.terminal_step,
        ),
        chunk_summary: FixedShapeChunkSummary::from_public_chunk(
            &handoff.public_chunk,
            public_chunk_digest,
            chunk_relation_digest,
        ),
    };
    let state_out = Rv64imChunkFoldState {
        carry: next_carry,
        transcript: transcript_out,
    };
    let main_circuit_witness = crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcWitness {
        handoff: handoff.clone(),
        state_in: running_state.clone(),
        state_out: state_out.clone(),
        replay_witness: replay_witness.clone(),
        terminal_step: step_shape.terminal_step,
    };
    let native_verified_step_statement =
        build_rv64im_main_recursion_construction2_verified_step_statement_from_relation(&Rv64imChunkStepIvcRelation {
            statement: native_step_statement.clone(),
            witness: main_circuit_witness.clone(),
        })
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let native_chunk_summary = native_verified_step_statement
        .fixed_shape_chunk_summary()
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let main_circuit_chunk_trace =
        crate::rv64im::main_relation_trace::build_rv64im_main_circuit_chunk_trace_from_authoritative_parts(
            native_verified_step_statement.chunk_index as usize,
            &main_circuit_witness.handoff,
            &native_chunk_summary,
            &main_circuit_witness.state_in.carry,
            &main_circuit_witness.state_out.carry,
            &main_circuit_witness.state_in.transcript,
            &main_circuit_witness.state_out.transcript,
            &main_circuit_witness.replay_witness,
        )
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let advice = crate::rv64im::main_recursion::Rv64imMainRecursionFPrimeAdvice::from_parts(
        vk_fs.clone(),
        chunk_count_in,
        crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state()
            .carry
            .terminal_handle
            .0,
        running_state.carry.terminal_handle.0,
        crate::rv64im::main_recursion::RV64IM_MAIN_RECURSION_TRIVIAL_PC,
        crate::rv64im::main_recursion::Rv64imMainRecursionSideLaneWitness::zero(),
        crate::rv64im::main_recursion::Rv64imMainRecursionPhiSide::zero(),
        step_statement_chain_digest_in,
        bridge_handoff_chain_digest_in,
        running_state.clone(),
        build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
            &vk_fs,
            chunk_count_in,
            crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(&running_state.carry),
            step_statement_chain_digest_in,
            bridge_handoff_chain_digest_in,
            running_state.carry.terminal_handle.0,
        )
        .x_out,
        None,
        native_verified_step_statement,
        step_shape.terminal_step,
        handoff,
        state_out,
        main_circuit_chunk_trace,
        crate::rv64im::construction2::build_rv64im_main_recursion_construction2_pi_fold_from_trace(&trace)
            .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?,
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let payload =
        build_rv64im_main_recursion_f_prime_payload(&advice, &spartan_shape.cover_shape, &spartan_shape.claim_cover)
            .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if payload.step_shape != *step_shape {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion compressed-chain dummy setup derived a payload step shape that does not match the canonical chain step shape"
                .into(),
        ));
    }
    let statement = build_rv64im_main_recursion_step_spartan_statement_from_payload(&advice)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    Ok(Rv64imMainRecursionFPrimeBackendRelation {
        f_prime_advice: advice,
        spartan_statement: statement,
        payload,
    })
}

fn main_recursion_x_out_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    label: &str,
    chunk_count_value: u64,
    z_0: &[AllocatedNum<SpartanF>; 4],
    z_0_value: &[SpartanF; 4],
    z_next: &[AllocatedNum<SpartanF>; 4],
    z_next_value: &[SpartanF; 4],
    pc_next_value: u64,
    accumulator_instance_digest: &[AllocatedNum<SpartanF>; 4],
    accumulator_instance_digest_value: &[SpartanF; 4],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs().map_err(|_| SynthesisError::Unsatisfiable)?;
    let mut transcript = Poseidon2TranscriptCircuit::new(
        cs.namespace(|| format!("{label}_init")),
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out",
    )?;
    transcript.append_message(
        cs.namespace(|| format!("{label}_version")),
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/version",
        b"v4",
    )?;
    transcript.append_message(
        cs.namespace(|| format!("{label}_vk_fs")),
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/vk_fs",
        &vk_fs.expected_digest(),
    )?;
    transcript.append_u64s(
        cs.namespace(|| format!("{label}_meta")),
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/meta",
        &[chunk_count_value, pc_next_value],
    )?;
    transcript.append_fields(
        cs.namespace(|| format!("{label}_z_0")),
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/z_0",
        z_0,
        z_0_value,
    )?;
    transcript.append_fields(
        cs.namespace(|| format!("{label}_z_i")),
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/z_i",
        z_next,
        z_next_value,
    )?;
    transcript.append_fields(
        cs.namespace(|| format!("{label}_accumulator_instance_digest")),
        b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/accumulator_instance_digest",
        accumulator_instance_digest,
        accumulator_instance_digest_value,
    )?;
    transcript.digest32(cs.namespace(|| format!("{label}_digest")))
}

fn allocated_digest_field_values(digest: &[AllocatedNum<SpartanF>; 4]) -> Result<[SpartanF; 4], SynthesisError> {
    Ok([
        digest[0]
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?,
        digest[1]
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?,
        digest[2]
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?,
        digest[3]
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?,
    ])
}

fn enforce_inactive_side_lane_constraints<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    label: &str,
    side_claim_count: u64,
    phi_side_commitment_count: u64,
) -> Result<(), SynthesisError> {
    if !crate::rv64im::main_recursion::RV64IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE {
        let side_claim_count_input = alloc_const_field_values(
            &mut cs.namespace(|| format!("{label}_side_claim_count")),
            &[SpartanF::from_canonical_u64(side_claim_count)],
            &format!("{label}_side_claim_count"),
        )?
        .into_iter()
        .next()
        .ok_or(SynthesisError::Unsatisfiable)?;
        cs.enforce(
            || format!("{label}_side_claim_count_zero"),
            |lc| lc + side_claim_count_input.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
    }
    if !crate::rv64im::main_recursion::RV64IM_MAIN_RECURSION_PHI_SIDE_ACTIVE {
        let phi_side_commitment_count_input = alloc_const_field_values(
            &mut cs.namespace(|| format!("{label}_phi_side_commitment_count")),
            &[SpartanF::from_canonical_u64(phi_side_commitment_count)],
            &format!("{label}_phi_side_commitment_count"),
        )?
        .into_iter()
        .next()
        .ok_or(SynthesisError::Unsatisfiable)?;
        cs.enforce(
            || format!("{label}_phi_side_commitment_count_zero"),
            |lc| lc + phi_side_commitment_count_input.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc,
        );
    }
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_step_spartan_inactive_side_lane_constraints(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    enforce_inactive_side_lane_constraints(
        &mut cs.namespace(|| "inactive_side_lane"),
        "inactive_side_lane",
        backend_relation.f_prime_advice.side_witness().claim_count(),
        backend_relation.payload.phi_side_commitment_words.len() as u64,
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if !cs.is_satisfied() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            cs.which_is_unsatisfied()
                .map(|name| name.to_string())
                .unwrap_or_else(|| "unknown unsatisfied inactive side-lane constraint".to_string()),
        ));
    }
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_x_out_gadget_parity(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    ensure_main_recursion_step_spartan_statement_binding(backend_relation)?;
    let statement = &backend_relation.spartan_statement;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let folded_accumulator_digest = digest_const_inputs(
        &mut cs.namespace(|| "folded_accumulator_digest"),
        statement.folded_accumulator_digest,
        "folded_accumulator_digest",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let z_0 = digest_const_inputs(&mut cs.namespace(|| "z_0"), *backend_relation.payload.z_0(), "z_0")
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let z_next = digest_const_inputs(
        &mut cs.namespace(|| "z_next"),
        *backend_relation.payload.z_next(),
        "z_next",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let expected_x_out = digest_const_inputs(
        &mut cs.namespace(|| "expected_x_out"),
        statement.x_out.bytes(),
        "expected_x_out",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let x_out_digest = main_recursion_x_out_circuit(
        &mut cs.namespace(|| "x_out_digest"),
        "x_out_digest",
        backend_relation.f_prime_advice.chunk_count_in() + 1,
        &z_0,
        &digest32_as_spartan_fields(*backend_relation.payload.z_0()),
        &z_next,
        &digest32_as_spartan_fields(*backend_relation.payload.z_next()),
        backend_relation.payload.pc_next(),
        &folded_accumulator_digest,
        &digest32_as_spartan_fields(statement.folded_accumulator_digest),
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "x_out_eq"),
        &x_out_digest,
        &expected_x_out,
        "x_out_eq",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if !cs.is_satisfied() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            cs.which_is_unsatisfied()
                .map(|name| name.to_string())
                .unwrap_or_else(|| "unknown unsatisfied x_out gadget constraint".to_string()),
        ));
    }
    Ok(())
}

fn synthesize_rv64im_main_recursion_step_body<CS: ConstraintSystem<SpartanF>>(
    circuit: &Rv64imMainRecursionStepCircuit,
    cs: &mut CS,
    public_inputs: &[AllocatedNum<SpartanF>],
    public_cursor: &mut usize,
) -> Result<Rv64imMainRecursionStepPublicVar, SynthesisError> {
    let witness = &circuit.backend_relation.f_prime_advice;
    let payload = &circuit.backend_relation.payload;
    let statement = &circuit.backend_relation.spartan_statement;
    let chunk_index_input = next_public_u64(public_inputs, public_cursor)?;
    let carry_state_in_digest_input = next_public_digest(public_inputs, public_cursor, "carry_state_in_digest")?;
    let folded_accumulator_in_digest_input =
        next_public_digest(public_inputs, public_cursor, "folded_accumulator_in_digest")?;
    let step_statement_chain_digest_in_public =
        next_public_digest(public_inputs, public_cursor, "step_statement_chain_digest_in")?;
    let bridge_handoff_chain_digest_in_public =
        next_public_digest(public_inputs, public_cursor, "bridge_handoff_chain_digest_in")?;
    let carry_state_out_digest_input = next_public_digest(public_inputs, public_cursor, "carry_state_out_digest")?;
    let x_out_input = next_public_digest(public_inputs, public_cursor, "x_out")?;
    let folded_accumulator_out_digest_input =
        next_public_digest(public_inputs, public_cursor, "folded_accumulator_out_digest")?;
    let step_statement_chain_digest_out_input =
        next_public_digest(public_inputs, public_cursor, "step_statement_chain_digest_out")?;
    let bridge_handoff_chain_digest_out_input =
        next_public_digest(public_inputs, public_cursor, "bridge_handoff_chain_digest_out")?;
    let terminal_handle_digest_out_input =
        next_public_digest(public_inputs, public_cursor, "terminal_handle_digest_out")?;

    let chunk_count_in = AllocatedNum::alloc(cs.namespace(|| "chunk_count_in"), || {
        Ok(SpartanF::from_canonical_u64(witness.chunk_count_in()))
    })?;
    cs.enforce(
        || "chunk_count_in_eq",
        |lc| lc + chunk_index_input.get_variable() - chunk_count_in.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    let step_statement_input = digest_const_inputs(
        &mut cs.namespace(|| "step_statement_digest"),
        witness.step_statement_digest(),
        "step_statement_digest",
    )?;
    let bridge_handoff_input = digest_const_inputs(
        &mut cs.namespace(|| "bridge_handoff_digest"),
        witness.bridge_handoff_digest(),
        "bridge_handoff_digest",
    )?;
    let step_statement_chain_in_expected = digest_const_inputs(
        &mut cs.namespace(|| "step_statement_chain_digest_in"),
        witness.step_statement_chain_digest_in(),
        "step_statement_chain_digest_in",
    )?;
    let bridge_handoff_chain_in_expected = digest_const_inputs(
        &mut cs.namespace(|| "bridge_handoff_chain_digest_in"),
        witness.bridge_handoff_chain_digest_in(),
        "bridge_handoff_chain_digest_in",
    )?;
    let carry_state_in_digest_expected = digest_const_inputs(
        &mut cs.namespace(|| "carry_state_in_digest"),
        rv64im_chunk_step_recursive_carry_state_digest(
            &payload.state_in_claims,
            &witness.running_state().transcript,
            witness.running_state().carry.terminal_handle.0,
        ),
        "carry_state_in_digest",
    )?;
    let folded_accumulator_in_digest_expected = digest_const_inputs(
        &mut cs.namespace(|| "folded_accumulator_in_digest"),
        crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(
            &witness.running_state().carry,
        ),
        "folded_accumulator_in_digest",
    )?;
    let z_0_input = digest_const_inputs(&mut cs.namespace(|| "z_0"), *payload.z_0(), "z_0")?;
    let z_i_input = digest_const_inputs(&mut cs.namespace(|| "z_i"), *payload.z_i(), "z_i")?;
    let z_next_input = digest_const_inputs(&mut cs.namespace(|| "z_next"), *payload.z_next(), "z_next")?;
    let pc_i_input = alloc_const_field_values(
        &mut cs.namespace(|| "pc_i"),
        &[SpartanF::from_canonical_u64(payload.pc_i())],
        "pc_i",
    )?
    .into_iter()
    .next()
    .ok_or(SynthesisError::Unsatisfiable)?;
    let pc_next_input = alloc_const_field_values(
        &mut cs.namespace(|| "pc_next"),
        &[SpartanF::from_canonical_u64(payload.pc_next())],
        "pc_next",
    )?
    .into_iter()
    .next()
    .ok_or(SynthesisError::Unsatisfiable)?;
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )?;
    let state_out_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_out"),
        &payload.state_out_claims,
        &payload.fixed_transcript_out,
        witness.fresh_state_out().carry.terminal_handle.0,
        "state_out",
    )?;
    let carry_state_out_digest_expected = digest_const_inputs(
        &mut cs.namespace(|| "carry_state_out_digest"),
        rv64im_chunk_step_recursive_carry_state_digest(
            &payload.state_out_claims,
            payload.fixed_transcript_out(),
            witness.fresh_state_out().carry.terminal_handle.0,
        ),
        "carry_state_out_digest",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "carry_state_in_digest_eq"),
        &carry_state_in_digest_input,
        &carry_state_in_digest_expected,
        "carry_state_in_digest_eq",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "folded_accumulator_in_digest_eq"),
        &folded_accumulator_in_digest_input,
        &folded_accumulator_in_digest_expected,
        "folded_accumulator_in_digest_eq",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "step_statement_chain_digest_in_eq"),
        &step_statement_chain_digest_in_public,
        &step_statement_chain_in_expected,
        "step_statement_chain_digest_in_eq",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "bridge_handoff_chain_digest_in_eq"),
        &bridge_handoff_chain_digest_in_public,
        &bridge_handoff_chain_in_expected,
        "bridge_handoff_chain_digest_in_eq",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "carry_state_out_digest_eq"),
        &carry_state_out_digest_input,
        &carry_state_out_digest_expected,
        "carry_state_out_digest_eq",
    )?;
    let canonical_initial_z = digest_const_inputs(
        &mut cs.namespace(|| "canonical_initial_z"),
        crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state()
            .carry
            .terminal_handle
            .0,
        "canonical_initial_z",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "z_0_eq_initial"),
        &z_0_input,
        &canonical_initial_z,
        "z_0_eq_initial",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "z_i_eq_state_in_terminal_handle"),
        &z_i_input,
        &state_in_var.terminal_handle,
        "z_i_eq_state_in_terminal_handle",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "z_next_eq_state_out_terminal_handle"),
        &z_next_input,
        &state_out_var.terminal_handle,
        "z_next_eq_state_out_terminal_handle",
    )?;
    cs.enforce(
        || "pc_i_eq_trivial_pc",
        |lc| lc + pc_i_input.get_variable(),
        |lc| lc + CS::one(),
        |lc| {
            lc + (
                SpartanF::from_canonical_u64(crate::rv64im::main_recursion::RV64IM_MAIN_RECURSION_TRIVIAL_PC),
                CS::one(),
            )
        },
    );
    cs.enforce(
        || "pc_next_eq_trivial_pc",
        |lc| lc + pc_next_input.get_variable(),
        |lc| lc + CS::one(),
        |lc| {
            lc + (
                SpartanF::from_canonical_u64(crate::rv64im::main_recursion::RV64IM_MAIN_RECURSION_TRIVIAL_PC),
                CS::one(),
            )
        },
    );
    let live_folded_accumulator_out_digest = synthesize_rv64im_main_recursion_step_chunk_replay(
        &mut cs.namespace(|| "payload_chunk_replay"),
        witness,
        payload,
        &state_in_var,
        &state_out_var,
    )?
    .live_folded_accumulator_out_digest;

    let folded_step_statement_chain = chunk_step_ivc_digest_chain_fold_circuit(
        &mut cs.namespace(|| "step_statement_chain_fold"),
        &step_statement_chain_digest_in_public,
        &step_statement_input,
        0x7276_3634_7374_6d74,
        "step_statement_chain_fold",
    )?;

    let folded_bridge_handoff_chain = chunk_step_ivc_digest_chain_fold_circuit(
        &mut cs.namespace(|| "bridge_handoff_chain_fold"),
        &bridge_handoff_chain_digest_in_public,
        &bridge_handoff_input,
        0x7276_3634_62686467,
        "bridge_handoff_chain_fold",
    )?;
    enforce_inactive_side_lane_constraints(
        &mut cs.namespace(|| "inactive_side_lane"),
        "inactive_side_lane",
        witness.side_witness().claim_count(),
        payload.phi_side_commitment_words.len() as u64,
    )?;
    let x_out_digest = main_recursion_x_out_circuit(
        &mut cs.namespace(|| "x_out_digest"),
        "x_out_digest",
        witness.chunk_count_in() + 1,
        &z_0_input,
        &digest32_as_spartan_fields(*payload.z_0()),
        &z_next_input,
        &digest32_as_spartan_fields(*payload.z_next()),
        payload.pc_next(),
        &live_folded_accumulator_out_digest,
        &digest32_as_spartan_fields(statement.folded_accumulator_digest),
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "x_out_eq"),
        &x_out_input,
        &x_out_digest,
        "x_out_eq",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "folded_accumulator_out_digest_eq"),
        &folded_accumulator_out_digest_input,
        &live_folded_accumulator_out_digest,
        "folded_accumulator_out_digest_eq",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "step_statement_chain_digest_out_eq"),
        &step_statement_chain_digest_out_input,
        &folded_step_statement_chain,
        "step_statement_chain_digest_out_eq",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "bridge_handoff_chain_digest_out_eq"),
        &bridge_handoff_chain_digest_out_input,
        &folded_bridge_handoff_chain,
        "bridge_handoff_chain_digest_out_eq",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "terminal_handle_digest_out_eq"),
        &terminal_handle_digest_out_input,
        &state_out_var.terminal_handle,
        "terminal_handle_digest_out_eq",
    )?;

    Ok(Rv64imMainRecursionStepPublicVar {
        chunk_index: chunk_index_input,
        carry_state_in_digest: carry_state_in_digest_input,
        step_statement_chain_digest_in: step_statement_chain_digest_in_public,
        step_statement_chain_digest_out: folded_step_statement_chain,
        bridge_handoff_chain_digest_in: bridge_handoff_chain_digest_in_public,
        bridge_handoff_chain_digest_out: folded_bridge_handoff_chain,
        carry_state_out_digest: carry_state_out_digest_input,
        folded_accumulator_out_digest: live_folded_accumulator_out_digest,
        terminal_handle_digest_out: state_out_var.terminal_handle,
    })
}

pub fn debug_check_rv64im_main_recursion_step_spartan_circuit(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if !cs.is_satisfied() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            cs.which_is_unsatisfied()
                .map(|name| name.to_string())
                .unwrap_or_else(|| "unknown unsatisfied recursive-step constraint".to_string()),
        ));
    }
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_step_spartan_embedded_body(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let relation_public_inputs = alloc_private_field_values(
        &mut cs.namespace(|| "embedded_public_inputs"),
        &circuit.expected_public_values(),
        "embedded_public_inputs",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let mut relation_public_cursor = 0usize;
    synthesize_rv64im_main_recursion_step_body(
        &circuit,
        &mut cs.namespace(|| "embedded_body"),
        &relation_public_inputs,
        &mut relation_public_cursor,
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if relation_public_cursor != relation_public_inputs.len() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion embedded step body did not consume all expected public values".into(),
        ));
    }
    if !cs.is_satisfied() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            cs.which_is_unsatisfied()
                .map(|name| name.to_string())
                .unwrap_or_else(|| "unknown unsatisfied embedded recursive-step constraint".to_string()),
        ));
    }
    Ok(())
}

pub fn debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepSpartanCircuitShape, Rv64imMainRecursionStepSpartanError> {
    let circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let shape = ShapeCS::<Rv64imSpartan2DeciderEngine>::r1cs_shape(&circuit)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let sizes = shape.sizes();
    let num_inputs = sizes[8];
    let num_aux = sizes[1] + sizes[2] + sizes[3];
    let num_constraints = sizes[0];
    Ok(Rv64imMainRecursionStepSpartanCircuitShape {
        num_inputs,
        num_aux,
        num_constraints,
        constraint_fingerprint: format!(
            "inputs:{} aux:{} constraints:{} | padded_cons:{} padded_aux:{}",
            num_inputs,
            num_aux,
            num_constraints,
            sizes[4],
            sizes[5] + sizes[6] + sizes[7],
        ),
    })
}

pub fn debug_compare_rv64im_main_recursion_step_spartan_circuit_shapes(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    left: &Rv64imMainRecursionFPrimeBackendRelation,
    right: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Option<String>, Rv64imMainRecursionStepSpartanError> {
    let left_circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, left)?;
    let right_circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, right)?;

    let mut left_cs = TestConstraintSystem::<SpartanF>::new();
    left_circuit
        .synthesize(&mut left_cs, &[], &[], None)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;

    let mut right_cs = TestConstraintSystem::<SpartanF>::new();
    right_circuit
        .synthesize(&mut right_cs, &[], &[], None)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;

    Ok(match left_cs.delta(&right_cs, false) {
        Delta::Equal => None,
        delta => Some(format!("{delta:?}")),
    })
}

pub fn debug_compare_rv64im_main_recursion_step_spartan_shape_only_skeleton(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Option<String>, Rv64imMainRecursionStepSpartanError> {
    let live_circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let dummy_relation = dummy_backend_relation_from_chain_step(
        spartan_shape,
        &backend_relation.payload.step_shape,
        backend_relation.f_prime_advice.chunk_count_in(),
        backend_relation
            .f_prime_advice
            .step_statement_chain_digest_in(),
        backend_relation
            .f_prime_advice
            .bridge_handoff_chain_digest_in(),
        backend_relation.f_prime_advice.running_state(),
    )?;
    let skeleton_circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, &dummy_relation)?;

    let mut live_cs = TestConstraintSystem::<SpartanF>::new();
    live_circuit
        .synthesize(&mut live_cs, &[], &[], None)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;

    let mut skeleton_cs = TestConstraintSystem::<SpartanF>::new();
    skeleton_circuit
        .synthesize(&mut skeleton_cs, &[], &[], None)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;

    Ok(match live_cs.delta(&skeleton_cs, false) {
        Delta::Equal => None,
        delta => Some(format!("{delta:?}")),
    })
}

pub fn debug_check_rv64im_main_recursion_step_spartan_chunk_replay_surface(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let replay_chunk = backend_relation
        .payload
        .effective_chunk_replay_surface(
            &backend_relation.f_prime_advice.running_state().transcript,
            &backend_relation
                .f_prime_advice
                .running_state()
                .carry
                .main
                .claims,
        )
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if !backend_relation
        .payload
        .chunk_cover
        .covers_replay_surface(&replay_chunk)
    {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step payload replay surface is not dominated by the carried chunk cover".into(),
        ));
    }
    if replay_chunk.pi_ccs.ccs_outputs.len() < replay_chunk.fresh_claims.len() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step replay surface has fewer CCS outputs than fresh claims".into(),
        ));
    }
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_step_spartan_pi_ccs_replay_lengths(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let replay_chunk = backend_relation
        .payload
        .effective_chunk_replay_surface(
            &backend_relation.f_prime_advice.running_state().transcript,
            &backend_relation
                .f_prime_advice
                .running_state()
                .carry
                .main
                .claims,
        )
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let (params, _, structure) = rv64im_cached_root_main_lane_context()
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;

    if replay_chunk.pi_ccs.replay_proof.sumcheck_rounds.len()
        != replay_chunk.pi_ccs.row_chals.len() + replay_chunk.pi_ccs.alpha_prime.len()
    {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step Pi_CCS FE replay round count does not match row_chals + alpha_prime".into(),
        ));
    }
    if replay_chunk.pi_ccs.replay_proof.sumcheck_rounds_nc.len()
        != replay_chunk.pi_ccs.s_col.len() + replay_chunk.pi_ccs.alpha_prime_nc.len()
    {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step Pi_CCS NC replay round count does not match s_col + alpha_prime_nc".into(),
        ));
    }
    if replay_chunk.pi_ccs.row_chals.len() != dims.ell_n {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step Pi_CCS row challenge count does not match ell_n".into(),
        ));
    }
    if replay_chunk.pi_ccs.s_col.len() != dims.ell_m {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step Pi_CCS column challenge count does not match ell_m".into(),
        ));
    }
    Ok(())
}

impl SpartanCircuit<Rv64imSpartan2DeciderEngine> for Rv64imMainRecursionStepCircuit {
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
        let _ = synthesize_rv64im_main_recursion_step_body(self, cs, &public_inputs, &mut public_cursor)?;

        if public_cursor != public_inputs.len() {
            mark_unsatisfied(
                &mut cs.namespace(|| "step_public_cursor_len_mismatch"),
                "step_public_cursor_len_mismatch",
            )?;
        }
        let _ = &self.spartan_shape;
        Ok(())
    }
}

fn build_rv64im_main_recursion_step_circuit(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepCircuit, Rv64imMainRecursionStepSpartanError> {
    let _ = rv64im_cached_root_main_lane_context()
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if !spartan_shape.matches_payload(&backend_relation.payload) {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step circuit requires a canonical recursive-step payload matching the explicit Spartan shape".into(),
        ));
    }
    if backend_relation.payload.step_shape.state_in_claim_count
        != backend_relation
            .f_prime_advice
            .running_state()
            .carry
            .main
            .claims
            .len() as u64
        || backend_relation.payload.step_shape.state_out_claim_count
            != backend_relation
                .f_prime_advice
                .fresh_state_out()
                .carry
                .main
                .claims
                .len() as u64
    {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step circuit payload/state claim counts are out of sync".into(),
        ));
    }
    ensure_main_recursion_step_spartan_statement_binding(backend_relation)?;
    Ok(Rv64imMainRecursionStepCircuit {
        spartan_shape: spartan_shape.clone(),
        backend_relation: backend_relation.clone(),
    })
}

fn build_rv64im_main_recursion_step_shape_only_circuit(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
) -> Result<Rv64imMainRecursionStepCircuit, Rv64imMainRecursionStepSpartanError> {
    let seed_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state();
    let dummy_relation = dummy_backend_relation_from_chain_step(
        spartan_shape,
        &spartan_shape.cover_shape,
        0,
        crate::rv64im::chunk_step_ivc::rv64im_step_statement_chain_digest_init(),
        crate::rv64im::chunk_step_ivc::rv64im_bridge_handoff_chain_digest_init(),
        &seed_state,
    )?;
    build_rv64im_main_recursion_step_circuit(spartan_shape, &dummy_relation)
}

pub fn setup_rv64im_main_recursion_step_spartan_shape_cached(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
) -> Result<Rv64imMainRecursionStepSpartanKeyPair, Rv64imMainRecursionStepSpartanError> {
    let cache_key = rv64im_main_recursion_step_setup_cache_key(spartan_shape)?;
    let cache = RV64IM_MAIN_RECURSION_STEP_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| {
            Rv64imMainRecursionStepSpartanError::Setup("rv64im main recursion step setup cache poisoned".into())
        })?
        .get(&cache_key)
        .cloned()
    {
        return Ok(keys);
    }
    let circuit = build_rv64im_main_recursion_step_shape_only_circuit(spartan_shape)?;
    let keys = Arc::new(
        Rv64imSpartan2DeciderSnark::setup(circuit)
            .map_err(|err| Rv64imMainRecursionStepSpartanError::Setup(err.to_string()))?,
    );
    cache
        .lock()
        .map_err(|_| {
            Rv64imMainRecursionStepSpartanError::Setup("rv64im main recursion step setup cache poisoned".into())
        })?
        .insert(cache_key, keys.clone());
    Ok(keys)
}

pub fn setup_rv64im_main_recursion_step_spartan_cached(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepSpartanKeyPair, Rv64imMainRecursionStepSpartanError> {
    build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    setup_rv64im_main_recursion_step_spartan_shape_cached(spartan_shape)
}

pub fn prove_rv64im_main_recursion_step_spartan(
    pk: &Rv64imMainRecursionStepSpartanProverKey,
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepSpartanProof, Rv64imMainRecursionStepSpartanError> {
    let circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let prep = Rv64imSpartan2DeciderSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let proof = Rv64imSpartan2DeciderSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prove(err.to_string()))?;
    let snark_data =
        bincode::serialize(&proof).map_err(|err| Rv64imMainRecursionStepSpartanError::Encode(err.to_string()))?;
    Ok(Rv64imMainRecursionStepSpartanProof { snark_data })
}

pub fn verify_rv64im_main_recursion_step_spartan(
    vk: &Rv64imMainRecursionStepSpartanVerifierKey,
    statement: &Rv64imMainRecursionStepSpartanStatement,
    proof: &Rv64imMainRecursionStepSpartanProof,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let proof: Rv64imSpartan2DeciderSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Verify(err.to_string()))?;
    let statement_public_values = main_recursion_step_public_values(statement);
    if public_values.len() < statement_public_values.len()
        || public_values[public_values.len() - statement_public_values.len()..] != statement_public_values
    {
        return Err(Rv64imMainRecursionStepSpartanError::PublicIoMismatch);
    }
    Ok(())
}

pub fn verify_rv64im_main_recursion_step_spartan_published_target(
    vk: &Rv64imMainRecursionStepSpartanVerifierKey,
    published_target: &Rv64imMainRecursionStepSpartanPublishedTarget,
    proof: &Rv64imMainRecursionStepSpartanProof,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let extracted = verify_rv64im_main_recursion_step_spartan_and_extract_published_target(vk, proof)?;
    if &extracted != published_target {
        return Err(Rv64imMainRecursionStepSpartanError::PublicIoMismatch);
    }
    Ok(())
}

pub fn verify_rv64im_main_recursion_step_spartan_and_extract_published_target(
    vk: &Rv64imMainRecursionStepSpartanVerifierKey,
    proof: &Rv64imMainRecursionStepSpartanProof,
) -> Result<Rv64imMainRecursionStepSpartanPublishedTarget, Rv64imMainRecursionStepSpartanError> {
    let proof: Rv64imSpartan2DeciderSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Verify(err.to_string()))?;
    Rv64imMainRecursionStepSpartanPublishedTarget::from_public_values(&public_values)
}

pub fn prove_rv64im_main_recursion_step_spartan_chain(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<Rv64imMainRecursionStepSpartanChainProof, Rv64imMainRecursionStepSpartanError> {
    let Some(first) = backend_relations.first() else {
        return Ok(Vec::new());
    };
    let keys = setup_rv64im_main_recursion_step_spartan_cached(spartan_shape, first)?;
    let (pk, _) = &*keys;
    let mut step_proofs = Vec::with_capacity(backend_relations.len());
    for relation in backend_relations {
        step_proofs.push(prove_rv64im_main_recursion_step_spartan(pk, spartan_shape, relation)?);
    }
    Ok(step_proofs)
}

pub fn verify_rv64im_main_recursion_step_spartan_published_target_chain(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    published_targets: &[Rv64imMainRecursionStepSpartanPublishedTarget],
    step_proofs: &[Rv64imMainRecursionStepSpartanProof],
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let extracted =
        verify_rv64im_main_recursion_step_spartan_chain_and_extract_published_targets(spartan_shape, step_proofs)?;
    if published_targets.len() != extracted.len() {
        return Err(Rv64imMainRecursionStepSpartanError::Verify(
            "rv64im main recursion step published-target chain length mismatch".into(),
        ));
    }
    if published_targets != extracted.as_slice() {
        return Err(Rv64imMainRecursionStepSpartanError::PublicIoMismatch);
    }
    Ok(())
}

pub fn verify_rv64im_main_recursion_step_spartan_chain_and_extract_published_targets(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    step_proofs: &[Rv64imMainRecursionStepSpartanProof],
) -> Result<Vec<Rv64imMainRecursionStepSpartanPublishedTarget>, Rv64imMainRecursionStepSpartanError> {
    let keys = setup_rv64im_main_recursion_step_spartan_shape_cached(spartan_shape)?;
    let (_, vk) = &*keys;
    let initial_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state();
    let mut expected_chunk_index = 0u64;
    let mut expected_carry_state_in_digest = rv64im_chunk_step_recursive_carry_state_digest(
        &initial_state.carry.main.claims,
        &initial_state.transcript,
        initial_state.carry.terminal_handle.0,
    );
    let mut expected_folded_accumulator_in_digest =
        crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(&initial_state.carry);
    let mut expected_step_statement_chain_digest =
        crate::rv64im::chunk_step_ivc::rv64im_step_statement_chain_digest_init();
    let mut expected_bridge_handoff_chain_digest =
        crate::rv64im::chunk_step_ivc::rv64im_bridge_handoff_chain_digest_init();
    let mut published_targets = Vec::with_capacity(step_proofs.len());
    for step_proof in step_proofs {
        let published_target = verify_rv64im_main_recursion_step_spartan_and_extract_published_target(vk, step_proof)?;
        if published_target.chunk_index != expected_chunk_index {
            return Err(Rv64imMainRecursionStepSpartanError::Verify(
                "rv64im main recursion step published-target chain chunk index does not match the folded prefix".into(),
            ));
        }
        if published_target.carry_state_in_digest != expected_carry_state_in_digest {
            return Err(Rv64imMainRecursionStepSpartanError::Verify(
                "rv64im main recursion step published-target chain carry-state input does not match the folded prefix"
                    .into(),
            ));
        }
        if published_target.folded_accumulator_in_digest != expected_folded_accumulator_in_digest {
            return Err(Rv64imMainRecursionStepSpartanError::Verify(
                "rv64im main recursion step published-target chain folded-accumulator input does not match the folded prefix"
                    .into(),
            ));
        }
        if published_target.step_statement_chain_digest_in != expected_step_statement_chain_digest {
            return Err(Rv64imMainRecursionStepSpartanError::Verify(
                "rv64im main recursion step published-target chain step-statement digest input does not match the folded prefix"
                    .into(),
            ));
        }
        if published_target.bridge_handoff_chain_digest_in != expected_bridge_handoff_chain_digest {
            return Err(Rv64imMainRecursionStepSpartanError::Verify(
                "rv64im main recursion step published-target chain bridge-handoff digest input does not match the folded prefix"
                    .into(),
            ));
        }
        expected_chunk_index += 1;
        expected_carry_state_in_digest = published_target.carry_state_out_digest;
        expected_folded_accumulator_in_digest = published_target.folded_accumulator_out_digest;
        expected_step_statement_chain_digest = published_target.step_statement_chain_digest;
        expected_bridge_handoff_chain_digest = published_target.bridge_handoff_chain_digest;
        published_targets.push(published_target);
    }
    Ok(published_targets)
}

pub fn verify_rv64im_main_recursion_step_spartan_chain(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
    step_proofs: &[Rv64imMainRecursionStepSpartanProof],
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    validate_rv64im_main_recursion_step_spartan_chain_shape(spartan_shape, backend_relations)?;
    let published_targets = backend_relations
        .iter()
        .map(build_rv64im_main_recursion_step_spartan_published_target)
        .collect::<Result<Vec<_>, _>>()?;
    verify_rv64im_main_recursion_step_spartan_published_target_chain(spartan_shape, &published_targets, step_proofs)
}
