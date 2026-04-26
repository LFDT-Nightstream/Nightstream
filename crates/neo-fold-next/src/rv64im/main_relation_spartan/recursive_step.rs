//! Owns the fixed-step Spartan backend for RV64IM main recursion.
//!
//! This circuit proves one carried `U_i -> U_{i+1}` transition. Public IO binds
//! the Construction-2 verifier key digest, final `u_i`, `x_out`, folded
//! accumulator digest, and bridge handoff digest. The fresh padded chunk payload (`pi_ccs`,
//! `pi_rlc`, `pi_dec`, fresh claims/witnesses) is consumed in the constraint
//! body by `synthesize_rv64im_main_recursion_step_chunk_replay`, which reuses
//! the inner verifier body `synthesize_rv64im_chunk_nifs_verifier_body`
//! directly. In the current specialization the program counter is fixed to `1`,
//! so no live PC witness/range gadget remains in the circuit.

mod authoritative_surface;
mod chunk_replay;
mod construction2_public;
mod construction2_witness;
mod debug_checks;
mod diagnostics;
mod exports;
mod public_target;
mod step_handle;
mod synthesize_support;
mod terminal_statement;

use std::fmt::Write as _;
use std::time::Instant;

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError};
use ff::Field;
use neo_math::F;
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::chunk_step_recursive::{
    build_rv64im_main_recursion_step_spartan_statement as build_rv64im_main_recursion_step_spartan_statement_from_payload,
    Rv64imMainRecursionFPrimeBackendRelation, Rv64imMainRecursionStepSpartanShape,
};
use super::recursive_cover::{
    alloc_recursive_cover_state, recursive_accumulator_instance_digest_circuit_from_claims,
    Rv64imRecursiveCoverStateVar,
};
use super::{
    alloc_const_field_values, alloc_private_field_values, digest32_as_spartan_fields, digest_const_inputs,
    enforce_digest_eq, next_public_digest, Rv64imMainRecursionStepSpartanStatement,
};
use crate::finalize::FixedShapeChunkSummary;
use crate::proof::{Carry, ChunkInput, StepInput};
use crate::rv64im::chunk_fold_step::{Rv64imAccumulatorHandle, Rv64imChunkFoldCarry};
use crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcRelation;
use crate::rv64im::construction2::{
    build_rv64im_main_recursion_construction2_verified_step_statement_from_relation,
    Rv64imMainRecursionConstruction2PublicBoundary,
};
use crate::rv64im::final_relation::{Rv64imChunkFoldState, Rv64imChunkFoldTranscriptSnapshot};
use crate::rv64im::ivc_snark::{Rv64imDeciderEngine, SpartanCircuit, SpartanF};
use crate::rv64im::kernel::{
    rv64im_cached_root_main_lane_optimized_cache, rv64im_root_main_lane_context_for_claim_count,
    rv64im_root_main_lane_context_for_step_cap,
};
use crate::rv64im::kernel::{
    Rv64imChunkBridgeHandoff, Rv64imPreparedStepBridgeBinding, Rv64imVerifiedKernelChunkHandoff,
};
use crate::rv64im::main_recursion::{
    build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs,
    build_rv64im_main_recursion_verifier_key_fs_for_step_cap,
};
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::rv64im::main_relation_spartan::chunk_step_ivc::Rv64imChunkStepIvcShape;
use crate::rv64im::main_relation_spartan::chunk_step_recursive::build_rv64im_main_recursion_f_prime_payload;
use chunk_replay::synthesize_rv64im_main_recursion_step_chunk_replay;
use construction2_public::enforce_digest_eq_when_non_base;
use construction2_witness::construction2_current_input_x_from_live_step;
pub(super) use step_handle::fixed_shape_recursive_step_handle_digest_circuit;
use step_handle::fixed_shape_recursive_step_handle_digest_circuit_from_vars;
use synthesize_support::{emit_synthesize_trace, mark_unsatisfied};
use terminal_statement::{
    build_terminal_f_prime_verified_step_statement_digest, construction2_verified_step_statement_digest_circuit,
};

pub use exports::*;
pub(crate) use public_target::terminal_f_prime_r2_public_values_from_parts;
pub use public_target::Rv64imMainRecursionStepSpartanPublishedTarget;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionStepSpartanCircuitShape {
    pub num_inputs: usize,
    pub num_aux: usize,
    pub num_constraints: usize,
    pub constraint_fingerprint: String,
}

fn format_spartan_digest_hex(digest: [u8; 32]) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[derive(Debug, Error)]
pub enum Rv64imMainRecursionStepSpartanError {
    #[error("rv64im main recursion step prepare failed: {0}")]
    Prepare(String),
    #[error("rv64im main recursion step verify failed: {0}")]
    Verify(String),
}

#[derive(Clone)]
pub(crate) struct Rv64imMainRecursionStepCircuit {
    spartan_shape: Rv64imMainRecursionStepSpartanShape,
    backend_relation: Rv64imMainRecursionFPrimeBackendRelation,
}

impl Rv64imMainRecursionStepCircuit {
    fn expected_public_values(&self) -> Vec<SpartanF> {
        let target = build_rv64im_main_recursion_step_spartan_published_target(&self.backend_relation)
            .expect("recursive-step circuit must be built from a canonical backend relation");
        target.terminal_f_prime_r2_public_values()
    }
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
    if backend_relation.construction2_u_next.x_i() != &canonical_statement.x_out {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion published target requires backend construction2_u_next.x_i to match F' output"
                .into(),
        ));
    }
    backend_relation
        .f_prime_advice
        .construction2_input_fresh_instance()
        .ok_or_else(|| {
            Rv64imMainRecursionStepSpartanError::Prepare(
                "rv64im main recursion target requires a Construction-2 input fresh instance".into(),
            )
        })?;
    let terminal_verified_step_statement_digest =
        build_terminal_f_prime_verified_step_statement_digest(backend_relation)?;
    Ok(Rv64imMainRecursionStepSpartanPublishedTarget {
        vk_fs_digest: backend_relation
            .f_prime_advice
            .verifier_key_fs()
            .expected_digest(),
        chunk_count: backend_relation.f_prime_advice.chunk_count_in() + 1,
        z_0: *backend_relation.payload.z_0(),
        z_i: *backend_relation.payload.z_next(),
        pc: backend_relation.payload.pc_next(),
        x_out: canonical_statement.x_out,
        construction2_u_i: Rv64imMainRecursionConstruction2PublicBoundary::from_fresh_instance(
            &backend_relation.construction2_u_next,
        ),
        folded_accumulator_out_digest: canonical_statement.folded_accumulator_digest,
        bridge_handoff_digest: backend_relation.f_prime_advice.bridge_handoff_digest(),
        terminal_verified_step_statement_digest,
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

fn zero_step_inputs(
    fresh_claim_shapes: &[crate::rv64im::main_relation_spartan::chunk_step_recursive::Rv64imCcsClaimShape],
    fresh_witness_shapes: &[crate::rv64im::main_relation_spartan::chunk_step_recursive::Rv64imCcsWitnessShape],
    live_len: usize,
) -> Vec<StepInput> {
    (0..live_len)
        .map(|idx| StepInput {
            label: String::new(),
            mcs: fresh_claim_shapes[idx].zero_claim(),
            witness: fresh_witness_shapes[idx].zero_witness(),
        })
        .collect()
}

fn dummy_backend_relation_from_chain_step(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    step_shape: &Rv64imChunkStepIvcShape,
    chunk_count_in: u64,
    running_state: &Rv64imChunkFoldState,
) -> Result<Rv64imMainRecursionFPrimeBackendRelation, Rv64imMainRecursionStepSpartanError> {
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs_for_step_cap(step_shape.fresh_claim_count as usize)
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
    let step_cap = usize::try_from(step_shape.fresh_claim_count).map_err(|_| {
        Rv64imMainRecursionStepSpartanError::Prepare("rv64im main recursion step step_cap overflow".into())
    })?;
    let (params, log, structure) = rv64im_root_main_lane_context_for_step_cap(step_cap)
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
            &params,
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
        &params,
        structure,
        log,
        optimized_cache,
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let next_carry = Rv64imChunkFoldCarry::from_main(
        Carry {
            claims: trace.children.clone(),
            witnesses: trace.z_split.clone(),
        },
        Rv64imAccumulatorHandle(crate::rv64im::chunk_relation::rv64im_step_handle(
            running_state.carry.terminal_handle.0,
            chunk_count_in as usize,
            handoff.public_chunk.start_index,
            handoff.public_chunk.steps.len(),
            chunk_relation_digest,
        )),
    );
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
    let canonical_full_width =
        crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_canonical_full_width(
            &vk_fs,
            &crate::rv64im::main_recursion::Rv64imMainRecursionPhiSide::zero(),
        )
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let canonical_u_i = crate::rv64im::construction2::build_rv64im_main_recursion_construction2_default_fresh_instance(
        &vk_fs,
        canonical_full_width,
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let initial_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state_for_step_cap(
        vk_fs
            .step_cap()
            .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?,
    );
    let advice = crate::rv64im::main_recursion::Rv64imMainRecursionFPrimeAdvice::from_parts(
        vk_fs.clone(),
        chunk_count_in,
        initial_state.carry.terminal_handle.0,
        running_state.carry.terminal_handle.0,
        crate::rv64im::main_recursion::RV64IM_MAIN_RECURSION_TRIVIAL_PC,
        crate::rv64im::main_recursion::Rv64imMainRecursionSideLaneWitness::zero(),
        crate::rv64im::main_recursion::Rv64imMainRecursionPhiSide::zero(),
        running_state.clone(),
        build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
            &vk_fs,
            chunk_count_in,
            crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(&running_state.carry),
            running_state.carry.terminal_handle.0,
        )
        .x_out,
        Some(canonical_u_i),
        native_verified_step_statement,
        step_shape.terminal_step,
        handoff,
        state_out,
        replay_witness,
        crate::rv64im::construction2::build_rv64im_main_recursion_construction2_pi_fold_from_trace(&trace)
            .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?,
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let payload = build_rv64im_main_recursion_f_prime_payload(&advice, &spartan_shape)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if !step_shape.covers_recursive_step_shape(&payload.step_shape)
        || !step_shape.canonical_recursive_step_shape_equal(&payload.step_shape)
    {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion compressed-chain dummy setup derived a payload step shape whose fixed-shape fields drifted from the requested canonical chain step shape"
                .into(),
        ));
    }
    let statement = build_rv64im_main_recursion_step_spartan_statement_from_payload(&advice)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let construction2_u_next = crate::rv64im::main_recursion::evaluate_rv64im_main_recursion_f_prime_advice(&advice)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?
        .construction2_u_next()
        .clone();
    Ok(Rv64imMainRecursionFPrimeBackendRelation {
        f_prime_advice: advice,
        spartan_statement: statement,
        construction2_u_next,
        payload,
    })
}

fn main_recursion_x_out_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    label: &str,
    vk_fs_digest: [u8; 32],
    chunk_count_halves: &[AllocatedNum<SpartanF>; 2],
    chunk_count_half_values: &[SpartanF; 2],
    z_0: &[AllocatedNum<SpartanF>; 4],
    z_0_value: &[SpartanF; 4],
    z_next: &[AllocatedNum<SpartanF>; 4],
    z_next_value: &[SpartanF; 4],
    pc_next_halves: &[AllocatedNum<SpartanF>; 2],
    pc_next_half_values: &[SpartanF; 2],
    accumulator_instance_digest: &[AllocatedNum<SpartanF>; 4],
    accumulator_instance_digest_value: &[SpartanF; 4],
    exact_initial_prefix: Option<ExactInitialXOutPrefix>,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
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
        &vk_fs_digest,
    )?;
    if let Some(prefix) = exact_initial_prefix {
        let expected_chunk_count_halves = u64_halves_as_spartan_fields(prefix.next_chunk_count);
        let expected_pc_next_halves = u64_halves_as_spartan_fields(prefix.pc_next);
        if chunk_count_half_values != &expected_chunk_count_halves
            || pc_next_half_values != &expected_pc_next_halves
            || z_0_value != &prefix.z_0
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        transcript.append_u64s(
            cs.namespace(|| format!("{label}_meta")),
            b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/meta",
            &[prefix.next_chunk_count, prefix.pc_next],
        )?;
        transcript.append_const_fields(
            cs.namespace(|| format!("{label}_z_0")),
            b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/z_0",
            &prefix.z_0,
        )?;
    } else {
        let meta_halves = [
            chunk_count_halves[0].clone(),
            chunk_count_halves[1].clone(),
            pc_next_halves[0].clone(),
            pc_next_halves[1].clone(),
        ];
        let meta_half_values = [
            chunk_count_half_values[0],
            chunk_count_half_values[1],
            pc_next_half_values[0],
            pc_next_half_values[1],
        ];
        transcript.append_u64_halves(
            cs.namespace(|| format!("{label}_meta")),
            b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/meta",
            &meta_halves,
            &meta_half_values,
            2,
        )?;
        transcript.append_fields(
            cs.namespace(|| format!("{label}_z_0")),
            b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/z_0",
            z_0,
            z_0_value,
        )?;
    }
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

#[derive(Clone, Copy)]
struct ExactInitialXOutPrefix {
    next_chunk_count: u64,
    pc_next: u64,
    z_0: [SpartanF; 4],
}

fn exact_initial_x_out_prefix(step_cap: usize) -> ExactInitialXOutPrefix {
    let initial_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state_for_step_cap(step_cap);
    ExactInitialXOutPrefix {
        next_chunk_count: 1,
        pc_next: 1,
        z_0: digest32_as_spartan_fields(initial_state.carry.terminal_handle.0),
    }
}

pub(super) fn ensure_unit_program_counter(pc: u64) -> Result<(), SynthesisError> {
    if pc == 1 {
        Ok(())
    } else {
        Err(SynthesisError::Unsatisfiable)
    }
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

fn enforce_u64_halves_eq_constant<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    halves: &[AllocatedNum<SpartanF>; 2],
    value: u64,
    label: &str,
) {
    let expected_halves = u64_halves_as_spartan_fields(value);
    for (idx, (half, expected)) in halves.iter().zip(expected_halves.iter()).enumerate() {
        enforce_allocated_num_eq_constant(cs, half, *expected, &format!("{label}_{idx}"));
    }
}

pub(crate) fn allocated_digest_field_values(
    digest: &[AllocatedNum<SpartanF>; 4],
) -> Result<[SpartanF; 4], SynthesisError> {
    Ok([
        digest[0].get_value().unwrap_or(SpartanF::ZERO),
        digest[1].get_value().unwrap_or(SpartanF::ZERO),
        digest[2].get_value().unwrap_or(SpartanF::ZERO),
        digest[3].get_value().unwrap_or(SpartanF::ZERO),
    ])
}

fn private_digest_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest: [u8; 32],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    alloc_private_field_values(cs, &digest32_as_spartan_fields(digest), label)?
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)
}

fn enforce_digest_bit_image<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest_bits: &[AllocatedNum<SpartanF>],
    digest: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<(), SynthesisError> {
    if digest_bits.len() != 256 {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, bit) in digest_bits.iter().enumerate() {
        cs.enforce(
            || format!("{label}_bit_{idx}_boolean"),
            |lc| lc + bit.get_variable(),
            |lc| lc + bit.get_variable() - CS::one(),
            |lc| lc,
        );
    }
    for (limb_idx, limb_bits) in digest_bits.chunks_exact(64).enumerate() {
        let mut acc = LinearCombination::<SpartanF>::zero();
        for (bit_idx, bit) in limb_bits.iter().enumerate() {
            acc = acc + (SpartanF::from_canonical_u64(1u64 << bit_idx), bit.get_variable());
        }
        cs.enforce(
            || format!("{label}_limb_{limb_idx}_packs_bits"),
            |_| acc,
            |lc| lc + CS::one(),
            |lc| lc + digest[limb_idx].get_variable(),
        );
    }
    Ok(())
}

pub(super) fn u64_halves_as_spartan_fields(value: u64) -> [SpartanF; 2] {
    [
        SpartanF::from_canonical_u64(value & 0xFFFF_FFFF),
        SpartanF::from_canonical_u64(value >> 32),
    ]
}

fn private_u64_halves<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: u64,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 2], SynthesisError> {
    alloc_private_field_values(cs, &u64_halves_as_spartan_fields(value), label)?
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)
}

fn next_public_values(
    public_inputs: &[AllocatedNum<SpartanF>],
    cursor: &mut usize,
    count: usize,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    if *cursor + count > public_inputs.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let out = public_inputs[*cursor..*cursor + count].to_vec();
    *cursor += count;
    Ok(out)
}

fn next_public_u64_halves(
    public_inputs: &[AllocatedNum<SpartanF>],
    cursor: &mut usize,
) -> Result<[AllocatedNum<SpartanF>; 2], SynthesisError> {
    if *cursor + 2 > public_inputs.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let out = [public_inputs[*cursor].clone(), public_inputs[*cursor + 1].clone()];
    *cursor += 2;
    Ok(out)
}

fn enforce_initial_transcript_when_base<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    next_chunk_count_halves: &[AllocatedNum<SpartanF>; 2],
    next_chunk_count: u64,
    state_in_var: &Rv64imRecursiveCoverStateVar,
    label: &str,
) -> Result<(), SynthesisError> {
    let combined_value = SpartanF::from_canonical_u64(next_chunk_count) - SpartanF::ONE;
    let is_initial_value = if next_chunk_count == 1 {
        SpartanF::ONE
    } else {
        SpartanF::ZERO
    };
    let inverse_value = if combined_value == SpartanF::ZERO {
        SpartanF::ZERO
    } else {
        combined_value.invert().unwrap()
    };
    let is_initial = AllocatedNum::alloc(cs.namespace(|| format!("{label}_is_initial")), || Ok(is_initial_value))?;
    let inverse = AllocatedNum::alloc(cs.namespace(|| format!("{label}_not_initial_inverse")), || {
        Ok(inverse_value)
    })?;
    cs.enforce(
        || format!("{label}_is_initial_bit"),
        |lc| lc + is_initial.get_variable(),
        |lc| lc + CS::one() - (SpartanF::ONE, is_initial.get_variable()),
        |lc| lc,
    );
    cs.enforce(
        || format!("{label}_combined_times_is_initial"),
        |lc| {
            lc + next_chunk_count_halves[0].get_variable()
                + (
                    SpartanF::from_canonical_u64(1u64 << 32),
                    next_chunk_count_halves[1].get_variable(),
                )
                - CS::one()
        },
        |lc| lc + is_initial.get_variable(),
        |lc| lc,
    );
    cs.enforce(
        || format!("{label}_combined_inverse"),
        |lc| {
            lc + next_chunk_count_halves[0].get_variable()
                + (
                    SpartanF::from_canonical_u64(1u64 << 32),
                    next_chunk_count_halves[1].get_variable(),
                )
                - CS::one()
        },
        |lc| lc + inverse.get_variable(),
        |lc| lc + CS::one() - (SpartanF::ONE, is_initial.get_variable()),
    );

    let expected = crate::rv64im::final_relation::rv64im_chunk_fold_initial_transcript_snapshot();
    let expected_state = expected
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    for (idx, (lane, expected_lane)) in state_in_var
        .transcript_state
        .iter()
        .zip(expected_state.iter())
        .enumerate()
    {
        cs.enforce(
            || format!("{label}_state_{idx}"),
            |lc| lc + lane.get_variable() - (*expected_lane, CS::one()),
            |lc| lc + is_initial.get_variable(),
            |lc| lc,
        );
    }
    cs.enforce(
        || format!("{label}_absorbed"),
        |lc| {
            lc + state_in_var.transcript_absorbed.get_variable()
                - (SpartanF::from_canonical_u64(expected.absorbed as u64), CS::one())
        },
        |lc| lc + is_initial.get_variable(),
        |lc| lc,
    );
    Ok(())
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

pub(crate) fn synthesize_rv64im_main_recursion_step_body_with_outputs<CS: ConstraintSystem<SpartanF>>(
    circuit: &Rv64imMainRecursionStepCircuit,
    cs: &mut CS,
    public_inputs: &[AllocatedNum<SpartanF>],
    public_cursor: &mut usize,
    trace_prefix: Option<&str>,
) -> Result<(), SynthesisError> {
    let witness = &circuit.backend_relation.f_prime_advice;
    let payload = &circuit.backend_relation.payload;
    let started = Instant::now();
    let vk_fs_digest_input = next_public_digest(public_inputs, public_cursor, "vk_fs_digest")?;
    let chunk_count_input = next_public_u64_halves(public_inputs, public_cursor)?;
    let z_0_public_input = next_public_digest(public_inputs, public_cursor, "z_0")?;
    let z_i_public_input = next_public_digest(public_inputs, public_cursor, "z_i")?;
    let pc_input = next_public_u64_halves(public_inputs, public_cursor)?;
    let x_out_input = next_public_digest(public_inputs, public_cursor, "x_out")?;
    let x_out_field_image_input = next_public_values(public_inputs, public_cursor, 256)?;
    let folded_accumulator_out_digest_input =
        next_public_digest(public_inputs, public_cursor, "folded_accumulator_out_digest")?;
    let bridge_handoff_digest_input = next_public_digest(public_inputs, public_cursor, "bridge_handoff_digest")?;
    let terminal_verified_step_statement_digest_input =
        next_public_digest(public_inputs, public_cursor, "terminal_verified_step_statement_digest")?;
    emit_synthesize_trace(trace_prefix, "public_inputs", started);
    let started = Instant::now();
    let next_chunk_count = witness.chunk_count_in() + 1;
    let chunk_index_halves = private_u64_halves(
        &mut cs.namespace(|| "chunk_index_halves"),
        next_chunk_count,
        "chunk_index_halves",
    )?;
    let z_0_input = z_0_public_input.clone();
    let z_i_input = private_digest_inputs(&mut cs.namespace(|| "z_i"), *payload.z_i(), "z_i")?;
    let z_next_input = z_i_public_input.clone();
    let pc_next_halves = private_u64_halves(
        &mut cs.namespace(|| "pc_next_halves"),
        payload.pc_next(),
        "pc_next_halves",
    )?;
    let terminal_halted_out_halves = private_u64_halves(
        &mut cs.namespace(|| "terminal_halted_out_halves"),
        u64::from(witness.bridge_handoff_halted_out()),
        "terminal_halted_out_halves",
    )?;
    let terminal_halted_out_half_values = u64_halves_as_spartan_fields(u64::from(witness.bridge_handoff_halted_out()));
    let step_handle_meta_values = [
        SpartanF::from_canonical_u64(payload.handoff.public_chunk.start_index as u64),
        SpartanF::from_canonical_u64(payload.handoff.public_chunk.steps.len() as u64),
    ];
    let step_handle_meta = alloc_private_field_values(
        &mut cs.namespace(|| "step_handle_meta"),
        &step_handle_meta_values,
        "step_handle_meta",
    )?;
    emit_synthesize_trace(trace_prefix, "private_witness_inputs", started);
    let started = Instant::now();
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
    emit_synthesize_trace(trace_prefix, "alloc_cover_states", started);
    let started = Instant::now();
    let initial_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state_for_step_cap(
        witness
            .verifier_key_fs()
            .step_cap()
            .map_err(|_| SynthesisError::Unsatisfiable)?,
    );
    let canonical_initial_z = digest_const_inputs(
        &mut cs.namespace(|| "canonical_initial_z"),
        initial_state.carry.terminal_handle.0,
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
    enforce_initial_transcript_when_base(
        &mut cs.namespace(|| "initial_transcript_gate"),
        &chunk_index_halves,
        next_chunk_count,
        &state_in_var,
        "initial_transcript_gate",
    )?;
    ensure_unit_program_counter(payload.pc_i())?;
    ensure_unit_program_counter(payload.pc_next())?;
    let expected_vk_fs_digest = digest_const_inputs(
        &mut cs.namespace(|| "expected_vk_fs_digest"),
        witness.verifier_key_fs().expected_digest(),
        "expected_vk_fs_digest",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "vk_fs_digest_eq"),
        &vk_fs_digest_input,
        &expected_vk_fs_digest,
        "vk_fs_digest_eq",
    )?;
    let expected_bridge_handoff_digest = private_digest_inputs(
        &mut cs.namespace(|| "expected_bridge_handoff_digest"),
        witness.bridge_handoff_digest(),
        "expected_bridge_handoff_digest",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "bridge_handoff_digest_eq"),
        &bridge_handoff_digest_input,
        &expected_bridge_handoff_digest,
        "bridge_handoff_digest_eq",
    )?;
    for idx in 0..2 {
        cs.enforce(
            || format!("chunk_count_public_eq_{idx}"),
            |lc| lc + chunk_count_input[idx].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + chunk_index_halves[idx].get_variable(),
        );
        cs.enforce(
            || format!("pc_public_eq_{idx}"),
            |lc| lc + pc_input[idx].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + pc_next_halves[idx].get_variable(),
        );
    }
    cs.enforce(
        || "terminal_halted_out_low_bit_boolean",
        |lc| lc + terminal_halted_out_halves[0].get_variable(),
        |lc| lc + terminal_halted_out_halves[0].get_variable() - CS::one(),
        |lc| lc,
    );
    cs.enforce(
        || "terminal_halted_out_high_half_zero",
        |lc| lc + terminal_halted_out_halves[1].get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    let exact_initial_x_out_prefix = None;
    emit_synthesize_trace(trace_prefix, "bind_state_and_pc", started);
    let started = Instant::now();
    let chunk_replay = synthesize_rv64im_main_recursion_step_chunk_replay(
        &mut cs.namespace(|| "payload_chunk_replay"),
        witness,
        payload,
        &state_in_var,
        &state_out_var,
        &bridge_handoff_digest_input,
        trace_prefix,
    )?;
    let statement_chunk_index = witness.chunk_count_in();
    let statement_chunk_index_halves = private_u64_halves(
        &mut cs.namespace(|| "terminal_verified_step_chunk_index_halves"),
        statement_chunk_index,
        "terminal_verified_step_chunk_index_halves",
    )?;
    let statement_chunk_index_half_values = u64_halves_as_spartan_fields(statement_chunk_index);
    let step_hi = (payload.handoff.public_chunk.start_index as u64)
        .checked_add(payload.handoff.public_chunk.steps.len() as u64)
        .ok_or(SynthesisError::Unsatisfiable)?;
    let step_hi_halves = private_u64_halves(
        &mut cs.namespace(|| "terminal_verified_step_step_hi_halves"),
        step_hi,
        "terminal_verified_step_step_hi_halves",
    )?;
    let step_hi_half_values = u64_halves_as_spartan_fields(step_hi);
    let two32 = SpartanF::from_canonical_u64(1u64 << 32);
    cs.enforce(
        || "terminal_verified_step_chunk_index_closes_public_chunk_count",
        |lc| {
            lc + statement_chunk_index_halves[0].get_variable()
                + (two32, statement_chunk_index_halves[1].get_variable())
                + CS::one()
                - chunk_index_halves[0].get_variable()
                - (two32, chunk_index_halves[1].get_variable())
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );
    cs.enforce(
        || "terminal_verified_step_step_hi_closes_public_chunk_span",
        |lc| {
            lc + step_hi_halves[0].get_variable() + (two32, step_hi_halves[1].get_variable())
                - chunk_replay.pi_ccs.public_chunk_start_index.get_variable()
                - chunk_replay.pi_ccs.public_step_count.get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );
    let live_verified_step_statement_digest = construction2_verified_step_statement_digest_circuit(
        &mut cs.namespace(|| "terminal_verified_step_statement_digest"),
        "terminal_verified_step_statement_digest",
        &statement_chunk_index_halves,
        &statement_chunk_index_half_values,
        &chunk_replay.pi_ccs.public_chunk_start_index_halves,
        &chunk_replay.pi_ccs.public_chunk_start_index_half_values,
        &step_hi_halves,
        &step_hi_half_values,
        &terminal_halted_out_halves,
        &terminal_halted_out_half_values,
        &state_in_var.terminal_handle,
        &digest32_as_spartan_fields(witness.running_state().carry.terminal_handle.0),
        &state_out_var.terminal_handle,
        &digest32_as_spartan_fields(witness.fresh_state_out().carry.terminal_handle.0),
        &chunk_replay.pi_ccs.public_chunk_digest,
        &digest32_as_spartan_fields(payload.handoff.public_chunk_digest),
        &chunk_replay.chunk_relation_digest,
        &chunk_replay.chunk_relation_digest_values,
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "terminal_verified_step_statement_digest_eq"),
        &terminal_verified_step_statement_digest_input,
        &live_verified_step_statement_digest,
        "terminal_verified_step_statement_digest_eq",
    )?;
    let current_input_x_i = construction2_current_input_x_from_live_step(
        &mut cs.namespace(|| "construction2_current_input"),
        witness,
        &statement_chunk_index_halves,
        trace_prefix,
    )?;
    let live_folded_accumulator_in_digest = recursive_accumulator_instance_digest_circuit_from_claims(
        &mut cs.namespace(|| "live_folded_accumulator_in_digest"),
        &chunk_replay.state_in_claims,
        &state_in_var.terminal_handle,
        "live_folded_accumulator_in_digest",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "state_in_folded_accumulator_digest_eq_live"),
        &state_in_var.folded_accumulator_digest,
        &live_folded_accumulator_in_digest,
        "state_in_folded_accumulator_digest_eq_live",
    )?;
    let folded_accumulator_in_digest_values = digest32_as_spartan_fields(witness.folded_accumulator_in_digest());
    let current_x_i_digest = main_recursion_x_out_circuit(
        &mut cs.namespace(|| "current_x_i_digest"),
        "current_x_i_digest",
        witness.verifier_key_fs().expected_digest(),
        &statement_chunk_index_halves,
        &statement_chunk_index_half_values,
        &z_0_input,
        &digest32_as_spartan_fields(*payload.z_0()),
        &z_i_input,
        &digest32_as_spartan_fields(*payload.z_i()),
        &pc_input,
        &u64_halves_as_spartan_fields(payload.pc_i()),
        &state_in_var.folded_accumulator_digest,
        &folded_accumulator_in_digest_values,
        None,
    )?;
    enforce_digest_eq_when_non_base(
        &mut cs.namespace(|| "current_construction2_u_i_x_i_eq"),
        &current_input_x_i,
        &current_x_i_digest,
        &statement_chunk_index_halves,
        "current_construction2_u_i_x_i_eq",
    );
    let live_folded_accumulator_out_digest = chunk_replay.live_folded_accumulator_out_digest;
    let expected_step_handle = fixed_shape_recursive_step_handle_digest_circuit_from_vars(
        &mut cs.namespace(|| "expected_step_handle"),
        "expected_step_handle",
        &state_in_var.terminal_handle,
        &digest32_as_spartan_fields(witness.running_state().carry.terminal_handle.0),
        &chunk_index_halves,
        next_chunk_count,
        &step_handle_meta[0],
        step_handle_meta_values[0],
        &step_handle_meta[1],
        step_handle_meta_values[1],
        &chunk_replay.chunk_relation_digest,
        &chunk_replay.chunk_relation_digest_values,
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "state_out_terminal_handle_eq_expected_step_handle"),
        &state_out_var.terminal_handle,
        &expected_step_handle,
        "state_out_terminal_handle_eq_expected_step_handle",
    )?;
    emit_synthesize_trace(trace_prefix, "payload_chunk_replay", started);

    let started = Instant::now();
    enforce_inactive_side_lane_constraints(
        &mut cs.namespace(|| "inactive_side_lane"),
        "inactive_side_lane",
        witness.side_witness().claim_count(),
        payload.phi_side_commitment_words.len() as u64,
    )?;
    let live_folded_accumulator_out_digest_values = allocated_digest_field_values(&live_folded_accumulator_out_digest)?;
    let x_out_digest = main_recursion_x_out_circuit(
        &mut cs.namespace(|| "x_out_digest"),
        "x_out_digest",
        witness.verifier_key_fs().expected_digest(),
        &chunk_index_halves,
        &u64_halves_as_spartan_fields(next_chunk_count),
        &z_0_input,
        &digest32_as_spartan_fields(*payload.z_0()),
        &z_next_input,
        &digest32_as_spartan_fields(*payload.z_next()),
        &pc_next_halves,
        &u64_halves_as_spartan_fields(payload.pc_next()),
        &live_folded_accumulator_out_digest,
        &live_folded_accumulator_out_digest_values,
        exact_initial_x_out_prefix,
    )?;
    emit_synthesize_trace(trace_prefix, "inactive_side_lane_and_x_out", started);
    let started = Instant::now();
    enforce_digest_eq(
        &mut cs.namespace(|| "x_out_eq"),
        &x_out_input,
        &x_out_digest,
        "x_out_eq",
    )?;
    enforce_digest_bit_image(
        &mut cs.namespace(|| "x_out_field_image_eq"),
        &x_out_field_image_input,
        &x_out_digest,
        "x_out_field_image_eq",
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "folded_accumulator_out_digest_eq"),
        &folded_accumulator_out_digest_input,
        &live_folded_accumulator_out_digest,
        "folded_accumulator_out_digest_eq",
    )?;
    emit_synthesize_trace(trace_prefix, "public_output_eq", started);
    Ok(())
}

pub(crate) fn synthesize_rv64im_main_recursion_step_body<CS: ConstraintSystem<SpartanF>>(
    circuit: &Rv64imMainRecursionStepCircuit,
    cs: &mut CS,
    public_inputs: &[AllocatedNum<SpartanF>],
    public_cursor: &mut usize,
    trace_prefix: Option<&str>,
) -> Result<(), SynthesisError> {
    synthesize_rv64im_main_recursion_step_body_with_outputs(circuit, cs, public_inputs, public_cursor, trace_prefix)
}

impl SpartanCircuit<Rv64imDeciderEngine> for Rv64imMainRecursionStepCircuit {
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
        synthesize_rv64im_main_recursion_step_body(self, cs, &public_inputs, &mut public_cursor, None)?;

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

pub(crate) fn build_rv64im_main_recursion_step_circuit(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepCircuit, Rv64imMainRecursionStepSpartanError> {
    build_rv64im_main_recursion_step_circuit_inner(spartan_shape, backend_relation)
}

pub(crate) fn build_rv64im_terminal_f_prime_r2_circuit(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepCircuit, Rv64imMainRecursionStepSpartanError> {
    build_rv64im_main_recursion_step_circuit_inner(spartan_shape, backend_relation)
}

fn build_rv64im_main_recursion_step_circuit_inner(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepCircuit, Rv64imMainRecursionStepSpartanError> {
    let _ = rv64im_root_main_lane_context_for_claim_count(
        backend_relation
            .f_prime_advice
            .running_state()
            .carry
            .main
            .claims
            .len(),
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if !spartan_shape.matches_payload(&backend_relation.payload) {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step circuit requires a canonical recursive-step payload matching the explicit Spartan shape".into(),
        ));
    }
    let canonical_step_image =
        crate::rv64im::main_recursion::evaluate_rv64im_main_recursion_f_prime_advice(&backend_relation.f_prime_advice)
            .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if backend_relation.construction2_u_next.x_i() != canonical_step_image.construction2_u_next().x_i() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step circuit requires construction2_u_next.x_i to match the terminal F' output hash"
                .into(),
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
    let step_cap = usize::try_from(spartan_shape.cover_shape.fresh_claim_count).map_err(|_| {
        Rv64imMainRecursionStepSpartanError::Prepare("rv64im main recursion shape step_cap overflow".into())
    })?;
    let seed_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state_for_step_cap(step_cap);
    let dummy_relation =
        dummy_backend_relation_from_chain_step(spartan_shape, &spartan_shape.cover_shape, 0, &seed_state)?;
    build_rv64im_main_recursion_step_circuit(spartan_shape, &dummy_relation)
}
