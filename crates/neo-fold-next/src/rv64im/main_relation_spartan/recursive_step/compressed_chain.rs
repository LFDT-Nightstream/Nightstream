//! Owns compressed-chain wrapper/debug/setup/prove for the RV64IM recursive-step Spartan backend.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem, SynthesisError};
use neo_transcript::{Poseidon2Transcript, Transcript};
use spartan2::{
    bellpepper::{r1cs::SpartanShape, shape_cs::ShapeCS},
    provider::goldi::F as SpartanF,
    traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};

use super::*;
use crate::rv64im::main_relation_spartan::chunk_step_recursive::rv64im_chunk_step_recursive_carry_state_digest;

static RV64IM_MAIN_RECURSION_STEP_COMPRESSED_CHAIN_SETUP_CACHE: OnceLock<
    Mutex<HashMap<[u8; 32], Rv64imMainRecursionStepSpartanKeyPair>>,
> = OnceLock::new();
const RV64IM_MAIN_RECURSION_STEP_PUBLIC_IO_ARITY: usize = 20;

#[derive(Clone, Debug, PartialEq)]
pub struct Rv64imMainRecursionStepSpartanCompressedChainProveMetrics {
    pub setup_ms: f64,
    pub prep_prove_ms: f64,
    pub prove_ms: f64,
    pub serialize_ms: f64,
    pub snark_bytes: usize,
}

#[derive(Clone)]
pub(super) struct Rv64imMainRecursionStepCompressedChainCircuit {
    chain_shape: Rv64imMainRecursionStepSpartanCompressedChainShape,
    backend_relations: Vec<Rv64imMainRecursionFPrimeBackendRelation>,
}

fn rv64im_main_recursion_step_compressed_chain_setup_cache_key(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
) -> Result<[u8; 32], Rv64imMainRecursionStepSpartanError> {
    let mut tr =
        Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_step_spartan/compressed_chain_setup_cache_key");
    tr.append_message(
        b"neo.fold.next/rv64im/main_recursion_step_spartan/compressed_chain_setup_cache_key/version",
        b"v3",
    );
    tr.append_message(
        b"neo.fold.next/rv64im/main_recursion_step_spartan/compressed_chain_setup_cache_key/chain_shape",
        &chain_shape.expected_digest(),
    );
    Ok(tr.digest32())
}

impl Rv64imMainRecursionStepCompressedChainCircuit {
    fn canonical_statement(&self) -> Rv64imMainRecursionStepSpartanStatement {
        build_rv64im_main_recursion_step_spartan_statement(&self.backend_relations)
            .expect("compressed-chain circuit must be built from a canonical backend relation chain")
    }

    fn expected_public_values(&self) -> Vec<SpartanF> {
        main_recursion_step_public_values(&self.canonical_statement())
    }
}

pub fn debug_check_rv64im_main_recursion_step_spartan_compressed_chain_circuit(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    for (step_index, relation) in backend_relations.iter().enumerate() {
        debug_check_rv64im_main_recursion_step_spartan_embedded_body(spartan_shape, relation).map_err(|err| {
            Rv64imMainRecursionStepSpartanError::Prepare(format!(
                "compressed-chain embedded step body failed before wrapper synthesis at step {step_index}: {err}"
            ))
        })?;
    }
    let chain_shape =
        build_rv64im_main_recursion_step_spartan_compressed_chain_shape(spartan_shape, backend_relations)?;
    let circuit =
        build_rv64im_main_recursion_step_compressed_chain_circuit_from_relations(&chain_shape, backend_relations)?;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if !cs.is_satisfied() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            cs.which_is_unsatisfied()
                .map(|name| name.to_string())
                .unwrap_or_else(|| "unknown unsatisfied compressed-chain constraint".to_string()),
        ));
    }
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_step_spartan_compressed_chain_shape_only_setup(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_shape_only_circuit(chain_shape)?;
    let _ = setup_rv64im_main_recursion_step_spartan_compressed_chain_cached(chain_shape, None)?;
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_step_spartan_compressed_chain_shape_only_circuit(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let backend_relations = build_dummy_backend_relation_chain(chain_shape)?;
    for (step_index, relation) in backend_relations.iter().enumerate() {
        debug_check_rv64im_main_recursion_step_spartan_embedded_body(&chain_shape.spartan_shape, relation).map_err(
            |err| {
                Rv64imMainRecursionStepSpartanError::Prepare(format!(
                    "shape-only compressed-chain embedded step body failed before wrapper synthesis at step {step_index}: {err}"
                ))
            },
        )?;
    }
    let circuit =
        build_rv64im_main_recursion_step_compressed_chain_circuit_from_relations(chain_shape, &backend_relations)?;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if !cs.is_satisfied() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            cs.which_is_unsatisfied()
                .map(|name| name.to_string())
                .unwrap_or_else(|| "unknown unsatisfied shape-only compressed-chain constraint".to_string()),
        ));
    }
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_step_spartan_compressed_chain_parity(
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let initial_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state();
    let initial_state_claims = backend_relations
        .first()
        .map(|relation| relation.payload.state_in_claims.clone())
        .unwrap_or_else(|| initial_state.carry.main.claims.clone());
    let expected_initial_carry_state_digest = rv64im_chunk_step_recursive_carry_state_digest(
        &initial_state_claims,
        &initial_state.transcript,
        initial_state.carry.terminal_handle.0,
    );
    if let Some(first) = backend_relations.first() {
        let actual_initial_carry_state_digest = rv64im_chunk_step_recursive_carry_state_digest(
            &first.payload.state_in_claims,
            &first.f_prime_advice.running_state().transcript,
            first.f_prime_advice.running_state().carry.terminal_handle.0,
        );
        if actual_initial_carry_state_digest != expected_initial_carry_state_digest {
            return Err(Rv64imMainRecursionStepSpartanError::Prepare(
                "rv64im main recursion compressed-chain parity mismatch: first step carry-state input does not match the padded initial seed".into(),
            ));
        }
    }

    let mut expected_step_statement_chain = crate::rv64im::chunk_step_ivc::rv64im_step_statement_chain_digest_init();
    let mut expected_bridge_handoff_chain = crate::rv64im::chunk_step_ivc::rv64im_bridge_handoff_chain_digest_init();
    let mut expected_carry_state_out_digest = None;
    let mut expected_folded_accumulator_digest =
        crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(&initial_state.carry);
    let mut expected_terminal_handle_digest = initial_state.carry.terminal_handle.0;
    for (step_index, relation) in backend_relations.iter().enumerate() {
        if relation.f_prime_advice.chunk_count_in() != step_index as u64 {
            return Err(Rv64imMainRecursionStepSpartanError::Prepare(
                "rv64im main recursion compressed-chain parity mismatch: chunk_count_in does not match the chain position"
                    .into(),
            ));
        }
        let actual_carry_state_in_digest = rv64im_chunk_step_recursive_carry_state_digest(
            &relation.payload.state_in_claims,
            &relation.f_prime_advice.running_state().transcript,
            relation
                .f_prime_advice
                .running_state()
                .carry
                .terminal_handle
                .0,
        );
        if let Some(expected_digest) = expected_carry_state_out_digest {
            if actual_carry_state_in_digest != expected_digest {
                return Err(Rv64imMainRecursionStepSpartanError::Prepare(
                    "rv64im main recursion compressed-chain parity mismatch: carry-state input does not match the previous padded carry-state output"
                        .into(),
                ));
            }
        }
        if relation.f_prime_advice.step_statement_chain_digest_in() != expected_step_statement_chain {
            return Err(Rv64imMainRecursionStepSpartanError::Prepare(
                "rv64im main recursion compressed-chain parity mismatch: step-statement chain input does not match the folded prefix"
                    .into(),
            ));
        }
        if relation.f_prime_advice.bridge_handoff_chain_digest_in() != expected_bridge_handoff_chain {
            return Err(Rv64imMainRecursionStepSpartanError::Prepare(
                "rv64im main recursion compressed-chain parity mismatch: bridge-handoff chain input does not match the folded prefix"
                    .into(),
            ));
        }
        ensure_main_recursion_step_spartan_statement_binding(relation).map_err(|err| {
            Rv64imMainRecursionStepSpartanError::Prepare(format!(
                "rv64im main recursion compressed-chain parity mismatch: {err}"
            ))
        })?;
        expected_carry_state_out_digest = Some(rv64im_chunk_step_recursive_carry_state_digest(
            &relation.payload.state_out_claims,
            &relation.payload.fixed_transcript_out,
            relation
                .f_prime_advice
                .fresh_state_out()
                .carry
                .terminal_handle
                .0,
        ));
        expected_folded_accumulator_digest = relation.spartan_statement.folded_accumulator_digest;
        expected_step_statement_chain = relation.spartan_statement.step_statement_chain_digest;
        expected_bridge_handoff_chain = relation.spartan_statement.bridge_handoff_chain_digest;
        expected_terminal_handle_digest = relation.spartan_statement.terminal_handle_digest;
    }
    let expected_statement = build_rv64im_main_recursion_step_spartan_statement(backend_relations)?;
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs()
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let canonical_final_statement = build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
        &vk_fs,
        backend_relations.len() as u64,
        expected_folded_accumulator_digest,
        expected_step_statement_chain,
        expected_bridge_handoff_chain,
        expected_terminal_handle_digest,
    );
    if expected_statement != canonical_final_statement {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion compressed-chain parity mismatch: final chain statement does not match the canonical native F' image".into(),
        ));
    }
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_step_spartan_compressed_chain_public_io(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let statement = build_rv64im_main_recursion_step_spartan_statement(backend_relations)?;
    let circuit =
        build_rv64im_main_recursion_step_compressed_chain_circuit_from_relations(chain_shape, backend_relations)?;
    let expected = main_recursion_step_public_values(&statement);
    if expected.len() != RV64IM_MAIN_RECURSION_STEP_PUBLIC_IO_ARITY {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(format!(
            "rv64im main recursion compressed-chain canonical public IO arity drifted: statement emits {} fields, expected {}",
            expected.len(),
            RV64IM_MAIN_RECURSION_STEP_PUBLIC_IO_ARITY,
        )));
    }
    let actual = circuit.expected_public_values();
    if actual != expected {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(format!(
            "rv64im main recursion compressed-chain public IO mismatch: circuit exposed {} fields, statement requires {}",
            actual.len(),
            expected.len(),
        )));
    }
    Ok(())
}

pub fn debug_measure_rv64im_main_recursion_step_spartan_compressed_chain_circuit_shape(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<Rv64imMainRecursionStepSpartanCircuitShape, Rv64imMainRecursionStepSpartanError> {
    let circuit =
        build_rv64im_main_recursion_step_compressed_chain_circuit_from_relations(chain_shape, backend_relations)?;
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

pub fn debug_check_rv64im_main_recursion_step_spartan_compressed_chain_statement_binding(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
    statement: &Rv64imMainRecursionStepSpartanStatement,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let canonical_statement = build_rv64im_main_recursion_step_spartan_statement(backend_relations)?;
    if statement != &canonical_statement {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion compressed-chain circuit requires the canonical final statement derived from the live backend relations"
                .into(),
        ));
    }
    let _ = build_rv64im_main_recursion_step_compressed_chain_circuit_from_relations(chain_shape, backend_relations)?;
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_step_spartan_shape_only_chain_parity(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let backend_relations = build_dummy_backend_relation_chain(chain_shape)?;
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_parity(&backend_relations).map_err(|err| {
        Rv64imMainRecursionStepSpartanError::Prepare(format!(
            "shape-only dummy compressed-chain parity failed before setup: {err}"
        ))
    })
}

impl SpartanCircuit<Rv64imSpartan2DeciderEngine> for Rv64imMainRecursionStepCompressedChainCircuit {
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
        let x_out_input = next_public_digest(&public_inputs, &mut public_cursor, "x_out")?;
        let folded_accumulator_out_digest_input =
            next_public_digest(&public_inputs, &mut public_cursor, "folded_accumulator_out_digest")?;
        let step_statement_chain_out_input =
            next_public_digest(&public_inputs, &mut public_cursor, "step_statement_chain_digest_out")?;
        let bridge_handoff_chain_out_input =
            next_public_digest(&public_inputs, &mut public_cursor, "bridge_handoff_chain_digest_out")?;
        let terminal_handle_out_input =
            next_public_digest(&public_inputs, &mut public_cursor, "terminal_handle_digest_out")?;
        let initial_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state();
        let initial_state_claims = self
            .backend_relations
            .first()
            .map(|relation| relation.payload.state_in_claims.clone())
            .unwrap_or_else(|| initial_state.carry.main.claims.clone());
        let initial_carry_state_digest = digest_const_inputs(
            &mut cs.namespace(|| "initial_carry_state_digest"),
            rv64im_chunk_step_recursive_carry_state_digest(
                &initial_state_claims,
                &initial_state.transcript,
                initial_state.carry.terminal_handle.0,
            ),
            "initial_carry_state_digest",
        )?;
        let initial_folded_accumulator_digest = digest_const_inputs(
            &mut cs.namespace(|| "initial_folded_accumulator_digest"),
            crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(&initial_state.carry),
            "initial_folded_accumulator_digest",
        )?;
        let initial_step_statement_chain = digest_const_inputs(
            &mut cs.namespace(|| "initial_step_statement_chain_seed"),
            crate::rv64im::chunk_step_ivc::rv64im_step_statement_chain_digest_init(),
            "initial_step_statement_chain_seed",
        )?;
        let initial_bridge_handoff_chain = digest_const_inputs(
            &mut cs.namespace(|| "initial_bridge_handoff_chain_seed"),
            crate::rv64im::chunk_step_ivc::rv64im_bridge_handoff_chain_digest_init(),
            "initial_bridge_handoff_chain_seed",
        )?;
        let initial_terminal_handle = digest_const_inputs(
            &mut cs.namespace(|| "initial_terminal_handle"),
            initial_state.carry.terminal_handle.0,
            "initial_terminal_handle",
        )?;
        let mut final_folded_accumulator_digest_value = digest32_as_spartan_fields(
            crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(&initial_state.carry),
        );
        let mut final_terminal_handle_value = digest32_as_spartan_fields(initial_state.carry.terminal_handle.0);

        let mut previous_step: Option<Rv64imMainRecursionStepPublicVar> = None;
        for (step_index, relation) in self.backend_relations.iter().enumerate() {
            let step_circuit = build_rv64im_main_recursion_step_circuit(&self.chain_shape.spartan_shape, relation)
                .map_err(|_| SynthesisError::Unsatisfiable)?;
            let relation_public_inputs = alloc_private_field_values(
                &mut cs.namespace(|| format!("step_{step_index}_public_inputs")),
                &step_circuit.expected_public_values(),
                &format!("step_{step_index}_public_inputs"),
            )?;
            let mut relation_public_cursor = 0usize;
            let step_public = synthesize_rv64im_main_recursion_step_body(
                &step_circuit,
                &mut cs.namespace(|| format!("step_{step_index}")),
                &relation_public_inputs,
                &mut relation_public_cursor,
            )?;
            if relation_public_cursor != relation_public_inputs.len() {
                mark_unsatisfied(
                    &mut cs.namespace(|| format!("step_{step_index}_public_cursor_len_mismatch")),
                    &format!("step_{step_index}_public_cursor_len_mismatch"),
                )?;
            }
            if let Some(previous) = previous_step.as_ref() {
                enforce_digest_eq(
                    &mut cs.namespace(|| format!("step_{step_index}_accumulator_chain")),
                    &previous.carry_state_out_digest,
                    &step_public.carry_state_in_digest,
                    &format!("step_{step_index}_accumulator_chain"),
                )?;
                enforce_digest_eq(
                    &mut cs.namespace(|| format!("step_{step_index}_statement_chain")),
                    &previous.step_statement_chain_digest_out,
                    &step_public.step_statement_chain_digest_in,
                    &format!("step_{step_index}_statement_chain"),
                )?;
                enforce_digest_eq(
                    &mut cs.namespace(|| format!("step_{step_index}_bridge_chain")),
                    &previous.bridge_handoff_chain_digest_out,
                    &step_public.bridge_handoff_chain_digest_in,
                    &format!("step_{step_index}_bridge_chain"),
                )?;
            } else {
                cs.enforce(
                    || "initial_chunk_index_eq",
                    |lc| lc + step_public.chunk_index.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc,
                );
                enforce_digest_eq(
                    &mut cs.namespace(|| "initial_carry_state_chain"),
                    &initial_carry_state_digest,
                    &step_public.carry_state_in_digest,
                    "initial_carry_state_chain",
                )?;
                enforce_digest_eq(
                    &mut cs.namespace(|| "initial_step_statement_chain"),
                    &initial_step_statement_chain,
                    &step_public.step_statement_chain_digest_in,
                    "initial_step_statement_chain",
                )?;
                enforce_digest_eq(
                    &mut cs.namespace(|| "initial_bridge_handoff_chain"),
                    &initial_bridge_handoff_chain,
                    &step_public.bridge_handoff_chain_digest_in,
                    "initial_bridge_handoff_chain",
                )?;
            }
            final_folded_accumulator_digest_value =
                digest32_as_spartan_fields(relation.spartan_statement.folded_accumulator_digest);
            final_terminal_handle_value = digest32_as_spartan_fields(relation.spartan_statement.terminal_handle_digest);
            previous_step = Some(step_public);
        }

        let final_folded_accumulator_digest = previous_step
            .as_ref()
            .map(|step| step.folded_accumulator_out_digest.clone())
            .unwrap_or(initial_folded_accumulator_digest);
        let final_step_statement_chain = previous_step
            .as_ref()
            .map(|step| step.step_statement_chain_digest_out.clone())
            .unwrap_or(initial_step_statement_chain);
        let final_bridge_handoff_chain = previous_step
            .as_ref()
            .map(|step| step.bridge_handoff_chain_digest_out.clone())
            .unwrap_or(initial_bridge_handoff_chain);
        let final_terminal_handle = previous_step
            .as_ref()
            .map(|step| step.terminal_handle_digest_out.clone())
            .unwrap_or(initial_terminal_handle);
        let initial_z = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state()
            .carry
            .terminal_handle
            .0;
        let final_z_0 = digest_const_inputs(&mut cs.namespace(|| "final_z_0"), initial_z, "final_z_0")?;
        let final_x_out = main_recursion_x_out_circuit(
            &mut cs.namespace(|| "final_x_out"),
            "final_x_out",
            self.chain_shape.step_shapes.len() as u64,
            &final_z_0,
            &digest32_as_spartan_fields(initial_z),
            &final_terminal_handle,
            &final_terminal_handle_value,
            crate::rv64im::main_recursion::RV64IM_MAIN_RECURSION_TRIVIAL_PC,
            &final_folded_accumulator_digest,
            &final_folded_accumulator_digest_value,
        )?;
        enforce_digest_eq(
            &mut cs.namespace(|| "x_out_output_eq"),
            &x_out_input,
            &final_x_out,
            "x_out_output_eq",
        )?;
        enforce_digest_eq(
            &mut cs.namespace(|| "folded_accumulator_output_eq"),
            &folded_accumulator_out_digest_input,
            &final_folded_accumulator_digest,
            "folded_accumulator_output_eq",
        )?;
        enforce_digest_eq(
            &mut cs.namespace(|| "step_statement_chain_output_eq"),
            &step_statement_chain_out_input,
            &final_step_statement_chain,
            "step_statement_chain_output_eq",
        )?;
        enforce_digest_eq(
            &mut cs.namespace(|| "bridge_handoff_chain_output_eq"),
            &bridge_handoff_chain_out_input,
            &final_bridge_handoff_chain,
            "bridge_handoff_chain_output_eq",
        )?;
        enforce_digest_eq(
            &mut cs.namespace(|| "terminal_handle_output_eq"),
            &terminal_handle_out_input,
            &final_terminal_handle,
            "terminal_handle_output_eq",
        )?;
        if public_cursor != public_inputs.len() {
            mark_unsatisfied(
                &mut cs.namespace(|| "compressed_chain_public_cursor_len_mismatch"),
                "compressed_chain_public_cursor_len_mismatch",
            )?;
        }
        Ok(())
    }
}

fn build_rv64im_main_recursion_step_compressed_chain_circuit(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
) -> Result<Rv64imMainRecursionStepCompressedChainCircuit, Rv64imMainRecursionStepSpartanError> {
    let backend_relations = build_dummy_backend_relation_chain(chain_shape)?;
    Ok(Rv64imMainRecursionStepCompressedChainCircuit {
        chain_shape: chain_shape.clone(),
        backend_relations,
    })
}

pub(super) fn build_rv64im_main_recursion_step_compressed_chain_circuit_from_relations(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<Rv64imMainRecursionStepCompressedChainCircuit, Rv64imMainRecursionStepSpartanError> {
    if backend_relations.len() != chain_shape.step_shapes.len() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion compressed-chain circuit requires one live backend relation per structural chain step"
                .into(),
        ));
    }
    for (step_shape, relation) in chain_shape.step_shapes.iter().zip(backend_relations.iter()) {
        if &relation.payload.step_shape != step_shape {
            return Err(Rv64imMainRecursionStepSpartanError::Prepare(
                "rv64im main recursion compressed-chain circuit received backend relations with a mismatched live step shape"
                    .into(),
            ));
        }
        let _ = build_rv64im_main_recursion_step_circuit(&chain_shape.spartan_shape, relation)?;
    }
    let _ = build_rv64im_main_recursion_step_spartan_statement(backend_relations)?;
    Ok(Rv64imMainRecursionStepCompressedChainCircuit {
        chain_shape: chain_shape.clone(),
        backend_relations: backend_relations.to_vec(),
    })
}

fn build_dummy_backend_relation_chain(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
) -> Result<Vec<Rv64imMainRecursionFPrimeBackendRelation>, Rv64imMainRecursionStepSpartanError> {
    let initial_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state();
    let mut backend_relations = Vec::with_capacity(chain_shape.step_shapes.len());
    let mut step_statement_chain_digest = crate::rv64im::chunk_step_ivc::rv64im_step_statement_chain_digest_init();
    let mut bridge_handoff_chain_digest = crate::rv64im::chunk_step_ivc::rv64im_bridge_handoff_chain_digest_init();
    let mut running_state = initial_state;

    for (step_index, step_shape) in chain_shape.step_shapes.iter().enumerate() {
        let relation = dummy_backend_relation_from_chain_step(
            &chain_shape.spartan_shape,
            step_shape,
            step_index as u64,
            step_statement_chain_digest,
            bridge_handoff_chain_digest,
            &running_state,
        )?;
        step_statement_chain_digest = relation.spartan_statement.step_statement_chain_digest;
        bridge_handoff_chain_digest = relation.spartan_statement.bridge_handoff_chain_digest;
        running_state = relation.f_prime_advice.fresh_state_out().clone();
        backend_relations.push(relation);
    }
    Ok(backend_relations)
}

fn diagnose_compressed_chain_setup_failure(circuit: &Rv64imMainRecursionStepCompressedChainCircuit) -> String {
    if let Err(err) = debug_check_rv64im_main_recursion_step_spartan_compressed_chain_parity(&circuit.backend_relations)
    {
        return format!("shape-only dummy compressed-chain parity failed before setup fallback: {err}");
    }
    if let Err(err) = debug_check_rv64im_main_recursion_step_spartan_compressed_chain_wrapper_only(
        &circuit.chain_shape,
        &circuit.backend_relations,
    ) {
        return format!("compressed-chain wrapper-only reduction failed before setup fallback: {err}");
    }
    for (step_index, relation) in circuit.backend_relations.iter().enumerate() {
        if let Err(err) = debug_check_rv64im_main_recursion_x_out_gadget_parity(relation) {
            return format!(
                "shape-only dummy step {step_index} x_out gadget parity failed before setup fallback: {err}"
            );
        }
        if let Err(err) =
            crate::rv64im::main_relation_spartan::debug_check_rv64im_chunk_step_recursive_effective_chunk_trace_matches_native(
                relation,
            )
        {
            return format!(
                "shape-only dummy step {step_index} replay surface drifted from the native trace before setup fallback: {err}"
            );
        }
    }
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    match circuit.synthesize(&mut cs, &[], &[], None) {
        Ok(()) if cs.is_satisfied() => "test constraint system remained satisfied during setup fallback".into(),
        Ok(()) => cs
            .which_is_unsatisfied()
            .map(|name| format!("setup fallback unsatisfied at {name}"))
            .unwrap_or_else(|| "setup fallback found an unknown unsatisfied compressed-chain constraint".into()),
        Err(err) => format!("setup fallback synthesize failed: {err}"),
    }
}

fn setup_rv64im_main_recursion_step_spartan_compressed_chain_cached(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
    backend_relations: Option<&[Rv64imMainRecursionFPrimeBackendRelation]>,
) -> Result<Rv64imMainRecursionStepSpartanKeyPair, Rv64imMainRecursionStepSpartanError> {
    let cache_key = rv64im_main_recursion_step_compressed_chain_setup_cache_key(chain_shape)?;
    let cache = RV64IM_MAIN_RECURSION_STEP_COMPRESSED_CHAIN_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| {
            Rv64imMainRecursionStepSpartanError::Setup(
                "rv64im main recursion step compressed-chain setup cache poisoned".into(),
            )
        })?
        .get(&cache_key)
        .cloned()
    {
        return Ok(keys);
    }
    let circuit = if let Some(backend_relations) = backend_relations {
        build_rv64im_main_recursion_step_compressed_chain_circuit_from_relations(chain_shape, backend_relations)?
    } else {
        build_rv64im_main_recursion_step_compressed_chain_circuit(chain_shape)?
    };
    let keys = Arc::new(Rv64imSpartan2DeciderSnark::setup(circuit.clone()).map_err(|err| {
        let detail = diagnose_compressed_chain_setup_failure(&circuit);
        Rv64imMainRecursionStepSpartanError::Setup(format!("{err}; {detail}"))
    })?);
    cache
        .lock()
        .map_err(|_| {
            Rv64imMainRecursionStepSpartanError::Setup(
                "rv64im main recursion step compressed-chain setup cache poisoned".into(),
            )
        })?
        .insert(cache_key, keys.clone());
    Ok(keys)
}

pub fn prove_rv64im_main_recursion_step_spartan_compressed_chain(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<Rv64imMainRecursionStepSpartanCompressedChainProof, Rv64imMainRecursionStepSpartanError> {
    let chain_shape =
        build_rv64im_main_recursion_step_spartan_compressed_chain_shape(spartan_shape, backend_relations)?;
    let circuit =
        build_rv64im_main_recursion_step_compressed_chain_circuit_from_relations(&chain_shape, backend_relations)?;
    let keys = setup_rv64im_main_recursion_step_spartan_compressed_chain_cached(&chain_shape, Some(backend_relations))?;
    let (pk, _) = &*keys;
    let prep = Rv64imSpartan2DeciderSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let proof = Rv64imSpartan2DeciderSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prove(err.to_string()))?;
    let snark_data =
        bincode::serialize(&proof).map_err(|err| Rv64imMainRecursionStepSpartanError::Encode(err.to_string()))?;
    Ok(Rv64imMainRecursionStepSpartanCompressedChainProof { snark_data })
}

pub fn debug_profile_rv64im_main_recursion_step_spartan_compressed_chain_prove_stages(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<Rv64imMainRecursionStepSpartanCompressedChainProveMetrics, Rv64imMainRecursionStepSpartanError> {
    let chain_shape =
        build_rv64im_main_recursion_step_spartan_compressed_chain_shape(spartan_shape, backend_relations)?;
    let circuit =
        build_rv64im_main_recursion_step_compressed_chain_circuit_from_relations(&chain_shape, backend_relations)?;

    let started = Instant::now();
    let keys = setup_rv64im_main_recursion_step_spartan_compressed_chain_cached(&chain_shape, Some(backend_relations))?;
    let setup_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let (pk, _) = &*keys;
    let started = Instant::now();
    let prep = Rv64imSpartan2DeciderSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let prep_prove_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let started = Instant::now();
    let proof = Rv64imSpartan2DeciderSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prove(err.to_string()))?;
    let prove_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let started = Instant::now();
    let snark_data =
        bincode::serialize(&proof).map_err(|err| Rv64imMainRecursionStepSpartanError::Encode(err.to_string()))?;
    let serialize_ms = started.elapsed().as_secs_f64() * 1_000.0;

    Ok(Rv64imMainRecursionStepSpartanCompressedChainProveMetrics {
        setup_ms,
        prep_prove_ms,
        prove_ms,
        serialize_ms,
        snark_bytes: snark_data.len(),
    })
}

pub fn verify_rv64im_main_recursion_step_spartan_compressed_chain(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
    statement: &Rv64imMainRecursionStepSpartanStatement,
    proof: &Rv64imMainRecursionStepSpartanCompressedChainProof,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let keys = setup_rv64im_main_recursion_step_spartan_compressed_chain_cached(chain_shape, None)?;
    let (_, vk) = &*keys;
    let proof: Rv64imSpartan2DeciderSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Verify(err.to_string()))?;
    if public_values != main_recursion_step_public_values(statement) {
        return Err(Rv64imMainRecursionStepSpartanError::PublicIoMismatch);
    }
    Ok(())
}
