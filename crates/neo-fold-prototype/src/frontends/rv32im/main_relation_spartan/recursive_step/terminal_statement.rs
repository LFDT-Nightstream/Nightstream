//! Owns terminal verified-step statement digests for the recursive step.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::rv32im::construction2::Rv32imMainRecursionConstruction2VerifiedStepStatement;
use crate::rv32im::main_relation_spartan::chunk_step::Rv32imMainRecursionFPrimeBackendRelation;
use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::transcript::Poseidon2TranscriptCircuit;

use super::Rv32imMainRecursionStepSpartanError;

pub(super) fn build_terminal_f_prime_verified_step_statement(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imMainRecursionConstruction2VerifiedStepStatement, Rv32imMainRecursionStepSpartanError> {
    let step_lo = backend_relation.payload.handoff.public_chunk.start_index as u64;
    let public_step_count = backend_relation.payload.handoff.public_chunk.steps.len() as u64;
    let step_hi = step_lo.checked_add(public_step_count).ok_or_else(|| {
        Rv32imMainRecursionStepSpartanError::Prepare(
            "rv32im terminal F' verified-step statement step span overflow".into(),
        )
    })?;
    let chunk_relation_digest = crate::rv32im::chunk::transition::rv32im_chunk_relation_digest_from_fold_digest(
        backend_relation.payload.handoff.public_chunk_digest,
        backend_relation.payload.pi_ccs.replay.header_digest,
        backend_relation.payload.handoff.bridge_handoff_digest,
    );
    Ok(Rv32imMainRecursionConstruction2VerifiedStepStatement {
        chunk_index: backend_relation.f_prime_advice.chunk_count_in(),
        step_lo,
        step_hi,
        halted_out: backend_relation.f_prime_advice.bridge_handoff_halted_out(),
        state_in: backend_relation
            .f_prime_advice
            .running_state()
            .carry
            .terminal_handle
            .0,
        state_out: backend_relation
            .f_prime_advice
            .fresh_state_out()
            .carry
            .terminal_handle
            .0,
        public_chunk_digest: backend_relation.payload.handoff.public_chunk_digest,
        chunk_relation_digest,
    })
}

pub(super) fn build_terminal_f_prime_verified_step_statement_digest(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<[u8; 32], Rv32imMainRecursionStepSpartanError> {
    Ok(build_terminal_f_prime_verified_step_statement(backend_relation)?.expected_digest())
}

pub(super) fn construction2_verified_step_statement_digest_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    label: &str,
    chunk_index_halves: &[AllocatedNum<SpartanF>; 2],
    chunk_index_half_values: &[SpartanF; 2],
    step_lo_halves: &[AllocatedNum<SpartanF>; 2],
    step_lo_half_values: &[SpartanF; 2],
    step_hi_halves: &[AllocatedNum<SpartanF>; 2],
    step_hi_half_values: &[SpartanF; 2],
    halted_out_halves: &[AllocatedNum<SpartanF>; 2],
    halted_out_half_values: &[SpartanF; 2],
    state_in: &[AllocatedNum<SpartanF>; 4],
    state_in_value: &[SpartanF; 4],
    state_out: &[AllocatedNum<SpartanF>; 4],
    state_out_value: &[SpartanF; 4],
    public_chunk_digest: &[AllocatedNum<SpartanF>; 4],
    public_chunk_digest_value: &[SpartanF; 4],
    chunk_relation_digest: &[AllocatedNum<SpartanF>; 4],
    chunk_relation_digest_value: &[SpartanF; 4],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut transcript = Poseidon2TranscriptCircuit::new(
        cs.namespace(|| format!("{label}_init")),
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement",
    )?;
    transcript.append_message(
        cs.namespace(|| format!("{label}_version")),
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/version",
        b"v2",
    )?;
    let meta_halves = [
        chunk_index_halves[0].clone(),
        chunk_index_halves[1].clone(),
        step_lo_halves[0].clone(),
        step_lo_halves[1].clone(),
        step_hi_halves[0].clone(),
        step_hi_halves[1].clone(),
        halted_out_halves[0].clone(),
        halted_out_halves[1].clone(),
    ];
    let meta_half_values = [
        chunk_index_half_values[0],
        chunk_index_half_values[1],
        step_lo_half_values[0],
        step_lo_half_values[1],
        step_hi_half_values[0],
        step_hi_half_values[1],
        halted_out_half_values[0],
        halted_out_half_values[1],
    ];
    transcript.append_u64_halves(
        cs.namespace(|| format!("{label}_meta")),
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/meta",
        &meta_halves,
        &meta_half_values,
        4,
    )?;
    transcript.append_fields(
        cs.namespace(|| format!("{label}_state_in")),
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/state_in",
        state_in,
        state_in_value,
    )?;
    transcript.append_fields(
        cs.namespace(|| format!("{label}_state_out")),
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/state_out",
        state_out,
        state_out_value,
    )?;
    transcript.append_fields(
        cs.namespace(|| format!("{label}_public_chunk_digest")),
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/public_chunk_digest",
        public_chunk_digest,
        public_chunk_digest_value,
    )?;
    transcript.append_fields(
        cs.namespace(|| format!("{label}_chunk_relation_digest")),
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/chunk_relation_digest",
        chunk_relation_digest,
        chunk_relation_digest_value,
    )?;
    transcript.digest32(cs.namespace(|| format!("{label}_digest")))
}
