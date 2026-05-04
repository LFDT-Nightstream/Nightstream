//! Owns fixed-shape recursive step-handle digest gadgets.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;

use crate::rv32im::main_relation_spartan::digest32_as_spartan_fields;
use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::transcript::hash_field_linear_combinations_raw;

pub(in crate::rv32im::main_relation_spartan) fn fixed_shape_recursive_step_handle_digest_circuit<
    CS: ConstraintSystem<SpartanF>,
>(
    cs: &mut CS,
    label: &str,
    previous_handle: &[AllocatedNum<SpartanF>; 4],
    previous_handle_values: &[SpartanF; 4],
    next_chunk_count_halves: &[AllocatedNum<SpartanF>; 2],
    next_chunk_count: u64,
    chunk_start_index: &AllocatedNum<SpartanF>,
    chunk_start_index_value: SpartanF,
    public_step_count: &AllocatedNum<SpartanF>,
    public_step_count_value: SpartanF,
    chunk_relation_digest: &[u8; 32],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let chunk_relation_digest_fields = digest32_as_spartan_fields(*chunk_relation_digest);
    let mut field_terms = Vec::with_capacity(4 + 3 + chunk_relation_digest_fields.len());
    let mut field_constants = Vec::with_capacity(field_terms.capacity());
    let mut field_values = Vec::with_capacity(field_terms.capacity());

    for (lane, value) in previous_handle.iter().zip(previous_handle_values.iter()) {
        field_terms.push(vec![(lane.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_values.push(*value);
    }

    let chunk_index_value = SpartanF::from_canonical_u64(
        next_chunk_count
            .checked_sub(1)
            .ok_or(SynthesisError::Unsatisfiable)?,
    );
    field_terms.push(vec![
        (next_chunk_count_halves[0].get_variable(), SpartanF::ONE),
        (
            next_chunk_count_halves[1].get_variable(),
            SpartanF::from_canonical_u64(1u64 << 32),
        ),
    ]);
    field_constants.push(-SpartanF::ONE);
    field_values.push(chunk_index_value);

    for (lane, value) in [
        (chunk_start_index, chunk_start_index_value),
        (public_step_count, public_step_count_value),
    ] {
        field_terms.push(vec![(lane.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_values.push(value);
    }

    for value in chunk_relation_digest_fields {
        field_terms.push(Vec::new());
        field_constants.push(value);
        field_values.push(value);
    }

    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

pub(in crate::rv32im::main_relation_spartan) fn fixed_shape_recursive_step_handle_digest_circuit_from_vars<
    CS: ConstraintSystem<SpartanF>,
>(
    cs: &mut CS,
    label: &str,
    previous_handle: &[AllocatedNum<SpartanF>; 4],
    previous_handle_values: &[SpartanF; 4],
    next_chunk_count_halves: &[AllocatedNum<SpartanF>; 2],
    next_chunk_count: u64,
    chunk_start_index: &AllocatedNum<SpartanF>,
    chunk_start_index_value: SpartanF,
    public_step_count: &AllocatedNum<SpartanF>,
    public_step_count_value: SpartanF,
    chunk_relation_digest: &[AllocatedNum<SpartanF>; 4],
    chunk_relation_digest_values: &[SpartanF; 4],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut field_terms = Vec::with_capacity(4 + 3 + chunk_relation_digest.len());
    let mut field_constants = Vec::with_capacity(field_terms.capacity());
    let mut field_values = Vec::with_capacity(field_terms.capacity());

    for (lane, value) in previous_handle.iter().zip(previous_handle_values.iter()) {
        field_terms.push(vec![(lane.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_values.push(*value);
    }

    let chunk_index_value = SpartanF::from_canonical_u64(
        next_chunk_count
            .checked_sub(1)
            .ok_or(SynthesisError::Unsatisfiable)?,
    );
    field_terms.push(vec![
        (next_chunk_count_halves[0].get_variable(), SpartanF::ONE),
        (
            next_chunk_count_halves[1].get_variable(),
            SpartanF::from_canonical_u64(1u64 << 32),
        ),
    ]);
    field_constants.push(-SpartanF::ONE);
    field_values.push(chunk_index_value);

    for (lane, value) in [
        (chunk_start_index, chunk_start_index_value),
        (public_step_count, public_step_count_value),
    ] {
        field_terms.push(vec![(lane.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_values.push(value);
    }

    for (lane, value) in chunk_relation_digest
        .iter()
        .zip(chunk_relation_digest_values.iter())
    {
        field_terms.push(vec![(lane.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_values.push(*value);
    }

    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}
