//! Digest transition circuits for direct-CCS public state.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::finalize::digest32_as_fields;
use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::transcript::hash_field_linear_combinations_raw;
use neo_math::F;

use super::super::public_io::{
    direct_terminal_current_boundary_digest_range, direct_terminal_public_trace_out_digest_range,
    direct_terminal_x_out_digest_range, enforce_digest_eq_constant, enforce_digest_fields_public_io,
};
use super::fields::{
    direct_domain_spartan_fields, field_to_spartan, push_constant_spartan_fields, spartan_one, spartan_zero,
    u64_halves_as_spartan_fields,
};

pub(crate) fn enforce_direct_public_trace_transition<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    public_inputs: &[AllocatedNum<SpartanF>],
    public_trace_in_digest: [u8; 32],
    latest_chunk_digest: &[AllocatedNum<SpartanF>; 4],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut field_terms = Vec::new();
    let mut field_constants = Vec::new();
    let mut field_values = Vec::new();
    push_constant_spartan_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        direct_domain_spartan_fields(b"neo.fold.next/direct_ccs/public_trace_update/v1"),
    );
    for value in digest32_as_fields(public_trace_in_digest).iter().copied() {
        field_terms.push(Vec::new());
        field_constants.push(field_to_spartan(value));
        field_values.push(field_to_spartan(value));
    }
    for value in latest_chunk_digest {
        field_terms.push(vec![(value.get_variable(), spartan_one())]);
        field_constants.push(spartan_zero());
        field_values.push(value.get_value().unwrap_or(spartan_zero()));
    }
    let digest = hash_field_linear_combinations_raw(
        cs.namespace(|| "direct_public_trace_transition"),
        &field_terms,
        &field_constants,
        &field_values,
    )?;
    enforce_digest_fields_public_io(
        &mut cs.namespace(|| "direct_public_trace_out_digest_public"),
        &digest,
        public_inputs,
        direct_terminal_public_trace_out_digest_range(),
        "direct_public_trace_out_digest_public",
    )?;
    Ok(digest)
}

pub(crate) fn enforce_direct_current_boundary_transition<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    public_inputs: &[AllocatedNum<SpartanF>],
    boundary_in_digest: [u8; 32],
    latest_chunk_digest: &[AllocatedNum<SpartanF>; 4],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut field_terms = Vec::new();
    let mut field_constants = Vec::new();
    let mut field_values = Vec::new();
    push_constant_spartan_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        direct_domain_spartan_fields(b"neo.fold.next/direct_ccs/current_boundary_update/v1"),
    );
    for value in digest32_as_fields(boundary_in_digest).iter().copied() {
        field_terms.push(Vec::new());
        field_constants.push(field_to_spartan(value));
        field_values.push(field_to_spartan(value));
    }
    for value in latest_chunk_digest {
        field_terms.push(vec![(value.get_variable(), spartan_one())]);
        field_constants.push(spartan_zero());
        field_values.push(value.get_value().unwrap_or(spartan_zero()));
    }
    let digest = hash_field_linear_combinations_raw(
        cs.namespace(|| "direct_current_boundary_transition"),
        &field_terms,
        &field_constants,
        &field_values,
    )?;
    enforce_digest_fields_public_io(
        &mut cs.namespace(|| "direct_current_boundary_out_digest_public"),
        &digest,
        public_inputs,
        direct_terminal_current_boundary_digest_range(),
        "direct_current_boundary_out_digest_public",
    )?;
    Ok(digest)
}

pub(crate) fn enforce_direct_state_x_in_digest<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    vk_fs_digest: [u8; 32],
    mat_digest: &[F; 4],
    chunk_count: u64,
    step_count: u64,
    initial_boundary_digest: [u8; 32],
    current_boundary_digest: [u8; 32],
    pc: u64,
    semantic_accumulator_digest: &[AllocatedNum<SpartanF>; 4],
    construction2_accumulator_digest: &[AllocatedNum<SpartanF>; 4],
    public_trace_digest: [u8; 32],
    expected_x: [u8; 32],
    label: &str,
) -> Result<(), SynthesisError> {
    let digest = direct_state_x_digest_circuit(
        &mut cs.namespace(|| format!("{label}_hash")),
        vk_fs_digest,
        mat_digest,
        chunk_count,
        step_count,
        initial_boundary_digest,
        DirectDigestInput::Constant(current_boundary_digest),
        pc,
        semantic_accumulator_digest,
        construction2_accumulator_digest,
        DirectDigestInput::Constant(public_trace_digest),
    )?;
    enforce_digest_eq_constant(&mut cs.namespace(|| format!("{label}_eq")), &digest, expected_x, label)
}

pub(crate) fn enforce_direct_state_x_out_public_digest<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    public_inputs: &[AllocatedNum<SpartanF>],
    vk_fs_digest: [u8; 32],
    mat_digest: &[F; 4],
    chunk_count: u64,
    step_count: u64,
    initial_boundary_digest: [u8; 32],
    current_boundary_digest: &[AllocatedNum<SpartanF>; 4],
    pc: u64,
    semantic_accumulator_digest: &[AllocatedNum<SpartanF>; 4],
    construction2_accumulator_digest: &[AllocatedNum<SpartanF>; 4],
    public_trace_digest: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<(), SynthesisError> {
    let digest = direct_state_x_digest_circuit(
        &mut cs.namespace(|| format!("{label}_hash")),
        vk_fs_digest,
        mat_digest,
        chunk_count,
        step_count,
        initial_boundary_digest,
        DirectDigestInput::Allocated(current_boundary_digest),
        pc,
        semantic_accumulator_digest,
        construction2_accumulator_digest,
        DirectDigestInput::Allocated(public_trace_digest),
    )?;
    enforce_digest_fields_public_io(
        &mut cs.namespace(|| format!("{label}_public")),
        &digest,
        public_inputs,
        direct_terminal_x_out_digest_range(),
        label,
    )
}

enum DirectDigestInput<'a> {
    Constant([u8; 32]),
    Allocated(&'a [AllocatedNum<SpartanF>; 4]),
}

fn direct_state_x_digest_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    vk_fs_digest: [u8; 32],
    mat_digest: &[F; 4],
    chunk_count: u64,
    step_count: u64,
    initial_boundary_digest: [u8; 32],
    current_boundary_digest: DirectDigestInput<'_>,
    pc: u64,
    semantic_accumulator_digest: &[AllocatedNum<SpartanF>; 4],
    construction2_accumulator_digest: &[AllocatedNum<SpartanF>; 4],
    public_trace_digest: DirectDigestInput<'_>,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut field_terms = Vec::new();
    let mut field_constants = Vec::new();
    let mut field_values = Vec::new();
    push_constant_spartan_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        direct_domain_spartan_fields(b"neo.fold.next/direct_ccs/f_prime_x_out/v2"),
    );
    push_constant_spartan_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        digest32_as_fields(vk_fs_digest)
            .into_iter()
            .map(field_to_spartan),
    );
    push_constant_spartan_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        mat_digest.iter().copied().map(field_to_spartan),
    );
    push_constant_spartan_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        u64_halves_as_spartan_fields(chunk_count),
    );
    push_constant_spartan_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        u64_halves_as_spartan_fields(step_count),
    );
    push_constant_spartan_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        digest32_as_fields(initial_boundary_digest)
            .into_iter()
            .map(field_to_spartan),
    );
    push_digest_input(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        current_boundary_digest,
    );
    push_constant_spartan_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        u64_halves_as_spartan_fields(pc),
    );
    for value in semantic_accumulator_digest {
        field_terms.push(vec![(value.get_variable(), spartan_one())]);
        field_constants.push(spartan_zero());
        field_values.push(value.get_value().unwrap_or(spartan_zero()));
    }
    for value in construction2_accumulator_digest {
        field_terms.push(vec![(value.get_variable(), spartan_one())]);
        field_constants.push(spartan_zero());
        field_values.push(value.get_value().unwrap_or(spartan_zero()));
    }
    push_digest_input(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        public_trace_digest,
    );
    hash_field_linear_combinations_raw(
        cs.namespace(|| "direct_state_x_digest"),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

fn push_digest_input(
    field_terms: &mut Vec<Vec<(bellpepper_core::Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    digest: DirectDigestInput<'_>,
) {
    match digest {
        DirectDigestInput::Constant(digest) => {
            push_constant_spartan_fields(
                field_terms,
                field_constants,
                field_values,
                digest32_as_fields(digest).into_iter().map(field_to_spartan),
            );
        }
        DirectDigestInput::Allocated(digest) => {
            for value in digest {
                field_terms.push(vec![(value.get_variable(), spartan_one())]);
                field_constants.push(spartan_zero());
                field_values.push(value.get_value().unwrap_or(spartan_zero()));
            }
        }
    }
}
