//! Shared circuit helpers for the direct CCS/R1CS terminal F' surface.

mod accumulator;
mod fields;

pub(crate) use accumulator::{
    direct_accumulator_digest_circuit_from_claims, direct_accumulator_digest_from_claims,
    direct_accumulator_digest_from_claims_with_base,
};
pub(crate) use fields::{digest32_as_spartan_fields, field_to_spartan, spartan_zero, u64_halves_as_spartan_fields};
use fields::{direct_domain_spartan_fields, push_constant_spartan_fields, spartan_one};

use crate::construction2::Construction2FreshInstance;
use crate::finalize::digest32_as_fields;
use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::ce_consistency::enforce_paper_ce_claim_consistency;
use crate::superneo_circuit::claim::CircuitCeClaim;
use crate::superneo_circuit::transcript::hash_field_linear_combinations_raw;
use crate::superneo_circuit::witness::alloc_packed_witness;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ccs::{CcsStructure, CcsWitness};
use neo_math::{D, F};
use neo_params::NeoParams;

pub(crate) fn enforce_digest_fields_public_io<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest: &[AllocatedNum<SpartanF>; 4],
    public_inputs: &[AllocatedNum<SpartanF>],
    range: std::ops::Range<usize>,
    label: &str,
) -> Result<(), SynthesisError> {
    if range.len() != 4 || range.end > public_inputs.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, digest_lane) in digest.iter().enumerate() {
        cs.enforce(
            || format!("{label}_{idx}"),
            |lc| lc + digest_lane.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + public_inputs[range.start + idx].get_variable(),
        );
    }
    Ok(())
}

pub(crate) fn enforce_digest_eq_constant<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest: &[AllocatedNum<SpartanF>; 4],
    expected: [u8; 32],
    label: &str,
) -> Result<(), SynthesisError> {
    for (idx, expected) in digest32_as_spartan_fields(expected).into_iter().enumerate() {
        cs.enforce(
            || format!("{label}_{idx}"),
            |lc| lc + digest[idx].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (expected, CS::one()),
        );
    }
    Ok(())
}

pub(crate) fn alloc_digest_constant<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest: [u8; 32],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    digest32_as_spartan_fields(digest)
        .into_iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(value)))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)
}

pub(crate) fn public_digest_input(
    public_inputs: &[AllocatedNum<SpartanF>],
    range: std::ops::Range<usize>,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    if range.len() != 4 || range.end > public_inputs.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok([
        public_inputs[range.start].clone(),
        public_inputs[range.start + 1].clone(),
        public_inputs[range.start + 2].clone(),
        public_inputs[range.start + 3].clone(),
    ])
}

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

pub(crate) fn direct_terminal_accumulator_digest_range() -> std::ops::Range<usize> {
    280..284
}

pub(crate) fn direct_terminal_current_boundary_digest_range() -> std::ops::Range<usize> {
    16..20
}

pub(crate) fn direct_terminal_x_out_digest_range() -> std::ops::Range<usize> {
    20..24
}

pub(crate) fn direct_terminal_public_trace_out_digest_range() -> std::ops::Range<usize> {
    284..288
}

pub(crate) fn direct_terminal_construction2_accumulator_digest_range() -> std::ops::Range<usize> {
    288..292
}

pub(crate) fn enforce_direct_construction2_input_u_i<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    input_u_i: &Construction2FreshInstance,
    expected_x_i: &crate::construction2::Construction2EncodedPublicInput,
    chunk_count_in: u64,
    expected_kappa: usize,
) -> Result<(), SynthesisError> {
    let x_i = input_u_i.x_i().bytes();
    let x_i = digest32_as_spartan_fields(x_i)
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("construction2_input_u_i_x_{idx}")), || {
                Ok(value)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let x_i: [AllocatedNum<SpartanF>; 4] = x_i.try_into().map_err(|_| SynthesisError::Unsatisfiable)?;
    enforce_digest_eq_constant(
        &mut cs.namespace(|| "construction2_input_u_i_x_matches_x_in"),
        &x_i,
        expected_x_i.bytes(),
        "construction2_input_u_i_x_matches_x_in",
    )?;

    let commitment = input_u_i.commitment().commitment();
    if chunk_count_in == 0 && !input_u_i.is_canonical_zero_for(expected_kappa, input_u_i.x_i()) {
        return Err(SynthesisError::Unsatisfiable);
    }
    if chunk_count_in != 0
        && (commitment.d != D
            || commitment.kappa == 0
            || commitment
                .d
                .checked_mul(commitment.kappa)
                .map_or(true, |len| len != commitment.data.len()))
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let d = AllocatedNum::alloc(cs.namespace(|| "construction2_input_u_i_commitment_d"), || {
        Ok(SpartanF::from_canonical_u64(commitment.d as u64))
    })?;
    let kappa = AllocatedNum::alloc(cs.namespace(|| "construction2_input_u_i_commitment_kappa"), || {
        Ok(SpartanF::from_canonical_u64(commitment.kappa as u64))
    })?;
    cs.enforce(
        || "construction2_input_u_i_d",
        |lc| lc + d.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(D as u64), CS::one()),
    );
    let expected_kappa = commitment.data.len() / D;
    cs.enforce(
        || "construction2_input_u_i_kappa",
        |lc| lc + kappa.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(expected_kappa as u64), CS::one()),
    );
    for (idx, value) in commitment.data.iter().copied().enumerate() {
        let data = AllocatedNum::alloc(
            cs.namespace(|| format!("construction2_input_u_i_commitment_data_{idx}")),
            || Ok(field_to_spartan(value)),
        )?;
        if chunk_count_in == 0 {
            cs.enforce(
                || format!("construction2_x_only_u_i_data_{idx}"),
                |lc| lc + data.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
        } else {
            cs.enforce(
                || format!("construction2_carried_u_i_data_{idx}"),
                |lc| lc + data.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + data.get_variable(),
            );
        }
    }
    Ok(())
}

pub(crate) fn enforce_direct_terminal_final_ce_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[CircuitCeClaim],
    witnesses: &[CcsWitness<F>],
) -> Result<(), SynthesisError> {
    if claims.len() != witnesses.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (claim, witness)) in claims.iter().zip(witnesses.iter()).enumerate() {
        let witness = alloc_packed_witness(
            &mut cs.namespace(|| format!("final_claim_{idx}_witness")),
            witness,
            &format!("final_claim_{idx}_witness"),
        )?;
        enforce_paper_ce_claim_consistency(
            &mut cs.namespace(|| format!("final_claim_{idx}_ce_consistency")),
            params,
            structure,
            structure,
            &witness,
            claim,
            SpartanF::from_canonical_u64(7),
            &format!("final_claim_{idx}_ce_consistency"),
        )?;
    }
    Ok(())
}
