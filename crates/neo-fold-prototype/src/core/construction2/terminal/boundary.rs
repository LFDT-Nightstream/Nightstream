//! Owns Construction-2 terminal public boundary allocation and digest checks.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_math::D;

use super::constraints::{enforce_allocated_num_eq_constant, native_to_spartan};
use super::types::{Construction2TerminalBoundaryInputs, Construction2TerminalBoundaryView};
use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::transcript::hash_field_linear_combinations_raw;
use crate::superneo_nifs_circuit::{digest32_as_spartan_fields, enforce_digest_eq};

pub(crate) fn terminal_boundary_public_values(boundary: &Construction2TerminalBoundaryView<'_>) -> Vec<SpartanF> {
    let mut values = Vec::with_capacity(14 + boundary.commitment_data.len());
    values.extend(digest32_as_spartan_fields(boundary.fresh_instance_digest));
    values.extend(digest32_as_spartan_fields(boundary.commitment_digest));
    values.push(SpartanF::from_canonical_u64(boundary.commitment_d));
    values.push(SpartanF::from_canonical_u64(boundary.commitment_kappa));
    values.extend(boundary.commitment_data.iter().map(native_to_spartan));
    values.extend(digest32_as_spartan_fields(boundary.x_i_bytes));
    values
}

pub(crate) fn alloc_terminal_boundary_public_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    label_prefix: &str,
    boundary: &Construction2TerminalBoundaryView<'_>,
) -> Result<Construction2TerminalBoundaryInputs, SynthesisError> {
    let fresh_instance_digest = alloc_digest_public_inputs(
        cs,
        &format!("{label_prefix}_fresh_instance_digest"),
        boundary.fresh_instance_digest,
    )?;
    let commitment_digest = alloc_digest_public_inputs(
        cs,
        &format!("{label_prefix}_commitment_digest"),
        boundary.commitment_digest,
    )?;
    let commitment_d = AllocatedNum::alloc_input(cs.namespace(|| format!("{label_prefix}_commitment_d")), || {
        Ok(SpartanF::from_canonical_u64(boundary.commitment_d))
    })?;
    let commitment_kappa =
        AllocatedNum::alloc_input(cs.namespace(|| format!("{label_prefix}_commitment_kappa")), || {
            Ok(SpartanF::from_canonical_u64(boundary.commitment_kappa))
        })?;
    let commitment_data = boundary
        .commitment_data
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc_input(cs.namespace(|| format!("{label_prefix}_commitment_data_{idx}")), || {
                Ok(native_to_spartan(value))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let x_i = alloc_digest_public_inputs(cs, &format!("{label_prefix}_x_i"), boundary.x_i_bytes)?;
    Ok(Construction2TerminalBoundaryInputs {
        fresh_instance_digest,
        commitment_digest,
        commitment_d,
        commitment_kappa,
        commitment_data,
        x_i,
    })
}

pub(crate) fn enforce_terminal_boundary_digests<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    boundary: &Construction2TerminalBoundaryInputs,
    commitment_raw_tag: u64,
    public_boundary_raw_tag: u64,
    label_prefix: &str,
) -> Result<(), SynthesisError> {
    enforce_allocated_num_eq_constant(
        &mut cs.namespace(|| format!("{label_prefix}_commitment_d_eq")),
        &boundary.commitment_d,
        SpartanF::from_canonical_u64(D as u64),
        &format!("{label_prefix}_commitment_d_eq"),
    );
    let expected_commitment_digest = construction2_commitment_digest_circuit(
        &mut cs.namespace(|| format!("{label_prefix}_expected_commitment_digest")),
        commitment_raw_tag,
        &boundary.commitment_d,
        &boundary.commitment_kappa,
        &boundary.commitment_data,
        &format!("{label_prefix}_expected_commitment_digest"),
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| format!("{label_prefix}_commitment_digest_eq")),
        &boundary.commitment_digest,
        &expected_commitment_digest,
        &format!("{label_prefix}_commitment_digest_eq"),
    )?;
    let expected_fresh_instance_digest = construction2_public_boundary_digest_circuit(
        &mut cs.namespace(|| format!("{label_prefix}_expected_fresh_instance_digest")),
        public_boundary_raw_tag,
        &boundary.commitment_digest,
        &boundary.x_i,
        &format!("{label_prefix}_expected_fresh_instance_digest"),
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| format!("{label_prefix}_fresh_instance_digest_eq")),
        &boundary.fresh_instance_digest,
        &expected_fresh_instance_digest,
        &format!("{label_prefix}_fresh_instance_digest_eq"),
    )
}

fn construction2_public_boundary_digest_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    raw_tag: u64,
    commitment_digest: &[AllocatedNum<SpartanF>; 4],
    x_i: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut field_terms = Vec::with_capacity(9);
    let mut field_constants = Vec::with_capacity(9);
    let mut field_values = Vec::with_capacity(9);

    field_terms.push(Vec::new());
    field_constants.push(SpartanF::from_canonical_u64(raw_tag));
    field_values.push(SpartanF::from_canonical_u64(raw_tag));
    for lane in commitment_digest.iter().chain(x_i.iter()) {
        field_terms.push(vec![(lane.get_variable(), SpartanF::from_canonical_u64(1))]);
        field_constants.push(SpartanF::from_canonical_u64(0));
        field_values.push(lane.get_value().unwrap_or(SpartanF::from_canonical_u64(0)));
    }

    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

fn construction2_commitment_digest_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    raw_tag: u64,
    d: &AllocatedNum<SpartanF>,
    kappa: &AllocatedNum<SpartanF>,
    data: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut field_terms = Vec::with_capacity(3 + data.len());
    let mut field_constants = Vec::with_capacity(3 + data.len());
    let mut field_values = Vec::with_capacity(3 + data.len());

    field_terms.push(Vec::new());
    field_constants.push(SpartanF::from_canonical_u64(raw_tag));
    field_values.push(SpartanF::from_canonical_u64(raw_tag));
    for value in [d, kappa].into_iter().chain(data.iter()) {
        field_terms.push(vec![(value.get_variable(), SpartanF::from_canonical_u64(1))]);
        field_constants.push(SpartanF::from_canonical_u64(0));
        field_values.push(value.get_value().unwrap_or(SpartanF::from_canonical_u64(0)));
    }

    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

fn alloc_digest_public_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    label: &str,
    digest: [u8; 32],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let fields = digest32_as_spartan_fields(digest);
    let values = fields
        .into_iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("{label}_{idx}")), || Ok(value)))
        .collect::<Result<Vec<_>, _>>()?;
    values.try_into().map_err(|_| SynthesisError::Unsatisfiable)
}
