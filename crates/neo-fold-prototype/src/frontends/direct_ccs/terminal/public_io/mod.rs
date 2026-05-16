//! Public-input digest lane helpers for the direct terminal circuit.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::spartan_backend::SpartanF;

use super::gadgets::digest32_as_spartan_fields;

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
