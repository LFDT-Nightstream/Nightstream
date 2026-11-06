//! Generic execution of the Lean-authored compact 4-matrix program.
//!
//! This module owns wire decoding and row interpretation. It does not select
//! phases, applications, row schedules, or matrix formulas.

use p3_goldilocks::Goldilocks;

use super::PackageError;

const POSEIDON_WIDTH: usize = neo_ccs::crypto::poseidon2_goldilocks::WIDTH;

mod form;
use form::Form;
pub(crate) use form::MatrixRun;
pub(crate) use permutation::{external_outputs, permutation_rows, Entries};
mod permutation;
mod template;

pub(super) const MEANINGFUL_PORTS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Entry {
    pub(super) column: usize,
    pub(super) coefficient: Goldilocks,
}

pub(super) type RowView<'a> = [&'a [MatrixRun]; MEANINGFUL_PORTS];

pub(super) fn external_layer(state: &[Form], logical_width: usize) -> Result<Vec<Form>, PackageError> {
    if state.len() != POSEIDON_WIDTH {
        return Err(PackageError::Malformed("matrix Poseidon2 state"));
    }
    template::outputs("poseidon2-external-v1", state, logical_width)
}

pub(super) fn validate_form(form: &Form, logical_width: usize) -> Result<(), PackageError> {
    form.validate(logical_width)
}
