//! Shared arithmetic for the matrix-conformance reference path.

use super::GOLDILOCKS_MODULUS;

pub(super) struct ReferenceLayout {
    pub(super) unpadded_rows: usize,
    pub(super) unpadded_constant: usize,
    pub(super) public_columns: usize,
    pub(super) domain_size: usize,
    pub(super) final_columns: usize,
}

impl ReferenceLayout {
    pub(super) fn map_column(&self, column: usize) -> usize {
        if column < self.unpadded_constant {
            column
        } else {
            self.domain_size + (column - self.unpadded_constant)
        }
    }

    pub(super) fn constant_column(&self) -> usize {
        self.domain_size
    }
}

pub(super) fn word(value: u64) -> usize {
    usize::try_from(value).expect("reference word fits usize")
}

pub(super) fn add_mod(left: u64, right: u64) -> u64 {
    ((u128::from(left) + u128::from(right)) % u128::from(GOLDILOCKS_MODULUS)) as u64
}

pub(super) fn mul_mod(left: u64, right: u64) -> u64 {
    ((u128::from(left) * u128::from(right)) % u128::from(GOLDILOCKS_MODULUS)) as u64
}

pub(super) fn changed_word(value: u64) -> u64 {
    if value + 1 == GOLDILOCKS_MODULUS {
        0
    } else {
        value + 1
    }
}
