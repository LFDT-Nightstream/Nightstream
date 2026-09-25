//! Shared public authority prefix for every phased F-prime circuit.
//!
//! `x_out` is the after-state Poseidon2 digest. The suffix carries the
//! before-state digest and both exact schedule cursors. Component circuits
//! must recompute the digests from their private state values.

use std::ops::Range;

use crate::paper::f_prime::public_input_link::{
    FPrimePublicInputLayout, F_PRIME_ENC_INST_BITS, F_PRIME_ENC_INST_OFFSET,
};

const CURSOR_BITS: usize = 64;
const PHASE_SUFFIX_BITS: usize = F_PRIME_ENC_INST_BITS + 2 * CURSOR_BITS;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingPublicLayout {
    after_state_digest_bits: Range<usize>,
    before_state_digest_bits: Range<usize>,
    before_cursor_bits: Range<usize>,
    after_cursor_bits: Range<usize>,
    padding_columns: Range<usize>,
}

impl NebulaFPrimeStreamingPublicLayout {
    pub fn production() -> Self {
        let after_state_digest_bits = F_PRIME_ENC_INST_OFFSET..F_PRIME_ENC_INST_OFFSET + F_PRIME_ENC_INST_BITS;
        let before_state_digest_bits = after_state_digest_bits.end..after_state_digest_bits.end + F_PRIME_ENC_INST_BITS;
        let before_cursor_bits = before_state_digest_bits.end..before_state_digest_bits.end + CURSOR_BITS;
        let after_cursor_bits = before_cursor_bits.end..before_cursor_bits.end + CURSOR_BITS;
        let columns = FPrimePublicInputLayout::with_suffix(PHASE_SUFFIX_BITS).total_len();
        let padding_columns = after_cursor_bits.end..columns;
        Self {
            after_state_digest_bits,
            before_state_digest_bits,
            before_cursor_bits,
            after_cursor_bits,
            padding_columns,
        }
    }

    pub fn after_state_digest_bits(&self) -> Range<usize> {
        self.after_state_digest_bits.clone()
    }

    pub fn before_state_digest_bits(&self) -> Range<usize> {
        self.before_state_digest_bits.clone()
    }

    pub fn before_cursor_bits(&self) -> Range<usize> {
        self.before_cursor_bits.clone()
    }

    pub fn after_cursor_bits(&self) -> Range<usize> {
        self.after_cursor_bits.clone()
    }

    pub fn padding_columns(&self) -> Range<usize> {
        self.padding_columns.clone()
    }

    pub fn logical_columns(&self) -> usize {
        self.after_cursor_bits.end
    }

    pub fn columns(&self) -> usize {
        self.padding_columns.end
    }

    pub(crate) fn f_prime_layout(&self) -> FPrimePublicInputLayout {
        FPrimePublicInputLayout::with_suffix(PHASE_SUFFIX_BITS)
    }
}
