use crate::layout::COL_ONE;

use super::tagged_r1cs_builder::WasmTaggedR1csBuilder as R1csBuilder;
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

#[derive(Clone, Copy, Debug)]
pub struct ConditionalSelectCols {
    pub selector: usize,
    pub cond: usize,
    pub cond_is_zero: usize,
    pub lhs: usize,
    pub rhs: usize,
    pub out: usize,
    pub scratch_out_delta: usize,
    pub scratch_inverse: usize,
}

/// Constraints for `selector * (out - (cond != 0 ? lhs : rhs)) = 0`.
///
/// The zero-test and scratch product rows are intentionally global: callers
/// must populate `cond_is_zero`, `scratch_inverse`, and `scratch_out_delta` on
/// every witness row.
pub fn add_conditional_select_gadget(b: &mut R1csBuilder, cols: ConditionalSelectCols) {
    push_zero_test_gadget(b, cols.cond, cols.scratch_inverse, cols.cond_is_zero);

    // scratch_out_delta = (cond != 0) * (lhs - rhs)
    b.push_row(
        [(COL_ONE, F::ONE), (cols.cond_is_zero, -F::ONE)],
        [(cols.lhs, F::ONE), (cols.rhs, -F::ONE)],
        [(cols.scratch_out_delta, F::ONE)],
    );

    push_gated_linear_zero(
        b,
        cols.selector,
        // (out - rhs) - (cond != 0) * (lhs - rhs) = 0
        [
            (cols.out, F::ONE),
            (cols.rhs, -F::ONE),
            (cols.scratch_out_delta, -F::ONE),
        ],
    );
}

pub(crate) fn push_gated_linear_zero<const N: usize>(b: &mut R1csBuilder, selector: usize, terms: [(usize, F); N]) {
    b.push_row([(selector, F::ONE)], terms, []);
}

/// Constrains `is_zero` to be exactly the zero-test of `value`.
///
/// The witness must set `inverse = value^{-1}` when `value != 0`, and `inverse = 0`
/// when `value == 0`.
pub fn push_zero_test_gadget(b: &mut R1csBuilder, value: usize, inverse: usize, is_zero: usize) {
    b.push_row(
        [(value, F::ONE)],
        [(inverse, F::ONE)],
        [(super::layout::COL_ONE, F::ONE), (is_zero, -F::ONE)],
    );
    b.push_row([(value, F::ONE)], [(is_zero, F::ONE)], []);
}

pub(crate) fn zero_test_witness_u64(value: u64) -> (F, F) {
    zero_test_witness_field(F::from_u64(value))
}

/// Zero-test witness for an arbitrary field element.
///
/// Returns `(is_zero, inverse)` consistent with [`push_zero_test_gadget`]:
/// when `value == 0` the inverse is unconstrained and we return `0`; when
/// `value != 0` `is_zero` is `0` and we return the field inverse.
pub(crate) fn zero_test_witness_field(value: F) -> (F, F) {
    if value == F::ZERO {
        (F::ONE, F::ZERO)
    } else {
        (F::ZERO, value.try_inverse().expect("nonzero field inverse"))
    }
}

/// Enforce `(Σ gate_cols) · (word - Σ_{i<4} bytes[i] · 2^(8i)) = 0`.
///
/// Each entry in `gate_cols` is summed with coefficient `1` on the
/// left of the CCS row. For a single-column gate, pass `[selector_col]`.
/// For a one-hot opcode gate (the constraint should fire on any of
/// several mutually exclusive opcodes), pass the corresponding
/// selector columns — exactly one is `1` per row, so `Σ gate_cols` is
/// `1` when the constraint should fire and `0` otherwise.
pub(crate) fn push_u32_le_bytes_decomp<const N: usize>(
    b: &mut R1csBuilder,
    gate_cols: impl IntoIterator<Item = usize>,
    word: usize,
    bytes: [usize; N],
) {
    debug_assert_eq!(N, 4);
    b.push_row(
        gate_cols.into_iter().map(|col| (col, F::ONE)),
        [
            (word, F::ONE),
            (bytes[0], -F::ONE),
            (bytes[1], -F::from_u64(1 << 8)),
            (bytes[2], -F::from_u64(1 << 16)),
            (bytes[3], -F::from_u64(1 << 24)),
        ],
        [],
    );
}
