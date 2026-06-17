use super::tagged_r1cs_builder::WasmTaggedR1csBuilder as R1csBuilder;
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

pub(crate) fn push_gated_linear_zero<const N: usize>(b: &mut R1csBuilder, selector: usize, terms: [(usize, F); N]) {
    b.push_row([(selector, F::ONE)], terms, []);
}

/// Constrains `is_zero` to be exactly the zero-test of `value`.
///
/// The witness must set `inverse = value^{-1}` when `value != 0`, and `inverse = 0`
/// when `value == 0`.
pub fn push_zero_test_gadget(b: &mut R1csBuilder, value: usize, inverse: usize, is_zero: usize) {
    push_zero_test_expr_gadget(b, [(value, F::ONE)], inverse, is_zero);
}

/// Constrains `is_zero` to be exactly the zero-test of a linear expression.
///
/// The witness must set `inverse` to the expression's field inverse when it
/// is nonzero, and `inverse = 0` when it is zero.
pub(crate) fn push_zero_test_expr_gadget<const N: usize>(
    b: &mut R1csBuilder,
    expr: [(usize, F); N],
    inverse: usize,
    is_zero: usize,
) {
    b.push_row(
        expr,
        [(inverse, F::ONE)],
        [(super::layout::COL_ONE, F::ONE), (is_zero, -F::ONE)],
    );
    b.push_row(expr, [(is_zero, F::ONE)], []);
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

/// Unsigned `x >= y` for two values known to be in `[0, 2^32)`.
///
/// Enforces the borrow-bit (ge) decomposition `x - y + 2^32 = low + ge·2^32`, where
/// `low` is also known to be in `[0, 2^32)` and `ge` is boolean.
///
/// Since `x, y < 2^32`, the left side is a genuine integer in `(0, 2^33)` so
/// the split is unique and `ge ?= (x >= y)`.
///
/// `gate_cols` are assumed to be one-hot selectors.
pub(crate) fn push_unsigned_ge_gadget(
    b: &mut R1csBuilder,
    gate_cols: impl IntoIterator<Item = usize>,
    x: usize,
    y: usize,
    low: usize,
    ge: usize,
) {
    let shift = F::from_u64(1u64 << 32);
    let mut terms = vec![(x, F::ONE), (y, -F::ONE)];

    terms.push((super::layout::COL_ONE, shift));
    terms.push((low, -F::ONE));
    terms.push((ge, -shift));
    b.push_row(gate_cols.into_iter().map(|col| (col, F::ONE)), terms, []);
}

/// Witness for [`push_unsigned_ge_gadget`]: returns `(low, ge)` for the
/// borrow-bit split of `x - y + 2^32`. `x` and `y` must be `< 2^32`.
pub(crate) fn unsigned_ge_witness(x: u64, y: u64) -> (F, F) {
    debug_assert!(x < (1u64 << 32) && y < (1u64 << 32));
    let shifted = x + (1u64 << 32) - y;
    let ge = shifted >> 32;
    let low = shifted & 0xffff_ffff;
    (F::from_u64(low), F::from_u64(ge))
}
