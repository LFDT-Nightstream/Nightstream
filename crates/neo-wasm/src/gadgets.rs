//! WASM-only algebraic gadgets not yet represented by shared descriptors.

use super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

pub(crate) fn push_gated_linear_zero<const N: usize>(
    b: &mut WasmTaggedR1csBuilder<'_>,
    selector: usize,
    terms: [(usize, F); N],
) {
    b.push_row([(selector, F::ONE)], terms, []);
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
    b: &mut WasmTaggedR1csBuilder<'_>,
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

/// Gated unsigned comparison for two caller-bounded integer expressions.
///
/// On active rows, `lhs` and `rhs` must evaluate as canonical integers in
/// `[0, 2^32)`. They may be linear combinations of columns, but callers must
/// ensure those combinations do not rely on field wraparound and do not become
/// negative. This gadget only proves the comparison once those bounds are true.
///
/// It enforces `lhs - rhs + 2^32 = low + ge * 2^32`, where `low` is `U32` and
/// `ge` is boolean. Since `lhs, rhs < 2^32`, the shifted value is in
/// `[1, 2^33)`, below the Goldilocks modulus, so the split is unique:
/// `ge = 1` iff `lhs >= rhs`.
///
/// `gate_cols` activate the row through their sum. Callers normally pass
/// mutually exclusive opcode selectors.
pub(crate) fn push_unsigned_ge_gadget(
    b: &mut WasmTaggedR1csBuilder<'_>,
    gate_cols: impl IntoIterator<Item = usize>,
    lhs: impl IntoIterator<Item = (usize, F)>,
    rhs: impl IntoIterator<Item = (usize, F)>,
    low: usize,
    ge: usize,
) {
    let shift = F::from_u64(1u64 << 32);
    // diff = lhs - rhs, then `diff + 2^32 = low + ge·2^32`.
    let mut terms: Vec<(usize, F)> = lhs.into_iter().collect();
    terms.extend(rhs.into_iter().map(|(col, coeff)| (col, -coeff)));
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
