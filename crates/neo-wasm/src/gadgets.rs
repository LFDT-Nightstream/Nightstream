use super::tagged_r1cs_builder::WasmTaggedR1csBuilder as R1csBuilder;
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

#[derive(Clone, Copy, Debug)]
pub struct ConditionalSelectCols {
    pub selector: usize,
    pub cond: usize,
    pub lhs: usize,
    pub rhs: usize,
    pub out: usize,
    pub scratch_out_delta: usize,
}

/// Constraints for:
/// cond ? lhs : rhs
///
/// TODO: this currently *assumes* that cond is boolean
pub fn add_conditional_select_gadget(b: &mut R1csBuilder, cols: ConditionalSelectCols) {
    // the core equation is:
    //
    // selector * [ cond * (lhs - rhs) ] = selector * (out - rhs)

    // one auxiliary variable/row for:
    //   cond * (lhs - rhs)
    b.push_row(
        [(cols.cond, F::ONE)],
        [(cols.lhs, F::ONE), (cols.rhs, -F::ONE)],
        [(cols.scratch_out_delta, F::ONE)],
    );

    push_gated_linear_zero(
        b,
        cols.selector,
        // (out - rhs) - cond * (lhs - rhs)] = 0
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
    if value == 0 {
        (F::ONE, F::ZERO)
    } else {
        (
            F::ZERO,
            F::from_u64(value)
                .try_inverse()
                .expect("nonzero field inverse"),
        )
    }
}

pub(crate) fn push_u32_le_bytes<const N: usize>(b: &mut R1csBuilder, selector: usize, word: usize, bytes: [usize; N]) {
    debug_assert_eq!(N, 4);
    push_gated_linear_zero(
        b,
        selector,
        [
            (word, F::ONE),
            (bytes[0], -F::ONE),
            (bytes[1], -F::from_u64(1 << 8)),
            (bytes[2], -F::from_u64(1 << 16)),
            (bytes[3], -F::from_u64(1 << 24)),
        ],
    );
}
