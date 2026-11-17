use super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

pub(crate) fn push_gated_linear_zero<const N: usize>(
    b: &mut WasmTaggedR1csBuilder<'_>,
    selector: usize,
    terms: [(usize, F); N],
) {
    b.push_gated_linear_zero(selector, terms);
}

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

pub(crate) fn push_unsigned_ge_of_u32_operands(
    b: &mut WasmTaggedR1csBuilder<'_>,
    gate_cols: impl IntoIterator<Item = usize>,
    lhs: impl IntoIterator<Item = (usize, F)>,
    rhs: impl IntoIterator<Item = (usize, F)>,
    low: usize,
    ge: usize,
) {
    let shift = F::from_u64(1u64 << 32);
    let mut terms: Vec<(usize, F)> = lhs.into_iter().collect();
    terms.extend(rhs.into_iter().map(|(col, coeff)| (col, -coeff)));
    terms.push((super::layout::COL_ONE, shift));
    terms.push((low, -F::ONE));
    terms.push((ge, -shift));
    b.push_row(gate_cols.into_iter().map(|col| (col, F::ONE)), terms, []);
}

pub(crate) fn unsigned_ge_witness(x: u64, y: u64) -> (F, F) {
    debug_assert!(x < (1u64 << 32) && y < (1u64 << 32));
    let shifted = x + (1u64 << 32) - y;
    let ge = shifted >> 32;
    let low = shifted & 0xffff_ffff;
    (F::from_u64(low), F::from_u64(ge))
}
