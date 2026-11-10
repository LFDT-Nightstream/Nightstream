use super::builder::{bits_u64, CircuitBuilder, ColumnId, Lc};
use super::{ops_lc, Inputs};
use crate::isa::WasmOpcode;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Allocate `left - right - borrow_in + 2^32` for each 32-bit half as 33
/// bits. The top bit of a half is 1 iff that prefix does not borrow.
pub(super) fn no_borrow_bits(
    builder: &mut CircuitBuilder,
    left: &[ColumnId],
    right: &[ColumnId],
    borrow_in: &Lc,
) -> Result<Vec<ColumnId>, String> {
    let mut borrow = borrow_in.clone();
    let mut tops = Vec::new();
    for (left, right) in left.chunks(32).zip(right.chunks(32)) {
        let (l, r, previous) = (left.to_vec(), right.to_vec(), borrow.clone());
        let bits = builder.alloc_bits("borrow_difference", 33, move |z| {
            Ok((i128::from(bits_u64(z, &l)?)
                - i128::from(bits_u64(z, &r)?)
                - i128::from(previous.evaluate(z).as_canonical_u64())
                + (1i128 << 32)) as u64)
        });
        builder.enforce_linear_zero(
            Lc::from_bits(&bits)
                .minus(&Lc::from_bits(left))
                .plus(&Lc::from_bits(right))
                .plus(&borrow)
                .minus(&builder.one().scaled(F::from_u64(1 << 32))),
        );
        borrow = builder.one().minus(&Lc::var(bits[32]));
        tops.push(bits[32]);
    }
    Ok(tops)
}

/// All 16 comparisons share one subtraction `l - r - c`. `c = 1` turns
/// `<` into `<=`; an inverted result gives `>` and `>=`. A signed result is
/// the unsigned result XOR both sign bits.
pub(super) fn enforce_comparisons(builder: &mut CircuitBuilder, inputs: &Inputs) -> Result<(), String> {
    let narrow = comparison_operations(false);
    let wide = comparison_operations(true);
    let filter = |table: &[(WasmOpcode, bool, bool, bool)], pick: fn(&(WasmOpcode, bool, bool, bool)) -> bool| {
        table
            .iter()
            .filter(|entry| pick(entry))
            .map(|entry| entry.0)
            .collect::<Vec<_>>()
    };
    let both = narrow.iter().chain(&wide).copied().collect::<Vec<_>>();
    let narrow_gate = ops_lc(&filter(&narrow, |_| true));
    let wide_gate = ops_lc(&filter(&wide, |_| true));
    let activation = narrow_gate.clone().plus(&wide_gate);
    let borrow_in = ops_lc(&filter(&both, |entry| entry.1));
    let invert = ops_lc(&filter(&both, |entry| entry.3));
    let tops = no_borrow_bits(builder, &inputs.left64, &inputs.right64, &borrow_in)?;
    let narrow_top = builder.alloc_product(&narrow_gate, &Lc::var(tops[0]));
    let wide_top = builder.alloc_product(&wide_gate, &Lc::var(tops[1]));
    let borrow = activation
        .clone()
        .minus(&Lc::var(narrow_top))
        .minus(&Lc::var(wide_top));
    let narrow_signs = builder.alloc_xor(&Lc::var(inputs.left64[31]), &Lc::var(inputs.right64[31]))?;
    let wide_signs = builder.alloc_xor(&Lc::var(inputs.left64[63]), &Lc::var(inputs.right64[63]))?;
    let narrow_flip = builder.alloc_product(&ops_lc(&filter(&narrow, |entry| entry.2)), &Lc::var(narrow_signs));
    let wide_flip = builder.alloc_product(&ops_lc(&filter(&wide, |entry| entry.2)), &Lc::var(wide_signs));
    let signed = builder.alloc_xor(&borrow, &Lc::var(narrow_flip).plus(&Lc::var(wide_flip)))?;
    let result = builder.alloc_xor(&Lc::var(signed), &invert)?;
    builder.enforce_equal_when(&activation, &Lc::var(inputs.output64[0]), &Lc::var(result));
    builder.enforce_bits_zero_when(&activation, &inputs.output64[1..32]);
    builder.enforce_bits_zero_when(&wide_gate, &inputs.output64[32..]);
    Ok(())
}

/// (operation, borrow in, signed, invert): `<` is `l - r` borrows, `<=` is
/// `l - r - 1` borrows, `>` is not `<=`, and `>=` is not `<`.
pub(super) fn comparison_operations(wide: bool) -> [(WasmOpcode, bool, bool, bool); 8] {
    use WasmOpcode::*;
    if wide {
        [
            (I64LtS, false, true, false),
            (I64LtU, false, false, false),
            (I64GtS, true, true, true),
            (I64GtU, true, false, true),
            (I64LeS, true, true, false),
            (I64LeU, true, false, false),
            (I64GeS, false, true, true),
            (I64GeU, false, false, true),
        ]
    } else {
        [
            (I32LtS, false, true, false),
            (I32LtU, false, false, false),
            (I32GtS, true, true, true),
            (I32GtU, true, false, true),
            (I32LeS, true, true, false),
            (I32LeU, true, false, false),
            (I32GeS, false, true, true),
            (I32GeU, false, false, true),
        ]
    }
}
