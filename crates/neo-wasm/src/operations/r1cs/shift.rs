use super::builder::{CircuitBuilder, ColumnId, Lc};
use super::{ops_lc, Inputs};
use crate::isa::WasmOpcode;

/// One 64-bit left barrel shifter for both widths. Right shifts and `rotr`
/// reverse the input and the output, so every operation shifts left. The
/// fill of the low bits is zero, the wrapped bit (rotations), or the sign
/// (`shr_s`).
///
/// An i32 operand becomes 64 bits: two copies for rotations, the sign for
/// `shr_s`, and zeros otherwise. The 32-bit result is the low half of the
/// 64-bit result when the count is below 32, so i32 operations disable the
/// sixth stage and bind only the low 32 output bits.
pub(super) fn enforce_shifts(builder: &mut CircuitBuilder, inputs: &Inputs) -> Result<(), String> {
    use WasmOpcode::*;
    const WIDTH: usize = 64;
    const NARROW: usize = 32;
    let value = &inputs.left64;
    let amount = &inputs.right64;
    let output = &inputs.output64;
    let wide = ops_lc(&[I64Shl, I64ShrU, I64ShrS, I64Rotl, I64Rotr]);
    let activation = ops_lc(&[I32Shl, I32ShrU, I32ShrS, I32Rotl, I32Rotr]).plus(&wide);
    let reverse = ops_lc(&[I32ShrU, I32ShrS, I32Rotr, I64ShrU, I64ShrS, I64Rotr]);
    let wide_reverse = ops_lc(&[I64ShrU, I64ShrS, I64Rotr]);
    let rotate = ops_lc(&[I32Rotl, I32Rotr, I64Rotl, I64Rotr]);

    let narrow_sign = Lc::var(builder.alloc_product(&ops_lc(&[I32ShrS]), &Lc::var(value[NARROW - 1])));
    let mut prepared = value[..NARROW]
        .iter()
        .map(|&bit| Lc::var(bit))
        .collect::<Vec<_>>();
    for index in NARROW..WIDTH {
        let copy = builder.alloc_product(&ops_lc(&[I32Rotl, I32Rotr]), &Lc::var(value[index - NARROW]));
        prepared.push(Lc::var(builder.alloc_select(
            &wide,
            &Lc::var(value[index]),
            &Lc::var(copy).plus(&narrow_sign),
        )));
    }
    let sign = Lc::var(builder.alloc_product(&ops_lc(&[I32ShrS, I64ShrS]), &prepared[WIDTH - 1]));

    let mut current = vec![Lc::zero(); WIDTH];
    for low in 0..WIDTH / 2 {
        let high = WIDTH - 1 - low;
        let chosen = builder.alloc_select(&reverse, &prepared[high], &prepared[low]);
        current[low] = Lc::var(chosen);
        current[high] = prepared[low]
            .clone()
            .plus(&prepared[high])
            .minus(&Lc::var(chosen));
    }
    for stage in 0..WIDTH.trailing_zeros() as usize {
        let amount_bit = if (1 << stage) < NARROW {
            Lc::var(amount[stage])
        } else {
            Lc::var(builder.alloc_product(&wide, &Lc::var(amount[stage])))
        };
        let distance = 1usize << stage;
        let mut next = Vec::with_capacity(WIDTH);
        for index in 0..WIDTH {
            let source = if index >= distance {
                current[index - distance].clone()
            } else {
                let wrapped = builder.alloc_product(&rotate, &current[WIDTH - distance + index]);
                Lc::var(wrapped).plus(&sign)
            };
            next.push(Lc::var(builder.alloc_select(&amount_bit, &source, &current[index])));
        }
        current = next;
    }
    let shifted = current
        .iter()
        .map(|lc| ColumnId(lc.terms[0].0))
        .collect::<Vec<_>>();
    let reversed = shifted.iter().rev().copied().collect::<Vec<_>>();
    let forward = activation.minus(&reverse);
    let wide_forward = wide.minus(&wide_reverse);
    builder.enforce_bits_equal_when(&forward, &output[..NARROW], &shifted[..NARROW]);
    builder.enforce_bits_equal_when(&wide_forward, &output[NARROW..], &shifted[NARROW..]);
    builder.enforce_bits_equal_when(&reverse, &output[..NARROW], &reversed[..NARROW]);
    builder.enforce_bits_equal_when(&wide_reverse, &output[NARROW..], &reversed[NARROW..]);
    Ok(())
}
