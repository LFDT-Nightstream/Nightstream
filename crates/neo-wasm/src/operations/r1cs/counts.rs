use super::builder::{CircuitBuilder, ColumnId, Lc};
use super::{opcode_lc, ops_lc};
use crate::isa::WasmOpcode;

pub(super) fn enforce_zero_counts(
    builder: &mut CircuitBuilder,
    input: &[ColumnId],
    output: &[ColumnId],
    wide: bool,
) -> Result<(), String> {
    let (leading, trailing) = if wide {
        (WasmOpcode::I64Clz, WasmOpcode::I64Ctz)
    } else {
        (WasmOpcode::I32Clz, WasmOpcode::I32Ctz)
    };
    let activation = ops_lc(&[leading, trailing]);
    let width = input.len();
    let mut source = vec![Lc::zero(); width];
    for low in 0..width / 2 {
        let high = width - 1 - low;
        let chosen = builder.alloc_select(&opcode_lc(leading), &Lc::var(input[high]), &Lc::var(input[low]));
        source[low] = Lc::var(chosen);
        source[high] = Lc::var(input[low])
            .plus(&Lc::var(input[high]))
            .minus(&Lc::var(chosen));
    }
    let mut previous = activation.clone();
    let mut count = Lc::zero();
    for bit in &source {
        let next = builder.alloc_product(&previous, &builder.one().minus(bit));
        count = count.plus(&Lc::var(next));
        previous = Lc::var(next);
    }
    enforce_count_output(builder, &activation, output, &count);
    Ok(())
}

pub(super) fn enforce_popcount(
    builder: &mut CircuitBuilder,
    input: &[ColumnId],
    output: &[ColumnId],
    operation: WasmOpcode,
) -> Result<(), String> {
    let count = input
        .iter()
        .fold(Lc::zero(), |sum, &bit| sum.plus(&Lc::var(bit)));
    enforce_count_output(builder, &opcode_lc(operation), output, &count);
    Ok(())
}

/// A count is at most 64. Pack only the low 32 output bits and force the
/// rest to zero: a 64-bit packing can wrap modulo p.
pub(super) fn enforce_count_output(builder: &mut CircuitBuilder, gate: &Lc, output: &[ColumnId], count: &Lc) {
    builder.enforce_equal_when(gate, &Lc::from_bits(&output[..32]), count);
    builder.enforce_bits_zero_when(gate, &output[32..]);
}
