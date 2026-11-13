use super::builder::{CircuitBuilder, ColumnId, Lc};
use super::opcode_lc;
use crate::isa::WasmOpcode;

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
