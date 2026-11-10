use super::builder::{CircuitBuilder, Lc};
use super::{ops_lc, Inputs};
use crate::isa::WasmOpcode;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// One product per bit serves AND, OR and XOR of both widths: AND = l*r,
/// OR = l + r - l*r, XOR = l + r - 2*l*r.
pub(super) fn enforce_bitwise(builder: &mut CircuitBuilder, inputs: &Inputs) -> Result<(), String> {
    use WasmOpcode::*;
    for index in 0..64 {
        let gate = |narrow: WasmOpcode, wide: WasmOpcode| {
            if index < 32 {
                ops_lc(&[narrow, wide])
            } else {
                ops_lc(&[wide])
            }
        };
        let left = Lc::var(inputs.left64[index]);
        let right = Lc::var(inputs.right64[index]);
        let output = Lc::var(inputs.output64[index]);
        let product = Lc::var(builder.alloc_product(&left, &right));
        let sum = left.plus(&right);
        builder.enforce_equal_when(&gate(I32And, I64And), &output, &product);
        builder.enforce_equal_when(&gate(I32Or, I64Or), &output, &sum.clone().minus(&product));
        builder.enforce_equal_when(
            &gate(I32Xor, I64Xor),
            &output,
            &sum.minus(&product.scaled(F::from_u64(2))),
        );
    }
    Ok(())
}
