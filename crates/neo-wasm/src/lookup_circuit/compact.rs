//! Compact, verifier-fixed R1CS semantics for the 48 operation-table families.

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::builder::{selector_lc, Bit, Lc, LookupR1csBuilder};
use crate::isa::WasmOpcode;
use crate::layout::{
    selector_col, COL_DIV_OVERFLOW_COND, COL_OP_TABLE_ENABLED, COL_STACK_READ0_VALUE_HI, COL_STACK_READ0_VALUE_LO,
    COL_STACK_READ1_VALUE_HI, COL_STACK_READ1_VALUE_LO, COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO,
};
use crate::range_check::range_checked_bit_columns;
use crate::tagged_r1cs_builder::WasmR1csRow;

const LIMB_BITS: usize = 16;
const PRODUCT_BITS: usize = 32;
const CARRY_BITS: usize = 18;

struct Inputs {
    left32: Vec<Bit>,
    right32: Vec<Bit>,
    output32: Vec<Bit>,
    left64: Vec<Bit>,
    right64: Vec<Bit>,
    output64: Vec<Bit>,
}

pub(super) fn synthesize(base_assignment: &[F]) -> Result<(Vec<WasmR1csRow>, Vec<F>), String> {
    let inputs = Inputs::new()?;
    let mut builder = LookupR1csBuilder::new(base_assignment);

    enforce_bitwise(&mut builder, &inputs.left32, &inputs.right32, &inputs.output32, false)?;
    enforce_bitwise(&mut builder, &inputs.left64, &inputs.right64, &inputs.output64, true)?;
    enforce_comparisons(&mut builder, &inputs.left32, &inputs.right32, &inputs.output32, false)?;
    enforce_comparisons(&mut builder, &inputs.left64, &inputs.right64, &inputs.output64, true)?;
    enforce_shifts(&mut builder, &inputs.left32, &inputs.right32, &inputs.output32, false)?;
    enforce_shifts(&mut builder, &inputs.left64, &inputs.right64, &inputs.output64, true)?;
    enforce_mul(
        &mut builder,
        &inputs.left32,
        &inputs.right32,
        &inputs.output32,
        WasmOpcode::I32Mul,
    )?;
    enforce_mul(
        &mut builder,
        &inputs.left64,
        &inputs.right64,
        &inputs.output64,
        WasmOpcode::I64Mul,
    )?;
    enforce_zero_counts(&mut builder, &inputs.left32, &inputs.output32, false)?;
    enforce_zero_counts(&mut builder, &inputs.left64, &inputs.output64, true)?;
    enforce_popcount(&mut builder, &inputs.left32, &inputs.output32, WasmOpcode::I32Popcnt)?;
    enforce_popcount(&mut builder, &inputs.left64, &inputs.output64, WasmOpcode::I64Popcnt)?;
    enforce_div_rem(&mut builder, &inputs.left32, &inputs.right32, &inputs.output32, false)?;
    enforce_div_rem(&mut builder, &inputs.left64, &inputs.right64, &inputs.output64, true)?;

    Ok(builder.finish())
}

impl Inputs {
    fn new() -> Result<Self, String> {
        let left32 = column_bits(COL_STACK_READ0_VALUE_LO)?;
        let right32 = column_bits(COL_STACK_READ1_VALUE_LO)?;
        let output32 = column_bits(COL_STACK_WRITE0_VALUE_LO)?;
        let left64 = wide_bits(COL_STACK_READ0_VALUE_LO, COL_STACK_READ0_VALUE_HI)?;
        let right64 = wide_bits(COL_STACK_READ1_VALUE_LO, COL_STACK_READ1_VALUE_HI)?;
        let output64 = wide_bits(COL_STACK_WRITE0_VALUE_LO, COL_STACK_WRITE0_VALUE_HI)?;
        Ok(Self {
            left32,
            right32,
            output32,
            left64,
            right64,
            output64,
        })
    }
}

fn enforce_bitwise(
    builder: &mut LookupR1csBuilder,
    left: &[Bit],
    right: &[Bit],
    output: &[Bit],
    wide: bool,
) -> Result<(), String> {
    let operations = if wide {
        [WasmOpcode::I64And, WasmOpcode::I64Or, WasmOpcode::I64Xor]
    } else {
        [WasmOpcode::I32And, WasmOpcode::I32Or, WasmOpcode::I32Xor]
    };
    for operation in operations {
        let gate = opcode_lc(operation);
        for ((&left, &right), &output) in left.iter().zip(right).zip(output) {
            let masked_left = builder.mask_bit(&gate, left)?;
            let product = builder.alloc_and(&Lc::var(masked_left), &Lc::var(right))?;
            match operation {
                WasmOpcode::I32And | WasmOpcode::I64And => {
                    builder.enforce_equal_when(&gate, &Lc::var(output), &Lc::var(product));
                }
                WasmOpcode::I32Or | WasmOpcode::I64Or => {
                    let sum = Lc::var(left).plus(&Lc::var(right)).minus(&Lc::var(output));
                    builder.enforce_product(gate.clone(), sum, Lc::var(product));
                }
                WasmOpcode::I32Xor | WasmOpcode::I64Xor => {
                    let sum = Lc::var(left).plus(&Lc::var(right)).minus(&Lc::var(output));
                    builder.enforce_product(gate.clone(), sum, Lc::var(product).scaled(F::from_u64(2)));
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

fn enforce_comparisons(
    builder: &mut LookupR1csBuilder,
    left: &[Bit],
    right: &[Bit],
    output: &[Bit],
    wide: bool,
) -> Result<(), String> {
    let operations = comparison_operations(wide);
    let selectors = operations
        .iter()
        .map(|&(operation, _, _, _)| opcode_selector(operation))
        .collect::<Vec<_>>();
    let activation = selector_lc(&selectors);
    let inactive = Lc::one().minus(&activation);
    let selected = selected_opcode(
        builder,
        &selectors,
        &operations.iter().map(|entry| entry.0).collect::<Vec<_>>(),
    )?;
    let config = selected.and_then(|selected| {
        operations
            .iter()
            .find(|(operation, _, _, _)| *operation == selected)
            .copied()
    });

    let mut selected_left = Vec::with_capacity(left.len());
    let mut selected_right = Vec::with_capacity(right.len());
    for bit in 0..left.len() {
        let (left_value, right_value) = if let Some((_, swap, _, _)) = config {
            if swap {
                (builder.value_bit(right[bit])?, builder.value_bit(left[bit])?)
            } else {
                (builder.value_bit(left[bit])?, builder.value_bit(right[bit])?)
            }
        } else {
            (false, false)
        };
        let left_wire = builder.alloc_bit(left_value);
        let right_wire = builder.alloc_bit(right_value);
        for &(operation, swap, _, _) in &operations {
            let gate = opcode_lc(operation);
            let left_source = if swap { right[bit] } else { left[bit] };
            let right_source = if swap { left[bit] } else { right[bit] };
            builder.enforce_equal_when(&gate, &Lc::var(left_wire), &Lc::var(left_source));
            builder.enforce_equal_when(&gate, &Lc::var(right_wire), &Lc::var(right_source));
        }
        builder.enforce_zero_when(&inactive, &Lc::var(left_wire));
        builder.enforce_zero_when(&inactive, &Lc::var(right_wire));
        selected_left.push(left_wire);
        selected_right.push(right_wire);
    }

    let signed = selector_lc(
        &operations
            .iter()
            .filter(|(_, _, signed, _)| *signed)
            .map(|(operation, _, _, _)| opcode_selector(*operation))
            .collect::<Vec<_>>(),
    );
    let last = left.len() - 1;
    selected_left[last] = builder.alloc_xor(&Lc::var(selected_left[last]), &signed)?;
    selected_right[last] = builder.alloc_xor(&Lc::var(selected_right[last]), &signed)?;
    let less = enforce_less_than(builder, &selected_left, &selected_right)?;
    let invert = selector_lc(
        &operations
            .iter()
            .filter(|(_, _, _, invert)| *invert)
            .map(|(operation, _, _, _)| opcode_selector(*operation))
            .collect::<Vec<_>>(),
    );
    let result = builder.alloc_xor(&Lc::var(less), &invert)?;
    builder.enforce_equal_when(&activation, &Lc::var(output[0]), &Lc::var(result));
    for &bit in &output[1..] {
        builder.enforce_zero_when(&activation, &Lc::var(bit));
    }
    Ok(())
}

fn enforce_shifts(
    builder: &mut LookupR1csBuilder,
    value: &[Bit],
    amount: &[Bit],
    output: &[Bit],
    wide: bool,
) -> Result<(), String> {
    let operations = if wide {
        [
            WasmOpcode::I64Shl,
            WasmOpcode::I64ShrU,
            WasmOpcode::I64ShrS,
            WasmOpcode::I64Rotl,
            WasmOpcode::I64Rotr,
        ]
    } else {
        [
            WasmOpcode::I32Shl,
            WasmOpcode::I32ShrU,
            WasmOpcode::I32ShrS,
            WasmOpcode::I32Rotl,
            WasmOpcode::I32Rotr,
        ]
    };
    let selectors = operations.map(opcode_selector);
    let activation = selector_lc(&selectors);
    let inactive = Lc::one().minus(&activation);
    let selected = selected_opcode(builder, &selectors, &operations)?;
    let mut current = value.to_vec();
    for (stage, &amount_bit) in amount
        .iter()
        .take(value.len().trailing_zeros() as usize)
        .enumerate()
    {
        let distance = 1usize << stage;
        let mut next = Vec::with_capacity(value.len());
        for index in 0..value.len() {
            let shifted_value = selected
                .and_then(|operation| shift_source(operation, &current, index, distance))
                .map(|source| builder.value_bit(source))
                .transpose()?
                .unwrap_or(false);
            let shifted = builder.alloc_bit(shifted_value);
            for operation in operations {
                let source = shift_source(operation, &current, index, distance);
                match source {
                    Some(source) => {
                        builder.enforce_equal_when(&opcode_lc(operation), &Lc::var(shifted), &Lc::var(source));
                    }
                    None => builder.enforce_zero_when(&opcode_lc(operation), &Lc::var(shifted)),
                }
            }
            builder.enforce_zero_when(&inactive, &Lc::var(shifted));
            next.push(builder.alloc_mux(&Lc::var(amount_bit), &Lc::var(shifted), &Lc::var(current[index]))?);
        }
        current = next;
    }
    for (&actual, &expected) in output.iter().zip(&current) {
        builder.enforce_equal_when(&activation, &Lc::var(actual), &Lc::var(expected));
    }
    Ok(())
}

fn enforce_mul(
    builder: &mut LookupR1csBuilder,
    left: &[Bit],
    right: &[Bit],
    output: &[Bit],
    operation: WasmOpcode,
) -> Result<(), String> {
    let gate = opcode_lc(operation);
    let product_value = bits_u64(builder, left)?.wrapping_mul(bits_u64(builder, right)?);
    let product_output = allocate_u64_bits(builder, product_value, output.len());
    let limbs = left.len() / LIMB_BITS;
    let mut products = vec![Vec::<Lc>::new(); limbs];
    for left_limb in 0..limbs {
        for right_limb in 0..limbs - left_limb {
            let product = allocate_product_bits(builder, limb(&left, left_limb), limb(&right, right_limb))?;
            products[left_limb + right_limb].push(Lc::from_bits(&product));
        }
    }
    let mut carry = Lc::zero();
    let mut carry_value = 0u64;
    for index in 0..limbs {
        let product_value = products[index]
            .iter()
            .map(|product| lc_u64(builder, product))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<u64>();
        let total = product_value + carry_value;
        carry_value = total >> LIMB_BITS;
        let carry_bits = allocate_u64_bits(builder, carry_value, CARRY_BITS);
        let mut equation = carry
            .clone()
            .minus(&Lc::from_bits(limb(&product_output, index)));
        for product in &products[index] {
            equation = equation.plus(product);
        }
        equation = equation.add_scaled(&Lc::from_bits(&carry_bits), -F::from_u64(1 << LIMB_BITS));
        builder.enforce_linear_zero(equation);
        carry = Lc::from_bits(&carry_bits);
    }
    for (&actual, &expected) in output.iter().zip(&product_output) {
        builder.enforce_equal_when(&gate, &Lc::var(actual), &Lc::var(expected));
    }
    Ok(())
}

fn enforce_zero_counts(
    builder: &mut LookupR1csBuilder,
    input: &[Bit],
    output: &[Bit],
    wide: bool,
) -> Result<(), String> {
    let (leading, trailing) = if wide {
        (WasmOpcode::I64Clz, WasmOpcode::I64Ctz)
    } else {
        (WasmOpcode::I32Clz, WasmOpcode::I32Ctz)
    };
    let activation = selector_lc(&[opcode_selector(leading), opcode_selector(trailing)]);
    let inactive = Lc::one().minus(&activation);
    let selected = selected_opcode(
        builder,
        &[opcode_selector(leading), opcode_selector(trailing)],
        &[leading, trailing],
    )?;
    let mut previous = activation.clone();
    let mut prefix = Vec::with_capacity(input.len());
    for index in 0..input.len() {
        let source_index = if selected == Some(leading) {
            input.len() - 1 - index
        } else {
            index
        };
        let source = builder.alloc_bit(selected.is_some() && builder.value_bit(input[source_index])?);
        builder.enforce_equal_when(
            &opcode_lc(leading),
            &Lc::var(source),
            &Lc::var(input[input.len() - 1 - index]),
        );
        builder.enforce_equal_when(&opcode_lc(trailing), &Lc::var(source), &Lc::var(input[index]));
        builder.enforce_zero_when(&inactive, &Lc::var(source));
        let next = builder.alloc_and(&previous, &Lc::one().minus(&Lc::var(source)))?;
        prefix.push(next);
        previous = Lc::var(next);
    }
    let count = prefix
        .iter()
        .fold(Lc::zero(), |sum, &bit| sum.plus(&Lc::var(bit)));
    builder.enforce_equal_when(&activation, &Lc::from_bits(output), &count);
    Ok(())
}

fn enforce_popcount(
    builder: &mut LookupR1csBuilder,
    input: &[Bit],
    output: &[Bit],
    operation: WasmOpcode,
) -> Result<(), String> {
    let count = input
        .iter()
        .fold(Lc::zero(), |sum, &bit| sum.plus(&Lc::var(bit)));
    builder.enforce_equal_when(&opcode_lc(operation), &Lc::from_bits(output), &count);
    Ok(())
}

fn enforce_div_rem(
    builder: &mut LookupR1csBuilder,
    dividend: &[Bit],
    divisor: &[Bit],
    output: &[Bit],
    wide: bool,
) -> Result<(), String> {
    let operations = if wide {
        [
            WasmOpcode::I64DivU,
            WasmOpcode::I64DivS,
            WasmOpcode::I64RemU,
            WasmOpcode::I64RemS,
        ]
    } else {
        [
            WasmOpcode::I32DivU,
            WasmOpcode::I32DivS,
            WasmOpcode::I32RemU,
            WasmOpcode::I32RemS,
        ]
    };
    let enabled = Lc::var(COL_OP_TABLE_ENABLED);
    let mut active_bits = Vec::with_capacity(operations.len());
    for operation in operations {
        active_bits.push((operation, builder.alloc_and(&enabled, &opcode_lc(operation))?));
    }
    let activation = selector_lc(&active_bits.iter().map(|(_, bit)| *bit).collect::<Vec<_>>());
    let signed_remainder = if wide { WasmOpcode::I64RemS } else { WasmOpcode::I32RemS };
    let signed_remainder_active = active_bits
        .iter()
        .find_map(|(operation, bit)| (*operation == signed_remainder).then_some(*bit))
        .expect("signed remainder belongs to the selected width");
    let special_remainder = builder.alloc_and(&Lc::var(signed_remainder_active), &Lc::var(COL_DIV_OVERFLOW_COND))?;
    let regular_activation = activation.clone().minus(&Lc::var(special_remainder));
    let signed = selector_lc(
        &active_bits
            .iter()
            .filter(|(operation, _)| {
                matches!(
                    operation,
                    WasmOpcode::I32DivS | WasmOpcode::I32RemS | WasmOpcode::I64DivS | WasmOpcode::I64RemS
                )
            })
            .map(|(_, bit)| *bit)
            .collect::<Vec<_>>(),
    );
    let selected = active_bits.iter().find_map(|(operation, bit)| {
        builder
            .value_bit(*bit)
            .ok()
            .filter(|value| *value)
            .map(|_| *operation)
    });
    let (quotient_value, remainder_value) = division_values(builder, dividend, divisor, selected, wide)?;
    let quotient = allocate_u64_bits(builder, quotient_value, dividend.len());
    let remainder = allocate_u64_bits(builder, remainder_value, dividend.len());
    let inactive = Lc::one().minus(&activation);
    for &bit in quotient.iter().chain(&remainder) {
        builder.enforce_zero_when(&inactive, &Lc::var(bit));
        builder.enforce_zero_when(&Lc::var(special_remainder), &Lc::var(bit));
    }
    for (operation, active) in &active_bits {
        let expected = if matches!(
            operation,
            WasmOpcode::I32DivU | WasmOpcode::I32DivS | WasmOpcode::I64DivU | WasmOpcode::I64DivS
        ) {
            &quotient
        } else {
            &remainder
        };
        for (&actual, &expected) in output.iter().zip(expected) {
            builder.enforce_equal_when(&Lc::var(*active), &Lc::var(actual), &Lc::var(expected));
        }
    }

    let dividend_negative = builder.alloc_and(&signed, &Lc::var(*dividend.last().unwrap()))?;
    let divisor_negative = builder.alloc_and(&signed, &Lc::var(*divisor.last().unwrap()))?;
    let quotient_expected_negative = builder.alloc_xor(&Lc::var(dividend_negative), &Lc::var(divisor_negative))?;
    let quotient_actual_negative = builder.alloc_and(&signed, &Lc::var(*quotient.last().unwrap()))?;
    let remainder_actual_negative = builder.alloc_and(&signed, &Lc::var(*remainder.last().unwrap()))?;
    let dividend_abs = conditional_abs(builder, dividend, &Lc::var(dividend_negative))?;
    let divisor_abs = conditional_abs(builder, divisor, &Lc::var(divisor_negative))?;
    let quotient_abs = conditional_abs(builder, &quotient, &Lc::var(quotient_actual_negative))?;
    let remainder_abs = conditional_abs(builder, &remainder, &Lc::var(remainder_actual_negative))?;
    let quotient_sign_mismatch =
        builder.alloc_xor(&Lc::var(quotient_expected_negative), &Lc::var(quotient_actual_negative))?;
    let remainder_sign_mismatch =
        builder.alloc_xor(&Lc::var(dividend_negative), &Lc::var(remainder_actual_negative))?;
    for &bit in &quotient_abs {
        builder.enforce_zero_when(&Lc::var(quotient_sign_mismatch), &Lc::var(bit));
    }
    for &bit in &remainder_abs {
        builder.enforce_zero_when(&Lc::var(remainder_sign_mismatch), &Lc::var(bit));
    }
    let dividend_active = mask_bits(builder, &regular_activation, &dividend_abs)?;
    let divisor_active = mask_bits(builder, &regular_activation, &divisor_abs)?;
    enforce_exact_division_identity(
        builder,
        &dividend_active,
        &divisor_active,
        &quotient_abs,
        &remainder_abs,
    )?;
    let less = enforce_less_than(builder, &remainder_abs, &divisor_active)?;
    builder.enforce_equal_when(&regular_activation, &Lc::var(less), &Lc::one());
    Ok(())
}

fn enforce_exact_division_identity(
    builder: &mut LookupR1csBuilder,
    dividend: &[Bit],
    divisor: &[Bit],
    quotient: &[Bit],
    remainder: &[Bit],
) -> Result<(), String> {
    let limbs = dividend.len() / LIMB_BITS;
    let mut products = vec![Vec::<Lc>::new(); 2 * limbs - 1];
    for left_limb in 0..limbs {
        for right_limb in 0..limbs {
            let product = allocate_product_bits(builder, limb(divisor, left_limb), limb(quotient, right_limb))?;
            products[left_limb + right_limb].push(Lc::from_bits(&product));
        }
    }
    let mut carry = Lc::zero();
    let mut carry_value = 0u64;
    for index in 0..2 * limbs {
        let product_value = products
            .get(index)
            .into_iter()
            .flatten()
            .map(|product| lc_u64(builder, product))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<u64>();
        let remainder_value = if index < limbs {
            bits_u64(builder, limb(remainder, index))?
        } else {
            0
        };
        let total = product_value + remainder_value + carry_value;
        let mut equation = carry.clone();
        if let Some(group) = products.get(index) {
            for product in group {
                equation = equation.plus(product);
            }
        }
        if index < limbs {
            equation = equation.plus(&Lc::from_bits(limb(remainder, index)));
            equation = equation.minus(&Lc::from_bits(limb(dividend, index)));
        }
        if index + 1 < 2 * limbs {
            carry_value = total >> LIMB_BITS;
            let next = allocate_u64_bits(builder, carry_value, CARRY_BITS);
            equation = equation.add_scaled(&Lc::from_bits(&next), -F::from_u64(1 << LIMB_BITS));
            carry = Lc::from_bits(&next);
        }
        builder.enforce_linear_zero(equation);
    }
    Ok(())
}

fn enforce_less_than(builder: &mut LookupR1csBuilder, left: &[Bit], right: &[Bit]) -> Result<Bit, String> {
    let mut borrow = false;
    let mut borrow_lc = Lc::zero();
    let mut last = None;
    for (&left, &right) in left.iter().zip(right) {
        let left_value = builder.value_bit(left)?;
        let right_value = builder.value_bit(right)?;
        let signed = i8::from(left_value) - i8::from(right_value) - i8::from(borrow);
        let difference = builder.alloc_bit(signed.rem_euclid(2) == 1);
        let next_borrow = builder.alloc_bit(signed < 0);
        let equation = Lc::var(left)
            .minus(&Lc::var(right))
            .minus(&borrow_lc)
            .minus(&Lc::var(difference))
            .add_scaled(&Lc::var(next_borrow), F::from_u64(2));
        builder.enforce_linear_zero(equation);
        borrow = signed < 0;
        borrow_lc = Lc::var(next_borrow);
        last = Some(next_borrow);
    }
    last.ok_or_else(|| "lookup comparison width is zero".to_owned())
}

fn conditional_abs(builder: &mut LookupR1csBuilder, input: &[Bit], negative: &Lc) -> Result<Vec<Bit>, String> {
    let negative_value = builder.eval_bit(negative)?;
    let value = bits_u64(builder, input)?;
    let mask = if input.len() == u64::BITS as usize {
        u64::MAX
    } else {
        (1u64 << input.len()) - 1
    };
    let absolute = if negative_value {
        0u64.wrapping_sub(value) & mask
    } else {
        value
    };
    let output = allocate_u64_bits(builder, absolute, input.len());
    let not_negative = Lc::one().minus(negative);
    let limbs = input.len() / LIMB_BITS;
    let mut carry = Lc::zero();
    let mut carry_value = 0u64;
    for index in 0..limbs {
        builder.enforce_equal_when(
            &not_negative,
            &Lc::from_bits(limb(&output, index)),
            &Lc::from_bits(limb(input, index)),
        );
        let total = bits_u64(builder, limb(input, index))? + bits_u64(builder, limb(&output, index))? + carry_value;
        carry_value = if negative_value { total >> LIMB_BITS } else { 0 };
        let next = builder.alloc_bit(carry_value == 1);
        let equation = Lc::from_bits(limb(input, index))
            .plus(&Lc::from_bits(limb(&output, index)))
            .plus(&carry)
            .add_scaled(&Lc::var(next), -F::from_u64(1 << LIMB_BITS));
        builder.enforce_zero_when(negative, &equation);
        builder.enforce_zero_when(&not_negative, &Lc::var(next));
        carry = Lc::var(next);
    }
    builder.enforce_equal_when(negative, &carry, &Lc::one());
    Ok(output)
}

fn mask_bits(builder: &mut LookupR1csBuilder, gate: &Lc, bits: &[Bit]) -> Result<Vec<Bit>, String> {
    bits.iter()
        .map(|&bit| builder.mask_bit(gate, bit))
        .collect()
}

fn allocate_product_bits(builder: &mut LookupR1csBuilder, left: &[Bit], right: &[Bit]) -> Result<Vec<Bit>, String> {
    let value = bits_u64(builder, left)? * bits_u64(builder, right)?;
    let output = allocate_u64_bits(builder, value, PRODUCT_BITS);
    builder.enforce_product(Lc::from_bits(left), Lc::from_bits(right), Lc::from_bits(&output));
    Ok(output)
}

fn allocate_u64_bits(builder: &mut LookupR1csBuilder, value: u64, width: usize) -> Vec<Bit> {
    (0..width)
        .map(|bit| builder.alloc_bit(((value >> bit) & 1) == 1))
        .collect()
}

fn division_values(
    builder: &LookupR1csBuilder,
    dividend: &[Bit],
    divisor: &[Bit],
    operation: Option<WasmOpcode>,
    wide: bool,
) -> Result<(u64, u64), String> {
    let Some(operation) = operation else {
        return Ok((0, 0));
    };
    let dividend = bits_u64(builder, dividend)?;
    let divisor = bits_u64(builder, divisor)?;
    if divisor == 0 {
        return Err("active division has a zero divisor".into());
    }
    if wide {
        if matches!(operation, WasmOpcode::I64DivS | WasmOpcode::I64RemS) {
            let left = dividend as i64;
            let right = divisor as i64;
            if operation == WasmOpcode::I64RemS && left == i64::MIN && right == -1 {
                return Ok((0, 0));
            }
            let quotient = left
                .checked_div(right)
                .ok_or_else(|| "active signed i64 division overflowed".to_owned())?;
            let remainder = left
                .checked_rem(right)
                .ok_or_else(|| "active signed i64 remainder overflowed".to_owned())?;
            Ok((quotient as u64, remainder as u64))
        } else {
            Ok((dividend / divisor, dividend % divisor))
        }
    } else if matches!(operation, WasmOpcode::I32DivS | WasmOpcode::I32RemS) {
        let left = dividend as u32 as i32;
        let right = divisor as u32 as i32;
        if operation == WasmOpcode::I32RemS && left == i32::MIN && right == -1 {
            return Ok((0, 0));
        }
        let quotient = left
            .checked_div(right)
            .ok_or_else(|| "active signed i32 division overflowed".to_owned())?;
        let remainder = left
            .checked_rem(right)
            .ok_or_else(|| "active signed i32 remainder overflowed".to_owned())?;
        Ok((quotient as u32 as u64, remainder as u32 as u64))
    } else {
        let left = dividend as u32;
        let right = divisor as u32;
        Ok((u64::from(left / right), u64::from(left % right)))
    }
}

fn comparison_operations(wide: bool) -> [(WasmOpcode, bool, bool, bool); 8] {
    if wide {
        [
            (WasmOpcode::I64LtS, false, true, false),
            (WasmOpcode::I64LtU, false, false, false),
            (WasmOpcode::I64GtS, true, true, false),
            (WasmOpcode::I64GtU, true, false, false),
            (WasmOpcode::I64LeS, true, true, true),
            (WasmOpcode::I64LeU, true, false, true),
            (WasmOpcode::I64GeS, false, true, true),
            (WasmOpcode::I64GeU, false, false, true),
        ]
    } else {
        [
            (WasmOpcode::I32LtS, false, true, false),
            (WasmOpcode::I32LtU, false, false, false),
            (WasmOpcode::I32GtS, true, true, false),
            (WasmOpcode::I32GtU, true, false, false),
            (WasmOpcode::I32LeS, true, true, true),
            (WasmOpcode::I32LeU, true, false, true),
            (WasmOpcode::I32GeS, false, true, true),
            (WasmOpcode::I32GeU, false, false, true),
        ]
    }
}

fn shift_source(operation: WasmOpcode, current: &[Bit], index: usize, distance: usize) -> Option<Bit> {
    let width = current.len();
    match operation {
        WasmOpcode::I32Shl | WasmOpcode::I64Shl => index.checked_sub(distance).map(|source| current[source]),
        WasmOpcode::I32ShrU | WasmOpcode::I64ShrU => current.get(index + distance).copied(),
        WasmOpcode::I32ShrS | WasmOpcode::I64ShrS => current
            .get(index + distance)
            .copied()
            .or_else(|| current.last().copied()),
        WasmOpcode::I32Rotl | WasmOpcode::I64Rotl => Some(current[(index + width - distance) % width]),
        WasmOpcode::I32Rotr | WasmOpcode::I64Rotr => Some(current[(index + distance) % width]),
        _ => unreachable!(),
    }
}

fn selected_opcode(
    builder: &LookupR1csBuilder,
    selectors: &[usize],
    operations: &[WasmOpcode],
) -> Result<Option<WasmOpcode>, String> {
    for (&selector, &operation) in selectors.iter().zip(operations) {
        if builder.value_bit(selector)? {
            return Ok(Some(operation));
        }
    }
    Ok(None)
}

fn limb(bits: &[Bit], index: usize) -> &[Bit] {
    &bits[index * LIMB_BITS..(index + 1) * LIMB_BITS]
}

fn bits_u64(builder: &LookupR1csBuilder, bits: &[Bit]) -> Result<u64, String> {
    bits.iter()
        .enumerate()
        .try_fold(0u64, |value, (bit, &column)| {
            Ok(value | (u64::from(builder.value_bit(column)?) << bit))
        })
}

fn lc_u64(builder: &LookupR1csBuilder, lc: &Lc) -> Result<u64, String> {
    lc.terms
        .iter()
        .try_fold(0u64, |value, &(column, coefficient)| {
            let coefficient = coefficient.as_canonical_u64();
            let bit = u64::from(builder.value_bit(column)?);
            value
                .checked_add(bit * coefficient)
                .ok_or_else(|| "lookup limb value overflow".to_owned())
        })
}

fn column_bits(column: usize) -> Result<Vec<Bit>, String> {
    range_checked_bit_columns(column)
        .map(Iterator::collect)
        .ok_or_else(|| format!("missing range-check bits for application column {column}"))
}

fn wide_bits(low: usize, high: usize) -> Result<Vec<Bit>, String> {
    let mut bits = column_bits(low)?;
    bits.extend(column_bits(high)?);
    Ok(bits)
}

fn opcode_selector(operation: WasmOpcode) -> usize {
    selector_col(operation).expect("operation-table opcode has a selector")
}

fn opcode_lc(operation: WasmOpcode) -> Lc {
    Lc::var(opcode_selector(operation))
}
