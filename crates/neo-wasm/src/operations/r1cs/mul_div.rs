use super::builder::{bits_u64, eval_bit, selector_lc, CircuitBuilder, ColumnId, Lc};
use super::compare::no_borrow_bits;
use super::{limb, opcode_lc, Inputs, CARRY_BITS, LIMB_BITS};
use crate::isa::WasmOpcode;
use crate::layout::{COL_DIV_OVERFLOW_COND, COL_DIV_TRAP};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// One 64-bit multiplier with field limb products. `i32.mul` reads the low
/// two limbs, which depend only on the low input limbs.
pub(super) fn enforce_mul(builder: &mut CircuitBuilder, inputs: &Inputs) -> Result<(), String> {
    let narrow = opcode_lc(WasmOpcode::I32Mul);
    let wide = opcode_lc(WasmOpcode::I64Mul);
    let limbs = 4;
    let mut products = vec![Vec::<ColumnId>::new(); limbs];
    for left_limb in 0..limbs {
        for right_limb in 0..limbs - left_limb {
            products[left_limb + right_limb].push(builder.alloc_limb_product(
                &Lc::from_bits(limb(&inputs.left64, left_limb)),
                &Lc::from_bits(limb(&inputs.right64, right_limb)),
            ));
        }
    }
    let mut carry = Lc::zero();
    for index in 0..limbs {
        let gate = if index < 2 {
            narrow.clone().plus(&wide)
        } else {
            wide.clone()
        };
        let (terms, previous, active) = (products[index].clone(), carry.clone(), gate.clone());
        let carry_bits = builder.alloc_bits("carry", CARRY_BITS, move |z| {
            let total = terms
                .iter()
                .map(|&column| z[column.0].as_canonical_u64())
                .sum::<u64>()
                + previous.evaluate(z).as_canonical_u64();
            Ok(if eval_bit(z, &active)? { total >> LIMB_BITS } else { 0 })
        });
        let mut equation = carry.minus(&Lc::from_bits(limb(&inputs.output64, index)));
        for &product in &products[index] {
            equation = equation.plus(&Lc::var(product));
        }
        equation = equation.add_scaled(&Lc::from_bits(&carry_bits), -F::from_u64(1 << LIMB_BITS));
        builder.enforce_zero_when(&gate, &equation);
        builder.enforce_bits_zero_when(&builder.one().minus(&gate), &carry_bits);
        carry = Lc::from_bits(&carry_bits);
    }
    Ok(())
}

pub(super) fn enforce_div_rem(
    builder: &mut CircuitBuilder,
    dividend: &[ColumnId],
    divisor: &[ColumnId],
    output: &[ColumnId],
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
    // A trapping row (zero divisor, or signed overflow for div_s) has no result.
    let enabled = builder.one().minus(&Lc::var(COL_DIV_TRAP));
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
    let (active, left, right) = (active_bits.clone(), dividend.to_vec(), divisor.to_vec());
    let values = move |z: &[F]| {
        let mut selected = None;
        for &(operation, column) in &active {
            if eval_bit(z, &Lc::var(column))? {
                selected = Some(operation);
                break;
            }
        }
        division_values(bits_u64(z, &left)?, bits_u64(z, &right)?, selected, wide)
    };
    let quotient_values = values.clone();
    let quotient = builder.alloc_bits("quotient", dividend.len(), move |z| Ok(quotient_values(z)?.0));
    let remainder = builder.alloc_bits("remainder", dividend.len(), move |z| Ok(values(z)?.1));
    let inactive = builder.one().minus(&activation);
    for bits in [&quotient, &remainder] {
        builder.enforce_bits_zero_when(&inactive, bits);
        builder.enforce_bits_zero_when(&Lc::var(special_remainder), bits);
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
        builder.enforce_bits_equal_when(&Lc::var(*active), output, expected);
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
    builder.enforce_bits_zero_when(&Lc::var(quotient_sign_mismatch), &quotient_abs);
    builder.enforce_bits_zero_when(&Lc::var(remainder_sign_mismatch), &remainder_abs);
    enforce_exact_division_identity(
        builder,
        &regular_activation,
        &dividend_abs,
        &divisor_abs,
        &quotient_abs,
        &remainder_abs,
    )?;
    // remainder < divisor: the subtraction remainder - divisor borrows.
    let tops = no_borrow_bits(builder, &remainder_abs, &divisor_abs, &Lc::zero())?;
    builder.enforce_zero_when(&regular_activation, &Lc::var(*tops.last().unwrap()));
    Ok(())
}

pub(super) fn enforce_exact_division_identity(
    builder: &mut CircuitBuilder,
    gate: &Lc,
    dividend: &[ColumnId],
    divisor: &[ColumnId],
    quotient: &[ColumnId],
    remainder: &[ColumnId],
) -> Result<(), String> {
    let limbs = dividend.len() / LIMB_BITS;
    let mut products = vec![Vec::<ColumnId>::new(); 2 * limbs - 1];
    for left_limb in 0..limbs {
        for right_limb in 0..limbs {
            products[left_limb + right_limb].push(builder.alloc_limb_product(
                &Lc::from_bits(limb(divisor, left_limb)),
                &Lc::from_bits(limb(quotient, right_limb)),
            ));
        }
    }
    let mut carry = Lc::zero();
    for index in 0..2 * limbs {
        let mut equation = carry.clone();
        for &product in products.get(index).into_iter().flatten() {
            equation = equation.plus(&Lc::var(product));
        }
        if index < limbs {
            equation = equation.plus(&Lc::from_bits(limb(remainder, index)));
            equation = equation.minus(&Lc::from_bits(limb(dividend, index)));
        }
        if index + 1 < 2 * limbs {
            let terms = products.get(index).cloned().unwrap_or_default();
            let remainder_limb = if index < limbs {
                Lc::from_bits(limb(remainder, index))
            } else {
                Lc::zero()
            };
            let (previous, active) = (carry.clone(), gate.clone());
            let next = builder.alloc_bits("division_carry", CARRY_BITS, move |z| {
                let total = terms
                    .iter()
                    .map(|&column| z[column.0].as_canonical_u64())
                    .sum::<u64>()
                    + previous.evaluate(z).as_canonical_u64()
                    + remainder_limb.evaluate(z).as_canonical_u64();
                Ok(if eval_bit(z, &active)? { total >> LIMB_BITS } else { 0 })
            });
            builder.enforce_bits_zero_when(&builder.one().minus(gate), &next);
            equation = equation.add_scaled(&Lc::from_bits(&next), -F::from_u64(1 << LIMB_BITS));
            carry = Lc::from_bits(&next);
        }
        builder.enforce_zero_when(gate, &equation);
    }
    Ok(())
}

pub(super) fn conditional_abs(
    builder: &mut CircuitBuilder,
    input: &[ColumnId],
    negative: &Lc,
) -> Result<Vec<ColumnId>, String> {
    let (source, sign) = (input.to_vec(), negative.clone());
    let output = builder.alloc_bits("absolute", input.len(), move |z| {
        let value = bits_u64(z, &source)?;
        let mask = if source.len() == 64 {
            u64::MAX
        } else {
            (1u64 << source.len()) - 1
        };
        Ok(if eval_bit(z, &sign)? {
            0u64.wrapping_sub(value) & mask
        } else {
            value
        })
    });
    let not_negative = builder.one().minus(negative);
    let limbs = input.len() / LIMB_BITS;
    let mut carry = Lc::zero();
    for index in 0..limbs {
        builder.enforce_equal_when(
            &not_negative,
            &Lc::from_bits(limb(&output, index)),
            &Lc::from_bits(limb(input, index)),
        );
        let total = Lc::from_bits(limb(input, index))
            .plus(&Lc::from_bits(limb(&output, index)))
            .plus(&carry);
        let sign = negative.clone();
        let next = builder.alloc_bit("absolute_carry", move |z| {
            Ok(eval_bit(z, &sign)? && total.evaluate(z).as_canonical_u64() >> LIMB_BITS == 1)
        });
        let equation = Lc::from_bits(limb(input, index))
            .plus(&Lc::from_bits(limb(&output, index)))
            .plus(&carry)
            .add_scaled(&Lc::var(next), -F::from_u64(1 << LIMB_BITS));
        builder.enforce_zero_when(negative, &equation);
        builder.enforce_zero_when(&not_negative, &Lc::var(next));
        carry = Lc::var(next);
    }
    builder.enforce_equal_when(negative, &carry, &builder.one());
    Ok(output)
}

/// Quotient and remainder that the assignment writes for the active div/rem
/// operation. `rem_s(MIN, -1)` is 0; trapping rows never reach here.
fn division_values(
    dividend: u64,
    divisor: u64,
    operation: Option<WasmOpcode>,
    wide: bool,
) -> Result<(u64, u64), String> {
    let Some(operation) = operation else {
        return Ok((0, 0));
    };
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
