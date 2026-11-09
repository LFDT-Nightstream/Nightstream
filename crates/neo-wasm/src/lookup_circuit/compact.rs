//! Compact, verifier-fixed R1CS semantics for the 48 operation-table families.

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::builder::{selector_lc, Bit, Lc, LookupR1csBuilder, LookupR1csRow};
use crate::isa::WasmOpcode;
use crate::layout::{
    selector_col, COL_DIV_OVERFLOW_COND, COL_OP_TABLE_ENABLED, COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO,
    COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO,
};
use crate::range_check::range_checked_bit_columns;

const LIMB_BITS: usize = 16;
const CARRY_BITS: usize = 18;

struct Inputs {
    left32: Vec<Bit>,
    right32: Vec<Bit>,
    output32: Vec<Bit>,
    left64: Vec<Bit>,
    right64: Vec<Bit>,
    output64: Vec<Bit>,
}

type Section = fn(&mut LookupR1csBuilder, &Inputs) -> Result<(), String>;

const SECTIONS: &[(&str, Section)] = &[
    ("bitwise", |b, i| enforce_bitwise(b, i)),
    ("compare", |b, i| enforce_comparisons(b, i)),
    ("shift", |b, i| enforce_shifts(b, i)),
    ("mul", |b, i| enforce_mul(b, i)),
    ("zcount32", |b, i| enforce_zero_counts(b, &i.left32, &i.output32, false)),
    ("zcount64", |b, i| enforce_zero_counts(b, &i.left64, &i.output64, true)),
    ("popcnt32", |b, i| {
        enforce_popcount(b, &i.left32, &i.output32, WasmOpcode::I32Popcnt)
    }),
    ("popcnt64", |b, i| {
        enforce_popcount(b, &i.left64, &i.output64, WasmOpcode::I64Popcnt)
    }),
    ("divrem32", |b, i| {
        enforce_div_rem(b, &i.left32, &i.right32, &i.output32, false)
    }),
    ("divrem64", |b, i| {
        enforce_div_rem(b, &i.left64, &i.right64, &i.output64, true)
    }),
];

pub(super) fn synthesize(base_assignment: &[F]) -> Result<(Vec<LookupR1csRow>, Vec<F>, Vec<usize>), String> {
    let inputs = Inputs::new()?;
    let mut builder = LookupR1csBuilder::new(base_assignment);
    for (_, section) in SECTIONS {
        section(&mut builder, &inputs)?;
    }
    Ok(builder.finish())
}

impl Inputs {
    fn new() -> Result<Self, String> {
        let left32 = column_bits(COL_STACK_READ_VALUE_LO[0])?;
        let right32 = column_bits(COL_STACK_READ_VALUE_LO[1])?;
        let output32 = column_bits(COL_STACK_WRITE0_VALUE_LO)?;
        let left64 = wide_bits(COL_STACK_READ_VALUE_LO[0], COL_STACK_READ_VALUE_HI[0])?;
        let right64 = wide_bits(COL_STACK_READ_VALUE_LO[1], COL_STACK_READ_VALUE_HI[1])?;
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

/// One product per bit serves AND, OR and XOR of both widths: AND = l*r,
/// OR = l + r - l*r, XOR = l + r - 2*l*r.
fn enforce_bitwise(builder: &mut LookupR1csBuilder, inputs: &Inputs) -> Result<(), String> {
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

/// Allocate `left - right - borrow_in + 2^32` for each 32-bit half as 33
/// bits. The top bit of a half is 1 iff that prefix does not borrow.
fn no_borrow_bits(
    builder: &mut LookupR1csBuilder,
    left: &[Bit],
    right: &[Bit],
    borrow_in: &Lc,
) -> Result<Vec<Bit>, String> {
    let mut borrow = borrow_in.clone();
    let mut tops = Vec::new();
    for (left, right) in left.chunks(32).zip(right.chunks(32)) {
        let value = (i128::from(bits_u64(builder, left)?)
            - i128::from(bits_u64(builder, right)?)
            - i128::from(builder.eval(&borrow).as_canonical_u64())
            + (1i128 << 32)) as u64;
        let bits = allocate_u64_bits(builder, value, 33);
        builder.enforce_linear_zero(
            Lc::from_bits(&bits)
                .minus(&Lc::from_bits(left))
                .plus(&Lc::from_bits(right))
                .plus(&borrow)
                .minus(&Lc::one().scaled(F::from_u64(1 << 32))),
        );
        borrow = Lc::one().minus(&Lc::var(bits[32]));
        tops.push(bits[32]);
    }
    Ok(tops)
}

/// All 16 comparisons share one subtraction `l - r - c`. `c = 1` turns
/// `<` into `<=`; an inverted result gives `>` and `>=`. A signed result is
/// the unsigned result XOR both sign bits.
fn enforce_comparisons(builder: &mut LookupR1csBuilder, inputs: &Inputs) -> Result<(), String> {
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

/// One 64-bit left barrel shifter for both widths. Right shifts and `rotr`
/// reverse the input and the output, so every operation shifts left. The
/// fill of the low bits is zero, the wrapped bit (rotations), or the sign
/// (`shr_s`).
///
/// An i32 operand becomes 64 bits: two copies for rotations, the sign for
/// `shr_s`, and zeros otherwise. The 32-bit result is the low half of the
/// 64-bit result when the count is below 32, so i32 operations disable the
/// sixth stage and bind only the low 32 output bits.
fn enforce_shifts(builder: &mut LookupR1csBuilder, inputs: &Inputs) -> Result<(), String> {
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
    let shifted = current.iter().map(|lc| lc.terms[0].0).collect::<Vec<_>>();
    let reversed = shifted.iter().rev().copied().collect::<Vec<_>>();
    let forward = activation.minus(&reverse);
    let wide_forward = wide.minus(&wide_reverse);
    builder.enforce_bits_equal_when(&forward, &output[..NARROW], &shifted[..NARROW]);
    builder.enforce_bits_equal_when(&wide_forward, &output[NARROW..], &shifted[NARROW..]);
    builder.enforce_bits_equal_when(&reverse, &output[..NARROW], &reversed[..NARROW]);
    builder.enforce_bits_equal_when(&wide_reverse, &output[NARROW..], &reversed[NARROW..]);
    Ok(())
}

/// One 64-bit multiplier with field limb products. `i32.mul` reads the low
/// two limbs, which depend only on the low input limbs.
fn enforce_mul(builder: &mut LookupR1csBuilder, inputs: &Inputs) -> Result<(), String> {
    let narrow = opcode_lc(WasmOpcode::I32Mul);
    let wide = opcode_lc(WasmOpcode::I64Mul);
    let limbs = 4;
    let mut products = vec![Vec::<usize>::new(); limbs];
    for left_limb in 0..limbs {
        for right_limb in 0..limbs - left_limb {
            products[left_limb + right_limb].push(builder.alloc_limb_product(
                &Lc::from_bits(limb(&inputs.left64, left_limb)),
                &Lc::from_bits(limb(&inputs.right64, right_limb)),
            ));
        }
    }
    let mut carry = Lc::zero();
    let mut carry_value = 0u64;
    for index in 0..limbs {
        let gate = if index < 2 {
            narrow.clone().plus(&wide)
        } else {
            wide.clone()
        };
        let total = products[index]
            .iter()
            .map(|&product| builder.eval(&Lc::var(product)).as_canonical_u64())
            .sum::<u64>()
            + carry_value;
        carry_value = if builder.eval_bit(&gate)? {
            total >> LIMB_BITS
        } else {
            0
        };
        let carry_bits = allocate_u64_bits(builder, carry_value, CARRY_BITS);
        let mut equation = carry.minus(&Lc::from_bits(limb(&inputs.output64, index)));
        for &product in &products[index] {
            equation = equation.plus(&Lc::var(product));
        }
        equation = equation.add_scaled(&Lc::from_bits(&carry_bits), -F::from_u64(1 << LIMB_BITS));
        builder.enforce_zero_when(&gate, &equation);
        builder.enforce_bits_zero_when(&Lc::one().minus(&gate), &carry_bits);
        carry = Lc::from_bits(&carry_bits);
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
        let next = builder.alloc_product(&previous, &Lc::one().minus(bit));
        count = count.plus(&Lc::var(next));
        previous = Lc::var(next);
    }
    enforce_count_output(builder, &activation, output, &count);
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
    enforce_count_output(builder, &opcode_lc(operation), output, &count);
    Ok(())
}

/// A count is at most 64. Pack only the low 32 output bits and force the
/// rest to zero: a 64-bit packing can wrap modulo p.
fn enforce_count_output(builder: &mut LookupR1csBuilder, gate: &Lc, output: &[Bit], count: &Lc) {
    builder.enforce_equal_when(gate, &Lc::from_bits(&output[..32]), count);
    builder.enforce_bits_zero_when(gate, &output[32..]);
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

fn enforce_exact_division_identity(
    builder: &mut LookupR1csBuilder,
    gate: &Lc,
    dividend: &[Bit],
    divisor: &[Bit],
    quotient: &[Bit],
    remainder: &[Bit],
) -> Result<(), String> {
    let active = builder.eval_bit(gate)?;
    let limbs = dividend.len() / LIMB_BITS;
    let mut products = vec![Vec::<usize>::new(); 2 * limbs - 1];
    for left_limb in 0..limbs {
        for right_limb in 0..limbs {
            products[left_limb + right_limb].push(builder.alloc_limb_product(
                &Lc::from_bits(limb(divisor, left_limb)),
                &Lc::from_bits(limb(quotient, right_limb)),
            ));
        }
    }
    let mut carry = Lc::zero();
    let mut carry_value = 0u64;
    for index in 0..2 * limbs {
        let mut equation = carry.clone();
        let mut total = carry_value;
        for &product in products.get(index).into_iter().flatten() {
            equation = equation.plus(&Lc::var(product));
            total += builder.eval(&Lc::var(product)).as_canonical_u64();
        }
        if index < limbs {
            total += bits_u64(builder, limb(remainder, index))?;
            equation = equation.plus(&Lc::from_bits(limb(remainder, index)));
            equation = equation.minus(&Lc::from_bits(limb(dividend, index)));
        }
        if index + 1 < 2 * limbs {
            carry_value = if active { total >> LIMB_BITS } else { 0 };
            let next = allocate_u64_bits(builder, carry_value, CARRY_BITS);
            builder.enforce_bits_zero_when(&Lc::one().minus(gate), &next);
            equation = equation.add_scaled(&Lc::from_bits(&next), -F::from_u64(1 << LIMB_BITS));
            carry = Lc::from_bits(&next);
        }
        builder.enforce_zero_when(gate, &equation);
    }
    Ok(())
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

/// (operation, borrow in, signed, invert): `<` is `l - r` borrows, `<=` is
/// `l - r - 1` borrows, `>` is not `<=`, and `>=` is not `<`.
fn comparison_operations(wide: bool) -> [(WasmOpcode, bool, bool, bool); 8] {
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

fn ops_lc(operations: &[WasmOpcode]) -> Lc {
    selector_lc(
        &operations
            .iter()
            .map(|&operation| opcode_selector(operation))
            .collect::<Vec<_>>(),
    )
}

fn opcode_lc(operation: WasmOpcode) -> Lc {
    Lc::var(opcode_selector(operation))
}
