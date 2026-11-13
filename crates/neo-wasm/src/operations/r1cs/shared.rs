use std::sync::Arc;

use neo_application::ColumnWidth;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::builder::{bits_u64, eval_bit, CircuitBuilder, ColumnId, Lc};
use super::{opcode_lc, ops_lc, Inputs};
use crate::layout::COL_DIV_TRAP;
use crate::WasmOpcode;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Op {
    Mul,
    And,
    Or,
    Xor,
    Shl,
    ShrU,
    ShrS,
    Rotl,
    Rotr,
    DivU,
    DivS,
    RemU,
    RemS,
    Clz,
    Ctz,
}

const OPS: [(Op, WasmOpcode, WasmOpcode); 15] = {
    use Op::*;
    use WasmOpcode as W;
    [
        (Mul, W::I32Mul, W::I64Mul),
        (And, W::I32And, W::I64And),
        (Or, W::I32Or, W::I64Or),
        (Xor, W::I32Xor, W::I64Xor),
        (Shl, W::I32Shl, W::I64Shl),
        (ShrU, W::I32ShrU, W::I64ShrU),
        (ShrS, W::I32ShrS, W::I64ShrS),
        (Rotl, W::I32Rotl, W::I64Rotl),
        (Rotr, W::I32Rotr, W::I64Rotr),
        (DivU, W::I32DivU, W::I64DivU),
        (DivS, W::I32DivS, W::I64DivS),
        (RemU, W::I32RemU, W::I64RemU),
        (RemS, W::I32RemS, W::I64RemS),
        (Clz, W::I32Clz, W::I64Clz),
        (Ctz, W::I32Ctz, W::I64Ctz),
    ]
};

pub(super) fn opcodes() -> Vec<WasmOpcode> {
    OPS.iter().flat_map(|&(_, a, b)| [a, b]).collect()
}

fn divides(op: Op) -> bool {
    matches!(op, Op::DivU | Op::DivS | Op::RemU | Op::RemS)
}

fn shifts(op: Op) -> bool {
    matches!(op, Op::Shl | Op::ShrU | Op::ShrS | Op::Rotl | Op::Rotr)
}

struct Advice {
    words: [u64; 9],
    abs_carries: [u64; 3],
    delta: u64,
    borrow: u64,
    carries: [u64; 5],
    exponent: u64,
}

struct Assignment {
    left: Vec<ColumnId>,
    right: Vec<ColumnId>,
}

impl Assignment {
    fn advice(&self, z: &[F]) -> Result<Advice, String> {
        use Op::*;
        let mut selected = None;
        for &(op, narrow, wide) in &OPS {
            for (opcode, is64) in [(narrow, false), (wide, true)] {
                if eval_bit(z, &opcode_lc(opcode))? && selected.replace((op, is64)).is_some() {
                    return Err("multiple integer operations selected".into());
                }
            }
        }
        let (mut a, mut b) = (bits_u64(z, &self.left)?, bits_u64(z, &self.right)?);
        if let Some((op, false)) = selected {
            if a > u64::from(u32::MAX) || (!matches!(op, Clz | Ctz) && b > u64::from(u32::MAX)) {
                return Err("i32 integer operand has a nonzero high word".into());
            }
            if matches!(op, ShrS | DivS | RemS) {
                a = a as u32 as i32 as i64 as u64;
            }
            if matches!(op, DivS | RemS) {
                b = b as u32 as i32 as i64 as u64;
            }
            if matches!(op, Rotl | Rotr) {
                a |= a << 32;
            }
        }
        if selected.is_some_and(|(op, _)| divides(op)) && eval_bit(z, &Lc::var(COL_DIV_TRAP))? {
            selected = None;
        }
        let mut advice = Advice {
            words: [0, a, b, 0, 0, 0, 0, 0, 0],
            abs_carries: [0; 3],
            delta: 0,
            borrow: 0,
            carries: [0; 5],
            exponent: 0,
        };
        let Some((op, is64)) = selected else {
            return Ok(advice);
        };
        let signed = matches!(op, DivS | RemS);
        let aa = if signed { (a as i64).unsigned_abs() } else { a };
        let bb = if signed { (b as i64).unsigned_abs() } else { b };
        let shift = (b & if is64 { 63 } else { 31 }) as u32;
        let mut power = 1u64 << shift;
        let result = match op {
            Clz => u64::from(if is64 {
                a.leading_zeros()
            } else {
                (a as u32).leading_zeros()
            }),
            Ctz => u64::from(if is64 {
                a.trailing_zeros()
            } else {
                (a as u32).trailing_zeros()
            }),
            Mul => a.wrapping_mul(b),
            And => a & b,
            Or => a | b,
            Xor => a ^ b,
            Shl => a.wrapping_shl(shift),
            ShrU => a >> shift,
            ShrS => ((a as i64) >> shift) as u64,
            Rotl => a.rotate_left(shift),
            Rotr => a.rotate_right(shift),
            DivU | DivS | RemU | RemS => {
                if bb == 0 {
                    return Err("nontrapping division has a zero divisor".into());
                }
                let (q, r) = (aa / bb, aa % bb);
                if matches!(op, DivU | DivS) {
                    if signed && (a ^ b) >> 63 != 0 {
                        q.wrapping_neg()
                    } else {
                        q
                    }
                } else if signed && a >> 63 != 0 {
                    r.wrapping_neg()
                } else {
                    r
                }
            }
        };
        if matches!(op, Clz | Ctz) {
            power = 1u64 << (result & 63);
        }
        advice.exponent = if shifts(op) {
            u64::from(shift)
        } else if matches!(op, Clz | Ctz) && a != 0 {
            result
        } else {
            0
        };
        let rr = if signed { (result as i64).unsigned_abs() } else { result };
        advice.words[..4].copy_from_slice(&[result, aa, bb, rr]);
        for (index, value) in [a, b, result].into_iter().enumerate() {
            advice.abs_carries[index] = u64::from(signed && value >> 63 != 0 && value as u32 != 0);
        }
        advice.words[4..].copy_from_slice(&match op {
            Clz => {
                if a == 0 {
                    [0; 5]
                } else {
                    [a, power, 0, a * power, 0]
                }
            }
            Ctz => {
                if a == 0 {
                    [0; 5]
                } else {
                    [power, a / power, 0, a, 0]
                }
            }
            Mul => [a, b, 0, result, ((a as u128 * b as u128) >> 64) as u64],
            DivU | DivS | RemU | RemS => [aa / bb, bb, aa % bb, aa, 0],
            Shl => [a, power, 0, result, ((a as u128 * power as u128) >> 64) as u64],
            Rotl => {
                let spill = ((a as u128 * power as u128) >> 64) as u64;
                [a, power, spill, result, spill]
            }
            Rotr => {
                let spill = ((result as u128 * power as u128) >> 64) as u64;
                [result, power, spill, a, spill]
            }
            ShrU => [result, power, a % power, a, 0],
            ShrS => {
                let mask = if a >> 63 != 0 { u64::MAX } else { 0 };
                [result ^ mask, power, (a ^ mask) % power, a ^ mask, 0]
            }
            And | Or | Xor => [0; 5],
        });
        if divides(op) || matches!(op, ShrU | ShrS | Rotl | Rotr) {
            let [_, y, u, _, _] = advice.words[4..].try_into().unwrap();
            advice.delta = y - u - 1;
            advice.borrow = (u64::from(u as u32) + u64::from(advice.delta as u32) + 1) >> 32;
        }
        let mask = (1u64 << 22) - 1;
        let [x, y, u, _, _] = advice.words[4..].try_into().unwrap();
        let mut carry = 0;
        for column in 0..5 {
            let mut total = carry + if column < 3 { (u >> (22 * column)) & mask } else { 0 };
            for i in 0..3 {
                if column >= i && column - i < 3 {
                    total += ((x >> (22 * i)) & mask) * ((y >> (22 * (column - i))) & mask);
                }
            }
            carry = total >> 22;
            advice.carries[column] = carry;
        }
        Ok(advice)
    }
}

type Word = Vec<ColumnId>;
fn words(value: &[ColumnId]) -> [Lc; 2] {
    [Lc::from_bits(&value[..32]), Lc::from_bits(&value[32..])]
}
fn sum(values: impl IntoIterator<Item = Lc>) -> Lc {
    values
        .into_iter()
        .fold(Lc::zero(), |a, b| a.clone().plus(&b))
}

fn bounded_product(builder: &mut CircuitBuilder, a: &Lc, b: &Lc, width: u8) -> Lc {
    let (left, right) = (a.clone(), b.clone());
    let column = builder.allocate("product", ColumnWidth::Bits(width), 1, move |z, out| {
        out[0] = left.evaluate(z) * right.evaluate(z);
        Ok(())
    })[0];
    builder.push_row(a.clone(), b.clone(), Lc::var(column));
    Lc::var(column)
}

fn magnitude(builder: &mut CircuitBuilder, source: &[Lc; 2], sign: &Lc, target: &Word, carry: ColumnId) {
    let base = F::from_u64(1 << 32);
    let output = words(target);
    builder.push_row(
        sign.clone().scaled(F::from_u64(2)),
        source[0].clone(),
        source[0]
            .clone()
            .plus(&Lc::var(carry).scaled(base))
            .minus(&output[0]),
    );
    builder.push_row(
        sign.clone().scaled(F::from_u64(2)),
        source[1].clone(),
        source[1]
            .clone()
            .plus(&sign.clone().scaled(base))
            .minus(&Lc::var(carry))
            .minus(&output[1]),
    );
}

fn bind(builder: &mut CircuitBuilder, gate: &Lc, target: &[ColumnId], source: &[Lc; 2]) {
    for (target, source) in words(target).iter().zip(source) {
        builder.enforce_equal_when(gate, target, source);
    }
}

pub(super) fn enforce(builder: &mut CircuitBuilder, inputs: &Inputs) -> Result<(), String> {
    use Op::*;
    let assignment = Arc::new(Assignment {
        left: inputs.left64.clone(),
        right: inputs.right64.clone(),
    });
    let advice = (0..9)
        .map(|index| {
            let assignment = Arc::clone(&assignment);
            builder.alloc_bits("integer_word", 64, move |z| Ok(assignment.advice(z)?.words[index]))
        })
        .collect::<Vec<_>>();
    let abs_carries = (0..3)
        .map(|index| {
            let assignment = Arc::clone(&assignment);
            builder.alloc_bits("absolute_carry", 1, move |z| {
                Ok(assignment.advice(z)?.abs_carries[index])
            })[0]
        })
        .collect::<Vec<_>>();
    let delta = {
        let assignment = Arc::clone(&assignment);
        builder.alloc_bits("remainder_delta", 64, move |z| Ok(assignment.advice(z)?.delta))
    };
    let borrow = {
        let assignment = Arc::clone(&assignment);
        builder.alloc_bits("remainder_borrow", 1, move |z| Ok(assignment.advice(z)?.borrow))[0]
    };
    let carries = (0..5)
        .map(|index| {
            let assignment = Arc::clone(&assignment);
            builder.alloc_bits("integer_carry", 24, move |z| Ok(assignment.advice(z)?.carries[index]))
        })
        .collect::<Vec<_>>();
    let exponent = {
        let assignment = Arc::clone(&assignment);
        builder.alloc_bits("integer_exponent", 6, move |z| Ok(assignment.advice(z)?.exponent))
    };
    let count32 = ops_lc(&[WasmOpcode::I32Clz, WasmOpcode::I32Ctz]);
    let count64 = ops_lc(&[WasmOpcode::I64Clz, WasmOpcode::I64Ctz]);
    let zero32 = Lc::var(builder.alloc_product(&count32, &Lc::var(advice[0][5])));
    let zero64 = Lc::var(builder.alloc_product(&count64, &Lc::var(advice[0][6])));
    let count_zero = zero32.clone().plus(&zero64);
    builder.enforce_zero_when(&count32, &sum(advice[0][6..].iter().map(|&bit| Lc::var(bit))));
    builder.enforce_zero_when(&count64, &sum(advice[0][7..].iter().map(|&bit| Lc::var(bit))));
    builder.enforce_zero_when(&zero32, &sum(advice[0][..5].iter().map(|&bit| Lc::var(bit))));
    builder.enforce_zero_when(&zero64, &sum(advice[0][..6].iter().map(|&bit| Lc::var(bit))));
    builder.enforce_zero_when(&count_zero, &sum(inputs.left64.iter().map(|&bit| Lc::var(bit))));
    let gates = OPS
        .iter()
        .map(|&(op, a, b)| {
            let gate = ops_lc(&[a, b]);
            if matches!(op, Clz | Ctz) {
                Lc::var(builder.alloc_product(&gate, &builder.one().minus(&count_zero)))
            } else if divides(op) {
                Lc::var(builder.alloc_product(&gate, &builder.one().minus(&Lc::var(COL_DIV_TRAP))))
            } else {
                gate
            }
        })
        .collect::<Vec<_>>();
    let gate = |ops: &[Op]| {
        sum(OPS
            .iter()
            .zip(&gates)
            .filter(|((op, _, _), _)| ops.contains(op))
            .map(|(_, g)| g.clone()))
    };
    let active = sum(gates.iter().cloned()).plus(&count_zero);
    let arithmetic = gate(&[Mul, Shl, ShrU, ShrS, Rotl, Rotr, DivU, DivS, RemU, RemS, Clz, Ctz]);
    let signed = gate(&[DivS, RemS]);
    let is32 = ops_lc(&OPS.iter().map(|entry| entry.1).collect::<Vec<_>>());
    let is64 = ops_lc(&OPS.iter().map(|entry| entry.2).collect::<Vec<_>>());
    let raw_a = words(&inputs.left64);
    let raw_b = words(&inputs.right64);
    builder.enforce_zero_when(&is32, &sum(inputs.left64[32..].iter().map(|&bit| Lc::var(bit))));
    let binary32 = ops_lc(
        &OPS.iter()
            .filter(|(op, _, _)| !matches!(op, Clz | Ctz))
            .map(|entry| entry.1)
            .collect::<Vec<_>>(),
    );
    builder.enforce_zero_when(&binary32, &sum(inputs.right64[32..].iter().map(|&bit| Lc::var(bit))));
    let sign32_a = Lc::var(builder.alloc_product(
        &ops_lc(&[WasmOpcode::I32ShrS, WasmOpcode::I32DivS, WasmOpcode::I32RemS]),
        &Lc::var(inputs.left32[31]),
    ));
    let sign32_b = Lc::var(builder.alloc_product(
        &ops_lc(&[WasmOpcode::I32DivS, WasmOpcode::I32RemS]),
        &Lc::var(inputs.right32[31]),
    ));
    let repeat = bounded_product(
        builder,
        &ops_lc(&[WasmOpcode::I32Rotl, WasmOpcode::I32Rotr]),
        &raw_a[0],
        32,
    );
    let a = [
        raw_a[0].clone(),
        raw_a[1]
            .clone()
            .plus(&sign32_a.clone().scaled(F::from_u64(u32::MAX.into())))
            .plus(&repeat),
    ];
    let b = [
        raw_b[0].clone(),
        raw_b[1]
            .clone()
            .plus(&sign32_b.clone().scaled(F::from_u64(u32::MAX.into()))),
    ];
    let sign_a = Lc::var(inputs.left64[63]).plus(&sign32_a);
    let sign_b = Lc::var(inputs.right64[63]).plus(&sign32_b);
    let abs_a = Lc::var(builder.alloc_product(&signed, &sign_a));
    let abs_b = Lc::var(builder.alloc_product(&signed, &sign_b));
    let abs_r = Lc::var(builder.alloc_product(&signed, &Lc::var(advice[0][63])));
    magnitude(builder, &a, &abs_a, &advice[1], abs_carries[0]);
    magnitude(builder, &b, &abs_b, &advice[2], abs_carries[1]);
    magnitude(builder, &words(&advice[0]), &abs_r, &advice[3], abs_carries[2]);
    let both = Lc::var(builder.alloc_product(&sign_a, &sign_b));
    let opposite = sign_a
        .clone()
        .plus(&sign_b)
        .minus(&both.clone().scaled(F::from_u64(2)));
    let quotient_sign = Lc::var(builder.alloc_product(&gate(&[DivS]), &opposite));
    let remainder_sign = Lc::var(builder.alloc_product(&gate(&[RemS]), &sign_a));
    let mismatch = quotient_sign.clone().plus(&remainder_sign).minus(&abs_r);
    for word in words(&advice[3]) {
        builder.enforce_zero_when(&mismatch, &word);
    }
    let shift_sign = Lc::var(builder.alloc_product(&gate(&[ShrS]), &sign_a));
    let complement = |builder: &mut CircuitBuilder, source: [Lc; 2]| -> [Lc; 2] {
        source.map(|word| {
            let product = bounded_product(builder, &shift_sign, &word, 32);
            word.clone()
                .minus(&product.clone().scaled(F::from_u64(2)))
                .plus(&shift_sign.clone().scaled(F::from_u64(u32::MAX.into())))
        })
    };
    let a_complement = complement(builder, a.clone());
    let r_complement = complement(builder, words(&advice[0]));
    let shift32 = ops_lc(
        &OPS.iter()
            .filter(|entry| shifts(entry.0))
            .map(|entry| entry.1)
            .collect::<Vec<_>>(),
    );
    let shift_upper = Lc::var(builder.alloc_product(&Lc::var(inputs.right64[5]), &builder.one().minus(&shift32)));
    let shift_amount = Lc::from_bits(&inputs.right32[..5]).plus(&shift_upper.scaled(F::from_u64(32)));
    let shift_gate = gate(&[Shl, ShrU, ShrS, Rotl, Rotr]);
    let count_gate = gate(&[Clz, Ctz]);
    builder.enforce_equal_when(&shift_gate, &Lc::from_bits(&exponent), &shift_amount);
    builder.enforce_equal_when(&count_gate, &Lc::from_bits(&exponent), &Lc::from_bits(&advice[0][..6]));
    builder.enforce_zero_when(
        &builder.one().minus(&shift_gate).minus(&count_gate),
        &Lc::from_bits(&exponent),
    );
    let mut power = builder.one().plus(&Lc::var(exponent[0]));
    for bit in 1..5 {
        let factor = builder
            .one()
            .plus(&Lc::var(exponent[bit]).scaled(F::from_u64((1 << (1 << bit)) - 1)));
        power = bounded_product(builder, &power, &factor, 1 << (bit + 1));
    }
    let high_power = bounded_product(builder, &Lc::var(exponent[5]), &power, 32);
    let power_words = [power.minus(&high_power), high_power];
    let zero = [Lc::zero(), Lc::zero()];
    let [x, y, u, v, w] = [&advice[4], &advice[5], &advice[6], &advice[7], &advice[8]];
    bind(builder, &gate(&[Mul, Shl, Rotl, Clz]), x, &words(&advice[1]));
    bind(builder, &gate(&[Ctz]), x, &power_words);
    bind(builder, &gate(&[DivU, DivS]), x, &words(&advice[3]));
    bind(builder, &gate(&[ShrU, ShrS, Rotr]), x, &r_complement);
    bind(builder, &gate(&[Mul, DivU, DivS, RemU, RemS]), y, &words(&advice[2]));
    bind(builder, &gate(&[Shl, ShrU, ShrS, Rotl, Rotr, Clz]), y, &power_words);
    bind(builder, &gate(&[Mul, Shl, Clz, Ctz]), u, &zero);
    bind(builder, &gate(&[RemU, RemS]), u, &words(&advice[3]));
    bind(builder, &gate(&[Rotl, Rotr]), u, &words(w));
    bind(builder, &gate(&[Mul, Shl, Rotl]), v, &words(&advice[0]));
    bind(builder, &gate(&[DivU, DivS, RemU, RemS, Ctz]), v, &words(&advice[1]));
    bind(builder, &gate(&[ShrU, ShrS, Rotr]), v, &a_complement);
    bind(
        builder,
        &gate(&[DivU, DivS, RemU, RemS, ShrU, ShrS, Clz, Ctz]),
        w,
        &zero,
    );
    let clz32 = Lc::var(builder.alloc_product(&opcode_lc(WasmOpcode::I32Clz), &builder.one().minus(&count_zero)));
    let clz64 = Lc::var(builder.alloc_product(&opcode_lc(WasmOpcode::I64Clz), &builder.one().minus(&count_zero)));
    builder.enforce_equal_when(&clz32, &Lc::var(v[31]), &builder.one());
    builder.enforce_zero_when(&clz32, &Lc::from_bits(&v[32..]));
    builder.enforce_equal_when(&clz64, &Lc::var(v[63]), &builder.one());
    builder.enforce_equal_when(&gate(&[Ctz]), &Lc::var(y[0]), &builder.one());
    let unused = builder.one().minus(&arithmetic);
    builder.enforce_zero_when(&unused, &sum(advice[4..].iter().flatten().map(|&bit| Lc::var(bit))));
    let bounded = gate(&[DivU, DivS, RemU, RemS, ShrU, ShrS, Rotl, Rotr]);
    let [ul, uh] = words(u);
    let [yl, yh] = words(y);
    builder.enforce_zero_when(
        &bounded,
        &ul.clone()
            .plus(&Lc::from_bits(&delta[..32]))
            .plus(&builder.one())
            .minus(&yl)
            .minus(&Lc::var(borrow).scaled(F::from_u64(1 << 32))),
    );
    builder.enforce_zero_when(
        &bounded,
        &uh.clone()
            .plus(&Lc::from_bits(&delta[32..]))
            .plus(&Lc::var(borrow))
            .minus(&yh),
    );
    builder.enforce_zero_when(
        &builder.one().minus(&bounded),
        &sum(delta
            .iter()
            .chain(std::iter::once(&borrow))
            .map(|&bit| Lc::var(bit))),
    );
    let limbs = |bits: &[ColumnId]| bits.chunks(22).map(Lc::from_bits).collect::<Vec<_>>();
    let (xl, yl, ul) = (limbs(x), limbs(y), limbs(u));
    let output = v.iter().chain(w.iter()).copied().collect::<Vec<_>>();
    let output = limbs(&output);
    let mut columns = vec![Lc::zero(); 6];
    for i in 0..3 {
        for j in 0..3 {
            let product = bounded_product(builder, &xl[i], &yl[j], 44);
            columns[i + j] = columns[i + j].clone().plus(&product);
        }
    }
    let mut carry = Lc::zero();
    for index in 0..6 {
        let next = if index < 5 {
            Lc::from_bits(&carries[index])
        } else {
            Lc::zero()
        };
        let addend = ul.get(index).cloned().unwrap_or_else(Lc::zero);
        builder.enforce_linear_zero(
            columns[index]
                .clone()
                .plus(&addend)
                .plus(&carry)
                .minus(&output[index])
                .minus(&next.clone().scaled(F::from_u64(1 << 22))),
        );
        carry = next;
    }
    let bit_products = inputs
        .left64
        .iter()
        .zip(&inputs.right64)
        .map(|(&a, &b)| Lc::var(builder.alloc_product(&Lc::var(a), &Lc::var(b))))
        .collect::<Vec<_>>();
    for op in [And, Or, Xor] {
        for half in 0..2 {
            let start = half * 32;
            let expected = sum((0..32).map(|bit| {
                let index = start + bit;
                let product = &bit_products[index];
                let value = if op == And {
                    product.clone()
                } else {
                    Lc::var(inputs.left64[index])
                        .plus(&Lc::var(inputs.right64[index]))
                        .minus(
                            &product
                                .clone()
                                .scaled(F::from_u64(if op == Xor { 2 } else { 1 })),
                        )
                };
                value.clone().scaled(F::from_u64(1 << bit))
            }));
            builder.enforce_equal_when(&gate(&[op]), &words(&advice[0])[half], &expected);
        }
    }
    builder.enforce_equal_when(&active, &Lc::from_bits(&inputs.output32), &words(&advice[0])[0]);
    builder.enforce_equal_when(&is64, &Lc::from_bits(&inputs.output64[32..]), &words(&advice[0])[1]);
    builder.enforce_zero_when(&is32, &Lc::from_bits(&inputs.output64[32..]));
    builder.enforce_zero_when(
        &opcode_lc(WasmOpcode::I32DivS),
        &Lc::var(advice[0][31]).minus(&Lc::var(advice[0][63])),
    );
    builder.enforce_zero_when(
        &builder.one().minus(&active),
        &sum(advice[0].iter().map(|&bit| Lc::var(bit))),
    );
    Ok(())
}
