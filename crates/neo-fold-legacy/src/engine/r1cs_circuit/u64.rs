//! u64 arithmetic gadgets — increment, add, equality — over bit-decomposed values.
//!
//! Used by F' R1CS to enforce counter transitions:
//!   - `chunk_count_out = chunk_count_in + 1`
//!   - `step_count_out  = step_count_in + K`
//!   - `nifs_metadata_field == declared_constant`
//!
//! Each u64 is represented as 64 base-field bits (laid out little-endian).
//! The gadgets enforce range (bitness) and the arithmetic relation via carry
//! witnesses.
//!
//! ## Cost
//!
//! - `enforce_u64_bitness`: 64 mult constraints (1 per bit).
//! - `enforce_u64_increment`: 64 linear constraints + 63 carry-bit allocations.
//! - `enforce_u64_add`: same shape.
//! - `enforce_u64_equality`: 64 linear constraints.
//! - `enforce_u64_constant`: 64 linear constraints.
//!
//! No K-mults; pure base-field arithmetic.

use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};

const U64_BITS: usize = 64;

/// Allocate a u64 value as 64 base-field bits (little-endian) and enforce
/// bitness on each bit. Returns the bit Var array.
pub fn alloc_u64_bits(builder: &mut R1csBuilder, value: u64) -> [Var; U64_BITS] {
    let mut bits = [Var::ONE; U64_BITS];
    for (i, slot) in bits.iter_mut().enumerate() {
        let bit_val = (value >> i) & 1;
        *slot = builder.alloc(F::from_u64(bit_val));
        enforce_bit(builder, *slot);
    }
    bits
}

/// Decompose an existing F-valued `Var` into 64 canonical Goldilocks bits.
///
/// Enforces: (1) each output is a bit, (2) `var == Σ 2^i · bits[i]` (mod p),
/// and (3) the decomposition is **canonical** — that is, the integer
/// `Σ 2^i · bits[i]` is `< p`. The canonicity check rules out the
/// malleability where `value + p < 2^64` allows a second valid 64-bit
/// representation in F-arithmetic.
///
/// `p = 2^64 − 2^32 + 1`, so the only non-canonical 64-bit value of an
/// F-element `x` is `x + p` for `x < 2^32 − 1`. The check is:
///
/// ```text
///   NOT (hi == 0xFFFF_FFFF AND lo > 0)
/// ```
///
/// where `hi`, `lo` are the upper and lower 32-bit halves.
pub fn decompose_var_to_u64_bits(builder: &mut R1csBuilder, var: Var) -> [Var; U64_BITS] {
    if let Some(bits) = builder.canonical_u64_decomposition(var) {
        return bits;
    }

    let row_start = builder.rows();
    let (bits, high_is_max, inverse) = decompose_var_to_u64_bits_inner(builder, var);
    let source_rows = row_start..builder.rows();
    debug_assert_eq!(source_rows.len(), 69, "canonical-u64 source schedule changed");
    builder.record_canonical_u64_decomposition(var, bits, high_is_max, inverse, source_rows);
    bits
}

fn decompose_var_to_u64_bits_inner(builder: &mut R1csBuilder, var: Var) -> ([Var; U64_BITS], Var, Var) {
    // 1. Take canonical u64 from the witness value.
    let raw: F = builder.witness()[var.col()];
    use p3_field::PrimeField64;
    let canonical_u64 = raw.as_canonical_u64();

    // 2. Allocate 64 bits matching the canonical representation.
    let bits = alloc_u64_bits(builder, canonical_u64);

    // 3. Enforce `var == Σ 2^i · bits[i]`.
    let mut sum_lc = Lc::zero();
    let mut pow2 = F::ONE;
    for &b in &bits {
        sum_lc.add_term(b, pow2);
        pow2 = pow2 + pow2;
    }
    builder.enforce_eq(&Lc::from_var(var), &sum_lc);

    // 4. Canonicity check: NOT (hi == 0xFFFF_FFFF AND lo > 0).
    //
    // Encode hi_is_max = (hi == 0xFFFF_FFFF) and force `hi_is_max * lo == 0`.
    let goldilocks_high_max = F::from_u64(0xFFFF_FFFF);
    let mut hi_lc = Lc::zero();
    let mut pow2_hi = F::ONE;
    for &b in &bits[32..] {
        hi_lc.add_term(b, pow2_hi);
        pow2_hi = pow2_hi + pow2_hi;
    }
    let mut lo_lc = Lc::zero();
    let mut pow2_lo = F::ONE;
    for &b in &bits[..32] {
        lo_lc.add_term(b, pow2_lo);
        pow2_lo = pow2_lo + pow2_lo;
    }
    let hi_val = builder.eval(&hi_lc);

    // hi_is_max ∈ {0, 1}, with hi_is_max = 1 ⟺ hi == 0xFFFF_FFFF.
    let hi_is_max_val = if hi_val == goldilocks_high_max { F::ONE } else { F::ZERO };
    let hi_is_max = builder.alloc(hi_is_max_val);
    enforce_bit(builder, hi_is_max);

    // Inverse witness for `hi != 0xFFFF_FFFF`. When equal, the inverse is
    // unused; we provide ZERO as a placeholder.
    let diff_val = hi_val - goldilocks_high_max;
    let inv_val = if diff_val == F::ZERO {
        F::ZERO
    } else {
        diff_val.inverse()
    };
    let inv = builder.alloc(inv_val);

    // Constraint A: `hi_is_max * (hi - 0xFFFF_FFFF) = 0`.
    let diff_lc = hi_lc
        .clone()
        .add_scaled(&Lc::from_const(goldilocks_high_max), -F::ONE);
    builder.enforce(&Lc::from_var(hi_is_max), &diff_lc, &Lc::zero());

    // Constraint B: `(hi - 0xFFFF_FFFF) * inv == 1 - hi_is_max`.
    let mut one_minus_lc = Lc::from_const(F::ONE);
    one_minus_lc.add_term(hi_is_max, -F::ONE);
    builder.enforce(&diff_lc, &Lc::from_var(inv), &one_minus_lc);

    // Constraint C: `hi_is_max * lo == 0` — the canonicity gate.
    builder.enforce(&Lc::from_var(hi_is_max), &lo_lc, &Lc::zero());

    (bits, hi_is_max, inv)
}

/// Enforce that a previously-allocated bit array represents valid bits.
/// Used when bits come from elsewhere in the witness.
pub fn enforce_u64_bitness(builder: &mut R1csBuilder, bits: &[Var; U64_BITS]) {
    for &b in bits {
        enforce_bit(builder, b);
    }
}

/// Enforce `out_bits == in_bits + 1` (mod 2^64) using carry witnesses.
///
/// The addition is bit-by-bit with carries. Each bit:
///   `out[i] + 2·carry[i] == in[i] + carry[i-1]` (carry[-1] = 1 for increment by 1).
///
/// Allocates 63 carry bits (carry out of bits 0..62; carry out of bit 63 is
/// dropped since we're mod 2^64).
pub fn enforce_u64_increment(builder: &mut R1csBuilder, in_bits: &[Var; U64_BITS], out_bits: &[Var; U64_BITS]) {
    let mut carry_in = Lc::from_var(Var::ONE); // +1
    for i in 0..U64_BITS {
        let sum_value = builder.eval(&Lc::from_var(in_bits[i])) + builder.eval(&carry_in);

        // Allocate carry_out (except for the last bit where we drop it).
        let carry_out = if i + 1 < U64_BITS {
            let v = (sum_value_to_int(sum_value) >> 1) & 1;
            let cv = builder.alloc(F::from_u64(v));
            enforce_bit(builder, cv);
            Some(cv)
        } else {
            None
        };

        // Constraint: in[i] + carry_in == out[i] + 2 · carry_out
        let lhs = Lc::from_var(in_bits[i]).add_scaled(&carry_in, F::ONE);
        let mut rhs = Lc::from_var(out_bits[i]);
        if let Some(c) = carry_out {
            rhs.add_term(c, F::from_u64(2));
        }
        builder.enforce_eq(&lhs, &rhs);

        carry_in = match carry_out {
            Some(c) => Lc::from_var(c),
            None => Lc::zero(),
        };
    }
}

/// Enforce `out_bits == lhs_bits + rhs_bits` (mod 2^64).
///
/// Bit-by-bit add with carries. Allocates 63 carry bits.
pub fn enforce_u64_add(
    builder: &mut R1csBuilder,
    lhs_bits: &[Var; U64_BITS],
    rhs_bits: &[Var; U64_BITS],
    out_bits: &[Var; U64_BITS],
) {
    let mut carry_in = Lc::zero();
    for i in 0..U64_BITS {
        let sum_value = builder.eval(&Lc::from_var(lhs_bits[i]))
            + builder.eval(&Lc::from_var(rhs_bits[i]))
            + builder.eval(&carry_in);

        let carry_out = if i + 1 < U64_BITS {
            let v = (sum_value_to_int(sum_value) >> 1) & 1;
            let cv = builder.alloc(F::from_u64(v));
            enforce_bit(builder, cv);
            Some(cv)
        } else {
            None
        };

        let lhs_lc = Lc::from_var(lhs_bits[i])
            .add_scaled(&Lc::from_var(rhs_bits[i]), F::ONE)
            .add_scaled(&carry_in, F::ONE);
        let mut rhs_lc = Lc::from_var(out_bits[i]);
        if let Some(c) = carry_out {
            rhs_lc.add_term(c, F::from_u64(2));
        }
        builder.enforce_eq(&lhs_lc, &rhs_lc);

        carry_in = match carry_out {
            Some(c) => Lc::from_var(c),
            None => Lc::zero(),
        };
    }
}

/// Enforce `lhs_bits == rhs_bits` (bitwise).
pub fn enforce_u64_equality(builder: &mut R1csBuilder, lhs: &[Var; U64_BITS], rhs: &[Var; U64_BITS]) {
    for (l, r) in lhs.iter().zip(rhs.iter()) {
        builder.enforce_eq(&Lc::from_var(*l), &Lc::from_var(*r));
    }
}

/// Enforce `bits == expected` (bitwise, against a known u64 constant).
pub fn enforce_u64_constant(builder: &mut R1csBuilder, bits: &[Var; U64_BITS], expected: u64) {
    for (i, &b) in bits.iter().enumerate() {
        let expected_bit = (expected >> i) & 1;
        builder.enforce_eq(&Lc::from_var(b), &Lc::from_const(F::from_u64(expected_bit)));
    }
}

// ── helpers ──────────────────────────────────────────────────────────────

/// Convert a small-magnitude F-value to a u64 for carry computation.
/// Only valid when the value is known to fit in a u64.
fn sum_value_to_int(v: F) -> u64 {
    use p3_field::PrimeField64;
    v.as_canonical_u64()
}
