//! Boolean gadgets: bitness and small-range checks.
//!
//! Mechanical. No paper math.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};

/// Enforce that `v ∈ {0, 1}` via `v · (v - 1) = 0`.
pub fn enforce_bit(builder: &mut R1csBuilder, v: Var) {
    builder.record_boolean(v);
    let v_lc = Lc::from_var(v);
    let mut v_minus_one = Lc::from_var(v);
    v_minus_one.add_constant(-F::ONE);
    builder.enforce(&v_lc, &v_minus_one, &Lc::zero());
}

/// Enforce that `v ∈ {0, 1, …, b - 1}` via `Π_{a=0..b-1} (v - a) = 0`.
///
/// For `b = 2` this is pure bitness (one constraint).
/// For `b > 2` chains multiplications; emits `b - 1` constraints and
/// `b - 2` auxiliary multiplication variables.
pub fn enforce_low_norm(builder: &mut R1csBuilder, v: Var, b: u32) {
    if b == 2 {
        enforce_bit(builder, v);
        return;
    }
    let mut acc = Lc::from_var(v);
    for a in 1..b {
        let mut v_minus_a = Lc::from_var(v);
        v_minus_a.add_constant(-F::from_u64(a as u64));
        if a + 1 == b {
            builder.enforce(&acc, &v_minus_a, &Lc::zero());
        } else {
            let next = builder.alloc_mul(&acc, &v_minus_a);
            acc = Lc::from_var(next);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_constraint_accepts_exact_bits_and_rejects_other_values() {
        for value in [F::ZERO, F::ONE] {
            let mut builder = R1csBuilder::new();
            let v = builder.alloc(value);
            enforce_bit(&mut builder, v);
            assert!(builder.is_satisfied(), "bitness rejected valid bit {value:?}");
        }

        let mut builder = R1csBuilder::new();
        let v = builder.alloc(F::from_u64(2));
        enforce_bit(&mut builder, v);
        assert!(!builder.is_satisfied(), "bitness accepted value 2");
    }

    #[test]
    fn low_norm_accepts_exact_range_and_rejects_outside() {
        for value in [0u64, 1, 2] {
            let mut builder = R1csBuilder::new();
            let v = builder.alloc(F::from_u64(value));
            enforce_low_norm(&mut builder, v, 3);
            assert!(builder.is_satisfied(), "b=3 low-norm check rejected {value}");
        }

        let mut builder = R1csBuilder::new();
        let v = builder.alloc(F::from_u64(3));
        enforce_low_norm(&mut builder, v, 3);
        assert!(!builder.is_satisfied(), "b=3 low-norm check accepted 3");
    }

    #[test]
    fn low_norm_constraint_is_tied_to_the_input_wire() {
        let mut builder = R1csBuilder::new();
        let v = builder.alloc(F::ONE);
        enforce_low_norm(&mut builder, v, 4);
        assert!(builder.is_satisfied(), "baseline should satisfy");

        builder.tamper_witness(v.col(), F::from_u64(4));
        assert!(
            !builder.is_satisfied(),
            "tampering an accepted value outside the range must break the constraint"
        );
    }
}
