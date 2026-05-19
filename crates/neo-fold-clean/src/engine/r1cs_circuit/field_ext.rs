//! Extension-field 𝕂 = 𝔽[X]/(X² − W) arithmetic as R1CS gadgets.
//!
//! For `neo-math`'s `K = BinomialExtensionField<Goldilocks, 2>`, `W = 7`
//! (see `p3-goldilocks::extension`). One 𝕂-element is represented in the
//! witness as two consecutive base-field columns `(c0, c1)`, where the
//! algebraic value is `c0 + c1 · X` with `X² = W`.
//!
//! ## Operations
//!
//! - `KVar` — pair of `Var`s representing one 𝕂-element.
//! - `KLc` — pair of linear combinations over 𝕂.
//! - `klc_add`, `klc_add_scaled` — linear, emit no constraints.
//! - `enforce_k_mul` — Karatsuba-form `out = a · b` in 𝕂, emitting
//!   3 mult-constraints + 2 linear-equalities.
//!
//! ## Soundness
//!
//! Mechanical. No paper claims. The W constant is read at runtime from
//! [`<p3_goldilocks::Goldilocks as BinomiallyExtendable<2>>::W`] to keep this
//! gadget byte-identical with native 𝕂 arithmetic.

use neo_math::{Fq, F};
use p3_field::extension::BinomiallyExtendable;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};

/// `K = F[X]/(X² − W)`. For Goldilocks-quadratic, `W = 7`.
fn w_constant() -> F {
    <Fq as BinomiallyExtendable<2>>::W
}

/// One 𝕂-element: low limb (constant term) + high limb (coefficient of X).
#[derive(Clone, Copy, Debug)]
pub struct KVar {
    pub c0: Var,
    pub c1: Var,
}

impl KVar {
    pub fn new(c0: Var, c1: Var) -> Self {
        Self { c0, c1 }
    }

    pub fn alloc(builder: &mut R1csBuilder, value_c0: F, value_c1: F) -> Self {
        Self {
            c0: builder.alloc(value_c0),
            c1: builder.alloc(value_c1),
        }
    }
}

/// Two linear combinations representing one 𝕂-element. No allocation.
#[derive(Clone, Debug)]
pub struct KLc {
    pub c0: Lc,
    pub c1: Lc,
}

impl KLc {
    pub fn zero() -> Self {
        Self {
            c0: Lc::zero(),
            c1: Lc::zero(),
        }
    }

    pub fn from_var(v: KVar) -> Self {
        Self {
            c0: Lc::from_var(v.c0),
            c1: Lc::from_var(v.c1),
        }
    }

    pub fn from_base_const(c: F) -> Self {
        Self {
            c0: Lc::from_const(c),
            c1: Lc::zero(),
        }
    }
}

/// `out = a + s · b`. Linear; no constraints emitted.
pub fn klc_add_scaled(a: &KLc, b: &KLc, s: F) -> KLc {
    KLc {
        c0: a.c0.clone().add_scaled(&b.c0, s),
        c1: a.c1.clone().add_scaled(&b.c1, s),
    }
}

pub fn klc_add(a: &KLc, b: &KLc) -> KLc {
    klc_add_scaled(a, b, F::ONE)
}

/// Allocate a fresh `KVar` constrained to equal `lc`. Linear; no K-mult.
pub fn alloc_klc(builder: &mut R1csBuilder, lc: &KLc) -> KVar {
    let c0 = builder.eval(&lc.c0);
    let c1 = builder.eval(&lc.c1);
    let v = KVar::alloc(builder, c0, c1);
    builder.enforce_eq(&Lc::from_var(v.c0), &lc.c0);
    builder.enforce_eq(&Lc::from_var(v.c1), &lc.c1);
    v
}

/// The three Karatsuba intermediates `enforce_k_mul` allocates. Exposed
/// so parity tests can compare a K-mul's internal wires against the
/// `KMulView` slot the F' source image reserves. `enforce_k_mul`
/// itself drops these and only returns the output `KVar`.
#[derive(Clone, Copy, Debug)]
pub struct KMulIntermediates {
    /// `p = a.c0 · b.c0`.
    pub p: Var,
    /// `q = a.c1 · b.c1`.
    pub q: Var,
    /// `r = (a.c0 + a.c1) · (b.c0 + b.c1)`.
    pub r: Var,
}

/// Allocate `out = a · b` in 𝕂 and emit the constraints. Same shape as
/// [`enforce_k_mul`] but returns the three Karatsuba intermediates as
/// well, for tests and audit tooling that need to bind them to a
/// bit-backed `KMulView` slot.
pub fn enforce_k_mul_with_intermediates(builder: &mut R1csBuilder, a: &KLc, b: &KLc) -> (KVar, KMulIntermediates) {
    let p_var = builder.alloc_mul(&a.c0, &b.c0);
    let q_var = builder.alloc_mul(&a.c1, &b.c1);
    let sum_a = a.c0.clone().add_scaled(&a.c1, F::ONE);
    let sum_b = b.c0.clone().add_scaled(&b.c1, F::ONE);
    let r_var = builder.alloc_mul(&sum_a, &sum_b);

    let w = w_constant();
    let p = builder.witness()[p_var.col()];
    let q = builder.witness()[q_var.col()];
    let r = builder.witness()[r_var.col()];
    let out_c0_value = p + w * q;
    let out_c1_value = r - p - q;
    let out_c0 = builder.alloc(out_c0_value);
    let out_c1 = builder.alloc(out_c1_value);

    // out_c0 = p + W · q
    let lhs = Lc::from_var(out_c0);
    let rhs = Lc::from_var(p_var).add_scaled(&Lc::from_var(q_var), w);
    builder.enforce_eq(&lhs, &rhs);

    // out_c1 = r - p - q
    let lhs = Lc::from_var(out_c1);
    let rhs = Lc::from_var(r_var)
        .add_scaled(&Lc::from_var(p_var), -F::ONE)
        .add_scaled(&Lc::from_var(q_var), -F::ONE);
    builder.enforce_eq(&lhs, &rhs);

    builder.record_k_mul(p_var, q_var, r_var);
    (
        KVar::new(out_c0, out_c1),
        KMulIntermediates {
            p: p_var,
            q: q_var,
            r: r_var,
        },
    )
}

/// Allocate `out = a · b` in 𝕂 and emit the constraints.
///
/// `(a0 + a1·X) · (b0 + b1·X) = (a0·b0 + W·a1·b1) + (a0·b1 + a1·b0) · X`.
///
/// Uses the Karatsuba-like 3-multiplication trick:
///
/// ```text
///     p = a0 · b0
///     q = a1 · b1
///     r = (a0 + a1) · (b0 + b1)
///     out_c0 = p + W · q
///     out_c1 = r - p - q
/// ```
///
/// 3 multiplication constraints + 2 linear equalities. For audit/test
/// access to the `p, q, r` intermediates, use
/// [`enforce_k_mul_with_intermediates`].
pub fn enforce_k_mul(builder: &mut R1csBuilder, a: &KLc, b: &KLc) -> KVar {
    let (out, _) = enforce_k_mul_with_intermediates(builder, a, b);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_math::{KExtensions, K};

    fn alloc_k(builder: &mut R1csBuilder, value: K) -> KVar {
        let [c0, c1] = value.as_coeffs();
        KVar::alloc(builder, c0, c1)
    }

    fn read_k(builder: &R1csBuilder, value: KVar) -> K {
        K::from_coeffs([builder.witness()[value.c0.col()], builder.witness()[value.c1.col()]])
    }

    #[test]
    fn k_mul_matches_native_extension_arithmetic() {
        let cases = [
            (K::ZERO, K::ONE),
            (K::ONE, K::ONE),
            (
                K::from_coeffs([F::from_u64(3), F::from_u64(5)]),
                K::from_coeffs([F::from_u64(7), F::from_u64(11)]),
            ),
            (
                K::from_coeffs([-F::ONE, F::from_u64(2)]),
                K::from_coeffs([F::from_u64(4), -F::ONE]),
            ),
        ];

        for (a, b) in cases {
            let mut builder = R1csBuilder::new();
            let a_var = alloc_k(&mut builder, a);
            let b_var = alloc_k(&mut builder, b);
            let out = enforce_k_mul(&mut builder, &KLc::from_var(a_var), &KLc::from_var(b_var));

            assert_eq!(read_k(&builder, out), a * b, "native K multiplication mismatch");
            assert!(
                builder.is_satisfied(),
                "K multiplication gadget unsatisfied (first bad row: {:?})",
                builder.first_unsatisfied_row()
            );
        }
    }

    #[test]
    fn k_mul_rejects_tampered_output_limb() {
        let mut builder = R1csBuilder::new();
        let a = alloc_k(&mut builder, K::from_coeffs([F::from_u64(7), F::from_u64(11)]));
        let b = alloc_k(&mut builder, K::from_coeffs([F::from_u64(13), F::from_u64(17)]));
        let out = enforce_k_mul(&mut builder, &KLc::from_var(a), &KLc::from_var(b));
        assert!(builder.is_satisfied(), "baseline should satisfy");

        let target = out.c0.col();
        builder.tamper_witness(target, builder.witness()[target] + F::ONE);
        assert!(
            !builder.is_satisfied(),
            "tampered K multiplication output must violate constraints"
        );
    }

    #[test]
    fn alloc_klc_binds_linear_combinations_to_witness_values() {
        let mut builder = R1csBuilder::new();
        let x = builder.alloc(F::from_u64(10));
        let y = builder.alloc(F::from_u64(3));
        let lc = KLc {
            c0: Lc::from_var(x).add_scaled(&Lc::from_var(y), F::from_u64(2)),
            c1: Lc::from_var(x).add_scaled(&Lc::from_var(y), -F::ONE),
        };

        let out = alloc_klc(&mut builder, &lc);
        assert_eq!(builder.witness()[out.c0.col()], F::from_u64(16));
        assert_eq!(builder.witness()[out.c1.col()], F::from_u64(7));
        assert!(builder.is_satisfied(), "linear K allocation should satisfy");

        builder.tamper_witness(out.c1.col(), F::from_u64(8));
        assert!(
            !builder.is_satisfied(),
            "tampered allocated linear-combination output must fail"
        );
    }
}
