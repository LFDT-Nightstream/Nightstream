//! Ring action `ρ · c` in `R_𝔽 = 𝔽[X]/Φ(X)` as an R1CS gadget.
//!
//! For SuperNeo Goldilocks Appendix B.2, `Φ(X) = X^54 + X^27 + 1` (the 81st
//! cyclotomic polynomial). The reduction `X^k mod Φ` is *not* a simple
//! permute-and-add (that would require a power-of-two cyclotomic per
//! Remark 1); we precompute a reduction table and use it in the gadget.
//!
//! ## What this gadget owns
//!
//! - The reduction table `Φ_TABLE[k][m]` = coefficient of `X^m` in
//!   `X^k mod Φ`, for `k ∈ [0, 2D-2]`. Built once at first use.
//! - [`enforce_ring_mul`] — given `ρ` and `c` as length-D arrays of
//!   `Var`s in `R_𝔽`'s coefficient basis, emit `D²` multiplication rows
//!   plus `D` linear-equality rows for the output coefficients.
//!
//! ## Soundness
//!
//! Mechanical. The reduction table is built by running the native
//! `reduce_mod_phi_81` once per basis monomial `X^k`. The gadget's output
//! satisfies `out = Rq(ρ).mul(&Rq(c))` byte-for-byte.

use std::sync::OnceLock;

use neo_math::ring::{D, PHI_MID_DEGREE};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};

const TABLE_LEN: usize = 2 * D - 1;

/// `Φ_TABLE[k][m]` = coefficient of `X^m` in `X^k mod Φ`, for `k ∈ [0, 2D-2]`.
fn phi_reduction_table() -> &'static [[F; D]; TABLE_LEN] {
    static TABLE: OnceLock<[[F; D]; TABLE_LEN]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [[F::ZERO; D]; TABLE_LEN];
        for (k, row) in t.iter_mut().enumerate().take(D) {
            row[k] = F::ONE;
        }
        for k in D..TABLE_LEN {
            let mut tail = [F::ZERO; TABLE_LEN];
            tail[k] = F::ONE;
            reduce_in_place(&mut tail);
            t[k][..D].copy_from_slice(&tail[..D]);
        }
        t
    })
}

/// Coefficient of `X^m` in `X^k mod Φ`, for `k ∈ [0, 2D-2]`.
///
/// This is the public, read-only view of the same reduction table used by
/// [`enforce_ring_mul`]. Phase 1.4 structure builders use it to emit
/// decoded-level output constraints without duplicating the cyclotomic
/// reduction logic.
pub fn phi_reduction_coeff(k: usize, m: usize) -> F {
    assert!(k < TABLE_LEN, "ring reduction degree k={k} out of range 0..{TABLE_LEN}");
    assert!(m < D, "ring reduction output lane m={m} out of range 0..{D}");
    phi_reduction_table()[k][m]
}

/// Mirror of `neo_math::ring::reduce_mod_phi_81`. Kept as a private copy here
/// so this gadget builds without crossing into neo-math's private surface.
fn reduce_in_place(coeffs: &mut [F; TABLE_LEN]) {
    for i in (D..TABLE_LEN).rev() {
        let t = coeffs[i];
        if t == F::ZERO {
            continue;
        }
        coeffs[i] = F::ZERO;
        coeffs[i - D] -= t;
        let idx_27 = i - PHI_MID_DEGREE;
        if idx_27 < D {
            coeffs[idx_27] -= t;
        } else {
            coeffs[idx_27 - D] += t;
            if idx_27 - PHI_MID_DEGREE < D {
                coeffs[idx_27 - PHI_MID_DEGREE] += t;
            }
        }
    }
}

/// The `D²` Karatsuba-style product wires `enforce_ring_mul` allocates.
/// Exposed so parity tests can compare a ring multiplication's internal
/// wires against a `RingActionTraceImage`'s product subregion.
/// `enforce_ring_mul` itself drops these and only returns the output
/// wires.
#[derive(Clone, Debug)]
pub struct RingMulProducts {
    /// `prods[i][j]` = witness column of `ρ[i] · c[j]`.
    pub prods: Vec<Vec<Var>>,
}

/// Allocate `out = ρ · c` in `R_𝔽` and emit constraints. Same shape as
/// [`enforce_ring_mul`] but returns the `D²` product wires as well, for
/// tests and audit tooling that need to bind them to a bit-backed
/// ring-action `RingActionTraceImage` slot.
pub fn enforce_ring_mul_with_products(
    builder: &mut R1csBuilder,
    rho: &[Var; D],
    c: &[Var; D],
) -> ([Var; D], RingMulProducts) {
    let table = phi_reduction_table();

    // Allocate d² products prod[i][j] = ρ[i] · c[j], each via one mult constraint.
    let mut prods: Vec<Vec<Var>> = Vec::with_capacity(D);
    for i in 0..D {
        let mut row = Vec::with_capacity(D);
        for j in 0..D {
            let v = builder.alloc_mul(&Lc::from_var(rho[i]), &Lc::from_var(c[j]));
            row.push(v);
        }
        prods.push(row);
    }

    // For each output coefficient m, constrain out[m] = Σ table[i+j][m] · prods[i][j].
    let mut out: [Option<Var>; D] = [None; D];
    for (m, out_slot) in out.iter_mut().enumerate() {
        let mut combo = Lc::zero();
        for i in 0..D {
            for j in 0..D {
                let coeff = table[i + j][m];
                if coeff != F::ZERO {
                    combo.add_term(prods[i][j], coeff);
                }
            }
        }
        let mut value = F::ZERO;
        for (col, coeff) in &combo.terms {
            value += *coeff * builder.witness()[*col];
        }
        value += combo.constant;
        let v = builder.alloc(value);
        builder.enforce_eq(&Lc::from_var(v), &combo);
        *out_slot = Some(v);
    }

    let output = out.map(|opt| opt.expect("ring_mul out slot must be populated"));
    builder.record_ring_mul(crate::engine::r1cs_circuit::builder::RingMulAuditEntry {
        rho: rho.to_vec(),
        c: c.to_vec(),
        output: output.to_vec(),
        products: prods.clone(),
    });
    (output, RingMulProducts { prods })
}

/// Allocate `out = ρ · c` in `R_𝔽` and emit constraints.
///
/// Inputs are length-`D` arrays of `Var`s in the coefficient basis.
/// Returns the length-`D` output coefficient wires.
///
/// Cost: `D²` mult-constraints (one per coefficient product) + `D` linear
/// equality constraints (one per output coefficient). For audit/test
/// access to the `D²` intermediate product wires, use
/// [`enforce_ring_mul_with_products`].
pub fn enforce_ring_mul(builder: &mut R1csBuilder, rho: &[Var; D], c: &[Var; D]) -> [Var; D] {
    let (output, _) = enforce_ring_mul_with_products(builder, rho, c);
    output
}

/// Convenience: allocate `out_coeffs` from an existing slice of `F` values,
/// then enforce `out = ρ · c` (no equality to a target — produces fresh wires).
///
/// Useful for tests that build the gadget on raw `F` arrays.
pub fn alloc_and_enforce_ring_mul(builder: &mut R1csBuilder, rho_vals: &[F; D], c_vals: &[F; D]) -> [Var; D] {
    let rho = collect_vars(builder, rho_vals);
    let c = collect_vars(builder, c_vals);
    enforce_ring_mul(builder, &rho, &c)
}

fn collect_vars(builder: &mut R1csBuilder, vals: &[F; D]) -> [Var; D] {
    let mut out = [Var::ONE; D];
    for (slot, &v) in out.iter_mut().zip(vals.iter()) {
        *slot = builder.alloc(v);
    }
    out
}
