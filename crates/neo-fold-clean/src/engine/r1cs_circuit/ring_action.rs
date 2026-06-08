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
//! - [`enforce_ring_mul`] — legacy full-product shape: given `ρ` and `c`
//!   as length-D arrays of `Var`s in `R_𝔽`'s coefficient basis, emit
//!   `D²` multiplication rows plus `D` linear-equality rows for the output
//!   coefficients, and record all `D²` products for strict trace audits.
//! - [`enforce_ring_mul_toom3`] — production verifier shape: mirrors
//!   `Rq::mul`'s 3-way multiplication and emits `5·18² + D` rows unless
//!   the audit trail is enabled.
//!
//! ## Soundness
//!
//! Mechanical. The reduction table is built by running the native
//! `reduce_mod_phi_81` once per basis monomial `X^k`. The gadget's output
//! satisfies `out = Rq(ρ).mul(&Rq(c))` byte-for-byte.

use std::sync::OnceLock;

use neo_math::ring::{D, PHI_MID_DEGREE};
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};

const TABLE_LEN: usize = 2 * D - 1;
const TOOM3_SPLIT: usize = D / 3;
const TOOM3_CHUNK_OUT: usize = 2 * TOOM3_SPLIT - 1;

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

/// Optimized `out = ρ · c` gadget for production verifier paths.
///
/// This mirrors `neo_math::ring::Rq::mul`'s 3-way Toom/Karatsuba
/// multiplication: split each length-54 polynomial into three length-18
/// chunks, evaluate at `0, 1, -1, 2, ∞`, schoolbook-multiply the five
/// length-18 pairs, interpolate, then reduce modulo `Φ₈₁`.
///
/// Cost without the audit trail:
/// - `5 * 18² = 1620` multiplication rows.
/// - `54` output equality rows.
///
/// The legacy [`enforce_ring_mul`] uses `54² + 54 = 2970` rows and records
/// every pairwise product. When the builder audit trail is enabled we
/// deliberately fall back to that legacy shape so the strict source-image
/// product-parity tests still observe the full `D²` product matrix.
pub fn enforce_ring_mul_toom3(builder: &mut R1csBuilder, rho: &[Var; D], c: &[Var; D]) -> [Var; D] {
    if builder.audit_trail_enabled() {
        return enforce_ring_mul(builder, rho, c);
    }
    enforce_ring_mul_toom3_inner(builder, rho, c)
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

/// Convenience variant for the optimized production gadget.
pub fn alloc_and_enforce_ring_mul_toom3(builder: &mut R1csBuilder, rho_vals: &[F; D], c_vals: &[F; D]) -> [Var; D] {
    let rho = collect_vars(builder, rho_vals);
    let c = collect_vars(builder, c_vals);
    enforce_ring_mul_toom3(builder, &rho, &c)
}

fn collect_vars(builder: &mut R1csBuilder, vals: &[F; D]) -> [Var; D] {
    let mut out = [Var::ONE; D];
    for (slot, &v) in out.iter_mut().zip(vals.iter()) {
        *slot = builder.alloc(v);
    }
    out
}

fn enforce_ring_mul_toom3_inner(builder: &mut R1csBuilder, rho: &[Var; D], c: &[Var; D]) -> [Var; D] {
    let a0 = chunk_at(rho, 0);
    let a1 = chunk_at(rho, TOOM3_SPLIT);
    let a2 = chunk_at(rho, 2 * TOOM3_SPLIT);
    let b0 = chunk_at(c, 0);
    let b1 = chunk_at(c, TOOM3_SPLIT);
    let b2 = chunk_at(c, 2 * TOOM3_SPLIT);

    let two = F::from_u64(2);
    let four = F::from_u64(4);

    let p0 = schoolbook_product_lcs(builder, &a0, &b0);
    let p1 = schoolbook_product_lcs(
        builder,
        &eval_chunk_lcs(&a0, &a1, &a2, F::ONE, F::ONE, F::ONE),
        &eval_chunk_lcs(&b0, &b1, &b2, F::ONE, F::ONE, F::ONE),
    );
    let pm1 = schoolbook_product_lcs(
        builder,
        &eval_chunk_lcs(&a0, &a1, &a2, F::ONE, -F::ONE, F::ONE),
        &eval_chunk_lcs(&b0, &b1, &b2, F::ONE, -F::ONE, F::ONE),
    );
    let p2 = schoolbook_product_lcs(
        builder,
        &eval_chunk_lcs(&a0, &a1, &a2, F::ONE, two, four),
        &eval_chunk_lcs(&b0, &b1, &b2, F::ONE, two, four),
    );
    let p4 = schoolbook_product_lcs(builder, &a2, &b2);

    let half = two.inverse();
    let sixth = F::from_u64(6).inverse();
    let sixteen = F::from_u64(16);

    let mut c0 = Vec::with_capacity(TOOM3_CHUNK_OUT);
    let mut c1 = Vec::with_capacity(TOOM3_CHUNK_OUT);
    let mut c2 = Vec::with_capacity(TOOM3_CHUNK_OUT);
    let mut c3 = Vec::with_capacity(TOOM3_CHUNK_OUT);
    let mut c4 = Vec::with_capacity(TOOM3_CHUNK_OUT);

    for idx in 0..TOOM3_CHUNK_OUT {
        let c0_i = p0[idx].clone();
        let c4_i = p4[idx].clone();

        let mut c2_i = Lc::zero();
        c2_i = c2_i.add_scaled(&p1[idx], half);
        c2_i = c2_i.add_scaled(&pm1[idx], half);
        c2_i = c2_i.add_scaled(&c0_i, -F::ONE);
        c2_i = c2_i.add_scaled(&c4_i, -F::ONE);

        let mut s_i = Lc::zero();
        s_i = s_i.add_scaled(&p1[idx], half);
        s_i = s_i.add_scaled(&pm1[idx], -half);

        let mut c3_i = Lc::zero();
        c3_i = c3_i.add_scaled(&p2[idx], F::ONE);
        c3_i = c3_i.add_scaled(&c0_i, -F::ONE);
        c3_i = c3_i.add_scaled(&c2_i, -four);
        c3_i = c3_i.add_scaled(&c4_i, -sixteen);
        c3_i = c3_i.add_scaled(&s_i, -two);
        c3_i = Lc::zero().add_scaled(&c3_i, sixth);

        let mut c1_i = s_i;
        c1_i = c1_i.add_scaled(&c3_i, -F::ONE);

        c0.push(c0_i);
        c1.push(c1_i);
        c2.push(c2_i);
        c3.push(c3_i);
        c4.push(c4_i);
    }

    let mut raw = vec![Lc::zero(); TABLE_LEN];
    add_chunk_lcs_at(&mut raw, 0, &c0);
    add_chunk_lcs_at(&mut raw, TOOM3_SPLIT, &c1);
    add_chunk_lcs_at(&mut raw, 2 * TOOM3_SPLIT, &c2);
    add_chunk_lcs_at(&mut raw, 3 * TOOM3_SPLIT, &c3);
    add_chunk_lcs_at(&mut raw, 4 * TOOM3_SPLIT, &c4);
    reduce_lcs_mod_phi_81(&mut raw);

    let mut out: [Option<Var>; D] = [None; D];
    for (idx, out_slot) in out.iter_mut().enumerate() {
        let value = builder.eval(&raw[idx]);
        let v = builder.alloc(value);
        builder.enforce_eq(&Lc::from_var(v), &raw[idx]);
        *out_slot = Some(v);
    }
    out.map(|slot| slot.expect("toom3 ring_mul out slot must be populated"))
}

fn chunk_at(vars: &[Var; D], offset: usize) -> Vec<Lc> {
    (0..TOOM3_SPLIT)
        .map(|idx| Lc::from_var(vars[offset + idx]))
        .collect()
}

fn eval_chunk_lcs(a0: &[Lc], a1: &[Lc], a2: &[Lc], c0: F, c1: F, c2: F) -> Vec<Lc> {
    (0..TOOM3_SPLIT)
        .map(|idx| {
            let mut out = Lc::zero();
            out = out.add_scaled(&a0[idx], c0);
            out = out.add_scaled(&a1[idx], c1);
            out = out.add_scaled(&a2[idx], c2);
            out
        })
        .collect()
}

fn schoolbook_product_lcs(builder: &mut R1csBuilder, lhs: &[Lc], rhs: &[Lc]) -> Vec<Lc> {
    debug_assert_eq!(lhs.len(), TOOM3_SPLIT);
    debug_assert_eq!(rhs.len(), TOOM3_SPLIT);
    let mut out = vec![Lc::zero(); TOOM3_CHUNK_OUT];
    for i in 0..TOOM3_SPLIT {
        for j in 0..TOOM3_SPLIT {
            let product = builder.alloc_mul(&lhs[i], &rhs[j]);
            out[i + j].add_term(product, F::ONE);
        }
    }
    out
}

fn add_chunk_lcs_at(dst: &mut [Lc], offset: usize, src: &[Lc]) {
    for (idx, lc) in src.iter().enumerate() {
        dst[offset + idx] = dst[offset + idx].clone().add_scaled(lc, F::ONE);
    }
}

fn reduce_lcs_mod_phi_81(coeffs: &mut [Lc]) {
    debug_assert_eq!(coeffs.len(), TABLE_LEN);
    for i in (D..TABLE_LEN).rev() {
        let t = coeffs[i].clone();
        coeffs[i] = Lc::zero();
        coeffs[i - D] = coeffs[i - D].clone().add_scaled(&t, -F::ONE);
        let idx_27 = i - PHI_MID_DEGREE;
        if idx_27 < D {
            coeffs[idx_27] = coeffs[idx_27].clone().add_scaled(&t, -F::ONE);
        } else {
            coeffs[idx_27 - D] = coeffs[idx_27 - D].clone().add_scaled(&t, F::ONE);
            if idx_27 - PHI_MID_DEGREE < D {
                coeffs[idx_27 - PHI_MID_DEGREE] = coeffs[idx_27 - PHI_MID_DEGREE]
                    .clone()
                    .add_scaled(&t, F::ONE);
            }
        }
    }
}
