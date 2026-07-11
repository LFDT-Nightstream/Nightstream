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

use crate::engine::r1cs_circuit::builder::{Lc, PolynomialEvaluationTrace, R1csBuilder, Var};

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

// ── Projection-checked batched ring action (encoding.md candidate E) ──────
//
// The gadgets above materialize all `D²` partial products of one `ρ · c`
// — the measured wall of the folded `enc(F')` regime (~197k committed
// bits per pair at U64). The projection check replaces materialization
// with a polynomial-identity test at a transcript challenge `β ∈ K`:
//
//   out = Σ_i ρ_i · c_i (mod Φ)
//     ⟺  Σ_i ρ_i(X)·c_i(X) = q(X)·Φ(X) + out(X)      (q = the quotient)
//     ⟸  the same identity at a random β, except w.p. ≤ (2D−2)/|K|
//         over β — Schwartz–Zippel, sound only if β is sampled AFTER
//         ρ_i, c_i, out, q are committed (commit-then-challenge, the
//         same discipline as γ and the fold challenges).
//
// Committed material per input pair drops from O(D²) products to O(D)
// evaluation terms; the β power ladder is shared across every pair of a
// step. The caller owns β's transcript binding. The authoritative NIFS.V
// commitment client supplies transcript-derived β and quotient wires;
// other clients and the final low-norm lowering remain Road A work.

use crate::engine::r1cs_circuit::field_ext::{alloc_klc, enforce_k_mul, KLc, KVar};

/// Quotient length for the batched identity: `deg(Σ ρ_i·c_i) ≤ 2D − 2`,
/// so `deg q = 2D − 2 − D = D − 2` → `D − 1` coefficients.
pub const PROJECTION_QUOTIENT_LEN: usize = D - 1;

/// `β^0 .. β^top` as constrained `KVar`s (one K-mult per power).
/// `top = D` covers everything the batched check needs: evaluations use
/// `β^0..β^{D−1}`, and `Φ(β)` reads `β^{27}` and `β^{54}`.
pub fn enforce_beta_ladder(builder: &mut R1csBuilder, beta: KVar, top: usize) -> Vec<KVar> {
    let one_c0 = builder.alloc(F::ONE);
    builder.enforce_eq(&Lc::from_var(one_c0), &Lc::from_const(F::ONE));
    let zero_c1 = builder.alloc(F::ZERO);
    builder.enforce_zero(&Lc::from_var(zero_c1));
    let mut powers = Vec::with_capacity(top + 1);
    powers.push(KVar::new(one_c0, zero_c1));
    for k in 1..=top {
        let prev = powers[k - 1];
        powers.push(enforce_k_mul(builder, &KLc::from_var(prev), &KLc::from_var(beta)));
    }
    powers
}

/// `p(β) = Σ_j coeffs[j] · β^j` as a constrained `KVar`. Two committed
/// product wires per coefficient (the `j = 0` term is free: `β^0 = 1`).
pub fn enforce_eval_at_beta(builder: &mut R1csBuilder, coeffs: &[Var], powers: &[KVar]) -> KVar {
    assert!(coeffs.len() <= powers.len(), "ladder too short for this polynomial");
    let row_start = builder.rows();
    let column_start = builder.cols();
    let mut sum = KLc::zero();
    for (j, &coeff) in coeffs.iter().enumerate() {
        if j == 0 {
            sum.c0.add_term(coeff, F::ONE);
            continue;
        }
        let p0 = builder.alloc_mul(&Lc::from_var(coeff), &Lc::from_var(powers[j].c0));
        let p1 = builder.alloc_mul(&Lc::from_var(coeff), &Lc::from_var(powers[j].c1));
        sum.c0.add_term(p0, F::ONE);
        sum.c1.add_term(p1, F::ONE);
    }
    let output = alloc_klc(builder, &sum);
    builder.record_polynomial_evaluation(PolynomialEvaluationTrace {
        row_start,
        row_end: builder.rows(),
        allocated_columns: (column_start..builder.cols()).collect(),
        coefficient_cols: coeffs.iter().map(|value| value.col()).collect(),
        power_cols: powers[..coeffs.len()]
            .iter()
            .map(|power| [power.c0.col(), power.c1.col()])
            .collect(),
        output_cols: [output.c0.col(), output.c1.col()],
    });
    output
}

/// Evaluations tied to an exact ordered list of polynomial wire columns.
/// Private fields prevent a caller from pairing cached values with different
/// transcript-derived polynomials.
pub struct PolynomialEvaluationsAtBeta {
    source_columns: Vec<[usize; D]>,
    evaluations: Vec<KVar>,
}

pub fn enforce_polynomial_evaluations_at_beta(
    builder: &mut R1csBuilder,
    polynomials: &[[Var; D]],
    powers: &[KVar],
) -> PolynomialEvaluationsAtBeta {
    PolynomialEvaluationsAtBeta {
        source_columns: polynomials
            .iter()
            .map(|polynomial| polynomial.map(Var::col))
            .collect(),
        evaluations: polynomials
            .iter()
            .map(|polynomial| enforce_eval_at_beta(builder, polynomial, powers))
            .collect(),
    }
}

/// Enforce `out = Σ_i ρ_i · c_i (mod Φ)` via the projection identity at
/// β: `Σ_i ρ_i(β)·c_i(β) = q(β)·Φ(β) + out(β)`.
///
/// The caller supplies the shared `enforce_beta_ladder(β, D)` powers,
/// the committed operand coefficient wires, and the committed quotient
/// wires (prover side: [`projection_quotient`]). β MUST be a
/// transcript challenge sampled after every operand and the quotient
/// are committed — this gadget only enforces the algebra.
pub fn enforce_ring_action_projection_batch(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    pairs: &[(&[Var; D], &[Var; D])],
    out: &[Var; D],
    quotient: &[Var; PROJECTION_QUOTIENT_LEN],
) {
    let rho_polynomials = pairs.iter().map(|(rho, _)| **rho).collect::<Vec<_>>();
    let rho_evaluations = enforce_polynomial_evaluations_at_beta(builder, &rho_polynomials, powers);
    enforce_ring_action_projection_batch_with_rho_evaluations(builder, powers, &rho_evaluations, pairs, out, quotient);
}

/// Projection identity using rho evaluations constrained once and reused by
/// every commitment, X, and evaluation-vector client in the same NIFS step.
pub fn enforce_ring_action_projection_batch_with_rho_evaluations(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    rho_evaluations: &PolynomialEvaluationsAtBeta,
    pairs: &[(&[Var; D], &[Var; D])],
    out: &[Var; D],
    quotient: &[Var; PROJECTION_QUOTIENT_LEN],
) {
    let identity_row_start = builder.rows();
    assert!(powers.len() > D, "ladder must reach β^D for Φ(β)");
    assert_eq!(rho_evaluations.evaluations.len(), pairs.len(), "rho evaluation count");
    // Σ_i ρ_i(β) · c_i(β), one K-mult per pair.
    let mut lhs = KLc::zero();
    for (pair_index, (rho, c)) in pairs.iter().enumerate() {
        assert_eq!(
            rho_evaluations.source_columns[pair_index],
            rho.map(Var::col),
            "cached rho evaluation must match the identity's rho wires"
        );
        let rho_eval = rho_evaluations.evaluations[pair_index];
        let c_eval = enforce_eval_at_beta(builder, c.as_slice(), powers);
        let term = enforce_k_mul(builder, &KLc::from_var(rho_eval), &KLc::from_var(c_eval));
        lhs.c0.add_term(term.c0, F::ONE);
        lhs.c1.add_term(term.c1, F::ONE);
    }

    // q(β)·Φ(β) + out(β), with Φ(β) = β^D + β^{PHI_MID_DEGREE} + 1 as a
    // linear form over the shared ladder.
    let out_eval = enforce_eval_at_beta(builder, out.as_slice(), powers);
    let q_eval = enforce_eval_at_beta(builder, quotient.as_slice(), powers);
    let mut phi_beta = KLc::zero();
    phi_beta.c0.add_term(powers[D].c0, F::ONE);
    phi_beta.c0.add_term(powers[PHI_MID_DEGREE].c0, F::ONE);
    phi_beta.c0.add_constant(F::ONE);
    phi_beta.c1.add_term(powers[D].c1, F::ONE);
    phi_beta.c1.add_term(powers[PHI_MID_DEGREE].c1, F::ONE);
    let q_phi = enforce_k_mul(builder, &KLc::from_var(q_eval), &phi_beta);

    let mut rhs = KLc::zero();
    rhs.c0.add_term(q_phi.c0, F::ONE);
    rhs.c0.add_term(out_eval.c0, F::ONE);
    rhs.c1.add_term(q_phi.c1, F::ONE);
    rhs.c1.add_term(out_eval.c1, F::ONE);

    builder.enforce_eq(&lhs.c0, &rhs.c0);
    builder.enforce_eq(&lhs.c1, &rhs.c1);
    builder.record_row_family("nifs.pi_rlc.projection_identity", identity_row_start);
}

/// Prover-side companion of [`enforce_ring_action_projection_batch`]:
/// the reduced result `out` and the division quotient `q` with
/// `Σ_i ρ_i(X)·c_i(X) = q(X)·Φ(X) + out(X)` exactly (monic long
/// division by `Φ = X^D + X^{27} + 1`).
pub fn projection_quotient(pairs: &[([F; D], [F; D])]) -> ([F; D], [F; PROJECTION_QUOTIENT_LEN]) {
    let mut p = [F::ZERO; TABLE_LEN];
    for (rho, c) in pairs {
        for i in 0..D {
            if rho[i] == F::ZERO {
                continue;
            }
            for j in 0..D {
                p[i + j] += rho[i] * c[j];
            }
        }
    }
    let mut q = [F::ZERO; PROJECTION_QUOTIENT_LEN];
    for k in (D..TABLE_LEN).rev() {
        let t = p[k];
        if t == F::ZERO {
            continue;
        }
        q[k - D] = t;
        p[k] = F::ZERO;
        p[k - PHI_MID_DEGREE] -= t;
        p[k - D] -= t;
    }
    let mut out = [F::ZERO; D];
    out.copy_from_slice(&p[..D]);
    (out, q)
}
