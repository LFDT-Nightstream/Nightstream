//! `claim.y_ring[j] == multilinear_eval(M_j · Z, claim.r)` as R1CS rows.
//!
//! Ports the prototype's `enforce_claim_y_ring_from_point_var` (Bellpepper)
//! into the clean's `R1csBuilder` / `KVar` world.
//!
//! ## Layout
//!
//! - `claim.r ∈ K^{log n}` is unfolded to `chi_r ∈ K^n` in-circuit via
//!   `2^log_n - 1` K-mults (cheap when n is small, as it is for the
//!   F'-image structure).
//! - For each CCS matrix `M_j` and real ring lane rho ∈ 0..D:
//!     row_component(row) = Σ_(logical_col, coeff) coeff · Z[logical_col]
//!                          where (logical_col, coeff) come from
//!                          `row_ring_projection_terms(M_j, row, m, rho)`
//!     y_ring[j][rho] = Σ_row chi_r[row] · row_component(row)   (in K)
//!   Each row contributes 2 base-field mults (one per K-lane).
//! - Native `compute_y_from_Z_and_r` pads each y_ring[j] from the `D`
//!   real coefficients up to `d_pad = D.next_power_of_two()` K-lanes with
//!   `K::ZERO`; the gadget binds those padding lanes `D..d_pad` to zero
//!   so every claim wire is constrained (parity with native exact `Vec`
//!   equality).
//! - The clean's `CeClaimWires` stores `y_ring[j]` as flat 2-limb base
//!   columns: index as `y_ring[j][rho * K_LIMBS + limb]` with K_LIMBS = 2,
//!   so the claim carries `d_pad * K_LIMBS` base wires per matrix.

use neo_ccs::sparse::CcsMatrix;
use neo_math::ring::Rq;
use neo_math::{superneo_bar_block, D, F};
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder};
use crate::engine::r1cs_circuit::field_ext::{enforce_k_mul, klc_add, KLc, KVar};
use crate::lifecycle::Preprocessing;
use crate::paper::reductions::pi_dec_circuit::CeClaimWires;

use super::witness::FinalWitnessWires;

/// Number of base-field columns one K-element occupies in the
/// `CeClaimWires::y_ring` flat layout (`[lane * K_LIMBS + limb]`).
const K_LIMBS: usize = 2;

#[derive(Debug)]
pub(crate) struct YRingError {
    what: &'static str,
    expected: usize,
    got: usize,
}

impl YRingError {
    pub(crate) fn what(&self) -> &'static str {
        self.what
    }
    pub(crate) fn expected(&self) -> usize {
        self.expected
    }
    pub(crate) fn got(&self) -> usize {
        self.got
    }
}

/// Wire-equality binding `claim.ct[j] == claim.y_ring[j][lane=0]` for
/// every CCS matrix. Per SuperNeo paper Theorem 5, `ct(y_j) = M̄_j z(r)`
/// — the constant term of the K-valued ring evaluation is the field-
/// level multilinear eval. The constant term is the `lane=0` K-element,
/// which in the flat-limb layout is `(y_ring[j][0], y_ring[j][1])`.
///
/// `enforce_y_ring_from_z_at_r` already binds `y_ring[j][lane]` to
/// `multilinear_eval(M_j · Z, r)[lane]`; this gadget closes the loop
/// by binding `ct` to the constant-term slice of `y_ring`. Together,
/// `ct[j] = M̄_j z(r)` follows transitively.
pub(crate) fn enforce_ct_from_y_ring(builder: &mut R1csBuilder, claim: &CeClaimWires) -> Result<(), YRingError> {
    if claim.ct.len() != claim.y_ring.len() {
        return Err(YRingError {
            what: "ct length (# of CCS matrices)",
            expected: claim.y_ring.len(),
            got: claim.ct.len(),
        });
    }
    for (j, ct) in claim.ct.iter().enumerate() {
        let row = &claim.y_ring[j];
        if row.len() < K_LIMBS {
            return Err(YRingError {
                what: "y_ring[j] lane-0 limbs",
                expected: K_LIMBS,
                got: row.len(),
            });
        }
        builder.enforce_eq(&Lc::from_var(ct.c0), &Lc::from_var(row[0]));
        builder.enforce_eq(&Lc::from_var(ct.c1), &Lc::from_var(row[1]));
    }
    Ok(())
}

pub(crate) fn enforce_y_ring_from_z_at_r(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    witness: &FinalWitnessWires,
    claim: &CeClaimWires,
) -> Result<(), YRingError> {
    let structure = prep.structure();
    let expected_m = structure.m;
    let n = structure.n;

    if claim.y_ring.len() != structure.matrices.len() {
        return Err(YRingError {
            what: "y_ring outer length (# of CCS matrices)",
            expected: structure.matrices.len(),
            got: claim.y_ring.len(),
        });
    }

    // ── r shape guard ─────────────────────────────────────────────────
    // The chi_r tensor below has `2^|r|` leaves. The honest fold engine
    // sizes the row-domain point as `log2(next_pow2(n).max(2))` — the
    // min-one-round padding shared with `neo_reductions::api::ell_n_for_ccs`
    // / `engines::utils`. A short or long `r` would build a wrong-size
    // tensor and evaluate M·Z at the wrong point, so reject it before
    // unfolding. Mirrors the native `check_ce_relation` guard.
    let expected_r_len = n.next_power_of_two().max(2).trailing_zeros() as usize;
    if claim.r.len() != expected_r_len {
        return Err(YRingError {
            what: "claim.r length",
            expected: expected_r_len,
            got: claim.r.len(),
        });
    }

    // ── 1. Build chi_r tensor in-circuit. Matches neo_ccs::tensor_point
    //    exactly: at each round j, the array doubles by taking two
    //    copies, scaling the FIRST copy by (1 - r[j]) and the SECOND
    //    copy by r[j]. This is the bit-LSB convention:
    //    `chi_r[i] = Π_j (r[j] if bit_j(i) else 1 - r[j])` with `bit_0`
    //    the LSB. Interleaving `(even, odd)` per-entry would swap
    //    positions `1` and `2` for `log_n ≥ 2` and break the y_ring
    //    consistency check.
    let chi_r_wires = build_chi_tensor(builder, &claim.r);
    let row_cap = n.min(chi_r_wires.len());

    // ── 2. y_ring[j][rho] = Σ_row chi_r[row] · row_component(row) ──────
    //
    // Native `compute_y_from_Z_and_r` returns each y_ring[j] as the `D`
    // real ring coefficients (`[K; D]`) padded up to `d_pad = 1 << ell_d
    // = D.next_power_of_two()` K-lanes with `K::ZERO`. `alloc_ce_claim`
    // flattens that faithfully, so the claim carries `d_pad * K_LIMBS`
    // base wires per matrix. Bind ALL of them — lanes `0..D` to `M·Z(r)`
    // and the padding lanes `D..d_pad` to zero — and reject any other
    // length, matching the native `check_ce_relation`'s exact `Vec`
    // equality. (A `< D*K_LIMBS` guard with a `0..D` loop left the
    // padding lanes — and any overlong tail — allocated but
    // unconstrained on the authoritative in-circuit boundary.)
    let d_pad = D.next_power_of_two();
    let expected_y_ring_limbs = d_pad * K_LIMBS;
    for (matrix_idx, matrix) in structure.matrices.iter().enumerate() {
        if claim.y_ring[matrix_idx].len() != expected_y_ring_limbs {
            return Err(YRingError {
                what: "y_ring[j] inner length (d_pad * K_LIMBS)",
                expected: expected_y_ring_limbs,
                got: claim.y_ring[matrix_idx].len(),
            });
        }
        for rho in 0..d_pad {
            let y_c0 = claim.y_ring[matrix_idx][rho * K_LIMBS];
            let y_c1 = claim.y_ring[matrix_idx][rho * K_LIMBS + 1];
            if rho >= D {
                // Padding lane: native fills `D..d_pad` with `K::ZERO`.
                builder.enforce_eq(&Lc::from_var(y_c0), &Lc::zero());
                builder.enforce_eq(&Lc::from_var(y_c1), &Lc::zero());
                continue;
            }
            let mut acc = klc_from_base_const(F::ZERO);
            for row in 0..row_cap {
                let row_terms = row_ring_projection_terms(matrix, row, expected_m, rho)?;
                if row_terms.is_empty() {
                    continue;
                }
                let mut row_component = Lc::zero();
                for (packed_col, coeff) in row_terms {
                    let z_var = witness.packed_entry(packed_col).ok_or(YRingError {
                        what: "witness packed entry",
                        expected: expected_m.div_ceil(D) * D,
                        got: packed_col,
                    })?;
                    row_component.add_term(z_var, coeff);
                }
                let chi_row = &chi_r_wires[row];
                // K-base-mul: (chi_row.c0 + chi_row.c1 X) * row_component
                //   = (chi_row.c0 * row_component) + (chi_row.c1 * row_component) X
                let term_c0 = builder.alloc_mul(&chi_row.c0, &row_component);
                let term_c1 = builder.alloc_mul(&chi_row.c1, &row_component);
                acc = klc_add(&acc, &KLc::from_var(KVar::new(term_c0, term_c1)));
            }
            builder.enforce_eq(&acc.c0, &Lc::from_var(y_c0));
            builder.enforce_eq(&acc.c1, &Lc::from_var(y_c1));
        }
    }
    Ok(())
}

/// Bind the optional NC-channel sidecar `y_zcol` to the opened witness:
/// `claim.y_zcol[rho] == Σ_{col % D = rho} Z[col] · chi_s[col]`, where
/// `chi_s = tensor_point(claim.s_col)`.
///
/// `s_col/y_zcol` are not part of SuperNeo Definition 13's CE tuple, but
/// this implementation carries them inside the accumulator digest and the
/// recursive continuity checks. If present, they must therefore be
/// recomputed from authoritative terminal data instead of trusted as
/// digest-only sidecar fields.
pub(crate) fn enforce_y_zcol_from_z_at_s_col(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    witness: &FinalWitnessWires,
    claim: &CeClaimWires,
) -> Result<(), YRingError> {
    let has_nc_channel = !(claim.s_col.is_empty() && claim.y_zcol.is_empty());
    if !has_nc_channel {
        return Ok(());
    }
    if claim.s_col.is_empty() || claim.y_zcol.is_empty() {
        return Err(YRingError {
            what: "incomplete NC channel (s_col/y_zcol)",
            expected: 2,
            got: 1,
        });
    }

    let expected_m = prep.structure().m;
    let expected_s_col_len = expected_m.next_power_of_two().max(2).trailing_zeros() as usize;
    if claim.s_col.len() != expected_s_col_len {
        return Err(YRingError {
            what: "claim.s_col length",
            expected: expected_s_col_len,
            got: claim.s_col.len(),
        });
    }

    let d_pad = D.next_power_of_two();
    let expected_y_zcol_limbs = d_pad * K_LIMBS;
    if claim.y_zcol_lanes != d_pad || claim.y_zcol.len() != expected_y_zcol_limbs {
        return Err(YRingError {
            what: "claim.y_zcol length (d_pad * K_LIMBS)",
            expected: expected_y_zcol_limbs,
            got: claim.y_zcol.len(),
        });
    }

    let chi_s_wires = build_chi_tensor(builder, &claim.s_col);
    let col_cap = expected_m.min(chi_s_wires.len());
    for rho in 0..d_pad {
        let y_c0 = claim.y_zcol[rho * K_LIMBS];
        let y_c1 = claim.y_zcol[rho * K_LIMBS + 1];
        if rho >= D {
            builder.enforce_eq(&Lc::from_var(y_c0), &Lc::zero());
            builder.enforce_eq(&Lc::from_var(y_c1), &Lc::zero());
            continue;
        }

        let mut acc = klc_from_base_const(F::ZERO);
        let mut logical_col = rho;
        while logical_col < col_cap {
            let z_var = witness
                .logical_entry(expected_m, logical_col)
                .ok_or(YRingError {
                    what: "witness logical entry",
                    expected: expected_m,
                    got: logical_col,
                })?;
            let z_lc = Lc::from_var(z_var);
            let chi_col = &chi_s_wires[logical_col];
            let term_c0 = builder.alloc_mul(&chi_col.c0, &z_lc);
            let term_c1 = builder.alloc_mul(&chi_col.c1, &z_lc);
            acc = klc_add(&acc, &KLc::from_var(KVar::new(term_c0, term_c1)));
            logical_col += D;
        }
        builder.enforce_eq(&acc.c0, &Lc::from_var(y_c0));
        builder.enforce_eq(&acc.c1, &Lc::from_var(y_c1));
    }
    Ok(())
}

/// Sparse `(packed_col, coeff)` pairs realising `(M_j · Z)[row, rho]`
/// under SuperNeo's ring-bar projection on each complete `D`-coefficient
/// block. `packed_col` ranges over `ceil(effective_m / D) * D`: the matrix is
/// zero-extended past `effective_m`, but the final witness ring element is
/// not truncated, matching native `compute_y_from_Z_and_r`.
///
/// Mirrors the prototype's private `row_ring_projection_terms` so the
/// in-circuit eval matches the native `compute_y_from_Z_and_r`.
fn row_ring_projection_terms(
    matrix: &CcsMatrix<F>,
    row: usize,
    effective_m: usize,
    rho: usize,
) -> Result<Vec<(usize, F)>, YRingError> {
    if rho >= D {
        return Err(YRingError {
            what: "rho < D",
            expected: D,
            got: rho,
        });
    }
    let block_count = effective_m.div_ceil(D);
    let mut terms = Vec::new();
    for blk in 0..block_count {
        let base = blk * D;
        let mut a = [F::ZERO; D];
        for (off, coeff) in a.iter_mut().enumerate() {
            *coeff = matrix_entry_base_f(matrix, row, base + off);
        }
        if a.iter().all(|value| *value == F::ZERO) {
            continue;
        }
        let a_bar = Rq(superneo_bar_block(a));
        for off in 0..D {
            let packed_col = base + off;
            let mut basis = [F::ZERO; D];
            basis[off] = F::ONE;
            let coeff = a_bar.mul(&Rq(basis)).0[rho];
            if coeff != F::ZERO {
                terms.push((packed_col, coeff));
            }
        }
    }
    Ok(terms)
}

fn matrix_entry_base_f(matrix: &CcsMatrix<F>, row: usize, col: usize) -> F {
    if row >= matrix.rows() || col >= matrix.cols() {
        return F::ZERO;
    }
    match matrix {
        CcsMatrix::Identity { .. } => {
            if row == col {
                F::ONE
            } else {
                F::ZERO
            }
        }
        CcsMatrix::Csc(csc) => {
            let start = csc.col_ptr[col];
            let end = csc.col_ptr[col + 1];
            let mut acc = F::ZERO;
            for idx in start..end {
                if csc.row_idx[idx] == row {
                    acc += csc.vals[idx];
                }
            }
            acc
        }
        CcsMatrix::CscWithSeededPhi81 { csc, blocks } => {
            let mut acc = F::ZERO;
            for index in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                if csc.row_idx[index] == row {
                    acc += csc.vals[index];
                }
            }
            for block in blocks {
                acc += block.entry::<F>(row, col);
            }
            acc
        }
    }
}

fn build_chi_tensor(builder: &mut R1csBuilder, point: &[KVar]) -> Vec<KLc> {
    let mut chi_wires: Vec<KLc> = vec![klc_from_base_const(F::ONE)];
    for point_j in point {
        let point_j_klc = KLc::from_var(*point_j);
        let one_minus_point_j_klc = klc_one_minus(point_j);

        let old_len = chi_wires.len();
        let mut next_wires = Vec::with_capacity(old_len * 2);
        // First half: scale by (1 - point[j]).
        for chi_i in &chi_wires {
            let scaled = enforce_k_mul(builder, chi_i, &one_minus_point_j_klc);
            next_wires.push(KLc::from_var(scaled));
        }
        // Second half: scale by point[j].
        for chi_i in &chi_wires {
            let scaled = enforce_k_mul(builder, chi_i, &point_j_klc);
            next_wires.push(KLc::from_var(scaled));
        }
        chi_wires = next_wires;
    }
    chi_wires
}

/// `K::from_coeffs([c, 0])` lifted into a `KLc`. No constraints.
fn klc_from_base_const(c: F) -> KLc {
    KLc {
        c0: Lc::from_const(c),
        c1: Lc::zero(),
    }
}

/// `1 - r` as a `KLc`, given `r: KVar`. No constraints (pure linear).
fn klc_one_minus(r: &KVar) -> KLc {
    let mut c0 = Lc::from_const(F::ONE);
    c0 = c0.add_scaled(&Lc::from_var(r.c0), -F::ONE);
    let c1 = Lc::zero().add_scaled(&Lc::from_var(r.c1), -F::ONE);
    KLc { c0, c1 }
}
