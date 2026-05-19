//! Π_DEC.V — in-circuit verifier (paper §7.5 step 2).
//!
//! Reduction:  CE(B, ℒ)   →   CE(b, ℒ)^k     where B = b^k
//!
//! The verifier has no random coins. Soundness comes from re-deriving the
//! parent from the children via the b-ary homomorphism and rejecting on
//! mismatch.
//!
//! ## What this gadget owns
//!
//! - Allocation of parent and children CE-claim wires inside an
//!   [`R1csBuilder`].
//! - Linear-combination constraints that enforce, lane-by-lane:
//!     - `parent.c.data[ℓ]    == Σ_{i∈[k]} b^{i-1} · child_i.c.data[ℓ]`
//!     - `parent.X[r,c]       == Σ_{i∈[k]} b^{i-1} · child_i.X[r,c]`
//!     - `parent.y_ring[j][ℓ] == Σ_{i∈[k]} b^{i-1} · child_i.y_ring[j][ℓ]`
//!   (`y_ring` lanes hold `K`-elements; the sum is enforced separately on
//!   each of the `s` base-field coefficients of `K`.)
//!
//! ## What this gadget does NOT own
//!
//! - Sampling, transcript, or any random challenges (Π_DEC has none).
//! - Validation of the children's `r` — `r` is shared between parent and
//!   children by paper definition; the wiring constraint (parent.r == child_i.r)
//!   is enforced when the gadget is composed inside Π_RLC.V → Π_DEC.V chain.
//!   Standalone callers wishing to lock parent.r ≡ child.r should add the
//!   equality after [`alloc_dec_inputs`] returns.
//! - Canonicality of `split_b(x)`. Strict mode mirrors native
//!   `verify_dec_public`: it checks public b-ary recomposition of `X`, but
//!   does not range-check the child `X` digits.

use neo_ajtai::Commitment;
use neo_math::{KExtensions, F, K};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

use crate::engine::r1cs_circuit::boolean;
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::paper::digest::digest32_as_fields;
use crate::paper::params::Params;
use crate::paper::relations::CeClaim;

/// Wires for one CE claim inside the Π_DEC.V gadget. Returned by
/// [`alloc_dec_inputs`] so callers can compose further constraints without
/// re-walking the witness layout.
///
/// The shape fields (`c_d`, `c_kappa`, `x_rows`, `x_cols`, `m_in`,
/// `y_ring_lanes`, `y_zcol_lanes`) are non-wire `usize` parameters that
/// the verifier already pinned implicitly via `check_shapes`; they are
/// re-exposed so downstream gadgets (e.g. the decider's CE-continuity
/// gate) can branch on them without re-walking the underlying CE claim.
#[derive(Clone, Debug)]
pub struct CeClaimWires {
    /// `d * kappa` columns, column-major (matches `Commitment::data`).
    pub c_data: Vec<Var>,
    /// Ajtai dimension `d` of the commitment.
    pub c_d: usize,
    /// Ajtai dimension `kappa` of the commitment.
    pub c_kappa: usize,
    /// `rows * cols` columns, row-major.
    pub x: Vec<Var>,
    pub x_rows: usize,
    pub x_cols: usize,
    /// Public input length the claim was constructed under.
    pub m_in: usize,
    /// `t` outer × `d` lanes × `s` base-field columns per K-element.
    /// Index as `y_ring[j][lane * s + limb]`.
    pub y_ring: Vec<Vec<Var>>,
    pub y_ring_lanes: usize,
    /// CE evaluation point `r ∈ K^{log n}`. Shared between parent and all
    /// children in a Π_DEC.V step; enforced via [`enforce_r_consistency`].
    pub r: Vec<KVar>,
    /// NC column-domain point `s_col ∈ K^{log m}`. Shared between parent
    /// and all children (NC channel doesn't decompose `s_col`); enforced
    /// via [`enforce_s_col_consistency`].
    pub s_col: Vec<KVar>,
    /// `d` lanes × `s` base-field columns of `y_zcol` (the NC output
    /// column). Index as `y_zcol[lane * s + limb]`. **Not** subject to
    /// b-ary recomposition: Π_CCS outputs mix MCS digit-decomposed and ME
    /// linear y_zcols, so `Σ b^{i-1} · child.y_zcol ≠ parent.y_zcol` in
    /// general. Children's y_zcol values are re-bound by the next step's
    /// Π_CCS NC terminal identity. Mirrors native `verify_dec_public`.
    pub y_zcol: Vec<Var>,
    pub y_zcol_lanes: usize,
    /// `fold_digest` field of the CE claim, projected to four base-field
    /// lanes. Allocated from `claim.fold_digest` so the decider's
    /// CE-continuity gate can pin it equal to the next step's running's
    /// `fold_digest_fields`.
    pub fold_digest_fields: [Var; 4],
}

/// Wires for the full DEC input set: one parent + `k_rho` children.
#[derive(Clone, Debug)]
pub struct DecInputWires {
    pub parent: CeClaimWires,
    pub children: Vec<CeClaimWires>,
}

/// Allocate witness variables for parent + k children CE claims and return
/// their wire handles. No constraints are emitted by this function.
///
/// Callers must call [`enforce_dec_v`] (and optionally [`enforce_x_bitness`])
/// to actually constrain the relationship.
pub fn alloc_dec_inputs(builder: &mut R1csBuilder, parent: &CeClaim, children: &[CeClaim]) -> DecInputWires {
    let parent_wires = alloc_ce_claim(builder, parent);
    let child_wires = children
        .iter()
        .map(|c| alloc_ce_claim(builder, c))
        .collect();
    DecInputWires {
        parent: parent_wires,
        children: child_wires,
    }
}

/// Emit Π_DEC.V constraints on already-allocated input wires.
///
/// Returns `Err` if shapes drift (children disagree with parent, or there
/// are not exactly `pp.k_rho()` of them).
pub fn enforce_dec_v(builder: &mut R1csBuilder, pp: &Params, wires: &DecInputWires) -> Result<(), Error> {
    let k = pp.k_rho() as usize;
    if wires.children.len() != k {
        return Err(Error::ChildCount {
            expected: k,
            got: wires.children.len(),
        });
    }
    check_shapes(&wires.parent, &wires.children)?;

    // b^{i-1} as F-scalars, i = 1..=k.
    let b = F::from_u64(pp.b() as u64);
    let mut b_pows = Vec::with_capacity(k);
    let mut p = F::ONE;
    for _ in 0..k {
        b_pows.push(p);
        p *= b;
    }

    enforce_lane_combination(
        builder,
        &wires.parent.c_data,
        &wires.children,
        &b_pows,
        ChildField::Commitment,
    );
    enforce_lane_combination(builder, &wires.parent.x, &wires.children, &b_pows, ChildField::X);
    for j in 0..wires.parent.y_ring.len() {
        enforce_lane_combination_y(builder, j, &wires.parent.y_ring[j], &wires.children, &b_pows);
    }
    // NOTE: No y_zcol b-ary recomposition. Native `verify_dec_public` does
    // not enforce `parent.y_zcol == Σ b^{i-1} · child.y_zcol`. The identity
    // doesn't hold in production because Π_CCS outputs mix MCS y_zcol
    // (digit-decomposed) and ME y_zcol (linear) — their Π_RLC combination
    // doesn't telescope under Π_DEC's b-ary split. Children's y_zcol are
    // re-bound by the next step's Π_CCS NC terminal identity.
    Ok(())
}

/// Optional: enforce `child_i.X[k] ∈ {0, …, b-1}`. For `b = 2` this is
/// pure bitness (`x · (1 - x) = 0`); for `b > 2` we enforce
/// `Π_{a=0..b-1} (x - a) = 0` via repeated multiplication. Caller decides
/// whether to use this gadget — strict Construction-2 builds it on top of
/// [`enforce_dec_v`] when a fresh F' instance comes from a low-norm split.
pub fn enforce_x_bitness(builder: &mut R1csBuilder, pp: &Params, wires: &DecInputWires) {
    let b = pp.b();
    for child in &wires.children {
        for &x_var in &child.x {
            boolean::enforce_low_norm(builder, x_var, b);
        }
    }
}

/// Enforce `parent.r == child_i.r` for every child `i`. The CE evaluation
/// point is shared between parent and children by paper §7.5 definition.
///
/// Returns `Err` if any child's `r` length differs from the parent's
/// (`alloc_dec_inputs` may have allocated mismatched shapes).
pub fn enforce_r_consistency(builder: &mut R1csBuilder, wires: &DecInputWires) -> Result<(), Error> {
    for (idx, child) in wires.children.iter().enumerate() {
        if child.r.len() != wires.parent.r.len() {
            return Err(Error::ShapeMismatch {
                what: "r length",
                expected: wires.parent.r.len(),
                got: child.r.len(),
                idx,
            });
        }
        for (p, c) in wires.parent.r.iter().zip(child.r.iter()) {
            builder.enforce_eq(&Lc::from_var(p.c0), &Lc::from_var(c.c0));
            builder.enforce_eq(&Lc::from_var(p.c1), &Lc::from_var(c.c1));
        }
    }
    Ok(())
}

/// Strict Π_DEC.V — mirrors native `verify_dec_public`.
///
/// Composes:
///   1. [`enforce_dec_v`] — `(c, X, y_ring)` b-ary recomposition.
///   2. [`enforce_r_consistency`] — `parent.r == child_i.r` for all `i`.
///   3. [`enforce_s_col_consistency`] — `parent.s_col == child_i.s_col`.
///
/// Notably absent (and absent on the native side):
///   - **No `y_zcol` b-ary check.** Π_CCS outputs mix MCS digit-decomposed
///     and ME linear y_zcols; after Π_RLC the parent's y_zcol doesn't
///     telescope under Π_DEC's b-ary split. Children's y_zcols are re-bound
///     by the next step's Π_CCS NC terminal identity.
///   - **No x bitness check.** `decompose_balanced_fixed_d_digits_k`
///     produces signed digits (e.g. -1 ↦ p-1 in F), so an unsigned
///     `{0..b-1}` check would reject honest provers. Low-norm soundness is
///     carried by Ajtai-commitment binding, not by a verifier-side range
///     check on `child.X`. [`enforce_x_bitness`] remains available for
///     callers that have an unsigned range invariant to enforce.
pub fn enforce_dec_v_strict(builder: &mut R1csBuilder, pp: &Params, wires: &DecInputWires) -> Result<(), Error> {
    enforce_dec_v(builder, pp, wires)?;
    enforce_r_consistency(builder, wires)?;
    enforce_s_col_consistency(builder, wires)?;
    enforce_inactive_x_zero(builder, wires)?;
    Ok(())
}

/// Reject parent + children whose `X` has non-zero entries in columns
/// `[ceil(m_in / D), x.cols())`. Children become the next running
/// accumulator; without this, a terminal state could carry a non-canonical
/// accumulator that no downstream Π_CCS would re-validate. Mirrors the
/// native-side `pi_dec::validate_inactive_x_zero`.
pub fn enforce_inactive_x_zero(builder: &mut R1csBuilder, wires: &DecInputWires) -> Result<(), Error> {
    enforce_inactive_x_zero_one(builder, &wires.parent, 0)?;
    for (idx, child) in wires.children.iter().enumerate() {
        enforce_inactive_x_zero_one(builder, child, idx)?;
    }
    Ok(())
}

fn enforce_inactive_x_zero_one(builder: &mut R1csBuilder, claim: &CeClaimWires, idx: usize) -> Result<(), Error> {
    // CeClaimWires::x_cols equals the underlying CE claim's `m_in` (set in
    // `alloc_ce_claim` from `claim.X.cols()`, which the SplitNc shape check
    // forces to `m_in`).
    let active_cols = crate::paper::relations::superneo_public_x_cols(claim.x_cols);
    if active_cols > claim.x_cols {
        return Err(Error::ShapeMismatch {
            what: "active X columns",
            expected: claim.x_cols,
            got: active_cols,
            idx,
        });
    }
    for r in 0..claim.x_rows {
        for c in active_cols..claim.x_cols {
            builder.enforce_eq(&Lc::from_var(claim.x[r * claim.x_cols + c]), &Lc::zero());
        }
    }
    Ok(())
}

/// Validate that parent + every child have SplitNc-shaped `y_ring` rows and
/// `y_zcol` of exactly `d_pad = 2^ell_d` K-element lanes.
///
/// `enforce_dec_v`'s generic [`check_shapes`] only enforces parent ↔ child
/// length parity — that's enough for the b-ary recomposition algebra but
/// doesn't catch the case where both sides are silently un-padded. The
/// SplitNc verifier path requires the Ajtai-padded shape; NIFS.V layers
/// this check on top of [`enforce_dec_v_strict`] before consuming the
/// wires.
pub fn enforce_split_nc_d_pad_shape(wires: &DecInputWires, d_pad: usize) -> Result<(), Error> {
    let label = |what: &'static str, got: usize, idx: usize| Error::ShapeMismatch {
        what,
        expected: d_pad,
        got,
        idx,
    };
    for (j, row) in wires.parent.y_ring.iter().enumerate() {
        if wires.parent.y_ring_lanes != d_pad || row.len() != d_pad * K_LIMBS {
            return Err(label("parent.y_ring[j] lane count", wires.parent.y_ring_lanes, j));
        }
    }
    if wires.parent.y_zcol_lanes != d_pad || wires.parent.y_zcol.len() != d_pad * K_LIMBS {
        return Err(label("parent.y_zcol lane count", wires.parent.y_zcol_lanes, 0));
    }
    for (idx, child) in wires.children.iter().enumerate() {
        for row in child.y_ring.iter() {
            if child.y_ring_lanes != d_pad || row.len() != d_pad * K_LIMBS {
                return Err(label("child.y_ring[j] lane count", child.y_ring_lanes, idx));
            }
        }
        if child.y_zcol_lanes != d_pad || child.y_zcol.len() != d_pad * K_LIMBS {
            return Err(label("child.y_zcol lane count", child.y_zcol_lanes, idx));
        }
    }
    Ok(())
}

/// Enforce `parent.s_col == child_i.s_col` for every child `i`. The NC
/// column-domain point is shared by every claim in a Π_DEC.V step (parent
/// and all children), mirroring `r` for the FE channel.
pub fn enforce_s_col_consistency(builder: &mut R1csBuilder, wires: &DecInputWires) -> Result<(), Error> {
    for (idx, child) in wires.children.iter().enumerate() {
        if child.s_col.len() != wires.parent.s_col.len() {
            return Err(Error::ShapeMismatch {
                what: "s_col length",
                expected: wires.parent.s_col.len(),
                got: child.s_col.len(),
                idx,
            });
        }
        for (p, c) in wires.parent.s_col.iter().zip(child.s_col.iter()) {
            builder.enforce_eq(&Lc::from_var(p.c0), &Lc::from_var(c.c0));
            builder.enforce_eq(&Lc::from_var(p.c1), &Lc::from_var(c.c1));
        }
    }
    Ok(())
}

// ── private helpers ───────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum ChildField {
    Commitment,
    X,
}

fn enforce_lane_combination(
    builder: &mut R1csBuilder,
    parent_lanes: &[Var],
    children: &[CeClaimWires],
    b_pows: &[F],
    field: ChildField,
) {
    let n = parent_lanes.len();
    for lane in 0..n {
        let mut combo = Lc::zero();
        for (idx, child) in children.iter().enumerate() {
            let child_var = match field {
                ChildField::Commitment => child.c_data[lane],
                ChildField::X => child.x[lane],
            };
            combo.add_term(child_var, b_pows[idx]);
        }
        builder.enforce_eq(&Lc::from_var(parent_lanes[lane]), &combo);
    }
}

fn enforce_lane_combination_y(
    builder: &mut R1csBuilder,
    j: usize,
    parent_lanes: &[Var],
    children: &[CeClaimWires],
    b_pows: &[F],
) {
    for lane in 0..parent_lanes.len() {
        let mut combo = Lc::zero();
        for (idx, child) in children.iter().enumerate() {
            combo.add_term(child.y_ring[j][lane], b_pows[idx]);
        }
        builder.enforce_eq(&Lc::from_var(parent_lanes[lane]), &combo);
    }
}

fn alloc_ce_claim(builder: &mut R1csBuilder, claim: &CeClaim) -> CeClaimWires {
    let c_data = builder.alloc_vec(&claim.c.data);
    let x_rows = claim.X.rows();
    let x_cols = claim.X.cols();
    let mut x = Vec::with_capacity(x_rows * x_cols);
    for r in 0..x_rows {
        for c in 0..x_cols {
            x.push(builder.alloc(claim.X[(r, c)]));
        }
    }
    let y_ring = claim
        .y_ring
        .iter()
        .map(|row| {
            let mut lane_vars = Vec::with_capacity(row.len() * K_LIMBS);
            for elem in row {
                for limb in elem.as_basis_coefficients_slice() {
                    lane_vars.push(builder.alloc(*limb));
                }
            }
            lane_vars
        })
        .collect::<Vec<_>>();
    let y_ring_lanes = claim.y_ring.first().map(|row| row.len()).unwrap_or(0);
    let r = claim
        .r
        .iter()
        .map(|k| {
            let [c0, c1] = k.as_coeffs();
            KVar::alloc(builder, c0, c1)
        })
        .collect();
    let s_col = claim
        .s_col
        .iter()
        .map(|k| {
            let [c0, c1] = k.as_coeffs();
            KVar::alloc(builder, c0, c1)
        })
        .collect();
    let y_zcol_lanes = claim.y_zcol.len();
    let mut y_zcol = Vec::with_capacity(y_zcol_lanes * K_LIMBS);
    for elem in &claim.y_zcol {
        for limb in elem.as_basis_coefficients_slice() {
            y_zcol.push(builder.alloc(*limb));
        }
    }
    // Allocate the CE claim's fold_digest as four base-field wires so
    // downstream gadgets (decider CE-continuity gate) can pin it equal
    // to the next step's running's `fold_digest_fields`.
    let fold_digest_lanes = digest32_as_fields(claim.fold_digest);
    let fold_digest_fields: [Var; 4] = [
        builder.alloc(fold_digest_lanes[0]),
        builder.alloc(fold_digest_lanes[1]),
        builder.alloc(fold_digest_lanes[2]),
        builder.alloc(fold_digest_lanes[3]),
    ];
    CeClaimWires {
        c_data,
        c_d: claim.c.d,
        c_kappa: claim.c.kappa,
        x,
        x_rows,
        x_cols,
        m_in: claim.m_in,
        y_ring,
        y_ring_lanes,
        r,
        s_col,
        y_zcol,
        y_zcol_lanes,
        fold_digest_fields,
    }
}

fn check_shapes(parent: &CeClaimWires, children: &[CeClaimWires]) -> Result<(), Error> {
    for (idx, child) in children.iter().enumerate() {
        if child.c_data.len() != parent.c_data.len() {
            return Err(Error::ShapeMismatch {
                what: "commitment lane count",
                expected: parent.c_data.len(),
                got: child.c_data.len(),
                idx,
            });
        }
        if child.x.len() != parent.x.len() || child.x_rows != parent.x_rows || child.x_cols != parent.x_cols {
            return Err(Error::ShapeMismatch {
                what: "X dimensions",
                expected: parent.x.len(),
                got: child.x.len(),
                idx,
            });
        }
        if child.y_ring.len() != parent.y_ring.len() {
            return Err(Error::ShapeMismatch {
                what: "y_ring outer length (t)",
                expected: parent.y_ring.len(),
                got: child.y_ring.len(),
                idx,
            });
        }
        for (j, (p, c)) in parent.y_ring.iter().zip(child.y_ring.iter()).enumerate() {
            if p.len() != c.len() {
                return Err(Error::ShapeMismatch {
                    what: "y_ring[j] lane length",
                    expected: p.len(),
                    got: c.len(),
                    idx: j,
                });
            }
        }
        if child.s_col.len() != parent.s_col.len() {
            return Err(Error::ShapeMismatch {
                what: "s_col length",
                expected: parent.s_col.len(),
                got: child.s_col.len(),
                idx,
            });
        }
        if child.y_zcol.len() != parent.y_zcol.len() || child.y_zcol_lanes != parent.y_zcol_lanes {
            return Err(Error::ShapeMismatch {
                what: "y_zcol lane length",
                expected: parent.y_zcol.len(),
                got: child.y_zcol.len(),
                idx,
            });
        }
    }
    Ok(())
}

const K_LIMBS: usize = <K as BasedVectorSpace<F>>::DIMENSION;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("\u{03A0}_DEC.V: child count {got} does not match params.k_rho() {expected}")]
    ChildCount { expected: usize, got: usize },
    #[error("\u{03A0}_DEC.V: shape mismatch — {what} expected {expected}, got {got} (at idx {idx})")]
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
        idx: usize,
    },
}

// `Commitment` import kept for documentation linkage; not referenced directly
// in the lane-combine path.
#[allow(dead_code)]
fn _commitment_assoc(_c: &Commitment) {}
