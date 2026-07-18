//! Π_DEC.V — in-circuit verifier (paper §7.5 step 2).
//!
//! Reduction:  CE(B, ℒ)   →   CE(b, ℒ)^k     where B = b^k
//!
//! The verifier has no random coins. Soundness comes from re-deriving the
//! parent from the children via the b-ary homomorphism and rejecting on
//! mismatch.
//!
//! Owns: parent/child allocation, strict shape validation, b-ary
//! recomposition, and equality of fields shared by definition.
//!
//! Does not own: transcript challenges, Π_RLC parent construction, or the
//! surrounding accumulator-authority link.
//!
//! Emits constraints: yes, direct recomposition/equality rows.
//!
//! Authority boundary: the parent is accepted only when every authoritative
//! coordinate is reconstructed from the supplied children; no child digest is
//! used as a substitute.
//!
//! | Constraint family | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | Allocation/shape | Parent and children have the fixed CE carrier shape | yes | this file | concrete refinement open |
//! | Commitment/X/y recomposition | `parent = sum_i b^i child_i` lane-wise | yes | this file | PiDEC semantics |
//! | Shared fields | Parent/children agree on non-decomposed fields | yes | this file | PiDEC semantics |
//! | `y_zcol` recomposition | `parent = sum_i b^i child_i` for the raw optimized projection | **not currently emitted** | allocation omits child sidecars | delayed parent-projection refinement open |
//! | Advice recomposition | Product-commitment advice follows the same radix map | yes | `product_commitment_circuit` | concrete refinement open |
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
//! - Equality constraints that enforce
//!   `parent.fold_digest == child_i.fold_digest`.
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

pub mod stage;

use neo_ajtai::Commitment;
use neo_math::ring::D;
use neo_math::{KExtensions, F, K};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::boolean;
use crate::engine::r1cs_circuit::builder::{
    CenteredUnitTrace, PiDecAdvAudit, PiDecClaimAudit, PiDecCommitmentAudit, PiDecStrictAudit,
};
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::paper::params::Params;
use crate::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;
use crate::paper::relations::product_commitment_circuit::{
    alloc_adv, enforce_adv_recomposition, validate_adv_shape, AdvCommitmentWires,
};
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
///
/// `aux_openings` / Pattern-A metadata are deliberately represented only
/// by shape counters. The clean SplitNc/NIFS circuit does not implement
/// those sidecar fields; strict DEC rejects them before relying on this
/// wire bundle.
#[derive(Clone, Debug)]
pub struct CeClaimWires {
    /// `d * kappa` columns, column-major (matches `Commitment::data`).
    pub c_data: Vec<Var>,
    /// Ajtai dimension `d` of the commitment.
    pub c_d: usize,
    pub c_d_var: Var,
    /// Ajtai dimension `kappa` of the commitment.
    pub c_kappa: usize,
    pub c_kappa_var: Var,
    /// Nebula coordinates of the same product commitment as `c_data`.
    pub adv: Option<AdvCommitmentWires>,
    /// `rows * cols` columns, row-major.
    pub x: Vec<Var>,
    pub x_rows: usize,
    pub x_rows_var: Var,
    pub x_cols: usize,
    pub x_cols_var: Var,
    /// Public input length the claim was constructed under.
    pub m_in: usize,
    pub m_in_var: Var,
    /// Unsupported in the clean SplitNc/NIFS circuit. Must be zero.
    pub aux_openings_len: usize,
    /// Unsupported Pattern-A metadata. Must be zero.
    pub c_step_coords_len: usize,
    pub u_offset: usize,
    pub u_len: usize,
    /// `t` outer × `d` lanes × `s` base-field columns per K-element.
    /// Index as `y_ring[j][lane * s + limb]`.
    pub y_ring: Vec<Vec<Var>>,
    pub y_ring_lanes: usize,
    /// SuperNeo scalar/constant-term view of `y_ring`. Per Theorem 5 of
    /// the SuperNeo paper, `ct(y_j) = M̄_j z(r)` — the constant term of
    /// the K-valued ring evaluation equals the field-level multilinear
    /// eval. In flat-limb form, `ct[j] == (y_ring[j][0], y_ring[j][1])`
    /// (the lane-0 K-element of `y_ring[j]`). One `KVar` per CCS matrix.
    /// Bound to the y_ring layout by
    /// `paper::decider_ce_relation::evaluation::enforce_ct_from_y_ring`.
    pub ct: Vec<KVar>,
    /// CE evaluation point `r ∈ K^{log n}`. Shared between parent and all
    /// children in a Π_DEC.V step; enforced via [`enforce_r_consistency`].
    pub r: Vec<KVar>,
    /// NC column-domain point `s_col ∈ K^{log m}`. Shared between parent
    /// and all children (NC channel doesn't decompose `s_col`); enforced
    /// via [`enforce_s_col_consistency`].
    pub s_col: Vec<KVar>,
    /// Flattened `y_zcol` sidecar, indexed as
    /// `y_zcol[lane * s + limb]`. The authoritative Π_RLC parent retains its
    /// full fixed-width sidecar. Ordinary Π_DEC children leave this vector
    /// empty because strict Π_DEC neither reads nor validates child `y_zcol`.
    /// Terminal-decider children reattach the sidecar at the terminal CE
    /// relation, where it is checked directly against the opened witness.
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
/// their wire handles. Allocation also emits fail-closed carrier rows: one
/// inactive-X zero sentinel and five fixed metadata pins per claim, plus a
/// rejection row when a fold-digest limb is noncanonical.
///
/// Callers must call [`enforce_dec_v`] (and optionally [`enforce_x_bitness`])
/// to actually constrain the relationship.
pub fn alloc_dec_inputs(builder: &mut R1csBuilder, parent: &CeClaim, children: &[CeClaim]) -> DecInputWires {
    let parent_wires = alloc_ce_claim(builder, parent);
    let child_wires = children
        .iter()
        .map(|c| alloc_dec_child_claim(builder, c))
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
    let row_start = builder.rows();
    enforce_dec_v_inner(builder, pp, wires)?;
    builder.record_row_family(stage::RECOMPOSITION, row_start);
    Ok(())
}

fn enforce_dec_v_inner(builder: &mut R1csBuilder, pp: &Params, wires: &DecInputWires) -> Result<(), Error> {
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

    let family_start = builder.rows();
    enforce_lane_combination(
        builder,
        &wires.parent.c_data,
        &wires.children,
        &b_pows,
        ChildField::Commitment,
    );
    builder.record_row_family(stage::RECOMPOSITION_COMMITMENT, family_start);

    let family_start = builder.rows();
    let child_adv: Vec<Option<AdvCommitmentWires>> = wires
        .children
        .iter()
        .map(|child| child.adv.clone())
        .collect();
    enforce_adv_recomposition(builder, wires.parent.adv.as_ref(), &child_adv, &b_pows)
        .map_err(Error::ProductCommitment)?;
    builder.record_row_family(stage::RECOMPOSITION_ADVICE, family_start);

    let family_start = builder.rows();
    enforce_active_x_combination(builder, &wires.parent, &wires.children, &b_pows);
    builder.record_row_family(stage::RECOMPOSITION_X, family_start);

    let family_start = builder.rows();
    for j in 0..wires.parent.y_ring.len() {
        enforce_lane_combination_y(builder, j, &wires.parent.y_ring[j], &wires.children, &b_pows);
    }
    builder.record_row_family(stage::RECOMPOSITION_Y_RING, family_start);
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

/// Enforce that each active packed child `X` entry lies in the centered
/// CE(b) alphabet `{-(b-1), ..., +(b-1)}`. Π_DEC outputs CE(b) children;
/// b-ary recomposition alone would allow out-of-alphabet child public
/// projections that cancel in the parent.
pub fn enforce_child_x_balanced_alphabet(
    builder: &mut R1csBuilder,
    pp: &Params,
    wires: &DecInputWires,
) -> Result<(), Error> {
    let b = pp.b();
    if b < 2 {
        return Err(Error::ShapeMismatch {
            what: "child X alphabet bound",
            expected: 2,
            got: b as usize,
            idx: 0,
        });
    }
    for (idx, child) in wires.children.iter().enumerate() {
        let active_cols = crate::paper::relations::superneo_public_x_cols(child.m_in);
        if active_cols > child.x_cols {
            return Err(Error::ShapeMismatch {
                what: "child active X columns",
                expected: child.x_cols,
                got: active_cols,
                idx,
            });
        }
        for r in 0..child.x_rows {
            for c in 0..active_cols {
                enforce_centered_alphabet(builder, child.x[r * child.x_cols + c], b);
            }
        }
    }
    Ok(())
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
///   4. [`enforce_child_x_balanced_alphabet`] — child `X` active entries
///      remain in the centered CE(b) alphabet.
///   5. [`enforce_ct_consistency`] — every claim's cached `ct[j]` is the
///      lane-0 K-element of `y_ring[j]`.
///   6. `y_ring[D..] == 0` — SplitNc's padded CE representation is canonical.
///   7. [`enforce_fold_digest_consistency`] — children carry the same
///      transcript digest as their Π_DEC parent.
///
/// Notably absent (and absent on the native side):
///   - **No `y_zcol` b-ary check.** The optimized NC table stores raw packed
///     witness coordinates, so its projection is linear and does telescope
///     through Π_RLC and Π_DEC. Omitting the check is a known authority gap,
///     not a semantic impossibility. The planned delayed-parent refinement
///     must bind this projection before the old point is discarded.
///   - **No unsigned x bitness check.** `decompose_balanced_fixed_d_digits_k`
///     produces signed digits (e.g. -1 ↦ p-1 in F), so an unsigned
///     `{0..b-1}` check would reject honest provers. Strict mode enforces the
///     centered CE(b) alphabet instead. [`enforce_x_bitness`] remains
///     available for callers that have an unsigned range invariant to enforce.
pub fn enforce_dec_v_strict(builder: &mut R1csBuilder, pp: &Params, wires: &DecInputWires) -> Result<(), Error> {
    let row_start = builder.rows();
    let first_allocated_column = builder.cols();

    let phase_start = builder.rows();
    enforce_dec_v_inner(builder, pp, wires)?;
    builder.record_row_family(stage::RECOMPOSITION, phase_start);

    let phase_start = builder.rows();
    enforce_shape_metadata_consistency(builder, wires);
    builder.record_row_family(stage::SHAPE, phase_start);

    let phase_start = builder.rows();
    enforce_r_consistency(builder, wires)?;
    builder.record_row_family(stage::R, phase_start);

    let phase_start = builder.rows();
    enforce_s_col_consistency(builder, wires)?;
    builder.record_row_family(stage::S_COL, phase_start);

    let phase_start = builder.rows();
    enforce_inactive_x_zero(builder, wires)?;
    builder.record_row_family(stage::INACTIVE_X, phase_start);

    let phase_start = builder.rows();
    enforce_child_x_balanced_alphabet(builder, pp, wires)?;
    builder.record_row_family(stage::ALPHABET, phase_start);

    let phase_start = builder.rows();
    enforce_ct_consistency(builder, wires)?;
    builder.record_row_family(stage::CT, phase_start);

    let phase_start = builder.rows();
    enforce_y_ring_padding_zero(builder, wires);
    builder.record_row_family(stage::Y_RING_PADDING, phase_start);

    let phase_start = builder.rows();
    enforce_fold_digest_consistency(builder, wires)?;
    builder.record_row_family(stage::FOLD_DIGEST, phase_start);
    builder.record_row_family(stage::VERIFY, row_start);

    builder.record_pi_dec_strict(PiDecStrictAudit {
        row_start,
        row_end: builder.rows(),
        first_allocated_column,
        radix: pp.b(),
        parent: pi_dec_claim_audit(&wires.parent),
        children: wires.children.iter().map(pi_dec_claim_audit).collect(),
    });
    Ok(())
}

fn commitment_audit(wires: &AdvCommitmentWires) -> PiDecAdvAudit {
    let coordinate =
        |commitment: &crate::paper::relations::product_commitment_circuit::CommitmentWires| PiDecCommitmentAudit {
            d_col: commitment.d_var.col(),
            kappa_col: commitment.kappa_var.col(),
            data_cols: commitment.data.iter().map(|wire| wire.col()).collect(),
        };
    PiDecAdvAudit {
        ops: coordinate(&wires.ops),
        is: coordinate(&wires.is),
        fs: coordinate(&wires.fs),
    }
}

fn pi_dec_claim_audit(wires: &CeClaimWires) -> PiDecClaimAudit {
    PiDecClaimAudit {
        commitment: PiDecCommitmentAudit {
            d_col: wires.c_d_var.col(),
            kappa_col: wires.c_kappa_var.col(),
            data_cols: wires.c_data.iter().map(|wire| wire.col()).collect(),
        },
        adv: wires.adv.as_ref().map(commitment_audit),
        x_cols: wires.x.iter().map(|wire| wire.col()).collect(),
        x_rows: wires.x_rows,
        x_width: wires.x_cols,
        x_rows_col: wires.x_rows_var.col(),
        x_width_col: wires.x_cols_var.col(),
        m_in: wires.m_in,
        m_in_col: wires.m_in_var.col(),
        y_ring_cols: wires
            .y_ring
            .iter()
            .map(|row| row.iter().map(|wire| wire.col()).collect())
            .collect(),
        ct_cols: wires
            .ct
            .iter()
            .map(|wire| [wire.c0.col(), wire.c1.col()])
            .collect(),
        r_cols: wires
            .r
            .iter()
            .map(|wire| [wire.c0.col(), wire.c1.col()])
            .collect(),
        s_col_cols: wires
            .s_col
            .iter()
            .map(|wire| [wire.c0.col(), wire.c1.col()])
            .collect(),
        fold_digest_cols: wires.fold_digest_fields.map(Var::col),
    }
}

/// Enforce parent/child equality for non-wire CE shape metadata as rows.
/// `check_shapes` still fail-closes malformed vector lengths before any
/// indexing; this helper prevents scalar metadata (`c.d`, `c.kappa`,
/// `X.rows`, `X.cols`, `m_in`) from becoming an unconstrained side channel
/// when the malformed values are large enough to synthesize a circuit.
pub fn enforce_shape_metadata_consistency(builder: &mut R1csBuilder, wires: &DecInputWires) {
    for child in &wires.children {
        enforce_var_eq(builder, wires.parent.c_d_var, child.c_d_var);
        enforce_var_eq(builder, wires.parent.c_kappa_var, child.c_kappa_var);
        enforce_var_eq(builder, wires.parent.x_rows_var, child.x_rows_var);
        enforce_var_eq(builder, wires.parent.x_cols_var, child.x_cols_var);
        enforce_var_eq(builder, wires.parent.m_in_var, child.m_in_var);
    }
}

/// Enforce the SuperNeo implementation invariant `ct[j] == y_ring[j][0]`
/// for the parent and every child. The paper CE instance carries
/// `y_ring`; `ct` is a cached scalar/constant-term view used by later
/// verifier equations. Keeping it derived here prevents it from becoming
/// a shadow authoritative field.
pub fn enforce_ct_consistency(builder: &mut R1csBuilder, wires: &DecInputWires) -> Result<(), Error> {
    enforce_ct_consistency_one(builder, &wires.parent, 0)?;
    for (idx, child) in wires.children.iter().enumerate() {
        enforce_ct_consistency_one(builder, child, idx)?;
    }
    Ok(())
}

fn enforce_ct_consistency_one(builder: &mut R1csBuilder, claim: &CeClaimWires, idx: usize) -> Result<(), Error> {
    if claim.ct.len() != claim.y_ring.len() {
        return Err(Error::ShapeMismatch {
            what: "ct length",
            expected: claim.y_ring.len(),
            got: claim.ct.len(),
            idx,
        });
    }
    for (j, (ct, row)) in claim.ct.iter().zip(claim.y_ring.iter()).enumerate() {
        if row.len() < K_LIMBS {
            return Err(Error::ShapeMismatch {
                what: "y_ring[j] constant-term limbs",
                expected: K_LIMBS,
                got: row.len(),
                idx: j,
            });
        }
        builder.enforce_eq(&Lc::from_var(ct.c0), &Lc::from_var(row[0]));
        builder.enforce_eq(&Lc::from_var(ct.c1), &Lc::from_var(row[1]));
    }
    Ok(())
}

/// Enforce that Π_DEC does not mint fresh transcript authorities. The
/// children are a b-ary decomposition of the parent CE claim; their
/// `fold_digest` is the Π_CCS transcript digest carried through the CE
/// claim and must remain equal to the parent's digest.
pub fn enforce_fold_digest_consistency(builder: &mut R1csBuilder, wires: &DecInputWires) -> Result<(), Error> {
    for child in &wires.children {
        for lane in 0..wires.parent.fold_digest_fields.len() {
            builder.enforce_eq(
                &Lc::from_var(child.fold_digest_fields[lane]),
                &Lc::from_var(wires.parent.fold_digest_fields[lane]),
            );
        }
    }
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
    enforce_unique_zero_wires(
        builder,
        (0..claim.x_rows).flat_map(|r| (active_cols..claim.x_cols).map(move |c| claim.x[r * claim.x_cols + c])),
    );
    Ok(())
}

/// Validate that parent + every child have exactly `t` SplitNc-shaped
/// `y_ring` rows and that the Π_RLC parent has `y_zcol` of exactly
/// `d_pad = 2^ell_d` K-element lanes.
///
/// `enforce_dec_v`'s generic [`check_shapes`] only enforces parent ↔ child
/// length parity — that's enough for the b-ary recomposition algebra but
/// doesn't catch the case where both sides carry extra rows or silently
/// un-padded rows. The SplitNc verifier path requires the structure-owned
/// matrix count plus Ajtai-padded lane shape; NIFS.V layers this check on
/// top of [`enforce_dec_v_strict`] before consuming the wires.
pub fn enforce_split_nc_d_pad_shape(wires: &DecInputWires, t: usize, d_pad: usize) -> Result<(), Error> {
    let label = |what: &'static str, got: usize, idx: usize| Error::ShapeMismatch {
        what,
        expected: d_pad,
        got,
        idx,
    };
    if wires.parent.y_ring.len() != t {
        return Err(Error::ShapeMismatch {
            what: "parent.y_ring outer length",
            expected: t,
            got: wires.parent.y_ring.len(),
            idx: 0,
        });
    }
    for (j, row) in wires.parent.y_ring.iter().enumerate() {
        if wires.parent.y_ring_lanes != d_pad || row.len() != d_pad * K_LIMBS {
            return Err(label("parent.y_ring[j] lane count", wires.parent.y_ring_lanes, j));
        }
    }
    if wires.parent.y_zcol_lanes != d_pad || wires.parent.y_zcol.len() != d_pad * K_LIMBS {
        return Err(label("parent.y_zcol lane count", wires.parent.y_zcol_lanes, 0));
    }
    for (idx, child) in wires.children.iter().enumerate() {
        if child.y_ring.len() != t {
            return Err(Error::ShapeMismatch {
                what: "child.y_ring outer length",
                expected: t,
                got: child.y_ring.len(),
                idx,
            });
        }
        for row in child.y_ring.iter() {
            if child.y_ring_lanes != d_pad || row.len() != d_pad * K_LIMBS {
                return Err(label("child.y_ring[j] lane count", child.y_ring_lanes, idx));
            }
        }
    }
    Ok(())
}

/// Enforce canonical SplitNc `y_ring` padding: lanes `D..` must be zero.
///
/// Native Π_CCS / terminal CE construction computes `y_ring` as the real
/// `D` ring coefficients padded up to `d_pad = 2^ell_d` with zeros. Π_DEC's
/// b-ary recomposition alone would allow children to carry nonzero padding
/// lanes that cancel in the parent. Since children are the next running
/// accumulator, they must be canonical outputs, not just recomposition-valid.
fn enforce_y_ring_padding_zero(builder: &mut R1csBuilder, wires: &DecInputWires) {
    enforce_y_ring_padding_zero_one(builder, &wires.parent);
    for child in &wires.children {
        enforce_y_ring_padding_zero_one(builder, child);
    }
}

fn enforce_y_ring_padding_zero_one(builder: &mut R1csBuilder, claim: &CeClaimWires) {
    for row in &claim.y_ring {
        for limb in row.iter().skip(D * K_LIMBS) {
            builder.enforce_eq(&Lc::from_var(*limb), &Lc::zero());
        }
    }
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
            };
            combo.add_term(child_var, b_pows[idx]);
        }
        builder.enforce_eq(&Lc::from_var(parent_lanes[lane]), &combo);
    }
}

fn enforce_active_x_combination(
    builder: &mut R1csBuilder,
    parent: &CeClaimWires,
    children: &[CeClaimWires],
    b_pows: &[F],
) {
    let active_cols = crate::paper::relations::superneo_public_x_cols(parent.m_in);
    for row in 0..parent.x_rows {
        for col in 0..active_cols {
            let lane = row * parent.x_cols + col;
            let mut combo = Lc::zero();
            for (child, coeff) in children.iter().zip(b_pows.iter().copied()) {
                combo.add_term(child.x[lane], coeff);
            }
            builder.enforce_eq(&Lc::from_var(parent.x[lane]), &combo);
        }
    }
}

fn enforce_unique_zero_wires(builder: &mut R1csBuilder, wires: impl Iterator<Item = Var>) {
    let mut constrained = std::collections::HashSet::new();
    for wire in wires {
        if constrained.insert(wire.col()) {
            builder.enforce_eq(&Lc::from_var(wire), &Lc::zero());
        }
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

pub(crate) fn alloc_ce_claim(builder: &mut R1csBuilder, claim: &CeClaim) -> CeClaimWires {
    alloc_ce_claim_from_y_zcol(builder, claim, &claim.y_zcol)
}

/// Allocate the strict-Π_DEC child core. Child `y_zcol` is deliberately not
/// materialized because it is outside the Π_DEC acceptance predicate.
pub(crate) fn alloc_dec_child_claim(builder: &mut R1csBuilder, claim: &CeClaim) -> CeClaimWires {
    alloc_ce_claim_from_y_zcol(builder, claim, &[])
}

fn alloc_ce_claim_from_y_zcol(builder: &mut R1csBuilder, claim: &CeClaim, y_zcol_values: &[K]) -> CeClaimWires {
    let c_data = builder.alloc_vec(&claim.c.data);
    let adv = alloc_adv(builder, claim.adv.as_ref());
    let x_rows = claim.X.rows();
    let x_cols = claim.X.cols();
    let mut x = Vec::with_capacity(x_rows * x_cols);
    let active_cols = crate::paper::relations::superneo_public_x_cols(claim.m_in);
    let inactive_nonzero = (0..x_rows).any(|r| (active_cols..x_cols).any(|c| claim.X[(r, c)] != F::ZERO));
    let allocation_start = builder.rows();
    let inactive_zero = builder.alloc(if inactive_nonzero { F::ONE } else { F::ZERO });
    builder.enforce_eq(&Lc::from_var(inactive_zero), &Lc::zero());
    builder.record_row_family(pi_rlc_stage::ROW_SHAPE_ALLOCATE_INACTIVE_X_SENTINEL, allocation_start);
    for r in 0..x_rows {
        for c in 0..x_cols {
            x.push(if c < active_cols {
                builder.alloc(claim.X[(r, c)])
            } else {
                inactive_zero
            });
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
    // `ct[j]` is the SuperNeo scalar/constant-term view of `y_ring[j]`.
    // Native value (= `ct_from_y_digits(y_ring[j])` = `y_ring[j][0]`)
    // is filled here; the wire-equality binding `ct[j] == y_ring[j][lane=0]`
    // is enforced by
    // `paper::decider_ce_relation::evaluation::enforce_ct_from_y_ring`.
    let ct = claim
        .ct
        .iter()
        .map(|k| {
            let [c0, c1] = k.as_coeffs();
            KVar::alloc(builder, c0, c1)
        })
        .collect();
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
    let y_zcol_lanes = y_zcol_values.len();
    let mut y_zcol = Vec::with_capacity(y_zcol_lanes * K_LIMBS);
    for elem in y_zcol_values {
        for limb in elem.as_basis_coefficients_slice() {
            y_zcol.push(builder.alloc(*limb));
        }
    }
    // Allocate the CE claim's fold_digest as four canonical base-field wires
    // so downstream gadgets (decider CE-continuity gate) can pin it equal to
    // the next step's running's `fold_digest_fields`. Reject noncanonical
    // byte limbs instead of reducing them modulo F; proof bytes should not be
    // able to alias a different transcript digest in-circuit.
    let allocation_start = builder.rows();
    let fold_digest_lanes = canonical_digest32_fields_or_unsat(builder, claim.fold_digest);
    builder.record_row_family(
        pi_rlc_stage::ROW_SHAPE_ALLOCATE_FOLD_DIGEST_CANONICALITY,
        allocation_start,
    );
    let fold_digest_fields: [Var; 4] = [
        builder.alloc(fold_digest_lanes[0]),
        builder.alloc(fold_digest_lanes[1]),
        builder.alloc(fold_digest_lanes[2]),
        builder.alloc(fold_digest_lanes[3]),
    ];
    let allocation_start = builder.rows();
    let c_d_var = alloc_usize(builder, claim.c.d);
    let c_kappa_var = alloc_usize(builder, claim.c.kappa);
    let x_rows_var = alloc_usize(builder, x_rows);
    let x_cols_var = alloc_usize(builder, x_cols);
    let m_in_var = alloc_usize(builder, claim.m_in);
    builder.record_row_family(pi_rlc_stage::ROW_SHAPE_ALLOCATE_METADATA, allocation_start);

    CeClaimWires {
        c_data,
        c_d: claim.c.d,
        c_d_var,
        c_kappa: claim.c.kappa,
        c_kappa_var,
        adv,
        x,
        x_rows,
        x_rows_var,
        x_cols,
        x_cols_var,
        m_in: claim.m_in,
        m_in_var,
        aux_openings_len: claim.aux_openings.len(),
        c_step_coords_len: claim.c_step_coords.len(),
        u_offset: claim.u_offset,
        u_len: claim.u_len,
        y_ring,
        y_ring_lanes,
        ct,
        r,
        s_col,
        y_zcol,
        y_zcol_lanes,
        fold_digest_fields,
    }
}

fn canonical_digest32_fields_or_unsat(builder: &mut R1csBuilder, bytes: [u8; 32]) -> [F; 4] {
    let mut fields = [F::ZERO; 4];
    for (lane, out) in fields.iter_mut().enumerate() {
        let start = lane * 8;
        let value = u64::from_le_bytes(
            bytes[start..start + 8]
                .try_into()
                .expect("8-byte digest limb"),
        );
        if value >= F::ORDER_U64 {
            builder.enforce_eq(&Lc::zero(), &Lc::from_const(F::ONE));
            return [F::ZERO; 4];
        }
        *out = F::from_u64(value);
    }
    fields
}

fn alloc_usize(builder: &mut R1csBuilder, value: usize) -> Var {
    let v = builder.alloc(F::from_u64(value as u64));
    builder.enforce_eq(&Lc::from_var(v), &Lc::from_const(F::from_u64(value as u64)));
    v
}

fn enforce_var_eq(builder: &mut R1csBuilder, a: Var, b: Var) {
    builder.enforce_eq(&Lc::from_var(a), &Lc::from_var(b));
}

fn enforce_centered_alphabet(builder: &mut R1csBuilder, v: Var, b: u32) {
    debug_assert!(b >= 2, "caller gates b >= 2");
    let row_start = builder.rows();
    let column_start = builder.cols();
    let bound = b as i64 - 1;
    let alphabet: Vec<i64> = (-bound..=bound).collect();
    let mut acc: Option<Lc> = None;
    let total = alphabet.len();
    for (i, a) in alphabet.iter().enumerate() {
        let mut factor = Lc::from_var(v);
        let neg_a = if *a >= 0 {
            -F::from_u64(*a as u64)
        } else {
            F::from_u64((-*a) as u64)
        };
        factor.add_constant(neg_a);
        match acc.take() {
            None => acc = Some(factor),
            Some(prev) => {
                if i + 1 == total {
                    builder.enforce(&prev, &factor, &Lc::zero());
                    if b == 2 {
                        builder.record_centered_unit_trace(CenteredUnitTrace {
                            row_start,
                            row_end: builder.rows(),
                            allocated_columns: (column_start..builder.cols()).collect(),
                            value_col: v.col(),
                        });
                    }
                    return;
                }
                let next = builder.alloc_mul(&prev, &factor);
                acc = Some(Lc::from_var(next));
            }
        }
    }
}

fn check_shapes(parent: &CeClaimWires, children: &[CeClaimWires]) -> Result<(), Error> {
    reject_unsupported_sidecar_fields(parent, 0)?;
    validate_adv_shape(parent.adv.as_ref(), parent.c_d, parent.c_kappa, "parent").map_err(Error::ProductCommitment)?;
    if parent.c_d != D {
        return Err(Error::ShapeMismatch {
            what: "parent commitment d",
            expected: D,
            got: parent.c_d,
            idx: 0,
        });
    }
    let parent_c_len = D * parent.c_kappa;
    if parent.c_data.len() != parent_c_len {
        return Err(Error::ShapeMismatch {
            what: "parent commitment lane count",
            expected: parent_c_len,
            got: parent.c_data.len(),
            idx: 0,
        });
    }
    if parent.x_rows != D {
        return Err(Error::ShapeMismatch {
            what: "parent X rows",
            expected: D,
            got: parent.x_rows,
            idx: 0,
        });
    }
    if parent.x_cols != parent.m_in {
        return Err(Error::ShapeMismatch {
            what: "parent X cols vs m_in",
            expected: parent.m_in,
            got: parent.x_cols,
            idx: 0,
        });
    }
    for (idx, child) in children.iter().enumerate() {
        reject_unsupported_sidecar_fields(child, idx)?;
        validate_adv_shape(child.adv.as_ref(), child.c_d, child.c_kappa, &format!("child[{idx}]"))
            .map_err(Error::ProductCommitment)?;
        if child.c_d != parent.c_d {
            return Err(Error::ShapeMismatch {
                what: "child commitment d",
                expected: parent.c_d,
                got: child.c_d,
                idx,
            });
        }
        if child.c_kappa != parent.c_kappa {
            return Err(Error::ShapeMismatch {
                what: "child commitment kappa",
                expected: parent.c_kappa,
                got: child.c_kappa,
                idx,
            });
        }
        let child_c_len = D * child.c_kappa;
        if child.c_data.len() != child_c_len {
            return Err(Error::ShapeMismatch {
                what: "child commitment lane count",
                expected: child_c_len,
                got: child.c_data.len(),
                idx,
            });
        }
        if child.m_in != parent.m_in {
            return Err(Error::ShapeMismatch {
                what: "child m_in",
                expected: parent.m_in,
                got: child.m_in,
                idx,
            });
        }
        if child.x_rows != D {
            return Err(Error::ShapeMismatch {
                what: "child X rows",
                expected: D,
                got: child.x_rows,
                idx,
            });
        }
        if child.x_cols != child.m_in {
            return Err(Error::ShapeMismatch {
                what: "child X cols vs m_in",
                expected: child.m_in,
                got: child.x_cols,
                idx,
            });
        }
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
    }
    Ok(())
}

fn reject_unsupported_sidecar_fields(claim: &CeClaimWires, idx: usize) -> Result<(), Error> {
    if claim.aux_openings_len != 0 {
        return Err(Error::ShapeMismatch {
            what: "aux_openings",
            expected: 0,
            got: claim.aux_openings_len,
            idx,
        });
    }
    if claim.c_step_coords_len != 0 {
        return Err(Error::ShapeMismatch {
            what: "c_step_coords",
            expected: 0,
            got: claim.c_step_coords_len,
            idx,
        });
    }
    if claim.u_offset != 0 {
        return Err(Error::ShapeMismatch {
            what: "u_offset",
            expected: 0,
            got: claim.u_offset,
            idx,
        });
    }
    if claim.u_len != 0 {
        return Err(Error::ShapeMismatch {
            what: "u_len",
            expected: 0,
            got: claim.u_len,
            idx,
        });
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
    #[error("Pi_DEC.V: invalid product commitment: {0}")]
    ProductCommitment(String),
}

// `Commitment` import kept for documentation linkage; not referenced directly
// in the lane-combine path.
#[allow(dead_code)]
fn _commitment_assoc(_c: &Commitment) {}
