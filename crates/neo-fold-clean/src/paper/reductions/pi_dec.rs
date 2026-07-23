//! Π_DEC — SuperNeo §7.5. Decomposition (norm reduction).
//!
//! Reduction:  CE(B, ℒ)   →   CE(b, ℒ)^k     where B = b^k
//!
//! Soundness: reduction of knowledge (Theorem 7, proof in §D.6). The verifier
//! has no random coins here; soundness comes from the verifier's
//! reconstruction checks.
//!
//! ## Read this file top-to-bottom
//!
//! - `prove` runs `split_b`, commits each child, and returns `(Children, Proof)`.
//! - `verify` uses `combine_b_pows` to re-derive parent commitments and y's
//!   from the children, and rejects on any mismatch.

use neo_ajtai::AjtaiSModule;
use neo_ccs::Mat;
use neo_math::balanced::within_nc_bound;
use neo_math::{D, F, K};
use neo_reductions::optimized_engine::{OptimizedStructureCache, PiDecProverPrecompute};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::optimized as engine;
use crate::paper::params::Params;
use crate::paper::relations::{
    recompose_adv, superneo_inactive_x_zero, superneo_public_x_cols, CeClaim, DecMixer, LaneScheme, Structure,
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("\u{03A0}_DEC: child count {got} does not match params.k_rho() {expected}")]
    ChildCount { expected: usize, got: usize },
    #[error("\u{03A0}_DEC: accelerator witness count {got} does not match child count {expected}")]
    AcceleratorWitnessCount { expected: usize, got: usize },
    #[error("\u{03A0}_DEC: verifier rejected the children reconstruction")]
    VerifyRejected,
    #[error("\u{03A0}_DEC: inactive X columns must be zero in {0}")]
    InactiveX(&'static str),
    #[error("\u{03A0}_DEC: child X active entries must lie in the CE(b) alphabet")]
    ChildXLowNorm,
    #[error("\u{03A0}_DEC: child fold_digest must equal parent fold_digest")]
    FoldDigest,
    #[error("\u{03A0}_DEC: noncanonical fold_digest byte limb in {owner} at lane {lane}")]
    FoldDigestCanonicality { owner: &'static str, lane: usize },
    #[error("\u{03A0}_DEC: r length must match the SplitNc row point in {0}")]
    RShape(&'static str),
    #[error("\u{03A0}_DEC: cached ct must equal the constant term of y_ring in {0}")]
    CtConsistency(&'static str),
    #[error("\u{03A0}_DEC: y_ring row count must match structure.t in {0}")]
    YRingShape(&'static str),
    #[error("\u{03A0}_DEC: child s_col must equal parent s_col")]
    SColConsistency,
    #[error(
        "\u{03A0}_DEC: adv presence must be all-or-nothing across parent and children ({present}/{total} present)"
    )]
    AdvPresence { present: usize, total: usize },
    #[error("\u{03A0}_DEC: children adv must recompose to the parent adv component-wise")]
    AdvRecomposition,
    #[error("\u{03A0}_DEC: adv-bearing parent requires a LaneScheme to commit child lane slices")]
    AdvLaneSchemeMissing,
    #[error("\u{03A0}_DEC: lane scheme rejected a child witness: {0}")]
    AdvLaneCommit(#[from] crate::paper::relations::LaneSchemeError),
    #[error("\u{03A0}_DEC: s_col length must match the SplitNc column point in {0}")]
    SColShape(&'static str),
    #[error("\u{03A0}_DEC: y_ring padding lanes must be zero in {0}")]
    YRingPadding(&'static str),
    #[error("\u{03A0}_DEC: unsupported sidecar field {field} in {owner}")]
    UnsupportedSidecar {
        owner: &'static str,
        field: &'static str,
    },
    #[error(transparent)]
    Engine(#[from] engine::Error),
}

/// Output of one Π_DEC step — k CE claims of norm b plus their k witness
/// matrices `Z_i = split_b(parent_witness)[i]`. Witnesses are prover-only;
/// the verifier sees only `Children::claims`.
#[derive(Clone, Debug)]
pub struct Children {
    pub claims: Vec<CeClaim>,
    pub witnesses: Vec<Mat<F>>,
}

/// Wire-format proof: just the children CE claims. The verifier reconstructs
/// `parent.c = Σ b^{i-1} c_i` and the y's from these and checks equality.
#[derive(Clone, Debug)]
pub struct Proof {
    pub children: Vec<CeClaim>,
}

// ──────────────────────────────────────────────────────────────────────────
// Prover (§7.5)
// ──────────────────────────────────────────────────────────────────────────

pub fn prove(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &AjtaiSModule,
    lanes: Option<&LaneScheme>,
    combine: DecMixer,
    parent: &CeClaim,
    parent_witness: &Mat<F>,
) -> Result<(Children, Proof), Error> {
    prove_inner(pp, s, cache, log, lanes, combine, parent, parent_witness, None)
}

pub(crate) fn prove_with_precompute(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &AjtaiSModule,
    lanes: Option<&LaneScheme>,
    combine: DecMixer,
    parent: &CeClaim,
    parent_witness: &Mat<F>,
    precompute: &PiDecProverPrecompute,
) -> Result<(Children, Proof), Error> {
    prove_inner(
        pp,
        s,
        cache,
        log,
        lanes,
        combine,
        parent,
        parent_witness,
        Some(precompute),
    )
}

/// Π_DEC prover from accelerator-produced split witnesses and commitments.
///
/// The caller must validate that the digit planes are low norm and recompose
/// to `parent_witness`. This function retains canonical child construction,
/// lane attachment, and every public consistency check.
#[allow(clippy::too_many_arguments)]
pub fn prove_from_split_material(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    lanes: Option<&LaneScheme>,
    child_adv: Option<Vec<neo_ccs::LaneCommitments<neo_ajtai::Commitment>>>,
    combine: DecMixer,
    parent: &CeClaim,
    z_split: Vec<Mat<F>>,
    digit_nonzero: Vec<bool>,
    child_commitments: Vec<neo_ajtai::Commitment>,
    precomputed_y_ring: Vec<Vec<[K; D]>>,
) -> Result<(Children, Proof), Error> {
    if precomputed_y_ring.len() != z_split.len() || precomputed_y_ring.iter().any(|rows| rows.len() != s.t()) {
        return Err(Error::YRingShape("accelerator output"));
    }
    if digit_nonzero.len() != z_split.len()
        || digit_nonzero
            .iter()
            .zip(&precomputed_y_ring)
            .any(|(&nonzero, rows)| !nonzero && rows.iter().flatten().any(|&value| value != K::ZERO))
    {
        return Err(Error::YRingPadding("accelerator output"));
    }
    let (mut children, witnesses) = engine::prove_pi_dec_from_split(
        pp,
        s,
        cache,
        parent,
        z_split,
        digit_nonzero,
        child_commitments,
        &precomputed_y_ring,
        None,
        |commitments, b| combine(commitments, b),
    )?;
    if let Some(child_adv) = child_adv {
        if child_adv.len() != children.len() {
            return Err(Error::AdvPresence {
                present: child_adv.len(),
                total: children.len(),
            });
        }
        for (child, adv) in children.iter_mut().zip(child_adv) {
            child.adv = Some(adv);
        }
    } else {
        attach_child_adv(lanes, parent, &mut children, &witnesses)?;
    }
    validate_child_count(pp, children.len())?;
    validate_inactive_x_zero(parent, &children)?;
    validate_child_x_low_norm(pp, &children)?;
    validate_adv_recomposition(pp, combine, parent, &children)?;
    Ok((
        Children {
            claims: children.clone(),
            witnesses,
        },
        Proof { children },
    ))
}

/// Π_DEC boundary for accelerator-owned child witnesses.
///
/// Complete public claims are verified canonically here while the backend
/// retains ownership of the private witness buffers.
#[doc(hidden)]
pub fn prove_from_accelerator_claims(
    pp: &Params,
    s: &Structure,
    combine: DecMixer,
    parent: &CeClaim,
    claims: Vec<CeClaim>,
    witnesses: Vec<Mat<F>>,
) -> Result<(Children, Proof), Error> {
    if witnesses.len() != claims.len() {
        return Err(Error::AcceleratorWitnessCount {
            expected: claims.len(),
            got: witnesses.len(),
        });
    }
    let proof = Proof { children: claims };
    let claims = verify(pp, s, combine, parent, &proof)?;
    Ok((Children { claims, witnesses }, proof))
}

#[allow(clippy::too_many_arguments)]
fn prove_inner(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &AjtaiSModule,
    lanes: Option<&LaneScheme>,
    combine: DecMixer,
    parent: &CeClaim,
    parent_witness: &Mat<F>,
    precompute: Option<&PiDecProverPrecompute>,
) -> Result<(Children, Proof), Error> {
    let (mut children, witnesses) =
        engine::prove_pi_dec(pp, s, cache, log, parent, parent_witness, precompute, |cs, b| {
            combine(cs, b)
        })?;
    attach_child_adv(lanes, parent, &mut children, &witnesses)?;
    validate_child_count(pp, children.len())?;
    validate_inactive_x_zero(parent, &children)?;
    validate_child_x_low_norm(pp, &children)?;
    validate_adv_recomposition(pp, combine, parent, &children)?;
    Ok((
        Children {
            claims: children.clone(),
            witnesses,
        },
        Proof { children },
    ))
}

/// Spec §5.2 R2 (Π_DEC prover side): an adv-bearing parent's children each
/// carry the lane commitments of their own digit witness — `adv_{i,L} =
/// A_L · Z_i[L]` — so the tuples recompose to the parent by the same
/// `b`-power linearity as `c`, and each child opens its slices at the
/// terminal decider (R3).
fn attach_child_adv(
    lanes: Option<&LaneScheme>,
    parent: &CeClaim,
    children: &mut [CeClaim],
    witnesses: &[Mat<F>],
) -> Result<(), Error> {
    if parent.adv.is_none() {
        return Ok(());
    }
    let Some(lanes) = lanes else {
        return Err(Error::AdvLaneSchemeMissing);
    };
    for (child, witness) in children.iter_mut().zip(witnesses.iter()) {
        child.adv = Some(lanes.commit(witness)?);
    }
    Ok(())
}

/// Spec §5.2 R2 (Π_DEC verifier side): the children's tuples must
/// recompose to the parent's, component-wise, under the same `Σ b^{i−1}`
/// combiner that reconstructs `parent.c` — pure public arithmetic, no
/// lane scheme needed. Presence is all-or-nothing; a plain parent with
/// plain children passes as `None == None`.
fn validate_adv_recomposition(
    pp: &Params,
    combine: DecMixer,
    parent: &CeClaim,
    children: &[CeClaim],
) -> Result<(), Error> {
    let advs: Vec<_> = children.iter().map(|child| child.adv.clone()).collect();
    let recomposed = recompose_adv(combine, pp.b(), &advs).map_err(|e| Error::AdvPresence {
        present: e.present,
        total: e.total,
    })?;
    if recomposed != parent.adv {
        return Err(Error::AdvRecomposition);
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Verifier (§7.5)
// ──────────────────────────────────────────────────────────────────────────

pub fn verify(
    pp: &Params,
    s: &Structure,
    combine: DecMixer,
    parent: &CeClaim,
    proof: &Proof,
) -> Result<Vec<CeClaim>, Error> {
    validate_child_count(pp, proof.children.len())?;
    validate_fold_digest_canonical("parent", parent)?;
    for child in &proof.children {
        validate_fold_digest_canonical("child", child)?;
    }
    validate_r_shape(s, parent, &proof.children)?;
    validate_y_ring_shape(s, parent, &proof.children)?;
    validate_inactive_x_zero(parent, &proof.children)?;
    validate_child_x_low_norm(pp, &proof.children)?;
    validate_supported_sidecars(parent, &proof.children)?;
    validate_s_col_shape(s, parent, &proof.children)?;
    validate_s_col_consistency(parent, &proof.children)?;
    validate_ct_consistency(parent, &proof.children)?;
    validate_y_ring_padding_zero(parent, &proof.children)?;
    validate_fold_digest_consistency(parent, &proof.children)?;
    validate_adv_recomposition(pp, combine, parent, &proof.children)?;
    let ok = engine::verify_pi_dec(pp, s, parent, &proof.children, |cs, b| combine(cs, b));
    if !ok {
        return Err(Error::VerifyRejected);
    }
    Ok(proof.children.clone())
}

fn validate_r_shape(s: &Structure, parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    validate_r_shape_one("parent", s, parent)?;
    for child in children {
        validate_r_shape_one("child", s, child)?;
    }
    Ok(())
}

fn validate_r_shape_one(owner: &'static str, s: &Structure, claim: &CeClaim) -> Result<(), Error> {
    let expected = s.n.next_power_of_two().max(2).trailing_zeros() as usize;
    if claim.r.len() != expected {
        return Err(Error::RShape(owner));
    }
    Ok(())
}

fn validate_child_count(pp: &Params, got: usize) -> Result<(), Error> {
    let expected = pp.k_rho() as usize;
    if got != expected {
        return Err(Error::ChildCount { expected, got });
    }
    Ok(())
}

/// Reject parent + children whose `X` has non-zero entries in columns
/// `[ceil(m_in / D), x.cols())`. Children become the next running
/// accumulator; without this, a terminal state could carry a non-canonical
/// accumulator that no downstream Π_CCS would re-validate. Mirrors the
/// circuit-side `pi_dec_circuit::enforce_inactive_x_zero`.
fn validate_inactive_x_zero(parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    if !superneo_inactive_x_zero(&parent.X, parent.m_in) {
        return Err(Error::InactiveX("parent"));
    }
    for child in children {
        if !superneo_inactive_x_zero(&child.X, child.m_in) {
            return Err(Error::InactiveX("child"));
        }
    }
    Ok(())
}

/// Π_DEC outputs CE(b) children. The public projection `X_i` is part of
/// each child CE claim, so its active packed entries must stay in the same
/// centered alphabet as a low-norm child witness. Recomposition alone would
/// allow canceling out-of-alphabet child `X` values.
fn validate_child_x_low_norm(pp: &Params, children: &[CeClaim]) -> Result<(), Error> {
    let b = pp.b();
    for child in children {
        let active_cols = superneo_public_x_cols(child.m_in);
        if active_cols > child.X.cols() {
            return Err(Error::ChildXLowNorm);
        }
        for r in 0..child.X.rows() {
            for c in 0..active_cols {
                if !within_nc_bound(child.X[(r, c)], b) {
                    return Err(Error::ChildXLowNorm);
                }
            }
        }
    }
    Ok(())
}

/// Π_DEC decomposes one parent CE claim into children; it must not let a
/// child introduce a fresh Π_CCS transcript digest. The native prover fills
/// each child from `parent.fold_digest`, and the circuit-side DEC verifier
/// enforces the same equality lane-by-lane.
fn validate_fold_digest_consistency(parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    for child in children {
        if child.fold_digest != parent.fold_digest {
            return Err(Error::FoldDigest);
        }
    }
    Ok(())
}

fn validate_fold_digest_canonical(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    for (lane, chunk) in claim.fold_digest.chunks_exact(8).enumerate() {
        let value = u64::from_le_bytes(chunk.try_into().expect("fold_digest lanes are 8 bytes"));
        if value >= F::ORDER_U64 {
            return Err(Error::FoldDigestCanonicality { owner, lane });
        }
    }
    Ok(())
}

fn validate_supported_sidecars(parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    validate_supported_sidecars_one("parent", parent)?;
    for child in children {
        validate_supported_sidecars_one("child", child)?;
    }
    Ok(())
}

fn validate_supported_sidecars_one(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    if !claim.aux_openings.is_empty() {
        return Err(Error::UnsupportedSidecar {
            owner,
            field: "aux_openings",
        });
    }
    if !claim.c_step_coords.is_empty() {
        return Err(Error::UnsupportedSidecar {
            owner,
            field: "c_step_coords",
        });
    }
    if claim.u_offset != 0 {
        return Err(Error::UnsupportedSidecar {
            owner,
            field: "u_offset",
        });
    }
    if claim.u_len != 0 {
        return Err(Error::UnsupportedSidecar { owner, field: "u_len" });
    }
    Ok(())
}

fn validate_s_col_shape(s: &Structure, parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    validate_s_col_shape_one("parent", s, parent)?;
    for child in children {
        validate_s_col_shape_one("child", s, child)?;
    }
    Ok(())
}

fn validate_s_col_shape_one(owner: &'static str, s: &Structure, claim: &CeClaim) -> Result<(), Error> {
    let expected = crate::paper::construction2::running::split_nc_column_point_len(s.m);
    if claim.s_col.len() != expected {
        return Err(Error::SColShape(owner));
    }
    Ok(())
}

fn validate_s_col_consistency(parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    for child in children {
        if child.s_col != parent.s_col {
            return Err(Error::SColConsistency);
        }
    }
    Ok(())
}

fn validate_ct_consistency(parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    validate_ct_consistency_one("parent", parent)?;
    for child in children {
        validate_ct_consistency_one("child", child)?;
    }
    Ok(())
}

fn validate_y_ring_shape(s: &Structure, parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    validate_y_ring_shape_one("parent", s, parent)?;
    for child in children {
        validate_y_ring_shape_one("child", s, child)?;
    }
    Ok(())
}

fn validate_y_ring_shape_one(owner: &'static str, s: &Structure, claim: &CeClaim) -> Result<(), Error> {
    if claim.y_ring.len() != s.t() {
        return Err(Error::YRingShape(owner));
    }
    Ok(())
}

fn validate_ct_consistency_one(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    if claim.ct.len() != claim.y_ring.len() {
        return Err(Error::CtConsistency(owner));
    }
    for (ct, row) in claim.ct.iter().zip(&claim.y_ring) {
        if row.first().copied().unwrap_or_default() != *ct {
            return Err(Error::CtConsistency(owner));
        }
    }
    Ok(())
}

fn validate_y_ring_padding_zero(parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    validate_y_ring_padding_zero_one("parent", parent)?;
    for child in children {
        validate_y_ring_padding_zero_one("child", child)?;
    }
    Ok(())
}

fn validate_y_ring_padding_zero_one(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    for row in &claim.y_ring {
        for &lane in row.iter().skip(D) {
            if lane != K::default() {
                return Err(Error::YRingPadding(owner));
            }
        }
    }
    Ok(())
}
