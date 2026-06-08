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
use neo_reductions::optimized_engine::OptimizedStructureCache;
use p3_field::PrimeField64;
use thiserror::Error;

use crate::engine::optimized as engine;
use crate::paper::params::Params;
use crate::paper::relations::{superneo_inactive_x_zero, superneo_public_x_cols, CeClaim, DecMixer, Structure};

#[derive(Debug, Error)]
pub enum Error {
    #[error("\u{03A0}_DEC: child count {got} does not match params.k_rho() {expected}")]
    ChildCount { expected: usize, got: usize },
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
    combine: DecMixer,
    parent: &CeClaim,
    parent_witness: &Mat<F>,
) -> Result<(Children, Proof), Error> {
    let (children, witnesses) =
        engine::prove_pi_dec(pp, s, cache, log, parent, parent_witness, |cs, b| combine(cs, b))?;
    validate_child_count(pp, children.len())?;
    validate_inactive_x_zero(parent, &children)?;
    validate_child_x_low_norm(pp, &children)?;
    Ok((
        Children {
            claims: children.clone(),
            witnesses,
        },
        Proof { children },
    ))
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
    let expected = s.m.next_power_of_two().max(2).trailing_zeros() as usize;
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
