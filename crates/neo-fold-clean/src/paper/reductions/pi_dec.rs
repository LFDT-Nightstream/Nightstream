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
use neo_math::F;
use thiserror::Error;

use crate::engine::optimized as engine;
use crate::paper::params::Params;
use crate::paper::relations::{superneo_inactive_x_zero, CeClaim, DecMixer, Structure};

#[derive(Debug, Error)]
pub enum Error {
    #[error("\u{03A0}_DEC: child count {got} does not match params.k_rho() {expected}")]
    ChildCount { expected: usize, got: usize },
    #[error("\u{03A0}_DEC: verifier rejected the children reconstruction")]
    VerifyRejected,
    #[error("\u{03A0}_DEC: inactive X columns must be zero in {0}")]
    InactiveX(&'static str),
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
    log: &AjtaiSModule,
    combine: DecMixer,
    parent: &CeClaim,
    parent_witness: &Mat<F>,
) -> Result<(Children, Proof), Error> {
    let (children, witnesses) = engine::prove_pi_dec(pp, s, log, parent, parent_witness, |cs, b| combine(cs, b))?;
    validate_child_count(pp, children.len())?;
    validate_inactive_x_zero(parent, &children)?;
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
    let ok = engine::verify_pi_dec(pp, s, parent, &proof.children, |cs, b| combine(cs, b));
    if !ok {
        return Err(Error::VerifyRejected);
    }
    validate_inactive_x_zero(parent, &proof.children)?;
    Ok(proof.children.clone())
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
