//! Commitment-action helpers for Π_RLC and Π_DEC.
//!
//! These are not paper Definitions — they're API contracts for "how does
//! the commitment scheme combine commitments under a homomorphic action."
//! The lifecycle verifier fixes the canonical Ajtai action at preprocess
//! time; low-level reduction tests may still pass the function pointers
//! explicitly to exercise Π_RLC / Π_DEC in isolation.

use neo_ajtai::{s_mul_add, scale_commitment_add_inplace, Commitment};
use neo_ccs::Mat;
use neo_math::ring::{cf_inv, Rq};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

/// Π_RLC commitment mixer: `Σρ_i c_i` over the K+k input commitments.
pub type RlcMixer = fn(&[Mat<F>], &[Commitment]) -> Commitment;

/// Π_DEC commitment combiner: `Σ b^{i-1} c_i` over the k child commitments.
pub type DecMixer = fn(&[Commitment], u32) -> Commitment;

/// `Σ ρ_i · c_i` under the Ajtai S-module action. ρ_i is the polynomial
/// recovered from the rotation matrix's first column.
pub fn ajtai_rlc_mixer(rhos: &[Mat<F>], commits: &[Commitment]) -> Commitment {
    debug_assert!(!commits.is_empty(), "commit_mix: empty commitments");
    debug_assert_eq!(rhos.len(), commits.len(), "commit_mix: |rhos| != |commits|");
    let mut acc = Commitment::zeros(commits[0].d, commits[0].kappa);
    for (rho, c) in rhos.iter().zip(commits.iter()) {
        let rq = rot_matrix_to_rq(rho);
        s_mul_add(&mut acc, &rq, c);
    }
    acc
}

/// `Σ b^{i-1} · c_i` for `i = 1..=k` (k = `commits.len()`).
pub fn ajtai_dec_mixer(commits: &[Commitment], b: u32) -> Commitment {
    debug_assert!(!commits.is_empty(), "dec_mix: empty commitments");
    let mut acc = Commitment::zeros(commits[0].d, commits[0].kappa);
    let base = F::from_u64(b as u64);
    let mut pow = F::ONE;
    for c in commits {
        scale_commitment_add_inplace(&mut acc, pow, c);
        pow *= base;
    }
    acc
}

/// Recover the polynomial a rotation matrix represents — its first column
/// is the polynomial's coefficient vector — via the inverse coefficient map.
fn rot_matrix_to_rq(mat: &Mat<F>) -> Rq {
    let mut coeffs = [F::default(); D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

// ── Nebula `adv` tuple mirrors (spec §5.2 R2) ──────────────────────────────
//
// The tuple is folded by the *same* public arithmetic as `c` — these
// helpers apply the caller's mixer component-wise and own the presence
// rule: a fold is either entirely adv-bearing or entirely not. A mixed
// batch has no well-defined combined tuple and is rejected before any
// arithmetic (the §5.1 all-or-nothing shape, lifted from one claim to a
// fold's input set).

use neo_ccs::LaneCommitments;

/// A fold input set that mixes adv-bearing and plain claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdvPresenceError {
    /// How many of the inputs carried a tuple.
    pub present: usize,
    /// Total inputs in the fold.
    pub total: usize,
}

/// Π_RLC mirror: `adv*_L = Σ ρ_i · adv_{i,L}` per lane, under the same
/// mixer that combines `c`. Returns `None` when no input carries a tuple.
pub fn mix_adv(
    mix: RlcMixer,
    rho_mats: &[Mat<F>],
    inputs: &[Option<LaneCommitments<Commitment>>],
) -> Result<Option<LaneCommitments<Commitment>>, AdvPresenceError> {
    let advs = require_homogeneous(inputs)?;
    let Some(advs) = advs else { return Ok(None) };
    let component = |pick: fn(&LaneCommitments<Commitment>) -> &Commitment| {
        let cs: Vec<Commitment> = advs.iter().map(|adv| pick(adv).clone()).collect();
        mix(rho_mats, &cs)
    };
    Ok(Some(LaneCommitments {
        ops: component(|adv| &adv.ops),
        is: component(|adv| &adv.is),
        fs: component(|adv| &adv.fs),
    }))
}

/// Π_DEC mirror: `adv*_L = Σ b^{i−1} · adv_{i,L}` per lane, under the same
/// combiner that reconstructs `parent.c`. Returns `None` when no child
/// carries a tuple.
pub fn recompose_adv(
    combine: DecMixer,
    b: u32,
    children: &[Option<LaneCommitments<Commitment>>],
) -> Result<Option<LaneCommitments<Commitment>>, AdvPresenceError> {
    let advs = require_homogeneous(children)?;
    let Some(advs) = advs else { return Ok(None) };
    let component = |pick: fn(&LaneCommitments<Commitment>) -> &Commitment| {
        let cs: Vec<Commitment> = advs.iter().map(|adv| pick(adv).clone()).collect();
        combine(&cs, b)
    };
    Ok(Some(LaneCommitments {
        ops: component(|adv| &adv.ops),
        is: component(|adv| &adv.is),
        fs: component(|adv| &adv.fs),
    }))
}

/// All-or-nothing presence across a fold's inputs; `Ok(Some(..))` yields
/// the tuples in input order, `Ok(None)` means a plain (non-Nebula) fold.
fn require_homogeneous(
    inputs: &[Option<LaneCommitments<Commitment>>],
) -> Result<Option<Vec<&LaneCommitments<Commitment>>>, AdvPresenceError> {
    let present = inputs.iter().filter(|adv| adv.is_some()).count();
    if present == 0 {
        return Ok(None);
    }
    if present != inputs.len() {
        return Err(AdvPresenceError {
            present,
            total: inputs.len(),
        });
    }
    Ok(Some(
        inputs
            .iter()
            .map(|adv| adv.as_ref().expect("all present"))
            .collect(),
    ))
}
