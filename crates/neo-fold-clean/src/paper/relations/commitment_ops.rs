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
