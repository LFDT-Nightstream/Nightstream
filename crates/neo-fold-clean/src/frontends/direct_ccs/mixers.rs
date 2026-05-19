//! Standard Ajtai-action commitment mixers for direct-CCS users.
//!
//! Both Π_RLC and Π_DEC need a closure that combines commitments under
//! the protocol's commitment-scheme action. For the audited Ajtai
//! homomorphism, the actions are:
//!
//! - **Π_RLC**: `Σ ρ_i · c_i` where ρ_i is the *polynomial* a rotation
//!   matrix represents (not a scalar). Use `cf_inv` to recover the
//!   polynomial coefficients from the matrix's first column, then
//!   `s_mul_add` for the polynomial-times-commitment multiplication.
//! - **Π_DEC**: `Σ b^{i-1} · c_i` for `i = 1..k`. The base `b` is a
//!   small integer, so this is just per-lane scalar multiplication via
//!   `scale_commitment_add_inplace`.
//!
//! These functions match `RlcMixer`/`DecMixer` `fn(...)` types exactly,
//! so users pass them by name rather than declaring local closures.

use neo_ajtai::{s_mul_add, scale_commitment_add_inplace, Commitment};
use neo_ccs::Mat;
use neo_math::ring::{cf_inv, Rq};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

/// `Σ ρ_i · c_i` under the Ajtai S-module action. ρ_i is the polynomial
/// recovered from the rotation matrix's first column.
pub fn ajtai_rlc_mixer(rhos: &[Mat<F>], commits: &[Commitment]) -> Commitment {
    debug_assert!(!commits.is_empty(), "commit_mix: empty commitments");
    debug_assert_eq!(rhos.len(), commits.len(), "commit_mix: |rhos| \u{2260} |commits|");
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
