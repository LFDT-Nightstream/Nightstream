//! Helpers for linear Ajtai-opening projections.
//!
//! For a seeded PP and a chosen set of vectors `u_i ∈ F^d`, we can express batched opening checks
//! as a single inner product over the witness matrix `Z`.
//!
//! Concretely, for `c = Commit(pp, Z)`, this module computes a weight vector `w_u` such that:
//!   Σ_i ⟨u_i, c_i⟩ = ⟨w_u, Z⟩
//! where `c_i` is the i-th Ajtai commitment column and `Z` is interpreted in row-major order.
//!
//! This is used by closure proofs to avoid recomputing `Commit(pp, Z)` directly.

#![forbid(unsafe_code)]

use neo_math::{F, D};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use p3_field::PrimeCharacteristicRing;

/// Compute the row-major weight vector `w_u` for a seeded PP and vectors `u_vecs` (one per κ row).
///
/// Returns a vector of length `D * m`, indexed as `w_u[row * m + col]`.
pub fn compute_opening_weights_for_u_seeded(seed: [u8; 32], m: usize, u_vecs: &[[F; D]]) -> Vec<F> {
    let kappa = u_vecs.len();
    let d = D;

    let mut w = vec![F::ZERO; d * m];
    if m == 0 || kappa == 0 {
        return w;
    }

    let (chunk_size, chunk_seeds_by_row) = crate::commit::seeded_pp_chunk_seeds(seed, kappa, m);

    for (i, u_i) in u_vecs.iter().enumerate() {
        let chunk_seeds = &chunk_seeds_by_row[i];
        for (chunk_idx, &chunk_seed) in chunk_seeds.iter().enumerate() {
            let start = chunk_idx * chunk_size;
            let end = core::cmp::min(m, start + chunk_size);

            let mut rng = ChaCha8Rng::from_seed(chunk_seed);
            let mut nxt = [F::ZERO; D];
            for col_idx in start..end {
                let a_ij = crate::commit::sample_uniform_rq(&mut rng);
                let mut rot_col = neo_math::cf(a_ij);
                for row in 0..d {
                    let mut dot = F::ZERO;
                    for r in 0..d {
                        dot += u_i[r] * rot_col[r];
                    }
                    w[row * m + col_idx] += dot;
                    crate::commit::rot_step(&rot_col, &mut nxt);
                    core::mem::swap(&mut rot_col, &mut nxt);
                }
            }
        }
    }

    w
}
