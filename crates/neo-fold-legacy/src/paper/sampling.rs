//! Definition 17 (Strong Sampling Set) and Theorem 9 (expansion factor T).
//!
//! Owns: the typed challenge form for Π_RLC ρ_i, and the soundness gate
//! `count·T·(b−1) < B` from Definition 14. Both are invariants the verifier
//! can re-derive from `Params` alone.
//!
//! Does not own: the production rotation-matrix sampler (lives in
//! `neo-reductions::common`); paper-layer code should reach for it through
//! [`sample_rho`] / [`StrongSet`] only.
//!
//! ## Why `count` is a runtime value
//!
//! In Construction 2 the prover folds `K` fresh CCS instances per step on
//! top of `k_rho = pp.k_rho()` carried claims. `K` is a per-call choice (the
//! frontend decides), not a `NeoParams` field, so the bound check has to
//! happen at the call site.

use thiserror::Error;

use crate::paper::params::Params;

#[derive(Debug, Error)]
pub enum SamplingError {
    /// `count·T·(b−1) < B` (Definition 14) failed; ρ_i drawn from `𝒞`
    /// would not preserve norm bounds. **Hard rejection** — never warn.
    #[error(
        "\u{03A0}_RLC norm bound violated: count={count}\u{00B7}T={t}\u{00B7}(b-1)={bm1} = {lhs} must be < B = {b_pow_k}"
    )]
    RlcBoundViolated {
        count: usize,
        t: u128,
        bm1: u32,
        lhs: u128,
        b_pow_k: u128,
    },
}

/// The strong sampling set 𝒞 ⊆ 𝑅_𝔽 — opaque handle.
///
/// Construction goes through [`StrongSet::for_params_and_count`], which
/// validates the Π_RLC bound up front for the specific `count` of ρ_i it
/// will be used to sample.
#[derive(Clone, Debug)]
pub struct StrongSet {
    /// Expansion factor T (Theorem 9).
    expansion_t: u128,
    /// The `count` this set was validated for. Other counts must be
    /// re-validated.
    count: usize,
}

impl StrongSet {
    /// Build the strong sampling set for the given `Params` and `count` of
    /// rotations to sample, validating the Π_RLC norm bound (Definition 14)
    /// at construction time.
    pub fn for_params_and_count(pp: &Params, count: usize) -> Result<Self, SamplingError> {
        let t = pp.T() as u128;
        check_rlc_bound(pp, count, t)?;
        Ok(Self { expansion_t: t, count })
    }

    pub fn expansion_t(&self) -> u128 {
        self.expansion_t
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

/// Π_RLC challenge ρ_i — a validated rotation matrix from 𝒞 (Definition 17).
///
/// Re-exported from `neo_reductions::common` so paper and engine speak the
/// same wire-format type. The validation invariants (rotation-matrix shape,
/// alphabet membership) are enforced at construction inside `neo_reductions`.
pub use neo_reductions::common::RotRho;

/// Definition 14 Π_RLC bound: `count·T·(b−1) < B`.
///
/// `count` is the number of ρ_i sampled in this step (paper: `K + k`).
pub fn check_rlc_bound(pp: &Params, count: usize, t: u128) -> Result<(), SamplingError> {
    let bm1 = pp.b().saturating_sub(1);
    let lhs = (count as u128)
        .saturating_mul(t)
        .saturating_mul(bm1 as u128);
    let b_pow_k = pp.big_b() as u128;
    if lhs < b_pow_k {
        Ok(())
    } else {
        Err(SamplingError::RlcBoundViolated {
            count,
            t,
            bm1,
            lhs,
            b_pow_k,
        })
    }
}

// `RotRho` values come into existence via `engine::optimized::sample_rho_n`,
// which the paper layer reaches through `pi_rlc::prove` / `verify`. There is
// no per-element sampling helper here — sampling is batched at the call site
// so the bound check `count·T·(b−1) < B` runs once per fold, not N times.
