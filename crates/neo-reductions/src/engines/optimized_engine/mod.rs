//! Optimized engine implementation for PiCCS.
//!
//! `paper_joint` is the canonical prover. Device selection can change only
//! evaluator implementation; it cannot change protocol messages.

#![allow(non_snake_case)]

use crate::engines::utils::digest_ccs_matrices_with_sparse_cache;
use crate::error::PiCcsError;
use crate::superneo_eval::{build_superneo_eval_cache, SuperneoEvalCache};
use neo_ccs::CcsStructure;
use neo_math::F;
use neo_math::K;
use p3_goldilocks::Goldilocks;
use std::sync::Arc;

mod common;
pub(crate) mod paper_joint;
mod prove;
mod rlc;
mod sparse;
mod verify;

// Re-export commonly used items
pub use crate::engines::pi_ccs_protocol::{Challenges, PiCcsProof};
pub use sparse::{CscMat, SparseCache};

#[derive(Debug, Clone, Copy, Default)]
pub struct PiCcsProvePerf {
    pub bind_ms: f64,
    pub sample_challenges_ms: f64,
    pub sumcheck_ms: f64,
    pub output_materialize_ms: f64,
    pub total_ms: f64,
}

/// Prover-only data shared by adjacent Π_CCS and Π_DEC phases.
#[derive(Debug, Clone)]
pub struct PiDecProverPrecompute {
    pub row_chals: Vec<K>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PiCcsVerifyPerf {
    pub bind_ms: f64,
    pub bind_header_instances_ms: f64,
    pub bind_header_prefix_ms: f64,
    pub bind_header_poly_ms: f64,
    pub bind_header_public_instances_ms: f64,
    pub bind_me_inputs_ms: f64,
    pub bind_sample_challenges_ms: f64,
    pub sumcheck_ms: f64,
    pub output_checks_ms: f64,
    pub terminal_ms: f64,
    pub total_ms: f64,
}

pub use common::{
    dec_reduction_optimized, dec_reduction_optimized_with_digit_flags, dec_reduction_optimized_with_superneo_cache,
};
pub use rlc::{
    rlc_combine_claims, rlc_mix_witnesses, rlc_reduction_optimized, rlc_reduction_optimized_with_commit_mix,
    rlc_reduction_optimized_with_mixers,
};
// The normal optimized interface implements PaddedRowIdentity.
pub use prove::optimized_prove as pi_ccs_prove;
pub use prove::optimized_prove_with_cache;
pub use prove::optimized_prove_with_cache_and_instance_digest_and_me_input_handle_and_perf;
pub use prove::optimized_prove_with_cache_and_instance_digest_and_perf;
pub use prove::optimized_prove_with_cache_and_perf;
pub use verify::optimized_verify as pi_ccs_verify;
pub use verify::optimized_verify_with_cache;
pub use verify::optimized_verify_with_cache_and_instance_digest_and_me_input_handle_and_perf;
pub use verify::optimized_verify_with_cache_and_instance_digest_and_perf;
pub use verify::optimized_verify_with_cache_and_perf;

/// Wrapper for simple case (k=1, no ME inputs)
pub use prove::optimized_prove_simple as pi_ccs_prove_simple;

/// Canonical oracle types exposed only for the independent PaperExact audit
/// suite. Normal clients use the prover and verifier entrypoints.
#[cfg(feature = "paper-exact")]
pub mod canonical_audit {
    pub use super::paper_joint::OptimizedPaperJointOracle;
}

#[derive(Clone)]
pub struct OptimizedStructureCache {
    sparse: Arc<SparseCache<F>>,
    superneo: Arc<SuperneoEvalCache>,
    matrix_digest: [Goldilocks; 4],
    /// Shape fingerprint of the source structure: `(n, m, t)`.
    /// Full cache validation also uses shared ownership or the matrix digest.
    shape: (usize, usize, usize),
}

impl OptimizedStructureCache {
    pub fn build(s: &CcsStructure<F>) -> Result<Self, PiCcsError> {
        Self::build_with_sparse(s, Arc::new(SparseCache::build(s)))
    }

    pub fn build_shared(structure: Arc<CcsStructure<F>>) -> Result<Self, PiCcsError> {
        let sparse = Arc::new(SparseCache::from_shared_structure(Arc::clone(&structure)));
        Self::build_with_sparse(structure.as_ref(), sparse)
    }

    fn build_with_sparse(s: &CcsStructure<F>, sparse: Arc<SparseCache<F>>) -> Result<Self, PiCcsError> {
        #[cfg(feature = "perf-timers")]
        let t_total = std::time::Instant::now();
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let (superneo, matrix_digest) = {
            let sparse_for_digest = Arc::clone(&sparse);
            let (superneo, matrix_digest) = rayon::join(
                || {
                    #[cfg(feature = "perf-timers")]
                    let t_superneo = std::time::Instant::now();
                    let out = build_superneo_eval_cache(s).ok_or_else(|| {
                        PiCcsError::InvalidInput(format!(
                            "optimized cache requires SuperNeo-compatible CCS shape (m={}, matrices={})",
                            s.m,
                            s.matrices.len()
                        ))
                    });
                    #[cfg(feature = "perf-timers")]
                    eprintln!(
                        "OptimizedStructureCache::build: superneo           {:.2?}",
                        t_superneo.elapsed()
                    );
                    out
                },
                || {
                    #[cfg(feature = "perf-timers")]
                    let t_digest = std::time::Instant::now();
                    let out = digest_ccs_matrices_with_sparse_cache(s, Some(sparse_for_digest.as_ref()))
                        .try_into()
                        .map_err(|digest: Vec<Goldilocks>| {
                            PiCcsError::ProtocolError(format!(
                                "optimized cache expected 4 CCS digest limbs, got {}",
                                digest.len()
                            ))
                        });
                    #[cfg(feature = "perf-timers")]
                    eprintln!(
                        "OptimizedStructureCache::build: matrix digest      {:.2?}",
                        t_digest.elapsed()
                    );
                    out
                },
            );
            (superneo?, matrix_digest?)
        };
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let (superneo, matrix_digest) = {
            #[cfg(feature = "perf-timers")]
            let t_superneo = std::time::Instant::now();
            let superneo = build_superneo_eval_cache(s).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "optimized cache requires SuperNeo-compatible CCS shape (m={}, matrices={})",
                    s.m,
                    s.matrices.len()
                ))
            })?;
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "OptimizedStructureCache::build: superneo           {:.2?}",
                t_superneo.elapsed()
            );
            #[cfg(feature = "perf-timers")]
            let t_digest = std::time::Instant::now();
            let matrix_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(s, Some(sparse.as_ref()))
                .try_into()
                .map_err(|digest: Vec<Goldilocks>| {
                    PiCcsError::ProtocolError(format!(
                        "optimized cache expected 4 CCS digest limbs, got {}",
                        digest.len()
                    ))
                })?;
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "OptimizedStructureCache::build: matrix digest      {:.2?}",
                t_digest.elapsed()
            );
            (superneo, matrix_digest)
        };
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "OptimizedStructureCache::build: TOTAL              {:.2?}",
            t_total.elapsed()
        );
        Ok(Self {
            sparse,
            superneo: Arc::new(superneo),
            matrix_digest,
            shape: (s.n, s.m, s.matrices.len()),
        })
    }

    pub fn sparse(&self) -> &SparseCache<F> {
        self.sparse.as_ref()
    }

    pub fn superneo(&self) -> &SuperneoEvalCache {
        self.superneo.as_ref()
    }

    /// Shared handles for constructing oracles outside this crate (e.g.
    /// accelerator backends building an `OptimizedOracle` for snapshots).
    pub fn sparse_arc(&self) -> Arc<SparseCache<F>> {
        self.sparse.clone()
    }

    pub fn superneo_arc(&self) -> Arc<SuperneoEvalCache> {
        self.superneo.clone()
    }

    pub fn matrix_digest(&self) -> &[Goldilocks; 4] {
        &self.matrix_digest
    }

    /// Check that this cache belongs to the selected CCS structure.
    ///
    /// A shared cache owns the same immutable `Arc` as production
    /// preprocessing, so pointer identity is sufficient and constant-time.
    /// Standalone caches recompute the canonical digest to keep the public
    /// low-level API safe against same-shape cache substitution.
    pub fn validate_structure(&self, structure: &CcsStructure<F>) -> Result<(), PiCcsError> {
        if self.shape != (structure.n, structure.m, structure.t()) {
            return Err(PiCcsError::InvalidInput(
                "optimized structure cache shape does not match the selected CCS structure".into(),
            ));
        }
        if self.sparse.shares_structure(structure) {
            return Ok(());
        }
        let digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, None)
            .try_into()
            .map_err(|digest: Vec<Goldilocks>| {
                PiCcsError::ProtocolError(format!(
                    "optimized cache expected 4 CCS digest limbs, got {}",
                    digest.len()
                ))
            })?;
        if digest != self.matrix_digest {
            return Err(PiCcsError::InvalidInput(
                "optimized structure cache matrix digest does not match the selected CCS structure".into(),
            ));
        }
        Ok(())
    }

    /// `(n, m, t)` of the structure this cache was built from. Used as
    /// a cheap pre-digest fingerprint when validating that the cache
    /// still matches its owning `Structure`.
    pub fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }
}
