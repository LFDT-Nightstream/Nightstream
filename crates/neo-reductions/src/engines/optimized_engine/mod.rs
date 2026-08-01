//! Optimized engine implementation for PiCCS.
//!
//! `paper_rectangular` is the canonical prover. The
//! [`legacy_split_nc`] namespace contains the old block/lane, replay, and
//! deferred accelerator protocol. Code outside that namespace must not use
//! those computations as evidence of paper equivalence.

#![allow(non_snake_case)]

use crate::engines::utils::digest_ccs_matrices_with_sparse_cache;
use crate::error::PiCcsError;
use crate::superneo_eval::{build_superneo_eval_cache, SuperneoEvalCache};
use neo_ccs::CcsStructure;
use neo_math::F;
use neo_math::K;
use p3_goldilocks::Goldilocks;
use std::sync::Arc;

// Common types and utility functions shared across engines
mod backend;
mod block_lane_entrypoints;
mod block_lane_replay;
mod block_lane_terminal;
mod common;
mod delayed_projection;
mod digit_table;
mod legacy_types;
mod paper_rectangular;
mod phase_trace;
mod proof_assembly;
mod replay_binding;
mod replay_entrypoints;
mod replay_validation;
mod rlc;
mod row_poly;
mod sparse;
mod terminal_identities;
mod terminal_outputs;
mod transcript_segments;

mod oracle;
mod prove;
mod verify;

// Re-export commonly used items
pub use crate::engines::pi_ccs_protocol::{Challenges, PiCcsProof, PiCcsProofVariant};
pub use sparse::{CscMat, SparseCache};

#[derive(Debug, Clone, Copy, Default)]
pub struct PiCcsProvePerf {
    pub bind_ms: f64,
    pub sample_challenges_ms: f64,
    pub fe_sumcheck_ms: f64,
    pub nc_sumcheck_ms: f64,
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
    pub fe_sumcheck_ms: f64,
    pub nc_sumcheck_ms: f64,
    pub output_checks_ms: f64,
    pub terminal_ms: f64,
    pub total_ms: f64,
}

// Re-export the independent PiDEC and PiRLC computations. These are not part
// of the legacy PiCCS split protocol.
pub(crate) use common::claimed_initial_sum_from_inputs_with_k_mcs;
pub use common::{
    dec_reduction_paper_exact, dec_reduction_paper_exact_with_commit_check,
    dec_reduction_paper_exact_with_sparse_cache, dec_reduction_paper_exact_with_superneo_cache,
    dec_reduction_paper_exact_with_superneo_cache_and_digit_flags, recomposed_z_from_Z, rlc_reduction_paper_exact,
    rlc_reduction_paper_exact_with_commit_mix,
};
pub use rlc::{
    rlc_combine_claims, rlc_mix_witnesses, rlc_reduction_optimized, rlc_reduction_optimized_with_commit_mix,
    rlc_reduction_optimized_with_mixers,
};
pub(crate) use terminal_identities::{rhs_terminal_identity_fe_with_k_mcs, rhs_terminal_identity_nc};
// The normal optimized interface implements PaperRectangularV1.
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
    pub use super::paper_rectangular::{OptimizedPaperRectangularFeOracle, OptimizedPaperRectangularNcOracle};
}

/// Explicit access to the superseded block/lane SplitNC protocol.
///
/// This namespace exists only for the current accelerator and recursive-circuit
/// migration. It does not implement `PaperRectangularV1`, and callers must not
/// compare it with `PaperExact` as a paper-equivalence oracle.
pub mod legacy_split_nc {
    pub use super::backend::{
        BackendTranscriptMode, FeEvalTable, FeMcsRowTables, FePhaseTraceRequest, FeRowRoundSummary, FeRowRoundTrace,
        FeSumcheckBackend, NcColRoundTrace, NcColTraceRequest, NcFinalizedColState, NcPhaseRoundTrace,
        NcSumcheckBackend, PiCcsPhaseBackend, PiCcsPhaseProofLog, PiCcsPhaseSummary, PiCcsPhaseTrace,
        PiCcsPhaseTraceRequest, PiCcsTerminalOutputSurfaces, TranscriptSnapshot,
    };
    pub use super::block_lane_entrypoints::{
        optimized_prove_block_lane_delayed_with_cache_and_instance_digest_and_me_input_handle_and_perf,
        optimized_verify_block_lane_delayed_with_cache_and_instance_digest_and_me_input_handle_and_perf,
    };
    pub use super::common::{
        chi_ajtai_at_bool_point, chi_row_at_bool_point, claimed_initial_sum_from_inputs_with_k_mcs, eq_points,
        q_at_point_paper_exact, q_eval_at_ext_point_paper_exact, q_eval_at_ext_point_paper_exact_with_inputs,
        sum_q_over_hypercube_paper_exact,
    };
    pub use super::delayed_projection::{
        beta_power_selector as delayed_beta_power_selector, claimed_initial_sum as delayed_claimed_initial_sum,
        parent_evaluation as delayed_parent_evaluation, terminal_rhs as delayed_terminal_rhs,
        validate_input as validate_delayed_projection_input, DelayedProjectionChallenges, DelayedProjectionConfig,
        DelayedProjectionInput,
    };
    pub use super::digit_table::{build_nc_digit_table_compact, NcDigitMasks, NcDigitTable};
    pub use super::legacy_types::{
        PiCcsReplayOutputs, PiCcsReplayProofWitness, PiCcsReplayTerminalState, PiCcsReplayWitnessOutputs,
        PiCcsTerminalOutputShell,
    };
    pub use super::oracle::{BlockLaneNcPending, OptimizedOracle as CcsOracle};
    pub use super::proof_assembly::PiCcsDeferredProof;
    pub use super::prove::{
        optimized_defer_prove_with_device_backends_and_transcript_mode,
        optimized_defer_prove_with_phase_backend_and_transcript_mode, optimized_prove_with_device_backends,
        optimized_prove_with_device_backends_and_transcript_mode,
        optimized_prove_with_phase_backend_and_transcript_mode,
    };
    pub use super::replay_entrypoints::{
        optimized_replay_outputs_with_cache_and_instance_digest_and_perf, optimized_replay_outputs_with_cache_and_perf,
        optimized_replay_terminal_state_with_cache_and_instance_digest_and_perf,
        optimized_replay_terminal_state_with_cache_and_perf,
        optimized_replay_terminal_state_with_cache_instance_digest_and_me_input_handle_and_perf,
        optimized_replay_terminal_state_with_phase_backend_and_transcript_mode,
        optimized_replay_trace_with_cache_and_instance_digest_and_perf,
        optimized_replay_trace_with_cache_instance_digest_and_me_input_handle_and_perf,
        optimized_replay_witness_with_cache_and_instance_digest_and_perf, optimized_replay_witness_with_cache_and_perf,
    };
    pub use super::terminal_identities::{
        rhs_terminal_identity_fe, rhs_terminal_identity_fe_with_k_mcs, rhs_terminal_identity_nc,
    };

    /// Legacy CPU oracle internals used by accelerator parity tests.
    pub mod oracle {
        pub use super::super::oracle::*;
    }
}

#[derive(Clone)]
pub struct OptimizedStructureCache {
    sparse: Arc<SparseCache<F>>,
    superneo: Arc<SuperneoEvalCache>,
    mat_digest: [Goldilocks; 4],
    /// Shape fingerprint of the source structure: `(n, m, t)` where
    /// `t = matrices.len()`. Used by downstream code to assert this
    /// cache is still describing the structure it was built from
    /// (e.g. after a caller mutated a public `Preprocessing.structure`
    /// field).
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
        let (superneo, mat_digest) = {
            let sparse_for_digest = Arc::clone(&sparse);
            let (superneo, mat_digest) = rayon::join(
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
            (superneo?, mat_digest?)
        };
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let (superneo, mat_digest) = {
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
            let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(s, Some(sparse.as_ref()))
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
            (superneo, mat_digest)
        };
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "OptimizedStructureCache::build: TOTAL              {:.2?}",
            t_total.elapsed()
        );
        let mut superneo = superneo;
        superneo.set_mat_digest(mat_digest);
        Ok(Self {
            sparse,
            superneo: Arc::new(superneo),
            mat_digest,
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

    pub fn mat_digest(&self) -> &[Goldilocks; 4] {
        &self.mat_digest
    }

    /// `(n, m, t)` of the structure this cache was built from. Used as
    /// a cheap pre-digest fingerprint when validating that the cache
    /// still matches its owning `Structure`.
    pub fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }
}
