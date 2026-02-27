//! Shard-level folding: CPU (Π_CCS) + memory sidecar (Twist/Shout) via Route A.
//!
//! High-level flow (per step):
//! 1. Bind CCS header + carried ME inputs.
//! 2. Prove/verify a *batched* time/row sum-check that shares `r_time` across CCS + Twist/Shout time oracles.
//! 3. Finish CCS Ajtai rounds using the CCS oracle state after the batched rounds.
//! 4. Finalize the memory sidecar at the shared `r_time` (and optionally produce Twist `r_val` claims).
//! 5. Fold all `r_time` ME claims (CCS outputs + memory claims) via Π_RLC → Π_DEC into `k_rho` children.
//! 6. If Twist produces `r_val` ME claims, fold them in a separate Π_RLC → Π_DEC lane.
//!
//! Notes:
//! - CCS-only folding is supported by passing steps with empty LUT/MEM vectors.
//! - Index→OneHot adapter is integrated via the Shout address-domain proving flow.

#![allow(non_snake_case)]

use crate::finalize::ObligationFinalizer;
use crate::memory_sidecar::sumcheck_ds::{run_sumcheck_prover_ds, verify_sumcheck_rounds_ds};
use crate::memory_sidecar::utils::RoundOraclePrefix;
use crate::pi_ccs::{self as ccs, FoldingMode};
pub use crate::shard_proof_types::{
    BatchedTimeProof, FoldStep, MemOrLutProof, MemSidecarProof, RlcDecProof, ShardFoldOutputs, ShardFoldWitnesses,
    ShardObligations, ShardProof, ShardSegmentKind, ShardSegmentMeta, ShoutProofK, StepProof, TwistProofK,
};
use crate::PiCcsError;
#[cfg(target_arch = "wasm32")]
use js_sys::Date;
use neo_ajtai::{
    get_global_pp_for_dims, get_global_pp_seeded_params_for_dims, has_global_pp_for_dims, sample_uniform_rq,
    seeded_pp_chunk_seeds, try_get_loaded_global_pp_for_dims, Commitment as Cmt,
};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{KExtensions, D, F, K};
use neo_memory::riscv::trace::{Rv32DecodeSidecarLayout, Rv32TraceLayout};
use neo_memory::ts_common as ts;
use neo_memory::witness::{LutTableSpec, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::oracle::SparseCache;
use neo_reductions::engines::utils;
use neo_reductions::paper_exact_engine::{build_me_outputs_paper_exact, claimed_initial_sum_from_inputs_with_k_mcs};
use neo_reductions::sumcheck::{poly_eval_k, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
type TimePoint = f64;
#[cfg(not(target_arch = "wasm32"))]
type TimePoint = Instant;

#[inline]
fn time_now() -> TimePoint {
    #[cfg(target_arch = "wasm32")]
    {
        Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Instant::now()
    }
}

#[inline]
fn elapsed_ms(start: TimePoint) -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        Date::now() - start
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        start.elapsed().as_secs_f64() * 1_000.0
    }
}

#[path = "shard/ccs_only_batched.rs"]
mod ccs_only_batched;
#[path = "shard/core_utils.rs"]
mod core_utils;
#[path = "shard/mixed_batched.rs"]
mod mixed_batched;
#[path = "shard/poseidon_lane_helpers.rs"]
mod poseidon_lane_helpers;
#[path = "shard/prover.rs"]
mod prover;
#[path = "shard/rlc_dec.rs"]
mod rlc_dec;
#[path = "shard/route_a_segment.rs"]
mod route_a_segment;
#[path = "shard/verifier_and_api.rs"]
mod verifier_and_api;
#[path = "shard/verify_consistency.rs"]
mod verify_consistency;

pub use ccs_only_batched::{fold_shard_prove_ccs_only_batched, fold_shard_verify_ccs_only_batched};
pub use core_utils::{absorb_step_memory, check_step_linking, CommitMixers, StepLinkingConfig};
pub use verifier_and_api::*;

pub(crate) use core_utils::*;
pub(crate) use mixed_batched::*;
pub(crate) use poseidon_lane_helpers::*;
pub(crate) use prover::*;
pub(crate) use rlc_dec::*;
pub(crate) use route_a_segment::*;
pub(crate) use verify_consistency::*;
