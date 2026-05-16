//! Fibonacci F' frontend — app step in, foldable lifecycle out.
//!
//! Two entry points, sharing one canonical [`RecursiveStepImagePlan`]
//! (the verifier-owned structure description):
//!
//! 1. **Encoded path** — callers who already have an
//!    [`EncodedFibonacciFPrimeStep`] in hand hand it directly to
//!    [`build_instance`] / [`prove_encoded_steps`]. Used by test
//!    fixtures and downstream chain builders that own their own
//!    encoder.
//! 2. **Compiler path** — callers feed app-level
//!    [`FibonacciAppStepInput`]s through
//!    [`compile_fibonacci_step`]. The compiler hides
//!    `FibonacciFPrimeStepInput` and the NIFS / K-mul / ring-action /
//!    Poseidon-trace plumbing. Branches internally on
//!    `ctx.chain_state.chunk_count`: base step (`== 0`, no prior fold)
//!    or recursive step (`> 0`, caller supplies a real per-step
//!    `StepProof::Recursive` via `ctx.fold_for_step`).
//!
//! Unlike [`crate::frontends::direct_ccs`] (which takes a user R1CS and
//! folds raw application CCS instances), this frontend folds the
//! **augmented** step `enc(F'_i)` HyperNova §6.3 Construction 2 calls
//! for: each fresh CCS instance is the bit-encoding of a step that
//! contains the previous fold's NIFS.V trace in-circuit. The chain is
//! self-verifying; the terminal decider stays O(1) in chain length
//! (see [`crate::engine::decider::synthesize_last_step_terminal_r1cs`]).
//!
//! ## API surface
//!
//! - [`FibonacciFPrimePreprocessing`] — wrapped lifecycle preprocessing
//!   plus the canonical [`RecursiveStepImagePlan`] the compiler reads.
//! - [`preprocess`] — production entry; caller supplies a canonical
//!   [`RecursiveStepImagePlan`] plus protocol params, and the Ajtai
//!   setup is read from the canonical global registry for the derived
//!   structure's CCS shape.
//! - [`preprocess_seeded`] — test/demo helper that derives params from
//!   the plan-derived CCS shape via [`crate::config::r1cs_params`] and
//!   installs the Ajtai setup deterministically from a caller-supplied
//!   seed.
//! - [`build_instance`] — turn one encoded F' step into a foldable
//!   `CcsInstance` under matched preprocessing.
//! - [`prove_encoded_steps`] — fold a sequence of encoded F' steps
//!   through `lifecycle::prove`, one step per batch.
//! - [`compile_fibonacci_step`] / [`start_fibonacci_chain`] — the
//!   app-step compiler. See [`compiler`] for the detailed contract,
//!   in particular the per-step F'-transcript expectation on
//!   `ctx.fold_for_step.proof`.

pub mod compiler;
pub mod encoder;
pub mod image;
pub mod instance;
pub mod lifecycle;
pub mod recursive_plan;
pub mod structure;

pub use compiler::{
    compile_fibonacci_step, start_fibonacci_chain, FibonacciAppState, FibonacciAppStepInput, FibonacciAppStepOutput,
    FibonacciAppWitness, FibonacciChainState, FibonacciCompiledStep, FibonacciCompilerContext, FibonacciCompilerError,
    FibonacciFoldForStep,
};
pub use instance::build_instance;
pub use lifecycle::prove_encoded_steps;

use thiserror::Error;

use crate::frontends::direct_ccs::{ajtai, ajtai_dec_mixer, ajtai_rlc_mixer};
use crate::frontends::fibonacci_f_prime::image::FibonacciFPrimeImageLayout;
use crate::frontends::fibonacci_f_prime::recursive_plan::{build_recursive_step_image_config, RecursiveStepImagePlan};
use crate::frontends::fibonacci_f_prime::structure::build_fibonacci_f_prime_structure;
use crate::lifecycle::{preprocess as lifecycle_preprocess, Preprocessing};
use crate::paper::params::Params;

/// Lifecycle preprocessing pinned to one encoded-F' CCS structure.
///
/// All encoded F' steps folded through a single chain must share the
/// same `FPrimeStructure.ccs` shape; [`build_instance`] enforces that
/// by `structure_digest` equality. Wrapping the [`Preprocessing`] in a
/// named type makes that contract explicit at the type level and keeps
/// the encoded-F' path visually distinct from `direct_ccs::Preprocessing`
/// at call sites.
///
/// `plan` is the verifier-owned [`RecursiveStepImagePlan`] whose CCS
/// structure `prep` was derived from. The compiler reads it to drive
/// both base- and recursive-step paths (NIFS payload shape, accumulator
/// preimage layout, etc.) so a single canonical plan flows through the
/// whole chain.
pub struct FibonacciFPrimePreprocessing {
    pub prep: Preprocessing,
    pub plan: RecursiveStepImagePlan,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(
        "encoded F' instance's CCS structure (digest {step_digest:?}) does not match preprocessing's structure (digest {prep_digest:?})"
    )]
    StructureMismatch {
        prep_digest: [neo_math::F; 4],
        step_digest: [neo_math::F; 4],
    },
    #[error(transparent)]
    Params(#[from] neo_params::ParamsError),
    #[error(transparent)]
    Relations(#[from] crate::paper::relations::RelationError),
    #[error(transparent)]
    Lifecycle(#[from] crate::lifecycle::Error),
}

/// Build [`FibonacciFPrimePreprocessing`] from a verifier-owned
/// [`RecursiveStepImagePlan`] and caller-supplied protocol params.
///
/// **Soundness**: the CCS structure is derived *here* from the
/// canonical plan the verifier already trusts (vk_fs / compiler-config
/// territory). A prover-supplied encoded step is never consulted at
/// this point. Cross-checking that prover steps match this structure
/// is [`build_instance`]'s job (via `structure_digest` equality).
///
/// The Ajtai setup is verifier-owned global protocol configuration.
/// This function reads the already-registered canonical setup for the
/// derived structure's `(D, cols)` shape; it does not accept a
/// proof-supplied setup. Use [`preprocess_seeded`] when you want a
/// deterministic test/demo setup derived from a seed.
pub fn preprocess(plan: &RecursiveStepImagePlan, params: Params) -> Result<FibonacciFPrimePreprocessing, Error> {
    let (structure, public_input_len) = derive_canonical_structure(plan);
    let prep = lifecycle_preprocess(
        params,
        structure,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        Some(public_input_len),
    )?;
    Ok(FibonacciFPrimePreprocessing {
        prep,
        plan: plan.clone(),
    })
}

/// Test/demo helper: derive structure from the verifier-owned `plan`,
/// derive params from the resulting CCS shape via
/// [`crate::config::r1cs_params`], install an Ajtai PP for the shape
/// deterministically from `seed`, then build preprocessing.
///
/// Production callers should install the canonical Ajtai setup out of
/// band and then call [`preprocess`]. This helper exists for
/// deterministic tests and examples, mirroring
/// [`crate::frontends::direct_ccs::preprocess_seeded`].
pub fn preprocess_seeded(plan: &RecursiveStepImagePlan, seed: u64) -> Result<FibonacciFPrimePreprocessing, Error> {
    let (structure, public_input_len) = derive_canonical_structure(plan);
    let params = crate::config::r1cs_params(structure.n, structure.m)?;
    let _ = ajtai::setup_seeded(&params, &structure, seed);
    let prep = lifecycle_preprocess(
        params,
        structure,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        Some(public_input_len),
    )?;
    Ok(FibonacciFPrimePreprocessing {
        prep,
        plan: plan.clone(),
    })
}

/// Run the verifier-owned plan through the canonical image-layout +
/// structure builders. Returns `(structure, public_input_len)`; the
/// caller wraps them into [`Preprocessing`]. Centralised here so
/// `preprocess` and `preprocess_seeded` share one derivation.
fn derive_canonical_structure(plan: &RecursiveStepImagePlan) -> (neo_ccs::CcsStructure<neo_math::F>, usize) {
    let layout = FibonacciFPrimeImageLayout::new(build_recursive_step_image_config(plan));
    let public_input_len = 1 + layout.boundary.bits;
    let structure = build_fibonacci_f_prime_structure(layout).ccs;
    (structure, public_input_len)
}
