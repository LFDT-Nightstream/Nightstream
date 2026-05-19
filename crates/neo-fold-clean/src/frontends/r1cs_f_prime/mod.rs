//! R1CS F' frontend — fixed-shape R1CS circuit in, foldable lifecycle out.
//!
//! Mirrors [`fibonacci_f_prime`] but lets the user
//! pick the per-step app circuit. The verifier pins one R1CS shape
//! [`R1cs`]; every step supplies a satisfying assignment `z = [x | w]`
//! and the compiler emits an encoded F' step whose CCS structure
//! enforces every R1CS row via the existing mixed product gate.
//!
//! The image, the F' shell (boundary / state / Poseidon transitions /
//! selector / NIFS payload), the lifecycle plumbing, and the terminal
//! verifier path are the **same** as Fibonacci's. Only the appended
//! structure rows and the app-private region's interpretation differ —
//! `app_private` is repurposed as the bit-decomposed R1CS assignment.

pub mod compiler;
pub mod encoder;
pub mod instance;
pub mod lifecycle;
pub mod structure;

pub use compiler::{
    compile_step, start_chain, R1csChainState, R1csCompiledStep, R1csCompilerContext, R1csCompilerError,
    R1csFPrimeStepInput, R1csFoldForStep,
};
pub use encoder::{assignment_to_bits, encode_r1cs_f_prime_step, R1csEncoderInput};
pub use instance::build_instance;
pub use lifecycle::{prove_encoded_steps, R1csChainBuilder};
pub use structure::{build_r1cs_f_prime_structure, R1csRowAnchors, R1csShape, SparseR1cs};

use std::sync::Arc;

use thiserror::Error;

use crate::frontends::direct_ccs::{ajtai, ajtai_dec_mixer, ajtai_rlc_mixer, R1cs};
use crate::frontends::f_prime::image::FPrimeImageLayout;
use crate::frontends::f_prime::recursive_plan::{build_recursive_step_image_config, RecursiveStepImagePlan};
use crate::frontends::f_prime::structure::FPrimeStructure;
use crate::lifecycle::{preprocess as lifecycle_preprocess, Preprocessing};
use crate::paper::params::Params;

/// Lifecycle preprocessing pinned to one R1CS-shape + canonical plan.
///
/// `prep` carries the CCS structure derived from `(plan, r1cs)`; every
/// encoded step folded through one chain must share its
/// `structure_digest` (enforced by [`build_instance`]). `plan` is the
/// verifier-owned canonical [`RecursiveStepImagePlan`]; the compiler
/// reads it for both base and recursive branches. `r1cs` is the
/// verifier-pinned circuit shape used to derive the structure's R1CS
/// rows.
///
/// `structure` caches the full [`FPrimeStructure`] (shell rows +
/// per-constraint R1CS product rows) built once at preprocess time and
/// shared with every encoded step via [`Arc`]. The R1CS rows scale with
/// `r1cs.n()` and the shell's bitness rows scale with `plan.limbs`
/// (~1.5M rows for SHA-256-sized circuits), so re-deriving them per
/// step would dominate chain build time. `anchors` is kept alongside
/// for tests / external auditing — the compiler never reads it.
pub struct R1csFPrimePreprocessing {
    pub prep: Preprocessing,
    pub plan: RecursiveStepImagePlan,
    pub r1cs: R1csShape,
    pub structure: Arc<FPrimeStructure>,
    pub anchors: R1csRowAnchors,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(
        "encoded R1CS-F' instance's CCS structure (digest {step_digest:?}) does not match preprocessing's structure (digest {prep_digest:?})"
    )]
    StructureMismatch {
        prep_digest: [neo_math::F; 4],
        step_digest: [neo_math::F; 4],
    },
    #[error("R1CS-F' preprocessing: plan.limbs = {plan_limbs} does not match r1cs.m() * 64 + 1 = {expected}")]
    PlanLimbsMismatch { plan_limbs: usize, expected: usize },
    #[error(
        "R1CS-F' preprocessing: plan.state_x_out is None, but the R1CS frontend requires `state_x_out = Some(..)` so the chain's `state_x_out` Poseidon hash can absorb the app-level public input via `app_public_input_var_indices`"
    )]
    PlanMissingStateXOut,
    #[error(
        "R1CS-F' preprocessing: plan.state_x_out.app_public_input_var_indices = {actual:?} does not match the required range `0..r1cs.m_in` = 0..{m_in}. The frontend binds every public-input variable into `state_x_out`; misconfigured indices would silently miss the binding (different `x` could produce the same `public_output_digest`)."
    )]
    PlanAppPublicInputMismatch { actual: Vec<usize>, m_in: usize },
    #[error(transparent)]
    Compiler(#[from] compiler::R1csCompilerError),
    #[error("R1CS-F' chain builder: cannot finish before appending any steps")]
    ChainEmpty,
    #[error("R1CS-F' chain builder: expected active lifecycle state while deriving the next recursive fold")]
    ChainExpectedActiveState,
    #[error(transparent)]
    Frontend(#[from] crate::frontends::direct_ccs::FrontendError),
    #[error(transparent)]
    Params(#[from] neo_params::ParamsError),
    #[error(transparent)]
    Relations(#[from] crate::paper::relations::RelationError),
    #[error(transparent)]
    Lifecycle(#[from] crate::lifecycle::Error),
}

/// Build [`R1csFPrimePreprocessing`] from a verifier-owned plan + R1CS
/// shape + caller-supplied protocol params.
///
/// The CCS structure is derived from the canonical plan + R1CS shape;
/// no prover input is consulted here.
pub fn preprocess(
    r1cs: &R1cs,
    plan: &RecursiveStepImagePlan,
    params: Params,
) -> Result<R1csFPrimePreprocessing, Error> {
    r1cs.validate_shape()?;
    let r1cs = R1csShape::Dense(r1cs.clone());
    let (structure, anchors, public_input_len) = derive_structure(plan, &r1cs)?;
    let prep = lifecycle_preprocess(
        params,
        structure.ccs.clone(),
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        Some(public_input_len),
    )?;
    Ok(R1csFPrimePreprocessing {
        prep,
        plan: plan.clone(),
        r1cs,
        structure,
        anchors,
    })
}

/// Build [`R1csFPrimePreprocessing`] from a verifier-owned plan + sparse
/// R1CS shape + caller-supplied protocol params.
pub fn preprocess_sparse(
    r1cs: &SparseR1cs,
    plan: &RecursiveStepImagePlan,
    params: Params,
) -> Result<R1csFPrimePreprocessing, Error> {
    r1cs.validate_shape()?;
    let r1cs = R1csShape::Sparse(r1cs.clone());
    let (structure, anchors, public_input_len) = derive_structure(plan, &r1cs)?;
    let prep = lifecycle_preprocess(
        params,
        structure.ccs.clone(),
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        Some(public_input_len),
    )?;
    Ok(R1csFPrimePreprocessing {
        prep,
        plan: plan.clone(),
        r1cs,
        structure,
        anchors,
    })
}

/// Test/demo helper: derive the structure, install an Ajtai PP
/// deterministically from `seed`, and build preprocessing under
/// `config::r1cs_params` for the derived CCS shape.
pub fn preprocess_seeded(
    r1cs: &R1cs,
    plan: &RecursiveStepImagePlan,
    seed: u64,
) -> Result<R1csFPrimePreprocessing, Error> {
    r1cs.validate_shape()?;
    let r1cs = R1csShape::Dense(r1cs.clone());
    let (structure, anchors, public_input_len) = derive_structure(plan, &r1cs)?;
    let params = crate::config::r1cs_params(structure.ccs.n, structure.ccs.m)?;
    let _ = ajtai::setup_seeded(&params, &structure.ccs, seed);
    let prep = lifecycle_preprocess(
        params,
        structure.ccs.clone(),
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        Some(public_input_len),
    )?;
    Ok(R1csFPrimePreprocessing {
        prep,
        plan: plan.clone(),
        r1cs,
        structure,
        anchors,
    })
}

/// Test/demo helper for sparse R1CS shapes.
pub fn preprocess_sparse_seeded(
    r1cs: &SparseR1cs,
    plan: &RecursiveStepImagePlan,
    seed: u64,
) -> Result<R1csFPrimePreprocessing, Error> {
    r1cs.validate_shape()?;
    let r1cs = R1csShape::Sparse(r1cs.clone());
    let (structure, anchors, public_input_len) = derive_structure(plan, &r1cs)?;
    let params = crate::config::r1cs_params(structure.ccs.n, structure.ccs.m)?;
    let _ = ajtai::setup_seeded(&params, &structure.ccs, seed);
    let prep = lifecycle_preprocess(
        params,
        structure.ccs.clone(),
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        Some(public_input_len),
    )?;
    Ok(R1csFPrimePreprocessing {
        prep,
        plan: plan.clone(),
        r1cs,
        structure,
        anchors,
    })
}

/// Test/demo helper variant that lets the caller supply a custom
/// [`Params`] (e.g. a smaller test profile that preserves the
/// protocol's algebraic correctness at lower cryptographic security).
/// Production paths must use [`preprocess`] / [`preprocess_seeded`]
/// with the Appendix B.2 params.
pub fn preprocess_seeded_with_params(
    r1cs: &R1cs,
    plan: &RecursiveStepImagePlan,
    params: Params,
    seed: u64,
) -> Result<R1csFPrimePreprocessing, Error> {
    r1cs.validate_shape()?;
    let r1cs = R1csShape::Dense(r1cs.clone());
    let (structure, anchors, public_input_len) = derive_structure(plan, &r1cs)?;
    let _ = ajtai::setup_seeded(&params, &structure.ccs, seed);
    let prep = lifecycle_preprocess(
        params,
        structure.ccs.clone(),
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        Some(public_input_len),
    )?;
    Ok(R1csFPrimePreprocessing {
        prep,
        plan: plan.clone(),
        r1cs,
        structure,
        anchors,
    })
}

/// Test/demo helper variant for sparse R1CS with custom params.
pub fn preprocess_sparse_seeded_with_params(
    r1cs: &SparseR1cs,
    plan: &RecursiveStepImagePlan,
    params: Params,
    seed: u64,
) -> Result<R1csFPrimePreprocessing, Error> {
    r1cs.validate_shape()?;
    let r1cs = R1csShape::Sparse(r1cs.clone());
    let (structure, anchors, public_input_len) = derive_structure(plan, &r1cs)?;
    let _ = ajtai::setup_seeded(&params, &structure.ccs, seed);
    let prep = lifecycle_preprocess(
        params,
        structure.ccs.clone(),
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        Some(public_input_len),
    )?;
    Ok(R1csFPrimePreprocessing {
        prep,
        plan: plan.clone(),
        r1cs,
        structure,
        anchors,
    })
}

/// Run the verifier-owned `(plan, r1cs)` through the canonical
/// image-layout + R1CS-aware structure builder, after validating that
/// the plan binds the full R1CS public input into `state_x_out`.
///
/// Returns the cached [`FPrimeStructure`] (wrapped in [`Arc`] so the
/// compiler can hand it to every encoded step) plus the R1CS row
/// anchors and public-input length. Building this is the dominant cost
/// of preprocessing for large R1CS shapes — sharing the result across
/// every step of a chain is the whole point of caching.
fn derive_structure(
    plan: &RecursiveStepImagePlan,
    r1cs: &R1csShape,
) -> Result<(Arc<FPrimeStructure>, R1csRowAnchors, usize), Error> {
    let expected_limbs = r1cs.m() * crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS + 1;
    if plan.limbs != expected_limbs {
        return Err(Error::PlanLimbsMismatch {
            plan_limbs: plan.limbs,
            expected: expected_limbs,
        });
    }

    // The verifier-owned plan MUST bind every R1CS public-input
    // variable into `state_x_out` — otherwise the chain's
    // `public_output_digest` does not commit to `x`, and two
    // assignments with different public input can produce the same
    // verifier-visible output (the load-bearing property pinned by
    // `r1cs_compiler_public_output_depends_on_public_input`). We reject
    // here at preprocess time so a misconfigured plan never compiles
    // a step instead of failing silently downstream.
    let state_x_out = plan
        .state_x_out
        .as_ref()
        .ok_or(Error::PlanMissingStateXOut)?;
    let expected_indices: Vec<usize> = (0..r1cs.m_in()).collect();
    if state_x_out.app_public_input_var_indices != expected_indices {
        return Err(Error::PlanAppPublicInputMismatch {
            actual: state_x_out.app_public_input_var_indices.clone(),
            m_in: r1cs.m_in(),
        });
    }

    let layout = FPrimeImageLayout::new(build_recursive_step_image_config(plan));
    let public_input_len = 1 + layout.boundary.bits;
    let (structure, anchors) = structure::build_r1cs_f_prime_structure(layout, r1cs);
    Ok((Arc::new(structure), anchors, public_input_len))
}
