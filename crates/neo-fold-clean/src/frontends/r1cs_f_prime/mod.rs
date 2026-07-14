//! R1CS F' frontend — fixed-shape R1CS circuit in, foldable lifecycle out.
//!
//! Mirrors [`fibonacci_f_prime`] but lets the user
//! pick the per-step app circuit. The verifier pins one R1CS shape
//! [`R1cs`]; every step supplies a satisfying assignment `z = [x | w]`
//! and the compiler emits an encoded F' step whose CCS structure
//! enforces every R1CS row via the existing mixed product gate.
//!
//! The image, the F' shell (boundary / state / Poseidon transitions /
//! accumulator handles / optional source-image NIFS payload), the lifecycle plumbing, and the terminal
//! verifier path are the **same** as Fibonacci's. Only the appended
//! structure rows and the app-private region's interpretation differ —
//! `app_private` is repurposed as the bit-decomposed R1CS assignment.

pub mod compiler;
pub mod encoder;
pub mod full_relation;
pub mod instance;
pub mod ivc;
pub mod lifecycle;
pub mod lowering;
mod selective;
mod selective_audit;
pub mod structure;
mod ternary_encoding;

pub use compiler::{
    compile_chunk, compile_step, start_chain, R1csChainState, R1csCompiledStep, R1csCompilerContext, R1csCompilerError,
    R1csFPrimeStepInput, R1csFoldForStep,
};
pub use encoder::{assignment_to_bits, encode_r1cs_f_prime_step, R1csEncoderInput};
pub use full_relation::{
    semantic_state_digest_fields, FullFPrimeBranchExecution, FullFPrimeContext, FullFPrimeError, FullFPrimeExecution,
    FullFPrimeRelation, FullFPrimeShape,
};
pub use instance::build_instance;
pub use lifecycle::{prove_encoded_steps, R1csChainBuilder};
pub use lowering::{
    build_fixed_shape_low_norm_r1cs, build_fixed_shape_low_norm_r1cs_with_shared_private_prefix,
    build_multi_branch_low_norm_r1cs, build_multi_branch_low_norm_r1cs_with_alignment, lower_field_r1cs,
    lower_sparse_r1cs_to_low_norm, FieldR1csLoweringError, FixedR1csBranch, FixedShapeLowNormR1cs, LowNormR1cs,
    LowNormR1csError, LoweredFieldR1cs, MultiBranchLowNormR1cs,
};
pub(crate) use selective::{
    audit_multi_branch_selective_low_norm_shape_with_alignment,
    audit_multi_branch_selective_low_norm_shape_with_shared_bit_prefix, SelectiveLowNormShape,
};
pub use selective::{
    audit_multi_branch_selective_low_norm_width_with_alignment,
    audit_multi_branch_selective_low_norm_width_with_shared_bit_prefix,
    build_multi_branch_selective_low_norm_r1cs_with_alignment,
    build_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix,
};
pub use selective_audit::{
    SelectiveArmWidthAudit, SelectiveFamilyWidthAudit, SelectiveLowNormWidthAudit, SelectiveTraceWidthAudit,
};
pub use structure::{build_r1cs_f_prime_structure, R1csRowAnchors, R1csShape, SparseR1cs};

use std::sync::Arc;

use neo_reductions::optimized_engine::OptimizedStructureCache;
use thiserror::Error;

use crate::frontends::direct_ccs::{ajtai, ajtai_dec_mixer, ajtai_rlc_mixer, R1cs};
use crate::frontends::f_prime::image::FPrimeImageLayout;
use crate::frontends::f_prime::recursive_plan::{build_recursive_step_image_config, RecursiveStepImagePlan};
use crate::frontends::f_prime::structure::FPrimeStructure;
use crate::lifecycle::{
    preprocess as lifecycle_preprocess, preprocess_with_test_log_and_optimized_cache, Preprocessing,
};
use crate::paper::construction2::SemanticStateMode;
use crate::paper::digest::structure_digest_from_mat_digest;
use crate::paper::params::Params;

/// Read the plan and decide whether the chain built on top will carry
/// application state. The frontend installs this on
/// [`Preprocessing::semantic_state_mode`] so the verifier can apply the
/// stateless invariant where appropriate. `Stateful` covers both
/// transition-linked app state (`semantic_state_in/out_var_indices`) and
/// output/public-only binding (`app_public_input*_indices`): in either
/// case `state_x_out` absorbs an independent semantic digest that must
/// be authenticated by the generated F' structure.
pub(crate) fn semantic_state_mode_for_plan(plan: &RecursiveStepImagePlan) -> SemanticStateMode {
    let Some(state_x_out) = plan.state_x_out.as_ref() else {
        return SemanticStateMode::Stateless;
    };
    if state_x_out.semantic_state_in_var_indices.is_empty()
        && state_x_out.semantic_state_out_var_indices.is_empty()
        && state_x_out.app_public_input_var_indices.is_empty()
        && state_x_out.app_public_input_bit_var_indices.is_empty()
    {
        SemanticStateMode::Stateless
    } else {
        SemanticStateMode::Stateful
    }
}

/// Read the plan's initial-semantic-state anchor, defaulting to
/// `empty_semantic_state_digest()` for stateless plans. This is the
/// value that gets baked into `vk_fs_digest` AND into the F' image's
/// CCS structure's base-step constraint.
pub(crate) fn initial_semantic_state_digest_for_plan(plan: &RecursiveStepImagePlan) -> [u8; 32] {
    plan.state_x_out
        .as_ref()
        .and_then(|sxo| sxo.initial_semantic_state_digest_anchor)
        .unwrap_or_else(crate::paper::digest::empty_semantic_state_digest)
}

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
    /// Lifecycle-layer preprocessing. Kept `pub` because `Preprocessing`
    /// already locks down its soundness-critical internals (mode/anchor
    /// are `pub(crate)`); the lifecycle layer is read by many call
    /// sites and forcing all of them through an accessor adds churn
    /// without buying assurance.
    pub prep: Preprocessing,
    /// The four mutually-coupled F' image fields. **Private** so an
    /// external caller cannot assemble a Frankenstein
    /// `R1csFPrimePreprocessing` by swapping in a `plan` that doesn't
    /// match `structure`, an `r1cs` that doesn't match `plan`, or
    /// `anchors` from a different layout. Read access goes through
    /// [`Self::plan`] / [`Self::r1cs`] / [`Self::structure`] /
    /// [`Self::anchors`].
    plan: RecursiveStepImagePlan,
    r1cs: R1csShape,
    structure: Arc<FPrimeStructure>,
    anchors: R1csRowAnchors,
}

impl R1csFPrimePreprocessing {
    /// Read-only view of the verifier-owned recursive-step plan.
    pub fn plan(&self) -> &RecursiveStepImagePlan {
        &self.plan
    }

    /// Read-only view of the verifier-owned R1CS shape this
    /// preprocessing was built for.
    pub fn r1cs(&self) -> &R1csShape {
        &self.r1cs
    }

    /// Read-only view of the cached F' image structure shared with
    /// every encoded step in a chain.
    pub fn structure(&self) -> &Arc<FPrimeStructure> {
        &self.structure
    }

    /// Read-only view of the R1CS row anchors (used by audit
    /// tooling and tests).
    pub fn anchors(&self) -> &R1csRowAnchors {
        &self.anchors
    }
}

/// Verifier-derived, internally-consistent `(plan, r1cs, structure)` bundle.
///
/// This is the split preprocessing boundary for callers that need to
/// first inspect the derived F' shape (for example to choose
/// shape-dependent params) and then build lifecycle preprocessing
/// without re-deriving the same large structure. Fields stay private so
/// external callers cannot forge a mismatched tuple.
pub struct R1csFPrimeDerivedStructure {
    plan: RecursiveStepImagePlan,
    r1cs: R1csShape,
    structure: Arc<FPrimeStructure>,
    anchors: R1csRowAnchors,
    public_input_len: usize,
}

impl R1csFPrimeDerivedStructure {
    /// Read-only view of the derived F' structure.
    pub fn structure(&self) -> &FPrimeStructure {
        &self.structure
    }

    /// Read-only view of the plan that produced this structure.
    pub fn plan(&self) -> &RecursiveStepImagePlan {
        &self.plan
    }

    /// Read-only view of the R1CS shape used to derive the structure.
    pub fn r1cs(&self) -> &R1csShape {
        &self.r1cs
    }
}

/// Verifier-owned, cache-bearing R1CS-F' structure artifact.
///
/// This is the product-facing boundary for reusing expensive preprocessing
/// work. The optimized cache is built inside the constructor from the same
/// derived F' structure it is stored beside; there is deliberately no public
/// constructor that accepts a standalone cache.
pub struct R1csFPrimePreparedStructure {
    plan: RecursiveStepImagePlan,
    r1cs: R1csShape,
    structure: Arc<FPrimeStructure>,
    anchors: R1csRowAnchors,
    public_input_len: usize,
    optimized_cache: OptimizedStructureCache,
    structure_digest: [neo_math::F; 4],
}

impl R1csFPrimePreparedStructure {
    pub fn structure(&self) -> &FPrimeStructure {
        &self.structure
    }

    pub fn plan(&self) -> &RecursiveStepImagePlan {
        &self.plan
    }

    pub fn r1cs(&self) -> &R1csShape {
        &self.r1cs
    }

    pub fn structure_digest(&self) -> &[neo_math::F; 4] {
        &self.structure_digest
    }
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
    #[error(
        "R1CS-F' preprocessing: plan.limbs = {plan_limbs} does not match expected app-private width + 1 = {expected}"
    )]
    PlanLimbsMismatch { plan_limbs: usize, expected: usize },
    #[error(
        "R1CS-F' preprocessing: typed app-private layout provides {got} variable widths but r1cs.m() = {expected}"
    )]
    PlanAppPrivateWidthCountMismatch { got: usize, expected: usize },
    #[error("R1CS-F' preprocessing: typed app-private width at variable {index} is {width}; widths must be in 1..=64")]
    PlanAppPrivateWidthInvalid { index: usize, width: usize },
    #[error(
        "R1CS-F' preprocessing: typed app-private variable {index} uses a {width}-bit slot but the R1CS shape only proves a {proven_width}-bit bound"
    )]
    PlanAppPrivateWidthTooNarrow {
        index: usize,
        width: usize,
        proven_width: usize,
    },
    #[error(
        "R1CS-F' preprocessing: packed public-input variable {index} is not explicitly Boolean-constrained by the R1CS shape"
    )]
    PlanPackedPublicInputBooleanUnconstrained { index: usize },
    #[error(
        "R1CS-F' preprocessing: packed public-input variable {index} uses an app-private slot width of {width}; packed public-input mode requires a 1-bit slot"
    )]
    PlanPackedPublicInputWidthNotOne { index: usize, width: usize },
    #[error(
        "R1CS-F' preprocessing: plan.state_x_out is None, but the R1CS frontend requires `state_x_out = Some(..)` so the chain can bind app-level public input through semantic state"
    )]
    PlanMissingStateXOut,
    #[error(
        "R1CS-F' preprocessing: public input binding mismatch. \
         full-lane indices = {actual:?}, bit-packed indices = {actual_bit:?}; \
         exactly one mode must bind the required range `0..r1cs.m_in` = 0..{m_in}. \
         Misconfigured indices would silently miss the binding (different `x` could produce the same `public_output_digest`)."
    )]
    PlanAppPublicInputMismatch {
        actual: Vec<usize>,
        actual_bit: Vec<usize>,
        m_in: usize,
    },
    #[error(
        "R1CS-F' preprocessing: semantic-state input variables require an outgoing semantic binding. \
         Use semantic_state_out_var_indices for explicit app state, or app-public binding fields for output-only public data."
    )]
    PlanSemanticStatePartial,
    #[error(
        "R1CS-F' preprocessing: plan declares semantic-state input variables but no \
         `initial_semantic_state_digest_anchor`. Stateful chains MUST anchor the base step's \
         first app-state to a verifier-owned digest (baked into the F' image's CCS structure \
         via the base-gated `is_base * (state_in.semantic_state_digest_in_lane - anchor) == 0` \
         constraint)."
    )]
    PlanSemanticStateMissingAnchor,
    #[error(
        "R1CS-F' preprocessing: plan supplies an `initial_semantic_state_digest_anchor` but \
         no semantic-state input/output or app-public semantic output. An anchor with nothing to bind to is meaningless."
    )]
    PlanSemanticStateAnchorWithoutIndices,
    #[error(
        "R1CS-F' lifecycle: `prove_encoded_steps` is stateless-only and a stateful preprocessing \
         was supplied. Use `R1csChainBuilder` instead — it threads per-step semantic digests \
         consistently with the F' image's state-out binding rows."
    )]
    ProveEncodedStepsStatefulUnsupported,
    #[error("R1CS-F' preprocessing: semantic state variable index {index} is out of range for r1cs.m() = {m}")]
    PlanSemanticStateIndexOutOfRange { index: usize, m: usize },
    #[error(
        "R1CS-F' preprocessing: public input variable {index} is not bound by the explicit semantic-state in/out hashes"
    )]
    PlanPublicInputNotSemanticBound { index: usize },
    #[error(transparent)]
    Compiler(#[from] compiler::R1csCompilerError),
    #[error("R1CS-F' chain builder: cannot finish before appending any steps")]
    ChainEmpty,
    #[error("R1CS-F' chain builder: chunk must contain at least one assignment (got empty)")]
    EmptyChunk,
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
    #[error(transparent)]
    Nifs(#[from] crate::paper::nifs::Error),
    #[error(transparent)]
    OptimizedCacheBuild(#[from] neo_reductions::error::PiCcsError),
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
    let prep = lifecycle_preprocess(params, structure.ccs.clone(), Some(public_input_len))?
        .with_f_prime_recursive_link()
        .with_semantic_state_mode(semantic_state_mode_for_plan(plan))
        .with_initial_semantic_state_digest(initial_semantic_state_digest_for_plan(plan))?;
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
    let prep = lifecycle_preprocess(params, structure.ccs.clone(), Some(public_input_len))?
        .with_f_prime_recursive_link()
        .with_semantic_state_mode(semantic_state_mode_for_plan(plan))
        .with_initial_semantic_state_digest(initial_semantic_state_digest_for_plan(plan))?;
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
/// `config::ccs_params` for the derived CCS shape.
pub fn preprocess_seeded(
    r1cs: &R1cs,
    plan: &RecursiveStepImagePlan,
    seed: u64,
) -> Result<R1csFPrimePreprocessing, Error> {
    r1cs.validate_shape()?;
    let r1cs = R1csShape::Dense(r1cs.clone());
    let (structure, anchors, public_input_len) = derive_structure(plan, &r1cs)?;
    let params = crate::config::ccs_params(
        structure.ccs.n,
        structure.ccs.m,
        structure.ccs.t(),
        structure.ccs.max_degree(),
    )?;
    let _ = ajtai::setup_seeded(&params, &structure.ccs, seed);
    let prep = lifecycle_preprocess(params, structure.ccs.clone(), Some(public_input_len))?
        .with_f_prime_recursive_link()
        .with_semantic_state_mode(semantic_state_mode_for_plan(plan))
        .with_initial_semantic_state_digest(initial_semantic_state_digest_for_plan(plan))?;
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
    let params = crate::config::ccs_params(
        structure.ccs.n,
        structure.ccs.m,
        structure.ccs.t(),
        structure.ccs.max_degree(),
    )?;
    let _ = ajtai::setup_seeded(&params, &structure.ccs, seed);
    let prep = lifecycle_preprocess(params, structure.ccs.clone(), Some(public_input_len))?
        .with_f_prime_recursive_link()
        .with_semantic_state_mode(semantic_state_mode_for_plan(plan))
        .with_initial_semantic_state_digest(initial_semantic_state_digest_for_plan(plan))?;
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
    let prep = lifecycle_preprocess(params, structure.ccs.clone(), Some(public_input_len))?
        .with_f_prime_recursive_link()
        .with_semantic_state_mode(semantic_state_mode_for_plan(plan))
        .with_initial_semantic_state_digest(initial_semantic_state_digest_for_plan(plan))?;
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
    let prep = lifecycle_preprocess(params, structure.ccs.clone(), Some(public_input_len))?
        .with_f_prime_recursive_link()
        .with_semantic_state_mode(semantic_state_mode_for_plan(plan))
        .with_initial_semantic_state_digest(initial_semantic_state_digest_for_plan(plan))?;
    Ok(R1csFPrimePreprocessing {
        prep,
        plan: plan.clone(),
        r1cs,
        structure,
        anchors,
    })
}

/// Derive and validate the sparse R1CS-F' structure without building
/// lifecycle preprocessing yet.
///
/// Use this when params are selected from the derived F' shape and the
/// caller wants to avoid deriving the large structure twice. The return
/// value is opaque and can only be constructed by this function, so
/// [`preprocess_sparse_seeded_derived_with_params`] can safely consume
/// it without re-checking the structure rows.
pub fn derive_sparse_preprocessing_structure(
    r1cs: &SparseR1cs,
    plan: &RecursiveStepImagePlan,
) -> Result<R1csFPrimeDerivedStructure, Error> {
    r1cs.validate_shape()?;
    let r1cs = R1csShape::Sparse(r1cs.clone());
    let (structure, anchors, public_input_len) = derive_structure(plan, &r1cs)?;
    Ok(R1csFPrimeDerivedStructure {
        plan: plan.clone(),
        r1cs,
        structure,
        anchors,
        public_input_len,
    })
}

/// Derive and prepare a dense R1CS-F' structure artifact.
pub fn prepare_preprocessing_structure(
    r1cs: &R1cs,
    plan: &RecursiveStepImagePlan,
) -> Result<R1csFPrimePreparedStructure, Error> {
    r1cs.validate_shape()?;
    let r1cs = R1csShape::Dense(r1cs.clone());
    let (structure, anchors, public_input_len) = derive_structure(plan, &r1cs)?;
    prepare_structure_parts(plan.clone(), r1cs, structure, anchors, public_input_len)
}

/// Derive and prepare a sparse R1CS-F' structure artifact.
pub fn prepare_sparse_preprocessing_structure(
    r1cs: &SparseR1cs,
    plan: &RecursiveStepImagePlan,
) -> Result<R1csFPrimePreparedStructure, Error> {
    prepare_derived_structure(derive_sparse_preprocessing_structure(r1cs, plan)?)
}

/// Add verifier-owned optimized cache material to an already-derived
/// R1CS-F' structure.
pub fn prepare_derived_structure(derived: R1csFPrimeDerivedStructure) -> Result<R1csFPrimePreparedStructure, Error> {
    let R1csFPrimeDerivedStructure {
        plan,
        r1cs,
        structure,
        anchors,
        public_input_len,
    } = derived;
    prepare_structure_parts(plan, r1cs, structure, anchors, public_input_len)
}

fn prepare_structure_parts(
    plan: RecursiveStepImagePlan,
    r1cs: R1csShape,
    structure: Arc<FPrimeStructure>,
    anchors: R1csRowAnchors,
    public_input_len: usize,
) -> Result<R1csFPrimePreparedStructure, Error> {
    let optimized_cache = OptimizedStructureCache::build(&structure.ccs)?;
    let structure_digest = structure_digest_from_mat_digest(&structure.ccs, optimized_cache.mat_digest());
    Ok(R1csFPrimePreparedStructure {
        plan,
        r1cs,
        structure,
        anchors,
        public_input_len,
        optimized_cache,
        structure_digest,
    })
}

/// Build sparse R1CS-F' preprocessing from a previously derived
/// structure bundle.
///
/// This is equivalent to [`preprocess_sparse_seeded_with_params`] but
/// skips the expensive second structure derivation. The derived bundle
/// carries the validated plan/R1CS/structure triple, so the resulting
/// preprocessing preserves the same ownership and soundness boundary as
/// the one-shot API.
pub fn preprocess_sparse_seeded_derived_with_params(
    derived: R1csFPrimeDerivedStructure,
    params: Params,
    seed: u64,
) -> Result<R1csFPrimePreprocessing, Error> {
    preprocess_seeded_prepared_with_params(prepare_derived_structure(derived)?, params, seed)
}

/// Build R1CS-F' preprocessing from a prepared structure artifact without
/// rebuilding the optimized engine cache.
pub fn preprocess_seeded_prepared_with_params(
    prepared: R1csFPrimePreparedStructure,
    params: Params,
    seed: u64,
) -> Result<R1csFPrimePreprocessing, Error> {
    let log = ajtai::setup_seeded(&params, &prepared.structure.ccs, seed);
    let prep = preprocess_with_test_log_and_optimized_cache(
        params,
        std::sync::Arc::new(prepared.structure.ccs.clone()),
        log,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        Some(prepared.public_input_len),
        prepared.optimized_cache,
    )?
    .with_f_prime_recursive_link()
    .with_semantic_state_mode(semantic_state_mode_for_plan(&prepared.plan))
    .with_initial_semantic_state_digest(initial_semantic_state_digest_for_plan(&prepared.plan))?;
    Ok(R1csFPrimePreprocessing {
        prep,
        plan: prepared.plan,
        r1cs: prepared.r1cs,
        structure: prepared.structure,
        anchors: prepared.anchors,
    })
}

/// Run the verifier-owned `(plan, r1cs)` through the canonical
/// image-layout + R1CS-aware structure builder, after validating that
/// the plan binds the R1CS public input into the carried semantic state.
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
    validate_plan(plan, r1cs)?;
    let layout = FPrimeImageLayout::new(build_recursive_step_image_config(plan));
    let public_input_len = 1 + layout.boundary.bits;
    let (structure, anchors) = structure::build_r1cs_f_prime_structure(layout, r1cs);
    Ok((Arc::new(structure), anchors, public_input_len))
}

/// Validate the application-state contract shared by both the historical
/// image compiler and the authoritative recursive R1CS-IVC relation.
pub(crate) fn validate_plan(plan: &RecursiveStepImagePlan, r1cs: &R1csShape) -> Result<(), Error> {
    let boolean_vars = r1cs.boolean_constrained_variables();
    let proven_widths = if plan.app_private_var_widths.is_empty() {
        Vec::new()
    } else {
        r1cs.conservative_app_private_var_widths()
    };
    let state_x_out = plan
        .state_x_out
        .as_ref()
        .ok_or(Error::PlanMissingStateXOut)?;
    let expected_indices: Vec<usize> = (0..r1cs.m_in()).collect();
    let full_public_input_ok = state_x_out.app_public_input_var_indices == expected_indices
        && state_x_out.app_public_input_bit_var_indices.is_empty();
    let packed_public_input_ok = state_x_out.app_public_input_var_indices.is_empty()
        && state_x_out.app_public_input_bit_var_indices == expected_indices;
    if !plan.app_private_var_widths.is_empty() && plan.app_private_var_widths.len() != r1cs.m() {
        return Err(Error::PlanAppPrivateWidthCountMismatch {
            got: plan.app_private_var_widths.len(),
            expected: r1cs.m(),
        });
    }
    if let Some((index, &width)) = plan
        .app_private_var_widths
        .iter()
        .enumerate()
        .find(|(_, &width)| !(1..=crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS).contains(&width))
    {
        return Err(Error::PlanAppPrivateWidthInvalid { index, width });
    }
    if !plan.app_private_var_widths.is_empty() {
        if let Some((index, &width)) = plan
            .app_private_var_widths
            .iter()
            .enumerate()
            .find(|(index, &width)| width < proven_widths[*index])
        {
            return Err(Error::PlanAppPrivateWidthTooNarrow {
                index,
                width,
                proven_width: proven_widths[index],
            });
        }
        // One-bit slots need no separate explicit-Boolean-row gate: the
        // `PlanAppPrivateWidthTooNarrow` check above already requires every
        // declared width to cover the derivation's proven width, and the
        // derivation (`conservative_app_private_var_widths`) only proves
        // width 1 when the R1CS rows force the variable into {0, 1} — via
        // an explicit Boolean row or the determining-row corner rule.
    }
    let expected_app_private_bits = if plan.app_private_var_widths.is_empty() {
        r1cs.m() * crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS
    } else {
        plan.app_private_var_widths.iter().sum()
    };
    let expected_limbs = expected_app_private_bits + 1;
    if plan.limbs != expected_limbs {
        return Err(Error::PlanLimbsMismatch {
            plan_limbs: plan.limbs,
            expected: expected_limbs,
        });
    }

    // The verifier-owned plan MUST bind every R1CS public-input
    // variable into the carried semantic state. `state_x_out` absorbs
    // that semantic digest, so two assignments with different public
    // input cannot produce the same verifier-visible chain output (the
    // load-bearing property pinned by
    // `r1cs_compiler_public_output_depends_on_public_input`). We reject
    // here at preprocess time so a misconfigured plan never compiles a
    // step instead of failing silently downstream.
    if !full_public_input_ok && !packed_public_input_ok {
        return Err(Error::PlanAppPublicInputMismatch {
            actual: state_x_out.app_public_input_var_indices.clone(),
            actual_bit: state_x_out.app_public_input_bit_var_indices.clone(),
            m_in: r1cs.m_in(),
        });
    }
    if packed_public_input_ok {
        for &index in &state_x_out.app_public_input_bit_var_indices {
            let width = plan
                .app_private_var_widths
                .get(index)
                .copied()
                .unwrap_or(crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS);
            if width != 1 {
                return Err(Error::PlanPackedPublicInputWidthNotOne { index, width });
            }
            // R1CS variable 0 is the conventional constant-one lane in
            // Boolean-heavy frontends (Bellpepper/SHA). The structure pins
            // that lane when packed public bits are used, so it is safe to
            // pack as a bit even though it has no `z0 * (1-z0)` row.
            if index != 0 && !boolean_vars[index] {
                return Err(Error::PlanPackedPublicInputBooleanUnconstrained { index });
            }
        }
    }
    let has_semantic_in = !state_x_out.semantic_state_in_var_indices.is_empty();
    let has_explicit_semantic_out = !state_x_out.semantic_state_out_var_indices.is_empty();
    let has_app_public_semantic_out = !state_x_out.app_public_input_var_indices.is_empty()
        || !state_x_out.app_public_input_bit_var_indices.is_empty();
    let has_semantic_out = has_explicit_semantic_out || has_app_public_semantic_out;
    if has_semantic_in && !has_semantic_out {
        return Err(Error::PlanSemanticStatePartial);
    }
    for &index in state_x_out
        .semantic_state_in_var_indices
        .iter()
        .chain(state_x_out.semantic_state_out_var_indices.iter())
    {
        if index >= r1cs.m() {
            return Err(Error::PlanSemanticStateIndexOutOfRange { index, m: r1cs.m() });
        }
    }
    // When explicit semantic-state output variables are configured, the
    // app-public digest path is intentionally suppressed. Every declared
    // public R1CS input must therefore be represented by either the
    // incoming or outgoing semantic hash, except for z[0] when the
    // structure pins it as the conventional constant-one lane.
    if has_explicit_semantic_out {
        for index in state_x_out
            .app_public_input_var_indices
            .iter()
            .chain(state_x_out.app_public_input_bit_var_indices.iter())
            .copied()
        {
            let sem_bound = state_x_out.semantic_state_in_var_indices.contains(&index)
                || state_x_out.semantic_state_out_var_indices.contains(&index);
            if !sem_bound && index != 0 {
                return Err(Error::PlanPublicInputNotSemanticBound { index });
            }
        }
    }
    // Stateful transition inputs require a verifier-owned initial
    // anchor. Output-only public bindings use the preprocessing's
    // default empty seed unless the caller provides an explicit anchor.
    let has_anchor = state_x_out.initial_semantic_state_digest_anchor.is_some();
    if has_semantic_in && !has_anchor {
        return Err(Error::PlanSemanticStateMissingAnchor);
    }
    if !has_semantic_in && !has_semantic_out && has_anchor {
        return Err(Error::PlanSemanticStateAnchorWithoutIndices);
    }

    Ok(())
}
