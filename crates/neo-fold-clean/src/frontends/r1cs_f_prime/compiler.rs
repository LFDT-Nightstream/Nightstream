//! R1CS-specific compilation of fixed-shape assignments into encoded `F'` steps.
//!
//! Owns: assignment shape/satisfaction checks, semantic-state digests, public
//! input binding, and step/chunk compiler entrypoints.
//!
//! Does not own: shared `F'` chain mechanics, image row formulas, lifecycle
//! folding, or NIFS verification.
//!
//! Emits constraints: no. It validates native assignments and fills a cached
//! encoded relation.
//!
//! Authority boundary: the verifier-owned R1CS shape determines valid
//! assignments; shared prior-fold verification remains authoritative for the
//! recursive link.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Chain initialization | [`start_chain`] | no | Preprocessed verifier shape |
//! | Step/chunk compilation | [`compile_step`], [`compile_chunk`] | no | Satisfying fixed-shape assignments |
//! | Semantic state | semantic-state digest helpers | no | Verifier-selected state columns |

use std::sync::Arc;

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use crate::frontends::direct_ccs::FrontendError;
use crate::frontends::f_prime::compiler::{
    assemble_shared_chunk_traces, assemble_step_from_shared, canonical_base_accumulator_digest,
    canonical_ce_shape_and_child_count, nifs_ce_shape_from_claim, nifs_ce_view_from_claim,
    nifs_payload_inputs_for_source_image, perp_nifs_ce_view, start_f_prime_chain_context, verify_prior_fold,
    FPrimeChainState, FPrimeCompilerContext, FPrimeFoldForStep, FPrimeFoldPostSummary, FPrimeShellCompilerError,
};
use crate::frontends::f_prime::encoder::EncodedFPrimeStep;
use crate::frontends::f_prime::image::{NifsCeClaimShape, NifsCeClaimView};
use crate::frontends::f_prime::recursive_plan::build_semantic_state_preimage_fields;
use crate::frontends::f_prime::recursive_plan::RecursiveStepImagePlan;
use crate::frontends::r1cs_f_prime::encoder::{assignment_to_bits, encode_r1cs_f_prime_step, R1csEncoderInput};
use crate::frontends::r1cs_f_prime::R1csFPrimePreprocessing;
use crate::paper::construction2::TRIVIAL_PC;
use crate::paper::f_prime::poseidon_trace::encode_poseidon_trace;

/// `pc` for an R1CS-F' chain.
///
/// HyperNova / SuperNeo's `pc` (program counter) selects which `F'_j`
/// the chain folds. The R1CS frontend pins one R1CS shape per chain
/// (one `pc`, one `F'_j`), so `pc` is constant. We bind it through
/// `paper::construction2::TRIVIAL_PC` directly rather than aliasing
/// it under a frontend-specific name; the underlying constant is the
/// same `TRIVIAL_PC = 1`, but the name now reflects which frontend
/// committed to it.
const R1CS_F_PRIME_PC: u64 = TRIVIAL_PC;

/// R1CS-facing alias for the shared F'-shell fold authority.
pub type R1csFoldForStep = FPrimeFoldForStep;

/// R1CS-facing alias for the shared F'-shell chain state.
pub type R1csChainState = FPrimeChainState;

/// Per-step input to [`compile_step`]: one satisfying R1CS assignment
/// `z = [x | w]` of length `r1cs.m()`.
#[derive(Clone, Debug)]
pub struct R1csFPrimeStepInput {
    pub assignment: Vec<F>,
}

/// Output of [`compile_step`].
#[derive(Debug)]
pub struct R1csCompiledStep {
    pub encoded: EncodedFPrimeStep,
    /// `state_x_out` digest committed in this step's boundary region.
    pub public_output_digest: [F; 4],
    /// Carried semantic-state digest consumed by this step.
    pub semantic_state_digest_in: [F; 4],
    /// Carried semantic-state digest produced by this step.
    pub semantic_state_digest_out: [F; 4],
}

/// R1CS-facing alias for the shared F'-shell compiler context.
pub type R1csCompilerContext = FPrimeCompilerContext;

/// Initialise a compiler context for a fresh R1CS chain.
///
/// Thin wrapper over
/// [`crate::frontends::f_prime::compiler::start_f_prime_chain_context`]
/// that pins the R1CS frontend's `pc` and sources `limbs` from the
/// canonical plan (sized by the R1CS's variable count `m()`).
pub fn start_chain(prep: &R1csFPrimePreprocessing) -> Result<R1csCompilerContext, R1csCompilerError> {
    Ok(start_f_prime_chain_context(
        &prep.prep,
        R1CS_F_PRIME_PC,
        prep.plan.limbs,
    )?)
}

/// Compile one R1CS app step into a foldable [`EncodedFPrimeStep`].
/// K=1 wrapper around [`compile_chunk`]; preserves the legacy
/// single-assignment API and surface error type used by per-step
/// callers.
pub fn compile_step(
    prep: &R1csFPrimePreprocessing,
    ctx: &mut R1csCompilerContext,
    input: R1csFPrimeStepInput,
) -> Result<R1csCompiledStep, R1csCompilerError> {
    match compile_chunk(prep, ctx, vec![input]) {
        Ok(mut compiled) => {
            debug_assert_eq!(compiled.len(), 1);
            Ok(compiled.pop().expect("compile_chunk(1) returns 1 step"))
        }
        // Map the chunk's positional error back to the legacy
        // single-step error so existing K=1 callers keep matching on
        // `R1csCompilerError::Unsatisfied(_)`.
        Err(R1csCompilerError::UnsatisfiedAt { source, .. }) => Err(R1csCompilerError::Unsatisfied(source)),
        Err(e) => Err(e),
    }
}

/// Compile a SuperNeo same-shape chunk of `inputs.len()` R1CS app
/// assignments into `inputs.len()` foldable [`EncodedFPrimeStep`]s, all
/// rooted at the same pre-step chain state. The chain advances
/// **once** (chunk_count += 1, step_count += K) at the end of the
/// chunk; every emitted `EncodedFPrimeStep` carries the same
/// post-step chain coordinates (chunk_count, step_count, z_i,
/// public_trace, acc_digest), so the resulting K CCS instances form
/// one "chunk rooted at one prior Construction-2 state."
///
/// Branches on `ctx.chain_state.chunk_count`:
/// - `0` → base step. Caller must NOT supply `ctx.fold_for_step`.
/// - `> 0` → recursive step. Caller MUST supply `ctx.fold_for_step`;
///   the compiler re-verifies the prior fold's NIFS proof **once**
///   under the per-step F' transcript (which now absorbs `K =
///   inputs.len()` via the chunk digest) before emitting the K
///   encoded images.
pub fn compile_chunk(
    prep: &R1csFPrimePreprocessing,
    ctx: &mut R1csCompilerContext,
    inputs: Vec<R1csFPrimeStepInput>,
) -> Result<Vec<R1csCompiledStep>, R1csCompilerError> {
    if inputs.is_empty() {
        return Err(FPrimeShellCompilerError::EmptyChunk.into());
    }
    let rows_in_chunk = inputs.len();

    // ── App-level satisfaction check (all K assignments) ──────────
    #[cfg(feature = "perf-timers")]
    let t_satisfaction = std::time::Instant::now();
    for (idx, input) in inputs.iter().enumerate() {
        if input.assignment.len() != prep.r1cs.m() {
            return Err(R1csCompilerError::AssignmentLength {
                got: input.assignment.len(),
                expected: prep.r1cs.m(),
            });
        }
        if prep.anchors().constant_lane_pinned && input.assignment[0] != F::ONE {
            return Err(R1csCompilerError::ConstantLaneNotOne {
                got: input.assignment[0],
            });
        }
        prep.r1cs
            .is_satisfied_by(&input.assignment)
            .map_err(|e| R1csCompilerError::UnsatisfiedAt { index: idx, source: e })?;
    }
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[r1cs-compile] app satisfaction (x{:>2})       {:>7.2}s",
        rows_in_chunk,
        t_satisfaction.elapsed().as_secs_f64()
    );
    // Canonical chain-link check for stateful chains. Lives here (not
    // in `R1csChainBuilder::append_chunk` nor `finalize_compile_chunk`)
    // because this is the single funnel both `compile_step` and the
    // chain builder pass through, AND it fires before
    // `PriorFoldMissingForRecursiveStep` would mask the soundness
    // signal. A mismatch here means step `i+1` claimed an input
    // semantic state that doesn't equal step `i`'s output — direct
    // chain disconnection, rejected before any expensive compile work.
    let semantic = semantic_state_digests_for_inputs(prep, &inputs)?;
    if let Some(s) = semantic.as_ref() {
        if let Some(input) = s.input {
            if ctx.chain_state.chunk_count > 0 && ctx.chain_state.semantic_state_digest != input {
                return Err(R1csCompilerError::SemanticStateInputMismatch {
                    expected: ctx.chain_state.semantic_state_digest,
                    got: input,
                });
            }
        }
    }

    // ── Branch: base (chunk_count == 0) or recursive (> 0) ──────────
    let is_base = ctx.chain_state.chunk_count == 0;

    if is_base {
        if ctx.fold_for_step.is_some() || ctx.fold_summary_for_step.is_some() {
            return Err(FPrimeShellCompilerError::BaseStepUnexpectedPriorFold.into());
        }
        compile_base_chunk(prep, ctx, inputs, rows_in_chunk, semantic)
    } else {
        match (
            ctx.fold_for_step.as_ref().cloned(),
            ctx.fold_summary_for_step.as_ref().cloned(),
        ) {
            (Some(_), Some(_)) => Err(FPrimeShellCompilerError::ConflictingPriorFoldInputs.into()),
            (Some(fold), None) => {
                if ctx.fold_for_step_needs_native_verify {
                    // The prior-fold transcript reconstruction needs the K of the
                    // current step (the NIFS proof's transcript absorbed
                    // chunk_digest with K from native).
                    #[cfg(feature = "perf-timers")]
                    let t_verify = std::time::Instant::now();
                    verify_prior_fold(&prep.prep, ctx, &fold, rows_in_chunk)?;
                    #[cfg(feature = "perf-timers")]
                    eprintln!(
                        "[r1cs-compile] verify_prior_fold          {:>7.2}s",
                        t_verify.elapsed().as_secs_f64()
                    );
                }
                compile_recursive_chunk(prep, ctx, inputs, fold, rows_in_chunk, semantic)
            }
            (None, Some(summary)) => {
                if ctx.fold_for_step_needs_native_verify {
                    return Err(FPrimeShellCompilerError::UnverifiedPriorFoldSummary.into());
                }
                compile_recursive_chunk_from_summary(prep, ctx, inputs, summary, rows_in_chunk, semantic)
            }
            (None, None) => Err(FPrimeShellCompilerError::PriorFoldMissingForRecursiveStep {
                chunk_count: ctx.chain_state.chunk_count,
            }
            .into()),
        }
    }
}

fn compile_base_chunk(
    prep: &R1csFPrimePreprocessing,
    ctx: &mut R1csCompilerContext,
    inputs: Vec<R1csFPrimeStepInput>,
    rows_in_chunk: usize,
    semantic: Option<SemanticStateDigests>,
) -> Result<Vec<R1csCompiledStep>, R1csCompilerError> {
    let plan = prep.plan.clone();
    let (ce_shape, child_count) = canonical_ce_shape_and_child_count(&plan)?;

    let perp_view = perp_nifs_ce_view(&ce_shape);
    let new_acc_digest = canonical_base_accumulator_digest(&prep.prep)?;
    finalize_compile_chunk(
        prep,
        ctx,
        inputs,
        plan,
        /* is_base = */ true,
        perp_view,
        new_acc_digest,
        child_count,
        rows_in_chunk,
        semantic,
    )
}

fn compile_recursive_chunk(
    prep: &R1csFPrimePreprocessing,
    ctx: &mut R1csCompilerContext,
    inputs: Vec<R1csFPrimeStepInput>,
    fold: R1csFoldForStep,
    rows_in_chunk: usize,
    semantic: Option<SemanticStateDigests>,
) -> Result<Vec<R1csCompiledStep>, R1csCompilerError> {
    let plan = prep.plan.clone();
    let (canonical_ce_shape, child_count) = canonical_ce_shape_and_child_count(&plan)?;
    let (ce_view, new_acc_digest) = if let Some(summary) = fold.post_summary {
        compile_surface_from_summary(summary, canonical_ce_shape, child_count)?
    } else {
        let post_running = &fold.post_running;
        let post_parent = post_running
            .parent_authority
            .as_ref()
            .ok_or(FPrimeShellCompilerError::PostRunningMissingParentAuthority)?;
        let actual_shape = nifs_ce_shape_from_claim(post_parent, ctx.public_input_len);
        if actual_shape != canonical_ce_shape {
            return Err(FPrimeShellCompilerError::PostParentShapeMismatch {
                canonical: canonical_ce_shape,
                actual: actual_shape,
            }
            .into());
        }
        if post_running.claims.len() as u64 != child_count {
            return Err(FPrimeShellCompilerError::PostRunningClaimsCountMismatch {
                canonical: child_count,
                actual: post_running.claims.len() as u64,
            }
            .into());
        }
        (
            nifs_ce_view_from_claim(post_parent, ctx.public_input_len),
            crate::paper::digest::digest32_as_fields(
                post_running
                    .accumulator_digest(prep.prep.structure())
                    .map_err(|_| FPrimeShellCompilerError::PostRunningMissingParentAuthority)?,
            ),
        )
    };
    finalize_compile_chunk(
        prep,
        ctx,
        inputs,
        plan,
        /* is_base = */ false,
        ce_view,
        new_acc_digest,
        child_count,
        rows_in_chunk,
        semantic,
    )
}

fn compile_recursive_chunk_from_summary(
    prep: &R1csFPrimePreprocessing,
    ctx: &mut R1csCompilerContext,
    inputs: Vec<R1csFPrimeStepInput>,
    summary: FPrimeFoldPostSummary,
    rows_in_chunk: usize,
    semantic: Option<SemanticStateDigests>,
) -> Result<Vec<R1csCompiledStep>, R1csCompilerError> {
    let plan = prep.plan.clone();
    let (canonical_ce_shape, child_count) = canonical_ce_shape_and_child_count(&plan)?;
    let (ce_view, new_acc_digest) = compile_surface_from_summary(summary, canonical_ce_shape, child_count)?;
    finalize_compile_chunk(
        prep,
        ctx,
        inputs,
        plan,
        /* is_base = */ false,
        ce_view,
        new_acc_digest,
        child_count,
        rows_in_chunk,
        semantic,
    )
}

fn compile_surface_from_summary(
    summary: FPrimeFoldPostSummary,
    canonical_ce_shape: NifsCeClaimShape,
    child_count: u64,
) -> Result<(NifsCeClaimView, [F; 4]), R1csCompilerError> {
    if summary.parent_shape != canonical_ce_shape {
        return Err(FPrimeShellCompilerError::PostParentShapeMismatch {
            canonical: canonical_ce_shape,
            actual: summary.parent_shape,
        }
        .into());
    }
    if summary.child_count != child_count {
        return Err(FPrimeShellCompilerError::PostRunningClaimsCountMismatch {
            canonical: child_count,
            actual: summary.child_count,
        }
        .into());
    }
    Ok((perp_nifs_ce_view(&canonical_ce_shape), summary.acc_digest))
}

/// Compose one encoded R1CS-F' step around the shared shell assembly.
///
/// R1CS plans that bind app-public data are semantic-stateful today, so
/// they must compile as serial K=1 chunks. That keeps the carried
/// semantic digest unambiguous: one proven app public value becomes one
/// outgoing Construction-2 state coordinate.
fn finalize_compile_chunk(
    prep: &R1csFPrimePreprocessing,
    ctx: &mut R1csCompilerContext,
    inputs: Vec<R1csFPrimeStepInput>,
    plan: RecursiveStepImagePlan,
    is_base: bool,
    ce_view: NifsCeClaimView,
    new_acc_digest: [F; 4],
    _child_count: u64,
    rows_in_chunk: usize,
    semantic: Option<SemanticStateDigests>,
) -> Result<Vec<R1csCompiledStep>, R1csCompilerError> {
    debug_assert_eq!(inputs.len(), rows_in_chunk);
    // For stateful-transition plans, seed the base compiler state to
    // the first input digest. Output-only plans keep the preprocessing
    // seed and simply publish their first output digest.
    if let Some(semantic) = semantic {
        if ctx.chain_state.chunk_count == 0 {
            if let Some(input) = semantic.input {
                ctx.chain_state.semantic_state_digest = input;
            }
        }
    }

    // Compute the chunk-shared trace ONCE: chunk_digest, the
    // app-input-independent boundary Poseidon trace, the delayed
    // accumulator handle state, and the post-step chain advance.
    // These are identical for every assignment in the chunk, so
    // recomputing them per assignment (the previous behavior) was the
    // dominant redundant compile cost.
    #[cfg(feature = "perf-timers")]
    let t_shared = std::time::Instant::now();
    let shared = assemble_shared_chunk_traces(ctx, is_base, new_acc_digest, rows_in_chunk);
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[r1cs-compile] shared chunk traces (once)   {:>7.2}s",
        t_shared.elapsed().as_secs_f64()
    );

    let mut compiled = Vec::with_capacity(rows_in_chunk);
    for input in inputs.into_iter() {
        // Bind this assignment's R1CS public input `x =
        // assignment[..m_in]` into the outgoing semantic-state digest.
        // That digest is then absorbed by `state_x_out`, so the native
        // verifier and the F' CCS agree on the recursive-link hash while
        // still learning this specific `x` was proven.
        #[cfg(feature = "perf-timers")]
        let t_lanes = std::time::Instant::now();
        let app_public_input = state_x_out_app_preimage_lanes_for_assignment(prep.plan(), &input.assignment)?;
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-compile] app preimage lanes           {:>7.2}s",
            t_lanes.elapsed().as_secs_f64()
        );
        #[cfg(feature = "perf-timers")]
        let t_assembly = std::time::Instant::now();
        let semantic_out = semantic.map(|s| s.output);
        let assembly = assemble_step_from_shared(&shared, ctx, &[], semantic_out);
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-compile] step state_x_out             {:>7.2}s",
            t_assembly.elapsed().as_secs_f64()
        );

        #[cfg(feature = "perf-timers")]
        let t_bits = std::time::Instant::now();
        let assignment_bits = assignment_to_plan_bits(&prep.plan, &input.assignment)?;
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-compile] assignment_to_bits           {:>7.2}s",
            t_bits.elapsed().as_secs_f64()
        );
        #[cfg(feature = "perf-timers")]
        let t_one_shot = std::time::Instant::now();
        let mut one_shot_traces = Vec::new();
        if let Some(state_x_out) = prep.plan.state_x_out.as_ref() {
            if !state_x_out.semantic_state_in_var_indices.is_empty() {
                one_shot_traces.push(semantic_state_trace_for_assignment(
                    &input.assignment,
                    &state_x_out.semantic_state_in_var_indices,
                ));
            }
            if !state_x_out.semantic_state_out_var_indices.is_empty() {
                one_shot_traces.push(semantic_state_trace_for_assignment(
                    &input.assignment,
                    &state_x_out.semantic_state_out_var_indices,
                ));
            } else if !state_x_out.app_public_input_var_indices.is_empty()
                || !state_x_out.app_public_input_bit_var_indices.is_empty()
            {
                one_shot_traces.push(encode_poseidon_trace(&build_semantic_state_preimage_fields(
                    &app_public_input,
                )));
            }
        }
        one_shot_traces.push(assembly.traces.state_x_out);
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-compile] one-shot traces              {:>7.2}s",
            t_one_shot.elapsed().as_secs_f64()
        );
        let encoder_input = R1csEncoderInput {
            plan: plan.clone(),
            boundary_bits: assembly.boundary_bits,
            state_in: assembly.state_in,
            state_out: assembly.state_out,
            chunk_digest: assembly.chunk_digest,
            assignment_bits,
            is_base,
            nifs_payloads: nifs_payload_inputs_for_source_image(&plan, ce_view.clone()),
            kmul_views: vec![],
            ring_action_pairs: vec![],
            one_shot_traces,
            sponge_trace: None,
        };

        #[cfg(feature = "perf-timers")]
        let t_encode = std::time::Instant::now();
        let encoded = encode_r1cs_f_prime_step(encoder_input, Arc::clone(&prep.structure));
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-compile] encode step                  {:>7.2}s",
            t_encode.elapsed().as_secs_f64()
        );

        compiled.push(R1csCompiledStep {
            encoded,
            public_output_digest: assembly.public_output_digest,
            semantic_state_digest_in: assembly.state_in.semantic_state_digest_in,
            semantic_state_digest_out: assembly.state_out.new_semantic_state_digest,
        });
    }

    // Advance the chain state **once** for the whole chunk. In
    // stateful mode, the semantic lane is the app-state output digest;
    // stateless mode keeps the legacy accumulator digest there.
    ctx.chain_state = if let Some(semantic) = semantic {
        FPrimeChainState {
            semantic_state_digest: semantic.output,
            ..shared.next_chain_state
        }
    } else {
        shared.next_chain_state
    };
    ctx.fold_for_step = None;
    ctx.fold_summary_for_step = None;
    ctx.fold_for_step_needs_native_verify = true;

    Ok(compiled)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SemanticStateDigests {
    pub input: Option<[F; 4]>,
    pub output: [F; 4],
}

pub(crate) fn semantic_state_digests_for_inputs(
    prep: &R1csFPrimePreprocessing,
    inputs: &[R1csFPrimeStepInput],
) -> Result<Option<SemanticStateDigests>, R1csCompilerError> {
    let Some(state_x_out) = prep.plan.state_x_out.as_ref() else {
        return Ok(None);
    };
    let has_input = !state_x_out.semantic_state_in_var_indices.is_empty();
    let has_output = !state_x_out.semantic_state_out_var_indices.is_empty()
        || !state_x_out.app_public_input_var_indices.is_empty()
        || !state_x_out.app_public_input_bit_var_indices.is_empty();
    if !has_input && !has_output {
        return Ok(None);
    }
    if inputs.len() != 1 {
        return Err(R1csCompilerError::StatefulChunkMustBeSerial { got: inputs.len() });
    }
    let assignment = &inputs[0].assignment;
    if assignment.len() != prep.r1cs.m() {
        return Err(R1csCompilerError::AssignmentLength {
            got: assignment.len(),
            expected: prep.r1cs.m(),
        });
    }
    let input =
        has_input.then(|| semantic_state_digest_for_assignment(assignment, &state_x_out.semantic_state_in_var_indices));
    let output = if !state_x_out.semantic_state_out_var_indices.is_empty() {
        semantic_state_digest_for_assignment(assignment, &state_x_out.semantic_state_out_var_indices)
    } else {
        let app_public_lanes = state_x_out_app_preimage_lanes_for_assignment(prep.plan(), assignment)?;
        semantic_state_digest_for_fields(&app_public_lanes)
    };
    Ok(Some(SemanticStateDigests { input, output }))
}

pub(crate) fn semantic_state_digest_for_assignment(assignment: &[F], indices: &[usize]) -> [F; 4] {
    semantic_state_trace_for_assignment(assignment, indices).digest_native
}

pub(crate) fn semantic_state_out_preimage_for_assignment(
    prep: &R1csFPrimePreprocessing,
    assignment: &[F],
) -> Result<Option<Vec<F>>, R1csCompilerError> {
    let Some(state_x_out) = prep.plan.state_x_out.as_ref() else {
        return Ok(None);
    };
    if !state_x_out.semantic_state_out_var_indices.is_empty() {
        let values = state_x_out
            .semantic_state_out_var_indices
            .iter()
            .map(|&idx| assignment[idx])
            .collect::<Vec<_>>();
        return Ok(Some(build_semantic_state_preimage_fields(&values)));
    }
    if !state_x_out.app_public_input_var_indices.is_empty() || !state_x_out.app_public_input_bit_var_indices.is_empty()
    {
        let app_public_lanes = state_x_out_app_preimage_lanes_for_assignment(prep.plan(), assignment)?;
        return Ok(Some(build_semantic_state_preimage_fields(&app_public_lanes)));
    }
    Ok(None)
}

pub(crate) fn semantic_state_in_preimage_for_assignment(
    prep: &R1csFPrimePreprocessing,
    assignment: &[F],
) -> Option<Vec<F>> {
    let state_x_out = prep.plan.state_x_out.as_ref()?;
    if state_x_out.semantic_state_in_var_indices.is_empty() {
        return None;
    }
    let values = state_x_out
        .semantic_state_in_var_indices
        .iter()
        .map(|&idx| assignment[idx])
        .collect::<Vec<_>>();
    Some(build_semantic_state_preimage_fields(&values))
}

pub(super) fn semantic_state_digest_for_fields(fields: &[F]) -> [F; 4] {
    encode_poseidon_trace(&build_semantic_state_preimage_fields(fields)).digest_native
}

fn semantic_state_trace_for_assignment(
    assignment: &[F],
    indices: &[usize],
) -> crate::paper::f_prime::poseidon_trace::PoseidonTraceImage {
    let values: Vec<F> = indices.iter().map(|&idx| assignment[idx]).collect();
    encode_poseidon_trace(&build_semantic_state_preimage_fields(&values))
}

pub(super) fn state_x_out_app_preimage_lanes_for_assignment(
    plan: &RecursiveStepImagePlan,
    assignment: &[F],
) -> Result<Vec<F>, R1csCompilerError> {
    let Some(state_x_out) = plan.state_x_out.as_ref() else {
        return Ok(Vec::new());
    };

    let mut lanes = Vec::new();
    for &index in &state_x_out.app_public_input_var_indices {
        lanes.push(assignment[index]);
    }

    for chunk in state_x_out.app_public_input_bit_var_indices.chunks(64) {
        let mut packed = 0u64;
        for (bit_index, &index) in chunk.iter().enumerate() {
            let value = assignment[index];
            if value == F::from_u64(0) {
                continue;
            }
            if value == F::from_u64(1) {
                packed |= 1u64 << bit_index;
                continue;
            }
            return Err(R1csCompilerError::PackedPublicInputNotBit { index, value });
        }
        lanes.push(F::from_u64(packed));
    }

    Ok(lanes)
}

fn assignment_to_plan_bits(plan: &RecursiveStepImagePlan, assignment: &[F]) -> Result<Vec<F>, R1csCompilerError> {
    if plan.app_private_var_widths.is_empty() {
        return Ok(assignment_to_bits(assignment));
    }

    let mut bits = Vec::with_capacity(plan.limbs.saturating_sub(1));
    for (index, (&value, &width)) in assignment
        .iter()
        .zip(plan.app_private_var_widths.iter())
        .enumerate()
    {
        if !(1..=POSEIDON2_GOLDILOCKS_BITS).contains(&width) {
            panic!("R1CS-F' typed app-private widths must be in 1..=64");
        }
        let raw = value.as_canonical_u64();
        if width == 1 && raw >= 2 {
            return Err(R1csCompilerError::TypedBooleanVariableNotBit { index, value });
        }
        for bit in 0..width {
            bits.push(if ((raw >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
        }
    }
    Ok(bits)
}

// ─────────────────────────────────────────────────────────────────────────
// Error.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum R1csCompilerError {
    /// Caller passed an assignment of the wrong length.
    #[error("R1CS compiler: assignment length {got} \u{2260} r1cs.m() = {expected}")]
    AssignmentLength { got: usize, expected: usize },

    /// The R1CS rejected the assignment at the given row.
    #[error("R1CS compiler: app-level R1CS unsatisfied: {0}")]
    Unsatisfied(#[source] FrontendError),

    /// The R1CS rejected the assignment at `index` within a chunk.
    #[error("R1CS compiler: app-level R1CS unsatisfied at chunk index {index}: {source}")]
    UnsatisfiedAt {
        index: usize,
        #[source]
        source: FrontendError,
    },

    #[error("R1CS compiler: stateful semantic mode requires K=1 serial chunks (got K={got})")]
    StatefulChunkMustBeSerial { got: usize },

    #[error("R1CS compiler: semantic state input digest mismatch (expected {expected:?}, got {got:?})")]
    SemanticStateInputMismatch { expected: [F; 4], got: [F; 4] },

    #[error("R1CS compiler: packed public-input variable z[{index}] is not Boolean (got {value:?})")]
    PackedPublicInputNotBit { index: usize, value: F },

    #[error("R1CS compiler: typed Boolean app variable z[{index}] is not Boolean (got {value:?})")]
    TypedBooleanVariableNotBit { index: usize, value: F },

    #[error("R1CS compiler: conventional constant lane z[0] must be ONE when the plan relies on it (got {got:?})")]
    ConstantLaneNotOne { got: F },

    /// Protocol-generic shell-level failure (preprocessing missing
    /// `public_input_len`, boundary shape mismatch, prior-fold
    /// rejection, plan shape mismatch, etc.). Carries the shell error
    /// verbatim so callers can match on the underlying variant.
    #[error(transparent)]
    Shell(#[from] FPrimeShellCompilerError),
}
