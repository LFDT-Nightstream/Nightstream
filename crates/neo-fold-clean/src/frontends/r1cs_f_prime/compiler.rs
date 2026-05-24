//! R1CS app-step compiler — fixed-shape R1CS assignments into encoded F'.
//!
//! This frontend owns only the R1CS-specific layer: assignment length
//! checks, R1CS satisfiability, assignment-bit encoding, and binding
//! public input `x = z[..m_in]` into `state_x_out`.
//!
//! Shared F' mechanics live in
//! [`crate::frontends::f_prime::compiler`]: chain state,
//! prior-fold verification, NIFS CE views, canonical plan validation,
//! and unified Poseidon trace assembly.

use std::sync::Arc;

use neo_math::F;
use thiserror::Error;

use crate::frontends::direct_ccs::FrontendError;
use crate::frontends::f_prime::compiler::{
    assemble_shared_chunk_traces, assemble_step_from_shared, canonical_ce_shape_and_child_count,
    nifs_ce_view_from_claim, perp_nifs_ce_view, start_f_prime_chain_context, verify_prior_fold, FPrimeChainState,
    FPrimeCompilerContext, FPrimeFoldForStep, FPrimeShellCompilerError,
};
use crate::frontends::f_prime::encoder::{EncodedFPrimeStep, NifsPayloadInput};
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
    let semantic = semantic_state_digests_for_inputs(prep, &inputs)?;
    if let Some(semantic) = semantic {
        if ctx.chain_state.chunk_count > 0 && ctx.chain_state.semantic_state_digest != semantic.input {
            return Err(R1csCompilerError::SemanticStateInputMismatch {
                expected: ctx.chain_state.semantic_state_digest,
                got: semantic.input,
            });
        }
    }

    // ── Branch: base (chunk_count == 0) or recursive (> 0) ──────────
    let is_base = ctx.chain_state.chunk_count == 0;

    if is_base {
        if ctx.fold_for_step.is_some() {
            return Err(FPrimeShellCompilerError::BaseStepUnexpectedPriorFold.into());
        }
        compile_base_chunk(prep, ctx, inputs, rows_in_chunk, semantic)
    } else {
        let fold =
            ctx.fold_for_step
                .as_ref()
                .cloned()
                .ok_or(FPrimeShellCompilerError::PriorFoldMissingForRecursiveStep {
                    chunk_count: ctx.chain_state.chunk_count,
                })?;
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
        compile_recursive_chunk(prep, ctx, inputs, fold, rows_in_chunk, semantic)
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
    let recursive_c_data = perp_view.c_data.clone();
    finalize_compile_chunk(
        prep,
        ctx,
        inputs,
        plan,
        /* is_base = */ true,
        perp_view,
        recursive_c_data,
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
    let post_running = &fold.post_running;
    let post_parent = post_running
        .parent_authority
        .as_ref()
        .ok_or(FPrimeShellCompilerError::PostRunningMissingParentAuthority)?;

    let plan = prep.plan.clone();
    let (canonical_ce_shape, child_count) = canonical_ce_shape_and_child_count(&plan)?;

    let actual_shape = NifsCeClaimShape {
        c_data_entries: post_parent.c.data.len(),
        x_rows: post_parent.X.rows(),
        x_active_cols: crate::paper::relations::superneo_public_x_cols(post_parent.m_in),
        r_len: post_parent.r.len(),
        y_ring_inner_lens: post_parent.y_ring.iter().map(|row| row.len()).collect(),
        y_zcol_len: post_parent.y_zcol.len(),
        s_col_len: post_parent.s_col.len(),
    };
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

    let ce_view = nifs_ce_view_from_claim(post_parent, ctx.public_input_len);
    let recursive_c_data = ce_view.c_data.clone();
    finalize_compile_chunk(
        prep,
        ctx,
        inputs,
        plan,
        /* is_base = */ false,
        ce_view,
        recursive_c_data,
        child_count,
        rows_in_chunk,
        semantic,
    )
}

/// Compose K encoded R1CS-F' steps around the shared shell assembly.
/// Every step in the chunk sees the **same** pre-step `ctx` and the
/// **same** `rows_in_chunk`, so the K post-step chain coordinates are
/// identical. Each step's `state_x_out` still binds its own
/// `app_public_input`, giving K distinct `public_output_digest`s
/// (the verifier learns "this specific `x_i` was proven" for each).
/// The chain advances **once** at the end of the chunk.
fn finalize_compile_chunk(
    prep: &R1csFPrimePreprocessing,
    ctx: &mut R1csCompilerContext,
    inputs: Vec<R1csFPrimeStepInput>,
    plan: RecursiveStepImagePlan,
    is_base: bool,
    ce_view: NifsCeClaimView,
    recursive_c_data: Vec<F>,
    child_count: u64,
    rows_in_chunk: usize,
    semantic: Option<SemanticStateDigests>,
) -> Result<Vec<R1csCompiledStep>, R1csCompilerError> {
    debug_assert_eq!(inputs.len(), rows_in_chunk);
    if let Some(semantic) = semantic {
        if ctx.chain_state.chunk_count == 0 {
            ctx.chain_state.semantic_state_digest = semantic.input;
        } else if ctx.chain_state.semantic_state_digest != semantic.input {
            return Err(R1csCompilerError::SemanticStateInputMismatch {
                expected: ctx.chain_state.semantic_state_digest,
                got: semantic.input,
            });
        }
    }

    // Compute the chunk-shared traces ONCE: chunk_digest, the four
    // app-input-independent Poseidon traces (boundary, public_trace,
    // base / recursive accumulator), and the post-step chain advance.
    // These are identical for every assignment in the chunk, so
    // recomputing them per assignment (the previous behavior) was the
    // dominant redundant compile cost.
    #[cfg(feature = "perf-timers")]
    let t_shared = std::time::Instant::now();
    let shared = assemble_shared_chunk_traces(ctx, is_base, &recursive_c_data, child_count, rows_in_chunk);
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[r1cs-compile] shared chunk traces (once)   {:>7.2}s",
        t_shared.elapsed().as_secs_f64()
    );

    let mut compiled = Vec::with_capacity(rows_in_chunk);
    for input in inputs.into_iter() {
        // Bind this assignment's R1CS public input `x =
        // assignment[..m_in]` into its `state_x_out` Poseidon hash.
        // For a K-deposit step every assignment shares the post-step
        // chain coordinates but absorbs its own `x`, so the K
        // boundary digests are distinct and the verifier learns each
        // specific `x_i`. Only `state_x_out` / `boundary_bits` are
        // recomputed here; the four shared traces are cloned.
        let app_public_input: Vec<F> = input.assignment[..prep.r1cs.m_in()].to_vec();
        #[cfg(feature = "perf-timers")]
        let t_assembly = std::time::Instant::now();
        let semantic_out = semantic.map(|s| s.output);
        let assembly = assemble_step_from_shared(&shared, ctx, &app_public_input, semantic_out);
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-compile] step state_x_out             {:>7.2}s",
            t_assembly.elapsed().as_secs_f64()
        );

        #[cfg(feature = "perf-timers")]
        let t_bits = std::time::Instant::now();
        let assignment_bits = assignment_to_bits(&input.assignment);
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-compile] assignment_to_bits           {:>7.2}s",
            t_bits.elapsed().as_secs_f64()
        );
        let mut one_shot_traces = vec![
            assembly.traces.boundary,
            assembly.traces.public_trace,
            assembly.traces.base_accumulator,
            assembly.traces.recursive_accumulator,
        ];
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
            }
        }
        one_shot_traces.push(assembly.traces.state_x_out);
        let encoder_input = R1csEncoderInput {
            plan: plan.clone(),
            boundary_bits: assembly.boundary_bits,
            state_in: assembly.state_in,
            state_out: assembly.state_out,
            chunk_digest: assembly.chunk_digest,
            assignment_bits,
            is_base,
            nifs_payloads: vec![NifsPayloadInput::Ce(ce_view.clone())],
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

    Ok(compiled)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SemanticStateDigests {
    pub input: [F; 4],
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
    let has_output = !state_x_out.semantic_state_out_var_indices.is_empty();
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
    Ok(Some(SemanticStateDigests {
        input: semantic_state_digest_for_assignment(assignment, &state_x_out.semantic_state_in_var_indices),
        output: semantic_state_digest_for_assignment(assignment, &state_x_out.semantic_state_out_var_indices),
    }))
}

pub(crate) fn semantic_state_digest_for_assignment(assignment: &[F], indices: &[usize]) -> [F; 4] {
    semantic_state_trace_for_assignment(assignment, indices).digest_native
}

fn semantic_state_trace_for_assignment(
    assignment: &[F],
    indices: &[usize],
) -> crate::paper::f_prime::poseidon_trace::PoseidonTraceImage {
    let values: Vec<F> = indices.iter().map(|&idx| assignment[idx]).collect();
    encode_poseidon_trace(&build_semantic_state_preimage_fields(&values))
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

    /// Protocol-generic shell-level failure (preprocessing missing
    /// `public_input_len`, boundary shape mismatch, prior-fold
    /// rejection, plan shape mismatch, etc.). Carries the shell error
    /// verbatim so callers can match on the underlying variant.
    #[error(transparent)]
    Shell(#[from] FPrimeShellCompilerError),
}
