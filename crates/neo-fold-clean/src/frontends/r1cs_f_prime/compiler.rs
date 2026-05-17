//! R1CS app-step compiler — fixed-shape R1CS assignments into encoded F'.
//!
//! This frontend owns only the R1CS-specific layer: assignment length
//! checks, R1CS satisfiability, assignment-bit encoding, and binding
//! public input `x = z[..m_in]` into `state_x_out`.
//!
//! Shared F' mechanics live in
//! [`crate::frontends::f_prime_shell::compiler`]: chain state,
//! prior-fold verification, NIFS CE views, canonical plan validation,
//! and unified Poseidon trace assembly.

use std::sync::Arc;

use neo_math::F;
use thiserror::Error;

use crate::frontends::direct_ccs::FrontendError;
use crate::frontends::f_prime_shell::compiler::{
    assemble_unified_step_traces, canonical_ce_shape_and_child_count, nifs_ce_view_from_claim, perp_nifs_ce_view,
    start_f_prime_chain_context, verify_prior_fold, FPrimeChainState, FPrimeCompilerContext, FPrimeFoldForStep,
    FPrimeShellCompilerError,
};
use crate::frontends::f_prime_shell::encoder::{EncodedFPrimeStep, NifsPayloadInput};
use crate::frontends::f_prime_shell::image::{NifsCeClaimShape, NifsCeClaimView};
use crate::frontends::f_prime_shell::recursive_plan::RecursiveStepImagePlan;
use crate::frontends::r1cs_f_prime::encoder::{assignment_to_bits, encode_r1cs_f_prime_step, R1csEncoderInput};
use crate::frontends::r1cs_f_prime::R1csFPrimePreprocessing;
use crate::paper::construction2::TRIVIAL_PC;

/// `pc` for an R1CS-F' chain.
///
/// HyperNova / SuperNeo's `pc` (program counter) selects which `F'_j`
/// the chain folds. The R1CS frontend pins one R1CS shape per chain
/// (one `pc`, one `F'_j`), so `pc` is constant. We bind it through
/// `paper::construction2::TRIVIAL_PC` rather than reusing
/// `frontends::fibonacci_f_prime::FIBONACCI_PC` so the R1CS frontend
/// doesn't implicitly claim to be the Fibonacci frontend; the
/// underlying constant is the same `TRIVIAL_PC = 1`, but the name now
/// reflects which frontend committed to it.
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
}

/// R1CS-facing alias for the shared F'-shell compiler context.
pub type R1csCompilerContext = FPrimeCompilerContext;

/// Initialise a compiler context for a fresh R1CS chain.
///
/// Thin wrapper over
/// [`crate::frontends::f_prime_shell::compiler::start_f_prime_chain_context`]
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
///
/// Branches on `ctx.chain_state.chunk_count`:
/// - `0` → base step. Caller must NOT supply `ctx.fold_for_step`.
/// - `> 0` → recursive step. Caller MUST supply `ctx.fold_for_step`;
///   the compiler re-verifies the prior fold's NIFS proof under the
///   per-step F' transcript before emitting the encoded image.
pub fn compile_step(
    prep: &R1csFPrimePreprocessing,
    ctx: &mut R1csCompilerContext,
    input: R1csFPrimeStepInput,
) -> Result<R1csCompiledStep, R1csCompilerError> {
    // ── App-level satisfaction check ────────────────────────────────
    #[cfg(feature = "perf-timers")]
    let t_satisfaction = std::time::Instant::now();
    if input.assignment.len() != prep.r1cs.m() {
        return Err(R1csCompilerError::AssignmentLength {
            got: input.assignment.len(),
            expected: prep.r1cs.m(),
        });
    }
    prep.r1cs
        .is_satisfied_by(&input.assignment)
        .map_err(R1csCompilerError::Unsatisfied)?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[r1cs-compile] app satisfaction             {:>7.2}s",
        t_satisfaction.elapsed().as_secs_f64()
    );

    // ── Branch: base (chunk_count == 0) or recursive (> 0) ──────────
    let is_base = ctx.chain_state.chunk_count == 0;

    if is_base {
        if ctx.fold_for_step.is_some() {
            return Err(FPrimeShellCompilerError::BaseStepUnexpectedPriorFold.into());
        }
        compile_base_step(prep, ctx, input)
    } else {
        let fold =
            ctx.fold_for_step
                .as_ref()
                .cloned()
                .ok_or(FPrimeShellCompilerError::PriorFoldMissingForRecursiveStep {
                    chunk_count: ctx.chain_state.chunk_count,
                })?;
        // The prior-fold transcript is protocol-generic; the shared
        // shell helper performs the per-step F' transcript reconstruction
        // and NIFS verification.
        #[cfg(feature = "perf-timers")]
        let t_verify = std::time::Instant::now();
        verify_prior_fold(&prep.prep, ctx, &fold)?;
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-compile] verify_prior_fold          {:>7.2}s",
            t_verify.elapsed().as_secs_f64()
        );
        compile_recursive_step(prep, ctx, input, fold)
    }
}

fn compile_base_step(
    prep: &R1csFPrimePreprocessing,
    ctx: &mut R1csCompilerContext,
    input: R1csFPrimeStepInput,
) -> Result<R1csCompiledStep, R1csCompilerError> {
    let plan = prep.plan.clone();
    let (ce_shape, child_count) = canonical_ce_shape_and_child_count(&plan)?;

    let perp_view = perp_nifs_ce_view(&ce_shape);
    let recursive_c_data = perp_view.c_data.clone();
    finalize_compile(
        prep,
        ctx,
        input,
        plan,
        /* is_base = */ true,
        perp_view,
        recursive_c_data,
        child_count,
    )
}

fn compile_recursive_step(
    prep: &R1csFPrimePreprocessing,
    ctx: &mut R1csCompilerContext,
    input: R1csFPrimeStepInput,
    fold: R1csFoldForStep,
) -> Result<R1csCompiledStep, R1csCompilerError> {
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
    finalize_compile(
        prep,
        ctx,
        input,
        plan,
        /* is_base = */ false,
        ce_view,
        recursive_c_data,
        child_count,
    )
}

/// Compose the encoded R1CS-F' step around the shared shell assembly.
fn finalize_compile(
    prep: &R1csFPrimePreprocessing,
    ctx: &mut R1csCompilerContext,
    input: R1csFPrimeStepInput,
    plan: RecursiveStepImagePlan,
    is_base: bool,
    ce_view: NifsCeClaimView,
    recursive_c_data: Vec<F>,
    child_count: u64,
) -> Result<R1csCompiledStep, R1csCompilerError> {
    // Bind the R1CS public input `x = assignment[..m_in]` into the
    // chain's verifier-visible `state_x_out` Poseidon hash. Without
    // this binding two assignments with different `x` but the same
    // R1CS shape produce the same `public_output_digest` — the
    // verifier learns only "some assignment satisfies the R1CS,"
    // not "this specific `x` was proven."
    let app_public_input: Vec<F> = input.assignment[..prep.r1cs.m_in()].to_vec();
    #[cfg(feature = "perf-timers")]
    let t_assembly = std::time::Instant::now();
    let assembly = assemble_unified_step_traces(ctx, is_base, &recursive_c_data, child_count, &app_public_input);
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[r1cs-compile] assemble traces              {:>7.2}s",
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
    let encoder_input = R1csEncoderInput {
        plan,
        boundary_bits: assembly.boundary_bits,
        state_in: assembly.state_in,
        state_out: assembly.state_out,
        chunk_digest: assembly.chunk_digest,
        assignment_bits,
        is_base,
        nifs_payloads: vec![NifsPayloadInput::Ce(ce_view)],
        kmul_views: vec![],
        ring_action_pairs: vec![],
        one_shot_traces: vec![
            assembly.traces.boundary,
            assembly.traces.public_trace,
            assembly.traces.base_accumulator,
            assembly.traces.recursive_accumulator,
            assembly.traces.state_x_out,
        ],
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

    ctx.chain_state = assembly.next_chain_state;
    ctx.fold_for_step = None;

    Ok(R1csCompiledStep {
        encoded,
        public_output_digest: assembly.public_output_digest,
    })
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

    /// Protocol-generic shell-level failure (preprocessing missing
    /// `public_input_len`, boundary shape mismatch, prior-fold
    /// rejection, plan shape mismatch, etc.). Carries the shell error
    /// verbatim so callers can match on the underlying variant.
    #[error(transparent)]
    Shell(#[from] FPrimeShellCompilerError),
}
