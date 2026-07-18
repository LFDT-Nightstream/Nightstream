//! App-step compiler — base + recursive paths under the unified plan.
//!
//! ## Goal
//!
//! Turn a Fibonacci app step into an [`EncodedFPrimeStep`] the
//! lifecycle can fold. The shared shell
//! ([`neo_fold_clean::frontends::f_prime::compiler`]) assembles the unified
//! F' traces (chunk digest, state_in / state_out, the canonical Poseidon
//! traces, boundary bits, next chain state); this frontend supplies the
//! Fibonacci transition check, the app-state output, and the encoder
//! input (`FPrimeStepInput`) that splices the shell-built pieces with
//! Fibonacci's `app_private_carries` and `is_base` lane.
//!
//! The user's per-step API:
//!
//! - [`FibonacciAppState`] — `{ prev, curr, step_index }`
//! - [`FibonacciAppWitness`] — `{ next }`
//! - [`FibonacciAppStepInput`] — `{ state_in, witness }`
//! - [`FibonacciAppStepOutput`] — `{ state_out, public_output_digest }`
//! - [`FibonacciCompiledStep`] — `{ app_output, encoded }`
//!
//! ## Branching: base vs recursive
//!
//! Both paths emit foldable encoded F' steps under the **same canonical
//! plan** (`prep.plan`, which the verifier owns). The selector is
//! `ctx.chain_state.chunk_count`:
//!
//! - **Base step** (`chunk_count == 0`): caller must NOT pass
//!   `fold_for_step` (rejected with
//!   `FibonacciCompilerError::Shell(FPrimeShellCompilerError::BaseStepUnexpectedPriorFold)`).
//!   The shell assembles `is_base = 1` with the base accumulator handle
//!   taken from `AccumulatorHandle::empty()`; this frontend fills the NIFS
//!   payload with a **perp** view via the shell's
//!   [`perp_nifs_ce_view`][`neo_fold_clean::frontends::f_prime::compiler::perp_nifs_ce_view`]
//!   — deterministic filler, NOT authority. The unified structure's
//!   selector picks the constant base accumulator digest into
//!   `state_out.new_acc_digest`, so the perp payload's `c_data` does
//!   not enter authority via this path. (Audit: any future weakening
//!   of the selector binding must keep the perp payload out of
//!   authority.)
//!
//! - **Recursive step** (`chunk_count > 0`): caller MUST supply
//!   `fold_for_step` (rejected with
//!   `FibonacciCompilerError::Shell(FPrimeShellCompilerError::PriorFoldMissingForRecursiveStep)`
//!   otherwise). The shell's
//!   [`verify_prior_fold`][`neo_fold_clean::frontends::f_prime::compiler::verify_prior_fold`]
//!   reruns NIFS.V on the prior fold and rejects if the derived
//!   `post_running` differs from `fold.post_running`. This frontend
//!   then fills the NIFS payload via the shell's
//!   [`nifs_ce_view_from_claim`][`neo_fold_clean::frontends::f_prime::compiler::nifs_ce_view_from_claim`]
//!   from `post_running.parent_authority`'s real CE claim and the
//!   shell assembles `is_base = 0` traces. The recursive accumulator
//!   handle is derived from the full verified `post_running`, not from
//!   a short parent-commitment digest.
//!
//! ## Internal NIFS verification (recursive only)
//!
//! The shell's `verify_prior_fold` calls `paper::nifs::verify` on
//! `(fold.pre_running, fold.latest, fold.proof)` under the **per-step
//! F' transcript** (`F_PRIME_STEP_TRANSCRIPT_LABEL` + the state-bound
//! F'-step context absorbs over `ctx.chain_state`) — the same transcript
//! prefix `paper::f_prime::native::prove` initialises for the fold
//! inside the lifecycle's `extend` call. Callers must therefore source
//! `fold.proof` from a per-step `StepProof::Recursive`
//! (e.g. `audit.steps[i].fold` before finalisation), **not** from
//! `finish_uncompressed_with_audit`'s terminal fold (which lives under
//! a different transcript label).
//!
//! ## Ownership table — who computes what?
//!
//! | Field / artifact                                                                  | Owner                                                                                                                                                    | Visibility to app caller |
//! |-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
//! | Fibonacci `state_in` (`prev`, `curr`, `step_index`) / `witness` (`next`)          | App caller                                                                                                                                               | Provided                 |
//! | Fibonacci transition check (`next == prev + curr`) and app `state_out`            | This frontend                                                                                                                                            | Hidden / returned via `app_output` |
//! | Prior fold authority boundary (`pre_running`, `latest`, `proof`, `post_running`)  | Caller hands once per step via `ctx.fold_for_step` (recursive only)                                                                                      | Threaded via `ctx`       |
//! | F' chain header / state and prior-fold verification                               | `f_prime::compiler` ([`FPrimeCompilerContext`], `verify_prior_fold`)                                                                               | Hidden under `ctx`       |
//! | Image plan (NIFS shape, accumulator options, canonical CE shape validation)       | `prep.plan` + `f_prime::compiler::canonical_ce_shape_and_child_count`; recursive path shape-validates `post_running.parent_authority` against `prep.plan` and rejects on mismatch | Hidden                   |
//! | NIFS payload views                                                                | `f_prime::compiler` (`perp_nifs_ce_view` for base, `nifs_ce_view_from_claim` for recursive); selected by Fibonacci branch logic                    | Hidden                   |
//! | Poseidon trace (`state_x_out`)                                                  | `f_prime::compiler::assemble_unified_step_traces`                                                                                                  | Hidden                   |
//! | `is_base` lane + `app_private_carries`                                            | This frontend                                                                                                                                            | Hidden                   |
//! | `FPrimeStepInput` (splices shell assembly with `app_private_carries`, `is_base`, NIFS payload view) and `EncodedFPrimeStep` encoder output | This frontend                                                                                                                                            | **Never exposed** / returned via `encoded` |

use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use neo_fold_clean::frontends::f_prime::compiler::{
    assemble_unified_step_traces, canonical_ce_shape_and_child_count, nifs_ce_view_from_claim,
    nifs_payload_inputs_for_source_image, perp_nifs_ce_view, start_f_prime_chain_context, verify_prior_fold,
    FPrimeChainState, FPrimeCompilerContext, FPrimeFoldForStep, FPrimeShellCompilerError,
};
use neo_fold_clean::frontends::f_prime::encoder::{
    encode_f_prime_step_with_structure, EncodedFPrimeStep, FPrimeStepInput,
};
use neo_fold_clean::frontends::f_prime::image::NifsCeClaimView;
use neo_fold_clean::frontends::f_prime::recursive_plan::RecursiveStepImagePlan;
use neo_fold_clean::paper::construction2::TRIVIAL_PC;
use neo_fold_clean::paper::digest::AccumulatorHandle;

// ─────────────────────────────────────────────────────────────────────────
// App surface — the only types the caller writes per step.
// ─────────────────────────────────────────────────────────────────────────

/// Fibonacci app state at the boundary of one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FibonacciAppState {
    pub prev: F,
    pub curr: F,
    pub step_index: u64,
}

/// Fibonacci app witness for one step. `next == prev + curr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FibonacciAppWitness {
    pub next: F,
}

/// Per-step input to [`compile_fibonacci_step`].
#[derive(Clone, Copy, Debug)]
pub struct FibonacciAppStepInput {
    pub state_in: FibonacciAppState,
    pub witness: FibonacciAppWitness,
}

/// App-level output of one compile.
#[derive(Clone, Copy, Debug)]
pub struct FibonacciAppStepOutput {
    pub state_out: FibonacciAppState,
    pub public_output_digest: [F; 4],
}

/// Combined output of [`compile_fibonacci_step`].
#[derive(Debug)]
pub struct FibonacciCompiledStep {
    pub app_output: FibonacciAppStepOutput,
    pub encoded: EncodedFPrimeStep,
}

// ─────────────────────────────────────────────────────────────────────────
// Compiler context.
//
// The state types live in `neo_fold_clean::frontends::f_prime::compiler`
// because every F' app frontend (Fibonacci, R1CS, …) threads the same
// shape. Fibonacci re-exports them under app-named aliases so call
// sites keep their existing names.
// ─────────────────────────────────────────────────────────────────────────

/// Fibonacci-facing alias for the shared F'-shell compiler context.
pub type FibonacciCompilerContext = FPrimeCompilerContext;

/// Fibonacci-facing alias for the shared F'-shell chain state.
pub type FibonacciChainState = FPrimeChainState;

/// Fibonacci-facing alias for the shared F'-shell fold authority.
pub type FibonacciFoldForStep = FPrimeFoldForStep;

// ─────────────────────────────────────────────────────────────────────────
// Public entrypoints.
// ─────────────────────────────────────────────────────────────────────────

/// Initialize a [`FibonacciCompilerContext`] for a fresh chain.
///
/// Thin wrapper over
/// [`neo_fold_clean::frontends::f_prime::compiler::start_f_prime_chain_context`]
/// that pins Fibonacci's `pc`; the app-private width comes from the
/// verifier-owned plan so tiny test fixtures can pad the source image
/// without changing Fibonacci's public transition.
pub fn start_fibonacci_chain(
    prep: &super::FibonacciFPrimePreprocessing,
) -> Result<FibonacciCompilerContext, FibonacciCompilerError> {
    Ok(start_f_prime_chain_context(&prep.prep, FIBONACCI_PC, prep.plan.limbs)?)
}

/// Compile one Fibonacci app step into a foldable
/// [`EncodedFPrimeStep`].
///
/// Branches on `ctx.chain_state.chunk_count`:
/// - `0` → base step. Caller must NOT supply `ctx.fold_for_step`.
/// - `> 0` → recursive step. Caller MUST supply `ctx.fold_for_step`.
///
/// **Recursive step — internal NIFS verification:** the compiler calls
/// `nifs::verify` on `(fold.pre_running, fold.latest, fold.proof)` and
/// rejects if the derived running differs from `fold.post_running`.
/// This means a mutated proof (NIFS rejection) or a mutated
/// `post_running` (post-state mismatch) is caught by the compiler,
/// not by downstream lifecycle verification. The transcript is the
/// **per-step F' transcript** — see [`verify_prior_fold`] for the
/// reconstruction rule; callers must source `fold.proof` from a
/// per-step `StepProof::Recursive`, not from
/// `finish_uncompressed_with_audit`'s terminal fold.
///
/// On success, returns [`FibonacciCompiledStep`] and updates
/// `ctx.chain_state` for the next call. The caller is responsible for
/// writing the next `fold_for_step` before the next recursive compile.
pub fn compile_fibonacci_step(
    prep: &super::FibonacciFPrimePreprocessing,
    ctx: &mut FibonacciCompilerContext,
    input: FibonacciAppStepInput,
) -> Result<FibonacciCompiledStep, FibonacciCompilerError> {
    // ── Transition check ────────────────────────────────────────────────
    let expected_next = input.state_in.prev + input.state_in.curr;
    if input.witness.next != expected_next {
        return Err(FibonacciCompilerError::TransitionMismatch {
            got: input.witness.next,
            expected: expected_next,
        });
    }

    // ── Branch: base (chunk_count == 0) or recursive (> 0) ──────────────
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
        // Re-runs NIFS.V on (pre_running, latest, proof) under the
        // **per-step F' transcript** the lifecycle's `paper::f_prime::native::prove`
        // uses for this same fold. Catches:
        // - mutated NIFS proof (NIFS.V rejects → PriorFoldVerificationFailed)
        // - mutated post_running (claims / parent_authority differ
        //   from derived → PriorFoldPostRunningMismatch)
        verify_prior_fold(&prep.prep, ctx, &fold, 1)?;
        compile_recursive_step(prep, ctx, input, fold)
    }
}

fn compile_base_step(
    prep: &super::FibonacciFPrimePreprocessing,
    ctx: &mut FibonacciCompilerContext,
    input: FibonacciAppStepInput,
) -> Result<FibonacciCompiledStep, FibonacciCompilerError> {
    let plan = prep.plan.clone();
    let (ce_shape, child_count) = canonical_ce_shape_and_child_count(&plan)?;

    // Base step: perp NIFS payload, `is_base = 1`. The perp payload is
    // deterministic filler only; the authoritative accumulator handle
    // is the empty base handle selected by the shell.
    let perp_view = perp_nifs_ce_view(&ce_shape);
    let new_acc_digest = AccumulatorHandle::empty().digest_fields();
    finalize_compile(
        prep,
        ctx,
        input,
        plan,
        /* is_base = */ true,
        perp_view,
        new_acc_digest,
        child_count,
    )
}

fn compile_recursive_step(
    prep: &super::FibonacciFPrimePreprocessing,
    ctx: &mut FibonacciCompilerContext,
    input: FibonacciAppStepInput,
    fold: FibonacciFoldForStep,
) -> Result<FibonacciCompiledStep, FibonacciCompilerError> {
    let post_running = &fold.post_running;
    let post_parent = post_running
        .parent_authority
        .as_ref()
        .ok_or(FPrimeShellCompilerError::PostRunningMissingParentAuthority)?;

    // Recursive path uses **the same canonical plan** as the base
    // path. This is HyperNova Construction 2's fixed-`pc` invariant:
    // for one `pc`, base and recursive branches live in one `F'_j`
    // structure, so both compiler paths must agree on `prep.plan`.
    let plan = prep.plan.clone();
    let (canonical_ce_shape, child_count) = canonical_ce_shape_and_child_count(&plan)?;

    // Shape guard: the prover's `post_parent` must already match the
    // verifier-owned canonical CE shape. Truncating or padding would
    // either drop authority (`c_data`, `r`, `y_ring`, …) or write
    // unrelated values into committed payload bits — neither is safe.
    let actual_shape = neo_fold_clean::frontends::f_prime::image::NifsCeClaimShape {
        c_data_entries: post_parent.c.data.len(),
        x_rows: post_parent.X.rows(),
        x_active_cols: neo_fold_clean::paper::relations::superneo_public_x_cols(post_parent.m_in),
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
    // child_count is a baked Constant in the recursive accumulator's
    // preimage source list. A mismatch with the real
    // `post_running.claims.len()` would still produce a syntactically
    // valid F' image, but the recursive accumulator trace would hash
    // `H(tag, plan.child_count, ...)` while the lifecycle's authority
    // hashes `H(tag, running.claims.len(), ...)` — the two
    // `new_acc_digest`s disagree and downstream chain consistency
    // breaks. Reject here so the failure mode is named.
    if post_running.claims.len() as u64 != child_count {
        return Err(FPrimeShellCompilerError::PostRunningClaimsCountMismatch {
            canonical: child_count,
            actual: post_running.claims.len() as u64,
        }
        .into());
    }

    let ce_view = nifs_ce_view_from_claim(post_parent, ctx.public_input_len);
    let new_acc_digest = AccumulatorHandle::from_running_parts(&post_running.claims, Some(post_parent)).digest_fields();
    finalize_compile(
        prep,
        ctx,
        input,
        plan,
        /* is_base = */ false,
        ce_view,
        new_acc_digest,
        child_count,
    )
}

/// Compose the encoded F' step around the shared shell assembly.
///
/// `new_acc_digest` is the exact ordered-child accumulator handle carried
/// through state and checked by the consumer step/terminal fold, so the
/// producer image does not emit a dedicated accumulator Poseidon trace.
fn finalize_compile(
    prep: &super::FibonacciFPrimePreprocessing,
    ctx: &mut FibonacciCompilerContext,
    input: FibonacciAppStepInput,
    plan: RecursiveStepImagePlan,
    is_base: bool,
    ce_view: NifsCeClaimView,
    new_acc_digest: [F; 4],
    _child_count: u64,
) -> Result<FibonacciCompiledStep, FibonacciCompilerError> {
    // Fibonacci does not bind app public input into `state_x_out` — the
    // boundary digest alone commits to the chain's verifier-visible
    // output. R1CS passes its public assignment prefix here instead.
    // Fibonacci is K=1 per step (one app row per F' step).
    let assembly = assemble_unified_step_traces(ctx, is_base, new_acc_digest, &[], 1);

    let nifs_payloads = nifs_payload_inputs_for_source_image(&plan, ce_view);
    let encoder_input = FPrimeStepInput {
        plan,
        boundary_bits: assembly.boundary_bits,
        state_in: assembly.state_in,
        state_out: assembly.state_out,
        chunk_digest: assembly.chunk_digest,
        // `limbs - 1` carry bits; for canonical Fibonacci limbs=3 → 2
        // carries. The canonical plan does not yet algebraically bind
        // carries to the Fibonacci limb arithmetic (`witness.next =
        // prev + curr`); they're free witness bits at this layer. We
        // populate honest zeros (the canonical low-norm choice).
        app_private_carries: vec![F::ZERO; ctx.limbs.saturating_sub(1)],
        is_base,
        nifs_payloads,
        kmul_views: vec![],
        ring_action_pairs: vec![],
        one_shot_traces: vec![assembly.traces.state_x_out],
        sponge_trace: None,
    };

    let encoded = encode_f_prime_step_with_structure(encoder_input, prep.structure.clone());

    ctx.chain_state = assembly.next_chain_state;
    ctx.fold_for_step = None;

    let app_state_out = FibonacciAppState {
        prev: input.state_in.curr,
        curr: input.witness.next,
        step_index: input.state_in.step_index + 1,
    };

    Ok(FibonacciCompiledStep {
        app_output: FibonacciAppStepOutput {
            state_out: app_state_out,
            public_output_digest: assembly.public_output_digest,
        },
        encoded,
    })
}

// ─────────────────────────────────────────────────────────────────────────
// Error.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum FibonacciCompilerError {
    /// Fibonacci app-level transition rejected:
    /// `witness.next != state_in.prev + state_in.curr`.
    #[error("fibonacci compiler: witness.next ({got}) \u{2260} state_in.prev + state_in.curr ({expected})")]
    TransitionMismatch { got: F, expected: F },

    /// Protocol-generic shell-level failure (prior-fold rejection, plan
    /// shape mismatch, missing parent authority, etc.). Carries the
    /// shell error verbatim so callers can match on the underlying
    /// variant.
    #[error(transparent)]
    Shell(#[from] FPrimeShellCompilerError),
}

// ─────────────────────────────────────────────────────────────────────────
// Module-level constants.
// ─────────────────────────────────────────────────────────────────────────

/// Default `pc` for a Fibonacci F' chain — ℓ = 1, so `pc = TRIVIAL_PC`.
pub const FIBONACCI_PC: u64 = TRIVIAL_PC;
