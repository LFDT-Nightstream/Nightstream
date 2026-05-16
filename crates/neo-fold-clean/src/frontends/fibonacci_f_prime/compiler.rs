//! App-step compiler — base + recursive paths under the unified plan.
//!
//! ## Goal
//!
//! Turn a Fibonacci app step into an [`EncodedFibonacciFPrimeStep`] the
//! lifecycle can fold, without exposing the internal F' encoding
//! machinery (region plans, NIFS payload views, Karatsuba views,
//! Poseidon trace splices) to the caller.
//!
//! The user's per-step API:
//!
//! - [`FibonacciAppState`] — `{ prev, curr, step_index }`
//! - [`FibonacciAppWitness`] — `{ next }`
//! - [`FibonacciAppStepInput`] — `{ state_in, witness }`
//! - [`FibonacciAppStepOutput`] — `{ state_out, public_output_digest }`
//! - [`FibonacciCompiledStep`] — `{ app_output, encoded }`
//!
//! Everything else is owned by the compiler.
//!
//! ## Branching: base vs recursive
//!
//! Both paths emit foldable encoded F' steps under the **same canonical
//! plan** (`prep.plan`, which the verifier owns). The selector is
//! `ctx.chain_state.chunk_count`:
//!
//! - **Base step** (`chunk_count == 0`): caller must NOT pass
//!   `fold_for_step` (rejected with
//!   [`FibonacciCompilerError::BaseStepUnexpectedPriorFold`]). Compiler
//!   emits `is_base = 1`, writes
//!   `new_acc_digest = digest32_as_fields(accumulator_digest_from_claims(b, &[]))`
//!   (the empty-accumulator digest), and fills the NIFS payload with a
//!   **perp** view — deterministic filler, NOT authority. The unified
//!   structure's selector picks the base accumulator trace's digest
//!   into `state_out.new_acc_digest`, so the perp payload's `c_data`
//!   does not enter authority via this path. (Audit: any future
//!   weakening of the selector binding must keep the perp payload out
//!   of authority. See [`perp_nifs_ce_view`] for the contract.)
//!
//! - **Recursive step** (`chunk_count > 0`): caller MUST supply
//!   `fold_for_step` (rejected with
//!   [`FibonacciCompilerError::PriorFoldMissingForRecursiveStep`]
//!   otherwise). The compiler verifies the prior fold via internal
//!   `nifs::verify`, then populates the NIFS payload from
//!   `post_running.parent_authority`'s real CE claim, emits
//!   `is_base = 0`, and writes the recursive accumulator's digest as
//!   `new_acc_digest`.
//!
//! ## Internal NIFS verification (recursive only)
//!
//! The compiler calls `paper::nifs::verify` on
//! `(fold.pre_running, fold.latest, fold.proof)` and rejects if the
//! derived post-fold running differs from `fold.post_running`. The
//! verification uses the **per-step F' transcript**
//! (`F_PRIME_STEP_TRANSCRIPT_LABEL` + the six F'-step context absorbs
//! over `ctx.chain_state`) — the same transcript prefix
//! `paper::f_prime::native::prove` initialises for the fold inside the
//! lifecycle's `extend` call. Callers must therefore source
//! `fold.proof` from a per-step `StepProof::Recursive`
//! (e.g. `audit.steps[i].fold` before finalisation), **not** from
//! `finish_uncompressed_with_audit`'s terminal fold (which lives under
//! a different transcript label).
//!
//! ## Ownership table — who computes what?
//!
//! | Field / artifact                                  | Owner                                  | Visibility to app caller |
//! |---------------------------------------------------|----------------------------------------|--------------------------|
//! | Fibonacci `state_in` (`prev`, `curr`, `step_index`) | App caller                             | Provided                 |
//! | Fibonacci `witness` (`next`)                      | App caller                             | Provided                 |
//! | Fibonacci transition check (`next == prev + curr`) | Compiler                               | Hidden                   |
//! | Fibonacci `state_out`                             | Compiler                               | Returned via `app_output` |
//! | F' chain header (`vk_fs_digest`, `structure_digest`, `z_0`, `pc`) | Verifier (set once in [`FibonacciCompilerContext`]) | Hidden under `ctx`       |
//! | F' chain state (`chunk_count`, `step_count`, `z_i`, `acc_digest`, `public_trace`) | Compiler (threaded inside `ctx`)       | Hidden under `ctx`       |
//! | `chunk_digest` for the step                       | Compiler (shape-only digest)           | Hidden                   |
//! | Prior fold authority boundary (`pre_running`, `latest`, `proof`, `post_running`) | Caller hands once per step via `ctx.fold_for_step` (recursive only) | Threaded via `ctx`       |
//! | Image plan (NIFS shape, accumulator options)      | `prep.plan` — same canonical plan for base and recursive paths (HyperNova Construction 2 fixed-`pc` invariant). Recursive path shape-validates `post_running.parent_authority` against `prep.plan` and rejects on mismatch. | Hidden                   |
//! | NIFS payload views                                | **Base:** [`perp_nifs_ce_view`] over `prep.plan`'s shape. **Recursive:** [`nifs_ce_view_from_claim`] over `post_running.parent_authority` | Hidden                   |
//! | `is_base` lane                                    | Compiler (`true` for base, `false` for recursive) | Hidden                   |
//! | Poseidon traces (boundary, public_trace, base_acc, recursive_acc, state_x_out) | Compiler (5 traces under the unified plan) | Hidden                   |
//! | `app_private_carries`                             | Compiler                               | Hidden                   |
//! | `FibonacciFPrimeStepInput`                        | Internal artifact assembled by compiler | **Never exposed**        |
//! | `EncodedFibonacciFPrimeStep`                      | Compiler output                        | Returned via `encoded`   |

use neo_ajtai::Commitment;
use neo_ccs::matrix::Mat;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use crate::frontends::fibonacci_f_prime::encoder::{
    encode_fibonacci_f_prime_step, EncodedFibonacciFPrimeStep, FibonacciFPrimeStepInput, NifsPayloadInput,
};
use crate::frontends::fibonacci_f_prime::image::{
    NifsCeClaimShape, NifsCeClaimView, NifsPayloadShape, StateIn, StateOut,
};
use crate::frontends::fibonacci_f_prime::recursive_plan::{
    build_accumulator_preimage_fields, build_boundary_update_preimage_fields,
    build_public_trace_update_preimage_fields, build_state_x_out_preimage_fields, RecursiveStepImagePlan,
};
use crate::paper::construction2::{LatestInstance, ProofState, RunningInstance, State as PaperState, TRIVIAL_PC};
use crate::paper::digest::{
    accumulator_digest_from_claims, digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest,
    initial_boundary_digest, public_trace_seed_digest, structure_digest,
};
use crate::paper::f_prime::native::f_prime_step_transcript;
use crate::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use crate::paper::nifs::NifsProof;
use crate::paper::relations::{CcsClaim, CeClaim};

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
    pub encoded: EncodedFibonacciFPrimeStep,
}

// ─────────────────────────────────────────────────────────────────────────
// Compiler context.
// ─────────────────────────────────────────────────────────────────────────

/// Verifier-owned compiler context, built once per chain and threaded
/// through every [`compile_fibonacci_step`] call.
///
/// Carries the chain header (constant) plus the F' chain state (mutated
/// per step) plus the prior fold's authority boundary (`None` at step
/// 0; `Some(..)` for every subsequent step). Production callers
/// construct via [`start_fibonacci_chain`].
#[derive(Clone, Debug)]
pub struct FibonacciCompilerContext {
    // Chain header — constant across steps.
    pub vk_fs_digest: [F; 4],
    pub structure_digest: [F; 4],
    pub z_0: [F; 4],
    pub pc: u64,
    pub public_input_len: usize,
    pub commitment_d: usize,
    pub commitment_kappa: usize,
    pub boundary_bits: usize,
    pub limbs: usize,
    // Threaded F' chain state — compiler updates each step.
    pub chain_state: FibonacciChainState,
    // Prior fold authority boundary — caller writes between steps.
    pub fold_for_step: Option<FibonacciFoldForStep>,
}

/// F'-level chain state threaded between successive compile calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FibonacciChainState {
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_i: [F; 4],
    pub acc_digest: [F; 4],
    pub public_trace: [F; 4],
}

/// Authority boundary of the prior fold the compiler embeds into this
/// step's encoded image.
///
/// All four components matter:
/// - `pre_running` and `latest` are the **inputs** to the prior fold,
/// - `proof` is the NIFS witness that authorises the transition,
/// - `post_running` is the **output**, and is what this F' step
///   actually commits to via its NIFS payload + accumulator hash.
///
/// The recursive plan is derived from `post_running.parent_authority`'s
/// real CE shape; the NIFS payload view is filled from the same
/// post-fold parent.
#[derive(Clone, Debug)]
pub struct FibonacciFoldForStep {
    pub pre_running: RunningInstance,
    pub latest: LatestInstance,
    pub proof: NifsProof,
    pub post_running: RunningInstance,
}

// ─────────────────────────────────────────────────────────────────────────
// Public entrypoints.
// ─────────────────────────────────────────────────────────────────────────

/// Initialize a [`FibonacciCompilerContext`] for a fresh chain.
///
/// Derives the chain header (vk_fs_digest, structure_digest, z_0, pc)
/// from `prep` and sets the chain state to the base case (counters at
/// 0, `z_i = z_0`, empty accumulator digest, public-trace seed,
/// `fold_for_step = None`).
pub fn start_fibonacci_chain(
    prep: &super::FibonacciFPrimePreprocessing,
) -> Result<FibonacciCompilerContext, FibonacciCompilerError> {
    let structure_digest = structure_digest(&prep.prep.structure);
    let public_input_len = prep
        .prep
        .public_input_len
        .ok_or(FibonacciCompilerError::PreprocessingMissingPublicInputLen)?;
    let z_0 = digest32_as_fields(initial_boundary_digest(&structure_digest, Some(public_input_len)));
    let public_trace = digest32_as_fields(public_trace_seed_digest(&structure_digest));
    let acc_digest = digest32_as_fields(accumulator_digest_from_claims(prep.prep.params.b(), &[]));
    let vk_fs_digest = digest32_as_fields(prep.prep.vk.digest());

    // Boundary bits = 4 lanes × POSEIDON2_GOLDILOCKS_BITS = 256.
    let boundary_bits = 4 * POSEIDON2_GOLDILOCKS_BITS;
    if public_input_len != 1 + boundary_bits {
        return Err(FibonacciCompilerError::UnsupportedPublicInputShape {
            got: public_input_len,
            expected: 1 + boundary_bits,
        });
    }

    Ok(FibonacciCompilerContext {
        vk_fs_digest,
        structure_digest,
        z_0,
        pc: FIBONACCI_PC,
        public_input_len,
        commitment_d: neo_math::D,
        commitment_kappa: prep.prep.params.kappa() as usize,
        boundary_bits,
        // Fibonacci canonical: 3 limbs ⇒ 2 carry bits.
        limbs: 3,
        chain_state: FibonacciChainState {
            chunk_count: 0,
            step_count: 0,
            z_i: z_0,
            acc_digest,
            public_trace,
        },
        fold_for_step: None,
    })
}

/// Compile one Fibonacci app step into a foldable
/// [`EncodedFibonacciFPrimeStep`].
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
            return Err(FibonacciCompilerError::BaseStepUnexpectedPriorFold);
        }
        compile_base_step(prep, ctx, input)
    } else {
        let fold =
            ctx.fold_for_step
                .as_ref()
                .cloned()
                .ok_or(FibonacciCompilerError::PriorFoldMissingForRecursiveStep {
                    chunk_count: ctx.chain_state.chunk_count,
                })?;
        // Re-runs NIFS.V on (pre_running, latest, proof) under the
        // **per-step F' transcript** the lifecycle's `paper::f_prime::native::prove`
        // uses for this same fold. Catches:
        // - mutated NIFS proof (NIFS.V rejects → PriorFoldVerificationFailed)
        // - mutated post_running (claims / parent_authority differ
        //   from derived → PriorFoldPostRunningMismatch)
        verify_prior_fold(prep, ctx, &fold)?;
        compile_recursive_step(prep, ctx, input, fold)
    }
}

/// Re-derive `post_running` from `(pre_running, latest, proof)` via
/// `nifs::verify`. Reject on (a) NIFS rejection or (b) derived ≠
/// caller-supplied `post_running`.
///
/// The transcript is the per-step F' transcript
/// (`F_PRIME_STEP_TRANSCRIPT_LABEL` plus the six F'-step context
/// absorbs over `ctx.chain_state` + this step's `chunk_digest`), so it
/// matches what `paper::f_prime::native::prove` initialised for the
/// same fold. Callers must therefore source `fold.proof` from a per-step
/// `StepProof::Recursive` (e.g. `audit.steps[i].fold`) — terminal-fold
/// proofs from `finish_uncompressed_with_audit` use a different
/// transcript label and will be rejected here.
fn verify_prior_fold(
    prep: &super::FibonacciFPrimePreprocessing,
    ctx: &FibonacciCompilerContext,
    fold: &FibonacciFoldForStep,
) -> Result<(), FibonacciCompilerError> {
    // Reconstruct the per-step F' transcript prefix. `proof` carries an
    // `Initial` placeholder because `f_prime_step_transcript` only reads
    // digest fields (z_0, z_i, …); the actual proof variant is irrelevant
    // to the absorb sequence.
    let state_in = PaperState {
        chunk_count: ctx.chain_state.chunk_count,
        step_count: ctx.chain_state.step_count,
        z_0: digest_fields_as_digest32(ctx.z_0),
        z_i: digest_fields_as_digest32(ctx.chain_state.z_i),
        pc: ctx.pc,
        acc_digest: digest_fields_as_digest32(ctx.chain_state.acc_digest),
        public_trace: digest_fields_as_digest32(ctx.chain_state.public_trace),
        proof: ProofState::Initial,
    };
    let chunk_digest = chunk_digest_for_shape(
        ctx.chain_state.step_count,
        ctx.commitment_d,
        ctx.commitment_kappa,
        ctx.public_input_len,
    );
    let mut tr = f_prime_step_transcript(&prep.prep.vk, &prep.prep.structure, &state_in, chunk_digest);

    let derived = crate::paper::nifs::verify(
        &mut tr,
        &prep.prep.params,
        &prep.prep.structure,
        prep.prep.mix_rhos_commits,
        prep.prep.combine_b_pows,
        &fold.latest.claims(),
        &fold.pre_running,
        &fold.proof,
    )
    .map_err(|e| FibonacciCompilerError::PriorFoldVerificationFailed { reason: e.to_string() })?;

    // NIFS.V returns claims + parent_authority; witnesses are
    // prover-side only. Compare the authority-bearing pair.
    if derived.claims != fold.post_running.claims || derived.parent_authority != fold.post_running.parent_authority {
        return Err(FibonacciCompilerError::PriorFoldPostRunningMismatch);
    }
    Ok(())
}

fn compile_base_step(
    prep: &super::FibonacciFPrimePreprocessing,
    ctx: &mut FibonacciCompilerContext,
    input: FibonacciAppStepInput,
) -> Result<FibonacciCompiledStep, FibonacciCompilerError> {
    let plan = prep.plan.clone();
    let (ce_shape, child_count) = {
        let acc = plan
            .accumulator
            .as_ref()
            .ok_or(FibonacciCompilerError::CanonicalPlanMissingAccumulator)?;
        if !acc.unified {
            return Err(FibonacciCompilerError::CanonicalPlanNotUnified);
        }
        let ce_shape = match &plan.nifs_payload_shapes[acc.ce_claim_payload_index] {
            NifsPayloadShape::CeClaim(s) => s.clone(),
            _ => return Err(FibonacciCompilerError::CanonicalPlanPayloadNotCeClaim),
        };
        (ce_shape, acc.child_count)
    };

    // Base step: perp NIFS payload, `is_base = 1`. The recursive
    // accumulator trace must still be emitted (unified mode binds its
    // digest output); we feed it the perp payload's zero `c_data` so
    // the trace is satisfying but the selector discards it.
    let perp_view = perp_nifs_ce_view(&ce_shape);
    let recursive_c_data = perp_view.c_data.clone();
    finalize_compile(
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
    prep: &super::FibonacciFPrimePreprocessing,
    ctx: &mut FibonacciCompilerContext,
    input: FibonacciAppStepInput,
    fold: FibonacciFoldForStep,
) -> Result<FibonacciCompiledStep, FibonacciCompilerError> {
    let post_running = &fold.post_running;
    let post_parent = post_running
        .parent_authority
        .as_ref()
        .ok_or(FibonacciCompilerError::PostRunningMissingParentAuthority)?;

    // Recursive path uses **the same canonical plan** as the base
    // path. This is HyperNova Construction 2's fixed-`pc` invariant:
    // for one `pc`, base and recursive branches live in one `F'_j`
    // structure, so both compiler paths must agree on `prep.plan`.
    let plan = prep.plan.clone();
    let (canonical_ce_shape, child_count) = {
        let acc = plan
            .accumulator
            .as_ref()
            .ok_or(FibonacciCompilerError::CanonicalPlanMissingAccumulator)?;
        if !acc.unified {
            return Err(FibonacciCompilerError::CanonicalPlanNotUnified);
        }
        let ce_shape = match &plan.nifs_payload_shapes[acc.ce_claim_payload_index] {
            NifsPayloadShape::CeClaim(s) => s.clone(),
            _ => return Err(FibonacciCompilerError::CanonicalPlanPayloadNotCeClaim),
        };
        (ce_shape, acc.child_count)
    };

    // Shape guard: the prover's `post_parent` must already match the
    // verifier-owned canonical CE shape. Truncating or padding would
    // either drop authority (`c_data`, `r`, `y_ring`, …) or write
    // unrelated values into committed payload bits — neither is safe.
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
        return Err(FibonacciCompilerError::PostParentShapeMismatch {
            canonical: canonical_ce_shape,
            actual: actual_shape,
        });
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
        return Err(FibonacciCompilerError::PostRunningClaimsCountMismatch {
            canonical: child_count,
            actual: post_running.claims.len() as u64,
        });
    }

    let ce_view = nifs_ce_view_from_claim(post_parent, ctx.public_input_len);
    let recursive_c_data = ce_view.c_data.clone();
    finalize_compile(
        ctx,
        input,
        plan,
        /* is_base = */ false,
        ce_view,
        recursive_c_data,
        child_count,
    )
}

/// Shared trace assembly + image emission for base and recursive paths.
///
/// `recursive_c_data` supplies the `c_data` lanes the recursive
/// accumulator preimage absorbs (taken from the NIFS CE view); the
/// selector picks the matching trace's digest based on `is_base`.
fn finalize_compile(
    ctx: &mut FibonacciCompilerContext,
    input: FibonacciAppStepInput,
    plan: RecursiveStepImagePlan,
    is_base: bool,
    ce_view: NifsCeClaimView,
    recursive_c_data: Vec<F>,
    child_count: u64,
) -> Result<FibonacciCompiledStep, FibonacciCompilerError> {
    let chunk_digest = chunk_digest_for_shape(
        ctx.chain_state.step_count,
        ctx.commitment_d,
        ctx.commitment_kappa,
        ctx.public_input_len,
    );
    let state_in = StateIn {
        vk_fs_digest: ctx.vk_fs_digest,
        structure_digest: ctx.structure_digest,
        z_0: ctx.z_0,
        z_i_in: ctx.chain_state.z_i,
        acc_digest_in: ctx.chain_state.acc_digest,
        public_trace_in: ctx.chain_state.public_trace,
    };

    // Five Poseidon traces under the unified plan: boundary_update,
    // public_trace_update, base_acc (`H(tag, 0)`), recursive_acc
    // (`H(tag, child_count, c_data...)`), state_x_out. The selector
    // chooses between the two accumulator traces' digests based on
    // `is_base`.
    let boundary_trace = encode_poseidon_trace(&build_boundary_update_preimage_fields(state_in.z_i_in, chunk_digest));
    let public_trace_trace = encode_poseidon_trace(&build_public_trace_update_preimage_fields(
        state_in.public_trace_in,
        chunk_digest,
    ));
    let base_accumulator_trace = encode_poseidon_trace(&build_accumulator_preimage_fields(0, &[]));
    let recursive_accumulator_trace =
        encode_poseidon_trace(&build_accumulator_preimage_fields(child_count, &recursive_c_data));

    let new_acc_digest = if is_base {
        base_accumulator_trace.digest_native
    } else {
        recursive_accumulator_trace.digest_native
    };
    let new_z_i = boundary_trace.digest_native;
    let new_public_trace = public_trace_trace.digest_native;
    let new_chunk_count = ctx.chain_state.chunk_count + 1;
    let new_step_count = ctx.chain_state.step_count + 1;

    let state_out = StateOut {
        new_chunk_count,
        new_step_count,
        new_z_i,
        new_public_trace,
        new_acc_digest,
    };

    let state_x_out_trace = encode_poseidon_trace(&build_state_x_out_preimage_fields(
        ctx.vk_fs_digest,
        ctx.structure_digest,
        new_chunk_count,
        new_step_count,
        ctx.z_0,
        new_z_i,
        ctx.pc,
        new_acc_digest,
        new_public_trace,
    ));
    let public_output_digest = state_x_out_trace.digest_native;
    let boundary_bits_buf = boundary_bits_from_digest(public_output_digest, ctx.boundary_bits);

    let encoder_input = FibonacciFPrimeStepInput {
        plan,
        boundary_bits: boundary_bits_buf,
        state_in,
        state_out,
        chunk_digest,
        // `limbs - 1` carry bits; for canonical Fibonacci limbs=3 → 2
        // carries. The canonical plan does not yet algebraically bind
        // carries to the Fibonacci limb arithmetic (`witness.next =
        // prev + curr`); they're free witness bits at this layer. We
        // populate honest zeros (the canonical low-norm choice).
        app_private_carries: vec![F::ZERO; ctx.limbs.saturating_sub(1)],
        is_base,
        nifs_payloads: vec![NifsPayloadInput::Ce(ce_view)],
        kmul_views: vec![],
        ring_action_pairs: vec![],
        one_shot_traces: vec![
            boundary_trace,
            public_trace_trace,
            base_accumulator_trace,
            recursive_accumulator_trace,
            state_x_out_trace,
        ],
        sponge_trace: None,
    };

    let encoded = encode_fibonacci_f_prime_step(encoder_input);

    ctx.chain_state = FibonacciChainState {
        chunk_count: new_chunk_count,
        step_count: new_step_count,
        z_i: new_z_i,
        acc_digest: new_acc_digest,
        public_trace: new_public_trace,
    };
    ctx.fold_for_step = None;

    let app_state_out = FibonacciAppState {
        prev: input.state_in.curr,
        curr: input.witness.next,
        step_index: input.state_in.step_index + 1,
    };

    Ok(FibonacciCompiledStep {
        app_output: FibonacciAppStepOutput {
            state_out: app_state_out,
            public_output_digest,
        },
        encoded,
    })
}

// ─────────────────────────────────────────────────────────────────────────
// Plan + view derivation helpers.
// ─────────────────────────────────────────────────────────────────────────

/// Build a perp (inactive, deterministic-filler) NIFS CE claim view
/// matching `shape`.
///
/// **Soundness contract.** This view is *not* a real `CeClaim`. The
/// unified-mode F' structure binds the NIFS payload region bit-by-bit,
/// so the perp payload's bits must be deterministic — but the
/// payload's *authority* (its `c_data`, commitment, fold_digest, etc.)
/// must not enter the chain's accumulator. The selector achieves this:
/// when `is_base = 1` the structure forces
/// `state_out.new_acc_digest = base_trace.digest`, which depends only
/// on the constant preimage `(tag, 0)`. The recursive accumulator
/// trace's preimage still sources its `c_data` lanes from this
/// payload, but its digest is *discarded* by the selector.
///
/// Audit hooks: any future change that weakens the selector binding
/// (e.g. allowing both branches' digests to enter authority through a
/// different code path) must keep the perp payload's `c_data` out of
/// the new authority path, or this filler becomes a soundness bug.
fn perp_nifs_ce_view(shape: &NifsCeClaimShape) -> NifsCeClaimView {
    let y_ring: Vec<Vec<[F; 2]>> = shape
        .y_ring_inner_lens
        .iter()
        .map(|&len| vec![[F::ZERO; 2]; len])
        .collect();
    NifsCeClaimView {
        d: 0,
        kappa: 0,
        c_data: vec![F::ZERO; shape.c_data_entries],
        x_rows: shape.x_rows as u64,
        // The encoder fills `x_active_cols` × `x_rows` entries; the
        // legacy `x_cols` header is informational. We mirror
        // `x_active_cols` so the deterministic-filler view stays
        // internally consistent.
        x_cols: shape.x_active_cols as u64,
        x_active_cols: shape.x_active_cols as u64,
        x_active_flat: vec![F::ZERO; shape.x_rows * shape.x_active_cols],
        r: vec![[F::ZERO; 2]; shape.r_len],
        y_ring,
        y_zcol: vec![[F::ZERO; 2]; shape.y_zcol_len],
        s_col: vec![[F::ZERO; 2]; shape.s_col_len],
        m_in: 0,
        fold_digest_fields: [F::ZERO; 4],
    }
}

/// Build the full NIFS CE-claim view from the real post-fold parent
/// authority. Every field is derived honestly from `post_parent`; no
/// zero placeholders for authority-bearing data.
fn nifs_ce_view_from_claim(post_parent: &CeClaim, _public_input_len: usize) -> NifsCeClaimView {
    let active_cols = crate::paper::relations::superneo_public_x_cols(post_parent.m_in);
    let x_active_flat: Vec<F> = collect_active_x(&post_parent.X, active_cols);
    let r_pairs: Vec<[F; 2]> = post_parent.r.iter().map(k_to_pair).collect();
    let y_ring_pairs: Vec<Vec<[F; 2]>> = post_parent
        .y_ring
        .iter()
        .map(|row| row.iter().map(k_to_pair).collect())
        .collect();
    let y_zcol_pairs: Vec<[F; 2]> = post_parent.y_zcol.iter().map(k_to_pair).collect();
    let s_col_pairs: Vec<[F; 2]> = post_parent.s_col.iter().map(k_to_pair).collect();

    NifsCeClaimView {
        d: post_parent.c.d as u64,
        kappa: post_parent.c.kappa as u64,
        c_data: post_parent.c.data.clone(),
        x_rows: post_parent.X.rows() as u64,
        x_cols: post_parent.X.cols() as u64,
        x_active_cols: active_cols as u64,
        x_active_flat,
        r: r_pairs,
        y_ring: y_ring_pairs,
        y_zcol: y_zcol_pairs,
        s_col: s_col_pairs,
        m_in: post_parent.m_in as u64,
        fold_digest_fields: digest32_as_fields(post_parent.fold_digest),
    }
}

/// Flatten the active-columns prefix of `X` row-major.
fn collect_active_x(x: &Mat<F>, active_cols: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(x.rows() * active_cols);
    for row in 0..x.rows() {
        for col in 0..active_cols {
            out.push(x[(row, col)]);
        }
    }
    out
}

/// Convert a `K` (Goldilocks degree-2 extension) into the `[F; 2]`
/// shape `NifsCeClaimView` expects.
fn k_to_pair(k: &K) -> [F; 2] {
    let coeffs = neo_math::field::KExtensions::as_coeffs(k);
    [coeffs[0], coeffs[1]]
}

/// `chunk_digest` for the canonical Fibonacci shape: shape-only, no
/// circularity with the step's own commitment data. Synthesises a
/// minimal [`CcsClaim`] with the right (d, kappa, m_in) — these are
/// the only fields [`f_prime_chunk_claim_digest`] reads.
fn chunk_digest_for_shape(start_index: u64, d: usize, kappa: usize, m_in: usize) -> [F; 4] {
    let shape_claim = CcsClaim {
        c: Commitment {
            d,
            kappa,
            data: Vec::new(),
        },
        x: Vec::new(),
        m_in,
    };
    f_prime_chunk_public_digest(start_index, std::slice::from_ref(&shape_claim))
}

/// Decompose the 4-lane Goldilocks digest into `boundary_bits` boolean
/// field elements, little-endian per lane.
fn boundary_bits_from_digest(digest: [F; 4], boundary_bits: usize) -> Vec<F> {
    let mut bits = vec![F::ZERO; boundary_bits];
    for (m, lane) in digest.iter().enumerate() {
        let value = lane.as_canonical_u64();
        for j in 0..POSEIDON2_GOLDILOCKS_BITS {
            let pos = m * POSEIDON2_GOLDILOCKS_BITS + j;
            if pos >= boundary_bits {
                break;
            }
            bits[pos] = if (value >> j) & 1 == 1 { F::ONE } else { F::ZERO };
        }
    }
    bits
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

    /// Caller set `ctx.fold_for_step = Some(..)` on the base step
    /// (`ctx.chain_state.chunk_count == 0`). The base step has no
    /// prior fold to consume; a stale fold either reflects a bug in
    /// the caller's chain-state tracking or an attempt to fold prior
    /// authority into the chain's first step.
    #[error("fibonacci compiler: base step (chunk_count == 0) must not carry a prior fold")]
    BaseStepUnexpectedPriorFold,

    /// Caller left `ctx.fold_for_step = None` on a recursive step
    /// (`ctx.chain_state.chunk_count > 0`). The recursive path needs
    /// the prior fold's authority boundary to populate the NIFS payload
    /// and validate the post-fold running.
    #[error(
        "fibonacci compiler: recursive step at chunk_count = {chunk_count} requires `ctx.fold_for_step = Some(..)`"
    )]
    PriorFoldMissingForRecursiveStep { chunk_count: u64 },

    /// `fold_for_step.post_running.parent_authority` is `None` — the
    /// caller handed an `RunningInstance` with empty claims as the
    /// post-fold result, which can't host a NIFS payload.
    #[error(
        "fibonacci compiler: post-fold running has no parent_authority; expected a non-empty post-fold running with Π_RLC parent"
    )]
    PostRunningMissingParentAuthority,

    /// `nifs::verify` rejected the prior fold's
    /// `(pre_running, latest, proof)` triple. The caller-supplied
    /// proof does not authorize the claimed fold under the per-step F'
    /// transcript. Catches mutated / forged NIFS proofs.
    #[error("fibonacci compiler: prior fold's NIFS proof failed to verify: {reason}")]
    PriorFoldVerificationFailed { reason: String },

    /// `nifs::verify` accepted the prior fold but the derived
    /// `post_running` does not match the caller-supplied
    /// `fold.post_running`. Catches a prover that mutated
    /// `post_running` after the proof was produced.
    #[error(
        "fibonacci compiler: derived post-fold running does not match caller-supplied `fold.post_running` (claims or parent_authority differ)"
    )]
    PriorFoldPostRunningMismatch,

    /// `prep.plan.accumulator` is `None`. The compiler only supports
    /// chains whose canonical plan has an accumulator (with `unified =
    /// true` — see [`Self::CanonicalPlanNotUnified`]). A plan without
    /// accumulator can't host the chain-wide accumulator digest.
    #[error("fibonacci compiler: prep.plan must carry `accumulator = Some(..)`; got `None`")]
    CanonicalPlanMissingAccumulator,

    /// `prep.plan.accumulator.unified` is `false`. The base + recursive
    /// compiler branches require the unified-mode structure (with both
    /// accumulator preimages + the `is_base` selector); legacy single-
    /// accumulator plans are not foldable through the compiler.
    #[error("fibonacci compiler: prep.plan.accumulator.unified must be true; got false (legacy plan)")]
    CanonicalPlanNotUnified,

    /// `prep.plan.nifs_payload_shapes[acc.ce_claim_payload_index]` is
    /// not a `CeClaim`. The compiler hard-wires CE-claim payloads
    /// (matching the canonical Fibonacci recursive shape).
    #[error("fibonacci compiler: prep.plan.nifs_payload_shapes[acc.ce_claim_payload_index] must be `CeClaim`")]
    CanonicalPlanPayloadNotCeClaim,

    /// `fold.post_running.parent_authority`'s CE-claim shape does not
    /// match the verifier-owned canonical CE shape encoded in
    /// `prep.plan`. Per HyperNova Construction 2's fixed-`pc`
    /// invariant, base and recursive branches share one `F'_j`
    /// structure, so a prover-supplied `post_parent` that doesn't fit
    /// the canonical NIFS payload region cannot be admitted into the
    /// chain — truncating or padding would either drop authority
    /// (`c_data`, `r`, `y_ring`, …) or write unrelated values into
    /// committed payload bits.
    #[error(
        "fibonacci compiler: post-fold parent CE shape does not match canonical (canonical={canonical:?}, actual={actual:?})"
    )]
    PostParentShapeMismatch {
        canonical: NifsCeClaimShape,
        actual: NifsCeClaimShape,
    },

    /// `fold.post_running.claims.len()` differs from
    /// `prep.plan.accumulator.child_count`. `child_count` is baked
    /// into the recursive accumulator preimage as a `Constant` —
    /// mismatch breaks the F' image's accumulator-digest binding to
    /// the lifecycle's authority running.
    #[error(
        "fibonacci compiler: post-fold running.claims.len() = {actual} does not match canonical child_count = {canonical}"
    )]
    PostRunningClaimsCountMismatch { canonical: u64, actual: u64 },

    /// Preprocessing's `public_input_len` is `None`; the compiler can't
    /// derive `z_0` / boundary sizing without a fixed length.
    #[error("fibonacci compiler: preprocessing.public_input_len must be Some(..) for the compiler to derive z_0")]
    PreprocessingMissingPublicInputLen,

    /// Preprocessing's `public_input_len` does not match the
    /// canonical Fibonacci boundary layout (1 + 4*64 = 257).
    #[error(
        "fibonacci compiler: public_input_len = {got} doesn't match the canonical Fibonacci boundary shape (expected {expected})"
    )]
    UnsupportedPublicInputShape { got: usize, expected: usize },
}

// ─────────────────────────────────────────────────────────────────────────
// Module-level constants.
// ─────────────────────────────────────────────────────────────────────────

/// Default `pc` for a Fibonacci F' chain — ℓ = 1, so `pc = TRIVIAL_PC`.
pub const FIBONACCI_PC: u64 = TRIVIAL_PC;
