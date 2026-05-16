//! Unified-mode compiler integration tests.
//!
//! Each test is designed to fail under a plausible bad compiler so the
//! suite has real teeth rather than rubber-stamping internal helpers.
//!
//! - `compiler_accepts_real_intermediate_fold_proof` — feeds a real
//!   intermediate `StepProof::Recursive` (NOT a terminal-fold proof
//!   from `finish_uncompressed_with_audit`). Fails under the older
//!   `FINAL_FOLD_TRANSCRIPT_LABEL` path; passes once the compiler
//!   reconstructs the per-step F' transcript.
//! - `compiler_recursive_step_emits_unified_structure` — confirms the
//!   recursive compile path actually emits a unified-mode image (five
//!   one-shot traces, `unified_accumulator_selector = Some(..)`,
//!   `is_base = 0`). Catches a compiler that records `is_base` but
//!   leaves the structure in legacy single-accumulator shape.
//! - `compiler_base_step_uses_empty_accumulator_digest` — base step's
//!   `new_acc_digest` equals `digest32_as_fields(accumulator_digest_from_claims(b, &[]))`.
//! - `compiler_base_step_emits_perp_nifs_payload` — the base step's
//!   NIFS payload region is the canonical perp view (zero `c_data` of
//!   `prep.plan` shape, zero `r`, zero `y_ring`, …).
//! - `compiler_base_step_rejects_unexpected_prior_fold` — base path
//!   refuses a caller that left a stale `fold_for_step`.
//! - `compiler_recursive_step_sets_is_base_false` — recursive image's
//!   committed `is_base` bit is `0`.
//! - `compiler_chain_builds_from_scratch_and_verify_uncompressed_accepts`
//!   — a single base step compiled end-to-end through the lifecycle is
//!   accepted by the production non-replay verifier.
//! - `compiler_two_step_chain_builds_from_scratch_and_verify_uncompressed_accepts`
//!   — ignored by default (runs ~500 s under the canonical big plan,
//!   well above the 5-min default cap). Run manually with `--ignored`.
//!   This is the load-bearing production IVC path: compile a base step,
//!   fold it through the lifecycle, derive the next step's NIFS proof
//!   from a shape-equivalent placeholder extend, compile a recursive
//!   step with that fold, re-extend the original audit with the real
//!   compiled recursive instance, finalise, and run **both** verifier
//!   surfaces — `verify_uncompressed_audit` (chain-replay) AND
//!   `verify_uncompressed` (production terminal-only). The
//!   chain-replay form guarantees the per-step F'-transcript chain
//!   matches; the terminal-only form is what an on-chain or
//!   compressed-snark verifier would port to. Both accepting on a
//!   compiler-built chain is what closes the SuperNeo / HyperNova
//!   §6.3 Construction 2 IVC milestone for Fibonacci.
//! - `compiler_base_and_recursive_steps_share_structure` — load-bearing.
//!   Both paths' encoded steps share `structure_digest`, pinning the
//!   fixed-`F'_j` invariant for a single `pc`.

#![allow(non_snake_case)]

#[path = "../support/mod.rs"]
mod support;

use std::sync::OnceLock;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::frontends::fibonacci_f_prime::image::NifsCeClaimShape;
use neo_fold_clean::frontends::fibonacci_f_prime::{
    self, compile_fibonacci_step, start_fibonacci_chain, FibonacciAppState, FibonacciAppStepInput, FibonacciAppWitness,
    FibonacciChainState, FibonacciCompilerError, FibonacciFPrimePreprocessing, FibonacciFoldForStep,
};
use neo_fold_clean::lifecycle;
use neo_fold_clean::paper::construction2::{FoldProof, ProofState};
use neo_fold_clean::paper::digest::{accumulator_digest_from_claims, digest32_as_fields, structure_digest};

use support::fibonacci_f_prime::canonical_threaded_plan;

/// Convenience: a Fibonacci app input where `next == prev + curr`.
fn valid_app_step(prev_u: u64, curr_u: u64, step_index: u64) -> FibonacciAppStepInput {
    let prev = F::from_u64(prev_u);
    let curr = F::from_u64(curr_u);
    FibonacciAppStepInput {
        state_in: FibonacciAppState { prev, curr, step_index },
        witness: FibonacciAppWitness { next: prev + curr },
    }
}

/// Shared cache of a single bootstrap result — every test that needs a
/// real intermediate `StepProof::Recursive` clones from this. The big
/// canonical plan makes each lifecycle fold expensive (~25-50 s), so
/// 5 tests × 3 folds × 8-way parallelism = quadratic-ish contention
/// without caching. Caching once flattens that to one bootstrap +
/// cheap per-test clones.
///
/// The cached values must outlive every test on `'static`, which
/// `OnceLock` gives us automatically.
struct BootstrapShared {
    prep: FibonacciFPrimePreprocessing,
    fold: FibonacciFoldForStep,
    recursive_has_selector: bool,
    recursive_one_shot_count: usize,
    recursive_is_base: bool,
    base_structure_digest: [F; 4],
    recursive_structure_digest: [F; 4],
}

static BOOTSTRAP: OnceLock<BootstrapShared> = OnceLock::new();

fn shared_bootstrap() -> &'static BootstrapShared {
    BOOTSTRAP.get_or_init(|| bootstrap_real_intermediate_fold_uncached(0xC0DE_5EED))
}

/// Build a real per-step fold + matching compiler chain state from a
/// compiler-emitted base step.
///
/// Returns `(prep, fold, chain_state)` where `fold.proof` is a real
/// intermediate `StepProof::Recursive` (extracted from
/// `audit.steps[1]` of a non-finalised lifecycle audit) and `chain_state`
/// mirrors lifecycle state at the start of step 1 so the compiler's
/// per-step transcript reconstruction matches the prover side.
fn bootstrap_real_intermediate_fold_uncached(seed: u64) -> BootstrapShared {
    let plan = canonical_threaded_plan();
    let prep = fibonacci_f_prime::preprocess_seeded(&plan, seed).expect("preprocess");
    let mut ctx = start_fibonacci_chain(&prep).expect("start chain");

    let compiled_base = compile_fibonacci_step(&prep, &mut ctx, valid_app_step(1, 1, 0)).expect("base compile");
    let base_structure_digest = structure_digest(&compiled_base.encoded.structure.ccs);
    let base_instance = fibonacci_f_prime::build_instance(&prep, &compiled_base.encoded).expect("base instance");

    let audit_after_base = lifecycle::prove(&prep.prep, [vec![base_instance.clone()]]).expect("base lifecycle prove");

    let pre_state = audit_after_base.proof.state.clone();
    let (pre_running, latest) = match &pre_state.proof {
        ProofState::Active { running, latest } => (running.clone(), latest.clone()),
        _ => panic!("expected Active state at the start of step 1"),
    };
    let chain_state = FibonacciChainState {
        chunk_count: pre_state.chunk_count,
        step_count: pre_state.step_count,
        z_i: digest32_as_fields(pre_state.z_i),
        acc_digest: digest32_as_fields(pre_state.acc_digest),
        public_trace: digest32_as_fields(pre_state.public_trace),
    };

    let audit_after_recursive =
        lifecycle::extend(&prep.prep, audit_after_base, vec![base_instance]).expect("derive recursive fold proof");
    let proof = match &audit_after_recursive.steps[1].fold {
        FoldProof::Recursive(p) => p.clone(),
        _ => panic!("expected Recursive at audit.steps[1]"),
    };
    let post_running = match &audit_after_recursive.proof.state.proof {
        ProofState::Active { running, .. } => running.clone(),
        _ => panic!("expected Active state after final extend"),
    };

    let fold = FibonacciFoldForStep {
        pre_running,
        latest,
        proof,
        post_running,
    };

    let mut recursive_ctx = start_fibonacci_chain(&prep).expect("start chain for recursive compile");
    recursive_ctx.chain_state = chain_state;
    recursive_ctx.fold_for_step = Some(fold.clone());
    let compiled_recursive =
        compile_fibonacci_step(&prep, &mut recursive_ctx, valid_app_step(1, 1, 1)).expect("recursive compile");
    let recursive_config = &compiled_recursive.encoded.image.layout.config;

    BootstrapShared {
        prep,
        fold,
        recursive_has_selector: recursive_config.unified_accumulator_selector.is_some(),
        recursive_one_shot_count: recursive_config.poseidon_one_shot_preimage_lens.len(),
        recursive_is_base: compiled_recursive.encoded.image.decode_is_base(),
        base_structure_digest,
        recursive_structure_digest: structure_digest(&compiled_recursive.encoded.structure.ccs),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Per-step transcript: real intermediate fold proof.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn compiler_accepts_real_intermediate_fold_proof() {
    // The fold proof here is `audit.steps[n-1].fold` — a real
    // intermediate `StepProof::Recursive`, NOT the terminal NIFS from
    // `finish_uncompressed_with_audit`. Pre-fix, the compiler verified
    // under `FINAL_FOLD_TRANSCRIPT_LABEL` and this would reject; under
    // the per-step F' transcript it should accept.
    let shared = shared_bootstrap();
    assert!(
        !shared.recursive_is_base,
        "shared bootstrap must have compiled a recursive step with a real intermediate StepProof::Recursive"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Recursive path emits a unified-mode image.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn compiler_recursive_step_emits_unified_structure() {
    let shared = shared_bootstrap();
    assert!(
        shared.recursive_has_selector,
        "recursive compile must emit a unified-mode image (UnifiedAccumulatorSelector = Some)"
    );
    assert_eq!(
        shared.recursive_one_shot_count, 5,
        "unified mode requires 5 one-shot traces (boundary, public_trace, base_acc, recursive_acc, state_x_out)"
    );
    assert!(!shared.recursive_is_base, "recursive step must commit `is_base = 0`");
}

// ─────────────────────────────────────────────────────────────────────────
// Base path: shape, payload, error mode.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn compiler_base_step_uses_empty_accumulator_digest() {
    let plan = canonical_threaded_plan();
    let prep = fibonacci_f_prime::preprocess_seeded(&plan, 0xC0DE_0003).expect("preprocess");
    let mut ctx = start_fibonacci_chain(&prep).expect("start chain");

    let compiled = compile_fibonacci_step(&prep, &mut ctx, valid_app_step(1, 1, 0)).expect("base compile");
    let state_out = compiled.encoded.image.decode_state_out();
    let expected_empty = digest32_as_fields(accumulator_digest_from_claims(prep.prep.params.b(), &[]));
    assert_eq!(
        state_out.new_acc_digest, expected_empty,
        "base step must commit `new_acc_digest = digest32_as_fields(accumulator_digest_from_claims(b, &[]))`"
    );
    assert!(
        compiled.encoded.image.decode_is_base(),
        "base step must commit `is_base = 1`"
    );
}

#[test]
fn compiler_base_step_emits_perp_nifs_payload() {
    let plan = canonical_threaded_plan();
    let prep = fibonacci_f_prime::preprocess_seeded(&plan, 0xC0DE_0004).expect("preprocess");
    let mut ctx = start_fibonacci_chain(&prep).expect("start chain");

    let compiled = compile_fibonacci_step(&prep, &mut ctx, valid_app_step(1, 1, 0)).expect("base compile");

    // Pull the canonical CE shape out of the prep's plan.
    let ce_shape: NifsCeClaimShape = {
        use neo_fold_clean::frontends::fibonacci_f_prime::image::NifsPayloadShape;
        match &prep.plan.nifs_payload_shapes[0] {
            NifsPayloadShape::CeClaim(s) => s.clone(),
            other => panic!("expected CeClaim shape, got {other:?}"),
        }
    };

    let decoded = compiled.encoded.image.decode_nifs_ce_claim_at(0, &ce_shape);
    assert_eq!(decoded.d, 0, "perp.d must be 0");
    assert_eq!(decoded.kappa, 0, "perp.kappa must be 0");
    assert!(
        decoded.c_data.iter().all(|f| *f == F::ZERO),
        "perp.c_data must be all zeros"
    );
    assert_eq!(decoded.x_rows, ce_shape.x_rows as u64);
    assert_eq!(decoded.x_active_cols, ce_shape.x_active_cols as u64);
    assert!(
        decoded.x_active_flat.iter().all(|f| *f == F::ZERO),
        "perp.x_active_flat must be all zeros"
    );
    assert!(
        decoded.r.iter().all(|pair| pair == &[F::ZERO; 2]),
        "perp.r must be all zeros"
    );
    assert!(
        decoded
            .y_ring
            .iter()
            .all(|row| row.iter().all(|pair| pair == &[F::ZERO; 2])),
        "perp.y_ring must be all zeros"
    );
    assert!(
        decoded.y_zcol.iter().all(|pair| pair == &[F::ZERO; 2]),
        "perp.y_zcol must be all zeros"
    );
    assert!(
        decoded.s_col.iter().all(|pair| pair == &[F::ZERO; 2]),
        "perp.s_col must be all zeros"
    );
    assert_eq!(decoded.m_in, 0, "perp.m_in must be 0");
    assert_eq!(
        decoded.fold_digest_fields,
        [F::ZERO; 4],
        "perp.fold_digest_fields must be all zeros"
    );
}

#[test]
fn compiler_base_step_rejects_unexpected_prior_fold() {
    let shared = shared_bootstrap();
    let mut ctx = start_fibonacci_chain(&shared.prep).expect("start chain");
    // Intentionally do NOT advance chain_state; chunk_count stays at 0
    // (= base path) while fold_for_step is supplied.
    ctx.fold_for_step = Some(shared.fold.clone());

    let err = compile_fibonacci_step(&shared.prep, &mut ctx, valid_app_step(1, 1, 0)).expect_err("must reject");
    assert!(
        matches!(err, FibonacciCompilerError::BaseStepUnexpectedPriorFold),
        "expected BaseStepUnexpectedPriorFold, got {err:?}"
    );
}

#[test]
fn compiler_recursive_step_sets_is_base_false() {
    let shared = shared_bootstrap();
    assert!(!shared.recursive_is_base, "recursive step must commit `is_base = 0`");
}

// ─────────────────────────────────────────────────────────────────────────
// End-to-end: base step accepted by `verify_uncompressed`.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn compiler_chain_builds_from_scratch_and_verify_uncompressed_accepts() {
    let plan = canonical_threaded_plan();
    let prep = fibonacci_f_prime::preprocess_seeded(&plan, 0xC0DE_0007).expect("preprocess");
    let mut ctx = start_fibonacci_chain(&prep).expect("start chain");

    let compiled = compile_fibonacci_step(&prep, &mut ctx, valid_app_step(1, 1, 0)).expect("base compile");

    // Fold the compiler-emitted base step through the lifecycle.
    let inst = fibonacci_f_prime::build_instance(&prep, &compiled.encoded).expect("build_instance");
    let audit = lifecycle::prove(&prep.prep, [vec![inst]]).expect("lifecycle prove");
    let finalized = neo_fold_clean::finish_uncompressed(&prep.prep, audit).expect("finalize");

    lifecycle::verify_uncompressed(&prep.prep, &finalized).expect("verify_uncompressed");
}

#[test]
#[ignore = "full compiler-built two-step IVC: base compile + lifecycle fold + recursive compile + final extend + finalize + both `verify_uncompressed_audit` (chain-replay) and `verify_uncompressed` (production terminal-only). Runs ~500 s on dev hardware under the canonical big plan, well above the 5-min default cap. Run manually with `cargo test --release -p neo-fold-clean --test system_fibonacci_compiler_unified_structure -- --ignored compiler_two_step_chain_builds_from_scratch_and_verify_uncompressed_accepts` after small-params lands."]
fn compiler_two_step_chain_builds_from_scratch_and_verify_uncompressed_accepts() {
    let plan = canonical_threaded_plan();
    let prep = fibonacci_f_prime::preprocess_seeded(&plan, 0xC0DE_0009).expect("preprocess");
    let mut ctx = start_fibonacci_chain(&prep).expect("start chain");

    // Step 0: compile the base step and fold it once so it becomes the
    // `latest` that step 1 must fold into the running accumulator.
    let compiled_base = compile_fibonacci_step(&prep, &mut ctx, valid_app_step(1, 1, 0)).expect("base compile");
    let base_instance = fibonacci_f_prime::build_instance(&prep, &compiled_base.encoded).expect("base instance");
    let audit_after_base = lifecycle::prove(&prep.prep, [vec![base_instance.clone()]]).expect("base lifecycle prove");

    let pre_state_for_recursive = audit_after_base.proof.state.clone();
    let (pre_running, latest) = match &pre_state_for_recursive.proof {
        ProofState::Active { running, latest } => (running.clone(), latest.clone()),
        _ => panic!("after base step the lifecycle state must be Active"),
    };

    // Derive the real per-step NIFS proof for step 1. The fresh instance
    // used here is a shape-equivalent placeholder: F' chunk digests bind
    // only `(start_index, fresh.len, d, kappa, m_in)`, and every compiler
    // output under `prep` has the same shape. This avoids a circular
    // dependency between "compile step 1" and "need step 1's fold proof".
    // `audit_after_base` is cloned so the placeholder extend doesn't
    // contaminate the chain we re-extend with the real step 1 below.
    let audit_after_placeholder_recursive =
        lifecycle::extend(&prep.prep, audit_after_base.clone(), vec![base_instance.clone()])
            .expect("derive recursive fold proof with shape-equivalent placeholder");
    let recursive_proof = match &audit_after_placeholder_recursive.steps[1].fold {
        FoldProof::Recursive(p) => p.clone(),
        _ => panic!("step 1 must produce a Recursive fold proof"),
    };
    let post_running = match &audit_after_placeholder_recursive.proof.state.proof {
        ProofState::Active { running, .. } => running.clone(),
        _ => panic!("after placeholder recursive step the lifecycle state must be Active"),
    };
    assert!(
        !post_running.claims.is_empty(),
        "step 1 fold must produce a non-empty running accumulator"
    );

    // Step 1: feed the derived fold authority back into the compiler and
    // compile the actual recursive app step from the compiler-threaded ctx.
    ctx.chain_state = FibonacciChainState {
        chunk_count: pre_state_for_recursive.chunk_count,
        step_count: pre_state_for_recursive.step_count,
        z_i: digest32_as_fields(pre_state_for_recursive.z_i),
        acc_digest: digest32_as_fields(pre_state_for_recursive.acc_digest),
        public_trace: digest32_as_fields(pre_state_for_recursive.public_trace),
    };
    ctx.fold_for_step = Some(FibonacciFoldForStep {
        pre_running,
        latest,
        proof: recursive_proof,
        post_running,
    });
    let compiled_recursive =
        compile_fibonacci_step(&prep, &mut ctx, valid_app_step(1, 1, 1)).expect("recursive compile");
    assert!(!compiled_recursive.encoded.image.decode_is_base());

    let recursive_instance =
        fibonacci_f_prime::build_instance(&prep, &compiled_recursive.encoded).expect("recursive instance");
    assert_eq!(
        base_instance.claim.m_in, recursive_instance.claim.m_in,
        "base and recursive instances must use the same public-input split"
    );
    assert_eq!(
        structure_digest(&compiled_base.encoded.structure.ccs),
        structure_digest(&compiled_recursive.encoded.structure.ccs),
        "base and recursive compiler outputs must build instances under the same verifier-owned structure"
    );

    // Production-shape end-to-end: extend the original (un-placeholder)
    // audit with the *real* compiled recursive instance, finalise, and
    // run **both** verifier surfaces:
    //
    // - `verify_uncompressed_audit` — chain-replay verifier; walks
    //   every per-step `StepProof::Recursive` under the F' transcript,
    //   then re-runs the terminal fold.
    // - `verify_uncompressed` — production terminal-only verifier; the
    //   one a thin on-chain (or compressed-snark) IVC verifier ports
    //   to. Reads `proof.final_fold.terminal_inputs` plus
    //   `proof.state` and is independent of the historical
    //   `audit.steps[]`.
    //
    // Both must accept a compiler-built base+recursive chain — they
    // pin different soundness properties and a real production
    // deployment will rely on `verify_uncompressed` (the audit form
    // doesn't survive Spartan compression).
    let audit_after_recursive =
        lifecycle::extend(&prep.prep, audit_after_base, vec![recursive_instance]).expect("recursive lifecycle extend");
    let finalized =
        neo_fold_clean::finish_uncompressed_with_audit(&prep.prep, audit_after_recursive).expect("finalize");

    lifecycle::verify_uncompressed_audit(&prep.prep, &finalized).expect("verify_uncompressed_audit");
    lifecycle::verify_uncompressed(&prep.prep, &finalized.proof).expect("verify_uncompressed");
}

// ─────────────────────────────────────────────────────────────────────────
// Load-bearing: base and recursive must share one structure_digest.
// ─────────────────────────────────────────────────────────────────────────

/// HyperNova Construction 2 fixed-`pc` invariant: for one `pc`, the
/// base and recursive branches share one `F'_j` structure. The
/// compiler enforces this by using `prep.plan` for **both** paths and
/// shape-validating the prover-supplied `post_parent` against it.
#[test]
fn compiler_base_and_recursive_steps_share_structure() {
    let shared = shared_bootstrap();

    assert_eq!(
        shared.base_structure_digest, shared.recursive_structure_digest,
        "base and recursive compiles must share one structure_digest \
         (HyperNova §6.3 Construction 2 fixed-`F'_j` invariant)"
    );
}
