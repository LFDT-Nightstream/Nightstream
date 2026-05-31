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
//!   — ignored by default (runs ~80 s on the current optimized path,
//!   but remains too heavy for the default compiler-regression suite).
//!   Run manually with `--ignored`.
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
use std::time::Instant;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::f_prime::compiler::FPrimeShellCompilerError;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImageLayout, NifsCeClaimShape, NifsPayloadShape};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_recursive_step_image_config, AccumulatorPlanOptions, RecursiveStepImagePlan, StateXOutPlanOptions,
};
use neo_fold_clean::lifecycle;
use neo_fold_clean::paper::construction2::{FoldProof, ProofState};
use neo_fold_clean::paper::digest::{accumulator_digest_from_claims, digest32_as_fields, structure_digest};
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_params::{goldilocks_paper_b2, NeoParams};
use support::fibonacci_f_prime::{
    self, compile_fibonacci_step, start_fibonacci_chain, FibonacciAppState, FibonacciAppStepInput, FibonacciAppWitness,
    FibonacciChainBuilder, FibonacciChainState, FibonacciCompilerError, FibonacciFPrimePreprocessing,
    FibonacciFoldForStep,
};

use support::fibonacci_f_prime::{canonical_threaded_plan, BOUNDARY_BITS};

/// One phase of an IVC benchmark, timed and logged via `--nocapture`.
/// Returns `(result, elapsed_seconds)` so the caller can both keep the
/// produced value and feed the duration into a summary block. Bind the
/// duration with `_` when you only want the log line.
///
/// Mirrors the SHA-256 system test's `phase` helper so the two benchmarks
/// share one output format. SHA-256's helper returns only `R` because its
/// benchmark variant uses inline `Instant::now()` blocks for the rows it
/// summarises; returning `(R, f64)` here keeps the call site uniform.
fn phase<R>(label: &str, f: impl FnOnce() -> R) -> (R, f64) {
    let t = Instant::now();
    let out = f();
    let elapsed_s = t.elapsed().as_secs_f64();
    eprintln!("[fib-ivc] {label:<32} {:>7.2}s", elapsed_s);
    (out, elapsed_s)
}

#[derive(Clone, Copy)]
struct FibIvcReference {
    kappa: u32,
    lambda: u32,
    params_m: u64,
    b: u32,
    k_rho: u32,
    t_sampling: u32,
    structure_n: usize,
    structure_m: usize,
    structure_t: usize,
    plan_limbs: usize,
    boundary_bits: usize,
    preprocess_s: f64,
    step0_s: f64,
    step1_s: f64,
    drop_compiled_s: f64,
    finish_s: f64,
    audit_verify_s: f64,
    terminal_verify_s: f64,
    drop_rest_s: f64,
    total_s: f64,
    prove_total_s: f64,
    verify_total_s: f64,
    amortized_recursive_s: f64,
    untimed_s: f64,
}

/// Local reference captured on 2026-05-31 after the terminal-CE
/// shared-`r` verifier optimization. Shape fields are deterministic and
/// asserted below; timings are wall-clock reference values and are only
/// printed as deltas because they depend on the machine and thermal state.
const FIB_IVC_REFERENCE: FibIvcReference = FibIvcReference {
    kappa: 18,
    lambda: 107,
    params_m: 1_073_741_824,
    b: 2,
    k_rho: 14,
    t_sampling: 216,
    structure_n: 6_210_611,
    structure_m: 6_117_572,
    structure_t: 8,
    plan_limbs: 3,
    boundary_bits: 256,
    preprocess_s: 10.61,
    step0_s: 12.79,
    step1_s: 26.40,
    drop_compiled_s: 0.04,
    finish_s: 13.66,
    audit_verify_s: 1.42,
    terminal_verify_s: 8.76,
    drop_rest_s: 0.07,
    total_s: 85.57,
    prove_total_s: 52.86,
    verify_total_s: 10.18,
    amortized_recursive_s: 26.40,
    untimed_s: 11.81,
};

fn assert_fib_ivc_reference_shape(prep: &FibonacciFPrimePreprocessing, plan: &RecursiveStepImagePlan) {
    let params = &prep.prep.params;
    let structure = prep.prep.structure();
    assert_eq!(params.kappa(), FIB_IVC_REFERENCE.kappa, "Fibonacci IVC kappa changed");
    assert_eq!(
        params.lambda(),
        FIB_IVC_REFERENCE.lambda,
        "Fibonacci IVC lambda changed"
    );
    assert_eq!(params.m(), FIB_IVC_REFERENCE.params_m, "Fibonacci IVC params.m changed");
    assert_eq!(params.b(), FIB_IVC_REFERENCE.b, "Fibonacci IVC norm bound b changed");
    assert_eq!(params.k_rho(), FIB_IVC_REFERENCE.k_rho, "Fibonacci IVC k_rho changed");
    assert_eq!(params.T(), FIB_IVC_REFERENCE.t_sampling, "Fibonacci IVC T changed");
    assert_eq!(
        structure.n, FIB_IVC_REFERENCE.structure_n,
        "Fibonacci IVC structure.n changed"
    );
    assert_eq!(
        structure.m, FIB_IVC_REFERENCE.structure_m,
        "Fibonacci IVC structure.m changed"
    );
    assert_eq!(
        structure.t(),
        FIB_IVC_REFERENCE.structure_t,
        "Fibonacci IVC structure.t changed"
    );
    assert_eq!(
        plan.limbs, FIB_IVC_REFERENCE.plan_limbs,
        "Fibonacci IVC plan.limbs changed"
    );
    assert_eq!(
        plan.boundary_bits, FIB_IVC_REFERENCE.boundary_bits,
        "Fibonacci IVC plan.boundary_bits changed"
    );
}

fn print_reference_row(label: &str, current_s: f64, reference_s: f64) {
    let delta_s = current_s - reference_s;
    let pct = if reference_s == 0.0 {
        0.0
    } else {
        delta_s * 100.0 / reference_s
    };
    eprintln!("[fib-ivc] {label:<27} {current_s:>8.2}s  {reference_s:>8.2}s  {delta_s:>+8.2}s  ({pct:>+6.1}%)");
}

fn print_fib_ivc_reference_comparison(current: FibIvcReference) {
    eprintln!();
    eprintln!("[fib-ivc] ───────────── reference comparison ─────────────");
    eprintln!("[fib-ivc] baseline: 2026-05-31 local post terminal-CE shared-r verifier optimization");
    eprintln!("[fib-ivc] metric                       current  reference         Δ");
    print_reference_row("preprocess", current.preprocess_s, FIB_IVC_REFERENCE.preprocess_s);
    print_reference_row("step 0 base", current.step0_s, FIB_IVC_REFERENCE.step0_s);
    print_reference_row("step 1 recursive", current.step1_s, FIB_IVC_REFERENCE.step1_s);
    print_reference_row(
        "drop compiled steps",
        current.drop_compiled_s,
        FIB_IVC_REFERENCE.drop_compiled_s,
    );
    print_reference_row("terminal fold", current.finish_s, FIB_IVC_REFERENCE.finish_s);
    print_reference_row(
        "audit replay verify",
        current.audit_verify_s,
        FIB_IVC_REFERENCE.audit_verify_s,
    );
    print_reference_row(
        "terminal verify",
        current.terminal_verify_s,
        FIB_IVC_REFERENCE.terminal_verify_s,
    );
    print_reference_row("drop prep + proof", current.drop_rest_s, FIB_IVC_REFERENCE.drop_rest_s);
    print_reference_row("total", current.total_s, FIB_IVC_REFERENCE.total_s);
    print_reference_row("prove wall", current.prove_total_s, FIB_IVC_REFERENCE.prove_total_s);
    print_reference_row("verify wall", current.verify_total_s, FIB_IVC_REFERENCE.verify_total_s);
    print_reference_row(
        "amortized recursive",
        current.amortized_recursive_s,
        FIB_IVC_REFERENCE.amortized_recursive_s,
    );
    print_reference_row("untimed remainder", current.untimed_s, FIB_IVC_REFERENCE.untimed_s);
    eprintln!("[fib-ivc] ─────────────────────────────────────────────────");
}

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
    fold: FibonacciFoldForStep,
    recursive_has_selector: bool,
    recursive_one_shot_count: usize,
    recursive_is_base: bool,
    base_structure_digest: [F; 4],
    recursive_structure_digest: [F; 4],
}

static CANONICAL_PREP: OnceLock<FibonacciFPrimePreprocessing> = OnceLock::new();
static BOOTSTRAP: OnceLock<BootstrapShared> = OnceLock::new();

fn shared_canonical_prep() -> &'static FibonacciFPrimePreprocessing {
    CANONICAL_PREP.get_or_init(|| {
        let plan = canonical_threaded_plan();
        fibonacci_f_prime::preprocess_seeded(&plan, 0xC0DE_5EED).expect("preprocess")
    })
}

fn shared_bootstrap() -> &'static BootstrapShared {
    BOOTSTRAP.get_or_init(bootstrap_real_intermediate_fold_uncached)
}

/// Build a real per-step fold + matching compiler chain state from a
/// compiler-emitted base step.
///
/// Returns `(prep, fold, chain_state)` where `fold.proof` is a real
/// intermediate `StepProof::Recursive` (extracted from
/// `audit.steps[1]` of a non-finalised lifecycle audit) and `chain_state`
/// mirrors lifecycle state at the start of step 1 so the compiler's
/// per-step transcript reconstruction matches the prover side.
fn bootstrap_real_intermediate_fold_uncached() -> BootstrapShared {
    let prep = shared_canonical_prep();
    let mut ctx = start_fibonacci_chain(prep).expect("start chain");

    let compiled_base = compile_fibonacci_step(prep, &mut ctx, valid_app_step(1, 1, 0)).expect("base compile");
    let base_structure_digest = structure_digest(&compiled_base.encoded.structure.ccs);
    let base_instance = fibonacci_f_prime::build_instance(prep, &compiled_base.encoded).expect("base instance");

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
        semantic_state_digest: digest32_as_fields(pre_state.semantic_state_digest),
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

    let mut recursive_ctx = start_fibonacci_chain(prep).expect("start chain for recursive compile");
    recursive_ctx.chain_state = chain_state;
    recursive_ctx.fold_for_step = Some(fold.clone());
    let compiled_recursive =
        compile_fibonacci_step(prep, &mut recursive_ctx, valid_app_step(1, 1, 1)).expect("recursive compile");
    let recursive_config = &compiled_recursive.encoded.image.layout.config;

    BootstrapShared {
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
    let prep = shared_canonical_prep();
    let mut ctx = start_fibonacci_chain(prep).expect("start chain");

    let compiled = compile_fibonacci_step(prep, &mut ctx, valid_app_step(1, 1, 0)).expect("base compile");
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
    let prep = shared_canonical_prep();
    let mut ctx = start_fibonacci_chain(prep).expect("start chain");

    let compiled = compile_fibonacci_step(prep, &mut ctx, valid_app_step(1, 1, 0)).expect("base compile");

    // Pull the canonical CE shape out of the prep's plan.
    let ce_shape: NifsCeClaimShape = {
        use neo_fold_clean::frontends::f_prime::image::NifsPayloadShape;
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
    let prep = shared_canonical_prep();
    let mut ctx = start_fibonacci_chain(prep).expect("start chain");
    // Intentionally do NOT advance chain_state; chunk_count stays at 0
    // (= base path) while fold_for_step is supplied.
    ctx.fold_for_step = Some(shared.fold.clone());

    let err = compile_fibonacci_step(prep, &mut ctx, valid_app_step(1, 1, 0)).expect_err("must reject");
    assert!(
        matches!(
            err,
            FibonacciCompilerError::Shell(FPrimeShellCompilerError::BaseStepUnexpectedPriorFold)
        ),
        "expected Shell(BaseStepUnexpectedPriorFold), got {err:?}"
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
    let prep = shared_canonical_prep();

    // Run the production base path entirely through the builder so the
    // `FibonacciChainBuilder::finish() -> verify_uncompressed` surface
    // gets default coverage.
    let mut builder = FibonacciChainBuilder::new(prep).expect("start builder");
    let compiled = builder
        .append_step(valid_app_step(1, 1, 0))
        .expect("base append");
    assert!(
        compiled.encoded.image.decode_is_base(),
        "single-step builder chain must compile the base branch"
    );

    let finalized = builder.finish().expect("finish");
    lifecycle::verify_uncompressed(&prep.prep, &finalized).expect("verify_uncompressed");
}

// ─────────────────────────────────────────────────────────────────────────
// Light base-step test for `FibonacciChainBuilder`.
//
// Confirms the prover-side wrapper does the same compile + lifecycle
// prove the manual two-line dance above does — without the finalize +
// verify pass (covered by the test directly above).
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn fibonacci_chain_builder_appends_base_step() {
    let prep = shared_canonical_prep();

    let mut builder = FibonacciChainBuilder::new(prep).expect("start chain");
    assert!(builder.audit().is_none(), "fresh builder must not own an audit yet");

    let compiled = builder
        .append_step(valid_app_step(1, 1, 0))
        .expect("base step");
    assert!(
        compiled.encoded.image.decode_is_base(),
        "first builder step must take the base branch"
    );
    assert!(
        builder.audit().is_some(),
        "after one append the builder must own an audit"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Tiny-params recursive builder test.
//
// Pins `FibonacciChainBuilder::prepare_next_fold` (the recursive-append
// orchestration) in the default suite. Under the canonical big plan one
// prove + extend exceeds the 5-min cap; under a test-only smaller
// `Params` profile (kappa = 4, m = 2^16, lambda = 60) the full
// base → recursive flow fits comfortably. The Goldilocks ring + Π_RLC
// constants are unchanged, so every algebraic identity holds bit-for-bit;
// only the Ajtai-SIS security parameter is reduced.
//
// `c_data_entries = 216` and `child_count = 14` are the params-derived
// fixed-point CE shape (KAPPA * D and K_RHO respectively); `r_len = 21`
// is the empirically converged value under this image size. If `Params`
// or the limb count ever change, this test fails with
// `PostParentShapeMismatch` and the error message names the new
// `actual` shape to copy into these constants.
// ─────────────────────────────────────────────────────────────────────────

fn tiny_fibonacci_params() -> Params {
    let inner = NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        /* kappa  */ 4,
        /* m      */ 1u64 << 16,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        /* lambda */ 60,
    )
    .expect("tiny NeoParams must satisfy the Π_RLC guard");
    Params::test_only_from_neo_params(inner)
}

fn tiny_fibonacci_lifecycle_plan() -> RecursiveStepImagePlan {
    const TINY_C_DATA_ENTRIES: usize = 216;
    const TINY_CHILD_COUNT: u64 = 14;
    const TINY_R_LEN: usize = 21;

    let ce_shape = NifsCeClaimShape {
        c_data_entries: TINY_C_DATA_ENTRIES,
        x_rows: 54,
        x_active_cols: 5,
        r_len: TINY_R_LEN,
        y_ring_inner_lens: vec![64; 8],
        y_zcol_len: 64,
        s_col_len: TINY_R_LEN,
    };

    let probe_plan = RecursiveStepImagePlan {
        limbs: 3,
        boundary_bits: BOUNDARY_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: vec![NifsPayloadShape::CeClaim(ce_shape)],
        accumulator: Some(AccumulatorPlanOptions {
            ce_claim_payload_index: 0,
            c_data_entries: TINY_C_DATA_ENTRIES,
            child_count: TINY_CHILD_COUNT,
            unified: true,
        }),
        state_x_out: None,
    };

    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let boundary_start = probe_layout.boundary.offset;
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|i| boundary_start + i * POSEIDON2_GOLDILOCKS_BITS);

    let mut plan = probe_plan;
    plan.state_x_out = Some(StateXOutPlanOptions {
        pc: 1,
        public_x_out_lane_bit_starts,
        app_public_input_var_indices: Vec::new(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
        initial_semantic_state_digest_anchor: None,
    });
    plan
}

#[test]
fn fibonacci_chain_builder_appends_recursive_step_under_tiny_params() {
    let plan = tiny_fibonacci_lifecycle_plan();
    let prep = fibonacci_f_prime::preprocess_seeded_with_params(&plan, tiny_fibonacci_params(), 0xC0DE_00B1)
        .expect("preprocess");

    let mut builder = FibonacciChainBuilder::new(&prep).expect("start builder");

    let compiled_base = builder
        .append_step(valid_app_step(1, 1, 0))
        .expect("base append");
    let base_out = compiled_base.app_output.state_out;

    let recursive_input = FibonacciAppStepInput {
        state_in: base_out,
        witness: FibonacciAppWitness {
            next: base_out.prev + base_out.curr,
        },
    };
    let compiled_recursive = builder
        .append_step(recursive_input)
        .expect("recursive append");

    assert!(compiled_base.encoded.image.decode_is_base());
    assert!(!compiled_recursive.encoded.image.decode_is_base());

    assert_eq!(
        structure_digest(&compiled_base.encoded.structure.ccs),
        structure_digest(&compiled_recursive.encoded.structure.ccs),
        "builder base and recursive outputs must share one verifier-owned F'_j structure"
    );
    assert_eq!(
        builder
            .audit()
            .expect("audit after recursive append")
            .steps
            .len(),
        2,
        "builder must extend the lifecycle once per appended app step"
    );
    assert_eq!(
        builder.context().chain_state.step_count,
        2,
        "builder must thread compiler chain state across base and recursive appends"
    );
}

#[test]
#[ignore = "production-shape two-step compiler chain runs ~80s; run explicitly with --ignored"]
fn compiler_two_step_chain_builds_from_scratch_and_verify_uncompressed_accepts() {
    let total = Instant::now();

    // 1. Build the canonical F' plan (microseconds — not timed).
    let plan = canonical_threaded_plan();

    // 2. Preprocess R1CS-F'. Single largest one-time cost in the test.
    //    Builds: the F' structure (bitness rows + R1CS recompose rows),
    //    optimized engine cache, structure_digest, vk, and the Ajtai PP.
    let (prep, prep_s) = phase("preprocess R1CS-F'", || {
        fibonacci_f_prime::preprocess_seeded(&plan, 0xC0DE_0009).expect("preprocess")
    });
    let p = &prep.prep.params;
    let s = prep.prep.structure();
    eprintln!(
        "[fib-ivc]   params:    kappa={}, lambda={}, m={}, b={}, k_rho={}, T={}",
        p.kappa(),
        p.lambda(),
        p.m(),
        p.b(),
        p.k_rho(),
        p.T(),
    );
    eprintln!(
        "[fib-ivc]   structure: n={}, m={}, t={}, plan.limbs={}, boundary_bits={}",
        s.n,
        s.m,
        s.t(),
        plan.limbs,
        plan.boundary_bits,
    );
    assert_fib_ivc_reference_shape(&prep, &plan);
    let current_kappa = p.kappa();
    let current_lambda = p.lambda();
    let current_params_m = p.m();
    let current_b = p.b();
    let current_k_rho = p.k_rho();
    let current_t_sampling = p.T();
    let current_structure_n = s.n;
    let current_structure_m = s.m;
    let current_structure_t = s.t();
    let current_plan_limbs = plan.limbs;
    let current_boundary_bits = plan.boundary_bits;

    // `FibonacciChainBuilder` owns the compile → prove → derive-next-fold
    // → compile → extend dance. Each `append_step` after the first
    // re-extends a cloned audit with the previous `latest` to obtain the
    // recursive fold authority, feeds that into the compiler, then
    // extends the real audit with the compiled instance.
    let mut builder = FibonacciChainBuilder::new(&prep).expect("start chain");

    // 3. Step 0 (base): compile + base lifecycle prove. The base branch
    //    runs the Π_CCS + Π_RLC + Π_DEC prover once over the first
    //    deposited instance; no NIFS fold and no prior-fold authority.
    let (compiled_base, step0_s) = phase("step 0 append (base)", || {
        builder
            .append_step(valid_app_step(1, 1, 0))
            .expect("base step")
    });
    assert!(
        compiled_base.encoded.image.decode_is_base(),
        "step 0 must take the base branch"
    );

    // 4. Step 1 (recursive): heaviest phase. Inside this test-support
    //    `append_step` (see tests/support/fibonacci_f_prime/lifecycle.rs):
    //      a. prepare_next_fold — clones the real audit and calls
    //         `lifecycle::extend` on it to derive the NIFS proof +
    //         post_running that the recursive compile embeds (one full
    //         prove pass). The cloned audit is then discarded — only
    //         the fold authority is kept on the compiler context.
    //      b. compile_fibonacci_step — re-runs NIFS.V on that fold
    //         under the per-step F' transcript, encodes the image with
    //         the prior fold authority embedded, satisfaction self-check
    //         against the cached structure.
    //      c. `lifecycle::extend` on the REAL audit with the new
    //         compiled instance (another full prove pass).
    //
    //    So this test-support builder pays ≈ 2× prove + 1× (compile +
    //    NIFS.V) per recursive step. The R1csChainBuilder used by the
    //    SHA-256 benchmark instead stashes the post-fold audit from (a)
    //    and swaps the compiled instance in at deposit time — that one
    //    pays 1× prove. Comparing fib-ivc and sha-ivc step-1 numbers
    //    directly is therefore apples-to-oranges; the prove count is
    //    different.
    let (compiled_recursive, step1_s) = phase("step 1 append (recursive)", || {
        builder
            .append_step(valid_app_step(1, 1, 1))
            .expect("recursive step")
    });
    assert!(
        !compiled_recursive.encoded.image.decode_is_base(),
        "step 1 must take the recursive branch"
    );

    // 5. HyperNova fixed-`F'_j` invariant: base and recursive R1CS-F'
    //    compiles share one verifier-owned structure (`prep.plan`), so
    //    their encoded steps share one `structure_digest`. This is the
    //    load-bearing IVC property.
    assert_eq!(
        structure_digest(&compiled_base.encoded.structure.ccs),
        structure_digest(&compiled_recursive.encoded.structure.ccs),
        "base and recursive compiler outputs must build instances under the same verifier-owned structure"
    );

    // 6. Drop the per-step compiled outputs (no longer needed by the
    //    terminal verifier path). The `Arc<FPrimeStructure>` they hold
    //    is `prep.structure`, so this only frees per-step image +
    //    witness, not the structure itself.
    let drop_compiled = Instant::now();
    drop(compiled_base);
    drop(compiled_recursive);
    let drop_compiled_s = drop_compiled.elapsed().as_secs_f64();
    eprintln!("[fib-ivc] {:<32} {:>7.2}s", "drop compiled steps", drop_compiled_s);

    // 7. Production-shape end-to-end: finalize the chain (one terminal
    //    NIFS fold), then run **both** verifier surfaces:
    //
    //    - `verify_uncompressed_audit` — chain-replay verifier; walks
    //      every per-step `StepProof::Recursive` under the F'
    //      transcript, then re-runs the terminal fold. Verify cost
    //      grows with chain length.
    //    - `verify_uncompressed` — production terminal-only verifier;
    //      the one a thin on-chain (or compressed-snark) IVC verifier
    //      ports to. Reads `proof.final_fold.terminal_inputs` plus
    //      `proof.state` and is independent of the historical
    //      `audit.steps[]`. Verify cost is constant-ish in chain length.
    //
    //    Both must accept a compiler-built base+recursive chain — they
    //    pin different soundness properties and a real production
    //    deployment will rely on `verify_uncompressed` (the audit form
    //    doesn't survive Spartan compression).
    let (finalized, finish_s) = phase("finish_with_audit() (terminal fold)", || {
        builder.finish_with_audit().expect("finalize")
    });

    let (_, audit_verify_s) = phase("verify_uncompressed_audit (replay)", || {
        lifecycle::verify_uncompressed_audit(&prep.prep, &finalized).expect("verify_uncompressed_audit")
    });
    let (_, terminal_verify_s) = phase("verify_uncompressed (terminal)", || {
        lifecycle::verify_uncompressed(&prep.prep, &finalized.proof).expect("verify_uncompressed")
    });

    // 8. Drop the remaining heavy allocations explicitly so the wall
    //    time is attributed to a labeled phase rather than to the
    //    implicit end-of-scope drop after the TOTAL line.
    let drop_rest = Instant::now();
    drop(finalized);
    drop(prep);
    let drop_rest_s = drop_rest.elapsed().as_secs_f64();
    eprintln!("[fib-ivc] {:<32} {:>7.2}s", "drop prep + finalized", drop_rest_s);

    let total_s = total.elapsed().as_secs_f64();
    eprintln!("[fib-ivc] {:<32} {:>7.2}s", "TOTAL (incl. drops)", total_s);

    // 9. Summary: roll up the timed phases into the numbers a reader of
    //    these logs actually wants — prover wall, verifier wall, and the
    //    one number that scales with chain length (the recursive step).
    //    Spartan compression is explicitly excluded; this is the IVC
    //    surface that `verify_uncompressed` accepts.
    let recursive_steps = 1.0_f64; // one recursive step in a two-step chain
    let prove_total = step0_s + step1_s + finish_s;
    let verify_total = audit_verify_s + terminal_verify_s;
    let amortized_recursive = step1_s / recursive_steps;
    let untimed = total_s
        - (prep_s + step0_s + step1_s + drop_compiled_s + finish_s + audit_verify_s + terminal_verify_s + drop_rest_s);
    eprintln!();
    eprintln!("[fib-ivc] ───────────────────────── summary ─────────────────────────");
    eprintln!("[fib-ivc] preprocess (one-time):       {prep_s:>7.2}s");
    eprintln!("[fib-ivc] prove wall:                  {prove_total:>7.2}s  (base + recursive + finish)");
    eprintln!("[fib-ivc]   step 0 base:                 {step0_s:>7.2}s  (1× lifecycle prove)");
    eprintln!(
        "[fib-ivc]   step 1 recursive:            {step1_s:>7.2}s  (≈ 2× prove + compile + NIFS.V; test-support builder)"
    );
    eprintln!("[fib-ivc]   terminal fold (finish):      {finish_s:>7.2}s  (1× NIFS prove)");
    eprintln!("[fib-ivc] verify wall:                 {verify_total:>7.2}s");
    eprintln!("[fib-ivc]   audit replay:                {audit_verify_s:>7.2}s  (scales with chain length)");
    eprintln!("[fib-ivc]   terminal only:               {terminal_verify_s:>7.2}s  ← production verify cost");
    eprintln!(
        "[fib-ivc] amortized recursive step:    {amortized_recursive:>7.2}s/op  (step 1 / {} recursive step{})",
        recursive_steps as u64,
        if recursive_steps as u64 == 1 { "" } else { "s" },
    );
    eprintln!("[fib-ivc] excludes:                    Spartan compression, on-chain wrapping");
    eprintln!("[fib-ivc] untimed remainder:           {untimed:>7.2}s  (test harness, drops outside labeled blocks)");
    eprintln!("[fib-ivc] ────────────────────────────────────────────────────────────");

    print_fib_ivc_reference_comparison(FibIvcReference {
        kappa: current_kappa,
        lambda: current_lambda,
        params_m: current_params_m,
        b: current_b,
        k_rho: current_k_rho,
        t_sampling: current_t_sampling,
        structure_n: current_structure_n,
        structure_m: current_structure_m,
        structure_t: current_structure_t,
        plan_limbs: current_plan_limbs,
        boundary_bits: current_boundary_bits,
        preprocess_s: prep_s,
        step0_s,
        step1_s,
        drop_compiled_s,
        finish_s,
        audit_verify_s,
        terminal_verify_s,
        drop_rest_s,
        total_s,
        prove_total_s: prove_total,
        verify_total_s: verify_total,
        amortized_recursive_s: amortized_recursive,
        untimed_s: untimed,
    });
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
