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
//!   recursive compile path emits the delayed-handle unified image
//!   (only the `state_x_out` one-shot trace, no source-image selector,
//!   `is_base = 0`). Catches a compiler that records `is_base` but
//!   leaves the structure in legacy single-accumulator shape.
//! - `compiler_base_step_uses_canonical_zero_accumulator_digest` — base
//!   step commits the fixed-shape Construction-2 zero accumulator.
//! - `compiler_base_step_elides_source_image_nifs_payload` — the base
//!   step reserves no source-image NIFS payload columns; verifier-plan
//!   shape metadata still exists for compiler validation.
//! - `compiler_base_step_rejects_unexpected_prior_fold` — base path
//!   refuses a caller that left a stale `fold_for_step`.
//! - `compiler_recursive_step_sets_is_base_false` — recursive image's
//!   committed `is_base` bit is `0`.
//! - `compiler_chain_builds_from_scratch_and_verify_uncompressed_accepts`
//!   — a single base step compiled end-to-end through the lifecycle is
//!   accepted by the production non-replay verifier.
//! - `compiler_two_step_chain_builds_from_scratch_and_rejects_terminal_only`
//!   — ignored by default because it is a perf/integration snapshot, not
//!   a small semantic unit test. It currently runs well under the
//!   project-wide 5-minute cap on the optimized path.
//!   Run manually with `--ignored`.
//!   This is the load-bearing encoded-shell integration path: compile a
//!   base step, fold it through the lifecycle, derive the next step's
//!   NIFS proof from a shape-equivalent placeholder extend, compile a
//!   recursive step with that fold, re-extend the original audit with
//!   the real compiled recursive instance, finalise, and check both
//!   verifier surfaces: `verify_uncompressed_audit` accepts the
//!   chain-replay proof, while terminal-only `verify_uncompressed`
//!   rejects the multi-chunk F' projection until the compressed decider
//!   proves the recursive F'/NIFS.V induction. It proves the current
//!   strict low-norm shell is foldable, and it exposes the shell's cost
//!   wall. It is not the final compact HyperNova F' verifier shape; that
//!   path lives in `paper::f_prime::r1cs` and still needs a production
//!   low-norm `enc(F')` boundary.
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
use neo_fold_clean::paper::construction2::{FoldProof, LaneCommitmentMode, ProofState, RunningInstance};
use neo_fold_clean::paper::digest::{digest32_as_fields, structure_digest, AccumulatorHandle};
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

/// Local reference captured on 2026-06-01 after the terminal-CE flat batch
/// evaluator, sparse CSC cleanup, nested PiDEC y-ring batching, parallel
/// chi-table expansion, cached Fibonacci F' structure reuse, parallel
/// active-block SuperNeo y-evaluation, Poseidon2 tree matrix digest,
/// parallel SuperNeo matrix-cache build, compact NC digit tables, and
/// semantic-Boolean-only F' shell rows, delayed accumulator-handle binding,
/// duplicate public-trace removal, boundary-update elision, compact
/// `state_x_out` domain/preimage trimming, and transitive `z_0` binding
/// through `vk_fs_digest`, stateless duplicate-semantic omission in
/// `state_x_out`, plus branch-reduced digit-monomial reduction.
/// Updated after the seeded Ajtai signed-unit `commit_many` batching path,
/// which reuses each sampled PP column across all terminal child witnesses
/// instead of committing every signed-unit child independently.
/// Shape
/// fields are deterministic and asserted below; timings are wall-clock
/// reference values and are only printed as deltas because they depend on the
/// machine and thermal state.
const FIB_IVC_REFERENCE: FibIvcReference = FibIvcReference {
    kappa: 18,
    lambda: 106,
    params_m: 1_073_741_824,
    b: 2,
    k_rho: 14,
    t_sampling: 216,
    // Includes the derived-is_base region: +64 inverse-lane boolean rows
    // and +2 counter-link rows (n), +64 inverse-lane columns (m).
    structure_n: 2_393,
    structure_m: 134_852,
    structure_t: 8,
    plan_limbs: 3,
    boundary_bits: 256,
    preprocess_s: 0.08,
    step0_s: 0.03,
    step1_s: 0.16,
    drop_compiled_s: 0.00,
    finish_s: 0.11,
    audit_verify_s: 0.12,
    terminal_verify_s: 0.10,
    drop_rest_s: 0.00,
    total_s: 0.64,
    prove_total_s: 0.31,
    verify_total_s: 0.22,
    amortized_recursive_s: 0.16,
    untimed_s: 0.03,
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
    eprintln!(
        "[fib-ivc] baseline: 2026-06-01 local post flat CE evaluator + sparse CSC + PiDEC/chi batching + cached F' structure + parallel active-block y_eval + compact NC digit tables + semantic-Boolean F' shell rows + delayed accumulator handle + compact state_x_out + branch-reduced digit monomials + batched signed-unit Ajtai commit_many"
    );
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

fn print_fib_ivc_poseidon_trace_budget(prep: &FibonacciFPrimePreprocessing) {
    let layout = &prep.structure.layout;
    let names = ["state_x_out"];
    eprintln!();
    eprintln!("[fib-ivc] ───────────── F' one-shot Poseidon trace budget ─────────────");
    eprintln!("[fib-ivc] trace                    preimage   perms     digits    %cols");
    let mut total_bits = 0usize;
    for (idx, trace_layout) in layout.one_shot_poseidon_layouts.iter().enumerate() {
        let name = names.get(idx).copied().unwrap_or("one_shot");
        let preimage_len = layout.config.poseidon_one_shot_preimage_lens[idx];
        total_bits += trace_layout.trace_len;
        eprintln!(
            "[fib-ivc] {name:<24} {preimage_len:>8} {perms:>7} {bits:>10} {pct:>7.1}%",
            perms = trace_layout.absorbs,
            bits = trace_layout.trace_len,
            pct = trace_layout.trace_len as f64 * 100.0 / prep.prep.structure().m as f64,
        );
    }
    eprintln!(
        "[fib-ivc] {total:<24} {blank:>8} {blank:>7} {total_bits:>10} {pct:>7.1}%",
        total = "one-shot total",
        blank = "",
        pct = total_bits as f64 * 100.0 / prep.prep.structure().m as f64,
    );
    eprintln!(
        "[fib-ivc] {total:<24} {blank:>8} {blank:>7} {bits:>10} {pct:>7.1}%",
        total = "poseidon region",
        blank = "",
        bits = layout.poseidon.bits,
        pct = layout.poseidon.bits as f64 * 100.0 / prep.prep.structure().m as f64,
    );
    eprintln!("[fib-ivc] accumulator handle is carried in state_out; no producer-side parent.c_data hash trace");
    eprintln!("[fib-ivc] ─────────────────────────────────────────────────────────────");
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
/// real intermediate `StepProof::Recursive` clones from this. Even after
/// the canonical F' shell was shrunk, deriving an intermediate lifecycle
/// fold is still the expensive part of these compiler tests; caching once
/// keeps the default regression suite focused on semantic coverage rather
/// than repeatedly rebuilding the same fold authority.
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
        ProofState::Active { running, latest } => (
            running.materialize().expect("pre-running materialization"),
            latest.clone(),
        ),
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
        FoldProof::Recursive(p) => p
            .materialize()
            .expect("recursive NIFS proof materialization"),
        _ => panic!("expected Recursive at audit.steps[1]"),
    };
    let post_running = match &audit_after_recursive.proof.state.proof {
        ProofState::Active { running, .. } => running.materialize().expect("post-running materialization"),
        _ => panic!("expected Active state after final extend"),
    };

    let fold = FibonacciFoldForStep {
        pre_running,
        latest,
        proof,
        post_running,
        post_summary: None,
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
#[ignore = "the canonical recursive bootstrap exceeds the 5-minute test cap; this compiler-adapter diagnostic requires an explicit longer-run approval"]
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
// Recursive path emits the delayed-handle unified image.
// ─────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "the canonical recursive bootstrap exceeds the 5-minute test cap; bounded one-joint circuit tests cover the recursive structure"]
fn compiler_recursive_step_emits_unified_structure() {
    let shared = shared_bootstrap();
    assert!(
        !shared.recursive_has_selector,
        "recursive compile must use delayed accumulator-handle binding (no UnifiedAccumulatorSelector)"
    );
    assert_eq!(
        shared.recursive_one_shot_count, 1,
        "delayed-handle unified mode emits only the state_x_out one-shot trace"
    );
    assert!(!shared.recursive_is_base, "recursive step must commit `is_base = 0`");
}

// ─────────────────────────────────────────────────────────────────────────
// Base path: shape, payload, error mode.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn compiler_base_step_uses_canonical_zero_accumulator_digest() {
    let prep = shared_canonical_prep();
    let mut ctx = start_fibonacci_chain(prep).expect("start chain");

    let compiled = compile_fibonacci_step(prep, &mut ctx, valid_app_step(1, 1, 0)).expect("base compile");
    let state_out = compiled.encoded.image.decode_state_out();
    let m_in = prep
        .prep
        .public_input_len
        .expect("Fibonacci preprocessing pins public input width");
    let canonical = RunningInstance::canonical_zero(
        &prep.prep.params,
        prep.prep.structure(),
        m_in,
        LaneCommitmentMode::Plain,
    )
    .expect("construct canonical base accumulator");
    let expected =
        AccumulatorHandle::from_running_parts(&canonical.claims, canonical.parent_authority.as_ref()).digest_fields();
    assert_eq!(
        state_out.new_acc_digest, expected,
        "base step must commit the fixed-shape Construction-2 zero accumulator"
    );
    assert!(
        compiled.encoded.image.decode_is_base(),
        "base step must commit `is_base = 1`"
    );
}

#[test]
fn compiler_base_step_elides_source_image_nifs_payload() {
    let prep = shared_canonical_prep();
    let mut ctx = start_fibonacci_chain(prep).expect("start chain");

    let compiled = compile_fibonacci_step(prep, &mut ctx, valid_app_step(1, 1, 0)).expect("base compile");

    assert_eq!(
        compiled.encoded.image.layout.nifs_payloads.bits, 0,
        "unified delayed-handle mode must not reserve source-image NIFS payload columns",
    );
    assert!(
        !prep.plan.nifs_payload_shapes.is_empty(),
        "the verifier plan still keeps CE-shape metadata for compiler validation",
    );
}

#[test]
#[ignore = "the canonical recursive bootstrap exceeds the 5-minute test cap; this compiler-adapter diagnostic requires an explicit longer-run approval"]
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
#[ignore = "the canonical recursive bootstrap exceeds the 5-minute test cap; bounded recursive image tests cover the is_base bit"]
fn compiler_recursive_step_sets_is_base_false() {
    let shared = shared_bootstrap();
    assert!(!shared.recursive_is_base, "recursive step must commit `is_base = 0`");
}

// ─────────────────────────────────────────────────────────────────────────
// End-to-end: base step accepted by `verify_uncompressed`.
// ─────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "the canonical finalization fixture exceeds the 5-minute test cap; lifecycle finalization has a bounded active-path suite"]
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
// Reduced-security recursive builder diagnostic.
//
// Pins `FibonacciChainBuilder::prepare_next_fold` (the recursive-append
// orchestration) in the default suite. Under the canonical big plan one
// prove + extend exceeds the 5-min cap; under a test-only smaller
// `Params` profile (kappa = 4, m = 2^24, lambda = 60) the full
// base → recursive flow fits comfortably. The Goldilocks ring + Π_RLC
// constants are unchanged, so every algebraic identity holds bit-for-bit;
// only the Ajtai-SIS security parameter is reduced.
//
// `c_data_entries = 216` and `child_count = 14` are the params-derived
// fixed-point CE shape (KAPPA * D and K_RHO respectively); `r_len = 12`
// is the empirically converged value after source
// NIFS payload elision. The reduced-security fixture uses a wider dummy app-private
// region to avoid colliding with the canonical test SRS dimensions while
// keeping the same public input shape. If `Params`
// or the limb count ever change, this test fails with
// `PostParentShapeMismatch` and the error message names the new
// `actual` shape to copy into these constants.
// ─────────────────────────────────────────────────────────────────────────

fn reduced_security_fibonacci_params() -> Params {
    let inner = NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        /* kappa  */ 4,
        /* m      */ 1u64 << 24,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        /* lambda */ 60,
    )
    .expect("reduced-security NeoParams must satisfy the Π_RLC guard");
    Params::test_only_from_neo_params(inner)
}

fn reduced_security_fibonacci_lifecycle_plan() -> RecursiveStepImagePlan {
    const TINY_C_DATA_ENTRIES: usize = 216;
    const TINY_CHILD_COUNT: u64 = 14;
    const TINY_R_LEN: usize = 12;
    const TINY_LIMBS: usize = 57;

    let ce_shape = NifsCeClaimShape {
        c_data_entries: TINY_C_DATA_ENTRIES,
        x_rows: 54,
        x_active_cols: 5,
        r_len: TINY_R_LEN,
        y_ring_inner_lens: vec![64; 8],
    };

    let probe_plan = RecursiveStepImagePlan {
        limbs: TINY_LIMBS,
        app_private_var_widths: Vec::new(),
        boundary_bits: BOUNDARY_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        projection_batches: Vec::new(),
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
        app_public_input_bit_var_indices: Vec::new(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
        initial_semantic_state_digest_anchor: None,
    });
    plan
}

#[test]
#[ignore = "the selected one-joint 24-row domain makes this recursive builder fixture exceed the 5-minute test cap; bounded NIFS and lifecycle suites cover the active transition"]
fn fibonacci_chain_builder_appends_recursive_step_under_reduced_security_params() {
    let plan = reduced_security_fibonacci_lifecycle_plan();
    let prep =
        fibonacci_f_prime::preprocess_seeded_with_params(&plan, reduced_security_fibonacci_params(), 0xC0DE_00B1)
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
#[ignore = "encoded-shell two-step compiler chain is a perf/integration snapshot; run explicitly with --ignored"]
fn compiler_two_step_chain_builds_from_scratch_and_rejects_terminal_only() {
    let total = Instant::now();

    // 1. Build the canonical F' plan (microseconds — not timed).
    let plan = canonical_threaded_plan();

    // 2. Preprocess R1CS-F'. Single largest one-time cost in the test.
    //    Builds: the F' structure (semantic Boolean shell rows + R1CS recompose rows),
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
    print_fib_ivc_poseidon_trace_budget(&prep);
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

    // 7. Encoded-shell end-to-end: finalize the chain (one terminal
    //    NIFS fold), then check both verifier surfaces:
    //
    //    - `verify_uncompressed_audit` — chain-replay verifier; walks
    //      every per-step `StepProof::Recursive` under the F'
    //      transcript, then re-runs the terminal fold. Verify cost
    //      grows with chain length.
    //    - `verify_uncompressed` — terminal-only verifier. It must
    //      reject this two-chunk F' projection because it has dropped
    //      the intermediate F' witnesses and NIFS.V messages that
    //      HyperNova's induction needs. Single-chunk F' remains accepted
    //      by the small test above.
    //
    //    Audit replay must accept the compiler-built base+recursive
    //    chain. Terminal-only must fail closed until the compressed
    //    decider proves the recursive F'/NIFS.V induction.
    let (finalized, finish_s) = phase("finish_with_audit() (terminal fold)", || {
        builder.finish_with_audit().expect("finalize")
    });

    let (_, audit_verify_s) = phase("verify_uncompressed_audit (replay)", || {
        lifecycle::verify_uncompressed_audit(&prep.prep, &finalized).expect("verify_uncompressed_audit")
    });
    let (_, terminal_verify_s) = phase("verify_uncompressed (expected reject)", || {
        let err = lifecycle::verify_uncompressed(&prep.prep, &finalized.proof)
            .expect_err("terminal-only verifier must reject multi-chunk R1CS-F' projection");
        assert!(
            matches!(
                err,
                lifecycle::Error::TerminalOnlyMultiChunkUnsupported { chunk_count: 2 }
            ),
            "expected TerminalOnlyMultiChunkUnsupported(2), got {err:?}"
        );
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
    //    Spartan compression is explicitly excluded. For two chunks,
    //    audit replay is the accepting surface; terminal-only rejection
    //    is expected and timed separately.
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
    eprintln!("[fib-ivc] verify/check wall:           {verify_total:>7.2}s");
    eprintln!("[fib-ivc]   audit replay:                {audit_verify_s:>7.2}s  (scales with chain length)");
    eprintln!(
        "[fib-ivc]   terminal-only reject:        {terminal_verify_s:>7.2}s  (expected multi-chunk F' fail-closed)"
    );
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
#[ignore = "the canonical recursive bootstrap exceeds the 5-minute test cap; this fixed-structure compiler diagnostic requires an explicit longer-run approval"]
fn compiler_base_and_recursive_steps_share_structure() {
    let shared = shared_bootstrap();

    assert_eq!(
        shared.base_structure_digest, shared.recursive_structure_digest,
        "base and recursive compiles must share one structure_digest \
         (HyperNova §6.3 Construction 2 fixed-`F'_j` invariant)"
    );
}
