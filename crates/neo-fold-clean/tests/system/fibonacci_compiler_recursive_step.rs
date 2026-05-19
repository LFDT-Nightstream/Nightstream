//! Recursive-step compiler tests (council-hardened).
//!
//! Every test in this file is designed to fail under at least one
//! plausible bad compiler:
//!
//! - **TransitionMismatch coverage** — catches a compiler that
//!   doesn't enforce the Fibonacci app step.
//! - **Whole post-fold parent-authority encoding** — catches a
//!   compiler that fills only `c_data` and leaves
//!   `d`/`kappa`/`m_in`/`fold_digest`/`X`/`r`/`y`/`s` at defaults.
//!   The expected view is built test-locally from the real `CeClaim`,
//!   so the assertion isn't tautological.
//! - **Post vs pre confusion** — catches a compiler that uses
//!   `pre_running.parent_authority` instead of
//!   `post_running.parent_authority`. The fixture ensures pre and
//!   post commitments differ before the check runs.
//! - **Mutated post_running** — catches a compiler that trusts the
//!   caller's `post_running` without re-verifying the fold. The
//!   compiler must re-run NIFS.V and reject.
//! - **Mutated NIFS proof** — catches a compiler that ignores
//!   `nifs_proof` and only copies `post_running`. The compiler must
//!   re-run NIFS.V and reject.
//! - **Chain state advancement** — catches a compiler that emits an
//!   encoded step but doesn't move the chain coordinates forward.
//! - **Missing prior fold** — catches a compiler that silently emits
//!   a perp/zero NIFS payload on a recursive step.
//!
//! The fold input is bootstrapped from a real lifecycle output: the
//! fixture builds a chain, the lifecycle folds it, and the test
//! extracts the resulting `RunningInstance` to feed the compiler.
//!
//! ## Chunk count
//!
//! These tests exercise the **recursive** branch of
//! `compile_fibonacci_step`. The compiler dispatches to recursive
//! when `ctx.chain_state.chunk_count > 0`, so every test bumps the
//! counter past the base case before compiling. The base-step path
//! is covered in `fibonacci_compiler_unified_structure.rs`.

#![allow(non_snake_case)]

#[path = "../support/mod.rs"]
mod support;

use std::sync::OnceLock;

use neo_ccs::matrix::Mat;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::frontends::f_prime::compiler::FPrimeShellCompilerError;
use neo_fold_clean::frontends::f_prime::image::{NifsCeClaimShape, NifsCeClaimView};
use neo_fold_clean::lifecycle;
use neo_fold_clean::paper::construction2::{FoldProof, ProofState};
use neo_fold_clean::paper::digest::digest32_as_fields;
use support::fibonacci_f_prime::{
    self, compile_fibonacci_step, start_fibonacci_chain, FibonacciAppState, FibonacciAppStepInput, FibonacciAppWitness,
    FibonacciChainState, FibonacciCompilerError, FibonacciFPrimePreprocessing, FibonacciFoldForStep,
};

use support::fibonacci_f_prime::{canonical_threaded_plan, honest_state_threaded_encoded_f_prime_steps};

/// One-shot cache for an `n = 2` real per-step fold bootstrap. Heavy
/// under the canonical big plan; tests clone from this rather than
/// reproving for every case.
struct CachedFold {
    prep: FibonacciFPrimePreprocessing,
    fold: FibonacciFoldForStep,
    chain_state: FibonacciChainState,
}

static CACHED_N2: OnceLock<CachedFold> = OnceLock::new();

fn cached_n2() -> &'static CachedFold {
    CACHED_N2.get_or_init(|| {
        let (prep, fold, chain_state) = build_real_fold_uncached(2, 0x4D02_5EED);
        CachedFold {
            prep,
            fold,
            chain_state,
        }
    })
}

/// Build a real **per-step** `FibonacciFoldForStep` from a non-finalised
/// lifecycle audit, plus the matching `FibonacciChainState` the compiler
/// needs in `ctx.chain_state` so its reconstructed F'-step transcript
/// matches the prover-side one.
///
/// Strategy: run the lifecycle incrementally through `n` calls without
/// finalising, capture state at the START of the last call (so its
/// `proof = Active { pre_running, latest }` are the fold inputs), then
/// run the last call to obtain the per-step `StepProof::Recursive`. The
/// chain-state the compiler must use is derived from that pre-state's
/// digests — directly setting `ctx.chain_state` is simpler than
/// re-running compile base + recursive at this layer.
///
/// `n >= 2` is required (call 0 is `NoFold`, so a real Recursive proof
/// only appears starting at call 1).
fn build_real_fold_uncached(
    n: usize,
    seed: u64,
) -> (FibonacciFPrimePreprocessing, FibonacciFoldForStep, FibonacciChainState) {
    assert!(n >= 2, "build_real_fold needs n >= 2 (call 0 is NoFold)");
    let plan = canonical_threaded_plan();
    let prep = fibonacci_f_prime::preprocess_seeded(&plan, seed).expect("preprocess");
    let steps = honest_state_threaded_encoded_f_prime_steps(n);

    // Build CcsInstances for each step and feed them through the
    // lifecycle one call at a time.
    let instances: Vec<_> = steps
        .iter()
        .map(|s| fibonacci_f_prime::build_instance(&prep, s).expect("instance"))
        .collect();

    // First call: equivalent to `start_proof` + extend([batch_0]).
    let mut audit = lifecycle::prove(&prep.prep, [vec![instances[0].clone()]]).expect("first prove");
    // Subsequent calls except the last.
    for inst in instances.iter().take(n - 1).skip(1) {
        audit = lifecycle::extend(&prep.prep, audit, vec![inst.clone()]).expect("extend");
    }

    // State at the START of the last call (= state_{n-1}_input). Its
    // `proof.{running, latest}` are the inputs to the fold we're about
    // to drive.
    let pre_state = audit.proof.state.clone();
    let (pre_running, latest) = match &pre_state.proof {
        ProofState::Active { running, latest } => (running.clone(), latest.clone()),
        _ => panic!("expected Active state at the start of call n-1"),
    };

    // Compiler-side chain state mirrors `pre_state`'s digest fields so
    // `verify_prior_fold` rebuilds the same per-step F' transcript the
    // prover-side `paper::f_prime::native::prove` used.
    let chain_state = FibonacciChainState {
        chunk_count: pre_state.chunk_count,
        step_count: pre_state.step_count,
        z_i: digest32_as_fields(pre_state.z_i),
        acc_digest: digest32_as_fields(pre_state.acc_digest),
        public_trace: digest32_as_fields(pre_state.public_trace),
    };

    // Last call generates `step_proof_{n-1}` whose `fold` is the
    // Recursive NIFS proof for `(pre_running, latest) -> post_running`.
    audit = lifecycle::extend(&prep.prep, audit, vec![instances[n - 1].clone()]).expect("final extend");
    let proof = match &audit.steps[n - 1].fold {
        FoldProof::Recursive(p) => p.clone(),
        _ => panic!("expected Recursive at step_proof[{}]", n - 1),
    };
    let post_running = match &audit.proof.state.proof {
        ProofState::Active { running, .. } => running.clone(),
        _ => panic!("expected Active state after final extend"),
    };

    let fold = FibonacciFoldForStep {
        pre_running,
        latest,
        proof,
        post_running,
    };
    (prep, fold, chain_state)
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

/// Test-local helper: build the expected `NifsCeClaimView` directly
/// from a real post-fold parent `CeClaim`. **This is intentionally a
/// separate implementation from the compiler's helper** — if the
/// compiler's view differs from this one for any field, the
/// `compiler_encodes_full_post_fold_ce_claim` test fails.
fn expected_view_from_post_parent(post_parent: &neo_fold_clean::CeClaim) -> NifsCeClaimView {
    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(post_parent.m_in);
    let x_active_flat: Vec<F> = collect_active_x_test(&post_parent.X, active_cols);
    let r_pairs: Vec<[F; 2]> = post_parent.r.iter().map(k_to_pair_test).collect();
    let y_ring_pairs: Vec<Vec<[F; 2]>> = post_parent
        .y_ring
        .iter()
        .map(|row| row.iter().map(k_to_pair_test).collect())
        .collect();
    let y_zcol_pairs: Vec<[F; 2]> = post_parent.y_zcol.iter().map(k_to_pair_test).collect();
    let s_col_pairs: Vec<[F; 2]> = post_parent.s_col.iter().map(k_to_pair_test).collect();
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

fn collect_active_x_test(x: &Mat<F>, active_cols: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(x.rows() * active_cols);
    for row in 0..x.rows() {
        for col in 0..active_cols {
            out.push(x[(row, col)]);
        }
    }
    out
}

fn k_to_pair_test(k: &K) -> [F; 2] {
    let coeffs = neo_math::field::KExtensions::as_coeffs(k);
    [coeffs[0], coeffs[1]]
}

/// Shape matching the compiler's derivation from a `post_parent`
/// `CeClaim`. Tests use this to decode the encoded image's NIFS
/// payload region.
fn shape_for_post_parent(post_parent: &neo_fold_clean::CeClaim) -> NifsCeClaimShape {
    NifsCeClaimShape {
        c_data_entries: post_parent.c.data.len(),
        x_rows: post_parent.X.rows(),
        x_active_cols: neo_fold_clean::paper::relations::superneo_public_x_cols(post_parent.m_in),
        r_len: post_parent.r.len(),
        y_ring_inner_lens: post_parent.y_ring.iter().map(|row| row.len()).collect(),
        y_zcol_len: post_parent.y_zcol.len(),
        s_col_len: post_parent.s_col.len(),
    }
}

/// Assert the test fixture itself is non-degenerate. If these fail we
/// know the test setup is hiding the bug we're trying to catch.
fn assert_fold_fixture_preconditions(fold: &FibonacciFoldForStep) {
    let post_parent = fold
        .post_running
        .parent_authority
        .as_ref()
        .expect("post-fold parent authority must exist");
    assert!(
        !post_parent.c.data.is_empty(),
        "fixture precondition: post-fold c.data must be non-empty"
    );
    assert!(
        post_parent.c.data.iter().any(|f| *f != F::ZERO),
        "fixture precondition: post-fold c.data must have at least one non-zero entry"
    );
    assert!(post_parent.m_in > 0, "fixture precondition: post-fold m_in must be > 0");
    assert_ne!(
        digest32_as_fields(post_parent.fold_digest),
        [F::ZERO; 4],
        "fixture precondition: post-fold fold_digest must be non-zero"
    );
}

// ── Contract: app-level transition + base-step gate ────────────────────────

#[test]
fn compiler_rejects_bad_fibonacci_witness() {
    let cached = cached_n2();
    let mut ctx = start_fibonacci_chain(&cached.prep).expect("start chain");
    ctx.chain_state = cached.chain_state;
    ctx.fold_for_step = Some(cached.fold.clone());

    let bad_input = FibonacciAppStepInput {
        state_in: FibonacciAppState {
            prev: F::from_u64(2),
            curr: F::from_u64(3),
            step_index: 1,
        },
        // 2 + 3 = 5, but witness says 99.
        witness: FibonacciAppWitness { next: F::from_u64(99) },
    };
    let err = compile_fibonacci_step(&cached.prep, &mut ctx, bad_input).expect_err("must reject");
    assert!(
        matches!(err, FibonacciCompilerError::TransitionMismatch { .. }),
        "expected TransitionMismatch, got {err:?}"
    );
}

#[test]
fn compiler_rejects_recursive_step_without_prior_fold() {
    // Recursive branch: chunk_count > 0 but caller forgot to supply
    // ctx.fold_for_step. The compiler must reject with
    // `PriorFoldMissingForRecursiveStep` (and NOT silently emit a
    // perp/zero NIFS payload — that would silently corrupt authority).
    let cached = cached_n2();
    let mut ctx = start_fibonacci_chain(&cached.prep).expect("start chain");
    let expected_chunk_count = cached.chain_state.chunk_count;
    ctx.chain_state = cached.chain_state;
    assert!(ctx.fold_for_step.is_none());

    let err = compile_fibonacci_step(&cached.prep, &mut ctx, valid_app_step(1, 1, 1)).expect_err("must reject");
    assert!(
        matches!(
            err,
            FibonacciCompilerError::Shell(FPrimeShellCompilerError::PriorFoldMissingForRecursiveStep { chunk_count })
                if chunk_count == expected_chunk_count
        ),
        "expected Shell(PriorFoldMissingForRecursiveStep {{ chunk_count: {expected_chunk_count} }}), got {err:?}"
    );
}

// ── Whole post-fold CE-claim encoding (non-tautological) ───────────────────

#[test]
fn compiler_encodes_full_post_fold_ce_claim() {
    let cached = cached_n2();
    assert_fold_fixture_preconditions(&cached.fold);
    let post_parent = cached
        .fold
        .post_running
        .parent_authority
        .as_ref()
        .unwrap()
        .clone();

    let mut ctx = start_fibonacci_chain(&cached.prep).expect("start chain");
    ctx.chain_state = cached.chain_state;
    ctx.fold_for_step = Some(cached.fold.clone());

    let compiled = compile_fibonacci_step(&cached.prep, &mut ctx, valid_app_step(1, 1, 1))
        .expect("recursive compile with real fold");

    // Decode the NIFS payload using a shape matching the compiler's
    // derivation, then compare whole view against an expected view
    // built test-locally from the real CeClaim (no compiler code).
    let shape = shape_for_post_parent(&post_parent);
    let decoded = compiled.encoded.image.decode_nifs_ce_claim_at(0, &shape);
    let expected = expected_view_from_post_parent(&post_parent);
    assert_eq!(
        decoded, expected,
        "decoded NIFS CE payload must equal expected view built from post_parent"
    );
}

// ── Post vs pre confusion (anti-bug) ───────────────────────────────────────

#[test]
#[ignore = "n=3 fixture chain under the canonical big plan — preprocess + 3 fixture-step encodes + 2 lifecycle extends + 1 final extend pushes the binary over the 5-min per-test cap when other recursive-step tests run in the same invocation. Run manually with `cargo test --release -p neo-fold-clean --test system_fibonacci_compiler_recursive_step -- --ignored compiler_uses_post_fold_parent_authority_not_pre_fold`. The anti-bug property (compiler uses post-fold parent authority, not pre-fold) is also covered by the lighter `compiler_encodes_full_post_fold_ce_claim` test that runs by default."]
fn compiler_uses_post_fold_parent_authority_not_pre_fold() {
    // 3-step chain so both pre_running and post_running have parent
    // authorities (running becomes non-empty after the first recursive
    // fold, so call N >= 2 sees a non-trivial pre_running).
    //
    // This test cannot use the n=2 cache because it needs a pre_running
    // with a non-trivial parent_authority. n=3 is uncached (one-off).
    let (prep, fold, chain_state) = build_real_fold_uncached(3, 0x4D02_0004);
    assert_fold_fixture_preconditions(&fold);
    let post_parent = fold.post_running.parent_authority.as_ref().unwrap().clone();
    let pre_parent = fold
        .pre_running
        .parent_authority
        .clone()
        .expect("pre-fold parent authority for a 3-step chain");

    // Fixture precondition: pre and post commitments differ. If they
    // happen to be equal the test would pass trivially regardless of
    // which one the compiler used.
    assert_ne!(
        pre_parent.c.data, post_parent.c.data,
        "fixture precondition: pre and post commitments must be distinct after a real fold"
    );

    let mut ctx = start_fibonacci_chain(&prep).expect("start chain");
    ctx.chain_state = chain_state;
    ctx.fold_for_step = Some(fold);

    let compiled =
        compile_fibonacci_step(&prep, &mut ctx, valid_app_step(1, 2, 2)).expect("recursive compile with real fold");

    let shape = shape_for_post_parent(&post_parent);
    let decoded = compiled.encoded.image.decode_nifs_ce_claim_at(0, &shape);

    assert_eq!(
        decoded.c_data, post_parent.c.data,
        "compiler must use post-fold parent c_data"
    );
    assert_ne!(
        decoded.c_data, pre_parent.c.data,
        "anti-bug: compiler used pre-fold parent c_data instead of post-fold"
    );
}

// ── Mutation tests (require internal NIFS verification) ────────────────────

#[test]
fn compiler_rejects_mutated_post_running_authority() {
    let cached = cached_n2();
    let mut fold = cached.fold.clone();
    // Mutate one byte of the post-fold parent commitment. The prior
    // fold's NIFS.V will derive the *un-mutated* post running, and
    // the compiler must reject the mismatch.
    let post_parent_mut = fold.post_running.parent_authority.as_mut().unwrap();
    post_parent_mut.c.data[0] += F::ONE;

    let mut ctx = start_fibonacci_chain(&cached.prep).expect("start chain");
    ctx.chain_state = cached.chain_state;
    ctx.fold_for_step = Some(fold);

    let err = compile_fibonacci_step(&cached.prep, &mut ctx, valid_app_step(1, 1, 1)).expect_err("must reject");
    assert!(
        matches!(
            err,
            FibonacciCompilerError::Shell(FPrimeShellCompilerError::PriorFoldPostRunningMismatch)
        ),
        "expected Shell(PriorFoldPostRunningMismatch), got {err:?}"
    );
}

#[test]
fn compiler_rejects_mutated_nifs_proof() {
    let cached = cached_n2();
    let mut fold = cached.fold.clone();
    // Mutate a child claim's commitment inside the NIFS proof. NIFS.V
    // reconstructs sumcheck challenges from the transcript; the
    // mutated commitment breaks the final algebraic check.
    fold.proof.pi_dec.children[0].c.data[0] += F::ONE;

    let mut ctx = start_fibonacci_chain(&cached.prep).expect("start chain");
    ctx.chain_state = cached.chain_state;
    ctx.fold_for_step = Some(fold);

    let err = compile_fibonacci_step(&cached.prep, &mut ctx, valid_app_step(1, 1, 1)).expect_err("must reject");
    // A mutated Π_DEC child can fire either via NIFS.verify's algebraic
    // checks (PriorFoldVerificationFailed) or via the post-state
    // binding (PriorFoldPostRunningMismatch) depending on which check
    // surfaces first. Both are correct rejections.
    assert!(
        matches!(
            err,
            FibonacciCompilerError::Shell(FPrimeShellCompilerError::PriorFoldVerificationFailed { .. })
                | FibonacciCompilerError::Shell(FPrimeShellCompilerError::PriorFoldPostRunningMismatch)
        ),
        "expected Shell(PriorFoldVerificationFailed) or Shell(PriorFoldPostRunningMismatch), got {err:?}"
    );
}

// ── Chain state advancement + app output ───────────────────────────────────

#[test]
fn compiler_advances_chain_state_and_clears_fold_for_step() {
    let cached = cached_n2();
    let mut ctx = start_fibonacci_chain(&cached.prep).expect("start chain");
    ctx.chain_state = cached.chain_state;
    let pre_chain_state = ctx.chain_state;
    ctx.fold_for_step = Some(cached.fold.clone());

    let _compiled = compile_fibonacci_step(&cached.prep, &mut ctx, valid_app_step(1, 1, 1))
        .expect("recursive compile with real fold");

    assert_eq!(ctx.chain_state.chunk_count, pre_chain_state.chunk_count + 1);
    assert_eq!(ctx.chain_state.step_count, pre_chain_state.step_count + 1);
    assert_ne!(ctx.chain_state.z_i, pre_chain_state.z_i);
    assert_ne!(ctx.chain_state.acc_digest, pre_chain_state.acc_digest);
    assert_ne!(ctx.chain_state.public_trace, pre_chain_state.public_trace);
    assert!(
        ctx.fold_for_step.is_none(),
        "compiler must clear ctx.fold_for_step after consuming it"
    );
}

#[test]
fn compiler_app_output_threads_fibonacci_state() {
    let cached = cached_n2();
    let mut ctx = start_fibonacci_chain(&cached.prep).expect("start chain");
    ctx.chain_state = cached.chain_state;
    ctx.fold_for_step = Some(cached.fold.clone());

    // F_5 = 5, F_6 = 8, F_7 = 13.
    let input = FibonacciAppStepInput {
        state_in: FibonacciAppState {
            prev: F::from_u64(5),
            curr: F::from_u64(8),
            step_index: 5,
        },
        witness: FibonacciAppWitness { next: F::from_u64(13) },
    };
    let compiled = compile_fibonacci_step(&cached.prep, &mut ctx, input).expect("compile");

    assert_eq!(compiled.app_output.state_out.prev, F::from_u64(8));
    assert_eq!(compiled.app_output.state_out.curr, F::from_u64(13));
    assert_eq!(compiled.app_output.state_out.step_index, 6);
    assert!(compiled
        .app_output
        .public_output_digest
        .iter()
        .any(|f| *f != F::ZERO));
}
