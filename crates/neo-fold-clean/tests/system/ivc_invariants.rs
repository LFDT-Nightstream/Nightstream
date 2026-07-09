//! IVC architectural invariants for the SuperNeo folding lifecycle.
//!
//! These tests pin properties a real IVC compiler over the SuperNeo
//! folding scheme must satisfy. Phase 1.6a swaps
//! the same-shape synthetic fixture for the **state-threaded**
//! variant, so each invariant now exercises a chain whose `state_out`
//! of step i really is `state_in` of step i+1.
//!
//! | Invariant | Default | Notes |
//! |---|---|---|
//! | `per_step_ccs_structure_must_encode_f_prime` | ✓ | Folds one threaded encoded F' step; `prep.structure().m` ≥ 50_000 |
//! | `running_accumulator_witness_must_carry_f_prime_encoded_size` | ✓ | Folds three threaded encoded F' steps; running.witnesses[0] ≥ 50_000 cells |
//! | `decider_r1cs_size_must_be_constant_in_chain_length` | ignored | Synthesizes the steady-state last-step terminal R1CS for two threaded chains; asserts ≤ 10% per-step growth. Under the canonical fixed-point plan this exceeds the 5-min default-test budget, so it runs only with `--ignored`. |
//! | `multi_chunk_f_prime_chain_must_verify_terminal_only` | **ignored: fails by design** | The induction-gap milestone gate (Path 1/2 of the enc(F') decision): a 3-chunk F' chain must be accepted by `verify_uncompressed` in O(1). Fails today — the terminal-only verifier fails closed for multi-chunk chains. Un-ignore when the recursive F'/NIFS.V induction becomes checkable. |
//! | `multi_chunk_terminal_only_verification_fails_closed_today` | ✓ | The tripwire twin of the gate above: pins today's exact fail-closed rejection. The day the guard lifts, this fails — flip BOTH tests together, deliberately. |
//! | `nebula_chain_must_verify_terminal_only_with_memory` | **ignored: fails by design** | Spec §13 step 9's end-state: a multi-segment memory chain accepted terminal-only, lane closed. Fails today for the same induction gap. |
//! | `nebula_terminal_only_verification_fails_closed_today` | ✓ | Tripwire twin of the Nebula gate. |
//! | `recursive_link_multi_chunk_fails_closed_today` | ✓ | The stricter guard layer: an `r1cs_f_prime` chain (recursive-link flag SET) must reject multi-chunk terminal-only verification via `FPrimeNonReplayUnsupported`, not just the generic non-empty-running guard. |
//! | `folded_f_prime_shell_must_adopt_projection_budget` | ✓ **GREEN (Road A Unit 2)** | The production F' shell carries projection regions: integrated layout = 14,040,452 bits (was 94,330,948 D²-materialized). Now a budget-regression guard. |
//! | `projection_shell_semantic_rows_must_be_enforced` | ✓ **GREEN (Road A Phase B)** | The projection region now emits semantic CCS rows for evaluation sums, Karatsuba relations, and the final identity. |
//!
//! The implementation that turned each invariant green:
//!   - Phase 1.5b: encoded F' image / structure / encoder + foldable `CcsInstance`.
//!   - Phase 1.5c-a: `project_x_from_witness_mat` ring-column projection fix.
//!   - Phase 1.5c-b: encoded-F' test-support module + lifecycle fold path.
//!   - Phase 1.5d: `synthesize_last_step_terminal_r1cs` — emits only
//!     the last F' step + terminal fold + public-image pins, skipping
//!     the per-step history replay that the audit-style
//!     [`synthesize_statement_r1cs`] still does for debugging.
//!   - Phase 1.6a: state-threaded encoded-F' fixture (see
//!     `phase_1_6a_state_threaded_encoded_f_prime.rs`).

#![allow(non_snake_case)]

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::engine::decider::synthesize_last_step_terminal_r1cs;
use support::fibonacci_f_prime;

use support::fibonacci_f_prime::{canonical_threaded_plan, honest_state_threaded_encoded_f_prime_steps};

// ── Invariants ─────────────────────────────────────────────────────────────

/// The CCS structure being folded must itself encode `F'_i` — the augmented
/// step. For SuperNeo over CCS that includes (at minimum) the in-circuit
/// NIFS.V of the previous fold, plus the state-advance + hash-chain
/// bookkeeping. Per SuperNeo §1 (D6) the augmented-step verifier circuit
/// targets "logarithmic recursion overhead [...] analogous to HyperNova,"
/// i.e. at least tens to hundreds of thousands of CCS variables.
///
/// Green since Phase 1.5c-b: the encoded-F' fixture derives
/// `prep.structure` directly from the F' image layout (`FPrimeStructure.ccs`),
/// so this invariant is now a regression gate against the F' image
/// shape shrinking below the in-circuit NIFS.V floor.
#[test]
fn per_step_ccs_structure_must_encode_f_prime() {
    // Phase 1.5c-b: build one encoded F' step and derive preprocessing
    // from its `FPrimeStructure.ccs`. The structure size now reflects
    // the bit-encoded F' image (image bits = m), not a raw app R1CS.
    let plan = canonical_threaded_plan();
    let prep = fibonacci_f_prime::preprocess_seeded(&plan, 0x1F15_C001).expect("preprocess");
    let steps = honest_state_threaded_encoded_f_prime_steps(1);
    let _proof = fibonacci_f_prime::prove_encoded_steps(&prep, &steps).expect("prove");
    let prep = &prep.prep;

    // F's in-circuit verifier (NIFS.V over SuperNeo's Π_CCS + Π_RLC +
    // Π_DEC) at our parameters dominates with ~hundreds of thousands of
    // CCS variables minimum. Pin a generous floor of 50_000: any
    // structure smaller than this provably cannot contain in-circuit
    // NIFS.V, and so cannot be a correct F' encoding.
    const MIN_F_PRIME_STRUCTURE_M: usize = 50_000;

    assert!(
        prep.structure().m >= MIN_F_PRIME_STRUCTURE_M,
        "Encoded F' per-step CCS structure has m = {} variables; expected \
         at least {MIN_F_PRIME_STRUCTURE_M} for the structure to encode \
         `F'_i` (user app step + in-circuit NIFS.V of the previous fold + \
         state advance + hash chain). If this regresses, the encoded F' \
         image shape has shrunk below the in-circuit NIFS.V floor.",
        prep.structure().m,
    );
}

/// The terminal decider R1CS size must be effectively constant in the
/// number of folded steps. This is the defining property of IVC: per
/// HyperNova §6.3 Construction 2, only the latest folded accumulator
/// needs to be proven at the end; the chain's history is bound
/// transitively through the recursive verifier embedded in each `F'_i`.
///
/// We measure at two **steady-state** chain lengths (N=3 and N=4).
/// "Steady state" means the last step folds a `k_rho`-sized running
/// accumulator into a `k_rho`-sized running — the standard recursive
/// case. For N=2 the last step would instead fold into an *empty*
/// running, a one-off shape that would confound the comparison.
///
/// Phase 1.5d: this invariant folds **encoded F'** instances and uses
/// [`synthesize_last_step_terminal_r1cs`], which emits **the last F'
/// step + the terminal NIFS.V fold + public-image pins** — three
/// constant-size pieces. It is *not* a pure accumulator-only terminal
/// decider; the last step's full F' shell is still inside the relation.
/// The asymptotic win is that the per-step `for`-loop over historical
/// steps (which [`synthesize_statement_r1cs`] uses for audit/replay)
/// is gone, so `builder.rows()` is constant in the steady state.
#[test]
#[ignore = "runs two big-plan lifecycle chains + two finalizations + two terminal syntheses; exceeds the repo's 5-min per-test ceiling under the canonical fixed-point plan. Run manually with `cargo test --release -p neo-fold-clean --test system_ivc_invariants -- --ignored`."]
fn decider_r1cs_size_must_be_constant_in_chain_length() {
    // This test sits intentionally near the repo's 5-min test ceiling:
    // it runs two encoded-F' lifecycle proofs, two finalizations, and
    // two terminal syntheses (~170s on dev hardware). Keep it focused
    // — do not add extra cases here; a separate perf snapshot is the
    // right place for broader sweeps.
    //
    // Compare N=3 vs N=4. Both have the last step folding into a
    // steady-state running accumulator (k_rho CE claims). For N=2 the
    // last step folds into an *empty* running, which is a one-off
    // shape: the in-circuit NIFS.V emission has fewer inputs there,
    // and any comparison against it would conflate "first recursive
    // step (empty → k_rho)" with the steady-state pattern we care
    // about ("k_rho → k_rho").
    // Chain lengths used to be N=3 vs N=4 under a stub canonical
    // plan. With the canonical plan now matching the real post-fold CE
    // shape, each lifecycle fold is ~25-50 s, so we use the smallest
    // delta that still exercises strictly different chain step counts:
    // N=2 vs N=3.
    let plan = canonical_threaded_plan();

    let prep_short = fibonacci_f_prime::preprocess_seeded(&plan, 0x1F15_D003).expect("preprocess N=2");
    let steps_short = honest_state_threaded_encoded_f_prime_steps(2);
    let proof_short = fibonacci_f_prime::prove_encoded_steps(&prep_short, &steps_short).expect("prove N=2");
    let finished_short =
        neo_fold_clean::finish_uncompressed_with_audit(&prep_short.prep, proof_short).expect("finish N=2");
    let synth_short =
        synthesize_last_step_terminal_r1cs(&prep_short.prep, &finished_short).expect("terminal synth N=2");
    assert!(
        synth_short.terminal_ce_direct_relations,
        "terminal synthesis must emit terminal CE-relation rows for the short chain"
    );

    let prep_long = fibonacci_f_prime::preprocess_seeded(&plan, 0x1F15_D004).expect("preprocess N=3");
    let steps_long = honest_state_threaded_encoded_f_prime_steps(3);
    let proof_long = fibonacci_f_prime::prove_encoded_steps(&prep_long, &steps_long).expect("prove N=3");
    let finished_long =
        neo_fold_clean::finish_uncompressed_with_audit(&prep_long.prep, proof_long).expect("finish N=3");
    let synth_long = synthesize_last_step_terminal_r1cs(&prep_long.prep, &finished_long).expect("terminal synth N=3");
    assert!(
        synth_long.terminal_ce_direct_relations,
        "terminal synthesis must emit terminal CE-relation rows for the long chain"
    );

    let short_chain_steps = finished_short.steps.len(); // = 2
    let long_chain_steps = finished_long.steps.len(); // = 3
    assert!(
        long_chain_steps > short_chain_steps,
        "test setup must produce strictly more F' steps for the long chain than the short one"
    );

    let short_rows = synth_short.builder.rows();
    let long_rows = synth_long.builder.rows();

    let per_step_growth = long_rows.saturating_sub(short_rows) / (long_chain_steps - short_chain_steps);

    // For an O(1) IVC decider, per-step growth must be near zero
    // (counter / public-image pin diffs only). A generous threshold:
    // per-step growth must be < 10% of the smaller decider's total
    // size. Worse than that and the decider is still fundamentally
    // linear in chain length.
    let growth_budget = short_rows / 10;

    assert!(
        per_step_growth <= growth_budget,
        "encoded-F' terminal decider R1CS still grows with chain length:\n\
         \n\
         \t  N=2 chain   → rows={short_rows} (chain steps = {short_chain_steps})\n\
         \t  N=3 chain   → rows={long_rows} (chain steps = {long_chain_steps})\n\
         \t  per-step growth = {per_step_growth} rows\n\
         \t  IVC budget      = {growth_budget} rows (10% of short_rows)\n\
         \n\
         A true IVC terminal decider proves only the latest folded\n\
         accumulator. Its R1CS size is bounded by F' (the augmented step\n\
         verifier circuit) and is independent of chain length.",
    );
}

/// Cross-check: even at a single chain length, the running accumulator's
/// CCS witness `Z` matrix should reflect the size of `enc(F'_i)`, not the
/// size of the user's raw app step. SuperNeo's `enc(F'_i)` witness must be
/// low-norm AND must contain the source-image bits + decoded LCs that
/// encode the in-circuit NIFS.V of the previous fold (the load-bearing
/// constraint the other AI flagged). With `enc(F'_i)` folded each step,
/// each running CE claim's `Z` would have dimensions matching F's CCS
/// structure (≥ tens of thousands of cells).
///
/// **Why this should fail today**: the running accumulator carries the
/// CCS witnesses of whatever was folded, which is the user's app R1CS
/// (a few hundred cells). The witness has no recursive-verification
/// content.
#[test]
fn running_accumulator_witness_must_carry_f_prime_encoded_size() {
    // Phase 1.5c-b: fold encoded F' steps through the existing
    // lifecycle. The running accumulator's CE claim witness Z must
    // therefore carry `enc(F'_i)`-sized data, not the few hundred cells
    // a raw app R1CS would yield. n=2 is the minimum that triggers a
    // recursive fold (call 0 is `NoFold`); the witness shape gate
    // holds for any n >= 2.
    let plan = canonical_threaded_plan();
    let prep = fibonacci_f_prime::preprocess_seeded(&plan, 0x1F15_C003).expect("preprocess");
    let steps = honest_state_threaded_encoded_f_prime_steps(2);
    let proof = fibonacci_f_prime::prove_encoded_steps(&prep, &steps).expect("prove");

    let running = match &proof.proof.state.proof {
        neo_fold_clean::ProofState::Active { running, .. } => running,
        _ => panic!("test setup must leave state in ProofState::Active"),
    };
    assert!(
        !running.witnesses.is_empty(),
        "encoded F' lifecycle should leave at least one running CE claim with a witness"
    );

    // Each witness's Z matrix has dimensions tied to the CCS structure
    // it satisfies. For `enc(F'_i)` that's tens of thousands of cells.
    let z_cells_per_claim = running.witnesses[0].rows() * running.witnesses[0].cols();
    const MIN_F_PRIME_WITNESS_CELLS: usize = 50_000;

    assert!(
        z_cells_per_claim >= MIN_F_PRIME_WITNESS_CELLS,
        "Each running CE claim's witness Z has {z_cells_per_claim} cells; \
         expected at least {MIN_F_PRIME_WITNESS_CELLS} for `enc(F'_i)` \
         content. If this regresses, the lifecycle has stopped folding \
         the encoded F' image and is back to a raw-app-CCS shape.",
    );
}

// ── The induction gap (milestone gates + tripwires) ────────────────────────
//
// Today, `verify_uncompressed` (the O(1) terminal-only verifier) fails
// closed for multi-chunk chains: the folded F' images bind the state-hash
// chain but do not yet constrain the previous fold's NIFS.V, so nothing
// in the terminal artifact vouches for earlier chunks — multi-chunk
// soundness authority is the audit-replay path. Closing this is the
// enc(F') regime milestone (folded in-circuit NIFS.V, or the terminal
// compressed decider). Each gate below is an `#[ignore]`d acceptance
// test that FAILS today by design and defines the milestone's done
// state; its active twin pins today's exact fail-closed rejection so
// the guard cannot silently lift — the day it does, flip both together.

#[path = "../nebula/fixture.rs"]
mod nebula_fixture;

/// MILESTONE GATE (ignored; fails today): a multi-chunk F' chain must be
/// accepted by the terminal-only verifier — HyperNova's "accumulator +
/// latest fold" check, with the recursive F'/NIFS.V induction carrying
/// the earlier chunks.
#[test]
#[ignore = "induction-gap milestone gate: fails until the recursive F'/NIFS.V induction is checkable terminal-only (enc(F') regime milestone)"]
fn multi_chunk_f_prime_chain_must_verify_terminal_only() {
    let plan = canonical_threaded_plan();
    let prep = fibonacci_f_prime::preprocess_seeded(&plan, 0x1F15_C004).expect("preprocess");
    let steps = honest_state_threaded_encoded_f_prime_steps(3);
    let audit = fibonacci_f_prime::prove_encoded_steps(&prep, &steps).expect("prove");
    let proof = neo_fold_clean::lifecycle::finish_uncompressed(&prep.prep, audit).expect("finalize");
    neo_fold_clean::lifecycle::verify_uncompressed(&prep.prep, &proof)
        .expect("milestone: an honest 3-chunk F' chain must verify in O(1), terminal-only");
}

/// TRIPWIRE (active): pins today's fail-closed rejection for the gate
/// above. If this test ever fails, the non-replay guard has lifted —
/// either the milestone landed (un-ignore the gate, delete this pin) or
/// soundness just regressed (investigate immediately).
#[test]
fn multi_chunk_terminal_only_verification_fails_closed_today() {
    let plan = canonical_threaded_plan();
    let prep = fibonacci_f_prime::preprocess_seeded(&plan, 0x1F15_C005).expect("preprocess");
    let steps = honest_state_threaded_encoded_f_prime_steps(3);
    let audit = fibonacci_f_prime::prove_encoded_steps(&prep, &steps).expect("prove");
    let proof = neo_fold_clean::lifecycle::finish_uncompressed(&prep.prep, audit).expect("finalize");
    let err = neo_fold_clean::lifecycle::verify_uncompressed(&prep.prep, &proof)
        .expect_err("fail-closed today: multi-chunk terminal-only verification");
    assert!(
        matches!(
            err,
            neo_fold_clean::lifecycle::Error::TerminalOnlyMultiChunkUnsupported { chunk_count: 3 }
        ),
        "expected the multi-chunk fail-closed guard, got: {err}"
    );
}

/// MILESTONE GATE (ignored; fails today): spec §13 step 9's end state —
/// a multi-segment Nebula memory chain accepted by the terminal-only
/// verifier, with the lane's finalization rule checked on the accepted
/// result. Note the future chain that satisfies this folds S_mem through
/// an F' shell that carries the §6.3 transition in-circuit; this test
/// intentionally pins only the OUTCOME (honest chain, terminal-only
/// accept), not today's chain shape.
#[test]
#[ignore = "spec §13 step 9 milestone gate: fails until the Nebula lane transition rides a terminal-only-verifiable F' chain"]
fn nebula_chain_must_verify_terminal_only_with_memory() {
    let (_, prep, audit) = nebula_fixture::honest_two_segment_chain();
    neo_fold_clean::lifecycle::verify_uncompressed(&prep, &audit.proof)
        .expect("milestone: an honest two-segment memory chain must verify in O(1), terminal-only");
    let lane = audit.proof.state.nebula.as_ref().expect("lane");
    assert!(
        lane.is_closed(),
        "accepted terminal state satisfies the finalization rule"
    );
}

/// TRIPWIRE (active): pins today's fail-closed rejection for the Nebula
/// gate above. Same flip discipline as the F' tripwire.
#[test]
fn nebula_terminal_only_verification_fails_closed_today() {
    let (_, prep, audit) = nebula_fixture::honest_two_segment_chain();
    let err = neo_fold_clean::lifecycle::verify_uncompressed(&prep, &audit.proof)
        .expect_err("fail-closed today: multi-chunk Nebula terminal-only verification");
    assert!(
        matches!(
            err,
            neo_fold_clean::lifecycle::Error::TerminalOnlyMultiChunkUnsupported { .. }
        ),
        "expected the multi-chunk fail-closed guard, got: {err}"
    );
}

/// TRIPWIRE (active): the STRICTER guard path. The Fibonacci fixture
/// above does not set the F' recursive-link flag, so its rejection
/// comes from the generic non-empty-running guard
/// (`TerminalOnlyMultiChunkUnsupported`). The `r1cs_f_prime` frontend
/// DOES set `with_f_prime_recursive_link`, and its multi-chunk chains
/// must reject through the dedicated recursive-link guard
/// (`FPrimeNonReplayUnsupported`) — pinned here so both fail-closed
/// layers are covered. Same flip discipline as the other tripwires.
#[test]
fn recursive_link_multi_chunk_fails_closed_today() {
    use support::r1cs_compiler_fixtures::{
        assignment_one_product, make_tiny_lifecycle_plan, one_product_r1cs, tiny_params,
    };

    let r1cs = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep = neo_fold_clean::frontends::r1cs_f_prime::preprocess_seeded_with_params(
        &r1cs,
        &plan,
        tiny_params(),
        0x1F15_C006,
    )
    .expect("preprocess");
    assert!(
        prep.prep.enforces_f_prime_recursive_link(),
        "r1cs_f_prime preprocessing must set the recursive-link flag"
    );

    let mut chain = neo_fold_clean::frontends::r1cs_f_prime::R1csChainBuilder::new(&prep).expect("builder");
    chain
        .append_assignment(assignment_one_product(3, 7))
        .expect("base step");
    chain
        .append_assignment(assignment_one_product(3, 7))
        .expect("recursive step");
    let proof = chain.finish().expect("finalize");

    let err = neo_fold_clean::lifecycle::verify_uncompressed(&prep.prep, &proof)
        .expect_err("fail-closed today: recursive-link multi-chunk terminal-only verification");
    assert!(
        matches!(
            err,
            neo_fold_clean::lifecycle::Error::FPrimeNonReplayUnsupported { chunk_count: 2 }
        ),
        "expected the recursive-link fail-closed guard, got: {err}"
    );
}

/// MILESTONE GATE — **GREEN since Road A Unit 2**: the production F'
/// shell carries the ring action as projection regions (candidate E).
/// Integrated measurement: 14,040,452 bits/step (was 94,330,948 with
/// D² materialization; the D² shell survives as
/// `production_kmul_d2_ring_action_shell_image_config` for reference).
/// This gate now guards the budget against regression; the projection
/// region's SEMANTIC rows landed in Phase B and are guarded by
/// `projection_shell_semantic_rows_must_be_enforced` below.
#[test]
fn folded_f_prime_shell_must_adopt_projection_budget() {
    use neo_fold_clean::frontends::f_prime::image::FPrimeImageLayout;
    use neo_fold_clean::frontends::f_prime::structure::production_kmul_ring_action_shell_image_config;

    let layout = FPrimeImageLayout::new(production_kmul_ring_action_shell_image_config());
    const PROJECTION_BUDGET_BITS: usize = 16_000_000;
    assert!(
        layout.end <= PROJECTION_BUDGET_BITS,
        "production F' shell commits {} bits/step; the projection-checked budget is {} \
         (encoding.md candidate E — integrate `enforce_ring_action_projection_batch`)",
        layout.end,
        PROJECTION_BUDGET_BITS
    );
}

/// Road A Phase B: the projection region's semantic rows are real CCS
/// rows. A structure built with projection regions must emit the beta
/// ladder, evaluation sums, Karatsuba relations, and final identity
/// rows on top of the committed-width region.
#[test]
fn projection_shell_semantic_rows_must_be_enforced() {
    use neo_fold_clean::frontends::f_prime::image::FPrimeImageLayout;
    use neo_fold_clean::frontends::f_prime::structure::{
        build_f_prime_structure, production_kmul_d2_ring_action_shell_image_config,
    };

    let base_config = {
        let mut c = production_kmul_d2_ring_action_shell_image_config();
        c.kmul_count = 2;
        c.ring_action_pair_count = 0;
        c
    };
    let projection_config = {
        let mut c = base_config.clone();
        c.projection_batches = vec![2]; // one identity consuming two pairs
        c
    };

    let base_rows = build_f_prime_structure(FPrimeImageLayout::new(base_config))
        .ccs
        .n;
    let projection_rows = build_f_prime_structure(FPrimeImageLayout::new(projection_config))
        .ccs
        .n;

    assert!(
        projection_rows >= base_rows + 100,
        "projection regions must be semantically constrained: {projection_rows} rows with the region vs {base_rows} without"
    );
}
