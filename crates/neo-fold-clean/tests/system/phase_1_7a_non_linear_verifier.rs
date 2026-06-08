//! Phase 1.7a — `verify_uncompressed` is a non-replay IVC verifier.
//!
//! After Phase 1.7 (revised) `verify_uncompressed` no longer iterates
//! `proof.steps` or `proof.public_batches`. Authority comes from re-running
//! the **terminal NIFS fold** (HyperNova §6.3 Construction 2; SuperNeo §7)
//! on the prover-snapshotted pre-final inputs (`final_fold.terminal_inputs`),
//! then binding the derived post-fold state to the recorded `proof.state`
//! plus a witness-side commitment/projection/low-norm cross-check.
//!
//! These tests pin both halves of the contract:
//!
//! 1. **Acceptance** — a finalized encoded-F' chain verifies; tampering
//!    *historical* `proof.steps` is **not** the IVC verifier's job and
//!    must still pass under this verifier (caught later by
//!    `paper::decider::validate_witness`).
//!
//! 2. **Rejection** — tampering any field on the authoritative path
//!    (NIFS proof, terminal-fold inputs, recorded running claims, recorded
//!    chain coordinates, recorded acc_digest, recorded witnesses) must
//!    surface as a named error.

#![allow(non_snake_case)]

#[path = "../support/mod.rs"]
mod support;

use neo_math::F;
use std::sync::{Mutex, OnceLock};

use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::ProofState;
use support::fibonacci_f_prime;

/// Shared cache of an `n = 1` finalized encoded-F' audit. The big
/// canonical plan makes preprocessing + folding expensive (~100s per
/// chain), and the ignored verifier tests would otherwise build many copies.
/// We cache the audit form (which contains everything needed to
/// derive `Uncompressed` via `audit.proof.clone()`); per-test
/// tampering happens on a clone, leaving the shared cache pristine.
struct CachedChain {
    prep: fibonacci_f_prime::FibonacciFPrimePreprocessing,
    audit: neo_fold_clean::UncompressedAudit,
}

static CHAIN_N1: OnceLock<CachedChain> = OnceLock::new();
static VERIFIER_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Keep the high-memory terminal-verifier checks from running in
/// parallel under libtest. They all share one cached chain, but each
/// verifier call still replays terminal NIFS/Π_CCS work.
fn run_serial<R>(f: impl FnOnce() -> R) -> R {
    let _guard = VERIFIER_TEST_LOCK
        .lock()
        .expect("phase_1_7a verifier test mutex poisoned");
    f()
}

fn cached_chain_n1() -> &'static CachedChain {
    CHAIN_N1.get_or_init(|| {
        let plan = fibonacci_f_prime::canonical_threaded_plan();
        let prep = fibonacci_f_prime::preprocess_seeded(&plan, 0x1F17_5EED).expect("preprocess");
        let mut builder = fibonacci_f_prime::FibonacciChainBuilder::new(&prep).expect("start builder");
        append_one_fibonacci_step(&mut builder);
        let audit = builder.finish_with_audit().expect("finish");
        CachedChain { prep, audit }
    })
}

fn append_one_fibonacci_step(builder: &mut fibonacci_f_prime::FibonacciChainBuilder<'_>) {
    builder
        .append_step(valid_app_step(1, 1, 0))
        .expect("append base step");
}

fn append_two_fibonacci_steps(builder: &mut fibonacci_f_prime::FibonacciChainBuilder<'_>) {
    let base = builder
        .append_step(valid_app_step(1, 1, 0))
        .expect("append base step");
    let state = base.app_output.state_out;
    builder
        .append_step(fibonacci_f_prime::FibonacciAppStepInput {
            state_in: state,
            witness: fibonacci_f_prime::FibonacciAppWitness {
                next: state.prev + state.curr,
            },
        })
        .expect("append recursive step");
}

fn valid_app_step(prev_u: u64, curr_u: u64, step_index: u64) -> fibonacci_f_prime::FibonacciAppStepInput {
    let prev = F::from_u64(prev_u);
    let curr = F::from_u64(curr_u);
    fibonacci_f_prime::FibonacciAppStepInput {
        state_in: fibonacci_f_prime::FibonacciAppState { prev, curr, step_index },
        witness: fibonacci_f_prime::FibonacciAppWitness { next: prev + curr },
    }
}

/// Get `(prep, Uncompressed)` from the shared cache, cloning the
/// terminal-only proof for tampering use.
fn finalized_encoded_f_prime_proof() -> (
    &'static fibonacci_f_prime::FibonacciFPrimePreprocessing,
    neo_fold_clean::Uncompressed,
) {
    let cached = cached_chain_n1();
    (&cached.prep, cached.audit.proof.clone())
}

/// Get `(prep, UncompressedAudit)` from the shared cache, cloning the
/// full audit so per-test tampering doesn't poison the cache.
fn finalized_encoded_f_prime_audit_proof() -> (
    &'static fibonacci_f_prime::FibonacciFPrimePreprocessing,
    neo_fold_clean::UncompressedAudit,
) {
    let cached = cached_chain_n1();
    (&cached.prep, cached.audit.clone())
}

// ── Acceptance ─────────────────────────────────────────────────────────────

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_accepts_finalized_encoded_f_prime_chain() {
    run_serial(|| {
        let (prep, finished) = finalized_encoded_f_prime_proof();
        neo_fold_clean::verify_uncompressed(&prep.prep, &finished)
            .expect("finalized encoded-F' chain verifies under the non-replay IVC verifier");
    });
}

/// The non-replay verifier deliberately ignores the audit trail
/// (`steps`, `public_batches`). The Phase 1.7-revised type split makes
/// this structural: [`Uncompressed`] doesn't even carry those fields.
///
/// This test verifies the same property at the
/// [`UncompressedAudit`] level: tampering a historical step's NIFS
/// payload leaves the terminal-only projection (`audit.proof`)
/// untouched, so `verify_uncompressed` still accepts it; the audit
/// verifier (`verify_uncompressed_audit`) rejects it.
#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_ignores_audit_trail_that_verify_uncompressed_audit_catches() {
    run_serial(|| {
        let (prep, mut audit) = finalized_encoded_f_prime_audit_proof();
        assert!(
            !audit.public_batches.is_empty() && !audit.public_batches[0].is_empty(),
            "test setup: audit must carry at least one public batch"
        );

        // Mutate audit-only history while leaving `audit.proof` untouched.
        // The terminal-only verifier consumes only the projected proof, but
        // audit replay must reject because it replays the recorded batches.
        audit.public_batches[0][0].x[0] += F::ONE;

        // (a) Non-replay IVC verifier accepts: it reads only `audit.proof`.
        neo_fold_clean::verify_uncompressed(&prep.prep, &audit.proof)
            .expect("non-replay verifier must not reject audit-trail tampers");

        // (b) Chain-replay verifier rejects: it reads the audit trail.
        assert!(
            neo_fold_clean::verify_uncompressed_audit(&prep.prep, &audit).is_err(),
            "verify_uncompressed_audit (chain-replay) must reject a tampered historical step"
        );
    });
}

// ── Rejection — recorded final-running tampers ─────────────────────────────

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_running_witness_entry() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        tamper_running_witness_entry(
            &mut finished,
            /* index = */ 0,
            /* row = */ 0,
            /* col = */ 0,
        );
        assert!(
            matches!(
                neo_fold_clean::verify_uncompressed(&prep.prep, &finished),
                Err(neo_fold_clean::Error::FinalAccumulatorWitnessCommitmentMismatch { index: 0 })
            ),
            "tampered running witness must surface as a commitment mismatch"
        );
    });
}

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_running_claim_commitment() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        tamper_running_claim_commitment(&mut finished, /* index = */ 0);
        // The tampered c.data also changes the derived acc_digest, but
        // bind_derived_state_to_recorded fires first.
        assert!(
            neo_fold_clean::verify_uncompressed(&prep.prep, &finished).is_err(),
            "tampered running claim.c must be rejected"
        );
    });
}

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_running_claim_public_input() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        tamper_running_claim_public_input(&mut finished, /* index = */ 0);
        // Either the derived-state binding (claim.X differs) or the
        // witness-projection cross-check will fire; both are correct
        // rejections.
        assert!(
            neo_fold_clean::verify_uncompressed(&prep.prep, &finished).is_err(),
            "tampered running claim.X must be rejected"
        );
    });
}

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_recorded_acc_digest() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        finished.state.acc_digest = [0xA5; 32];
        assert!(
            neo_fold_clean::verify_uncompressed(&prep.prep, &finished).is_err(),
            "tampered recorded acc_digest must be rejected (post-state binding or acc-digest check)"
        );
    });
}

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_unfinalized_proof_state() {
    run_serial(|| {
        let plan = fibonacci_f_prime::canonical_threaded_plan();
        let prep = fibonacci_f_prime::preprocess_seeded(&plan, 0x1F17_A105).expect("preprocess");
        let mut builder = fibonacci_f_prime::FibonacciChainBuilder::new(&prep).expect("start builder");
        append_two_fibonacci_steps(&mut builder);
        let unfinished = builder.into_audit().expect("unfinished audit");

        match &unfinished.proof.state.proof {
            ProofState::Active { latest, .. } => assert!(
                !latest.instances.is_empty(),
                "test setup: pre-finalize state must carry a trailing latest"
            ),
            ProofState::Initial => panic!("test setup: 2-step encoded-F' proof must be Active"),
        }
        assert!(
            matches!(
                neo_fold_clean::verify_uncompressed(&prep.prep, &unfinished.proof),
                Err(neo_fold_clean::Error::NotFinalized)
            ),
            "pre-finalize proof must be rejected as NotFinalized"
        );
    });
}

// ── Rejection — chain coordinates ──────────────────────────────────────────

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_chunk_count() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        finished.state.chunk_count += 1;
        // chunk_count is absorbed into state_x_out and also linked as
        // recorded Construction-2 state. A prover-carried relabel must
        // fail when the verifier re-runs the terminal fold and binds the
        // derived state back to proof.state.
        assert!(
            neo_fold_clean::verify_uncompressed(&prep.prep, &finished).is_err(),
            "tampered chunk_count must be rejected by derived-state binding"
        );
    });
}

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_z_i() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        finished.state.z_i[0] ^= 0xFF;
        assert!(
            neo_fold_clean::verify_uncompressed(&prep.prep, &finished).is_err(),
            "tampered z_i must be rejected via terminal-fold x_out check"
        );
    });
}

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_z_0() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        finished.state.z_0[0] ^= 0xFF;
        // z_0 is intentionally omitted from the compact state_x_out digest:
        // it is verifier-derived from preprocessing and linked as recorded
        // Construction-2 state. The non-replay verifier must still reject a
        // prover-carried z_0 relabel before it becomes a self-consistent
        // terminal state.
        let err =
            neo_fold_clean::verify_uncompressed(&prep.prep, &finished).expect_err("tampered z_0 must be rejected");
        assert!(
            matches!(err, neo_fold_clean::Error::PostStateMismatch),
            "tampered z_0 should reject through compact-state anchor binding, got {err:?}"
        );
    });
}

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_pc() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        finished.state.pc += 1;
        // pc is absorbed into state_x_out and also pinned as the
        // single-program selector in F'. The non-replay verifier must
        // reject a prover-carried pc relabel before it becomes a
        // self-consistent terminal state.
        let err = neo_fold_clean::verify_uncompressed(&prep.prep, &finished).expect_err("tampered pc must be rejected");
        assert!(
            matches!(err, neo_fold_clean::Error::PostStateMismatch),
            "tampered pc should reject through compact-state anchor binding, got {err:?}"
        );
    });
}

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_public_trace() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        finished.state.public_trace[0] ^= 0xFF;
        assert!(
            neo_fold_clean::verify_uncompressed(&prep.prep, &finished).is_err(),
            "tampered public_trace must be rejected via terminal-fold x_out check"
        );
    });
}

// ── Rejection — terminal-fold inputs and NIFS proof ────────────────────────

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_terminal_nifs_proof() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        let final_fold = finished.final_fold.as_mut().expect("final_fold present");
        support::mutate_ce_claim(&mut final_fold.nifs.pi_dec.children[0]);
        assert!(
            neo_fold_clean::verify_uncompressed(&prep.prep, &finished).is_err(),
            "tampered terminal NIFS proof must be rejected by the in-verifier NIFS.V"
        );
    });
}

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_terminal_latest_x() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        let final_fold = finished.final_fold.as_mut().expect("final_fold present");
        let latest_claim = &mut final_fold.terminal_inputs.latest.instances[0].claim;
        latest_claim.x[0] += F::ONE;
        assert!(
            neo_fold_clean::verify_uncompressed(&prep.prep, &finished).is_err(),
            "tampered terminal-latest public input x must be rejected (Π_CCS sumcheck fails)"
        );
    });
}

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_cleared_terminal_latest() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        let final_fold = finished.final_fold.as_mut().expect("final_fold present");
        final_fold.terminal_inputs.latest.instances.clear();
        assert!(
            neo_fold_clean::verify_uncompressed(&prep.prep, &finished).is_err(),
            "cleared terminal-latest must be rejected by NIFS.V shape check"
        );
    });
}

#[test]
#[ignore = "canonical verifier replay is memory-heavy; run explicitly with --ignored"]
fn verify_uncompressed_rejects_tampered_terminal_pre_final_running_commitment() {
    run_serial(|| {
        let (prep, mut finished) = finalized_encoded_f_prime_proof();
        let final_fold = finished.final_fold.as_mut().expect("final_fold present");
        let pre_running = &mut final_fold.terminal_inputs.pre_final_running;
        // Mutate one claim.c byte. The verifier-derived post_running depends
        // on this input; the resulting acc_digest / claims chain will not
        // match the recorded `proof.state.running`.
        pre_running.claims[0].c.data[0] += F::ONE;
        assert!(
            neo_fold_clean::verify_uncompressed(&prep.prep, &finished).is_err(),
            "tampered terminal-input pre_final_running commitment must be rejected"
        );
    });
}

// ── Tamper helpers ─────────────────────────────────────────────────────────

fn tamper_running_witness_entry(finished: &mut neo_fold_clean::Uncompressed, index: usize, row: usize, col: usize) {
    let running = running_mut(finished);
    running.witnesses[index][(row, col)] += F::ONE;
}

fn tamper_running_claim_commitment(finished: &mut neo_fold_clean::Uncompressed, index: usize) {
    let running = running_mut(finished);
    running.claims[index].c.data[0] += F::ONE;
}

fn tamper_running_claim_public_input(finished: &mut neo_fold_clean::Uncompressed, index: usize) {
    let running = running_mut(finished);
    running.claims[index].X[(0, 0)] += F::ONE;
}

fn running_mut(
    finished: &mut neo_fold_clean::Uncompressed,
) -> &mut neo_fold_clean::paper::construction2::RunningInstance {
    match &mut finished.state.proof {
        ProofState::Active { running, .. } => running,
        ProofState::Initial => panic!("test setup: encoded-F' proof must be Active after finalization"),
    }
}
