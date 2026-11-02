//! The option-B lifecycle end to end on the Poseidon2 hash chain: each step
//! folds the previous step's instance (Lean `Spec/Folding/Light`,
//! `Spec/Folding/Refresh`), the step circuit's rows hold on its words, and
//! the terminal (`OptionB.Terminal.HoldsFor`) accepts after every step. A
//! small cap reaches every pair of fold kind and relation in a few steps.

#[path = "../option_b_support/mod.rs"]
mod support;

use super::{Envelope, OptionB, OptionBError};
use crate::application::{poseidon2_hash_chain_v1, Affine, ApplicationBuilder, ApplicationCircuit};
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use support::{hash_chain_step, packages, words};

const MINIMUM_SECURITY_BITS: u32 = 114;

fn prover(cap: u64) -> OptionB {
    let mut prover = OptionB::new(packages(), MINIMUM_SECURITY_BITS).expect("option-B prover");
    prover.cap = cap;
    prover
}

/// Runs `steps` hash-chain steps from `z0`, verifies after each, and returns
/// the last proof and the fold kinds' program counters.
fn run(prover: &OptionB, steps: u64) -> (Envelope, Vec<u64>) {
    let z0 = words([10, 11, 12, 13]);
    let mut envelope = prover.initial(z0).expect("initial proof");
    prover
        .verify(0, z0, z0, &envelope)
        .expect("zero steps verify");
    let mut circuits = Vec::new();
    for step in 0..steps {
        let message = words([20 + step, 21 + step, 22 + step, 23 + step]);
        let next = hash_chain_step(envelope.current, message);
        envelope = prover.extend(&envelope, &message, next).expect("step");
        circuits.push(envelope.pc);
        prover
            .verify(step + 1, z0, next, &envelope)
            .unwrap_or_else(|error| panic!("step {step} verifies: {error}"));
    }
    (envelope, circuits)
}

/// Cap 2: two light folds into slot 0, the second over an accumulator whose
/// witness lies outside {-1, 0, 1}; a refresh of slot 0 under the light
/// relation; then a light fold of the refresh instance into slot 1 under the
/// refresh relation. Each tampered proof fails exactly one check.
#[test]
fn light_folds_and_a_refresh_verify() {
    let prover = prover(2);
    let (envelope, circuits) = run(&prover, 5);
    assert_eq!(circuits, [1, 1, 1, 2, 1]);

    let (z0, current) = (envelope.z0, envelope.current);
    let rejects = |envelope: &Envelope, expected: &str| match prover.verify(5, z0, current, envelope) {
        Err(error) => assert!(error.to_string().contains(expected), "{expected}: {error}"),
        Ok(()) => panic!("{expected}: accepted"),
    };
    let wrong_current = [current[0] + F::ONE, current[1], current[2], current[3]];
    assert!(matches!(
        prover.verify(5, z0, wrong_current, &envelope),
        Err(OptionBError::Statement(_))
    ));

    let mut deeper = envelope.clone();
    deeper.slots[1].depth += 1;
    rejects(&deeper, "public input differs from the hash of the state");

    let mut tampered = envelope.clone();
    tampered.slots[1].accumulator.witness[(0, 0)] += F::ONE;
    rejects(&tampered, "option-B slot 1");

    let mut narrow = envelope.clone();
    narrow.slots[0].accumulator.claim.X = Mat::zero(D, 1, F::ZERO);
    rejects(&narrow, "option-B slot 0: claim shape");

    let recommitted = |column: usize| {
        let mut tampered = envelope.clone();
        let fresh = tampered.fresh.as_mut().expect("fresh instance");
        let entry = &mut fresh.witness.Z[(column % D, column / D)];
        *entry = if *entry == F::ZERO { F::ONE } else { F::ZERO };
        fresh.claim.c = commit_production_signed_unit_prefix_matrix(&fresh.witness.Z).expect("commitment");
        tampered
    };
    rejects(&recommitted(300), "a relation row fails");
    let blocks = envelope
        .fresh
        .as_ref()
        .expect("fresh instance")
        .witness
        .Z
        .cols();
    rejects(&recommitted(blocks * D - 1), "nonzero witness tail");

    let mut tampered = envelope.clone();
    let fresh = tampered.fresh.as_mut().expect("fresh instance");
    let mut data = fresh.claim.c.data().to_vec();
    data[0] += F::ONE;
    fresh.claim.c = Commitment::new(fresh.claim.c.kappa(), data).expect("commitment");
    rejects(&tampered, "fixed-key commitment differs from the witness");
}

/// Cap 0: every fold is a refresh, so step 2 refreshes slot 1 under the
/// refresh relation.
#[test]
fn refresh_folds_under_both_relations_verify() {
    let (_, circuits) = run(&prover(0), 3);
    assert_eq!(circuits, [1, 2, 2]);
}

/// The addition application with two witness words, so the shared words after
/// them move: the next state is the current state plus `[w0, w1, w0, w1]`.
fn addition() -> ApplicationCircuit {
    let builder = ApplicationBuilder::new(2).expect("builder");
    let input = builder.input_state();
    let private = builder.private_inputs().to_vec();
    let outputs = std::array::from_fn(|lane| Affine::from(input[lane]) + Affine::from(private[lane % 2]));
    builder.finish(outputs).expect("addition application")
}

/// Cap 1 with a native application in place of Lean's: the base step, a
/// light step and a refresh step, each verified; a step that claims a wrong
/// next state fails the application's rows.
fn native_application_verifies(application: ApplicationCircuit) {
    let packages = packages()
        .with_application(application.records().clone(), application.private_input_count())
        .expect("packages with the application");
    let mut prover = OptionB::new(packages, MINIMUM_SECURITY_BITS).expect("option-B prover");
    prover.cap = 1;
    let z0 = words([10, 11, 12, 13]);
    let mut envelope = prover.initial(z0).expect("initial proof");
    let mut circuits = Vec::new();
    for step in 0..3 {
        let message = &words([20 + step, 21 + step, 22 + step, 23 + step])[..application.private_input_count()];
        let next = application
            .execute(envelope.current, message)
            .expect("application step")
            .output_state();
        let wrong = [next[0] + F::ONE, next[1], next[2], next[3]];
        let rejected = prover.extend(&envelope, message, wrong).expect("step");
        assert!(
            matches!(
                prover.verify(step + 1, z0, wrong, &rejected),
                Err(OptionBError::Fresh("a relation row fails"))
            ),
            "step {step} with a wrong next state"
        );
        envelope = prover.extend(&envelope, message, next).expect("step");
        circuits.push(envelope.pc);
        prover
            .verify(step + 1, z0, next, &envelope)
            .unwrap_or_else(|error| panic!("step {step} verifies: {error}"));
    }
    assert_eq!(circuits, [1, 1, 2]);
}

#[test]
fn native_hash_chain_verifies() {
    native_application_verifies(poseidon2_hash_chain_v1().expect("hash chain"));
}

#[test]
fn native_addition_verifies() {
    native_application_verifies(addition());
}

/// The CPU time of each option-B step with the production cap, against
/// `poseidon_base_step_matches_lean_and_verifies` (today's one-circuit fold)
/// on the same machine. Steps 1 to 3 are light; a refresh step runs at the
/// cap, so this test reaches one with cap 2.
#[test]
#[ignore = "Timing run; run this test separately under the 300-second cap."]
fn option_b_step_times() {
    let started = std::time::Instant::now();
    let prover = prover(2);
    eprintln!("option-B preparation elapsed={:?}", started.elapsed());
    let z0 = words([10, 11, 12, 13]);
    let mut envelope = prover.initial(z0).expect("initial proof");
    for step in 0..6 {
        let message = words([20 + step, 21 + step, 22 + step, 23 + step]);
        let next = hash_chain_step(envelope.current, message);
        let proving = std::time::Instant::now();
        envelope = prover.extend(&envelope, &message, next).expect("step");
        let proved = proving.elapsed();
        let verifying = std::time::Instant::now();
        prover
            .verify(step + 1, z0, next, &envelope)
            .expect("verifies");
        eprintln!(
            "option-B step={step} circuit={} prove={proved:?} verify={:?}",
            envelope.pc,
            verifying.elapsed()
        );
    }
}
