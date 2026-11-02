//! The option-B lifecycle end to end on the Poseidon2 hash chain: each step
//! folds the previous step's instance (Lean `Spec/Folding/Light`,
//! `Spec/Folding/Refresh`), the step circuit's rows hold on its words, and
//! the terminal (`OptionB.Terminal.HoldsFor`) accepts after every step. A
//! small cap reaches every pair of fold kind and relation in a few steps.

#[path = "../option_b_support/mod.rs"]
mod support;

use super::{Envelope, OptionB, OptionBError};
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
