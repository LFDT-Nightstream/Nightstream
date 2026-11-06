//! The option-B lifecycle end to end on the Poseidon2 hash chain: each step
//! folds the previous step's instance (Lean `Spec/Folding/Light`,
//! `Spec/Folding/Refresh`), the step circuit's rows hold on its words, and
//! the terminal (`OptionB.Terminal.HoldsFor`) accepts after every step. A
//! small cap reaches every pair of fold kind and relation in a few steps.

#[path = "../option_b_support/mod.rs"]
mod support;

mod final_ir;
mod setup_runs;

use super::{Envelope, OptionB, OptionBError};
use crate::application::{poseidon2_hash_chain_v1, Affine, ApplicationBuilder, ApplicationCircuit};
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use support::{hash_chain_step, packages, words};

const MINIMUM_SECURITY_BITS: u32 = 109;

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
/// light step and a refresh step, each verified; `extend` refuses a step that
/// claims a wrong next state, because the application's rows fail.
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
        assert!(
            matches!(
                prover.extend(&envelope, message, wrong),
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

/// The CPU time of each option-B step. Steps 1 to 3 are light; a refresh step
/// runs at the cap, so this test reaches one with cap 2.
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

/// Removes a setup folder when it drops, so a failed run frees its disk too.
struct SetupDir(std::path::PathBuf);

impl Drop for SetupDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// CW-6.4: the final program (the finish package's rows and both layer-1
/// verifiers) accepts the last fold of three steps, and the shrink proof of
/// it verifies. A run with a changed row column, current word or key word,
/// or with slot 1 carrying slot 0's layer-1 proof, fails its own check; a
/// final proof for another current state rejects.
#[test]
fn final_proof_verifies() {
    use super::final_program::Finish;
    use neo_spartan::shrink::Program;
    let started = std::time::Instant::now();
    // Keys for a minimum below the production one have fewer layer bits, so
    // a production verifier must refuse them (the end of this test).
    let mut prover = OptionB::new(packages(), 105).expect("option-B prover");
    prover.cap = 2;
    let finish = Finish::new(
        nightstream_fprime::load_option_b_finish(
            &support::artifact("nightstream-fprime-option-b-finish-v1.json"),
            nightstream_fprime::OPTION_B_FINISH_IDENTITY,
        )
        .expect("finish package"),
    )
    .expect("finish rows");
    let z0 = words([10, 11, 12, 13]);
    let mut envelope = prover.initial(z0).expect("initial proof");
    for step in 0..3 {
        let message = words([20 + step, 21 + step, 22 + step, 23 + step]);
        let next = hash_chain_step(envelope.current, message);
        envelope = prover.extend(&envelope, &message, next).expect("step");
    }
    let dir = SetupDir(std::env::temp_dir().join(format!("option-b-final-{}", std::process::id())));
    let (witness, keys) = {
        let setups = prover.compression_setups(&dir.0).expect("layer-1 setups");
        eprintln!("compression setups elapsed={:?}", started.elapsed());
        let witnessing = std::time::Instant::now();
        let witness = prover
            .final_witness(&envelope, &setups, &finish)
            .expect("final witness");
        eprintln!("final witness elapsed={:?}", witnessing.elapsed());
        (witness, setups.keys().clone())
    };
    drop(dir);
    assert_eq!(
        (
            nightstream_fprime::PUBLIC_WORDS / D,
            nightstream_fprime::PI_CCS_V1_2_ROUND_COUNT
        ),
        (5, 28),
        "the layer-1 relations take Production.lean's public blocks and point variables"
    );
    assert!(
        keys.bound < 1 << 16,
        "each slot's honest bound lies below Production.normLimit"
    );
    let (iteration, current) = (envelope.iteration, envelope.current);
    let statement = (iteration, z0, current);
    let relations = [
        super::OptionB::layer1_relation(&keys, 0).expect("light relation"),
        super::OptionB::layer1_relation(&keys, 1).expect("refresh relation"),
    ];
    let run = |witness: &super::final_program::FinalWitness| {
        prover
            .final_program(&finish, [&relations[0], &relations[1]], statement, Some(witness))
            .run(&mut neo_spartan::Native)
    };
    run(&witness).expect("the final program accepts the honest witness");
    let rejects = |witness: super::final_program::FinalWitness, expected: &str| match run(&witness) {
        Err(error) => assert!(error.to_string().contains(expected), "{expected}: {error}"),
        Ok(()) => panic!("{expected}: accepted"),
    };
    let changed = |column: usize| {
        let mut columns = witness.columns.clone();
        columns[column] = columns[column].wrapping_add(1) % 18_446_744_069_414_584_321;
        super::final_program::FinalWitness {
            columns,
            layer1: witness.layer1.clone(),
        }
    };
    rejects(changed(finish.package().first_fresh()), "finish package");
    rejects(changed(287), "current word");
    rejects(changed(275), "key word");
    rejects(
        super::final_program::FinalWitness {
            columns: witness.columns.clone(),
            layer1: [witness.layer1[0].clone(), witness.layer1[0].clone()],
        },
        "layer-1",
    );

    let proving = std::time::Instant::now();
    let proof = prover
        .prove_final(&keys, &finish, statement, &witness)
        .expect("final proof");
    eprintln!("final proof elapsed={:?}", proving.elapsed());
    eprintln!("final proof bytes: {}", proof.shrink.to_bytes().len());
    let verifying = std::time::Instant::now();
    prover
        .verify_final(&keys, &finish, iteration, z0, current, &proof)
        .expect("the final proof verifies");
    eprintln!("final verification elapsed={:?}", verifying.elapsed());
    let wrong_current = [current[0] + F::ONE, current[1], current[2], current[3]];
    let mut moved = proof.clone();
    moved.current = wrong_current;
    assert!(prover
        .verify_final(&keys, &finish, iteration, z0, wrong_current, &moved)
        .is_err());
    assert!(
        self::prover(2)
            .verify_final(&keys, &finish, iteration, z0, current, &proof)
            .is_err(),
        "a verifier at the production minimum refuses keys made for 105 bits"
    );
}
