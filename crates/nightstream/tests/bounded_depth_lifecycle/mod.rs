//! The bounded-depth lifecycle end to end on the Poseidon2 hash chain: each step
//! folds the previous step's instance (Lean `Spec/Folding/Light`,
//! `Spec/Folding/Refresh`), the step circuit's rows hold on its words, and
//! the terminal (`BoundedDepth.Terminal.HoldsFor`) accepts after every step. A
//! small cap reaches every pair of fold kind and relation in a few steps.

#[path = "../bounded_depth_support/mod.rs"]
mod support;

mod factored_rows;
mod final_ir;
mod guard_fusion;
mod packed_boolean;
mod packed_product;
mod setup_runs;

use super::{BoundedDepth, BoundedDepthError, Envelope, LIGHT_FOLD_CAP};
use crate::application::{poseidon2_hash_chain_v1, Affine, ApplicationBuilder, ApplicationCircuit};
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use support::{hash_chain_step, packages, words};

const MINIMUM_SECURITY_BITS: u32 = 109;

fn prover(cap: u64) -> BoundedDepth {
    let mut prover = BoundedDepth::new(packages(), MINIMUM_SECURITY_BITS).expect("bounded-depth prover");
    prover.cap = cap;
    prover
}

/// Runs `steps` hash-chain steps from `z0`, verifies after each, and returns
/// the last proof and the fold kinds' program counters.
fn run(prover: &BoundedDepth, steps: u64) -> (Envelope, Vec<u64>) {
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
        Err(BoundedDepthError::Statement(_))
    ));

    let mut deeper = envelope.clone();
    deeper.slots[1].depth += 1;
    rejects(&deeper, "public input differs from the hash of the state");

    let mut tampered = envelope.clone();
    tampered.slots[1].accumulator.witness[(0, 0)] += F::ONE;
    rejects(&tampered, "bounded-depth slot 1");

    let mut narrow = envelope.clone();
    narrow.slots[0].accumulator.claim.X = Mat::zero(D, 1, F::ZERO);
    rejects(&narrow, "bounded-depth slot 0: claim shape");

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

/// The slot check holds a witness to `(17 + depth) · 216`, the Lean
/// `accumulatorBound` of the production parameters, on both signs, and
/// refuses a depth above the cap. No accepted proof can isolate this check in
/// `verify`: a slot above its bound needs a second opening of its commitment.
#[test]
fn a_slot_is_held_to_the_bound_of_its_depth() {
    let prover = prover(LIGHT_FOLD_CAP);
    let mut slot = prover
        .initial(words([10, 11, 12, 13]))
        .expect("initial proof")
        .slots[0]
        .clone();
    assert_eq!(
        prover.slot_within_depth_bound(&slot),
        Ok(()),
        "the compact zero witness"
    );
    let q = F::ORDER_U64;
    for depth in [0, 1, LIGHT_FOLD_CAP] {
        let bound = (17 + depth) * 216;
        slot.depth = depth;
        for (coefficient, accepted) in [
            (bound, true),
            (bound + 1, false),
            (q - bound, true),
            (q - bound - 1, false),
        ] {
            slot.accumulator.witness.as_mut_slice()[0] = F::from_u64(coefficient);
            assert_eq!(
                prover.slot_within_depth_bound(&slot).is_ok(),
                accepted,
                "depth {depth}, coefficient {coefficient}"
            );
        }
    }
    slot.accumulator.witness.as_mut_slice()[0] = F::ZERO;
    slot.depth = LIGHT_FOLD_CAP + 1;
    assert_eq!(
        prover.slot_within_depth_bound(&slot),
        Err("depth is above the light fold cap")
    );
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
        .with_application(
            application.records().clone(),
            application.private_input_count(),
            application.shape(),
        )
        .expect("packages with the application");
    let mut prover = BoundedDepth::new(packages, MINIMUM_SECURITY_BITS).expect("bounded-depth prover");
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
                Err(BoundedDepthError::Fresh("a relation row fails"))
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

/// Σ coefficient · (point³ + unit) over every entry of every plan row, with
/// unit `4 · row + port` and point `16,777,259 · unit + column + 1`.
fn plan_sum(rows: &nightstream_fprime::PlanRows) -> F {
    let mut sum = F::ZERO;
    for row in 0..rows.row_count() {
        for port in 0..4 {
            let unit = (4 * row + port) as u64;
            for (column, coefficient) in rows.port(row, port) {
                let point = F::from_u64(unit * 16_777_259 + column as u64 + 1);
                sum += coefficient * (point * point * point + F::from_u64(unit));
            }
        }
    }
    sum
}

/// The Rust hash chain in place of Lean's gives Lean's step packages: the same
/// blocks, the same plan rows entry for entry, and the same slot for every
/// column. So each permutation is a block with no slot for its dead locals.
#[test]
fn native_hash_chain_packages_are_lean_packages() {
    let lean = packages();
    let application = poseidon2_hash_chain_v1().expect("hash chain");
    assert_eq!(application.shape().blocks.len(), 5);
    let native = lean
        .with_application(application.records().clone(), 4, application.shape())
        .expect("packages with the application");
    for (lean, native) in [(lean.light(), native.light()), (lean.refresh(), native.refresh())] {
        assert_eq!(native.block_count(), lean.block_count());
        let (lean_rows, native_rows) = (lean.plan_rows().expect("rows"), native.plan_rows().expect("rows"));
        assert_eq!(native_rows.row_count(), lean_rows.row_count());
        assert_eq!(plan_sum(&native_rows), plan_sum(&lean_rows));
        let (lean_layout, native_layout) = (lean.layout().expect("layout"), native.layout().expect("layout"));
        assert_eq!(native_layout.width(), lean_layout.width());
        let end = lean.sbox_start() + 150 * lean.block_count();
        assert!((0..end).all(|column| native_layout.slot(column) == lean_layout.slot(column)));
    }
}

/// `Circuit::compile` refuses a block whose rows are not the permutation of
/// its lanes: a lane with another constant, and a block one local late.
#[test]
fn a_block_that_is_not_its_permutation_is_refused() {
    use crate::application::ApplicationError;
    let honest = poseidon2_hash_chain_v1().expect("hash chain");
    let forged = |change: &dyn Fn(&mut nightstream_fprime::NativeBlock)| {
        let mut shape = honest.shape().clone();
        change(&mut shape.blocks[1]);
        ApplicationCircuit::from_records(
            honest.records().clone(),
            honest.private_input_count(),
            honest.prepared_output_forms().expect("forms"),
            honest.prepared_output_rows(),
            shape,
        )
        .expect("application")
    };
    assert!(crate::application::check_blocks(&honest).is_ok());
    for application in [
        forged(&|block| block.lanes[3].constant += F::ONE),
        forged(&|block| block.first_local += 1),
    ] {
        assert!(matches!(
            crate::Circuit::compile(application),
            Err(crate::Error::Application(ApplicationError::Block(1)))
        ));
    }
}

/// The CPU time of each bounded-depth step. Steps 1 to 3 are light; a refresh step
/// runs at the cap, so this test reaches one with cap 2.
#[test]
#[ignore = "Timing run; run this test separately under the 300-second cap."]
fn bounded_depth_step_times() {
    let started = std::time::Instant::now();
    let prover = prover(2);
    eprintln!("bounded-depth preparation elapsed={:?}", started.elapsed());
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
            "bounded-depth step={step} circuit={} prove={proved:?} verify={:?}",
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
    let mut prover = BoundedDepth::new(packages(), 105).expect("bounded-depth prover");
    prover.cap = 2;
    let finish = Finish::new(
        nightstream_fprime::load_bounded_depth_finish(
            &support::artifact("nightstream-fprime-bounded-depth-finish-v1.json"),
            nightstream_fprime::BOUNDED_DEPTH_FINISH_IDENTITY,
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
    let dir = SetupDir(std::env::temp_dir().join(format!("bounded-depth-final-{}", std::process::id())));
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
        super::BoundedDepth::layer1_relation(&keys, 0).expect("light relation"),
        super::BoundedDepth::layer1_relation(&keys, 1).expect("refresh relation"),
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
    // `witness.columns` holds one word for each live column, in column order.
    let live = finish.live();
    let changed = |column: usize| {
        assert!(live[column], "the final program reads column {column}");
        let cell = live[..column].iter().filter(|&&live| live).count();
        let mut columns = witness.columns.clone();
        columns[cell] = columns[cell].wrapping_add(1) % 18_446_744_069_414_584_321;
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
