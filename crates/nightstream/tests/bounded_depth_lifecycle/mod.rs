//! Lean `Spec/Folding/Light`, `Spec/Folding/Refresh`, and
//! `BoundedDepth.Terminal.HoldsFor`.

#[path = "../bounded_depth_support/mod.rs"]
mod support;

mod checked_rows;
mod compression;
mod compression_error;
mod encodings;
mod factored_rows;
mod fold_gate;
mod guard_fusion;
#[cfg(feature = "metal")]
mod metal_parity;
mod packed_boolean;
mod packed_product;

use super::{BoundedDepth, BoundedDepthError, CompiledSteps, Envelope, LIGHT_FOLD_CAP};
use crate::application::{poseidon2_hash_chain_v1, Affine, ApplicationBuilder, ApplicationCircuit};
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F};
use neo_params::nightstream_p64_k14 as profile;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use support::{fold_gate_bits, hash_chain_step, packages, words};

fn compiled() -> CompiledSteps {
    CompiledSteps::new(packages()).expect("compiled steps")
}

fn prover(cap: u64) -> BoundedDepth {
    let mut prover = BoundedDepth::new(&compiled(), fold_gate_bits(), crate::engine::Backend::Optimized)
        .expect("bounded-depth prover");
    prover.cap = cap;
    prover
}

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
        envelope = prover
            .extend(&envelope, &message, next, None)
            .expect("step");
        circuits.push(envelope.pc);
        prover
            .verify(step + 1, z0, next, &envelope)
            .unwrap_or_else(|error| panic!("step {step} verifies: {error}"));
    }
    (envelope, circuits)
}

#[test]
fn packed_refresh_matches_a_full_step_and_raw_verification() {
    let eager = prover(2);
    let packed = CompiledSteps::with_refresh_storage(packages(), true).unwrap();
    let mut packed = BoundedDepth::new(&packed, fold_gate_bits(), crate::engine::Backend::Optimized).unwrap();
    packed.cap = 2;

    let (expected, _) = run(&eager, 4);
    let (proof, circuits) = run(&packed, 4);
    assert_eq!(circuits, [1, 1, 1, 2]);
    let bytes = packed.encode_proof(&proof).unwrap();
    assert_eq!(bytes, eager.encode_proof(&expected).unwrap());
    let read = packed.decode_proof(&bytes).unwrap();
    packed.verify(4, read.z0, read.current, &read).unwrap();
}

#[test]
fn a_raw_light_proof_verifies_with_the_refresh_rows_still_packed() {
    use crate::engine::{Backend, Engine};

    let prover = prover(2);
    let (proof, circuits) = run(&prover, 2);
    assert_eq!(circuits, [1, 1]);
    let bytes = prover.encode_proof(&proof).unwrap();

    for engine in [
        Engine::Optimized,
        #[cfg(feature = "metal")]
        Engine::Metal,
    ] {
        let steps = CompiledSteps::with_refresh_storage(packages(), true).unwrap();
        let verifier = BoundedDepth::new(&steps, fold_gate_bits(), Backend::new(engine).unwrap()).unwrap();
        let refresh = &steps.steps()[1];
        let decoded = verifier.decode_proof(&bytes).unwrap();
        assert!(
            refresh.remains_packed(),
            "{engine:?} restored the refresh rows before verification"
        );
        verifier
            .verify(2, proof.z0, proof.current, &decoded)
            .unwrap();
        assert!(
            refresh.remains_packed(),
            "{engine:?} restored the refresh rows during verification"
        );
    }
}

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
    rejects(
        &recommitted(nightstream_fprime::PUBLIC_WORDS + 30),
        "a relation row fails",
    );
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

/// Lean `accumulatorBound`.
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
        let bound = (1 + depth + u64::from(profile::K_RHO)) * u64::from(profile::T) * u64::from(profile::B_BASE - 1);
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

#[test]
fn refresh_folds_under_both_relations_verify() {
    let (_, circuits) = run(&prover(0), 3);
    assert_eq!(circuits, [1, 2, 2]);
}

fn addition() -> ApplicationCircuit {
    let builder = ApplicationBuilder::new(2).expect("builder");
    let input = builder.input_state();
    let private = builder.private_inputs().to_vec();
    let outputs = std::array::from_fn(|lane| Affine::from(input[lane]) + Affine::from(private[lane % 2]));
    builder.finish(outputs).expect("addition application")
}

#[test]
fn deferred_native_refresh_witness_matches_the_eager_proof() {
    let application = addition();
    let compile = |defer| {
        let packages = packages()
            .with_application_and_refresh_witness(
                application.records().clone(),
                application.private_input_count(),
                application.shape(),
                defer,
            )
            .unwrap();
        CompiledSteps::with_refresh_storage(packages, defer).unwrap()
    };
    let eager = compile(false);
    let deferred = compile(true);
    assert_eq!(deferred.identity(), eager.identity());
    let prove = |steps: &CompiledSteps| {
        let mut prover = BoundedDepth::new(steps, fold_gate_bits(), crate::engine::Backend::Optimized).unwrap();
        prover.cap = 1;
        let mut envelope = prover.initial(words([10, 11, 12, 13])).unwrap();
        let mut circuits = Vec::new();
        for step in 0..3 {
            let message = [F::from_u64(20 + step), F::from_u64(21 + step)];
            let next = application
                .execute(envelope.current, &message)
                .unwrap()
                .output_state();
            envelope = prover.extend(&envelope, &message, next, None).unwrap();
            circuits.push(envelope.pc);
        }
        assert_eq!(circuits, [1, 1, 2]);
        let bytes = prover.encode_proof(&envelope).unwrap();
        let decoded = prover.decode_proof(&bytes).unwrap();
        prover
            .verify(3, decoded.z0, decoded.current, &decoded)
            .unwrap();
        bytes
    };
    assert_eq!(prove(&deferred), prove(&eager));
}

fn native_application_verifies(application: ApplicationCircuit) {
    let packages = packages()
        .with_application(
            application.records().clone(),
            application.private_input_count(),
            application.shape(),
        )
        .expect("packages with the application");
    let mut prover = BoundedDepth::new(
        &CompiledSteps::new(packages).unwrap(),
        fold_gate_bits(),
        crate::engine::Backend::Optimized,
    )
    .expect("bounded-depth prover");
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
                prover.extend(&envelope, message, wrong, None),
                Err(BoundedDepthError::Fresh("a relation row fails"))
            ),
            "step {step} with a wrong next state"
        );
        envelope = prover.extend(&envelope, message, next, None).expect("step");
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

#[test]
fn native_hash_chain_packages_are_lean_packages() {
    let lean = packages();
    let application = poseidon2_hash_chain_v1().expect("hash chain");
    assert_eq!(application.shape().blocks.len(), 5);
    let native = lean
        .with_application(application.records().clone(), 4, application.shape())
        .expect("packages with the application");
    let (lean, native) = (lean.expand().expect("expanded"), native.expand().expect("expanded"));
    for (lean, native) in [(&lean.light, &native.light), (&lean.refresh, &native.refresh)] {
        let (lean_rows, native_rows) = (lean.rows(), native.rows());
        let (lean_layout, native_layout) = (lean.layout(), native.layout());
        assert_eq!(native.block_count(), lean.block_count());
        assert_eq!(native_rows.row_count(), lean_rows.row_count());
        assert_eq!(plan_sum(native_rows), plan_sum(lean_rows));
        assert_eq!(native_layout.width(), lean_layout.width());
        let end = lean.sbox_start() + 150 * lean.block_count();
        assert!((0..end).all(|column| native_layout.slot(column) == lean_layout.slot(column)));
    }
}

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
    assert!(crate::application::check_blocks_match_permutation_rows(&honest).is_ok());
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
        envelope = prover
            .extend(&envelope, &message, next, None)
            .expect("step");
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
