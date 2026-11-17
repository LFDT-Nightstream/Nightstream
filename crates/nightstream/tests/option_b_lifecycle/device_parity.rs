//! A device engine (`Engine::Metal`, `Engine::Cuda`) against
//! `Engine::Optimized` on option B: after each step the proof bytes are
//! equal, and so is everything that the final program reads; the device
//! final proof verifies. Every device session of a process shares one device
//! budget (`BUFFER_LIMIT_BYTES` of its backend), so run these tests one at a
//! time (`--test-threads=1`). The printed times are of one run.

use std::time::Instant;

use super::super::final_program::Finish;
use super::super::{Envelope, OptionB, OptionBError, Setups};
use super::{hash_chain_step, packages, support, words, SetupDir, MINIMUM_SECURITY_BITS};
use crate::engine::{Backend, Engine};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

/// A CPU and a device prover with light fold cap `cap`.
fn provers(engine: Engine, cap: u64) -> [OptionB; 2] {
    [Engine::Optimized, engine].map(|engine| {
        let backend = Backend::new(engine).expect("engine");
        let mut prover = OptionB::new(packages(), MINIMUM_SECURITY_BITS, backend).expect("option-B prover");
        prover.cap = cap;
        prover
    })
}

/// Start a peak measurement of the device bytes, when the backend counts
/// them.
fn reset_peak(engine: Engine) {
    #[cfg(feature = "cuda")]
    if engine == Engine::Cuda {
        neo_prover_cuda::CudaSession::reset_peak_allocated_bytes();
    }
    let _ = engine;
}

/// The most device bytes held since `reset_peak`, when the backend counts
/// them.
fn peak_bytes(engine: Engine) -> Option<usize> {
    #[cfg(feature = "cuda")]
    if engine == Engine::Cuda {
        return Some(neo_prover_cuda::CudaSession::peak_allocated_bytes());
    }
    let _ = engine;
    None
}

/// Cap 2 reaches each fold under each relation in five steps: light folds
/// into slot 0 over the zero accumulator and over a dense one, a refresh of
/// slot 0, and a light fold into slot 1. Both verifiers accept each device
/// proof.
fn steps_equal_cpu_steps(engine: Engine) {
    let [cpu, device] = provers(engine, 2);
    let z0 = words([10, 11, 12, 13]);
    let (mut on_cpu, mut on_device) = (cpu.initial(z0).expect("initial"), device.initial(z0).expect("initial"));
    let mut circuits = Vec::new();
    for step in 0..5 {
        let message = words([20 + step, 21 + step, 22 + step, 23 + step]);
        let next = hash_chain_step(on_cpu.current, message);
        let started = Instant::now();
        on_cpu = cpu.extend(&on_cpu, &message, next).expect("CPU step");
        let cpu_seconds = started.elapsed().as_secs_f64();
        reset_peak(engine);
        let started = Instant::now();
        on_device = device
            .extend(&on_device, &message, next)
            .expect("device step");
        let device_seconds = started.elapsed().as_secs_f64();
        let peak = peak_bytes(engine);
        assert_eq!(
            cpu.encode_proof(&on_cpu).expect("CPU bytes"),
            device.encode_proof(&on_device).expect("device bytes"),
            "step {step}"
        );
        let started = Instant::now();
        device
            .verify(step + 1, z0, next, &on_device)
            .expect("the device verifier accepts");
        let device_verify = started.elapsed().as_secs_f64();
        let started = Instant::now();
        cpu.verify(step + 1, z0, next, &on_device)
            .expect("the CPU verifier accepts");
        eprintln!(
            "step {step} circuit {}: prove CPU {cpu_seconds:.2} s, {engine:?} {device_seconds:.2} s; \
             verify CPU {:.2} s, {engine:?} {device_verify:.2} s; peak device bytes of the step {peak:?}",
            on_device.pc,
            started.elapsed().as_secs_f64()
        );
        circuits.push(on_device.pc);
    }
    assert_eq!(circuits, [1, 1, 1, 2, 1]);

    // A changed fresh witness with its own commitment fails a row on both
    // engines.
    let mut tampered = on_device.clone();
    let fresh = tampered.fresh.as_mut().expect("fresh instance");
    let entry = &mut fresh.witness.Z[(300 % D, 300 / D)];
    *entry = if *entry == F::ZERO { F::ONE } else { F::ZERO };
    fresh.claim.c = device
        .backend
        .commit(&[&fresh.witness.Z])
        .expect("commitment")
        .remove(0);
    for prover in [&cpu, &device] {
        match prover.verify(5, z0, on_device.current, &tampered) {
            Err(OptionBError::Fresh(reason)) => assert_eq!(reason, "a relation row fails"),
            other => panic!("tampered fresh witness: {:?}", other.err()),
        }
    }
}

/// Three device steps, the finish rows and the layer-1 setups of both
/// relations, which take no engine.
fn three_steps(prover: &OptionB) -> (Envelope, Finish, SetupDir, Setups) {
    let finish = Finish::new(
        nightstream_fprime::load_option_b_finish(
            &support::artifact("nightstream-fprime-option-b-finish-v1.json"),
            nightstream_fprime::OPTION_B_FINISH_IDENTITY,
        )
        .expect("finish package"),
    )
    .expect("finish rows");
    let mut envelope = prover.initial(words([10, 11, 12, 13])).expect("initial");
    for step in 0..3 {
        let message = words([20 + step, 21 + step, 22 + step, 23 + step]);
        envelope = prover
            .extend(&envelope, &message, hash_chain_step(envelope.current, message))
            .expect("step");
    }
    let dir = SetupDir(std::env::temp_dir().join(format!(
        "option-b-{:?}-final-{}",
        prover.backend.engine(),
        std::process::id()
    )));
    let started = Instant::now();
    let setups = prover.compression_setups(&dir.0).expect("layer-1 setups");
    eprintln!("layer-1 setups: {:.2} s", started.elapsed().as_secs_f64());
    (envelope, finish, dir, setups)
}

/// What the final program reads of three steps is equal on both engines:
/// the finish package's columns on the last refresh fold's words, and both
/// layer-1 proofs. The shrink proof is not compared: its grinding takes the
/// first witness that a parallel search finds (Plonky3 `find_any`), so its
/// bytes can differ between two runs on one engine.
fn final_witness_equals_cpu_final_witness(engine: Engine) {
    let [cpu, device] = provers(engine, 2);
    let (envelope, finish, _dir, setups) = three_steps(&device);
    let [on_cpu, on_device] = [&cpu, &device].map(|prover| {
        let started = Instant::now();
        let witness = prover
            .final_witness(&envelope, &setups, &finish)
            .expect("final witness");
        eprintln!(
            "final witness on {:?}: {:.2} s",
            prover.backend.engine(),
            started.elapsed().as_secs_f64()
        );
        witness
    });
    assert_eq!(on_cpu.columns, on_device.columns, "finish columns");
    for (cpu, device) in on_cpu.layer1.iter().zip(&on_device.layer1) {
        assert_eq!(cpu.to_bytes(), device.to_bytes(), "layer-1 proofs");
    }
}

/// The device final proof of three steps verifies on both engines, and not
/// for another current state.
fn final_proof_verifies(engine: Engine) {
    let [cpu, device] = provers(engine, 2);
    let (envelope, finish, _dir, setups) = three_steps(&device);
    let statement = (envelope.iteration, envelope.z0, envelope.current);
    let started = Instant::now();
    let witness = device
        .final_witness(&envelope, &setups, &finish)
        .expect("final witness");
    let witnessed = started.elapsed().as_secs_f64();
    let proof = device
        .prove_final(setups.keys(), &finish, statement, &witness)
        .expect("final proof");
    eprintln!(
        "final proof on {engine:?}: witness {witnessed:.2} s, shrink {:.2} s",
        started.elapsed().as_secs_f64() - witnessed
    );
    let (iteration, z0, current) = statement;
    for prover in [&cpu, &device] {
        prover
            .verify_final(setups.keys(), &finish, iteration, z0, current, &proof)
            .expect("the final proof verifies");
    }
    let wrong = [current[0] + F::ONE, current[1], current[2], current[3]];
    assert!(device
        .verify_final(setups.keys(), &finish, iteration, z0, wrong, &proof)
        .is_err());
}

/// One test module per device engine; each runs the same checks.
macro_rules! device_tests {
    ($module:ident, $feature:literal, $engine:expr) => {
        #[cfg(feature = $feature)]
        mod $module {
            use super::*;

            #[test]
            fn steps_equal_cpu_steps() {
                super::steps_equal_cpu_steps($engine);
            }

            #[test]
            #[ignore = "about 250 s with the CPU layer-1 setups; run alone with --ignored --test-threads=1"]
            fn final_witness_equals_cpu_final_witness() {
                super::final_witness_equals_cpu_final_witness($engine);
            }

            #[test]
            #[ignore = "about 200 s with the CPU layer-1 setups; run alone with --ignored --test-threads=1"]
            fn final_proof_verifies() {
                super::final_proof_verifies($engine);
            }
        }
    };
}

device_tests!(metal, "metal", Engine::Metal);
device_tests!(cuda, "cuda", Engine::Cuda);
