//! `Engine::Metal` against `Engine::Optimized` on the bounded-depth scheme: after each step
//! the proof bytes are equal, and so are the final witness and the final
//! proof. Every Metal session of a process shares one
//! device budget (`neo_prover_metal::BUFFER_LIMIT_BYTES`), so run these tests
//! one at a time (`--test-threads=1`). The printed times are of one run.

use std::time::Instant;

use super::super::final_program::Finish;
use super::super::{BoundedDepth, BoundedDepthError, Envelope, Setups};
use super::{compiled, hash_chain_step, packages, support, words, SetupDir, MINIMUM_SECURITY_BITS};
use crate::engine::{Backend, Engine};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

/// A CPU and a Metal prover with light fold cap `cap`.
fn provers(cap: u64) -> [BoundedDepth; 2] {
    [Engine::Optimized, Engine::Metal].map(|engine| {
        let backend = Backend::new(engine).expect("engine");
        let mut prover = BoundedDepth::new(&compiled(), MINIMUM_SECURITY_BITS, backend).expect("bounded-depth prover");
        prover.cap = cap;
        prover
    })
}

/// Cap 2 reaches each fold under each relation in five steps: light folds
/// into slot 0 over the zero accumulator and over a dense one, a refresh of
/// slot 0, and a light fold into slot 1. Both verifiers accept each Metal
/// proof.
#[test]
fn metal_steps_equal_cpu_steps() {
    let [cpu, metal] = provers(2);
    let z0 = words([10, 11, 12, 13]);
    let (mut on_cpu, mut on_metal) = (cpu.initial(z0).expect("initial"), metal.initial(z0).expect("initial"));
    let mut circuits = Vec::new();
    for step in 0..5 {
        let message = words([20 + step, 21 + step, 22 + step, 23 + step]);
        let next = hash_chain_step(on_cpu.current, message);
        let started = Instant::now();
        on_cpu = cpu.extend(&on_cpu, &message, next, None).expect("CPU step");
        let cpu_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        on_metal = metal
            .extend(&on_metal, &message, next, None)
            .expect("Metal step");
        let metal_seconds = started.elapsed().as_secs_f64();
        assert_eq!(
            cpu.encode_proof(&on_cpu).expect("CPU bytes"),
            metal.encode_proof(&on_metal).expect("Metal bytes"),
            "step {step}"
        );
        let started = Instant::now();
        metal
            .verify(step + 1, z0, next, &on_metal)
            .expect("the Metal verifier accepts");
        let metal_verify = started.elapsed().as_secs_f64();
        let started = Instant::now();
        cpu.verify(step + 1, z0, next, &on_metal)
            .expect("the CPU verifier accepts");
        eprintln!(
            "step {step} circuit {}: prove CPU {cpu_seconds:.2} s, Metal {metal_seconds:.2} s; \
             verify CPU {:.2} s, Metal {metal_verify:.2} s",
            on_metal.pc,
            started.elapsed().as_secs_f64()
        );
        circuits.push(on_metal.pc);
    }
    assert_eq!(circuits, [1, 1, 1, 2, 1]);

    // A changed fresh witness with its own commitment fails a row on both
    // engines.
    let mut tampered = on_metal.clone();
    let fresh = tampered.fresh.as_mut().expect("fresh instance");
    let entry = &mut fresh.witness.Z[(300 % D, 300 / D)];
    *entry = if *entry == F::ZERO { F::ONE } else { F::ZERO };
    fresh.claim.c = metal
        .backend
        .commit(&[&fresh.witness.Z])
        .expect("commitment")
        .remove(0);
    for prover in [&cpu, &metal] {
        match prover.verify(5, z0, on_metal.current, &tampered) {
            Err(BoundedDepthError::Fresh(reason)) => assert_eq!(reason, "a relation row fails"),
            other => panic!("tampered fresh witness: {:?}", other.err()),
        }
    }
}

/// Three Metal steps, the finish rows and the layer-1 setups of both
/// relations, which take no engine.
fn three_steps(prover: &BoundedDepth) -> (Envelope, Finish, SetupDir, Setups) {
    let finish = Finish::new(
        nightstream_fprime::load_bounded_depth_finish(
            &support::artifact("nightstream-fprime-bounded-depth-finish-v1.json"),
            nightstream_fprime::BOUNDED_DEPTH_FINISH_IDENTITY,
        )
        .expect("finish package"),
    )
    .expect("finish rows");
    let mut envelope = prover.initial(words([10, 11, 12, 13])).expect("initial");
    for step in 0..3 {
        let message = words([20 + step, 21 + step, 22 + step, 23 + step]);
        envelope = prover
            .extend(&envelope, &message, hash_chain_step(envelope.current, message), None)
            .expect("step");
    }
    let dir = SetupDir(std::env::temp_dir().join(format!("option-b-metal-final-{}", std::process::id())));
    let started = Instant::now();
    let setups = prover.compression_setups(&dir.0).expect("layer-1 setups");
    eprintln!("layer-1 setups: {:.2} s", started.elapsed().as_secs_f64());
    (envelope, finish, dir, setups)
}

/// What the final program reads of three steps is equal on both engines:
/// the finish package's columns on the last refresh fold's words, and both
/// layer-1 proofs.
#[test]
#[ignore = "about 250 s with the CPU layer-1 setups; run alone with --ignored --test-threads=1"]
fn metal_final_witness_equals_cpu_final_witness() {
    let [cpu, metal] = provers(2);
    let (envelope, finish, _dir, setups) = three_steps(&metal);
    let [on_cpu, on_metal] = [&cpu, &metal].map(|prover| {
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
    assert_eq!(on_cpu.columns, on_metal.columns, "finish columns");
    for (cpu, metal) in on_cpu.layer1.iter().zip(&on_metal.layer1) {
        assert_eq!(cpu.to_bytes(), metal.to_bytes(), "layer-1 proofs");
    }
}

/// The Metal final proof of three steps has the bytes of the CPU final proof.
/// It verifies on both engines, and not for another current state.
#[test]
#[ignore = "about 200 s with the CPU layer-1 setups; run alone with --ignored --test-threads=1"]
fn metal_final_proof_equals_cpu_final_proof() {
    let [cpu, metal] = provers(2);
    let (envelope, finish, _dir, setups) = three_steps(&metal);
    let statement = (envelope.iteration, envelope.z0, envelope.current);
    let started = Instant::now();
    let witness = metal
        .final_witness(&envelope, &setups, &finish)
        .expect("final witness");
    let witnessed = started.elapsed().as_secs_f64();
    let proof = metal
        .prove_final(setups.keys(), &finish, statement, &witness)
        .expect("final proof");
    eprintln!(
        "final proof on Metal: witness {witnessed:.2} s, shrink {:.2} s",
        started.elapsed().as_secs_f64() - witnessed
    );
    let on_cpu = cpu
        .prove_final(setups.keys(), &finish, statement, &witness)
        .expect("CPU final proof");
    assert!(
        proof.shrink.to_bytes() == on_cpu.shrink.to_bytes(),
        "the Metal final proof differs from the CPU final proof"
    );
    let (iteration, z0, current) = statement;
    for prover in [&cpu, &metal] {
        prover
            .verify_final(setups.keys(), &finish, iteration, z0, current, &proof)
            .expect("the final proof verifies");
    }
    let wrong = [current[0] + F::ONE, current[1], current[2], current[3]];
    assert!(metal
        .verify_final(setups.keys(), &finish, iteration, z0, wrong, &proof)
        .is_err());
}
