//! `Engine::Metal` against `Engine::Optimized` on the bounded-depth scheme: after each step
//! the proof bytes are equal. Every Metal session of a process shares one
//! device budget (`neo_prover_metal::BUFFER_LIMIT_BYTES`), so run these tests
//! one at a time (`--test-threads=1`). The printed times are of one run.

use std::time::Instant;

use super::super::{BoundedDepth, BoundedDepthError};
use super::{hash_chain_step, packages, words, MINIMUM_SECURITY_BITS};
use crate::engine::{Backend, Engine};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

/// A CPU and a Metal prover with light fold cap `cap`.
fn provers(cap: u64) -> [BoundedDepth; 2] {
    [Engine::Optimized, Engine::Metal].map(|engine| {
        let backend = Backend::new(engine).expect("engine");
        let mut prover = BoundedDepth::new(packages(), MINIMUM_SECURITY_BITS, backend).expect("bounded-depth prover");
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
    let column = nightstream_fprime::PUBLIC_WORDS + 30;
    let entry = &mut fresh.witness.Z[(column % D, column / D)];
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
