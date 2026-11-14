//! A compiled circuit expands its step packages and builds its finish
//! circuit once. Its provers and verifiers read that one copy, so making
//! them expands nothing again.

use std::sync::Arc;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::Circuit;
use crate::application::{Affine, ApplicationBuilder};
use crate::Engine;

fn circuit() -> Circuit {
    let mut builder = ApplicationBuilder::new(2).unwrap();
    let [left, right] = [0, 1].map(|index| Affine::from(builder.private_inputs()[index]));
    let product = builder.multiply(left, right).unwrap();
    let mut state = builder.input_state().map(Affine::from);
    state[0] = state[0].clone() + Affine::from(product) * F::from_u64(3);
    Circuit::compile(builder.finish(state).unwrap()).unwrap()
}

#[test]
fn provers_and_verifiers_read_the_circuit_expansion() {
    let circuit = circuit();
    let provers = [0, 1].map(|_| circuit.prover(Engine::Optimized, 109).unwrap());
    let verifier = circuit.verifier(Engine::Optimized, 109).unwrap();
    let lifecycles = [&provers[0].lifecycle, &provers[1].lifecycle, &verifier.lifecycle];
    for (index, step) in circuit.steps.steps().iter().enumerate() {
        let expanded = step.expanded();
        for lifecycle in lifecycles {
            let shared = lifecycle.relations()[index].step();
            assert!(Arc::ptr_eq(shared, step), "circuit {index}: another compiled step");
            let rows = shared.expanded().rows();
            assert_eq!(rows.columns().as_ptr(), expanded.rows().columns().as_ptr());
            assert!(std::ptr::eq(shared.expanded().layout(), expanded.layout()));
        }
    }
    for finish in [&provers[0].finish, &provers[1].finish, &verifier.finish] {
        assert!(Arc::ptr_eq(finish, &circuit.finish), "another finish circuit");
    }
}
