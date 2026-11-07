mod common;

use std::num::NonZeroUsize;

use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::memory_semantics::preload_host_event_tables;
use neo_wasm::nebula::{plan_ordered_slots, step_inputs, WasmApplication, WasmLimits};
use neo_wasm::{preload_from_program_artifacts, top_level_initial_state, WasmOpcode};
use nightstream::nebula::{Circuit, Geometry, ProveError};

fn bits(value: u64) -> usize {
    (u64::BITS - value.leading_zeros()) as usize
}

#[test]
fn terminal_end_cannot_claim_an_unrelated_pc() {
    let reference = common::checked_main(r#"(module (func (export "main") (result i32) i32.const 7))"#);
    assert_eq!(reference.run.results, ["7"]);
    let mut forged = reference.trace.clone();
    let last = forged.last_mut().unwrap();
    assert_eq!(last.opcode, WasmOpcode::End);
    assert!(last.state_after.activation_finished);
    last.state_after.pc += 41;

    let per_step = NonZeroUsize::MIN;
    let n = forged.len().next_power_of_two();
    let inputs = step_inputs(&forged, n, per_step).unwrap();
    let witnesses = inputs.iter().map(|step| step.z.clone()).collect::<Vec<_>>();
    let mut preload = preload_from_program_artifacts(&reference.artifacts);
    let export_fref = forged[0].state_before.host_events.turn_export_fref;
    preload_host_event_tables(&mut preload, &HostEventBindings::import_free(export_fref));
    let (regions, used) = WasmLimits::of(&[&preload], &witnesses).regions();
    let slots = plan_ordered_slots(per_step);
    let low = n.trailing_zeros() as usize;
    let r = bits(used[0] - 1).max(low);
    let mu = bits(used[1] - 1).max(r);
    let geometry = Geometry::new(r, mu, slots.len(), n, 1).unwrap();
    let app = WasmApplication::with_slots(&geometry, regions, per_step, &slots).unwrap();
    let circuit = Circuit::compile(geometry, app.clone()).unwrap();
    let engine = nightstream::Engine::Optimized;
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    let program = app.program(&preload).unwrap();
    let initial = top_level_initial_state(&reference.artifacts.tables, forged[0].state_before.pc);
    let honest_inputs = step_inputs(&reference.trace, n, per_step).unwrap();
    let mut honest = prover.start(&program, initial).unwrap();
    honest.segment(&honest_inputs).unwrap();
    let honest_proof = honest.finish().unwrap();
    verifier
        .verify(
            &program,
            &initial,
            &reference.trace.last().unwrap().state_after,
            &honest_proof,
        )
        .unwrap();

    let mut run = prover.start(&program, initial).unwrap();
    assert!(matches!(run.segment(&inputs), Err(ProveError::Proof(_))));
}
