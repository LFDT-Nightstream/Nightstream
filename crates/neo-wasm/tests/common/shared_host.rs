use std::num::NonZeroUsize;

use neo_wasm::memory_semantics::preload_host_event_tables;
use neo_wasm::nebula::{plan_ordered_slots, step_inputs, WasmApplication, WasmLimits};
use neo_wasm::{WasmProgramArtifacts, WasmVmStep};
use nightstream::nebula::{Circuit, Geometry};

pub fn check(trace: &[WasmVmStep]) {
    for rows in [7, 8, 9] {
        let rows = NonZeroUsize::new(rows).unwrap();
        let mut geometry = neo_wasm::nebula::WasmSize::Small.geometry();
        geometry.b_ops = plan_ordered_slots(rows).unwrap().len();
        let app = WasmApplication::new(
            &geometry,
            neo_wasm::nebula::WasmSize::Small
                .limits()
                .regions()
                .unwrap()
                .0,
            rows,
        )
        .unwrap();
        for step in step_inputs(trace, trace.len().div_ceil(rows.get()) + 1, rows).unwrap() {
            app.check(&step.z).unwrap();
        }
    }
}

pub fn prove(trace: &[WasmVmStep], artifacts: &WasmProgramArtifacts) {
    let rows = NonZeroUsize::new(9).unwrap();
    let n = trace.len().div_ceil(rows.get()).next_power_of_two();
    let inputs = step_inputs(trace, n, rows).unwrap();
    let witnesses = step_inputs(trace, n * rows.get(), NonZeroUsize::MIN)
        .unwrap()
        .into_iter()
        .map(|step| step.z)
        .collect::<Vec<_>>();
    let mut preload = neo_wasm::preload_from_program_artifacts(artifacts);
    preload_host_event_tables(&mut preload, &artifacts.host_event_bindings, artifacts);
    let (regions, used) = WasmLimits::of(&[&preload], &witnesses)
        .unwrap()
        .regions()
        .unwrap();
    let slots = plan_ordered_slots(rows).unwrap();
    let bits = |value: u64| (u64::BITS - value.leading_zeros()) as usize;
    let r = bits(used[0] - 1).max(n.trailing_zeros() as usize);
    let mu = bits(used[1] - 1).max(r);
    let geometry = Geometry::new(r, mu, slots.len(), n, 1).unwrap();
    let app = WasmApplication::with_slots(&geometry, regions, rows, &slots).unwrap();
    let program = app.program(&preload).unwrap();
    let circuit = Circuit::compile(geometry, app).unwrap();
    let prover = circuit.prover(nightstream::Engine::Optimized, 109).unwrap();
    let verifier = circuit
        .verifier(nightstream::Engine::Optimized, 109)
        .unwrap();
    let initial = trace[0].state_before;
    let last = trace.last().unwrap().state_after;
    let mut run = prover.start(&program, initial).unwrap();
    run.segment(&inputs).unwrap();
    let proof = run.finish().unwrap();
    verifier.verify(&program, &initial, &last, &proof).unwrap();
    for changed in [
        {
            let mut changed = last;
            changed.comm_chain[0] ^= 1;
            changed
        },
        {
            let mut changed = last;
            changed.host_events.host_frame_depth += 1;
            changed
        },
        {
            let mut changed = last;
            changed.trapped = !changed.trapped;
            changed
        },
    ] {
        assert!(verifier
            .verify(&program, &initial, &changed, &proof)
            .is_err());
    }
    eprintln!("shared_host_public_proof	109	{}	{geometry:?}", trace.len());
}
