mod common;

use std::num::NonZeroUsize;

use neo_wasm::memory_semantics::preload_host_event_tables;
use neo_wasm::nebula::{plan_ordered_slots, step_inputs, WasmApplication, WasmLimits};
use nightstream::nebula::{Circuit, Geometry};

#[test]
fn guarded_native_public_proof_binds_import_and_export_events() {
    let setup = common::host_event_fixture::host_event_lifecycle_setup();
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&setup.component_bytes).unwrap();
    let per_step = NonZeroUsize::new(4).unwrap();
    let n = setup
        .trace
        .len()
        .div_ceil(per_step.get())
        .next_power_of_two();
    let inputs = step_inputs(&setup.trace, n, per_step).unwrap();
    let mut preload = neo_wasm::preload_from_program_artifacts(&artifacts);
    preload_host_event_tables(&mut preload, &setup.bindings, &artifacts);
    let witnesses = step_inputs(&setup.trace, n * per_step.get(), NonZeroUsize::MIN)
        .unwrap()
        .into_iter()
        .map(|step| step.z)
        .collect::<Vec<_>>();
    let (regions, used) = WasmLimits::of(&[&preload], &witnesses)
        .unwrap()
        .regions()
        .unwrap();
    let slots = plan_ordered_slots(per_step).unwrap();
    let bits = |value: u64| (u64::BITS - value.leading_zeros()) as usize;
    let r = bits(used[0] - 1).max(n.trailing_zeros() as usize);
    let mu = bits(used[1] - 1).max(r);
    let geometry = Geometry::new(r, mu, slots.len(), n, 1).unwrap();
    let application = WasmApplication::with_slots(&geometry, regions, per_step, &slots).unwrap();
    let program = application.program(&preload).unwrap();
    let circuit = Circuit::compile(geometry, application).unwrap();
    let prover = circuit.prover(nightstream::Engine::Optimized, 109).unwrap();
    let verifier = circuit
        .verifier(nightstream::Engine::Optimized, 109)
        .unwrap();
    let initial = neo_wasm::host_event_top_level_initial_state(
        &artifacts.tables,
        setup.trace[0].state_before.pc,
        &setup.bindings,
        setup.run_fref,
        Default::default(),
    )
    .unwrap();
    assert_eq!(initial, setup.trace[0].state_before);
    let mut run = prover.start(&program, initial).unwrap();
    run.segment(&inputs).unwrap();
    let proof = run.finish().unwrap();
    let last = setup.trace.last().unwrap().state_after;
    verifier.verify(&program, &initial, &last, &proof).unwrap();
    let bytes = prover.encode_proof(&proof).unwrap();
    let decoded = verifier.decode_proof(&bytes).unwrap();
    verifier
        .verify(&program, &initial, &last, &decoded)
        .unwrap();
    let mut changed = last;
    changed.comm_chain[0] ^= 1;
    assert!(verifier
        .verify(&program, &initial, &changed, &decoded)
        .is_err());
    eprintln!(
        "guarded_host_public_proof\t1\tminimum_security_bits\t109\ttrace_rows\t{}\t{geometry:?}",
        setup.trace.len()
    );
}
