mod common;

use std::num::NonZeroUsize;

use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::memory_semantics::preload_host_event_tables;
use neo_wasm::nebula::{plan_ordered_slots, step_inputs, WasmApplication, WasmLimits};
use neo_wasm::{preload_from_program_artifacts, top_level_initial_state};
use nightstream::nebula::{Circuit, Geometry};

fn bit_width(value: u64) -> usize {
    (u64::BITS - value.leading_zeros()) as usize
}

#[test]
fn dynamic_shared_alu_public_proofs_bind_programs_and_accept_traps() {
    let sources = [
        r#"(module (func (export "main") (result i32)
            i32.const 8 i32.clz drop i64.const 8 i64.ctz drop
            i64.const -17 i64.const 5 i64.div_s i32.wrap_i64
            i32.const 5 i32.add))"#,
        r#"(module (func (export "main") (result i32)
            i64.const -17 i64.const 5 i64.rem_s i32.wrap_i64
            i32.const 4 i32.add))"#,
        r#"(module (func (export "main") (result i32)
            i64.const -9223372036854775808 i64.const -1 i64.div_s
            i32.wrap_i64))"#,
        r#"(module
            (memory 1)
            (func $mix (param i64) (result i64)
                local.get 0 i64.const 5 i64.div_s)
            (func (export "main") (result i32)
                i32.const 1 i64.const -17 call $mix
                i64.const 7 i64.xor i64.store align=1
                i32.const 1 i64.load align=1 i32.wrap_i64
                i32.const 0 i32.lt_s
                if (result i32) i32.const 2 else i32.const 99 end))"#,
    ];
    let cases = sources.map(common::checked_main);
    assert_eq!(cases[0].run.results, ["2"]);
    assert_eq!(cases[1].run.results, ["2"]);
    assert!(cases[2].trace.last().unwrap().state_after.trapped);
    assert_eq!(cases[3].run.results, ["2"]);
    let per_step = NonZeroUsize::new(4).unwrap();
    let n = cases
        .iter()
        .map(|case| case.trace.len())
        .max()
        .unwrap()
        .div_ceil(4)
        .next_power_of_two();
    let inputs = cases
        .iter()
        .map(|case| step_inputs(&case.trace, n, per_step).unwrap())
        .collect::<Vec<_>>();
    let preloads = cases
        .iter()
        .map(|case| {
            let mut preload = preload_from_program_artifacts(&case.artifacts);
            let fref = case.trace[0].state_before.host_events.turn_export_fref;
            preload_host_event_tables(&mut preload, &HostEventBindings::import_free(fref), &case.artifacts);
            preload
        })
        .collect::<Vec<_>>();
    let witnesses = cases
        .iter()
        .flat_map(|case| {
            step_inputs(&case.trace, n * per_step.get(), NonZeroUsize::MIN)
                .unwrap()
                .into_iter()
                .map(|step| step.z)
        })
        .collect::<Vec<_>>();
    let (regions, used) = WasmLimits::of(&preloads.iter().collect::<Vec<_>>(), &witnesses)
        .unwrap()
        .regions()
        .unwrap();
    let slots = plan_ordered_slots(per_step).unwrap();
    let r = bit_width(used[0] - 1).max(n.trailing_zeros() as usize);
    let mu = bit_width(used[1] - 1).max(r);
    let geometry = Geometry::new(r, mu, slots.len(), n, 1).unwrap();
    let app = WasmApplication::with_slots(&geometry, regions, per_step, &slots).unwrap();
    let circuit = Circuit::compile(geometry, app.clone()).unwrap();
    let prover = circuit.prover(nightstream::Engine::Optimized, 109).unwrap();
    let verifier = circuit
        .verifier(nightstream::Engine::Optimized, 109)
        .unwrap();
    let programs = preloads
        .iter()
        .map(|preload| app.program(preload).unwrap())
        .collect::<Vec<_>>();
    let initials = cases
        .iter()
        .map(|case| top_level_initial_state(&case.artifacts.tables, case.trace[0].state_before.pc).unwrap())
        .collect::<Vec<_>>();
    let mut proofs = Vec::new();
    for index in 0..cases.len() {
        let mut run = prover.start(&programs[index], initials[index]).unwrap();
        run.segment(&inputs[index]).unwrap();
        let proof = run.finish().unwrap();
        let last = &cases[index].trace.last().unwrap().state_after;
        verifier
            .verify(&programs[index], &initials[index], last, &proof)
            .unwrap();
        let bytes = prover.encode_proof(&proof).unwrap();
        let decoded = verifier.decode_proof(&bytes).unwrap();
        verifier
            .verify(&programs[index], &initials[index], last, &decoded)
            .unwrap();
        proofs.push(proof);
    }
    for expected in [0, 1, 3] {
        for actual in [0, 1, 3] {
            if expected != actual {
                let last = &cases[expected].trace.last().unwrap().state_after;
                assert!(verifier
                    .verify(&programs[expected], &initials[expected], last, &proofs[actual])
                    .is_err());
            }
        }
    }
    let mut wrong = cases[2].trace.last().unwrap().state_after;
    wrong.trapped = false;
    assert!(verifier
        .verify(&programs[2], &initials[2], &wrong, &proofs[2])
        .is_err());
    eprintln!("shared_alu_public_proofs\t4\tminimum_security_bits\t109\t{geometry:?}");
}
