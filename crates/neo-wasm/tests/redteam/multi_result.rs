use super::*;
use nightstream::nebula::Circuit;

const TWO_RESULTS: &str = r#"(module
    (func (export "main") (result i32 i32)
        i32.const 1
        i32.const 2))"#;
const THREE_RESULTS: &str = r#"(module
    (func (export "main") (result i32 i32 i32)
        i32.const 1
        i32.const 2
        i32.const 3))"#;
const NO_RESULT: &str = r#"(module
    (func (export "main")
        i32.const 7
        drop))"#;

fn public_verifier_accepts_disabled_output(wat: &str, change: fn(&mut neo_wasm::WasmOutputState)) -> bool {
    let checked = common::checked_main(wat);
    let mut trace = checked.trace;
    let artifacts = checked.artifacts;
    let capture = trace
        .iter()
        .rposition(|row| row.state_after.activation_finished && !row.state_before.activation_finished)
        .unwrap();
    for (index, row) in trace.iter_mut().enumerate().skip(capture) {
        if index > capture {
            change(&mut row.state_before.output);
        }
        change(&mut row.state_after.output);
    }
    let last = trace.last().unwrap().state_after;
    assert!(last.is_terminal());

    let (export_fref, entry_pc) = artifacts
        .tables
        .function_entries
        .iter()
        .copied()
        .max_by_key(|(_, pc)| *pc)
        .unwrap();
    let mut preload = preload_from_program_artifacts(&artifacts);
    preload_host_event_tables(
        &mut preload,
        &HostEventBindings::import_free(u32::try_from(export_fref).unwrap()),
        &artifacts,
    );
    let n = trace.len().next_power_of_two();
    let per_step = NonZeroUsize::MIN;
    let inputs = step_inputs(&trace, n, per_step).unwrap();
    let traced = Traced {
        states: inputs.iter().map(|row| row.next).collect(),
        rows: inputs.iter().map(|row| row.z.clone()).collect(),
        inputs,
        real: trace.len(),
        preload,
        initial: top_level_initial_state(&artifacts.tables, entry_pc).unwrap(),
        last,
    };
    let setup = Setup::of(vec![traced], n, per_step);
    let (app, geometry) = setup.complete();
    let traced = &setup.programs[0];
    let program = app.program(&traced.preload).unwrap();
    let circuit = Circuit::compile(geometry, app).unwrap();
    let prover = circuit.prover(nightstream::Engine::Optimized, 109).unwrap();
    let verifier = circuit
        .verifier(nightstream::Engine::Optimized, 109)
        .unwrap();
    let mut proving = prover.start(&program, traced.initial).unwrap();
    if proving.segment(&traced.inputs).is_err() {
        return false;
    }
    let proof = proving.finish().unwrap();
    verifier
        .verify(&program, &traced.initial, &traced.last, &proof)
        .is_ok()
}

#[test]
fn resultless_export_cannot_carry_an_inactive_value() {
    assert!(
        !public_verifier_accepts_disabled_output(NO_RESULT, |output| output.value_lo = 3),
        "the public verifier accepted a value for a disabled result"
    );
}

#[test]
fn an_export_with_three_results_is_rejected_before_capture() {
    let wasm = wat::parse_str(THREE_RESULTS).unwrap();
    let error = neo_wasm::collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "main", &[]).unwrap_err();
    assert!(matches!(error, neo_wasm::WasmBuildError::Unsupported(message) if message.contains("at most two")));
}

#[test]
fn valid_two_result_export_has_a_public_proof() {
    let wasm = wat::parse_str(TWO_RESULTS).unwrap();
    let run = neo_wasm::collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "main", &[]).unwrap();
    assert_eq!(run.results, ["1", "2"]);
    let artifacts = run.artifacts().clone();
    let trace = neo_wasm::traces_from_wasmtime_steps(&run.steps).unwrap();
    let last = trace.last().unwrap().state_after;
    assert!(last.is_terminal() && last.output.enabled && !last.trapped);
    assert_eq!(last.output.value_lo, 2);

    let (export_fref, entry_pc) = artifacts
        .tables
        .function_entries
        .iter()
        .copied()
        .max_by_key(|(_, pc)| *pc)
        .unwrap();
    let mut preload = preload_from_program_artifacts(&artifacts);
    preload_host_event_tables(
        &mut preload,
        &HostEventBindings::import_free(u32::try_from(export_fref).unwrap()),
        &artifacts,
    );
    let n = trace.len().next_power_of_two();
    let per_step = NonZeroUsize::MIN;
    let inputs = step_inputs(&trace, n, per_step).unwrap();
    let traced = Traced {
        states: inputs.iter().map(|row| row.next).collect(),
        rows: inputs.iter().map(|row| row.z.clone()).collect(),
        inputs,
        real: trace.len(),
        preload,
        initial: top_level_initial_state(&artifacts.tables, entry_pc).unwrap(),
        last,
    };
    let setup = Setup::of(vec![traced], n, per_step);
    let (app, geometry) = setup.complete();
    let traced = &setup.programs[0];
    let program = app.program(&traced.preload).unwrap();
    let circuit = Circuit::compile(geometry, app).unwrap();
    let prover = circuit.prover(nightstream::Engine::Optimized, 109).unwrap();
    let verifier = circuit
        .verifier(nightstream::Engine::Optimized, 109)
        .unwrap();
    let mut proving = prover.start(&program, traced.initial).unwrap();
    let result = proving.segment(&traced.inputs);
    assert!(
        result.is_ok(),
        "the public prover rejected a valid WASM export with two results: {result:?}"
    );
    let proof = proving.finish().unwrap();
    assert!(
        verifier
            .verify(&program, &traced.initial, &traced.last, &proof)
            .is_ok(),
        "the public verifier rejected a valid WASM export with two results"
    );
}
