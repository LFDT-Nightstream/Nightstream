use super::*;
use neo_wasm::{StackValueAccess, WasmOpcode, WasmOutputState};
use nightstream::nebula::Circuit;

fn change_access(access: &mut Option<StackValueAccess>, old: StackValueAccess, bad: StackValueAccess) {
    if let Some(access) = access {
        if access.value_lo == old.value_lo && access.value_hi == old.value_hi {
            access.value_lo = bad.value_lo;
            access.value_hi = bad.value_hi;
        }
    }
}

fn change_output(output: &mut WasmOutputState, bad: StackValueAccess) {
    if output.enabled {
        output.value_lo = bad.value_lo;
        output.value_hi = bad.value_hi.unwrap();
    }
}

fn public_verifier_accepts_count_alias(wat: &str, opcode: WasmOpcode, reference: &str, count: u32) -> bool {
    let mut checked = common::checked_main(wat);
    assert_eq!(checked.run.results, [reference]);
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == opcode)
        .unwrap();
    let old = checked.trace[row_index].stack_write0.unwrap();
    let mut bad = old;
    bad.value_lo = count + 1;
    bad.value_hi = Some(u32::MAX);
    assert_ne!(bad, old);
    checked.trace[row_index].stack_write0 = Some(bad);
    let mut output_captured = false;
    for row in &mut checked.trace[row_index + 1..] {
        change_access(&mut row.stack_read0, old, bad);
        change_access(&mut row.stack_write0, old, bad);
        change_output(&mut row.state_before.output, bad);
        output_captured |= row.output_captured;
        change_output(&mut row.state_after.output, bad);
    }
    assert!(output_captured);
    let last = checked.trace.last().unwrap().state_after;
    assert!(last.is_terminal());
    assert_eq!(last.output.value_lo, bad.value_lo);
    assert_eq!(last.output.value_hi, bad.value_hi.unwrap());

    let (export_fref, entry_pc) = checked
        .artifacts
        .tables
        .function_entries
        .iter()
        .copied()
        .max_by_key(|(_, pc)| *pc)
        .unwrap();
    let mut preload = preload_from_program_artifacts(&checked.artifacts);
    preload_host_event_tables(
        &mut preload,
        &HostEventBindings::import_free(u32::try_from(export_fref).unwrap()),
    );
    let n = checked.trace.len().next_power_of_two();
    let per_step = NonZeroUsize::MIN;
    let inputs = step_inputs(&checked.trace, n, per_step).unwrap();
    let traced = Traced {
        states: inputs.iter().map(|row| row.next).collect(),
        rows: inputs.iter().map(|row| row.z.clone()).collect(),
        inputs,
        real: checked.trace.len(),
        preload,
        initial: top_level_initial_state(&checked.artifacts.tables, entry_pc),
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
    prover
        .start(&program, traced.initial)
        .ok()
        .and_then(|mut run| {
            run.segment(&traced.inputs).ok()?;
            run.finish().ok()
        })
        .is_some_and(|proof| {
            verifier
                .verify(&program, &traced.initial, &traced.last, &proof)
                .is_ok()
        })
}

#[test]
fn i64_clz_must_not_accept_a_field_modulus_alias() {
    let wat = r#"(module
        (func (export "main") (result i64)
            i64.const 1
            i64.clz))"#;
    assert!(
        !public_verifier_accepts_count_alias(wat, WasmOpcode::I64Clz, "63", 63),
        "the public verifier accepted q+63 as i64.clz(1), which Wasmtime returns as 63"
    );
}

#[test]
fn i64_ctz_must_not_accept_a_field_modulus_alias() {
    let wat = r#"(module
        (func (export "main") (result i64)
            i64.const 1
            i64.ctz))"#;
    assert!(
        !public_verifier_accepts_count_alias(wat, WasmOpcode::I64Ctz, "0", 0),
        "the public verifier accepted q as i64.ctz(1), which Wasmtime returns as 0"
    );
}

#[test]
fn i64_popcnt_must_not_accept_a_field_modulus_alias() {
    let wat = r#"(module
        (func (export "main") (result i64)
            i64.const 1
            i64.popcnt))"#;
    assert!(
        !public_verifier_accepts_count_alias(wat, WasmOpcode::I64Popcnt, "1", 1),
        "the public verifier accepted q+1 as i64.popcnt(1), which Wasmtime returns as 1"
    );
}
