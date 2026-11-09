use super::*;
use neo_wasm::layout::{COL_SIGN_EXT_BIT, COL_SIGN_EXT_LOW7};
use neo_wasm::{StackValueAccess, WasmOpcode, WasmOutputState};
use nightstream::nebula::Circuit;
use p3_field::PrimeCharacteristicRing;

const EXTEND8: &str = r#"(module
    (func (export "main") (result i32)
        i32.const 128
        i32.extend8_s))"#;
const EXTEND16: &str = r#"(module
    (func (export "main") (result i32)
        i32.const 32768
        i32.extend16_s))"#;
const EXTEND_I32_WIDE: &str = r#"(module
    (func (export "main") (result i64)
        i32.const -2147483648
        i64.extend_i32_s))"#;
const EXTEND8_WIDE: &str = r#"(module
    (func (export "main") (result i64)
        i64.const 128
        i64.extend8_s))"#;
const EXTEND16_WIDE: &str = r#"(module
    (func (export "main") (result i64)
        i64.const 32768
        i64.extend16_s))"#;
const EXTEND32_WIDE: &str = r#"(module
    (func (export "main") (result i64)
        i64.const 2147483648
        i64.extend32_s))"#;
const LOAD32: &str = r#"(module
    (memory 1)
    (data (i32.const 0) "\00\00\00\80")
    (func (export "main") (result i64)
        i32.const 0
        i64.load32_s))"#;
const LOAD8: &str = r#"(module
    (memory 1)
    (data (i32.const 0) "\80")
    (func (export "main") (result i32)
        i32.const 0
        i32.load8_s))"#;
const LOAD16: &str = r#"(module
    (memory 1)
    (data (i32.const 0) "\00\80")
    (func (export "main") (result i32)
        i32.const 0
        i32.load16_s))"#;
const LOAD8_WIDE: &str = r#"(module
    (memory 1)
    (data (i32.const 0) "\80")
    (func (export "main") (result i64)
        i32.const 0
        i64.load8_s))"#;
const LOAD16_WIDE: &str = r#"(module
    (memory 1)
    (data (i32.const 0) "\00\80")
    (func (export "main") (result i64)
        i32.const 0
        i64.load16_s))"#;

fn change_access(access: &mut Option<StackValueAccess>, old: StackValueAccess, bad: StackValueAccess) {
    if let Some(access) = access {
        if access.value_lo == old.value_lo && access.value_hi == old.value_hi {
            access.value_lo = bad.value_lo;
            access.value_hi = bad.value_hi;
        }
    }
}

fn change_output(output: &mut WasmOutputState, low: Option<u32>, high: Option<u32>) {
    if output.enabled {
        if let Some(low) = low {
            output.value_lo = low;
        }
        if let Some(high) = high {
            output.value_hi = high;
        }
    }
}

fn public_verifier_accepts_bad_sign(
    wat: &str,
    opcode: WasmOpcode,
    reference_result: &str,
    low: Option<u32>,
    high: Option<u32>,
) -> bool {
    let mut checked = common::checked_main(wat);
    assert_eq!(checked.run.results, [reference_result]);
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == opcode)
        .unwrap();
    let old = checked.trace[row_index].stack_write0.unwrap();
    let mut bad = old;
    if let Some(low) = low {
        bad.value_lo = low;
    }
    if let Some(high) = high {
        bad.value_hi = Some(high);
    }
    assert_ne!(bad, old);
    checked.trace[row_index].stack_write0 = Some(bad);

    let mut output_captured = false;
    for row in &mut checked.trace[row_index + 1..] {
        change_access(&mut row.stack_read0, old, bad);
        change_access(&mut row.stack_write0, old, bad);
        change_output(&mut row.state_before.output, low, high);
        output_captured |= row.output_captured;
        change_output(&mut row.state_after.output, low, high);
    }
    assert!(output_captured);
    let last = checked.trace.last().unwrap().state_after;
    assert!(last.is_terminal());
    if let Some(low) = low {
        assert_eq!(last.output.value_lo, low);
    }
    if let Some(high) = high {
        assert_eq!(last.output.value_hi, high);
    }

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
        &checked.artifacts,
    );
    let n = checked.trace.len().next_power_of_two();
    let per_step = NonZeroUsize::MIN;
    let mut inputs = step_inputs(&checked.trace, n, per_step).unwrap();
    inputs[row_index].z[COL_SIGN_EXT_LOW7] = F::from_u64(128);
    inputs[row_index].z[COL_SIGN_EXT_BIT] = F::ZERO;
    let core_width = neo_wasm::build_wasm_relation()
        .unwrap()
        .r1cs()
        .structure()
        .m();
    let mut core = inputs[row_index].z[..core_width].to_vec();
    neo_wasm::write_range_check_bits(&mut core);
    inputs[row_index].z[..core_width].copy_from_slice(&core);
    let traced = Traced {
        states: inputs.iter().map(|row| row.next).collect(),
        rows: inputs.iter().map(|row| row.z.clone()).collect(),
        inputs,
        real: checked.trace.len(),
        preload,
        initial: top_level_initial_state(&checked.artifacts.tables, entry_pc).unwrap(),
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
fn sign_extension_cannot_return_an_unsigned_value_for_a_negative_byte() {
    assert!(
        !public_verifier_accepts_bad_sign(EXTEND8, WasmOpcode::I32Extend8S, "-128", Some(128), None),
        "the public verifier accepted output 128 for i32.extend8_s(128), which Wasmtime returns as -128"
    );
}

#[test]
fn signed_load_cannot_clear_the_high_limb_of_a_negative_word() {
    assert!(
        !public_verifier_accepts_bad_sign(LOAD32, WasmOpcode::I64Load32S, "-2147483648", None, Some(0)),
        "the public verifier accepted +2147483648 for i64.load32_s(0x80000000), which Wasmtime returns as -2147483648"
    );
}

#[test]
fn signed_byte_load_cannot_return_an_unsigned_value() {
    assert!(
        !public_verifier_accepts_bad_sign(LOAD8, WasmOpcode::I32Load8S, "-128", Some(128), None),
        "the public verifier accepted output 128 for i32.load8_s(0x80), which Wasmtime returns as -128"
    );
}

#[test]
fn signed_halfword_load_cannot_return_an_unsigned_value() {
    assert!(
        !public_verifier_accepts_bad_sign(LOAD16, WasmOpcode::I32Load16S, "-32768", Some(32768), None),
        "the public verifier accepted output 32768 for i32.load16_s(0x8000), which Wasmtime returns as -32768"
    );
}

#[test]
fn wide_signed_byte_load_cannot_clear_both_sign_extensions() {
    assert!(
        !public_verifier_accepts_bad_sign(LOAD8_WIDE, WasmOpcode::I64Load8S, "-128", Some(128), Some(0)),
        "the public verifier accepted +128 for i64.load8_s(0x80), which Wasmtime returns as -128"
    );
}

#[test]
fn sign_extension_of_a_negative_halfword_cannot_return_an_unsigned_value() {
    assert!(
        !public_verifier_accepts_bad_sign(EXTEND16, WasmOpcode::I32Extend16S, "-32768", Some(32768), None),
        "the public verifier accepted +32768 for i32.extend16_s(32768), which Wasmtime returns as -32768"
    );
}

#[test]
fn wide_signed_halfword_load_cannot_clear_both_sign_extensions() {
    assert!(
        !public_verifier_accepts_bad_sign(LOAD16_WIDE, WasmOpcode::I64Load16S, "-32768", Some(32768), Some(0)),
        "the public verifier accepted +32768 for i64.load16_s(0x8000), which Wasmtime returns as -32768"
    );
}

#[test]
fn wide_extend_i32_cannot_clear_the_sign_limb() {
    assert!(
        !public_verifier_accepts_bad_sign(EXTEND_I32_WIDE, WasmOpcode::I64ExtendI32S, "-2147483648", None, Some(0),),
        "the public verifier accepted +2147483648 for i64.extend_i32_s(-2147483648)"
    );
}

#[test]
fn wide_extend8_cannot_clear_both_sign_extensions() {
    assert!(
        !public_verifier_accepts_bad_sign(EXTEND8_WIDE, WasmOpcode::I64Extend8S, "-128", Some(128), Some(0)),
        "the public verifier accepted +128 for i64.extend8_s(128)"
    );
}

#[test]
fn wide_extend16_cannot_clear_both_sign_extensions() {
    assert!(
        !public_verifier_accepts_bad_sign(EXTEND16_WIDE, WasmOpcode::I64Extend16S, "-32768", Some(32768), Some(0)),
        "the public verifier accepted +32768 for i64.extend16_s(32768)"
    );
}

#[test]
fn wide_extend32_cannot_clear_the_sign_limb() {
    assert!(
        !public_verifier_accepts_bad_sign(EXTEND32_WIDE, WasmOpcode::I64Extend32S, "-2147483648", None, Some(0),),
        "the public verifier accepted +2147483648 for i64.extend32_s(2147483648)"
    );
}
