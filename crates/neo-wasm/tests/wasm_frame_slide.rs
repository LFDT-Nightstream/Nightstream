//! Branches and returns that drop operands slide the kept ones down with move rows.

mod common;

use neo_wasm::{
    build_wasm_relation_layout, preload_from_program_artifacts, sanity_check_memory_rows, WasmAuxOpcode,
    WasmFrameMoveState, WasmOpcode, WasmRowKind,
};

fn main_returning(result: &str, body: &str) -> String {
    format!("(module (func (export \"main\") (result {result}) {body}))")
}

fn moves(run: &common::CheckedWasmRun) -> usize {
    run.trace
        .iter()
        .filter(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::FrameMove))
        .count()
}

#[test]
fn branches_drop_operands_under_the_carried_ones() {
    for (result, body, expected_moves) in [
        (
            "i32",
            "block (result i32) i32.const 99 i32.const 42 i32.const 1 br_if 0 drop end",
            1,
        ),
        (
            "i32",
            "block (result i32) i32.const 5 i32.const 6 i32.const 7 br 0 end",
            1,
        ),
        (
            "i32",
            "block (result i32) i32.const 5 i32.const 6 i32.const 0 br_table 0 0 end",
            1,
        ),
        (
            "i32",
            "block (result i32) i32.const 5 block i32.const 6 i32.const 1 br_if 1 drop end end",
            1,
        ),
        ("i32", "block i32.const 99 i32.const 1 br_if 0 drop end i32.const 3", 0),
        ("i32", "block i32.const 99 i32.const 0 br_if 0 drop end i32.const 3", 0),
        (
            "i64",
            "block (result i64) i64.const 1 i64.const 0x1234567890 i32.const 1 br_if 0 drop end i64.const 0 i64.add",
            1,
        ),
    ] {
        let run = common::checked_main(&main_returning(result, body));
        assert_eq!(moves(&run), expected_moves, "{body}");
    }
}

#[test]
fn loop_back_edges_drop_temporaries() {
    let run = common::checked_main(&main_returning(
        "i32",
        "(local i32) loop i32.const 99 local.get 0 i32.const 1 i32.add local.tee 0 i32.const 3 i32.lt_u br_if 0 drop end local.get 0",
    ));
    let back_edges: Vec<_> = run
        .trace
        .iter()
        .filter(|row| row.opcode == WasmOpcode::BrIf && row.control_choice == 1)
        .collect();
    assert_eq!(back_edges.len(), 2);
    for row in back_edges {
        assert_eq!(
            row.state_before.sp,
            row.state_after.sp + 2,
            "the taken back edge must discard both the condition and the temporary"
        );
        assert_eq!(row.state_after.frame_move.distance, 1);
        assert_eq!(row.state_after.frame_move.remaining, 0);
    }
}

#[test]
fn nested_returns_drop_operands_under_the_results() {
    let run = common::checked_main(
        r#"(module
            (func $f (param i32) (result i32) (local i32)
              i32.const 7 block local.get 0 i32.const 1 i32.add return end drop i32.const 0)
            (func (export "main") (result i32) i32.const 3 i32.const 40 call $f i32.add))"#,
    );
    assert_eq!(moves(&run), 1);
}

#[test]
fn activation_returns_capture_only_their_results() {
    // A clean activation return binds its result count through its static slide.
    for (wat, output) in [
        (main_returning("i32", "i32.const 7 i32.const 9 return"), Some(9)),
        (
            main_returning("i32", "block i32.const 7 i32.const 9 return end i32.const 0"),
            Some(9),
        ),
        ("(module (func (export \"main\") i32.const 7 return))".to_string(), None),
        // A tail call makes the callee the activation's outermost frame.
        (
            "(module (func $f i32.const 7 return) (func (export \"main\") return_call $f))".to_string(),
            None,
        ),
        (
            "(module (func $f (result i32) i32.const 7 i32.const 9 return)
               (func (export \"main\") (result i32) return_call $f))"
                .to_string(),
            Some(9),
        ),
    ] {
        let run = common::checked_main(&wat);
        let state = common::final_state(&run.trace).output;
        assert_eq!(state.enabled, output.is_some(), "{wat}");
        if let Some(value) = output {
            assert_eq!(state.value_lo, value, "{wat}");
        }
    }
}

#[test]
fn normalization_rejects_return_slide_larger_than_frame() {
    let run = common::checked_main("(module (func (export \"main\") i32.const 7 return))");
    let mut steps = run.run.steps.clone();
    let ret = steps
        .iter_mut()
        .find(|step| step.opcode_decoded == Some(WasmOpcode::Return))
        .expect("return step");
    ret.frame_slide = Some((0, 2));

    let error = neo_wasm::traces_from_wasmtime_steps(&steps).expect_err("slide exceeds the one-cell frame");
    assert!(matches!(error, neo_wasm::WasmBuildError::Trace(_)), "{error}");
    assert!(error.to_string().contains("return result count underflow"), "{error}");
}

#[test]
fn a_clean_return_cannot_claim_a_dropped_operand() {
    let run = common::checked_main("(module (func $f i32.const 7 return) (func (export \"main\") return_call $f))");
    let index = run
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::Return)
        .expect("the callee's return");
    let mut forged = run.trace[index].clone();
    forged.output_captured = true;
    forged.state_after.output.enabled = true;
    forged.state_after.output.value_lo = 7;
    // The row is locally consistent; its claimed slide distance is not the return's.
    common::assert_satisfied(
        &neo_wasm::witness_builder::build_witness_vector(&forged),
        "clean return claiming a result",
    );
    reject_memory(&run, index, &forged, "a resultless return capturing its temporary");
}

fn slide_row(run: &common::CheckedWasmRun, opcode: WasmOpcode) -> usize {
    run.trace
        .iter()
        .position(|row| row.opcode == opcode && row.state_after.frame_move.distance != 0)
        .expect("a sliding row")
}

fn reject_memory(run: &common::CheckedWasmRun, index: usize, forged: &neo_wasm::WasmVmStep, label: &str) {
    let mut prefix = run.witnesses[..=index].to_vec();
    prefix[index] = neo_wasm::witness_builder::build_witness_vector(forged);
    let result = sanity_check_memory_rows(
        build_wasm_relation_layout(),
        &prefix,
        &preload_from_program_artifacts(&run.artifacts),
    );
    assert!(result.is_err(), "{label}");
}

#[test]
fn a_branch_cannot_skip_its_slide() {
    let run = common::checked_main(&main_returning(
        "i32",
        "block (result i32) i32.const 5 i32.const 6 i32.const 7 br 0 end",
    ));
    let index = slide_row(&run, WasmOpcode::Br);
    let mut forged = run.trace[index].clone();
    forged.state_after.frame_move = WasmFrameMoveState::ZERO;
    forged.state_after.sp = forged.state_before.sp;
    // The row is locally consistent; only the slide ROM rejects it.
    common::assert_satisfied(
        &neo_wasm::witness_builder::build_witness_vector(&forged),
        "branch row with no slide",
    );
    reject_memory(&run, index, &forged, "a branch must take its edge's slide");
}

#[test]
fn a_slide_must_drop_sp_by_its_distance() {
    let run = common::checked_main(&main_returning(
        "i32",
        "block (result i32) i32.const 5 i32.const 6 i32.const 7 br 0 end",
    ));
    let index = slide_row(&run, WasmOpcode::Br);
    let mut forged = run.trace[index].clone();
    forged.state_after.sp += 1;
    common::assert_rejected(
        &neo_wasm::witness_builder::build_witness_vector(&forged),
        "slide sp must drop by the distance",
    );
}

#[test]
fn a_return_cannot_skip_its_slide() {
    let run = common::checked_main(
        r#"(module
            (func $f (result i32) i32.const 7 i32.const 9 return)
            (func (export "main") (result i32) call $f))"#,
    );
    let index = slide_row(&run, WasmOpcode::Return);
    let mut forged = run.trace[index].clone();
    forged.state_after.frame_move.remaining = 2;
    forged.state_after.frame_move.distance -= 1;
    forged.state_after.sp += 1;
    common::assert_satisfied(
        &neo_wasm::witness_builder::build_witness_vector(&forged),
        "return row keeping both operands",
    );
    reject_memory(&run, index, &forged, "a return must take its static slide");
}
