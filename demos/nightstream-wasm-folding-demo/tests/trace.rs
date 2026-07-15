use nightstream_wasm_folding_demo::{
    preprocess_program, prove_prepared, trace_program, ProofMode, TraceRequest, BR_TABLE_WAT, COUNTER_WAT,
    MUTABLE_GLOBAL_WAT, TABLE_DISPATCH_WAT, TRAPPING_DIVISION_WAT,
};

#[test]
fn counter_trace_comes_from_normalized_wasm_rows() {
    let (response, _) = trace_program(TraceRequest {
        source: COUNTER_WAT.to_string(),
    })
    .expect("trace counter loop");

    assert_eq!(response.execution.results, ["5"]);
    assert!(response.execution.halted);
    assert!(!response.execution.trapped);
    assert!(!response.rows.is_empty());
    assert_eq!(response.execution.normalized_rows, response.rows.len());
    assert!(response
        .rows
        .iter()
        .any(|row| row.opcode == "Call" && row.call_depth_before == 0 && row.call_depth_after == 1));
    assert!(response.rows.iter().any(|row| row.call_depth_before == 1));
    assert!(response.rows.iter().any(|row| row.opcode == "BrIf"));
    assert!(!response.program.has_linear_memory);
}

#[test]
fn table_dispatch_trace_exercises_indirect_calls_and_mutation() {
    let (response, _) = trace_program(TraceRequest {
        source: TABLE_DISPATCH_WAT.to_string(),
    })
    .expect("trace mutable function table");

    assert_eq!(response.execution.results, ["12"]);
    assert!(response.execution.halted);
    assert!(!response.execution.trapped);
    assert!(!response.program.has_linear_memory);
    for opcode in ["RefFunc", "TableSet", "CallIndirect"] {
        assert!(
            response.rows.iter().any(|row| row.opcode == opcode),
            "missing {opcode} row"
        );
    }
    assert!(response.rows.iter().any(|row| row.call_depth_after == 2));
}

#[test]
fn mutable_global_trace_exercises_global_reads_and_writes() {
    let (response, _) = trace_program(TraceRequest {
        source: MUTABLE_GLOBAL_WAT.to_string(),
    })
    .expect("trace mutable global state");

    assert_eq!(response.execution.results, ["17"]);
    assert!(response.execution.halted);
    assert!(!response.execution.trapped);
    assert!(!response.program.has_linear_memory);
    assert_eq!(
        response
            .rows
            .iter()
            .filter(|row| row.opcode == "GlobalGet")
            .count(),
        3
    );
    assert_eq!(
        response
            .rows
            .iter()
            .filter(|row| row.opcode == "GlobalSet")
            .count(),
        2
    );
    assert_eq!(
        response
            .rows
            .iter()
            .filter_map(|row| row.global_read)
            .collect::<Vec<_>>(),
        [10, 13, 17]
    );
    assert_eq!(
        response
            .rows
            .iter()
            .filter_map(|row| row.global_write)
            .collect::<Vec<_>>(),
        [13, 17]
    );
    assert!(response
        .rows
        .iter()
        .filter(|row| row.global_index.is_some())
        .all(|row| row.global_index == Some(0)));
    assert!(response
        .rows
        .iter()
        .any(|row| row.opcode == "Call" && row.call_depth_after == 1));
}

#[test]
fn trapping_division_trace_exposes_the_terminal_trap() {
    let (response, _) = trace_program(TraceRequest {
        source: TRAPPING_DIVISION_WAT.to_string(),
    })
    .expect("trace division by zero");

    assert!(response.execution.results.is_empty());
    assert!(response.execution.halted);
    assert!(response.execution.trapped);
    assert!(!response.program.has_linear_memory);
    let trap = response.rows.last().expect("terminal trap row");
    assert_eq!(trap.opcode, "I32DivU");
    assert!(trap.halted);
    assert!(trap.trapped);
    assert_eq!(trap.call_depth_before, 1);
    assert_eq!(
        trap.stack_reads
            .iter()
            .map(|value| value.lo)
            .collect::<Vec<_>>(),
        [42, 0]
    );
}

#[test]
fn br_table_trace_exposes_the_selected_control_edge() {
    let (response, _) = trace_program(TraceRequest {
        source: BR_TABLE_WAT.to_string(),
    })
    .expect("trace br_table dispatch");

    assert_eq!(response.execution.results, ["20"]);
    assert!(response.execution.halted);
    assert!(!response.execution.trapped);
    assert!(!response.program.has_linear_memory);
    let branch = response
        .rows
        .iter()
        .find(|row| row.opcode == "BrTable")
        .expect("br_table row");
    assert_eq!(branch.control_choice, 2);
    assert_eq!(branch.call_depth_before, 1);
}

#[test]
fn invalid_wat_is_reported_at_the_api_boundary() {
    let error = trace_program(TraceRequest {
        source: "(not-wasm)".to_string(),
    })
    .err()
    .expect("invalid WAT must fail");

    assert!(error.starts_with("invalid WAT:"));
}

#[test]
#[ignore = "builds and verifies the batch-32 recursive demo proof"]
fn counter_fast_mode_uses_authoritative_ivc_without_memory() {
    let (_, traced) = trace_program(TraceRequest {
        source: COUNTER_WAT.to_string(),
    })
    .expect("trace counter loop");
    let (preparation, prepared) =
        preprocess_program(traced, ProofMode::IvcNoMemory).expect("preprocess counter loop without memory consistency");
    let response = prove_prepared(&prepared).expect("prove counter loop without memory consistency");

    assert!(!preparation.memory_consistency);
    assert_eq!(preparation.batch_size, 32);
    assert_eq!(preparation.folds, 2);
    assert_eq!(response.result, ["5"]);
    assert!(!response.memory_consistency);
    assert_eq!(response.batch_size, 32);
    assert_eq!(response.folds, 2);
    assert!((33..=64).contains(&response.normalized_rows));
    assert_eq!(response.padded_rows, 64 - response.normalized_rows);
}

#[test]
#[ignore = "builds and verifies the table-dispatch recursive demo proof"]
fn table_dispatch_fast_mode_uses_authoritative_ivc_without_memory() {
    let (_, traced) = trace_program(TraceRequest {
        source: TABLE_DISPATCH_WAT.to_string(),
    })
    .expect("trace table dispatch");
    let (preparation, prepared) = preprocess_program(traced, ProofMode::IvcNoMemory)
        .expect("preprocess table dispatch without memory consistency");
    let response = prove_prepared(&prepared).expect("prove table dispatch without memory consistency");

    assert_eq!(response.result, ["12"]);
    assert!(!response.memory_consistency);
    assert_eq!(response.batch_size, 32);
    assert_eq!(response.folds, preparation.folds);
    assert_eq!(response.padded_rows, preparation.padded_rows);
}

#[test]
#[ignore = "builds and verifies the mutable-global recursive demo proof"]
fn mutable_global_fast_mode_uses_authoritative_ivc_without_memory() {
    let (_, traced) = trace_program(TraceRequest {
        source: MUTABLE_GLOBAL_WAT.to_string(),
    })
    .expect("trace mutable global state");
    let (preparation, prepared) = preprocess_program(traced, ProofMode::IvcNoMemory)
        .expect("preprocess mutable global state without memory consistency");
    let response = prove_prepared(&prepared).expect("prove mutable global state without memory consistency");

    assert_eq!(response.result, ["17"]);
    assert!(!response.memory_consistency);
    assert_eq!(response.batch_size, 32);
    assert_eq!(response.folds, preparation.folds);
    assert_eq!(response.padded_rows, preparation.padded_rows);
}

#[test]
#[ignore = "builds and verifies the trapping recursive demo proof"]
fn trapping_division_fast_mode_proves_the_terminal_trap() {
    let (_, traced) = trace_program(TraceRequest {
        source: TRAPPING_DIVISION_WAT.to_string(),
    })
    .expect("trace division by zero");
    let (preparation, prepared) = preprocess_program(traced, ProofMode::IvcNoMemory)
        .expect("preprocess trapping division without memory consistency");
    let response = prove_prepared(&prepared).expect("prove trapping division without memory consistency");

    assert!(response.result.is_empty());
    assert!(response.trapped);
    assert!(!response.memory_consistency);
    assert_eq!(response.batch_size, 32);
    assert_eq!(response.folds, 1);
    assert_eq!(response.folds, preparation.folds);
    assert_eq!(response.padded_rows, preparation.padded_rows);
}

#[test]
#[ignore = "builds and verifies the br_table recursive demo proof"]
fn br_table_recursive_mode_proves_the_selected_branch() {
    let (_, traced) = trace_program(TraceRequest {
        source: BR_TABLE_WAT.to_string(),
    })
    .expect("trace br_table dispatch");
    let (preparation, prepared) =
        preprocess_program(traced, ProofMode::IvcNoMemory).expect("preprocess br_table without memory consistency");
    let response = prove_prepared(&prepared).expect("prove br_table without memory consistency");

    assert_eq!(response.result, ["20"]);
    assert!(!response.trapped);
    assert!(!response.memory_consistency);
    assert_eq!(response.batch_size, 32);
    assert_eq!(response.folds, preparation.folds);
    assert_eq!(response.padded_rows, preparation.padded_rows);
}

#[test]
#[ignore = "builds and replay-verifies the no-NIFS.V folding audit"]
fn br_table_folding_audit_mode_replays_the_proof() {
    let (_, traced) = trace_program(TraceRequest {
        source: BR_TABLE_WAT.to_string(),
    })
    .expect("trace br_table dispatch");
    let (preparation, prepared) =
        preprocess_program(traced, ProofMode::FoldingAudit).expect("preprocess br_table folding audit");
    let response = prove_prepared(&prepared).expect("prove br_table folding audit");

    assert_eq!(preparation.mode, "folding_audit");
    assert_eq!(preparation.batch_size, 64);
    assert_eq!(response.mode, "folding_audit");
    assert_eq!(response.result, ["20"]);
    assert!(!response.trapped);
    assert!(!response.memory_consistency);
    assert_eq!(response.batch_size, 64);
    assert_eq!(response.folds, preparation.folds);
    assert_eq!(response.padded_rows, preparation.padded_rows);
}
