#![allow(dead_code)]

pub mod audit;

use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use neo_wasm::layout::COLUMN_SPECS;
use neo_wasm::{
    build_wasm_relation_layout, collect_wasmtime_steps, extract_wasm_program_artifacts, opcode_info_from_code,
    preload_from_program_artifacts, sanity_check_lookup_row, sanity_check_memory_rows, top_level_initial_state_digest,
    traces_from_wasmtime_steps, witness_builder::build_witness_vector, LinearMemoryAccess, StackValueAccess,
    WasmCountdownState, WasmOpcode, WasmOutputState, WasmPcEdgeKind, WasmProgramArtifacts, WasmRowKind, WasmStepState,
    WasmVmSpec, WasmVmStep, WasmtimeTraceRun,
};

pub struct CheckedWasmRun {
    pub wasm: Vec<u8>,
    pub artifacts: WasmProgramArtifacts,
    pub run: WasmtimeTraceRun,
    pub trace: Vec<WasmVmStep>,
    pub witnesses: Vec<Vec<F>>,
}

pub fn checked_main(wat_src: &str) -> CheckedWasmRun {
    checked_wasm_run(wat_src, "main", &[])
}

pub fn checked_wasm_run(wat_src: &str, export: &str, params: &[i32]) -> CheckedWasmRun {
    let wasm = wat::parse_str(wat_src).expect("valid WAT");
    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = collect_wasmtime_steps(&wasm, export, params).expect("wasmtime trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");
    let witnesses = sanity_check_trace(&trace, &artifacts, &run.initial_locals);
    ccs_check_trace(&trace);
    assert_output_matches_reference(&trace, &run.results);
    CheckedWasmRun {
        wasm,
        artifacts,
        run,
        trace,
        witnesses,
    }
}

/// Cross-check the proof-bound output (the `output` carried into the final
/// semantic state by the normalizer) against wasmtime's reference return
/// value, which reaches us through `func.call_async` without touching the
/// step stream or the normalizer. This is the only end-to-end check that
/// the value the proof binds is the value the program actually produced.
fn assert_output_matches_reference(trace: &[WasmVmStep], results: &[String]) {
    let output = final_state(trace).output;
    if !output.enabled {
        return;
    }
    let [reference] = results else {
        panic!("captured output but reference results are not single-valued: {results:?}");
    };
    let reference: i128 = reference
        .parse()
        .unwrap_or_else(|err| panic!("non-integer reference result '{reference}': {err}"));
    let got = (u64::from(output.value_hi) << 32) | u64::from(output.value_lo);
    // The reference is a decimal string with the result type erased, so a
    // negative value's bit pattern is ambiguous between i32 and i64.
    let as_i64_bits = (reference as i64) as u64;
    let as_i32_bits = u64::from((reference as i32) as u32);
    assert!(
        got == as_i64_bits || got == as_i32_bits,
        "proof-bound output {got:#x} does not match wasmtime reference result {reference}",
    );
}

pub fn sanity_check_trace(
    trace: &[WasmVmStep],
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
) -> Vec<Vec<F>> {
    let layout = build_wasm_relation_layout();
    let mut witnesses = Vec::with_capacity(trace.len());
    for row in trace {
        let witness = build_witness_vector(row);
        sanity_check_lookup_row(&layout.auxiliary, &witness)
            .unwrap_or_else(|err| panic!("lookup semantics rejected {:?}: {err}", row.opcode));
        witnesses.push(witness);
    }
    let preload = preload_from_program_artifacts(artifacts, initial_locals);
    sanity_check_memory_rows(layout, &witnesses, &preload)
        .unwrap_or_else(|err| panic!("memory semantics rejected trace: {err}"));
    witnesses
}

/// Hand-build a single program row for direct row-CCS tests. Stack lanes are
/// given as logical (slot, value) pairs; addresses are doubled into the
/// physical layout and `value_hi` limbs are dropped, so wide rows built here
/// carry zero high limbs. Trap synthesis (zero divisor, signed overflow)
/// mirrors the trace normalizer.
#[allow(clippy::too_many_arguments)]
pub fn step(
    cycle: u64,
    pc_before: u64,
    opcode_code: u16,
    sp_before: u64,
    sp_after: u64,
    stack_read0: Option<StackValueAccess>,
    stack_read1: Option<StackValueAccess>,
    stack_read2: Option<StackValueAccess>,
    stack_write0: Option<StackValueAccess>,
    linear_memory: Option<LinearMemoryAccess>,
    linear_memory_offset: u64,
    halted: bool,
) -> WasmVmStep {
    fn state(pc: u64, sp: u64, halted: bool) -> WasmStepState {
        WasmStepState {
            pc,
            sp,
            output: WasmOutputState::ZERO,
            call_stack_depth: 0,
            memory_pages: None,
            max_memory_pages: None,
            locals_fbp: 0,
            halted,
            trapped: false,
            param_init: WasmCountdownState::ZERO,
            host_args: WasmCountdownState::ZERO,
            host_result_pending: false,
            host_callee_fref: 0,
        }
    }

    fn physical(access: Option<StackValueAccess>) -> Option<StackValueAccess> {
        access.map(|lane| StackValueAccess::new(lane.addr_lo * 2, lane.value_lo))
    }

    let opcode = opcode_info_from_code(opcode_code).opcode;
    let div_zero_trap = opcode.traps_on_zero_divisor()
        && stack_read1.is_some_and(|lane| lane.value_lo == 0 && lane.value_hi.unwrap_or(0) == 0);
    let (min_lo, min_hi) = if opcode.uses_wide_values() {
        (0, 0x8000_0000)
    } else {
        (0x8000_0000, 0)
    };
    let neg1_hi = if opcode.uses_wide_values() { u32::MAX } else { 0 };
    let div_overflow_trap = opcode.traps_on_signed_overflow()
        && stack_read0.is_some_and(|lane| lane.value_lo == min_lo && lane.value_hi.unwrap_or(0) == min_hi)
        && stack_read1.is_some_and(|lane| lane.value_lo == u32::MAX && lane.value_hi.unwrap_or(0) == neg1_hi);
    let div_trap = div_zero_trap || div_overflow_trap;
    WasmVmStep {
        cycle,
        row_kind: WasmRowKind::Program,
        state_before: state(pc_before, sp_before, false),
        state_after: WasmStepState {
            trapped: matches!(opcode, WasmOpcode::Unreachable) || div_trap,
            ..state(pc_before + 1, sp_after, halted)
        },
        control_choice: 0,
        pc_edge_kind: match opcode {
            WasmOpcode::Return | WasmOpcode::End => WasmPcEdgeKind::ReturnLike,
            WasmOpcode::CallIndirect => WasmPcEdgeKind::DynamicCallIndirect,
            WasmOpcode::Unreachable => WasmPcEdgeKind::Terminal,
            _ => WasmPcEdgeKind::Static,
        },
        wide_values_enabled: opcode_info_from_code(opcode_code).opcode.uses_wide_values(),
        opcode: opcode_info_from_code(opcode_code).opcode,
        info: opcode_info_from_code(opcode_code),
        stack_reads_override: None,
        stack_writes_override: None,
        output_captured: false,
        current_function_ref: 0,
        current_function_num_locals: 0,
        stack_read0: physical(stack_read0),
        stack_read1: physical(stack_read1),
        stack_read2: physical(stack_read2),
        stack_write0: physical(stack_write0),
        linear_memory,
        linear_memory_offset,
        local_index: None,
        local_read_value: None,
        local_read_value_hi: None,
        local_write_value: None,
        local_write_value_hi: None,
        global_index: None,
        global_read_value: None,
        global_read_value_hi: None,
        global_write_value: None,
        global_write_value_hi: None,
        table_id: None,
        table_index: None,
        table_value: None,
        function_ref: None,
        target_function_is_guest: false,
        function_type_id: None,
        call_indirect_type_index: None,
        expected_type_id: None,
        table_size: None,
        call_param_count: None,
        call_result_count: None,
        call_stack_push: None,
        call_stack_pop: None,
    }
}

pub fn assert_satisfied(z: &[F], label: &str) {
    let layout = build_wasm_relation_layout();
    sanity_check_lookup_row(&layout.auxiliary, z)
        .unwrap_or_else(|e| panic!("{label}: expected lookup semantics satisfied, got: {e}"));
    let vm = WasmVmSpec::default();
    let ccs = &vm.core_ccs_spec().structure;
    let m_in = vm.core_ccs_spec().m_in;
    // Keep aux bits consistent with any caller-mutated declared columns.
    let mut z = z.to_vec();
    neo_wasm::write_range_check_bits(&mut z);
    let (x, w) = (&z[..m_in], &z[m_in..]);
    check_ccs_rowwise_zero(ccs, x, w).unwrap_or_else(|e| panic!("{label}: expected CCS satisfied, got: {e}"));
}

pub fn assert_rejected(z: &[F], label: &str) {
    let vm = WasmVmSpec::default();
    let ccs = &vm.core_ccs_spec().structure;
    let m_in = vm.core_ccs_spec().m_in;
    // Keep aux bits consistent so in-range forgeries exercise semantic rows.
    let mut z = z.to_vec();
    neo_wasm::write_range_check_bits(&mut z);
    let (x, w) = (&z[..m_in], &z[m_in..]);
    assert!(
        check_ccs_rowwise_zero(ccs, x, w).is_err(),
        "{label}: expected CCS rejection, but the witness was accepted"
    );
}

pub fn ccs_check_trace(trace: &[WasmVmStep]) {
    let vm = WasmVmSpec::default();
    let ccs = &vm.core_ccs_spec().structure;
    let catalog = vm.constraint_catalog();
    for (idx, row) in trace.iter().enumerate() {
        let witness = build_witness_vector(row);
        let m_in = vm.core_ccs_spec().m_in;
        let (x, w) = (&witness[..m_in], &witness[m_in..]);
        check_ccs_rowwise_zero(ccs, x, w).unwrap_or_else(|err| {
            let detail = err.to_string();
            let row_idx = detail
                .split_once("row ")
                .and_then(|(_, rest)| rest.split_once(':'))
                .and_then(|(row, _)| row.parse::<usize>().ok());
            let tag = row_idx.and_then(|row| catalog.row_tags.get(row));
            let terms = row_idx
                .and_then(|row| catalog.rows.get(row))
                .map(|row| {
                    format!(
                        "A={}; B={}; C={}",
                        format_terms(&row.a_terms),
                        format_terms(&row.b_terms),
                        format_terms(&row.c_terms)
                    )
                })
                .unwrap_or_else(|| "terms unavailable".to_string());
            panic!(
                "trace row {idx} {:?} pc {}->{} sp {}->{} halted={} failed CCS satisfaction: {err}; tag={tag:?}; {terms}",
                row.opcode, row.state_before.pc, row.state_after.pc, row.state_before.sp, row.state_after.sp, row.state_after.halted
            );
        });
    }
}

pub fn verifier_initial_state_digest(artifacts: &WasmProgramArtifacts) -> [u8; 32] {
    let entry_pc = single_function_entry_pc(artifacts);
    top_level_initial_state_digest(&artifacts.tables, entry_pc)
}

/// Prover-disclosed final VM state for `neo_wasm::verify`.
pub fn final_state(trace: &[WasmVmStep]) -> WasmStepState {
    trace.last().expect("non-empty trace").state_after
}

pub fn single_function_entry_pc(artifacts: &WasmProgramArtifacts) -> u64 {
    let mut entries = artifacts
        .tables
        .function_entries
        .iter()
        .map(|&(_, entry_pc)| entry_pc)
        .collect::<Vec<_>>();
    entries.sort_unstable();
    entries.dedup();
    assert_eq!(
        entries.len(),
        1,
        "test helper can only infer the entry PC for single-entry wasm fixtures"
    );
    entries[0]
}

pub fn entry_pc_for_function_ref(artifacts: &WasmProgramArtifacts, function_ref: u64) -> u64 {
    artifacts
        .tables
        .function_entries
        .iter()
        .find(|&&(fref, _)| fref == function_ref)
        .map(|&(_, pc)| pc)
        .unwrap_or_else(|| panic!("function_ref {function_ref} not in function_entries"))
}

fn format_terms(terms: &[(usize, F)]) -> String {
    terms
        .iter()
        .map(|(col, coeff)| {
            let name = COLUMN_SPECS.get(*col).map(|spec| spec.name).unwrap_or("?");
            format!("{coeff:?}*{name}[{col}]")
        })
        .collect::<Vec<_>>()
        .join(" + ")
}
