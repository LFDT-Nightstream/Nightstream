#![allow(dead_code)]

use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use neo_wasm::layout::COLUMN_SPECS;
use neo_wasm::{
    build_wasm_lookup_binding_layout, collect_wasmtime_steps, extract_wasm_program_artifacts,
    preload_from_program_artifacts, sanity_check_lookup_row, sanity_check_memory_rows, top_level_initial_state_digest,
    traces_from_wasmtime_steps, witness_builder::build_witness_vector, WasmProgramArtifacts, WasmStepState,
    WasmStepTrace, WasmVmSpec, WasmtimeTraceRun,
};

pub struct CheckedWasmRun {
    pub wasm: Vec<u8>,
    pub artifacts: WasmProgramArtifacts,
    pub run: WasmtimeTraceRun,
    pub trace: Vec<WasmStepTrace>,
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
fn assert_output_matches_reference(trace: &[WasmStepTrace], results: &[String]) {
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
    trace: &[WasmStepTrace],
    artifacts: &WasmProgramArtifacts,
    initial_locals: &[u32],
) -> Vec<Vec<F>> {
    let layout = build_wasm_lookup_binding_layout();
    let mut witnesses = Vec::with_capacity(trace.len());
    for row in trace {
        let witness = build_witness_vector(row);
        sanity_check_lookup_row(layout, &witness)
            .unwrap_or_else(|err| panic!("lookup semantics rejected {:?}: {err}", row.opcode));
        witnesses.push(witness);
    }
    let preload = preload_from_program_artifacts(artifacts, initial_locals);
    sanity_check_memory_rows(layout, &witnesses, &preload)
        .unwrap_or_else(|err| panic!("memory semantics rejected trace: {err}"));
    witnesses
}

pub fn assert_satisfied(z: &[F], label: &str) {
    let layout = build_wasm_lookup_binding_layout();
    sanity_check_lookup_row(layout, z)
        .unwrap_or_else(|e| panic!("{label}: expected lookup semantics satisfied, got: {e}"));
    let vm = WasmVmSpec::default();
    let ccs = &vm.core_ccs_spec().structure;
    let m_in = vm.core_ccs_spec().m_in;
    let (x, w) = (&z[..m_in], &z[m_in..]);
    check_ccs_rowwise_zero(ccs, x, w).unwrap_or_else(|e| panic!("{label}: expected CCS satisfied, got: {e}"));
}

pub fn assert_rejected(z: &[F], label: &str) {
    let vm = WasmVmSpec::default();
    let ccs = &vm.core_ccs_spec().structure;
    let m_in = vm.core_ccs_spec().m_in;
    let (x, w) = (&z[..m_in], &z[m_in..]);
    assert!(
        check_ccs_rowwise_zero(ccs, x, w).is_err(),
        "{label}: expected CCS rejection, but the witness was accepted"
    );
}

pub fn ccs_check_trace(trace: &[WasmStepTrace]) {
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
pub fn final_state(trace: &[WasmStepTrace]) -> WasmStepState {
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
