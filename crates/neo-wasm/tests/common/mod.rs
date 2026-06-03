#![allow(dead_code)]

use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use neo_wasm::layout::COLUMN_SPECS;
use neo_wasm::{
    build_wasm_lookup_binding_layout, collect_wasmtime_steps, extract_wasm_program_artifacts,
    preload_from_program_artifacts, sanity_check_lookup_row, sanity_check_memory_rows, traces_from_wasmtime_steps,
    witness_builder::build_witness_vector, WasmProgramArtifacts, WasmStepTrace, WasmVmSpec, WasmtimeTraceRun,
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
    CheckedWasmRun {
        wasm,
        artifacts,
        run,
        trace,
        witnesses,
    }
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
                row.opcode, row.pc_before, row.pc_after, row.sp_before, row.sp_after, row.halted
            );
        });
    }
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
