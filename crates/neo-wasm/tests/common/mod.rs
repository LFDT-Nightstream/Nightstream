#![allow(dead_code)]

use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use neo_wasm::layout::COLUMN_SPECS;
use neo_wasm::{
    build_wasm_lookup_binding_layout, builder::build_witness_vector, collect_wasmtime_steps, preload_from_wasmtime_run,
    sanity_check_lookup_row, sanity_check_memory_rows, traces_from_wasmtime_steps, WasmStepTrace, WasmVmSpec,
    WasmtimeTraceRun,
};

pub struct CheckedWasmRun {
    pub wasm: Vec<u8>,
    pub run: WasmtimeTraceRun,
    pub trace: Vec<WasmStepTrace>,
    pub witnesses: Vec<Vec<F>>,
}

pub fn checked_main(wat_src: &str) -> CheckedWasmRun {
    checked_wasm_run(wat_src, "main", &[])
}

pub fn checked_wasm_run(wat_src: &str, export: &str, params: &[i32]) -> CheckedWasmRun {
    let wasm = wat::parse_str(wat_src).expect("valid WAT");
    let run = collect_wasmtime_steps(&wasm, export, params).expect("wasmtime trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");
    let witnesses = sanity_check_trace(&trace, &run);
    ccs_check_trace(&trace);
    CheckedWasmRun {
        wasm,
        run,
        trace,
        witnesses,
    }
}

pub fn sanity_check_trace(trace: &[WasmStepTrace], run: &WasmtimeTraceRun) -> Vec<Vec<F>> {
    let layout = build_wasm_lookup_binding_layout();
    let mut witnesses = Vec::with_capacity(trace.len());
    for row in trace {
        let witness = build_witness_vector(row);
        sanity_check_lookup_row(layout, &witness)
            .unwrap_or_else(|err| panic!("lookup semantics rejected {:?}: {err}", row.opcode));
        witnesses.push(witness);
    }
    let preload = preload_from_wasmtime_run(run, &run.initial_locals);
    sanity_check_memory_rows(layout, &witnesses, &preload)
        .unwrap_or_else(|err| panic!("memory semantics rejected trace: {err}"));
    witnesses
}

pub fn ccs_check_trace(trace: &[WasmStepTrace]) {
    let vm = WasmVmSpec::default();
    let ccs = &vm.core_ccs_spec().structure;
    let catalog = vm.constraint_catalog();
    for (idx, row) in trace.iter().enumerate() {
        let witness = build_witness_vector(row);
        let (x, w) = (&witness[..1], &witness[1..]);
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
