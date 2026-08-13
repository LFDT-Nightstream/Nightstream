//! Batched R1CS coverage for a grammar-mode trace.

mod common;

use common::grammar_fixture::grammar_lifecycle_setup;

#[test]
fn grammar_trace_satisfies_batched_ccs() {
    let setup = grammar_lifecycle_setup();
    let batch_size = 8;
    let batched = neo_wasm::batch::build_batched_wasm_ccs(batch_size).expect("batched CCS");
    let vm = neo_wasm::WasmVmSpec::default();
    let rows_per_step = vm.core_ccs_spec().structure.n;
    let catalog = vm.constraint_catalog();
    let batches = neo_wasm::batch::batch_count(setup.trace.len(), batch_size);

    for batch in 0..batches {
        let witness = neo_wasm::batch::build_batched_witness(&setup.trace, batch_size, batch);
        batched
            .sparse_r1cs
            .is_satisfied_by(&witness)
            .unwrap_or_else(|error| {
                let row = error
                    .to_string()
                    .split_once("row ")
                    .and_then(|(_, rest)| rest.split_once(|c: char| !c.is_ascii_digit()))
                    .and_then(|(digits, _)| digits.parse::<usize>().ok());
                let owner = row.map(|row| {
                    if row < batch_size * rows_per_step {
                        format!(
                            "step {} constraint {:?}",
                            batch * batch_size + row / rows_per_step,
                            catalog.row_tags.get(row % rows_per_step)
                        )
                    } else {
                        format!("link row {}", row - batch_size * rows_per_step)
                    }
                });
                panic!("batch {batch} rejected: {error} ({owner:?})");
            });
    }
}
