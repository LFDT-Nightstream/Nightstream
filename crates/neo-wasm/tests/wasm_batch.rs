//! Batched WASM relation shape, witness, and continuity tests.

mod common;

use neo_math::F;
use neo_wasm::batch::{batch_count, build_batched_wasm_ccs, build_batched_witness};
use neo_wasm::layout::{COL_LOCALS_FBP_AFTER, COL_PC_BEFORE, COL_SP_BEFORE};
use neo_wasm::{WasmVmSpec, WasmVmStep};
use p3_field::PrimeCharacteristicRing;

const SIMPLE_ADD_WAT: &str = r#"
(module (func (export "main") (result i32)
   i32.const 7
   i32.const 9
   i32.add))
"#;

fn assert_batched_satisfaction(trace: &[WasmVmStep], batch_size: usize) {
    let relation = build_batched_wasm_ccs(batch_size).expect("batched relation");
    for batch in 0..batch_count(trace.len(), batch_size) {
        let witness = build_batched_witness(trace, batch_size, batch);
        relation
            .sparse_r1cs
            .is_satisfied_by(&witness)
            .unwrap_or_else(|error| panic!("batch {batch} rejected: {error}"));
    }
}

#[test]
fn batch_one_matches_the_single_step_relation() {
    let batched = build_batched_wasm_ccs(1).expect("batch one");
    let single = WasmVmSpec::default().core_ccs_spec().clone();
    assert_eq!(batched.sparse_r1cs.m, single.structure.m);
    assert_eq!(batched.sparse_r1cs.n, single.structure.n);
    assert_eq!(batched.sparse_r1cs.m_in, single.m_in);
}

#[test]
fn batch_shape_adds_one_link_row_per_state_pair() {
    let single = build_batched_wasm_ccs(1).expect("single");
    let links = neo_wasm::build_wasm_relation_layout()
        .auxiliary
        .ivc_state_links
        .iter()
        .map(|link| link.column_pairs.len())
        .sum::<usize>();
    for size in [2, 4, 10] {
        let batched = build_batched_wasm_ccs(size).expect("batched");
        assert_eq!(batched.sparse_r1cs.m, size * single.sparse_r1cs.m);
        assert_eq!(batched.sparse_r1cs.n, size * single.sparse_r1cs.n + (size - 1) * links);
    }
}

#[test]
fn valid_and_padded_batches_satisfy_the_relation() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    for size in [1, 2, 3, 5, 7] {
        assert_batched_satisfaction(&checked.trace, size);
    }
}

#[test]
fn initial_digest_is_the_canonical_entry_state_digest() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    let entry = common::single_function_entry_pc(&checked.artifacts);
    let state = neo_wasm::top_level_initial_state(&checked.artifacts.tables, entry);
    assert_eq!(
        neo_wasm::semantic_state_digest(state),
        neo_wasm::top_level_initial_state_digest(&checked.artifacts.tables, entry)
    );
}

#[test]
fn continuity_rows_reject_changed_pc_sp_and_frame_base() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    let size = 2;
    let relation = build_batched_wasm_ccs(size).expect("batched");
    let width = relation.sparse_r1cs.m / size;

    for column in [width + COL_PC_BEFORE, width + COL_SP_BEFORE, COL_LOCALS_FBP_AFTER] {
        let mut witness = build_batched_witness(&checked.trace, size, 0);
        witness[column] += F::ONE;
        relation
            .sparse_r1cs
            .is_satisfied_by(&witness)
            .expect_err("changed linked state must fail");
    }
}

#[test]
fn replicated_steps_use_the_shared_constant_column() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    let size = 2;
    let relation = build_batched_wasm_ccs(size).expect("batched");
    let width = relation.sparse_r1cs.m / size;
    let mut witness = build_batched_witness(&checked.trace, size, 0);
    witness[width] = F::from_u64(7);
    relation
        .sparse_r1cs
        .is_satisfied_by(&witness)
        .expect("the second block-local constant is not referenced");
}
