//! Batching tests: verify the batched wasm R1CS satisfies for valid
//! witnesses and rejects inconsistent cross-step states.

mod common;

use neo_math::F;
use neo_wasm::batch::{batch_count, build_batched_wasm_ccs, build_batched_witness};
use neo_wasm::layout::{COL_LOCALS_FBP_AFTER, COL_PC_BEFORE, COL_SP_BEFORE};
use neo_wasm::{build_wasm_relation, WasmVmStep};
use p3_field::PrimeCharacteristicRing;

const SIMPLE_ADD_WAT: &str = r#"
(module (func (export "main") (result i32)
   i32.const 7
   i32.const 9
   i32.add))
"#;

fn satisfies_batched_ccs(traces: &[WasmVmStep], batch_size: usize) {
    let batched = build_batched_wasm_ccs(batch_size).expect("build batched CCS");
    let n_batches = batch_count(traces.len(), batch_size);
    for batch_idx in 0..n_batches {
        let witness = build_batched_witness(traces, batch_size, batch_idx);
        batched
            .sparse_r1cs
            .is_satisfied_by(&witness)
            .unwrap_or_else(|err| panic!("batched CCS rejected witness for batch {batch_idx}: {err}"));
    }
}

#[test]
fn batched_at_one_includes_lookup_constraints() {
    let single = build_batched_wasm_ccs(1).expect("single-step shape via batch");
    let core = build_wasm_relation()
        .expect("valid WASM relation")
        .r1cs()
        .clone();
    assert!(
        single.sparse_r1cs.m > core.structure().m(),
        "lookup advice must be included"
    );
    assert!(
        single.sparse_r1cs.n > core.structure().n(),
        "lookup constraints must be included"
    );
    assert!(single.widths[core.structure().m()..]
        .iter()
        .all(|&width| width == 1));
    assert_eq!(single.sparse_r1cs.m_in, core.public_input_count());
}

#[test]
fn batched_lookup_constraints_reject_forged_xor_outputs() {
    let checked = common::checked_main(
        r#"(module (func (export "main") (result i32)
            i32.const 6 i32.const 3 i32.xor drop
            i64.const 4294967302 i64.const 8589934595 i64.xor drop
            i32.const 0))"#,
    );
    let core = build_wasm_relation().expect("core relation");
    for batch_size in [1, 3] {
        satisfies_batched_ccs(&checked.trace, batch_size);
        let batched = build_batched_wasm_ccs(batch_size).expect("lookup relation");
        for (index, row) in checked
            .trace
            .iter()
            .enumerate()
            .filter(|(_, row)| matches!(row.opcode, neo_wasm::WasmOpcode::I32Xor | neo_wasm::WasmOpcode::I64Xor))
        {
            let mut forged = checked.trace.clone();
            let output = forged[index].stack_write0.as_mut().expect("XOR output");
            if row.opcode == neo_wasm::WasmOpcode::I64Xor {
                *output.value_hi.as_mut().expect("i64 high limb") ^= 1;
            } else {
                output.value_lo ^= 1;
            }
            // Rebuild both the VM columns and lookup advice from the forged
            // result. Range checks and the core's output bindings still hold.
            let base = neo_wasm::build_witness_vector(&forged[index]);
            let m_in = core.r1cs().public_input_count();
            neo_ccs::check_ccs_rowwise_zero(core.r1cs().structure(), &base[..m_in], &base[m_in..])
                .expect("the core alone does not constrain XOR semantics");
            let witness = build_batched_witness(&forged, batch_size, index / batch_size);
            batched
                .sparse_r1cs
                .is_satisfied_by(&witness)
                .expect_err("lookup rows must reject a forged XOR result");
        }
    }
}

#[test]
fn batched_shape_grows_with_batch_size() {
    let single = build_batched_wasm_ccs(1).expect("single");
    let n_links_per_boundary = {
        let layout = neo_wasm::build_wasm_relation_layout();
        // One link row per state-continuity column pair.
        layout.auxiliary.continuity.link_count()
    };
    for n in [2usize, 4, 10] {
        let batched = build_batched_wasm_ccs(n).expect("batched");
        assert_eq!(batched.sparse_r1cs.m, n * single.sparse_r1cs.m);
        let expected_n = n * single.sparse_r1cs.n + (n - 1) * n_links_per_boundary;
        assert_eq!(
            batched.sparse_r1cs.n, expected_n,
            "n at batch_size={n} should be {expected_n}"
        );
        assert_eq!(batched.widths.len(), n * single.widths.len());
    }
}

#[test]
fn batched_witness_satisfies_batched_ccs_at_dividing_sizes() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main");
    for batch_size in [1, 2, 4] {
        satisfies_batched_ccs(&checked.trace, batch_size);
    }
}

#[test]
fn batched_witness_satisfies_batched_ccs_with_padding() {
    // Sizes that don't divide trace_len exercise the padding path.
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main");
    assert_ne!(checked.trace.len() % 3, 0, "padding test needs a non-dividing size");
    for batch_size in [3, 5, 7] {
        satisfies_batched_ccs(&checked.trace, batch_size);
    }
}

#[test]
fn cross_step_link_rejects_inconsistent_pc() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main");
    let batch_size = 2;
    assert!(checked.trace.len() >= 2, "test needs at least 2 trace rows");

    let batched = build_batched_wasm_ccs(batch_size).expect("batched");
    let mut witness = build_batched_witness(&checked.trace, batch_size, 0);

    // Tamper with step 1's pc_before to break the pc continuity link.
    let m_single = batched.sparse_r1cs.m / batch_size;
    witness[m_single + COL_PC_BEFORE] += F::ONE;

    batched
        .sparse_r1cs
        .is_satisfied_by(&witness)
        .expect_err("batched CCS must reject pc_after[0] != pc_before[1]");
}

#[test]
fn cross_step_link_rejects_inconsistent_locals_fbp() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main");
    let batch_size = 2;
    let batched = build_batched_wasm_ccs(batch_size).expect("batched");
    let mut witness = build_batched_witness(&checked.trace, batch_size, 0);

    witness[COL_LOCALS_FBP_AFTER] += F::ONE;

    batched
        .sparse_r1cs
        .is_satisfied_by(&witness)
        .expect_err("batched CCS must reject locals_fbp_after[0] != locals_fbp_before[1]");
}

#[test]
fn cross_step_link_rejects_inconsistent_sp() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main");
    let batch_size = 2;
    let batched = build_batched_wasm_ccs(batch_size).expect("batched");
    let mut witness = build_batched_witness(&checked.trace, batch_size, 0);

    let m_single = batched.sparse_r1cs.m / batch_size;
    witness[m_single + COL_SP_BEFORE] += F::ONE;

    batched
        .sparse_r1cs
        .is_satisfied_by(&witness)
        .expect_err("batched CCS must reject sp_after[0] != sp_before[1]");
}

// TODO: the intra-batch 1-constant columns are currently ignored and
// replaced with the global one at 0. The cost is minimal, but for the sake of
// cleanliness we should probably remove them.
//
// Batching is currently done in a straight-forward naive way, so maybe batch
// that with other improvements like inlining the duplicated variables (using
// the output column of the previous step directly rather than the input column
// that is equal to it).
#[test]
fn block_local_constant_is_an_unreferenced_dont_care() {
    // Replicated rows must read the shared global `COL_ONE`, not block-local
    // constant slots after block 0.
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main");
    let batch_size = 2;
    let batched = build_batched_wasm_ccs(batch_size).expect("batched");
    let mut witness = build_batched_witness(&checked.trace, batch_size, 0);

    let m_single = batched.sparse_r1cs.m / batch_size;
    witness[m_single /* + COL_ONE = 0 */] = F::from_u64(7);

    batched
        .sparse_r1cs
        .is_satisfied_by(&witness)
        .expect("block-local constant slots after block 0 must be unreferenced");
}
