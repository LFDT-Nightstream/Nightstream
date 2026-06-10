//! Batching tests: verify the batched wasm R1CS satisfies for valid
//! witnesses, rejects inconsistent cross-step states, and round-trips
//! through prove/verify at various batch sizes.

mod common;

use neo_fold_clean::frontends::r1cs_f_prime::{R1csChainBuilder, R1csCompilerError};
use neo_math::F;
use neo_wasm::batch::{batch_count, build_batched_wasm_ccs, build_batched_witness};
use neo_wasm::{prove_batched, verify, WasmStepTrace, WasmVmSpec};
use p3_field::PrimeCharacteristicRing;

const SIMPLE_ADD_WAT: &str = r#"
(module (func (export "main") (result i32)
   i32.const 7
   i32.const 9
   i32.add))
"#;

fn satisfies_batched_ccs(traces: &[WasmStepTrace], batch_size: usize) {
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
fn batched_at_one_matches_single_step_shape() {
    let single = build_batched_wasm_ccs(1).expect("single-step shape via batch");
    let core = WasmVmSpec::default().core_ccs_spec().clone();
    assert_eq!(single.sparse_r1cs.m, core.structure.m, "m must match single-step");
    assert_eq!(
        single.sparse_r1cs.n, core.structure.n,
        "n must match single-step (no link rows at N=1)"
    );
    assert_eq!(single.sparse_r1cs.m_in, core.m_in);
}

#[test]
fn batched_shape_grows_with_batch_size() {
    let single = build_batched_wasm_ccs(1).expect("single");
    let n_links_per_boundary = {
        let layout = neo_wasm::build_wasm_lookup_binding_layout();
        // 1 local-constant link + one per state-continuity column pair.
        1 + layout
            .cross_step_links
            .iter()
            .map(|l| l.column_pairs.len())
            .sum::<usize>()
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
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    for batch_size in [1, 2, 4] {
        satisfies_batched_ccs(&checked.trace, batch_size);
    }
}

#[test]
fn batched_witness_satisfies_batched_ccs_with_padding() {
    // Sizes that don't divide trace_len exercise the padding path.
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    assert_ne!(checked.trace.len() % 3, 0, "padding test needs a non-dividing size");
    for batch_size in [3, 5, 7] {
        satisfies_batched_ccs(&checked.trace, batch_size);
    }
}

#[test]
fn initial_state_digest_covers_all_cross_step_inputs() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    let entry_pc = common::single_function_entry_pc(&checked.artifacts);
    let initial_state = neo_wasm::top_level_initial_state(&checked.artifacts.tables, entry_pc);

    let digest = neo_wasm::initial_semantic_state_digest(initial_state);
    assert_eq!(
        digest,
        neo_wasm::top_level_initial_state_digest(&checked.artifacts.tables, entry_pc)
    );
}

#[test]
fn cross_step_link_rejects_inconsistent_pc() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    let batch_size = 2;
    assert!(checked.trace.len() >= 2, "test needs at least 2 trace rows");

    let batched = build_batched_wasm_ccs(batch_size).expect("batched");
    let mut witness = build_batched_witness(&checked.trace, batch_size, 0);

    // Tamper with step 1's pc_before to break the pc continuity link.
    let m_single = batched.sparse_r1cs.m / batch_size;
    let pc_before_col = {
        let layout = neo_wasm::build_wasm_lookup_binding_layout();
        layout.state.pc_before.0
    };
    witness[m_single + pc_before_col] += F::ONE;

    batched
        .sparse_r1cs
        .is_satisfied_by(&witness)
        .expect_err("batched CCS must reject pc_after[0] != pc_before[1]");
}

#[test]
fn cross_step_link_rejects_inconsistent_locals_fbp() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    let batch_size = 2;
    let batched = build_batched_wasm_ccs(batch_size).expect("batched");
    let mut witness = build_batched_witness(&checked.trace, batch_size, 0);

    let locals_fbp_after_col = {
        let layout = neo_wasm::build_wasm_lookup_binding_layout();
        layout.frame.locals_fbp_after.0
    };
    witness[locals_fbp_after_col] += F::ONE;

    batched
        .sparse_r1cs
        .is_satisfied_by(&witness)
        .expect_err("batched CCS must reject locals_fbp_after[0] != locals_fbp_before[1]");
}

#[test]
fn cross_step_link_rejects_inconsistent_sp() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    let batch_size = 2;
    let batched = build_batched_wasm_ccs(batch_size).expect("batched");
    let mut witness = build_batched_witness(&checked.trace, batch_size, 0);

    let m_single = batched.sparse_r1cs.m / batch_size;
    let sp_before_col = {
        let layout = neo_wasm::build_wasm_lookup_binding_layout();
        layout.state.sp_before.0
    };
    witness[m_single + sp_before_col] += F::ONE;

    batched
        .sparse_r1cs
        .is_satisfied_by(&witness)
        .expect_err("batched CCS must reject sp_after[0] != sp_before[1]");
}

#[test]
fn local_constant_link_rejects_non_one_constant() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    let batch_size = 2;
    let batched = build_batched_wasm_ccs(batch_size).expect("batched");
    let mut witness = build_batched_witness(&checked.trace, batch_size, 0);

    // Replace step 1's local constant with a different value; the link
    // row should catch it.
    let m_single = batched.sparse_r1cs.m / batch_size;
    witness[m_single /* + COL_ONE = 0 */] = F::from_u64(7);

    batched
        .sparse_r1cs
        .is_satisfied_by(&witness)
        .expect_err("local-constant link must reject z[m_single + COL_ONE] != 1");
}

#[test]
fn semantic_state_rejects_rewound_cross_batch_boundary() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    let batch_size = 2;
    assert!(
        batch_count(checked.trace.len(), batch_size) >= 2,
        "test needs at least two batches"
    );
    let digest = common::verifier_initial_state_digest(&checked.artifacts);
    let prep = neo_wasm::preprocess_seeded_batched(&WasmVmSpec::default(), batch_size, digest).expect("prep");
    let mut chain = R1csChainBuilder::new(&prep).expect("chain");

    chain
        .append_assignment(build_batched_witness(&checked.trace, batch_size, 0))
        .expect("first batch");
    let err = chain
        .append_assignment(build_batched_witness(&checked.trace, batch_size, 0))
        .expect_err("rewound second batch must not match the carried output state");

    match err {
        neo_fold_clean::frontends::r1cs_f_prime::Error::Compiler(R1csCompilerError::SemanticStateInputMismatch {
            ..
        }) => {}
        other => panic!("expected SemanticStateInputMismatch, got {other:?}"),
    }
}

#[test]
fn semantic_state_rejects_wrong_initial_state_digest() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    let batch_size = 2;
    let mut digest = common::verifier_initial_state_digest(&checked.artifacts);
    digest[0] ^= 0xA5;
    let prep = neo_wasm::preprocess_seeded_batched(&WasmVmSpec::default(), batch_size, digest).expect("prep");
    let mut chain = R1csChainBuilder::new(&prep).expect("chain");
    let witness = build_batched_witness(&checked.trace, batch_size, 0);

    // The base-step path panics (rather than returning Err) when the
    // trace-derived `_before` digest disagrees with the verifier-baked
    // anchor; match on the encoder's structure-violation message so an
    // unrelated panic upstream of the digest check can't masquerade as
    // a successful rejection.
    let panic_payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| chain.append_assignment(witness)))
        .expect_err("wrong verifier-owned initial state digest must reject the first batch");
    let panic_msg = panic_payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic_payload.downcast_ref::<&'static str>().copied())
        .unwrap_or("<non-string panic>");
    assert!(
        panic_msg.contains("encoded R1CS F' step must satisfy its structure"),
        "expected encoder structure-violation panic, got: {panic_msg}"
    );
}

#[test]
#[ignore = "folding proof; gated by the 5-min test cap"]
fn batched_prove_verify_simple_add() {
    let checked = common::checked_wasm_run(SIMPLE_ADD_WAT, "main", &[]);
    // Cover both dividing (2, 4) and padding-required (3) sizes.
    for batch_size in [2usize, 3, 4] {
        let vm = WasmVmSpec::default();
        let digest = common::verifier_initial_state_digest(&checked.artifacts);
        let prep = neo_wasm::preprocess_seeded_batched(&vm, batch_size, digest).expect("prep");
        let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove");
        verify(&prep, &proof).expect("verify");
    }
}
