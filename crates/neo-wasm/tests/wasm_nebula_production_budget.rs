//! Production-shape width gate for the authoritative WASM + Nebula relation.

mod common;

use std::time::Instant;

use neo_fold_clean::frontends::nebula::f_prime::ROAD_A_COMMITTED_BIT_BUDGET;
use neo_fold_clean::{config, Params};
use neo_math::D;

#[test]
fn wasm_nebula_relation_stays_within_production_budget() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const 7))"#,
    );
    let entry_pc = common::single_function_entry_pc(&checked.artifacts);
    let params = Params::for_ccs_shape_with(
        ROAD_A_COMMITTED_BIT_BUDGET,
        13,
        8,
        config::MIN_EFFECTIVE_LAMBDA,
        config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("production WASM parameters");
    let started = Instant::now();
    let prep = neo_wasm::nebula::preprocess_seeded(
        params,
        neo_wasm::WasmNebulaProfile::production(),
        &checked.artifacts,
        &checked.run.initial_locals,
        entry_pc,
        0x57a5_7000,
    )
    .expect("production WASM + Nebula fixed point");
    let structure = prep.inner().relation().structure();
    println!(
        "WASM + Nebula production relation: rows={} columns={} elapsed={:.2?}",
        structure.n,
        structure.m,
        started.elapsed()
    );
    assert!(structure.m <= ROAD_A_COMMITTED_BIT_BUDGET);
    assert_eq!((structure.n, structure.m), (3_260_306, 15_839_550));
    assert!(
        structure.n < structure.m,
        "SplitNc production relation must remain rectangular"
    );
    assert_eq!(structure.t(), 13, "production matrix count drifted");
    assert_eq!(structure.max_degree(), 8, "production CCS degree drifted");
    let extension = prep
        .inner()
        .prep
        .params
        .validate_ccs_shape(structure.n.max(structure.m), structure.t(), structure.max_degree())
        .expect("production parameters must cover the emitted CCS shape");
    assert!(
        extension.slack_bits >= config::EXTENSION_SAFETY_MARGIN_BITS as i32,
        "production WASM relation has {} extension-safety bits, expected at least {}",
        extension.slack_bits,
        config::EXTENSION_SAFETY_MARGIN_BITS,
    );

    let plan = prep.inner().plan();
    let budget = plan.error_budget();
    let params = &prep.inner().prep.params;
    let d4_factor = params
        .ccs_soundness_factor(structure.n.max(structure.m), structure.t(), structure.max_degree())
        .expect("exact SuperNeo D.4 soundness factor");
    let q_h = 2f64.powi(budget.max_fs_query_bits as i32);
    let n_seg = plan.params().seg_max as f64;
    let n_f = n_seg * plan.params().steps_per_segment() as f64;
    let log2_k = 2.0 * (params.q() as f64).log2();
    let fold_inputs = 1 + params.k_rho() as usize;
    let projection_pairs = fold_inputs * (4 * params.kappa() as usize + 46 + 2 * 15 + 2);
    let pipeline_bits = log2_k - (q_h * n_f * d4_factor as f64).log2();
    let projection_bits = log2_k - (q_h * n_f * projection_pairs as f64 * (2 * D - 2) as f64).log2();
    let fingerprint_bits = log2_k - (q_h * n_seg * budget.m_seg as f64).log2();
    let challenge_set_bits = D as f64 * 5f64.log2();
    let mixing_bits = challenge_set_bits - (q_h * n_f * fold_inputs as f64).log2();
    let term_bits = [
        pipeline_bits,
        projection_bits,
        fingerprint_bits,
        mixing_bits,
        100.0,
        160.0,
        128.0,
    ];
    let end_to_end_bits = -(term_bits.iter().map(|bits| 2f64.powf(-bits)).sum::<f64>()).log2();
    println!(
        "WASM + Nebula security: D.4={d4_factor} pipe={pipeline_bits:.2} projection={projection_bits:.2} fingerprint={fingerprint_bits:.2} mixing={mixing_bits:.2} total={end_to_end_bits:.2}"
    );
    assert!(
        end_to_end_bits >= budget.end_to_end_target_bits as f64,
        "maximum-chain WASM security is {end_to_end_bits:.2} bits, below the declared {}-bit target",
        budget.end_to_end_target_bits,
    );
    assert_eq!(prep.inner().prep.params.k_rho(), 14);
    assert_eq!(prep.profile().batch_size(), 3);
    assert_eq!(prep.profile().memory().b_ops, 192);
    assert_eq!(prep.lookup_auxiliary_columns_per_instruction(), 4_694);
    assert_eq!(prep.total_lookup_auxiliary_columns(), 14_082);
}
