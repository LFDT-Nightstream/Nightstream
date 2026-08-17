use std::collections::BTreeMap;

use neo_fold_clean::config::{BIG_B, B_BASE, K_RHO};
use neo_fold_clean::frontends::nebula::f_prime::{
    production_pi_rlc_family_body_compiler_audit, production_pi_rlc_family_body_decoder_runs,
    production_pi_rlc_family_body_low_norm_shape_audit, production_pi_rlc_family_body_row_ledger,
    NebulaFPrimePiRlcBodyRewriteKind,
};

#[test]
fn production_streaming_profile_uses_nightstream_goldilocks_k16() {
    assert_eq!(B_BASE, 2);
    assert_eq!(K_RHO, 16);
    assert_eq!(BIG_B, 1 << 16);
}

#[test]
#[ignore = "exact Nightstream Goldilocks k_rho=16 PiRLC body shape audit"]
fn production_pi_rlc_body_b2_k16_shape_fits_joint_domain() {
    let shape = production_pi_rlc_family_body_low_norm_shape_audit().expect("Nightstream k_rho=16 PiRLC body shape");
    eprintln!(
        "PiRLC Nightstream k_rho=16 body: b={}, rows={}, columns={}, public={}, coordinates={}",
        shape.norm_base, shape.rows, shape.columns, shape.public_columns, shape.total_coordinates,
    );
    assert_eq!(shape.norm_base, 2);
    assert_eq!(shape.public_columns, 648);
    assert!(shape.rows <= 1 << 24, "PiRLC rows exceed the joint domain");
    assert!(shape.columns <= 1 << 24, "PiRLC columns exceed the joint domain");
}

#[test]
#[ignore = "exact Nightstream Goldilocks k_rho=16 PiRLC compiler ledger"]
fn production_pi_rlc_body_b2_k16_compiler_ledger_is_complete() {
    let ledger = production_pi_rlc_family_body_row_ledger().expect("Nightstream k_rho=16 PiRLC compiler ledger");
    eprintln!(
        "PiRLC Nightstream k_rho=16 ledger: rows={}, columns={}, source_rows={:?}, rewrites={}, fixed_runs={}, retained_runs={}, rewrite_batches={}",
        ledger.rows(),
        ledger.columns(),
        ledger.source_rows(),
        ledger.rewrite_count(),
        ledger.fixed_runs().len(),
        ledger.retained_runs().len(),
        ledger.rewrite_batches().len(),
    );
    assert_eq!(ledger.rows(), 491_046);
    assert_eq!(ledger.columns(), 8_858_862);
    assert_eq!(ledger.source_rows(), [1_300_897, 1_302_097]);
    assert_eq!(ledger.rewrite_count(), 14_638);
    assert_eq!(ledger.fixed_runs().len(), 8);
    assert_eq!(ledger.retained_runs().len(), 22);
    assert_eq!(ledger.linear_definition_counts(), [4_520, 4_520]);
    assert_eq!(ledger.rewrite_batches().len(), 40);

    let mut rewrites = BTreeMap::<&str, (usize, usize)>::new();
    for batch in ledger.rewrite_batches() {
        let name = match batch.kind() {
            NebulaFPrimePiRlcBodyRewriteKind::Poseidon2 => "poseidon2",
            NebulaFPrimePiRlcBodyRewriteKind::ShiftedTernaryCanonical => "shifted_ternary_canonical",
            NebulaFPrimePiRlcBodyRewriteKind::LinearDefinition => {
                panic!("linear definitions must use complement ownership")
            }
        };
        let entry = rewrites.entry(name).or_default();
        entry.0 += 1;
        entry.1 += batch.count();
    }
    eprintln!("PiRLC Nightstream k_rho=16 rewrite batches: {rewrites:?}");
}

#[test]
#[ignore = "exact Nightstream Goldilocks k_rho=16 PiRLC normalized layout audit"]
fn production_pi_rlc_body_b2_k16_layout_is_exact() {
    let compiler = production_pi_rlc_family_body_compiler_audit().expect("Nightstream k_rho=16 PiRLC compiler audit");
    eprintln!("PiRLC Nightstream k_rho=16 layout: {:#?}", compiler.layout());
    for (arm, mapping) in compiler.rows().arms().iter().enumerate() {
        for source_row in [0, 49_626, 49_667, 163_501, 163_609] {
            let run = mapping
                .source_runs()
                .iter()
                .find(|run| run.source_rows().contains(&source_row))
                .expect("source row owner");
            eprintln!(
                "arm={arm} source_row={source_row} source_run={:?} emitted_start={:?}",
                run.source_rows(),
                run.emitted_start(),
            );
        }
        let canonical = compiler
            .rows()
            .rewrites()
            .iter()
            .find(|rewrite| {
                rewrite.arm() == arm
                    && rewrite
                        .source_rows()
                        .iter()
                        .any(|rows| rows.contains(&49_667))
            })
            .expect("first canonical rewrite");
        eprintln!(
            "arm={arm} first_canonical_source={:?} first_canonical_emitted={:?}",
            canonical.source_rows(),
            canonical.emitted_rows(),
        );
    }

    let decoders = production_pi_rlc_family_body_decoder_runs().expect("Nightstream k_rho=16 PiRLC decoders");
    for decoder in decoders {
        eprintln!("arm={} final_columns={}", decoder.arm(), decoder.final_columns());
        for source_column in [
            641, 1_559, 2_477, 2_531, 52_103, 52_144, 52_226, 164_140, 164_142, 164_250, 164_358, 164_466, 165_384,
            166_302, 166_303,
        ] {
            let resolution = decoder
                .runs()
                .iter()
                .find_map(|run| run.resolution_at(source_column))
                .expect("source column resolution");
            eprintln!(
                "arm={} source_column={source_column} resolution={resolution:?}",
                decoder.arm()
            );
        }
    }
}
