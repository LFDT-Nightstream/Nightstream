//! Native/circuit parity and selective-lowering pins for SIS bulk bindings.

use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment, lower_field_r1cs,
};
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{
    accumulator_digest, commit_fields, enforce_accumulator_digest, enforce_commit_fields, SisAccumulatorConfig,
    CCS_CLAIM_SIS_CONFIG, CE_CLAIM_SIS_CONFIG, NEBULA_LEAF_SIS_CONFIG, PI_CCS_OUTPUTS_SIS_CONFIG,
    PI_RLC_PROJECTION_SIS_CONFIG, PROTOCOL_BINDING_KAPPA, SIS_DIGEST_COMPRESSION_CONFIG,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xA7; 32],
    kappa: 1,
    domain: 0x5349_5354_4553_5431,
};

#[test]
fn protocol_binding_maps_match_estimated_two_level_profile() {
    let long_maps = [
        CCS_CLAIM_SIS_CONFIG,
        CE_CLAIM_SIS_CONFIG,
        PI_CCS_OUTPUTS_SIS_CONFIG,
        PI_RLC_PROJECTION_SIS_CONFIG,
        NEBULA_LEAF_SIS_CONFIG,
    ];
    for config in long_maps {
        assert_eq!(config.kappa, PROTOCOL_BINDING_KAPPA);
    }
    assert_eq!(SIS_DIGEST_COMPRESSION_CONFIG.kappa, 1);

    let all_maps = [
        long_maps[0],
        long_maps[1],
        long_maps[2],
        long_maps[3],
        long_maps[4],
        SIS_DIGEST_COMPRESSION_CONFIG,
    ];
    for (index, config) in all_maps.iter().enumerate() {
        assert!(
            all_maps[..index]
                .iter()
                .all(|prior| prior.seed != config.seed && prior.domain != config.domain),
            "every estimated map must have an independent seed and domain"
        );
    }
}

#[test]
fn sis_accumulator_matches_native_and_rejects_tampering() {
    let values = [F::from_u64(3), F::from_u64(F::ORDER_U64 / 2), -F::ONE];
    let native = commit_fields(CONFIG, &values).expect("native SIS accumulator");
    let mut builder = R1csBuilder::new();
    let fields = builder.alloc_vec(&values);
    let commitment = enforce_commit_fields(&mut builder, CONFIG, &fields).expect("SIS accumulator circuit");
    let circuit_data: Vec<F> = commitment
        .data
        .iter()
        .map(|wire| builder.witness()[wire.col()])
        .collect();

    assert_eq!(circuit_data, native.data);
    assert!(builder.is_satisfied());
    assert_eq!(
        builder.cols(),
        306,
        "three fields, balanced trits, check intermediates, and one D-wide output"
    );
    assert_eq!(builder.rows(), 305, "balanced decomposition plus D output equations");
    assert!(
        builder.nonzero_entries() < 10_000,
        "the seeded ring map must not materialize its rotated coefficients"
    );

    let input_col = fields[1].col();
    let input = builder.witness()[input_col];
    builder.tamper_witness(input_col, input + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "input mutation must break canonical recomposition"
    );
    builder.tamper_witness(input_col, input);

    let output_col = commitment.data[0].col();
    let output = builder.witness()[output_col];
    builder.tamper_witness(output_col, output + F::ONE);
    assert!(!builder.is_satisfied(), "output mutation must break the Ajtai equation");
}

#[test]
fn sis_accumulator_hash_then_fs_matches_native_and_exposes_cost() {
    let values = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
    let native = accumulator_digest(CONFIG, &values).expect("native SIS digest");
    assert_ne!(
        native,
        accumulator_digest(CONFIG, &[values[0], values[1], values[2], F::ZERO]).expect("length-separated SIS digest"),
        "the SIS digest must bind the input field count"
    );
    let mut builder = R1csBuilder::new();
    let fields = builder.alloc_vec(&values);
    let wires = enforce_accumulator_digest(&mut builder, CONFIG, &fields).expect("SIS digest circuit");
    let circuit_digest: [F; 4] = std::array::from_fn(|lane| builder.witness()[wires.digest[lane].col()]);

    assert_eq!(circuit_digest, native);
    assert!(builder.is_satisfied());

    for wire in [
        wires.commitment.data[0],
        wires.digest_compression.data[0],
        wires.digest[0],
    ] {
        let original = builder.witness()[wire.col()];
        builder.tamper_witness(wire.col(), original + F::ONE);
        assert!(!builder.is_satisfied(), "every binding layer must be load-bearing");
        builder.tamper_witness(wire.col(), original);
        assert!(builder.is_satisfied());
    }
    eprintln!(
        "SIS accumulator (3 fields, kappa=1): rows={}, cols={}, nnz={}",
        builder.rows(),
        builder.cols(),
        builder.nonzero_entries()
    );
}

#[test]
fn sis_accumulator_reuses_authoritative_trits_in_selective_lowering() {
    let values = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
    let mut builder = R1csBuilder::new();
    let fields = builder.alloc_vec(&values);
    enforce_accumulator_digest(&mut builder, CONFIG, &fields).expect("SIS digest circuit");
    let lowered = lower_field_r1cs(builder, &[]).expect("field lowering");
    let (shape, field_assignment) = lowered.into_parts();
    let first_private_field = shape.m_in;
    let arms = [shape.clone(), shape];
    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&arms, 0, 1, 0).expect("selective low-norm lowering");
    let mut encoded = relation
        .encode(0, &field_assignment)
        .expect("encoded SIS arm");

    assert!(relation.is_satisfied(&encoded));
    let source_slot = relation
        .field_slot(0, first_private_field)
        .expect("SIS source field slot");
    assert_eq!(source_slot.1, 41, "SIS must reuse one balanced-ternary field slot");
    eprintln!(
        "selective SIS accumulator (3 fields, kappa=1): rows={}, committed_coordinates={}",
        relation.structure().n,
        relation.structure().m
    );

    encoded[source_slot.0] = F::from_u64(2);
    assert!(!relation.is_satisfied(&encoded), "a non-unit SIS trit must be rejected");
}
