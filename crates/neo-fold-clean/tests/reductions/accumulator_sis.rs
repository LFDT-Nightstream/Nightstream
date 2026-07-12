//! Native/circuit parity and selective-lowering pins for SIS bulk bindings.

#[path = "../gadgets/checked_program_artifact_support.rs"]
#[allow(dead_code)]
mod checked_program_artifact_support;
#[path = "../system/full_history_manifest_identity_support.rs"]
#[allow(dead_code)]
mod full_history_manifest_identity_support;

use neo_ajtai::commit_row_major_seeded;
use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::builder::RowFamilyRange;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment, lower_field_r1cs,
};
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{
    accumulator_digest, commit_fields, enforce_accumulator_digest, enforce_commit_fields, SisAccumulatorConfig,
    CCS_CLAIM_SIS_CONFIG, CE_CLAIM_SIS_CONFIG, NEBULA_LEAF_SIS_CONFIG, PI_CCS_OUTPUTS_SIS_CONFIG,
    PI_RLC_PROJECTION_SIS_CONFIG, PROTOCOL_BINDING_KAPPA, SIS_DIGEST_COMPRESSION_CONFIG,
};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xA7; 32],
    kappa: 1,
    domain: 0x5349_5354_4553_5431,
};

#[test]
fn seeded_phi81_blocks_change_authoritative_row_identity() {
    fn fixture(seed: [u8; 32]) -> R1csBuilder {
        let mut builder = R1csBuilder::new();
        let fields = builder.alloc_vec(&[F::from_u64(3), F::from_u64(5), F::from_u64(8)]);
        enforce_commit_fields(
            &mut builder,
            SisAccumulatorConfig {
                seed,
                kappa: 1,
                domain: 0x5345_4544_5F49_4454,
            },
            &fields,
        )
        .expect("seeded Phi81 fixture");
        assert!(builder.is_satisfied());
        builder
    }

    let left = fixture([0x51; 32]);
    let right = fixture([0xA2; 32]);
    assert_eq!(
        left.sparse_triplets(),
        right.sparse_triplets(),
        "the legacy sparse-only surface erases the seeded A coefficient source"
    );
    assert_eq!(left.seeded_phi81_a_blocks().len(), 1);
    assert_eq!(right.seeded_phi81_a_blocks().len(), 1);

    let left_range = RowFamilyRange {
        name: "seeded-identity-regression",
        row_start: 0,
        row_end: left.rows(),
    };
    let right_range = RowFamilyRange {
        row_end: right.rows(),
        ..left_range
    };
    assert_ne!(
        full_history_manifest_identity_support::range_hash(&left, &left_range),
        full_history_manifest_identity_support::range_hash(&right, &right_range),
        "the authoritative range hash must bind implicit seeded A rows"
    );

    let left_program = checked_program_artifact_support::normalize(&left);
    let right_program = checked_program_artifact_support::normalize(&right);
    assert_ne!(
        left_program.instructions, right_program.instructions,
        "checked-program normalization must materialize implicit seeded A coefficients"
    );
    let seeded_row = left.seeded_phi81_a_blocks()[0].row_start();
    match &left_program.instructions[seeded_row] {
        checked_program_artifact_support::Instruction::Define(checked_program_artifact_support::Definition {
            rhs: checked_program_artifact_support::Rhs::Product(a, b),
            ..
        }) => {
            assert!(!a.is_empty(), "seeded A row must be present");
            assert_eq!(b, &[(0, 1)], "seeded row multiplier");
        }
        checked_program_artifact_support::Instruction::Check(row) => {
            assert!(!row.a.is_empty(), "seeded A row must be present");
            assert_eq!(row.b, [(0, 1)], "seeded row multiplier");
        }
        instruction => panic!("unexpected seeded row normalization: {instruction:?}"),
    }
}

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
        426,
        "three fields, canonical shifted-base-3 trits, borrow witnesses, and one D-wide output"
    );
    assert_eq!(
        builder.rows(),
        428,
        "canonical shifted-base-3 decompositions plus D output equations"
    );
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
fn sis_accumulator_rejects_noncanonical_shifted_base3_opening() {
    let midpoint = F::ORDER_U64 / 2;
    let mut builder = R1csBuilder::new();
    let field = builder.alloc(F::from_u64(midpoint));
    let commitment = enforce_commit_fields(&mut builder, CONFIG, &[field]).expect("SIS commitment circuit");
    assert!(builder.is_satisfied(), "canonical shifted-base-3 opening");

    // If N is the canonical shifted-base-3 representative, N+p encodes the
    // same field residue and still fits in 41 trits for this fixture. Recompute
    // every auxiliary and the Ajtai output so only the terminal borrow remains
    // false. A reconstruction-only circuit accepts this forged opening.
    let modulus = F::ORDER_U64 as u128;
    let shift = (3u128.pow(41) - 1) / 2;
    let canonical_n = (midpoint as u128 + shift) % modulus;
    let mut remaining = canonical_n + modulus;
    assert!(remaining < 3u128.pow(41), "alternate representative fits in 41 trits");
    let alternate: [F; 41] = core::array::from_fn(|_| {
        let trit = remaining % 3;
        remaining /= 3;
        match trit {
            0 => -F::ONE,
            1 => F::ZERO,
            2 => F::ONE,
            _ => unreachable!("base-3 digit"),
        }
    });
    assert_eq!(remaining, 0);

    let digit_columns = builder
        .balanced_ternary_digit_columns(field)
        .expect("recorded balanced-ternary decomposition");
    assert_eq!(digit_columns.len(), 41);
    let negative_start = digit_columns[40] + 1;
    let borrow_start = negative_start + 41;
    for (index, (&column, &digit)) in digit_columns.iter().zip(&alternate).enumerate() {
        builder.tamper_witness(column, digit);
        builder.tamper_witness(negative_start + index, if digit == -F::ONE { F::ONE } else { F::ZERO });
    }

    let mut bound = F::ORDER_U64 - 1;
    let mut borrow = false;
    for (index, &digit) in alternate.iter().enumerate() {
        let trit = if digit == -F::ONE {
            0
        } else if digit == F::ZERO {
            1
        } else {
            2
        };
        let next = trit + u64::from(borrow) > bound % 3;
        bound /= 3;
        if index + 1 < 41 {
            builder.tamper_witness(borrow_start + index, if next { F::ONE } else { F::ZERO });
        } else {
            assert!(next, "N+p must leave a terminal borrow");
        }
        borrow = next;
    }

    let mut message = Mat::zero(D, 1, F::ZERO);
    for (row, &digit) in alternate.iter().enumerate() {
        message[(row, 0)] = digit;
    }
    let forged = commit_row_major_seeded(CONFIG.seed, D, CONFIG.kappa, 1, &message);
    for (wire, value) in commitment.data.iter().zip(forged.data) {
        builder.tamper_witness(wire.col(), value);
    }

    assert!(
        !builder.is_satisfied(),
        "a field residue must have exactly one accepted shifted-base-3 opening"
    );
    assert_eq!(
        builder.first_unsatisfied_row(),
        Some(125),
        "after recomputing every forged auxiliary, only the terminal borrow row must reject"
    );
}

#[test]
fn sis_shifted_base3_boundaries_accept_and_auxiliaries_are_load_bearing() {
    let shift = ((3u128.pow(41) - 1) / 2) as u64;
    let values = [
        0,
        1,
        F::ORDER_U64 - 1,
        F::ORDER_U64 - shift,
        F::ORDER_U64 - 1 - shift,
        F::ORDER_U64 / 2,
        F::ORDER_U64 / 2 + 1,
        0xdead_beef_cafe_babe % F::ORDER_U64,
    ];

    for value in values {
        let mut builder = R1csBuilder::new();
        let field = builder.alloc(F::from_u64(value));
        enforce_commit_fields(&mut builder, CONFIG, &[field]).expect("boundary SIS commitment");
        assert!(
            builder.is_satisfied(),
            "canonical shifted-base-3 opening must accept residue {value}"
        );
    }

    // `x = p-1-M` maps to the exact comparator boundary N=p-1.
    let mut builder = R1csBuilder::new();
    let field = builder.alloc(F::from_u64(F::ORDER_U64 - 1 - shift));
    enforce_commit_fields(&mut builder, CONFIG, &[field]).expect("N=p-1 SIS commitment");
    assert!(builder.is_satisfied(), "N=p-1 boundary");
    let digits = builder
        .balanced_ternary_digit_columns(field)
        .expect("recorded boundary decomposition");
    let negative_start = digits[40] + 1;
    let borrow_start = negative_start + 41;

    for column in negative_start..borrow_start + 40 {
        let original = builder.witness()[column];
        let tampered = if original == F::ZERO { F::ONE } else { F::ZERO };
        builder.tamper_witness(column, tampered);
        assert!(
            !builder.is_satisfied(),
            "canonicality auxiliary column {column} must be load-bearing"
        );
        builder.tamper_witness(column, original);
        assert!(builder.is_satisfied());
    }

    let first_digit = digits[0];
    let original = builder.witness()[first_digit];
    builder.tamper_witness(first_digit, F::from_u64(2));
    assert!(!builder.is_satisfied(), "a trit outside {{-1,0,1}} must be rejected");
    builder.tamper_witness(first_digit, original);
    assert!(builder.is_satisfied());
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
    assert!(
        relation.structure().n < 100_000 && relation.structure().m < 100_000,
        "shifted-base-3 canonicality must not reintroduce the 400k-coordinate lowering blow-up"
    );

    encoded[source_slot.0] = F::from_u64(2);
    assert!(!relation.is_satisfied(&encoded), "a non-unit SIS trit must be rejected");
}
