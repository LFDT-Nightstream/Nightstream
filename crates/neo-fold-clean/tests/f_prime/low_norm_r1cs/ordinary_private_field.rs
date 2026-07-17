//! Exact Rust side of the ordinary-private 41-coordinate contract.
//!
//! Owns: boundary fixtures, width floor, overflow regression, role separation,
//! local alphabet tamper rejection, estimator/materializer/profile parity, and
//! selector-formula ownership.
//!
//! Does not own: fresh CCS verifier authority or permission to remove the
//! local centered-unit rows.
//!
//! | Family | Obligation | Expected result |
//! |---|---|---|
//! | finite encoding | Lean `targetValue` / `encodeDigit` at `0,1,2,p-1` | exact 41 digits and source round trip |
//! | width | `3^40 < p < 3^41` | 41 is minimal for a three-symbol alphabet |
//! | local rows | every ordinary digit satisfies `d^3-d=0` | any digit set to `2` is rejected |
//! | role separation | ordinary / canonical-u64 / SIS | distinct materializers and cost families |
//! | selector formula | one inactive weighted decode binding per word | 41 local centered rows remain represented |
//! | fresh assignment | encoded `z` public split and `Z[rho, block] = z[block*D+rho]` | exact outer CCS boundary |
//! | adversarial index | coordinate `D` set to `2` | rejected at the same encoded index |

use neo_ajtai::{has_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule};
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::f_prime::gadget_native::{
    encode_ordinary_private_field, encode_r1cs_gadget_native, estimate_r1cs_gadget_native,
    estimate_selector_gated_r1cs_gadget_native, profile_r1cs_gadget_native_stages, GadgetNativeCenteredFamily,
    GadgetNativeError, GadgetNativePlanTestMutation, GadgetNativeSourceRole, ORDINARY_PRIVATE_DIGITS,
    ORDINARY_PRIVATE_RADIX_40, ORDINARY_PRIVATE_RADIX_41, ORDINARY_PRIVATE_SHIFT,
};
use neo_fold_clean::paper::relations::RelationError;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn expected_digits(value: u64) -> [F; ORDINARY_PRIVATE_DIGITS] {
    let mut target = (u128::from(value) + ORDINARY_PRIVATE_SHIFT) % u128::from(F::ORDER_U64);
    std::array::from_fn(|_| {
        let trit = target % 3;
        target /= 3;
        match trit {
            0 => -F::ONE,
            1 => F::ZERO,
            2 => F::ONE,
            _ => unreachable!(),
        }
    })
}

fn ordinary_fixture(
    values: &[u64],
) -> (
    neo_fold_clean::engine::r1cs_circuit::R1csSnapshot,
    neo_fold_clean::engine::r1cs_circuit::R1csEncodingTrace,
    Vec<usize>,
) {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let mut columns = Vec::new();
    for (index, &value) in values.iter().enumerate() {
        builder.begin_encoding_stage(match index {
            0 => "test.ordinary.first",
            _ => "test.ordinary.rest",
        });
        columns.push(builder.alloc(F::from_u64(value)).col());
    }
    builder.begin_encoding_stage("complete");
    (builder.snapshot(), builder.encoding_trace().clone(), columns)
}

fn fresh_assignment_fixture() -> neo_fold_clean::frontends::f_prime::gadget_native::EncodedGadgetNativeR1cs {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.fresh.public");
    let public_bit = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public_bit);
    builder.begin_encoding_stage("test.fresh.private");
    builder.alloc(F::from_u64(0x1234_5678));
    builder.alloc(F::from_u64(0x9abc_def0));
    builder.begin_encoding_stage("complete");

    encode_r1cs_gadget_native(&builder.snapshot(), builder.encoding_trace(), &[public_bit.col()])
        .expect("fresh assignment fixture")
}

fn fresh_assignment_commitment(
    encoded: &neo_fold_clean::frontends::f_prime::gadget_native::EncodedGadgetNativeR1cs,
) -> (neo_fold_clean::Params, AjtaiSModule) {
    let params = neo_fold_clean::config::ccs_params(
        encoded.structure.n,
        encoded.structure.m,
        encoded.structure.t(),
        encoded.structure.max_degree(),
    )
    .expect("shape-compatible SuperNeo params");
    let columns = encoded.assignment.len().div_ceil(D);
    if !has_global_pp_for_dims(D, columns) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x4652_4553_485f_5a31u64.to_le_bytes());
        set_global_pp_seeded(D, params.kappa() as usize, columns, seed).expect("Ajtai setup");
    }
    let log = AjtaiSModule::from_global_for_dims(D, columns).expect("Ajtai module");
    (params, log)
}

#[test]
fn ordinary_private_matches_lean_boundaries_and_uses_u128_shift() {
    let values = [0, 1, 2, F::ORDER_U64 - 1];
    let (source, trace, columns) = ordinary_fixture(&values);
    let estimate = estimate_r1cs_gadget_native(&source, &trace, &[]).expect("ordinary estimate");
    let encoded = encode_r1cs_gadget_native(&source, &trace, &[]).expect("ordinary materialization");

    assert!(u128::from(F::ORDER_U64 - 1) + ORDINARY_PRIVATE_SHIFT > u128::from(u64::MAX));
    assert_eq!(estimate.ordinary_private_field_source_cols, values.len());
    assert_eq!(estimate.canonical_binary_field_source_cols, 0);
    assert_eq!(
        estimate.ordinary_private_encoded_cols,
        values.len() * ORDINARY_PRIVATE_DIGITS
    );
    assert_eq!(estimate.encoded_cols, encoded.structure.m);
    assert_eq!(estimate.encoded_rows, encoded.structure.n);
    assert!(encoded.is_satisfied());
    assert_eq!(
        encoded
            .decode_source()
            .expect("exact source reconstruction"),
        source.witness()
    );

    for (&value, &source_column) in values.iter().zip(&columns) {
        let expected = expected_digits(value);
        assert_eq!(encode_ordinary_private_field(F::from_u64(value)), expected);
        let range = encoded
            .plan
            .encoded_range_for_source_column(source_column)
            .expect("ordinary encoded range");
        assert_eq!(range.len(), ORDINARY_PRIVATE_DIGITS);
        assert_eq!(&encoded.assignment[range], &expected);
        assert_eq!(
            encoded.plan.source_role_for_column(source_column),
            Some(GadgetNativeSourceRole::OrdinaryPrivateField)
        );
    }
}

#[test]
fn three_symbol_width_floor_is_exact() {
    assert_eq!(ORDINARY_PRIVATE_DIGITS, 41);
    assert_eq!(ORDINARY_PRIVATE_RADIX_40, 3u128.pow(40));
    assert_eq!(ORDINARY_PRIVATE_RADIX_41, 3u128.pow(41));
    assert!(ORDINARY_PRIVATE_RADIX_40 < u128::from(F::ORDER_U64));
    assert!(u128::from(F::ORDER_U64) < ORDINARY_PRIVATE_RADIX_41);
    assert_eq!(ORDINARY_PRIVATE_SHIFT, (ORDINARY_PRIVATE_RADIX_41 - 1) / 2);
}

#[test]
fn local_centered_rows_and_materialization_kind_fail_closed() {
    let (source, trace, columns) = ordinary_fixture(&[F::ORDER_U64 - 1]);
    let encoded = encode_r1cs_gadget_native(&source, &trace, &[]).expect("ordinary materialization");
    let range = encoded
        .plan
        .encoded_range_for_source_column(columns[0])
        .expect("ordinary range");
    assert_eq!(
        encoded
            .plan
            .coordinate_gate_schedule()
            .centered_pairing_for(GadgetNativeCenteredFamily::OrdinaryPrivateField)
            .coordinates,
        ORDINARY_PRIVATE_DIGITS
    );

    let mut digit_tamper = encoded.clone();
    digit_tamper.assignment[range.start] = F::from_u64(2);
    assert!(!digit_tamper.is_satisfied());
    assert!(matches!(
        digit_tamper.decode_source(),
        Err(GadgetNativeError::UnsatisfiedEncoding { .. })
    ));

    let mut kind_tamper = encoded;
    kind_tamper
        .plan
        .apply_test_mutation(GadgetNativePlanTestMutation::OrdinaryAsSis {
            source_column: columns[0],
        });
    assert!(matches!(
        kind_tamper.plan.validate_materialization_for_test(),
        Err(GadgetNativeError::SourceMaterializationMismatch { column }) if column == columns[0]
    ));
}

#[test]
fn fresh_ccs_instance_preserves_public_split_and_exact_packed_indices() {
    let encoded = fresh_assignment_fixture();
    let (params, log) = fresh_assignment_commitment(&encoded);
    let instance = encoded
        .to_fresh_ccs_instance(&params, &log)
        .expect("all encoded coordinates satisfy b=2");

    assert!(encoded.assignment.len() > D + 1, "fixture must cross one ring block");
    assert_eq!(instance.claim.m_in, encoded.plan.public_input_len());
    assert_eq!(instance.claim.x, encoded.public_input());
    assert_eq!(instance.witness.Z.rows(), D);
    assert_eq!(instance.witness.Z.cols(), encoded.assignment.len().div_ceil(D));

    for (index, &coordinate) in encoded.assignment.iter().enumerate() {
        assert_eq!(
            instance.witness.Z[(index % D, index / D)],
            coordinate,
            "packed coordinate {index}"
        );
    }
    for index in encoded.assignment.len()..instance.witness.Z.rows() * instance.witness.Z.cols() {
        assert_eq!(
            instance.witness.Z[(index % D, index / D)],
            F::ZERO,
            "padding coordinate {index}"
        );
    }
    assert_eq!(
        instance
            .witness
            .private_values(instance.claim.m_in, encoded.assignment.len())
            .expect("packed private suffix")
            .as_ref(),
        encoded.private_witness()
    );
}

#[test]
fn fresh_ccs_instance_reports_coordinate_two_at_exact_encoded_index() {
    let mut encoded = fresh_assignment_fixture();
    let (params, log) = fresh_assignment_commitment(&encoded);
    encoded.assignment[D] = F::from_u64(2);

    assert!(matches!(
        encoded.to_fresh_ccs_instance(&params, &log),
        Err(RelationError::NormBoundViolated { idx, b: 2 }) if idx == D
    ));
}

#[test]
fn direct_canonical_u64_and_sis_materializers_remain_separate() {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.direct_u64");
    let direct = builder.alloc(F::from_u64(0x1234_5678_9abc_def0));
    let _ = decompose_var_to_u64_bits(&mut builder, direct);
    builder.begin_encoding_stage("complete");
    let direct_source = builder.snapshot();
    let direct_trace = builder.encoding_trace();
    let direct_estimate = estimate_r1cs_gadget_native(&direct_source, direct_trace, &[]).expect("direct estimate");
    let direct_encoded = encode_r1cs_gadget_native(&direct_source, direct_trace, &[]).expect("direct lowering");
    assert_eq!(direct_estimate.canonical_binary_field_source_cols, 1);
    assert_eq!(
        direct_estimate.ordinary_private_field_source_cols, 1,
        "the decomposition inverse remains an ordinary private scalar; the traced field itself must not"
    );
    assert_eq!(
        direct_encoded.plan.source_role_for_column(direct.col()),
        Some(GadgetNativeSourceRole::CanonicalU64)
    );
    assert_eq!(
        direct_encoded
            .plan
            .encoded_range_for_source_column(direct.col())
            .expect("direct range")
            .len(),
        64
    );

    let (sis_source, sis_trace, sis_field) = super::balanced_ternary_relation(F::from_u64(19));
    let sis_estimate = estimate_r1cs_gadget_native(&sis_source, &sis_trace, &[]).expect("SIS estimate");
    let sis_encoded = encode_r1cs_gadget_native(&sis_source, &sis_trace, &[]).expect("SIS lowering");
    assert!(
        sis_estimate.ordinary_private_field_source_cols > 0,
        "untraced SIS commitment outputs are independent ordinary scalars"
    );
    assert_eq!(sis_estimate.balanced_ternary_field_source_cols, 1);
    assert_eq!(
        sis_encoded.plan.source_role_for_column(sis_field),
        Some(GadgetNativeSourceRole::SisOpening)
    );
    assert_eq!(
        sis_encoded
            .plan
            .coordinate_gate_schedule()
            .centered_pairing_for(GadgetNativeCenteredFamily::SisOpening)
            .coordinates,
        41
    );
}

#[test]
fn stage_profile_and_selector_formula_reconcile_ordinary_costs() {
    let (base_source, base_trace, _) = ordinary_fixture(&[0]);
    let (recursive_source, recursive_trace, _) = ordinary_fixture(&[1, 2]);
    let base = estimate_r1cs_gadget_native(&base_source, &base_trace, &[]).expect("base estimate");
    let base_profile = profile_r1cs_gadget_native_stages(&base_source, &base_trace, &[]).expect("base profile");
    assert_eq!(base_profile.total, base);
    assert_eq!(
        base_profile
            .stages
            .iter()
            .map(|stage| stage.ordinary_private_field_source_cols)
            .sum::<usize>(),
        1
    );
    assert_eq!(
        base_profile
            .stages
            .iter()
            .map(|stage| stage.ordinary_private_centered_pairing.total_rows())
            .sum::<usize>(),
        base.ordinary_private_centered_pairing.total_rows()
    );

    let fixed = estimate_selector_gated_r1cs_gadget_native(
        &base_source,
        &base_trace,
        &[],
        &recursive_source,
        &recursive_trace,
        &[],
    )
    .expect("selector formula");
    assert_eq!(fixed.ordinary_private_field_slots, 3);
    assert_eq!(fixed.ordinary_private_coordinates, 3 * ORDINARY_PRIVATE_DIGITS);
    assert_eq!(fixed.ordinary_private_inactive_binding_rows, 3);
    assert_eq!(
        fixed.ordinary_private_centered_pairing.coordinates,
        3 * ORDINARY_PRIVATE_DIGITS
    );
    assert_eq!(
        fixed.ordinary_private_centered_pairing.total_rows(),
        (3 * ORDINARY_PRIVATE_DIGITS).div_ceil(2)
    );
}
