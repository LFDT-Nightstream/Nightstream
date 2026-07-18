//! Differential and cost-tree tests for exact Pi_RLC projection compaction.
//!
//! Owns: an isolated 31-role/15-pair production-shape fixture, honest inverse
//! parity, forged-assignment rejection, independent trace/source-row mutation
//! rejection, and exact compact leaf counts.
//!
//! Does not own: transcript sampling, the exact-or-bad-root reduction, or
//! permission to trust profiler totals without source replay.
//!
//! Emits constraints: yes, by materializing the production compact lowering.
//!
//! Authority boundary: every test regenerates authoritative source rows. Role
//! metadata is assigned only after synthesis and must pass the same exact
//! manifest gate as production.
//!
//! | Test branch | Mathematical obligation | Expected compact shape |
//! |---|---|---|
//! | evaluation leaves | bind all 34 retained output limbs | 102 rows, 68 carries per identity |
//! | fused terminal leaf | direct 15-pair K identity minus q/Phi and output | 4 rows, 2 carries per identity |
//! | K-product leaves | fully absorbed by the fused terminal identity | zero encoded rows/columns |
//! | complete identity | exact sum of leaves | 5,248 rows / 8,044 columns |

use std::sync::OnceLock;

use neo_fold_clean::engine::r1cs_circuit::ring_action::{
    enforce_beta_ladder, enforce_polynomial_evaluations_at_beta,
    enforce_ring_action_projection_batch_with_rho_evaluations_and_stages, projection_quotient,
    ProjectionIdentityStageLabels,
};
use neo_fold_clean::engine::r1cs_circuit::{
    KVar, PolynomialEvaluationTraceTestMutation, ProjectionIdentityRole, ProjectionIdentityTraceTestMutation,
    R1csBuilder, R1csEncodingTrace, R1csSnapshot, Var,
};
use neo_fold_clean::frontends::f_prime::gadget_native::{
    encode_r1cs_gadget_native, estimate_r1cs_gadget_native, profile_r1cs_gadget_native_stages, GadgetNativeError,
    GadgetNativeStageEstimate, GadgetNativeStageProfile,
};
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage;
use neo_math::ring::D;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

const PAIRS: usize = 15;
const IDENTITIES: usize = 31;
const RETAINED_PER_IDENTITY: usize = 34;
const SYNTHETIC_PER_IDENTITY: usize = 70;
const PRODUCT_ROWS_PER_IDENTITY: usize = 106;
const ENCODED_COLUMNS_PER_IDENTITY: usize = 8_044;
const ENCODED_ROWS_PER_IDENTITY: usize = 5_248;

struct Fixture {
    source: R1csSnapshot,
    trace: R1csEncodingTrace,
    retained_output: usize,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(build_fixture)
}

fn build_fixture() -> Fixture {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.pi_rlc.setup");

    let beta = K::from_coeffs([F::from_u64(7), F::from_u64(11)]);
    let [beta_c0, beta_c1] = beta.as_coeffs();
    let beta = KVar::alloc(&mut builder, beta_c0, beta_c1);
    let powers = enforce_beta_ladder(&mut builder, beta, D);

    let rho_values = (0..PAIRS)
        .map(|pair| std::array::from_fn(|coefficient| F::from_u64((pair as u64 + 3) * (coefficient as u64 + 5) + 1)))
        .collect::<Vec<[F; D]>>();
    let input_values = (0..PAIRS)
        .map(|pair| std::array::from_fn(|coefficient| F::from_u64((pair as u64 + 13) * (coefficient as u64 + 17) + 9)))
        .collect::<Vec<[F; D]>>();
    let rho = rho_values
        .iter()
        .map(|values| alloc_polynomial(&mut builder, values))
        .collect::<Vec<_>>();
    let inputs = input_values
        .iter()
        .map(|values| alloc_polynomial(&mut builder, values))
        .collect::<Vec<_>>();
    let rho_evaluations = enforce_polynomial_evaluations_at_beta(&mut builder, &rho, &powers);

    let native_pairs = rho_values
        .iter()
        .copied()
        .zip(input_values.iter().copied())
        .collect::<Vec<_>>();
    let (output_values, quotient_values) = projection_quotient(&native_pairs);
    let output = alloc_polynomial(&mut builder, &output_values);
    let quotient = alloc_quotient(&mut builder, &quotient_values);
    let pairs = rho
        .iter()
        .zip(&inputs)
        .map(|(rho, input)| (rho, input))
        .collect::<Vec<_>>();

    for identity in 0..IDENTITIES {
        enforce_ring_action_projection_batch_with_rho_evaluations_and_stages(
            &mut builder,
            &powers,
            &rho_evaluations,
            &pairs,
            &output,
            &quotient,
            Some(labels_for_identity(identity)),
        );
    }
    builder.begin_encoding_stage("complete");
    assert!(builder.is_satisfied(), "isolated source projection relation");

    let source = builder.snapshot();
    let mut trace = builder.encoding_trace().clone();
    for (identity, role) in expected_roles().into_iter().enumerate() {
        trace.apply_projection_identity_trace_test_mutation(
            identity,
            ProjectionIdentityTraceTestMutation::Role { role },
        );
    }
    let retained_output =
        trace.polynomial_evaluations()[trace.projection_identities()[0].output_evaluation].output_cols[0];
    Fixture {
        source,
        trace,
        retained_output,
    }
}

fn alloc_polynomial(builder: &mut R1csBuilder, values: &[F; D]) -> [Var; D] {
    builder
        .alloc_vec(values)
        .try_into()
        .expect("ring polynomial width")
}

fn alloc_quotient(
    builder: &mut R1csBuilder,
    values: &[F; neo_fold_clean::engine::r1cs_circuit::ring_action::PROJECTION_QUOTIENT_LEN],
) -> [Var; neo_fold_clean::engine::r1cs_circuit::ring_action::PROJECTION_QUOTIENT_LEN] {
    builder
        .alloc_vec(values)
        .try_into()
        .expect("projection quotient width")
}

fn labels_for_identity(identity: usize) -> ProjectionIdentityStageLabels {
    match identity {
        0..=17 => stage::COMMITMENT_IDENTITY_STAGES,
        18..=22 => stage::X_IDENTITY_STAGES,
        23..=28 => stage::Y_RING_IDENTITY_STAGES,
        29 => stage::Y_ZCOL_LIMB0_IDENTITY_STAGES,
        30 => stage::Y_ZCOL_LIMB1_IDENTITY_STAGES,
        _ => unreachable!("fixed projection role count"),
    }
}

fn expected_roles() -> Vec<ProjectionIdentityRole> {
    let mut roles = Vec::with_capacity(IDENTITIES);
    roles.extend((0..18).map(|lane| ProjectionIdentityRole::CommitmentLane { lane }));
    roles.extend((0..5).map(|column| ProjectionIdentityRole::ActiveXColumn { column }));
    for row in 0..3 {
        roles.extend((0..2).map(|limb| ProjectionIdentityRole::YRingLimb { row, limb }));
    }
    roles.extend((0..2).map(|limb| ProjectionIdentityRole::YZColLimb { limb }));
    roles
}

#[test]
fn exact_projection_compaction_materializes_and_rejects_a_forged_retained_output() {
    let fixture = fixture();
    let estimate = estimate_r1cs_gadget_native(&fixture.source, &fixture.trace, &[])
        .expect("exact projection compaction estimate");
    let mut encoded = encode_r1cs_gadget_native(&fixture.source, &fixture.trace, &[])
        .expect("exact projection compaction materialization");
    assert!(encoded.is_satisfied());
    assert_eq!(encoded.structure.m, estimate.encoded_cols);
    assert_eq!(encoded.structure.n, estimate.encoded_rows);
    assert_eq!(
        encoded.decode_source().expect("exact source inverse"),
        fixture.source.witness()
    );

    let carry_bit = encoded
        .plan
        .first_synthetic_product_sum_field_range()
        .expect("projection carry slot")
        .start;
    encoded.assignment[carry_bit] = F::ONE - encoded.assignment[carry_bit];
    assert!(encoded.first_unsatisfied_row().is_some());

    let mut encoded = encode_r1cs_gadget_native(&fixture.source, &fixture.trace, &[])
        .expect("fresh exact projection compaction materialization");

    let retained_bit = encoded
        .plan
        .encoded_range_for_source_column(fixture.retained_output)
        .expect("retained evaluation output slot")
        .start;
    encoded.assignment[retained_bit] = F::ONE - encoded.assignment[retained_bit];
    assert!(encoded.first_unsatisfied_row().is_some());
    assert!(matches!(
        encoded.decode_source(),
        Err(GadgetNativeError::UnsatisfiedEncoding { .. })
    ));
}

#[test]
fn exact_projection_compaction_rejects_trace_and_source_row_mutations() {
    let fixture = fixture();

    let mut wrong_role = fixture.trace.clone();
    wrong_role.apply_projection_identity_trace_test_mutation(
        0,
        ProjectionIdentityTraceTestMutation::Role {
            role: ProjectionIdentityRole::CommitmentLane { lane: 1 },
        },
    );
    assert!(matches!(
        estimate_r1cs_gadget_native(&fixture.source, &wrong_role, &[]),
        Err(GadgetNativeError::ProjectionIdentityManifest { .. })
    ));

    let mut wrong_equation = fixture.trace.clone();
    let evaluation = wrong_equation.projection_identities()[0]
        .input_evaluations
        .start;
    let column = wrong_equation.polynomial_evaluations()[evaluation].coefficient_cols[0] + 1;
    wrong_equation.apply_polynomial_evaluation_trace_test_mutation(
        evaluation,
        PolynomialEvaluationTraceTestMutation::CoefficientColumn { offset: 0, column },
    );
    assert!(matches!(
        estimate_r1cs_gadget_native(&fixture.source, &wrong_equation, &[]),
        Err(GadgetNativeError::ProjectionIdentityTrace(_))
    ));

    let mut wrong_row = fixture.source.clone();
    let projection = &fixture.trace.projection_identities()[0];
    let row = projection.source_rows.start;
    wrong_row.apply_a_row_test_mutation(row, Var::ONE.col(), F::ONE);
    assert!(matches!(
        estimate_r1cs_gadget_native(&wrong_row, &fixture.trace, &[]),
        Err(GadgetNativeError::ProjectionIdentityTrace(_))
    ));

    let mut wrong_w = fixture.source.clone();
    let first_product = &fixture.trace.k_muls()[projection.pair_products.start];
    wrong_w.apply_a_row_test_mutation(
        first_product.source_rows.start + 3,
        first_product.intermediates[1].col(),
        F::ONE,
    );
    assert!(matches!(
        estimate_r1cs_gadget_native(&wrong_w, &fixture.trace, &[]),
        Err(GadgetNativeError::ProjectionIdentityTrace(_))
    ));

    let mut wrong_final_sign = fixture.source.clone();
    let quotient_phi = &fixture.trace.k_muls()[projection.quotient_phi_product];
    wrong_final_sign.apply_a_row_test_mutation(
        projection.final_limb_rows.start,
        quotient_phi.output[0].col(),
        F::from_u64(2),
    );
    assert!(matches!(
        estimate_r1cs_gadget_native(&wrong_final_sign, &fixture.trace, &[]),
        Err(GadgetNativeError::ProjectionIdentityTrace(_))
    ));
}

#[test]
fn exact_projection_compaction_stage_profile_exposes_evaluation_fusion_and_zero_k_nodes() {
    let fixture = fixture();
    let profile = profile_r1cs_gadget_native_stages(&fixture.source, &fixture.trace, &[])
        .expect("exact projection stage profile");
    let stages = profile.aggregate_by_label();

    assert_family_profile(&stages, 18, stage::COMMITMENT_IDENTITY_STAGES);
    assert_family_profile(&stages, 5, stage::X_IDENTITY_STAGES);
    assert_family_profile(&stages, 6, stage::Y_RING_IDENTITY_STAGES);
    assert_family_profile(&stages, 1, stage::Y_ZCOL_LIMB0_IDENTITY_STAGES);
    assert_family_profile(&stages, 1, stage::Y_ZCOL_LIMB1_IDENTITY_STAGES);

    let y_zcol_parent_stages = labels_as_prefix_totals(&profile, stage::Y_ZCOL_IDENTITY_STAGES);
    assert_family_profile(&y_zcol_parent_stages, 2, stage::Y_ZCOL_IDENTITY_STAGES);

    let labels = [
        stage::COMMITMENT_IDENTITY_STAGES,
        stage::X_IDENTITY_STAGES,
        stage::Y_RING_IDENTITY_STAGES,
        stage::Y_ZCOL_LIMB0_IDENTITY_STAGES,
        stage::Y_ZCOL_LIMB1_IDENTITY_STAGES,
    ];
    let retained = labels
        .iter()
        .flat_map(|labels| {
            [
                labels.input_evaluations,
                labels.output_evaluation,
                labels.quotient_evaluation,
            ]
        })
        .map(|label| stage_at(&stages, label).ordinary_private_field_source_cols)
        .sum::<usize>();
    let synthetic = labels
        .iter()
        .flat_map(|labels| {
            [
                labels.input_evaluations,
                labels.output_evaluation,
                labels.quotient_evaluation,
                labels.final_limb_checks,
            ]
        })
        .map(|label| stage_at(&stages, label).synthetic_product_sum_fields)
        .sum::<usize>();
    let product_rows = labels
        .iter()
        .flat_map(|labels| {
            [
                labels.input_evaluations,
                labels.output_evaluation,
                labels.quotient_evaluation,
                labels.final_limb_checks,
            ]
        })
        .map(|label| stage_at(&stages, label).product_sum_rows)
        .sum::<usize>();
    let encoded_columns = labels
        .iter()
        .flat_map(|labels| {
            [
                labels.input_evaluations,
                labels.output_evaluation,
                labels.quotient_evaluation,
                labels.final_limb_checks,
            ]
        })
        .map(|label| stage_at(&stages, label).encoded_cols)
        .sum::<usize>();
    let encoded_rows = labels
        .iter()
        .flat_map(|labels| {
            [
                labels.input_evaluations,
                labels.output_evaluation,
                labels.quotient_evaluation,
                labels.final_limb_checks,
            ]
        })
        .map(|label| stage_at(&stages, label).encoded_rows)
        .sum::<usize>();

    assert_eq!(retained, IDENTITIES * RETAINED_PER_IDENTITY);
    assert_eq!(synthetic, IDENTITIES * SYNTHETIC_PER_IDENTITY);
    assert_eq!(product_rows, IDENTITIES * PRODUCT_ROWS_PER_IDENTITY);
    assert_eq!(encoded_columns, IDENTITIES * ENCODED_COLUMNS_PER_IDENTITY);
    assert_eq!(encoded_rows, IDENTITIES * ENCODED_ROWS_PER_IDENTITY);
}

fn labels_as_prefix_totals(
    profile: &GadgetNativeStageProfile,
    labels: ProjectionIdentityStageLabels,
) -> Vec<GadgetNativeStageEstimate> {
    [
        labels.input_evaluations,
        labels.rho_times_input,
        labels.output_evaluation,
        labels.quotient_evaluation,
        labels.quotient_times_phi,
        labels.final_limb_checks,
    ]
    .into_iter()
    .map(|label| {
        profile
            .aggregate_prefix(label)
            .expect("identity phase parent")
    })
    .collect()
}

fn assert_family_profile(
    stages: &[GadgetNativeStageEstimate],
    identities: usize,
    labels: ProjectionIdentityStageLabels,
) {
    let inputs = stage_at(stages, labels.input_evaluations);
    assert_eq!(inputs.product_sum_batches, identities);
    assert_eq!(inputs.product_sum_identities, identities * PAIRS * 2);
    assert_eq!(inputs.product_sum_rows, identities * PAIRS * 2 * 3);
    assert_eq!(inputs.synthetic_product_sum_fields, identities * PAIRS * 2 * 2);
    assert_eq!(inputs.ordinary_private_field_source_cols, identities * PAIRS * 2);

    let output = stage_at(stages, labels.output_evaluation);
    assert_evaluation_leaf(output, identities, 106);
    let quotient = stage_at(stages, labels.quotient_evaluation);
    assert_evaluation_leaf(quotient, identities, 104);

    for label in [labels.rho_times_input, labels.quotient_times_phi] {
        let k_products = stage_at(stages, label);
        assert_eq!(k_products.encoded_cols, 0);
        assert_eq!(k_products.encoded_rows, 0);
        assert_eq!(k_products.product_sum_identities, 0);
        assert_eq!(k_products.product_sum_rows, 0);
        assert_eq!(k_products.synthetic_product_sum_fields, 0);
    }

    let final_limb = stage_at(stages, labels.final_limb_checks);
    assert_eq!(final_limb.product_sum_identities, identities * 2);
    assert_eq!(final_limb.product_sum_rows, identities * 4);
    assert_eq!(final_limb.synthetic_product_sum_fields, identities * 2);
    assert_eq!(final_limb.canonical_binary_field_source_cols, 0);
    assert_eq!(final_limb.encoded_cols, identities * 2 * 95);
    assert_eq!(final_limb.encoded_rows, identities * 131);
}

fn assert_evaluation_leaf(stage: &GadgetNativeStageEstimate, identities: usize, removed_columns: usize) {
    assert_eq!(stage.product_sum_identities, identities * 2);
    assert_eq!(stage.product_sum_rows, identities * 6);
    assert_eq!(stage.synthetic_product_sum_fields, identities * 4);
    assert_eq!(stage.ordinary_private_field_source_cols, identities * 2);
    assert_eq!(stage.gadget_derived_source_cols, identities * removed_columns);
    assert_eq!(stage.encoded_cols, identities * 462);
    assert_eq!(stage.encoded_rows, identities * 301);
}

fn stage_at<'a>(stages: &'a [GadgetNativeStageEstimate], label: &str) -> &'a GadgetNativeStageEstimate {
    stages
        .iter()
        .find(|stage| stage.label == label)
        .expect("profile leaf")
}
