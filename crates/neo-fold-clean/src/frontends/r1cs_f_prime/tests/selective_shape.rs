use super::*;
use std::fmt::Write as _;

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder};
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, SelectiveEmittedRowFamily};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const CENTERED_SEPTENARY_ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/CenteredSeptenary/Generated/RustEncoderCases.lean";

fn centered_septenary_fixture_sources() -> [u64; 4] {
    [0, 1, F::ORDER_U64 / 2, F::ORDER_U64 - 1]
}

fn centered_digit_residue(digit: i8) -> u64 {
    if digit < 0 {
        F::ORDER_U64 - u64::from(digit.unsigned_abs())
    } else {
        digit as u64
    }
}

fn render_centered_septenary_rust_artifact() -> String {
    let sources = centered_septenary_fixture_sources();
    let source_values = sources
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let cases = sources
        .iter()
        .map(|&source| {
            let digits =
                crate::frontends::r1cs_f_prime::ternary_encoding::balanced_septenary_digits(F::from_u64(source), 0)
                    .expect("septenary fixture encoding");
            let digits = digits
                .into_iter()
                .map(centered_digit_residue)
                .map(|digit| digit.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("({source}, [{digits}])")
        })
        .collect::<Vec<_>>()
        .join(",\n    ");

    let mut rendered = String::new();
    rendered.push_str("import Nightstream.Implementation.R1CS.Core.Semantics\n\n");
    rendered.push_str("/-! Generated centered-septenary Rust encoder cases. Do not hand-edit. -/\n\n");
    rendered.push_str("namespace Nightstream.Implementation.R1CS.Artifacts.CenteredSeptenary.RustEncoderCases\n\n");
    rendered.push_str("def schemaVersion : Nat := 1\n");
    writeln!(rendered, "def sources : List Nat := [{source_values}]").expect("render sources");
    writeln!(rendered, "def cases : List (Nat × List Nat) :=\n  [{cases}]").expect("render cases");
    rendered.push_str("\nend Nightstream.Implementation.R1CS.Artifacts.CenteredSeptenary.RustEncoderCases\n");
    rendered
}

fn bit_relation() -> SparseR1cs {
    let mut builder = R1csBuilder::new();
    let public = builder.alloc(F::ONE);
    let first_private = builder.alloc(F::ZERO);
    let second_private = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public);
    enforce_bit(&mut builder, first_private);
    enforce_bit(&mut builder, second_private);
    lower_field_r1cs(builder, &[public])
        .expect("field lowering")
        .into_parts()
        .0
}

#[test]
fn lightweight_shape_matches_complete_audit_exactly() {
    let arm = bit_relation();
    let arms = [arm.clone(), arm.clone(), arm];

    for shared_private_fields in [1, 2] {
        let prepared = super::super::prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
            arms.to_vec(),
            shared_private_fields,
            1,
            D,
            7,
            2,
        )
        .expect("owned prepared relation");
        let summary = prepared.shape_summary();
        let shape =
            audit_multi_branch_selective_low_norm_shape_with_shared_bit_prefix(&arms, shared_private_fields, 1, D, 7)
                .expect("complete shape audit");

        assert!(summary.matches(&shape));
        assert_eq!(summary.rows, shape.compiler_audit.rows().total_rows());
        assert_eq!(summary.columns, summary.total_coordinates.next_multiple_of(D));

        let relation = prepared.finish().expect("emit prepared relation");
        assert_eq!(
            (relation.structure().n, relation.structure().m),
            (summary.rows, summary.columns)
        );
        assert_eq!(relation.public_input_len(), summary.public_input_len);
        assert_eq!(
            relation
                .selective_compiler_audit()
                .expect("compiler audit")
                .width()
                .total_coordinates,
            summary.total_coordinates,
        );

        let mut wrong_summary = summary;
        wrong_summary.total_coordinates += 1;
        assert!(!wrong_summary.matches(&shape));
    }
}

#[test]
fn radix_four_general_fields_use_exact_septenary_words() {
    let mut builder = R1csBuilder::new();
    let public = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public);
    let value = F::from_u64(9_223_372_034_707_292_160);
    let input = builder.alloc(value);
    let square = builder.alloc(value * value);
    builder.enforce(&Lc::from_var(input), &Lc::from_var(input), &Lc::from_var(square));
    let input_col = input.col();
    let (arm, source_assignment) = lower_field_r1cs(builder, &[public])
        .expect("field lowering")
        .into_parts();

    let prepared = super::super::prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        vec![arm.clone(), arm],
        0,
        0,
        D,
        0,
        4,
    )
    .expect("radix-four prepared relation");
    let relation = prepared.finish().expect("radix-four selective relation");
    let snapshot = relation.selective_snapshot().expect("selective snapshot");
    assert_eq!(
        snapshot
            .arm(0)
            .expect("first arm")
            .slot(input_col)
            .expect("input slot")
            .len(),
        23
    );

    let encoded = relation
        .encode(0, &source_assignment)
        .expect("radix-four encoding");
    assert!(relation.is_satisfied(&encoded));
    assert!(
        encoded.iter().any(|&digit| digit == F::from_u64(2)
            || digit == -F::from_u64(2)
            || digit == F::from_u64(3)
            || digit == -F::from_u64(3)),
        "fixture must exercise the larger radix-four alphabet"
    );
    assert_eq!(
        snapshot
            .encode(0, &source_assignment)
            .expect("snapshot encoding"),
        encoded,
        "the audited encoder must replay the exact septenary assignment"
    );
}

#[test]
fn radix_four_domain_audit_partitions_emitted_arm_rows() {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.radix_four.domain");
    let public = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public);
    let private_bit = builder.alloc(F::ONE);
    enforce_bit(&mut builder, private_bit);
    let centered = builder.alloc(F::ONE);
    let centered_square = builder.alloc_mul(&Lc::from_var(centered), &Lc::from_var(centered));
    builder.enforce(
        &Lc::from_var(centered_square),
        &Lc::from_var(centered),
        &Lc::from_var(centered),
    );
    builder.record_centered_unit(centered);
    let centered_other = builder.alloc(-F::ONE);
    let centered_other_square = builder.alloc_mul(&Lc::from_var(centered_other), &Lc::from_var(centered_other));
    builder.enforce(
        &Lc::from_var(centered_other_square),
        &Lc::from_var(centered_other),
        &Lc::from_var(centered_other),
    );
    builder.record_centered_unit(centered_other);
    let centered_tail = builder.alloc(F::ZERO);
    let centered_tail_square = builder.alloc_mul(&Lc::from_var(centered_tail), &Lc::from_var(centered_tail));
    builder.enforce(
        &Lc::from_var(centered_tail_square),
        &Lc::from_var(centered_tail),
        &Lc::from_var(centered_tail),
    );
    builder.record_centered_unit(centered_tail);
    let ordinary = builder.alloc(F::from_u64(9));
    let _ordinary_square = builder.alloc_mul(&Lc::from_var(ordinary), &Lc::from_var(ordinary));
    let (arm, source_assignment) = lower_field_r1cs(builder, &[public])
        .expect("field lowering")
        .into_parts();

    let relation = super::super::prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        vec![arm.clone(), arm],
        0,
        0,
        D,
        0,
        4,
    )
    .expect("radix-four prepared relation")
    .finish()
    .expect("radix-four selective relation");
    let encoded = relation
        .encode(0, &source_assignment)
        .expect("radix-four encoding");
    assert!(relation.is_satisfied(&encoded));

    let snapshot = relation.selective_snapshot().expect("selective snapshot");
    let compiler = relation.selective_compiler_audit().expect("compiler audit");
    for (arm_index, arm) in compiler.width().arms.iter().enumerate() {
        let domain = arm
            .physical_stages
            .iter()
            .fold([0usize; 4], |mut total, stage| {
                total[0] += stage.source_boolean_coordinates;
                total[1] += stage.outer_norm_coordinates;
                total[2] += stage.boolean_domain_rows;
                total[3] += stage.centered_unit_domain_coordinates;
                total
            });
        assert_eq!(domain.iter().sum::<usize>(), arm.branch_coordinates);
        assert!(domain[0] > 0, "fixture must use source Boolean coverage");
        assert!(domain[1] > 0, "fixture must use the outer norm");
        assert!(domain[3] > 0, "fixture must emit a centered-unit row");
        let emitted = compiler
            .rows()
            .emitted_runs()
            .iter()
            .filter(|run| run.family() == SelectiveEmittedRowFamily::ArmDomain && run.arm() == Some(arm_index))
            .map(|run| run.emitted_rows().len())
            .sum::<usize>();
        assert_eq!(domain[2] + domain[3].div_ceil(2), emitted);
    }

    let arm_zero_domain = compiler
        .rows()
        .emitted_runs()
        .iter()
        .find(|run| run.family() == SelectiveEmittedRowFamily::ArmDomain && run.arm() == Some(0))
        .expect("first arm domain");
    let arm_zero_mapping = &compiler.rows().arms()[0];
    let audited_pair_row = arm_zero_mapping
        .centered_domain_pair_row()
        .expect("first arm centered pair row");
    let audited_tail_row = arm_zero_mapping
        .centered_domain_tail_row()
        .expect("first arm centered tail row");
    assert!(arm_zero_domain.emitted_rows().contains(&audited_pair_row));
    assert!(arm_zero_domain.emitted_rows().contains(&audited_tail_row));
    let selector = snapshot.selector_cols()[0];
    let pair = arm_zero_domain
        .emitted_rows()
        .find_map(|row| {
            let artifact = snapshot
                .materialize_row(row)
                .expect("materialized domain row");
            let port = |index| artifact.matrix_row().port(index).expect("selective port");
            assert_eq!(port(GENERAL_SELECTOR)[0].column(), selector);
            assert_eq!(port(EVAL_SELECTOR)[0].column(), selector);
            assert_eq!(port(GENERAL_SELECTOR)[0].coefficient(), F::ONE);
            assert_eq!(port(EVAL_SELECTOR)[0].coefficient(), F::ONE);
            match (port(CENTERED_UNIT).first(), port(A).first()) {
                (Some(left), Some(right)) => Some((row, left.column(), right.column())),
                _ => None,
            }
        })
        .expect("fixture must emit one packed centered pair");
    assert_eq!(pair.0, audited_pair_row);
    for column in [pair.1, pair.2] {
        let mut tampered = encoded.clone();
        tampered[column] = F::from_u64(2);
        assert_eq!(relation.first_unsatisfied_row(&tampered), Some(pair.0));
    }
    let tail = snapshot
        .materialize_row(audited_tail_row)
        .expect("materialized centered tail row");
    let tail_port = |index| tail.matrix_row().port(index).expect("selective port");
    assert_eq!(tail_port(GENERAL_SELECTOR)[0].column(), selector);
    assert_eq!(tail_port(EVAL_SELECTOR)[0].column(), selector);
    assert_eq!(tail_port(CENTERED_UNIT).len(), 1);
    assert!(tail_port(A).is_empty());
    let mut tampered = encoded.clone();
    tampered[tail_port(CENTERED_UNIT)[0].column()] = F::from_u64(2);
    assert_eq!(relation.first_unsatisfied_row(&tampered), Some(audited_tail_row));
}

#[test]
fn centered_septenary_rust_encoder_artifact_matches_committed_file() {
    let rendered = render_centered_septenary_rust_artifact();
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), CENTERED_SEPTENARY_ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write centered-septenary artifact review file");
        panic!("centered-septenary Rust encoder artifact drifted; inspect {expected}");
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_centered_septenary_rust_encoder_artifact() {
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), CENTERED_SEPTENARY_ARTIFACT_REL_PATH);
    std::fs::write(&path, render_centered_septenary_rust_artifact())
        .expect("write centered-septenary Rust encoder artifact");
    let expected = format!("{path}.expected");
    if std::path::Path::new(&expected).exists() {
        std::fs::remove_file(expected).expect("remove reviewed expected artifact");
    }
}
