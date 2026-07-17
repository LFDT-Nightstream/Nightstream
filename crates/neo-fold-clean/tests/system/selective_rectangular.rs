//! Rectangular SplitNc regression for the selective low-norm compiler.

#[path = "../support/mod.rs"]
mod support;

use neo_ccs::{CcsMatrix, CscMat, GeometricRowRun};
use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::pi_rlc_challenge_stage;
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::poseidon2::enforce_poseidon2_permutation;
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcRelation;
use neo_fold_clean::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_low_norm_width_with_alignment,
    build_multi_branch_selective_low_norm_r1cs_with_alignment, lower_field_r1cs, LowNormR1csError,
    SelectiveEmittedRowFamily, SelectiveRewriteKind, SelectiveSourceRowDisposition, SparseR1cs,
};
use neo_fold_clean::paper::f_prime::r1cs::{F_PRIME_PUBLIC_INPUT_LEN, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN};
use neo_fold_clean::paper::f_prime::stage as fprime_stage;
use neo_fold_clean::paper::nifs::circuit::stage as nifs_stage;
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{
    enforce_commit_fields, SIS_DIGEST_COMPRESSION_CONFIG,
};
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::stage as pi_ccs_stage;
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use support::r1cs_compiler_fixtures::{make_tiny_lifecycle_plan, one_product_r1cs, tiny_params};

#[test]
fn selective_poseidon_lowering_is_rectangular_and_binds_alignment_tail() {
    let mut shapes = Vec::new();
    let mut assignments = Vec::new();
    for arm in 0..3u64 {
        let mut builder = R1csBuilder::new();
        let input = core::array::from_fn(|lane| builder.alloc(F::from_u64(arm * 17 + lane as u64 + 1)));
        let output = enforce_poseidon2_permutation(&mut builder, &input);
        let output_bits = decompose_var_to_u64_bits(&mut builder, output[0]);
        let lowered = lower_field_r1cs(builder, &[output_bits[0]]).expect("field lowering");
        let (shape, assignment) = lowered.into_parts();
        shapes.push(shape);
        assignments.push(assignment);
    }

    let audit =
        audit_multi_branch_selective_low_norm_width_with_alignment(&shapes, 0, D, 0).expect("selective width audit");
    assert!(audit
        .arms
        .iter()
        .all(|arm| arm.traces.poseidon2_columns == 87));

    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, 0).expect("selective relation");
    let structure = relation.structure();
    assert!(
        structure.n < structure.m,
        "SplitNc must preserve the semantic-row domain"
    );
    assert_eq!(structure.t(), 13, "only semantic matrices belong in the relation");
    assert!(
        !structure.matrices[0].is_identity(),
        "the NC identity matrix is obsolete"
    );
    assert_eq!(structure.m, audit.total_coordinates.next_multiple_of(D));

    let row_audit = relation
        .selective_compiler_audit()
        .expect("selective compiler audit")
        .rows();
    assert_eq!(row_audit.total_rows(), structure.n);
    let prefix = row_audit.prefix_rows();
    assert_eq!(prefix.start, 0);
    let mut emitted_cursor = prefix.end;
    assert_eq!(row_audit.arms().len(), shapes.len());
    for (arm_index, (arm_rows, shape)) in row_audit.arms().iter().zip(&shapes).enumerate() {
        let emitted = arm_rows.emitted_rows();
        let retained = arm_rows.retained_emitted_rows();
        assert_eq!(emitted.start, emitted_cursor, "arm {arm_index} emitted-row gap");
        assert_eq!(retained.start, emitted.start, "arm {arm_index} retained prefix");

        let source_runs = arm_rows.source_runs();
        assert!(
            source_runs.len() < shape.n,
            "arm {arm_index} source mapping must remain run-compressed"
        );
        let mut source_cursor = 0usize;
        let mut retained_source_rows = 0usize;
        let mut skipped_source_rows = 0usize;
        let mut retained_targets = Vec::new();
        for (run_index, run) in source_runs.iter().enumerate() {
            let source = run.source_rows();
            assert_eq!(source.start, source_cursor, "arm {arm_index} source-run gap");
            assert!(!source.is_empty(), "arm {arm_index} empty source run");
            if let Some(emitted_start) = run.emitted_start() {
                retained_source_rows += source.len();
                retained_targets.extend(emitted_start..emitted_start + source.len());
            } else {
                skipped_source_rows += source.len();
            }
            if let Some(next) = source_runs.get(run_index + 1) {
                assert_ne!(
                    run.emitted_start().is_some(),
                    next.emitted_start().is_some(),
                    "arm {arm_index} adjacent equal-disposition runs were not compressed"
                );
            }
            source_cursor = source.end;
        }
        assert_eq!(source_cursor, shape.n, "arm {arm_index} source partition");
        assert_eq!(
            retained_targets,
            retained.clone().collect::<Vec<_>>(),
            "arm {arm_index} retained source rows must map once in source order"
        );
        assert_eq!(
            retained_source_rows + skipped_source_rows,
            shape.n,
            "arm {arm_index} source rows must be retained or skipped exactly once"
        );
        assert!(
            skipped_source_rows != 0,
            "arm {arm_index} fixture must exercise trace replacement"
        );
        assert!(retained.end <= emitted.end, "arm {arm_index} retained rows escape arm");
        emitted_cursor = emitted.end;
    }
    let ring_padding = row_audit.ring_padding_rows();
    assert_eq!(ring_padding.start, emitted_cursor);
    assert_eq!(ring_padding.end, row_audit.total_rows());

    let alignment_tail = audit.total_coordinates..structure.m;
    assert!(!alignment_tail.is_empty(), "fixture must exercise ring alignment");
    for arm in 0..3 {
        let encoded = relation.encode(arm, &assignments[arm]).expect("encode arm");
        assert!(relation.is_satisfied(&encoded), "arm {arm} must satisfy the relation");
        assert!(encoded[alignment_tail.clone()]
            .iter()
            .all(|value| *value == F::ZERO));

        for coordinate in alignment_tail.clone() {
            let mut padding_tamper = encoded.clone();
            padding_tamper[coordinate] = F::ONE;
            assert!(
                !relation.is_satisfied(&padding_tamper),
                "alignment coordinate {coordinate} must be zero-bound"
            );
        }

        let mut output_tamper = encoded;
        output_tamper[1] = if output_tamper[1] == F::ZERO { F::ONE } else { F::ZERO };
        assert!(
            !relation.is_satisfied(&output_tamper),
            "public Poseidon output must be bound"
        );
    }
}

#[test]
fn selective_f_prime_public_carrier_precedes_selectors() {
    let mut shapes = Vec::new();
    let mut assignments = Vec::new();
    for arm in 0..3 {
        let mut builder = R1csBuilder::new();
        let public_bits = (0..F_PRIME_PUBLIC_INPUT_LEN - 1)
            .map(|_| {
                let bit = builder.alloc(F::ZERO);
                enforce_bit(&mut builder, bit);
                bit
            })
            .collect::<Vec<_>>();
        let lowered = lower_field_r1cs(builder, &public_bits).expect("field lowering");
        let (shape, assignment) = lowered.into_parts();
        assert_eq!(shape.m_in, F_PRIME_PUBLIC_INPUT_LEN, "arm {arm} logical width");
        shapes.push(shape);
        assignments.push(assignment);
    }

    let residue = F_PRIME_PUBLIC_INPUT_LEN % D;
    let audit = audit_multi_branch_selective_low_norm_width_with_alignment(&shapes, 0, D, residue)
        .expect("selective width audit");
    assert_eq!(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, 270);
    assert_eq!(audit.logical_public_coordinates, 256);
    assert_eq!(audit.public_carrier_padding, 13);
    assert_eq!(audit.public_coordinates, 269);
    assert_eq!(audit.selector_coordinates, 3);
    assert_eq!(audit.alignment_padding, 38);
    assert_eq!(audit.branch_start, 311);

    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, residue).expect("selective relation");
    assert_eq!(relation.public_input_len(), F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
    assert_eq!(relation.selector_cols(), &[270, 271, 272]);
    let emitted_audit = relation
        .selective_compiler_audit()
        .expect("selective compiler audit");
    assert_eq!(emitted_audit.width(), &audit);
    let layout = emitted_audit.layout();
    assert_eq!(layout.logical_public_input_len(), F_PRIME_PUBLIC_INPUT_LEN);
    assert_eq!(layout.public_input_len(), F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
    assert!(layout.public_padding_columns().iter().copied().eq(257..270));
    assert_eq!(layout.selector_columns(), &[270, 271, 272]);
    assert!(layout
        .private_alignment_padding_columns()
        .iter()
        .copied()
        .eq(273..311));
    assert_eq!(layout.shared_private_columns(), 311..311);
    assert_eq!(layout.branch_columns().start, 311);
    assert_eq!(layout.total_columns(), relation.structure().m);

    for arm in 0..3 {
        let encoded = relation.encode(arm, &assignments[arm]).expect("encode arm");
        assert!(relation.is_satisfied(&encoded));
        for &coordinate in layout.public_padding_columns() {
            assert_eq!(encoded[coordinate], F::ZERO);
            let mut tampered = encoded.clone();
            tampered[coordinate] = F::ONE;
            assert!(!relation.is_satisfied(&tampered));
        }
        for &coordinate in layout.private_alignment_padding_columns() {
            assert_eq!(encoded[coordinate], F::ZERO);
            let mut tampered = encoded.clone();
            tampered[coordinate] = F::ONE;
            assert!(!relation.is_satisfied(&tampered));
        }
        for coordinate in layout.ring_alignment_padding_columns() {
            assert_eq!(encoded[coordinate], F::ZERO);
            let mut tampered = encoded.clone();
            tampered[coordinate] = F::ONE;
            assert!(!relation.is_satisfied(&tampered));
        }
        for (selector_index, &coordinate) in layout.selector_columns().iter().enumerate() {
            assert_eq!(
                encoded[coordinate],
                if selector_index == arm { F::ONE } else { F::ZERO }
            );
            let mut tampered = encoded.clone();
            tampered[coordinate] = if selector_index == arm { F::ZERO } else { F::ONE };
            assert!(!relation.is_satisfied(&tampered));
        }
    }
}

#[test]
fn tiny_fixed_point_tracks_the_aligned_public_carrier() {
    let app = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let relation = R1csIvcRelation::compile_fixed_point(&tiny_params(), &app.into(), &plan)
        .expect("compile diagnostic fixed point");

    assert_eq!(relation.public_input_len(), F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
    assert_eq!(relation.public_input_len() % D, 0);
    assert!(relation.structure().m >= relation.public_input_len());

    let audit = relation.compilation_audit();
    assert!(audit.rounds().len() >= 2, "fixed point requires an observed repeat");
    let terminal_round = audit.rounds().last().expect("terminal fixed-point round");
    let source_stages = audit.source_arm_physical_stages();
    assert_eq!(source_stages.len(), terminal_round.arms.len());
    for (arm, (stages, shape)) in source_stages.iter().zip(&terminal_round.arms).enumerate() {
        assert!(!stages.is_empty(), "fixed-point arm {arm} omitted physical provenance");
        let expected_root = if arm == 0 {
            fprime_stage::BASE_ROOT
        } else {
            fprime_stage::RECURSIVE_ROOT
        };
        assert_eq!(stages[0].path(), expected_root, "fixed-point arm {arm} root");

        let mut next_row = 0usize;
        for stage in stages {
            assert_eq!(
                stage.row_start(),
                next_row,
                "fixed-point arm {arm} has a row gap or overlap"
            );
            assert!(
                stage.row_end() >= stage.row_start(),
                "fixed-point arm {arm} has a reversed range"
            );
            next_row = stage.row_end();

            let path = stage.path();
            let allowed = if arm == 0 {
                fprime_stage::BASE_ALL.contains(&path)
            } else {
                fprime_stage::RECURSIVE_ALL.contains(&path)
                    || pi_ccs_stage::ALL.contains(&path)
                    || pi_rlc_stage::LIFECYCLE_ALL.contains(&path)
                    || pi_rlc_stage::ALL.contains(&path)
                    || pi_rlc_challenge_stage::ALL.contains(&path)
                    || nifs_stage::ALL.contains(&path)
            };
            assert!(allowed, "fixed-point arm {arm} contains undeclared stage {path}");
        }
        assert_eq!(
            next_row, shape.rows,
            "fixed-point arm {arm} does not cover all source rows"
        );
    }
    let layout = audit.layout();
    assert_eq!(layout.logical_public_input_len(), F_PRIME_PUBLIC_INPUT_LEN);
    assert_eq!(layout.public_input_len(), F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
    assert!(layout.public_padding_columns().iter().copied().eq(257..270));
    assert_eq!(layout.selector_columns(), &[270, 271, 272]);
    assert!(layout
        .private_alignment_padding_columns()
        .iter()
        .copied()
        .eq(273..311));
    assert_eq!(layout.branch_columns().start, 311);

    let same_polynomial = |left: &neo_ccs::SparsePoly<F>, right: &neo_ccs::SparsePoly<F>| {
        left.arity() == right.arity()
            && left.terms().len() == right.terms().len()
            && left
                .terms()
                .iter()
                .zip(right.terms())
                .all(|(left, right)| left.coeff == right.coeff && left.exps == right.exps)
    };
    for adjacent in audit.rounds().windows(2) {
        assert_eq!(adjacent[0].output.rows, adjacent[1].input.rows);
        assert_eq!(adjacent[0].output.columns, adjacent[1].input.columns);
        assert_eq!(adjacent[0].output.public_input_len, adjacent[1].input.public_input_len);
        assert!(same_polynomial(
            &adjacent[0].output.polynomial,
            &adjacent[1].input.polynomial
        ));
    }
    let terminal = &audit.rounds().last().expect("terminal round").output;
    assert_eq!(
        (terminal.rows, terminal.columns),
        (relation.structure().n, relation.structure().m)
    );
    assert_eq!(terminal.public_input_len, relation.public_input_len());
    assert!(same_polynomial(&terminal.polynomial, &relation.structure().f));
}

#[test]
fn selective_source_geometric_rows_are_remapped() {
    let empty = || CscMat::from_triplets(Vec::new(), 1, 3);
    let a = CcsMatrix::csc_with_compact_rows(
        empty(),
        Vec::new(),
        vec![GeometricRowRun::new(0, 1, 2, F::from_u64(2), F::from_u64(3))],
    )
    .expect("compact source A matrix");
    let b = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 0, F::ONE)], 1, 3));
    let c = CcsMatrix::Csc(CscMat::from_triplets(
        vec![(0, 1, F::from_u64(2)), (0, 2, F::from_u64(6))],
        1,
        3,
    ));
    let source = SparseR1cs::new(a, b, c, 1, 3, 1).expect("source shape");

    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[source.clone(), source], 0, D, 0)
        .expect("selective relation");
    let encoded = relation
        .encode(0, &[F::ONE, F::from_u64(7), F::from_u64(11)])
        .expect("encode source assignment");

    assert!(
        relation.is_satisfied(&encoded),
        "the compact source coefficient must survive selective slot remapping"
    );
}

fn shifted_canonical_arm(split_trace_stage: bool) -> SparseR1cs {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("test.shifted_canonical");
    let field = builder.alloc(F::ZERO);
    let commitment =
        enforce_commit_fields(&mut builder, SIS_DIGEST_COMPRESSION_CONFIG, &[field]).expect("one-field SIS commitment");
    assert_eq!(builder.rows(), 180, "fixture source-row census");
    if split_trace_stage {
        builder.append_physical_stage_checkpoint_for_test("test.shifted_canonical.inner", 43);
    }
    builder.begin_encoding_stage("complete");
    lower_field_r1cs(builder, &[commitment.d_var, commitment.kappa_var])
        .expect("lower shifted-canonical arm")
        .into_parts()
        .0
}

fn linear_definition_arm() -> SparseR1cs {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("test.linear_definition");
    let x = builder.alloc(F::from_u64(7));
    let y = builder.alloc(F::from_u64(10));
    let mut rhs = Lc::from_var(x);
    rhs.add_constant(F::from_u64(3));
    builder.enforce_eq(&Lc::from_var(y), &rhs);
    builder.begin_encoding_stage("complete");
    lower_field_r1cs(builder, &[])
        .expect("lower affine definition")
        .into_parts()
        .0
}

fn repeated_stage_arm() -> SparseR1cs {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("test.root");
    builder.begin_encoding_stage("test.repeated");
    let first = builder.alloc(F::ONE);
    enforce_bit(&mut builder, first);
    builder.begin_encoding_stage("test.repeated");
    let second = builder.alloc(F::ZERO);
    enforce_bit(&mut builder, second);
    builder.begin_encoding_stage("complete");
    lower_field_r1cs(builder, &[])
        .expect("lower repeated stages")
        .into_parts()
        .0
}

#[test]
fn shifted_ternary_rewrite_links_123_source_rows_to_82_emitted_rows() {
    let first = shifted_canonical_arm(false);
    let second = shifted_canonical_arm(false);
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[first, second], 0, D, 0)
        .expect("selective canonical relation");
    let rows = relation
        .selective_compiler_audit()
        .expect("selective compiler audit")
        .rows();

    for arm in 0..2 {
        let rewrite = rows
            .rewrites()
            .iter()
            .find(|rewrite| rewrite.arm() == arm && rewrite.kind() == SelectiveRewriteKind::ShiftedTernaryCanonical)
            .expect("one shifted-ternary rewrite");
        assert_eq!(rewrite.source_rows(), &[2..84, 85..126]);
        assert_eq!(
            rewrite
                .source_rows()
                .iter()
                .map(|range| range.len())
                .sum::<usize>(),
            123
        );
        assert_eq!(rewrite.emitted_rows().len(), 82);
        assert_eq!(rewrite.source_stage_occurrence(), Some(0));

        let linked_source = rows.arms()[arm]
            .source_runs()
            .iter()
            .filter(|run| run.disposition().rewrite_id() == Some(rewrite.id()))
            .map(|run| run.source_rows())
            .collect::<Vec<_>>();
        assert_eq!(linked_source, vec![2..84, 85..126]);
        assert_eq!(rows.arms()[arm].emitted_rows().len(), 139);
    }
}

#[test]
fn linear_definition_rewrite_is_explicitly_source_to_empty() {
    let first = linear_definition_arm();
    let second = linear_definition_arm();
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[first, second], 0, D, 0)
        .expect("selective affine relation");
    let rows = relation
        .selective_compiler_audit()
        .expect("selective compiler audit")
        .rows();

    for arm in 0..2 {
        let rewrite = rows
            .rewrites()
            .iter()
            .find(|rewrite| rewrite.arm() == arm && rewrite.kind() == SelectiveRewriteKind::LinearDefinition)
            .expect("one linear-definition rewrite");
        assert_eq!(rewrite.source_rows(), &[0..1]);
        assert!(rewrite.emitted_rows().is_empty());
        assert_eq!(rewrite.source_stage_occurrence(), Some(0));
        assert_eq!(rows.arms()[arm].source_runs().len(), 1);
        assert_eq!(
            rows.arms()[arm].source_runs()[0].disposition(),
            SelectiveSourceRowDisposition::LinearDefinition(rewrite.id())
        );
        assert!(rows.arms()[arm].emitted_rows().is_empty());
    }
}

#[test]
fn row_ledger_partitions_every_source_and_emitted_row_once() {
    let first = shifted_canonical_arm(false);
    let second = shifted_canonical_arm(false);
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[first, second], 0, D, 0)
        .expect("selective canonical relation");
    let rows = relation
        .selective_compiler_audit()
        .expect("selective compiler audit")
        .rows();

    for arm in rows.arms() {
        let mut source_cursor = 0usize;
        for run in arm.source_runs() {
            assert_eq!(run.source_rows().start, source_cursor);
            assert!(!run.source_rows().is_empty());
            source_cursor = run.source_rows().end;
        }
        assert_eq!(source_cursor, 180);
    }

    let mut emitted_cursor = 0usize;
    for run in rows.emitted_runs() {
        assert_eq!(run.emitted_rows().start, emitted_cursor);
        emitted_cursor = run.emitted_rows().end;
    }
    assert_eq!(emitted_cursor, rows.total_rows());
    assert_eq!(
        rows.rewrites()
            .iter()
            .map(|rewrite| rewrite.id().index())
            .collect::<Vec<_>>(),
        (0..rows.rewrites().len()).collect::<Vec<_>>()
    );
    assert!(rows
        .emitted_runs()
        .iter()
        .any(|run| run.family() == SelectiveEmittedRowFamily::ShiftedTernaryCanonical));
    for rewrite in rows.rewrites() {
        let source_rows = rows.arms()[rewrite.arm()]
            .source_runs()
            .iter()
            .filter(|run| run.disposition().rewrite_id() == Some(rewrite.id()))
            .map(|run| run.source_rows())
            .collect::<Vec<_>>();
        assert_eq!(source_rows, rewrite.source_rows());
        if !rewrite.emitted_rows().is_empty() {
            assert!(rows
                .emitted_runs()
                .iter()
                .any(|run| { run.rewrite_id() == Some(rewrite.id()) && run.emitted_rows() == rewrite.emitted_rows() }));
        }
    }
}

#[test]
fn repeated_stage_paths_remain_distinct_occurrences_in_the_row_ledger() {
    let first = repeated_stage_arm();
    let second = repeated_stage_arm();
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[first, second], 0, D, 0)
        .expect("selective repeated-stage relation");
    let audit = relation
        .selective_compiler_audit()
        .expect("selective compiler audit");

    assert_eq!(audit.source_arm_physical_stages()[0][0].rows(), 0..0);
    assert_eq!(audit.source_arm_physical_stages()[0][1].path(), "test.repeated");
    assert_eq!(audit.source_arm_physical_stages()[0][2].path(), "test.repeated");
    let runs = audit.rows().arms()[0].source_runs();
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].source_rows(), 0..1);
    assert_eq!(runs[0].stage_occurrence(), Some(1));
    assert_eq!(runs[1].source_rows(), 1..2);
    assert_eq!(runs[1].stage_occurrence(), Some(2));
    assert!(runs
        .iter()
        .all(|run| run.disposition() == SelectiveSourceRowDisposition::Retained));
}

#[test]
fn selective_rewrite_crossing_stage_occurrences_is_rejected() {
    let first = shifted_canonical_arm(true);
    let second = shifted_canonical_arm(true);
    let error = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[first, second], 0, D, 0)
        .expect_err("cross-stage trace must fail closed");
    match error {
        LowNormR1csError::SelectiveTrace(message) => {
            assert!(
                message.contains("arm 0 ShiftedTernaryCanonical rewrite 0")
                    && message.contains("source rows 2..84")
                    && message.contains("crosses physical stage occurrences Some(0) and Some(1)"),
                "unexpected error: {message}"
            );
        }
        other => panic!("unexpected error: {other}"),
    }
}
