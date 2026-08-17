//! Rectangular one-joint regression for the selective low-norm compiler.

#[path = "../support/mod.rs"]
mod support;

use neo_ccs::{CcsMatrix, CscMat, GeometricRowRun};
use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::pi_rlc_challenge_stage;
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::poseidon2::enforce_poseidon2_permutation;
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::{BalancedTernaryOpeningTraceEntry, Lc, R1csBuilder};
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
use neo_fold_clean::paper::reductions::pi_ccs_circuit::stage as pi_ccs_stage;
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
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
        "PaddedRowIdentity must preserve the semantic-row domain"
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
fn active_fixed_point_shape_stabilizes_after_accumulator_ce_compression() {
    let app = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let audit = R1csIvcRelation::audit_fixed_point_shape(&tiny_params(), &app.into(), &plan)
        .expect("audit active fixed-point shape");

    let width = audit.width();
    assert_eq!(
        audit
            .rounds()
            .iter()
            .map(|round| {
                (
                    round.input.rows,
                    round.input.columns,
                    round.output.rows,
                    round.output.columns,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (2, 270, 7_080_164, 11_543_040),
            (7_080_164, 11_543_040, 7_110_788, 11_997_504),
            (7_110_788, 11_997_504, 7_110_788, 11_997_504),
        ],
        "the selected one-joint production shape must stabilize at the measured fixed point",
    );
    assert_eq!(width.total_coordinates, 11_997_464);
    assert_eq!(width.branch_start, 311);
    assert_eq!(width.shared_private_coordinates, 0);
    assert_eq!(
        width
            .arms
            .iter()
            .map(|arm| {
                (
                    arm.branch_source_columns,
                    arm.eliminated_columns,
                    arm.retained_coordinates_before_aliases,
                    arm.decomposition_aliases,
                    arm.equality_aliases,
                    arm.branch_coordinates,
                    arm.derived_product_sums,
                    arm.derived_coordinates,
                    arm.total_branch_coordinates,
                    arm.traces.poseidon2_permutations,
                    arm.traces.poseidon2_coordinates,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (14_261, 11_631, 82_111, 448, 0, 81_663, 0, 0, 81_663, 22, 77_828),
            (
                10_713_961, 5_561_723, 14_601_537, 3_294_332, 912, 11_269_813, 17_740, 727_340, 11_997_153, 937,
                3_335_094,
            ),
            (
                10_713_961, 5_561_723, 14_601_537, 3_294_332, 912, 11_269_813, 17_740, 727_340, 11_997_153, 937,
                3_335_094,
            ),
        ],
        "each selector-disjoint arm must retain the measured compressed-width profile",
    );
    let terminal_round = audit.rounds().last().expect("terminal fixed-point round");
    assert_eq!(
        terminal_round.output.public_input_len,
        F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN
    );
    assert_eq!(terminal_round.output.public_input_len % D, 0);
    assert!(terminal_round.output.columns >= terminal_round.output.public_input_len);
    assert_eq!(terminal_round.output.polynomial.arity(), 13);
    let output_digest = audit.pi_ccs_output_digest();
    let output_profile = output_digest.profile();
    assert_eq!(output_profile.source_count(), 15);
    assert_eq!(output_profile.matrix_count(), 14);
    assert_eq!(output_profile.output_field_count(), 23_033);
    let output_sis = output_digest.sis();
    assert_eq!(output_sis.primary().block().word_starts().len(), 23_033);
    assert_eq!(output_sis.primary().input_columns().len(), 23_033);
    assert_eq!(output_sis.primary().output_columns().len(), 2 * D);
    assert_eq!(output_sis.compression().block().word_starts().len(), 2 * D);
    assert_eq!(
        output_sis.compression().input_columns(),
        output_sis.primary().output_columns()
    );
    assert_eq!(output_sis.compression().output_columns().len(), D);
    let output_prefix = output_digest.envelope_prefix();
    assert_eq!(output_prefix.columns().len(), 10);
    assert_eq!(output_prefix.values().len(), 10);
    assert_eq!(output_prefix.values()[8], F::from_u64(23_033));
    assert_eq!(output_prefix.values()[9], F::from_u64(2));
    assert_eq!(output_prefix.rows().end, output_digest.hash().row_start);
    assert_eq!(output_digest.hash().input_cols.len(), 64);
    assert_eq!(output_digest.hash().rounds.len(), 17);
    assert_eq!(
        output_digest.hash().input_cols,
        output_prefix
            .columns()
            .iter()
            .chain(output_sis.compression().output_columns())
            .copied()
            .collect::<Vec<_>>()
    );

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
    let stage_census = |arm: usize, path: &str| {
        let occurrences = source_stages[arm]
            .iter()
            .enumerate()
            .filter_map(|(occurrence, stage)| (stage.path() == path).then_some((occurrence, stage)))
            .collect::<Vec<_>>();
        assert!(occurrences.len() <= 1, "arm {arm} repeats accumulator leaf {path}");
        occurrences.first().map(|(occurrence, stage)| {
            let emitted = audit
                .rows()
                .emitted_runs()
                .iter()
                .filter(|run| run.arm() == Some(arm) && run.source_stage_occurrence() == Some(*occurrence))
                .map(|run| run.emitted_rows().len())
                .sum::<usize>();
            (*occurrence, stage.rows().len(), emitted)
        })
    };
    let accumulator_stage_census = (0..source_stages.len())
        .map(|arm| {
            (
                arm,
                stage_census(arm, fprime_stage::RECURSIVE_ACCUMULATOR_OUTPUT_CHILD_DIGESTS),
                stage_census(arm, fprime_stage::RECURSIVE_ACCUMULATOR_OUTPUT_AGGREGATE),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        accumulator_stage_census,
        vec![
            (0, None, None),
            (1, Some((7_819, 3_034_465, 513_786)), Some((7_820, 3_034, 434)),),
            (2, Some((7_819, 3_034_465, 513_786)), Some((7_820, 3_034, 434)),),
        ],
        "the selected protocol must bind exact outgoing children without a delayed pending-family stage",
    );
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
}

#[test]
#[ignore = "materializes the complete 7.1M-row by 12.7M-column fixed-point fixture; run explicitly after compiler changes"]
fn active_fixed_point_materializes_after_accumulator_ce_compression() {
    let app = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let relation = R1csIvcRelation::compile_fixed_point(&tiny_params(), &app.into(), &plan)
        .expect("materialize SIS-compressed active fixed point");

    let structure = relation.structure();
    assert_eq!(structure.n, 7_143_950);
    assert_eq!(structure.m, 12_678_066);
    assert_eq!(relation.public_input_len(), F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
    assert_eq!(structure.t(), 13);
    assert_eq!(structure.matrices.len(), 13);

    let audit = relation.compilation_audit();
    assert_eq!(audit.rounds().len(), 3);
    assert_eq!(audit.width().total_coordinates, 11_997_464);
    assert_eq!(audit.layout().total_columns(), structure.m);
    assert_eq!(audit.rows().total_rows(), structure.n);
    let terminal = audit
        .rounds()
        .last()
        .expect("materialized terminal fixed-point round");
    assert_eq!(terminal.output.rows, structure.n);
    assert_eq!(terminal.output.columns, structure.m);
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

fn shifted_canonical_arm_and_assignment(split_trace_stage: bool) -> (SparseR1cs, Vec<F>) {
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
}

fn shifted_canonical_arm(split_trace_stage: bool) -> SparseR1cs {
    shifted_canonical_arm_and_assignment(split_trace_stage).0
}

fn shifted_canonical_private_arm_and_assignment() -> (SparseR1cs, Vec<F>, BalancedTernaryOpeningTraceEntry) {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.shifted_canonical_private");
    let field = builder.alloc(F::ZERO);
    enforce_commit_fields(&mut builder, SIS_DIGEST_COMPRESSION_CONFIG, &[field]).expect("one-field SIS commitment");
    let opening = builder.encoding_trace().balanced_ternary_openings()[0].clone();
    builder.begin_encoding_stage("complete");
    let (shape, assignment) = lower_field_r1cs(builder, &[])
        .expect("lower private shifted-canonical arm")
        .into_parts();
    (shape, assignment, opening)
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
fn shifted_ternary_rewrite_links_124_source_rows_to_21_emitted_rows() {
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
        assert_eq!(rewrite.source_rows(), &[2..126]);
        assert_eq!(
            rewrite
                .source_rows()
                .iter()
                .map(|range| range.len())
                .sum::<usize>(),
            124
        );
        assert_eq!(rewrite.emitted_rows().len(), 21);
        assert_eq!(rewrite.source_stage_occurrence(), Some(0));

        let linked_source = rows.arms()[arm]
            .source_runs()
            .iter()
            .filter(|run| run.disposition().rewrite_id() == Some(rewrite.id()))
            .map(|run| run.source_rows())
            .collect::<Vec<_>>();
        assert_eq!(linked_source, vec![2..126]);
        assert_eq!(rows.arms()[arm].emitted_rows().len(), 77);
    }
}

#[test]
fn shifted_ternary_pairs_retain_only_endpoint_borrows_and_reject_tampering() {
    let (first, first_assignment, _) = shifted_canonical_private_arm_and_assignment();
    let (second, second_assignment, _) = shifted_canonical_private_arm_and_assignment();
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[first, second], 0, D, 0)
        .expect("selective canonical relation");
    let compiler_audit = relation
        .selective_compiler_audit()
        .expect("selective compiler audit");
    for (arm, openings) in compiler_audit.canonical_openings().iter().enumerate() {
        let [opening] = openings.as_slice() else {
            panic!("arm {arm} must own one memoized opening")
        };
        assert_eq!(opening.digit_coordinates().len(), 41);
        assert_eq!(opening.borrow_coordinates().len(), 20);
        assert_eq!(opening.coordinate_count(), 61);
        assert_eq!(opening.emitted_rows().len(), 21);
    }

    for (arm, source) in [first_assignment, second_assignment].iter().enumerate() {
        let encoded = relation.encode(arm, source).expect("encode canonical arm");
        assert!(relation.is_satisfied(&encoded), "honest arm {arm}");

        let borrow_starts = (1..=source.len() - 40)
            .filter(|&start| (0..40).all(|index| relation.field_slot(arm, start + index).is_some() == (index % 2 == 1)))
            .collect::<Vec<_>>();
        assert_eq!(borrow_starts.len(), 1, "one projected 40-borrow run");
        let borrow_start = borrow_starts[0];
        for index in 0..40 {
            let slot = relation.field_slot(arm, borrow_start + index);
            assert_eq!(
                slot.is_some(),
                index % 2 == 1,
                "only the endpoint after each two-trit chunk is retained"
            );
        }

        let endpoint = relation
            .field_slot(arm, borrow_start + 1)
            .expect("first chunk endpoint")
            .0;
        let mut tampered = encoded;
        tampered[endpoint] = if tampered[endpoint] == F::ZERO { F::ONE } else { F::ZERO };
        assert!(
            !relation.is_satisfied(&tampered),
            "the paired transition must bind its retained endpoint"
        );
    }
}

fn row_residual(structure: &neo_ccs::CcsStructure<F>, row: usize, assignment: &[F]) -> F {
    let point = structure
        .matrices
        .iter()
        .map(|matrix| {
            matrix
                .materialize_row(row)
                .expect("in-range selective row")
                .into_iter()
                .fold(F::ZERO, |sum, (column, coefficient)| {
                    sum + coefficient * assignment[column]
                })
        })
        .collect::<Vec<_>>();
    structure.f.eval(&point)
}

#[test]
fn shifted_ternary_pair_rows_match_every_local_transition() {
    let (first, _, opening) = shifted_canonical_private_arm_and_assignment();
    let (second, _, _) = shifted_canonical_private_arm_and_assignment();
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[first, second], 0, D, 0)
        .expect("selective canonical relation");
    let rewrite = relation
        .selective_compiler_audit()
        .expect("selective compiler audit")
        .rows()
        .rewrites()
        .iter()
        .find(|rewrite| rewrite.arm() == 0 && rewrite.kind() == SelectiveRewriteKind::ShiftedTernaryCanonical)
        .expect("one shifted-ternary rewrite")
        .clone();
    assert_eq!(rewrite.emitted_rows().len(), 21);

    let digit_slots = opening
        .digit_cols
        .map(|column| relation.field_slot(0, column).expect("retained digit").0);
    let borrow_slots = opening.borrow_cols.map(|column| {
        relation
            .field_slot(0, column)
            .map(|(coordinate, _)| coordinate)
    });
    let selector = relation.selector_cols()[0];
    let centered = [-F::ONE, F::ZERO, F::ONE];
    let trit = |digit: F| {
        if digit == -F::ONE {
            0u64
        } else if digit == F::ZERO {
            1
        } else {
            2
        }
    };
    let mut bound = F::ORDER_U64 - 1;

    for chunk in 0..21 {
        let digit_index = 2 * chunk;
        let bound_zero = bound % 3;
        bound /= 3;
        let has_second = digit_index + 1 < 41;
        let bound_one = if has_second {
            let value = bound % 3;
            bound /= 3;
            value
        } else {
            0
        };
        let row = rewrite.emitted_rows().start + chunk;
        let second_digits: &[F] = if has_second { &centered } else { &[-F::ONE] };
        let inputs: &[u64] = if chunk == 0 { &[0] } else { &[0, 1] };
        let outputs: &[F] = if chunk == 20 {
            &[F::ZERO]
        } else {
            &[-F::ONE, F::ZERO, F::ONE]
        };

        for &digit_zero in &centered {
            for &digit_one in second_digits {
                for &borrow_in in inputs {
                    let middle = u64::from(trit(digit_zero) + borrow_in > bound_zero);
                    let expected = u64::from(trit(digit_one) + middle > bound_one);
                    for &borrow_out in outputs {
                        let mut assignment = vec![F::ZERO; relation.structure().m];
                        assignment[0] = F::ONE;
                        assignment[selector] = F::ONE;
                        assignment[digit_slots[digit_index]] = digit_zero;
                        if has_second {
                            assignment[digit_slots[digit_index + 1]] = digit_one;
                        }
                        if chunk != 0 {
                            assignment[borrow_slots[digit_index - 1].expect("retained input endpoint")] =
                                F::from_u64(borrow_in);
                        }
                        if chunk != 20 {
                            assignment[borrow_slots[digit_index + 1].expect("retained output endpoint")] = borrow_out;
                        }
                        assert_eq!(
                            row_residual(relation.structure(), row, &assignment) == F::ZERO,
                            borrow_out == F::from_u64(expected),
                            "chunk={chunk} d0={} d1={} bin={borrow_in} bout={}",
                            trit(digit_zero),
                            trit(digit_one),
                            borrow_out.as_canonical_u64(),
                        );
                    }
                }
            }
        }
    }
    assert_eq!(bound, 0);
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

    let definition_arms = relation
        .selective_compiler_audit()
        .expect("selective compiler audit")
        .source_arm_linear_definitions();
    assert_eq!(definition_arms.len(), 2);
    for definitions in definition_arms {
        let [definition] = definitions.as_slice() else {
            panic!("one exact affine definition per arm");
        };
        assert_eq!(definition.source_row(), Some(0));
        assert_eq!(definition.target(), 2);
        assert_eq!(definition.constant(), F::from_u64(3));
        let [term] = definition.terms() else {
            panic!("one exact affine source term");
        };
        assert_eq!(term.column(), 1);
        assert_eq!(term.coefficient(), F::ONE);
    }

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
    for arm in &audit.width().arms {
        assert_eq!(
            arm.physical_stages
                .iter()
                .map(|stage| stage.allocated_coordinates)
                .sum::<usize>(),
            arm.branch_coordinates,
            "exclusive stage widths must cover each branch exactly",
        );
    }
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
