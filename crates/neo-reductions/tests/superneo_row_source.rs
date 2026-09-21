use neo_ccs::{CcsStructure, GeometricRowRun, Mat, SparsePoly};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::{
    build_superneo_eval_cache, SuperneoCompactRowOffsets, SuperneoEvalCache, SuperneoEvalCacheBuilder, SuperneoZBlocks,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn assert_offsets_equal(left: SuperneoCompactRowOffsets<'_>, right: SuperneoCompactRowOffsets<'_>) {
    use SuperneoCompactRowOffsets::*;
    match (left, right) {
        (Empty, Empty) => {}
        (U24(left), U24(right)) => assert_eq!(left, right),
        (U32(left), U32(right)) => assert_eq!(left, right),
        (
            U16Chunked {
                chunk_offsets: a,
                local_offsets: b,
                chunk_rows: c,
            },
            U16Chunked {
                chunk_offsets: x,
                local_offsets: y,
                chunk_rows: z,
            },
        ) => assert_eq!((a, b, c), (x, y, z)),
        _ => panic!("capacity hints changed offset encoding"),
    }
}

fn assert_cache_parts_equal(left: &SuperneoEvalCache, right: &SuperneoEvalCache) {
    assert_eq!(left.relation_shape(), right.relation_shape());
    for (left, right) in left.matrix_caches().iter().zip(right.matrix_caches()) {
        let a = left.compact_device_parts().unwrap();
        let b = right.compact_device_parts().unwrap();
        assert_offsets_equal(a.row_offsets, b.row_offsets);
        assert_offsets_equal(a.geometric_row_offsets, b.geometric_row_offsets);
        assert_eq!(
            (
                a.row_blocks,
                a.dense_row_blocks,
                a.dense_offsets,
                a.dense_locals,
                a.geometric_runs,
                a.identity
            ),
            (
                b.row_blocks,
                b.dense_row_blocks,
                b.dense_offsets,
                b.dense_locals,
                b.geometric_runs,
                b.identity
            )
        );
        let bytes = |values: &[F]| {
            values
                .iter()
                .flat_map(|value| value.as_canonical_u64().to_le_bytes())
                .collect::<Vec<_>>()
        };
        assert_eq!(bytes(a.dense_coefficients), bytes(b.dense_coefficients));
    }
}

#[test]
fn row_source_matches_matrix_derived_cache_and_implicit_zero_slot() {
    let rows = 4;
    let columns = 2 * D + 5;
    let entries = [
        vec![vec![], vec![], vec![]],
        vec![
            vec![(0, F::ONE), (D + 3, -F::ONE)],
            vec![(D - 1, F::from_u64(7))],
            vec![],
        ],
        vec![
            vec![(1, F::from_u64(3)), (D - 1, F::from_u64(11)), (columns - 1, F::ONE)],
            vec![],
            vec![],
        ],
        vec![vec![], vec![(D, -F::ONE)], vec![]],
    ];
    let mut matrices = vec![Mat::zero(rows, columns, F::ZERO); 3];
    let mut builder = SuperneoEvalCacheBuilder::new(rows, columns, matrices.len()).unwrap();
    for (row, forms) in entries.iter().enumerate() {
        for (matrix, form) in forms.iter().enumerate() {
            for &(column, coefficient) in form {
                matrices[matrix][(row, column)] = coefficient;
            }
            builder.push_row(matrix, row, form.iter().copied()).unwrap();
        }
    }
    let structure = CcsStructure::new(matrices, SparsePoly::new(3, vec![])).unwrap();
    let expected = build_superneo_eval_cache(&structure).unwrap();
    let actual = builder.finish().unwrap();
    assert_eq!(actual.relation_shape(), Some((rows, columns.div_ceil(D) * D, 3)));
    for (row, forms) in entries.iter().enumerate() {
        for (matrix, form) in forms.iter().enumerate() {
            let mut decoded = Vec::new();
            actual
                .matrix(matrix)
                .unwrap()
                .for_each_compact_explicit_row_coefficient(row, |block, local, value| {
                    decoded.push((block as usize * D + usize::from(local), value));
                });
            assert_eq!(&decoded, form);
        }
    }
    let zero = actual.matrix(2).unwrap().compact_device_parts().unwrap();
    assert!(matches!(zero.row_offsets, SuperneoCompactRowOffsets::Empty));
    assert!(zero.row_blocks.is_empty());
    let values = (0..columns)
        .map(|index| K::from(F::from_u64(index as u64 + 1)))
        .collect::<Vec<_>>();
    let witnesses = [SuperneoZBlocks::from_z(&values)];
    let weights = (0..rows)
        .map(|index| K::from_coeffs([F::from_u64(index as u64 + 3), F::ONE]))
        .collect::<Vec<_>>();
    let wanted = expected.eval_ring_linear_forms_for_real_z_blocks(&weights, rows, &witnesses);
    let got = actual.eval_ring_linear_forms_for_real_z_blocks(&weights, rows, &witnesses);
    assert_eq!(got, wanted);
    assert_eq!(got[0][2], [K::ZERO; D]);
    for hints in [[(4, 1, 2), (2, 1, 1), (0, 0, 0)], [(0, 0, 0); 3]] {
        let mut reserved = SuperneoEvalCacheBuilder::new(rows, columns, 3).unwrap();
        for (matrix, (blocks, dense, coefficients)) in hints.into_iter().enumerate() {
            reserved
                .reserve_matrix(matrix, blocks, dense, coefficients)
                .unwrap();
        }
        for (row, forms) in entries.iter().enumerate() {
            for (matrix, form) in forms.iter().enumerate() {
                reserved
                    .push_row(matrix, row, form.iter().copied())
                    .unwrap();
            }
        }
        let reserved = reserved.finish().unwrap();
        assert_cache_parts_equal(&actual, &reserved);
        assert_eq!(
            reserved.eval_ring_linear_forms_for_real_z_blocks(&weights, rows, &witnesses),
            wanted
        );
    }
}

#[test]
fn row_source_rejects_noncanonical_entries_and_incomplete_coverage() {
    for entries in [
        vec![(D, F::ONE)],
        vec![(0, F::ZERO)],
        vec![(1, F::ONE), (0, F::ONE)],
        vec![(0, F::ONE), (0, -F::ONE)],
    ] {
        for hint in [0, 2] {
            let mut builder = SuperneoEvalCacheBuilder::new(1, D, 1).unwrap();
            builder.reserve_matrix(0, hint, hint, hint).unwrap();
            assert!(builder.push_row(0, 0, entries.iter().copied()).is_err());
            assert!(builder.push_row(0, 0, []).is_err());
            assert!(builder.finish().is_err());
        }
    }
    let mut builder = SuperneoEvalCacheBuilder::new(1, D, 2).unwrap();
    builder.reserve_matrix(0, 0, 0, 0).unwrap();
    builder.reserve_matrix(1, 0, 0, 0).unwrap();
    builder.push_row(0, 0, []).unwrap();
    assert!(builder.finish().is_err());
    for (matrix, row) in [(1, 0), (0, 1)] {
        let mut builder = SuperneoEvalCacheBuilder::new(1, D, 2).unwrap();
        assert!(builder.push_row(matrix, row, []).is_err());
        assert!(builder.finish().is_err());
    }
    assert!(SuperneoEvalCacheBuilder::new(1, 0, 1).is_err());
    assert!(SuperneoEvalCacheBuilder::new(1, D, 0).is_err());
    let mut builder = SuperneoEvalCacheBuilder::new(1, D, 1).unwrap();
    assert!(builder.reserve_matrix(1, 0, 0, 0).is_err());
    assert!(builder.push_row(0, 0, []).is_err());
    assert!(builder.finish().is_err());
}

#[test]
fn unexpanded_runs_match_scalar_matrices_for_rows_and_openings() {
    let rows = 4;
    let columns = 2 * D + 5;
    let matrices = 3;
    let mut dense = vec![Mat::zero(rows, columns, F::ZERO); matrices];
    let mut builder = SuperneoEvalCacheBuilder::new(rows, columns, matrices).unwrap();
    let runs = [
        vec![],
        vec![GeometricRowRun::new(1, D - 2, 41, F::from_u64(7), F::from_u64(3))],
        vec![
            GeometricRowRun::new(2, 1, 41, F::ONE, F::from_u64(3)),
            GeometricRowRun::new(2, 1, 41, -F::ONE, F::from_u64(3)),
            GeometricRowRun::new(2, D + 1, columns - D - 1, F::from_u64(11), -F::ONE),
        ],
        vec![],
    ];
    for (row, row_runs) in runs.iter().enumerate() {
        for (matrix, values) in dense.iter_mut().enumerate() {
            let explicit = if row == 2 && matrix == 0 {
                vec![(D + 1, -F::from_u64(11))]
            } else {
                vec![]
            };
            let selected = if matrix == 0 { row_runs.as_slice() } else { &[] };
            for &(column, coefficient) in &explicit {
                values[(row, column)] += coefficient;
            }
            for run in selected {
                run.for_each_term(|row, column, coefficient| values[(row, column)] += coefficient);
            }
            builder
                .push_row_with_runs(matrix, row, explicit, selected.iter().cloned())
                .unwrap();
        }
    }
    let structure = CcsStructure::new(dense, SparsePoly::new(matrices, vec![])).unwrap();
    let expected = build_superneo_eval_cache(&structure).unwrap();
    let actual = builder.finish().unwrap();
    let parts = actual.matrix(0).unwrap().compact_device_parts().unwrap();
    assert_eq!(parts.geometric_runs.len(), 4);
    assert_eq!(
        parts.row_blocks.len(),
        1,
        "only the explicit scalar occupies a row block"
    );
    let values = (0..columns)
        .map(|i| K::from(F::from_u64(i as u64 + 1)))
        .collect::<Vec<_>>();
    let witnesses = [SuperneoZBlocks::from_z(&values)];
    let weights = (0..rows)
        .map(|i| K::from_coeffs([F::from_u64(i as u64 + 3), F::ONE]))
        .collect::<Vec<_>>();
    assert_eq!(
        actual.eval_ring_linear_forms_for_real_z_blocks(&weights, rows, &witnesses),
        expected.eval_ring_linear_forms_for_real_z_blocks(&weights, rows, &witnesses)
    );
    let lane_weights = std::array::from_fn(|i| K::from_coeffs([F::ONE, F::from_u64(i as u64 + 1)]));
    let matrix_weights = [K::ONE; 3];
    assert_eq!(
        actual.eval_weighted_row_table(&witnesses[0], &lane_weights, &matrix_weights, rows, rows * 2),
        expected.eval_weighted_row_table(&witnesses[0], &lane_weights, &matrix_weights, rows, rows * 2),
        "geometric-only rows must participate in the matrix mask"
    );
    assert!(matches!(
        actual
            .matrix(2)
            .unwrap()
            .compact_device_parts()
            .unwrap()
            .geometric_row_offsets,
        SuperneoCompactRowOffsets::Empty
    ));
}

#[test]
fn row_source_rejects_invalid_runs_and_cannot_finish_after_failure() {
    let mut encoded = serde_json::to_value(GeometricRowRun::new(0, 0, 1, F::ONE, F::ONE)).unwrap();
    encoded["len"] = serde_json::json!(0);
    let empty: GeometricRowRun<F> = serde_json::from_value(encoded).unwrap();
    for runs in [
        vec![empty],
        vec![GeometricRowRun::new(1, 0, 1, F::ONE, F::ONE)],
        vec![GeometricRowRun::new(0, D - 1, 2, F::ONE, F::ONE)],
        vec![GeometricRowRun::new(0, usize::MAX, 2, F::ONE, F::ONE)],
        vec![
            GeometricRowRun::new(0, 2, 1, F::ONE, F::ONE),
            GeometricRowRun::new(0, 1, 1, F::ONE, F::ONE),
        ],
    ] {
        let mut builder = SuperneoEvalCacheBuilder::new(1, D, 1).unwrap();
        assert!(builder.push_row_with_runs(0, 0, [], runs).is_err());
        assert!(builder.push_row(0, 0, []).is_err());
        assert!(builder.finish().is_err());
    }
}
