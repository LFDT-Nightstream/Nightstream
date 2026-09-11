use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::{
    build_superneo_eval_cache, SuperneoCompactRowOffsets, SuperneoEvalCacheBuilder, SuperneoZBlocks,
};
use p3_field::PrimeCharacteristicRing;

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
        assert_eq!(
            reserved
                .finish()
                .unwrap()
                .eval_ring_linear_forms_for_real_z_blocks(&weights, rows, &witnesses),
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
