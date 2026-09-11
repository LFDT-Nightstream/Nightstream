use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat, SparsePoly, V1_1Evaluations};
use neo_math::{superneo_bar_block, KExtensions, Rq, D, F, K};
use neo_params::NeoParams;
use neo_reductions::common::split_b_matrix_k_with_nonzero_flags;
use neo_reductions::optimized_engine::dec_reduction_optimized_with_digit_flags;
use neo_reductions::superneo_eval::{SuperneoEvalCacheBuilder, SuperneoZBlocks};
use p3_field::PrimeCharacteristicRing;

fn weight(point: &[K], row: usize) -> K {
    point
        .iter()
        .enumerate()
        .fold(K::ONE, |product, (bit, &coordinate)| {
            product
                * if row & (1 << bit) == 0 {
                    K::ONE - coordinate
                } else {
                    coordinate
                }
        })
}

fn matrix(values: &[F]) -> Mat<F> {
    let blocks = values.len() / D;
    Mat::from_row_major(
        D,
        blocks,
        (0..D)
            .flat_map(|lane| (0..blocks).map(move |block| values[block * D + lane]))
            .collect(),
    )
}

fn direct(matrices: &[Mat<F>], values: &[F], point: &[K]) -> V1_1Evaluations<K> {
    let mut pad = vec![K::ZERO; D];
    for row in 0..values.len() {
        let mut basis = [F::ZERO; D];
        basis[row % D] = F::ONE;
        let source = Rq(std::array::from_fn(|lane| values[row / D * D + lane]));
        let product = Rq(superneo_bar_block(basis)).mul(&source);
        for lane in 0..D {
            pad[lane] += weight(point, row) * K::from(product.0[lane]);
        }
    }
    let eval_a = matrices
        .iter()
        .map(|matrix| {
            let mut output = vec![K::ZERO; D];
            for row in 0..matrix.rows() {
                for block in 0..values.len() / D {
                    let original = std::array::from_fn(|lane| {
                        let column = block * D + lane;
                        if column < matrix.cols() {
                            matrix[(row, column)]
                        } else {
                            F::ZERO
                        }
                    });
                    let source = Rq(std::array::from_fn(|lane| values[block * D + lane]));
                    let product = Rq(superneo_bar_block(original)).mul(&source);
                    for lane in 0..D {
                        output[lane] += weight(point, row) * K::from(product.0[lane]);
                    }
                }
            }
            output
        })
        .collect();
    V1_1Evaluations { eval_k: pad, eval_a }
}

#[test]
fn real_witness_openings_match_direct_ring_rows() {
    let rows = 5;
    let logical = D + 3;
    let width = 2 * D;
    let mut matrices = vec![Mat::zero(rows, logical, F::ZERO); 14];
    for (index, matrix) in matrices.iter_mut().take(13).enumerate() {
        matrix[(index % rows, index * 7 % logical)] = F::from_u64(index as u64 + 2);
        matrix[(rows - 1, logical - 1)] = -F::from_u64(index as u64 + 1);
    }
    let structure = CcsStructure::new(matrices.clone(), SparsePoly::new(matrices.len(), vec![])).unwrap();
    let mut builder = SuperneoEvalCacheBuilder::new(rows, logical, matrices.len()).unwrap();
    for row in 0..rows {
        for (index, matrix) in matrices.iter().enumerate() {
            builder
                .push_row(
                    index,
                    row,
                    (0..logical).filter_map(|column| {
                        let value = matrix[(row, column)];
                        (value != F::ZERO).then_some((column, value))
                    }),
                )
                .unwrap();
        }
    }
    let cache = builder.finish().unwrap();
    let point = (0..width.next_power_of_two().trailing_zeros())
        .map(|index| K::from_coeffs([F::from_u64(u64::from(index) + 3), F::ONE]))
        .collect::<Vec<_>>();
    let mut parent_values = vec![F::ZERO; width];
    parent_values[0] = F::from_u64(257);
    parent_values[logical - 1] = -F::from_u64(9);
    parent_values[width - 1] = -F::from_u64(511); // Nonzero mixed-parent carrier tail.
    let parent_witness = matrix(&parent_values);
    let mut fresh_values = vec![F::ZERO; width];
    fresh_values[0] = F::ONE;
    fresh_values[logical - 1] = -F::ONE; // Higher Pad terms need the remaining zero lanes.
    let fresh_witness = Mat::compact_signed_unit(D, 2, matrix(&fresh_values).to_dense_vec());
    let zero_witness = Mat::virtual_constant(D, 2, F::ZERO);
    let blocks = [&parent_witness, &fresh_witness, &zero_witness]
        .into_iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, logical).unwrap())
        .collect::<Vec<_>>();
    let evaluated = cache.eval_real_v1_1_openings(&point, &blocks).unwrap();
    let zero_values = vec![F::ZERO; width];
    for (actual, values) in evaluated
        .iter()
        .zip([&parent_values, &fresh_values, &zero_values])
    {
        let expected = direct(&matrices, values, &point);
        assert_eq!(actual.eval_k, expected.eval_k);
        assert_eq!(actual.eval_a, expected.eval_a);
        assert_eq!(actual.eval_a[13], vec![K::ZERO; D]);
    }

    // Exercise the default normal D path with no supplied openings or forms.
    // This test owns evaluation arithmetic; fixed-key commitments have their
    // separate selected-source test.
    let params = NeoParams::nightstream_goldilocks_k16();
    let parent_opening = direct(&matrices, &parent_values, &point);
    let pad = |mut values: Vec<K>| {
        values.resize(D.next_power_of_two(), K::ZERO);
        values
    };
    let parent = CeClaim {
        c: Commitment::zeros(D, params.kappa as usize),
        X: Mat::from_row_major(D, 1, parent_values[..D].to_vec()),
        r: point.clone(),
        eval_k: pad(parent_opening.eval_k),
        eval_a: parent_opening.eval_a.into_iter().map(pad).collect(),
        m_in: D,
        fold_digest: [0; 32],
        adv: None,
    };
    let (digits, flags) = split_b_matrix_k_with_nonzero_flags(&parent_witness, params.k_rho as usize, 2).unwrap();
    let (children, evaluations_ok, public_ok) = dec_reduction_optimized_with_digit_flags(
        &structure,
        &params,
        &parent,
        &digits,
        &flags,
        D.next_power_of_two().trailing_zeros() as usize,
        Some(&cache),
        None,
        None,
    );
    assert!(evaluations_ok && public_ok);
    assert_eq!(children.len(), params.k_rho as usize);
    for (child, digit) in children.iter().zip(&digits) {
        let values = (0..width)
            .map(|column| digit[(column % D, column / D)])
            .collect::<Vec<_>>();
        let expected = direct(&matrices, &values, &point);
        assert_eq!(child.eval_k, pad(expected.eval_k));
        assert_eq!(child.eval_a, expected.eval_a.into_iter().map(pad).collect::<Vec<_>>());
    }

    assert!(cache.eval_real_v1_1_openings(&point[1..], &blocks).is_err());
    assert!(cache
        .eval_real_v1_1_openings(&point, &[SuperneoZBlocks::with_block_len(1)])
        .is_err());
    let imaginary = SuperneoZBlocks::from_z(&vec![K::from_coeffs([F::ZERO, F::ONE]); width]);
    assert!(cache.eval_real_v1_1_openings(&point, &[imaginary]).is_err());
}
