//! Spec-derived invariant tests for SuperNeoEval.spec.md
//!
//! Each test corresponds to a row in the Invariant Obligations table.

#[path = "common/mod.rs"]
mod common;

use common::seeded_rng;
use neo_ccs::{CcsMatrix, CcsStructure, CscMat, Mat, SeededPhi81LinearBlock, SparsePoly, Term};
use neo_math::{ct, superneo_bar_block, KExtensions, Rq, D, F, K};
use neo_reductions::superneo_eval::superneo_row_dot_transformed_matrix;
use p3_field::PrimeCharacteristicRing;
use rand::Rng;

/// Build a bar-transformed CcsMatrix from a dense F matrix.
fn build_bar_ccs_matrix(mat: &Mat<F>) -> neo_ccs::CcsMatrix<F> {
    let n = mat.rows();
    let m = mat.cols();
    let mut bar_data = vec![F::ZERO; n * m];

    for row in 0..n {
        let blocks = m.div_ceil(D);
        for blk in 0..blocks {
            let base = blk * D;
            let mut coeffs = [F::ZERO; D];
            for i in 0..D {
                if base + i < m {
                    coeffs[i] = mat[(row, base + i)];
                }
            }
            let bar = superneo_bar_block(coeffs);
            for i in 0..D {
                if base + i < m {
                    bar_data[row * m + base + i] = bar[i];
                }
            }
        }
    }

    let bar_mat = Mat::from_row_major(n, m, bar_data);
    neo_ccs::CcsMatrix::Csc(neo_ccs::CscMat::from_dense_row_major(&bar_mat))
}

/// Direct matrix-vector product (M * z)[row] using K extension field.
fn direct_row_dot(mat: &Mat<F>, row: usize, z: &[K]) -> K {
    let mut sum = K::ZERO;
    for c in 0..mat.cols().min(z.len()) {
        sum += K::from(mat[(row, c)]) * z[c];
    }
    sum
}

// ---------------------------------------------------------------------------
// 1. superneo_row_dot matches direct product for identity matrix
// ---------------------------------------------------------------------------

/// SuperNeoEval.spec.md: superneo_row_dot matches direct product for identity
#[test]
fn superneo_row_dot_identity() {
    let n = D;
    let mat = Mat::identity(n);
    let bar_mat = build_bar_ccs_matrix(&mat);

    let mut rng = seeded_rng(0xA001);
    let z: Vec<K> = (0..n)
        .map(|_| K::from(F::from_u64(rng.random::<u64>() % 1000)))
        .collect();

    for row in 0..n {
        let superneo = superneo_row_dot_transformed_matrix(&bar_mat, row, &z);
        assert_eq!(superneo, z[row], "identity matrix: row {row} mismatch");
    }
}

// ---------------------------------------------------------------------------
// 2. superneo_row_dot matches direct product for random matrix
// ---------------------------------------------------------------------------

/// SuperNeoEval.spec.md: superneo_row_dot matches direct for random matrix
#[test]
fn superneo_row_dot_random_base_field() {
    let mut rng = seeded_rng(0xA002);
    let n = 4;
    let m = D * 2;

    let data: Vec<F> = (0..n * m)
        .map(|_| F::from_u64(rng.random::<u64>() % 100))
        .collect();
    let mat = Mat::from_row_major(n, m, data);
    let bar_mat = build_bar_ccs_matrix(&mat);

    // z in base field (imag=0)
    let z: Vec<K> = (0..m)
        .map(|_| K::from(F::from_u64(rng.random::<u64>() % 100)))
        .collect();

    for row in 0..n {
        let superneo = superneo_row_dot_transformed_matrix(&bar_mat, row, &z);
        let direct = direct_row_dot(&mat, row, &z);
        assert_eq!(superneo, direct, "random matrix: row {row} mismatch");
    }
}

// ---------------------------------------------------------------------------
// 3. Out-of-bounds row returns K::ZERO
// ---------------------------------------------------------------------------

/// SuperNeoEval.spec.md: out-of-bounds row returns K::ZERO
#[test]
fn superneo_row_dot_out_of_bounds() {
    let n = 2;
    let m = D;
    let data = vec![F::ONE; n * m];
    let mat = Mat::from_row_major(n, m, data);
    let bar_mat = build_bar_ccs_matrix(&mat);

    let z = vec![K::ONE; m];

    let result = superneo_row_dot_transformed_matrix(&bar_mat, n + 5, &z);
    assert_eq!(result, K::ZERO, "out-of-bounds row should return K::ZERO");
}

// ---------------------------------------------------------------------------
// 4. Zero row produces K::ZERO
// ---------------------------------------------------------------------------

/// SuperNeoEval.spec.md: zero row produces K::ZERO
#[test]
fn superneo_row_dot_zero_row() {
    let n = 2;
    let m = D;
    let mut data = vec![F::ZERO; n * m];
    for i in 0..m {
        data[m + i] = F::from_u64((i + 1) as u64);
    }
    let mat = Mat::from_row_major(n, m, data);
    let bar_mat = build_bar_ccs_matrix(&mat);

    let z: Vec<K> = (0..m).map(|i| K::from(F::from_u64(i as u64 + 1))).collect();

    let result_row0 = superneo_row_dot_transformed_matrix(&bar_mat, 0, &z);
    assert_eq!(result_row0, K::ZERO, "zero row should produce K::ZERO");

    let result_row1 = superneo_row_dot_transformed_matrix(&bar_mat, 1, &z);
    assert_ne!(result_row1, K::ZERO, "nonzero row should produce nonzero result");
}

// ---------------------------------------------------------------------------
// 5. Theorem 3 kernel: ct(bar(a) * b) = <a, b>
// ---------------------------------------------------------------------------

/// SuperNeoEval.spec.md: Theorem 3 inner product via ct(bar(a)*b)
#[test]
fn theorem_3_inner_product_kernel() {
    let mut rng = seeded_rng(0xA005);

    for _ in 0..10 {
        let a: [F; D] = core::array::from_fn(|_| F::from_u64(rng.random::<u64>() % 100));
        let b: [F; D] = core::array::from_fn(|_| F::from_u64(rng.random::<u64>() % 100));

        let mut dot = F::ZERO;
        for i in 0..D {
            dot += a[i] * b[i];
        }

        let bar_a = superneo_bar_block(a);
        let bar_ring = Rq(bar_a);
        let b_ring = Rq(b);
        let product = bar_ring.mul(&b_ring);
        let ct_val = ct(&product);

        assert_eq!(ct_val, dot, "Theorem 3: ct(bar(a)*b) should equal <a,b>");
    }
}

// ---------------------------------------------------------------------------
// 6. Partial blocks handled correctly
// ---------------------------------------------------------------------------

/// SuperNeoEval.spec.md: partial block (m not multiple of D)
#[test]
fn superneo_row_dot_partial_block() {
    let n = 1;
    let m = D + 1;
    let data: Vec<F> = (0..n * m).map(|i| F::from_u64((i + 1) as u64)).collect();
    let mat = Mat::from_row_major(n, m, data);
    let bar_mat = build_bar_ccs_matrix(&mat);

    let z: Vec<K> = (0..m).map(|i| K::from(F::from_u64(i as u64 + 1))).collect();

    let superneo = superneo_row_dot_transformed_matrix(&bar_mat, 0, &z);
    let direct = direct_row_dot(&mat, 0, &z);
    assert_eq!(superneo, direct, "partial block: superneo should match direct");
}

#[test]
fn seeded_phi81_cache_matches_expanded_matrix_on_every_evaluation_surface() {
    let seed = [0x5C; 32];
    let kappa = 1;
    let rows = D;
    let word_width = 41;
    let cols = D * 3;
    let word_starts = vec![1, 42, 83];
    let message_cols = (word_starts.len() * word_width).div_ceil(D);
    let (chunk_size, chunk_seeds) = neo_ajtai::seeded_pp_chunk_seeds(seed, kappa, message_cols);
    let block = SeededPhi81LinearBlock::new_with_word_width(
        0,
        word_starts,
        word_width,
        kappa,
        message_cols,
        chunk_size,
        chunk_seeds,
    )
    .unwrap();
    let compact_matrix =
        CcsMatrix::csc_with_seeded_phi81(CscMat::from_triplets(Vec::new(), rows, cols), vec![block.clone()]).unwrap();
    let mut expanded_trips = Vec::new();
    block.for_each_term::<F, _>(|row, col, value| expanded_trips.push((row, col, value)));
    let expanded_matrix = CcsMatrix::Csc(CscMat::from_triplets(expanded_trips, rows, cols));
    let polynomial = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let compact = CcsStructure::new_sparse(vec![compact_matrix], polynomial.clone()).unwrap();
    let expanded = CcsStructure::new_sparse(vec![expanded_matrix], polynomial).unwrap();
    let compact_cache = neo_reductions::superneo_eval::build_superneo_eval_cache(&compact).unwrap();
    let expanded_cache = neo_reductions::superneo_eval::build_superneo_eval_cache(&expanded).unwrap();

    let z: Vec<K> = (0..cols)
        .map(|col| match (col * 31 + 7) % 3 {
            0 => -K::ONE,
            1 => K::ZERO,
            _ => K::ONE,
        })
        .collect();
    let chi: Vec<K> = (0..rows)
        .map(|row| K::from_coeffs([F::from_u64((row * 13 + 1) as u64), F::from_u64((row * 7 + 3) as u64)]))
        .collect();
    assert_eq!(
        neo_reductions::superneo_eval::eval_all_mats_cached(&compact_cache, &z, &chi, rows),
        neo_reductions::superneo_eval::eval_all_mats_cached(&expanded_cache, &z, &chi, rows)
    );
    assert_eq!(
        neo_reductions::superneo_eval::eval_all_mats_ring_cached(&compact_cache, &z, &chi, rows),
        neo_reductions::superneo_eval::eval_all_mats_ring_cached(&expanded_cache, &z, &chi, rows)
    );

    let compact_linear = compact_cache.build_linear_forms(&chi, rows);
    let expanded_linear = expanded_cache.build_linear_forms(&chi, rows);
    assert_eq!(compact_linear[0].eval_vec_k(&z), expanded_linear[0].eval_vec_k(&z));

    let compact_ring = compact_cache.build_ring_linear_forms(&chi, rows);
    let expanded_ring = expanded_cache.build_ring_linear_forms(&chi, rows);
    let seeded_ring = compact_cache.build_seeded_ring_linear_forms(&chi, rows);
    assert_eq!(
        seeded_ring[0].to_dense_block_coeffs(),
        compact_ring[0].to_dense_block_coeffs(),
        "seeded-only forms must exactly reconstruct a purely seeded matrix"
    );
    let row_challenges: Vec<K> = (0..6)
        .map(|index| {
            K::from_coeffs([
                F::from_u64((index * 17 + 2) as u64),
                F::from_u64((index * 11 + 5) as u64),
            ])
        })
        .collect();
    let tensor_chi = neo_ccs::utils::tensor_point_parallel::<K>(&row_challenges);
    let seeded_from_tensor = compact_cache.build_seeded_ring_linear_forms(&tensor_chi, rows);
    let seeded_from_challenges =
        compact_cache.build_seeded_ring_linear_forms_from_row_challenges(&row_challenges, rows);
    assert_eq!(
        seeded_from_challenges[0].to_dense_block_coeffs(),
        seeded_from_tensor[0].to_dense_block_coeffs(),
        "selective seeded-row evaluation must match the full tensor table"
    );
    let z_blocks = neo_reductions::superneo_eval::SuperneoZBlocks::from_z(&z);
    let mut compact_rows = vec![K::ZERO; rows];
    let mut expanded_rows = vec![K::ZERO; rows];
    compact_cache
        .matrix(0)
        .unwrap()
        .fill_row_dots_real_with_blocks(&mut compact_rows, &z_blocks);
    expanded_cache
        .matrix(0)
        .unwrap()
        .fill_row_dots_real_with_blocks(&mut expanded_rows, &z_blocks);
    assert_eq!(compact_rows, expanded_rows);
    assert_eq!(
        neo_reductions::superneo_eval::eval_ring_linear_forms_real_z_blocks(&compact_ring, &z_blocks),
        neo_reductions::superneo_eval::eval_ring_linear_forms_real_z_blocks(&expanded_ring, &z_blocks)
    );

    let weights: [K; D] = core::array::from_fn(|lane| K::from(F::from_u64((lane + 1) as u64)));
    let compact_weighted = compact_cache.build_weighted_matrix_caches(&weights);
    let expanded_weighted = expanded_cache.build_weighted_matrix_caches(&weights);
    for row in 0..rows {
        assert_eq!(
            compact_weighted[0].row_dot_with_blocks(row, &z_blocks),
            expanded_weighted[0].row_dot_with_blocks(row, &z_blocks),
            "weighted row {row}"
        );
    }

    let compact_transformed = compact.transform_matrices_superneo().unwrap();
    let expanded_transformed = expanded.transform_matrices_superneo().unwrap();
    let compact_transformed_cache =
        neo_reductions::superneo_eval::build_superneo_eval_cache(&compact_transformed).unwrap();
    let expanded_transformed_cache =
        neo_reductions::superneo_eval::build_superneo_eval_cache(&expanded_transformed).unwrap();
    let mut compact_transformed_rows = vec![K::ZERO; rows];
    let mut expanded_transformed_rows = vec![K::ZERO; rows];
    compact_transformed_cache
        .matrix(0)
        .unwrap()
        .fill_row_dots_real_with_blocks(&mut compact_transformed_rows, &z_blocks);
    expanded_transformed_cache
        .matrix(0)
        .unwrap()
        .fill_row_dots_real_with_blocks(&mut expanded_transformed_rows, &z_blocks);
    assert_eq!(compact_transformed_rows, expanded_transformed_rows);
    let compact_transformed_ring = compact_transformed_cache.build_ring_linear_forms(&chi, rows);
    let expanded_transformed_ring = expanded_transformed_cache.build_ring_linear_forms(&chi, rows);
    assert_eq!(
        neo_reductions::superneo_eval::eval_ring_linear_forms_real_z_blocks(&compact_transformed_ring, &z_blocks),
        neo_reductions::superneo_eval::eval_ring_linear_forms_real_z_blocks(&expanded_transformed_ring, &z_blocks)
    );

    let complex_z: Vec<K> = z
        .iter()
        .enumerate()
        .map(|(column, value)| {
            *value
                * K::from_coeffs([
                    F::from_u64((column % 11 + 1) as u64),
                    F::from_u64((column % 7 + 1) as u64),
                ])
        })
        .collect();
    let complex_blocks = neo_reductions::superneo_eval::SuperneoZBlocks::from_z(&complex_z);
    assert!(!complex_blocks.imag_all_zero());
    let matrix_coeffs = [K::from_coeffs([F::from_u64(5), F::from_u64(9)])];
    assert_eq!(
        compact_cache.eval_weighted_row_table(&complex_blocks, &weights, &matrix_coeffs, rows, rows),
        expanded_cache.eval_weighted_row_table(&complex_blocks, &weights, &matrix_coeffs, rows, rows),
        "complex carried witnesses must retain compact seeded rows"
    );
    assert_eq!(
        compact_transformed_cache.eval_weighted_row_table(&complex_blocks, &weights, &matrix_coeffs, rows, rows),
        expanded_transformed_cache.eval_weighted_row_table(&complex_blocks, &weights, &matrix_coeffs, rows, rows),
        "complex carried witnesses must retain transformed compact seeded rows"
    );
}

#[test]
fn seeded_phi81_matrix_digest_binds_input_word_width() {
    let seed = [0x6D; 32];
    let kappa = 1;
    let message_cols = 1;
    let (chunk_size, chunk_seeds) = neo_ajtai::seeded_pp_chunk_seeds(seed, kappa, message_cols);
    let block = |word_width| {
        SeededPhi81LinearBlock::new_with_word_width(
            0,
            vec![1],
            word_width,
            kappa,
            message_cols,
            chunk_size,
            chunk_seeds.clone(),
        )
        .unwrap()
    };
    let polynomial = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let structure = |block| {
        CcsStructure::new_sparse(
            vec![CcsMatrix::csc_with_seeded_phi81(CscMat::from_triplets(Vec::new(), D, D + 1), vec![block]).unwrap()],
            polynomial.clone(),
        )
        .unwrap()
    };

    let balanced = structure(block(41));
    let wider = structure(block(54));
    assert_ne!(
        neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(&balanced, None),
        neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(&wider, None),
    );
}
