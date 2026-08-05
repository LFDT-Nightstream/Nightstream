use neo_ccs::{
    matrix::Mat, poly::SparsePoly, CcsMatrix, CcsStructure, CscMat, GeometricRowRun, SeededPhi81LinearBlock, Term,
};
use neo_math::{superneo_bar_block, KExtensions, Rq};
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::{
    build_superneo_eval_cache, eval_all_mats_cached, eval_all_mats_cached_with_blocks, eval_all_mats_direct,
    eval_all_mats_ring_cached, eval_all_mats_ring_cached_with_blocks, eval_all_mats_transformed,
    eval_ring_linear_forms_real_z_blocks, SuperneoZBlocks,
};
use p3_field::PrimeCharacteristicRing;

fn chi_table(point: &[K]) -> Vec<K> {
    let n = 1usize << point.len();
    let mut out = vec![K::ZERO; n];
    for (idx, out_cell) in out.iter_mut().enumerate().take(n) {
        let mut w = K::ONE;
        for (bit, p) in point.iter().copied().enumerate() {
            let is_one = ((idx >> bit) & 1) == 1;
            w *= if is_one { p } else { K::ONE - p };
        }
        *out_cell = w;
    }
    out
}

fn rq_dot(lhs: &Rq, rhs: &Rq) -> F {
    lhs.0
        .iter()
        .zip(rhs.0.iter())
        .fold(F::ZERO, |acc, (&a, &b)| acc + a * b)
}

#[test]
fn column_mask_witness_blocks_match_row_major_signed_unit_blocks() {
    let cols = 5;
    let mut values = vec![F::ZERO; D * cols];
    let mut positive = vec![0u64; cols];
    let mut negative = vec![0u64; cols];
    for block in 0..cols {
        for lane in 0..D {
            let value = match (block * 11 + lane * 7) % 5 {
                0 => F::ONE,
                1 => F::ZERO - F::ONE,
                _ => F::ZERO,
            };
            values[lane * cols + block] = value;
            if value == F::ONE {
                positive[block] |= 1u64 << lane;
            } else if value == F::ZERO - F::ONE {
                negative[block] |= 1u64 << lane;
            }
        }
    }

    let row_major = Mat::compact_signed_unit(D, cols, values);
    let column_masks: Mat<F> = Mat::compact_signed_unit_from_column_masks(D, cols, &positive, &negative)
        .expect("valid signed-unit column masks");
    let expected_m = D * cols - 3;
    let row_major_blocks = SuperneoZBlocks::from_witness_mat(&row_major, expected_m).expect("row-major blocks");
    let column_mask_blocks = SuperneoZBlocks::from_witness_mat(&column_masks, expected_m).expect("column-mask blocks");

    assert_eq!(column_mask_blocks.re_plane_words(), row_major_blocks.re_plane_words());
    assert_eq!(column_mask_blocks.im_plane_words(), row_major_blocks.im_plane_words());
}

#[test]
fn transformed_eval_matches_direct_eval_for_sparse_mats() {
    let n = 8usize;
    let m = 2 * D;

    let mut m0 = Mat::zero(n, m, F::ZERO);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        for c in 0..m {
            if ((r * 17) + (c * 13)) % 19 == 0 {
                m0[(r, c)] = F::from_u64(((r + c) % 11 + 1) as u64);
            }
            if ((r * 7) + (c * 5)) % 23 == 0 {
                m1[(r, c)] = F::from_u64(((2 * r + c) % 13 + 1) as u64);
            }
        }
    }

    let s = CcsStructure::new(vec![m0, m1], SparsePoly::new(2, vec![])).expect("valid CCS");
    let s_bar = s.transform_matrices_superneo().expect("superneo transform");

    let z: Vec<K> = (0..m)
        .map(|i| K::from_coeffs([F::from_u64((i % 17 + 1) as u64), F::from_u64((i % 7) as u64)]))
        .collect();
    let r = vec![
        K::from_coeffs([F::from_u64(3), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(5), F::from_u64(2)]),
        K::from_coeffs([F::from_u64(7), F::from_u64(0)]),
    ];
    let chi_r = chi_table(&r);

    let direct = eval_all_mats_direct(&s, &z, &chi_r, n);
    let via_bar = eval_all_mats_transformed(&s_bar, &z, &chi_r, n);
    assert_eq!(direct, via_bar);
}

#[test]
fn geometric_rows_match_expanded_csc_in_direct_cached_and_transformed_evaluation() {
    let n = 4usize;
    let m = 2 * D;
    let run = GeometricRowRun::new(2, D - 9, 41, F::from_u64(7), F::from_u64(3));
    let mut expanded = Vec::new();
    run.for_each_term(|row, column, value| expanded.push((row, column, value)));
    expanded.push((0, 0, F::from_u64(11)));

    let expanded_matrix = CcsMatrix::Csc(CscMat::from_triplets(expanded, n, m));
    let structured_matrix = CcsMatrix::csc_with_compact_rows(
        CscMat::from_triplets(vec![(0, 0, F::from_u64(11))], n, m),
        Vec::new(),
        vec![run],
    )
    .expect("valid compact matrix");
    let polynomial = SparsePoly::new(1, vec![]);
    let expanded = CcsStructure::new_sparse(vec![expanded_matrix], polynomial.clone()).expect("expanded structure");
    let structured = CcsStructure::new_sparse(vec![structured_matrix], polynomial).expect("structured relation");

    let z = (0..m)
        .map(|index| K::from_coeffs([F::from_u64((index + 1) as u64), F::from_u64((index % 5) as u64)]))
        .collect::<Vec<_>>();
    let point = [K::from_coeffs([F::from_u64(3), F::from_u64(1)]); 2];
    let chi_r = chi_table(&point);
    let expected = eval_all_mats_direct(&expanded, &z, &chi_r, n);
    assert_eq!(eval_all_mats_direct(&structured, &z, &chi_r, n), expected);

    let cache = build_superneo_eval_cache(&structured).expect("structured SuperNeo cache");
    assert_eq!(eval_all_mats_cached(&cache, &z, &chi_r, n), expected);

    let transformed = structured
        .transform_matrices_superneo()
        .expect("transform structured matrix");
    assert_eq!(eval_all_mats_transformed(&transformed, &z, &chi_r, n), expected);
}

#[test]
fn transformed_eval_matches_direct_eval_for_identity_sentinel() {
    let n = D;
    let s = CcsStructure::new(vec![Mat::identity(n)], SparsePoly::new(1, vec![])).expect("valid identity CCS");
    let s_bar = s.transform_matrices_superneo().expect("superneo transform");

    let z: Vec<K> = (0..n)
        .map(|i| K::from_coeffs([F::from_u64((3 * i as u64) + 1), F::from_u64(i as u64 % 5)]))
        .collect();
    let r = vec![
        K::from_coeffs([F::from_u64(2), F::from_u64(0)]),
        K::from_coeffs([F::from_u64(3), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(5), F::from_u64(0)]),
        K::from_coeffs([F::from_u64(7), F::from_u64(2)]),
        K::from_coeffs([F::from_u64(11), F::from_u64(0)]),
        K::from_coeffs([F::from_u64(13), F::from_u64(1)]),
    ];
    let chi_r = chi_table(&r);

    let direct = eval_all_mats_direct(&s, &z, &chi_r, n);
    let via_bar = eval_all_mats_transformed(&s_bar, &z, &chi_r, n);
    assert_eq!(direct, via_bar);
}

#[test]
fn weighted_cache_coefficients_match_monomial_shift_dot() {
    let n = 1usize;
    let m = D;
    let mut row = [F::ZERO; D];
    for (i, slot) in row.iter_mut().enumerate() {
        *slot = F::from_u64(((17 * i + 5) % 29 + 1) as u64);
    }

    let mat = Mat::from_row_major(n, m, row.to_vec());
    let s = CcsStructure::new(vec![mat], SparsePoly::new(1, vec![])).expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("superneo-compatible width");
    let matrix = cache.matrix(0).expect("matrix cache");

    let mut weights = [K::ZERO; D];
    let mut weight_re = [F::ZERO; D];
    let mut weight_im = [F::ZERO; D];
    for i in 0..D {
        weight_re[i] = F::from_u64(((11 * i + 3) % 31 + 1) as u64);
        weight_im[i] = F::from_u64(((7 * i + 9) % 37 + 1) as u64);
        weights[i] = K::from_coeffs([weight_re[i], weight_im[i]]);
    }

    let weighted = matrix.compile_weighted_rows(&weights);
    let bar = Rq(superneo_bar_block(row));
    let weight_re = Rq(weight_re);
    let weight_im = Rq(weight_im);

    for basis in 0..D {
        let mut z = [F::ZERO; D];
        z[basis] = F::ONE;
        let z_blocks = SuperneoZBlocks::from_base_row_f(&z);
        let observed = weighted.row_dot_real_with_blocks(0, &z_blocks);

        let shifted = bar.mul_by_monomial(basis);
        let expected = K::from_coeffs([rq_dot(&weight_re, &shifted), rq_dot(&weight_im, &shifted)]);
        assert_eq!(observed, expected, "basis {basis}");
    }
}

#[test]
fn weighted_matrix_cache_matches_direct_weighted_ring_rows_for_complex_witness() {
    let n = 8usize;
    let m = 2 * D;

    let mut mat = Mat::zero(n, m, F::ZERO);
    for row in 0..n {
        for col in 0..m {
            if ((row * 17) + (col * 5)) % 19 == 0 {
                mat[(row, col)] = F::from_u64(((3 * row + 2 * col) % 31 + 1) as u64);
            }
        }
    }

    let s = CcsStructure::new(vec![mat], SparsePoly::new(1, vec![])).expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("cache should build for D-compatible width");
    let matrix = cache.matrix(0).expect("matrix cache");

    let mut weights = [K::ZERO; D];
    for (rho, weight) in weights.iter_mut().enumerate() {
        *weight = K::from_coeffs([
            F::from_u64(((11 * rho + 3) % 37 + 1) as u64),
            F::from_u64(((7 * rho + 5) % 41 + 1) as u64),
        ]);
    }
    let weighted = matrix.compile_weighted_rows(&weights);

    let z: Vec<K> = (0..m)
        .map(|col| {
            K::from_coeffs([
                F::from_u64(((13 * col + 2) % 29 + 1) as u64),
                F::from_u64(((5 * col + 7) % 23 + 1) as u64),
            ])
        })
        .collect();
    let z_blocks = SuperneoZBlocks::from_z(&z);
    assert!(!z_blocks.imag_all_zero());

    for row in 0..n {
        let direct = matrix.row_dot_ring_weighted_with_blocks(row, &z_blocks, &weights);
        let cached = weighted.row_dot_with_blocks(row, &z_blocks);
        assert_eq!(direct, cached, "row {row}");
    }
}

#[test]
fn weighted_row_table_matches_direct_weighted_rows_for_complex_witness() {
    let n = 16usize;
    let n_pad = 32usize;
    let m = 3 * D;

    let mut m0 = Mat::zero(n, m, F::ZERO);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    let mut m2 = Mat::zero(n, m, F::ZERO);
    for row in 0..n {
        for col in 0..m {
            if ((row * 17) + (col * 5)) % 19 == 0 {
                m0[(row, col)] = F::from_u64(((3 * row + 2 * col) % 31 + 1) as u64);
            }
            if ((row * 11) + (col * 13)) % 29 == 0 {
                m1[(row, col)] = F::from_u64(((row + 5 * col) % 37 + 1) as u64);
            }
            if ((row * 7) + (col * 3)) % 23 == 0 {
                m2[(row, col)] = F::from_u64(((2 * row + 7 * col) % 41 + 1) as u64);
            }
        }
    }

    let s = CcsStructure::new(vec![m0, m1, m2], SparsePoly::new(3, vec![])).expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("cache should build for D-compatible width");

    let mut weights = [K::ZERO; D];
    for (rho, weight) in weights.iter_mut().enumerate() {
        *weight = K::from_coeffs([
            F::from_u64(((11 * rho + 3) % 37 + 1) as u64),
            F::from_u64(((7 * rho + 5) % 41 + 1) as u64),
        ]);
    }
    let mat_coeffs = vec![
        K::from_coeffs([F::from_u64(3), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(5), F::from_u64(2)]),
        K::from_coeffs([F::ZERO, F::ZERO]),
    ];

    let z: Vec<K> = (0..m)
        .map(|col| {
            K::from_coeffs([
                F::from_u64(((13 * col + 2) % 29 + 1) as u64),
                F::from_u64(((5 * col + 7) % 23 + 1) as u64),
            ])
        })
        .collect();
    let z_blocks = SuperneoZBlocks::from_z(&z);
    assert!(!z_blocks.imag_all_zero());

    let mut direct = vec![K::ZERO; n_pad];
    for (row, out) in direct.iter_mut().take(n).enumerate() {
        let mut row_acc = K::ZERO;
        for (j, coeff) in mat_coeffs.iter().copied().enumerate() {
            if coeff == K::ZERO {
                continue;
            }
            let matrix = cache.matrix(j).expect("matrix cache");
            let y_alpha = matrix.row_dot_ring_weighted_with_blocks(row, &z_blocks, &weights);
            if y_alpha != K::ZERO {
                row_acc += coeff * y_alpha;
            }
        }
        *out = row_acc;
    }

    let projected = cache.eval_weighted_row_table(&z_blocks, &weights, &mat_coeffs, n, n_pad);
    assert_eq!(projected, direct);
}

#[test]
fn weighted_row_table_supports_more_rows_than_columns_without_identity() {
    let rows = 2 * D;
    let columns = D;
    let mut matrix = Mat::zero(rows, columns, F::ZERO);
    for row in 0..rows {
        matrix[(row, row % columns)] = F::from_u64((row + 1) as u64);
    }
    let structure = CcsStructure::new(vec![matrix], SparsePoly::new(1, vec![])).expect("valid rectangular CCS");
    let cache = build_superneo_eval_cache(&structure).expect("rectangular cache");
    let weights = core::array::from_fn(|index| {
        K::from_coeffs([F::from_u64((index + 2) as u64), F::from_u64((2 * index + 1) as u64)])
    });
    let z = (0..columns)
        .map(|column| {
            K::from_coeffs([
                F::from_u64((3 * column + 1) as u64),
                F::from_u64((5 * column + 2) as u64),
            ])
        })
        .collect::<Vec<_>>();
    let z_blocks = SuperneoZBlocks::from_z(&z);
    let expected = (0..rows)
        .map(|row| {
            cache
                .matrix(0)
                .expect("matrix cache")
                .row_dot_ring_weighted_with_blocks(row, &z_blocks, &weights)
        })
        .collect::<Vec<_>>();

    assert_eq!(
        cache.eval_weighted_row_table(&z_blocks, &weights, &[K::ONE], rows, rows),
        expected
    );
}

#[test]
fn weighted_identity_projection_matches_ring_formula() {
    let s = CcsStructure::new(vec![Mat::<F>::identity(D)], SparsePoly::new(1, vec![])).expect("identity CCS");
    let cache = build_superneo_eval_cache(&s).expect("identity cache");
    let weights = core::array::from_fn(|index| {
        K::from_coeffs([F::from_u64((3 * index + 5) as u64), F::from_u64((5 * index + 2) as u64)])
    });
    let z = core::array::from_fn::<_, D, _>(|index| {
        K::from_coeffs([
            F::from_u64((7 * index + 11) as u64),
            F::from_u64((11 * index + 3) as u64),
        ])
    });
    let z_blocks = SuperneoZBlocks::from_z(&z);
    let direct: [K; D] = core::array::from_fn(|row| {
        cache
            .matrix(0)
            .expect("identity matrix")
            .row_dot_ring_weighted_with_blocks(row, &z_blocks, &weights)
    });

    let weight_re = Rq(weights.map(|value| value.real()));
    let weight_im = Rq(weights.map(|value| value.imag()));
    let z_re = Rq(z.map(|value| value.real()));
    let z_im = Rq(z.map(|value| value.imag()));
    let rr = Rq(superneo_bar_block(weight_re.0)).mul(&z_re);
    let ir = Rq(superneo_bar_block(weight_im.0)).mul(&z_re);
    let ri = Rq(superneo_bar_block(weight_re.0)).mul(&z_im);
    let ii = Rq(superneo_bar_block(weight_im.0)).mul(&z_im);
    let extension_generator = K::from_coeffs([F::ZERO, F::ONE]);
    let expected = core::array::from_fn(|local| {
        K::from_coeffs([rr.0[local], ir.0[local]]) + extension_generator * K::from_coeffs([ri.0[local], ii.0[local]])
    });
    assert_eq!(direct, expected);
    assert_eq!(
        cache.eval_weighted_row_table(&z_blocks, &weights, &[K::ONE], D, D),
        direct
    );
}

#[test]
fn cached_superneo_eval_matches_direct_eval_for_sparse_mats() {
    let n = 32usize;
    let m = 2 * D;

    let mut m0 = Mat::zero(n, m, F::ZERO);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        for c in 0..m {
            if ((r * 13) + (c * 7)) % 17 == 0 {
                m0[(r, c)] = F::from_u64(((r + 2 * c) % 19 + 1) as u64);
            }
            if ((r * 5) + (c * 11)) % 29 == 0 {
                m1[(r, c)] = F::from_u64(((3 * r + c) % 23 + 1) as u64);
            }
        }
    }

    let s = CcsStructure::new(vec![m0, m1], SparsePoly::new(2, vec![])).expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("cache should build for D-compatible width");

    let z: Vec<K> = (0..m)
        .map(|i| K::from_coeffs([F::from_u64((i % 31 + 1) as u64), F::from_u64((i % 9) as u64)]))
        .collect();
    let r = vec![
        K::from_coeffs([F::from_u64(2), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(3), F::from_u64(0)]),
        K::from_coeffs([F::from_u64(5), F::from_u64(2)]),
        K::from_coeffs([F::from_u64(7), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(11), F::from_u64(0)]),
    ];
    let chi_r = chi_table(&r);

    let direct = eval_all_mats_direct(&s, &z, &chi_r, n);
    let cached = eval_all_mats_cached(&cache, &z, &chi_r, n);
    assert_eq!(direct, cached);

    let linear_forms = cache.build_linear_forms(&chi_r, n);
    let via_linear_forms: Vec<K> = linear_forms.iter().map(|lf| lf.eval_vec_k(&z)).collect();
    assert_eq!(cached, via_linear_forms);
}

#[test]
fn cached_superneo_linear_forms_match_base_row_evals() {
    let n = 32usize;
    let m = 2 * D;

    let mut m0 = Mat::zero(n, m, F::ZERO);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        for c in 0..m {
            if ((r * 29) + (c * 3)) % 31 == 0 {
                m0[(r, c)] = F::from_u64(((r + c) % 17 + 1) as u64);
            }
            if ((r * 11) + (c * 5)) % 37 == 0 {
                m1[(r, c)] = F::from_u64(((2 * r + c) % 19 + 1) as u64);
            }
        }
    }

    let s = CcsStructure::new(vec![m0, m1], SparsePoly::new(2, vec![])).expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("cache should build for D-compatible width");
    let r = vec![
        K::from_coeffs([F::from_u64(2), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(3), F::from_u64(0)]),
        K::from_coeffs([F::from_u64(5), F::from_u64(2)]),
        K::from_coeffs([F::from_u64(7), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(11), F::from_u64(0)]),
    ];
    let chi_r = chi_table(&r);
    let linear_forms = cache.build_linear_forms(&chi_r, n);

    let zi = {
        let mut data = Vec::with_capacity(D * m);
        for rho in 0..D {
            for c in 0..m {
                data.push(F::from_u64((1000 + 17 * rho as u64 + c as u64) % 257));
            }
        }
        Mat::from_row_major(D, m, data)
    };

    for rho in 0..D {
        let row = zi.row(rho);
        let z_row_k: Vec<K> = row.iter().copied().map(K::from).collect();
        let cached = eval_all_mats_cached(&cache, &z_row_k, &chi_r, n);
        let via_linear_forms: Vec<K> = linear_forms
            .iter()
            .map(|lf| lf.eval_vec_base_f(row))
            .collect();
        assert_eq!(cached, via_linear_forms);
    }
}

#[test]
fn cached_superneo_ring_constant_term_matches_scalar_eval() {
    let n = 32usize;
    let m = 2 * D;

    let mut m0 = Mat::zero(n, m, F::ZERO);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        for c in 0..m {
            if ((r * 23) + (c * 9)) % 31 == 0 {
                m0[(r, c)] = F::from_u64(((r + 2 * c) % 29 + 1) as u64);
            }
            if ((r * 7) + (c * 13)) % 37 == 0 {
                m1[(r, c)] = F::from_u64(((3 * r + c) % 31 + 1) as u64);
            }
        }
    }

    let s = CcsStructure::new(vec![m0, m1], SparsePoly::new(2, vec![])).expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("cache should build for D-compatible width");

    let z: Vec<K> = (0..m)
        .map(|i| K::from_coeffs([F::from_u64((i % 31 + 1) as u64), F::from_u64((i % 11 + 2) as u64)]))
        .collect();
    let r = vec![
        K::from_coeffs([F::from_u64(2), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(3), F::from_u64(0)]),
        K::from_coeffs([F::from_u64(5), F::from_u64(2)]),
        K::from_coeffs([F::from_u64(7), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(11), F::from_u64(0)]),
    ];
    let chi_r = chi_table(&r);

    let scalar = eval_all_mats_cached(&cache, &z, &chi_r, n);
    let ring = eval_all_mats_ring_cached(&cache, &z, &chi_r, n);
    assert_eq!(scalar.len(), ring.len());
    for j in 0..scalar.len() {
        assert_eq!(scalar[j], ring[j][0], "matrix {j}: scalar eval must equal ct(y_ring)");
    }
}

#[test]
fn cached_superneo_ring_linear_forms_match_ring_eval_for_real_witnesses() {
    let n = 32usize;
    let m = 2 * D;

    let mut m0 = Mat::zero(n, m, F::ZERO);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        for c in 0..m {
            if ((r * 23) + (c * 9)) % 31 == 0 {
                m0[(r, c)] = F::from_u64(((r + 2 * c) % 29 + 1) as u64);
            }
            if ((r * 7) + (c * 13)) % 37 == 0 {
                m1[(r, c)] = F::from_u64(((3 * r + c) % 31 + 1) as u64);
            }
        }
    }

    let s = CcsStructure::new(vec![m0, m1], SparsePoly::new(2, vec![])).expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("cache should build for D-compatible width");

    let z: Vec<K> = (0..m)
        .map(|i| K::from(F::from_u64((i % 31 + 1) as u64)))
        .collect();
    let r = vec![
        K::from_coeffs([F::from_u64(2), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(3), F::from_u64(0)]),
        K::from_coeffs([F::from_u64(5), F::from_u64(2)]),
        K::from_coeffs([F::from_u64(7), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(11), F::from_u64(0)]),
    ];
    let chi_r = chi_table(&r);

    let ring = eval_all_mats_ring_cached(&cache, &z, &chi_r, n);
    let forms = cache.build_ring_linear_forms(&chi_r, n);
    let z_blocks = SuperneoZBlocks::from_z(&z);
    let via_forms: Vec<[K; D]> = forms
        .iter()
        .map(|form| form.eval_real_z_blocks(&z_blocks))
        .collect();
    assert_eq!(ring, via_forms);

    let via_batched_forms = eval_ring_linear_forms_real_z_blocks(&forms, &z_blocks);
    assert_eq!(ring, via_batched_forms);
}

#[test]
fn ring_linear_forms_pair_digit_fast_path_matches_ring_eval() {
    let n = 16usize;
    let m = 3 * D;

    let mut m0 = Mat::zero(n, m, F::ZERO);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for row in 0..n {
        for col in 0..m {
            if ((row * 11) + (col * 7)) % 13 == 0 {
                m0[(row, col)] = F::from_u64(((row + 3 * col) % 17 + 1) as u64);
            }
            if ((row * 5) + (col * 19)) % 23 == 0 {
                m1[(row, col)] = F::from_u64(((2 * row + col) % 29 + 1) as u64);
            }
        }
    }

    let s = CcsStructure::new(vec![m0, m1], SparsePoly::new(2, vec![])).expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("cache should build for D-compatible width");

    let neg_one = F::ZERO - F::ONE;
    let z: Vec<K> = (0..m)
        .map(|col| {
            let v = match col % 7 {
                0 | 3 => F::ONE,
                1 | 5 => neg_one,
                _ => F::ZERO,
            };
            K::from(v)
        })
        .collect();
    let z_blocks = SuperneoZBlocks::from_z(&z);
    assert!(z_blocks.imag_all_zero());

    let r = vec![
        K::from_coeffs([F::from_u64(2), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(3), F::from_u64(2)]),
        K::from_coeffs([F::from_u64(5), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(7), F::from_u64(3)]),
    ];
    let chi_r = chi_table(&r);

    let ring = eval_all_mats_ring_cached(&cache, &z, &chi_r, n);
    let forms = cache.build_ring_linear_forms(&chi_r, n);
    let via_forms = eval_ring_linear_forms_real_z_blocks(&forms, &z_blocks);
    assert_eq!(ring, via_forms);
}

#[test]
fn ring_linear_forms_chunked_parallel_matches_serial_above_threshold() {
    let blocks = 4096usize;
    let n = 1usize;
    let m = blocks * D;

    let mut row0 = vec![F::ZERO; m];
    let mut row1 = vec![F::ZERO; m];
    for blk in 0..blocks {
        row0[blk * D + ((7 * blk + 3) % D)] = F::from_u64(((blk % 29) + 1) as u64);
        row1[blk * D + ((11 * blk + 5) % D)] = F::from_u64(((blk % 31) + 1) as u64);
    }

    let s = CcsStructure::new(
        vec![Mat::from_row_major(n, m, row0), Mat::from_row_major(n, m, row1)],
        SparsePoly::new(2, vec![]),
    )
    .expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("cache should build for D-compatible width");
    let forms = cache.build_ring_linear_forms(&[K::ONE], n);

    let neg_one = F::ZERO - F::ONE;
    let z: Vec<K> = (0..m)
        .map(|col| {
            let v = match col % 5 {
                0 => F::ONE,
                1 => neg_one,
                _ => F::ZERO,
            };
            K::from(v)
        })
        .collect();
    let z_blocks = SuperneoZBlocks::from_z(&z);

    let serial: Vec<[K; D]> = forms
        .iter()
        .map(|form| form.eval_real_z_blocks(&z_blocks))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("build local rayon pool");
    let chunked = pool.install(|| eval_ring_linear_forms_real_z_blocks(&forms, &z_blocks));
    assert_eq!(chunked, serial);

    let scalar = eval_all_mats_cached(&cache, &z, &[K::ONE], n);
    for (j, y_ring) in chunked.iter().enumerate() {
        assert_eq!(
            scalar[j], y_ring[0],
            "matrix {j}: constant term of chunked y_ring must match scalar CE eval"
        );
    }
}

#[test]
fn cached_superneo_eval_keeps_imaginary_only_blocks() {
    let n = 4usize;
    let m = D;
    let mut mat = Mat::zero(n, m, F::ZERO);
    mat[(2, 7)] = F::from_u64(9);

    let s = CcsStructure::new(vec![mat], SparsePoly::new(1, vec![])).expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("cache should build for D-compatible width");

    let mut z_real = vec![K::ZERO; m];
    z_real[7] = K::from(F::from_u64(5));
    let real_blocks = SuperneoZBlocks::from_z(&z_real);
    let i = K::from_coeffs([F::ZERO, F::ONE]);
    let complex_blocks = SuperneoZBlocks::linear_combination_real(&[real_blocks], &[i]);

    let mut z_complex = vec![K::ZERO; m];
    z_complex[7] = i * K::from(F::from_u64(5));
    let r = vec![
        K::from_coeffs([F::from_u64(3), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(5), F::from_u64(2)]),
    ];
    let chi_r = chi_table(&r);

    let direct = eval_all_mats_direct(&s, &z_complex, &chi_r, n);
    let cached = eval_all_mats_cached_with_blocks(&cache, &complex_blocks, &chi_r, n);
    assert_eq!(cached, direct);

    let ring = eval_all_mats_ring_cached_with_blocks(&cache, &complex_blocks, &chi_r, n);
    assert_eq!(ring[0][0], direct[0]);
}

#[test]
fn cached_superneo_eval_supports_nondivisible_width() {
    let n = 8usize;
    let m = D + 1;
    let s = CcsStructure::new(vec![Mat::zero(n, m, F::ZERO)], SparsePoly::new(1, vec![])).expect("valid CCS");
    assert!(build_superneo_eval_cache(&s).is_some());
}

#[test]
fn cached_superneo_ring_real_z_blocks_match_scalar_eval() {
    let n = 32usize;
    let m = 2 * D;

    let mut m0 = Mat::zero(n, m, F::ZERO);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        for c in 0..m {
            if ((r * 19) + (c * 5)) % 23 == 0 {
                m0[(r, c)] = F::from_u64(((r + c) % 17 + 1) as u64);
            }
            if ((r * 7) + (c * 17)) % 29 == 0 {
                m1[(r, c)] = F::from_u64(((2 * r + c) % 19 + 1) as u64);
            }
        }
    }

    let s = CcsStructure::new(vec![m0, m1], SparsePoly::new(2, vec![])).expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("cache should build for D-compatible width");
    let z: Vec<K> = (0..m)
        .map(|i| K::from_coeffs([F::from_u64((i % 37 + 1) as u64), F::ZERO]))
        .collect();
    let z_blocks = SuperneoZBlocks::from_z(&z);
    assert!(z_blocks.imag_all_zero());
    let r = vec![
        K::from_coeffs([F::from_u64(2), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(3), F::from_u64(0)]),
        K::from_coeffs([F::from_u64(5), F::from_u64(2)]),
        K::from_coeffs([F::from_u64(7), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(11), F::from_u64(0)]),
    ];
    let chi_r = chi_table(&r);

    let scalar = eval_all_mats_cached(&cache, &z, &chi_r, n);
    let ring = eval_all_mats_ring_cached_with_blocks(&cache, &z_blocks, &chi_r, n);
    assert_eq!(scalar.len(), ring.len());
    for j in 0..scalar.len() {
        assert_eq!(scalar[j], ring[j][0], "matrix {j}: scalar eval must equal ct(y_ring)");
    }
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
    .expect("valid seeded Phi81 block");
    let compact_matrix =
        CcsMatrix::csc_with_seeded_phi81(CscMat::from_triplets(Vec::new(), rows, cols), vec![block.clone()])
            .expect("valid compact matrix");
    let mut expanded_triplets = Vec::new();
    block.for_each_term::<F, _>(|row, column, value| expanded_triplets.push((row, column, value)));
    let expanded_matrix = CcsMatrix::Csc(CscMat::from_triplets(expanded_triplets, rows, cols));
    let polynomial = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let compact = CcsStructure::new_sparse(vec![compact_matrix], polynomial.clone()).expect("compact CCS");
    let expanded = CcsStructure::new_sparse(vec![expanded_matrix], polynomial).expect("expanded CCS");
    let compact_cache = build_superneo_eval_cache(&compact).expect("compact cache");
    let expanded_cache = build_superneo_eval_cache(&expanded).expect("expanded cache");

    let z: Vec<K> = (0..cols)
        .map(|column| match (column * 31 + 7) % 3 {
            0 => -K::ONE,
            1 => K::ZERO,
            _ => K::ONE,
        })
        .collect();
    let chi: Vec<K> = (0..rows)
        .map(|row| K::from_coeffs([F::from_u64((row * 13 + 1) as u64), F::from_u64((row * 7 + 3) as u64)]))
        .collect();
    assert_eq!(
        eval_all_mats_cached(&compact_cache, &z, &chi, rows),
        eval_all_mats_cached(&expanded_cache, &z, &chi, rows)
    );
    assert_eq!(
        eval_all_mats_ring_cached(&compact_cache, &z, &chi, rows),
        eval_all_mats_ring_cached(&expanded_cache, &z, &chi, rows)
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
        "seeded forms must reconstruct the compact seeded matrix"
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
        "selective seeded-row evaluation must match the tensor table"
    );

    let z_blocks = SuperneoZBlocks::from_z(&z);
    let mut compact_rows = vec![K::ZERO; rows];
    let mut expanded_rows = vec![K::ZERO; rows];
    compact_cache
        .matrix(0)
        .expect("compact matrix cache")
        .fill_row_dots_real_with_blocks(&mut compact_rows, &z_blocks);
    expanded_cache
        .matrix(0)
        .expect("expanded matrix cache")
        .fill_row_dots_real_with_blocks(&mut expanded_rows, &z_blocks);
    assert_eq!(compact_rows, expanded_rows);
    assert_eq!(
        eval_ring_linear_forms_real_z_blocks(&compact_ring, &z_blocks),
        eval_ring_linear_forms_real_z_blocks(&expanded_ring, &z_blocks)
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

    let compact_transformed = compact
        .transform_matrices_superneo()
        .expect("compact transform");
    let expanded_transformed = expanded
        .transform_matrices_superneo()
        .expect("expanded transform");
    let compact_transformed_cache = build_superneo_eval_cache(&compact_transformed).expect("compact transformed cache");
    let expanded_transformed_cache =
        build_superneo_eval_cache(&expanded_transformed).expect("expanded transformed cache");
    let mut compact_transformed_rows = vec![K::ZERO; rows];
    let mut expanded_transformed_rows = vec![K::ZERO; rows];
    compact_transformed_cache
        .matrix(0)
        .expect("compact transformed matrix cache")
        .fill_row_dots_real_with_blocks(&mut compact_transformed_rows, &z_blocks);
    expanded_transformed_cache
        .matrix(0)
        .expect("expanded transformed matrix cache")
        .fill_row_dots_real_with_blocks(&mut expanded_transformed_rows, &z_blocks);
    assert_eq!(compact_transformed_rows, expanded_transformed_rows);
    let compact_transformed_ring = compact_transformed_cache.build_ring_linear_forms(&chi, rows);
    let expanded_transformed_ring = expanded_transformed_cache.build_ring_linear_forms(&chi, rows);
    assert_eq!(
        eval_ring_linear_forms_real_z_blocks(&compact_transformed_ring, &z_blocks),
        eval_ring_linear_forms_real_z_blocks(&expanded_transformed_ring, &z_blocks)
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
    let complex_blocks = SuperneoZBlocks::from_z(&complex_z);
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
        "transformed complex witnesses must retain compact seeded rows"
    );
}
