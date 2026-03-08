use neo_ccs::{matrix::Mat, poly::SparsePoly, CcsStructure};
use neo_math::KExtensions;
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::{
    build_superneo_eval_cache, eval_all_mats_cached, eval_all_mats_direct, eval_all_mats_ring_cached,
    eval_all_mats_ring_cached_with_blocks, eval_all_mats_transformed, SuperneoZBlocks,
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
