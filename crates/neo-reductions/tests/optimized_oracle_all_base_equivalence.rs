#![allow(non_snake_case)]

use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::{CcsStructure, CcsWitness, Mat};
use neo_math::{from_complex, D, F, K};
use neo_params::{goldilocks_paper_b2, NeoParams};
use neo_reductions::engines::optimized_engine::oracle::{OptimizedOracle, SparseCache};
use neo_reductions::superneo_eval::build_superneo_eval_cache;
use neo_reductions::Challenges;
use p3_field::PrimeCharacteristicRing;
use std::sync::Arc;

fn build_params_for_b(b: u32, m: usize) -> NeoParams {
    let q: u64 = goldilocks_paper_b2::Q;
    let eta: u32 = goldilocks_paper_b2::ETA as u32;
    let d: u32 = goldilocks_paper_b2::D as u32;
    let kappa: u32 = 2;
    let m_u: u64 = m as u64;
    let k_rho: u32 = 12;
    let T: u32 = goldilocks_paper_b2::T;
    let s: u32 = goldilocks_paper_b2::EXTENSION_DEGREE;
    let lambda: u32 = 96;
    NeoParams::new(q, eta, d, kappa, m_u, b, k_rho, T, s, lambda).expect("params")
}

fn build_oracle_with_counts(b: u32, k_mcs: usize, k_me: usize) -> OptimizedOracle<'static, F> {
    build_oracle_with_superneo_shape_and_cache_mode(b, k_mcs, k_me, true)
}

fn build_oracle(b: u32) -> OptimizedOracle<'static, F> {
    build_oracle_with_counts(b, 1, 0)
}

fn build_oracle_with_superneo_shape_and_cache_mode(
    b: u32,
    k_mcs: usize,
    k_me: usize,
    with_cache: bool,
) -> OptimizedOracle<'static, F> {
    assert!(k_mcs >= 1);

    // SuperNeo-compatible width: m is a multiple of D.
    let n = 2 * D;
    let mat = Mat::identity(n);
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let s = Box::leak(Box::new(CcsStructure::new(vec![mat], f).expect("CCS")));

    let params = Box::leak(Box::new(build_params_for_b(b, n)));

    let mk_witness = |seed: u64| {
        let cols = n / D;
        let mut data = Vec::with_capacity(D * cols);
        for rho in 0..D {
            for blk in 0..cols {
                let c = blk * D + rho;
                let x = seed
                    ^ (rho as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                    ^ 0x94D0_49BB_1331_11EB;
                data.push(F::from_u64(x));
            }
        }
        Mat::from_row_major(D, cols, data)
    };

    let mcs_vec = Box::leak(Box::new(
        (0..k_mcs)
            .map(|i| CcsWitness {
                w: vec![F::ZERO; n],
                Z: mk_witness(30_000 + i as u64),
            })
            .collect::<Vec<_>>(),
    ));
    let me_vec = Box::leak(Box::new(
        (0..k_me)
            .map(|i| mk_witness(40_000 + i as u64))
            .collect::<Vec<_>>(),
    ));

    let ell_n = 7usize; // n_pad = 128 >= n
    let ell_d = 6usize; // d_pad = 64 >= D
    let d_sc = 5usize;

    let ch = Challenges {
        alpha: (0..ell_d)
            .map(|i| K::from(F::from_u64((i as u64) + 1)))
            .collect(),
        beta_a: (0..ell_d)
            .map(|i| K::from(F::from_u64((i as u64) + 11)))
            .collect(),
        beta_r: (0..ell_n)
            .map(|i| K::from(F::from_u64((i as u64) + 21)))
            .collect(),
        beta_m: Vec::new(),
        gamma: K::from(F::from_u64(7)),
    };

    let r_inputs = if k_me > 0 {
        let r = Box::leak(Box::new(
            (0..ell_n)
                .map(|i| K::from(F::from_u64((i as u64) + 101)))
                .collect::<Vec<_>>(),
        ));
        Some(r.as_slice())
    } else {
        None
    };

    let sparse = Arc::new(SparseCache::build(s));
    let superneo_cache = if with_cache {
        build_superneo_eval_cache(s)
            .map(Arc::new)
            .unwrap_or_else(|| panic!("expected D-compatible superneo cache for test"))
    } else {
        panic!("missing cache should hard-fail in SuperNeo-only mode")
    };
    OptimizedOracle::new_with_sparse_and_superneo_cache(
        s,
        params,
        mcs_vec.as_slice(),
        me_vec.as_slice(),
        ch,
        ell_d,
        ell_n,
        d_sc,
        r_inputs,
        sparse,
        superneo_cache,
    )
}

fn build_oracle_with_superneo_shape(b: u32, k_mcs: usize, k_me: usize) -> OptimizedOracle<'static, F> {
    build_oracle_with_superneo_shape_and_cache_mode(b, k_mcs, k_me, true)
}

#[test]
fn optimized_oracle_all_base_matches_generic_b2() {
    let oracle = build_oracle(2);
    assert!(oracle.__test_row_stream_all_base(), "expected all_base=true");

    let xs: Vec<K> = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
    ];
    let (base, generic) = oracle.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(base, generic);
}

#[test]
fn optimized_oracle_all_base_matches_generic_b3() {
    let oracle = build_oracle(3);
    assert!(oracle.__test_row_stream_all_base(), "expected all_base=true");

    let xs: Vec<K> = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
    ];
    let (base, generic) = oracle.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(base, generic);
}

#[test]
fn optimized_oracle_all_base_matches_generic_b2_k_mcs4_k_me2() {
    let oracle = build_oracle_with_counts(2, 4, 2);
    assert!(oracle.__test_row_stream_all_base(), "expected all_base=true");

    let xs: Vec<K> = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
    ];
    let (base, generic) = oracle.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(base, generic);
}

#[test]
fn optimized_oracle_all_base_matches_generic_b3_k_mcs4_k_me2() {
    let oracle = build_oracle_with_counts(3, 4, 2);
    assert!(oracle.__test_row_stream_all_base(), "expected all_base=true");

    let xs: Vec<K> = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
    ];
    let (base, generic) = oracle.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(base, generic);
}

#[test]
fn optimized_oracle_nonbase_fast_matches_generic_b2_k_mcs4_k_me2() {
    let oracle = build_oracle_with_counts(2, 4, 2);
    assert!(oracle.__test_row_stream_all_base(), "expected all_base=true");

    let xs: Vec<K> = vec![
        from_complex(F::from_u64(0), F::from_u64(1)),
        from_complex(F::from_u64(1), F::from_u64(2)),
        from_complex(F::from_u64(2), F::from_u64(3)),
        from_complex(F::from_u64(5), F::from_u64(1)),
    ];
    let (fast, generic) = oracle.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(fast, generic);
}

#[test]
fn optimized_oracle_nonbase_fast_matches_generic_b3_k_mcs4_k_me2() {
    let oracle = build_oracle_with_counts(3, 4, 2);
    assert!(oracle.__test_row_stream_all_base(), "expected all_base=true");

    let xs: Vec<K> = vec![
        from_complex(F::from_u64(0), F::from_u64(1)),
        from_complex(F::from_u64(1), F::from_u64(2)),
        from_complex(F::from_u64(2), F::from_u64(3)),
        from_complex(F::from_u64(5), F::from_u64(1)),
    ];
    let (fast, generic) = oracle.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(fast, generic);
}

#[test]
fn optimized_oracle_superneo_shape_matches_generic_b2_k_mcs4_k_me2() {
    let oracle = build_oracle_with_superneo_shape(2, 4, 2);
    assert!(oracle.__test_row_stream_all_base(), "expected all_base=true");

    let xs: Vec<K> = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
    ];
    let (base, generic) = oracle.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(base, generic);
}

#[test]
fn optimized_oracle_superneo_shape_matches_generic_b3_k_mcs4_k_me2() {
    let oracle = build_oracle_with_superneo_shape(3, 4, 2);
    assert!(oracle.__test_row_stream_all_base(), "expected all_base=true");

    let xs: Vec<K> = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
    ];
    let (base, generic) = oracle.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(base, generic);
}

#[test]
fn optimized_oracle_superneo_shape_explicit_cache_toggle_b2() {
    let with_cache = build_oracle_with_superneo_shape_and_cache_mode(2, 4, 2, true);
    assert!(with_cache.__test_row_stream_uses_superneo_rows());
    let missing_cache = std::panic::catch_unwind(|| {
        let _ = build_oracle_with_superneo_shape_and_cache_mode(2, 4, 2, false);
    });
    assert!(
        missing_cache.is_err(),
        "missing cache should hard-fail in SuperNeo-only mode"
    );

    let xs: Vec<K> = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
    ];
    let (fast, generic) = with_cache.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(fast, generic);
}

#[test]
fn optimized_oracle_superneo_shape_explicit_cache_toggle_b3() {
    let with_cache = build_oracle_with_superneo_shape_and_cache_mode(3, 4, 2, true);
    assert!(with_cache.__test_row_stream_uses_superneo_rows());
    let missing_cache = std::panic::catch_unwind(|| {
        let _ = build_oracle_with_superneo_shape_and_cache_mode(3, 4, 2, false);
    });
    assert!(
        missing_cache.is_err(),
        "missing cache should hard-fail in SuperNeo-only mode"
    );

    let xs: Vec<K> = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
    ];
    let (fast, generic) = with_cache.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(fast, generic);
}
