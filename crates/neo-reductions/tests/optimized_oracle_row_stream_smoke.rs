use std::sync::Arc;

use neo_ccs::{CcsStructure, CcsWitness, Mat, SparsePoly, Term};
use neo_math::{from_complex, D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::oracle::{OptimizedOracle, SparseCache};
use neo_reductions::engines::optimized_engine::Challenges;
use neo_reductions::engines::paper_exact_engine::q_eval_at_ext_point_fe_paper_exact_with_inputs;
use neo_reductions::sumcheck::{poly_eval_k, run_sumcheck_prover};
use neo_reductions::superneo_eval::build_superneo_eval_cache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn k(re: u64, im: u64) -> K {
    from_complex(F::from_u64(re), F::from_u64(im))
}

fn dense_mat<Ff: PrimeCharacteristicRing + Copy>(rows: usize, cols: usize, seed: u64) -> Mat<Ff> {
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            // Deterministic, mildly sparse-ish pattern.
            let v = if (r + 2 * c) % 5 == 0 {
                Ff::from_u64(seed + (r as u64) * 17 + (c as u64) * 23 + 1)
            } else {
                Ff::ZERO
            };
            data.push(v);
        }
    }
    Mat::from_row_major(rows, cols, data)
}

fn z_witness(seed: u64, m: usize) -> Mat<F> {
    assert!(m.is_multiple_of(D), "SuperNeo-only test requires m divisible by D");
    let cols = m / D;
    let mut data = Vec::with_capacity(D * cols);
    for rho in 0..D {
        for blk in 0..cols {
            let c = blk * D + rho;
            data.push(F::from_u64(seed + (rho as u64) * 19 + (c as u64) * 29));
        }
    }
    Mat::from_row_major(D, cols, data)
}

#[test]
fn optimized_oracle_row_stream_matches_paper_exact_q_at_challenge_point() {
    // SuperNeo-compatible shape covering the optimized row-polynomial term
    // classes: product-2, selector-times-power, and product-3.
    let n = D;
    let m = D;
    let params = NeoParams::goldilocks_auto_ccs_with(n, 4, 8, 96, 2).unwrap();
    let matrices = vec![
        Mat::<F>::identity(n),
        dense_mat::<F>(n, m, 10),
        dense_mat::<F>(n, m, 20),
        dense_mat::<F>(n, m, 30),
    ];
    let f = SparsePoly::new(
        /*t=*/ 4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1, 0], // x1 * x2
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 1, 0, 1], // -x1 * x3; keeps every term in one factored group
            },
            Term {
                coeff: F::from_u64(3),
                exps: vec![0, 1, 7, 0], // 3 * x1 * x2^7
            },
            Term {
                coeff: F::from_u64(5),
                exps: vec![0, 1, 1, 1], // 5 * x1 * x2 * x3
            },
        ],
    );
    let s = CcsStructure::new(matrices, f).unwrap();

    // Protocol dimensions (ell_d fixed by D, ell_n from n, d_sc from max degree/b).
    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &s).unwrap();
    assert_eq!(dims.ell_n, 6);
    assert_eq!(dims.ell_d, 6);
    let expected_d_sc = core::cmp::max(s.max_degree() as usize + 1, 2 * (params.b as usize));
    assert_eq!(dims.d_sc, expected_d_sc);

    // Two witnesses to activate the Eval block (k_total>=2).
    let mcs_witnesses = vec![
        CcsWitness {
            w: vec![],
            Z: z_witness(100, m),
        },
        CcsWitness {
            w: vec![],
            Z: z_witness(200, m),
        },
    ];
    let me_witnesses: Vec<Mat<F>> = Vec::new();

    // Public challenges (α, β, γ) and ME input r for Eval gating.
    let ch = Challenges {
        alpha: (0..dims.ell_d)
            .map(|i| k(1000 + i as u64, 2000 + i as u64))
            .collect(),
        beta_a: (0..dims.ell_d)
            .map(|i| k(3000 + i as u64, 4000 + i as u64))
            .collect(),
        beta_r: (0..dims.ell_n)
            .map(|i| k(5000 + i as u64, 6000 + i as u64))
            .collect(),
        beta_m: Vec::new(),
        gamma: k(7777, 8888),
    };
    let r_inputs: Vec<K> = (0..dims.ell_n)
        .map(|i| k(9000 + i as u64, 10000 + i as u64))
        .collect();

    // Brute-force initial sum over the Boolean hypercube for the FE-only polynomial Q_fe.
    let d_sz = 1usize << dims.ell_d;
    let n_sz = 1usize << dims.ell_n;
    let mut initial_sum = K::ZERO;
    for xa in 0..d_sz {
        let alpha_bool: Vec<K> = (0..dims.ell_d)
            .map(|bit| if ((xa >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
            .collect();
        for xr in 0..n_sz {
            let r_bool: Vec<K> = (0..dims.ell_n)
                .map(|bit| if ((xr >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
                .collect();
            let (q_at_bool, _) = q_eval_at_ext_point_fe_paper_exact_with_inputs(
                &s,
                &params,
                &mcs_witnesses,
                &me_witnesses,
                &alpha_bool,
                &r_bool,
                &ch,
                Some(&r_inputs),
            );
            initial_sum += q_at_bool;
        }
    }

    // Run sumcheck prover using the optimized oracle (streaming row phase).
    let sparse = Arc::new(SparseCache::build(&s));
    let mut oracle = OptimizedOracle::new_with_sparse(
        &s,
        &params,
        &mcs_witnesses,
        &me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        Some(&r_inputs),
        sparse,
    );
    let mut tr = Poseidon2Transcript::new(b"optimized_oracle_row_stream_smoke");
    let (rounds, challenges) = run_sumcheck_prover(&mut tr, &mut oracle, initial_sum).unwrap();

    // Compute the final running sum from the prover transcript.
    let mut running = initial_sum;
    for (coeffs, &chal) in rounds.iter().zip(challenges.iter()) {
        running = poly_eval_k(coeffs, chal);
    }

    // Cross-check: final running sum must equal Q_fe(α', r') computed directly from witnesses.
    let r_prime = &challenges[..dims.ell_n];
    let alpha_prime = &challenges[dims.ell_n..];
    let (q_at_point, _) = q_eval_at_ext_point_fe_paper_exact_with_inputs(
        &s,
        &params,
        &mcs_witnesses,
        &me_witnesses,
        alpha_prime,
        r_prime,
        &ch,
        Some(&r_inputs),
    );
    assert_eq!(running, q_at_point);
}

#[test]
fn optimized_oracle_superneo_shape_matches_paper_exact_q_at_challenge_point() {
    // SuperNeo-compatible shape: m is a multiple of D.
    let n = D;
    let m = D;
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).unwrap();
    let matrices = vec![
        Mat::<F>::identity(n),
        dense_mat::<F>(n, m, 111),
        dense_mat::<F>(n, m, 222),
    ];
    let f = SparsePoly::new(
        /*t=*/ 3,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1], // x1 * x2
            },
            Term {
                coeff: F::ONE,
                exps: vec![1, 0, 0], // +x0
            },
        ],
    );
    let s = CcsStructure::new(matrices, f).unwrap();

    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &s).unwrap();
    assert_eq!(dims.ell_d, 6);

    // Include ME witnesses to exercise Eval block and Ajtai precompute Y_eval path.
    let mcs_witnesses = vec![
        CcsWitness {
            w: vec![],
            Z: z_witness(10_000, m),
        },
        CcsWitness {
            w: vec![],
            Z: z_witness(20_000, m),
        },
    ];
    let me_witnesses: Vec<Mat<F>> = vec![z_witness(30_000, m), z_witness(40_000, m)];

    let ch = Challenges {
        alpha: (0..dims.ell_d)
            .map(|i| k(1_000 + i as u64, 2_000 + i as u64))
            .collect(),
        beta_a: (0..dims.ell_d)
            .map(|i| k(3_000 + i as u64, 4_000 + i as u64))
            .collect(),
        beta_r: (0..dims.ell_n)
            .map(|i| k(5_000 + i as u64, 6_000 + i as u64))
            .collect(),
        beta_m: Vec::new(),
        gamma: k(7_777, 8_888),
    };
    let r_inputs: Vec<K> = (0..dims.ell_n)
        .map(|i| k(9_000 + i as u64, 10_000 + i as u64))
        .collect();

    // FE initial sum by brute force over the Boolean hypercube.
    let d_sz = 1usize << dims.ell_d;
    let n_sz = 1usize << dims.ell_n;
    let mut initial_sum = K::ZERO;
    for xa in 0..d_sz {
        let alpha_bool: Vec<K> = (0..dims.ell_d)
            .map(|bit| if ((xa >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
            .collect();
        for xr in 0..n_sz {
            let r_bool: Vec<K> = (0..dims.ell_n)
                .map(|bit| if ((xr >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
                .collect();
            let (q_at_bool, _) = q_eval_at_ext_point_fe_paper_exact_with_inputs(
                &s,
                &params,
                &mcs_witnesses,
                &me_witnesses,
                &alpha_bool,
                &r_bool,
                &ch,
                Some(&r_inputs),
            );
            initial_sum += q_at_bool;
        }
    }

    let sparse = Arc::new(SparseCache::build(&s));
    let mut oracle = OptimizedOracle::new_with_sparse(
        &s,
        &params,
        &mcs_witnesses,
        &me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        Some(&r_inputs),
        sparse,
    );
    let mut tr = Poseidon2Transcript::new(b"optimized_oracle_superneo_shape_smoke");
    let (rounds, challenges) = run_sumcheck_prover(&mut tr, &mut oracle, initial_sum).unwrap();

    let mut running = initial_sum;
    for (coeffs, &chal) in rounds.iter().zip(challenges.iter()) {
        running = poly_eval_k(coeffs, chal);
    }

    let r_prime = &challenges[..dims.ell_n];
    let alpha_prime = &challenges[dims.ell_n..];
    let (q_at_point, _) = q_eval_at_ext_point_fe_paper_exact_with_inputs(
        &s,
        &params,
        &mcs_witnesses,
        &me_witnesses,
        alpha_prime,
        r_prime,
        &ch,
        Some(&r_inputs),
    );
    assert_eq!(running, q_at_point);
}

#[test]
fn optimized_oracle_superneo_shape_rectangular_matches_paper_exact_q_at_challenge_point() {
    // SuperNeo-compatible width with rectangular CCS: m multiple of D, n != m.
    let n = 32usize;
    let m = D;
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).unwrap();
    let matrices = vec![
        dense_mat::<F>(n, m, 123),
        dense_mat::<F>(n, m, 456),
        dense_mat::<F>(n, m, 789),
    ];
    let f = SparsePoly::new(
        /*t=*/ 3,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 0, 0], // +x0
            },
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1], // +x1*x2
            },
        ],
    );
    let s = CcsStructure::new(matrices, f).unwrap();

    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &s).unwrap();
    assert_eq!(dims.ell_d, 6);

    let mcs_witnesses = vec![
        CcsWitness {
            w: vec![],
            Z: z_witness(50_000, m),
        },
        CcsWitness {
            w: vec![],
            Z: z_witness(60_000, m),
        },
    ];
    let me_witnesses: Vec<Mat<F>> = vec![z_witness(70_000, m), z_witness(80_000, m)];

    let ch = Challenges {
        alpha: (0..dims.ell_d)
            .map(|i| k(1_100 + i as u64, 2_100 + i as u64))
            .collect(),
        beta_a: (0..dims.ell_d)
            .map(|i| k(3_100 + i as u64, 4_100 + i as u64))
            .collect(),
        beta_r: (0..dims.ell_n)
            .map(|i| k(5_100 + i as u64, 6_100 + i as u64))
            .collect(),
        beta_m: Vec::new(),
        gamma: k(7_701, 8_802),
    };
    let r_inputs: Vec<K> = (0..dims.ell_n)
        .map(|i| k(9_100 + i as u64, 10_100 + i as u64))
        .collect();

    let d_sz = 1usize << dims.ell_d;
    let n_sz = 1usize << dims.ell_n;
    let mut initial_sum = K::ZERO;
    for xa in 0..d_sz {
        let alpha_bool: Vec<K> = (0..dims.ell_d)
            .map(|bit| if ((xa >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
            .collect();
        for xr in 0..n_sz {
            let r_bool: Vec<K> = (0..dims.ell_n)
                .map(|bit| if ((xr >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
                .collect();
            let (q_at_bool, _) = q_eval_at_ext_point_fe_paper_exact_with_inputs(
                &s,
                &params,
                &mcs_witnesses,
                &me_witnesses,
                &alpha_bool,
                &r_bool,
                &ch,
                Some(&r_inputs),
            );
            initial_sum += q_at_bool;
        }
    }

    let sparse = Arc::new(SparseCache::build(&s));
    let mut oracle = OptimizedOracle::new_with_sparse(
        &s,
        &params,
        &mcs_witnesses,
        &me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        Some(&r_inputs),
        sparse,
    );
    let mut tr = Poseidon2Transcript::new(b"optimized_oracle_superneo_rectangular_shape_smoke");
    let (rounds, challenges) = run_sumcheck_prover(&mut tr, &mut oracle, initial_sum).unwrap();

    let mut running = initial_sum;
    for (coeffs, &chal) in rounds.iter().zip(challenges.iter()) {
        running = poly_eval_k(coeffs, chal);
    }

    let r_prime = &challenges[..dims.ell_n];
    let alpha_prime = &challenges[dims.ell_n..];
    let (q_at_point, _) = q_eval_at_ext_point_fe_paper_exact_with_inputs(
        &s,
        &params,
        &mcs_witnesses,
        &me_witnesses,
        alpha_prime,
        r_prime,
        &ch,
        Some(&r_inputs),
    );
    assert_eq!(running, q_at_point);
}

#[test]
fn optimized_oracle_explicit_superneo_cache_runs_canonical_path() {
    let n = 32usize;
    let m = D;
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).unwrap();
    let matrices = vec![
        dense_mat::<F>(n, m, 1234),
        dense_mat::<F>(n, m, 2345),
        dense_mat::<F>(n, m, 3456),
    ];
    let f = SparsePoly::new(
        /*t=*/ 3,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 0, 0],
            },
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1],
            },
        ],
    );
    let s = CcsStructure::new(matrices, f).unwrap();
    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &s).unwrap();

    let mcs_witnesses = vec![
        CcsWitness {
            w: vec![],
            Z: z_witness(101_000, m),
        },
        CcsWitness {
            w: vec![],
            Z: z_witness(102_000, m),
        },
    ];
    let me_witnesses: Vec<Mat<F>> = vec![z_witness(103_000, m)];

    let ch = Challenges {
        alpha: (0..dims.ell_d)
            .map(|i| k(1_010 + i as u64, 2_010 + i as u64))
            .collect(),
        beta_a: (0..dims.ell_d)
            .map(|i| k(3_010 + i as u64, 4_010 + i as u64))
            .collect(),
        beta_r: (0..dims.ell_n)
            .map(|i| k(5_010 + i as u64, 6_010 + i as u64))
            .collect(),
        beta_m: Vec::new(),
        gamma: k(7_010, 8_010),
    };
    let r_inputs: Vec<K> = (0..dims.ell_n)
        .map(|i| k(9_010 + i as u64, 10_010 + i as u64))
        .collect();

    let d_sz = 1usize << dims.ell_d;
    let n_sz = 1usize << dims.ell_n;
    let mut initial_sum = K::ZERO;
    for xa in 0..d_sz {
        let alpha_bool: Vec<K> = (0..dims.ell_d)
            .map(|bit| if ((xa >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
            .collect();
        for xr in 0..n_sz {
            let r_bool: Vec<K> = (0..dims.ell_n)
                .map(|bit| if ((xr >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
                .collect();
            let (q_at_bool, _) = q_eval_at_ext_point_fe_paper_exact_with_inputs(
                &s,
                &params,
                &mcs_witnesses,
                &me_witnesses,
                &alpha_bool,
                &r_bool,
                &ch,
                Some(&r_inputs),
            );
            initial_sum += q_at_bool;
        }
    }

    let sparse = Arc::new(SparseCache::build(&s));
    let superneo_cache = build_superneo_eval_cache(&s)
        .map(Arc::new)
        .unwrap_or_else(|| panic!("expected D-compatible superneo cache"));

    let run = || {
        let mut oracle = OptimizedOracle::new_with_sparse_and_superneo_cache(
            &s,
            &params,
            &mcs_witnesses,
            &me_witnesses,
            ch.clone(),
            dims.ell_d,
            dims.ell_n,
            dims.d_sc,
            Some(&r_inputs),
            sparse.clone(),
            superneo_cache.clone(),
        );
        assert!(oracle.__test_row_stream_uses_superneo_rows());
        let mut tr = Poseidon2Transcript::new(b"optimized_oracle_cache_equiv");
        let (rounds, challenges) = run_sumcheck_prover(&mut tr, &mut oracle, initial_sum)?;
        let mut running = initial_sum;
        for (coeffs, &chal) in rounds.iter().zip(challenges.iter()) {
            running = poly_eval_k(coeffs, chal);
        }
        Ok::<K, neo_reductions::sumcheck::SumcheckError>(running)
    };

    let _with_cache = run().expect("SuperNeo oracle should satisfy invariant");
}
