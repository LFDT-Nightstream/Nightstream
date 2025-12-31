use std::sync::Arc;

use neo_ccs::{CcsStructure, Mat, McsWitness, SparsePoly, Term};
use neo_math::{from_complex, D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::oracle::{OptimizedOracle, SparseCache};
use neo_reductions::engines::optimized_engine::{
    q_eval_at_ext_point_paper_exact_with_inputs, sum_q_over_hypercube_paper_exact, Challenges,
};
use neo_reductions::sumcheck::{poly_eval_k, run_sumcheck_prover};
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
    let mut data = Vec::with_capacity(D * m);
    for rho in 0..D {
        for c in 0..m {
            data.push(F::from_u64(seed + (rho as u64) * 19 + (c as u64) * 29));
        }
    }
    Mat::from_row_major(D, m, data)
}

#[test]
fn optimized_oracle_row_stream_matches_paper_exact_q_at_challenge_point() {
    // Small CCS instance: n=m=8, t=4 with M0=I, and f(x)=x1*x2 - x3.
    let n = 8usize;
    let m = 8usize;
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).unwrap();
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
                exps: vec![0, 0, 0, 1], // -x3
            },
        ],
    );
    let s = CcsStructure::new(matrices, f).unwrap();

    // Protocol dimensions (ell_d fixed by D, ell_n from n, d_sc from max degree/b).
    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &s).unwrap();
    assert_eq!(dims.ell_n, 3);
    assert_eq!(dims.ell_d, 6);
    assert_eq!(dims.d_sc, 6);

    // Two witnesses to activate the Eval block (k_total>=2).
    let mcs_witnesses = vec![
        McsWitness {
            w: vec![],
            Z: z_witness(100, m),
        },
        McsWitness {
            w: vec![],
            Z: z_witness(200, m),
        },
    ];
    let me_witnesses: Vec<Mat<F>> = Vec::new();

    // Public challenges (α, β, γ) and ME input r for Eval gating.
    let ch = Challenges {
        alpha: (0..dims.ell_d).map(|i| k(1000 + i as u64, 2000 + i as u64)).collect(),
        beta_a: (0..dims.ell_d).map(|i| k(3000 + i as u64, 4000 + i as u64)).collect(),
        beta_r: (0..dims.ell_n).map(|i| k(5000 + i as u64, 6000 + i as u64)).collect(),
        gamma: k(7777, 8888),
    };
    let r_inputs: Vec<K> = (0..dims.ell_n)
        .map(|i| k(9000 + i as u64, 10000 + i as u64))
        .collect();

    // Brute-force initial sum over the Boolean hypercube.
    let initial_sum = sum_q_over_hypercube_paper_exact(
        &s,
        &params,
        &mcs_witnesses,
        &me_witnesses,
        &ch,
        dims.ell_d,
        dims.ell_n,
        Some(&r_inputs),
    );

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

    // Cross-check: final running sum must equal Q(α', r') computed directly from witnesses.
    let r_prime = &challenges[..dims.ell_n];
    let alpha_prime = &challenges[dims.ell_n..];
    let (q_at_point, _) = q_eval_at_ext_point_paper_exact_with_inputs(
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
