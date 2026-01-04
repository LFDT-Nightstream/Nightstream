use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, McsInstance, McsWitness, MeInstance, SparsePoly, Term};
use neo_math::{from_complex, D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::oracle::{OptimizedOracle, SparseCache};
use neo_reductions::engines::optimized_engine::Challenges;
use neo_reductions::paper_exact_engine::build_me_outputs_paper_exact;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

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
fn optimized_oracle_outputs_match_paper_exact_builder() {
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

    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &s).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m).expect("Ajtai setup should succeed");
    let l = AjtaiSModule::new(Arc::new(pp));

    // One MCS + one ME witness, to exercise both output cases.
    let m_in = 4usize;
    let z_mcs = z_witness(100, m);
    let mcs_inst = McsInstance {
        c: l.commit(&z_mcs),
        x: vec![F::ONE; m_in],
        m_in,
    };
    let mcs_witnesses = vec![McsWitness {
        w: vec![F::ZERO; m.saturating_sub(m_in)],
        Z: z_mcs,
    }];

    let z_me = z_witness(200, m);
    let me_inputs_r: Vec<K> = (0..dims.ell_n)
        .map(|i| k(9000 + i as u64, 10000 + i as u64))
        .collect();
    let me_inputs = vec![MeInstance {
        c: l.commit(&z_me),
        X: l.project_x(&z_me, m_in),
        r: me_inputs_r.clone(),
        y: vec![vec![K::ZERO; 1usize << dims.ell_d]; s.t()],
        y_scalars: vec![K::ZERO; s.t()],
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    }];
    let me_witnesses = vec![z_me];

    let ch = Challenges {
        alpha: (0..dims.ell_d).map(|i| k(1000 + i as u64, 2000 + i as u64)).collect(),
        beta_a: (0..dims.ell_d).map(|i| k(3000 + i as u64, 4000 + i as u64)).collect(),
        beta_r: (0..dims.ell_n).map(|i| k(5000 + i as u64, 6000 + i as u64)).collect(),
        gamma: k(7777, 8888),
    };

    // Choose a finalized row challenge point r' (independent from ME input r).
    let r_prime: Vec<K> = (0..dims.ell_n)
        .map(|i| k(111 + i as u64, 222 + i as u64))
        .collect();

    // Build oracle, fold row rounds to set r', then build outputs from the Ajtai precomp.
    let sparse = Arc::new(SparseCache::build(&s));
    let mut oracle = OptimizedOracle::new_with_sparse(
        &s,
        &params,
        &mcs_witnesses,
        &me_witnesses,
        ch,
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        Some(&me_inputs_r),
        sparse,
    );
    for &r_i in &r_prime {
        oracle.fold(r_i);
    }

    let fold_digest = [7u8; 32];
    let out_fast =
        oracle.build_me_outputs_from_ajtai_precomp(core::slice::from_ref(&mcs_inst), &me_inputs, fold_digest, &l);
    let out_ref = build_me_outputs_paper_exact(
        &s,
        &params,
        core::slice::from_ref(&mcs_inst),
        &mcs_witnesses,
        &me_inputs,
        &me_witnesses,
        &r_prime,
        dims.ell_d,
        fold_digest,
        &l,
    );

    assert_eq!(out_fast, out_ref);
}
