#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use neo_reductions::pi_ccs_paper_exact as refimpl;
use neo_reductions::mojo_gpu_engine::engine::MojoGpuEngine;
use neo_reductions::mojo_gpu_engine::oracle::MojoOracle;
use neo_reductions::engines::pi_rlc_dec::RlcDecOps;
use neo_ccs::{CcsStructure, McsInstance, McsWitness, Mat, SparsePoly, Term};
use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule, Commitment, s_lincomb};
use neo_ccs::traits::SModuleHomomorphism;
use rand_chacha::rand_core::SeedableRng;
use neo_params::NeoParams;
use neo_math::{F, K, D};
use p3_field::PrimeCharacteristicRing;
use neo_reductions::optimized_engine::Challenges;

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

fn tiny_ccs_id(n: usize, m: usize) -> CcsStructure<F> {
    assert_eq!(n, m, "use square tiny ccs");
    let m0 = Mat::identity(n);
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]); // f(y0) = y0
    CcsStructure::new(vec![m0], f).unwrap()
}

#[test]
fn test_mojo_consistency() {
    let params = NeoParams::goldilocks_127();
    let n = 2usize; let m = 2usize;
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z = Mat::from_row_major(D, m, vec![F::ONE; D * m]);
    let w = McsWitness { w: vec![], Z: z.clone() };
    let c = l.commit(&w.Z);
    let inst = McsInstance { c: c.clone(), x: vec![], m_in: 0 };

    let r_p = vec![K::from(F::from_u64(7)); 1];
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    // 1. Test build_me_outputs
    let out_exact = refimpl::build_me_outputs_paper_exact(
        &s, &params,
        &[inst.clone()], &[w.clone()],
        &[], &[],
        &r_p, ell_d, [0u8; 32], &l,
    );

    let out_mojo = MojoGpuEngine::build_me_outputs_mojo(
        &s, &params,
        &[inst.clone()], &[w.clone()],
        &[], &[],
        &r_p, ell_d, [0u8; 32], &l,
    );
    
    assert_eq!(out_exact.len(), out_mojo.len());
    for (exact, mojo) in out_exact.iter().zip(out_mojo.iter()) {
        assert_eq!(exact.y, mojo.y, "y mismatch");
        assert_eq!(exact.y_scalars, mojo.y_scalars, "y_scalars mismatch");
        assert_eq!(exact.r, mojo.r, "r mismatch");
        assert_eq!(exact.c, mojo.c, "c mismatch");
        assert_eq!(exact.X.as_slice(), mojo.X.as_slice(), "X mismatch");
    }

    // 2. Test RLC
    let rhos = vec![Mat::identity(D)]; // 1 instance, Identity rho
    let me_input = out_exact[0].clone();
    let Zs = vec![z.clone()];
    
    let mix_commits = |rhos: &[Mat<F>], cmts: &[Commitment]| -> Commitment {
        // Convert rhos (Mat<F>) to RqEl for s_lincomb
        let mut rq_rhos = Vec::with_capacity(rhos.len());
        for rho_mat in rhos {
            let scalar = rho_mat[(0,0)]; // extract scalar
            let _val = if scalar == F::ONE { 1 } else { 0 }; // Hack for test
            if scalar == F::ONE {
                rq_rhos.push(neo_math::ring::Rq::one());
            } else {
                rq_rhos.push(neo_math::ring::Rq::zero());
            }
        }
        s_lincomb(&rq_rhos, cmts).unwrap()
    };

    let (rlc_exact, Z_mix_exact) = neo_reductions::engines::pi_rlc_dec::PaperExactRlcDec::rlc_with_commit(
        &s, &params, &rhos, &[me_input.clone()], &Zs, ell_d, mix_commits
    );

    let (rlc_mojo, Z_mix_mojo) = MojoGpuEngine::rlc_with_commit(
        &s, &params, &rhos, &[me_input.clone()], &Zs, ell_d, mix_commits
    );

    assert_eq!(Z_mix_exact.as_slice(), Z_mix_mojo.as_slice(), "RLC Z_mix mismatch");
    assert_eq!(rlc_exact.c, rlc_mojo.c, "RLC c mismatch");
    assert_eq!(rlc_exact.X.as_slice(), rlc_mojo.X.as_slice(), "RLC X mismatch");

    // 3. Test DEC
    let Z_split = vec![z.clone()]; // k=1
    let child_commits = vec![c.clone()];
    let combine_b = |cmts: &[Commitment], _b: u32| -> Commitment {
        cmts[0].clone() // Dummy combine for k=1
    };

    let (dec_exact, ok_y_e, ok_X_e, ok_c_e) = neo_reductions::engines::pi_rlc_dec::PaperExactRlcDec::dec_children_with_commit(
        &s, &params, &me_input, &Z_split, ell_d, &child_commits, combine_b
    );

    let (dec_mojo, ok_y_m, ok_X_m, ok_c_m) = MojoGpuEngine::dec_children_with_commit(
        &s, &params, &me_input, &Z_split, ell_d, &child_commits, combine_b
    );

    assert_eq!(ok_y_e, ok_y_m, "DEC ok_y mismatch");
    assert_eq!(ok_X_e, ok_X_m, "DEC ok_X mismatch");
    assert_eq!(ok_c_e, ok_c_m, "DEC ok_c mismatch");
    assert_eq!(dec_exact.len(), dec_mojo.len());
    // Compare children
    for (exact, mojo) in dec_exact.iter().zip(dec_mojo.iter()) {
         assert_eq!(exact.y, mojo.y, "DEC child y mismatch");
    }

    // 4. Test Oracle
    let ch = Challenges {
        alpha: vec![K::from(F::from_u64(2)); ell_d],
        beta_a: vec![K::from(F::from_u64(3)); ell_d],
        beta_r: vec![K::from(F::from_u64(4)); 1], // ell_n = 1
        gamma: K::from(F::from_u64(5)),
    };
    
    let witnesses = [w.clone()];
    let empty_me: [Mat<F>; 0] = [];
    
    let oracle_exact = neo_reductions::paper_exact_engine::oracle::PaperExactOracle::new(
        &s, &params, &witnesses, &empty_me, ch.clone(), ell_d, 1, 10, None
    );
    
    let mut oracle_mojo = MojoOracle::new(
        &s, &params, &witnesses, &empty_me, ch.clone(), ell_d, 1, 10, None
    );

    let r_eval = vec![K::from(F::from_u64(9)); 1 + ell_d]; // r'
    
    let (r_row, r_ajtai) = r_eval.split_at(1);
    let q_exact = oracle_exact.eval_q_ext(r_ajtai, r_row);
    let q_mojo = oracle_mojo.eval_q_ext(&r_eval);
    
    assert_eq!(q_exact, q_mojo, "Oracle eval_q mismatch");
}
