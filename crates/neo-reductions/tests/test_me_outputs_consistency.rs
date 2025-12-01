#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use neo_reductions::pi_ccs_paper_exact as refimpl;
use neo_reductions::optimized_engine::build_me_outputs_optimized;
use neo_ccs::{CcsStructure, McsInstance, McsWitness, Mat, SparsePoly, Term};
use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use rand_chacha::rand_core::SeedableRng;
use neo_params::NeoParams;
use neo_math::{F, K, D};
use p3_field::PrimeCharacteristicRing;

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
fn test_me_outputs_optimized_vs_paper_exact() {
    let params = NeoParams::goldilocks_127();
    let n = 2usize; let m = 2usize;
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z = Mat::from_row_major(D, m, vec![F::ONE; D * m]);
    let w = McsWitness { w: vec![], Z: z };
    let c = l.commit(&w.Z);
    let inst = McsInstance { c, x: vec![], m_in: 0 };

    let r_p = vec![K::from(F::from_u64(7)); 1];

    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    let out_exact = refimpl::build_me_outputs_paper_exact(
        &s, &params,
        &[inst.clone()], &[w.clone()],
        &[], &[],
        &r_p, ell_d, [0u8; 32], &l,
    );

    let out_opt = build_me_outputs_optimized(
        &s, &params,
        &[inst.clone()], &[w.clone()],
        &[], &[],
        &r_p, ell_d, [0u8; 32], &l,
    );
    
    // Compare outputs
    assert_eq!(out_exact.len(), out_opt.len());
    for (exact, opt) in out_exact.iter().zip(out_opt.iter()) {
        assert_eq!(exact.y, opt.y);
        assert_eq!(exact.y_scalars, opt.y_scalars);
        assert_eq!(exact.r, opt.r);
        assert_eq!(exact.c, opt.c);
        assert_eq!(exact.X.as_slice(), opt.X.as_slice());
    }
}
