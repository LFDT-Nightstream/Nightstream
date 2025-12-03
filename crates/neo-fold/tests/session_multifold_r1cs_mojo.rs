#![allow(non_snake_case)]

use neo_fold::session::{
    FoldingSession, Accumulator, me_from_z_balanced, ProveInput
};
use neo_fold::pi_ccs::FoldingMode;
use neo_ccs::{Mat, r1cs_to_ccs};
use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule};
use rand_chacha::rand_core::SeedableRng;
use neo_params::NeoParams;
use neo_math::{F, K, D};
use p3_field::PrimeCharacteristicRing;

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

#[test]
fn test_session_multifold_k3_three_steps_r1cs_mojo() {
    let n = 4usize;
    let A = Mat::identity(n);
    let B = Mat::identity(n);
    let C = Mat::identity(n);
    let ccs = r1cs_to_ccs(A, B, C);
    
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n)
        .expect("goldilocks_auto_r1cs_ccs should find valid params");
    
    setup_ajtai_for_dims(n);
    let l = AjtaiSModule::from_global_for_dims(D, n).expect("AjtaiSModule init");
    
    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &ccs)
        .expect("dims");
    let ell_n = dims.ell_n;
    
    let r: Vec<K> = vec![K::from(F::from_u64(3)); ell_n];
    
    let m_in = 2;
    let z_seed_1: Vec<F> = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    let z_seed_2: Vec<F> = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    
    let (me1, Z1) = me_from_z_balanced(&params, &ccs, &l, &z_seed_1, &r, m_in)
        .expect("seed1 ME ok");
    let (me2, Z2) = me_from_z_balanced(&params, &ccs, &l, &z_seed_2, &r, m_in)
        .expect("seed2 ME ok");
    
    // k=3: 2 seed MEs in accumulator + 1 new MCS per step
    let acc = Accumulator { 
        me: vec![me1, me2], 
        witnesses: vec![Z1, Z2] 
    };
    
    let mut session = FoldingSession::new(FoldingMode::Mojo, params, l.clone())
        .with_initial_accumulator(acc, &ccs)
        .expect("with_initial_accumulator");
    
    {
        let x: Vec<F> = vec![F::ZERO, F::ZERO];
        let w: Vec<F> = vec![F::ZERO, F::ZERO];
        let input = ProveInput { ccs: &ccs, public_input: &x, witness: &w, output_claims: &[] };
        session.add_step_from_io(&input).expect("add_step 1");
    }
    
    {
        let x: Vec<F> = vec![F::ZERO, F::ZERO];
        let w: Vec<F> = vec![F::ZERO, F::ZERO];
        let input = ProveInput { ccs: &ccs, public_input: &x, witness: &w, output_claims: &[] };
        session.add_step_from_io(&input).expect("add_step 2");
    }
    
    {
        let x: Vec<F> = vec![F::ZERO, F::ZERO];
        let w: Vec<F> = vec![F::ZERO, F::ZERO];
        let input = ProveInput { ccs: &ccs, public_input: &x, witness: &w, output_claims: &[] };
        session.add_step_from_io(&input).expect("add_step 3");
    }
    
    let run = session.fold_and_prove(&ccs).expect("fold_and_prove should produce a FoldRun");
    
    // Test has k=3 fold fan-in, but params.k_rho=12 (DEC produces 12 children, not 3)
    assert_eq!(run.steps.len(), 3, "should have three fold steps");
    for (i, step) in run.steps.iter().enumerate() {
        assert_eq!(step.dec_children.len(), params.k_rho as usize, 
                   "step {} should have k_rho={} DEC children", i, params.k_rho);
    }
    assert_eq!(run.final_outputs.len(), params.k_rho as usize, 
               "final outputs should have k_rho={}", params.k_rho);
    
    let mcss_public = session.mcss_public();
    let ok = session.verify(&ccs, &mcss_public, &run)
        .expect("verify should run");
    assert!(ok, "mojo verification should pass");
}
