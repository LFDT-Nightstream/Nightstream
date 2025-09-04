use neo_params::NeoParams;
use neo_math::F as Fq;
use neo_ccs::{Mat, McsInstance, McsWitness, r1cs_to_ccs};
use neo_ajtai::{setup as ajtai_setup, commit as ajtai_commit, decomp_b, DecompStyle, set_global_pp};
use neo_fold::fold_ccs_instances;
use p3_field::PrimeCharacteristicRing as _;
use rand::SeedableRng;

#[test]
fn fold_pipeline_smoke() {
    let params = NeoParams::goldilocks_127();

    // Build R1CS → CCS for the identity: 1 * z0 = z0
    let a = Mat::from_row_major(1, 2, vec![Fq::ONE, Fq::ZERO]);
    let b = Mat::from_row_major(1, 2, vec![Fq::ZERO, Fq::ONE]);
    let c = Mat::from_row_major(1, 2, vec![Fq::ZERO, Fq::ONE]);
    let ccs = r1cs_to_ccs(a, b, c);

    // Ajtai PP and publish globally
    let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
    let d = neo_math::ring::D;
    let kappa = 8;
    let m = ccs.m; // 2
    let pp = ajtai_setup(&mut rng, d, kappa, m).unwrap();
    set_global_pp(pp.clone()).unwrap();

    // Two MCS instances (k+1 = 2) with different z0
    let mut instances = Vec::new();
    let mut witnesses = Vec::new();
    for i in 0..2 {
        let z = vec![Fq::ONE, Fq::from_u64(5 + i as u64)]; // [const 1, z0]
        let z_cols = decomp_b(&z, params.b, d, DecompStyle::Balanced);
        let cmt = ajtai_commit(&pp, &z_cols);

        let mut z_mat = Mat::zero(d, m, Fq::ZERO);
        for col in 0..m { for row in 0..d { z_mat[(row, col)] = z_cols[col*d + row]; } }

        instances.push(McsInstance { c: cmt, x: vec![], m_in: 0 });
        witnesses.push(McsWitness { w: z, Z: z_mat });
    }

    // Fold: k+1=2 → Π_RLC reduces to 1 → Π_DEC expands to k=12 final instances
    let (out_mes, _witnesses, proof) = fold_ccs_instances(&params, &ccs, &instances, &witnesses).expect("fold ok");
    assert_eq!(out_mes.len(), params.k as usize, "should produce k final instances from Π_DEC");
    
    // Verify all output instances have consistent structure
    for me in &out_mes {
        assert_eq!(me.y.len(), ccs.t(), "y count must equal t for each ME");
        assert_eq!(me.r.len(), 0, "For n=1, r should be empty (ell=0)");
    }
    
    // Verify proof structure
    assert!(proof.pi_ccs_proof.sumcheck_rounds.len() <= ccs.n.trailing_zeros() as usize);
    assert!(proof.pi_rlc_proof.rho_elems.len() > 0, "Should have RLC coefficients");
    
    println!("✅ Folding pipeline test passed: {} → 1 → {} instances", instances.len(), out_mes.len());
}
