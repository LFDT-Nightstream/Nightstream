#![cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]

use neo_ajtai::{set_global_pp_seeded, AjtaiSModule};
use neo_ccs::{CcsStructure, Mat, SModuleHomomorphism, SparsePoly};
use neo_closure_proof::{
    compute_accumulator_digest_v2, compute_obligations_digest_v1, prove_whir_p3_ajtai_opening_plus_x_v1,
    verify_closure_v1_with_context, ClosureStatementV1,
};
use neo_fold::shard::ShardObligations;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

fn identity_ccs(m: usize) -> CcsStructure<F> {
    let mat = Mat::identity(m);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

#[test]
fn whir_p3_ajtai_opening_plus_x_roundtrip_and_rejects_x_mismatch() {
    let m = 16usize;
    let m_in = 4usize;
    let ccs = identity_ccs(m);
    let params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");

    let seed = [11u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, m, seed).expect("set_global_pp_seeded");
    let l = AjtaiSModule::from_global_for_dims(D, m).expect("committer");

    // Witness Z and its public projection X.
    let mut z = Mat::zero(D, m, F::ZERO);
    for r in 0..D {
        for c in 0..m {
            let v = ((r * 31 + c * 17) % 3) as u64; // 0..2
            z.set(r, c, F::from_u64(v));
        }
    }
    let mut x = Mat::zero(D, m_in, F::ZERO);
    for r in 0..D {
        for c in 0..m_in {
            x.set(r, c, z[(r, c)]);
        }
    }
    let cmt = l.commit(&z);

    let me = neo_ccs::MeInstance {
        c: cmt,
        X: x.clone(),
        r: Vec::<K>::new(),
        y: Vec::<Vec<K>>::new(),
        y_scalars: Vec::<K>::new(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    };
    let obligations = ShardObligations {
        main: vec![me],
        val: vec![],
    };

    let pp_id_digest = neo_ajtai::compute_pp_id_digest_v1(D, m, params.kappa as usize, seed);
    let acc_main = compute_accumulator_digest_v2(params.b, obligations.main.as_slice());
    let acc_val = compute_accumulator_digest_v2(params.b, obligations.val.as_slice());
    let obligations_digest = compute_obligations_digest_v1(acc_main, acc_val, pp_id_digest);

    let stmt = ClosureStatementV1::new([2u8; 32], pp_id_digest, obligations_digest);
    let proof = prove_whir_p3_ajtai_opening_plus_x_v1(&stmt, &params, &ccs, &obligations, &[z.clone()], &[])
        .expect("prove");
    verify_closure_v1_with_context(&stmt, &proof, Some(&params), Some(&ccs)).expect("verify");

    // Now mismatch X: keep opening consistent by changing both Z and commitment, but keep X fixed.
    let mut z_bad = z;
    z_bad.as_mut_slice()[0] += F::ONE;
    let cmt_bad = l.commit(&z_bad);
    let me_bad = neo_ccs::MeInstance {
        c: cmt_bad,
        X: x,
        r: Vec::<K>::new(),
        y: Vec::<Vec<K>>::new(),
        y_scalars: Vec::<K>::new(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    };
    let obligations_bad = ShardObligations {
        main: vec![me_bad],
        val: vec![],
    };
    let acc_main_bad = compute_accumulator_digest_v2(params.b, obligations_bad.main.as_slice());
    let acc_val_bad = compute_accumulator_digest_v2(params.b, obligations_bad.val.as_slice());
    let obligations_digest_bad = compute_obligations_digest_v1(acc_main_bad, acc_val_bad, pp_id_digest);
    let stmt_bad = ClosureStatementV1::new([3u8; 32], pp_id_digest, obligations_digest_bad);

    let proof_bad = prove_whir_p3_ajtai_opening_plus_x_v1(&stmt_bad, &params, &ccs, &obligations_bad, &[z_bad], &[])
        .expect("prove");
    assert!(
        verify_closure_v1_with_context(&stmt_bad, &proof_bad, Some(&params), Some(&ccs)).is_err(),
        "X mismatch must be rejected"
    );
}
