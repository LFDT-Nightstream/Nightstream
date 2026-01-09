#![cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]

use neo_ajtai::{set_global_pp_seeded, AjtaiSModule};
use neo_ccs::{CcsStructure, Mat, SModuleHomomorphism, SparsePoly};
use neo_closure_proof::{
    compute_accumulator_digest_v2, compute_obligations_digest_v1, prove_whir_p3_ajtai_opening_only_v1,
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
fn whir_p3_ajtai_opening_only_roundtrip_and_tamper() {
    let m = 16usize;
    let ccs = identity_ccs(m);
    let params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");

    let seed = [9u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, m, seed).expect("set_global_pp_seeded");
    let l = AjtaiSModule::from_global_for_dims(D, m).expect("committer");

    // Small bounded witness Z (not proving bounds here; just keep it sane).
    let mut z = Mat::zero(D, m, F::ZERO);
    for r in 0..D {
        for c in 0..m {
            let v = ((r * 31 + c * 17) % 3) as u64; // 0..2
            z.set(r, c, F::from_u64(v));
        }
    }
    let cmt = l.commit(&z);

    let me = neo_ccs::MeInstance {
        c: cmt,
        X: Mat::zero(D, 0, F::ZERO),
        r: Vec::<K>::new(),
        y: Vec::<Vec<K>>::new(),
        y_scalars: Vec::<K>::new(),
        m_in: 0,
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

    let stmt = ClosureStatementV1::new([1u8; 32], pp_id_digest, obligations_digest);
    let proof = prove_whir_p3_ajtai_opening_only_v1(&stmt, &params, &ccs, &obligations, &[z.clone()], &[])
        .expect("prove");

    verify_closure_v1_with_context(&stmt, &proof, Some(&params), Some(&ccs)).expect("verify");

    // Tamper witness: keep the same obligation commitment, but change Z.
    let mut z_bad = z;
    z_bad.as_mut_slice()[0] += F::ONE;
    let proof_bad =
        prove_whir_p3_ajtai_opening_only_v1(&stmt, &params, &ccs, &obligations, &[z_bad], &[]).expect("prove");

    assert!(
        verify_closure_v1_with_context(&stmt, &proof_bad, Some(&params), Some(&ccs)).is_err(),
        "tampered Z must be rejected"
    );
}
