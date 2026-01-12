#![forbid(unsafe_code)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, Mat, MeInstance, SparsePoly};
use neo_closure_proof::{
    compute_accumulator_digest_v2, compute_obligations_digest_v2, prove_obligations_digest_binding_proof_v1,
    verify_obligations_digest_binding_proof_v1, ClosureStatementV1,
};
use neo_fold::shard::ShardObligations;
use neo_math::{F as NeoF, K as NeoK};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn identity_ccs(m: usize) -> CcsStructure<NeoF> {
    let mat = Mat::identity(m);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

fn make_min_me(d: usize, kappa: usize) -> MeInstance<Cmt, NeoF, NeoK> {
    MeInstance {
        c: Cmt::zeros(d, kappa),
        X: Mat::from_row_major(d, 0, Vec::new()),
        r: Vec::new(),
        y: vec![vec![NeoK::ZERO; d]],
        y_scalars: vec![NeoK::ZERO],
        m_in: 0,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

#[test]
fn obligations_digest_binding_proof_v1_roundtrips_and_tampers_fail() {
    // Keep this test small: digest-binding proof cost grows with κ·d.
    let mut params = NeoParams::goldilocks_127();
    params.kappa = 1;

    let d = params.d as usize;
    let kappa = params.kappa as usize;

    let ccs = identity_ccs(1);

    let obligations = ShardObligations {
        main: vec![make_min_me(d, kappa)],
        val: Vec::new(),
    };

    let pp_id_digest = [7u8; 32];
    let acc_main = compute_accumulator_digest_v2(params.b, obligations.main.as_slice());
    let acc_val = compute_accumulator_digest_v2(params.b, obligations.val.as_slice());
    let obligations_digest = compute_obligations_digest_v2(acc_main, acc_val, pp_id_digest);

    let stmt = ClosureStatementV1::new([0u8; 32], pp_id_digest, obligations_digest);

    // Deterministic roots/claim for this minimal all-zeros obligation.
    let root_z_u64 = [0u64; 8];
    let claimed_sum_u64 = 0u64;

    let proof = prove_obligations_digest_binding_proof_v1(&stmt, &params, &ccs, &obligations, &root_z_u64, claimed_sum_u64)
        .expect("digest binding proof should be provable");

    verify_obligations_digest_binding_proof_v1(&stmt, &params, &ccs, &root_z_u64, &proof)
        .expect("digest binding proof should verify");

    // Tamper statement digest.
    let mut bad_stmt = stmt.clone();
    bad_stmt.obligations_digest[0] ^= 1;
    assert!(
        verify_obligations_digest_binding_proof_v1(&bad_stmt, &params, &ccs, &root_z_u64, &proof).is_err(),
        "verification must fail if the public obligations_digest changes"
    );

    // Tamper proof bytes.
    let mut bad_proof = proof.clone();
    let mid = bad_proof.len() / 2;
    bad_proof[mid] ^= 1;
    assert!(
        verify_obligations_digest_binding_proof_v1(&stmt, &params, &ccs, &root_z_u64, &bad_proof).is_err(),
        "verification must fail on proof tampering"
    );
}
