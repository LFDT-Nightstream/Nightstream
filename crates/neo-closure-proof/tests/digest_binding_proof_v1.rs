#![forbid(unsafe_code)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{Mat, MeInstance};
use neo_closure_proof::{
    compute_accumulator_digest_v2, compute_obligations_digest_v2, prove_obligations_digest_binding_proof_v1,
    verify_obligations_digest_binding_proof_v1, ClosureStatementV1,
};
use neo_fold::shard::ShardObligations;
use neo_math::{F as NeoF, K as NeoK};
use neo_params::NeoParams;

fn make_min_me(d: usize, kappa: usize) -> MeInstance<Cmt, NeoF, NeoK> {
    MeInstance {
        c: Cmt::zeros(d, kappa),
        X: Mat::from_row_major(d, 0, Vec::new()),
        r: Vec::new(),
        y: Vec::new(),
        y_scalars: Vec::new(),
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

    let obligations = ShardObligations {
        main: vec![make_min_me(d, kappa)],
        val: Vec::new(),
    };

    let pp_id_digest = [7u8; 32];
    let acc_main = compute_accumulator_digest_v2(params.b, obligations.main.as_slice());
    let acc_val = compute_accumulator_digest_v2(params.b, obligations.val.as_slice());
    let obligations_digest = compute_obligations_digest_v2(acc_main, acc_val, pp_id_digest);

    let stmt = ClosureStatementV1::new([0u8; 32], pp_id_digest, obligations_digest);

    let proof = prove_obligations_digest_binding_proof_v1(&stmt, &params, &obligations)
        .expect("digest binding proof should be provable");

    verify_obligations_digest_binding_proof_v1(&stmt, &params, &proof)
        .expect("digest binding proof should verify");

    // Tamper statement digest.
    let mut bad_stmt = stmt.clone();
    bad_stmt.obligations_digest[0] ^= 1;
    assert!(
        verify_obligations_digest_binding_proof_v1(&bad_stmt, &params, &proof).is_err(),
        "verification must fail if the public obligations_digest changes"
    );

    // Tamper proof bytes.
    let mut bad_proof = proof.clone();
    let mid = bad_proof.len() / 2;
    bad_proof[mid] ^= 1;
    assert!(
        verify_obligations_digest_binding_proof_v1(&stmt, &params, &bad_proof).is_err(),
        "verification must fail on proof tampering"
    );
}
