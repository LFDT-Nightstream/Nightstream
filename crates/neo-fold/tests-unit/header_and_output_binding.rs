use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_params::NeoParams;
use neo_math::F;
use neo_fold::pi_ccs::{pi_ccs_prove, pi_ccs_verify};
use crate::helpers::{DummyS, create_test_ccs, mk_mcs};
use p3_field::PrimeCharacteristicRing;

#[test]
fn tampering_fold_digest_or_r_is_rejected() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let s = create_test_ccs().ensure_identity_first().unwrap();
    let l = DummyS;
    let m_in = 1;

    let (mcs1, wit1) = mk_mcs(&params, vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)], m_in, &l);
    let (mcs2, wit2) = mk_mcs(&params, vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)], m_in, &l);
    let mut tr_k1 = Poseidon2Transcript::new(b"fixture/k1");
    let (me_outs, _) = neo_fold::pi_ccs_prove_simple(&mut tr_k1, &params, &s, &[mcs2.clone()], &[wit2.clone()], &l).unwrap();
    let me_input = me_outs[0].clone();

    let mut tr = Poseidon2Transcript::new(b"prove/k2");
    let (mut me_outputs, proof) = pi_ccs_prove(&mut tr, &params, &s, &[mcs1.clone()], &[wit1], &[me_input.clone()], &[wit2.Z.clone()], &l).unwrap();

    // 1) tweak header_digest
    let mut tr_v = Poseidon2Transcript::new(b"prove/k2");
    let bad_proof = { let mut p = proof.clone(); p.header_digest[0] ^= 0xFF; p };
    let err = pi_ccs_verify(&mut tr_v, &params, &s, &[mcs1.clone()], &[me_input.clone()], &me_outputs, &bad_proof).unwrap_err();
    assert!(format!("{:?}", err).contains("header digest mismatch"));

    // 2) tweak out_me.r (first output)
    let mut tr_v2 = Poseidon2Transcript::new(b"prove/k2");
    me_outputs[0].r[0] = me_outputs[0].r[0] + me_outputs[0].r[0]; // garbage r
    let err2 = pi_ccs_verify(&mut tr_v2, &params, &s, &[mcs1], &[me_input], &me_outputs, &proof).unwrap_err();
    assert!(format!("{:?}", err2).contains("incorrect r values"));
}
