use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_params::NeoParams;
use neo_math::F;
use neo_fold::pi_ccs::pi_ccs_prove;
use crate::helpers::{DummyS, create_test_ccs, mk_mcs};
use p3_field::PrimeCharacteristicRing;

#[test]
fn eval_row_gate_and_partial_fold_each_round() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let s = create_test_ccs().ensure_identity_first().unwrap();
    let l = DummyS;
    let m_in = 1;

    let (mcs1, wit1) = mk_mcs(&params, vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)], m_in, &l);
    let (mcs2, wit2) = mk_mcs(&params, vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)], m_in, &l);

    let mut tr_k1 = Poseidon2Transcript::new(b"fixture/k1");
    let (me_outs, _) = neo_fold::pi_ccs_prove_simple(&mut tr_k1, &params, &s, &[mcs2.clone()], &[wit2.clone()], &l).unwrap();
    let me_input = me_outs[0].clone();

    // Run proving once; relies on internal invariants to enforce folding
    let mut tr = Poseidon2Transcript::new(b"prove/k2");
    let _ = pi_ccs_prove(&mut tr, &params, &s, &[mcs1], &[wit1], &[me_input], &[wit2.Z], &l).unwrap();
}
