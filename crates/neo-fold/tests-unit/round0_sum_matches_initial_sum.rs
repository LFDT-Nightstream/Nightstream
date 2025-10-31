use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_params::NeoParams;
use neo_math::{F, K};
use neo_fold::pi_ccs::pi_ccs_prove;
use neo_fold::sumcheck::poly_eval_k;
use crate::helpers::{DummyS, create_test_ccs, mk_mcs};
use p3_field::PrimeCharacteristicRing;

#[test]
fn round0_sum_matches_initial_sum() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let s = create_test_ccs().ensure_identity_first().unwrap();
    let l = DummyS;
    let m_in = 1;

    let (mcs1, wit1) = mk_mcs(&params, vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)], m_in, &l);
    let (mcs2, wit2) = mk_mcs(&params, vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)], m_in, &l);

    let mut tr_k1 = Poseidon2Transcript::new(b"fixture/k1");
    let (me_outs, _proof_k1) = neo_fold::pi_ccs_prove_simple(&mut tr_k1, &params, &s, &[mcs2.clone()], &[wit2.clone()], &l).unwrap();
    let me_input = me_outs[0].clone();

    let mut tr = Poseidon2Transcript::new(b"prove/k2");
    let (me_outputs, proof) = pi_ccs_prove(&mut tr, &params, &s, &[mcs1], &[wit1], &[me_input], &[wit2.Z], &l).unwrap();

    // The prover includes sc_initial_sum in the proof
    let init = proof.sc_initial_sum.expect("prover must bind initial sum");
    let p0 = &proof.sumcheck_rounds[0];
    let chk = poly_eval_k(p0, K::ZERO) + poly_eval_k(p0, K::ONE);
    assert_eq!(chk, init, "Round-0 invariant: p(0)+p(1) must equal initial_sum");

    // avoid dead-code warning
    assert!(!me_outputs.is_empty());
}
