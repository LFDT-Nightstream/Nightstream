use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_params::NeoParams;
use neo_math::F;
use neo_fold::pi_ccs::{pi_ccs_prove, context};
use neo_fold::pi_ccs::terminal::rhs_Q_apr;
use neo_fold::pi_ccs::transcript::{bind_header_and_instances, bind_me_inputs, sample_challenges};
use neo_fold::pi_ccs::pi_ccs_derive_transcript_tail;
use crate::helpers::{DummyS, create_test_ccs, mk_mcs};
use p3_field::PrimeCharacteristicRing;

#[test]
fn terminal_rhs_matches_running_sum() {
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
    let (out_me, proof) = pi_ccs_prove(&mut tr, &params, &s, &[mcs1.clone()], &[wit1], &[me_input.clone()], &[wit2.Z.clone()], &l).unwrap();

    // Replay to get running_sum and (α', r')
    let dims = context::build_dims_and_policy(&params, &s).unwrap();
    let mut tr_v = Poseidon2Transcript::new(b"prove/k2");
    bind_header_and_instances(&mut tr_v, &params, &s, &[mcs1.clone()], dims.ell, dims.d_sc, 0).unwrap();
    bind_me_inputs(&mut tr_v, &[me_input.clone()]).unwrap();
    let ch = sample_challenges(&mut tr_v, dims.ell_d, dims.ell).unwrap();

    let tail = pi_ccs_derive_transcript_tail(&params, &s, &[mcs1.clone()], &proof).unwrap();
    let (r_prime, alpha_prime) = tail.r.split_at(dims.ell_n);

    // Q(α',r') from public outputs
    let rhs = rhs_Q_apr(&s, &ch, r_prime, alpha_prime, &[mcs1], &[me_input], &out_me, &params).unwrap();

    assert_eq!(tail.running_sum, rhs, "Final running sum must equal Q(α', r') computed from outputs.");
}
