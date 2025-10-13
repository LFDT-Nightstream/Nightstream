//! Pi-DEC transcript parity cross-checks.
//!
//! Ensures that prover and verifier absorb the same transcript items on honest inputs,
//! and that a simple tamper causes verification to fail while transcript parity diverges
//! (since we now absorb parent/digit X and fold digest explicitly).

#[allow(unused_imports)]
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_ccs::{CcsStructure, Mat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[allow(dead_code)]
fn make_dummy_ccs() -> CcsStructure<F> {
    // Minimal 1x1 CCS just to drive Pi-DEC structure; values don't matter
    let a = Mat::from_row_major(1, 1, vec![F::ONE]);
    let b = Mat::from_row_major(1, 1, vec![F::ONE]);
    let c = Mat::from_row_major(1, 1, vec![F::ONE]);
    neo_ccs::r1cs_to_ccs(a, b, c)
}

#[test]
#[cfg(feature = "fs-guard")]
fn pi_dec_transcript_parity_honest_and_tampered() {
    use neo_fold::pi_ccs::{pi_ccs_prove};
    use neo_fold::pi_rlc::{pi_rlc_prove};
    use neo_fold::pi_dec::{pi_dec, pi_dec_verify};
    use neo_params::NeoParams;
    use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule};
    use rand::SeedableRng;
    use neo_transcript::fs_guard as guard;

    // Setup Ajtai PP
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let pp = ajtai_setup(&mut rng, neo_math::D, 2, 1).expect("PP setup");
    let _ = set_global_pp(pp);

    let params = NeoParams::goldilocks_small_circuits();
    let s = make_dummy_ccs();
    let l = AjtaiSModule::from_global().expect("PP");

    // Build a tiny instance/witness
    let inst = neo_ccs::McsInstance { c: neo_ajtai::Commitment::zeros(neo_math::D, 2), x: vec![F::ZERO], m_in: 1 };
    let wit  = neo_ccs::McsWitness { w: vec![], Z: Mat::zero(neo_math::D, 1, F::ZERO) };

    // Prove Pi-CCS → Pi-RLC → Parent instance
    let mut tr_c = Poseidon2Transcript::new(b"neo/fold");
    let (me_list, _) = pi_ccs_prove(&mut tr_c, &params, &s, &[inst.clone(), inst.clone()], &[wit.clone(), wit.clone()], &l).expect("pi_ccs");
    let mut tr_r = Poseidon2Transcript::new(b"neo/fold");
    let (me_b, _) = pi_rlc_prove(&mut tr_r, &params, &me_list).expect("pi_rlc");
    let me_b_wit = neo_ccs::MeWitness { Z: Mat::zero(neo_math::D, 1, F::ZERO) };

    // Honest Pi-DEC prove
    guard::reset("pi_dec_parity");
    let mut tr_p = Poseidon2Transcript::new(b"neo/fold");
    let (digits, _digit_wits, proof_dec) = pi_dec(&mut tr_p, &params, &me_b, &me_b_wit, &s, &l)
        .expect("pi_dec");
    let spec = guard::take();

    // Honest verify: parity must match
    guard::reset("pi_dec_parity_verify_ok");
    let mut tr_v_ok = Poseidon2Transcript::new(b"neo/fold");
    let ok = pi_dec_verify(&mut tr_v_ok, &params, &me_b, &digits, &proof_dec, &l).expect("verify ok");
    assert!(ok, "honest verification must succeed");
    let ev_ok = guard::take();
    assert!(neo_transcript::fs_guard::first_mismatch(&spec, &ev_ok).is_none(), "prover and verifier transcripts should match in honest case");

    // Tamper parent X (absorbed on verifier): transcript parity (op/label/len) remains identical
    // by design, but structural checks reject. This confirms absorptions aren't drift-prone for FS.
    let mut me_tampered = me_b.clone();
    if me_tampered.X.rows() > 0 && me_tampered.X.cols() > 0 {
        me_tampered.X[(0, 0)] += F::ONE;
    }

    guard::reset("pi_dec_parity_verify_bad");
    let mut tr_v_bad = Poseidon2Transcript::new(b"neo/fold");
    let bad = pi_dec_verify(&mut tr_v_bad, &params, &me_tampered, &digits, &proof_dec, &l).expect("verify tampered");
    assert!(!bad, "tampered verification must fail");
    let ev_bad = guard::take();
    // Parity at (op,label,len) level remains identical (we don't compare contents in fs-guard)
    assert!(neo_transcript::fs_guard::first_mismatch(&spec, &ev_bad).is_none(), "transcript parity should remain identical at the event level");
}

// Provide a no-op variant so the test suite compiles without fs-guard feature
#[test]
#[cfg(not(feature = "fs-guard"))]
fn pi_dec_transcript_parity_honest_and_tampered() {
    eprintln!("fs-guard feature not enabled; skipping transcript parity test");
}
