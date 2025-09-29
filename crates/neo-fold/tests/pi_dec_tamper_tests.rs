//! Π_DEC tamper tests for parent instance binding

use neo_fold::pi_ccs::{pi_ccs_prove};
use neo_fold::pi_rlc::{pi_rlc_prove};
use neo_fold::pi_dec::{pi_dec, pi_dec_verify};
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_params::NeoParams;
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term, MeWitness, SModuleHomomorphism};
use neo_ajtai::{setup as ajtai_setup, set_global_pp};
use neo_math::{F, D};
use rand::SeedableRng;
use p3_field::PrimeCharacteristicRing;

fn make_ccs() -> CcsStructure<F> {
    // Small CCS: f = X2*X0 - X1
    let a = Mat::from_row_major(2, 2, vec![F::ONE, F::ZERO, F::ZERO, F::ONE]);
    let b = Mat::from_row_major(2, 2, vec![F::ONE, F::ZERO, F::ZERO, F::ONE]);
    let c = Mat::from_row_major(2, 2, vec![F::ONE, F::ZERO, F::ZERO, F::ONE]);
    let terms = vec![
        Term { coeff: F::ONE,  exps: vec![1, 0, 1] }, // X0 * X2
        Term { coeff: -F::ONE, exps: vec![0, 1, 0] }, // -X1
    ];
    let f = SparsePoly::new(3, terms);
    CcsStructure::new(vec![a, b, c], f).unwrap()
}

fn setup_env() {
    // Minimal Ajtai PP for commitments
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
    let pp = ajtai_setup(&mut rng, D, 2, 2).expect("setup");
    let _ = set_global_pp(pp);
}

fn make_instance_and_witness(seed: u64) -> (neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>) {
    use neo_ajtai::{decomp_b, DecompStyle, AjtaiSModule};
    let x = vec![F::from_u64(seed % 5)];
    let w = vec![F::from_u64((seed % 7) + 1)];
    let mut z = x.clone(); z.extend_from_slice(&w);
    let digits = decomp_b(&z, 2, D, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; D * z.len()];
    for col in 0..z.len() { for row in 0..D { row_major[row * z.len() + col] = digits[col * D + row]; } }
    let z_mat = Mat::from_row_major(D, z.len(), row_major);
    let l = AjtaiSModule::from_global().expect("PP");
    let c = l.commit(&z_mat);
    (neo_ccs::McsInstance { c, x, m_in: 1 }, neo_ccs::McsWitness { w, Z: z_mat })
}

#[test]
fn pi_dec_detects_parent_x_tamper() {
    use neo_ajtai::AjtaiSModule;
    setup_env();
    let params = NeoParams::goldilocks_small_circuits();
    let s = make_ccs();

    // Build three instances for a simple fold
    let (inst0, wit0) = make_instance_and_witness(0);
    let (inst1, wit1) = make_instance_and_witness(1);
    let (inst2, wit2) = make_instance_and_witness(2);
    let instances = vec![inst0.clone(), inst1.clone(), inst2.clone()];
    let witnesses = vec![wit0, wit1, wit2];
    let l = AjtaiSModule::from_global().expect("PP");

    // Π_CCS → Π_RLC → get parent me_B and its witness Z'
    let mut tr_c = Poseidon2Transcript::new(b"neo/fold");
    let (me_list, _) = pi_ccs_prove(&mut tr_c, &params, &s, &instances, &witnesses, &l).expect("pi_ccs");

    let mut tr_r = Poseidon2Transcript::new(b"neo/fold");
    let (me_b, pi_rlc_proof) = pi_rlc_prove(&mut tr_r, &params, &me_list).expect("pi_rlc");

    // Build parent witness Z' via the S-action recombination (same as pipeline)
    use neo_math::{Rq, cf_inv, SAction};
    let mut z_prime = Mat::zero(D, me_b.m_in + 1, F::ZERO); // cols matches m_in in pipeline pattern
    let rho_ring: Vec<Rq> = pi_rlc_proof.rho_elems.iter().map(|cs| cf_inv(*cs)).collect();
    for (idx, w) in witnesses.iter().enumerate() {
        let s_action = SAction::from_ring(rho_ring[idx]);
        for c in 0..z_prime.cols() {
            let mut col = [F::ZERO; D];
            for r in 0..D { col[r] = w.Z[(r, c % w.Z.cols())]; }
            let rot = s_action.apply_vec(&col);
            for r in 0..D { z_prime[(r, c)] += rot[r]; }
        }
    }
    let me_b_wit = MeWitness { Z: z_prime };

    // Π_DEC
    let mut tr_d = Poseidon2Transcript::new(b"neo/fold");
    let (digits, _digit_wits, proof_dec) = pi_dec(&mut tr_d, &params, &me_b, &me_b_wit, &s, &l).expect("pi_dec");

    // Tamper parent X matrix and verify must fail
    let mut me_tampered = me_b.clone();
    if me_tampered.X.rows() > 0 && me_tampered.X.cols() > 0 {
        me_tampered.X[(0, 0)] += F::ONE;
    }
    let mut tr_v = Poseidon2Transcript::new(b"neo/fold");
    let ok = pi_dec_verify(&mut tr_v, &params, &me_tampered, &digits, &proof_dec, &l).expect("verify");
    assert!(!ok, "Π_DEC must reject when parent X is tampered");
}
