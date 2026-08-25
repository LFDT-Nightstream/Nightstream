use std::panic::{catch_unwind, AssertUnwindSafe};

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::PiCcsProof;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

#[test]
fn raw_pi_ccs_verifier_rejects_malformed_ce_shape_without_panicking() {
    let s = CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, Vec::new())).expect("valid identity CCS");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("params");
    let commitment = Commitment::zeros(D, params.kappa as usize);
    let mcs = CcsClaim {
        c: commitment.clone(),
        x: vec![F::ZERO],
        m_in: 1,
        adv: None,
    };
    let ell_n = s.n.next_power_of_two().max(2).trailing_zeros() as usize;
    let malformed = CeClaim {
        c: commitment,
        X: Mat::zero(D, 1, F::ZERO),
        r: vec![K::ZERO; ell_n],
        eval_k: Vec::new(),
        eval_a: Vec::new(),
        m_in: 1,
        fold_digest: [0; 32],
        adv: None,
    };
    let proof = PiCcsProof::new(Vec::new());

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut tr = Poseidon2Transcript::new(b"redteam/raw_pi_ccs_shape");
        neo_reductions::pi_ccs_verify(
            &mut tr,
            &params,
            &s,
            core::slice::from_ref(&mcs),
            core::slice::from_ref(&malformed),
            &[],
            &proof,
        )
    }));

    let verdict = result.expect("public raw Pi_CCS verifier must return Err, not panic");
    assert!(verdict.is_err(), "malformed CE input must be rejected");
}

#[test]
fn public_pi_ccs_verifier_handles_documented_unpadded_v1_1_evaluations() {
    let structure =
        CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, Vec::new())).expect("valid identity CCS");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("params");
    let commitment = Commitment::zeros(D, params.kappa as usize);
    let mcs = CcsClaim {
        c: commitment.clone(),
        x: vec![F::ZERO],
        m_in: 1,
        adv: None,
    };
    let ell_n = structure.n.next_power_of_two().max(2).trailing_zeros() as usize;
    let claim = CeClaim {
        c: commitment,
        X: Mat::zero(D, 1, F::ZERO),
        r: vec![K::ZERO; ell_n],
        eval_k: vec![K::ZERO; D],
        eval_a: vec![vec![K::ZERO; D]],
        m_in: 1,
        fold_digest: [0; 32],
        adv: None,
    };
    let output = claim.clone();
    let proof = PiCcsProof::new(Vec::new());
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut transcript = Poseidon2Transcript::new(b"redteam/unpadded-v1_1-evaluations");
        neo_reductions::api::verify(
            neo_reductions::api::FoldingMode::Optimized,
            &mut transcript,
            &params,
            &structure,
            core::slice::from_ref(&mcs),
            core::slice::from_ref(&claim),
            core::slice::from_ref(&output),
            &proof,
        )
    }));

    let verdict = result.expect("validator-approved unpadded CE must not panic verifier");
    assert!(verdict.is_err(), "dummy proof is not expected to verify");
}
