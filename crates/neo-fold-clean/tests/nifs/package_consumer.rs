use std::{fs, path::PathBuf};

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{
    encode_pi_ccs_v1_1_public_input, load_pi_ccs_v1_1_package, pi_ccs_v1_1_state_hash,
    serialize_pi_ccs_v1_1_state_preimage, PiCcsV1_1ProofInputs,
};
use neo_fold_clean::paper::pi_ccs;
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim};
use neo_math::{D, F, K};
use nightstream_fprime::{
    PI_CCS_V1_1_MATRIX_COUNT, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT, PI_CCS_V1_1_SOURCE_COUNT,
};
use p3_field::PrimeCharacteristicRing;

// Last validated PiCCS-only identity. Do not replace it until the final
// package passes the matrix, assignment, parity, and mutation gates.
const PACKAGE_IDENTITY: [u64; 4] = [
    4_149_794_454_264_745_319,
    3_860_295_598_124_073_314,
    9_185_184_515_076_867_919,
    6_634_095_431_211_870_257,
];

fn package_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-v1.json")
}

fn parity_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-piccs-parity-v1.json")
}

fn fixture_commitment_key_words() -> Vec<u64> {
    let parity: serde_json::Value =
        serde_json::from_slice(&fs::read(parity_path()).expect("Lean parity bytes")).expect("Lean parity JSON");
    let input = parity[1].as_array().expect("Lean parity input");
    let authority = input[11]
        .as_array()
        .expect("Lean verifier-context authority");
    serde_json::from_value(authority[3].clone()).expect("Lean commitment-key authority words")
}

fn zero_ce_claim() -> CeClaim {
    CeClaim {
        c: Commitment::zeros(D, 18),
        X: Mat::zero(D, 1, F::ZERO),
        r: vec![K::ZERO; PI_CCS_V1_1_ROUND_COUNT],
        eval_k: vec![K::ZERO; D.next_power_of_two()],
        eval_a: vec![vec![K::ZERO; D.next_power_of_two()]; PI_CCS_V1_1_MATRIX_COUNT],
        m_in: D,
        fold_digest: [0; 32],
        adv: None,
    }
}

#[test]
fn application_consumes_identity_bound_lean_package_for_canonical_state() {
    let bytes = fs::read(package_path()).expect("Lean-emitted package bytes");
    let commitment_key_words = fixture_commitment_key_words();
    let (prover, verifier) = load_pi_ccs_v1_1_package(&bytes, PACKAGE_IDENTITY, &commitment_key_words)
        .expect("identity-bound application package runtime");
    let verifier_context = prover.verifier_context().clone();
    assert_eq!(prover.relation_identifier(), PACKAGE_IDENTITY);
    assert_eq!(verifier.relation_identifier(), PACKAGE_IDENTITY);
    assert_eq!(prover.verifier_context(), verifier.verifier_context());

    let fresh = CcsClaim {
        c: Commitment::zeros(D, 18),
        x: vec![F::ZERO; D],
        m_in: D,
        adv: None,
    };
    let proof = pi_ccs::Proof {
        sumcheck: pi_ccs::SumcheckProof::new(vec![
            vec![K::ZERO; PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT];
            PI_CCS_V1_1_ROUND_COUNT
        ]),
        outputs: vec![zero_ce_claim(); PI_CCS_V1_1_SOURCE_COUNT],
    };
    let proof_inputs =
        PiCcsV1_1ProofInputs::from_proof(std::slice::from_ref(&fresh), &proof).expect("exact v1_1 proof input shape");

    let running = vec![zero_ce_claim(); 16];
    let verifier_key_digest = verifier_context.digest().map(F::from_u64);
    let z0 = [F::from_u64(201), F::from_u64(202), F::from_u64(203), F::from_u64(204)];
    let current = [F::from_u64(301), F::from_u64(302), F::from_u64(303), F::from_u64(304)];
    let preimage = serialize_pi_ccs_v1_1_state_preimage(verifier_key_digest, 7, z0, current, &running, 1)
        .expect("canonical v1_1 state preimage");
    let digest = pi_ccs_v1_1_state_hash(&preimage).expect("Lean stateHash replay");
    let prior_public = encode_pi_ccs_v1_1_public_input(digest).expect("Lean encHash replay");
    let mut expected_public = prior_public.clone();
    expected_public.extend_from_slice(&digest);
    expected_public.extend_from_slice(&verifier_context.digest());
    let inputs = proof_inputs
        .into_package_inputs(preimage.clone(), preimage, prior_public, digest, verifier_context)
        .expect("typed package inputs");

    let proof = prover.prove(&inputs).expect("application package proof");
    verifier
        .verify(&proof, &expected_public)
        .expect("application package verification");
    let mut changed_public = expected_public;
    changed_public[0] = 0;
    assert!(verifier.verify(&proof, &changed_public).is_err());

    let stats = prover.matrix_stats();
    assert!(stats.a_nonzeros() > 0);
    assert!(stats.b_nonzeros() > 0);
    assert!(stats.c_nonzeros() > 0);
}
