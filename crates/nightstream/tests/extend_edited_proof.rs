//! Extending a proof whose fresh commitment was edited in its bytes.
//! The rule: every proof that `extend` hands back verifies.

use std::{fs, path::PathBuf, time::Instant};

use neo_math::D;
use nightstream::{application::poseidon2_hash_chain_v1, Circuit, Engine, Verifier};
use nightstream_fprime::{
    PI_CCS_V1_2_FRESH_COMMITMENT_WORDS, PI_CCS_V1_2_MATRIX_COUNT, PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS,
    PI_CCS_V1_2_ROUND_COUNT, PI_DEC_V1_2_CHILD_COUNT,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

fn compiled() -> Circuit {
    let reference = fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .expect("saved selected circuit reference");
    Circuit::compile(&reference, poseidon2_hash_chain_v1().unwrap()).unwrap()
}

/// Add one to the first word of the fresh commitment, which stays below the
/// modulus. The layout is the one `encode_proof` writes: the header and state,
/// the running claims, the running witnesses, then the fresh commitment.
fn edit_fresh_commitment(mut bytes: Vec<u8>) -> Vec<u8> {
    let header = 16 + 8 + 9 * 8;
    let claim = (PI_CCS_V1_2_FRESH_COMMITMENT_WORDS
        + PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS
        + 2 * PI_CCS_V1_2_ROUND_COUNT
        + (PI_CCS_V1_2_MATRIX_COUNT + 1) * 2 * D)
        * 8;
    let fresh_words = (PI_CCS_V1_2_FRESH_COMMITMENT_WORDS + PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS) * 8;
    let children = PI_DEC_V1_2_CHILD_COUNT;
    let witnesses = bytes.len() - header - children * claim - fresh_words;
    assert_eq!(witnesses % (children + 1), 0, "proof layout");
    let offset = header + children * claim + children * (witnesses / (children + 1));
    let word = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    let edited = if word + 1 < F::ORDER_U64 { word + 1 } else { 0 };
    bytes[offset..offset + 8].copy_from_slice(&edited.to_le_bytes());
    bytes
}

/// Prove one step on `engine`, edit the fresh commitment of its bytes, and
/// extend the decoded proof. Returns whether extend handed back a proof, and
/// the verifier's answer for that proof.
fn extend_edited(engine: Engine) -> Option<Result<(), String>> {
    let started = Instant::now();
    let circuit = compiled();
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = Verifier::from_package(&circuit, Engine::Optimized, 114).unwrap();
    let initial = [202, 203, 204, 205].map(F::from_u64);
    let message = [7, 11, 13, 17].map(F::from_u64);

    let proof = prover.prove(initial, &message).unwrap();
    let bytes = edit_fresh_commitment(prover.encode_proof(&proof).unwrap());
    let edited = verifier
        .decode_proof(&bytes)
        .expect("an edited word below the modulus decodes");
    let rejected = verifier
        .verify(edited.state(), &edited)
        .unwrap_err()
        .to_string();
    eprintln!("edited proof: {rejected}");
    assert!(
        rejected.contains("fresh opening: fixed-key commitment differs from the witness"),
        "the edit must hit the fresh commitment: {rejected}"
    );

    let outcome = match prover.extend(&edited, &message) {
        Err(error) => {
            eprintln!("{engine:?} extend failed: {error}");
            None
        }
        Ok(next) => {
            let verdict = verifier
                .verify(next.state(), &next)
                .map_err(|error| error.to_string());
            eprintln!("{engine:?} extend returned a proof; verify: {verdict:?}");
            Some(verdict)
        }
    };
    eprintln!("elapsed={:?}", started.elapsed());
    outcome
}

#[test]
#[ignore = "Full production profile; run this test separately under the 300-second cap."]
fn cpu_extend_of_an_edited_fresh_commitment_returns_no_unverifiable_proof() {
    if let Some(verdict) = extend_edited(Engine::Optimized) {
        verdict.expect("extend handed back a proof that verify rejects");
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Full production Metal profile; run this test separately under the 300-second cap."]
fn metal_extend_of_an_edited_fresh_commitment_returns_no_unverifiable_proof() {
    if let Some(verdict) = extend_edited(Engine::Metal) {
        verdict.expect("extend handed back a proof that verify rejects");
    }
}
