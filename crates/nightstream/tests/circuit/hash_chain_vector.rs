//! Generic hash-chain vector: two Poseidon2HashChainV1 links per step take the
//! ordinary assembly route and a key prefix wider than the selected package.
//! The proof verifies; a changed statement, witness or commitment is rejected.

use super::*;
use crate::application::{poseidon2_hash_chain, poseidon2_hash_chain_step};
use crate::folding::CcsInstance;
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_math::D;
use p3_field::PrimeCharacteristicRing;
use std::{fs, path::Path};

fn reference() -> Vec<u8> {
    fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .unwrap()
}

#[test]
#[ignore = "Full production-profile check; run this test separately under the 300-second cap."]
fn two_link_hash_chain_base_step_verifies_and_rejects_changes() {
    let circuit = Circuit::compile(&reference(), poseidon2_hash_chain(2).unwrap()).unwrap();
    let prover = circuit.prover(Engine::Optimized, 114).unwrap();
    let verifier = Verifier::from_package(&circuit, Engine::Optimized, 114).unwrap();
    let initial = [1, 2, 3, 4].map(F::from_u64);
    let message = [5, 6, 7, 8, 9, 10, 11, 12].map(F::from_u64);
    let first = poseidon2_hash_chain_step(initial, message[..4].try_into().unwrap());
    let output = poseidon2_hash_chain_step(first, message[4..].try_into().unwrap());
    let expected = Stage1State::new(1, initial, output);

    assert!(matches!(
        prover.prove(initial, &message[..4]),
        Err(Error::Application(ApplicationError::PrivateInputCount {
            expected: 8,
            actual: 4
        }))
    ));
    let proof = prover.prove(initial, &message).unwrap();
    assert_eq!(proof.state(), &expected);
    verifier.verify(&expected, &proof).unwrap();

    // Reassemble the same openings under a supplied statement.
    let running = proof.running().unwrap().clone();
    let fresh = proof.fresh().unwrap().clone();
    let reseal =
        |state: &Stage1State, fresh: CcsInstance| Stage1Envelope::from_parts(state.clone(), running.clone(), fresh);
    verifier
        .verify(&expected, &reseal(&expected, fresh.clone()))
        .unwrap();

    // The openings do not prove another endpoint, initial state or counter.
    let mut changed = output;
    changed[0] += F::ONE;
    for state in [
        Stage1State::new(1, initial, changed),
        Stage1State::new(1, changed, output),
        Stage1State::new(2, initial, output),
    ] {
        assert!(matches!(
            verifier.verify(&state, &reseal(&state, fresh.clone())),
            Err(Error::Verify(VerifyError::Fresh(
                "public input differs from the recomputed terminal state hash"
            )))
        ));
    }

    // Change one application coordinate to another signed unit. The old
    // commitment no longer opens, and a recomputed commitment opens to a
    // witness that violates the CCS rows.
    let column = circuit.compiled.package.application().private_range().start;
    let (lane, block) = (column % D, column / D);
    let mut witness = fresh.clone();
    let value = witness.witness.Z[(lane, block)];
    witness
        .witness
        .Z
        .set(lane, block, if value == F::ZERO { F::ONE } else { F::ZERO });
    assert!(matches!(
        verifier.verify(&expected, &reseal(&expected, witness.clone())),
        Err(Error::Verify(VerifyError::Fresh(
            "fixed-key commitment differs from the witness"
        )))
    ));
    witness.claim.c = commit_production_signed_unit_prefix_matrix(&witness.witness.Z).unwrap();
    assert!(matches!(
        verifier.verify(&expected, &reseal(&expected, witness)),
        Err(Error::Verify(VerifyError::FreshRelation(_)))
    ));

    // A changed commitment no longer opens to the witness.
    let mut commitment = fresh;
    commitment.claim.c.data[0] += F::ONE;
    assert!(matches!(
        verifier.verify(&expected, &reseal(&expected, commitment)),
        Err(Error::Verify(VerifyError::Fresh(
            "fixed-key commitment differs from the witness"
        )))
    ));
}
