//! Generic hash-chain vector: two Poseidon2HashChainV1 links per step take the
//! ordinary assembly route and a key prefix wider than the selected package.
//! The base proof and one recursive fold verify; a changed statement, witness
//! or commitment is rejected.

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

/// The native result of one application step: two links of the chain.
fn chain(state: [F; 4], message: &[F]) -> [F; 4] {
    let first = poseidon2_hash_chain_step(state, message[..4].try_into().unwrap());
    poseidon2_hash_chain_step(first, message[4..].try_into().unwrap())
}

#[test]
#[ignore = "Full production-profile check; run this test separately under the 300-second cap."]
fn two_link_hash_chain_base_step_verifies_and_rejects_changes() {
    let circuit = Circuit::compile(&reference(), poseidon2_hash_chain(2).unwrap()).unwrap();
    let prover = circuit.prover(Engine::Optimized, 114).unwrap();
    let verifier = Verifier::from_package(&circuit, Engine::Optimized, 114).unwrap();
    let initial = [1, 2, 3, 4].map(F::from_u64);
    let message = [5, 6, 7, 8, 9, 10, 11, 12].map(F::from_u64);
    let output = chain(initial, &message);
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
    let application = &circuit.compiled.application;
    let manifest: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../artifacts/shared-verifier-v1.json")).unwrap();
    let local = manifest["ports"]
        .as_array()
        .unwrap()
        .iter()
        .find(|port| port["name"] == "application_local")
        .unwrap();
    let counts = [
        1,
        application.private_input_count(),
        application.generated_range().len(),
        application.row_count(),
    ];
    let retained_start: usize = local["retained_start"]
        .as_array()
        .unwrap()
        .iter()
        .zip(counts)
        .map(|(coefficient, count)| coefficient.as_u64().unwrap() as usize * count)
        .sum();
    let retained_end = retained_start
        + application.generated_range().len() * manifest["geometry"]["field_slot_width"].as_u64().unwrap() as usize;
    let column = retained_start;
    assert!((retained_start..retained_end).contains(&column));
    assert!(retained_end <= circuit.compiled.package.logical_column_count());
    assert_ne!(column, circuit.compiled.package.application().private_range().start);
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

/// One recursive fold above the old key limit: PiDEC commits the children
/// under the wider prefix, and the terminal verifier accepts the extension.
fn two_link_hash_chain_folds(engine: Engine) {
    let circuit = Circuit::compile(&reference(), poseidon2_hash_chain(2).unwrap()).unwrap();
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = Verifier::from_package(&circuit, engine, 114).unwrap();
    let initial = [1, 2, 3, 4].map(F::from_u64);
    let first = [5, 6, 7, 8, 9, 10, 11, 12].map(F::from_u64);
    let second = [13, 14, 15, 16, 17, 18, 19, 20].map(F::from_u64);
    let base = prover.prove(initial, &first).unwrap();
    let proof = prover.extend(&base, &second).unwrap();
    let expected = Stage1State::new(2, initial, chain(chain(initial, &first), &second));
    assert_eq!(proof.state(), &expected);
    verifier.verify(&expected, &proof).unwrap();

    // The recursive openings do not prove another endpoint.
    let mut changed = expected.current();
    changed[0] += F::ONE;
    let state = Stage1State::new(2, initial, changed);
    let envelope = Stage1Envelope::from_parts(
        state.clone(),
        proof.running().unwrap().clone(),
        proof.fresh().unwrap().clone(),
    );
    assert!(matches!(
        verifier.verify(&state, &envelope),
        Err(Error::Verify(VerifyError::Fresh(
            "public input differs from the recomputed terminal state hash"
        )))
    ));
}

#[test]
// This covers the CPU PiDEC capacity check. It exceeded the 300-second cap on
// the development Mac (2026-09-25); run it only with owner approval for that run.
#[ignore = "Full production-profile CPU fold; exceeds the 300-second cap on this host."]
fn two_link_hash_chain_folds_on_cpu() {
    two_link_hash_chain_folds(Engine::Optimized);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Full production Metal fold; run this test separately under the 300-second cap."]
fn two_link_hash_chain_folds_on_metal() {
    two_link_hash_chain_folds(Engine::Metal);
}
