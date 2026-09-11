//! Normal selected PiCCS proving from an executed witness, without supplied images or openings.

mod stage1_actual;
pub mod stage1_values;

use neo_math::{F, K};
use neo_reductions::optimized_engine::optimized_verify_with_trace;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use stage1_actual::{artifact, ActualBase};
use stage1_values::words;
use std::{fs, time::Instant};

#[test]
#[ignore = "Full selected base PiCCS call; witness, commitment and both cache passes are inside the 300-second test cap."]
fn selected_pi_ccs_prover_from_actual_base_sources() {
    let started = Instant::now();
    let expected: Value = serde_json::from_str(include_str!("fixtures/stage1_base_pi_ccs.json")).unwrap();
    let fixture_bytes = fs::read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json")).unwrap();
    // Test-data provenance only; this SHA does not enter a protocol binding.
    assert_eq!(
        json!(format!("{:x}", Sha256::digest(&fixture_bytes))),
        expected["base_fixture_sha256"]
    );
    let ActualBase {
        package,
        params,
        fresh,
        running,
    } = stage1_actual::load(
        &artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
        &fixture_bytes,
    );
    drop(fixture_bytes);
    assert_eq!(
        json!(package.structural_identifier()),
        expected["structural_identifier"]
    );
    assert_eq!(json!(package.package_identity()), expected["package_identity"]);
    let witness = fresh.witness;
    let fresh = fresh.claim;
    let running_witnesses = running.witnesses;
    let running = running.claims;
    let proof = package
        .prove_pi_ccs(
            std::slice::from_ref(&fresh),
            std::slice::from_ref(&witness),
            &running,
            &running_witnesses,
        )
        .unwrap();
    println!("actual_selected_prover_elapsed={:?}", started.elapsed());
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 8], 0);
    let (valid, trace) = optimized_verify_with_trace(
        &mut transcript,
        params.inner(),
        package.structure(),
        std::slice::from_ref(&fresh),
        &running,
        &proof.outputs,
        &proof.sumcheck,
    )
    .unwrap();
    assert!(valid);
    assert_eq!(
        json!(proof
            .sumcheck
            .sumcheck_rounds
            .iter()
            .map(|round| words(round))
            .collect::<Vec<_>>()),
        expected["rounds"]
    );
    assert!(proof
        .outputs
        .iter()
        .all(|output| output.r == trace.round_challenges));
    let observed = stage1_values::ccs_phase(&proof, &trace, valid);
    assert_eq!(observed, expected["phase"]);
    assert!(proof
        .outputs
        .iter()
        .all(|output| output.eval_a[13].iter().all(|value| *value == K::ZERO)));
    println!("full_selected_pi_ccs_elapsed={:?}", started.elapsed());
}
