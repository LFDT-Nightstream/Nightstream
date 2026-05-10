mod support;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::paper::construction2::ProofState;

#[test]
fn verify_uncompressed_rejects_unfolded_trailing_latest() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 23)]])
        .expect("one-batch uncompressed proof");

    match &proof.state.proof {
        ProofState::Active { latest, .. } => {
            assert!(
                !latest.instances.is_empty(),
                "test requires the current one-step-lag state to contain a trailing latest"
            );
        }
        ProofState::Initial => panic!("one-batch proof must leave base state"),
    }

    assert!(
        neo_fold_clean::verify_uncompressed(&prep, &proof).is_err(),
        "verify_uncompressed accepted a proof whose trailing latest was never folded"
    );
}

#[test]
fn finish_uncompressed_folds_trailing_latest_and_verifies() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 29)]])
        .expect("one-batch uncompressed proof");

    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish uncompressed proof");
    assert!(
        finished.final_fold.is_some(),
        "one trailing latest needs a final fold proof"
    );
    match &finished.state.proof {
        ProofState::Active { latest, .. } => assert!(latest.instances.is_empty()),
        ProofState::Initial => panic!("finished one-batch proof must be active"),
    }

    neo_fold_clean::verify_uncompressed(&prep, &finished).expect("finished proof verifies");
}

#[test]
fn verify_uncompressed_rejects_tampered_final_fold() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 31)]])
        .expect("one-batch uncompressed proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish uncompressed proof");

    let final_fold = finished.final_fold.as_mut().expect("final fold proof");
    support::mutate_ce_claim(&mut final_fold.nifs.pi_dec.children[0]);

    assert!(
        neo_fold_clean::verify_uncompressed(&prep, &finished).is_err(),
        "verify_uncompressed accepted a tampered final fold proof"
    );
}

#[test]
fn verify_uncompressed_does_not_trust_recorded_acc_digest() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 33)]])
        .expect("one-batch uncompressed proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish uncompressed proof");

    finished.state.acc_digest = [0xA5; 32];

    neo_fold_clean::verify_uncompressed(&prep, &finished)
        .expect("recorded acc_digest is derived context, not verifier authority");
}

#[test]
fn finish_uncompressed_rejects_inconsistent_already_finalized_proof() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 37)]])
        .expect("one-batch uncompressed proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish uncompressed proof");

    match &mut finished.state.proof {
        ProofState::Active { latest, .. } => {
            latest.instances.push(support::toy_instance(&prep, 41));
        }
        ProofState::Initial => panic!("finished one-batch proof must be active"),
    }

    assert!(
        matches!(
            neo_fold_clean::finish_uncompressed(&prep, finished),
            Err(neo_fold_clean::Error::FinalizedProofInconsistent)
        ),
        "finish_uncompressed trusted an already-finalized proof with a non-empty latest"
    );
}

#[test]
fn prove_rejects_public_input_len_mismatch() {
    let prep = support::toy_preprocessing();
    let z = vec![F::ZERO; prep.structure.m];
    let mismatched =
        neo_fold_clean::CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, &prep.structure, &z, 0)
            .expect("mismatched public-input split instance");

    assert!(
        matches!(
            neo_fold_clean::prove(&prep, vec![vec![mismatched]]),
            Err(neo_fold_clean::Error::PublicInputLenMismatch { expected: 1, got: 0 })
        ),
        "prove accepted an instance whose m_in disagreed with preprocessing"
    );
}

#[test]
fn verify_uncompressed_rejects_public_batch_m_in_mismatch() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 43)]])
        .expect("one-batch uncompressed proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finish uncompressed proof");
    finished.public_batches[0][0].m_in = 0;

    assert!(
        matches!(
            neo_fold_clean::verify_uncompressed(&prep, &finished),
            Err(neo_fold_clean::Error::PublicInputLenMismatch { expected: 1, got: 0 })
        ),
        "verify_uncompressed accepted a public batch whose m_in disagreed with preprocessing"
    );
}

#[test]
fn compress_returns_unsupported_until_decider_lands() {
    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 47)]])
        .expect("one-batch uncompressed proof");

    assert!(
        matches!(
            neo_fold_clean::compress(&prep, proof),
            Err(neo_fold_clean::Error::Decider(
                neo_fold_clean::paper::decider::Error::Unsupported
            ))
        ),
        "compress should return an explicit unsupported error, not panic"
    );
}
