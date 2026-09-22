//! Complete recursive production, golden handoff and later terminal checks.
//! The baseline full producer exceeded the normal test cap. This gate needs
//! approval for its specific invocation; an ignored result is not evidence.

use std::time::Instant;

use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use p3_field::PrimeCharacteristicRing;

use super::*;
use crate::folding::{ajtai_dec_mixer, ajtai_rlc_mixer, transcript::Transcript};
use crate::lifecycle::{
    encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage, ProofState,
    Stage1Envelope, VerifyError,
};

#[test]
#[ignore = "The baseline full producer exceeded 300 seconds; this full two-fold invocation requires a specific longer-run approval."]
fn fresh_recursive_producer_matches_golden_and_folds_successor() {
    let started = Instant::now();
    let application = crate::application::poseidon2_hash_chain_v1().unwrap();
    let reference = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
    let (prepared, binding) = crate::assembly::prepare(&reference, &application).unwrap();
    let package =
        PreparedLifecycle::from_package(prepared.into(), binding, crate::engine::Backend::Optimized, 114).unwrap();
    eprintln!("recursive preparation elapsed={:?}", started.elapsed());

    let base = read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json"));
    let private: Vec<u64> = serde_json::from_value(base[2].clone()).unwrap();
    let initial: [u64; 4] = private[30..34].try_into().unwrap();
    let message: [u64; 4] = private[private.len() - 4..].try_into().unwrap();
    let initial = initial.map(F::from_u64);
    let message = message.map(F::from_u64);
    let base_output = application
        .execute(initial, &message)
        .unwrap()
        .output_state();
    eprintln!("fresh recursive source started elapsed={:?}", started.elapsed());
    let base_proof = package
        .extend_with_output(
            Stage1Envelope::initial(initial),
            &message.map(|value| value.as_canonical_u64()),
            base_output,
            None,
        )
        .unwrap();
    assert_eq!(base_proof.state(), &Stage1State::new(1, initial, base_output));
    assert_eq!(base_output.as_slice(), fields(&base[4][0]));
    eprintln!("fresh recursive source elapsed={:?}", started.elapsed());

    let (state, proof_state) = base_proof.into_parts();
    let ProofState::Active { running, mut latest } = proof_state else {
        panic!("base generation must produce an active source");
    };
    assert_eq!(latest.instances.len(), 1);
    let fresh = latest.instances.pop().unwrap();
    let prior = running.claims_only();
    let fresh_claim = fresh.claim.clone();
    let first_fold = Instant::now();
    eprintln!("first full C/R/D started elapsed={:?}", started.elapsed());
    let (next, proof) = package.prove(vec![fresh], running).unwrap();
    eprintln!("first full C/R/D elapsed={:?}", first_fold.elapsed());

    // Only assertions read expected proof bytes. They never enter the prover.
    let saved = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage1_actual_nifs");
    assert_eq!(proof.canonical_bytes(), fs::read(saved.join("proof.native")).unwrap());
    let expected_nifs = read(artifact("nightstream-fprime-stage1-base-nifs-result-v1.json"));
    let params = Params::for_ccs_shape(
        package.structure.n,
        package.structure.m,
        package.structure.t(),
        package.structure.max_degree(),
        114,
    )
    .unwrap();
    let mut transcript = Transcript::session();
    let verified = folding::verify(
        &mut transcript,
        &params,
        &package.structure,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        std::slice::from_ref(&fresh_claim),
        &prior,
        &proof,
    )
    .unwrap();
    assert_eq!(next.claims, verified.claims);
    assert_eq!(next.parent_authority, verified.parent_authority);
    let outgoing = transcript
        .snapshot()
        .state()
        .map(|value| value.as_canonical_u64());
    assert_eq!(json!(outgoing), expected_nifs[7][9]);
    assert_eq!(json!(outgoing), expected_nifs[9][14]);
    assert_eq!(transcript.snapshot().absorbed(), 0);

    let request: Value = serde_json::from_slice(
        &fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/stage1_recursive_states/nonzero-running.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let recursive_message: [F; 4] = fields(&request[3]).try_into().unwrap();
    let recursive_words = recursive_message.map(|value| value.as_canonical_u64());
    let successor_output = application
        .execute(state.current(), &recursive_message)
        .unwrap()
        .output_state();
    let packet = package
        .step_inputs(&state, &prior, &fresh_claim, &proof, &recursive_words, successor_output)
        .unwrap();
    check_next_metadata(&packet, &proof);
    let expected_packet = read(artifact(
        "nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json",
    ));
    let encoded = package
        .package
        .encode_stage1_v1_1_inputs(packet.pi_ccs(), packet.pi_dec(), packet.application_witness())
        .unwrap();
    let expected_private: Vec<u64> = serde_json::from_value(expected_packet[2].clone()).unwrap();
    let expected_public: Vec<u64> = serde_json::from_value(expected_packet[3].clone()).unwrap();
    assert_eq!(encoded.private_values(), expected_private);
    assert_eq!(encoded.public_values(), expected_public);
    assert_eq!(json!(packet.output_digest()), expected_packet[4][1]);
    let successor = package.complete_step(packet, next.witnesses, None).unwrap();
    assert_eq!(successor.state().iteration(), request[0].as_u64().unwrap());
    assert_eq!(successor.state().z0().as_slice(), fields(&request[1]));
    assert_eq!(successor.state().current().as_slice(), fields(&request[2]));
    eprintln!("golden successor completed elapsed={:?}", started.elapsed());

    // The newly produced successor is the actual input of the next full fold.
    let final_output = application
        .execute(successor.state().current(), &recursive_message)
        .unwrap()
        .output_state();
    let second_fold = Instant::now();
    eprintln!("second full fold started elapsed={:?}", started.elapsed());
    let final_proof = package
        .extend_with_output(successor, &recursive_words, final_output, None)
        .unwrap();
    eprintln!("second full fold and successor elapsed={:?}", second_fold.elapsed());
    let expected_state = Stage1State::new(3, initial, final_output);
    assert_eq!(final_proof.state(), &expected_state);
    let terminal = Instant::now();
    eprintln!("later terminal acceptance started elapsed={:?}", started.elapsed());
    package.verify(&expected_state, &final_proof).unwrap();
    eprintln!("later terminal acceptance elapsed={:?}", terminal.elapsed());
    eprintln!(
        "new recursive result={}",
        json!({
            "package_identity": package.package_identity(),
            "iteration": expected_state.iteration(),
            "initial": initial.map(|value| value.as_canonical_u64()),
            "current": final_output.map(|value| value.as_canonical_u64()),
            "running_claims": &final_proof.running().unwrap().claims,
            "fresh_claim": &final_proof.fresh().unwrap().claim,
        })
    );

    // Rebuilding the digest and commitment must not hide a false running opening.
    let changed = rehash_false_running_opening(&package, final_proof);
    eprintln!(
        "rehashed running-opening rejection started elapsed={:?}",
        started.elapsed()
    );
    assert!(matches!(
        package.verify(&expected_state, &changed),
        Err(VerifyError::Running {
            index: 0,
            reason: "Eval_K differs from the complete witness opening"
        })
    ));
    eprintln!("complete recursive gate elapsed={:?}", started.elapsed());
}

/// Keep this exact rehash/recommit attack shared by the full and staged gates.
pub(super) fn rehash_false_running_opening(package: &PreparedLifecycle, final_proof: Stage1Envelope) -> Stage1Envelope {
    let expected_state = *final_proof.state();
    let (_, proof_state) = final_proof.into_parts();
    let ProofState::Active {
        mut running,
        mut latest,
    } = proof_state
    else {
        panic!("recursive output must remain active");
    };
    let mut fresh = latest.instances.pop().unwrap();
    running.claims[0].eval_k[0] += K::ONE;
    let preimage = serialize_pi_ccs_v1_1_state_preimage(
        package.binding.verifier_context().digest().map(F::from_u64),
        expected_state.iteration(),
        expected_state.z0(),
        expected_state.current(),
        &running.claims,
        1,
    )
    .unwrap();
    fresh.claim.x = encode_pi_ccs_v1_1_public_input(pi_ccs_v1_1_state_hash(&preimage).unwrap())
        .unwrap()
        .into_iter()
        .map(F::from_u64)
        .collect();
    let (positive, negative) = fresh.witness.Z.packed_signed_unit_column_masks().unwrap();
    let mut positive = positive.to_vec();
    let mut negative = negative.to_vec();
    for (column, &value) in fresh.claim.x.iter().enumerate() {
        let mask = 1u64 << (column % D);
        positive[column / D] &= !mask;
        negative[column / D] &= !mask;
        if value == F::ONE {
            positive[column / D] |= mask;
        } else {
            assert_eq!(value, F::ZERO);
        }
    }
    fresh.witness.Z = Mat::compact_signed_unit_from_column_masks(D, positive.len(), &positive, &negative).unwrap();
    drop((positive, negative));
    fresh.claim.c = commit_production_signed_unit_prefix_matrix(&fresh.witness.Z).unwrap();
    Stage1Envelope::from_parts(expected_state, running, fresh)
}
