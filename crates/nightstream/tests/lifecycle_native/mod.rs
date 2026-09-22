//! Native migration checks. Saved Lean inputs are test data, never producer input.
use super::{PreparedLifecycle, Stage1State, Stage1StepInputs, StepInputError};
use crate::folding::{self, pi_ccs, pi_dec, pi_rlc, CcsClaim, CeClaim, NifsProof, Params, RunningInstance};
use neo_ajtai::Commitment;
use neo_ccs::{LaneCommitments, Mat};
use neo_math::{from_complex, D, F, K};
use nightstream_fprime::{load_poseidon2_hash_chain_v1_package, PI_CCS_V1_1_STATE_PREIMAGE_WORDS};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};
use std::{fs, path::PathBuf};

#[cfg(feature = "metal")]
mod terminal_engine;
fn output(current: [F; 4], message: [F; 4]) -> [F; 4] {
    crate::application::poseidon2_hash_chain_v1()
        .unwrap()
        .execute(current, &message)
        .unwrap()
        .output_state()
}
fn artifact(name: &str) -> PathBuf {
    let directory = if name == "nightstream-fprime-stage1-poseidon2-hash-chain-v1.json" {
        "artifacts"
    } else {
        "tests/fixtures/lean"
    };
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(directory)
        .join(name)
}

fn read(path: PathBuf) -> Value {
    serde_json::from_slice(&fs::read(path).expect("retained conformance input")).unwrap()
}

fn field(word: u64) -> F {
    assert!(word < F::ORDER_U64, "canonical fixture word");
    F::from_u64(word)
}

fn fields(value: &Value) -> Vec<F> {
    serde_json::from_value::<Vec<u64>>(value.clone())
        .unwrap()
        .into_iter()
        .map(field)
        .collect()
}

fn extensions(value: &Value) -> Vec<K> {
    serde_json::from_value::<Vec<[u64; 2]>>(value.clone())
        .unwrap()
        .into_iter()
        .map(|[low, high]| from_complex(field(low), field(high)))
        .collect()
}

fn evaluation(value: &Value) -> Vec<K> {
    let mut result = extensions(value);
    assert_eq!(result.len(), D);
    result.resize(D.next_power_of_two(), K::ZERO);
    result
}

fn commitment(value: &Value) -> Commitment {
    let data = fields(value);
    assert_eq!(data.len(), D * 22);
    Commitment { d: D, kappa: 22, data }
}

fn frame(words: [u64; 4]) -> [u8; 32] {
    let mut bytes = [0; 32];
    for (lane, word) in words.into_iter().enumerate() {
        bytes[lane * 8..lane * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn claim(value: &Value, fold_digest: [u8; 32]) -> CeClaim {
    let public = fields(&value[1]);
    assert_eq!(public.len(), 270);
    let mut x = Mat::zero(D, public.len() / D, F::ZERO);
    for (index, value) in public.into_iter().enumerate() {
        x[(index % D, index / D)] = value;
    }
    let point = extensions(&value[2]);
    assert_eq!(point.len(), 28);
    let matrices = value[4].as_array().unwrap();
    assert_eq!(matrices.len(), 14);
    CeClaim {
        c: commitment(&value[0]),
        X: x,
        r: point,
        eval_k: evaluation(&value[3]),
        eval_a: matrices.iter().map(evaluation).collect(),
        m_in: 270,
        fold_digest,
        adv: None,
    }
}

fn proof(actual: &Value) -> NifsProof {
    let input = &actual["pi_ccs_input"];
    let phase = &actual["pi_ccs_phase"];
    let outgoing: Vec<u64> = serde_json::from_value(phase[14].clone()).unwrap();
    let digest = frame(outgoing[..4].try_into().unwrap());
    let outputs = (0..17)
        .map(|source| {
            claim(
                &json!([
                    phase[10][source],
                    phase[11][source],
                    phase[6],
                    phase[12][source],
                    phase[13][source]
                ]),
                digest,
            )
        })
        .collect();
    let children = &actual["children"];
    NifsProof {
        pi_ccs: pi_ccs::Proof {
            sumcheck: pi_ccs::SumcheckProof::new(
                input[3]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(extensions)
                    .collect(),
            ),
            outputs,
        },
        pi_rlc: pi_rlc::Proof {
            combined: claim(&actual["pi_rlc_parent"], digest),
        },
        pi_dec: pi_dec::Proof {
            children: (0..16)
                .map(|child| {
                    claim(
                        &json!([
                            children[1][child],
                            children[2][child],
                            children[0],
                            children[3][child],
                            children[4][child]
                        ]),
                        digest,
                    )
                })
                .collect(),
        },
    }
}

fn check_next_metadata(packet: &Stage1StepInputs, original: &NifsProof) {
    let next = packet.next_running();
    assert!(
        next.witnesses.is_empty(),
        "this packet does not construct child openings"
    );
    assert_eq!(next.claims.len(), 16);
    let digest = frame(packet.output_digest());
    for (actual, original) in next.claims.iter().zip(&original.pi_dec.children) {
        let mut expected = original.clone();
        expected.fold_digest = digest;
        assert_eq!(*actual, expected, "only the next-state frame metadata changes");
    }
    let mut parent = original.pi_rlc.combined.clone();
    parent.fold_digest = digest;
    assert_eq!(next.parent_authority.as_ref(), Some(&parent));
}
struct Fixture {
    package: PreparedLifecycle,
    loaded: nightstream_fprime::LoadedPerApplicationPackage,
    expected: Value,
    wire: Vec<u8>,
    state: Stage1State,
    running: RunningInstance,
    fresh: CcsClaim,
    proof: NifsProof,
    message: [F; 4],
}

impl Fixture {
    fn load() -> Self {
        let bytes = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
        let source = load_poseidon2_hash_chain_v1_package(&bytes).unwrap();
        let binding = source.production_verifier_binding().unwrap();
        let package =
            PreparedLifecycle::from_package(source.into(), binding, crate::engine::Backend::Optimized).unwrap();
        let loaded = load_poseidon2_hash_chain_v1_package(&bytes).unwrap();
        let base = read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json"));
        let expected = read(artifact(
            "nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json",
        ));
        let saved = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage1_actual_nifs");
        let actual = read(saved.join("actual_result.json"));
        let proof = proof(&actual);
        let wire = fs::read(saved.join("proof.native")).unwrap();
        assert!(proof.canonical_bytes() == wire, "exact retained native proof");
        let params = Params::for_ccs_shape(
            package.structure().n,
            package.structure().m,
            package.structure().t(),
            package.structure().max_degree(),
        )
        .unwrap();
        let mut running = RunningInstance::canonical_zero(&params, package.structure(), 270)
            .unwrap()
            .claims_only();
        let prior_digest: [u64; 4] = serde_json::from_value(base[4][1].clone()).unwrap();
        for child in &mut running.claims {
            child.fold_digest = frame(prior_digest);
        }
        running.parent_authority.as_mut().unwrap().fold_digest = frame(prior_digest);
        let fresh = CcsClaim {
            c: commitment(&actual["pi_ccs_input"][1]),
            x: fields(&actual["pi_ccs_input"][2]),
            m_in: 270,
            adv: None,
        };
        let z0 = [202, 203, 204, 205].map(field); // Existing base fixture inputs.
        let current: [F; 4] = fields(&base[4][0]).try_into().unwrap();
        let state = Stage1State::new(1, z0, current);
        let message = [7, 11, 13, 17].map(field); // Same advice as the Lean reference.
        Self {
            package,
            loaded,
            expected,
            wire,
            state,
            running,
            fresh,
            proof,
            message,
        }
    }
    fn packet(&self) -> Stage1StepInputs {
        self.package
            .step_inputs(
                &self.state,
                &self.running,
                &self.fresh,
                &self.proof,
                &self.message.map(|f| f.as_canonical_u64()),
                output(self.state.current(), self.message),
            )
            .unwrap()
    }
}
#[test]
fn actual_nifs_builds_the_checked_successor_assignment() {
    let Fixture {
        package,
        loaded,
        expected,
        wire,
        state,
        running,
        fresh,
        proof,
        message,
        ..
    } = Fixture::load();
    let z0 = state.z0();
    let current = state.current();
    let packet = package
        .step_inputs(
            &state,
            &running,
            &fresh,
            &proof,
            &message.map(|f| f.as_canonical_u64()),
            output(current, message),
        )
        .unwrap();
    let encoded = loaded
        .encode_stage1_v1_1_inputs(packet.pi_ccs(), packet.pi_dec(), packet.application_witness())
        .unwrap();
    let private: Vec<u64> = serde_json::from_value(expected[2].clone()).unwrap();
    let public: Vec<u64> = serde_json::from_value(expected[3].clone()).unwrap();
    assert_eq!(encoded.private_values().len(), private.len());
    let first_difference = encoded
        .private_values()
        .iter()
        .zip(&private)
        .position(|(a, b)| a != b);
    assert_eq!(first_difference, None, "first differing caller-private input word");
    assert_eq!(encoded.public_values(), public, "every caller-public input word");
    let state_width = PI_CCS_V1_1_STATE_PREIMAGE_WORDS;
    assert_eq!(packet.output_preimage(), &private[state_width..2 * state_width]);
    assert_eq!(json!(packet.output_digest()), expected[4][1]);
    assert_eq!(json!(packet.next_public_input()), expected[4][2]);
    check_next_metadata(&packet, &proof);
    assert!(
        proof.canonical_bytes() == wire,
        "handoff does not mutate its source proof"
    );

    // Exercise the selected public consumer, then check its complete low-norm
    // logical carrier and public output. Independent row checks run separately.
    let assignment = package
        .execute_step_witness(packet.pi_ccs(), packet.pi_dec(), packet.application_witness())
        .unwrap();
    let logical = loaded.execute_logical_assignment(&assignment).unwrap();
    assert!(logical
        .balanced_values()
        .iter()
        .all(|value| (-1..=1).contains(value)));
    for (column, expected) in packet.next_public_input().iter().enumerate() {
        assert_eq!(logical.value(column).unwrap(), *expected);
    }
    drop((logical, assignment));

    for iteration in [0, F::ORDER_U64 - 1, F::ORDER_U64, u64::MAX] {
        let invalid = Stage1State::new(iteration, z0, current);
        assert!(package
            .step_inputs(
                &invalid,
                &running,
                &fresh,
                &proof,
                &message.map(|f| f.as_canonical_u64()),
                output(current, message)
            )
            .is_err());
    }
    let mut detached_current = current;
    detached_current[0] += F::ONE;
    let invalid = Stage1State::new(1, z0, detached_current);
    assert!(package
        .step_inputs(
            &invalid,
            &running,
            &fresh,
            &proof,
            &message.map(|f| f.as_canonical_u64()),
            output(current, message)
        )
        .is_err());
    let mut detached_initial = z0;
    detached_initial[0] += F::ONE;
    let invalid = Stage1State::new(1, detached_initial, current);
    assert!(package
        .step_inputs(
            &invalid,
            &running,
            &fresh,
            &proof,
            &message.map(|f| f.as_canonical_u64()),
            output(current, message)
        )
        .is_err());
    let mut detached_running = running.clone();
    detached_running.claims[0].fold_digest[0] ^= 1;
    assert!(package
        .step_inputs(
            &state,
            &detached_running,
            &fresh,
            &proof,
            &message.map(|f| f.as_canonical_u64()),
            output(current, message)
        )
        .is_err());
    let mut detached_parent = running.clone();
    detached_parent
        .parent_authority
        .as_mut()
        .unwrap()
        .fold_digest[0] ^= 1;
    assert!(package
        .step_inputs(
            &state,
            &detached_parent,
            &fresh,
            &proof,
            &message.map(|f| f.as_canonical_u64()),
            output(current, message)
        )
        .is_err());
    let mut detached_fresh = fresh.clone();
    detached_fresh.x[1] += F::ONE;
    assert!(package
        .step_inputs(
            &state,
            &running,
            &detached_fresh,
            &proof,
            &message.map(|f| f.as_canonical_u64()),
            output(current, message)
        )
        .is_err());
    let mut detached_child = proof.clone();
    detached_child.pi_dec.children[0].c.data[0] += F::ONE;
    assert!(package
        .step_inputs(
            &state,
            &running,
            &fresh,
            &detached_child,
            &message.map(|f| f.as_canonical_u64()),
            output(current, message)
        )
        .is_err());
    let mut detached_point = proof.clone();
    detached_point.pi_dec.children[0].r[0] += K::ONE;
    assert!(package
        .step_inputs(
            &state,
            &running,
            &fresh,
            &detached_point,
            &message.map(|f| f.as_canonical_u64()),
            output(current, message)
        )
        .is_err());
}

#[test]
fn selected_plain_step_rejects_auxiliary_commitments() {
    let fixture = Fixture::load();
    let _ = fixture.packet(); // The unchanged plain input is accepted.
    for value in [F::ZERO, F::ONE] {
        let mut commitment = Commitment::zeros(D, fixture.fresh.c.kappa);
        commitment.data[0] = value;
        let auxiliary = LaneCommitments {
            ops: commitment.clone(),
            is: commitment.clone(),
            fs: commitment,
        };
        let reject = |running: &RunningInstance, fresh: &CcsClaim, location: &str| {
            let error = fixture
                .package
                .step_inputs(
                    &fixture.state,
                    running,
                    fresh,
                    &fixture.proof,
                    &fixture.message.map(|f| f.as_canonical_u64()),
                    output(fixture.state.current(), fixture.message),
                )
                .unwrap_err();
            assert!(
                matches!(&error, StepInputError::Input(message)
                    if *message == "selected plain claims cannot carry auxiliary commitments"),
                "{location}, auxiliary word {value:?}: {error}"
            );
        };

        let mut fresh = fixture.fresh.clone();
        fresh.adv = Some(auxiliary.clone());
        reject(&fixture.running, &fresh, "fresh claim");

        let mut running = fixture.running.claims_only();
        running.claims.last_mut().unwrap().adv = Some(auxiliary.clone());
        reject(&running, &fixture.fresh, "running claim");

        let mut running = fixture.running.claims_only();
        running.parent_authority.as_mut().unwrap().adv = Some(auxiliary);
        reject(&running, &fixture.fresh, "supplied parent claim");
    }
}
#[test]
fn saved_proof_and_transcript_match_lean() {
    let fixture = Fixture::load();
    let params = Params::for_ccs_shape(
        fixture.package.structure.n,
        fixture.package.structure.m,
        fixture.package.structure.t(),
        fixture.package.structure.max_degree(),
    )
    .unwrap();
    let mut transcript = folding::transcript::Transcript::session();
    let running = folding::verify(
        &mut transcript,
        &params,
        &fixture.package.structure,
        folding::ajtai_rlc_mixer,
        folding::ajtai_dec_mixer,
        std::slice::from_ref(&fixture.fresh),
        &fixture.running,
        &fixture.proof,
    )
    .unwrap();
    assert_eq!(running.claims, fixture.proof.pi_dec.children);
    let expected = read(artifact("nightstream-fprime-stage1-base-nifs-result-v1.json"));
    let state = transcript.snapshot().state().map(|f| f.as_canonical_u64());
    assert_eq!(json!(state), expected[7][9]);
    assert_eq!(json!(state), expected[9][14]);
    assert_eq!(transcript.snapshot().absorbed(), 0);
    assert_eq!(fixture.proof.canonical_bytes(), fixture.wire);
    let mut invalid = fixture.proof.clone();
    invalid.pi_dec.children[0].X[(0, 0)] = F::from_u64(2);
    let mut transcript = folding::transcript::Transcript::session();
    assert!(folding::verify(
        &mut transcript,
        &params,
        &fixture.package.structure,
        folding::ajtai_rlc_mixer,
        folding::ajtai_dec_mixer,
        std::slice::from_ref(&fixture.fresh),
        &fixture.running,
        &invalid
    )
    .is_err());
}

mod base;

mod key_prefix;
mod matrix_workspace;
mod recursive;

mod staged;
