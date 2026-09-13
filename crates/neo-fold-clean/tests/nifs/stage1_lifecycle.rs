//! Compare the actual NIFS-to-successor handoff with the independent Lean caller packet.

use std::{
    fs,
    io::BufWriter,
    path::{Path, PathBuf},
    time::Instant,
};

use neo_ajtai::{nightstream_fprime_setup::commit_production_signed_units, Commitment};
use neo_ccs::Mat;
use neo_fold_clean::{
    paper::{
        construction2::{LaneCommitmentMode, RunningInstance},
        nifs::NifsProof,
        params::Params,
        pi_ccs, pi_dec, pi_rlc,
        relations::{CcsClaim, CeClaim},
    },
    stage1::{CompleteStepError, Stage1Envelope, Stage1State, Stage1StepInputs},
    Poseidon2HashChainV1Package,
};
use neo_math::{from_complex, D, F, K};
use nightstream_fprime::{load_poseidon2_hash_chain_v1_package, PI_CCS_V1_1_STATE_PREIMAGE_WORDS};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};

fn artifact(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts")
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

// The source is the saved actual execution. Byte equality below checks every
// native proof field, including its extra frame metadata and evaluation padding.
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
    package: Poseidon2HashChainV1Package,
    loaded: nightstream_fprime::LoadedPerApplicationPackage,
    base: Value,
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
        let package = Poseidon2HashChainV1Package::load(&bytes).unwrap();
        let loaded = load_poseidon2_hash_chain_v1_package(&bytes).unwrap();
        let base = read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json"));
        let expected = read(artifact(
            "nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json",
        ));
        let saved = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/nifs/fixtures/stage1_actual_nifs");
        let actual = read(saved.join("actual_result.json"));
        let proof = proof(&actual);
        let wire = fs::read(saved.join("proof.bin")).unwrap();
        assert!(proof.canonical_bytes() == wire, "exact retained native proof");
        let params = Params::for_ccs_shape(
            package.structure().n,
            package.structure().m,
            package.structure().t(),
            package.structure().max_degree(),
        )
        .unwrap();
        let mut running = RunningInstance::canonical_zero(&params, package.structure(), 270, LaneCommitmentMode::Plain)
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
            base,
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
            .step_inputs(&self.state, &self.running, &self.fresh, &self.proof, self.message)
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
        .step_inputs(&state, &running, &fresh, &proof, message)
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
            .step_inputs(&invalid, &running, &fresh, &proof, message)
            .is_err());
    }
    let mut detached_current = current;
    detached_current[0] += F::ONE;
    let invalid = Stage1State::new(1, z0, detached_current);
    assert!(package
        .step_inputs(&invalid, &running, &fresh, &proof, message)
        .is_err());
    let mut detached_initial = z0;
    detached_initial[0] += F::ONE;
    let invalid = Stage1State::new(1, detached_initial, current);
    assert!(package
        .step_inputs(&invalid, &running, &fresh, &proof, message)
        .is_err());
    let mut detached_running = running.clone();
    detached_running.claims[0].fold_digest[0] ^= 1;
    assert!(package
        .step_inputs(&state, &detached_running, &fresh, &proof, message)
        .is_err());
    let mut detached_parent = running.clone();
    detached_parent
        .parent_authority
        .as_mut()
        .unwrap()
        .fold_digest[0] ^= 1;
    assert!(package
        .step_inputs(&state, &detached_parent, &fresh, &proof, message)
        .is_err());
    let mut detached_fresh = fresh.clone();
    detached_fresh.x[1] += F::ONE;
    assert!(package
        .step_inputs(&state, &running, &detached_fresh, &proof, message)
        .is_err());
    let mut detached_child = proof.clone();
    detached_child.pi_dec.children[0].c.data[0] += F::ONE;
    assert!(package
        .step_inputs(&state, &running, &fresh, &detached_child, message)
        .is_err());
    let mut detached_point = proof.clone();
    detached_point.pi_dec.children[0].r[0] += K::ONE;
    assert!(package
        .step_inputs(&state, &running, &fresh, &detached_point, message)
        .is_err());
}

/// Capped fixture action. The large child matrices are supplied as explicit
/// paths; ordinary tests do not depend on a machine-local witness directory.
pub fn complete_envelope(digit_directory: &Path, output_directory: &Path) {
    let started = Instant::now();
    assert!(!output_directory.exists(), "use a fresh envelope output directory");
    let fixture = Fixture::load();
    let packet = fixture.packet();
    let expected_running = packet.next_running().clone();
    let expected_state = packet.next_state();
    let digits = (0..16)
        .map(|child| {
            let bytes = fs::read(digit_directory.join(format!("digit-{child}.json"))).unwrap();
            serde_json::from_slice::<Mat<F>>(&bytes).expect("validated compact actual child")
        })
        .collect::<Vec<_>>();
    println!("envelope_actual_inputs_elapsed={:?}", started.elapsed());

    // Reject the exact handoff defects before the valid completion. These
    // cases fail at the first child, without checking unchanged later children.
    let rejected = |witnesses| {
        fixture
            .package
            .complete_step(fixture.packet(), witnesses)
            .unwrap_err()
    };
    let mut missing = digits.clone();
    missing.pop();
    assert!(matches!(rejected(missing), CompleteStepError::Input(_)));
    let mut extra = digits.clone();
    extra.push(Mat::virtual_constant(D, digits[0].cols(), F::ZERO));
    assert!(matches!(rejected(extra), CompleteStepError::Input(_)));
    let mut wrong_shape = digits.clone();
    wrong_shape[0] = Mat::virtual_constant(D, 1, F::ZERO);
    assert!(matches!(
        rejected(wrong_shape),
        CompleteStepError::ChildWitness { index: 0, .. }
    ));
    let masks = digits[0]
        .packed_signed_unit_column_masks()
        .expect("actual active first child");
    let mut positive = masks.0.to_vec();
    let negative = masks.1.to_vec();
    // Flip coefficient zero while retaining a valid signed-unit encoding.
    if negative[0] & 1 == 0 {
        positive[0] ^= 1;
    } else {
        positive[0] |= 1;
    }
    let mut changed_negative = negative.clone();
    changed_negative[0] &= !1;
    let mut wrong_public = digits.clone();
    wrong_public[0] =
        Mat::compact_signed_unit_from_column_masks(D, digits[0].cols(), &positive, &changed_negative).unwrap();
    assert!(matches!(
        rejected(wrong_public),
        CompleteStepError::ChildWitness {
            index: 0,
            reason: "witness public projection differs from the verified child"
        }
    ));

    // A nonunit private value exercises the dense fallback without changing
    // the required public prefix. Zero allocation touches only these entries.
    let mut nonunit = Mat::zero(D, digits[0].cols(), F::ZERO);
    for row in 0..D {
        for column in 0..5 {
            nonunit[(row, column)] = digits[0][(row, column)];
        }
    }
    nonunit[(0, 5)] = F::from_u64(2);
    let mut bad_norm = digits.clone();
    bad_norm[0] = nonunit;
    assert!(matches!(
        rejected(bad_norm),
        CompleteStepError::ChildWitness {
            index: 0,
            reason: "witness does not have the fixed-key shape and strict signed-unit norm"
        }
    ));

    let mut positive = masks.0.to_vec();
    let mut negative = masks.1.to_vec();
    if negative[5] & 1 == 0 {
        positive[5] ^= 1;
    } else {
        positive[5] |= 1;
        negative[5] &= !1;
    }
    let mut detached_private = digits.clone();
    detached_private[0] =
        Mat::compact_signed_unit_from_column_masks(D, digits[0].cols(), &positive, &negative).unwrap();
    assert!(matches!(
        rejected(detached_private),
        CompleteStepError::ChildWitness {
            index: 0,
            reason: "fixed-key commitment differs from the verified child"
        }
    ));
    println!("envelope_handoff_rejections=passed elapsed={:?}", started.elapsed());

    // Compare the complete retained values with the actual supplied matrices.
    let expected_digits = digits.clone();
    let envelope = fixture
        .package
        .complete_step(packet, digits)
        .expect("complete actual successor envelope");
    assert!(!envelope.is_initial());
    assert_eq!(*envelope.state(), expected_state);
    let running = envelope.running().unwrap();
    assert_eq!(running.claims, expected_running.claims);
    assert_eq!(running.parent_authority, expected_running.parent_authority);
    assert_eq!(running.witnesses.len(), 16);
    for (actual, expected) in running.witnesses.iter().zip(&expected_digits) {
        assert_eq!((actual.rows(), actual.cols()), (expected.rows(), expected.cols()));
        if let (Some(a), Some(b)) = (
            actual.packed_signed_unit_column_masks(),
            expected.packed_signed_unit_column_masks(),
        ) {
            assert!(a == b, "all signed-unit coefficients are retained");
        } else if let (Some(a), Some(b)) = (actual.virtual_constant_value(), expected.virtual_constant_value()) {
            assert_eq!(a, b);
        } else {
            assert!(actual == expected, "every actual child witness is retained");
        }
    }
    drop(expected_digits);
    println!(
        "envelope_retained_children_and_fresh_commitment_elapsed={:?}",
        started.elapsed()
    );

    // Independent input is the Lean caller packet, not the native constructor.
    let private: Vec<u64> = serde_json::from_value(fixture.expected[2].clone()).unwrap();
    let public: Vec<u64> = serde_json::from_value(fixture.expected[3].clone()).unwrap();
    let physical = fixture.loaded.execute_witness(&private, &public).unwrap();
    let logical = fixture
        .loaded
        .execute_logical_assignment(&physical)
        .unwrap();
    drop(physical);
    let fresh = envelope.fresh().unwrap();
    assert!(fresh.witness.w.is_empty());
    assert!(fresh.claim.adv.is_none());
    assert_eq!(fresh.claim.m_in, 270);
    assert_eq!(fresh.claim.x, fields(&fixture.expected[4][2]));
    for (column, &value) in logical.balanced_values().iter().enumerate() {
        let expected = if value < 0 { -F::ONE } else { F::from_u64(value as u64) };
        assert_eq!(
            fresh.witness.Z[(column % D, column / D)],
            expected,
            "fresh carrier coordinate {column}"
        );
    }
    neo_reductions::common::validate_fresh_witness_tail_zero(
        &fresh.witness.Z,
        fixture.package.structure().m,
        "selected successor fresh witness",
    )
    .unwrap();
    let mut carrier = logical.balanced_values().to_vec();
    carrier.resize(fresh.witness.Z.rows() * fresh.witness.Z.cols(), 0);
    assert_eq!(
        commit_production_signed_units(&carrier).unwrap(),
        fresh.claim.c,
        "same complete reference assignment committed"
    );
    drop((logical, carrier));
    assert!(fixture.proof.canonical_bytes() == fixture.wire);

    let initial = Stage1Envelope::initial(fixture.state.z0());
    assert!(initial.is_initial());
    assert!(initial.running().is_none() && initial.fresh().is_none());
    assert_eq!(initial.state().iteration(), fixture.base[2][28].as_u64().unwrap());
    assert_eq!(
        initial.state().z0(),
        fields(&json!(&fixture.base[2].as_array().unwrap()[30..34])).as_slice()
    );
    assert_eq!(
        initial.state().current(),
        fields(&json!(&fixture.base[2].as_array().unwrap()[35..39])).as_slice()
    );

    fs::create_dir(output_directory).unwrap();
    serde_json::to_writer(
        BufWriter::new(fs::File::create(output_directory.join("fresh-witness.json")).unwrap()),
        &fresh.witness.Z,
    )
    .unwrap();
    serde_json::to_writer(
        BufWriter::new(fs::File::create(output_directory.join("fresh-claim.json")).unwrap()),
        &fresh.claim,
    )
    .unwrap();
    let record = json!({
        "schema": 1,
        "scope": "prover envelope; terminal CE evaluations and acceptance remain separate",
        "iteration": envelope.state().iteration(),
        "z0": envelope.state().z0().map(|value| value.as_canonical_u64()),
        "current": envelope.state().current().map(|value| value.as_canonical_u64()),
        "running_claims": running.claims,
        "running_parent": running.parent_authority,
        "child_witness_directory": digit_directory,
        "child_witness_count": running.witnesses.len(),
        "fresh_witness_file": "fresh-witness.json",
        "fresh_claim_file": "fresh-claim.json",
    });
    fs::write(
        output_directory.join("envelope.json"),
        serde_json::to_vec(&record).unwrap(),
    )
    .unwrap();
    println!("actual_successor_envelope=passed elapsed={:?}", started.elapsed());
}
