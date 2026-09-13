//! Explicit capped terminal checks on the retained complete selected envelope.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_matrix;
use neo_ccs::Mat;
use neo_fold_clean::{
    paper::{
        construction2::RunningInstance,
        relations::{CcsClaim, CcsInstance, CcsWitness, CeClaim},
    },
    stage1::{
        encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage, Stage1Envelope,
        Stage1State, VerifyError,
    },
    Poseidon2HashChainV1Package,
};
use neo_math::{D, F, K};
use nightstream_fprime::PI_CCS_V1_1_STATE_PREIMAGE_WORDS;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};

fn artifact(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts")
        .join(name)
}

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn state_words(value: &Value) -> [F; 4] {
    serde_json::from_value::<[u64; 4]>(value.clone())
        .unwrap()
        .map(|word| {
            assert!(word < F::ORDER_U64, "canonical state word");
            F::from_u64(word)
        })
}

fn change_units(witness: &Mat<F>, updates: impl IntoIterator<Item = (usize, F)>) -> Mat<F> {
    let (positive, negative) = witness
        .packed_signed_unit_column_masks()
        .expect("retained compact fresh witness");
    let mut positive = positive.to_vec();
    let mut negative = negative.to_vec();
    for (coordinate, value) in updates {
        let mask = 1u64 << (coordinate % D);
        positive[coordinate / D] &= !mask;
        negative[coordinate / D] &= !mask;
        if value == F::ONE {
            positive[coordinate / D] |= mask;
        } else if value == -F::ONE {
            negative[coordinate / D] |= mask;
        } else {
            assert_eq!(value, F::ZERO);
        }
    }
    Mat::compact_signed_unit_from_column_masks(D, witness.cols(), &positive, &negative).unwrap()
}

/// Each case runs separately under the existing native cap. Mutation cases
/// recompute the fresh commitment, so commitment agreement cannot hide a
/// missing CE evaluation or fresh-row check.
pub fn check(case: &str, envelope_directory: &Path, child_directory: &Path, output: &Path) {
    assert!(matches!(
        case,
        "accepted" | "ce-evaluation" | "ce-matrix-evaluation" | "fresh-private"
    ));
    assert!(!output.exists(), "use a fresh terminal evidence file");
    let started = Instant::now();
    let package = Poseidon2HashChainV1Package::load(
        &fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap(),
    )
    .unwrap();
    let reference = read(&artifact(
        "nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json",
    ));
    let private = reference[2].as_array().unwrap();
    let start = PI_CCS_V1_1_STATE_PREIMAGE_WORDS;
    let expected = Stage1State::new(
        private[start + 28].as_u64().unwrap(),
        state_words(&json!(&private[start + 30..start + 34])),
        state_words(&reference[4][0]),
    );
    let record = read(&envelope_directory.join("envelope.json"));
    assert_eq!(record["schema"], 1);
    let state = Stage1State::new(
        record["iteration"].as_u64().unwrap(),
        state_words(&record["z0"]),
        state_words(&record["current"]),
    );
    assert_eq!(state, expected, "independent Lean endpoint");
    let claims: Vec<CeClaim> = serde_json::from_value(record["running_claims"].clone()).unwrap();
    let witnesses = (0..16)
        .map(|child| {
            serde_json::from_slice::<Mat<F>>(&fs::read(child_directory.join(format!("digit-{child}.json"))).unwrap())
                .unwrap()
        })
        .collect();
    let mut running = RunningInstance::new(claims, witnesses, None);
    let fresh_claim: CcsClaim =
        serde_json::from_slice(&fs::read(envelope_directory.join("fresh-claim.json")).unwrap()).unwrap();
    let fresh_witness: Mat<F> =
        serde_json::from_slice(&fs::read(envelope_directory.join("fresh-witness.json")).unwrap()).unwrap();
    let mut fresh = CcsInstance {
        claim: fresh_claim,
        witness: CcsWitness {
            w: Vec::new(),
            Z: fresh_witness,
        },
    };
    println!("terminal_inputs_loaded case={case} elapsed={:?}", started.elapsed());

    if case == "accepted" {
        let initial = Stage1Envelope::initial(expected.z0());
        package
            .verify(&Stage1State::new(0, expected.z0(), expected.z0()), &initial)
            .unwrap();
        assert!(matches!(
            package.verify(&expected, &initial),
            Err(VerifyError::Statement(_))
        ));
        for iteration in [0, F::ORDER_U64] {
            let invalid = Stage1State::new(iteration, expected.z0(), expected.current());
            let proof = Stage1Envelope::from_parts(invalid, running.clone(), fresh.clone());
            assert!(matches!(
                package.verify(&invalid, &proof),
                Err(VerifyError::Statement(_))
            ));
        }
        // The final canonical counter is allowed at the terminal boundary.
        // This different statement fails its hash link, not the counter check.
        let last = Stage1State::new(F::ORDER_U64 - 1, expected.z0(), expected.current());
        let last_proof = Stage1Envelope::from_parts(last, running.clone(), fresh.clone());
        assert!(matches!(
            package.verify(&last, &last_proof),
            Err(VerifyError::Fresh(
                "public input differs from the recomputed terminal state hash"
            ))
        ));
        drop(last_proof);
        // These values are not part of the formal terminal statement or Z.
        // A valid terminal proof must not depend on its discarded parent cache,
        // next-step transcript frame metadata, or redundant scalar-witness cache.
        for claim in &mut running.claims {
            claim.fold_digest = [0; 32];
        }
        fresh.witness.w = vec![F::from_u64(2)];
    } else if matches!(case, "ce-evaluation" | "ce-matrix-evaluation") {
        if case == "ce-evaluation" {
            running.claims[0].eval_k[0] += K::ONE;
        } else {
            running.claims[0].eval_a[0][0] += K::ONE;
        }
        let context = state_words(&reference[1]);
        let preimage = serialize_pi_ccs_v1_1_state_preimage(
            context,
            state.iteration(),
            state.z0(),
            state.current(),
            &running.claims,
            1,
        )
        .unwrap();
        fresh.claim.x = encode_pi_ccs_v1_1_public_input(pi_ccs_v1_1_state_hash(&preimage).unwrap())
            .unwrap()
            .into_iter()
            .map(F::from_u64)
            .collect();
        fresh.witness.Z = change_units(&fresh.witness.Z, fresh.claim.x.iter().copied().enumerate());
        fresh.claim.c = commit_production_signed_unit_matrix(&fresh.witness.Z).unwrap();
    } else {
        // The first private coordinate follows the package's full public prefix.
        let coordinate = fresh.claim.m_in;
        let old = fresh.witness.Z[(coordinate % D, coordinate / D)];
        let changed = if old == F::ZERO { -F::ONE } else { F::ZERO };
        fresh.witness.Z = change_units(&fresh.witness.Z, [(coordinate, changed)]);
        fresh.claim.c = commit_production_signed_unit_matrix(&fresh.witness.Z).unwrap();
    }
    println!("terminal_case_prepared case={case} elapsed={:?}", started.elapsed());
    let envelope = Stage1Envelope::from_parts(state, running, fresh);

    if case == "accepted" {
        let mut changed = expected.current();
        changed[0] += F::ONE;
        let wrong = Stage1State::new(expected.iteration(), expected.z0(), changed);
        assert!(matches!(
            package.verify(&wrong, &envelope),
            Err(VerifyError::Statement(_))
        ));
    }
    let result = package.verify(&expected, &envelope);
    match case {
        "accepted" => assert!(result.is_ok(), "actual terminal acceptance: {result:?}"),
        "ce-evaluation" => assert!(
            matches!(
                result,
                Err(VerifyError::Running {
                    index: 0,
                    reason: "Eval_K differs from the complete witness opening"
                })
            ),
            "changed CE evaluation: {result:?}"
        ),
        "ce-matrix-evaluation" => assert!(
            matches!(
                result,
                Err(VerifyError::Running {
                    index: 0,
                    reason: "Eval_A differs from the complete witness openings"
                })
            ),
            "changed CE matrix evaluation: {result:?}"
        ),
        "fresh-private" => assert!(
            matches!(result, Err(VerifyError::FreshRelation(_))),
            "recommitted private witness: {result:?}"
        ),
        _ => unreachable!(),
    }
    let elapsed = started.elapsed();
    fs::write(
        output,
        serde_json::to_vec(&json!({
            "schema": 1, "case": case, "passed": true,
            "iteration": expected.iteration(), "seconds": elapsed.as_secs_f64(),
            "result": format!("{result:?}"),
            "scope": "selected terminal predicate on actual retained witnesses; no new proof backend",
        }))
        .unwrap(),
    )
    .unwrap();
    println!("selected_terminal_case=passed case={case} elapsed={elapsed:?}");
}
