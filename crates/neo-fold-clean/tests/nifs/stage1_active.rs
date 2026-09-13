//! Public active-extension fixture checks on the original iteration-2 sources.
//! Complete outputs are compared with caller-selected, separately checked
//! iteration-3 material. Expected files never enter the production call.

use std::{
    fs::{self, File},
    io::BufReader,
    path::Path,
    time::Instant,
};

use neo_ccs::Mat;
use neo_fold_clean::{
    paper::{
        nifs, pi_dec,
        relations::{CcsClaim, CeClaim},
    },
    stage1::{
        encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage, ExtendError,
        Stage1Envelope, Stage1State,
    },
};
use neo_math::{F, K};
use nightstream_fprime::PI_DEC_V1_1_CHILD_COUNT;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::de::DeserializeOwned;

use super::stage1_actual::{self, ActualSources};

fn load<T: DeserializeOwned>(path: &Path) -> T {
    serde_json::from_reader(BufReader::new(File::open(path).expect("retained fixture file")))
        .expect("complete typed fixture data")
}

/// One complete public call. Cache variants keep all semantic source claims,
/// complete witnesses and the message fixed. Each case needs its own result.
pub fn check_public_active(
    case: &str,
    package_path: &Path,
    source_path: &Path,
    expected_envelope: &Path,
    expected_children: &Path,
) {
    assert!(matches!(case, "original" | "parent-absent" | "parent-changed"));
    let started = Instant::now();
    // Reject missing reference data before starting the expensive producer.
    let expected: serde_json::Value = load(&expected_envelope.join("envelope.json"));
    assert_eq!(expected["schema"], 1);
    assert_eq!(expected["iteration"], 3);
    assert_eq!(expected["child_witness_count"], PI_DEC_V1_1_CHILD_COUNT);
    let expected_claims: Vec<CeClaim> = serde_json::from_value(expected["running_claims"].clone()).unwrap();
    let expected_parent: Option<CeClaim> = serde_json::from_value(expected["running_parent"].clone()).unwrap();
    let expected_fresh: CcsClaim = load(&expected_envelope.join("fresh-claim.json"));
    assert!(expected_envelope.join("fresh-witness.json").is_file());
    assert_eq!(expected_claims.len(), PI_DEC_V1_1_CHILD_COUNT);
    for child in 0..PI_DEC_V1_1_CHILD_COUNT {
        assert!(expected_children
            .join(format!("digit-{child}.json"))
            .is_file());
    }

    let ActualSources {
        package,
        mut fresh,
        mut running,
        ..
    } = stage1_actual::load_path(package_path, source_path);
    let source_record: serde_json::Value = load(&source_path.join("envelope.json"));
    let (iteration, z0, current, message): (u64, [u64; 4], [u64; 4], [u64; 4]) =
        load(&source_path.join("next-message-input.json"));
    assert_eq!(iteration, 2);
    assert_eq!(source_record["iteration"], iteration);
    assert_eq!(source_record["z0"], serde_json::json!(z0));
    assert_eq!(source_record["current"], serde_json::json!(current));
    let fields = |words: [u64; 4]| {
        words.map(|word| {
            assert!(word < F::ORDER_U64, "canonical state/advice word");
            F::from_u64(word)
        })
    };
    let state = Stage1State::new(iteration, fields(z0), fields(current));
    let message = fields(message);
    let expected_state = Stage1State::new(
        iteration + 1,
        fields(serde_json::from_value(expected["z0"].clone()).unwrap()),
        fields(serde_json::from_value(expected["current"].clone()).unwrap()),
    );

    if case == "parent-absent" {
        running.parent_authority = None;
    } else if case == "parent-changed" {
        let parent = running
            .parent_authority
            .as_mut()
            .expect("original staged parent");
        parent.c.data[0] += F::ONE;
        parent.X[(0, 0)] += F::ONE;
        parent.r[0] += K::ONE;
        parent.eval_k[0] += K::ONE;
        parent.eval_a[0][0] += K::ONE;
        parent.fold_digest[0] ^= 1;
    }
    if case != "original" {
        for claim in &mut running.claims {
            claim.fold_digest[0] ^= 1;
        }
        fresh.witness.w = vec![F::from_u64(2)];
    }
    // Semantic claims, complete source Z values and the application message
    // stay identical. Only non-authoritative caches change in the two cases.
    let envelope = Stage1Envelope::from_parts(state, running, fresh);
    println!(
        "public_active_inputs_loaded case={case} elapsed={:?}",
        started.elapsed()
    );
    let returned = package
        .extend(envelope, message)
        .expect("complete public active extension");
    println!(
        "public_active_extend_returned case={case} elapsed={:?}",
        started.elapsed()
    );

    assert!(!returned.is_initial());
    assert_eq!(*returned.state(), expected_state);
    let actual_running = returned.running().unwrap();
    assert_eq!(
        actual_running.claims, expected_claims,
        "every ordered child claim field"
    );
    assert_eq!(
        actual_running.parent_authority, expected_parent,
        "complete returned parent cache"
    );
    assert_eq!(actual_running.witnesses.len(), PI_DEC_V1_1_CHILD_COUNT);
    let compare_matrix = |actual: &Mat<F>, expected: &Mat<F>| {
        assert_eq!((actual.rows(), actual.cols()), (expected.rows(), expected.cols()));
        if let (Some(a), Some(b)) = (
            actual.packed_signed_unit_column_masks(),
            expected.packed_signed_unit_column_masks(),
        ) {
            assert!(a == b, "every signed-unit coefficient, including the full tail");
        } else if let (Some(a), Some(b)) = (actual.virtual_constant_value(), expected.virtual_constant_value()) {
            assert_eq!(a, b);
        } else {
            assert!(actual == expected, "every complete matrix coordinate");
        }
    };
    for (child, actual) in actual_running.witnesses.iter().enumerate() {
        let expected: Mat<F> = load(&expected_children.join(format!("digit-{child}.json")));
        compare_matrix(actual, &expected);
    }
    let actual_fresh = returned.fresh().unwrap();
    assert_eq!(actual_fresh.claim.c, expected_fresh.c, "fresh commitment");
    assert_eq!(actual_fresh.claim.x, expected_fresh.x, "every fresh public word");
    assert_eq!(actual_fresh.claim.m_in, expected_fresh.m_in, "fresh public width");
    assert_eq!(actual_fresh.claim.adv, expected_fresh.adv, "fresh advice metadata");
    assert!(actual_fresh.witness.w.is_empty());
    let expected_fresh_witness: Mat<F> = load(&expected_envelope.join("fresh-witness.json"));
    compare_matrix(&actual_fresh.witness.Z, &expected_fresh_witness);
    neo_reductions::common::validate_fresh_witness_tail_zero(
        &actual_fresh.witness.Z,
        package.structure().m,
        "public active output",
    )
    .unwrap();
    println!(
        "public_active_full_output_comparison=passed case={case} prior_iteration=2 output_iteration=3 elapsed={:?}",
        started.elapsed()
    );
}

/// Reject a rehashed nonunit child at the public family check before proving.
pub fn reject_active_child_public(package_path: &Path, source_path: &Path) {
    let started = Instant::now();
    let ActualSources {
        package,
        mut fresh,
        mut running,
        ..
    } = stage1_actual::load_path(package_path, source_path);
    let record: serde_json::Value = load(&source_path.join("envelope.json"));
    let (iteration, z0, current, message): (u64, [u64; 4], [u64; 4], [u64; 4]) =
        load(&source_path.join("next-message-input.json"));
    assert_eq!(iteration, 2, "actual iteration-2 rejection case");
    assert_eq!(record["iteration"], iteration);
    assert_eq!(record["z0"], serde_json::json!(z0));
    assert_eq!(record["current"], serde_json::json!(current));
    let fields = |words: [u64; 4]| {
        words.map(|word| {
            assert!(word < F::ORDER_U64, "canonical state/advice word");
            F::from_u64(word)
        })
    };
    let state = Stage1State::new(iteration, fields(z0), fields(current));
    let message = fields(message);
    let loaded = nightstream_fprime::load_poseidon2_hash_chain_v1_package(&fs::read(package_path).unwrap()).unwrap();
    let context = loaded
        .production_verifier_binding()
        .unwrap()
        .verifier_context()
        .digest();
    drop(loaded);

    let changed = F::from_u64(2);
    assert_ne!(running.claims[0].X[(0, 0)], changed);
    running.claims[0].X[(0, 0)] = changed;
    let preimage = serialize_pi_ccs_v1_1_state_preimage(
        context.map(F::from_u64),
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

    // The rewritten hash passes checked_prior_state. extend must rebuild
    // the parent and reject the nonunit child before any source proving.
    let envelope = Stage1Envelope::from_parts(state, running, fresh);
    let error = package
        .extend(envelope, message)
        .expect_err("nonunit child public input");
    assert!(
        matches!(
            &error,
            ExtendError::PriorFamily(nifs::Error::PiDec(pi_dec::Error::ChildXLowNorm))
        ),
        "expected the child-family rejection before proving, got {error:?}"
    );
    println!(
        "public_active_child_public_rejection=passed prior_iteration=2 elapsed={:?}",
        started.elapsed()
    );
}
