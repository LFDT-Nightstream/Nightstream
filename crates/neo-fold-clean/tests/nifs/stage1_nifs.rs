//! Actual selected C/R/D proving and strict comparison with a separate Lean result.

#[path = "nifs_actual_mutations.rs"]
mod mutations;
#[path = "stage1_actual.rs"]
mod stage1_actual;
#[path = "stage1_values.rs"]
pub(crate) mod stage1_values;

use neo_fold_clean::{
    engine::transcript::Transcript,
    paper::{
        nifs,
        relations::{ajtai_dec_mixer, ajtai_rlc_mixer},
    },
};
use neo_math::{D, F, K};
use neo_reductions::optimized_engine::optimized_verify_with_trace;
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::{PI_CCS_V1_1_SOURCE_COUNT, PI_DEC_V1_1_CHILD_COUNT};
use p3_field::PrimeCharacteristicRing;
use serde_json::{json, Value};
use std::{fs, path::Path, time::Instant};

use stage1_actual::ActualBase;
use stage1_values::{ccs_input, ccs_phase, claim_value, fields, running_value};

pub struct Observed {
    pub result: Value,
    pub wire: Vec<u8>,
}

/// Compute from actual source witnesses. There is no expected-result input.
pub fn prove(package_path: &Path, fixture_path: &Path) -> Observed {
    let started = Instant::now();
    let fixture_bytes = fs::read(fixture_path).expect("actual base fixture");
    let ActualBase {
        package,
        params,
        fresh,
        running,
    } = stage1_actual::load(package_path, &fixture_bytes);
    drop(fixture_bytes);
    let fresh_claim = fresh.claim.clone();
    let prior = running.claims_only();
    let (next, proof) = package
        .prove(vec![fresh], running)
        .expect("normal selected C/R/D prover");
    println!("actual_selected_nifs_prover_elapsed={:?}", started.elapsed());
    let mut transcript = Transcript::session();
    let verified = nifs::verify(
        &mut transcript,
        &params,
        package.structure(),
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        std::slice::from_ref(&fresh_claim),
        &prior,
        &proof,
    )
    .expect("normal selected NIFS verifier");
    assert_eq!(next.claims, verified.claims);
    assert_eq!(next.parent_authority, verified.parent_authority);
    assert_eq!(next.parent_authority.as_ref(), Some(&proof.pi_rlc.combined));
    assert_eq!(next.claims.len(), PI_DEC_V1_1_CHILD_COUNT);
    assert_eq!(next.witnesses.len(), PI_DEC_V1_1_CHILD_COUNT);
    assert_eq!(proof.pi_ccs.outputs.len(), PI_CCS_V1_1_SOURCE_COUNT);
    for witness in &next.witnesses {
        assert_eq!((witness.rows(), witness.cols()), (D, package.structure().m.div_ceil(D)));
        assert!(
            witness.packed_signed_unit_column_masks().is_some() || witness.virtual_constant_value() == Some(&F::ZERO)
        );
    }
    for claim in proof.pi_ccs.outputs.iter().chain(&next.claims) {
        assert!(claim.eval_a[13].iter().all(|&value| value == K::ZERO));
        assert!(claim.eval_k[D..].iter().all(|&value| value == K::ZERO));
        assert!(claim
            .eval_a
            .iter()
            .all(|family| family[D..].iter().all(|&value| value == K::ZERO)));
    }
    drop(next);

    let mut ccs_transcript = Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 8], 0);
    let (valid, trace) = optimized_verify_with_trace(
        &mut ccs_transcript,
        params.inner(),
        package.structure(),
        std::slice::from_ref(&fresh_claim),
        &prior.claims,
        &proof.pi_ccs.outputs,
        &proof.pi_ccs.sumcheck,
    )
    .unwrap();
    assert!(valid);
    assert!(proof
        .pi_ccs
        .outputs
        .iter()
        .all(|output| output.r == trace.round_challenges));
    // These existing rejection checks use the actual proof just produced.
    mutations::check(&params, package.structure(), &fresh_claim, &prior, &proof);
    let result = json!({
        "schema": 1,
        "structural_identifier": package.structural_identifier(),
        "package_identity": package.package_identity(),
        "pi_ccs_input": ccs_input(&fresh_claim, &prior.claims, &proof.pi_ccs),
        "pi_ccs_phase": ccs_phase(&proof.pi_ccs, &trace, valid),
        "pi_rlc_parent": claim_value(&proof.pi_rlc.combined, true),
        "children": running_value(&proof.pi_dec.children),
        "outgoing_state": fields(&transcript.snapshot().state()),
        "absorbed": transcript.snapshot().absorbed(),
    });
    println!("actual_selected_nifs_verified_elapsed={:?}", started.elapsed());
    Observed {
        result,
        wire: proof.canonical_bytes(),
    }
}

/// Compare saved actual outputs with the complete independent Lean check.
/// This never generates, substitutes, or repairs an expectation.
pub fn compare(actual: &Observed, expected: &Value) {
    assert_eq!(actual.result["schema"], 1);
    assert_eq!(expected.as_array().expect("full Lean C/R/D result").len(), 10);
    assert_eq!(expected[0], 1);
    assert_eq!(expected[5][0], 1);
    assert_eq!(expected[7][0], 1);
    assert_eq!(expected[9][0], 1);
    assert_eq!(expected[9][16][0], 1);
    assert_eq!(actual.result["package_identity"], expected[6][6]);
    assert_eq!(actual.result["pi_ccs_input"], expected[1]);
    assert_eq!(actual.result["pi_ccs_phase"], expected[5]);
    assert_eq!(
        actual.result["pi_rlc_parent"],
        json!([
            expected[7][3],
            expected[7][4],
            expected[7][5],
            expected[7][6],
            expected[7][7],
            1,
        ])
    );
    assert_eq!(actual.result["children"], expected[9][16][1]);
    assert_eq!(actual.result["outgoing_state"], expected[7][9]);
    assert_eq!(actual.result["outgoing_state"], expected[9][14]);
    assert_eq!(actual.result["absorbed"], 0);
    mutations::check_wire(&actual.wire, expected);
}

#[test]
#[ignore = "Full selected C/R/D call requires the independent Lean base result and includes all work within the 300-second cap."]
fn selected_nifs_prover_from_actual_base_sources() {
    // A missing result is an error before the expensive prover starts.
    let expected: Value = serde_json::from_slice(
        &fs::read(stage1_actual::artifact(
            "nightstream-fprime-stage1-base-nifs-result-v1.json",
        ))
        .expect("independent checked Lean base C/R/D result"),
    )
    .unwrap();
    let actual = prove(
        &stage1_actual::artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
        &stage1_actual::artifact("nightstream-fprime-stage1-base-step-fixture-v1.json"),
    );
    compare(&actual, &expected);
}
