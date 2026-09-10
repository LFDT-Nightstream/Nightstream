//! Replay the complete selected NIFS proof and prior-parent check.
//! The caller-selected package owns relation metadata. The previous checked
//! C/R result supplies the parent cache; current public input owns its frame digest.

use std::{fs, path::Path, time::Instant};

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::{
    construction2::RunningInstance,
    nifs::{self, NifsProof},
    params::Params,
    relations::{ajtai_dec_mixer, ajtai_rlc_mixer},
};
use neo_math::{F, K};
use serde_json::{json, Value};

use super::native_dec::{claim_value, running_value};
use super::native_driver::{canonical, commitment, extensions, fields, padded, public_matrix};

#[path = "../../../tests/nifs/nifs_actual_mutations.rs"]
mod mutations;

type Claim = CeClaim<Commitment, F, K>;

fn prior_parent(path: &Path, expected_identity: [u64; 4], frame_digest: [u8; 32]) -> Claim {
    let reference: Value =
        serde_json::from_slice(&fs::read(path).expect("checked prior C/R result")).expect("prior C/R schema");
    canonical(&reference);
    assert_eq!(reference.as_array().expect("prior C/R result").len(), 8);
    assert_eq!(reference[0], 1);
    assert_eq!(reference[5][0], 1);
    assert_eq!(reference[7][0], 1);
    assert_eq!(reference[6][6], json!(expected_identity), "same selected relation");
    assert_eq!(reference[7][5], reference[5][6], "prior R uses its actual C point");
    let result = &reference[7];
    assert_eq!(result.as_array().unwrap().len(), 11);
    assert_eq!(
        result[7]
            .as_array()
            .expect("prior matrix evaluations")
            .len(),
        14
    );
    Claim {
        c: commitment(&result[3]),
        X: public_matrix(&result[4]),
        r: extensions(&result[5]),
        eval_k: padded(&result[6]),
        eval_a: result[7].as_array().unwrap().iter().map(padded).collect(),
        m_in: 270,
        // This legacy field carries the incoming caller digest. It is not
        // an extra coordinate of the formal CE claim or the old C endpoint.
        fold_digest: frame_digest,
        adv: None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn check(
    params: &Params,
    structure: &CcsStructure<F>,
    fresh: &CcsClaim<Commitment, F>,
    running: Vec<Claim>,
    proof: &NifsProof,
    expected_identity: [u64; 4],
    prior_phase: &Path,
    reference: &Value,
    expected_state: [F; 8],
    output: &Path,
) {
    let started = Instant::now();
    assert_eq!(running.len(), 16);
    assert_eq!(proof.pi_ccs.outputs.len(), 17);
    assert_eq!(proof.pi_dec.children.len(), 16);
    let frame_digest = running[0].fold_digest;
    assert!(running
        .iter()
        .all(|claim| claim.fold_digest == frame_digest));
    let parent = prior_parent(prior_phase, expected_identity, frame_digest);
    let running = RunningInstance::new(running, Vec::new(), Some(parent));
    let mut transcript = Transcript::session();
    let verified = nifs::verify(
        &mut transcript,
        params,
        structure,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        std::slice::from_ref(fresh),
        &running,
        proof,
    )
    .expect("complete actual native NIFS verifier");
    assert!(
        verified.claims == proof.pi_dec.children,
        "exact final ordered NIFS claims"
    );
    assert!(
        verified.parent_authority.as_ref() == Some(&proof.pi_rlc.combined),
        "exact checked output parent cache"
    );
    assert!(
        verified.witnesses.is_empty(),
        "verifier output contains public claims only"
    );
    assert_eq!(transcript.snapshot().state(), expected_state);
    assert_eq!(transcript.snapshot().absorbed(), 0);
    let output_value = running_value(&verified.claims);
    assert!(
        reference[9][16] == json!([1, output_value]),
        "complete Lean NIFS running output"
    );
    let wire = proof.canonical_bytes();
    mutations::check_wire(&wire, reference);
    mutations::check(params, structure, fresh, &running, proof);
    let path = output.with_extension("nifs.json");
    let wire_path = output.with_extension("nifs.bin");
    assert!(!path.exists() && !wire_path.exists(), "fresh complete NIFS sinks");
    let record = json!([
        1,
        expected_identity,
        fields(&expected_state),
        claim_value(&proof.pi_rlc.combined, true),
        output_value
    ]);
    let mut bytes = serde_json::to_vec(&record).expect("complete NIFS output record");
    bytes.push(b'\n');
    fs::write(&path, bytes).expect("complete NIFS output sink");
    fs::write(&wire_path, &wire).expect("complete canonical proof sink");
    println!(
        "complete_native_nifs=passed sources=17 children=16 proof_bytes={} output={} elapsed={:?}",
        wire.len(),
        path.display(),
        started.elapsed()
    );
}
