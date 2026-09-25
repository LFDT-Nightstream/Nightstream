//! Production terminal rejection after a valid fresh relation and PiDEC recomposition.
use super::*;
use crate::lifecycle::VerifyError;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpeningRequest {
    phase: String,
    directory: PathBuf,
}

#[derive(Clone, Copy)]
enum Opening {
    K,
    A,
}

fn reject_balanced_opening(opening: Opening) {
    let request: OpeningRequest =
        serde_json::from_reader(std::io::stdin().lock()).expect("staged source directory on stdin");
    let (phase, reason) = match opening {
        Opening::K => ("opening-k", "Eval_K differs from the complete witness opening"),
        Opening::A => ("opening-a", "Eval_A differs from the complete witness openings"),
    };
    assert_eq!(request.phase, phase);
    let root = request.directory;
    let package = prepare();
    let directory = fold_dir(&root, 2);
    let record: fold::SavedNifs = load(&directory.join("nifs.json"));
    let (state, fresh, prior, mut proof) = record.verify(&package, &step_dir(&root, 2), 2);

    // PiDEC checks sum_i b^i * eval_i. Adding b to child 0 and subtracting
    // one from child 1 preserves that sum, but falsifies both openings.
    let radix = K::from(F::from_u64(params(&package).b() as u64));
    match opening {
        Opening::K => {
            proof.pi_dec.children[0].eval_k[0] += radix;
            proof.pi_dec.children[1].eval_k[0] -= K::ONE;
        }
        Opening::A => {
            proof.pi_dec.children[0].eval_a[0][0] += radix;
            proof.pi_dec.children[1].eval_a[0][0] -= K::ONE;
        }
    }
    let packet = package
        .step_inputs(
            &state,
            &prior,
            &fresh,
            &proof,
            &message().map(|value| value.as_canonical_u64()),
            output(state.current(), message()),
        )
        .expect("the balanced change must pass the complete NIFS verifier");
    let children = (0..16)
        .map(|child| load(&directory.join(format!("digit-{child}.json"))))
        .collect();
    // Execute the full F' witness, including the changed child evaluations,
    // output-state hash and public input, and recompute the fresh commitment.
    let envelope = package
        .complete_step(packet, children, None)
        .expect("the balanced change must have a valid fresh witness");
    assert_eq!(envelope.state(), &expected_state(3));
    let error = package
        .verify(&expected_state(3), &envelope)
        .expect_err("a balanced false opening must not be accepted");
    assert!(
        matches!(error, VerifyError::Running { index: 0, reason: actual } if actual == reason),
        "the production verifier must pass all earlier checks and reach {reason}; got {error:?}"
    );
    save(
        &root.join(format!("terminal-3-{phase}-rejected.json")),
        &json!({
            "schema": 1, "iteration": 3, "case": phase,
            "verifier": "nightstream::PreparedLifecycle::verify", "engine": "Optimized",
            "package_identity": package.package_identity(), "changed_children": [0, 1],
            "weighted_recomposition_preserved": true, "fresh_witness_rebuilt": true,
            "earlier_terminal_checks_passed": true, "rejected": true,
            "rejection": format!("{error:?}")
        }),
    );
}

#[test]
#[ignore = "Full-profile production opening rejection; provide the staged source directory on stdin and use the 300-second cap."]
fn rejects_balanced_eval_k_after_valid_fresh_relation() {
    reject_balanced_opening(Opening::K);
}

#[test]
#[ignore = "Full-profile production opening rejection; provide the staged source directory on stdin and use the 300-second cap."]
fn rejects_balanced_eval_a_after_valid_fresh_relation() {
    reject_balanced_opening(Opening::A);
}
