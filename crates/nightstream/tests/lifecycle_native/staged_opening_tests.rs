//! Production terminal rejection after a valid fresh relation and PiDEC recomposition.
use super::*;
use crate::lifecycle::VerifyError;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpeningRequest {
    phase: String,
    directory: PathBuf,
    step: Option<u64>,
}

#[derive(Clone, Copy)]
pub(crate) enum Opening {
    K,
    A,
}

fn request(phase: &str) -> (PathBuf, u64) {
    let request: OpeningRequest =
        serde_json::from_reader(std::io::stdin().lock()).expect("staged source directory on stdin");
    assert_eq!(request.phase, phase);
    let step = request.step.unwrap_or(3);
    assert!(matches!(step, 3 | 4), "selected terminal state is iteration 3 or 4");
    (request.directory, step)
}

pub(crate) fn labels(opening: Opening) -> (&'static str, &'static str) {
    match opening {
        Opening::K => ("opening-k", "Eval_K differs from the complete witness opening"),
        Opening::A => ("opening-a", "Eval_A differs from the complete witness openings"),
    }
}

pub(crate) fn change_openings(children: &mut [CeClaim], opening: Opening, radix: K) {
    // The two changes preserve sum_i b^i * eval_i.
    match opening {
        Opening::K => {
            children[0].eval_k[0] += radix;
            children[1].eval_k[0] -= K::ONE;
        }
        Opening::A => {
            children[0].eval_a[0][0] += radix;
            children[1].eval_a[0][0] -= K::ONE;
        }
    }
}

fn prepare_balanced_opening(opening: Opening) {
    let (phase, _) = labels(opening);
    let (root, step) = request(&format!("{phase}-prepare"));
    let package = prepare();
    let directory = fold_dir(&root, step - 1);
    let record: fold::SavedNifs = load(&directory.join("nifs.json"));
    let (state, fresh, prior, mut proof) = record.verify(&package, &step_dir(&root, step - 1), step - 1);

    change_openings(
        &mut proof.pi_dec.children,
        opening,
        K::from(F::from_u64(params(&package).b() as u64)),
    );
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
    assert_eq!(envelope.state(), &expected_state(step));
    save_envelope(&package, &envelope, &root.join(phase), Some(&directory));
}

fn reject_balanced_opening(opening: Opening) {
    let (phase, reason) = labels(opening);
    let (root, step) = request(phase);
    let package = prepare();
    let record: fold::SavedNifs = load(&fold_dir(&root, step - 1).join("nifs.json"));
    let (_, _, _, mut proof) = record.verify(&package, &step_dir(&root, step - 1), step - 1);
    change_openings(
        &mut proof.pi_dec.children,
        opening,
        K::from(F::from_u64(params(&package).b() as u64)),
    );
    let envelope = load_envelope(&package, &root.join(phase), step);
    for (actual, mut expected) in envelope
        .running()
        .unwrap()
        .claims
        .iter()
        .zip(proof.pi_dec.children)
    {
        // Loading recomputes the frame from the actual changed state preimage.
        expected.fold_digest = actual.fold_digest;
        assert_eq!(actual, &expected, "exact balanced child change");
    }
    let error = package
        .verify(&expected_state(step), &envelope)
        .expect_err("a balanced false opening must not be accepted");
    assert!(
        matches!(error, VerifyError::Running { index: 0, reason: actual } if actual == reason),
        "the production verifier must pass all earlier checks and reach {reason}; got {error:?}"
    );
    save(
        &root.join(format!("terminal-{step}-{phase}-rejected.json")),
        &json!({
            "schema": 1, "iteration": step, "case": phase,
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

#[test]
#[ignore = "Prepare the complete changed witness under its own 300-second cap before opening-k."]
fn prepares_balanced_eval_k_witness() {
    prepare_balanced_opening(Opening::K);
}

#[test]
#[ignore = "Prepare the complete changed witness under its own 300-second cap before opening-a."]
fn prepares_balanced_eval_a_witness() {
    prepare_balanced_opening(Opening::A);
}
