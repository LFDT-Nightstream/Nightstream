//! Successor construction and terminal checks consume only newly generated checkpoints.
use super::*;
use crate::lifecycle::VerifyError;

pub(super) fn successor(root: &Path, step: u64, engine: EvaluationEngine) {
    let directory = fold_dir(root, step);
    let package = prepare_with_engine(engine);
    let record: fold::SavedNifs = load(&directory.join("nifs.json"));
    let (state, fresh, prior, proof) = record.verify(&package, &step_dir(root, step), step);
    let message = message();
    let output = output(state.current(), message);
    let packet = package
        .step_inputs(
            &state,
            &prior,
            &fresh,
            &proof,
            &message.map(|value| value.as_canonical_u64()),
            output,
        )
        .unwrap();
    super::super::check_next_metadata(&packet, &proof);
    if step == 1 {
        let expected = read(artifact(
            "nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json",
        ));
        let encoded = package
            .package
            .encode_stage1_v1_1_inputs(packet.pi_ccs(), packet.pi_dec(), packet.application_witness())
            .unwrap();
        let private: Vec<u64> = serde_json::from_value(expected[2].clone()).unwrap();
        let public: Vec<u64> = serde_json::from_value(expected[3].clone()).unwrap();
        assert_eq!(encoded.private_values(), private);
        assert_eq!(encoded.public_values(), public);
        assert_eq!(json!(packet.output_digest()), expected[4][1]);
        assert_eq!(json!(packet.next_public_input()), expected[4][2]);
    }
    let digits = (0..16)
        .map(|child| load(&directory.join(format!("digit-{child}.json"))))
        .collect();
    let envelope = package.complete_step(packet, digits).unwrap();
    assert_eq!(envelope.state(), &expected_state(step + 1));
    save_envelope(&package, &envelope, &step_dir(root, step + 1), Some(&directory));
}
pub(super) fn accept(root: &Path, engine: EvaluationEngine) {
    let package = prepare_with_engine(engine);
    let envelope = load_envelope(&package, &step_dir(root, 3), 3);
    let expected = expected_state(3);
    package.verify(&expected, &envelope).unwrap();
    #[cfg(feature = "metal")]
    if let crate::engine::Backend::Metal(device) = &package.backend {
        let activity = device.lock().unwrap().activity();
        assert!(activity.dispatches > 0);
        eprintln!("terminal Metal activity={activity:?}");
    }
    let mut wrong = expected.current();
    wrong[0] += F::ONE;
    assert!(package
        .verify(&Stage1State::new(3, expected.z0(), wrong), &envelope)
        .is_err());
    save(
        &root.join("terminal-accepted.json"),
        &json!({
            "schema":1, "package_identity":package.package_identity(), "iteration":3,
            "z0":expected.z0().map(|value| value.as_canonical_u64()),
            "current":expected.current().map(|value| value.as_canonical_u64()),
            "running_claims":&envelope.running().unwrap().claims,
            "fresh_claim":&envelope.fresh().unwrap().claim,
            "engine":format!("{engine:?}"),
            "scope":"new Rust two-fold terminal execution; no later Lean comparison claimed"
        }),
    );
}
pub(super) fn mutation(root: &Path) {
    let package = prepare();
    let original = step_dir(root, 3);
    let envelope = load_envelope(&package, &original, 3);
    let changed = super::super::recursive::rehash_false_running_opening(&package, envelope);
    save_envelope(&package, &changed, &root.join("changed-step-3"), Some(&original));
}
pub(super) fn reject(root: &Path, engine: EvaluationEngine) {
    let package = prepare_with_engine(engine);
    let changed = load_envelope(&package, &root.join("changed-step-3"), 3);
    assert!(matches!(
        package.verify(&expected_state(3), &changed),
        Err(VerifyError::Running {
            index: 0,
            reason: "Eval_K differs from the complete witness opening"
        })
    ));
    save(
        &root.join("terminal-rejected.json"),
        &json!({"schema":1,"package_identity":package.package_identity(),"case":"rehashed and recommitted false running Eval_K","rejected":true}),
    );
}
