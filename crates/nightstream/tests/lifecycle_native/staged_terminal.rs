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
    let encoded = package
        .package
        .encode_stage1_v1_1_inputs(packet.pi_ccs(), packet.pi_dec(), packet.application_witness())
        .unwrap();
    if step == 1 {
        let expected = read(artifact(
            "nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json",
        ));
        let private: Vec<u64> = serde_json::from_value(expected[2].clone()).unwrap();
        let public: Vec<u64> = serde_json::from_value(expected[3].clone()).unwrap();
        assert_eq!(encoded.private_values(), private);
        assert_eq!(encoded.public_values(), public);
        assert_eq!(json!(packet.output_digest()), expected[4][1]);
        assert_eq!(json!(packet.next_public_input()), expected[4][2]);
    }
    save(
        &directory.join("caller-inputs.json"),
        &json!({
            "schema": 1,
            "verifier_context": package.binding.verifier_context().digest(),
            "private_values": encoded.private_values(),
            "public_values": encoded.public_values(),
            "output": output.map(|value| value.as_canonical_u64()),
            "output_digest": packet.output_digest(),
            "next_public_input": packet.next_public_input(),
        }),
    );
    // Use only the current native caller arrays. This is the same witness
    // executor used by complete_step; no Lean output enters either call.
    let started = Instant::now();
    let physical = package
        .package
        .execute_witness(encoded.private_values(), encoded.public_values())
        .unwrap();
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("physical.bin"))
        .unwrap();
    let mut writer = BufWriter::new(file);
    for word in physical
        .private_values()
        .iter()
        .copied()
        .chain(std::iter::once(1))
        .chain(physical.public_values().iter().copied())
    {
        writer.write_all(&word.to_le_bytes()).unwrap();
    }
    writer.flush().unwrap();
    drop((writer, physical, encoded));
    eprintln!(
        "native caller and complete physical export elapsed={:?}",
        started.elapsed()
    );
    let digits = (0..16)
        .map(|child| load(&directory.join(format!("digit-{child}.json"))))
        .collect();
    let envelope = package.complete_step(packet, digits, None).unwrap();
    assert_eq!(envelope.state(), &expected_state(step + 1));
    save_envelope(&package, &envelope, &step_dir(root, step + 1), Some(&directory));
}
pub(super) fn accept(root: &Path, step: u64, engine: EvaluationEngine) {
    assert_eq!(step, 3, "selected terminal state is iteration 3");
    let package = prepare_with_engine(engine);
    let envelope = load_envelope(&package, &step_dir(root, step), step);
    let expected = expected_state(step);
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
        .verify(&Stage1State::new(step, expected.z0(), wrong), &envelope)
        .is_err());
    save(
        &root.join(format!("terminal-{step}-accepted.json")),
        &json!({
            "schema":1, "package_identity":package.package_identity(), "iteration":step,
            "z0":expected.z0().map(|value| value.as_canonical_u64()),
            "current":expected.current().map(|value| value.as_canonical_u64()),
            "running_claims":&envelope.running().unwrap().claims,
            "fresh_claim":&envelope.fresh().unwrap().claim,
            "engine":format!("{engine:?}"),
            "scope":"new Rust terminal execution on its own generated successor; Lean comparison is a separate check"
        }),
    );
}
pub(super) fn mutation(root: &Path, step: u64) {
    assert_eq!(step, 3, "selected terminal state is iteration 3");
    let package = prepare();
    let original = step_dir(root, step);
    let envelope = load_envelope(&package, &original, step);
    let changed = super::super::recursive::rehash_false_running_opening(&package, envelope);
    save_envelope(
        &package,
        &changed,
        &root.join(format!("changed-step-{step}")),
        Some(&original),
    );
}
pub(super) fn reject(root: &Path, step: u64, engine: EvaluationEngine) {
    assert_eq!(step, 3, "selected terminal state is iteration 3");
    let package = prepare_with_engine(engine);
    let changed = load_envelope(&package, &root.join(format!("changed-step-{step}")), step);
    let error = package
        .verify(&expected_state(step), &changed)
        .expect_err("a rehashed false opening must be rejected");
    // Row evaluation checks the fresh relation before the running openings.
    // Changing its public digest without its private assignment breaks that relation.
    assert!(
        matches!(
            error,
            VerifyError::FreshRelation(
                neo_reductions::superneo_eval::SuperneoCachedRelationError::UnsatisfiedRow { .. }
            )
        ),
        "unexpected rejection: {error:?}"
    );
    save(
        &root.join(format!("terminal-{step}-rejected.json")),
        &json!({"schema":1,"package_identity":package.package_identity(),"iteration":step,"case":"fresh relation mismatch after public input rehash and recommit","rejected":true,"rejection":format!("{error:?}")}),
    );
}
