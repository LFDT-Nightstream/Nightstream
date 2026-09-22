//! Full selected base assignment and terminal execution. Run in its own capped invocation.
use super::*;
use crate::lifecycle::{ExtendError, Stage1Envelope};
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_units;
use std::time::Instant;
#[test]
#[ignore = "Full production assignment and terminal check; run separately under the 300-second cap."]
fn base_extension_matches_full_lean_assignment_and_terminal() {
    let started = Instant::now();
    let bytes = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
    let source = load_poseidon2_hash_chain_v1_package(&bytes).unwrap();
    let binding = source.production_verifier_binding().unwrap();
    let package = PreparedLifecycle::from_package(source.into(), binding, crate::engine::Backend::Optimized).unwrap();
    let loaded = load_poseidon2_hash_chain_v1_package(&bytes).unwrap();
    let reference: Value =
        serde_json::from_slice(&fs::read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json")).unwrap())
            .unwrap();
    assert_eq!(reference[0], 1);
    let private: Vec<u64> = serde_json::from_value(reference[2].clone()).unwrap();
    let public: Vec<u64> = serde_json::from_value(reference[3].clone()).unwrap();
    let z0 = private[30..34]
        .try_into()
        .map(|words: [u64; 4]| words.map(F::from_u64))
        .unwrap();
    let message = private[private.len() - 4..]
        .try_into()
        .map(|words: [u64; 4]| words.map(F::from_u64))
        .unwrap();
    let current = fields(&reference[4][0])
        .try_into()
        .map(|words: [F; 4]| words)
        .unwrap();
    let expected = Stage1State::new(1, z0, current);

    let initial = Stage1Envelope::initial(z0);
    package
        .verify(&Stage1State::new(0, z0, z0), &initial)
        .unwrap();
    let envelope = package
        .extend_with_output(initial, &message.map(|f| f.as_canonical_u64()), output(z0, message))
        .expect("public base extension");
    assert_eq!(envelope.state(), &expected);
    println!("public_base_constructed elapsed={:?}", started.elapsed());

    let running = envelope.running().unwrap();
    assert_eq!(running.claims.len(), 16);
    assert_eq!(running.witnesses.len(), 16);
    for (claim, witness) in running.claims.iter().zip(&running.witnesses) {
        assert!(claim.c.data.iter().all(|value| *value == F::ZERO));
        assert!((0..claim.X.cols()).all(|column| (0..D).all(|lane| claim.X[(lane, column)] == F::ZERO)));
        assert!(claim.r.iter().all(|value| *value == K::ZERO));
        assert!(claim.eval_k.iter().all(|value| *value == K::ZERO));
        assert!(claim.eval_a.iter().flatten().all(|value| *value == K::ZERO));
        assert_eq!(witness.virtual_constant_value(), Some(&F::ZERO));
    }

    let fresh = envelope.fresh().unwrap();
    assert!(fresh.witness.w.is_empty());
    assert_eq!(fresh.claim.x, fields(&reference[4][2]));
    let physical = loaded.execute_witness(&private, &public).unwrap();
    let logical = loaded.execute_logical_assignment(&physical).unwrap();
    drop(physical);
    assert_eq!(logical.len(), package.structure().m);
    for (column, &value) in logical.balanced_values().iter().enumerate() {
        let expected = match value {
            -1 => -F::ONE,
            0 => F::ZERO,
            1 => F::ONE,
            _ => panic!("nonunit reference"),
        };
        assert_eq!(
            fresh.witness.Z[(column % D, column / D)],
            expected,
            "complete coordinate {column}"
        );
    }
    neo_reductions::common::validate_fresh_witness_tail_zero(
        &fresh.witness.Z,
        package.structure().m,
        "public base extension",
    )
    .unwrap();
    let mut carrier = logical.balanced_values().to_vec();
    carrier.resize(fresh.witness.Z.rows() * fresh.witness.Z.cols(), 0);
    assert_eq!(commit_production_signed_units(&carrier).unwrap(), fresh.claim.c);
    drop((carrier, logical));
    println!("public_base_full_reference=passed elapsed={:?}", started.elapsed());

    for iteration in [0, F::ORDER_U64 - 1, F::ORDER_U64] {
        let invalid =
            Stage1Envelope::from_parts(Stage1State::new(iteration, z0, current), running.clone(), fresh.clone());
        assert!(matches!(
            package.extend_with_output(
                invalid,
                &message.map(|f| f.as_canonical_u64()),
                output(current, message)
            ),
            Err(ExtendError::Input(_)) | Err(ExtendError::StepInputs(_))
        ));
    }
    package
        .verify(&expected, &envelope)
        .expect("public base terminal acceptance");
    println!("public_base_lifecycle=passed elapsed={:?}", started.elapsed());
}
