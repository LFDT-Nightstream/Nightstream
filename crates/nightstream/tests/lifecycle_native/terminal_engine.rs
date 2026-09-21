use super::*;
use crate::engine::{Engine, Prover};
use crate::folding::{CcsInstance, CcsWitness};
use crate::lifecycle::{Stage1Envelope, VerifyError};
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_reductions::superneo_eval::SuperneoCachedRelationError;
use std::time::Instant;

#[test]
#[ignore = "Production CPU-produced base proof, Metal terminal acceptance and recommitted-witness rejection; use the 300-second cap."]
fn metal_terminal_accepts_cpu_proof_and_matches_cpu_rejection() {
    let started = Instant::now();
    let application = crate::application::poseidon2_hash_chain_v1().unwrap();
    let bytes = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
    let (source, binding) = crate::assembly::prepare(&bytes, &application).unwrap();
    drop(bytes);
    let mut package = PreparedLifecycle::from_package(source, binding, Prover::Optimized).unwrap();
    let fixture = read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json"));
    let private: Vec<u64> = serde_json::from_value(fixture[2].clone()).unwrap();
    let initial: [F; 4] = private[30..34]
        .iter()
        .copied()
        .map(F::from_u64)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let message: [F; 4] = private[private.len() - 4..]
        .iter()
        .copied()
        .map(F::from_u64)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let current = output(initial, message);
    assert_eq!(current.to_vec(), fields(&fixture[4][0]));
    let expected = Stage1State::new(1, initial, current);
    let proof = package
        .extend_with_output(
            Stage1Envelope::initial(initial),
            &message.map(|word| word.as_canonical_u64()),
            current,
        )
        .unwrap();
    eprintln!("CPU base proof built elapsed={:?}", started.elapsed());
    package.prover = Prover::new(Engine::Metal).unwrap();
    package.verify(&expected, &proof).unwrap();
    let Prover::Metal(device) = &package.prover else {
        unreachable!()
    };
    let activity = device.lock().unwrap().activity();
    assert!(
        activity.dispatches > 0,
        "terminal verification did not execute Metal arithmetic"
    );
    eprintln!(
        "CPU proof accepted by Metal elapsed={:?} activity={activity:?}",
        started.elapsed()
    );

    // Keep the true public statement, replace the rest of the fresh witness,
    // and recompute its commitment. Digest and commitment consistency alone
    // must not make this false opening acceptable.
    let fresh = proof.fresh().unwrap();
    let blocks = package.structure.m.div_ceil(D);
    let mut positive = vec![0u64; blocks];
    let negative = vec![0u64; blocks];
    for (column, &value) in fresh.claim.x.iter().enumerate() {
        assert!(value == F::ZERO || value == F::ONE);
        if value == F::ONE {
            positive[column / D] |= 1u64 << (column % D);
        }
    }
    let witness = Mat::compact_signed_unit_from_column_masks(D, blocks, &positive, &negative).unwrap();
    drop((positive, negative));
    let mut claim = fresh.claim.clone();
    claim.c = commit_production_signed_unit_prefix_matrix(&witness).unwrap();
    let bad = Stage1Envelope::from_parts(
        expected.clone(),
        proof.running().unwrap().clone(),
        CcsInstance {
            claim,
            witness: CcsWitness { w: vec![], Z: witness },
        },
    );
    let row = |result| match result {
        Err(VerifyError::FreshRelation(SuperneoCachedRelationError::UnsatisfiedRow { row })) => row,
        other => panic!("expected a row failure after valid public and commitment checks, got {other:?}"),
    };
    let metal_row = row(package.verify(&expected, &bad));
    package.prover = Prover::Optimized;
    let cpu_row = row(package.verify(&expected, &bad));
    assert_eq!(metal_row, cpu_row);
    eprintln!(
        "Recommitted false witness rejected at row={cpu_row} elapsed={:?}",
        started.elapsed()
    );
}
