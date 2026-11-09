//! Extension must not turn a self-consistent but unsatisfied predecessor into
//! an accepted recursive history.

use super::*;
use crate::engine::Backend;
use crate::lifecycle::{ProofState, Stage1Envelope, VerifyError};
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;

fn flip(matrix: &mut Mat<F>, position: usize) {
    let (row, column) = (position % D, position / D);
    let value = if matrix[(row, column)] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };
    matrix.set(row, column, value);
}

#[test]
#[ignore = "Production recursive proof; run separately under the 300-second cap."]
fn extension_rejects_a_recommitted_unsatisfied_predecessor() {
    let initial = [1, 2, 3, 4].map(F::from_u64);
    let message = [5, 6, 7, 8].map(F::from_u64);
    let words = message.map(|word| word.as_canonical_u64());
    let application = crate::application::poseidon2_hash_chain_v1().unwrap();
    let bytes = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
    let (source, binding) = crate::assembly::prepare(&bytes, &application).unwrap();
    let package = PreparedLifecycle::from_package(source.into(), binding, Backend::Optimized, 114).unwrap();

    let first = output(initial, message);
    let predecessor = package
        .extend_with_output(Stage1Envelope::initial(initial), &words, first, None)
        .unwrap();
    let honest = predecessor.snapshot();
    let (state, proof) = predecessor.into_parts();
    let ProofState::Active { running, mut fresh } = proof else {
        panic!("base proof is active");
    };
    flip(&mut fresh.witness.Z, fresh.claim.m_in);
    fresh.claim.c = commit_production_signed_unit_prefix_matrix(&fresh.witness.Z).unwrap();
    let forged = Stage1Envelope::from_parts(state, running, fresh);

    assert!(matches!(
        package.verify(&state, &forged),
        Err(VerifyError::FreshRelation(_))
    ));
    let second = output(first, message);
    let error = package
        .extend_with_output(forged, &words, second, None)
        .expect_err("extension must reject an unsatisfied predecessor");
    eprintln!("recommitted predecessor rejected: {error}");

    let retry = package
        .extend_with_output(honest, &words, second, None)
        .unwrap();
    package
        .verify(&Stage1State::new(2, initial, second), &retry)
        .unwrap();
}
