//! Same-input timing of the unchanged old lifecycle. Run separately from the new crate.

use std::{fs, path::PathBuf, time::Instant};

use neo_fold_legacy::{Poseidon2HashChainV1Package, Stage1Envelope, Stage1State};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[test]
#[ignore = "Performance comparison; run separately from the new lifecycle under the 300-second cap."]
fn unchanged_old_poseidon_base_lifecycle() {
    let started = Instant::now();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../nightstream/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
    let package = Poseidon2HashChainV1Package::load(&fs::read(path).unwrap()).unwrap();
    eprintln!("old preparation elapsed={:?}", started.elapsed());

    // Same saved Lean base-step inputs and expected output as circuit_lifecycle.rs.
    let initial = [202, 203, 204, 205].map(F::from_u64);
    let message = [7, 11, 13, 17].map(F::from_u64);
    let output = [
        10494771037735471283,
        2262554809432084833,
        4297319866765586853,
        18002659824166910177,
    ]
    .map(F::from_u64);
    let expected = Stage1State::new(1, initial, output);

    let proving = Instant::now();
    let proof = package
        .extend(Stage1Envelope::initial(initial), message)
        .unwrap();
    eprintln!("old base proving elapsed={:?}", proving.elapsed());
    assert_eq!(proof.state(), &expected);

    let verification = Instant::now();
    package.verify(&expected, &proof).unwrap();
    eprintln!("old terminal verification elapsed={:?}", verification.elapsed());
    let mut changed_output = output;
    changed_output[0] += F::ONE;
    assert!(package
        .verify(&Stage1State::new(1, initial, changed_output), &proof)
        .is_err());
    eprintln!("old public lifecycle elapsed={:?}", started.elapsed());
}
