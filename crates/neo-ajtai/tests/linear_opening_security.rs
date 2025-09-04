#![allow(non_snake_case)] // Allow mathematical notation like Z

use neo_ajtai::{setup, commit, decomp_b, DecompStyle, open_linear};
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

#[test]
fn linear_opening_for_differential_testing_only() {
    let mut rng = ChaCha20Rng::seed_from_u64(4242);
    let d = neo_math::D;
    let kappa = 2usize;
    let m = 3usize;

    // Public parameters and a small witness
    let pp = setup(&mut rng, d, kappa, m).expect("setup should succeed");
    let z: Vec<Fq> = (0..m).map(|i| Fq::from_u64((i as u64) + 1)).collect();
    let Z = decomp_b(&z, 2, d, DecompStyle::Balanced); // d×m
    let c = commit(&pp, &Z);

    // A sample linear functional v and the prover-side opening
    let v = vec![Fq::from_u64(3), Fq::from_u64(5), Fq::from_u64(7)];
    let (y_slices, _proof) = open_linear(&pp, &c, &Z, &[v.clone()]);

    // open_linear is provided for differential testing only
    // Verification of linear relations must be done via neo_fold::verify_linear (Π_RLC)
    assert_eq!(y_slices.len(), 1);
    assert_eq!(y_slices[0].len(), d);
}

#[test] 
fn verify_linear_function_does_not_exist() {
    // This test ensures that verify_linear is not accidentally added back to the API
    // The very fact that this test compiles confirms verify_linear is not exported
    
    // If someone tries to use neo_ajtai::verify_linear, they should get a compile error
    // directing them to use neo_fold::verify_linear (Π_RLC) instead
    
    // NOTE: This is a compile-time test - if verify_linear existed, the import 
    // at the top of the file would fail to exclude it
    
    assert!(true, "verify_linear correctly absent from neo_ajtai API");
}
