#![cfg(feature = "testing")]

mod test_helpers;

use neo_ajtai::{commit, setup};
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, SeedableRng};
use test_helpers::commit_spec;

#[test]
fn dense_commit_handles_various_patterns() {
    let mut rng = StdRng::seed_from_u64(42);
    let d = neo_math::D;
    let kappa = 4;
    let m = 8;

    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");

    #[allow(non_snake_case)]
    let mut Z = vec![Fq::ZERO; d * m];

    for i in 0..Z.len() {
        match i % 4 {
            0 => Z[i] = Fq::ZERO,
            1 => Z[i] = Fq::ONE,
            2 => Z[i] = Fq::ZERO - Fq::ONE,
            _ => Z[i] = Fq::ZERO,
        }
    }

    let commitment = commit(&pp, &Z);

    assert!(
        neo_ajtai::verify_open(&pp, &commitment, &Z),
        "Commitment should verify correctly"
    );

    println!("✅ Constant-time dense commit handles various input patterns correctly");
}

#[test]
fn sparse_digit_classification_works() {
    let d = neo_math::D;
    let all_sparse = vec![Fq::ZERO; d]
        .into_iter()
        .enumerate()
        .map(|(i, _)| match i % 3 {
            0 => Fq::ZERO,
            1 => Fq::ONE,
            _ => Fq::ZERO - Fq::ONE,
        })
        .collect::<Vec<_>>();

    let all_dense = vec![Fq::from_u64(42); d]
        .into_iter()
        .enumerate()
        .map(|(i, _)| Fq::from_u64(42 + (i % 100) as u64))
        .collect::<Vec<_>>();

    let mut rng = StdRng::seed_from_u64(42);
    let d = neo_math::D;
    let kappa = 2;
    let m = 1;

    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");

    let commitment_sparse = commit(&pp, &all_sparse);
    let commitment_dense = commit(&pp, &all_dense);

    assert!(neo_ajtai::verify_open(&pp, &commitment_sparse, &all_sparse));
    assert!(neo_ajtai::verify_open(&pp, &commitment_dense, &all_dense));

    println!("✅ Both sparse and dense digit patterns produce valid commitments");
}

#[test]
fn constant_time_commit_correctness() {
    let mut rng = StdRng::seed_from_u64(42);
    let d = neo_math::D;
    let kappa = 2;
    let m = 4;

    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");

    let test_cases = vec![
        vec![Fq::ZERO; d * m],
        vec![Fq::ONE; d * m],
        vec![Fq::ZERO - Fq::ONE; d * m],
        (0..d * m)
            .map(|i| match i % 3 {
                0 => Fq::ZERO,
                1 => Fq::ONE,
                _ => Fq::ZERO - Fq::ONE,
            })
            .collect(),
        (0..d * m)
            .map(|i| if i % 2 == 0 { Fq::ZERO } else { Fq::ONE })
            .collect(),
    ];

    for (i, z) in test_cases.iter().enumerate() {
        let commitment = commit(&pp, z);
        assert!(
            neo_ajtai::verify_open(&pp, &commitment, z),
            "Test case {} should verify correctly",
            i
        );
    }

    println!("✅ Constant-time dense commit works correctly for various digit patterns");
}

#[test]
fn dense_commit_matches_spec_with_mixed_digits() {
    let mut rng = StdRng::seed_from_u64(7);
    let d = neo_math::D;
    let kappa = 4;
    let m = 8;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    #[allow(non_snake_case)]
    let mut Z = vec![Fq::ZERO; d * m];
    for i in 0..Z.len() {
        Z[i] = match i % 16 {
            0..=7 => Fq::ONE,
            8..=11 => Fq::ZERO,
            12..=13 => Fq::ZERO - Fq::ONE,
            _ => Fq::from_u64(2),
        };
    }

    let c_actual = neo_ajtai::commit(&pp, &Z);

    let c_spec = commit_spec(&pp, &Z);
    assert_eq!(c_actual, c_spec, "Commit with mixed digits must match specification");

    assert!(
        neo_ajtai::verify_open(&pp, &c_actual, &Z),
        "Mixed digit commit should verify correctly"
    );

    println!("✅ Commit handles mixed digits correctly (uses dense path due to 2's)");
}

#[test]
fn dense_commit_matches_spec_all_patterns() {
    let mut rng = StdRng::seed_from_u64(13);
    let d = neo_math::D;
    let kappa = 2;
    let m = 4;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    #[allow(non_snake_case)]
    let Z_strict: Vec<Fq> = (0..d * m)
        .map(|i| match i % 3 {
            0 => Fq::ZERO,
            1 => Fq::ONE,
            _ => Fq::ZERO - Fq::ONE,
        })
        .collect();

    #[allow(non_snake_case)]
    let mut Z_mixed = Z_strict.clone();
    Z_mixed[0] = Fq::from_u64(2);

    let c_strict = neo_ajtai::commit(&pp, &Z_strict);
    let c_mixed = neo_ajtai::commit(&pp, &Z_mixed);

    assert!(
        neo_ajtai::verify_open(&pp, &c_strict, &Z_strict),
        "Strict digits should verify"
    );
    assert!(
        neo_ajtai::verify_open(&pp, &c_mixed, &Z_mixed),
        "Mixed digits should verify"
    );
    assert_ne!(c_strict, c_mixed, "Different inputs should produce different outputs");

    let c_spec_strict = commit_spec(&pp, &Z_strict);
    let c_spec_mixed = commit_spec(&pp, &Z_mixed);
    assert_eq!(c_strict, c_spec_strict, "Strict {{-1,0,1}} digits must match spec");
    assert_eq!(c_mixed, c_spec_mixed, "Mixed digits must match spec");

    println!("✅ Constant-time dense commit matches spec for {{-1,0,1}} vs mixed digit patterns");
}

#[test]
fn dense_commit_differential_testing() {
    let mut rng = StdRng::seed_from_u64(42);
    let d = neo_math::D;
    let kappa = 3;
    let m = 6;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    #[allow(non_snake_case)]
    let Z_sparse: Vec<Fq> = (0..d * m)
        .map(|i| match i % 4 {
            0 => Fq::ONE,
            1 => Fq::ZERO,
            2 => Fq::ZERO - Fq::ONE,
            _ => Fq::ZERO,
        })
        .collect();

    #[allow(non_snake_case)]
    let Z_mixed: Vec<Fq> = Z_sparse
        .iter()
        .enumerate()
        .map(|(i, &x)| if i % 10 == 0 { Fq::from_u64(2) } else { x })
        .collect();

    let c_sparse = neo_ajtai::commit(&pp, &Z_sparse);
    let c_mixed = neo_ajtai::commit(&pp, &Z_mixed);
    let c_sparse_spec = commit_spec(&pp, &Z_sparse);
    let c_mixed_spec = commit_spec(&pp, &Z_mixed);

    assert_eq!(c_sparse, c_sparse_spec, "Sparse {{-1,0,1}} digits should match spec");
    assert_eq!(c_mixed, c_mixed_spec, "Mixed digits should match spec");
    assert_ne!(c_sparse, c_mixed, "Different inputs should produce different outputs");

    println!("✅ Differential testing verification: constant-time implementation matches spec");
    println!("   Sparse {{-1,0,1}} input: matches spec ✓");
    println!("   Mixed input with 2's: matches spec ✓");
}

#[test]
fn sparsity_invariant_testing() {
    let mut rng = StdRng::seed_from_u64(123);
    let d = neo_math::D;
    let kappa = 2;
    let m = 4;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    #[allow(non_snake_case)]
    let Z_very_sparse: Vec<Fq> = (0..d * m)
        .map(|i| if i % 20 == 0 { Fq::ONE } else { Fq::ZERO })
        .collect();

    #[allow(non_snake_case)]
    let Z_dense: Vec<Fq> = (0..d * m)
        .map(|i| if i % 2 == 0 { Fq::ONE } else { Fq::ZERO - Fq::ONE })
        .collect();

    let c_sparse = neo_ajtai::commit(&pp, &Z_very_sparse);
    let c_dense = neo_ajtai::commit(&pp, &Z_dense);
    let c_sparse_spec = commit_spec(&pp, &Z_very_sparse);
    let c_dense_spec = commit_spec(&pp, &Z_dense);

    assert_eq!(c_sparse, c_sparse_spec, "Very sparse input should match spec");
    assert_eq!(c_dense, c_dense_spec, "Dense input should match spec");

    println!("✅ Sparsity invariance test:");
    println!("   Very sparse (95% zeros): matches spec ✓");
    println!("   Dense (0% zeros): matches spec ✓");
    println!("   Constant-time implementation produces correct results regardless of sparsity");
}
