#![cfg(feature = "testing")]

mod test_helpers;

use neo_ajtai::{commit_masked_ct, commit_precomp_ct, setup, verify_open};
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, SeedableRng};
use test_helpers::commit_spec;

#[test]
fn masked_ct_matches_spec_various_patterns() {
    let mut rng = StdRng::seed_from_u64(123);
    let d = neo_math::D;
    let kappa = 3;
    let m = 4;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    let mut z_mixed = vec![Fq::ZERO; d * m];
    for (i, z) in z_mixed.iter_mut().enumerate() {
        *z = match i % 5 {
            0 => Fq::ONE,
            1 => Fq::ZERO - Fq::ONE,
            2 => Fq::from_u64(2),
            3 => Fq::ZERO,
            _ => Fq::from_u64(7),
        };
    }

    let c_spec = commit_spec(&pp, &z_mixed);
    let c_masked = commit_masked_ct(&pp, &z_mixed);

    assert_eq!(c_spec, c_masked, "Masked CT commit should match spec for mixed pattern");
    assert!(verify_open(&pp, &c_masked, &z_mixed), "Masked CT commit should verify");

    let z_zeros = vec![Fq::ZERO; d * m];
    let c_spec_zeros = commit_spec(&pp, &z_zeros);
    let c_masked_zeros = commit_masked_ct(&pp, &z_zeros);

    assert_eq!(
        c_spec_zeros, c_masked_zeros,
        "Masked CT should match spec for all zeros"
    );
    assert!(verify_open(&pp, &c_masked_zeros, &z_zeros), "All zeros should verify");

    let z_sparse: Vec<Fq> = (0..d * m)
        .map(|i| match i % 3 {
            0 => Fq::ZERO,
            1 => Fq::ONE,
            _ => Fq::ZERO - Fq::ONE,
        })
        .collect();

    let c_spec_sparse = commit_spec(&pp, &z_sparse);
    let c_masked_sparse = commit_masked_ct(&pp, &z_sparse);

    assert_eq!(
        c_spec_sparse, c_masked_sparse,
        "Masked CT should match spec for sparse pattern"
    );
    assert!(
        verify_open(&pp, &c_masked_sparse, &z_sparse),
        "Sparse pattern should verify"
    );

    println!("✅ Masked constant-time commit matches spec across all patterns");
}

#[test]
fn precomp_ct_matches_spec_various_patterns() {
    let mut rng = StdRng::seed_from_u64(456);
    let d = neo_math::D;
    let kappa = 2;
    let m = 3;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    let z_dense: Vec<Fq> = (0..d * m)
        .map(|i| Fq::from_u64((i * 17 + 42) as u64))
        .collect();

    let c_spec = commit_spec(&pp, &z_dense);
    let c_precomp = commit_precomp_ct(&pp, &z_dense);

    assert_eq!(
        c_spec, c_precomp,
        "Precomp CT commit should match spec for dense pattern"
    );
    assert!(
        verify_open(&pp, &c_precomp, &z_dense),
        "Precomp CT commit should verify"
    );

    let z_ones = vec![Fq::ONE; d * m];
    let c_spec_ones = commit_spec(&pp, &z_ones);
    let c_precomp_ones = commit_precomp_ct(&pp, &z_ones);

    assert_eq!(c_spec_ones, c_precomp_ones, "Precomp CT should match spec for all ones");
    assert!(verify_open(&pp, &c_precomp_ones, &z_ones), "All ones should verify");

    let mut z_mixed = vec![Fq::ZERO; d * m];
    for i in 0..z_mixed.len() {
        z_mixed[i] = match i % 7 {
            0 => Fq::ZERO,
            1 => Fq::ONE,
            2 => Fq::ZERO - Fq::ONE,
            3 => Fq::from_u64(2),
            4 => Fq::from_u64(3),
            5 => Fq::from_u64(100),
            _ => Fq::from_u64(42),
        };
    }

    let c_spec_mixed = commit_spec(&pp, &z_mixed);
    let c_precomp_mixed = commit_precomp_ct(&pp, &z_mixed);

    assert_eq!(
        c_spec_mixed, c_precomp_mixed,
        "Precomp CT should match spec for mixed pattern"
    );
    assert!(
        verify_open(&pp, &c_precomp_mixed, &z_mixed),
        "Mixed pattern should verify"
    );

    println!("✅ Precomputed constant-time commit matches spec across all patterns");
}

#[test]
fn both_ct_variants_equivalent() {
    let mut rng = StdRng::seed_from_u64(789);
    let d = neo_math::D;
    let kappa = 3;
    let m = 2;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    let test_vectors = vec![
        (0..d * m)
            .map(|i| match i % 4 {
                0 => Fq::ZERO - Fq::ONE,
                1 => Fq::ZERO,
                2 => Fq::ONE,
                _ => Fq::from_u64(2),
            })
            .collect::<Vec<Fq>>(),
        (0..d * m)
            .map(|i| Fq::from_u64(((i * 13 + 7) % 100) as u64))
            .collect::<Vec<Fq>>(),
        {
            let mut v = vec![Fq::ZERO; d * m];
            for i in (0..v.len()).step_by(10) {
                v[i] = Fq::from_u64((i / 10) as u64 + 1);
            }
            v
        },
    ];

    for (pattern_num, z) in test_vectors.iter().enumerate() {
        let c_masked = commit_masked_ct(&pp, z);
        let c_precomp = commit_precomp_ct(&pp, z);
        let c_spec = commit_spec(&pp, z);

        assert_eq!(
            c_masked,
            c_precomp,
            "Pattern {}: Masked CT and Precomp CT should produce identical results",
            pattern_num + 1
        );
        assert_eq!(
            c_masked,
            c_spec,
            "Pattern {}: Masked CT should match specification",
            pattern_num + 1
        );
        assert_eq!(
            c_precomp,
            c_spec,
            "Pattern {}: Precomp CT should match specification",
            pattern_num + 1
        );

        assert!(
            verify_open(&pp, &c_masked, z),
            "Pattern {}: Masked CT should verify",
            pattern_num + 1
        );
        assert!(
            verify_open(&pp, &c_precomp, z),
            "Pattern {}: Precomp CT should verify",
            pattern_num + 1
        );
    }

    println!("✅ Both constant-time variants produce equivalent results for all test patterns");
}

#[test]
fn ct_variants_deterministic() {
    let mut rng = StdRng::seed_from_u64(999);
    let d = neo_math::D;
    let kappa = 2;
    let m = 2;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    let z: Vec<Fq> = (0..d * m)
        .map(|i| match i % 6 {
            0 => Fq::ZERO,
            1 => Fq::ONE,
            2 => Fq::ZERO - Fq::ONE,
            3 => Fq::from_u64(2),
            4 => Fq::from_u64(17),
            _ => Fq::from_u64(42),
        })
        .collect();

    let c_masked_1 = commit_masked_ct(&pp, &z);
    let c_masked_2 = commit_masked_ct(&pp, &z);
    assert_eq!(c_masked_1, c_masked_2, "Masked CT should be deterministic");

    let c_precomp_1 = commit_precomp_ct(&pp, &z);
    let c_precomp_2 = commit_precomp_ct(&pp, &z);
    assert_eq!(c_precomp_1, c_precomp_2, "Precomp CT should be deterministic");

    assert!(verify_open(&pp, &c_masked_1, &z), "Masked CT should verify");
    assert!(verify_open(&pp, &c_precomp_1, &z), "Precomp CT should verify");

    println!("✅ Both constant-time variants are deterministic");
}

#[test]
fn ct_variants_binding_property() {
    let mut rng = StdRng::seed_from_u64(1337);
    let d = neo_math::D;
    let kappa = 2;
    let m = 2;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    let z1: Vec<Fq> = (0..d * m).map(|i| Fq::from_u64(i as u64)).collect();
    let mut z2 = z1.clone();
    z2[0] = z2[0] + Fq::ONE;

    let c1_masked = commit_masked_ct(&pp, &z1);
    let c2_masked = commit_masked_ct(&pp, &z2);
    assert_ne!(
        c1_masked, c2_masked,
        "Masked CT: different inputs should produce different commitments"
    );

    let c1_precomp = commit_precomp_ct(&pp, &z1);
    let c2_precomp = commit_precomp_ct(&pp, &z2);
    assert_ne!(
        c1_precomp, c2_precomp,
        "Precomp CT: different inputs should produce different commitments"
    );

    assert!(verify_open(&pp, &c1_masked, &z1), "Masked CT c1 should verify with z1");
    assert!(verify_open(&pp, &c2_masked, &z2), "Masked CT c2 should verify with z2");
    assert!(
        verify_open(&pp, &c1_precomp, &z1),
        "Precomp CT c1 should verify with z1"
    );
    assert!(
        verify_open(&pp, &c2_precomp, &z2),
        "Precomp CT c2 should verify with z2"
    );

    assert!(
        !verify_open(&pp, &c1_masked, &z2),
        "Masked CT c1 should not verify with z2"
    );
    assert!(
        !verify_open(&pp, &c2_masked, &z1),
        "Masked CT c2 should not verify with z1"
    );
    assert!(
        !verify_open(&pp, &c1_precomp, &z2),
        "Precomp CT c1 should not verify with z2"
    );
    assert!(
        !verify_open(&pp, &c2_precomp, &z1),
        "Precomp CT c2 should not verify with z1"
    );

    println!("✅ Both constant-time variants satisfy the binding property");
}
