mod test_helpers;

use neo_ajtai::{commit, commit_masked_ct, commit_precomp_ct, setup, verify_open};
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, SeedableRng};
use test_helpers::{compute_single_ajtai_row, rows_for_coords};

#[test]
fn commit_and_verify_basic_patterns() {
    let mut rng = StdRng::seed_from_u64(42);
    let d = neo_math::D;
    let kappa = 3;
    let m = 4;

    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");

    let z_zeros = vec![Fq::ZERO; d * m];
    let c_zeros = commit(&pp, &z_zeros);
    assert!(verify_open(&pp, &c_zeros, &z_zeros), "All-zeros should verify");

    let z_ones = vec![Fq::ONE; d * m];
    let c_ones = commit(&pp, &z_ones);
    assert!(verify_open(&pp, &c_ones, &z_ones), "All-ones should verify");

    let z_sparse: Vec<Fq> = (0..d * m)
        .map(|i| match i % 3 {
            0 => Fq::ZERO,
            1 => Fq::ONE,
            _ => Fq::ZERO - Fq::ONE,
        })
        .collect();
    let c_sparse = commit(&pp, &z_sparse);
    assert!(verify_open(&pp, &c_sparse, &z_sparse), "Sparse pattern should verify");

    let z_dense: Vec<Fq> = (0..d * m)
        .map(|i| Fq::from_u64((i * 17 + 42) as u64))
        .collect();
    let c_dense = commit(&pp, &z_dense);
    assert!(verify_open(&pp, &c_dense, &z_dense), "Dense pattern should verify");

    let mut z_mixed = z_sparse.clone();
    for i in (0..z_mixed.len()).step_by(10) {
        z_mixed[i] = Fq::from_u64(2);
    }
    let c_mixed = commit(&pp, &z_mixed);
    assert!(verify_open(&pp, &c_mixed, &z_mixed), "Mixed pattern should verify");

    println!("✅ Basic commit/verify functionality works for all test patterns");
}

#[test]
fn commit_deterministic() {
    let mut rng = StdRng::seed_from_u64(123);
    let d = neo_math::D;
    let kappa = 2;
    let m = 3;

    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");

    let z: Vec<Fq> = (0..d * m).map(|i| Fq::from_u64((i % 100) as u64)).collect();

    let c1 = commit(&pp, &z);
    let c2 = commit(&pp, &z);

    assert_eq!(c1, c2, "Commit should be deterministic");

    assert!(verify_open(&pp, &c1, &z), "First commit should verify");
    assert!(verify_open(&pp, &c2, &z), "Second commit should verify");

    println!("✅ Commit is deterministic");
}

#[test]
fn verify_rejects_wrong_opening() {
    let mut rng = StdRng::seed_from_u64(456);
    let d = neo_math::D;
    let kappa = 2;
    let m = 2;

    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");

    let z_correct: Vec<Fq> = (0..d * m).map(|i| Fq::from_u64(i as u64)).collect();
    let commitment = commit(&pp, &z_correct);

    assert!(
        verify_open(&pp, &commitment, &z_correct),
        "Correct opening should verify"
    );

    let mut z_wrong = z_correct.clone();
    z_wrong[0] = z_wrong[0] + Fq::ONE;

    assert!(
        !verify_open(&pp, &commitment, &z_wrong),
        "Wrong opening should be rejected"
    );

    println!("✅ Verification correctly rejects tampered openings");
}

#[test]
fn different_inputs_produce_different_commitments() {
    let mut rng = StdRng::seed_from_u64(789);
    let d = neo_math::D;
    let kappa = 2;
    let m = 2;

    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");

    let z1: Vec<Fq> = (0..d * m).map(|i| Fq::from_u64(i as u64)).collect();
    let mut z2 = z1.clone();
    z2[0] = z2[0] + Fq::ONE;

    let c1 = commit(&pp, &z1);
    let c2 = commit(&pp, &z2);

    assert_ne!(c1, c2, "Different inputs should produce different commitments");

    assert!(
        verify_open(&pp, &c1, &z1),
        "First commitment should verify with first opening"
    );
    assert!(
        verify_open(&pp, &c2, &z2),
        "Second commitment should verify with second opening"
    );

    assert!(
        !verify_open(&pp, &c1, &z2),
        "First commitment should not verify with second opening"
    );
    assert!(
        !verify_open(&pp, &c2, &z1),
        "Second commitment should not verify with first opening"
    );

    println!("✅ Different inputs produce different commitments (binding property)");
}

#[test]
fn try_commit_error_handling() {
    let mut rng = StdRng::seed_from_u64(999);
    let d = neo_math::D;
    let kappa = 2;
    let m = 3;

    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");

    let z_correct = vec![Fq::ZERO; d * m];
    assert!(
        neo_ajtai::try_commit(&pp, &z_correct).is_ok(),
        "Correct size should work"
    );

    let z_short = vec![Fq::ZERO; d * m - 1];
    let result_short = neo_ajtai::try_commit(&pp, &z_short);
    assert!(result_short.is_err(), "Too short should fail");

    let z_long = vec![Fq::ZERO; d * m + 1];
    let result_long = neo_ajtai::try_commit(&pp, &z_long);
    assert!(result_long.is_err(), "Too long should fail");

    println!("✅ try_commit properly handles dimension errors");
}

#[test]
fn constant_time_variants_basic_functionality() {
    let mut rng = StdRng::seed_from_u64(2023);
    let d = neo_math::D;
    let kappa = 2;
    let m = 2;

    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");

    let z_test: Vec<Fq> = (0..d * m)
        .map(|i| match i % 4 {
            0 => Fq::ZERO,
            1 => Fq::ONE,
            2 => Fq::ZERO - Fq::ONE,
            _ => Fq::from_u64(2),
        })
        .collect();

    let c_masked = commit_masked_ct(&pp, &z_test);
    assert!(verify_open(&pp, &c_masked, &z_test), "Masked CT commit should verify");

    let c_precomp = commit_precomp_ct(&pp, &z_test);
    assert!(verify_open(&pp, &c_precomp, &z_test), "Precomp CT commit should verify");

    let c_masked_2 = commit_masked_ct(&pp, &z_test);
    let c_precomp_2 = commit_precomp_ct(&pp, &z_test);
    assert_eq!(c_masked, c_masked_2, "Masked CT should be deterministic");
    assert_eq!(c_precomp, c_precomp_2, "Precomp CT should be deterministic");

    let mut z_different = z_test.clone();
    z_different[0] = z_different[0] + Fq::ONE;

    let c_masked_diff = commit_masked_ct(&pp, &z_different);
    let c_precomp_diff = commit_precomp_ct(&pp, &z_different);

    assert_ne!(
        c_masked, c_masked_diff,
        "Masked CT: different inputs should produce different outputs"
    );
    assert_ne!(
        c_precomp, c_precomp_diff,
        "Precomp CT: different inputs should produce different outputs"
    );

    assert!(
        verify_open(&pp, &c_masked_diff, &z_different),
        "Masked CT different input should verify"
    );
    assert!(
        verify_open(&pp, &c_precomp_diff, &z_different),
        "Precomp CT different input should verify"
    );

    println!("✅ Constant-time commit variants work correctly and maintain basic security properties");
}

#[test]
fn streaming_matches_batch_computation() {
    use rand::{rngs::StdRng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let d = neo_math::D;
    let kappa = 4;
    let m = 3;

    let pp = setup(&mut rng, d, kappa, m).expect("setup ok");
    let z_len = d * m;
    let num_coords = d * kappa;

    let batch_rows = rows_for_coords(&pp, z_len, num_coords).expect("batch rows ok");

    let mut streaming_rows = Vec::new();
    for coord_idx in 0..num_coords {
        let row = compute_single_ajtai_row(&pp, coord_idx, z_len, num_coords).expect("streaming row ok");
        streaming_rows.push(row);
    }

    assert_eq!(batch_rows.len(), streaming_rows.len(), "Row count mismatch");

    for (i, (batch_row, streaming_row)) in batch_rows.iter().zip(streaming_rows.iter()).enumerate() {
        assert_eq!(batch_row.len(), streaming_row.len(), "Row {} length mismatch", i);

        for (j, (&batch_val, &streaming_val)) in batch_row.iter().zip(streaming_row.iter()).enumerate() {
            assert_eq!(
                batch_val, streaming_val,
                "Mismatch at row {} position {}: batch={:?} vs streaming={:?}",
                i, j, batch_val, streaming_val
            );
        }
    }

    println!(
        "✅ Streaming computation matches batch computation for {} rows",
        num_coords
    );
}

#[test]
fn rows_for_coords_matches_commit() {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let d = neo_math::D;
    let kappa = 4;
    let m = 3;

    let pp = setup(&mut rng, d, kappa, m).expect("setup ok");

    let mut z = vec![Fq::ZERO; d * m];
    for x in &mut z {
        *x = Fq::from_u64(rng.random::<u64>());
    }

    let z_len = d * m;
    let num_coords = d * kappa;
    let rows = rows_for_coords(&pp, z_len, num_coords).expect("rows ok");

    let mut c_flat = vec![Fq::ZERO; num_coords];
    for (row_i, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), z_len);
        let mut acc = Fq::ZERO;
        for (a, &b) in row.iter().zip(&z) {
            acc += (*a) * b;
        }
        c_flat[row_i] = acc;
    }

    let c = commit_masked_ct(&pp, &z);
    let mut c_flat_expected = vec![Fq::ZERO; num_coords];
    for i in 0..kappa {
        for r in 0..d {
            c_flat_expected[i * d + r] = c.data[i * d + r];
        }
    }

    assert_eq!(c_flat, c_flat_expected, "L·z must equal vec(commit(pp, Z))");

    println!("✅ rows_for_coords produces correct linear algebra representation");
}
