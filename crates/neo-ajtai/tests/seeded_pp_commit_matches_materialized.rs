use neo_ajtai::{
    commit_row_major, commit_row_major_seeded, commit_row_major_seeded_binary_cols, commit_row_major_seeded_many,
    set_global_pp_seeded, setup_par, AjtaiSModule,
};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_math::D;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn seeded_pp_commit_matches_materialized_pp() {
    let seed = [7u8; 32];
    let d = D;
    let kappa = 2;

    for &m in &[10usize, 300usize] {
        let mut rng = ChaCha8Rng::from_seed(seed);
        let pp = setup_par(&mut rng, d, kappa, m).expect("setup_par");

        let mut data = Vec::with_capacity(d * m);
        for r in 0..d {
            for c in 0..m {
                let x = (r as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                    ^ 0x94D0_49BB_1331_11EB;
                data.push(Fq::from_u64(x));
            }
        }
        let z = Mat::from_row_major(d, m, data);

        let c_materialized = commit_row_major(&pp, &z);
        let c_seeded = commit_row_major_seeded(seed, d, kappa, m, &z);
        assert_eq!(c_materialized, c_seeded, "m={}", m);
    }
}

#[test]
fn seeded_pp_commit_many_single_sparse_matches_single_and_materialized_pp() {
    let seed = [31u8; 32];
    let d = D;
    let kappa = 3;
    let m = 300usize;
    let mut rng = ChaCha8Rng::from_seed(seed);
    let pp = setup_par(&mut rng, d, kappa, m).expect("setup_par");

    let mut data = vec![Fq::ZERO; d * m];
    for c in [0usize, 7, 63, 128, 299] {
        for r in [0usize, 1, 17, 53] {
            data[r * m + c] = Fq::from_u64((r as u64 + 3) * (c as u64 + 5));
        }
    }
    let z = Mat::from_row_major(d, m, data);

    let c_materialized = commit_row_major(&pp, &z);
    let c_seeded = commit_row_major_seeded(seed, d, kappa, m, &z);
    let c_many = commit_row_major_seeded_many(seed, d, kappa, m, &[&z]);

    assert_eq!(c_seeded, c_materialized);
    assert_eq!(c_many, vec![c_materialized]);
}

#[test]
fn seeded_pp_binary_cols_commit_matches_seeded_mat_commit() {
    let seed = [9u8; 32];
    let d = D;
    let kappa = 3;
    let m = 257usize;
    let mut column_bits = vec![0u64; m];
    for c in 0..m {
        let mut mask = 0u64;
        for r in 0..d {
            let bit = (((c as u64).wrapping_mul(0x9E37_79B9) ^ (r as u64).wrapping_mul(0xBF58_476D)) & 1) as u64;
            if bit == 1 {
                mask |= 1u64 << r;
            }
        }
        column_bits[c] = mask;
    }
    let mut data = Vec::with_capacity(d * m);
    for r in 0..d {
        for c in 0..m {
            data.push(if (column_bits[c] >> r) & 1 == 1 {
                Fq::ONE
            } else {
                Fq::ZERO
            });
        }
    }
    let z = Mat::from_row_major(d, m, data);

    let c_seeded = commit_row_major_seeded(seed, d, kappa, m, &z);
    let c_binary = commit_row_major_seeded_binary_cols(seed, d, kappa, m, &column_bits);
    assert_eq!(c_seeded, c_binary);
}

#[test]
fn seeded_s_module_commit_uses_same_map_for_binary_and_nonbinary_inputs() {
    let seed = [13u8; 32];
    let d = D;
    let kappa = 4;
    let m = 263usize;
    set_global_pp_seeded(d, kappa, m, seed).expect("register seeded PP");
    let log = AjtaiSModule::from_global_for_dims(d, m).expect("seeded global module");

    let binary_data: Vec<Fq> = (0..d)
        .flat_map(|r| {
            (0..m).map(move |c| {
                if ((r * 17 + c * 31) & 1) == 1 {
                    Fq::ONE
                } else {
                    Fq::ZERO
                }
            })
        })
        .collect();
    let binary = Mat::from_row_major(d, m, binary_data);
    assert_eq!(log.commit(&binary), commit_row_major_seeded(seed, d, kappa, m, &binary));

    let nonbinary_data: Vec<Fq> = (0..d)
        .flat_map(|r| (0..m).map(move |c| Fq::from_u64((r as u64 + 3) * (c as u64 + 5))))
        .collect();
    let nonbinary = Mat::from_row_major(d, m, nonbinary_data);
    assert_eq!(
        log.commit(&nonbinary),
        commit_row_major_seeded(seed, d, kappa, m, &nonbinary)
    );
}

#[test]
fn seeded_s_module_commit_uses_same_map_for_signed_unit_inputs() {
    let seed = [23u8; 32];
    let d = D;
    let kappa = 4;
    let m = 271usize;
    set_global_pp_seeded(d, kappa, m, seed).expect("register seeded PP");
    let log = AjtaiSModule::from_global_for_dims(d, m).expect("seeded global module");

    let neg_one = Fq::ZERO - Fq::ONE;
    let data: Vec<Fq> = (0..d)
        .flat_map(|r| {
            (0..m).map(move |c| match (r * 19 + c * 37) % 5 {
                0 => Fq::ONE,
                1 => neg_one,
                _ => Fq::ZERO,
            })
        })
        .collect();
    let z = Mat::from_row_major(d, m, data);

    assert_eq!(log.commit(&z), commit_row_major_seeded(seed, d, kappa, m, &z));
}

#[test]
fn seeded_s_module_commit_many_uses_same_map_for_signed_unit_inputs() {
    let seed = [29u8; 32];
    let d = D;
    let kappa = 3;
    let m = 257usize;
    set_global_pp_seeded(d, kappa, m, seed).expect("register seeded PP");
    let log = AjtaiSModule::from_global_for_dims(d, m).expect("seeded global module");
    let neg_one = Fq::ZERO - Fq::ONE;

    let mats: Vec<Mat<Fq>> = (0..4)
        .map(|z_idx| {
            let data: Vec<Fq> = (0..d)
                .flat_map(|r| {
                    (0..m).map(move |c| match (z_idx + r * 11 + c * 17) % 7 {
                        0 => Fq::ONE,
                        1 => neg_one,
                        _ => Fq::ZERO,
                    })
                })
                .collect();
            Mat::from_row_major(d, m, data)
        })
        .collect();
    let refs: Vec<&Mat<Fq>> = mats.iter().collect();

    let committed = log.commit_many(&refs);
    let expected: Vec<_> = refs
        .iter()
        .map(|z| commit_row_major_seeded(seed, d, kappa, m, z))
        .collect();
    assert_eq!(committed, expected);
}
