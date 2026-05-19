use neo_ajtai::{commit_row_major, commit_row_major_seeded, commit_row_major_seeded_binary_cols, setup_par};
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
