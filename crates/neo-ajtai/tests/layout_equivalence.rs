#![allow(non_snake_case)]

use neo_ajtai::{commit, commit_row_major, decomp_b, decomp_b_row_major, setup_par, DecompStyle};
use neo_ccs::Mat;
use neo_math::D;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn decomp_b_row_major_is_transpose_of_decomp_b_b2_balanced() {
    let d = D;
    let m = 257usize;
    let b = 2u32;

    // Mix positive/negative and odd/even values to exercise b=2 fast paths.
    let z: Vec<Fq> = (0..m)
        .map(|i| {
            let v = (i as u64).wrapping_mul(17).wrapping_add(5);
            match i % 4 {
                0 => Fq::from_u64(v),
                1 => Fq::ZERO - Fq::from_u64(v),
                2 => Fq::from_u64(v.wrapping_add(1)),
                _ => Fq::ZERO - Fq::from_u64(v.wrapping_add(1)),
            }
        })
        .collect();

    let col_major = decomp_b(&z, b, d, DecompStyle::Balanced);
    let row_major = decomp_b_row_major(&z, b, d, DecompStyle::Balanced);
    assert_eq!(col_major.len(), d * m);
    assert_eq!(row_major.len(), d * m);

    for row in 0..d {
        for col in 0..m {
            assert_eq!(
                row_major[row * m + col],
                col_major[col * d + row],
                "(row,col)=({row},{col})"
            );
        }
    }
}

#[test]
fn decomp_b_row_major_is_transpose_of_decomp_b_b3_balanced() {
    let d = D;
    let m = 257usize;
    let b = 3u32;

    let z: Vec<Fq> = (0..m)
        .map(|i| {
            let v = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
            match i % 3 {
                0 => Fq::from_u64(v),
                1 => Fq::ZERO - Fq::from_u64(v),
                _ => Fq::from_u64(v.wrapping_add(2)),
            }
        })
        .collect();

    let col_major = decomp_b(&z, b, d, DecompStyle::Balanced);
    let row_major = decomp_b_row_major(&z, b, d, DecompStyle::Balanced);
    assert_eq!(col_major.len(), d * m);
    assert_eq!(row_major.len(), d * m);

    for row in 0..d {
        for col in 0..m {
            assert_eq!(
                row_major[row * m + col],
                col_major[col * d + row],
                "(row,col)=({row},{col})"
            );
        }
    }
}

#[test]
fn commit_row_major_matches_commit_over_column_major_buffer() {
    let d = D;
    let kappa = 3usize;

    for &m in &[1usize, 7usize, 300usize] {
        let mut rng = ChaCha8Rng::from_seed([7u8; 32]);
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
        let Z = Mat::from_row_major(d, m, data);

        let c_row_major = commit_row_major(&pp, &Z);

        // Convert to the legacy column-major layout (col*d + row).
        let mut col_major = vec![Fq::ZERO; d * m];
        for r in 0..d {
            for c in 0..m {
                col_major[c * d + r] = Z[(r, c)];
            }
        }
        let c_col_major = commit(&pp, &col_major);

        assert_eq!(c_row_major, c_col_major, "m={m}");
    }
}

