#![allow(non_snake_case)]

use neo_ajtai::{commit, commit_row_major, setup};
use neo_ccs::Mat;
use neo_math::D;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn commit_row_major_matches_commit_over_column_major_buffer() {
    let d = D;
    let kappa = 3usize;

    for &m in &[1usize, 7usize, 300usize] {
        let mut rng = ChaCha8Rng::from_seed([7u8; 32]);
        let pp = setup(&mut rng, d, kappa, m).expect("setup");

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
