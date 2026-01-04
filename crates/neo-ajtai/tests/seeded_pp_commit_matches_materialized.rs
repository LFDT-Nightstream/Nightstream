use neo_ajtai::{commit_row_major, commit_row_major_seeded, setup_par};
use neo_ccs::Mat;
use neo_math::D;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

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
