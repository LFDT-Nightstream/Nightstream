#![allow(non_snake_case)]

use neo_ajtai::{commit_row_major_seeded, setup_par, PP};
use neo_ccs::Mat;
use neo_math::ring::Rq as RqEl;
use neo_math::{D, Fq};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;

fn assert_pp_eq(a: &PP<RqEl>, b: &PP<RqEl>) {
    assert_eq!(a.kappa, b.kappa);
    assert_eq!(a.m, b.m);
    assert_eq!(a.d, b.d);
    assert_eq!(a.m_rows.len(), b.m_rows.len());
    for (row_idx, (ra, rb)) in a.m_rows.iter().zip(b.m_rows.iter()).enumerate() {
        assert_eq!(ra.len(), rb.len(), "row {row_idx} length mismatch");
        for (col_idx, (ea, eb)) in ra.iter().zip(rb.iter()).enumerate() {
            assert_eq!(*ea, *eb, "PP mismatch at (row,col)=({row_idx},{col_idx})");
        }
    }
}

#[test]
fn setup_par_is_deterministic_across_rayon_thread_counts() {
    let d = D;
    let kappa = 3usize;
    let m = 1024usize;

    let pp_1 = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool(1)")
        .install(|| {
            let mut rng = ChaCha8Rng::from_seed([1u8; 32]);
            setup_par(&mut rng, d, kappa, m).expect("setup_par")
        });

    let pp_4 = ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool(4)")
        .install(|| {
            let mut rng = ChaCha8Rng::from_seed([1u8; 32]);
            setup_par(&mut rng, d, kappa, m).expect("setup_par")
        });

    assert_pp_eq(&pp_1, &pp_4);
}

#[test]
fn commit_row_major_seeded_is_deterministic_across_rayon_thread_counts() {
    let d = D;
    let kappa = 3usize;
    let m = 1024usize;
    let seed = [7u8; 32];

    let data: Vec<Fq> = (0..(d * m))
        .map(|i| Fq::from_u64((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ 0x94D0_49BB_1331_11EB))
        .collect();
    let Z = Mat::from_row_major(d, m, data);

    let c_1 = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool(1)")
        .install(|| commit_row_major_seeded(seed, d, kappa, m, &Z));

    let c_4 = ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool(4)")
        .install(|| commit_row_major_seeded(seed, d, kappa, m, &Z));

    assert_eq!(c_1, c_4);
}
