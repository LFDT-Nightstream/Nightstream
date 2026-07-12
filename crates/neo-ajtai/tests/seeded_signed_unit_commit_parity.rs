//! Parity pins for the seeded signed-unit commit paths.
//!
//! `AjtaiSModule::commit_many` on signed-unit witnesses uses the seeded
//! streaming batch path; `commit` uses the per-witness streaming path; and
//! `setup_par` materializes the same key densely. All three must produce
//! identical commitments — any divergence is a key-derivation bug. (A
//! base-column cache for the batch path was tried on 2026-06-10 and
//! dropped: ChaCha8 regeneration in hot buffers measured faster than
//! streaming a multi-MB cache from memory. These pins stay so any future
//! retry has its parity oracle ready.)

use neo_ajtai::{commit_row_major, get_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_math::ring::D;
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;

const KAPPA: usize = 5;
/// Unique (d, m) so this test owns its global registry entry.
const M_COLS: usize = 73;

fn signed_unit_mat(pattern: u64) -> Mat<Fq> {
    let mut z = Mat::zero(D, M_COLS, Fq::ZERO);
    let mut state = pattern.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    for col in 0..M_COLS {
        for row in 0..D {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            match state >> 62 {
                0 => z[(row, col)] = Fq::ONE,
                1 => z[(row, col)] = Fq::ZERO - Fq::ONE,
                _ => {}
            }
        }
    }
    z
}

#[test]
fn cached_commit_many_matches_streaming_and_dense_oracles() {
    let seed = [0xA7u8; 32];
    set_global_pp_seeded(D, KAPPA, M_COLS, seed).expect("register seeded PP");
    let module = AjtaiSModule::from_global_for_dims(D, M_COLS).expect("module");

    let mats: Vec<Mat<Fq>> = vec![
        signed_unit_mat(1),
        signed_unit_mat(2),
        Mat::zero(D, M_COLS, Fq::ZERO), // all-zero witness in the batch
        signed_unit_mat(3),
    ];
    let refs: Vec<&Mat<Fq>> = mats.iter().collect();
    let packed_mats = mats
        .iter()
        .map(|matrix| Mat::compact_signed_unit(D, M_COLS, matrix.to_dense_vec()))
        .collect::<Vec<_>>();
    let packed_refs = packed_mats.iter().collect::<Vec<_>>();

    // Cold (builds the cache) and warm (reuses it) must agree.
    let cold = module.commit_many(&refs);
    let warm = module.commit_many(&refs);
    let packed = module.commit_many(&packed_refs);
    assert_eq!(cold, warm, "cache warm-up must not change commitments");
    assert_eq!(cold, packed, "bit-packed storage must not change commitments");

    // Oracle 1: the single-witness streaming signed-unit path.
    for (z, expected) in mats.iter().zip(&cold) {
        assert_eq!(
            &module.commit(z),
            expected,
            "cached commit_many diverged from the streaming single-commit path"
        );
    }

    // Oracle 2: dense multiplication against the PP materialized from the
    // same seed by `setup_par`.
    let pp = get_global_pp_for_dims(D, M_COLS).expect("materialize seeded PP");
    for (z, expected) in mats.iter().zip(&cold) {
        assert_eq!(
            &commit_row_major(&pp, z),
            expected,
            "cached commit_many diverged from the dense materialized-PP oracle"
        );
    }
}
