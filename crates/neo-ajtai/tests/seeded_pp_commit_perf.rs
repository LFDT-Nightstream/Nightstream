use std::sync::Arc;
use std::time::Instant;

use neo_ajtai::{
    commit_row_major_seeded_binary_cols, get_global_pp_for_dims, set_global_pp_seeded, setup_par, AjtaiSModule,
};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_math::D;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;

const CONSTRUCTION2_FULL_WIDTH_BITS: usize = 4_627_840;
const CONSTRUCTION2_KAPPA: usize = 16;
const CONSTRUCTION2_M: usize = (CONSTRUCTION2_FULL_WIDTH_BITS + D - 1) / D;
const SHA_FPRIME_SIGNED_UNIT_M: usize = 16_384;
const SHA_FPRIME_SIGNED_UNIT_CLAIMS: usize = 14;
const SHA_SERIAL_QUAD_SIGNED_UNIT_M: usize = 26_249;
const SHA_SERIAL_QUAD_KAPPA: usize = 18;

fn representative_construction2_column_bits() -> Vec<u64> {
    let mut out = vec![0u64; CONSTRUCTION2_M];
    for (col, mask) in out.iter_mut().enumerate() {
        let active = col.wrapping_mul(0x9E37_79B9_7F4A_7C15usize).rotate_left(11) % 100 < 29;
        if !active {
            continue;
        }
        let mut bits = 0u64;
        let mut state = (col as u64)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9)
            .wrapping_add(0x94D0_49BB_1331_11EB);
        while bits.count_ones() < 26 {
            state ^= state >> 30;
            state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let bit = (state % D as u64) as usize;
            bits |= 1u64 << bit;
        }
        *mask = bits;
    }
    out
}

fn representative_signed_unit_claims(count: usize, m: usize) -> Vec<Mat<Fq>> {
    let neg_one = Fq::ZERO - Fq::ONE;
    (0..count)
        .map(|claim| {
            let mut data = vec![Fq::ZERO; D * m];
            for col in 0..m {
                let active = col
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15usize)
                    .wrapping_add(claim.wrapping_mul(0xD1B5_4A32_D192_ED03usize))
                    .rotate_left(7)
                    % 100
                    < 18;
                if !active {
                    continue;
                }
                let lane_a = (col.wrapping_mul(17).wrapping_add(claim * 11)) % D;
                let lane_b = (col.wrapping_mul(37).wrapping_add(claim * 19 + 5)) % D;
                data[lane_a * m + col] = Fq::ONE;
                if lane_b != lane_a {
                    data[lane_b * m + col] = neg_one;
                }
            }
            Mat::from_row_major(D, m, data)
        })
        .collect()
}

fn run_perf_rounds<F>(rounds: usize, mut run_once: F) -> (f64, f64, f64)
where
    F: FnMut(),
{
    let mut elapsed_ms = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let started = Instant::now();
        run_once();
        elapsed_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    elapsed_ms.sort_by(|a, b| a.partial_cmp(b).expect("finite timing"));
    let best_ms = elapsed_ms[0];
    let median_ms = elapsed_ms[rounds / 2];
    let mean_ms = elapsed_ms.iter().sum::<f64>() / rounds as f64;
    (best_ms, median_ms, mean_ms)
}

#[test]
#[ignore]
fn seeded_pp_binary_cols_representative_construction2_perf_snapshot() {
    let seed = [0x5au8; 32];
    let column_bits = representative_construction2_column_bits();
    let warmup = commit_row_major_seeded_binary_cols(seed, D, CONSTRUCTION2_KAPPA, CONSTRUCTION2_M, &column_bits);

    let rounds = 5usize;
    let mut commitment = warmup.clone();
    let (best_ms, median_ms, mean_ms) = run_perf_rounds(rounds, || {
        commitment = commit_row_major_seeded_binary_cols(seed, D, CONSTRUCTION2_KAPPA, CONSTRUCTION2_M, &column_bits);
    });

    let nonzero = column_bits.iter().filter(|mask| **mask != 0).count();
    let total_popcount: usize = column_bits
        .iter()
        .map(|mask| mask.count_ones() as usize)
        .sum();
    let avg_popcount = total_popcount as f64 / column_bits.len() as f64;

    println!("seeded_pp_binary_cols_representative_construction2_perf_snapshot");
    println!("  d={D}");
    println!("  kappa={CONSTRUCTION2_KAPPA}");
    println!("  m={CONSTRUCTION2_M}");
    println!("  nonzero_cols={nonzero}");
    println!("  avg_popcount={avg_popcount:.2}");
    println!("  rounds={rounds}");
    println!("  best_ms={best_ms:.3}");
    println!("  median_ms={median_ms:.3}");
    println!("  mean_ms={mean_ms:.3}");
    println!("  commitment_words={}", commitment.data.len());
}

#[test]
#[ignore]
fn seeded_pp_binary_cols_representative_construction2_perf_snapshot_single_thread() {
    let seed = [0x5au8; 32];
    let column_bits = representative_construction2_column_bits();
    let pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("single-thread rayon pool");

    let warmup = pool
        .install(|| commit_row_major_seeded_binary_cols(seed, D, CONSTRUCTION2_KAPPA, CONSTRUCTION2_M, &column_bits));

    let rounds = 5usize;
    let mut commitment = warmup.clone();
    let (best_ms, median_ms, mean_ms) = run_perf_rounds(rounds, || {
        commitment = pool.install(|| {
            commit_row_major_seeded_binary_cols(seed, D, CONSTRUCTION2_KAPPA, CONSTRUCTION2_M, &column_bits)
        });
    });

    let nonzero = column_bits.iter().filter(|mask| **mask != 0).count();
    let total_popcount: usize = column_bits
        .iter()
        .map(|mask| mask.count_ones() as usize)
        .sum();
    let avg_popcount = total_popcount as f64 / column_bits.len() as f64;

    println!("seeded_pp_binary_cols_representative_construction2_perf_snapshot_single_thread");
    println!("  d={D}");
    println!("  kappa={CONSTRUCTION2_KAPPA}");
    println!("  m={CONSTRUCTION2_M}");
    println!("  nonzero_cols={nonzero}");
    println!("  avg_popcount={avg_popcount:.2}");
    println!("  rounds={rounds}");
    println!("  best_ms={best_ms:.3}");
    println!("  median_ms={median_ms:.3}");
    println!("  mean_ms={mean_ms:.3}");
    println!("  commitment_words={}", commitment.data.len());
}

#[test]
#[ignore]
fn seeded_pp_signed_unit_commit_many_sha_fprime_perf_snapshot() {
    let seed = [0x6du8; 32];
    let d = D;
    let kappa = CONSTRUCTION2_KAPPA;
    let m = SHA_FPRIME_SIGNED_UNIT_M;
    set_global_pp_seeded(d, kappa, m, seed).expect("register seeded PP");
    let log = AjtaiSModule::from_global_for_dims(d, m).expect("seeded global module");
    let claims = representative_signed_unit_claims(SHA_FPRIME_SIGNED_UNIT_CLAIMS, m);
    let refs = claims.iter().collect::<Vec<_>>();

    let warmup = log.commit_many(&refs);
    let rounds = 5usize;
    let mut commitments = warmup.clone();
    let (best_ms, median_ms, mean_ms) = run_perf_rounds(rounds, || {
        commitments = log.commit_many(&refs);
    });

    let nonzero_cols = (0..m)
        .filter(|&col| {
            claims
                .iter()
                .any(|claim| (0..D).any(|row| claim[(row, col)] != Fq::ZERO))
        })
        .count();
    let nonzero_entries: usize = claims
        .iter()
        .map(|claim| {
            claim
                .as_slice()
                .iter()
                .filter(|value| **value != Fq::ZERO)
                .count()
        })
        .sum();
    let avg_nonzero_per_claim = nonzero_entries as f64 / claims.len() as f64;

    println!("seeded_pp_signed_unit_commit_many_sha_fprime_perf_snapshot");
    println!("  d={d}");
    println!("  kappa={kappa}");
    println!("  m={m}");
    println!("  claims={}", claims.len());
    println!("  nonzero_cols={nonzero_cols}");
    println!("  avg_nonzero_entries_per_claim={avg_nonzero_per_claim:.2}");
    println!("  rounds={rounds}");
    println!("  best_ms={best_ms:.3}");
    println!("  median_ms={median_ms:.3}");
    println!("  mean_ms={mean_ms:.3}");
    println!(
        "  commitment_words={}",
        commitments.iter().map(|c| c.data.len()).sum::<usize>()
    );
}

#[test]
#[ignore]
fn seeded_vs_materialized_signed_unit_commit_many_sha_serial_quad_perf_snapshot() {
    let seed = [0x71u8; 32];
    let d = D;
    let kappa = SHA_SERIAL_QUAD_KAPPA;
    let m = SHA_SERIAL_QUAD_SIGNED_UNIT_M;
    let claims = representative_signed_unit_claims(SHA_FPRIME_SIGNED_UNIT_CLAIMS, m);
    let refs = claims.iter().collect::<Vec<_>>();

    set_global_pp_seeded(d, kappa, m, seed).expect("register seeded PP");
    let seeded = AjtaiSModule::from_global_for_dims(d, m).expect("seeded global module");
    let seeded_warmup = seeded.commit_many(&refs);

    let rounds = 5usize;
    let mut seeded_commitments = seeded_warmup.clone();
    let (seeded_best_ms, seeded_median_ms, seeded_mean_ms) = run_perf_rounds(rounds, || {
        seeded_commitments = seeded.commit_many(&refs);
    });

    let _loaded_global_pp = get_global_pp_for_dims(d, m).expect("materialize seeded global PP");
    let global_loaded_warmup = seeded.commit_many(&refs);
    assert_eq!(
        seeded_warmup, global_loaded_warmup,
        "loaded seeded-global path must commit to the same values"
    );
    let mut global_loaded_commitments = global_loaded_warmup.clone();
    let (global_loaded_best_ms, global_loaded_median_ms, global_loaded_mean_ms) = run_perf_rounds(rounds, || {
        global_loaded_commitments = seeded.commit_many(&refs);
    });
    assert_eq!(seeded_commitments, global_loaded_commitments);

    let mut rng = ChaCha8Rng::from_seed(seed);
    let pp = setup_par(&mut rng, d, kappa, m).expect("materialize PP");
    let materialized = AjtaiSModule::new(Arc::new(pp));
    let materialized_warmup = materialized.commit_many(&refs);
    assert_eq!(
        seeded_warmup, materialized_warmup,
        "seeded and materialized PP paths must commit to the same values"
    );

    let mut materialized_commitments = materialized_warmup.clone();
    let (mat_best_ms, mat_median_ms, mat_mean_ms) = run_perf_rounds(rounds, || {
        materialized_commitments = materialized.commit_many(&refs);
    });
    assert_eq!(seeded_commitments, materialized_commitments);

    let nonzero_entries: usize = claims
        .iter()
        .map(|claim| {
            claim
                .as_slice()
                .iter()
                .filter(|value| **value != Fq::ZERO)
                .count()
        })
        .sum();
    let avg_nonzero_per_claim = nonzero_entries as f64 / claims.len() as f64;
    let pp_words = d * kappa * m;
    let pp_mib = pp_words * std::mem::size_of::<Fq>() / (1024 * 1024);

    println!("seeded_vs_materialized_signed_unit_commit_many_sha_serial_quad_perf_snapshot");
    println!("  d={d}");
    println!("  kappa={kappa}");
    println!("  m={m}");
    println!("  claims={}", claims.len());
    println!("  avg_nonzero_entries_per_claim={avg_nonzero_per_claim:.2}");
    println!("  materialized_pp_words={pp_words}");
    println!("  materialized_pp_mib~={pp_mib}");
    println!("  rounds={rounds}");
    println!("  seeded_best_ms={seeded_best_ms:.3}");
    println!("  seeded_median_ms={seeded_median_ms:.3}");
    println!("  seeded_mean_ms={seeded_mean_ms:.3}");
    println!("  global_loaded_best_ms={global_loaded_best_ms:.3}");
    println!("  global_loaded_median_ms={global_loaded_median_ms:.3}");
    println!("  global_loaded_mean_ms={global_loaded_mean_ms:.3}");
    println!("  materialized_best_ms={mat_best_ms:.3}");
    println!("  materialized_median_ms={mat_median_ms:.3}");
    println!("  materialized_mean_ms={mat_mean_ms:.3}");
    println!(
        "  commitment_words={}",
        materialized_commitments
            .iter()
            .map(|c| c.data.len())
            .sum::<usize>()
    );
}
