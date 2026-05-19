use std::time::Instant;

use neo_ajtai::commit_row_major_seeded_binary_cols;
use neo_math::D;
use rayon::ThreadPoolBuilder;

const CONSTRUCTION2_FULL_WIDTH_BITS: usize = 4_627_840;
const CONSTRUCTION2_KAPPA: usize = 16;
const CONSTRUCTION2_M: usize = (CONSTRUCTION2_FULL_WIDTH_BITS + D - 1) / D;

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
