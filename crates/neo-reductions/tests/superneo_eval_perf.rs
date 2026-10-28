use std::hint::black_box;
use std::time::Instant;

use neo_ccs::{matrix::Mat, poly::SparsePoly, CcsStructure};
use neo_math::KExtensions;
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::{
    build_superneo_eval_cache, eval_all_mats_cached_with_blocks, eval_all_mats_superneo, SuperneoZBlocks,
};
use p3_field::PrimeCharacteristicRing;

fn chi_table(point: &[K]) -> Vec<K> {
    let n = 1usize << point.len();
    let mut out = vec![K::ZERO; n];
    for (idx, out_cell) in out.iter_mut().enumerate().take(n) {
        let mut w = K::ONE;
        for (bit, p) in point.iter().copied().enumerate() {
            let is_one = ((idx >> bit) & 1) == 1;
            w *= if is_one { p } else { K::ONE - p };
        }
        *out_cell = w;
    }
    out
}

fn sparseish_matrix(rows: usize, cols: usize, seed: u64) -> Mat<F> {
    let mut m = Mat::zero(rows, cols, F::ZERO);
    for r in 0..rows {
        for c in 0..cols {
            if ((r as u64 * 17) + (c as u64 * 13) + seed).is_multiple_of(37) {
                m[(r, c)] = F::from_u64(((r + 3 * c + (seed as usize % 19)) % 23 + 1) as u64);
            }
        }
    }
    m
}

#[test]
#[ignore = "perf report only; run with --release -- --ignored --nocapture"]
fn report_superneo_cache_eval_speedup() {
    let n = 256usize;
    let m = 4 * D;

    let mats = vec![
        sparseish_matrix(n, m, 11),
        sparseish_matrix(n, m, 22),
        sparseish_matrix(n, m, 33),
        sparseish_matrix(n, m, 44),
    ];
    let s = CcsStructure::new(mats, SparsePoly::new(4, vec![])).expect("valid CCS");
    let cache = build_superneo_eval_cache(&s).expect("D-compatible width should build cache");

    let z: Vec<K> = (0..m)
        .map(|i| K::from_coeffs([F::from_u64((i % 97 + 1) as u64), F::from_u64((i % 31) as u64)]))
        .collect();
    let z_blocks = SuperneoZBlocks::from_z(&z);

    let r = vec![
        K::from_coeffs([F::from_u64(2), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(3), F::from_u64(0)]),
        K::from_coeffs([F::from_u64(5), F::from_u64(2)]),
        K::from_coeffs([F::from_u64(7), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(11), F::from_u64(0)]),
        K::from_coeffs([F::from_u64(13), F::from_u64(2)]),
        K::from_coeffs([F::from_u64(17), F::from_u64(1)]),
        K::from_coeffs([F::from_u64(19), F::from_u64(0)]),
    ];
    let chi_r = chi_table(&r);

    let out_baseline = eval_all_mats_superneo(&s, &z, &chi_r, n);
    let out_cached = eval_all_mats_cached_with_blocks(&cache, &z_blocks, &chi_r, n);
    assert_eq!(out_baseline, out_cached, "cached SuperNeo eval must match baseline");

    let iters = 200usize;

    let mut checksum_baseline = K::ZERO;
    let start_baseline = Instant::now();
    for _ in 0..iters {
        let out = eval_all_mats_superneo(&s, &z, &chi_r, n);
        checksum_baseline += out[0];
        black_box(&out);
    }
    let elapsed_baseline = start_baseline.elapsed();

    let mut checksum_cached = K::ZERO;
    let start_cached = Instant::now();
    for _ in 0..iters {
        let out = eval_all_mats_cached_with_blocks(&cache, &z_blocks, &chi_r, n);
        checksum_cached += out[0];
        black_box(&out);
    }
    let elapsed_cached = start_cached.elapsed();

    assert_eq!(checksum_baseline, checksum_cached, "perf loop checksums diverged");

    let baseline_s = elapsed_baseline.as_secs_f64();
    let cached_s = elapsed_cached.as_secs_f64();
    let speedup = baseline_s / cached_s;

    eprintln!(
        "\\n[superneo-eval-perf] workload: n={n}, m={m}, mats={}, iters={iters}",
        s.t()
    );
    eprintln!("[superneo-eval-perf] baseline superneo: {:.6}s", baseline_s);
    eprintln!("[superneo-eval-perf] cached with pre-split blocks: {:.6}s", cached_s);
    eprintln!("[superneo-eval-perf] speedup: {:.3}x", speedup);
}
