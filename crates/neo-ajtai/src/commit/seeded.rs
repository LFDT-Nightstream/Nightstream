//! Seeded Ajtai setup and low-norm commitment paths.

use super::*;

fn seeded_pp_chunking(m: usize) -> (usize, usize) {
    let chunk_size = core::cmp::min(m, 1 << 15).max(1024);
    let num_chunks = m.div_ceil(chunk_size);
    (chunk_size, num_chunks)
}

#[inline]
fn seeded_pp_row_seeds(master_seed: [u8; 32], kappa: usize) -> Vec<[u8; 32]> {
    let mut rng = ChaCha8Rng::from_seed(master_seed);
    let mut row_seeds = vec![[0u8; 32]; kappa];
    for seed in row_seeds.iter_mut() {
        rng.fill_bytes(seed);
    }
    row_seeds
}

#[inline]
fn seeded_pp_chunk_seeds_for_row(row_seed: [u8; 32], num_chunks: usize) -> Vec<[u8; 32]> {
    let mut seed_rng = ChaCha8Rng::from_seed(row_seed);
    let mut chunk_seeds = vec![[0u8; 32]; num_chunks];
    for seed in chunk_seeds.iter_mut() {
        seed_rng.fill_bytes(seed);
    }
    chunk_seeds
}

/// Deterministically derive PP chunk seeds for a seeded PP (the same partitioning used by [`setup_par`]).
///
/// Returns `(chunk_size, chunk_seeds_by_row)`, where `chunk_seeds_by_row[row][chunk]` seeds the
/// ChaCha stream used to generate the PP ring elements for that chunk.
#[doc(hidden)]
pub fn seeded_pp_chunk_seeds(master_seed: [u8; 32], kappa: usize, m: usize) -> (usize, Vec<Vec<[u8; 32]>>) {
    let (chunk_size, num_chunks) = seeded_pp_chunking(m);
    let row_seeds = seeded_pp_row_seeds(master_seed, kappa);
    let chunk_seeds = row_seeds
        .into_iter()
        .map(|rs| seeded_pp_chunk_seeds_for_row(rs, num_chunks))
        .collect();
    (chunk_size, chunk_seeds)
}

#[inline]
fn commit_row_major_seeded_row(
    chunk_size: usize,
    chunk_seeds: &[[u8; 32]],
    m: usize,
    z_rows: &[&[Fq]],
    last_nonzero_by_col: &[usize],
) -> [Fq; D] {
    let mut acc = [Fq::ZERO; D];
    let mut nxt = [Fq::ZERO; D];
    for (chunk_idx, seed) in chunk_seeds.iter().copied().enumerate() {
        let start = chunk_idx * chunk_size;
        let end = core::cmp::min(m, start + chunk_size);
        let mut rng = ChaCha8Rng::from_seed(seed);
        for col_idx in start..end {
            let last_t = last_nonzero_by_col[col_idx];
            if last_t == usize::MAX {
                skip_uniform_rq_coeffs(&mut rng);
                continue;
            }
            let mut rot_col = sample_uniform_rq_coeffs(&mut rng);
            for t in 0..last_t {
                let mask = z_rows[t][col_idx];
                if mask != Fq::ZERO {
                    acc_mul_add_inplace(&mut acc, &rot_col, mask);
                }
                rot_step(&rot_col, &mut nxt);
                core::mem::swap(&mut rot_col, &mut nxt);
            }
            let mask = z_rows[last_t][col_idx];
            if mask != Fq::ZERO {
                acc_mul_add_inplace(&mut acc, &rot_col, mask);
            }
        }
    }
    acc
}

#[inline]
fn accumulate_binary_mask_sparse(acc: &mut [Fq; D], nxt: &mut [Fq; D], rot_col: &mut [Fq; D], mut mask: u64) {
    let mut rot_pos = 0usize;
    while mask != 0 {
        let next_pos = mask.trailing_zeros() as usize;
        let run_len = (mask >> next_pos).trailing_ones() as usize;
        let delta = next_pos - rot_pos;
        if delta == 0 {
            acc_add_inplace(acc, rot_col);
        } else if delta == 1 {
            rot_step_add_phi_81(rot_col, nxt, acc);
            core::mem::swap(rot_col, nxt);
        } else {
            rot_advance_add_phi_81(rot_col, delta, nxt, acc);
            core::mem::swap(rot_col, nxt);
        }
        for _ in 1..run_len {
            rot_step_add_phi_81(rot_col, nxt, acc);
            core::mem::swap(rot_col, nxt);
        }
        rot_pos = next_pos + run_len - 1;
        mask &= !(((1u64 << run_len) - 1) << next_pos);
    }
}

#[inline]
fn commit_row_major_seeded_binary_cols_chunk(seed: [u8; 32], start: usize, end: usize, column_bits: &[u64]) -> [Fq; D] {
    const VALID_MASK: u64 = (1u64 << D) - 1;

    let mut acc = [Fq::ZERO; D];
    let mut nxt = [Fq::ZERO; D];
    let mut rng = ChaCha8Rng::from_seed(seed);
    let mut batch_words = [0u64; SEEDED_RQ_BATCH * D];
    let mut rot_col = [Fq::ZERO; D];

    let mut col_idx = start;
    while col_idx < end {
        let batch = (end - col_idx).min(SEEDED_RQ_BATCH);
        let all_zero_batch = column_bits[col_idx..col_idx + batch]
            .iter()
            .all(|mask| (*mask & VALID_MASK) == 0);
        if all_zero_batch {
            if advance_uniform_rq_coeff_validity_batch(&mut rng, batch) {
                col_idx += batch;
                continue;
            }
            for _ in 0..batch {
                skip_uniform_rq_coeffs(&mut rng);
            }
            col_idx += batch;
            continue;
        }
        if fill_uniform_rq_coeff_words_batch(&mut rng, batch, &mut batch_words) {
            for batch_idx in 0..batch {
                let mask = column_bits[col_idx + batch_idx] & VALID_MASK;
                if mask == 0 {
                    continue;
                }
                let word_start = batch_idx * D;
                let word_end = word_start + D;
                copy_uniform_rq_coeffs_from_words(&batch_words[word_start..word_end], &mut rot_col);
                if mask.count_ones() >= DENSE_BINARY_MASK_THRESHOLD {
                    let product = RqEl(rot_col).mul(&binary_mask_poly(mask));
                    acc_add_inplace(&mut acc, &product.0);
                } else {
                    accumulate_binary_mask_sparse(&mut acc, &mut nxt, &mut rot_col, mask);
                }
            }
        } else {
            for batch_idx in 0..batch {
                let mask = column_bits[col_idx + batch_idx] & VALID_MASK;
                if mask == 0 {
                    skip_uniform_rq_coeffs(&mut rng);
                    continue;
                }
                rot_col = sample_uniform_rq_coeffs(&mut rng);
                if mask.count_ones() >= DENSE_BINARY_MASK_THRESHOLD {
                    let product = RqEl(rot_col).mul(&binary_mask_poly(mask));
                    acc_add_inplace(&mut acc, &product.0);
                } else {
                    accumulate_binary_mask_sparse(&mut acc, &mut nxt, &mut rot_col, mask);
                }
            }
        }
        col_idx += batch;
    }
    acc
}

#[inline]
fn commit_row_major_seeded_binary_cols_row(
    chunk_size: usize,
    chunk_seeds: &[[u8; 32]],
    m: usize,
    column_bits: &[u64],
) -> [Fq; D] {
    let mut acc = [Fq::ZERO; D];
    for (chunk_idx, seed) in chunk_seeds.iter().copied().enumerate() {
        let start = chunk_idx * chunk_size;
        let end = core::cmp::min(m, start + chunk_size);
        let chunk_acc = commit_row_major_seeded_binary_cols_chunk(seed, start, end, column_bits);
        acc_add_inplace(&mut acc, &chunk_acc);
    }
    acc
}

/// Commit to a **row-major** `Mat<Fq>` using a *seeded PP* without materializing the multi-GB PP matrix.
///
/// This produces the same commitment as:
/// - `setup_par(ChaCha8Rng::from_seed(seed), d, kappa, m)` followed by
/// - [`commit_row_major`].
#[allow(non_snake_case)]
#[doc(hidden)]
pub fn commit_row_major_seeded(seed: [u8; 32], d: usize, kappa: usize, m: usize, Z: &Mat<Fq>) -> Commitment {
    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(Z.rows(), d, "Z must be d×m");
    assert_eq!(Z.cols(), m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);
    if m == 0 {
        return C;
    }

    struct Acc {
        acc: [Fq; D],
    }

    impl Acc {
        #[inline]
        fn new() -> Self {
            Self { acc: [Fq::ZERO; D] }
        }
    }

    let z_rows: Vec<&[Fq]> = (0..d).map(|r| Z.row(r)).collect();
    let mut last_nonzero_by_col = vec![usize::MAX; m];
    for col_idx in 0..m {
        for t in (0..d).rev() {
            if z_rows[t][col_idx] != Fq::ZERO {
                last_nonzero_by_col[col_idx] = t;
                break;
            }
        }
    }
    let (chunk_size, chunk_seeds_by_row) = seeded_pp_chunk_seeds(seed, kappa, m);

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        let allow_parallel = rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none();
        let chunk_parallelism = chunk_seeds_by_row
            .first()
            .map_or(0usize, |chunk_seeds| chunk_seeds.len());
        if allow_parallel && kappa > 1 && kappa >= chunk_parallelism {
            C.data.par_chunks_mut(d).enumerate().for_each(|(i, col)| {
                let acc =
                    commit_row_major_seeded_row(chunk_size, &chunk_seeds_by_row[i], m, &z_rows, &last_nonzero_by_col);
                col.copy_from_slice(&acc);
            });
            return C;
        }
    }

    for i in 0..kappa {
        let chunk_seeds = &chunk_seeds_by_row[i];
        let acc = {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                if chunk_seeds.len() == 1 {
                    let mut st = Acc::new();
                    let mut rng = ChaCha8Rng::from_seed(chunk_seeds[0]);
                    let mut nxt = [Fq::ZERO; D];
                    for col_idx in 0..m {
                        let last_t = last_nonzero_by_col[col_idx];
                        if last_t == usize::MAX {
                            skip_uniform_rq_coeffs(&mut rng);
                            continue;
                        }
                        let mut rot_col = sample_uniform_rq_coeffs(&mut rng);
                        for t in 0..last_t {
                            let mask = z_rows[t][col_idx];
                            if mask != Fq::ZERO {
                                acc_mul_add_inplace(&mut st.acc, &rot_col, mask);
                            }
                            rot_step(&rot_col, &mut nxt);
                            core::mem::swap(&mut rot_col, &mut nxt);
                        }
                        let mask = z_rows[last_t][col_idx];
                        if mask != Fq::ZERO {
                            acc_mul_add_inplace(&mut st.acc, &rot_col, mask);
                        }
                    }
                    st
                } else {
                    (0..chunk_seeds.len())
                        .into_par_iter()
                        .fold(Acc::new, |mut st, chunk_idx| {
                            let start = chunk_idx * chunk_size;
                            let end = core::cmp::min(m, start + chunk_size);
                            let mut rng = ChaCha8Rng::from_seed(chunk_seeds[chunk_idx]);
                            let mut nxt = [Fq::ZERO; D];
                            for col_idx in start..end {
                                let last_t = last_nonzero_by_col[col_idx];
                                if last_t == usize::MAX {
                                    skip_uniform_rq_coeffs(&mut rng);
                                    continue;
                                }
                                let mut rot_col = sample_uniform_rq_coeffs(&mut rng);
                                for t in 0..last_t {
                                    let mask = z_rows[t][col_idx];
                                    if mask != Fq::ZERO {
                                        acc_mul_add_inplace(&mut st.acc, &rot_col, mask);
                                    }
                                    rot_step(&rot_col, &mut nxt);
                                    core::mem::swap(&mut rot_col, &mut nxt);
                                }
                                let mask = z_rows[last_t][col_idx];
                                if mask != Fq::ZERO {
                                    acc_mul_add_inplace(&mut st.acc, &rot_col, mask);
                                }
                            }
                            st
                        })
                        .reduce_with(|mut a, b| {
                            for r in 0..d {
                                a.acc[r] += b.acc[r];
                            }
                            a
                        })
                        .unwrap_or_else(Acc::new)
                }
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                let mut st = Acc::new();
                for chunk_idx in 0..chunk_seeds.len() {
                    let start = chunk_idx * chunk_size;
                    let end = core::cmp::min(m, start + chunk_size);
                    let mut rng = ChaCha8Rng::from_seed(chunk_seeds[chunk_idx]);
                    let mut nxt = [Fq::ZERO; D];
                    for col_idx in start..end {
                        let last_t = last_nonzero_by_col[col_idx];
                        if last_t == usize::MAX {
                            skip_uniform_rq_coeffs(&mut rng);
                            continue;
                        }
                        let mut rot_col = sample_uniform_rq_coeffs(&mut rng);
                        for t in 0..last_t {
                            let mask = z_rows[t][col_idx];
                            if mask != Fq::ZERO {
                                acc_mul_add_inplace(&mut st.acc, &rot_col, mask);
                            }
                            rot_step(&rot_col, &mut nxt);
                            core::mem::swap(&mut rot_col, &mut nxt);
                        }
                        let mask = z_rows[last_t][col_idx];
                        if mask != Fq::ZERO {
                            acc_mul_add_inplace(&mut st.acc, &rot_col, mask);
                        }
                    }
                }
                st
            }
        };

        C.col_mut(i).copy_from_slice(&acc.acc);
    }

    C
}

/// Commit to a binary row-major matrix encoded as one `u64` bitmask per column.
///
/// Bit `rho` of `column_bits[c]` is interpreted as `Z[rho, c] ∈ {0,1}`.
/// This is a prover-local fast path for binary witnesses that preserves the
/// exact seeded-PP commitment defined by [`commit_row_major_seeded`].
#[allow(non_snake_case)]
#[doc(hidden)]
pub fn commit_row_major_seeded_binary_cols_with_chunk_seeds(
    d: usize,
    kappa: usize,
    m: usize,
    column_bits: &[u64],
    chunk_size: usize,
    chunk_seeds_by_row: &[Vec<[u8; 32]>],
) -> Commitment {
    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(column_bits.len(), m, "binary column image must have length m");
    assert_eq!(chunk_seeds_by_row.len(), kappa, "chunk seed rows must match kappa");

    let mut C = Commitment::zeros(d, kappa);
    if m == 0 {
        return C;
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        let allow_parallel = rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none();
        let chunk_parallelism = chunk_seeds_by_row
            .first()
            .map_or(0usize, |chunk_seeds| chunk_seeds.len());
        if allow_parallel && kappa > 1 && kappa >= chunk_parallelism {
            C.data.par_chunks_mut(d).enumerate().for_each(|(i, col)| {
                let acc = commit_row_major_seeded_binary_cols_row(chunk_size, &chunk_seeds_by_row[i], m, column_bits);
                col.copy_from_slice(&acc);
            });
            return C;
        }
    }

    for i in 0..kappa {
        let chunk_seeds = &chunk_seeds_by_row[i];
        let acc = {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                if chunk_seeds.len() == 1 {
                    commit_row_major_seeded_binary_cols_row(chunk_size, chunk_seeds, m, column_bits)
                } else {
                    chunk_seeds
                        .par_iter()
                        .copied()
                        .enumerate()
                        .map(|(chunk_idx, seed)| {
                            let start = chunk_idx * chunk_size;
                            let end = core::cmp::min(m, start + chunk_size);
                            commit_row_major_seeded_binary_cols_chunk(seed, start, end, column_bits)
                        })
                        .reduce(
                            || [Fq::ZERO; D],
                            |mut a, b| {
                                acc_add_inplace(&mut a, &b);
                                a
                            },
                        )
                }
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                commit_row_major_seeded_binary_cols_row(chunk_size, chunk_seeds, m, column_bits)
            }
        };
        C.col_mut(i).copy_from_slice(&acc);
    }

    C
}

/// Commit to a binary row-major matrix encoded as one `u64` bitmask per column.
///
/// Bit `rho` of `column_bits[c]` is interpreted as `Z[rho, c] ∈ {0,1}`.
/// This is a prover-local fast path for binary witnesses that preserves the
/// exact seeded-PP commitment defined by [`commit_row_major_seeded`].
#[allow(non_snake_case)]
#[doc(hidden)]
pub fn commit_row_major_seeded_binary_cols(
    seed: [u8; 32],
    d: usize,
    kappa: usize,
    m: usize,
    column_bits: &[u64],
) -> Commitment {
    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(column_bits.len(), m, "binary column image must have length m");

    let (chunk_size, chunk_seeds_by_row) = seeded_pp_chunk_seeds(seed, kappa, m);
    commit_row_major_seeded_binary_cols_with_chunk_seeds(d, kappa, m, column_bits, chunk_size, &chunk_seeds_by_row)
}

/// Commit to many row-major matrices using one seeded PP stream.
///
/// All matrices must share the same `d×m` shape and are committed against the
/// same seeded PP `(seed, d, kappa, m)`.
#[allow(non_snake_case)]
#[doc(hidden)]
pub fn commit_row_major_seeded_many(
    seed: [u8; 32],
    d: usize,
    kappa: usize,
    m: usize,
    Zs: &[&Mat<Fq>],
) -> Vec<Commitment> {
    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    if Zs.is_empty() {
        return Vec::new();
    }
    for (idx, z) in Zs.iter().enumerate() {
        assert_eq!(z.rows(), d, "Zs[{idx}] must be d×m");
        assert_eq!(z.cols(), m, "Zs[{idx}] must be d×m");
    }

    let n = Zs.len();
    let mut out: Vec<Commitment> = (0..n).map(|_| Commitment::zeros(d, kappa)).collect();
    if m == 0 {
        return out;
    }

    // Fast row slices per matrix.
    let z_rows_all: Vec<Vec<&[Fq]>> = Zs
        .iter()
        .map(|z| (0..d).map(|r| z.row(r)).collect())
        .collect();

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let allow_parallel = rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none();

    // Per-(matrix,column) sparsity metadata.
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let last_nonzero_all: Vec<Vec<usize>> = if allow_parallel && n >= 4 {
        z_rows_all
            .par_iter()
            .map(|z_rows| {
                let mut last_nonzero_by_col = vec![usize::MAX; m];
                for col_idx in 0..m {
                    for t in (0..d).rev() {
                        if z_rows[t][col_idx] != Fq::ZERO {
                            last_nonzero_by_col[col_idx] = t;
                            break;
                        }
                    }
                }
                last_nonzero_by_col
            })
            .collect()
    } else {
        z_rows_all
            .iter()
            .map(|z_rows| {
                let mut last_nonzero_by_col = vec![usize::MAX; m];
                for col_idx in 0..m {
                    for t in (0..d).rev() {
                        if z_rows[t][col_idx] != Fq::ZERO {
                            last_nonzero_by_col[col_idx] = t;
                            break;
                        }
                    }
                }
                last_nonzero_by_col
            })
            .collect()
    };
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let last_nonzero_all: Vec<Vec<usize>> = z_rows_all
        .iter()
        .map(|z_rows| {
            let mut last_nonzero_by_col = vec![usize::MAX; m];
            for col_idx in 0..m {
                for t in (0..d).rev() {
                    if z_rows[t][col_idx] != Fq::ZERO {
                        last_nonzero_by_col[col_idx] = t;
                        break;
                    }
                }
            }
            last_nonzero_by_col
        })
        .collect();

    let (chunk_size, chunk_seeds_by_row) = seeded_pp_chunk_seeds(seed, kappa, m);

    for i in 0..kappa {
        let chunk_seeds = &chunk_seeds_by_row[i];
        let accs: Vec<[Fq; D]> = {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                if chunk_seeds.len() == 1 {
                    if n == 1 {
                        vec![commit_row_major_seeded_row(
                            chunk_size,
                            chunk_seeds,
                            m,
                            &z_rows_all[0],
                            &last_nonzero_all[0],
                        )]
                    } else {
                        let mut rng = ChaCha8Rng::from_seed(chunk_seeds[0]);
                        let base_cols: Vec<[Fq; D]> = (0..m).map(|_| sample_uniform_rq_coeffs(&mut rng)).collect();

                        if allow_parallel && n > 1 {
                            (0..n)
                                .into_par_iter()
                                .map(|z_idx| {
                                    let mut acc = [Fq::ZERO; D];
                                    let z_rows = &z_rows_all[z_idx];
                                    let last_nonzero = &last_nonzero_all[z_idx];
                                    let mut nxt = [Fq::ZERO; D];
                                    for col_idx in 0..m {
                                        let last_t = last_nonzero[col_idx];
                                        if last_t == usize::MAX {
                                            continue;
                                        }
                                        let mut rot_col = base_cols[col_idx];
                                        for t in 0..last_t {
                                            let mask = z_rows[t][col_idx];
                                            if mask != Fq::ZERO {
                                                acc_mul_add_inplace(&mut acc, &rot_col, mask);
                                            }
                                            rot_step(&rot_col, &mut nxt);
                                            core::mem::swap(&mut rot_col, &mut nxt);
                                        }
                                        let mask = z_rows[last_t][col_idx];
                                        if mask != Fq::ZERO {
                                            acc_mul_add_inplace(&mut acc, &rot_col, mask);
                                        }
                                    }
                                    acc
                                })
                                .collect()
                        } else {
                            let mut local = vec![[Fq::ZERO; D]; n];
                            let mut nxt = [Fq::ZERO; D];
                            for col_idx in 0..m {
                                let base_col = base_cols[col_idx];
                                for z_idx in 0..n {
                                    let mut rot_col = base_col;
                                    let last_t = last_nonzero_all[z_idx][col_idx];
                                    if last_t == usize::MAX {
                                        continue;
                                    }
                                    let z_rows = &z_rows_all[z_idx];
                                    for t in 0..last_t {
                                        let mask = z_rows[t][col_idx];
                                        if mask != Fq::ZERO {
                                            acc_mul_add_inplace(&mut local[z_idx], &rot_col, mask);
                                        }
                                        rot_step(&rot_col, &mut nxt);
                                        core::mem::swap(&mut rot_col, &mut nxt);
                                    }
                                    let mask = z_rows[last_t][col_idx];
                                    if mask != Fq::ZERO {
                                        acc_mul_add_inplace(&mut local[z_idx], &rot_col, mask);
                                    }
                                }
                            }
                            local
                        }
                    }
                } else if allow_parallel {
                    (0..chunk_seeds.len())
                        .into_par_iter()
                        .fold(
                            || vec![[Fq::ZERO; D]; n],
                            |mut local, chunk_idx| {
                                let start = chunk_idx * chunk_size;
                                let end = core::cmp::min(m, start + chunk_size);
                                let mut rng = ChaCha8Rng::from_seed(chunk_seeds[chunk_idx]);
                                let mut nxt = [Fq::ZERO; D];
                                for col_idx in start..end {
                                    let base_col = sample_uniform_rq_coeffs(&mut rng);
                                    for z_idx in 0..n {
                                        let mut rot_col = base_col;
                                        let last_t = last_nonzero_all[z_idx][col_idx];
                                        if last_t == usize::MAX {
                                            continue;
                                        }
                                        let z_rows = &z_rows_all[z_idx];
                                        for t in 0..last_t {
                                            let mask = z_rows[t][col_idx];
                                            if mask != Fq::ZERO {
                                                acc_mul_add_inplace(&mut local[z_idx], &rot_col, mask);
                                            }
                                            rot_step(&rot_col, &mut nxt);
                                            core::mem::swap(&mut rot_col, &mut nxt);
                                        }
                                        let mask = z_rows[last_t][col_idx];
                                        if mask != Fq::ZERO {
                                            acc_mul_add_inplace(&mut local[z_idx], &rot_col, mask);
                                        }
                                    }
                                }
                                local
                            },
                        )
                        .reduce(
                            || vec![[Fq::ZERO; D]; n],
                            |mut a, b| {
                                for z_idx in 0..n {
                                    for r in 0..d {
                                        a[z_idx][r] += b[z_idx][r];
                                    }
                                }
                                a
                            },
                        )
                } else {
                    let mut local = vec![[Fq::ZERO; D]; n];
                    for (chunk_idx, seed) in chunk_seeds.iter().copied().enumerate() {
                        let start = chunk_idx * chunk_size;
                        let end = core::cmp::min(m, start + chunk_size);
                        let mut rng = ChaCha8Rng::from_seed(seed);
                        let mut nxt = [Fq::ZERO; D];
                        for col_idx in start..end {
                            let base_col = sample_uniform_rq_coeffs(&mut rng);
                            for z_idx in 0..n {
                                let mut rot_col = base_col;
                                let last_t = last_nonzero_all[z_idx][col_idx];
                                if last_t == usize::MAX {
                                    continue;
                                }
                                let z_rows = &z_rows_all[z_idx];
                                for t in 0..last_t {
                                    let mask = z_rows[t][col_idx];
                                    if mask != Fq::ZERO {
                                        acc_mul_add_inplace(&mut local[z_idx], &rot_col, mask);
                                    }
                                    rot_step(&rot_col, &mut nxt);
                                    core::mem::swap(&mut rot_col, &mut nxt);
                                }
                                let mask = z_rows[last_t][col_idx];
                                if mask != Fq::ZERO {
                                    acc_mul_add_inplace(&mut local[z_idx], &rot_col, mask);
                                }
                            }
                        }
                    }
                    local
                }
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                let mut local = vec![[Fq::ZERO; D]; n];
                for chunk_idx in 0..chunk_seeds.len() {
                    let start = chunk_idx * chunk_size;
                    let end = core::cmp::min(m, start + chunk_size);
                    let mut rng = ChaCha8Rng::from_seed(chunk_seeds[chunk_idx]);
                    let mut nxt = [Fq::ZERO; D];
                    for col_idx in start..end {
                        let base_col = sample_uniform_rq_coeffs(&mut rng);
                        for z_idx in 0..n {
                            let mut rot_col = base_col;
                            let last_t = last_nonzero_all[z_idx][col_idx];
                            if last_t == usize::MAX {
                                continue;
                            }
                            let z_rows = &z_rows_all[z_idx];
                            for t in 0..last_t {
                                let mask = z_rows[t][col_idx];
                                if mask != Fq::ZERO {
                                    acc_mul_add_inplace(&mut local[z_idx], &rot_col, mask);
                                }
                                rot_step(&rot_col, &mut nxt);
                                core::mem::swap(&mut rot_col, &mut nxt);
                            }
                            let mask = z_rows[last_t][col_idx];
                            if mask != Fq::ZERO {
                                acc_mul_add_inplace(&mut local[z_idx], &rot_col, mask);
                            }
                        }
                    }
                }
                local
            }
        };

        for z_idx in 0..n {
            out[z_idx].col_mut(i).copy_from_slice(&accs[z_idx]);
        }
    }

    out
}
