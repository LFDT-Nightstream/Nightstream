use crate::error::{AjtaiError, AjtaiResult};
use crate::types::{Commitment, PP};
use neo_ccs::Mat;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as Fq;
use rand::{CryptoRng, RngCore};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use std::time::Instant;

/// Bring in ring & S-action APIs from neo-math.
use neo_math::ring::{cf, cf_inv as cf_unmap, Rq as RqEl, D, ETA};

// Compile-time guards: this file's rot_step assumes Φ₈₁ (η=81 ⇒ D=54)
const _: () = assert!(ETA == 81, "rot_step is specialized for η=81 (D=54)");
const _: () = assert!(D == 54, "D must be 54 when η=81");
const DENSE_BINARY_MASK_THRESHOLD: u32 = 32;
pub(crate) const SEEDED_RQ_BATCH: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SeededBinaryColsCommitAudit {
    pub total_ms: f64,
    pub fill_batch_ms: f64,
    pub fallback_sample_ms: f64,
    pub dense_total_ms: f64,
    pub sparse_total_ms: f64,
    pub accepted_batches: usize,
    pub fallback_batches: usize,
    pub zero_columns: usize,
    pub dense_columns: usize,
    pub sparse_columns: usize,
    pub sparse_popcount_1: usize,
    pub sparse_popcount_2: usize,
    pub sparse_popcount_3: usize,
    pub sparse_popcount_4: usize,
    pub sparse_popcount_5_plus: usize,
}

impl SeededBinaryColsCommitAudit {
    #[inline]
    fn accumulate(&mut self, other: Self) {
        self.total_ms += other.total_ms;
        self.fill_batch_ms += other.fill_batch_ms;
        self.fallback_sample_ms += other.fallback_sample_ms;
        self.dense_total_ms += other.dense_total_ms;
        self.sparse_total_ms += other.sparse_total_ms;
        self.accepted_batches += other.accepted_batches;
        self.fallback_batches += other.fallback_batches;
        self.zero_columns += other.zero_columns;
        self.dense_columns += other.dense_columns;
        self.sparse_columns += other.sparse_columns;
        self.sparse_popcount_1 += other.sparse_popcount_1;
        self.sparse_popcount_2 += other.sparse_popcount_2;
        self.sparse_popcount_3 += other.sparse_popcount_3;
        self.sparse_popcount_4 += other.sparse_popcount_4;
        self.sparse_popcount_5_plus += other.sparse_popcount_5_plus;
    }
}

#[inline]
fn record_sparse_popcount(audit: &mut SeededBinaryColsCommitAudit, popcount: u32) {
    match popcount {
        1 => audit.sparse_popcount_1 += 1,
        2 => audit.sparse_popcount_2 += 1,
        3 => audit.sparse_popcount_3 += 1,
        4 => audit.sparse_popcount_4 += 1,
        _ => audit.sparse_popcount_5_plus += 1,
    }
}

/// Sample a uniform element from F_q using rejection sampling to avoid bias.
#[inline]
fn sample_uniform_fq<R: RngCore + CryptoRng>(rng: &mut R) -> Fq {
    const Q: u64 = <Fq as PrimeField64>::ORDER_U64; // 2^64 - 2^32 + 1
    loop {
        let x = rng.next_u64();
        if x < Q {
            return Fq::from_u64(x);
        }
    }
}

/// Sample a uniform element from R_q by sampling D uniform coefficients in F_q and mapping with `cf^{-1}`.
#[doc(hidden)]
#[inline]
pub fn sample_uniform_rq<R: RngCore + CryptoRng>(rng: &mut R) -> RqEl {
    let coeffs = sample_uniform_rq_coeffs(rng);
    cf_unmap(coeffs)
}

/// Sample a uniform coefficient vector in F_q^D that corresponds to a uniform R_q element.
#[inline]
pub(crate) fn sample_uniform_rq_coeffs<R: RngCore + CryptoRng>(rng: &mut R) -> [Fq; D] {
    const Q: u64 = <Fq as PrimeField64>::ORDER_U64;
    let mut bytes = [0u8; D * 8];
    rng.fill_bytes(&mut bytes);
    core::array::from_fn(|idx| {
        let start = idx * 8;
        let x = u64::from_le_bytes(bytes[start..start + 8].try_into().expect("8-byte chunk"));
        if x < Q {
            Fq::from_u64(x)
        } else {
            sample_uniform_fq(rng)
        }
    })
}

/// Advance the seeded PP stream by one ring element without materializing coefficients.
#[inline]
pub(crate) fn skip_uniform_rq_coeffs<R: RngCore + CryptoRng>(rng: &mut R) {
    const Q: u64 = <Fq as PrimeField64>::ORDER_U64;
    let mut bytes = [0u8; D * 8];
    rng.fill_bytes(&mut bytes);
    for idx in 0..D {
        let start = idx * 8;
        let x = u64::from_le_bytes(bytes[start..start + 8].try_into().expect("8-byte chunk"));
        if x >= Q {
            let _ = sample_uniform_fq(rng);
        }
    }
}

#[inline]
fn raw_u64_rejects_goldilocks(x: u64) -> bool {
    (x >> 32) == u32::MAX as u64 && (x as u32) != 0
}

#[inline]
pub(crate) fn fill_uniform_rq_coeff_words_batch(
    rng: &mut ChaCha8Rng,
    count: usize,
    words: &mut [u64; SEEDED_RQ_BATCH * D],
) -> bool {
    debug_assert!(count <= SEEDED_RQ_BATCH);
    let checkpoint_word_pos = rng.get_word_pos();
    let used = count * D;
    let mut all_valid = true;
    for word in &mut words[..used] {
        let sampled = rng.next_u64();
        *word = sampled;
        all_valid &= !raw_u64_rejects_goldilocks(sampled);
    }
    if !all_valid {
        rng.set_word_pos(checkpoint_word_pos);
    }
    all_valid
}

#[inline]
pub(crate) fn advance_uniform_rq_coeff_validity_batch(rng: &mut ChaCha8Rng, count: usize) -> bool {
    debug_assert!(count <= SEEDED_RQ_BATCH);
    let checkpoint_word_pos = rng.get_word_pos();
    let mut all_valid = true;
    for _ in 0..(count * D) {
        all_valid &= !raw_u64_rejects_goldilocks(rng.next_u64());
    }
    if !all_valid {
        rng.set_word_pos(checkpoint_word_pos);
    }
    all_valid
}

#[inline(always)]
pub(crate) fn copy_uniform_rq_coeffs_from_words(words: &[u64], out: &mut [Fq; D]) {
    debug_assert_eq!(words.len(), D);
    for (idx, word) in words.iter().enumerate() {
        out[idx] = Fq::from_u64(*word);
    }
}

/// Rotation "one-step" for Φ₈₁(X) = X^54 + X^27 + 1
///
/// Turns column t into column t+1 in O(d) (no ring multiply).
/// For Φ₈₁, the step is: next[0] = -v_{d-1}, next[27] = v_{26} - v_{d-1},
/// next[k] = v_{k-1} for k ∈ {1,...,d-1}\{27}.
#[inline(always)]
fn rot_step_phi_81(cur: &[Fq; D], next: &mut [Fq; D]) {
    let last = cur[D - 1];
    // shift: next[k] = cur[k-1] for k>=1; next[0] = 0
    next[0] = Fq::ZERO;
    next[1..D].copy_from_slice(&cur[..(D - 1)]);
    // cyclotomic corrections for X^54 ≡ -X^27 - 1
    next[0] -= last; // -1 * last
    next[27] -= last; // -X^27 * last
}

#[inline(always)]
fn rot_step_add_phi_81(cur: &[Fq; D], next: &mut [Fq; D], acc: &mut [Fq; D]) {
    let last = cur[D - 1];
    let next0 = Fq::ZERO - last;
    next[0] = next0;
    acc[0] += next0;

    for idx in 1..27 {
        let value = cur[idx - 1];
        next[idx] = value;
        acc[idx] += value;
    }

    let next27 = cur[26] - last;
    next[27] = next27;
    acc[27] += next27;

    for idx in 28..D {
        let value = cur[idx - 1];
        next[idx] = value;
        acc[idx] += value;
    }
}

/// Rotation step for internal use by commit implementations.
///
/// This implementation is specialized for η=81 (D=54) as enforced by compile-time assertions.
/// Re-exported publicly only when the `testing` feature is enabled (see lib.rs).
#[inline(always)]
pub fn rot_step(cur: &[Fq; D], next: &mut [Fq; D]) {
    rot_step_phi_81(cur, next)
}

#[inline(always)]
fn rot_advance_add_phi_81(cur: &[Fq; D], delta: usize, next: &mut [Fq; D], acc: &mut [Fq; D]) {
    debug_assert!(delta < D);
    if delta == 0 {
        *next = *cur;
        acc_add_inplace(acc, cur);
        return;
    }
    next.fill(Fq::ZERO);
    next[delta..].copy_from_slice(&cur[..(D - delta)]);
    if delta < 27 {
        for src in (D - delta)..D {
            let coeff = cur[src];
            let exp = src + delta;
            next[exp - 54] -= coeff;
            next[exp - 27] -= coeff;
        }
    } else {
        for src in (D - delta)..(81 - delta) {
            let coeff = cur[src];
            let exp = src + delta;
            next[exp - 54] -= coeff;
            next[exp - 27] -= coeff;
        }
        for src in (81 - delta)..D {
            next[src + delta - 81] += cur[src];
        }
    }
    acc_add_inplace(acc, next);
}

#[inline(always)]
fn acc_add_inplace(acc: &mut [Fq; D], col: &[Fq; D]) {
    let mut r = 0usize;
    while r + 3 < D {
        acc[r] += col[r];
        acc[r + 1] += col[r + 1];
        acc[r + 2] += col[r + 2];
        acc[r + 3] += col[r + 3];
        r += 4;
    }
    while r < D {
        acc[r] += col[r];
        r += 1;
    }
}

#[inline(always)]
fn binary_mask_poly(mask: u64) -> RqEl {
    let mut coeffs = [Fq::ZERO; D];
    let mut bits = mask & ((1u64 << D) - 1);
    while bits != 0 {
        let idx = bits.trailing_zeros() as usize;
        coeffs[idx] = Fq::ONE;
        bits &= bits - 1;
    }
    RqEl(coeffs)
}

#[inline(always)]
fn acc_mul_add_inplace(acc: &mut [Fq; D], col: &[Fq; D], scalar: Fq) {
    // Fast paths for the common balanced-digit case (b ∈ {2,3} ⇒ scalar ∈ {-1,0,1}).
    //
    // NOTE: This is intentionally variable-time w.r.t. `scalar`. It is only used in the
    // seeded PP row-major commitment path, which is a prover-only performance hot loop.
    if scalar == Fq::ZERO {
        return;
    }
    if scalar == Fq::ONE {
        acc_add_inplace(acc, col);
        return;
    }
    let neg_one = Fq::ZERO - Fq::ONE;
    if scalar == neg_one {
        let mut r = 0usize;
        while r + 3 < D {
            acc[r] -= col[r];
            acc[r + 1] -= col[r + 1];
            acc[r + 2] -= col[r + 2];
            acc[r + 3] -= col[r + 3];
            r += 4;
        }
        while r < D {
            acc[r] -= col[r];
            r += 1;
        }
        return;
    }

    // Fallback: generic scalar multiply-add.
    // Unrolled to encourage LLVM auto-vectorization on platforms that support it.
    let mut r = 0usize;
    while r + 3 < D {
        acc[r] += col[r] * scalar;
        acc[r + 1] += col[r + 1] * scalar;
        acc[r + 2] += col[r + 2] * scalar;
        acc[r + 3] += col[r + 3] * scalar;
        r += 4;
    }
    while r < D {
        acc[r] += col[r] * scalar;
        r += 1;
    }
}

/// MUST: Setup(κ,m) → sample M ← R_q^{κ×m} uniformly (Def. 9).
pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, d: usize, kappa: usize, m: usize) -> AjtaiResult<PP<RqEl>> {
    // Ensure d matches the fixed ring dimension from neo-math
    if d != neo_math::ring::D {
        return Err(AjtaiError::InvalidDimensions(
            "d parameter must match ring dimension D".to_string(),
        ));
    }
    let mut rows = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        let mut row = Vec::with_capacity(m);
        for _ in 0..m {
            let coeffs: [Fq; D] = core::array::from_fn(|_| sample_uniform_fq(rng));
            row.push(cf_unmap(coeffs));
        }
        rows.push(row);
    }
    Ok(PP {
        kappa,
        m,
        d,
        m_rows: rows,
    })
}

/// Parallel version of [`setup`], primarily intended for large `m` where setup dominates runtime.
///
/// Implementation notes:
/// - Uses the provided `rng` only to generate one 32-byte seed per row.
/// - Each row is generated independently in parallel using `ChaCha8Rng` seeded from that seed.
/// - Output is deterministic given the input `rng` state, but will not match the sequential `setup`
///   output for the same RNG because the RNG stream is partitioned.
pub fn setup_par<R: RngCore + CryptoRng>(rng: &mut R, d: usize, kappa: usize, m: usize) -> AjtaiResult<PP<RqEl>> {
    // Ensure d matches the fixed ring dimension from neo-math
    if d != neo_math::ring::D {
        return Err(AjtaiError::InvalidDimensions(
            "d parameter must match ring dimension D".to_string(),
        ));
    }

    if m == 0 {
        return Ok(PP {
            kappa,
            m,
            d,
            m_rows: vec![Vec::new(); kappa],
        });
    }

    let mut row_seeds = vec![[0u8; 32]; kappa];
    for seed in row_seeds.iter_mut() {
        rng.fill_bytes(seed);
    }

    // Deterministic chunking: must NOT depend on runtime thread count, so a verifier can
    // re-derive the same PP from the same seed across environments.
    let chunk_size = core::cmp::min(m, 1 << 15).max(1024);
    let num_chunks = m.div_ceil(chunk_size);

    let mut rows = Vec::with_capacity(kappa);
    for row_seed in row_seeds {
        // Derive per-chunk seeds deterministically from the row seed.
        let mut seed_rng = ChaCha8Rng::from_seed(row_seed);
        let mut chunk_seeds = vec![[0u8; 32]; num_chunks];
        for seed in chunk_seeds.iter_mut() {
            seed_rng.fill_bytes(seed);
        }

        // Fill the row in place in parallel. This avoids extra copies of multi-GB buffers.
        let mut row = vec![RqEl::zero(); m];
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            row.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let mut chunk_rng = ChaCha8Rng::from_seed(chunk_seeds[chunk_idx]);
                    for el in chunk.iter_mut() {
                        let coeffs: [Fq; D] = core::array::from_fn(|_| sample_uniform_fq(&mut chunk_rng));
                        *el = cf_unmap(coeffs);
                    }
                });
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            for (chunk_idx, chunk) in row.chunks_mut(chunk_size).enumerate() {
                let mut chunk_rng = ChaCha8Rng::from_seed(chunk_seeds[chunk_idx]);
                for el in chunk.iter_mut() {
                    let coeffs: [Fq; D] = core::array::from_fn(|_| sample_uniform_fq(&mut chunk_rng));
                    *el = cf_unmap(coeffs);
                }
            }
        }

        rows.push(row);
    }

    Ok(PP {
        kappa,
        m,
        d,
        m_rows: rows,
    })
}

#[inline]
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

fn audit_commit_row_major_seeded_binary_cols_chunk(
    seed: [u8; 32],
    start: usize,
    end: usize,
    column_bits: &[u64],
) -> ([Fq; D], SeededBinaryColsCommitAudit) {
    const VALID_MASK: u64 = (1u64 << D) - 1;

    let total_started = Instant::now();
    let mut audit = SeededBinaryColsCommitAudit::default();
    let mut acc = [Fq::ZERO; D];
    let mut nxt = [Fq::ZERO; D];
    let mut rng = ChaCha8Rng::from_seed(seed);
    let mut batch_words = [0u64; SEEDED_RQ_BATCH * D];

    let mut col_idx = start;
    while col_idx < end {
        let batch = (end - col_idx).min(SEEDED_RQ_BATCH);
        let all_zero_batch = column_bits[col_idx..col_idx + batch]
            .iter()
            .all(|mask| (*mask & VALID_MASK) == 0);
        if all_zero_batch {
            audit.zero_columns += batch;
            let started = Instant::now();
            if advance_uniform_rq_coeff_validity_batch(&mut rng, batch) {
                audit.accepted_batches += 1;
                audit.fill_batch_ms += started.elapsed().as_secs_f64() * 1_000.0;
            } else {
                audit.fallback_batches += 1;
                for _ in 0..batch {
                    skip_uniform_rq_coeffs(&mut rng);
                }
                audit.fallback_sample_ms += started.elapsed().as_secs_f64() * 1_000.0;
            }
            col_idx += batch;
            continue;
        }
        let started = Instant::now();
        if fill_uniform_rq_coeff_words_batch(&mut rng, batch, &mut batch_words) {
            audit.accepted_batches += 1;
            audit.fill_batch_ms += started.elapsed().as_secs_f64() * 1_000.0;
            let mut dense_indices = Vec::with_capacity(batch);
            let mut sparse_indices = Vec::with_capacity(batch);
            for batch_idx in 0..batch {
                let mask = column_bits[col_idx + batch_idx] & VALID_MASK;
                if mask == 0 {
                    audit.zero_columns += 1;
                    continue;
                }
                let popcount = mask.count_ones();
                if popcount >= DENSE_BINARY_MASK_THRESHOLD {
                    audit.dense_columns += 1;
                    dense_indices.push(batch_idx);
                } else {
                    audit.sparse_columns += 1;
                    record_sparse_popcount(&mut audit, popcount);
                    sparse_indices.push(batch_idx);
                }
            }
            let started = Instant::now();
            for batch_idx in dense_indices {
                let mask = column_bits[col_idx + batch_idx] & VALID_MASK;
                let word_start = batch_idx * D;
                let word_end = word_start + D;
                let mut sampled = [Fq::ZERO; D];
                copy_uniform_rq_coeffs_from_words(&batch_words[word_start..word_end], &mut sampled);
                let product = RqEl(sampled).mul(&binary_mask_poly(mask));
                acc_add_inplace(&mut acc, &product.0);
            }
            audit.dense_total_ms += started.elapsed().as_secs_f64() * 1_000.0;
            let started = Instant::now();
            for batch_idx in sparse_indices {
                let mask = column_bits[col_idx + batch_idx] & VALID_MASK;
                let word_start = batch_idx * D;
                let word_end = word_start + D;
                let mut sampled = [Fq::ZERO; D];
                copy_uniform_rq_coeffs_from_words(&batch_words[word_start..word_end], &mut sampled);
                accumulate_binary_mask_sparse(&mut acc, &mut nxt, &mut sampled, mask);
            }
            audit.sparse_total_ms += started.elapsed().as_secs_f64() * 1_000.0;
        } else {
            audit.fallback_batches += 1;
            let mut dense_items = Vec::with_capacity(batch);
            let mut sparse_items = Vec::with_capacity(batch);
            audit.fallback_sample_ms += started.elapsed().as_secs_f64() * 1_000.0;
            let started = Instant::now();
            for batch_idx in 0..batch {
                let mask = column_bits[col_idx + batch_idx] & VALID_MASK;
                if mask == 0 {
                    audit.zero_columns += 1;
                    skip_uniform_rq_coeffs(&mut rng);
                    continue;
                }
                let sampled = sample_uniform_rq_coeffs(&mut rng);
                let popcount = mask.count_ones();
                if popcount >= DENSE_BINARY_MASK_THRESHOLD {
                    audit.dense_columns += 1;
                    dense_items.push((sampled, mask));
                } else {
                    audit.sparse_columns += 1;
                    record_sparse_popcount(&mut audit, popcount);
                    sparse_items.push((sampled, mask));
                }
            }
            audit.fallback_sample_ms += started.elapsed().as_secs_f64() * 1_000.0;
            let started = Instant::now();
            for (sampled, mask) in dense_items {
                let product = RqEl(sampled).mul(&binary_mask_poly(mask));
                acc_add_inplace(&mut acc, &product.0);
            }
            audit.dense_total_ms += started.elapsed().as_secs_f64() * 1_000.0;
            let started = Instant::now();
            for (mut sampled, mask) in sparse_items {
                accumulate_binary_mask_sparse(&mut acc, &mut nxt, &mut sampled, mask);
            }
            audit.sparse_total_ms += started.elapsed().as_secs_f64() * 1_000.0;
        }
        col_idx += batch;
    }

    audit.total_ms = total_started.elapsed().as_secs_f64() * 1_000.0;
    (acc, audit)
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

#[allow(non_snake_case)]
#[doc(hidden)]
pub fn audit_commit_row_major_seeded_binary_cols_with_chunk_seeds(
    d: usize,
    kappa: usize,
    m: usize,
    column_bits: &[u64],
    chunk_size: usize,
    chunk_seeds_by_row: &[Vec<[u8; 32]>],
) -> (Commitment, SeededBinaryColsCommitAudit) {
    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(column_bits.len(), m, "binary column image must have length m");
    assert_eq!(chunk_seeds_by_row.len(), kappa, "chunk seed rows must match kappa");

    let total_started = Instant::now();
    let mut C = Commitment::zeros(d, kappa);
    let mut audit = SeededBinaryColsCommitAudit::default();
    if m == 0 {
        return (C, audit);
    }

    for i in 0..kappa {
        let chunk_seeds = &chunk_seeds_by_row[i];
        let mut acc = [Fq::ZERO; D];
        for (chunk_idx, seed) in chunk_seeds.iter().copied().enumerate() {
            let start = chunk_idx * chunk_size;
            let end = core::cmp::min(m, start + chunk_size);
            let (chunk_acc, chunk_audit) =
                audit_commit_row_major_seeded_binary_cols_chunk(seed, start, end, column_bits);
            acc_add_inplace(&mut acc, &chunk_acc);
            audit.accumulate(chunk_audit);
        }
        C.col_mut(i).copy_from_slice(&acc);
    }
    audit.total_ms = total_started.elapsed().as_secs_f64() * 1_000.0;
    (C, audit)
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

// Variable-time optimization removed for security and simplicity

/// MUST: Commit(pp, Z) = cf(M · cf^{-1}(Z)) as c ∈ F_q^{d×κ}.  S-homomorphic over S by construction.
/// Uses constant-time dense computation for all inputs (audit-ready).
/// Returns error if Z dimensions don't match expected d×m.
#[allow(non_snake_case)]
pub fn try_commit(pp: &PP<RqEl>, Z: &[Fq]) -> AjtaiResult<Commitment> {
    // Z is d×m (column-major by (col*d + row)), output c is d×kappa (column-major)
    let d = pp.d;
    let m = pp.m;
    if Z.len() != d * m {
        return Err(AjtaiError::SizeMismatch {
            expected: d * m,
            actual: Z.len(),
        });
    }

    // 🚀 PERFORMANCE OPTIMIZATION: Use precomputed rotations for large m
    // For small m, masked CT is faster due to lower setup cost
    // For large m, precomputed CT amortizes the rotation computation cost
    const PRECOMP_THRESHOLD: usize = 256; // Threshold tuned for D=54: precomp pays off when m*D > 16k

    if m >= PRECOMP_THRESHOLD {
        Ok(commit_precomp_ct(pp, Z))
    } else {
        Ok(commit_masked_ct(pp, Z))
    }
}

/// Convenience wrapper that panics on dimension mismatch (for tests and controlled environments).
#[allow(non_snake_case)]
pub fn commit(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    try_commit(pp, Z).expect("commit: Z dimensions must match d×m")
}

/// Commit to a **row-major** `Mat<Fq>` without materializing a full column-major buffer.
///
/// This is equivalent to:
/// 1) transposing `Z` (row-major) into a column-major `Vec<Fq>`, then
/// 2) calling [`commit`].
///
/// It exists to avoid a multi-hundred-MB temporary allocation in prover hot paths.
#[allow(non_snake_case)]
pub fn try_commit_row_major(pp: &PP<RqEl>, Z: &Mat<Fq>) -> AjtaiResult<Commitment> {
    let d = pp.d;
    let m = pp.m;
    if Z.rows() != d || Z.cols() != m {
        return Err(AjtaiError::InvalidDimensions(format!(
            "Z must have shape d×m = {}×{} (got {}×{})",
            d,
            m,
            Z.rows(),
            Z.cols()
        )));
    }

    // Mirror `try_commit`'s dispatch choice, but operate directly on the row-major `Mat`.
    const PRECOMP_THRESHOLD: usize = 256;
    Ok(if m >= PRECOMP_THRESHOLD {
        commit_precomp_ct_row_major(pp, Z)
    } else {
        commit_masked_ct_row_major(pp, Z)
    })
}

/// Convenience wrapper that panics on dimension mismatch (for tests and controlled environments).
#[allow(non_snake_case)]
pub fn commit_row_major(pp: &PP<RqEl>, Z: &Mat<Fq>) -> Commitment {
    try_commit_row_major(pp, Z).expect("commit_row_major: Z dimensions must match d×m")
}

/// MUST: Verify opening by recomputing commitment (binding implies uniqueness).
#[must_use = "Ajtai verification must be checked; ignoring this result is a security bug"]
#[allow(non_snake_case)]
pub fn verify_open(pp: &PP<RqEl>, c: &Commitment, Z: &[Fq]) -> bool {
    &commit(pp, Z) == c
}

/// MUST: Verify split opening: c == Σ b^{i-1} c_i and Z == Σ b^{i-1} Z_i, with ||Z_i||_∞<b (range assertions done by caller).
#[must_use = "Ajtai verification must be checked; ignoring this result is a security bug"]
#[allow(non_snake_case)]
pub fn verify_split_open(pp: &PP<RqEl>, c: &Commitment, b: u32, c_is: &[Commitment], Z_is: &[Vec<Fq>]) -> bool {
    let k = c_is.len();
    if k != Z_is.len() {
        return false;
    }
    // Check shapes
    for ci in c_is {
        if ci.d != c.d || ci.kappa != c.kappa {
            return false;
        }
    }
    // Recompose commitment
    let mut acc = Commitment::zeros(c.d, c.kappa);
    let mut pow = Fq::ONE;
    let b_f = Fq::from_u64(b as u64);
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        for (a, &x) in acc.data.iter_mut().zip(&c_is[i].data) {
            *a += x * pow;
        }
        pow *= b_f;
    }
    if &acc != c {
        return false;
    }
    // Recompose Z and check commit again
    let d = pp.d;
    let m = pp.m;
    let mut Z = vec![Fq::ZERO; d * m];
    let mut pow = Fq::ONE;
    for Zi in Z_is {
        if Zi.len() != d * m {
            return false;
        }
        for (a, &x) in Z.iter_mut().zip(Zi) {
            *a += x * pow;
        }
        pow *= b_f;
    }
    &commit(pp, &Z) == c
}

/// S-homomorphism: ρ·L(Z) = L(ρ·Z).  We expose helpers for left-multiplying commitments.
pub fn s_mul_add_from_rot_col(acc: &mut Commitment, first_rot_col: &[Fq; D], c: &Commitment) {
    let d = c.d;
    let kappa = c.kappa;
    debug_assert_eq!(d, D, "Ajtai commitment columns must have D rows");
    debug_assert_eq!(acc.d, d);
    debug_assert_eq!(acc.kappa, kappa);

    let (acc_cols, acc_rem) = acc.data.as_chunks_mut::<D>();
    let (c_cols, c_rem) = c.data.as_chunks::<D>();
    debug_assert!(acc_rem.is_empty(), "accumulator commitment columns must be D-wide");
    debug_assert!(c_rem.is_empty(), "input commitment columns must be D-wide");
    debug_assert_eq!(acc_cols.len(), kappa);
    debug_assert_eq!(c_cols.len(), kappa);

    let mut rot_col = *first_rot_col;
    let mut nxt = [Fq::ZERO; D];
    for t in 0..D {
        for (dst, src) in acc_cols.iter_mut().zip(c_cols.iter()) {
            acc_mul_add_inplace(dst, &rot_col, src[t]);
        }
        rot_step(&rot_col, &mut nxt);
        rot_col = nxt;
    }
}

/// S-homomorphism: ρ·L(Z) = L(ρ·Z).  We expose helpers for left-multiplying commitments.
pub fn s_mul_add(acc: &mut Commitment, rho_ring: &RqEl, c: &Commitment) {
    let rot_col = cf(*rho_ring);
    s_mul_add_from_rot_col(acc, &rot_col, c);
}

/// Add a field-scalar multiple of a commitment into an accumulator.
///
/// This is the constant-ring-element `S`-action specialized to the common verifier-side case
/// where the multiplier is just a base-field scalar instead of a full rotation element.
pub fn scale_commitment_add_inplace(acc: &mut Commitment, scalar: Fq, c: &Commitment) {
    debug_assert_eq!(acc.d, c.d);
    debug_assert_eq!(acc.kappa, c.kappa);

    if scalar == Fq::ZERO {
        return;
    }
    if scalar == Fq::ONE {
        acc.add_inplace(c);
        return;
    }
    let neg_one = Fq::ZERO - Fq::ONE;
    if scalar == neg_one {
        for (dst, src) in acc.data.iter_mut().zip(c.data.iter()) {
            *dst -= *src;
        }
        return;
    }

    for (dst, src) in acc.data.iter_mut().zip(c.data.iter()) {
        *dst += *src * scalar;
    }
}

pub fn scale_commitment(scalar: Fq, c: &Commitment) -> Commitment {
    let mut out = Commitment::zeros(c.d, c.kappa);
    scale_commitment_add_inplace(&mut out, scalar, c);
    out
}

pub fn s_mul(rho_ring: &RqEl, c: &Commitment) -> Commitment {
    let d = c.d;
    let kappa = c.kappa;
    let mut out = Commitment::zeros(d, kappa);
    s_mul_add(&mut out, rho_ring, c);
    out
}

pub fn s_lincomb(rhos: &[RqEl], cs: &[Commitment]) -> AjtaiResult<Commitment> {
    if rhos.is_empty() {
        return Err(AjtaiError::EmptyInput);
    }
    if rhos.len() != cs.len() {
        return Err(AjtaiError::SizeMismatch {
            expected: rhos.len(),
            actual: cs.len(),
        });
    }
    if cs.is_empty() {
        return Err(AjtaiError::EmptyInput);
    }

    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    for (rho, c) in rhos.iter().zip(cs) {
        s_mul_add(&mut acc, rho, c);
    }
    Ok(acc)
}

/// Constant-time masked columns accumulation (streaming).
///
/// c = cf(M · cf^{-1}(Z)) computed as:
///   for i in 0..kappa, j in 0..m:
///     col <- cf(a_ij)       // column 0 of rot(a_ij)
///     for t in 0..d-1:
///       acc += Z[j*d + t] * col
///       col <- next column via rot_step()
///
/// **Constant-Time Guarantees:**
/// - Fixed iteration counts (no secret-dependent branching)
/// - No secret-dependent memory accesses
/// - Identical execution flow regardless of Z values (sparsity, magnitude)
/// - Assumes underlying field arithmetic is constant-time (true for Goldilocks)
///
/// This implements the identity cf(a·b) = rot(a)·cf(b) = Σ(t=0 to d-1) b_t · col_t(rot(a))
#[allow(non_snake_case)]
pub fn commit_masked_ct(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d;
    let m = pp.m;
    let kappa = pp.kappa;

    // CRITICAL SECURITY: Runtime dimension checks to prevent binding bugs
    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(Z.len(), d * m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);

    // For each Ajtai row i and message column j
    for i in 0..kappa {
        let acc_i = C.col_mut(i);
        for j in 0..m {
            // Start from col_0 = cf(a_ij)
            let mut col = cf(pp.m_rows[i][j]);
            let mut nxt = [Fq::ZERO; D];

            // Loop over all base-digits t (constant-time)
            let base = j * d;
            for t in 0..d {
                let mask = Z[base + t]; // any Fq digit (0, ±1, small, or general)
                                        // acc += mask * col   (branch-free masked add)
                for r in 0..d {
                    // single FMA-like op on the field
                    acc_i[r] += col[r] * mask;
                }
                // Advance to the next rotation column in O(d)
                rot_step(&col, &mut nxt);
                core::mem::swap(&mut col, &mut nxt); // Cheaper than copying [Fq; D]
            }
        }
    }
    C
}

/// Row-major variant of [`commit_masked_ct`].
#[allow(non_snake_case)]
fn commit_masked_ct_row_major(pp: &PP<RqEl>, Z: &Mat<Fq>) -> Commitment {
    let d = pp.d;
    let m = pp.m;
    let kappa = pp.kappa;

    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(Z.rows(), d, "Z must be d×m");
    assert_eq!(Z.cols(), m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);

    for i in 0..kappa {
        let acc_i = C.col_mut(i);
        for j in 0..m {
            let mut col = cf(pp.m_rows[i][j]);
            let mut nxt = [Fq::ZERO; D];

            for t in 0..d {
                let mask = Z.row(t)[j];
                for r in 0..d {
                    acc_i[r] += col[r] * mask;
                }
                rot_step(&col, &mut nxt);
                core::mem::swap(&mut col, &mut nxt);
            }
        }
    }
    C
}

/// Fill `cols` with the d rotation columns of rot(a): cols[t] = cf(a * X^t).
///
/// This is an internal building block for high-performance folding code paths that need
/// to batch multiple Ajtai commitments without materializing full digit matrices.
#[doc(hidden)]
#[inline]
pub fn precompute_rot_columns(a: RqEl, cols: &mut [[Fq; D]]) {
    let mut col = cf(a);
    let mut nxt = [Fq::ZERO; D];
    for t in 0..D {
        cols[t] = col;
        rot_step(&col, &mut nxt);
        core::mem::swap(&mut col, &mut nxt); // Avoid copying 54 elements
    }
}

/// Constant-time commit using precomputed rotation columns per (i,j).
///
/// Space/time trade: uses a stack-allocated `[[Fq; D]; D]` scratch per (i,j) to
/// remove per-step rot_step(), keeping the same constant-time masked adds.
///
/// **Constant-Time Guarantees:**
/// - Fixed iteration counts (no secret-dependent branching)  
/// - No secret-dependent memory accesses
/// - Identical execution flow regardless of Z values (sparsity, magnitude)
/// - Assumes underlying field arithmetic is constant-time (true for Goldilocks)
///
/// This implements the same identity cf(a·b) = rot(a)·cf(b) = Σ(t=0 to d-1) b_t · col_t(rot(a))
/// but precomputes all rotation columns once per (i,j) pair for better cache locality.
#[allow(non_snake_case)]
pub fn commit_precomp_ct(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d;
    let m = pp.m;
    let kappa = pp.kappa;

    // CRITICAL SECURITY: Runtime dimension checks to prevent binding bugs
    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(Z.len(), d * m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);

    if m == 0 {
        return C;
    }

    struct Acc {
        acc: [Fq; D],
        cols: Box<[[Fq; D]]>,
    }

    impl Acc {
        #[inline]
        fn new() -> Self {
            Self {
                acc: [Fq::ZERO; D],
                cols: vec![[Fq::ZERO; D]; D].into_boxed_slice(),
            }
        }
    }

    for i in 0..kappa {
        let row = &pp.m_rows[i];
        debug_assert_eq!(row.len(), m);

        let acc = {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                row.par_iter()
                    .zip(Z.par_chunks_exact(d))
                    .fold(Acc::new, |mut st, (&a_ij, z_col)| {
                        precompute_rot_columns(a_ij, &mut st.cols);
                        // Constant schedule: always loop over all t
                        for t in 0..d {
                            let mask = z_col[t];
                            let col_t = &st.cols[t];
                            for r in 0..d {
                                st.acc[r] += col_t[r] * mask;
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
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                let mut st = Acc::new();
                for (&a_ij, z_col) in row.iter().zip(Z.chunks_exact(d)) {
                    precompute_rot_columns(a_ij, &mut st.cols);
                    for t in 0..d {
                        let mask = z_col[t];
                        let col_t = &st.cols[t];
                        for r in 0..d {
                            st.acc[r] += col_t[r] * mask;
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

/// Row-major variant of [`commit_precomp_ct`].
#[allow(non_snake_case)]
fn commit_precomp_ct_row_major(pp: &PP<RqEl>, Z: &Mat<Fq>) -> Commitment {
    let d = pp.d;
    let m = pp.m;
    let kappa = pp.kappa;

    assert_eq!(d, D, "Ajtai dimension mismatch: runtime d != compile-time D");
    assert_eq!(Z.rows(), d, "Z must be d×m");
    assert_eq!(Z.cols(), m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);
    if m == 0 {
        return C;
    }

    struct Acc {
        acc: [Fq; D],
        cols: Box<[[Fq; D]]>,
    }

    impl Acc {
        #[inline]
        fn new() -> Self {
            Self {
                acc: [Fq::ZERO; D],
                cols: vec![[Fq::ZERO; D]; D].into_boxed_slice(),
            }
        }
    }

    // Grab row slices once; avoids repeated bounds checks in the inner column loop.
    let z_rows: Vec<&[Fq]> = (0..d).map(|r| Z.row(r)).collect();

    for i in 0..kappa {
        let row = &pp.m_rows[i];
        debug_assert_eq!(row.len(), m);

        let acc = {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                row.par_iter()
                    .enumerate()
                    .fold(Acc::new, |mut st, (j, &a_ij)| {
                        precompute_rot_columns(a_ij, &mut st.cols);
                        for t in 0..d {
                            let mask = z_rows[t][j];
                            let col_t = &st.cols[t];
                            for r in 0..d {
                                st.acc[r] += col_t[r] * mask;
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
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                let mut st = Acc::new();
                for (j, &a_ij) in row.iter().enumerate() {
                    precompute_rot_columns(a_ij, &mut st.cols);
                    for t in 0..d {
                        let mask = z_rows[t][j];
                        let col_t = &st.cols[t];
                        for r in 0..d {
                            st.acc[r] += col_t[r] * mask;
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
