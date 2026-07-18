use crate::{commit as ajtai_commit, setup_par, AjtaiError, Commitment, PP};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_math::ring::{Rq as RqEl, D};
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::{OnceLock, RwLock};

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

type Key = (usize, usize); // (d, m)
type PPRef = Arc<PP<RqEl>>;
const SIGNED_UNIT_INNER_PAR_COL_CHUNK: usize = 512;
const SIGNED_UNIT_INNER_PAR_BASE_COL_LIMIT: usize = 1 << 17;
const SIGNED_UNIT_MASK_CACHE_COL_LIMIT: usize = 1 << 15;
const SIGNED_UNIT_GRID_PAR_MAX_TASKS: usize = 1024;

struct SignedUnitColumnPlan {
    active_union: u64,
    pos_commit_masks: [u64; D],
    neg_commit_masks: [u64; D],
}

#[derive(Clone)]
struct RegistryEntry {
    kappa: usize,
    /// If present, this entry can be (re)loaded on demand via `setup_par` with a fixed seed.
    seed: Option<[u8; 32]>,
    /// If present, PP is currently materialized in memory.
    pp: Option<PPRef>,
}

static AJTAI_PP_REGISTRY: OnceLock<RwLock<HashMap<Key, RegistryEntry>>> = OnceLock::new();

fn registry() -> &'static RwLock<HashMap<Key, RegistryEntry>> {
    AJTAI_PP_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Materialize the exact public matrix used by the seeded commitment path.
///
/// Accelerator backends use this to upload the canonical map once instead
/// of duplicating the seeded-parameter derivation outside `neo-ajtai`.
#[doc(hidden)]
pub fn materialize_seeded_pp(seed: [u8; 32], d: usize, kappa: usize, m: usize) -> Result<PP<RqEl>, AjtaiError> {
    let mut rng = ChaCha8Rng::from_seed(seed);
    setup_par(&mut rng, d, kappa, m)
}

/// Initialize the global Ajtai PP once (call this right after setup()).
pub fn set_global_pp(pp: PP<RqEl>) -> Result<(), AjtaiError> {
    let key = (pp.d, pp.m);
    let mut w = registry()
        .write()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    if let Some(existing) = w.get(&key) {
        if existing.seed.is_some() {
            return Err(AjtaiError::InvalidInput(format!(
                "Ajtai PP seed is already registered for (d,m)=({},{}) so `set_global_pp` is disallowed",
                pp.d, pp.m
            )));
        }
        // Idempotent: keep the existing PP to avoid accidentally changing commitments mid-process.
        return Ok(());
    }
    w.insert(
        key,
        RegistryEntry {
            kappa: pp.kappa,
            seed: None,
            pp: Some(Arc::new(pp)),
        },
    );
    Ok(())
}

/// Register a deterministic seed for (d,kappa,m) and *optionally* keep PP unloaded until first use.
///
/// This enables `unload_global_pp_for_dims()` to free multi-GB PP allocations during
/// prover phases that do not require commitments (e.g. sum-check table building).
pub fn set_global_pp_seeded(d: usize, kappa: usize, m: usize, seed: [u8; 32]) -> Result<(), AjtaiError> {
    let key = (d, m);
    let mut w = registry()
        .write()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    if let Some(entry) = w.get_mut(&key) {
        // If a PP is already materialized, we must not allow changing/adding a seed unless it
        // exactly matches the existing seeded configuration. Otherwise, later unload/reload would
        // silently change commitments and break proofs.
        if let Some(pp) = entry.pp.as_ref() {
            match entry.seed {
                Some(existing_seed) if existing_seed == seed => {
                    if entry.kappa != kappa || pp.kappa != kappa {
                        return Err(AjtaiError::InvalidInput(format!(
                            "Ajtai seeded PP kappa mismatch for (d,m)=({},{}) (existing κ={}, requested κ={})",
                            d, m, entry.kappa, kappa
                        )));
                    }
                    return Ok(());
                }
                Some(_) => {
                    return Err(AjtaiError::InvalidInput(format!(
                        "Ajtai PP seed mismatch for already-loaded (d,m)=({},{}); refusing to overwrite",
                        d, m
                    )));
                }
                None => {
                    return Err(AjtaiError::InvalidInput(format!(
                        "Ajtai PP for (d,m)=({},{}) is already loaded without a seed; cannot register a seed",
                        d, m
                    )));
                }
            }
        }

        if let Some(existing_seed) = entry.seed {
            if existing_seed != seed {
                return Err(AjtaiError::InvalidInput(format!(
                    "Ajtai PP seed already registered for (d,m)=({},{}) and does not match the provided seed",
                    d, m
                )));
            }
            if entry.kappa != kappa {
                return Err(AjtaiError::InvalidInput(format!(
                    "Ajtai seeded PP kappa mismatch for (d,m)=({},{}) (existing κ={}, requested κ={})",
                    d, m, entry.kappa, kappa
                )));
            }
            return Ok(());
        }

        entry.kappa = kappa;
        entry.seed = Some(seed);
        return Ok(());
    }

    w.insert(
        key,
        RegistryEntry {
            kappa,
            seed: Some(seed),
            pp: None,
        },
    );
    Ok(())
}

/// True if the PP for (d,m) can be reloaded (i.e. a seed is registered).
pub fn has_seed_for_dims(d: usize, m: usize) -> bool {
    registry()
        .read()
        .ok()
        .map(|r| r.get(&(d, m)).map(|e| e.seed.is_some()).unwrap_or(false))
        .unwrap_or(false)
}

/// Get `(kappa, seed)` for a seeded PP entry.
///
/// Returns an error if the entry does not exist or is not seeded.
pub fn get_global_pp_seeded_params_for_dims(d: usize, m: usize) -> Result<(usize, [u8; 32]), AjtaiError> {
    let r = registry()
        .read()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    let entry = r
        .get(&(d, m))
        .ok_or_else(|| AjtaiError::InvalidInput("Ajtai PP not initialized for requested (d,m)".to_string()))?;
    let seed = entry
        .seed
        .ok_or_else(|| AjtaiError::InvalidInput("Ajtai PP seed not registered for requested (d,m)".to_string()))?;
    Ok((entry.kappa, seed))
}

/// Drop the materialized PP for (d,m) from memory, keeping any registered seed.
///
/// Returns `Ok(true)` if PP was present and was unloaded.
pub fn unload_global_pp_for_dims(d: usize, m: usize) -> Result<bool, AjtaiError> {
    let key = (d, m);
    let mut w = registry()
        .write()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    let Some(entry) = w.get_mut(&key) else {
        return Ok(false);
    };
    if entry.seed.is_none() {
        return Err(AjtaiError::InvalidInput(format!(
            "Ajtai PP for (d,m)=({},{}) is not seeded; refusing to unload because it cannot be reloaded",
            d, m
        )));
    }
    let had = entry.pp.is_some();
    entry.pp = None;
    Ok(had)
}

/// If the PP for (d,m) is already materialized, return it without loading.
pub fn try_get_loaded_global_pp_for_dims(d: usize, m: usize) -> Option<PPRef> {
    registry()
        .read()
        .ok()
        .and_then(|r| r.get(&(d, m)).and_then(|e| e.pp.as_ref().cloned()))
}

fn get_or_load_global_pp_for_dims(d: usize, m: usize) -> Result<PPRef, AjtaiError> {
    // Fast path: already loaded.
    if let Ok(r) = registry().read() {
        if let Some(entry) = r.get(&(d, m)) {
            if let Some(pp) = entry.pp.as_ref() {
                return Ok(pp.clone());
            }
        }
    }

    // Slow path: load from seed if available.
    let (seed, kappa) = {
        let r = registry()
            .read()
            .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
        let entry = r
            .get(&(d, m))
            .ok_or_else(|| AjtaiError::InvalidInput("Ajtai PP not initialized for requested (d,m)".to_string()))?;
        let seed = entry
            .seed
            .ok_or_else(|| AjtaiError::InvalidInput("Ajtai PP seed not registered for requested (d,m)".to_string()))?;
        (seed, entry.kappa)
    };

    let mut rng = ChaCha8Rng::from_seed(seed);
    let pp = setup_par(&mut rng, d, kappa, m)?;
    let pp = Arc::new(pp);

    let mut w = registry()
        .write()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    let entry = w.entry((d, m)).or_insert_with(|| RegistryEntry {
        kappa,
        seed: Some(seed),
        pp: None,
    });
    entry.kappa = kappa;
    entry.seed = Some(seed);
    entry.pp = Some(pp.clone());
    Ok(pp)
}

fn pack_binary_column_bits(z: &Mat<Fq>, d: usize, m: usize) -> Option<Vec<u64>> {
    if d > u64::BITS as usize || z.rows() != d || z.cols() != m {
        return None;
    }

    let mut column_bits = vec![0u64; m];
    for row in 0..d {
        let bit = 1u64 << row;
        for col in 0..m {
            let value = z[(row, col)];
            if value == Fq::ZERO {
                continue;
            }
            if value != Fq::ONE {
                return None;
            }
            column_bits[col] |= bit;
        }
    }
    Some(column_bits)
}

fn pack_signed_unit_column_bits(z: &Mat<Fq>, d: usize, m: usize) -> Option<(Vec<u64>, Vec<u64>)> {
    if d > u64::BITS as usize || z.rows() != d || z.cols() != m {
        return None;
    }

    let neg_one = Fq::ZERO - Fq::ONE;
    let mut pos_bits = vec![0u64; m];
    let mut neg_bits = vec![0u64; m];
    for row in 0..d {
        let bit = 1u64 << row;
        for col in 0..m {
            let value = z[(row, col)];
            if value == Fq::ZERO {
                continue;
            }
            if value == Fq::ONE {
                pos_bits[col] |= bit;
                continue;
            }
            if value == neg_one {
                neg_bits[col] |= bit;
                continue;
            }
            return None;
        }
    }
    Some((pos_bits, neg_bits))
}

fn commit_packed_signed_unit_column_bits(
    d: usize,
    kappa: usize,
    m: usize,
    pos_bits: &[u64],
    neg_bits: &[u64],
    chunk_size: usize,
    chunk_seeds_by_row: &[Vec<[u8; 32]>],
) -> Commitment {
    let mut out = ajtai_commit::commit_row_major_seeded_binary_cols_with_chunk_seeds(
        d,
        kappa,
        m,
        pos_bits,
        chunk_size,
        chunk_seeds_by_row,
    );
    if neg_bits.iter().any(|&mask| mask != 0) {
        let neg = ajtai_commit::commit_row_major_seeded_binary_cols_with_chunk_seeds(
            d,
            kappa,
            m,
            neg_bits,
            chunk_size,
            chunk_seeds_by_row,
        );
        for (dst, src) in out.data.iter_mut().zip(neg.data.iter()) {
            *dst -= *src;
        }
    }
    out
}

#[inline(always)]
fn acc_add_signed_unit(acc: &mut [Fq; D], col: &[Fq; D], subtract: bool) {
    let mut idx = 0usize;
    if subtract {
        while idx + 3 < D {
            acc[idx] -= col[idx];
            acc[idx + 1] -= col[idx + 1];
            acc[idx + 2] -= col[idx + 2];
            acc[idx + 3] -= col[idx + 3];
            idx += 4;
        }
        while idx < D {
            acc[idx] -= col[idx];
            idx += 1;
        }
    } else {
        while idx + 3 < D {
            acc[idx] += col[idx];
            acc[idx + 1] += col[idx + 1];
            acc[idx + 2] += col[idx + 2];
            acc[idx + 3] += col[idx + 3];
            idx += 4;
        }
        while idx < D {
            acc[idx] += col[idx];
            idx += 1;
        }
    }
}

#[inline(always)]
fn advance_rot_col(rot_col: &mut [Fq; D], delta: usize) {
    match delta {
        0 => {}
        1 => rot_col_step_phi_81(rot_col),
        _ => *rot_col = mul_coeffs_by_monomial(rot_col, delta),
    }
}

#[inline(always)]
fn rot_col_step_phi_81(col: &mut [Fq; D]) {
    let last = col[D - 1];
    for idx in (1..D).rev() {
        col[idx] = col[idx - 1];
    }
    col[0] = Fq::ZERO - last;
    col[D / 2] -= last;
}

#[inline(always)]
fn mul_coeffs_by_monomial(input: &[Fq; D], j: usize) -> [Fq; D] {
    debug_assert!(j < D, "signed-unit column rotation must be below D");
    if j == 0 {
        return *input;
    }

    let mut out = [Fq::ZERO; D];
    let first_reduced = D - j;
    let first_wrap = (D + D / 2).saturating_sub(j).min(D);

    for i in 0..first_reduced {
        out[i + j] = input[i];
    }

    for i in first_reduced..first_wrap {
        let reduced = i + j - D;
        out[reduced] -= input[i];
        out[reduced + D / 2] -= input[i];
    }

    // In R_q = F_q[X]/(X^54 + X^27 + 1), the second reduction range
    // collapses to a single positive wrapped monomial.
    for i in first_wrap..D {
        out[i + j - D - D / 2] += input[i];
    }

    out
}

fn build_signed_unit_column_plans(m: usize, packed: &[(Vec<u64>, Vec<u64>)]) -> Option<Vec<SignedUnitColumnPlan>> {
    if packed.len() > u64::BITS as usize {
        return None;
    }

    let valid_mask = (1u64 << D) - 1;
    let mut plans = Vec::with_capacity(m);
    for col_idx in 0..m {
        let mut active_union = 0u64;
        let mut pos_commit_masks = [0u64; D];
        let mut neg_commit_masks = [0u64; D];

        for (commit_idx, (pos_bits, neg_bits)) in packed.iter().enumerate() {
            let commit_bit = 1u64 << commit_idx;
            let pos = pos_bits[col_idx] & valid_mask;
            let neg = neg_bits[col_idx] & valid_mask;
            active_union |= pos | neg;

            let mut mask = pos;
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                pos_commit_masks[bit] |= commit_bit;
                mask &= mask - 1;
            }

            let mut mask = neg;
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                neg_commit_masks[bit] |= commit_bit;
                mask &= mask - 1;
            }
        }

        plans.push(SignedUnitColumnPlan {
            active_union,
            pos_commit_masks,
            neg_commit_masks,
        });
    }
    Some(plans)
}

#[inline(always)]
fn accumulate_signed_unit_column_plan(accs: &mut [[Fq; D]], base_col: &[Fq; D], plan: &SignedUnitColumnPlan) {
    if plan.active_union == 0 {
        return;
    }

    let mut active_union = plan.active_union;
    let mut rot_col = *base_col;
    let mut rot_pos = 0usize;
    while active_union != 0 {
        let next_pos = active_union.trailing_zeros() as usize;
        advance_rot_col(&mut rot_col, next_pos - rot_pos);

        let mut pos_mask = plan.pos_commit_masks[next_pos];
        while pos_mask != 0 {
            let commit_idx = pos_mask.trailing_zeros() as usize;
            acc_add_signed_unit(&mut accs[commit_idx], &rot_col, false);
            pos_mask &= pos_mask - 1;
        }

        let mut neg_mask = plan.neg_commit_masks[next_pos];
        while neg_mask != 0 {
            let commit_idx = neg_mask.trailing_zeros() as usize;
            acc_add_signed_unit(&mut accs[commit_idx], &rot_col, true);
            neg_mask &= neg_mask - 1;
        }
        rot_pos = next_pos;
        active_union &= active_union - 1;
    }
}

#[inline(always)]
fn accumulate_signed_unit_masks_many_for_col(
    accs: &mut [[Fq; D]],
    base_col: &[Fq; D],
    col_idx: usize,
    packed: &[(Vec<u64>, Vec<u64>)],
) {
    let valid_mask = (1u64 << D) - 1;
    if packed.len() > u64::BITS as usize {
        accumulate_signed_unit_masks_many_for_col_slow(accs, base_col, col_idx, packed, valid_mask);
        return;
    }

    let mut active_union = 0u64;
    let mut pos_commit_masks = [0u64; D];
    let mut neg_commit_masks = [0u64; D];

    for (commit_idx, (pos_bits, neg_bits)) in packed.iter().enumerate() {
        let commit_bit = 1u64 << commit_idx;
        let pos = pos_bits[col_idx] & valid_mask;
        let neg = neg_bits[col_idx] & valid_mask;
        active_union |= pos | neg;

        let mut mask = pos;
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            pos_commit_masks[bit] |= commit_bit;
            mask &= mask - 1;
        }

        let mut mask = neg;
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            neg_commit_masks[bit] |= commit_bit;
            mask &= mask - 1;
        }
    }
    if active_union == 0 {
        return;
    }

    let mut rot_col = *base_col;
    let mut rot_pos = 0usize;
    while active_union != 0 {
        let next_pos = active_union.trailing_zeros() as usize;
        advance_rot_col(&mut rot_col, next_pos - rot_pos);

        let mut pos_mask = pos_commit_masks[next_pos];
        while pos_mask != 0 {
            let commit_idx = pos_mask.trailing_zeros() as usize;
            acc_add_signed_unit(&mut accs[commit_idx], &rot_col, false);
            pos_mask &= pos_mask - 1;
        }

        let mut neg_mask = neg_commit_masks[next_pos];
        while neg_mask != 0 {
            let commit_idx = neg_mask.trailing_zeros() as usize;
            acc_add_signed_unit(&mut accs[commit_idx], &rot_col, true);
            neg_mask &= neg_mask - 1;
        }
        rot_pos = next_pos;
        active_union &= active_union - 1;
    }
}

#[inline(always)]
fn accumulate_signed_unit_masks_many_for_col_slow(
    accs: &mut [[Fq; D]],
    base_col: &[Fq; D],
    col_idx: usize,
    packed: &[(Vec<u64>, Vec<u64>)],
    valid_mask: u64,
) {
    let mut active_union = 0u64;
    for (pos_bits, neg_bits) in packed {
        active_union |= (pos_bits[col_idx] | neg_bits[col_idx]) & valid_mask;
    }
    if active_union == 0 {
        return;
    }

    let mut rot_col = *base_col;
    let mut rot_pos = 0usize;
    while active_union != 0 {
        let next_pos = active_union.trailing_zeros() as usize;
        advance_rot_col(&mut rot_col, next_pos - rot_pos);
        let bit = 1u64 << next_pos;
        for (acc, (pos_bits, neg_bits)) in accs.iter_mut().zip(packed.iter()) {
            if (pos_bits[col_idx] & bit) != 0 {
                acc_add_signed_unit(acc, &rot_col, false);
            } else if (neg_bits[col_idx] & bit) != 0 {
                acc_add_signed_unit(acc, &rot_col, true);
            }
        }
        rot_pos = next_pos;
        active_union &= active_union - 1;
    }
}

#[inline(always)]
fn signed_unit_column_active_union(col_idx: usize, packed: &[(Vec<u64>, Vec<u64>)], valid_mask: u64) -> u64 {
    let mut active_union = 0u64;
    for (pos_bits, neg_bits) in packed {
        active_union |= (pos_bits[col_idx] | neg_bits[col_idx]) & valid_mask;
    }
    active_union
}

#[inline]
fn add_signed_unit_accs(dst: &mut [[Fq; D]], src: &[[Fq; D]]) {
    debug_assert_eq!(dst.len(), src.len());
    for (dst_acc, src_acc) in dst.iter_mut().zip(src.iter()) {
        for lane in 0..D {
            dst_acc[lane] += src_acc[lane];
        }
    }
}

fn commit_signed_unit_row_many_chunk(
    rng: &mut ChaCha8Rng,
    start: usize,
    end: usize,
    packed: &[(Vec<u64>, Vec<u64>)],
    column_plans: Option<&[SignedUnitColumnPlan]>,
    out: &mut [[Fq; D]],
) {
    let mut batch_words = [0u64; ajtai_commit::SEEDED_RQ_BATCH * D];
    let mut base_col = [Fq::ZERO; D];
    let valid_mask = (1u64 << D) - 1;

    let mut col_idx = start;
    while col_idx < end {
        let batch = (end - col_idx).min(ajtai_commit::SEEDED_RQ_BATCH);
        let mut active_offsets = 0u64;
        for offset in 0..batch {
            let col = col_idx + offset;
            let is_zero = match column_plans {
                Some(plans) => plans[col].active_union == 0,
                None => signed_unit_column_active_union(col, packed, valid_mask) == 0,
            };
            if !is_zero {
                active_offsets |= 1u64 << offset;
            }
        }

        if active_offsets == 0 {
            if ajtai_commit::advance_uniform_rq_coeff_validity_batch(rng, batch) {
                col_idx += batch;
                continue;
            }
            for _ in 0..batch {
                ajtai_commit::skip_uniform_rq_coeffs(rng);
            }
            col_idx += batch;
            continue;
        }

        if ajtai_commit::fill_uniform_rq_coeff_words_batch(rng, batch, &mut batch_words) {
            for offset in 0..batch {
                if (active_offsets & (1u64 << offset)) == 0 {
                    continue;
                }
                let col = col_idx + offset;
                let word_start = offset * D;
                let word_end = word_start + D;
                ajtai_commit::copy_uniform_rq_coeffs_from_words(&batch_words[word_start..word_end], &mut base_col);
                if let Some(plans) = column_plans {
                    accumulate_signed_unit_column_plan(out, &base_col, &plans[col]);
                } else {
                    accumulate_signed_unit_masks_many_for_col(out, &base_col, col, packed);
                }
            }
        } else {
            for offset in 0..batch {
                let col = col_idx + offset;
                if (active_offsets & (1u64 << offset)) == 0 {
                    ajtai_commit::skip_uniform_rq_coeffs(rng);
                    continue;
                }
                let base_col = ajtai_commit::sample_uniform_rq_coeffs(rng);
                if let Some(plans) = column_plans {
                    accumulate_signed_unit_column_plan(out, &base_col, &plans[col]);
                } else {
                    accumulate_signed_unit_masks_many_for_col(out, &base_col, col, packed);
                }
            }
        }
        col_idx += batch;
    }
}

fn commit_signed_unit_row_many(
    chunk_size: usize,
    chunk_seeds: &[[u8; 32]],
    m: usize,
    packed: &[(Vec<u64>, Vec<u64>)],
    column_plans: Option<&[SignedUnitColumnPlan]>,
    allow_inner_parallel: bool,
) -> Vec<[Fq; D]> {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if allow_inner_parallel && rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none() && m >= 4096
    {
        if chunk_seeds.len() == 1 && m <= SIGNED_UNIT_INNER_PAR_BASE_COL_LIMIT {
            let mut rng = ChaCha8Rng::from_seed(chunk_seeds[0]);
            let base_cols: Vec<[Fq; D]> = (0..m)
                .map(|_| ajtai_commit::sample_uniform_rq_coeffs(&mut rng))
                .collect();
            return base_cols
                .par_chunks(SIGNED_UNIT_INNER_PAR_COL_CHUNK)
                .enumerate()
                .map(|(chunk_idx, cols)| {
                    let mut local = vec![[Fq::ZERO; D]; packed.len()];
                    let start = chunk_idx * SIGNED_UNIT_INNER_PAR_COL_CHUNK;
                    for (offset, base_col) in cols.iter().enumerate() {
                        let col_idx = start + offset;
                        if let Some(plans) = column_plans {
                            accumulate_signed_unit_column_plan(&mut local, base_col, &plans[col_idx]);
                        } else {
                            accumulate_signed_unit_masks_many_for_col(&mut local, base_col, col_idx, packed);
                        }
                    }
                    local
                })
                .reduce(
                    || vec![[Fq::ZERO; D]; packed.len()],
                    |mut a, b| {
                        add_signed_unit_accs(&mut a, &b);
                        a
                    },
                );
        }

        if chunk_seeds.len() > 1 {
            return chunk_seeds
                .par_iter()
                .copied()
                .enumerate()
                .map(|(chunk_idx, seed)| {
                    let start = chunk_idx * chunk_size;
                    let end = core::cmp::min(m, start + chunk_size);
                    let mut local = vec![[Fq::ZERO; D]; packed.len()];
                    let mut rng = ChaCha8Rng::from_seed(seed);
                    commit_signed_unit_row_many_chunk(&mut rng, start, end, packed, column_plans, &mut local);
                    local
                })
                .reduce(
                    || vec![[Fq::ZERO; D]; packed.len()],
                    |mut a, b| {
                        add_signed_unit_accs(&mut a, &b);
                        a
                    },
                );
        }
    }

    let mut out = vec![[Fq::ZERO; D]; packed.len()];
    for (chunk_idx, seed) in chunk_seeds.iter().copied().enumerate() {
        let start = chunk_idx * chunk_size;
        let end = core::cmp::min(m, start + chunk_size);
        let mut rng = ChaCha8Rng::from_seed(seed);
        commit_signed_unit_row_many_chunk(&mut rng, start, end, packed, column_plans, &mut out);
    }
    out
}

fn commit_packed_signed_unit_column_bits_many(
    d: usize,
    kappa: usize,
    m: usize,
    packed: &[(Vec<u64>, Vec<u64>)],
    chunk_size: usize,
    chunk_seeds_by_row: &[Vec<[u8; 32]>],
) -> Vec<Commitment> {
    debug_assert_eq!(d, D);
    debug_assert_eq!(chunk_seeds_by_row.len(), kappa);
    if packed.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<Commitment> = (0..packed.len())
        .map(|_| Commitment::zeros(d, kappa))
        .collect();
    if m == 0 {
        return out;
    }
    let column_plans = if m <= SIGNED_UNIT_MASK_CACHE_COL_LIMIT {
        build_signed_unit_column_plans(m, packed)
    } else {
        None
    };
    let column_plans = column_plans.as_deref();

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let row_accs: Vec<(usize, Vec<[Fq; D]>)> = {
        let can_parallel = rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none();
        let num_chunks = chunk_seeds_by_row.first().map_or(0, Vec::len);
        if can_parallel
            && kappa > 1
            && num_chunks > 1
            && kappa.saturating_mul(num_chunks) <= SIGNED_UNIT_GRID_PAR_MAX_TASKS
        {
            let partials: Vec<(usize, Vec<[Fq; D]>)> = (0..(kappa * num_chunks))
                .into_par_iter()
                .map(|task_idx| {
                    let row = task_idx / num_chunks;
                    let chunk_idx = task_idx % num_chunks;
                    let start = chunk_idx * chunk_size;
                    let end = core::cmp::min(m, start + chunk_size);
                    let mut local = vec![[Fq::ZERO; D]; packed.len()];
                    let mut rng = ChaCha8Rng::from_seed(chunk_seeds_by_row[row][chunk_idx]);
                    commit_signed_unit_row_many_chunk(&mut rng, start, end, packed, column_plans, &mut local);
                    (row, local)
                })
                .collect();

            let mut rows = vec![vec![[Fq::ZERO; D]; packed.len()]; kappa];
            for (row, local) in partials {
                add_signed_unit_accs(&mut rows[row], &local);
            }
            rows.into_iter().enumerate().collect()
        } else if can_parallel && kappa > 4 {
            (0..kappa)
                .into_par_iter()
                .map(|row| {
                    (
                        row,
                        commit_signed_unit_row_many(
                            chunk_size,
                            &chunk_seeds_by_row[row],
                            m,
                            packed,
                            column_plans,
                            false,
                        ),
                    )
                })
                .collect()
        } else {
            (0..kappa)
                .map(|row| {
                    (
                        row,
                        commit_signed_unit_row_many(
                            chunk_size,
                            &chunk_seeds_by_row[row],
                            m,
                            packed,
                            column_plans,
                            can_parallel,
                        ),
                    )
                })
                .collect()
        }
    };

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let row_accs: Vec<(usize, Vec<[Fq; D]>)> = (0..kappa)
        .map(|row| {
            (
                row,
                commit_signed_unit_row_many(chunk_size, &chunk_seeds_by_row[row], m, packed, column_plans, false),
            )
        })
        .collect();

    for (row, accs) in row_accs {
        for (commitment, acc) in out.iter_mut().zip(accs.iter()) {
            commitment.col_mut(row).copy_from_slice(acc);
        }
    }
    out
}

fn commit_signed_unit_column_bits(seed: [u8; 32], d: usize, kappa: usize, m: usize, z: &Mat<Fq>) -> Option<Commitment> {
    let (pos_bits, neg_bits) = pack_signed_unit_column_bits(z, d, m)?;
    let (chunk_size, chunk_seeds_by_row) = ajtai_commit::seeded_pp_chunk_seeds(seed, kappa, m);
    Some(commit_packed_signed_unit_column_bits(
        d,
        kappa,
        m,
        &pos_bits,
        &neg_bits,
        chunk_size,
        &chunk_seeds_by_row,
    ))
}

/// Legacy: pick the sole PP if only one exists.
pub fn get_global_pp() -> Result<PPRef, AjtaiError> {
    let r = registry()
        .read()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    let mut it = r.iter();
    match (it.next(), it.next()) {
        (Some((&(d, m), _entry)), None) => {
            drop(r);
            get_or_load_global_pp_for_dims(d, m)
        }
        (None, _) => Err(AjtaiError::InvalidInput(
            "Ajtai PP not initialized (call set_global_pp())".to_string(),
        )),
        _ => Err(AjtaiError::InvalidInput(
            "Multiple Ajtai PPs present; use get_global_pp_for_dims()".to_string(),
        )),
    }
}

/// True if a PP for (d,m) is available.
pub fn has_global_pp_for_dims(d: usize, m: usize) -> bool {
    registry()
        .read()
        .map(|r| r.contains_key(&(d, m)))
        .unwrap_or(false)
}

/// Get the Ajtai PP for a specific (d,m).
pub fn get_global_pp_for_dims(d: usize, m: usize) -> Result<PPRef, AjtaiError> {
    get_or_load_global_pp_for_dims(d, m)
}

/// Get the Ajtai PP using `z_len = d*m`.
pub fn get_global_pp_for_z_len(z_len: usize) -> Result<PPRef, AjtaiError> {
    let d = neo_math::D;
    if z_len % d != 0 {
        return Err(AjtaiError::InvalidInput("z_len not multiple of D".to_string()));
    }
    get_global_pp_for_dims(d, z_len / d)
}

/// Concrete S-module homomorphism backed by Ajtai PP
#[derive(Clone)]
pub struct AjtaiSModule {
    pp: PpSource,
}

#[derive(Clone)]
enum PpSource {
    Owned(PPRef),
    Global { d: usize, m: usize },
}

impl AjtaiSModule {
    pub fn new(pp: PPRef) -> Self {
        Self {
            pp: PpSource::Owned(pp),
        }
    }
    /// Legacy: pick the sole PP if only one exists.
    pub fn from_global() -> Result<Self, AjtaiError> {
        let pp = get_global_pp()?;
        Ok(Self::new(pp))
    }
    /// New: pick PP that matches (d,m).
    pub fn from_global_for_dims(d: usize, m: usize) -> Result<Self, AjtaiError> {
        if !has_global_pp_for_dims(d, m) {
            return Err(AjtaiError::InvalidInput(
                "Ajtai PP not initialized for requested (d,m); call set_global_pp(...) or set_global_pp_seeded(...)"
                    .to_string(),
            ));
        }
        Ok(Self {
            pp: PpSource::Global { d, m },
        })
    }
    /// New: pick PP that matches `z_len = d*m`.
    pub fn from_global_for_z_len(z_len: usize) -> Result<Self, AjtaiError> {
        let d = neo_math::D;
        if z_len % d != 0 {
            return Err(AjtaiError::InvalidInput("z_len not multiple of D".to_string()));
        }
        let m = z_len / d;
        if !has_global_pp_for_dims(d, m) {
            return Err(AjtaiError::InvalidInput(
                "Ajtai PP not initialized for requested z_len; call set_global_pp(...) or set_global_pp_seeded(...)"
                    .to_string(),
            ));
        }
        Ok(Self {
            pp: PpSource::Global { d, m },
        })
    }

    /// Return κ for the underlying PP without requiring it to be materialized.
    pub fn kappa(&self) -> usize {
        match &self.pp {
            PpSource::Owned(pp) => pp.kappa,
            PpSource::Global { d, m } => registry()
                .read()
                .ok()
                .and_then(|r| r.get(&(*d, *m)).map(|e| e.kappa))
                .unwrap_or_else(|| {
                    get_or_load_global_pp_for_dims(*d, *m)
                        .expect("Ajtai PP load")
                        .kappa
                }),
        }
    }

    /// Return the Ajtai PP dimensions `(d, m)` without materializing a
    /// seeded global PP. Here `m` is the number of S-module columns, so a
    /// committed CCS witness has length `d * m`.
    pub fn dims(&self) -> (usize, usize) {
        match &self.pp {
            PpSource::Owned(pp) => (pp.d, pp.m),
            PpSource::Global { d, m } => (*d, *m),
        }
    }

    /// Return the registered deterministic setup parameters without
    /// materializing the public matrix.
    #[doc(hidden)]
    pub fn seeded_params(&self) -> Option<(usize, [u8; 32])> {
        let PpSource::Global { d, m } = &self.pp else {
            return None;
        };
        registry().read().ok().and_then(|entries| {
            entries
                .get(&(*d, *m))
                .and_then(|entry| entry.seed.map(|seed| (entry.kappa, seed)))
        })
    }

    /// Materialize the public matrix for verifier-side constraint emission.
    /// Commitment code should continue to use [`SModuleHomomorphism::commit`];
    /// this accessor exists for proof systems that must encode the same
    /// linear map as arithmetic constraints.
    pub fn verification_pp(&self) -> Result<Arc<PP<RqEl>>, AjtaiError> {
        self.materialize_pp()
    }

    /// Return the underlying Ajtai PP, loading a seeded global entry if needed.
    pub fn materialize_pp(&self) -> Result<Arc<PP<RqEl>>, AjtaiError> {
        match &self.pp {
            PpSource::Owned(pp) => Ok(pp.clone()),
            PpSource::Global { d, m } => get_or_load_global_pp_for_dims(*d, *m),
        }
    }
}

impl SModuleHomomorphism<Fq, Commitment> for AjtaiSModule {
    fn commit(&self, z: &Mat<Fq>) -> Commitment {
        match &self.pp {
            PpSource::Owned(pp) => ajtai_commit::commit_row_major(pp, z),
            PpSource::Global { d, m } => {
                // Prefer not to materialize PP for seeded entries.
                let (zd, zm) = (z.rows(), z.cols());
                let want_d = *d;
                let want_m = *m;
                assert_eq!(zd, want_d, "AjtaiSModule: Z.rows != d");
                assert_eq!(zm, want_m, "AjtaiSModule: Z.cols != m");

                if let Ok(r) = registry().read() {
                    if let Some(entry) = r.get(&(want_d, want_m)) {
                        if let Some(seed) = entry.seed {
                            // Seeded sparse witnesses are dramatically faster
                            // than dense materialized-PP multiplication. Keep
                            // this path even if the PP is currently loaded.
                            if let Some(column_bits) = pack_binary_column_bits(z, want_d, want_m) {
                                return ajtai_commit::commit_row_major_seeded_binary_cols(
                                    seed,
                                    want_d,
                                    entry.kappa,
                                    want_m,
                                    &column_bits,
                                );
                            }
                            if let Some(c) = commit_signed_unit_column_bits(seed, want_d, entry.kappa, want_m, z) {
                                return c;
                            }
                            if let Some(pp) = entry.pp.as_ref() {
                                return ajtai_commit::commit_row_major(pp, z);
                            }
                            return ajtai_commit::commit_row_major_seeded(seed, want_d, entry.kappa, want_m, z);
                        }
                        if let Some(pp) = entry.pp.as_ref() {
                            return ajtai_commit::commit_row_major(pp, z);
                        }
                    }
                }

                // Fallback: load PP if needed (non-seeded entry or registry inaccessible).
                let pp = get_or_load_global_pp_for_dims(want_d, want_m).expect("Ajtai PP load should succeed");
                ajtai_commit::commit_row_major(&pp, z)
            }
        }
    }

    fn commit_many(&self, zs: &[&Mat<Fq>]) -> Vec<Commitment> {
        if zs.is_empty() {
            return Vec::new();
        }
        match &self.pp {
            PpSource::Owned(pp) => zs
                .iter()
                .map(|z| ajtai_commit::commit_row_major(pp, z))
                .collect(),
            PpSource::Global { d, m } => {
                let want_d = *d;
                let want_m = *m;
                for (idx, z) in zs.iter().enumerate() {
                    assert_eq!(z.rows(), want_d, "AjtaiSModule: Zs[{idx}].rows != d");
                    assert_eq!(z.cols(), want_m, "AjtaiSModule: Zs[{idx}].cols != m");
                }

                if let Ok(r) = registry().read() {
                    if let Some(entry) = r.get(&(want_d, want_m)) {
                        if let Some(seed) = entry.seed {
                            if let Some(packed) = zs
                                .iter()
                                .map(|z| pack_signed_unit_column_bits(z, want_d, want_m))
                                .collect::<Option<Vec<_>>>()
                            {
                                let (chunk_size, chunk_seeds_by_row) =
                                    ajtai_commit::seeded_pp_chunk_seeds(seed, entry.kappa, want_m);
                                return commit_packed_signed_unit_column_bits_many(
                                    want_d,
                                    entry.kappa,
                                    want_m,
                                    &packed,
                                    chunk_size,
                                    &chunk_seeds_by_row,
                                );
                            }
                            if let Some(pp) = entry.pp.as_ref() {
                                return zs
                                    .iter()
                                    .map(|z| ajtai_commit::commit_row_major(pp, z))
                                    .collect();
                            }
                            return ajtai_commit::commit_row_major_seeded_many(seed, want_d, entry.kappa, want_m, zs);
                        }
                        if let Some(pp) = entry.pp.as_ref() {
                            return zs
                                .iter()
                                .map(|z| ajtai_commit::commit_row_major(pp, z))
                                .collect();
                        }
                    }
                }

                let pp = get_or_load_global_pp_for_dims(want_d, want_m).expect("Ajtai PP load should succeed");
                zs.iter()
                    .map(|z| ajtai_commit::commit_row_major(&pp, z))
                    .collect()
            }
        }
    }
}
