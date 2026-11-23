use p3_field::PrimeCharacteristicRing;

use super::{D, F, LIMB_BOUND, MAX_EXACT_TERMS};

const TILE_COLUMNS: usize = 128;
const GROUP_COLUMNS: usize = 1 << 16;
const LIMBS: usize = 2 * D;
const _: () = assert!(GROUP_COLUMNS as u128 * LIMB_BOUND <= 1u128 << f64::MANTISSA_DIGITS);

pub const ROW_WORKSPACE_WORDS: usize = TILE_COLUMNS * LIMBS + TILE_COLUMNS * D + LIMBS * D;

#[cfg(nightstream_sme2)]
mod sme2;

mod child_dgemm;

pub const MAX_BATCH_PLANES: usize = 6;
const CHILD_TILE_BYTES: usize = 165_888;
const CHILD_PLANE_BYTES: usize = 73_728;
const CHILD_STACK_AND_PAGE_SLACK: usize = 64 << 10;

pub fn child_batch_workspace_bytes(planes: usize) -> usize {
    assert!((1..=MAX_BATCH_PLANES).contains(&planes));
    CHILD_TILE_BYTES + planes * CHILD_PLANE_BYTES + CHILD_STACK_AND_PAGE_SLACK
}

#[derive(Clone, Copy)]
pub enum ChildBatchKernel {
    #[cfg(nightstream_sme2)]
    Sme2,
    Toom4,
}

impl ChildBatchKernel {
    pub fn select() -> Self {
        #[cfg(nightstream_sme2)]
        if sme2::available_on_this_thread() {
            return Self::Sme2;
        }
        Self::Toom4
    }

    pub fn jobs_per_task(self, jobs: usize, planes: usize, scratch_budget: usize) -> usize {
        match self {
            #[cfg(nightstream_sme2)]
            Self::Sme2 => 1,
            Self::Toom4 => jobs.div_ceil(scratch_budget / child_dgemm::workspace(planes)),
        }
    }

    pub fn try_dot(self, keys: &[u64], columns: &[u32], planes: &[(&[u64], &[u64])], out: &mut [[F; D]]) -> bool {
        match self {
            #[cfg(nightstream_sme2)]
            Self::Sme2 => sme2::child_batch_dot(keys, columns, planes, out),
            Self::Toom4 => child_dgemm::compute(keys, columns, planes, out),
        }
    }
}

pub fn packed_signed_ring_dot(key_words: &[u64], positive: &[u64], negative: &[u64]) -> [F; D] {
    #[cfg(nightstream_sme2)]
    if let Some(row) = sme2::try_signed_ring_dot(key_words, positive, negative) {
        return row;
    }
    signed_ring_dot(key_words, positive, negative, &mut vec![0.0; ROW_WORKSPACE_WORDS])
}

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_dgemm(
        layout: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}

fn check_row(key_words: &[u64], positive: &[u64], negative: &[u64]) {
    let columns = positive.len();
    assert!(columns <= MAX_EXACT_TERMS as usize);
    assert_eq!(negative.len(), columns);
    assert_eq!(key_words.len(), columns * D);
}

pub fn signed_ring_dot(key_words: &[u64], positive: &[u64], negative: &[u64], workspace: &mut [f64]) -> [F; D] {
    check_row(key_words, positive, negative);
    let columns = positive.len();
    assert_eq!(workspace.len(), ROW_WORKSPACE_WORDS);
    let (key_tile, workspace) = workspace.split_at_mut(TILE_COLUMNS * LIMBS);
    let (digit_tile, products) = workspace.split_at_mut(TILE_COLUMNS * D);
    let key_tile = key_tile.as_chunks_mut::<LIMBS>().0;
    let digit_tile = digit_tile.as_chunks_mut::<D>().0;
    let mut coefficients = [0i128; D];

    for start in (0..columns).step_by(GROUP_COLUMNS) {
        let end = (start + GROUP_COLUMNS).min(columns);
        let mut accumulated = false;
        for tile in (start..end).step_by(TILE_COLUMNS) {
            let mut active = 0;
            for column in tile..(tile + TILE_COLUMNS).min(end) {
                let (pos, neg) = (positive[column], negative[column]);
                if pos | neg == 0 {
                    continue;
                }
                for (limbs, &word) in key_tile[active]
                    .chunks_exact_mut(2)
                    .zip(&key_words[column * D..][..D])
                {
                    limbs[0] = f64::from(word as u32);
                    limbs[1] = f64::from((word >> 32) as u32);
                }
                for (lane, digit) in digit_tile[active].iter_mut().enumerate() {
                    *digit = f64::from(((pos >> lane) & 1) as i32 - ((neg >> lane) & 1) as i32);
                }
                active += 1;
            }
            if active > 0 {
                const ROW_MAJOR: i32 = 101;
                const NO_TRANSPOSE: i32 = 111;
                const TRANSPOSE: i32 = 112;
                // SAFETY: active <= TILE_COLUMNS. The disjoint row-major panels have strides LIMBS, D and D.
                #[allow(unsafe_code)]
                unsafe {
                    cblas_dgemm(
                        ROW_MAJOR,
                        TRANSPOSE,
                        NO_TRANSPOSE,
                        LIMBS as i32,
                        D as i32,
                        active as i32,
                        1.0,
                        key_tile.as_ptr().cast(),
                        LIMBS as i32,
                        digit_tile.as_ptr().cast(),
                        D as i32,
                        f64::from(u8::from(accumulated)),
                        products.as_mut_ptr(),
                        D as i32,
                    );
                }
                accumulated = true;
            }
        }
        if accumulated {
            for (key_lane, pair) in products.chunks_exact(2 * D).enumerate() {
                let (low, high) = pair.split_at(D);
                for lane in 0..D {
                    let product = i128::from(low[lane] as i64) + (i128::from(high[lane] as i64) << 32);
                    let degree = key_lane + lane;
                    if degree < D {
                        coefficients[degree] += product;
                    } else {
                        coefficients[degree - D] -= product;
                    }
                }
            }
        }
    }
    coefficients.map(F::from_i128)
}
