use p3_field::PrimeCharacteristicRing;

use super::{D, F, LIMB_BOUND, MAX_EXACT_TERMS};

const TILE_COLUMNS: usize = 128;
const GROUP_COLUMNS: usize = 1 << 16;
const LIMBS: usize = 2 * D;
const _: () = assert!(GROUP_COLUMNS as u128 * LIMB_BOUND <= 1u128 << f64::MANTISSA_DIGITS);

pub const ROW_WORKSPACE_WORDS: usize = TILE_COLUMNS * LIMBS + TILE_COLUMNS * D + LIMBS * D;

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

pub fn signed_ring_dot(key_words: &[u64], positive: &[u64], negative: &[u64], workspace: &mut [f64]) -> [F; D] {
    let columns = positive.len();
    assert!(columns <= MAX_EXACT_TERMS as usize);
    assert_eq!(negative.len(), columns);
    assert_eq!(key_words.len(), columns * D);
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
