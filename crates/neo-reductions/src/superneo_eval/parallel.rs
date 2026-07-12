use super::{is_all_zero, Rq, SuperneoZBlocks, F};
use neo_math::{KExtensions, D, K};
use p3_field::PrimeCharacteristicRing;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
const ACTIVE_BLOCK_PAR_THRESHOLD: usize = 4096;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
const ACTIVE_BLOCK_CHUNK: usize = 512;
#[inline]
pub(super) fn eval_active_blocks(
    active_blocks: &[usize],
    agg_re: &[Rq],
    agg_im: &[Rq],
    z_blocks: &SuperneoZBlocks,
) -> Option<[K; D]> {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if active_blocks.len() < ACTIVE_BLOCK_PAR_THRESHOLD || rayon::current_num_threads() <= 1 {
            return None;
        }
        let (out_re, out_im) = active_blocks
            .par_chunks(ACTIVE_BLOCK_CHUNK)
            .map(|blocks| {
                let mut local_re = [F::ZERO; D];
                let mut local_im = [F::ZERO; D];
                for &blk in blocks {
                    if !z_blocks.real_nonzero(blk) {
                        continue;
                    }
                    let re_nonzero = !is_all_zero(&agg_re[blk].0);
                    let im_nonzero = !is_all_zero(&agg_im[blk].0);
                    match (re_nonzero, im_nonzero) {
                        (true, true) => {
                            z_blocks.accumulate_real_pair(&mut local_re, &mut local_im, &agg_re[blk], &agg_im[blk], blk)
                        }
                        (true, false) => z_blocks.accumulate_real(&mut local_re, &agg_re[blk], blk),
                        (false, true) => z_blocks.accumulate_real(&mut local_im, &agg_im[blk], blk),
                        (false, false) => {}
                    }
                }
                (local_re, local_im)
            })
            .reduce(
                || ([F::ZERO; D], [F::ZERO; D]),
                |mut a, b| {
                    for i in 0..D {
                        a.0[i] += b.0[i];
                        a.1[i] += b.1[i];
                    }
                    a
                },
            );
        let mut out = [K::ZERO; D];
        for i in 0..D {
            out[i] = K::from_coeffs([out_re[i], out_im[i]]);
        }
        Some(out)
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    {
        let _ = (active_blocks, agg_re, agg_im, z_blocks);
        None
    }
}
