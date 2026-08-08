use super::{is_all_zero, Rq, SuperneoZBlocks, F};
use neo_math::{KExtensions, D, K};
use p3_field::PrimeCharacteristicRing;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

#[inline]
pub(super) fn eval_active_blocks(
    active_blocks: &[usize],
    agg_re: &[Rq],
    agg_im: &[Rq],
    z_blocks: &SuperneoZBlocks,
) -> Option<[K; D]> {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if rayon::current_num_threads() <= 1 || rayon::current_thread_index().is_some() {
            return None;
        }
        let (out_re, out_im) = active_blocks
            .par_iter()
            .map(|&blk| {
                let mut local_re = [F::ZERO; D];
                let mut local_im = [F::ZERO; D];
                if z_blocks.real_nonzero(blk) {
                    match (!is_all_zero(&agg_re[blk].0), !is_all_zero(&agg_im[blk].0)) {
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
