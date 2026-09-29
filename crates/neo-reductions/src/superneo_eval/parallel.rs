use super::{is_all_zero, RingEvalScratch, SuperneoZBlocks, F};
use neo_math::{KExtensions, D, K};
use p3_field::PrimeCharacteristicRing;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

#[inline]
pub(super) fn eval_active_blocks(scratch: &RingEvalScratch, z_blocks: &SuperneoZBlocks) -> Option<[K; D]> {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if rayon::current_num_threads() <= 1 || rayon::current_thread_index().is_some() {
            return None;
        }
        let (out_re, out_im) = scratch
            .active_blocks
            .par_iter()
            .map(|&blk| {
                let mut local_re = [F::ZERO; D];
                let mut local_im = [F::ZERO; D];
                if z_blocks.real_nonzero(blk) {
                    let (real, imaginary) = scratch.forms(blk);
                    match (!is_all_zero(&real.0), !is_all_zero(&imaginary.0)) {
                        (true, true) => {
                            z_blocks.accumulate_real_pair(&mut local_re, &mut local_im, &real, &imaginary, blk)
                        }
                        (true, false) => z_blocks.accumulate_real(&mut local_re, &real, blk),
                        (false, true) => z_blocks.accumulate_real(&mut local_im, &imaginary, blk),
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
        let _ = (scratch, z_blocks);
        None
    }
}

/// Evaluate every witness against the active scratch blocks. Each block's
/// forms are read once for all witnesses, and blocks are split across
/// workers; the result is `eval_ring_scratch_real_z_blocks` per witness.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
pub(super) fn eval_active_blocks_many(scratch: &RingEvalScratch, witnesses: &[&SuperneoZBlocks]) -> Vec<[K; D]> {
    const BLOCKS_PER_TASK: usize = 1 << 12;
    let zero = || vec![([F::ZERO; D], [F::ZERO; D]); witnesses.len()];
    scratch
        .active_blocks
        .par_chunks(BLOCKS_PER_TASK)
        .map(|blocks| {
            let mut sums = zero();
            for &blk in blocks {
                let mut forms = None;
                for (witness, (out_re, out_im)) in witnesses.iter().zip(sums.iter_mut()) {
                    if !witness.real_nonzero(blk) {
                        continue;
                    }
                    let (real, imaginary) = forms.get_or_insert_with(|| scratch.forms(blk));
                    match (!is_all_zero(&real.0), !is_all_zero(&imaginary.0)) {
                        (true, true) => witness.accumulate_real_pair(out_re, out_im, real, imaginary, blk),
                        (true, false) => witness.accumulate_real(out_re, real, blk),
                        (false, true) => witness.accumulate_real(out_im, imaginary, blk),
                        (false, false) => {}
                    }
                }
            }
            sums
        })
        .reduce(zero, |mut left, right| {
            for ((left_re, left_im), (right_re, right_im)) in left.iter_mut().zip(right) {
                for lane in 0..D {
                    left_re[lane] += right_re[lane];
                    left_im[lane] += right_im[lane];
                }
            }
            left
        })
        .into_iter()
        .map(|(real, imaginary)| core::array::from_fn(|lane| K::from_coeffs([real[lane], imaginary[lane]])))
        .collect()
}
