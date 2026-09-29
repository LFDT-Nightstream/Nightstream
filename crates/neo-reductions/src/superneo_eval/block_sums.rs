//! One task's opening sums of real witnesses against per-block ring forms.
//! Signed-unit witnesses keep exact raw shift sums and reduce once per task;
//! other witnesses add reduced ring products block by block.

use neo_math::{
    signed_sums::{SignedShiftSums, SplitRing},
    KExtensions, Rq, D, F, K,
};
use p3_field::PrimeCharacteristicRing;

use super::{is_all_zero, SuperneoZBlocks};

/// Blocks per task. Each block adds one signed-unit term set per witness,
/// far below the raw-sum limit.
pub(super) const TASK_BLOCKS: usize = 1 << 12;

/// Real and imaginary coefficient sums for each witness of a batch.
pub(super) type PairSums = Vec<([F; D], [F; D])>;

pub(super) struct TaskSums {
    reduced: PairSums,
    raw: Vec<(SignedShiftSums, SignedShiftSums)>,
}

impl TaskSums {
    pub(super) fn new(witnesses: usize) -> Self {
        Self {
            reduced: zero_pair_sums(witnesses),
            raw: vec![(SignedShiftSums::zero(), SignedShiftSums::zero()); witnesses],
        }
    }

    /// Add each witness's ring product with the block's real and imaginary
    /// forms. `forms` runs only when some witness is nonzero in the block.
    #[inline]
    pub(super) fn add_block(&mut self, witnesses: &[&SuperneoZBlocks], block: usize, forms: impl Fn() -> (Rq, Rq)) {
        let mut reduced_forms = None;
        let mut split_forms = None;
        for (index, witness) in witnesses.iter().enumerate() {
            if !witness.real_nonzero(block) {
                continue;
            }
            let (real, imaginary) = reduced_forms.get_or_insert_with(&forms);
            if let Some((positive, negative)) = witness.signed_unit_masks() {
                let (real, imaginary) =
                    split_forms.get_or_insert_with(|| (SplitRing::new(&real.0), SplitRing::new(&imaginary.0)));
                let (raw_real, raw_imaginary) = &mut self.raw[index];
                raw_real.add_signed_units(real, positive[block], negative[block]);
                raw_imaginary.add_signed_units(imaginary, positive[block], negative[block]);
                continue;
            }
            let (out_real, out_imaginary) = &mut self.reduced[index];
            match (!is_all_zero(&real.0), !is_all_zero(&imaginary.0)) {
                (true, true) => witness.accumulate_real_pair(out_real, out_imaginary, real, imaginary, block),
                (true, false) => witness.accumulate_real(out_real, real, block),
                (false, true) => witness.accumulate_real(out_imaginary, imaginary, block),
                (false, false) => {}
            }
        }
    }

    pub(super) fn finish(mut self) -> PairSums {
        for ((real, imaginary), (raw_real, raw_imaginary)) in self.reduced.iter_mut().zip(&self.raw) {
            let (raw_real, raw_imaginary) = (raw_real.reduce(), raw_imaginary.reduce());
            for lane in 0..D {
                real[lane] += raw_real[lane];
                imaginary[lane] += raw_imaginary[lane];
            }
        }
        self.reduced
    }
}

pub(super) fn zero_pair_sums(witnesses: usize) -> PairSums {
    vec![([F::ZERO; D], [F::ZERO; D]); witnesses]
}

pub(super) fn add_pair_sums(mut left: PairSums, right: PairSums) -> PairSums {
    for ((left_real, left_imaginary), (right_real, right_imaginary)) in left.iter_mut().zip(right) {
        for lane in 0..D {
            left_real[lane] += right_real[lane];
            left_imaginary[lane] += right_imaginary[lane];
        }
    }
    left
}

pub(super) fn to_extension(sums: PairSums) -> Vec<[K; D]> {
    sums.into_iter()
        .map(|(real, imaginary)| core::array::from_fn(|lane| K::from_coeffs([real[lane], imaginary[lane]])))
        .collect()
}
