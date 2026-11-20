use neo_math::{
    signed_sums::{SignedShiftSums, SplitRing},
    KExtensions, Rq, D, F, K,
};
use p3_field::PrimeCharacteristicRing;

use super::{is_all_zero, SuperneoZBlocks};

pub(super) const TASK_BLOCKS: usize = 1 << 12;

pub(super) type PairSums = Vec<([F; D], [F; D])>;

pub(super) struct TaskSums {
    reduced: PairSums,
    raw: Vec<(SignedShiftSums, SignedShiftSums)>,
}

pub(super) struct OpeningEvents {
    pub(super) real: Rq,
    pub(super) imaginary: Rq,
    pub(super) mask: u64,
}

impl TaskSums {
    pub(super) fn working_bytes(bands: usize, witnesses: usize) -> Option<usize> {
        let pair = size_of::<([F; D], [F; D])>();
        let raw = size_of::<(SignedShiftSums, SignedShiftSums)>();
        bands
            .checked_mul(raw + 3 * pair)?
            .checked_add(pair + size_of::<[K; D]>())?
            .checked_mul(witnesses)
    }

    pub(super) fn new(witnesses: usize) -> Self {
        Self {
            reduced: zero_pair_sums(witnesses),
            raw: vec![(SignedShiftSums::zero(), SignedShiftSums::zero()); witnesses],
        }
    }

    #[inline]
    pub(super) fn add_block(&mut self, witnesses: &[&SuperneoZBlocks], block: usize, forms: impl Fn() -> (Rq, Rq)) {
        self.add_opening_block(witnesses, block, None, forms);
    }

    #[inline]
    pub(super) fn add_opening_block(
        &mut self,
        witnesses: &[&SuperneoZBlocks],
        block: usize,
        events: Option<u64>,
        forms: impl Fn() -> (Rq, Rq),
    ) {
        let sparse = events.filter(|events| events.count_ones() <= 12);
        let mut reduced_forms = None;
        let mut split_forms = None;
        for (index, witness) in witnesses.iter().enumerate() {
            if !witness.real_nonzero(block) {
                continue;
            }
            let (real, imaginary) = reduced_forms.get_or_insert_with(&forms);
            if let Some((positive, negative)) = witness.signed_unit_masks() {
                let (raw_real, raw_imaginary) = &mut self.raw[index];
                if let Some(events) = sparse {
                    SignedShiftSums::add_signed_event_pair(
                        [raw_real, raw_imaginary],
                        [&real.0, &imaginary.0],
                        events,
                        positive[block],
                        negative[block],
                    );
                } else {
                    let (real, imaginary) =
                        split_forms.get_or_insert_with(|| (SplitRing::new(&real.0), SplitRing::new(&imaginary.0)));
                    raw_real.add_signed_units(real, positive[block], negative[block]);
                    raw_imaginary.add_signed_units(imaginary, positive[block], negative[block]);
                }
                continue;
            }
            if let (Some(events), super::RealBlockStorage::Dense { blocks, .. }) = (sparse, &witness.re) {
                const _: () = assert!(TASK_BLOCKS <= (1 << 12));
                let (a, b) = &mut self.raw[index];
                if let Some(digits) = blocks.event_digits(block) {
                    SignedShiftSums::add_prepared_event_pair([a, b], [&real.0, &imaginary.0], events, &digits);
                    continue;
                }
                let (a, b) = &mut self.reduced[index];
                add_sparse_digit_pair([a, b], [real, imaginary], events, &blocks.ring(block));
                continue;
            }
            self.add_full(index, witness, block, real, imaginary);
        }
    }

    fn add_full(&mut self, index: usize, witness: &SuperneoZBlocks, block: usize, real: &Rq, imaginary: &Rq) {
        let (out_real, out_imaginary) = &mut self.reduced[index];
        match (!is_all_zero(&real.0), !is_all_zero(&imaginary.0)) {
            (true, true) => witness.accumulate_real_pair(out_real, out_imaginary, real, imaginary, block),
            (true, false) => witness.accumulate_real(out_real, real, block),
            (false, true) => witness.accumulate_real(out_imaginary, imaginary, block),
            (false, false) => {}
        }
    }

    pub(super) fn add_opening_blocks<const M: usize>(
        matrices: &mut [Self; M],
        witnesses: &[&SuperneoZBlocks],
        block: usize,
        forms: &[OpeningEvents; M],
    ) {
        const _: () = assert!(TASK_BLOCKS <= (1 << 12));
        let sparse = forms
            .each_ref()
            .map(|form| form.mask != 0 && form.mask.count_ones() <= 12);
        let uses_sparse = sparse.into_iter().any(|active| active);
        let mut split: [Option<(SplitRing, SplitRing)>; M] = std::array::from_fn(|_| None);
        for (index, witness) in witnesses.iter().enumerate() {
            if !witness.real_nonzero(block) {
                continue;
            }
            let signs = witness.signed_unit_masks();
            let dense = match &witness.re {
                super::RealBlockStorage::Dense { blocks, .. } if uses_sparse => Some(blocks),
                _ => None,
            };
            let small = dense.and_then(|blocks| blocks.event_digits(block));
            for (((sums, form), &sparse), split) in matrices.iter_mut().zip(forms).zip(&sparse).zip(&mut split) {
                if form.mask == 0 {
                    continue;
                }
                let (real, imaginary) = (&form.real, &form.imaginary);
                let (out_real, out_imaginary) = &mut sums.raw[index];
                if let Some((positive, negative)) = signs {
                    if sparse {
                        SignedShiftSums::add_signed_event_pair(
                            [out_real, out_imaginary],
                            [&real.0, &imaginary.0],
                            form.mask,
                            positive[block],
                            negative[block],
                        );
                    } else {
                        let (real, imaginary) =
                            split.get_or_insert_with(|| (SplitRing::new(&real.0), SplitRing::new(&imaginary.0)));
                        out_real.add_signed_units(real, positive[block], negative[block]);
                        out_imaginary.add_signed_units(imaginary, positive[block], negative[block]);
                    }
                } else if sparse && dense.is_some() {
                    if let Some(digits) = &small {
                        SignedShiftSums::add_prepared_event_pair(
                            [out_real, out_imaginary],
                            [&real.0, &imaginary.0],
                            form.mask,
                            digits,
                        );
                    } else if let Some(blocks) = dense {
                        let (a, b) = &mut sums.reduced[index];
                        add_sparse_digit_pair([a, b], [real, imaginary], form.mask, &blocks.ring(block));
                    }
                } else {
                    sums.add_full(index, witness, block, real, imaginary);
                }
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

fn add_sparse_digit_pair(output: [&mut [F; D]; 2], forms: [&Rq; 2], mut events: u64, digits: &Rq) {
    let [real, imaginary] = output;
    while events != 0 {
        let degree = events.trailing_zeros() as usize;
        events &= events - 1;
        let [a, b] = forms.map(|f| f.0[degree]);
        if a == F::ZERO && b == F::ZERO {
            continue;
        }
        let (head, tail) = digits.0.split_at(D - degree);
        for ((x, y), &d) in real[degree..]
            .iter_mut()
            .zip(&mut imaginary[degree..])
            .zip(head)
        {
            *x += a * d;
            *y += b * d;
        }
        for ((x, y), &d) in real[..degree]
            .iter_mut()
            .zip(&mut imaginary[..degree])
            .zip(tail)
        {
            *x -= a * d;
            *y -= b * d;
        }
    }
}
