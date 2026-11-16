//! Owns opening coefficients and can reuse a completed SumCheck allocation.

use std::ops::Range;

use neo_math::{superneo_bar_block, KExtensions, Rq, D, F, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub(super) struct RingEvalScratch {
    values: Vec<K>,
    touched: Vec<bool>,
    pub(super) active_blocks: Vec<usize>,
}

pub(super) struct ScratchPart<'a> {
    first_block: usize,
    values: &'a mut [K],
    touched: &'a mut [bool],
    active_blocks: &'a mut Vec<usize>,
}

impl RingEvalScratch {
    pub(super) fn new(block_count: usize) -> Self {
        Self::reuse(Vec::new(), block_count)
    }

    pub(super) fn reuse(mut storage: Vec<K>, block_count: usize) -> Self {
        storage.clear();
        storage.resize(block_count * D, K::ZERO);
        Self {
            values: storage,
            touched: vec![false; block_count],
            active_blocks: Vec::new(),
        }
    }

    pub(super) fn ensure_block_count(&mut self, block_count: usize) {
        if self.values.len() == block_count * D {
            return;
        }
        self.values.resize(block_count * D, K::ZERO);
        self.touched.resize(block_count, false);
        self.active_blocks.clear();
    }

    pub(super) fn part(&mut self) -> ScratchPart<'_> {
        ScratchPart {
            first_block: 0,
            values: &mut self.values,
            touched: &mut self.touched,
            active_blocks: &mut self.active_blocks,
        }
    }

    /// Fills part `p`, the blocks `bounds[p]..bounds[p + 1]`, with `fill(p, part)`.
    pub(super) fn fill_parts(&mut self, bounds: &[usize], fill: impl Fn(usize, &mut ScratchPart<'_>) + Sync) {
        debug_assert_eq!(bounds.last(), Some(&self.touched.len()));
        let mut active = vec![Vec::new(); bounds.len() - 1];
        let mut values = self.values.as_mut_slice();
        let mut touched = self.touched.as_mut_slice();
        let mut parts = Vec::with_capacity(active.len());
        for (range, active_blocks) in bounds.windows(2).zip(&mut active) {
            let (part_values, rest) = core::mem::take(&mut values).split_at_mut((range[1] - range[0]) * D);
            values = rest;
            let (part_touched, rest) = core::mem::take(&mut touched).split_at_mut(range[1] - range[0]);
            touched = rest;
            parts.push(ScratchPart {
                first_block: range[0],
                values: part_values,
                touched: part_touched,
                active_blocks,
            });
        }
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        parts
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, part)| fill(index, part));
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        parts
            .iter_mut()
            .enumerate()
            .for_each(|(index, part)| fill(index, part));
        drop(parts);
        for part in active {
            self.active_blocks.extend(part);
        }
    }

    #[inline]
    pub(super) fn forms(&self, block: usize) -> (Rq, Rq) {
        forms(&self.values[block * D..(block + 1) * D])
    }

    pub(super) fn bar_active(&mut self) {
        self.part().bar_active();
    }

    pub(super) fn clear_active(&mut self) {
        for &block in &self.active_blocks {
            self.values[block * D..(block + 1) * D].fill(K::ZERO);
            self.touched[block] = false;
        }
        self.active_blocks.clear();
    }
}

impl ScratchPart<'_> {
    pub(super) fn blocks(&self) -> Range<usize> {
        self.first_block..self.first_block + self.touched.len()
    }

    #[inline]
    fn slot(&mut self, block: usize) -> &mut [K] {
        let local = block - self.first_block;
        if !self.touched[local] {
            self.touched[local] = true;
            self.active_blocks.push(block);
        }
        &mut self.values[local * D..(local + 1) * D]
    }

    #[inline]
    pub(super) fn add_coefficient(&mut self, block: usize, local: usize, real: F, imaginary: F) {
        self.slot(block)[local] += K::from_coeffs([real, imaginary]);
    }

    #[inline]
    pub(super) fn add_geometric(&mut self, block: usize, locals: Range<usize>, mut term: [F; 2], ratio: F) -> [F; 2] {
        for value in &mut self.slot(block)[locals] {
            *value += K::from_coeffs(term);
            term = [term[0] * ratio, term[1] * ratio];
        }
        term
    }

    #[inline]
    pub(super) fn add_event(&mut self, column: usize, term: [F; 2]) {
        self.slot(column / D)[column % D] += K::from_coeffs(term);
    }

    #[inline]
    pub(super) fn touch(&mut self, block: usize) {
        self.slot(block);
    }

    pub(super) fn resolve_geometric(&mut self, ratio: F) {
        let mut running = [F::ZERO; 2];
        for (&touched, values) in self.touched.iter().zip(self.values.chunks_exact_mut(D)) {
            if !touched {
                continue;
            }
            for value in values {
                let [real, imaginary] = value.as_coeffs();
                running = [running[0] * ratio + real, running[1] * ratio + imaginary];
                *value = K::from_coeffs(running);
            }
        }
    }

    pub(super) fn bar_active(&mut self) {
        for &block in self.active_blocks.iter() {
            let local = block - self.first_block;
            let values = &mut self.values[local * D..(local + 1) * D];
            let (real, imaginary) = forms(values);
            let real = superneo_bar_block(real.0);
            let imaginary = superneo_bar_block(imaginary.0);
            for (local, value) in values.iter_mut().enumerate() {
                *value = K::from_coeffs([real[local], imaginary[local]]);
            }
        }
    }
}

#[inline]
fn forms(values: &[K]) -> (Rq, Rq) {
    let mut real = Rq::zero();
    let mut imaginary = Rq::zero();
    for (local, value) in values.iter().enumerate() {
        [real.0[local], imaginary.0[local]] = value.as_coeffs();
    }
    (real, imaginary)
}
