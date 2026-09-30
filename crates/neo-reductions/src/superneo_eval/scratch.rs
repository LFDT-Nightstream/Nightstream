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

/// One contiguous block range of a scratch. Parts of one scratch are
/// disjoint, so they can be filled concurrently.
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

    /// The whole scratch as one part.
    pub(super) fn part(&mut self) -> ScratchPart<'_> {
        ScratchPart {
            first_block: 0,
            values: &mut self.values,
            touched: &mut self.touched,
            active_blocks: &mut self.active_blocks,
        }
    }

    /// Run `fill` on disjoint contiguous block ranges, one per worker thread.
    /// Touched blocks are appended in range order.
    pub(super) fn fill_parts(&mut self, fill: impl Fn(&mut ScratchPart<'_>) + Sync) {
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            let block_count = self.touched.len();
            let per_part = block_count.div_ceil(rayon::current_num_threads()).max(1);
            let mut parts = vec![Vec::new(); block_count.div_ceil(per_part)];
            self.values
                .par_chunks_mut(per_part * D)
                .zip(self.touched.par_chunks_mut(per_part))
                .zip(parts.par_iter_mut())
                .enumerate()
                .for_each(|(index, ((values, touched), active_blocks))| {
                    fill(&mut ScratchPart {
                        first_block: index * per_part,
                        values,
                        touched,
                        active_blocks,
                    })
                });
            for part in parts {
                self.active_blocks.extend(part);
            }
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        fill(&mut self.part());
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

    /// Mark one block of this part active and return its coefficients.
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

    /// Add `term * ratio^k` to the `k`th slot of `locals` in one block and
    /// return the term for the next slot.
    #[inline]
    pub(super) fn add_geometric(&mut self, block: usize, locals: Range<usize>, mut term: [F; 2], ratio: F) -> [F; 2] {
        for value in &mut self.slot(block)[locals] {
            *value += K::from_coeffs(term);
            term = [term[0] * ratio, term[1] * ratio];
        }
        term
    }

    /// Add a geometric event at one column. `resolve_geometric` later turns
    /// the events into run sums.
    #[inline]
    pub(super) fn add_event(&mut self, column: usize, term: [F; 2]) {
        self.slot(column / D)[column % D] += K::from_coeffs(term);
    }

    /// Mark a block covered by a run without adding a value to it.
    #[inline]
    pub(super) fn touch(&mut self, block: usize) {
        self.slot(block);
    }

    /// Replace every value by `ratio * previous + event`, so each start event
    /// spreads as a geometric run until its end event cancels it. Values must
    /// hold events only. Every block a run covers is touched, so the running
    /// sum is exactly zero across untouched blocks.
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

    pub(super) fn add_scaled(&mut self, block: usize, original: &Rq, real: F, imaginary: F) {
        for (value, &coefficient) in self.slot(block).iter_mut().zip(&original.0) {
            *value += K::from_coeffs([real * coefficient, imaginary * coefficient]);
        }
    }

    /// Replace every active block of this part by its `bar` image.
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
