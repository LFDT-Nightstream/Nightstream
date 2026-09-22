//! Owns opening coefficients and can reuse a completed SumCheck allocation.

use neo_math::{superneo_bar_block, KExtensions, Rq, D, F, K};
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Debug)]
pub(super) struct RingEvalScratch {
    values: Vec<K>,
    touched: Vec<bool>,
    pub(super) active_blocks: Vec<usize>,
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

    fn touch(&mut self, block: usize) {
        if !self.touched[block] {
            self.touched[block] = true;
            self.active_blocks.push(block);
        }
    }

    #[inline]
    pub(super) fn add_coefficient(&mut self, block: usize, local: usize, real: F, imaginary: F) {
        self.touch(block);
        self.values[block * D + local] += K::from_coeffs([real, imaginary]);
    }

    pub(super) fn add_scaled(&mut self, block: usize, original: &Rq, real: F, imaginary: F) {
        self.touch(block);
        for (value, &coefficient) in self.values[block * D..(block + 1) * D]
            .iter_mut()
            .zip(&original.0)
        {
            *value += K::from_coeffs([real * coefficient, imaginary * coefficient]);
        }
    }

    pub(super) fn add_forms(&mut self, block: usize, real: &Rq, imaginary: &Rq) {
        self.touch(block);
        for (local, value) in self.values[block * D..(block + 1) * D]
            .iter_mut()
            .enumerate()
        {
            *value += K::from_coeffs([real.0[local], imaginary.0[local]]);
        }
    }

    #[inline]
    pub(super) fn forms(&self, block: usize) -> (Rq, Rq) {
        let mut real = Rq::zero();
        let mut imaginary = Rq::zero();
        for (local, value) in self.values[block * D..(block + 1) * D].iter().enumerate() {
            [real.0[local], imaginary.0[local]] = value.as_coeffs();
        }
        (real, imaginary)
    }

    pub(super) fn bar_active(&mut self) {
        for &block in &self.active_blocks {
            let (real, imaginary) = self.forms(block);
            let real = superneo_bar_block(real.0);
            let imaginary = superneo_bar_block(imaginary.0);
            for (local, value) in self.values[block * D..(block + 1) * D]
                .iter_mut()
                .enumerate()
            {
                *value = K::from_coeffs([real[local], imaginary[local]]);
            }
        }
    }

    pub(super) fn clear_active(&mut self) {
        for &block in &self.active_blocks {
            self.values[block * D..(block + 1) * D].fill(K::ZERO);
            self.touched[block] = false;
        }
        self.active_blocks.clear();
    }
}
