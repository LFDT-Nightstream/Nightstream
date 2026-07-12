//! Device layout for Pi_CCS public Fiat-Shamir challenges.
//!
//! Owns the resident word buffer produced by the device transcript. Host
//! `Challenges` remain the proof/audit surface; this type is only for feeding
//! later device kernels without rebuilding the same words from CPU values.

use cuda_core::DeviceBuffer;

pub(crate) struct DevicePublicChallenges {
    words: DeviceBuffer<u64>,
    ell_d: usize,
    ell: usize,
    ell_m: usize,
}

impl DevicePublicChallenges {
    pub(crate) fn new(words: DeviceBuffer<u64>, ell_d: usize, ell: usize, ell_m: usize) -> Self {
        Self {
            words,
            ell_d,
            ell,
            ell_m,
        }
    }

    pub(crate) fn words(&self) -> &DeviceBuffer<u64> {
        &self.words
    }

    pub(crate) fn matches_shape(&self, ell_d: usize, ell: usize, ell_m: usize) -> bool {
        self.ell_d == ell_d && self.ell == ell && self.ell_m == ell_m
    }

    pub(crate) fn fe_point_words(&self) -> usize {
        2 * (self.ell_d + self.ell)
    }

    pub(crate) fn beta_a_word_offset(&self) -> usize {
        2 * self.ell_d
    }

    pub(crate) fn beta_a_words(&self) -> usize {
        2 * self.ell_d
    }

    pub(crate) fn gamma_word_offset(&self) -> usize {
        2 * (self.ell_d + self.ell)
    }
}
