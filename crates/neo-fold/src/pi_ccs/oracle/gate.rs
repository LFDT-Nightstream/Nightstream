//! Gate helpers and utility functions for sum-check oracle
//!
//! This module provides reusable functions for gate evaluation, folding,
//! and pair index mapping used throughout the oracle implementation.

use neo_math::K;
use p3_field::PrimeCharacteristicRing;

/// Evaluate equality gate for a pair at position k
/// Computes: (1-x)·weights[2k] + x·weights[2k+1]
#[inline]
pub fn gate_pair(weights: &[K], k: usize, x: K) -> K {
    debug_assert!(weights.len() >= 2 * (k + 1),
        "weights must have at least {} elements for pair {}", 2*(k+1), k);
    let (w0, w1) = (weights[2*k], weights[2*k+1]);
    (K::ONE - x) * w0 + x * w1
}

/// Fold a partial vector in-place with challenge r
/// Transforms: v[k] = (1-r)·v[2k] + r·v[2k+1] for k in 0..n/2
#[inline]
pub fn fold_partial_in_place(v: &mut [K], r: K) {
    let n2 = v.len() >> 1;
    for k in 0..n2 {
        let a = v[2*k];
        let b = v[2*k+1];
        v[k] = (K::ONE - r) * a + r * b;
    }
    // Note: truncation happens at the Vec call site
}

/// Map folded pair index k to full indices (j0, j1) in original domain
/// This handles the dynamic stride mapping during row rounds
#[inline]
pub fn pair_to_full_indices(pair_k: usize, round_idx: usize) -> (usize, usize) {
    let stride = 1usize << round_idx;
    let j0 = (pair_k & (stride - 1)) + ((pair_k >> round_idx) << (round_idx + 1));
    (j0, j0 + stride)
}

/// Read-only view over pairs in a weight vector.
#[derive(Copy, Clone)]
pub struct PairGate<'a> {
    pub w: &'a [K],
    pub half: usize,
}

impl<'a> PairGate<'a> {
    /// Create a new pair gate view
    pub fn new(w: &'a [K]) -> Self {
        debug_assert!(w.len() % 2 == 0, "weights must have even length");
        Self { w, half: w.len() >> 1 }
    }
    
    /// Evaluate gate for pair k at point x
    #[inline]
    pub fn eval(&self, k: usize, x: K) -> K { gate_pair(self.w, k, x) }
    
    /// Get the two weights for pair k
    #[inline]
    pub fn pair(&self, k: usize) -> (K, K) {
        debug_assert!(k < self.half, "pair index {} out of bounds", k);
        (self.w[2*k], self.w[2*k+1])
    }
}