//! Sparse representations for time-domain vectors.
//!
//! Route A memory sidecar polynomials are defined over a time hypercube of size `2^ell_n`,
//! but only a small prefix of time rows (`steps`) is typically “real”.
//! This module provides a compact representation of vectors that are mostly zero,
//! along with an in-place multilinear folding step used by sumcheck-style oracles.

use p3_field::PrimeCharacteristicRing;

/// Sparse vector over indices `[0, len)`, where `len` is a power of two.
///
/// Invariant:
/// - `entries` are strictly increasing by index
/// - all stored values are nonzero
#[derive(Clone, Debug)]
pub struct SparseIdxVec<R> {
    len: usize,
    entries: Vec<(usize, R)>,
}

impl<R> SparseIdxVec<R>
where
    R: PrimeCharacteristicRing + Copy + PartialEq,
{
    pub fn new(len: usize) -> Self {
        debug_assert!(len.is_power_of_two());
        Self {
            len,
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[(usize, R)] {
        &self.entries
    }

    /// Build a sparse vector from arbitrary entries.
    pub fn from_entries(len: usize, mut entries: Vec<(usize, R)>) -> Self {
        debug_assert!(len.is_power_of_two());
        entries.retain(|&(_, v)| v != R::ZERO);
        entries.sort_by_key(|(i, _)| *i);
        #[cfg(debug_assertions)]
        for w in entries.windows(2) {
            debug_assert!(w[0].0 < w[1].0, "SparseIdxVec entries must be strictly increasing");
        }
        Self { len, entries }
    }

    /// Get the value at `idx`, returning `0` if not present.
    pub fn get(&self, idx: usize) -> R {
        match self.entries.binary_search_by_key(&idx, |(i, _)| *i) {
            Ok(pos) => self.entries[pos].1,
            Err(_) => R::ZERO,
        }
    }

    /// One multilinear folding round on the least-significant index bit:
    ///
    /// For each parent index `p`:
    /// `out[p] = in[2p] * (1-r) + in[2p+1] * r`
    pub fn fold_round_in_place(&mut self, r: R) {
        debug_assert!(self.len.is_power_of_two());
        debug_assert!(self.len >= 2, "cannot fold len < 2");

        let one_minus_r = R::ONE - r;
        let mut out: Vec<(usize, R)> = Vec::with_capacity(self.entries.len());

        let mut i = 0usize;
        while i < self.entries.len() {
            let (idx, v) = self.entries[i];
            let parent = idx >> 1;

            let mut acc = R::ZERO;
            if (idx & 1) == 0 {
                // Even child present.
                acc += v * one_minus_r;
                i += 1;

                // If odd sibling is present, include it.
                if i < self.entries.len() {
                    let (idx2, v2) = self.entries[i];
                    if idx2 == idx + 1 {
                        acc += v2 * r;
                        i += 1;
                    }
                }
            } else {
                // Odd child present without even sibling.
                acc += v * r;
                i += 1;
            }

            if acc != R::ZERO {
                if let Some((last_p, last_v)) = out.last_mut() {
                    if *last_p == parent {
                        *last_v += acc;
                    } else {
                        out.push((parent, acc));
                    }
                } else {
                    out.push((parent, acc));
                }
            }
        }

        self.entries = out;
        self.len >>= 1;
    }

    /// Only valid when `len == 1`.
    pub fn singleton_value(&self) -> R {
        debug_assert_eq!(self.len, 1);
        match self.entries.as_slice() {
            [] => R::ZERO,
            [(_, v)] => *v,
            _ => panic!("SparseIdxVec invariant violated: len==1 but multiple entries"),
        }
    }
}
