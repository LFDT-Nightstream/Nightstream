//! Equality weight computations for eq(·,r) optimization.
//!
//! Half-table implementation to compute χ_r(row) = eq(row, r) without
//! materializing the full 2^ℓ tensor, using split lookup tables.

use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};
use rayon::prelude::*;
use crate::pi_ccs::sparse_matrix::Csr;

/// Row weight provider: returns χ_r(row) or an equivalent row weight.
pub trait RowWeight: Sync {
    fn w(&self, row: usize) -> K;
}

/// Half-table implementation of χ_r row weights to avoid materializing the full tensor.
pub struct HalfTableEq {
    lo: Vec<K>,
    hi: Vec<K>,
    split: usize,
}

impl HalfTableEq {
    pub fn new(r: &[K]) -> Self {
        let ell = r.len();
        let split = ell / 2; // lower split bits in lo, higher in hi
        let lo_len = 1usize << split;
        let hi_len = 1usize << (ell - split);

        // Precompute factors (1-r_i, r_i)
        let mut one_minus = Vec::with_capacity(ell);
        for &ri in r { one_minus.push(K::ONE - ri); }

        // Build lo table
        let mut lo = vec![K::ONE; lo_len];
        for mask in 0..lo_len {
            let mut acc = K::ONE;
            let mut m = mask;
            for i in 0..split {
                let bit = m & 1;
                acc *= if bit == 0 { one_minus[i] } else { r[i] };
                m >>= 1;
            }
            lo[mask] = acc;
        }

        // Build hi table
        let mut hi = vec![K::ONE; hi_len];
        for mask in 0..hi_len {
            let mut acc = K::ONE;
            let mut m = mask;
            for j in 0..(ell - split) {
                let idx = split + j;
                let bit = m & 1;
                acc *= if bit == 0 { one_minus[idx] } else { r[idx] };
                m >>= 1;
            }
            hi[mask] = acc;
        }

        Self { lo, hi, split }
    }
}

impl RowWeight for HalfTableEq {
    #[inline]
    fn w(&self, row: usize) -> K {
        let lo_mask = (1usize << self.split) - 1;
        let lo_idx = row & lo_mask;
        let hi_idx = row >> self.split;
        self.lo[lo_idx] * self.hi[hi_idx]
    }
}

/// Weighted version of CSR transpose multiply: v = A^T * w, where w(row) is provided on-the-fly.
///
/// Efficient parallel implementation using per-thread accumulators to avoid per-row Vec allocations.
pub fn spmv_csr_t_weighted_fk<F, W>(
    a: &Csr<F>,
    w: &W,
) -> Vec<K> 
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
    W: RowWeight + Sync,
{
    let cols = a.cols;
    let zero = K::ZERO;
    
    // Parallel fold: each thread builds a local accumulator, then reduce
    (0..a.rows).into_par_iter()
        .fold(|| vec![zero; cols], |mut acc, r| {
            let wr = w.w(r);
            for k in a.indptr[r]..a.indptr[r+1] {
                let c = a.indices[k];
                acc[c] += K::from(a.data[k]) * wr;
            }
            acc
        })
        .reduce(|| vec![zero; cols], |mut a, b| {
            for i in 0..cols { a[i] += b[i]; }
            a
        })
}

