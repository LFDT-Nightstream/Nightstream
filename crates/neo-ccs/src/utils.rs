use p3_field::Field;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

/// Validate n is a power-of-two.
pub fn validate_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Compute the tensor point r^b = ⊗_{i=1..ell} (r_i, 1-r_i) ∈ K^n where n = 2^ell.
pub fn tensor_point<K: Field>(r: &[K]) -> Vec<K> {
    let ell = r.len();
    let n = 1usize << ell;
    let mut out = vec![K::ONE; n];
    // Gray-code style expansion
    for (i, &ri) in r.iter().enumerate() {
        let stride = 1usize << i;
        let block = 1usize << (ell - i - 1);
        let one_minus = K::ONE - ri;
        let mut idx = 0usize;
        for _ in 0..block {
            for j in 0..stride {
                let a = out[idx + j];
                out[idx + j] = a * one_minus;
            }
            for j in 0..stride {
                let a = out[idx + stride + j];
                out[idx + stride + j] = a * ri;
            }
            idx += 2 * stride;
        }
    }
    out
}

/// Compute [`tensor_point`] with parallel stage updates for large domains.
pub fn tensor_point_parallel<K>(r: &[K]) -> Vec<K>
where
    K: Field + Send + Sync,
{
    let ell = r.len();
    let n = 1usize << ell;
    let mut out = vec![K::ONE; n];

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    const PAR_THRESHOLD: usize = 1 << 14;

    for (i, &ri) in r.iter().enumerate() {
        let stride = 1usize << i;
        let block_len = 2 * stride;
        let one_minus = K::ONE - ri;

        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            if n >= PAR_THRESHOLD && rayon::current_num_threads() > 1 {
                out.par_chunks_mut(block_len).for_each(|block| {
                    let (lo, hi) = block.split_at_mut(stride);
                    for slot in lo.iter_mut() {
                        *slot *= one_minus;
                    }
                    for slot in hi.iter_mut() {
                        *slot *= ri;
                    }
                });
                continue;
            }
        }

        let block_count = 1usize << (ell - i - 1);
        let mut idx = 0usize;
        for _ in 0..block_count {
            for j in 0..stride {
                out[idx + j] *= one_minus;
            }
            for j in 0..stride {
                out[idx + stride + j] *= ri;
            }
            idx += block_len;
        }
    }

    out
}

/// Multiply an F-matrix (n×m) by an F-vector (m) → F-vector (n).
pub fn mat_vec_mul_ff<F: Field>(m: &[F], n_rows: usize, n_cols: usize, v: &[F]) -> Vec<F> {
    debug_assert_eq!(n_cols, v.len());
    let mut out = vec![F::ZERO; n_rows];
    for r in 0..n_rows {
        let mut acc = F::ZERO;
        let row = &m[r * n_cols..(r + 1) * n_cols];
        for (a, b) in row.iter().zip(v.iter()) {
            acc += *a * *b;
        }
        out[r] = acc;
    }
    out
}

/// Multiply an F-matrix (d×m) by a K-vector (m) using the natural embedding F→K.
pub fn mat_vec_mul_fk<F: Field, K: Field + From<F>>(m: &[F], n_rows: usize, n_cols: usize, v: &[K]) -> Vec<K> {
    debug_assert_eq!(n_cols, v.len());
    let mut out = vec![K::ZERO; n_rows];
    for r in 0..n_rows {
        let mut acc = K::ZERO;
        let row = &m[r * n_cols..(r + 1) * n_cols];
        for (a_f, b_k) in row.iter().zip(v.iter()) {
            let a_k: K = (*a_f).into();
            acc += a_k * *b_k;
        }
        out[r] = acc;
    }
    out
}
