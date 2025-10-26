//! Tests for equality-weighted CSR operations in pi_ccs module
//! 
//! These tests verify the correctness of:
//! - Half-table χ_r weight computation
//! - Weighted sparse matrix-vector multiplication

use neo_fold::pi_ccs::{HalfTableEq, to_csr, spmv_csr_t_weighted_fk, RowWeight};
use neo_math::{F, K};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng, rand_core::RngCore};
use p3_field::PrimeCharacteristicRing;

fn rand_f(rng: &mut ChaCha8Rng) -> F {
    F::from_u64(rng.next_u64())
}

fn rand_k(rng: &mut ChaCha8Rng) -> K {
    let re = F::from_u64(rng.next_u64());
    let im = F::from_u64(rng.next_u64());
    neo_math::from_complex(re, im)
}

#[test]
fn half_table_weights_match_tensor_point() {
    let ell = 5usize; // n = 32
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);
    let mut r = Vec::with_capacity(ell);
    for _ in 0..ell { r.push(rand_k(&mut rng)); }
    let w = HalfTableEq::new(&r);
    let rb = neo_ccs::utils::tensor_point::<K>(&r);
    assert_eq!(rb.len(), 1usize << ell);
    for row in 0..rb.len() {
        assert_eq!(w.w(row), rb[row], "row weight mismatch at {}", row);
    }
}

#[test]
fn weighted_spmv_matches_dense_spmv_with_tensor_point() {
    let ell = 4usize; // n = 16 rows for manageable test
    let rows = 1usize << ell;
    let cols = 7usize;
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0DE);
    // Build a random sparse matrix over F
    let mut m = neo_ccs::Mat::zero(rows, cols, F::ZERO);
    // ~25% density
    for r in 0..rows {
        for c in 0..cols {
            let coin = rng.next_u32() % 4;
            if coin == 0 { m[(r, c)] = rand_f(&mut rng); }
        }
    }
    let csr = to_csr::<F>(&m, rows, cols);

    // Random r and corresponding weights
    let mut r_vec = vec![K::ZERO; ell];
    for i in 0..ell { r_vec[i] = rand_k(&mut rng); }
    let rb = neo_ccs::utils::tensor_point::<K>(&r_vec);
    let w = HalfTableEq::new(&r_vec);

    // Compute via dense χ_r vector directly from CSR (no helper)
    let mut v_dense = vec![K::ZERO; cols];
    for r_i in 0..rows {
        let wr = rb[r_i];
        let start = csr.indptr[r_i];
        let end = csr.indptr[r_i + 1];
        for k in start..end {
            let c = csr.indices[k];
            v_dense[c] += K::from(csr.data[k]) * wr;
        }
    }
    // Compute via weighted streaming
    let v_weighted = spmv_csr_t_weighted_fk(&csr, &w);
    assert_eq!(v_dense.len(), cols);
    assert_eq!(v_weighted.len(), cols);
    for c in 0..cols { assert_eq!(v_dense[c], v_weighted[c], "col {} mismatch", c); }
}
