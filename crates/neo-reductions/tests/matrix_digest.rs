use neo_ccs::{CcsMatrix, CcsStructure, Mat, SparsePoly};
use neo_math::F;
use neo_reductions::engines::optimized_engine::oracle::SparseCache;
use neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache;
use p3_field::PrimeCharacteristicRing;

fn sparse_two_matrix_structure() -> CcsStructure<F> {
    let n = 8usize;
    let m = 12usize;
    let mut m0 = Mat::zero(n, m, F::ZERO);
    let mut m1 = Mat::zero(n, m, F::ZERO);

    for r in 0..n {
        for c in 0..m {
            if (r + (3 * c)) % 7 == 0 {
                m0[(r, c)] = F::from_u64(((r * 11 + c * 5) % 17 + 1) as u64);
            }
            if ((5 * r) + c) % 11 == 0 {
                m1[(r, c)] = F::from_u64(((r * 13 + c * 7) % 19 + 1) as u64);
            }
        }
    }

    CcsStructure::new(vec![m0, m1], SparsePoly::new(2, vec![])).expect("valid CCS")
}

#[test]
fn cache_aware_matrix_digest_matches_with_or_without_sparse_cache() {
    let s = sparse_two_matrix_structure();
    let sparse = SparseCache::build(&s);

    let from_structure = digest_ccs_matrices_with_sparse_cache(&s, None);
    let from_cache = digest_ccs_matrices_with_sparse_cache(&s, Some(&sparse));

    assert_eq!(from_structure, from_cache);
}

#[test]
fn cache_aware_matrix_digest_changes_when_csc_value_changes() {
    let s = sparse_two_matrix_structure();
    let baseline = digest_ccs_matrices_with_sparse_cache(&s, Some(&SparseCache::build(&s)));

    let mut tampered = s.clone();
    let CcsMatrix::Csc(csc) = &mut tampered.matrices[0] else {
        panic!("expected sparse matrix");
    };
    csc.vals[0] += F::ONE;

    let changed = digest_ccs_matrices_with_sparse_cache(&tampered, Some(&SparseCache::build(&tampered)));
    assert_ne!(baseline, changed);
}

#[test]
fn cache_aware_matrix_digest_changes_when_matrix_order_changes() {
    let s = sparse_two_matrix_structure();
    let baseline = digest_ccs_matrices_with_sparse_cache(&s, Some(&SparseCache::build(&s)));

    let mut swapped = s.clone();
    swapped.matrices.swap(0, 1);

    let changed = digest_ccs_matrices_with_sparse_cache(&swapped, Some(&SparseCache::build(&swapped)));
    assert_ne!(baseline, changed);
}
