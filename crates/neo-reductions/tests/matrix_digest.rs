use neo_ccs::{CcsMatrix, CcsStructure, CscMat, GeometricRowRun, Mat, SeededPhi81LinearBlock, SparsePoly, Term};
use neo_math::{D, F};
use neo_reductions::engines::optimized_engine::SparseCache;
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

/// Structure with one nonzero at `(row, col)`; everything else (dims,
/// value, nonzero count) is held fixed so position-only mutations can
/// be isolated.
fn single_entry_structure(row: usize, col: usize) -> CcsStructure<F> {
    let mut m0 = Mat::zero(8, 12, F::ZERO);
    m0[(row, col)] = F::from_u64(42);
    CcsStructure::new(vec![m0], SparsePoly::new(1, vec![])).expect("valid CCS")
}

fn csc_vals(s: &CcsStructure<F>) -> &[F] {
    let CcsMatrix::Csc(csc) = &s.matrices[0] else {
        panic!("expected sparse matrix");
    };
    &csc.vals
}

/// A digest that only mixes dimensions, values, or nonzero counts —
/// not the sparse `(row, col)` placement — collapses these two
/// structures. The digest must bind `row_idx`.
#[test]
fn cache_aware_matrix_digest_changes_when_nonzero_moves_to_another_row() {
    let base = single_entry_structure(2, 5);
    let moved = single_entry_structure(3, 5);
    assert_eq!(csc_vals(&base), csc_vals(&moved), "value lists must be identical");

    let baseline = digest_ccs_matrices_with_sparse_cache(&base, Some(&SparseCache::build(&base)));
    let changed = digest_ccs_matrices_with_sparse_cache(&moved, Some(&SparseCache::build(&moved)));
    assert_ne!(
        baseline, changed,
        "digest must bind nonzero row placement, not just dims/values"
    );
}

/// Same as above for the column coordinate: moving the entry across
/// columns leaves `row_idx` and `vals` identical and only shifts
/// `col_ptr`. The digest must bind `col_ptr`.
#[test]
fn cache_aware_matrix_digest_changes_when_nonzero_moves_to_another_column() {
    let base = single_entry_structure(2, 5);
    let moved = single_entry_structure(2, 6);
    assert_eq!(csc_vals(&base), csc_vals(&moved), "value lists must be identical");

    let baseline = digest_ccs_matrices_with_sparse_cache(&base, Some(&SparseCache::build(&base)));
    let changed = digest_ccs_matrices_with_sparse_cache(&moved, Some(&SparseCache::build(&moved)));
    assert_ne!(
        baseline, changed,
        "digest must bind nonzero column placement, not just dims/values"
    );
}

fn geometric_structure(initial: F) -> CcsStructure<F> {
    let csc = CscMat::from_triplets(Vec::new(), 8, 108);
    let matrix = CcsMatrix::csc_with_compact_rows(
        csc,
        Vec::new(),
        vec![GeometricRowRun::new(3, 41, 41, initial, F::from_u64(3))],
    )
    .expect("valid geometric run");
    CcsStructure::new_sparse(vec![matrix], SparsePoly::new(1, vec![])).expect("valid geometric CCS")
}

#[test]
fn cache_aware_matrix_digest_binds_geometric_run_descriptors() {
    let baseline = geometric_structure(F::from_u64(7));
    let tampered = geometric_structure(F::from_u64(8));
    assert_ne!(
        digest_ccs_matrices_with_sparse_cache(&baseline, Some(&SparseCache::build(&baseline))),
        digest_ccs_matrices_with_sparse_cache(&tampered, Some(&SparseCache::build(&tampered))),
        "changing a compact coefficient must change the verifier-bound matrix digest"
    );
}

#[test]
fn cache_aware_matrix_digest_binds_seeded_phi81_word_width() {
    let seed = [0x6D; 32];
    let kappa = 1;
    let message_columns = 1;
    let (chunk_size, chunk_seeds) = neo_ajtai::seeded_pp_chunk_seeds(seed, kappa, message_columns);
    let block = |word_width| {
        SeededPhi81LinearBlock::new_with_word_width(
            0,
            vec![1],
            word_width,
            kappa,
            message_columns,
            chunk_size,
            chunk_seeds.clone(),
        )
        .expect("valid seeded Phi81 block")
    };
    let polynomial = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let structure = |block| {
        CcsStructure::new_sparse(
            vec![
                CcsMatrix::csc_with_seeded_phi81(CscMat::from_triplets(Vec::new(), D, D + 1), vec![block])
                    .expect("valid seeded matrix"),
            ],
            polynomial.clone(),
        )
        .expect("valid seeded CCS")
    };

    let balanced = structure(block(41));
    let wider = structure(block(54));
    assert_ne!(
        digest_ccs_matrices_with_sparse_cache(&balanced, None),
        digest_ccs_matrices_with_sparse_cache(&wider, None),
    );
}
