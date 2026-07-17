//! Tiny external fixtures for the borrowed selective-structure census.

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, GeometricRowRun, SeededPhi81LinearBlock, SparsePoly};
use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveMatrixTag, SelectiveStructureCensus, SelectiveStructureCensusError,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const PORTS: usize = 13;
const ROWS: usize = neo_math::D;
const COLUMNS: usize = 4;

fn plain_csc() -> CscMat<F> {
    CscMat::from_triplets(vec![(0, 0, F::ONE), (2, 1, F::from_u64(7))], ROWS, COLUMNS)
}

fn compact_matrix() -> CcsMatrix<F> {
    let block = SeededPhi81LinearBlock::new_with_word_width(0, vec![1], 1, 1, 1, 1, vec![vec![[0xa5; 32]]])
        .expect("tiny seeded block");
    let run = GeometricRowRun::new(ROWS - 1, 2, 2, F::from_u64(3), F::from_u64(5));
    CcsMatrix::csc_with_compact_rows(plain_csc(), vec![block], vec![run]).expect("tiny compact matrix")
}

fn compact_structure() -> CcsStructure<F> {
    let plain = CcsMatrix::Csc(plain_csc());
    let mut matrices = vec![plain.clone(); PORTS];
    matrices[1] = compact_matrix();
    CcsStructure::new_sparse(matrices, SparsePoly::new(PORTS, vec![])).expect("tiny selective structure")
}

#[test]
fn census_borrows_structure_and_counts_compact_payloads() {
    let structure = compact_structure();
    let census = SelectiveStructureCensus::new(&structure).expect("production-shaped census");

    assert!(core::ptr::eq(census.structure(), &structure));
    assert_eq!(census.port_count(), PORTS);
    assert_eq!(census.ports().len(), PORTS);

    let plain = census.port(0).expect("plain port");
    assert_eq!(plain.port(), 0);
    assert_eq!(plain.tag(), SelectiveMatrixTag::Csc);
    assert_eq!((plain.rows(), plain.columns()), (ROWS, COLUMNS));
    assert_eq!(plain.col_ptr_len(), COLUMNS + 1);
    assert_eq!(plain.nnz(), 2);
    assert_eq!(plain.seeded_block_count(), 0);
    assert_eq!(plain.seeded_metadata_bytes(), 0);
    assert_eq!(plain.geometric_run_count(), 0);
    assert_eq!(plain.conservative_raw_wire_bytes(), 108);

    let compact = census.port(1).expect("compact port");
    assert_eq!(compact.tag(), SelectiveMatrixTag::CscWithSeededPhi81);
    assert_eq!(compact.col_ptr_len(), COLUMNS + 1);
    assert_eq!(compact.nnz(), 2);
    assert_eq!(compact.seeded_block_count(), 1);
    assert_eq!(compact.seeded_metadata_bytes(), 112);
    assert_eq!(compact.geometric_run_count(), 1);
    assert_eq!(compact.conservative_raw_wire_bytes(), 260);

    assert_eq!(
        census.conservative_raw_wire_bytes(),
        3 * 8 + 12 * plain.conservative_raw_wire_bytes() + compact.conservative_raw_wire_bytes()
    );
}

#[test]
fn exact_tag_covers_all_rust_matrix_variants_and_production_rejects_identity() {
    let identity = CcsMatrix::Identity { n: 2 };
    let plain = CcsMatrix::Csc(plain_csc());
    let compact = compact_matrix();
    assert_eq!(SelectiveMatrixTag::from_matrix(&identity), SelectiveMatrixTag::Identity);
    assert_eq!(SelectiveMatrixTag::from_matrix(&plain), SelectiveMatrixTag::Csc);
    assert_eq!(
        SelectiveMatrixTag::from_matrix(&compact),
        SelectiveMatrixTag::CscWithSeededPhi81
    );

    let structure =
        CcsStructure::new_sparse(vec![identity; PORTS], SparsePoly::new(PORTS, vec![])).expect("identity fixture");
    assert_eq!(
        SelectiveStructureCensus::new(&structure).expect_err("identity is not a selective production port"),
        SelectiveStructureCensusError::IdentityPort { port: 0 }
    );
}

fn first_csc_mut(structure: &mut CcsStructure<F>) -> &mut CscMat<F> {
    match &mut structure.matrices[0] {
        CcsMatrix::Csc(csc) => csc,
        _ => panic!("fixture port zero must be CSC"),
    }
}

#[test]
fn census_rejects_nonmonotone_column_pointers() {
    let mut structure = compact_structure();
    let csc = first_csc_mut(&mut structure);
    csc.col_ptr[1] = 2;
    csc.col_ptr[2] = 1;

    assert!(matches!(
        SelectiveStructureCensus::new(&structure),
        Err(SelectiveStructureCensusError::ColumnPointerRange {
            port: 0,
            column: 1,
            start: 2,
            end: 1,
            nnz: 2,
        })
    ));
}

#[test]
fn census_rejects_noncanonical_or_out_of_range_rows() {
    let mut unordered = compact_structure();
    let csc = first_csc_mut(&mut unordered);
    csc.col_ptr = vec![0, 2, 2, 2, 2];
    csc.row_idx = vec![2, 1];
    assert!(matches!(
        SelectiveStructureCensus::new(&unordered),
        Err(SelectiveStructureCensusError::RowIndexOrder {
            port: 0,
            column: 0,
            entry: 1,
            previous: 2,
            row: 1,
        })
    ));

    let mut out_of_range = compact_structure();
    first_csc_mut(&mut out_of_range).row_idx[0] = ROWS as u32;
    assert!(matches!(
        SelectiveStructureCensus::new(&out_of_range),
        Err(SelectiveStructureCensusError::RowIndexOutOfBounds {
            port: 0,
            column: 0,
            entry: 0,
            row: ROWS,
            rows: ROWS,
        })
    ));
}

#[test]
fn census_rejects_explicit_zero_storage() {
    let mut structure = compact_structure();
    first_csc_mut(&mut structure).vals[0] = F::ZERO;
    assert_eq!(
        SelectiveStructureCensus::new(&structure).expect_err("zero CSC term must fail closed"),
        SelectiveStructureCensusError::ExplicitZero {
            port: 0,
            column: 0,
            entry: 0,
        }
    );
}
