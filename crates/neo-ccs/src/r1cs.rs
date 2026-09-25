use p3_field::Field;

use crate::{
    error::RelationError,
    matrix::Mat,
    poly::{SparsePoly, Term},
    relations::CcsStructure,
    sparse::CcsMatrix,
};

/// Minimal **R1CS → CCS** helper: given A, B, C ∈ F^{n×m}, produce CCS with
/// M_0=A, M_1=B, M_2=C and f(X0,X1,X2) = X0·X1 − X2 (elementwise).
///
/// This is the standard embedding: row-wise, `A z ∘ B z = C z`, i.e., `f=0`.
pub fn r1cs_to_ccs<F: Field>(a: Mat<F>, b: Mat<F>, c: Mat<F>) -> CcsStructure<F> {
    assert_eq!(a.rows(), b.rows());
    assert_eq!(a.rows(), c.rows());
    assert_eq!(a.cols(), b.cols());
    assert_eq!(a.cols(), c.cols());

    // Base polynomial f(X0,X1,X2) = X0 * X1 - X2
    let base_terms = vec![
        Term {
            coeff: F::ONE,
            exps: vec![1, 1, 0],
        }, // X1 * X2
        Term {
            coeff: -F::ONE,
            exps: vec![0, 0, 1],
        }, // -X3
    ];
    let f_base = SparsePoly::new(3, base_terms);

    CcsStructure::new(vec![a, b, c], f_base).expect("valid R1CS→CCS structure")
}

/// Sparse **R1CS -> CCS** helper.
///
/// This is the same row-wise embedding as [`r1cs_to_ccs`], but it preserves sparse
/// matrices instead of materializing dense zeros: `M_0=A`, `M_1=B`, `M_2=C` and
/// `f(X0,X1,X2) = X0 * X1 - X2`.
pub fn sparse_r1cs_to_ccs<F>(
    a: CcsMatrix<F>,
    b: CcsMatrix<F>,
    c: CcsMatrix<F>,
) -> Result<CcsStructure<F>, RelationError>
where
    F: Field,
{
    if a.rows() != b.rows() || a.rows() != c.rows() || a.cols() != b.cols() || a.cols() != c.cols() {
        return Err(RelationError::InvalidStructure);
    }

    let base_terms = vec![
        Term {
            coeff: F::ONE,
            exps: vec![1, 1, 0],
        },
        Term {
            coeff: -F::ONE,
            exps: vec![0, 0, 1],
        },
    ];
    let f_base = SparsePoly::new(3, base_terms);

    CcsStructure::new_sparse(vec![a, b, c], f_base)
}

/// Sparse native-selector R1CS embedding.
///
/// Given row-aligned sparse matrices `A`, `B`, `C`, and `S`, this constructs
/// the four-matrix CCS relation
///
/// `S z * ((A z) * (B z) - C z) = 0`.
///
/// A selector value of one enforces the source R1CS row. A selector value of
/// zero disables that row. The selector is a CCS matrix input. This
/// construction allocates no residual witness and adds no second row.
pub fn sparse_selected_r1cs_to_ccs<F>(
    a: CcsMatrix<F>,
    b: CcsMatrix<F>,
    c: CcsMatrix<F>,
    selector: CcsMatrix<F>,
) -> Result<CcsStructure<F>, RelationError>
where
    F: Field,
{
    let rows = a.rows();
    let columns = a.cols();
    if b.rows() != rows
        || c.rows() != rows
        || selector.rows() != rows
        || b.cols() != columns
        || c.cols() != columns
        || selector.cols() != columns
    {
        return Err(RelationError::InvalidStructure);
    }

    let polynomial = SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 1, 0, 1],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 1, 1],
            },
        ],
    );

    CcsStructure::new_sparse(vec![a, b, c, selector], polynomial)
}
