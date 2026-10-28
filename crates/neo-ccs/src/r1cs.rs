use p3_field::Field;

use crate::{
    error::RelationError,
    matrix::Mat,
    poly::{SparsePoly, Term},
    relations::CcsStructure,
    sparse::CcsMatrix,
};

pub fn r1cs_to_ccs<F: Field>(a: Mat<F>, b: Mat<F>, c: Mat<F>) -> CcsStructure<F> {
    assert_eq!(a.rows(), b.rows());
    assert_eq!(a.rows(), c.rows());
    assert_eq!(a.cols(), b.cols());
    assert_eq!(a.cols(), c.cols());

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

    CcsStructure::new(vec![a, b, c], f_base).expect("valid R1CS→CCS structure")
}

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
