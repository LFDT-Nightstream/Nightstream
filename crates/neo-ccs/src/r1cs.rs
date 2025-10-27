use p3_field::Field;

use crate::{matrix::Mat, poly::{SparsePoly, Term}, relations::CcsStructure};

/// Minimal **R1CS → CCS** helper: given A, B, C ∈ F^{n×m}, produce CCS with
/// M_1=A, M_2=B, M_3=C and f(X1,X2,X3) = X1·X2 − X3 (elementwise).
///
/// This is the standard embedding: row-wise, `A z ∘ B z = C z`, i.e., `f=0`.
pub fn r1cs_to_ccs<F: Field>(a: Mat<F>, b: Mat<F>, c: Mat<F>) -> CcsStructure<F> {
    assert_eq!(a.rows(), b.rows());
    assert_eq!(a.rows(), c.rows());
    assert_eq!(a.cols(), b.cols());
    assert_eq!(a.cols(), c.cols());

    let n = a.rows();
    let m = a.cols();

    // Base polynomial f(X1,X2,X3) = X1 * X2 - X3
    let base_terms = vec![
        Term { coeff: F::ONE,    exps: vec![1, 1, 0] }, // X1 * X2
        Term { coeff: -F::ONE,   exps: vec![0, 0, 1] }, // -X3
    ];
    let f_base = SparsePoly::new(3, base_terms);

    // Insert identity-first only when square; otherwise keep legacy 3-matrix CCS.
    if n == m {
        let i_n = Mat::<F>::identity(n);
        let f = f_base.insert_var_at_front();
        CcsStructure::new(vec![i_n, a, b, c], f).expect("valid identity-first CCS structure")
    } else {
        CcsStructure::new(vec![a, b, c], f_base).expect("valid R1CS→CCS structure")
    }
}
