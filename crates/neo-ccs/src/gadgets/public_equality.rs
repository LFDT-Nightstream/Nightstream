//! Public vector equality gadget.
//!
//! Public inputs:  [ lhs[0..len), rhs[0..len) ]
//! Witness inputs: [ 1 ]
//!
//! Enforces, for each i: lhs[i] - rhs[i] = 0  (encoded as (lhs[i]-rhs[i]) * 1 = 0)

use p3_goldilocks::Goldilocks;
use p3_field::PrimeCharacteristicRing;
use crate::{CcsStructure, Mat, r1cs_to_ccs};

type F = Goldilocks;

/// Build CCS for public vector equality constraint: lhs[k] - rhs[k] = 0 for all k.
/// 
/// This gadget enforces that two public vectors are element-wise equal.
/// All values are public inputs - no sensitive data is hidden.
/// 
/// # Arguments
/// * `len` - Length of the vectors to compare
/// 
/// # Returns
/// CCS structure with `len` constraints enforcing element-wise equality
pub fn public_equality_ccs(len: usize) -> CcsStructure<F> {
    if len == 0 {
        return r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
        );
    }

    let rows = len;

    // Columns:
    //   public:  lhs[0..len), rhs[0..len)      => 2*len
    //   witness: const=1                       => 1
    let pub_cols = 2 * len;
    let wit_cols = 1;
    let cols = pub_cols + wit_cols;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let     c = vec![F::ZERO; rows * cols];

    let col_lhs0   = 0usize;
    let col_rhs0   = len;
    let col_const  = pub_cols; // single witness const

    for i in 0..len {
        let r = i;
        // (lhs[i] - rhs[i]) * 1 = 0
        a[r * cols + (col_lhs0 + i)] = F::ONE;
        a[r * cols + (col_rhs0 + i)] = -F::ONE;
        b[r * cols +  col_const     ] = F::ONE;
        // c row is 0
    }

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Build witness for public vector equality gadget
/// Simply returns [1] since all values are public inputs
pub fn build_public_vec_eq_witness() -> Vec<F> {
    vec![F::ONE]
}

/// Build CCS with multiple public equality constraints based on bindings.
/// 
/// Each binding (pub_idx, wit_idx) creates a constraint: public[pub_idx] - witness[wit_idx] = 0
/// This allows binding specific witness variables to specific public inputs.
/// 
/// Used for Nova-style circuits where certain witness values must equal public inputs.
pub fn multiple_public_equality_constraints(
    bindings: &[(usize, usize)], // (public_index, witness_index) pairs
    witness_cols: usize,
    public_cols: usize,
) -> CcsStructure<F> {
    if bindings.is_empty() {
        return r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
        );
    }

    let rows = bindings.len();
    let total_cols = public_cols + witness_cols;

    let mut a = vec![F::ZERO; rows * total_cols];
    let mut b = vec![F::ZERO; rows * total_cols];
    let c = vec![F::ZERO; rows * total_cols];

    for (row, &(pub_idx, wit_idx)) in bindings.iter().enumerate() {
        // Constraint: public[pub_idx] - witness[wit_idx] = 0
        a[row * total_cols + pub_idx] = F::ONE;                    // +public[pub_idx]
        a[row * total_cols + public_cols + wit_idx] = -F::ONE;     // -witness[wit_idx] 
        b[row * total_cols + public_cols] = F::ONE;                // ×1 (witness const=1)
    }

    let a_mat = Mat::from_row_major(rows, total_cols, a);
    let b_mat = Mat::from_row_major(rows, total_cols, b);
    let c_mat = Mat::from_row_major(rows, total_cols, c);
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}