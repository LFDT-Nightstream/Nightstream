//! Ajtai commitment gadgets: opening and homomorphic linear combination.
//!
//! ## Opening gadget
//! Public:  c_open[0..L) (L = number of coordinates being opened)
//! Witness: [1, Z_digits[0..msg_len)]
//! Constraints: for every i,  <L_i, Z_digits> = c_open[i]
//!
//! ## Lincomb gadget
//! Public:  [ ρ, c_prev[0..L), c_step[0..L), c_next[0..L) ]
//! Witness: [ 1, u[0..L) ]
//! Constraints:
//!   - u[i] = ρ * c_step[i]
//!   - c_next[i] - c_prev[i] - u[i] = 0

use crate::{r1cs_to_ccs, CcsStructure, Mat};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

/// Build an opening gadget from precomputed rows (L_i).
/// Each `rows[i]` is the vector of coefficients for coordinate i,
/// applied to the digit-decomposed witness Z (length `msg_len`).
///
/// # Arguments
/// * `rows` - Precomputed L_i rows from Ajtai public parameters
/// * `msg_len` - Length of the digit-decomposed witness (d * m)
///
/// # Returns  
/// CCS structure enforcing c_open[i] = <L_i, Z_digits> for all i
///
/// # Public Layout
/// Public inputs:  c_open[0..L)
/// # Witness Layout  
/// Witness inputs: [1, Z_digits[0..msg_len)]
pub fn commitment_opening_from_rows_ccs(rows: &[Vec<F>], msg_len: usize) -> CcsStructure<F> {
    let l = rows.len();
    if l == 0 || msg_len == 0 {
        return r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
        );
    }

    // Validate row lengths
    #[cfg(debug_assertions)]
    for (i, r) in rows.iter().enumerate() {
        assert!(
            r.len() == msg_len,
            "rows[{}].len() = {} != msg_len = {}",
            i,
            r.len(),
            msg_len
        );
    }

    let rows_cnt = l;

    // Columns:
    //   public:   c_open[0..l)               => l
    //   witness:  const=1 | Z_digits[0..m)   => 1 + msg_len
    let pub_cols = l;
    let wit_cols = 1 + msg_len;
    let cols = pub_cols + wit_cols;

    let mut a = vec![F::ZERO; rows_cnt * cols];
    let mut b = vec![F::ZERO; rows_cnt * cols];
    let c = vec![F::ZERO; rows_cnt * cols];

    let col_c_open0 = 0usize;
    let col_const = pub_cols;
    let col_z0 = pub_cols + 1;

    // For each i: ( <L_i, Z> - c_open[i] ) * 1 = 0
    for i in 0..l {
        let r = i;

        // - c_open[i]
        a[r * cols + (col_c_open0 + i)] = -F::ONE;

        // + sum_j L_i[j] * Z[j]
        for (j, coeff) in rows[i].iter().enumerate() {
            if *coeff != F::ZERO {
                a[r * cols + (col_z0 + j)] = *coeff;
            }
        }

        // × 1
        b[r * cols + col_const] = F::ONE;
        // c row is 0
    }

    let a_mat = Mat::from_row_major(rows_cnt, cols, a);
    let b_mat = Mat::from_row_major(rows_cnt, cols, b);
    let c_mat = Mat::from_row_major(rows_cnt, cols, c);
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Homomorphic commitment linear combination:
/// checks, for all i:   c_next[i] = c_prev[i] + ρ * c_step[i]
///
/// # Arguments
/// * `commit_len` - Length of commitment coordinate vectors (L)
///
/// # Returns
/// CCS structure enforcing the linear combination with 2*L constraints
///
/// # Public Layout
/// Public inputs:  [ ρ, c_prev[0..l), c_step[0..l), c_next[0..l) ]
/// # Witness Layout
/// Witness inputs: [ 1, u[0..l) ]  with u[i] = ρ * c_step[i]
pub fn commitment_lincomb_ccs(commit_len: usize) -> CcsStructure<F> {
    let l = commit_len;
    if l == 0 {
        return r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
        );
    }

    // We'll mirror the public-ρ EV pattern so the same ρ can be passed publicly.
    // Rows: 2*l  (first l for mult, second l for linear)
    let rows = 2 * l;

    // Columns:
    //   public:  ρ | c_prev[l] | c_step[l] | c_next[l]   => 1 + 3*l
    //   witness: const=1 | u[l]                           => 1 + l
    let pub_cols = 1 + 3 * l;
    let wit_cols = 1 + l;
    let cols = pub_cols + wit_cols;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let mut c = vec![F::ZERO; rows * cols];

    // Public column offsets
    let col_rho = 0usize;
    let col_prev0 = 1usize;
    let col_step0 = 1 + l;
    let col_next0 = 1 + 2 * l;

    // Witness column offsets
    let col_const = pub_cols;
    let col_u0 = pub_cols + 1;

    // Rows 0..l-1: u[i] = ρ * c_step[i]
    for i in 0..l {
        let r = i;
        a[r * cols + col_rho] = F::ONE; // ρ
        b[r * cols + (col_step0 + i)] = F::ONE; // c_step[i]
        c[r * cols + (col_u0 + i)] = F::ONE; // u[i]
    }

    // Rows l..2*l-1: c_next[i] - c_prev[i] - u[i] = 0 (× 1)
    for i in 0..l {
        let r = l + i;
        a[r * cols + (col_next0 + i)] = F::ONE; // + c_next[i]
        a[r * cols + (col_prev0 + i)] = -F::ONE; // - c_prev[i]
        a[r * cols + (col_u0 + i)] = -F::ONE; // - u[i]
        b[r * cols + col_const] = F::ONE; // × 1
                                          // c row is 0
    }

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Build witness for Ajtai commitment opening.
///
/// Takes the base-b digits of the step witness and returns the opening witness.
/// The public input (c_step coordinates) must be computed separately.
///
/// Returns: [1, Z_digits[0..msg_len]]
pub fn build_opening_witness(z_digits: &[F]) -> Vec<F> {
    let mut w = Vec::with_capacity(1 + z_digits.len());
    w.push(F::ONE);
    w.extend_from_slice(z_digits);
    w
}

/// Build witness for commitment linear combination.
///
/// Computes the folded commitment c_next = c_prev + ρ * c_step and returns
/// both the witness and the resulting c_next coordinates.
///
/// Returns: (witness, c_next) where witness = [1, u[0..L]] and u[i] = ρ * c_step[i]
pub fn build_commitment_lincomb_witness(rho: F, c_prev: &[F], c_step: &[F]) -> (Vec<F>, Vec<F>) {
    assert_eq!(c_prev.len(), c_step.len(), "commitment length mismatch");
    let l = c_prev.len();
    let mut u = vec![F::ZERO; l];
    let mut c_next = vec![F::ZERO; l];
    for i in 0..l {
        u[i] = rho * c_step[i];
        c_next[i] = c_prev[i] + u[i];
    }
    let mut w = Vec::with_capacity(1 + l);
    w.push(F::ONE);
    w.extend_from_slice(&u);
    (w, c_next)
}

/// Build public input for commitment linear combination.
///
/// Returns the complete public input vector: [ρ, c_prev[0..L], c_step[0..L], c_next[0..L]]
pub fn build_commitment_lincomb_public_input(rho: F, c_prev: &[F], c_step: &[F], c_next: &[F]) -> Vec<F> {
    assert_eq!(c_prev.len(), c_step.len(), "commitment length mismatch");
    assert_eq!(c_prev.len(), c_next.len(), "commitment length mismatch");

    let l = c_prev.len();
    let mut public_input = Vec::with_capacity(1 + 3 * l);
    public_input.push(rho);
    public_input.extend_from_slice(c_prev);
    public_input.extend_from_slice(c_step);
    public_input.extend_from_slice(c_next);
    public_input
}
