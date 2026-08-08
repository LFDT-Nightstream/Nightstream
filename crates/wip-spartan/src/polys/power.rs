//! `PowPolynomial`: Represents multilinear extension of power polynomials

use crate::errors::SpartanError;
use core::iter::successors;
use ff::PrimeField;

/// Represents the multilinear extension polynomial (MLE) of the equality polynomial $pow(x,t)$, denoted as $\tilde{pow}(x, t)$.
///
/// The polynomial is defined by the formula:
/// $$
/// \tilde{power}(x, t) = \prod_{i=1}^m(1 + (t^{2^i} - 1) * x_i)
/// $$
#[allow(dead_code)]
pub struct PowPolynomial<Scalar: PrimeField> {
  t_pow: Vec<Scalar>,
}

impl<Scalar: PrimeField> PowPolynomial<Scalar> {
  /// Creates a new `PowPolynomial` from a Scalars `t`.
  pub fn new(t: &Scalar, ell: usize) -> Self {
    // t_pow = [t^{2^0}, t^{2^1}, ..., t^{2^{ell-1}}]
    let t_pow = successors(Some(*t), |p: &Scalar| Some(p.square()))
      .take(ell)
      .collect::<Vec<_>>();

    PowPolynomial { t_pow }
  }

  /// Evaluates the polynomial at a given point `r`.
  pub fn evaluate(&self, r: &[Scalar]) -> Result<Scalar, SpartanError> {
    if r.len() != self.t_pow.len() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "PowPolynomial: Expected {} elements in r, got {}",
          self.t_pow.len(),
          r.len()
        ),
      });
    }

    let mut acc = Scalar::ONE;
    for (i, &r_i) in r.iter().rev().enumerate() {
      acc *= Scalar::ONE + (self.t_pow[i] - Scalar::ONE) * r_i;
    }
    Ok(acc)
  }

  /// Computes two vectors such that their outer product equals the output of the `evals` function.
  /// This code ensures
  pub fn split_evals(&self, len_left: usize, len_right: usize) -> Vec<Scalar> {
    // Compute the number of elements in the left and right halves
    let ell = self.t_pow.len();
    assert_eq!(len_left * len_right, 1 << ell);

    let t = self.t_pow[0];

    // Compute the left and right halves of the evaluations
    // left = [1, t, t^2, ..., t^{2^{ell/2} - 1}]
    let left = successors(Some(Scalar::ONE), |p| Some(*p * t))
      .take(len_left)
      .collect::<Vec<_>>();

    // right = [1, t^{2^{ell/2}}, t^{2^{ell/2 + 1}}, ..., t^{2^{ell} - 1}]
    // take the last entry from left, multiply with t to get the second entry in right
    let left_last_times_t = left[left.len() - 1] * t;
    let mut right = vec![Scalar::ONE; len_right];
    right[0] = Scalar::ONE;
    right[1] = left_last_times_t;
    for i in 2..len_right {
      right[i] = right[i - 1] * left_last_times_t;
    }

    [left, right].concat()
  }
}
