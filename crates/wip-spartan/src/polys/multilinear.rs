//! Main components:
//! - `MultilinearPolynomial`: Dense representation of multilinear polynomials, represented by evaluations over all possible binary inputs.
//! - `SparsePolynomial`: Efficient representation of sparse multilinear polynomials, storing only non-zero evaluations.

use crate::{math::Math, polys::eq::EqPolynomial, start_span};
use core::ops::Index;
use ff::PrimeField;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, info_span};

/// A multilinear extension of a polynomial $Z(\cdot)$, denote it as $\tilde{Z}(x_1, ..., x_m)$
/// where the degree of each variable is at most one.
///
/// This is the dense representation of a multilinear poynomial.
/// Let it be $\mathbb{G}(\cdot): \mathbb{F}^m \rightarrow \mathbb{F}$, it can be represented uniquely by the list of
/// evaluations of $\mathbb{G}(\cdot)$ over the Boolean hypercube $\{0, 1\}^m$.
///
/// For example, a 3 variables multilinear polynomial can be represented by evaluation
/// at points $[0, 2^3-1]$.
///
/// The implementation follows
/// $$
/// \tilde{Z}(x_1, ..., x_m) = \sum_{e\in {0,1}^m}Z(e) \cdot \prod_{i=1}^m(x_i \cdot e_i + (1-x_i) \cdot (1-e_i))
/// $$
///
/// Vector $Z$ indicates $Z(e)$ where $e$ ranges from $0$ to $2^m-1$.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultilinearPolynomial<Scalar: PrimeField> {
  pub(crate) Z: Vec<Scalar>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

impl<Scalar: PrimeField> MultilinearPolynomial<Scalar> {
  /// Creates a new `MultilinearPolynomial` from the given evaluations.
  ///
  /// # Panics
  /// The number of evaluations must be a power of two.
  pub fn new(Z: Vec<Scalar>) -> Self {
    MultilinearPolynomial { Z }
  }

  /// Binds the polynomial's top variable using the given scalar.
  ///
  /// This operation modifies the polynomial in-place.
  pub fn bind_poly_var_top(&mut self, r: &Scalar) {
    assert!(
      self.Z.len() >= 2,
      "Vector Z must have at least two elements to bind the top variable."
    );

    let n = self.Z.len() / 2;

    let (left, right) = self.Z.split_at_mut(n);

    if crate::parallel::parallelism_enabled() {
      zip_with_for_each!((left.par_iter_mut(), right.par_iter()), |a, b| {
        *a += *r * (*b - *a);
      });
    } else {
      for (a, b) in left.iter_mut().zip(right.iter()) {
        *a += *r * (*b - *a);
      }
    }

    self.Z.truncate(n);
  }

  /// binds the polynomial's top variables using the given scalars.
  pub fn bind_with(poly: &[Scalar], L: &[Scalar], r_len: usize) -> Vec<Scalar> {
    assert_eq!(
      poly.len(),
      L.len() * r_len,
      "poly length ({}) must equal L.len() * r_len ({} * {}) = {}",
      poly.len(),
      L.len(),
      r_len,
      L.len() * r_len
    );

    if crate::parallel::parallelism_enabled() {
      (0..r_len)
        .into_par_iter()
        .map(|i| {
          let mut acc = Scalar::ZERO;
          for j in 0..L.len() {
            // row-major: index = j * r_len + i
            acc += L[j] * poly[j * r_len + i];
          }
          acc
        })
        .collect()
    } else {
      (0..r_len)
        .map(|i| {
          let mut acc = Scalar::ZERO;
          for j in 0..L.len() {
            acc += L[j] * poly[j * r_len + i];
          }
          acc
        })
        .collect()
    }
  }

  /// Evaluates the polynomial at the given point.
  /// Returns Z(r) in O(n) time.
  ///
  /// The point must have a value for each variable.
  #[allow(dead_code)]
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    // r must have a value for each variable
    let chis = EqPolynomial::evals_from_points(r);

    if crate::parallel::parallelism_enabled() {
      zip_with!(
        (chis.into_par_iter(), self.Z.par_iter()),
        |chi_i, Z_i| chi_i * Z_i
      )
      .sum()
    } else {
      chis
        .iter()
        .zip(self.Z.iter())
        .map(|(chi_i, z_i)| *chi_i * *z_i)
        .sum()
    }
  }

  /// Evaluates the polynomial with the given evaluations and point.
  pub fn evaluate_with(Z: &[Scalar], r: &[Scalar]) -> Scalar {
    let (_eval_span, eval_t) =
      start_span!("multilinear_evaluate_with", vars = r.len(), evals = Z.len());

    let evals = EqPolynomial::evals_from_points(r);
    let result = if crate::parallel::parallelism_enabled() {
      zip_with!((evals.into_par_iter(), Z.par_iter()), |a, b| a * b).sum()
    } else {
      evals.iter().zip(Z.iter()).map(|(a, b)| *a * *b).sum()
    };

    info!(elapsed_ms = %eval_t.elapsed().as_millis(), vars = r.len(), evals = Z.len(), "multilinear_evaluate_with");
    result
  }
}

impl<Scalar: PrimeField> Index<usize> for MultilinearPolynomial<Scalar> {
  type Output = Scalar;

  #[inline(always)]
  fn index(&self, _index: usize) -> &Scalar {
    &(self.Z[_index])
  }
}

/// Sparse multilinear polynomial, which means the $Z(\cdot)$ is zero at most points.
/// In our context, sparse polynomials are non-zeros over the hypercube at locations that map to "small" integers
/// We exploit this property to implement a time-optimal algorithm
pub(crate) struct SparsePolynomial<Scalar: PrimeField> {
  num_vars: usize,
  Z: Vec<Scalar>,
}

impl<Scalar: PrimeField> SparsePolynomial<Scalar> {
  pub fn new(num_vars: usize, Z: Vec<Scalar>) -> Self {
    SparsePolynomial { num_vars, Z }
  }

  // a time-optimal algorithm to evaluate sparse polynomials
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    assert_eq!(self.num_vars, r.len());

    // Guard against zero-dimension case
    if self.num_vars == 0 {
      return if self.Z.is_empty() {
        Scalar::ZERO
      } else {
        self.Z[0]
      };
    }

    let num_vars_z = self.Z.len().next_power_of_two().log_2();

    // Guard against underflow: ensure we have enough dimensions
    if self.num_vars < 1 + num_vars_z {
      // Fallback to full evaluation for edge cases
      let mut extended_z = self.Z.clone();
      extended_z.resize(1 << self.num_vars, Scalar::ZERO);
      return crate::polys::multilinear::MultilinearPolynomial::new(extended_z).evaluate(r);
    }

    let chis = EqPolynomial::evals_from_points(&r[self.num_vars - 1 - num_vars_z..]);
    let eval_partial: Scalar = self
      .Z
      .iter()
      .zip(chis.iter())
      .map(|(z, chi)| *z * *chi)
      .sum();

    let common = (0..self.num_vars - 1 - num_vars_z)
      .map(|i| Scalar::ONE - r[i])
      .product::<Scalar>();

    common * eval_partial
  }
}
