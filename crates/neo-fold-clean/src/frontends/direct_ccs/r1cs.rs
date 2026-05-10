//! User-facing R1CS shape for the direct-CCS frontend.
//!
//! `R1cs { a, b, c, m_in }` represents the standard R1CS relation
//! `(A·z) ∘ (B·z) = (C·z)` with `z = [x | w]` of length `a.cols()`,
//! split into `m_in`-element public input `x` and witness `w`.

use neo_ccs::{r1cs_to_ccs, Mat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::frontends::direct_ccs::FrontendError;
use crate::paper::relations::Structure;

/// User-facing R1CS shape.
///
/// All three matrices must share the same `n × m` shape. `n` is the
/// number of constraints, `m` is the assignment length (`= |x| + |w|`).
#[derive(Clone, Debug)]
pub struct R1cs {
    pub a: Mat<F>,
    pub b: Mat<F>,
    pub c: Mat<F>,
    /// Public-input split point: `z[..m_in] = x` is public, `z[m_in..] = w` is private.
    pub m_in: usize,
}

impl R1cs {
    /// Validate that A, B, C share shape and that `m_in ≤ m`.
    pub fn validate_shape(&self) -> Result<(), FrontendError> {
        let (ar, ac) = (self.a.rows(), self.a.cols());
        let (br, bc) = (self.b.rows(), self.b.cols());
        let (cr, cc) = (self.c.rows(), self.c.cols());
        if ar != br || ar != cr || ac != bc || ac != cc {
            return Err(FrontendError::ShapeMismatch {
                a_rows: ar,
                a_cols: ac,
                b_rows: br,
                b_cols: bc,
                c_rows: cr,
                c_cols: cc,
            });
        }
        if self.m_in > ac {
            return Err(FrontendError::PublicInputTooLarge { m_in: self.m_in, m: ac });
        }
        Ok(())
    }

    /// Number of variables in the assignment (`= |x| + |w|`).
    pub fn m(&self) -> usize {
        self.a.cols()
    }

    /// Number of R1CS constraints (rows in A, B, C).
    pub fn n(&self) -> usize {
        self.a.rows()
    }

    /// Translate this R1CS to the paper-layer `Structure` via the standard
    /// `(A·z) ∘ (B·z) = (C·z)` embedding (Definition 11 with three matrices
    /// and `f(X0, X1, X2) = X0·X1 - X2`).
    pub fn to_structure(&self) -> Structure {
        r1cs_to_ccs(self.a.clone(), self.b.clone(), self.c.clone())
    }

    /// Verify that an assignment satisfies the R1CS, row-by-row. Returns
    /// `Err(Unsatisfied { row })` at the first offending row, so users
    /// learn about witness bugs *before* they hit Π_CCS sumcheck.
    ///
    /// Uses an `&[F]` witness; field arithmetic, no commitment.
    pub fn is_satisfied_by(&self, z: &[F]) -> Result<(), FrontendError> {
        if z.len() != self.m() {
            return Err(FrontendError::AssignmentLength {
                got: z.len(),
                expected: self.m(),
            });
        }
        for row in 0..self.n() {
            let az = row_dot(&self.a, row, z);
            let bz = row_dot(&self.b, row, z);
            let cz = row_dot(&self.c, row, z);
            if az * bz != cz {
                return Err(FrontendError::Unsatisfied { row });
            }
        }
        Ok(())
    }
}

/// Compute `(M · z)[row]` = `Σ_j M[row, j] · z[j]`.
fn row_dot(m: &Mat<F>, row: usize, z: &[F]) -> F {
    let mut acc = F::ZERO;
    for j in 0..m.cols() {
        let coeff = m[(row, j)];
        if coeff != F::ZERO {
            acc += coeff * z[j];
        }
    }
    acc
}
