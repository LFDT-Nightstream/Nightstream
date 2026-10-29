//! Sparse R1CS value type for the batched and lookup-extended WASM relation.
//! Copied from `neo-fold-clean` (`frontends::r1cs_f_prime::SparseR1cs` and the
//! shape errors of `frontends::direct_ccs::FrontendError`) so this crate does not
//! depend on that folding stack.

use neo_ccs::CcsMatrix;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// Sparse R1CS shape for large app circuits.
#[derive(Clone, Debug)]
pub struct SparseR1cs {
    pub a: CcsMatrix<F>,
    pub b: CcsMatrix<F>,
    pub c: CcsMatrix<F>,
    pub n: usize,
    pub m: usize,
    pub m_in: usize,
}

impl SparseR1cs {
    pub fn new(
        a: CcsMatrix<F>,
        b: CcsMatrix<F>,
        c: CcsMatrix<F>,
        n: usize,
        m: usize,
        m_in: usize,
    ) -> Result<Self, FrontendError> {
        let out = Self { a, b, c, n, m, m_in };
        out.validate_shape()?;
        Ok(out)
    }

    pub fn validate_shape(&self) -> Result<(), FrontendError> {
        let (ar, ac) = (self.a.rows(), self.a.cols());
        let (br, bc) = (self.b.rows(), self.b.cols());
        let (cr, cc) = (self.c.rows(), self.c.cols());
        if ar != self.n || br != self.n || cr != self.n || ac != self.m || bc != self.m || cc != self.m {
            return Err(FrontendError::ShapeMismatch {
                a_rows: ar,
                a_cols: ac,
                b_rows: br,
                b_cols: bc,
                c_rows: cr,
                c_cols: cc,
            });
        }
        if self.m_in > self.m {
            return Err(FrontendError::PublicInputTooLarge {
                m_in: self.m_in,
                m: self.m,
            });
        }
        Ok(())
    }

    pub fn is_satisfied_by(&self, z: &[F]) -> Result<(), FrontendError> {
        if z.len() != self.m {
            return Err(FrontendError::AssignmentLength {
                got: z.len(),
                expected: self.m,
            });
        }
        let mut az = vec![F::ZERO; self.n];
        let mut bz = vec![F::ZERO; self.n];
        let mut cz = vec![F::ZERO; self.n];
        self.a.add_mul_into(z, &mut az, self.n);
        self.b.add_mul_into(z, &mut bz, self.n);
        self.c.add_mul_into(z, &mut cz, self.n);
        for row in 0..self.n {
            if az[row] * bz[row] != cz[row] {
                return Err(FrontendError::Unsatisfied { row });
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FrontendError {
    #[error(
        "R1CS shape: A is {a_rows}x{a_cols}, B is {b_rows}x{b_cols}, C is {c_rows}x{c_cols} — all three must agree"
    )]
    ShapeMismatch {
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
        c_rows: usize,
        c_cols: usize,
    },
    #[error("R1CS shape: m_in ({m_in}) > number of variables ({m})")]
    PublicInputTooLarge { m_in: usize, m: usize },
    #[error("R1CS unsatisfied at row {row}: (A·z)·(B·z) - (C·z) \u{2260} 0")]
    Unsatisfied { row: usize },
    #[error("witness assignment length {got} \u{2260} R1CS column count {expected}")]
    AssignmentLength { got: usize, expected: usize },
}
