//! Owns a minimal R1CS-to-CCS builder for VM core lanes.

use neo_ccs::{r1cs_to_ccs, CcsStructure, Mat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Debug)]
pub struct R1csRow {
    pub a_terms: Vec<(usize, F)>,
    pub b_terms: Vec<(usize, F)>,
    pub c_terms: Vec<(usize, F)>,
}

#[derive(Clone, Debug)]
pub struct R1csBuilder {
    width: usize,
    const_one_col: usize,
    rows: Vec<R1csRow>,
}

impl R1csBuilder {
    pub fn new(width: usize, const_one_col: usize) -> Result<Self, String> {
        if width == 0 {
            return Err("R1csBuilder width must be > 0".into());
        }
        if const_one_col >= width {
            return Err(format!(
                "R1csBuilder const_one_col {} is out of range for width {}",
                const_one_col, width
            ));
        }
        Ok(Self {
            width,
            const_one_col,
            rows: Vec::new(),
        })
    }

    pub fn const_one_col(&self) -> usize {
        self.const_one_col
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn push_row(
        &mut self,
        a_terms: impl IntoIterator<Item = (usize, F)>,
        b_terms: impl IntoIterator<Item = (usize, F)>,
        c_terms: impl IntoIterator<Item = (usize, F)>,
    ) -> &mut Self {
        self.rows.push(R1csRow {
            a_terms: a_terms.into_iter().collect(),
            b_terms: b_terms.into_iter().collect(),
            c_terms: c_terms.into_iter().collect(),
        });
        self
    }

    pub fn push_linear_zero(&mut self, terms: impl IntoIterator<Item = (usize, F)>) -> &mut Self {
        self.push_row(terms, [(self.const_one_col, F::ONE)], [])
    }

    pub fn push_boolean(&mut self, col: usize) -> &mut Self {
        self.push_row([(col, F::ONE)], [(col, F::ONE), (self.const_one_col, -F::ONE)], [])
    }

    pub fn build(self) -> Result<CcsStructure<F>, String> {
        if self.rows.is_empty() {
            return Err("R1csBuilder requires at least one row".into());
        }

        let n = self.rows.len();
        let m = self.width;
        let mut a = Mat::zero(n, m, F::ZERO);
        let mut b = Mat::zero(n, m, F::ZERO);
        let mut c = Mat::zero(n, m, F::ZERO);

        for (row_idx, row) in self.rows.iter().enumerate() {
            for &(col, coeff) in &row.a_terms {
                if col >= m {
                    return Err(format!("A term column {} out of range {}", col, m));
                }
                a[(row_idx, col)] += coeff;
            }
            for &(col, coeff) in &row.b_terms {
                if col >= m {
                    return Err(format!("B term column {} out of range {}", col, m));
                }
                b[(row_idx, col)] += coeff;
            }
            for &(col, coeff) in &row.c_terms {
                if col >= m {
                    return Err(format!("C term column {} out of range {}", col, m));
                }
                c[(row_idx, col)] += coeff;
            }
        }

        Ok(r1cs_to_ccs(a, b, c))
    }
}
