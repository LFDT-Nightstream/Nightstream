use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use serde::{Deserialize, Serialize};

use crate::error::{AjtaiError, AjtaiResult};

/// Public parameters for Ajtai: M ∈ R_q^{κ×m}, stored row-major.
/// Outside this crate `new` is the only constructor, so every value has its
/// declared shape.
#[derive(Clone, Debug)]
pub struct PP<RqEl> {
    pub(crate) kappa: usize,
    pub(crate) m: usize,
    pub(crate) d: usize,
    /// Ajtai matrix rows; each row is a vector of ring elements of length m.
    pub(crate) m_rows: Vec<Vec<RqEl>>,
}

impl<RqEl> PP<RqEl> {
    /// Require `d == D`, nonzero `kappa` and `m`, `kappa` rows, and `m`
    /// ring elements in every row.
    pub fn new(d: usize, kappa: usize, m: usize, m_rows: Vec<Vec<RqEl>>) -> AjtaiResult<Self> {
        if d != neo_math::ring::D || kappa == 0 || m == 0 {
            return Err(AjtaiError::InvalidDimensions(
                "Ajtai parameters need d = D and nonzero kappa and m".to_string(),
            ));
        }
        if m_rows.len() != kappa || m_rows.iter().any(|row| row.len() != m) {
            return Err(AjtaiError::InvalidDimensions(
                "Ajtai matrix rows do not match kappa x m".to_string(),
            ));
        }
        Ok(Self { kappa, m, d, m_rows })
    }

    pub fn kappa(&self) -> usize {
        self.kappa
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn rows(&self) -> &[Vec<RqEl>] {
        &self.m_rows
    }
}

/// Commitment c ∈ F_q^{d×κ}, stored as column-major flat matrix (κ columns, each length d).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment {
    pub d: usize,
    pub kappa: usize,
    /// data[c * d + i] = i-th row of column c
    pub data: Vec<Fq>,
}

impl Commitment {
    pub fn zeros(d: usize, kappa: usize) -> Self {
        Self {
            d,
            kappa,
            data: vec![Fq::ZERO; d * kappa],
        }
    }

    #[inline]
    pub fn col(&self, c: usize) -> &[Fq] {
        &self.data[c * self.d..(c + 1) * self.d]
    }

    #[inline]
    pub fn col_mut(&mut self, c: usize) -> &mut [Fq] {
        &mut self.data[c * self.d..(c + 1) * self.d]
    }

    pub fn add_inplace(&mut self, rhs: &Commitment) {
        debug_assert_eq!(self.d, rhs.d);
        debug_assert_eq!(self.kappa, rhs.kappa);
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a += *b;
        }
    }
}
