use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use serde::{Deserialize, Serialize};

use crate::error::{AjtaiError, AjtaiResult};

#[derive(Clone, Debug)]
pub struct PP<RqEl> {
    pub(crate) kappa: usize,
    pub(crate) m: usize,
    pub(crate) d: usize,
    pub(crate) m_rows: Vec<Vec<RqEl>>,
}

impl<RqEl> PP<RqEl> {
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment {
    pub d: usize,
    pub kappa: usize,
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
