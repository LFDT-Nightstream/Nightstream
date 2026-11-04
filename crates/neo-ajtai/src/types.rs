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

/// An Ajtai commitment: `kappa` columns of `D` field elements. The fields are
/// private, so `data` always holds exactly `D * kappa` elements. The serde form
/// keeps the field names `d`, `kappa` and `data`, and decoding checks the shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "CommitmentWire")]
pub struct Commitment {
    d: usize,
    kappa: usize,
    data: Vec<Fq>,
}

#[derive(Deserialize)]
struct CommitmentWire {
    d: usize,
    kappa: usize,
    data: Vec<Fq>,
}

impl TryFrom<CommitmentWire> for Commitment {
    type Error = AjtaiError;

    fn try_from(wire: CommitmentWire) -> AjtaiResult<Self> {
        if wire.d != neo_math::D {
            return Err(AjtaiError::InvalidDimensions(format!(
                "commitment column height {} differs from D = {}",
                wire.d,
                neo_math::D
            )));
        }
        Self::new(wire.kappa, wire.data)
    }
}

impl Commitment {
    /// `data` holds `kappa` columns of `D` elements, column after column.
    pub fn new(kappa: usize, data: Vec<Fq>) -> AjtaiResult<Self> {
        let expected = neo_math::D
            .checked_mul(kappa)
            .ok_or_else(|| AjtaiError::InvalidDimensions("commitment size overflows usize".into()))?;
        if data.len() != expected {
            return Err(AjtaiError::SizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            d: neo_math::D,
            kappa,
            data,
        })
    }

    pub fn zeros(kappa: usize) -> Self {
        Self {
            d: neo_math::D,
            kappa,
            data: vec![Fq::ZERO; neo_math::D * kappa],
        }
    }

    /// The column height, always `D`.
    pub fn d(&self) -> usize {
        self.d
    }

    pub fn kappa(&self) -> usize {
        self.kappa
    }

    pub fn data(&self) -> &[Fq] {
        &self.data
    }

    /// The elements, for in-place updates that keep the shape.
    pub fn data_mut(&mut self) -> &mut [Fq] {
        &mut self.data
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
