use p3_goldilocks::Goldilocks as Fq;
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

/// Public parameters for Ajtai: M ∈ R_q^{κ×m}, stored row-major.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PP<RqEl> {
    pub kappa: usize,
    pub m: usize,
    pub d: usize,
    /// Ajtai matrix rows; each row is a vector of ring elements of length m.
    pub m_rows: Vec<Vec<RqEl>>,
}

/// Commitment c ∈ F_q^{d×κ}, stored as column-major flat matrix (κ columns, each length d).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    pub d: usize,
    pub kappa: usize,
    /// data[c * d + i] = i-th row of column c
    pub data: Vec<Fq>,
}

impl Commitment {
    pub fn zeros(d: usize, kappa: usize) -> Self {
        Self { d, kappa, data: vec![Fq::ZERO; d * kappa] }
    }
    
    #[inline] 
    pub fn col(&self, c: usize) -> &[Fq] { 
        &self.data[c*self.d .. (c+1)*self.d] 
    }
    
    #[inline] 
    pub fn col_mut(&mut self, c: usize) -> &mut [Fq] { 
        &mut self.data[c*self.d .. (c+1)*self.d] 
    }

    pub fn add_inplace(&mut self, rhs: &Commitment) {
        debug_assert_eq!(self.d, rhs.d);
        debug_assert_eq!(self.kappa, rhs.kappa);
        for (a,b) in self.data.iter_mut().zip(rhs.data.iter()) { 
            *a += *b; 
        }
    }
}
