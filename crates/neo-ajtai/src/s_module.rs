use std::sync::OnceLock;
use neo_math::Fq;
use neo_math::ring::Rq as RqEl;
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use crate::{PP, Commitment, commit as ajtai_commit, AjtaiError};
use p3_field::PrimeCharacteristicRing;

static AJTAI_PP: OnceLock<PP<RqEl>> = OnceLock::new();

/// Initialize the global Ajtai PP once (call this right after setup()).
pub fn set_global_pp(pp: PP<RqEl>) -> Result<(), AjtaiError> {
    AJTAI_PP.set(pp).map_err(|_| AjtaiError::InvalidInput("Ajtai PP already initialized"))
}

/// Borrow the global Ajtai PP (used by folding).
pub fn get_global_pp() -> Result<&'static PP<RqEl>, AjtaiError> {
    AJTAI_PP.get().ok_or(AjtaiError::InvalidInput("Ajtai PP not initialized (call set_global_pp())"))
}

/// Concrete S-module homomorphism backed by Ajtai PP
#[derive(Clone, Copy)]
pub struct AjtaiSModule<'a> { pub pp: &'a PP<RqEl> }

impl<'a> AjtaiSModule<'a> {
    pub fn new(pp: &'a PP<RqEl>) -> Self { Self { pp } }
    pub fn from_global() -> Result<Self, AjtaiError> { Ok(Self { pp: get_global_pp()? }) }
}

impl<'a> SModuleHomomorphism<Fq, Commitment> for AjtaiSModule<'a> {
    fn commit(&self, z: &Mat<Fq>) -> Commitment {
        let d = self.pp.d; let m = self.pp.m;
        assert_eq!(z.rows(), d, "AjtaiSModule.commit: Z.rows != d");
        assert_eq!(z.cols(), m, "AjtaiSModule.commit: Z.cols != m");
        // Mat is row-major; Ajtai commit expects column-major (d×m)
        let mut col_major = vec![Fq::ZERO; d * m];
        for c in 0..m { for r in 0..d { col_major[c*d + r] = z[(r,c)]; } }
        ajtai_commit(self.pp, &col_major)
    }

    fn project_x(&self, z: &Mat<Fq>, min: usize) -> Mat<Fq> {
        let rows = z.rows(); let cols = min.min(z.cols());
        let mut data = Vec::with_capacity(rows*cols);
        for r in 0..rows { for c in 0..cols { data.push(z[(r,c)]); } }
        Mat::from_row_major(rows, cols, data)
    }
}
