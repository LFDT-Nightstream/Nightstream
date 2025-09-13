use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::sync::Arc;
use neo_math::Fq;
use neo_math::ring::Rq as RqEl;
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use crate::{PP, Commitment, commit as ajtai_commit, AjtaiError};
use p3_field::PrimeCharacteristicRing;

type Key = (usize, usize); // (d, m)
type PPRef = Arc<PP<RqEl>>;
static AJTAI_PP_REGISTRY: OnceLock<RwLock<HashMap<Key, PPRef>>> = OnceLock::new();

fn registry() -> &'static RwLock<HashMap<Key, PPRef>> {
    AJTAI_PP_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Initialize the global Ajtai PP once (call this right after setup()).
pub fn set_global_pp(pp: PP<RqEl>) -> Result<(), AjtaiError> {
    let key = (pp.d, pp.m);
    let mut w = registry()
        .write()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    w.entry(key).or_insert_with(|| Arc::new(pp));
    Ok(())
}

/// Legacy: pick the sole PP if only one exists.
pub fn get_global_pp() -> Result<PPRef, AjtaiError> {
    let r = registry()
        .read()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    let mut it = r.values();
    match (it.next(), it.next()) {
        (Some(pp), None) => Ok(pp.clone()),
        (None, _) => Err(AjtaiError::InvalidInput("Ajtai PP not initialized (call set_global_pp())".to_string())),
        _ => Err(AjtaiError::InvalidInput("Multiple Ajtai PPs present; use get_global_pp_for_dims()".to_string())),
    }
}

/// True if a PP for (d,m) is available.
pub fn has_global_pp_for_dims(d: usize, m: usize) -> bool {
    registry().read().map(|r| r.contains_key(&(d, m))).unwrap_or(false)
}

/// Get the Ajtai PP for a specific (d,m).
pub fn get_global_pp_for_dims(d: usize, m: usize) -> Result<PPRef, AjtaiError> {
    registry()
        .read()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?
        .get(&(d, m))
        .cloned()
        .ok_or(AjtaiError::InvalidInput("Ajtai PP not initialized for requested (d,m)".to_string()))
}

/// Get the Ajtai PP using `z_len = d*m`.
pub fn get_global_pp_for_z_len(z_len: usize) -> Result<PPRef, AjtaiError> {
    let d = neo_math::D;
    if z_len % d != 0 { return Err(AjtaiError::InvalidInput("z_len not multiple of D".to_string())); }
    get_global_pp_for_dims(d, z_len / d)
}

/// Concrete S-module homomorphism backed by Ajtai PP
#[derive(Clone)]
pub struct AjtaiSModule { pub pp: PPRef }

impl AjtaiSModule {
    pub fn new(pp: PPRef) -> Self { Self { pp } }
    /// Legacy: pick the sole PP if only one exists.
    pub fn from_global() -> Result<Self, AjtaiError> {
        Ok(Self { pp: get_global_pp()? })
    }
    /// New: pick PP that matches (d,m).
    pub fn from_global_for_dims(d: usize, m: usize) -> Result<Self, AjtaiError> {
        Ok(Self { pp: get_global_pp_for_dims(d, m)? })
    }
    /// New: pick PP that matches `z_len = d*m`.
    pub fn from_global_for_z_len(z_len: usize) -> Result<Self, AjtaiError> {
        Ok(Self { pp: get_global_pp_for_z_len(z_len)? })
    }
}

impl SModuleHomomorphism<Fq, Commitment> for AjtaiSModule {
    fn commit(&self, z: &Mat<Fq>) -> Commitment {
        let d = self.pp.d; let m = self.pp.m;
        assert_eq!(z.rows(), d, "AjtaiSModule.commit: Z.rows != d");
        assert_eq!(z.cols(), m, "AjtaiSModule.commit: Z.cols != m");
        // Mat is row-major; Ajtai commit expects column-major (d×m)
        let mut col_major = vec![Fq::ZERO; d * m];
        for c in 0..m { for r in 0..d { col_major[c*d + r] = z[(r,c)]; } }
        ajtai_commit(&self.pp, &col_major)
    }

    fn project_x(&self, z: &Mat<Fq>, min: usize) -> Mat<Fq> {
        let rows = z.rows(); let cols = min.min(z.cols());
        let mut data = Vec::with_capacity(rows*cols);
        for r in 0..rows { for c in 0..cols { data.push(z[(r,c)]); } }
        Mat::from_row_major(rows, cols, data)
    }
}
