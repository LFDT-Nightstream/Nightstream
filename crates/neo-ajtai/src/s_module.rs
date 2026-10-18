use crate::{commit as ajtai_commit, AjtaiError, Commitment, PP};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_math::ring::Rq as RqEl;
use neo_math::Fq;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::{OnceLock, RwLock};

type Key = (usize, usize);
type PPRef = Arc<PP<RqEl>>;

static AJTAI_PP_REGISTRY: OnceLock<RwLock<HashMap<Key, PPRef>>> = OnceLock::new();

fn registry() -> &'static RwLock<HashMap<Key, PPRef>> {
    AJTAI_PP_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn set_global_pp(pp: PP<RqEl>) -> Result<(), AjtaiError> {
    let key = (pp.d, pp.m);
    let mut w = registry()
        .write()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    if let Some(existing) = w.get(&key) {
        if existing.kappa == pp.kappa && existing.m_rows == pp.m_rows {
            return Ok(());
        }
        return Err(AjtaiError::InvalidInput(format!(
            "a different Ajtai PP is already registered for (d,m)=({},{})",
            pp.d, pp.m
        )));
    }
    w.insert(key, Arc::new(pp));
    Ok(())
}

pub fn get_global_pp() -> Result<PPRef, AjtaiError> {
    let r = registry()
        .read()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    let mut it = r.values();
    match (it.next(), it.next()) {
        (Some(pp), None) => Ok(pp.clone()),
        (None, _) => Err(AjtaiError::InvalidInput(
            "Ajtai PP not initialized (call set_global_pp())".to_string(),
        )),
        _ => Err(AjtaiError::InvalidInput(
            "Multiple Ajtai PPs present; use get_global_pp_for_dims()".to_string(),
        )),
    }
}

pub fn has_global_pp_for_dims(d: usize, m: usize) -> bool {
    registry()
        .read()
        .map(|r| r.contains_key(&(d, m)))
        .unwrap_or(false)
}

pub fn get_global_pp_for_dims(d: usize, m: usize) -> Result<PPRef, AjtaiError> {
    registry()
        .read()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?
        .get(&(d, m))
        .cloned()
        .ok_or_else(|| {
            AjtaiError::InvalidInput(
                "Ajtai PP not initialized for requested (d,m); call set_global_pp(...)".to_string(),
            )
        })
}

pub fn get_global_pp_for_z_len(z_len: usize) -> Result<PPRef, AjtaiError> {
    let d = neo_math::D;
    if z_len % d != 0 {
        return Err(AjtaiError::InvalidInput("z_len not multiple of D".to_string()));
    }
    get_global_pp_for_dims(d, z_len / d)
}

#[derive(Clone)]
pub struct AjtaiSModule {
    pp: PPRef,
}

impl AjtaiSModule {
    pub fn new(pp: PPRef) -> Self {
        Self { pp }
    }

    pub fn from_global() -> Result<Self, AjtaiError> {
        Ok(Self::new(get_global_pp()?))
    }
    pub fn from_global_for_dims(d: usize, m: usize) -> Result<Self, AjtaiError> {
        Ok(Self::new(get_global_pp_for_dims(d, m)?))
    }
    pub fn from_global_for_z_len(z_len: usize) -> Result<Self, AjtaiError> {
        Ok(Self::new(get_global_pp_for_z_len(z_len)?))
    }

    pub fn kappa(&self) -> usize {
        self.pp.kappa
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.pp.d, self.pp.m)
    }

    pub fn materialize_pp(&self) -> Result<Arc<PP<RqEl>>, AjtaiError> {
        Ok(self.pp.clone())
    }
}

impl SModuleHomomorphism<Fq, Commitment> for AjtaiSModule {
    fn commit(&self, z: &Mat<Fq>) -> Commitment {
        ajtai_commit::commit_row_major(&self.pp, z)
    }
}
