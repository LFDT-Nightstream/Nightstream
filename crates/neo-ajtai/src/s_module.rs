use crate::{commit as ajtai_commit, setup_par, AjtaiError, Commitment, PP};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_math::ring::Rq as RqEl;
use neo_math::Fq;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::{OnceLock, RwLock};

type Key = (usize, usize); // (d, m)
type PPRef = Arc<PP<RqEl>>;

#[derive(Clone)]
struct RegistryEntry {
    kappa: usize,
    /// If present, this entry can be (re)loaded on demand via `setup_par` with a fixed seed.
    seed: Option<[u8; 32]>,
    /// If present, PP is currently materialized in memory.
    pp: Option<PPRef>,
}

static AJTAI_PP_REGISTRY: OnceLock<RwLock<HashMap<Key, RegistryEntry>>> = OnceLock::new();

fn registry() -> &'static RwLock<HashMap<Key, RegistryEntry>> {
    AJTAI_PP_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Initialize the global Ajtai PP once (call this right after setup()).
pub fn set_global_pp(pp: PP<RqEl>) -> Result<(), AjtaiError> {
    let key = (pp.d, pp.m);
    let mut w = registry()
        .write()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    if let Some(existing) = w.get(&key) {
        if existing.seed.is_some() {
            return Err(AjtaiError::InvalidInput(format!(
                "Ajtai PP seed is already registered for (d,m)=({},{}) so `set_global_pp` is disallowed",
                pp.d, pp.m
            )));
        }
        // Idempotent: keep the existing PP to avoid accidentally changing commitments mid-process.
        return Ok(());
    }
    w.insert(
        key,
        RegistryEntry {
            kappa: pp.kappa,
            seed: None,
            pp: Some(Arc::new(pp)),
        },
    );
    Ok(())
}

/// Register a deterministic seed for (d,kappa,m) and *optionally* keep PP unloaded until first use.
///
/// This enables `unload_global_pp_for_dims()` to free multi-GB PP allocations during
/// prover phases that do not require commitments (e.g. sum-check table building).
pub fn set_global_pp_seeded(d: usize, kappa: usize, m: usize, seed: [u8; 32]) -> Result<(), AjtaiError> {
    let key = (d, m);
    let mut w = registry()
        .write()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    if let Some(entry) = w.get_mut(&key) {
        // If a PP is already materialized, we must not allow changing/adding a seed unless it
        // exactly matches the existing seeded configuration. Otherwise, later unload/reload would
        // silently change commitments and break proofs.
        if let Some(pp) = entry.pp.as_ref() {
            match entry.seed {
                Some(existing_seed) if existing_seed == seed => {
                    if entry.kappa != kappa || pp.kappa != kappa {
                        return Err(AjtaiError::InvalidInput(format!(
                            "Ajtai seeded PP kappa mismatch for (d,m)=({},{}) (existing κ={}, requested κ={})",
                            d, m, entry.kappa, kappa
                        )));
                    }
                    return Ok(());
                }
                Some(_) => {
                    return Err(AjtaiError::InvalidInput(format!(
                        "Ajtai PP seed mismatch for already-loaded (d,m)=({},{}); refusing to overwrite",
                        d, m
                    )));
                }
                None => {
                    return Err(AjtaiError::InvalidInput(format!(
                        "Ajtai PP for (d,m)=({},{}) is already loaded without a seed; cannot register a seed",
                        d, m
                    )));
                }
            }
        }

        if let Some(existing_seed) = entry.seed {
            if existing_seed != seed {
                return Err(AjtaiError::InvalidInput(format!(
                    "Ajtai PP seed already registered for (d,m)=({},{}) and does not match the provided seed",
                    d, m
                )));
            }
            if entry.kappa != kappa {
                return Err(AjtaiError::InvalidInput(format!(
                    "Ajtai seeded PP kappa mismatch for (d,m)=({},{}) (existing κ={}, requested κ={})",
                    d, m, entry.kappa, kappa
                )));
            }
            return Ok(());
        }

        entry.kappa = kappa;
        entry.seed = Some(seed);
        return Ok(());
    }

    w.insert(
        key,
        RegistryEntry {
            kappa,
            seed: Some(seed),
            pp: None,
        },
    );
    Ok(())
}

/// True if the PP for (d,m) can be reloaded (i.e. a seed is registered).
pub fn has_seed_for_dims(d: usize, m: usize) -> bool {
    registry()
        .read()
        .ok()
        .map(|r| r.get(&(d, m)).map(|e| e.seed.is_some()).unwrap_or(false))
        .unwrap_or(false)
}

/// Get `(kappa, seed)` for a seeded PP entry.
///
/// Returns an error if the entry does not exist or is not seeded.
pub fn get_global_pp_seeded_params_for_dims(d: usize, m: usize) -> Result<(usize, [u8; 32]), AjtaiError> {
    let r = registry()
        .read()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    let entry = r
        .get(&(d, m))
        .ok_or_else(|| AjtaiError::InvalidInput("Ajtai PP not initialized for requested (d,m)".to_string()))?;
    let seed = entry
        .seed
        .ok_or_else(|| AjtaiError::InvalidInput("Ajtai PP seed not registered for requested (d,m)".to_string()))?;
    Ok((entry.kappa, seed))
}

/// Drop the materialized PP for (d,m) from memory, keeping any registered seed.
///
/// Returns `Ok(true)` if PP was present and was unloaded.
pub fn unload_global_pp_for_dims(d: usize, m: usize) -> Result<bool, AjtaiError> {
    let key = (d, m);
    let mut w = registry()
        .write()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    let Some(entry) = w.get_mut(&key) else {
        return Ok(false);
    };
    if entry.seed.is_none() {
        return Err(AjtaiError::InvalidInput(format!(
            "Ajtai PP for (d,m)=({},{}) is not seeded; refusing to unload because it cannot be reloaded",
            d, m
        )));
    }
    let had = entry.pp.is_some();
    entry.pp = None;
    Ok(had)
}

/// If the PP for (d,m) is already materialized, return it without loading.
pub fn try_get_loaded_global_pp_for_dims(d: usize, m: usize) -> Option<PPRef> {
    registry()
        .read()
        .ok()
        .and_then(|r| r.get(&(d, m)).and_then(|e| e.pp.as_ref().cloned()))
}

fn get_or_load_global_pp_for_dims(d: usize, m: usize) -> Result<PPRef, AjtaiError> {
    // Fast path: already loaded.
    if let Ok(r) = registry().read() {
        if let Some(entry) = r.get(&(d, m)) {
            if let Some(pp) = entry.pp.as_ref() {
                return Ok(pp.clone());
            }
        }
    }

    // Slow path: load from seed if available.
    let (seed, kappa) = {
        let r = registry()
            .read()
            .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
        let entry = r
            .get(&(d, m))
            .ok_or_else(|| AjtaiError::InvalidInput("Ajtai PP not initialized for requested (d,m)".to_string()))?;
        let seed = entry
            .seed
            .ok_or_else(|| AjtaiError::InvalidInput("Ajtai PP seed not registered for requested (d,m)".to_string()))?;
        (seed, entry.kappa)
    };

    let mut rng = ChaCha8Rng::from_seed(seed);
    let pp = setup_par(&mut rng, d, kappa, m)?;
    let pp = Arc::new(pp);

    let mut w = registry()
        .write()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    let entry = w.entry((d, m)).or_insert_with(|| RegistryEntry {
        kappa,
        seed: Some(seed),
        pp: None,
    });
    entry.kappa = kappa;
    entry.seed = Some(seed);
    entry.pp = Some(pp.clone());
    Ok(pp)
}

/// Legacy: pick the sole PP if only one exists.
pub fn get_global_pp() -> Result<PPRef, AjtaiError> {
    let r = registry()
        .read()
        .map_err(|_| AjtaiError::Internal("PP registry poisoned".to_string()))?;
    let mut it = r.iter();
    match (it.next(), it.next()) {
        (Some((&(d, m), _entry)), None) => {
            drop(r);
            get_or_load_global_pp_for_dims(d, m)
        }
        (None, _) => Err(AjtaiError::InvalidInput(
            "Ajtai PP not initialized (call set_global_pp())".to_string(),
        )),
        _ => Err(AjtaiError::InvalidInput(
            "Multiple Ajtai PPs present; use get_global_pp_for_dims()".to_string(),
        )),
    }
}

/// True if a PP for (d,m) is available.
pub fn has_global_pp_for_dims(d: usize, m: usize) -> bool {
    registry()
        .read()
        .map(|r| r.contains_key(&(d, m)))
        .unwrap_or(false)
}

/// Get the Ajtai PP for a specific (d,m).
pub fn get_global_pp_for_dims(d: usize, m: usize) -> Result<PPRef, AjtaiError> {
    get_or_load_global_pp_for_dims(d, m)
}

/// Get the Ajtai PP using `z_len = d*m`.
pub fn get_global_pp_for_z_len(z_len: usize) -> Result<PPRef, AjtaiError> {
    let d = neo_math::D;
    if z_len % d != 0 {
        return Err(AjtaiError::InvalidInput("z_len not multiple of D".to_string()));
    }
    get_global_pp_for_dims(d, z_len / d)
}

/// Concrete S-module homomorphism backed by Ajtai PP
#[derive(Clone)]
pub struct AjtaiSModule {
    pp: PpSource,
}

#[derive(Clone)]
enum PpSource {
    Owned(PPRef),
    Global { d: usize, m: usize },
}

impl AjtaiSModule {
    pub fn new(pp: PPRef) -> Self {
        Self {
            pp: PpSource::Owned(pp),
        }
    }
    /// Legacy: pick the sole PP if only one exists.
    pub fn from_global() -> Result<Self, AjtaiError> {
        let pp = get_global_pp()?;
        Ok(Self::new(pp))
    }
    /// New: pick PP that matches (d,m).
    pub fn from_global_for_dims(d: usize, m: usize) -> Result<Self, AjtaiError> {
        if !has_global_pp_for_dims(d, m) {
            return Err(AjtaiError::InvalidInput(
                "Ajtai PP not initialized for requested (d,m); call set_global_pp(...) or set_global_pp_seeded(...)"
                    .to_string(),
            ));
        }
        Ok(Self {
            pp: PpSource::Global { d, m },
        })
    }
    /// New: pick PP that matches `z_len = d*m`.
    pub fn from_global_for_z_len(z_len: usize) -> Result<Self, AjtaiError> {
        let d = neo_math::D;
        if z_len % d != 0 {
            return Err(AjtaiError::InvalidInput("z_len not multiple of D".to_string()));
        }
        let m = z_len / d;
        if !has_global_pp_for_dims(d, m) {
            return Err(AjtaiError::InvalidInput(
                "Ajtai PP not initialized for requested z_len; call set_global_pp(...) or set_global_pp_seeded(...)"
                    .to_string(),
            ));
        }
        Ok(Self {
            pp: PpSource::Global {
                d,
                m,
            },
        })
    }

    /// Return κ for the underlying PP without requiring it to be materialized.
    pub fn kappa(&self) -> usize {
        match &self.pp {
            PpSource::Owned(pp) => pp.kappa,
            PpSource::Global { d, m } => registry()
                .read()
                .ok()
                .and_then(|r| r.get(&(*d, *m)).map(|e| e.kappa))
                .unwrap_or_else(|| get_or_load_global_pp_for_dims(*d, *m).expect("Ajtai PP load").kappa),
        }
    }
}

impl SModuleHomomorphism<Fq, Commitment> for AjtaiSModule {
    fn commit(&self, z: &Mat<Fq>) -> Commitment {
        match &self.pp {
            PpSource::Owned(pp) => ajtai_commit::commit_row_major(pp, z),
            PpSource::Global { d, m } => {
                // Prefer not to materialize PP for seeded entries.
                let (zd, zm) = (z.rows(), z.cols());
                let want_d = *d;
                let want_m = *m;
                assert_eq!(zd, want_d, "AjtaiSModule: Z.rows != d");
                assert_eq!(zm, want_m, "AjtaiSModule: Z.cols != m");

                // Fast path: if PP is already loaded, use it.
                if let Ok(r) = registry().read() {
                    if let Some(entry) = r.get(&(want_d, want_m)) {
                        if let Some(pp) = entry.pp.as_ref() {
                            return ajtai_commit::commit_row_major(pp, z);
                        }
                        if let Some(seed) = entry.seed {
                            return ajtai_commit::commit_row_major_seeded(seed, want_d, entry.kappa, want_m, z);
                        }
                    }
                }

                // Fallback: load PP if needed (non-seeded entry or registry inaccessible).
                let pp = get_or_load_global_pp_for_dims(want_d, want_m).expect("Ajtai PP load should succeed");
                ajtai_commit::commit_row_major(&pp, z)
            }
        }
    }

    fn project_x(&self, z: &Mat<Fq>, min: usize) -> Mat<Fq> {
        let rows = z.rows();
        let cols = min.min(z.cols());
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(z[(r, c)]);
            }
        }
        Mat::from_row_major(rows, cols, data)
    }
}
