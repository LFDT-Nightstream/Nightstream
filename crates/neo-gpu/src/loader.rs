use std::path::{Path, PathBuf};

use libloading::Library;

use crate::abi::{
    DeviceRequest, DeviceResponse, EvalsRequest, FlatFq, FoldRequest, SessionRequest, SnapshotRequest, ABI_VERSION,
    ABI_VERSION_SYMBOL, DEVICE_PROBE_SYMBOL, FE_CREATE_SYMBOL, FE_DESTROY_SYMBOL, FE_EVALS_AT_SYMBOL, FE_FOLD_SYMBOL,
    NC_CREATE_SYMBOL, NC_DESTROY_SYMBOL, NC_EVALS_AT_SYMBOL, NC_FOLD_SYMBOL, POSEIDON2_PERMUTE_BATCH_SYMBOL,
    POSEIDON2_PERMUTE_SYMBOL, POSEIDON2_STATE_WIDTH, SESSION_CLOSE_SYMBOL, SESSION_OPEN_SYMBOL,
};
use crate::{DeviceApi, MojoBackendConfig, NeoGpuError};

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type DeviceProbeFn = unsafe extern "C" fn(usize, *mut u64) -> i32;
type SessionOpenFn = unsafe extern "C" fn(usize, *mut u64) -> i32;
type SessionCloseFn = unsafe extern "C" fn(usize) -> i32;
type CreateEvaluatorFn = unsafe extern "C" fn(*const SnapshotRequest, *mut usize) -> i32;
type DestroyEvaluatorFn = unsafe extern "C" fn(usize, usize) -> i32;
type EvalsAtFn = unsafe extern "C" fn(*const EvalsRequest) -> i32;
type FoldFn = unsafe extern "C" fn(*const FoldRequest) -> i32;
type Poseidon2PermuteFn = unsafe extern "C" fn(usize, *mut FlatFq, u32) -> i32;
type Poseidon2PermuteBatchFn = unsafe extern "C" fn(usize, *mut FlatFq, u32, u32) -> i32;

fn platform_library_name() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "libnightstream_mojo_gpu.dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "libnightstream_mojo_gpu.so"
    }
    #[cfg(target_os = "windows")]
    {
        "nightstream_mojo_gpu.dll"
    }
}

fn resolve_library_path(cfg: &MojoBackendConfig) -> PathBuf {
    cfg.library_path
        .clone()
        .unwrap_or_else(|| PathBuf::from(platform_library_name()))
}

unsafe fn load_required<T: Copy>(lib: &Library, symbol: &'static [u8], name: &'static str) -> Result<T, NeoGpuError> {
    lib.get::<T>(symbol)
        .map(|sym| *sym)
        .map_err(|_| NeoGpuError::MissingSymbol { symbol: name })
}

unsafe fn load_optional<T: Copy>(lib: &Library, symbol: &'static [u8]) -> Option<T> {
    lib.get::<T>(symbol).ok().map(|sym| *sym)
}

struct OptionalEvaluatorFns {
    fe_create: Option<CreateEvaluatorFn>,
    fe_destroy: Option<DestroyEvaluatorFn>,
    fe_evals_at: Option<EvalsAtFn>,
    fe_fold: Option<FoldFn>,
    nc_create: Option<CreateEvaluatorFn>,
    nc_destroy: Option<DestroyEvaluatorFn>,
    nc_evals_at: Option<EvalsAtFn>,
    nc_fold: Option<FoldFn>,
    poseidon2_permute: Option<Poseidon2PermuteFn>,
    poseidon2_permute_batch: Option<Poseidon2PermuteBatchFn>,
}

impl OptionalEvaluatorFns {
    #[inline]
    fn supports_split_nc(&self) -> bool {
        self.fe_create.is_some()
            && self.fe_destroy.is_some()
            && self.fe_evals_at.is_some()
            && self.fe_fold.is_some()
            && self.nc_create.is_some()
            && self.nc_destroy.is_some()
            && self.nc_evals_at.is_some()
            && self.nc_fold.is_some()
    }

    #[inline]
    fn supports_poseidon2(&self) -> bool {
        self.poseidon2_permute.is_some()
    }

    #[inline]
    fn supports_poseidon2_batch(&self) -> bool {
        self.poseidon2_permute_batch.is_some()
    }

    #[inline]
    fn supports_cpu_direct_mode(&self) -> bool {
        self.supports_poseidon2() || self.supports_split_nc()
    }
}

pub struct MojoLibrary {
    path: PathBuf,
    _lib: Library,
    device_probe: DeviceProbeFn,
    session_open: SessionOpenFn,
    session_close: SessionCloseFn,
    evaluators: OptionalEvaluatorFns,
}

impl MojoLibrary {
    #[inline]
    fn supports_cpu_direct_mode(&self) -> bool {
        self.evaluators.supports_cpu_direct_mode()
    }

    pub fn load(cfg: &MojoBackendConfig) -> Result<Self, NeoGpuError> {
        let path = resolve_library_path(cfg);
        let lib = unsafe { Library::new(&path) }.map_err(|source| NeoGpuError::LoadLibrary {
            path: path.clone(),
            source,
        })?;

        let abi_version =
            unsafe { load_required::<AbiVersionFn>(&lib, ABI_VERSION_SYMBOL, "nightstream_gpu_abi_version")? };
        let observed = unsafe { abi_version() };
        if observed != ABI_VERSION {
            return Err(NeoGpuError::AbiMismatch {
                expected: ABI_VERSION,
                observed,
            });
        }

        let device_probe =
            unsafe { load_required::<DeviceProbeFn>(&lib, DEVICE_PROBE_SYMBOL, "nightstream_gpu_device_probe")? };
        let session_open =
            unsafe { load_required::<SessionOpenFn>(&lib, SESSION_OPEN_SYMBOL, "nightstream_gpu_session_open")? };
        let session_close =
            unsafe { load_required::<SessionCloseFn>(&lib, SESSION_CLOSE_SYMBOL, "nightstream_gpu_session_close")? };

        let evaluators = unsafe {
            OptionalEvaluatorFns {
                fe_create: load_optional::<CreateEvaluatorFn>(&lib, FE_CREATE_SYMBOL),
                fe_destroy: load_optional::<DestroyEvaluatorFn>(&lib, FE_DESTROY_SYMBOL),
                fe_evals_at: load_optional::<EvalsAtFn>(&lib, FE_EVALS_AT_SYMBOL),
                fe_fold: load_optional::<FoldFn>(&lib, FE_FOLD_SYMBOL),
                nc_create: load_optional::<CreateEvaluatorFn>(&lib, NC_CREATE_SYMBOL),
                nc_destroy: load_optional::<DestroyEvaluatorFn>(&lib, NC_DESTROY_SYMBOL),
                nc_evals_at: load_optional::<EvalsAtFn>(&lib, NC_EVALS_AT_SYMBOL),
                nc_fold: load_optional::<FoldFn>(&lib, NC_FOLD_SYMBOL),
                poseidon2_permute: load_optional::<Poseidon2PermuteFn>(&lib, POSEIDON2_PERMUTE_SYMBOL),
                poseidon2_permute_batch: load_optional::<Poseidon2PermuteBatchFn>(&lib, POSEIDON2_PERMUTE_BATCH_SYMBOL),
            }
        };

        Ok(Self {
            path,
            _lib: lib,
            device_probe,
            session_open,
            session_close,
            evaluators,
        })
    }

    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn probe_device(&self, api: DeviceApi, device_id: u32) -> Result<bool, NeoGpuError> {
        let req = DeviceRequest {
            api: api.as_u32(),
            device_id,
        };
        let mut resp_word = 0u64;
        let status = unsafe { (self.device_probe)((&req as *const DeviceRequest) as usize, &mut resp_word) };
        if status != 0 {
            return Ok(api == DeviceApi::Cpu && self.supports_cpu_direct_mode());
        }
        let resp = DeviceResponse {
            status: resp_word as u32 as i32,
            available: (resp_word >> 32) as u32 as i32,
        };
        if resp.available != 0 {
            return Ok(true);
        }
        Ok(false)
    }

    #[inline]
    pub fn supports_split_nc_api(&self) -> bool {
        self.evaluators.supports_split_nc()
    }

    #[inline]
    pub fn supports_poseidon2_api(&self) -> bool {
        self.evaluators.supports_poseidon2()
    }

    #[inline]
    pub fn supports_poseidon2_batch_api(&self) -> bool {
        self.evaluators.supports_poseidon2_batch()
    }

    pub fn open_session(self, cfg: &MojoBackendConfig) -> Result<MojoSession, NeoGpuError> {
        let req = SessionRequest {
            api: cfg.device_api.as_u32(),
            device_id: cfg.device_id,
        };
        let mut handle_word = 0u64;
        let status = unsafe { (self.session_open)((&req as *const SessionRequest) as usize, &mut handle_word) };
        let mut handle = handle_word as usize;
        if status == 0 && handle != 0 {
            return Ok(MojoSession {
                library: Some(self),
                handle,
                device_api: cfg.device_api,
                device_id: cfg.device_id,
            });
        }

        if cfg.device_api == DeviceApi::Cpu && self.supports_cpu_direct_mode() {
            handle = 1;
            return Ok(MojoSession {
                library: Some(self),
                handle,
                device_api: cfg.device_api,
                device_id: cfg.device_id,
            });
        }

        if !self.probe_device(cfg.device_api, cfg.device_id)? {
            return Err(NeoGpuError::DeviceUnavailable {
                api: cfg.device_api,
                device_id: cfg.device_id,
            });
        }

        if status != 0 || handle == 0 {
            return Err(NeoGpuError::SessionOpenFailed { status });
        }
        unreachable!()
    }
}

impl std::fmt::Debug for MojoLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MojoLibrary")
            .field("path", &self.path)
            .field("supports_split_nc_api", &self.supports_split_nc_api())
            .field("supports_poseidon2_api", &self.supports_poseidon2_api())
            .field("supports_poseidon2_batch_api", &self.supports_poseidon2_batch_api())
            .finish()
    }
}

pub struct MojoSession {
    library: Option<MojoLibrary>,
    handle: usize,
    device_api: DeviceApi,
    device_id: u32,
}

impl MojoSession {
    #[inline]
    pub fn handle(&self) -> usize {
        self.handle
    }

    #[inline]
    pub fn device_api(&self) -> DeviceApi {
        self.device_api
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn supports_split_nc_api(&self) -> bool {
        self.library
            .as_ref()
            .map(MojoLibrary::supports_split_nc_api)
            .unwrap_or(false)
    }

    #[inline]
    pub fn supports_poseidon2_api(&self) -> bool {
        self.library
            .as_ref()
            .map(MojoLibrary::supports_poseidon2_api)
            .unwrap_or(false)
    }

    #[inline]
    pub fn supports_poseidon2_batch_api(&self) -> bool {
        self.library
            .as_ref()
            .map(MojoLibrary::supports_poseidon2_batch_api)
            .unwrap_or(false)
    }

    pub fn permute_poseidon2_u64x8(
        &self,
        state: &[u64; POSEIDON2_STATE_WIDTH],
    ) -> Result<[u64; POSEIDON2_STATE_WIDTH], NeoGpuError> {
        let library = self
            .library
            .as_ref()
            .ok_or(NeoGpuError::UnsupportedOperation {
                op: "poseidon2_permute_u64x8",
            })?;
        let permute = library
            .evaluators
            .poseidon2_permute
            .ok_or(NeoGpuError::UnsupportedOperation {
                op: "poseidon2_permute_u64x8",
            })?;

        let mut state_flat = state.map(|limb| FlatFq { limb });
        let status = unsafe { permute(self.handle, state_flat.as_mut_ptr(), POSEIDON2_STATE_WIDTH as u32) };
        if status != 0 {
            return Err(NeoGpuError::OperationFailed {
                op: "poseidon2_permute_u64x8",
                status,
            });
        }
        Ok(state_flat.map(|x| x.limb))
    }

    pub fn permute_poseidon2_batch_u64x8(
        &self,
        states: &mut [[u64; POSEIDON2_STATE_WIDTH]],
    ) -> Result<(), NeoGpuError> {
        if states.is_empty() {
            return Ok(());
        }

        let library = self
            .library
            .as_ref()
            .ok_or(NeoGpuError::UnsupportedOperation {
                op: "poseidon2_permute_batch_u64x8",
            })?;

        if let Some(permute_batch) = library.evaluators.poseidon2_permute_batch {
            let state_ptr = states.as_mut_ptr().cast::<FlatFq>();
            let status = unsafe {
                permute_batch(
                    self.handle,
                    state_ptr,
                    states.len() as u32,
                    POSEIDON2_STATE_WIDTH as u32,
                )
            };
            if status != 0 {
                return Err(NeoGpuError::OperationFailed {
                    op: "poseidon2_permute_batch_u64x8",
                    status,
                });
            }
            return Ok(());
        }

        for state in states {
            *state = self.permute_poseidon2_u64x8(state)?;
        }
        Ok(())
    }
}

impl Drop for MojoSession {
    fn drop(&mut self) {
        let Some(library) = self.library.take() else {
            return;
        };
        let status = unsafe { (library.session_close)(self.handle) };
        if status != 0 {
            let _ = Err::<(), _>(NeoGpuError::SessionCloseFailed { status });
        }
    }
}

pub fn connect(cfg: &MojoBackendConfig) -> Result<MojoSession, NeoGpuError> {
    if cfg.device_api != DeviceApi::Auto {
        return MojoLibrary::load(cfg)?.open_session(cfg);
    }

    let mut last_err = None;
    for api in cfg.device_api.candidate_order() {
        let candidate_cfg = cfg.clone().with_device_api(*api);
        match MojoLibrary::load(&candidate_cfg).and_then(|lib| lib.open_session(&candidate_cfg)) {
            Ok(session) => return Ok(session),
            Err(err) => last_err = Some(err),
        }
    }

    Err(last_err.unwrap_or(NeoGpuError::DeviceUnavailable {
        api: cfg.device_api,
        device_id: cfg.device_id,
    }))
}
