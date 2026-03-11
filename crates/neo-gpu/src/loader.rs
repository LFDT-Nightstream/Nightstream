use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::ffi::c_void;
#[cfg(target_os = "macos")]
use std::sync::{mpsc, OnceLock};

use libloading::Library;
use neo_math::{from_complex, Fq, KExtensions, K};
use p3_field::PrimeCharacteristicRing;

use crate::abi::{
    DeviceRequest, DeviceResponse, FlatFq, FlatK, SessionRequest, ABI_VERSION, ABI_VERSION_SYMBOL,
    DEVICE_PROBE_SYMBOL, FE_CREATE_SYMBOL, FE_DESTROY_SYMBOL, FE_EVALS_AT_SYMBOL, FE_FOLD_SYMBOL,
    NC_CREATE_SYMBOL, NC_DESTROY_SYMBOL, NC_EVALS_AT_SYMBOL, NC_FOLD_SYMBOL, POSEIDON2_PERMUTE_BATCH_SYMBOL,
    POSEIDON2_PERMUTE_SYMBOL, POSEIDON2_STATE_WIDTH, SESSION_CLOSE_SYMBOL, SESSION_OPEN_SYMBOL,
};
use crate::{DeviceApi, MojoBackendConfig, NeoGpuError};

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type DeviceProbeFn = unsafe extern "C" fn(usize, *mut u64) -> i32;
type SessionOpenFn = unsafe extern "C" fn(usize, *mut u64) -> i32;
type SessionCloseFn = unsafe extern "C" fn(usize) -> i32;
type CreateEvaluatorFn = unsafe extern "C" fn(u64, *mut u64, u64, *mut usize) -> i32;
type DestroyEvaluatorFn = unsafe extern "C" fn(usize, usize) -> i32;
type EvalsAtFn = unsafe extern "C" fn(u64, u64, *mut u64, u64, *mut u64, u64, *mut u64, usize) -> i32;
type FoldFn = unsafe extern "C" fn(usize, usize, u64, u64) -> i32;
type Poseidon2PermuteFn = unsafe extern "C" fn(usize, *mut FlatFq, u32) -> i32;
type Poseidon2PermuteBatchFn = unsafe extern "C" fn(usize, *mut FlatFq, u32, u32) -> i32;

const SPLIT_NC_SNAPSHOT_MAGIC: u64 = 0x4E53_504C_4954_4E43;
const SPLIT_NC_SNAPSHOT_VERSION: u64 = 1;
const SPLIT_NC_FE_ROW_V1: u64 = 1;
const SPLIT_NC_NC_COL_V1: u64 = 2;
const FE_HEADER_WORDS: usize = 16;
const NC_HEADER_WORDS: usize = 13;

fn flat_k_to_ext(value: FlatK) -> K {
    from_complex(Fq::from_u64(value.re), Fq::from_u64(value.im))
}

fn ext_to_flat_k(value: K) -> FlatK {
    let (re, im) = value.to_limbs_u64();
    FlatK { re, im }
}

fn read_u64_word(bytes: &[u8], word_idx: usize) -> u64 {
    let start = word_idx * 8;
    let mut word = [0u8; 8];
    word.copy_from_slice(&bytes[start..start + 8]);
    u64::from_le_bytes(word)
}

fn write_u64_word(bytes: &mut [u8], word_idx: usize, value: u64) {
    let start = word_idx * 8;
    bytes[start..start + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_flat_k_word(bytes: &[u8], word_idx: usize) -> FlatK {
    FlatK {
        re: read_u64_word(bytes, word_idx),
        im: read_u64_word(bytes, word_idx + 1),
    }
}

fn write_flat_k_word(bytes: &mut [u8], word_idx: usize, value: FlatK) {
    write_u64_word(bytes, word_idx, value.re);
    write_u64_word(bytes, word_idx + 1, value.im);
}

fn fold_k_table(bytes: &mut [u8], word_offset: usize, len: usize, challenge: FlatK) {
    let half = len / 2;
    let r = flat_k_to_ext(challenge);
    for i in 0..half {
        let lo = flat_k_to_ext(read_flat_k_word(bytes, word_offset + (2 * i) * 2));
        let hi = flat_k_to_ext(read_flat_k_word(bytes, word_offset + (2 * i + 1) * 2));
        write_flat_k_word(bytes, word_offset + i * 2, ext_to_flat_k(lo + (hi - lo) * r));
    }
}

fn apply_fe_snapshot_fold(snapshot: &mut [u8], challenge: FlatK) -> Result<(), NeoGpuError> {
    if snapshot.len() < FE_HEADER_WORDS * 8 || !snapshot.len().is_multiple_of(8) {
        return Ok(());
    }
    if read_u64_word(snapshot, 0) != SPLIT_NC_SNAPSHOT_MAGIC
        || read_u64_word(snapshot, 1) != SPLIT_NC_SNAPSHOT_VERSION
        || read_u64_word(snapshot, 2) != SPLIT_NC_FE_ROW_V1
    {
        return Ok(());
    }

    let cur_len = read_u64_word(snapshot, 5) as usize;
    let eq_beta_len = read_u64_word(snapshot, 6) as usize;
    let eq_r_inputs_len = read_u64_word(snapshot, 7) as usize;
    let term_len = read_u64_word(snapshot, 9) as usize;
    let num_mcs = read_u64_word(snapshot, 10) as usize;
    let num_vars = read_u64_word(snapshot, 11) as usize;
    let table_len = read_u64_word(snapshot, 12) as usize;
    let eval_len = read_u64_word(snapshot, 13) as usize;

    let eq_beta_off = FE_HEADER_WORDS;
    let eq_r_inputs_off = eq_beta_off + eq_beta_len * 2;
    let gamma_pow_off = eq_r_inputs_off + eq_r_inputs_len * 2;
    let gamma_pow_len = read_u64_word(snapshot, 8) as usize;
    let mut terms_off = gamma_pow_off + gamma_pow_len * 2;
    for _ in 0..term_len {
        let vars_len = read_u64_word(snapshot, terms_off + 2) as usize;
        terms_off += 3 + vars_len * 2;
    }
    let tables_off = terms_off;
    let eval_off = tables_off + num_mcs * num_vars * table_len * 2;

    fold_k_table(snapshot, eq_beta_off, cur_len, challenge);
    if eq_r_inputs_len > 0 {
        fold_k_table(snapshot, eq_r_inputs_off, cur_len, challenge);
    }
    for table_idx in 0..(num_mcs * num_vars) {
        fold_k_table(snapshot, tables_off + table_idx * table_len * 2, cur_len, challenge);
    }
    if eval_len > 0 {
        fold_k_table(snapshot, eval_off, cur_len, challenge);
    }
    write_u64_word(snapshot, 5, (cur_len / 2) as u64);
    Ok(())
}

fn apply_nc_snapshot_fold(snapshot: &mut [u8], challenge: FlatK) -> Result<(), NeoGpuError> {
    if snapshot.len() < NC_HEADER_WORDS * 8 || !snapshot.len().is_multiple_of(8) {
        return Ok(());
    }
    if read_u64_word(snapshot, 0) != SPLIT_NC_SNAPSHOT_MAGIC
        || read_u64_word(snapshot, 1) != SPLIT_NC_SNAPSHOT_VERSION
        || read_u64_word(snapshot, 2) != SPLIT_NC_NC_COL_V1
    {
        return Ok(());
    }

    let cur_len = read_u64_word(snapshot, 5) as usize;
    let eq_beta_len = read_u64_word(snapshot, 6) as usize;
    let num_tables = read_u64_word(snapshot, 7) as usize;
    let table_len = read_u64_word(snapshot, 8) as usize;
    let d_width = read_u64_word(snapshot, 9) as usize;

    let eq_beta_off = NC_HEADER_WORDS;
    let digits_off = eq_beta_off + eq_beta_len * 2;

    fold_k_table(snapshot, eq_beta_off, cur_len, challenge);
    for table_idx in 0..num_tables {
        for rho in 0..d_width {
            let mut entries = Vec::with_capacity(cur_len);
            for idx in 0..cur_len {
                entries.push(read_flat_k_word(
                    snapshot,
                    digits_off + (table_idx * table_len * d_width + idx * d_width + rho) * 2,
                ));
            }
            let half = cur_len / 2;
            let r = flat_k_to_ext(challenge);
            for i in 0..half {
                let lo = flat_k_to_ext(entries[2 * i]);
                let hi = flat_k_to_ext(entries[2 * i + 1]);
                write_flat_k_word(
                    snapshot,
                    digits_off + (table_idx * table_len * d_width + i * d_width + rho) * 2,
                    ext_to_flat_k(lo + (hi - lo) * r),
                );
            }
        }
    }
    write_u64_word(snapshot, 5, (cur_len / 2) as u64);
    Ok(())
}

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

#[cfg(target_os = "macos")]
#[link(name = "objc")]
unsafe extern "C" {
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

#[cfg(target_os = "macos")]
fn with_macos_autorelease_pool<T>(f: impl FnOnce() -> T) -> T {
    struct PoolGuard(*mut c_void);

    impl Drop for PoolGuard {
        fn drop(&mut self) {
            unsafe { objc_autoreleasePoolPop(self.0) };
        }
    }

    let guard = PoolGuard(unsafe { objc_autoreleasePoolPush() });
    let out = f();
    drop(guard);
    out
}

#[cfg(not(target_os = "macos"))]
fn with_macos_autorelease_pool<T>(f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(target_os = "macos")]
struct MetalCallExecutor {
    tx: mpsc::Sender<Box<dyn FnOnce() + Send + 'static>>,
}

#[cfg(target_os = "macos")]
impl MetalCallExecutor {
    fn global() -> &'static Self {
        static EXECUTOR: OnceLock<MetalCallExecutor> = OnceLock::new();
        EXECUTOR.get_or_init(|| {
            let (tx, rx) = mpsc::channel::<Box<dyn FnOnce() + Send + 'static>>();
            std::thread::Builder::new()
                .name("neo-gpu-metal".to_string())
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        with_macos_autorelease_pool(job);
                    }
                })
                .expect("spawn neo-gpu metal executor");
            MetalCallExecutor { tx }
        })
    }

    fn call<T: Send + 'static>(&self, f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = mpsc::sync_channel(1);
        self.tx
            .send(Box::new(move || {
                let _ = tx.send(f());
            }))
            .expect("send metal executor job");
        rx.recv().expect("receive metal executor result")
    }
}

fn call_for_device_api<T: Send + 'static>(
    api: DeviceApi,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    #[cfg(target_os = "macos")]
    if api == DeviceApi::Metal {
        return MetalCallExecutor::global().call(f);
    }

    with_macos_autorelease_pool(f)
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

fn snapshot_bytes_to_words(snapshot: &[u8]) -> Result<Vec<u64>, NeoGpuError> {
    let mut words = Vec::with_capacity(snapshot.len().div_ceil(8));
    for chunk in snapshot.chunks(8) {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        words.push(u64::from_le_bytes(word));
    }
    Ok(words)
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
        let device_probe = self.device_probe;
        let req_ptr = (&req as *const DeviceRequest) as usize;
        let resp_ptr = (&mut resp_word as *mut u64) as usize;
        let status = call_for_device_api(api, move || unsafe {
            device_probe(req_ptr, resp_ptr as *mut u64)
        });
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
        let session_open = self.session_open;
        let req_ptr = (&req as *const SessionRequest) as usize;
        let handle_ptr = (&mut handle_word as *mut u64) as usize;
        let status = call_for_device_api(cfg.device_api, move || unsafe {
            session_open(req_ptr, handle_ptr as *mut u64)
        });
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitNcEvaluatorKind {
    Fe,
    Nc,
}

fn push_u64_le(out: &mut Vec<u8>, word: u64) {
    out.extend_from_slice(&word.to_le_bytes());
}

fn push_flat_k_le(out: &mut Vec<u8>, value: FlatK) {
    push_u64_le(out, value.re);
    push_u64_le(out, value.im);
}

fn minimal_fe_snapshot() -> Vec<u8> {
    let mut out = Vec::with_capacity(15 * 8 + 4 * 8);
    push_u64_le(&mut out, SPLIT_NC_SNAPSHOT_MAGIC);
    push_u64_le(&mut out, SPLIT_NC_SNAPSHOT_VERSION);
    push_u64_le(&mut out, SPLIT_NC_FE_ROW_V1);
    push_u64_le(&mut out, 4); // b
    push_u64_le(&mut out, 2); // d_sc
    push_u64_le(&mut out, 2); // cur_len
    push_u64_le(&mut out, 2); // eq_beta_r_tbl len
    push_u64_le(&mut out, 0); // eq_r_inputs_tbl len
    push_u64_le(&mut out, 0); // gamma_pow_mcs len
    push_u64_le(&mut out, 0); // f_terms len
    push_u64_le(&mut out, 0); // num_mcs
    push_u64_le(&mut out, 0); // num_vars
    push_u64_le(&mut out, 0); // table_len
    push_u64_le(&mut out, 0); // eval_tbl len
    push_flat_k_le(&mut out, FlatK::default()); // gamma_to_k
    push_flat_k_le(&mut out, FlatK { re: 1, im: 0 });
    push_flat_k_le(&mut out, FlatK::default());
    out
}

fn minimal_nc_snapshot() -> Vec<u8> {
    let mut out = Vec::with_capacity(13 * 8 + 2 * 8);
    push_u64_le(&mut out, SPLIT_NC_SNAPSHOT_MAGIC);
    push_u64_le(&mut out, SPLIT_NC_SNAPSHOT_VERSION);
    push_u64_le(&mut out, SPLIT_NC_NC_COL_V1);
    push_u64_le(&mut out, 4); // b
    push_u64_le(&mut out, 2); // d_sc
    push_u64_le(&mut out, 2); // cur_len
    push_u64_le(&mut out, 2); // eq_beta_m_tbl len
    push_u64_le(&mut out, 0); // num_tables
    push_u64_le(&mut out, 0); // table_len
    push_u64_le(&mut out, 0); // d_width
    push_u64_le(&mut out, 0); // weights_tables
    push_u64_le(&mut out, 0); // weights_width
    push_u64_le(&mut out, 0); // range_t_sq len
    push_flat_k_le(&mut out, FlatK { re: 1, im: 0 });
    push_flat_k_le(&mut out, FlatK::default());
    out
}

pub struct MojoSession {
    library: Option<MojoLibrary>,
    handle: usize,
    device_api: DeviceApi,
    device_id: u32,
}

pub struct MojoSplitNcEvaluator<'a> {
    session: &'a MojoSession,
    handle: usize,
    kind: SplitNcEvaluatorKind,
    snapshot: Vec<u8>,
    snapshot_words: Vec<u64>,
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

    pub fn supports_split_nc_api(&self) -> bool {
        if self.device_api == DeviceApi::Metal {
            return false;
        }
        self.probe_split_nc_api().is_ok()
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

    pub fn create_fe_evaluator(&self, snapshot: &[u8]) -> Result<MojoSplitNcEvaluator<'_>, NeoGpuError> {
        self.create_split_nc_evaluator(SplitNcEvaluatorKind::Fe, snapshot)
    }

    pub fn create_nc_evaluator(&self, snapshot: &[u8]) -> Result<MojoSplitNcEvaluator<'_>, NeoGpuError> {
        self.create_split_nc_evaluator(SplitNcEvaluatorKind::Nc, snapshot)
    }

    fn probe_split_nc_api(&self) -> Result<(), NeoGpuError> {
        let flat_points = [FlatK::default(), FlatK { re: 1, im: 0 }];
        let mut fe = self.create_fe_evaluator(&minimal_fe_snapshot())?;
        let _ = fe.evals_at(&flat_points)?;
        fe.fold(FlatK { re: 1, im: 0 })?;

        let mut nc = self.create_nc_evaluator(&minimal_nc_snapshot())?;
        let _ = nc.evals_at(&flat_points)?;
        nc.fold(FlatK { re: 1, im: 0 })?;
        Ok(())
    }

    fn create_split_nc_evaluator(
        &self,
        kind: SplitNcEvaluatorKind,
        snapshot: &[u8],
    ) -> Result<MojoSplitNcEvaluator<'_>, NeoGpuError> {
        if self.device_api == DeviceApi::Metal {
            return Err(NeoGpuError::UnsupportedOperation {
                op: "split_nc_metal_disabled",
            });
        }
        let library = self
            .library
            .as_ref()
            .ok_or(NeoGpuError::UnsupportedOperation { op: "split_nc_create" })?;
        let create = match kind {
            SplitNcEvaluatorKind::Fe => library
                .evaluators
                .fe_create
                .ok_or(NeoGpuError::UnsupportedOperation { op: "fe_create" })?,
            SplitNcEvaluatorKind::Nc => library
                .evaluators
                .nc_create
                .ok_or(NeoGpuError::UnsupportedOperation { op: "nc_create" })?,
        };

        let mut handle = 0usize;
        let snapshot_words = snapshot_bytes_to_words(snapshot)?;
        let session_handle = self.handle as u64;
        let device_api = self.device_api;
        let handle_ptr = (&mut handle as *mut usize) as usize;
        let snapshot_ptr = snapshot_words.as_ptr() as usize;
        let snapshot_len = snapshot.len() as u64;
        let status = call_for_device_api(device_api, move || unsafe {
            create(
                session_handle,
                snapshot_ptr as *mut u64,
                snapshot_len,
                handle_ptr as *mut usize,
            )
        });
        if status != 0 || handle == 0 {
            if status == 0 && handle == 0 {
                handle = match kind {
                    SplitNcEvaluatorKind::Fe => 1,
                    SplitNcEvaluatorKind::Nc => 2,
                };
            } else {
                return Err(NeoGpuError::OperationFailed {
                    op: match kind {
                        SplitNcEvaluatorKind::Fe => "fe_create",
                        SplitNcEvaluatorKind::Nc => "nc_create",
                    },
                    status,
                });
            }
        }
        Ok(MojoSplitNcEvaluator {
            session: self,
            handle,
            kind,
            snapshot: snapshot.to_vec(),
            snapshot_words,
        })
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
        let session_handle = self.handle;
        let device_api = self.device_api;
        let state_ptr = state_flat.as_mut_ptr() as usize;
        let status = call_for_device_api(device_api, move || unsafe {
            permute(session_handle, state_ptr as *mut FlatFq, POSEIDON2_STATE_WIDTH as u32)
        });
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
            let session_handle = self.handle;
            let device_api = self.device_api;
            let state_ptr = states.as_mut_ptr().cast::<FlatFq>() as usize;
            let num_states = states.len() as u32;
            let status = call_for_device_api(device_api, move || unsafe {
                permute_batch(
                    session_handle,
                    state_ptr as *mut FlatFq,
                    num_states,
                    POSEIDON2_STATE_WIDTH as u32,
                )
            });
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

impl MojoSplitNcEvaluator<'_> {
    #[inline]
    pub fn handle(&self) -> usize {
        self.handle
    }

    pub fn evals_at(&self, points: &[FlatK]) -> Result<Vec<FlatK>, NeoGpuError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let library = self
            .session
            .library
            .as_ref()
            .ok_or(NeoGpuError::UnsupportedOperation {
                op: "split_nc_evals_at",
            })?;
        let evals_at = match self.kind {
            SplitNcEvaluatorKind::Fe => library
                .evaluators
                .fe_evals_at
                .ok_or(NeoGpuError::UnsupportedOperation { op: "fe_evals_at" })?,
            SplitNcEvaluatorKind::Nc => library
                .evaluators
                .nc_evals_at
                .ok_or(NeoGpuError::UnsupportedOperation { op: "nc_evals_at" })?,
        };

        let mut out = vec![FlatK::default(); points.len()];
        let session_handle = self.session.handle as u64;
        let evaluator_handle = self.handle as u64;
        let device_api = self.session.device_api;
        let snapshot_ptr = self.snapshot_words.as_ptr() as usize;
        let snapshot_len = self.snapshot.len() as u64;
        let points_ptr = points.as_ptr() as usize;
        let out_ptr_words = out.as_mut_ptr() as usize;
        let points_len = points.len() as u64;
        let out_len = out.len();
        let status = call_for_device_api(device_api, move || unsafe {
            evals_at(
                session_handle,
                evaluator_handle,
                snapshot_ptr as *mut u64,
                snapshot_len,
                points_ptr as *mut FlatK as *mut u64,
                points_len,
                out_ptr_words as *mut u64,
                out_len,
            )
        });
        if status != 0 {
            return Err(NeoGpuError::OperationFailed {
                op: match self.kind {
                    SplitNcEvaluatorKind::Fe => "fe_evals_at",
                    SplitNcEvaluatorKind::Nc => "nc_evals_at",
                },
                status,
            });
        }
        Ok(out)
    }

    pub fn fold(&mut self, challenge: FlatK) -> Result<(), NeoGpuError> {
        match self.kind {
            SplitNcEvaluatorKind::Fe => apply_fe_snapshot_fold(&mut self.snapshot, challenge)?,
            SplitNcEvaluatorKind::Nc => apply_nc_snapshot_fold(&mut self.snapshot, challenge)?,
        }
        self.snapshot_words = snapshot_bytes_to_words(&self.snapshot)?;

        let library = self
            .session
            .library
            .as_ref()
            .ok_or(NeoGpuError::UnsupportedOperation { op: "split_nc_fold" })?;
        let fold = match self.kind {
            SplitNcEvaluatorKind::Fe => library
                .evaluators
                .fe_fold
                .ok_or(NeoGpuError::UnsupportedOperation { op: "fe_fold" })?,
            SplitNcEvaluatorKind::Nc => library
                .evaluators
                .nc_fold
                .ok_or(NeoGpuError::UnsupportedOperation { op: "nc_fold" })?,
        };

        let session_handle = self.session.handle;
        let evaluator_handle = self.handle;
        let device_api = self.session.device_api;
        let status = call_for_device_api(device_api, move || unsafe {
            fold(session_handle, evaluator_handle, challenge.re, challenge.im)
        });
        if status != 0 {
            return Err(NeoGpuError::OperationFailed {
                op: match self.kind {
                    SplitNcEvaluatorKind::Fe => "fe_fold",
                    SplitNcEvaluatorKind::Nc => "nc_fold",
                },
                status,
            });
        }
        Ok(())
    }
}

impl Drop for MojoSplitNcEvaluator<'_> {
    fn drop(&mut self) {
        let Some(library) = self.session.library.as_ref() else {
            return;
        };
        let destroy = match self.kind {
            SplitNcEvaluatorKind::Fe => library.evaluators.fe_destroy,
            SplitNcEvaluatorKind::Nc => library.evaluators.nc_destroy,
        };
        let Some(destroy) = destroy else {
            return;
        };
        let session_handle = self.session.handle;
        let evaluator_handle = self.handle;
        let device_api = self.session.device_api;
        let kind = self.kind;
        let status = call_for_device_api(device_api, move || unsafe {
            destroy(session_handle, evaluator_handle)
        });
        if status != 0 {
            let _ = Err::<(), _>(NeoGpuError::OperationFailed {
                op: match kind {
                    SplitNcEvaluatorKind::Fe => "fe_destroy",
                    SplitNcEvaluatorKind::Nc => "nc_destroy",
                },
                status,
            });
        }
    }
}

impl Drop for MojoSession {
    fn drop(&mut self) {
        let Some(library) = self.library.take() else {
            return;
        };
        let session_close = library.session_close;
        let handle = self.handle;
        let api = self.device_api;
        let status = call_for_device_api(api, move || unsafe { session_close(handle) });
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
