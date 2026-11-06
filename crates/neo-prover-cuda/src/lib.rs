//! Native CUDA prover backend for SuperNeo.
//!
//! The protocol crates remain authoritative for transcript order, proof
//! assembly, and verification. This crate owns CUDA device state and the
//! row evaluator. It is the CUDA port of `neo-prover-metal`: each kernel
//! file ports one Metal shader file, and each result equals the Metal and
//! the CPU result.

#![warn(clippy::undocumented_unsafe_blocks)]

pub mod poseidon2;

use thiserror::Error;

#[cfg(feature = "cuda")]
mod kernels;
#[cfg(feature = "cuda")]
mod rows;
#[cfg(feature = "cuda")]
mod session;
#[cfg(feature = "cuda")]
pub use rows::CudaRowProver;
#[cfg(feature = "cuda")]
pub use session::{CudaGkrLayer, CudaSession};

/// The owner's device memory ceiling: the Metal ceiling
/// (`neo_prover_metal::BUFFER_LIMIT_BYTES`), which is below the 24,564 MiB
/// of the RTX 4090 that the port targets. It is fixed, never read from the
/// device. Every session of a process allocates from it; CPU memory is
/// measured separately.
pub const BUFFER_LIMIT_BYTES: usize = 24_000_000_000;

pub type PoseidonDigest = [u64; 4];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaActivity {
    pub launches: u64,
    pub host_waits: u64,
    pub allocated_bytes: u64,
    pub uploaded_bytes: u64,
    pub downloaded_bytes: u64,
}

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA device creation failed: {0}")]
    Device(String),
    #[error("CUDA kernel module load failed: {0}")]
    Module(String),
    #[error("CUDA buffer allocation failed for {bytes} bytes")]
    Buffer { bytes: usize },
    #[error(
        "CUDA buffer request for {requested} bytes with {allocated} bytes allocated exceeds the {limit}-byte limit"
    )]
    MemoryLimit {
        requested: usize,
        allocated: usize,
        limit: usize,
    },
    #[error("CUDA command execution failed: {0}")]
    Execution(String),
    #[error("CUDA input shape mismatch: {0}")]
    Shape(&'static str),
    #[error("CUDA fixed-key commitment input: {0}")]
    Commitment(#[from] neo_ajtai::AjtaiError),
}

#[cfg(feature = "cuda")]
pub(crate) fn oracle_error(error: CudaError) -> neo_reductions::PiCcsError {
    match error {
        CudaError::Shape(reason) => neo_reductions::PiCcsError::InvalidInput(reason.into()),
        CudaError::Commitment(error) => neo_reductions::PiCcsError::InvalidInput(error.to_string()),
        error => neo_reductions::PiCcsError::BackendFailure {
            backend: "cuda",
            reason: error.to_string(),
        },
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "../tests/unit/oracle_errors.rs"]
mod oracle_errors;

#[cfg(feature = "perf-timers")]
pub mod perf_ranges {
    pub struct NvtxRange {
        id: Option<u64>,
    }

    impl NvtxRange {
        pub fn push(label: &str) -> Self {
            Self { id: nvtx::start(label) }
        }
    }

    impl Drop for NvtxRange {
        fn drop(&mut self) {
            if let Some(id) = self.id.take() {
                nvtx::end(id);
            }
        }
    }

    #[cfg(target_os = "linux")]
    mod nvtx {
        use std::{
            ffi::CString,
            os::raw::{c_char, c_int, c_void},
            sync::OnceLock,
        };

        type Start = unsafe extern "C" fn(*const c_char) -> u64;
        type End = unsafe extern "C" fn(u64) -> c_int;

        struct Api {
            start: Start,
            end: End,
        }

        #[link(name = "dl")]
        extern "C" {
            fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
            fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        }

        const RTLD_LAZY: c_int = 1;
        const RTLD_LOCAL: c_int = 0;
        static API: OnceLock<Option<Api>> = OnceLock::new();

        pub fn start(label: &str) -> Option<u64> {
            let Some(api) = API.get_or_init(load).as_ref() else {
                return None;
            };
            let Ok(label) = CString::new(label) else {
                return None;
            };
            let id = unsafe { (api.start)(label.as_ptr()) };
            (id != 0).then_some(id)
        }

        pub fn end(id: u64) {
            if let Some(api) = API.get_or_init(load).as_ref() {
                unsafe {
                    (api.end)(id);
                }
            }
        }

        fn load() -> Option<Api> {
            for lib in [
                "libnvToolsExt.so.1",
                "libnvToolsExt.so",
                "libnvtx3interop.so.1",
                "libnvtx3interop.so",
            ] {
                let lib = CString::new(lib).ok()?;
                let handle = unsafe { dlopen(lib.as_ptr(), RTLD_LAZY | RTLD_LOCAL) };
                if handle.is_null() {
                    continue;
                }
                let start = symbol::<Start>(handle, b"nvtxRangeStartA\0")?;
                let end = symbol::<End>(handle, b"nvtxRangeEnd\0")?;
                return Some(Api { start, end });
            }
            None
        }

        fn symbol<T>(handle: *mut c_void, name: &[u8]) -> Option<T> {
            let ptr = unsafe { dlsym(handle, name.as_ptr().cast()) };
            if ptr.is_null() {
                None
            } else {
                Some(unsafe { std::mem::transmute_copy(&ptr) })
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    mod nvtx {
        pub fn start(_label: &str) -> Option<u64> {
            None
        }

        pub fn end(_id: u64) {}
    }
}

#[cfg(feature = "perf-timers")]
#[macro_export]
macro_rules! perf_timed {
    ($label:expr, $body:block) => {{
        let __nvtx_range = $crate::perf_ranges::NvtxRange::push($label);
        let __timer = std::time::Instant::now();
        $body
        drop(__nvtx_range);
        let __epoch_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        eprintln!(
            "[neo-prover-cuda] {:<12} {:>8.2}ms @{}",
            $label,
            __timer.elapsed().as_secs_f64() * 1e3,
            __epoch_ns
        );
    }};
}
#[cfg(not(feature = "perf-timers"))]
#[macro_export]
macro_rules! perf_timed {
    ($label:expr, $body:block) => {
        $body
    };
}
