//! CUDA selection for the canonical SuperNeo NIFS prover.
//!
//! The former CUDA kernels implemented an incompatible protocol. The
//! current adapter checks CUDA availability and then runs the selected
//! one-joint protocol through the canonical host implementation. A future GPU
//! kernel must reproduce that protocol exactly before it can replace the host
//! computation.
//!
//! Build discipline: the `cuda` feature is only built through
//! `cargo +nightly-2026-04-03 oxide` (custom rustc codegen backend for
//! `#[cuda_module]` blocks). Plain `cargo` workspace builds keep it off.
//!
//! Public API: [`CudaNifsProver`].

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

/// Time a statement block to stderr under the workspace `perf-timers`
/// convention; free when the feature is off.
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

#[cfg(feature = "cuda")]
#[doc(hidden)]
pub mod adapter;

#[cfg(feature = "cuda")]
pub use adapter::CudaNifsProver;
