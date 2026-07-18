//! Narrow C ABI used by the iOS benchmark application.

use std::ptr;

use crate::{run_benchmark_json, BenchmarkConfig};

/// Runs a benchmark and returns either report JSON or error JSON in Rust-owned memory.
///
/// # Safety
///
/// Non-null input pointers must reference their declared byte lengths. Output
/// pointers must be writable, and returned buffers must be released exactly
/// once with [`neo_metal_benchmark_free_bytes`] using the returned length.
#[no_mangle]
pub unsafe extern "C" fn neo_metal_benchmark_run_json(
    config_ptr: *const u8,
    config_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
    error_ptr: *mut *mut u8,
    error_len: *mut usize,
) -> i32 {
    if out_ptr.is_null() || out_len.is_null() || error_ptr.is_null() || error_len.is_null() {
        return -1;
    }
    unsafe {
        *out_ptr = ptr::null_mut();
        *out_len = 0;
        *error_ptr = ptr::null_mut();
        *error_len = 0;
    }

    let config = if config_len == 0 {
        Ok(BenchmarkConfig::default())
    } else if config_ptr.is_null() {
        Err("configuration pointer is null while length is nonzero".to_owned())
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(config_ptr, config_len) };
        serde_json::from_slice(bytes).map_err(|error| format!("invalid benchmark configuration JSON: {error}"))
    };
    let result = match config {
        Ok(config) => run_benchmark_json(config).map_err(|error| error.to_string()),
        Err(error) => Err(error),
    };
    match result {
        Ok(json) => {
            unsafe { return_bytes(json.into_bytes(), out_ptr, out_len) };
            0
        }
        Err(error) => {
            let json = serde_json::json!({ "error": error }).to_string();
            unsafe { return_bytes(json.into_bytes(), error_ptr, error_len) };
            1
        }
    }
}

/// Releases a byte buffer returned by [`neo_metal_benchmark_run_json`].
///
/// # Safety
///
/// `ptr` and `len` must be an unchanged pair returned by this library and must
/// not have been freed previously. A null pointer is accepted as a no-op.
#[no_mangle]
pub unsafe extern "C" fn neo_metal_benchmark_free_bytes(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let slice = ptr::slice_from_raw_parts_mut(ptr, len);
    unsafe {
        drop(Box::<[u8]>::from_raw(slice));
    }
}

unsafe fn return_bytes(bytes: Vec<u8>, out_ptr: *mut *mut u8, out_len: *mut usize) {
    let mut bytes = bytes.into_boxed_slice();
    unsafe {
        *out_ptr = bytes.as_mut_ptr();
        *out_len = bytes.len();
    }
    std::mem::forget(bytes);
}
