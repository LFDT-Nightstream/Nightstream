//! Returns freed allocator pages to the operating system at proof-phase boundaries.

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn malloc_default_zone() -> *mut core::ffi::c_void;
    fn malloc_zone_pressure_relief(zone: *mut core::ffi::c_void, goal: usize) -> usize;
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
unsafe extern "C" {
    fn malloc_trim(pad: usize) -> core::ffi::c_int;
}

#[inline]
pub(crate) fn release_unused_pages() {
    #[cfg(target_os = "macos")]
    unsafe {
        let zone = malloc_default_zone();
        if !zone.is_null() {
            let _ = malloc_zone_pressure_relief(zone, 0);
        }
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        let _ = malloc_trim(0);
    }
}
