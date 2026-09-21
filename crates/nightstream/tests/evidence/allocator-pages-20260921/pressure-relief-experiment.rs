pub(crate) fn release_unused_pages() {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        // macOS can retain gigabytes of freed parsing and proof scratch.
        // SAFETY: the documented null zone and zero goal release unused pages
        // from all allocator zones. Allocations that are still live are retained.
        unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
    }
}
