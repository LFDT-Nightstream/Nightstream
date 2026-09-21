use crate::ir::WasmBuildError;

/// Resolve a naturally aligned host-event access of 1, 2, or 4 bytes.
/// Capture and replay share these checks; memory contents and overlap policy
/// belong to the caller. Guest instructions have separate alignment semantics.
pub(super) fn host_event_address(
    base: u32,
    byte_offset: u32,
    width: u8,
    memory_pages: Option<u32>,
) -> Result<u32, WasmBuildError> {
    debug_assert!(matches!(width, 1 | 2 | 4));
    let address = base.checked_add(byte_offset).ok_or_else(|| {
        WasmBuildError::Trace(format!(
            "host-event memory address overflows wasm32: {base} + {byte_offset}"
        ))
    })?;
    if address % u32::from(width) != 0 {
        return Err(WasmBuildError::Trace(format!(
            "host-event memory address {address} is not naturally aligned for width {width}"
        )));
    }
    let pages = memory_pages
        .ok_or_else(|| WasmBuildError::Trace("host-event memory access requires default linear memory".into()))?;
    if u64::from(address) + u64::from(width) > u64::from(pages) * 65536 {
        return Err(WasmBuildError::Trace(format!(
            "host-event memory access at byte address {address}, width {width} is out of bounds for {pages} memory pages"
        )));
    }
    Ok(address)
}
