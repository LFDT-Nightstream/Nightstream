//! Targeted live-memory capture and input recovery shared by host boundaries.
//! Captured bytes supply declared writes; they never initialize replay memory.

use super::{memory_address::host_event_address, runtime_read::read_memory_bytes};
use crate::host_event_bindings::{EventBlock, MemoryBase, SlotBinding};
use crate::WasmBuildError;
use std::collections::{BTreeMap, BTreeSet};
use wasmtime::{FrameHandle, StoreContextMut};

#[derive(Debug)]
pub(super) struct MemoryInputWrite {
    input: u8,
    address: u32,
    width: u8,
}

pub(super) fn memory_input_writes(
    events: &[EventBlock],
    mut pointer: impl FnMut(MemoryBase) -> Result<u32, WasmBuildError>,
    pages: Option<u32>,
    boundary: &str,
) -> Result<Vec<MemoryInputWrite>, WasmBuildError> {
    let mut writes = Vec::new();
    let mut covered = BTreeSet::new();
    for slot in events.iter().flat_map(|event| &event.block) {
        let (input, base, byte_offset, width) = match *slot {
            SlotBinding::MemoryWrite8 {
                input,
                base,
                byte_offset,
            } => (input, base, byte_offset, 1u8),
            SlotBinding::MemoryWrite16 {
                input,
                base,
                byte_offset,
            } => (input, base, byte_offset, 2),
            SlotBinding::MemoryWrite32 {
                input,
                base,
                byte_offset,
            } => (input, base, byte_offset, 4),
            _ => continue,
        };
        let address = host_event_address(pointer(base)?, byte_offset, width, pages)?;
        for offset in 0..width {
            if !covered.insert(address + u32::from(offset)) {
                return Err(WasmBuildError::Trace(format!(
                    "overlapping {boundary} memory writes at address {}; automatic recovery requires disjoint writes",
                    address + u32::from(offset)
                )));
            }
        }
        writes.push(MemoryInputWrite { input, address, width });
    }
    Ok(writes)
}

pub(super) fn capture_bytes<T>(
    writes: &[MemoryInputWrite],
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, T>,
) -> Result<BTreeMap<u32, u8>, WasmBuildError> {
    let mut captured = BTreeMap::new();
    for write in writes {
        let address = u64::from(write.address);
        let bytes = match write.width {
            1 => read_memory_bytes::<1, _>(0, address, frame, store)?.to_vec(),
            2 => read_memory_bytes::<2, _>(0, address, frame, store)?.to_vec(),
            4 => read_memory_bytes::<4, _>(0, address, frame, store)?.to_vec(),
            _ => unreachable!("memory input write width"),
        };
        for (offset, byte) in bytes.into_iter().enumerate() {
            captured.insert(write.address + offset as u32, byte);
        }
    }
    Ok(captured)
}

pub(super) fn recover_memory_inputs(
    inputs: &mut [Option<u32>],
    writes: &[MemoryInputWrite],
    captured: Option<&Result<BTreeMap<u32, u8>, String>>,
    boundary: &str,
) -> Result<(), WasmBuildError> {
    if writes.is_empty() {
        return Ok(());
    }
    let bytes = captured
        .ok_or_else(|| WasmBuildError::Trace(format!("missing {boundary} memory capture for declared memory writes")))?
        .as_ref()
        .map_err(|error| WasmBuildError::Trace(format!("{boundary} memory capture failed: {error}")))?;
    for write in writes {
        let mut value = [0; 4];
        for offset in 0..write.width {
            let address = write.address + u32::from(offset);
            value[usize::from(offset)] = *bytes.get(&address).ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "missing {boundary} memory byte at address {address} for input {} (width {})",
                    write.input, write.width
                ))
            })?;
        }
        bind_input(inputs, write.input, u32::from_le_bytes(value), boundary)?;
    }
    Ok(())
}

pub(super) fn bind_input(
    inputs: &mut [Option<u32>],
    input: u8,
    value: u32,
    boundary: &str,
) -> Result<(), WasmBuildError> {
    let slot = inputs
        .get_mut(usize::from(input))
        .ok_or_else(|| WasmBuildError::Trace(format!("{boundary} input {input} exceeds the declared input count")))?;
    if let Some(previous) = slot.replace(value) {
        if previous != value {
            return Err(WasmBuildError::Trace(format!(
                "conflicting mappings for {boundary} input {input}: {previous} and {value}"
            )));
        }
    }
    Ok(())
}

pub(super) fn finish_inputs(inputs: Vec<Option<u32>>, boundary: &str) -> Result<Vec<u64>, WasmBuildError> {
    inputs
        .into_iter()
        .enumerate()
        .map(|(input, value)| {
            value
                .map(u64::from)
                .ok_or_else(|| WasmBuildError::Trace(format!("{boundary} input {input} has no recoverable mapping")))
        })
        .collect()
}
