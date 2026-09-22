//! Captures candidate entry bytes from live memory and recovers declared inputs.
//! Normalization owns turn selection and replay; snapshots never seed replay RAM.
//!
//! The reason we need this is that inputs don't have corresponding wasm
//! instructions that we could trace, the memory is just written by the host and
//! observed by the engine.
//!
//! Because of that, we need to use the binding/script/template information to look at the memory before execution and capture those values.

use std::collections::{BTreeMap, BTreeSet};

use wasmtime::{FrameHandle, StoreContextMut};

use super::memory_address::host_event_address;
use super::{runtime_read::read_memory_bytes, LoweringTables, WasmtimeTraceStep};
use crate::host_event_bindings::{ExportTemplate, Limb, MemoryBase, SlotBinding};
use crate::ir::WasmBuildError;

pub(super) fn capture_entry_memory<T>(
    row: &mut WasmtimeTraceStep,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, T>,
    tables: &LoweringTables,
) {
    let Some(fref) = row.current_function_ref else {
        return;
    };
    let Some(template) = tables.artifacts.host_event_bindings.exports.get(&fref) else {
        return;
    };
    let entry_pc = tables
        .artifacts
        .trace
        .function_metas
        .get(&fref)
        .and_then(|meta| meta.entry_pc);
    if entry_pc.is_none() || entry_pc != row.pc.map(u64::from) {
        return;
    }
    // A function with an export binding may also be called by another guest
    // function. Keep capture errors on this candidate, not on the whole trace.
    row.entry_memory = Some(
        capture_bytes(template, &row.locals_words, row.memory_pages_before, frame, store)
            .map_err(|err| format!("fref {fref}, step {}: {err}", row.step)),
    );
}

fn capture_bytes<T>(
    template: &ExportTemplate,
    locals: &[(u32, u32)],
    pages: Option<u32>,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, T>,
) -> Result<BTreeMap<u32, u8>, WasmBuildError> {
    let mut captured = BTreeMap::new();
    for write in entry_memory_writes(template, locals, pages)? {
        let address = u64::from(write.address);
        let bytes = match write.width {
            1 => read_memory_bytes::<1, _>(0, address, frame, store)?.to_vec(),
            2 => read_memory_bytes::<2, _>(0, address, frame, store)?.to_vec(),
            4 => read_memory_bytes::<4, _>(0, address, frame, store)?.to_vec(),
            _ => unreachable!("entry write width"),
        };
        for (offset, byte) in bytes.into_iter().enumerate() {
            captured.insert(write.address + offset as u32, byte);
        }
    }
    Ok(captured)
}

pub(super) fn recover_entry_inputs(
    template: &ExportTemplate,
    locals: &[(u32, u32)],
    pages: Option<u32>,
    captured: Option<&Result<BTreeMap<u32, u8>, String>>,
) -> Result<Vec<u64>, WasmBuildError> {
    let mut inputs = vec![None; usize::from(template.entry_input_count)];
    for slot in template.entry.iter().flat_map(|event| &event.block) {
        if let SlotBinding::InputLocal { input, local, limb } = *slot {
            let &(lo, hi) = locals
                .get(usize::from(local))
                .ok_or_else(|| WasmBuildError::Trace(format!("entry local {local} is missing from the capture")))?;
            bind_input(
                &mut inputs,
                input,
                match limb {
                    Limb::Lo => lo,
                    Limb::Hi => hi,
                },
            )?;
        }
    }
    let writes = entry_memory_writes(template, locals, pages)?;
    if !writes.is_empty() {
        let bytes = captured
            .ok_or_else(|| {
                WasmBuildError::Trace(
                    "missing entry memory capture; configure capture bindings before invoking exports".into(),
                )
            })?
            .as_ref()
            .map_err(|error| WasmBuildError::Trace(format!("entry memory capture failed: {error}")))?;
        for write in writes {
            let mut value = [0; 4];
            for offset in 0..write.width {
                let address = write.address + u32::from(offset);
                value[usize::from(offset)] = *bytes.get(&address).ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "missing entry memory byte at address {address} for input {} (width {})",
                        write.input, write.width
                    ))
                })?;
            }
            bind_input(&mut inputs, write.input, u32::from_le_bytes(value))?;
        }
    }
    let entry = inputs
        .into_iter()
        .enumerate()
        .map(|(input, value)| {
            value
                .map(u64::from)
                .ok_or_else(|| WasmBuildError::Trace(format!("entry input {input} has no recoverable mapping")))
        })
        .collect::<Result<_, _>>()?;
    Ok(entry)
}

fn bind_input(inputs: &mut [Option<u32>], input: u8, value: u32) -> Result<(), WasmBuildError> {
    let slot = inputs
        .get_mut(usize::from(input))
        .ok_or_else(|| WasmBuildError::Trace(format!("entry input {input} exceeds the declared input count")))?;
    if let Some(previous) = slot.replace(value) {
        if previous != value {
            return Err(WasmBuildError::Trace(format!(
                "conflicting mappings for entry input {input}: {previous} and {value}"
            )));
        }
    }
    Ok(())
}

struct EntryMemoryWrite {
    input: u8,
    address: u32,
    width: u8,
}

pub(super) fn entry_memory_pointer(locals: &[(u32, u32)], local: u8) -> Result<u32, WasmBuildError> {
    let &(lo, hi) = locals
        .get(usize::from(local))
        .ok_or_else(|| WasmBuildError::Trace(format!("entry memory base local {local} is missing")))?;
    if hi != 0 {
        return Err(WasmBuildError::Trace(format!(
            "entry memory base local {local} is not a wasm32 pointer"
        )));
    }
    Ok(lo)
}

fn entry_memory_writes(
    template: &ExportTemplate,
    locals: &[(u32, u32)],
    pages: Option<u32>,
) -> Result<Vec<EntryMemoryWrite>, WasmBuildError> {
    let mut writes = Vec::new();
    let mut covered = BTreeSet::new();
    for slot in template.entry.iter().flat_map(|event| &event.block) {
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
        let MemoryBase::Local(local) = base else {
            return Err(WasmBuildError::Trace("entry memory writes require a local base".into()));
        };
        let lo = entry_memory_pointer(locals, local)?;
        let address = host_event_address(lo, byte_offset, width, pages)?;
        for offset in 0..width {
            if !covered.insert(address + u32::from(offset)) {
                return Err(WasmBuildError::Trace(format!(
                    "overlapping entry memory writes at address {}; automatic recovery requires disjoint writes",
                    address + u32::from(offset)
                )));
            }
        }
        writes.push(EntryMemoryWrite { input, address, width });
    }
    Ok(writes)
}
