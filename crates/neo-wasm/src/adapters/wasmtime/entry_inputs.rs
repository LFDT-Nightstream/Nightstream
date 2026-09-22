//! Captures candidate entry bytes from live memory and recovers declared inputs.
//! Normalization owns turn selection and replay; snapshots never seed replay RAM.
//!
//! The reason we need this is that inputs don't have corresponding wasm
//! instructions that we could trace, the memory is just written by the host and
//! observed by the engine.
//!
//! Because of that, we need to use the binding/script/template information to look at the memory before execution and capture those values.

use std::collections::BTreeMap;

use wasmtime::{FrameHandle, StoreContextMut};

use super::memory_address::memory_pointer;
use super::memory_inputs::{
    bind_input, capture_bytes, finish_inputs, memory_input_writes, recover_memory_inputs, MemoryInputWrite,
};
use super::{LoweringTables, WasmtimeTraceStep};
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
        entry_memory_writes(template, &row.locals_words, row.memory_pages_before)
            .and_then(|writes| capture_bytes(&writes, frame, store))
            .map_err(|err| format!("fref {fref}, step {}: {err}", row.step)),
    );
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
                "entry",
            )?;
        }
    }
    let writes = entry_memory_writes(template, locals, pages)?;
    recover_memory_inputs(&mut inputs, &writes, captured, "entry")?;
    finish_inputs(inputs, "entry")
}

fn entry_memory_writes(
    template: &ExportTemplate,
    locals: &[(u32, u32)],
    pages: Option<u32>,
) -> Result<Vec<MemoryInputWrite>, WasmBuildError> {
    memory_input_writes(
        &template.entry,
        |base| {
            let MemoryBase::Local(local) = base else {
                return Err(WasmBuildError::Trace("entry memory writes require a local base".into()));
            };
            memory_pointer(locals, local, "entry local")
        },
        pages,
        "entry",
    )
}
