//! Leaf readers that pull raw values and identities out of the live Wasmtime
//! `Store` / `Instance` / `FrameHandle` and normalize them into the lanes the
//! parent module records per step.
//!
//! Owns no trace-row assembly and no opcode classification — it only answers
//! "what does memory / a global / a table / the operand stack hold right now,
//! and what normalized funcref id does this value map to". Callers in the
//! parent module and `normalize` decide what to do with the answers.

use super::WasmtimeTraceState;
use crate::ir::{StackLaneAccess, WasmBuildError};
use std::collections::BTreeMap;
use wasmtime::{FrameHandle, Store, StoreContextMut, Val};

pub(crate) fn build_debug_function_id_map(
    instance: &wasmtime::Instance,
    mut store: impl wasmtime::AsContextMut<Data = WasmtimeTraceState>,
) -> Result<BTreeMap<usize, u32>, WasmBuildError> {
    let mut out = BTreeMap::new();
    let mut function_index = 0_u32;
    while let Some(func) = instance.debug_function(store.as_context_mut(), function_index) {
        let raw = func.to_raw(store.as_context_mut()) as usize;
        out.insert(raw, function_index.saturating_add(1));
        function_index = function_index.saturating_add(1);
    }
    Ok(out)
}

pub(crate) fn build_store_debug_function_id_map(
    store: &mut Store<WasmtimeTraceState>,
) -> Result<BTreeMap<usize, u32>, WasmBuildError> {
    let mut out = BTreeMap::new();
    for instance in store.debug_all_instances() {
        for (raw, function_id) in build_debug_function_id_map(&instance, &mut *store)? {
            out.insert(raw, function_id);
        }
    }
    Ok(out)
}

pub(crate) fn function_type_id_from_ref(
    function_ref: u32,
    store: &StoreContextMut<'_, WasmtimeTraceState>,
) -> Option<u32> {
    if function_ref == 0 {
        return Some(0);
    }
    store
        .data()
        .function_metas
        .get(&function_ref)
        .map(|meta| meta.type_id)
}

pub(crate) fn function_arity_from_ref(
    function_ref: u32,
    store: &StoreContextMut<'_, WasmtimeTraceState>,
) -> Option<(u8, u8)> {
    store
        .data()
        .function_metas
        .get(&function_ref)
        .map(|meta| (meta.param_count, meta.result_count))
}

pub(crate) fn normalize_value_lanes(
    val: Val,
    func_ref_ids: &BTreeMap<usize, u32>,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<(u32, u32), WasmBuildError> {
    match val {
        Val::I32(v) => Ok((v as u32, 0)),
        Val::I64(v) => {
            let bits = v as u64;
            Ok((bits as u32, (bits >> 32) as u32))
        }
        Val::FuncRef(None) => Ok((0, 0)),
        Val::FuncRef(Some(func)) => {
            let raw = func.to_raw(&mut *store) as usize;
            func_ref_ids
                .get(&raw)
                .copied()
                .map(|id| (id, 0))
                .ok_or_else(|| WasmBuildError::Trace(format!("missing normalized function id for funcref 0x{raw:x}")))
        }
        other => Err(WasmBuildError::Unsupported(format!(
            "unsupported Wasmtime operand stack value for current WASM row surface: {other:?}"
        ))),
    }
}

pub(crate) fn read_global_lanes(
    global_index: u32,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<(u32, u32), WasmBuildError> {
    let instance = frame
        .instance(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame instance: {err}")))?;
    let global = instance
        .debug_global(&mut *store, global_index)
        .ok_or_else(|| WasmBuildError::Trace(format!("missing Wasmtime global {} at current frame", global_index)))?;
    let func_ref_ids = store.data().func_ref_ids.clone();
    normalize_value_lanes(global.get(&mut *store), func_ref_ids.as_ref(), store)
}

pub(crate) fn read_table_funcref_u32(
    table_id: u32,
    table_index: u32,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<u32, WasmBuildError> {
    let instance = frame
        .instance(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame instance: {err}")))?;
    let table = instance
        .debug_table(&mut *store, table_id)
        .ok_or_else(|| WasmBuildError::Trace(format!("missing Wasmtime table {} at current frame", table_id)))?;
    match table.get(&mut *store, u64::from(table_index)) {
        Some(wasmtime::Ref::Func(None)) => Ok(0),
        Some(wasmtime::Ref::Func(Some(func))) => {
            let raw = func.to_raw(&mut *store) as usize;
            store.data().func_ref_ids.get(&raw).copied().ok_or_else(|| {
                WasmBuildError::Trace(format!("missing normalized function id for table funcref 0x{raw:x}"))
            })
        }
        Some(other) => Err(WasmBuildError::Unsupported(format!(
            "only funcref tables are supported right now, found value {other:?}"
        ))),
        None => Err(WasmBuildError::Trace(format!(
            "table.get out of bounds for table {} index {}",
            table_id, table_index
        ))),
    }
}

pub(crate) fn read_table_size(
    table_id: u32,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<u32, WasmBuildError> {
    let instance = frame
        .instance(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame instance: {err}")))?;
    let table = instance
        .debug_table(&mut *store, table_id)
        .ok_or_else(|| WasmBuildError::Trace(format!("missing Wasmtime table {} at current frame", table_id)))?;
    u32::try_from(table.size(&mut *store))
        .map_err(|_| WasmBuildError::Trace(format!("table {table_id} size exceeded u32")))
}

pub(crate) fn read_memory_pages_if_present(
    memory_index: u32,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<Option<u32>, WasmBuildError> {
    let instance = frame
        .instance(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame instance: {err}")))?;
    let Some(memory) = instance.debug_memory(&mut *store, memory_index) else {
        return Ok(None);
    };
    let pages = u32::try_from(memory.size(&mut *store))
        .map_err(|_| WasmBuildError::Trace(format!("memory {memory_index} page count exceeded u32")))?;
    Ok(Some(pages))
}

pub(crate) fn read_memory_bytes<const N: usize>(
    memory_index: u32,
    effective_address: u64,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<[u8; N], WasmBuildError> {
    let instance = frame
        .instance(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame instance: {err}")))?;

    let Some(memory) = instance.debug_memory(&mut *store, memory_index) else {
        return Err(WasmBuildError::Trace(format!(
            "missing Wasmtime memory {memory_index} at address {effective_address}"
        )));
    };

    let mut bytes = [0_u8; N];

    memory
        .read(&mut *store, effective_address as usize, &mut bytes)
        .map_err(|err| {
            WasmBuildError::Trace(format!(
                "failed to read Wasmtime memory at address {effective_address}: {err}"
            ))
        })?;

    Ok(bytes)
}

pub(crate) fn read_word(
    memory_index: u32,
    effective_address: u64,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<u32, WasmBuildError> {
    Ok(u32::from_le_bytes(read_memory_bytes::<4>(
        memory_index,
        effective_address,
        frame,
        store,
    )?))
}

pub(crate) fn read_byte(
    memory_index: u32,
    effective_address: u64,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<u8, WasmBuildError> {
    Ok(read_memory_bytes::<1>(memory_index, effective_address, frame, store)?[0])
}

pub(crate) fn read_halfword(
    memory_index: u32,
    effective_address: u64,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<u16, WasmBuildError> {
    Ok(u16::from_le_bytes(read_memory_bytes::<2>(
        memory_index,
        effective_address,
        frame,
        store,
    )?))
}

pub(crate) fn parse_stack_word(value: &str) -> Result<u32, WasmBuildError> {
    parse_signed_u32(value)
        .map_err(|err| WasmBuildError::Trace(format!("failed to parse Wasmtime operand stack value '{value}': {err}")))
}

pub(crate) fn parse_signed_u32(value: &str) -> Result<u32, WasmBuildError> {
    let parsed = value.parse::<i128>().map_err(|err| {
        WasmBuildError::Trace(format!("failed to parse signed i32-compatible value '{value}': {err}"))
    })?;
    Ok((parsed as i32) as u32)
}

pub(crate) fn read_lane(stack: &[u32], sp_before: u64, reads: u8, lane: usize) -> Option<StackLaneAccess> {
    let reads = reads as usize;
    if reads == 0 || lane >= reads {
        return None;
    }
    let stack_index = stack.len().checked_sub(reads)?.checked_add(lane)?;
    let slot = sp_before
        .checked_sub(reads as u64)?
        .checked_add(lane as u64)?;
    let addr = slot.checked_mul(2)?;
    stack
        .get(stack_index)
        .copied()
        .map(|value| StackLaneAccess { addr, value })
}

pub(crate) fn read_lane_hi(stack_hi: &[u32], reads: u8, lane: usize) -> Option<u32> {
    let reads = reads as usize;
    if reads == 0 || lane >= reads {
        return None;
    }
    let stack_index = stack_hi.len().checked_sub(reads)?.checked_add(lane)?;
    stack_hi.get(stack_index).copied()
}

pub(crate) fn val_to_string(val: Val) -> String {
    match val {
        Val::I32(x) => x.to_string(),
        Val::I64(x) => x.to_string(),
        Val::F32(x) => f32::from_bits(x).to_string(),
        Val::F64(x) => f64::from_bits(x).to_string(),
        Val::FuncRef(None) => "nullfuncref".to_string(),
        Val::FuncRef(Some(_)) => "funcref".to_string(),
        other => format!("{other:?}"),
    }
}
