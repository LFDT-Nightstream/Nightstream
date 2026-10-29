//! Verifier-side initial VM state for a top-level export entry.

use crate::adapters::wasmtime::WasmProgramTables;
use crate::comm_chain::CommChainState;
use crate::ir::{
    WasmBuildError, WasmCountdownState, WasmEventAbsorbState, WasmFrameMoveState, WasmHostEventState, WasmOutputState,
    WasmStepState,
};

/// Top-level VM state before executing a parameterless exported wasm function
/// of an import-free program: [`host_event_top_level_initial_state`] specialized to
/// the canonical import-free bindings (empty boundary template for the
/// invoked export, zero commitment chain).
///
/// The entry PC is an explicit verifier input: callers should resolve it from
/// the export they intend to prove.
pub fn top_level_initial_state(tables: &WasmProgramTables, entry_pc: u64) -> WasmStepState {
    let export_fref = export_fref_for_entry_pc(tables, entry_pc);
    host_event_top_level_initial_state(
        tables,
        entry_pc,
        &crate::host_event_bindings::HostEventBindings::import_free(export_fref),
        export_fref,
        CommChainState::default(),
    )
    .expect("canonical import-free bindings require a parameterless export")
}

/// The function ref whose body starts at `entry_pc`; the verifier-side
/// counterpart of the normalizer reading the entered export off the trace.
pub(crate) fn export_fref_for_entry_pc(tables: &WasmProgramTables, entry_pc: u64) -> u32 {
    tables
        .function_entries
        .iter()
        .find(|&&(_, pc)| pc == entry_pc)
        .map(|&(fref, _)| u32::try_from(fref).expect("function refs fit in u32"))
        .unwrap_or_else(|| panic!("entry pc {entry_pc} is not a function entry"))
}

/// Host-event initial state: seeds the commitment chain and loads the invoked
/// export's entry schedule. Event values remain
/// bound by the final commitment rather than this per-program anchor.
pub fn host_event_top_level_initial_state(
    tables: &WasmProgramTables,
    entry_pc: u64,
    bindings: &crate::host_event_bindings::HostEventBindings,
    export_fref: u32,
    initial_comm_chain: CommChainState,
) -> Result<WasmStepState, WasmBuildError> {
    bindings.validate_against_program(tables)?;
    let entry_fref = tables
        .function_entries
        .iter()
        .find(|&&(_, pc)| pc == entry_pc)
        .map(|&(fref, _)| u32::try_from(fref).expect("function refs fit in u32"))
        .ok_or_else(|| WasmBuildError::Trace(format!("entry pc {entry_pc} is not a function entry")))?;
    if entry_fref != export_fref {
        return Err(WasmBuildError::Trace(format!(
            "export fref {export_fref} enters at a different pc than selected entry pc {entry_pc} (which belongs to fref {entry_fref})"
        )));
    }
    let template = bindings.exports.get(&export_fref).ok_or_else(|| {
        WasmBuildError::Trace(format!(
            "host-event bindings have no export template for selected export fref {export_fref}"
        ))
    })?;
    let (_, total_locals) = export_frame_shape(tables, export_fref)?;
    // The first export's declared locals already occupy zero-initialized stack
    // cells. Entry inputs fill its parameters; later entries must clear reused cells.
    let mut state = WasmStepState {
        pc: entry_pc,
        sp: u64::from(total_locals),
        stack_frame_base: u64::from(total_locals),
        output: WasmOutputState::ZERO,
        call_stack_depth: 0,
        memory_pages: tables.initial_memory_pages,
        max_memory_pages: tables.max_memory_pages,
        locals_fbp: 0,
        activation_finished: false,
        trapped: false,
        frame_move: WasmFrameMoveState::ZERO,
        local_zero: WasmCountdownState::ZERO,
        host_callee_fref: 0,
        comm_chain: initial_comm_chain.canonical_u64(),
        event_absorb: WasmEventAbsorbState::ZERO,
        host_events: WasmHostEventState::ZERO,
    };
    state.host_callee_fref = export_fref;
    state.host_events.turn_export_fref = export_fref;
    state.host_events.events_remaining = template.entry.len() as u32;
    Ok(state)
}

/// `(parameters, parameters plus declared locals)` of a guest function.
fn export_frame_shape(tables: &WasmProgramTables, fref: u32) -> Result<(u8, u32), WasmBuildError> {
    let lookup = |rows: &[(u64, u64)], what: &str| {
        rows.iter()
            .find_map(|&(function_ref, value)| (function_ref == u64::from(fref)).then_some(value))
            .ok_or_else(|| WasmBuildError::Trace(format!("missing {what} for export fref {fref}")))
    };
    let (params, _, _) =
        crate::ir::function_call_metadata_shape(lookup(&tables.function_call_metadata, "call metadata")?);
    let total_locals = u32::try_from(lookup(&tables.function_local_counts, "local count")?)
        .map_err(|_| WasmBuildError::Trace(format!("local count of export fref {fref} does not fit u32")))?;
    if total_locals < u32::from(params) {
        return Err(WasmBuildError::Trace(format!(
            "export fref {fref} has a total local count smaller than its parameter count"
        )));
    }
    Ok((params, total_locals))
}
