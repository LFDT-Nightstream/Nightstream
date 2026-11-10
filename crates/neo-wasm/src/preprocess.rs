//! Verifier-side initial VM state for a top-level export entry, and the
//! continuity words that a row reads at its `_BEFORE` columns.

use crate::adapters::wasmtime::WasmProgramTables;
use crate::comm_chain::CommChainState;
use crate::ir::{
    WasmBuildError, WasmCountdownState, WasmEventAbsorbState, WasmFrameMoveState, WasmHostEventState, WasmOutputState,
    WasmStepState,
};
use crate::layout::{
    COL_ACTIVATION_FINISHED_BEFORE, COL_CALL_STACK_DEPTH_BEFORE, COL_COMM_CHAIN_BEFORE, COL_EVBUF_BEFORE,
    COL_FRAME_MOVE_ACTIVE_BEFORE, COL_FRAME_MOVE_DISTANCE_BEFORE, COL_FRAME_MOVE_REMAINING_BEFORE,
    COL_HOST_CALLEE_FREF_BEFORE, COL_HOST_EVENTS_REMAINING_BEFORE, COL_HOST_EVENT_ARGS_BASE_BEFORE,
    COL_HOST_EVENT_INDEX_BEFORE, COL_HOST_EVENT_SLOT_CURSOR_BEFORE, COL_LOCALS_FBP_BEFORE,
    COL_LOCAL_ZERO_ACTIVE_BEFORE, COL_LOCAL_ZERO_REMAINING_BEFORE, COL_MAX_MEMORY_PAGES_BEFORE,
    COL_MEMORY_PAGES_BEFORE, COL_OBJECT_ACTIVE_BEFORE, COL_OUTER_CHAIN_BEFORE, COL_OUTER_PREFIX_BEFORE,
    COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_BEFORE, COL_OUTPUT_VALUE_LO_BEFORE, COL_PC_BEFORE, COL_SP_BEFORE,
    COL_STACK_FRAME_BASE_BEFORE, COL_TRAPPED_BEFORE, COL_TURN_EXPORT_FREF_BEFORE,
};
use crate::relation_layout::build_wasm_relation_layout;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// One above the wasm32 page limit. Valid page counts cannot use this word,
/// so it gives `None` a distinct carried representation without another
/// continuity column.
pub(crate) const NO_DEFAULT_MEMORY_PAGE_WORD: u64 = 65_537;

/// The carried words for the current and maximum default-memory page counts.
/// A valid state either has no default memory or has both counts with
/// `current <= maximum <= 65_536`.
pub(crate) fn memory_page_state_words(pages: Option<u32>, maximum: Option<u32>) -> Option<[u64; 2]> {
    match (pages, maximum) {
        (None, None) => Some([NO_DEFAULT_MEMORY_PAGE_WORD; 2]),
        (Some(pages), Some(maximum)) if pages <= maximum && u64::from(maximum) < NO_DEFAULT_MEMORY_PAGE_WORD => {
            Some([u64::from(pages), u64::from(maximum)])
        }
        _ => None,
    }
}

/// Top-level VM state before executing a parameterless exported wasm function
/// of an import-free program: [`host_event_top_level_initial_state`] specialized to
/// the canonical import-free bindings (empty boundary template for the
/// invoked export, zero commitment chain).
///
/// The entry PC is an explicit verifier input: callers should resolve it from
/// the export they intend to prove.
pub fn top_level_initial_state(tables: &WasmProgramTables, entry_pc: u64) -> Result<WasmStepState, WasmBuildError> {
    let export_fref = export_fref_for_entry_pc(tables, entry_pc)?;
    host_event_top_level_initial_state(
        tables,
        entry_pc,
        &crate::host_event_bindings::HostEventBindings::import_free(export_fref),
        export_fref,
        CommChainState::default(),
    )
}

/// The function ref whose body starts at `entry_pc`; the verifier-side
/// counterpart of the normalizer reading the entered export off the trace.
pub(crate) fn export_fref_for_entry_pc(tables: &WasmProgramTables, entry_pc: u64) -> Result<u32, WasmBuildError> {
    let fref = tables
        .function_entries
        .iter()
        .find(|&&(_, pc)| pc == entry_pc)
        .map(|&(fref, _)| fref)
        .ok_or_else(|| WasmBuildError::Trace(format!("entry pc {entry_pc} is not a function entry")))?;
    u32::try_from(fref).map_err(|_| WasmBuildError::Trace("function reference does not fit u32".into()))
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
    let entry_fref = export_fref_for_entry_pc(tables, entry_pc)?;
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

/// The carried words of `state` in continuity-link order: the values that a
/// row starting from `state` holds at its `_BEFORE` columns.
pub fn continuity_words(state: WasmStepState) -> Vec<F> {
    continuity_values(state)
        .into_iter()
        .map(F::from_u64)
        .collect()
}

/// The carried integers of `state` in continuity-link order, before their
/// reduction to field words. A value of `q` or more stays as it is, so it is
/// not the word of the smaller value that it reduces to.
pub(crate) fn continuity_values(state: WasmStepState) -> Vec<u64> {
    build_wasm_relation_layout()
        .auxiliary
        .continuity
        .links()
        .map(|link| carried_state_field(state, link.next_step_column))
        .collect()
}

fn carried_state_field(state: WasmStepState, column: usize) -> u64 {
    let [memory_pages, max_memory_pages] =
        memory_page_state_words(state.memory_pages, state.max_memory_pages).unwrap_or([u64::MAX; 2]);
    for (columns, values) in [
        (
            COL_OUTER_CHAIN_BEFORE.as_slice(),
            state.event_absorb.outer_chain.as_slice(),
        ),
        (
            COL_OUTER_PREFIX_BEFORE.as_slice(),
            state.event_absorb.outer_prefix.as_slice(),
        ),
        (COL_COMM_CHAIN_BEFORE.as_slice(), state.comm_chain.as_slice()),
        (COL_EVBUF_BEFORE.as_slice(), state.event_absorb.evbuf.as_slice()),
    ] {
        if let Some(lane) = columns.iter().position(|&candidate| candidate == column) {
            return values[lane];
        }
    }

    match column {
        crate::layout::COL_HOST_FRAME_DEPTH_BEFORE => state.host_events.host_frame_depth,
        crate::layout::COL_HOST_AWAITING_BEFORE => u64::from(state.host_events.awaiting_host),
        crate::layout::COL_HOST_SUSPEND_PENDING_BEFORE => u64::from(state.host_events.suspend_pending),
        COL_OBJECT_ACTIVE_BEFORE => u64::from(state.event_absorb.object_active),
        COL_PC_BEFORE => state.pc,
        COL_SP_BEFORE => state.sp,
        COL_STACK_FRAME_BASE_BEFORE => state.stack_frame_base,
        COL_ACTIVATION_FINISHED_BEFORE => u64::from(state.activation_finished),
        COL_OUTPUT_ENABLED_BEFORE => u64::from(state.output.enabled),
        COL_OUTPUT_VALUE_LO_BEFORE => u64::from(state.output.value_lo),
        COL_OUTPUT_VALUE_HI_BEFORE => u64::from(state.output.value_hi),
        COL_CALL_STACK_DEPTH_BEFORE => state.call_stack_depth,
        COL_MEMORY_PAGES_BEFORE => memory_pages,
        COL_MAX_MEMORY_PAGES_BEFORE => max_memory_pages,
        COL_LOCALS_FBP_BEFORE => state.locals_fbp,
        COL_LOCAL_ZERO_ACTIVE_BEFORE => u64::from(state.local_zero.active),
        COL_LOCAL_ZERO_REMAINING_BEFORE => u64::from(state.local_zero.remaining),
        COL_FRAME_MOVE_ACTIVE_BEFORE => u64::from(state.frame_move.active),
        COL_FRAME_MOVE_REMAINING_BEFORE => u64::from(state.frame_move.remaining),
        COL_FRAME_MOVE_DISTANCE_BEFORE => state.frame_move.distance,
        COL_HOST_CALLEE_FREF_BEFORE => u64::from(state.host_callee_fref),
        COL_TURN_EXPORT_FREF_BEFORE => u64::from(state.host_events.turn_export_fref),
        COL_HOST_EVENTS_REMAINING_BEFORE => u64::from(state.host_events.events_remaining),
        COL_HOST_EVENT_INDEX_BEFORE => u64::from(state.host_events.event_index),
        COL_HOST_EVENT_ARGS_BASE_BEFORE => state.host_events.args_base,
        COL_HOST_EVENT_SLOT_CURSOR_BEFORE => u64::from(state.host_events.slot_cursor),
        COL_TRAPPED_BEFORE => u64::from(state.trapped),
        other => panic!("unsupported initial semantic-state column {other}"),
    }
}
