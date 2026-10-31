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
    COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_BEFORE, COL_OUTPUT_VALUE_LO_BEFORE, COL_PC_BEFORE,
    COL_PERM_PENDING_BEFORE, COL_PERM_ROUND_BEFORE, COL_PERM_STATE_BEFORE, COL_SP_BEFORE, COL_STACK_FRAME_BASE_BEFORE,
    COL_TRAPPED_BEFORE, COL_TURN_EXPORT_FREF_BEFORE,
};
use crate::relation_layout::build_wasm_relation_layout;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

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

/// The carried words of `state` in continuity-link order: the values that a
/// row starting from `state` holds at its `_BEFORE` columns.
pub fn continuity_words(state: WasmStepState) -> Vec<F> {
    build_wasm_relation_layout()
        .auxiliary
        .continuity
        .links()
        .map(|link| carried_state_field(state, link.next_step_column))
        .collect()
}

fn carried_state_field(state: WasmStepState, column: usize) -> F {
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
        (
            COL_PERM_STATE_BEFORE.as_slice(),
            state.event_absorb.perm_state.as_slice(),
        ),
    ] {
        if let Some(lane) = columns.iter().position(|&candidate| candidate == column) {
            return F::from_u64(values[lane]);
        }
    }

    match column {
        crate::layout::COL_HOST_FRAME_DEPTH_BEFORE => F::from_u64(state.host_events.host_frame_depth),
        crate::layout::COL_HOST_AWAITING_BEFORE => bool_field(state.host_events.awaiting_host),
        crate::layout::COL_HOST_SUSPEND_PENDING_BEFORE => bool_field(state.host_events.suspend_pending),
        COL_OBJECT_ACTIVE_BEFORE => bool_field(state.event_absorb.object_active),
        COL_PC_BEFORE => F::from_u64(state.pc),
        COL_SP_BEFORE => F::from_u64(state.sp),
        COL_STACK_FRAME_BASE_BEFORE => F::from_u64(state.stack_frame_base),
        COL_ACTIVATION_FINISHED_BEFORE => bool_field(state.activation_finished),
        COL_OUTPUT_ENABLED_BEFORE => bool_field(state.output.enabled),
        COL_OUTPUT_VALUE_LO_BEFORE => F::from_u64(u64::from(state.output.value_lo)),
        COL_OUTPUT_VALUE_HI_BEFORE => F::from_u64(u64::from(state.output.value_hi)),
        COL_CALL_STACK_DEPTH_BEFORE => F::from_u64(state.call_stack_depth),
        COL_MEMORY_PAGES_BEFORE => F::from_u64(u64::from(state.memory_pages.unwrap_or(0))),
        COL_MAX_MEMORY_PAGES_BEFORE => F::from_u64(u64::from(state.max_memory_pages.unwrap_or(0))),
        COL_LOCALS_FBP_BEFORE => F::from_u64(state.locals_fbp),
        COL_LOCAL_ZERO_ACTIVE_BEFORE => bool_field(state.local_zero.active),
        COL_LOCAL_ZERO_REMAINING_BEFORE => F::from_u64(u64::from(state.local_zero.remaining)),
        COL_FRAME_MOVE_ACTIVE_BEFORE => bool_field(state.frame_move.active),
        COL_FRAME_MOVE_REMAINING_BEFORE => F::from_u64(u64::from(state.frame_move.remaining)),
        COL_FRAME_MOVE_DISTANCE_BEFORE => F::from_u64(state.frame_move.distance),
        COL_HOST_CALLEE_FREF_BEFORE => F::from_u64(u64::from(state.host_callee_fref)),
        COL_TURN_EXPORT_FREF_BEFORE => F::from_u64(u64::from(state.host_events.turn_export_fref)),
        COL_HOST_EVENTS_REMAINING_BEFORE => F::from_u64(u64::from(state.host_events.events_remaining)),
        COL_HOST_EVENT_INDEX_BEFORE => F::from_u64(u64::from(state.host_events.event_index)),
        COL_HOST_EVENT_ARGS_BASE_BEFORE => F::from_u64(state.host_events.args_base),
        COL_HOST_EVENT_SLOT_CURSOR_BEFORE => F::from_u64(u64::from(state.host_events.slot_cursor)),
        COL_PERM_PENDING_BEFORE => bool_field(state.event_absorb.perm_pending),
        COL_PERM_ROUND_BEFORE => F::from_u64(u64::from(state.event_absorb.perm_round)),
        COL_TRAPPED_BEFORE => bool_field(state.trapped),
        other => panic!("unsupported initial semantic-state column {other}"),
    }
}

fn bool_field(value: bool) -> F {
    if value {
        F::ONE
    } else {
        F::ZERO
    }
}
