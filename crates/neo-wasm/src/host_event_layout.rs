//! Host-event state shared with the WASM relation.
//!
//! This region owns the carried transcript, event schedule, and row-level
//! interface columns that the base VM constrains or consumes. Poseidon and
//! gather scratch advice remains private to `ccs::host_event_chain`.

use crate::column_registry::define_column_region;

define_column_region! {
    region: "host_event_interface",
    start: crate::layout::WASM_COLUMN_COUNT,
    width: pub HOST_EVENT_COLUMN_COUNT,
    specs: pub HOST_EVENT_COLUMN_SPECS,
    indices: pub,
    columns: [
        COL_HOST_CALLEE_FREF_BEFORE: U32 =>
            "callee function ref of the most recent host call before this row (event attribution carry)",
        COL_HOST_CALLEE_FREF_AFTER: U32 =>
            "callee function ref of the most recent host call after this row (event attribution carry)",
        COL_TURN_EXPORT_FREF_BEFORE: U32 => "export function ref owning the current host-event turn before this row",
        COL_TURN_EXPORT_FREF_AFTER: U32 => "export function ref owning the current host-event turn after this row",
        COL_COMM_CHAIN_BEFORE: [Field; 4] => "host-event commitment chain before this row",
        COL_COMM_CHAIN_AFTER: [Field; 4] => "host-event commitment chain after this row",
        // The block buffer, pending flag, round cursor, and running state
        // form the carried interface of the permutation rows.
        COL_EVBUF_BEFORE: [Field; 8] => "host-event block buffer before this row",
        COL_EVBUF_AFTER: [Field; 8] => "host-event block buffer after this row",
        COL_PERM_PENDING_BEFORE: Boolean => "a filled host-event block awaits its perm rows before this row",
        COL_PERM_PENDING_AFTER: Boolean => "a filled host-event block awaits its perm rows after this row",
        COL_PERM_ROUND_BEFORE: Field =>
            "perm-group row position before this row (0 when idle; bounded by the position one-hot)",
        COL_PERM_ROUND_AFTER: Field =>
            "perm-group row position after this row (0 when idle; bounded by the position one-hot)",
        COL_PERM_ROUND_BEFORE_IS_ZERO: Boolean =>
            "zero-test flag for the perm-group row position before this row",
        COL_PERM_ROUND_BEFORE_INV: Field => "inverse witness for the perm-group row position before this row",
        COL_PERM_STATE_BEFORE: [Field; 12] => "chain permutation state before this row",
        COL_PERM_STATE_AFTER: [Field; 12] => "chain permutation state after this row",
        COL_GATHER_ACTIVE: Boolean => "row staging one expanded event block into the absorb buffer",
        COL_HOST_CALL_ACTIVE: Boolean => "non-trapping host-import call row",
        // Host-event schedule carry and the per-row verifier-owned ROM interface.
        COL_HOST_EVENTS_REMAINING_BEFORE: Field => "host events still owed in the current phase, before this row",
        COL_HOST_EVENTS_REMAINING_AFTER: Field => "host events still owed in the current phase, after this row",
        COL_HOST_EVENTS_REMAINING_BEFORE_IS_ZERO: Boolean => "zero-test flag for the owed host events before this row",
        COL_HOST_EVENTS_REMAINING_BEFORE_INV: Field => "inverse witness for the owed host events before this row",
        COL_HOST_EVENT_INDEX_BEFORE: Field => "current host-event index within the template, before this row",
        COL_HOST_EVENT_INDEX_AFTER: Field => "current host-event index within the template, after this row",
        COL_HOST_EVENT_ARGS_BASE_BEFORE: Field =>
            "stack slot index of the current call's first argument, before this row",
        COL_HOST_EVENT_ARGS_BASE_AFTER: Field =>
            "stack slot index of the current call's first argument, after this row",
        COL_HOST_EVENT_SLOT_CURSOR_BEFORE: Field => "next block word a gather row stages (0..=7), before this row",
        COL_HOST_EVENT_SLOT_CURSOR_AFTER: Field => "next block word a gather row stages (0..=7), after this row",
        COL_HOST_EVENT_SLOT_KIND: Field =>
            "host-event ROM slot binding kind (0 const, 1 arg, 2 result, 3 input-local, 4 output, 5 memory-read, 6 memory-write)",
        COL_HOST_EVENT_SLOT_ARG: Field => "host-event ROM slot argument or input index",
        COL_HOST_EVENT_SLOT_VARIANT: Field =>
            "encoded host-event ROM slot variant: value kinds use 0 lo / 1 hi; memory kinds use bit 0 for local base, bit 1 for byte width, bit 2 for half width, and bit 3 for captured-output base",
        COL_HOST_EVENT_SLOT_IMMEDIATE0: Field =>
            "host-event ROM slot immediate 0; see WasmHostEventRomEntry",
        COL_HOST_EVENT_SLOT_IMMEDIATE1: Field =>
            "host-event ROM slot immediate 1; see WasmHostEventRomEntry",
        COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT: Field =>
            "host-event ROM initial-schedule count for the called import or entered export (biased +1; reread to offset export-exit indices)",
        COL_HOST_EVENT_EXIT_SCHEDULE_COUNT: Field =>
            "host-event ROM exit-schedule count for the halting export",
        COL_GATHER_LOCAL_WRITE: Boolean =>
            "gather row writing an input word into an entry-frame locals lane (input-local slot); gates the hi-lane write (zero on lo rows)",
        COL_GATHER_LOCAL_WRITE_LO: Boolean =>
            "input-local gather row targeting the lo lane: gather_local_write · (1 - slot_variant)",
        COL_HOST_EVENT_EXIT_LATCH: Boolean =>
            "clean export-halt row: loads the export's exit-event schedule",
        COL_TURN_BOUNDARY: Boolean =>
            "multi-turn re-entry row: re-arms the output, loads the next export's entry schedule, jumps to its entry pc",
        COL_PC_FREF_ACTIVE: Boolean =>
            "pc -> function-ref ROM gate: every row except gather rows (post-halt exit gathers sit past the last pc)",
        COL_CI_HOST_CALL: Boolean => "non-trapping call_indirect row targeting a host import",
    ]
}
