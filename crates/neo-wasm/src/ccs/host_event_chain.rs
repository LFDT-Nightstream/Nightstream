//! In-circuit host-event chain gadget: binds the absorb buffer to host-event
//! gather rows and folds each filled block into the chain on the gather row
//! that stages its last word. The protocol constants and the reusable
//! commitment gadget live in `neo-application`, so the circuit and native
//! commitment share one protocol definition.
//!
//! Absorb: a non-advice gather row at slot seven (`ABSORB`) computes
//! `chain_after = commit_block(previous, [evbuf_before[0..7] | staged word])`
//! with one width-12 permutation, and clears the buffer. `previous` is the
//! chain before the row, or the restored outer chain when the same row copies
//! the final object-root lane. The commitment gadget is unconditional: every
//! row holds the permutation of its own inputs, and only `ABSORB` reads the
//! output. Opaque Enter saves and zeros the active chain; the final root-copy
//! slot restores the saved outer chain. Both contexts use the same hash path.
//!
//! Column ownership: the gadget's shared interface lives in
//! `host_event_layout` — the carried absorb state (buffer, pending flag,
//! object context) that continuity links,
//! the semantic digest, and `ccs/call.rs` refer to. The gadget-internal
//! witness columns (position one-hot, S-box powers, and gather decoding) are
//! allocated here in a private block right after
//! `NAMED_COLUMN_COUNT`. Their raw indices remain private; the witness-layout
//! registry exposes only their width metadata to generic range enforcement.

use super::super::layout::{
    COL_COMM_CHAIN_AFTER, COL_COMM_CHAIN_BEFORE, COL_EVBUF_AFTER, COL_EVBUF_BEFORE, COL_GATHER_ACTIVE,
    COL_HOST_CALLEE_FREF_AFTER, COL_HOST_EVENTS_REMAINING_BEFORE, COL_HOST_EVENTS_REMAINING_BEFORE_INV,
    COL_HOST_EVENTS_REMAINING_BEFORE_IS_ZERO, COL_OBJECT_ACTIVE_AFTER, COL_OBJECT_ACTIVE_BEFORE, COL_ONE,
    COL_OUTER_CHAIN_AFTER, COL_OUTER_CHAIN_BEFORE, COL_OUTER_PREFIX_AFTER, COL_OUTER_PREFIX_BEFORE, COL_STACK_READS,
    COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO,
    COL_STACK_WRITES, COL_TURN_BOUNDARY, COL_TURN_EXPORT_FREF_AFTER, COL_TURN_EXPORT_FREF_BEFORE,
};
use super::super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use super::call::host_call_gate_terms;
use super::host_event;
use crate::gadgets::push_gated_linear_zero;
use crate::ir::{WasmHostEventSlotKind, WasmVmStep};
use crate::layout::{COL_HOST_CLEAR_ACTIVATION_FINISHED, COL_HOST_RESUME, COL_HOST_SPLIT_CALL};
use neo_application::event_commitment::{commit_block, EventCommitment, EVENT_COMMITMENT_AUX_COLUMNS};
use neo_application::{define_column_region, ZeroTest};
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

pub(crate) const HOST_EVENTS_REMAINING_ZERO_TEST: ZeroTest = ZeroTest::column(
    COL_HOST_EVENTS_REMAINING_BEFORE,
    COL_HOST_EVENTS_REMAINING_BEFORE_INV,
    COL_HOST_EVENTS_REMAINING_BEFORE_IS_ZERO,
);

// Gadget-internal column block, allocated right after the named layout (the
// range-check bit columns follow it). Indices are private: the interface
// columns everything else uses are the named carried-state columns above.
const GKINDS: usize = WasmHostEventSlotKind::COUNT;
define_column_region! {
    region: "host_event_chain_aux",
    start: crate::witness_layout::HOST_EVENT_AUX_START,
    width: pub AUX_WIDTH,
    families: pub AUX_COLUMN_FAMILIES,
    indices: pub(self),
    columns: [
        ABSORB: Boolean => "gather row that fills an absorbing block and folds it into the chain",
        ABSORB_PREVIOUS: [Field; 4] => "chain the block folds into (the outer chain on an object exit)",
        ABSORB_OUTPUT: [Field; 4] => "event commitment of the previous chain and the filled block",
        ABSORB_PERMUTATION: [Field; EVENT_COMMITMENT_AUX_COLUMNS] => "width-12 permutation witnesses of the absorb",
        GATHER_WORD_POSITION: [Boolean; 8] => "gather block-word one-hot flags",
        GATHER_KIND: [Boolean; GKINDS] => "gather slot-kind one-hot flags",
        GARG_VAL: Field => "limb-selected stack argument value",
        GOUT_VAL: Field => "limb-selected export output value",
        GSLOT_VALUE: Field => "word staged by the current gather row",
        GK2_HI: Boolean => "result-slot high-limb write flag",
        GHC_PARAMS: Field => "host-event host-call and parameter-count product",
        G_ADVICE: Boolean => "advice-event slot flag",
        GMEM_LOCAL: Boolean => "memory pointer comes from an export local",
        GMEM_OUTPUT: Boolean => "memory pointer comes from the captured export output",
        GMEM_BYTE: Boolean => "byte-width host-event memory slot",
        GMEM_HALF: Boolean => "half-width host-event memory slot",
        TURN_SCHEDULE_COUNT_INV: Field => "turn-boundary nonempty-template inverse witness",
        OBJECT_ROOT_LANE: [Boolean; 4] => "opaque root copy-back lane selectors",
        OBJECT_SAVED_LANE: [Boolean; 4] => "suspended outer prefix lane selectors",
    ]
}

const GK_ARG: usize = GATHER_KIND[WasmHostEventSlotKind::Arg.index()];
const GK_RESULT: usize = GATHER_KIND[WasmHostEventSlotKind::Result.index()];
const GK_INPUT_LOCAL: usize = GATHER_KIND[WasmHostEventSlotKind::InputLocal.index()];
const GK_OUTPUT: usize = GATHER_KIND[WasmHostEventSlotKind::Output.index()];
const GK_MEMORY_READ: usize = GATHER_KIND[WasmHostEventSlotKind::MemoryRead.index()];
const GK_MEMORY_WRITE: usize = GATHER_KIND[WasmHostEventSlotKind::MemoryWrite.index()];
const GK_ENTER: usize = GATHER_KIND[WasmHostEventSlotKind::EnterOpaque.index()];
const GK_ROOT: usize = GATHER_KIND[WasmHostEventSlotKind::OpaqueRoot.index()];
const GK_SAVED: usize = GATHER_KIND[WasmHostEventSlotKind::OpaqueSaved.index()];

/// The gather column whose flag pins a non-popping stack read (arg slots).
/// The `sp' = sp - reads + writes` identity in `ccs.rs` exempts these reads
/// from popping. (Result slots WRITE: the Lo slot is a genuine push moving
/// sp through the counted port; the Hi slot uses only the un-counted
/// hi-word port and leaves sp alone.)
pub(super) const fn gather_arg_read_kind_col() -> usize {
    GK_ARG
}

pub(crate) const fn gather_memory_read_kind_col() -> usize {
    GK_MEMORY_READ
}

pub(crate) const fn gather_memory_write_kind_col() -> usize {
    GK_MEMORY_WRITE
}

pub(crate) const fn gather_memory_local_base_col() -> usize {
    GMEM_LOCAL
}

pub(crate) const fn gather_memory_output_base_col() -> usize {
    GMEM_OUTPUT
}

pub(crate) const fn gather_memory_byte_width_col() -> usize {
    GMEM_BYTE
}

pub(crate) const fn gather_memory_half_width_col() -> usize {
    GMEM_HALF
}

/// Product column carrying `host_call_active · call_param_count`: the sp
/// identity in `ccs.rs` pops all host-call args on the call row itself.
pub(super) const fn host_call_params_col() -> usize {
    GHC_PARAMS
}

/// The block absorb: the first seven words are already in the buffer, and the
/// row stages the eighth.
fn absorb_commitment() -> EventCommitment {
    EventCommitment {
        previous: ABSORB_PREVIOUS,
        block: core::array::from_fn(|word| if word < 7 { COL_EVBUF_BEFORE[word] } else { GSLOT_VALUE }),
        output: ABSORB_OUTPUT,
        auxiliary_start: ABSORB_PERMUTATION[0],
    }
}

pub(super) fn push_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    push_interface_constraints(b);
    push_host_event_gather_constraints(b);
    push_opaque_constraints(b);
    push_absorb_constraints(b);
    push_buffer_write_constraints(b);
    push_chain_update_constraints(b);
}

/// Shared host-event row shape and carried-state transitions.
fn push_interface_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.with_tag(host_event("host event interface"), |b| {
        b.push_linear_zero(
            [(super::super::layout::COL_HOST_CALL_ACTIVE, F::ONE)]
                .into_iter()
                .chain(host_call_gate_terms().map(|(column, coefficient)| (column, -coefficient))),
        );
        // Disable the pc-to-function lookup on gather, turn boundary,
        // padding, and resume rows, which do not execute a program
        // instruction.
        b.push_linear_zero([
            (super::super::layout::COL_PC_FREF_ACTIVE, F::ONE),
            (COL_ONE, -F::ONE),
            (COL_GATHER_ACTIVE, F::ONE),
            (COL_TURN_BOUNDARY, F::ONE),
            (super::super::layout::COL_PADDING_ACTIVE, F::ONE),
            (COL_HOST_RESUME, F::ONE),
        ]);
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (COL_TURN_BOUNDARY, -F::ONE),
                (COL_HOST_RESUME, -F::ONE),
            ],
            [
                (COL_TURN_EXPORT_FREF_AFTER, F::ONE),
                (COL_TURN_EXPORT_FREF_BEFORE, -F::ONE),
            ],
            [],
        );
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE)],
            [
                (COL_TURN_EXPORT_FREF_AFTER, F::ONE),
                (COL_HOST_CALLEE_FREF_AFTER, -F::ONE),
            ],
            [],
        );
        // Gather rows read the stack on arg slots and on memory slots whose
        // pointer base is an import argument; result slots
        // WRITE it — the lo slot through the counted port pair (the push),
        // the hi slot through the hi-word port alone (no sp effect). Both
        // preserve the rest of the VM state.
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE)],
            [
                (COL_STACK_READS, F::ONE),
                (GK_ARG, -F::ONE),
                (GK_MEMORY_READ, -F::ONE),
                (GK_MEMORY_WRITE, -F::ONE),
                (GMEM_LOCAL, F::ONE),
                (GMEM_OUTPUT, F::ONE),
            ],
            [],
        );
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE)],
            [(COL_STACK_WRITES, F::ONE), (GK_RESULT, -F::ONE), (GK2_HI, F::ONE)],
            [],
        );
    });
}

/// Host-event gather binding: each gather row stages exactly one block word,
/// whose value is pinned by the host-event ROM entry at
/// `(event_index, slot_cursor)` — a flat value, runtime input, or aligned
/// linear-memory access — and the per-call
/// event schedule is forced by ROM-loaded countdowns. This closes the
/// stage-B gap: with these rows, the commitment chain commits exactly the event
/// sequence obtained by applying the committed tables to the values at the
/// call site.
fn push_host_event_gather_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    use super::super::layout::{
        COL_ACTIVATION_FINISHED_AFTER, COL_ACTIVATION_FINISHED_BEFORE, COL_CALL_PARAM_COUNT as PARAM_COUNT,
        COL_GATHER_LOCAL_WRITE, COL_GATHER_LOCAL_WRITE_LO, COL_HOST_CALL_ACTIVE as HOST_CALL_ACTIVE,
        COL_HOST_EVENTS_REMAINING_AFTER as EVENTS_REMAINING_AFTER,
        COL_HOST_EVENTS_REMAINING_BEFORE as EVENTS_REMAINING_BEFORE,
        COL_HOST_EVENTS_REMAINING_BEFORE_IS_ZERO as EVENTS_REMAINING_IS_ZERO,
        COL_HOST_EVENT_ARGS_BASE_AFTER as ARGS_BASE_AFTER, COL_HOST_EVENT_ARGS_BASE_BEFORE as ARGS_BASE_BEFORE,
        COL_HOST_EVENT_BASE as EVENT_BASE, COL_HOST_EVENT_EXIT_LATCH,
        COL_HOST_EVENT_EXIT_SCHEDULE_COUNT as EXIT_SCHEDULE_COUNT, COL_HOST_EVENT_INDEX_AFTER as EVENT_INDEX_AFTER,
        COL_HOST_EVENT_INDEX_BEFORE as EVENT_INDEX_BEFORE,
        COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT as INITIAL_SCHEDULE_COUNT, COL_HOST_EVENT_SLOT_ARG as SLOT_ARG,
        COL_HOST_EVENT_SLOT_CURSOR_AFTER as SLOT_CURSOR_AFTER, COL_HOST_EVENT_SLOT_CURSOR_BEFORE as SLOT_CURSOR_BEFORE,
        COL_HOST_EVENT_SLOT_IMMEDIATE0 as SLOT_IMMEDIATE0, COL_HOST_EVENT_SLOT_IMMEDIATE1 as SLOT_IMMEDIATE1,
        COL_HOST_EVENT_SLOT_KIND as SLOT_KIND, COL_HOST_EVENT_SLOT_VARIANT as SLOT_VARIANT, COL_IS_PROGRAM_ROW,
        COL_LINEAR_MEM_ACCESS_BYTE0, COL_LINEAR_MEM_ACCESS_BYTE1, COL_LINEAR_MEM_BYTE_OFFSET, COL_LINEAR_MEM_LANE_ADDR,
        COL_LINEAR_MEM_LANE_VALUE, COL_LINEAR_MEM_OFFSET_IS_1, COL_LINEAR_MEM_OFFSET_IS_3, COL_LOCAL_INDEX,
        COL_LOCAL_VALUE, COL_LOCAL_VALUE_HI, COL_MEM_OOB, COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_BEFORE,
        COL_OUTPUT_VALUE_LO_BEFORE, COL_SP_BEFORE, COL_STACK_READ_ADDR_LO, COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE,
        COL_TURN_BOUNDARY, COL_WIDE_VALUES_ENABLED,
    };
    let ci_sel = super::super::layout::selector_col(crate::isa::WasmOpcode::CallIndirect).expect("ci selector");

    b.with_tag(host_event("host-event gather binding"), |b| {
        // Atomic host calls pop args here; split calls retain them until resume.
        // The sp identity consumes this product with ROM-bound arity.
        b.push_row(
            [(HOST_CALL_ACTIVE, F::ONE), (COL_HOST_SPLIT_CALL, -F::ONE)],
            [(PARAM_COUNT, F::ONE)],
            [(GHC_PARAMS, F::ONE)],
        );

        // Event schedule countdown: loaded from the event-count ROMs on the
        // host-call row (the whole call, args and result, is one atomic
        // event sequence), decremented at slot seven or an early Enter,
        // preserved elsewhere; program rows require it to be spent, and
        // gather rows require it to be live.
        HOST_EVENTS_REMAINING_ZERO_TEST.push_constraints(b);
        b.push_row([(COL_GATHER_ACTIVE, F::ONE)], [(EVENTS_REMAINING_IS_ZERO, F::ONE)], []);
        b.push_row([(COL_IS_PROGRAM_ROW, F::ONE)], [(EVENTS_REMAINING_BEFORE, F::ONE)], []);
        // Pre-count cells store count + 1 (presence bias): an undeclared
        // import's zero-filled cell loads the poisoned events_remaining = -1 = p-1.
        // See the count-family relation-layout comment for the full
        // ROM-address non-termination argument.
        b.push_row(
            [(HOST_CALL_ACTIVE, F::ONE)],
            [
                (EVENTS_REMAINING_AFTER, F::ONE),
                (INITIAL_SCHEDULE_COUNT, -F::ONE),
                (COL_ONE, F::ONE),
            ],
            [],
        );
        b.push_row(
            [(GATHER_WORD_POSITION[7], F::ONE), (GK_ENTER, F::ONE)],
            [
                (EVENTS_REMAINING_AFTER, F::ONE),
                (EVENTS_REMAINING_BEFORE, -F::ONE),
                (COL_ONE, F::ONE),
            ],
            [],
        );
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (HOST_CALL_ACTIVE, -F::ONE),
                (GATHER_WORD_POSITION[7], -F::ONE),
                (GK_ENTER, -F::ONE),
                (COL_HOST_EVENT_EXIT_LATCH, -F::ONE),
                (COL_TURN_BOUNDARY, -F::ONE),
                (COL_HOST_RESUME, -F::ONE),
            ],
            [(EVENTS_REMAINING_AFTER, F::ONE), (EVENTS_REMAINING_BEFORE, -F::ONE)],
            [],
        );
        // Turn boundary: the previous turn's schedules must be spent, and
        // the next export's entry schedule loads from the count ROM (keyed
        // by the repointed attribution, like the exit latch). The presence
        // bias (+1) binds the target to a DECLARED export template: internal
        // functions and imports read the export family's zero-filled 0 and
        // load the poisoned events_remaining = p-1 described above.
        b.push_row([(COL_TURN_BOUNDARY, F::ONE)], [(EVENTS_REMAINING_BEFORE, F::ONE)], []);
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE)],
            [
                (EVENTS_REMAINING_AFTER, F::ONE),
                (INITIAL_SCHEDULE_COUNT, -F::ONE),
                (COL_ONE, F::ONE),
            ],
            [],
        );
        // if col_turn_boundary then entry_len + exit_len != 0
        //
        // TODO: there may be a cleaner solution to this problem?
        //
        // basically this is here to force re-entrancy to leave a trace of how
        // many times the export was re-entered
        //
        // otherwise there would be no way of differentiating a proof of f^n
        // from a proof of f^m (and the final state may not reflect it either)
        //
        // Template validation requires a nonempty entry/exit pair to include
        // an absorbing event. That event may be in the exit schedule.
        //
        // note that if there is no-reentrancy then the template doesn't matter,
        // so the case of proving a single function is fine
        b.push_row(
            // Entry: 0 means no template, 1 is empty (entry len is x - 1).
            // The exit count is raw, so the sum below is entry_len + exit_len.
            [
                (INITIAL_SCHEDULE_COUNT, F::ONE),
                (COL_ONE, -F::ONE),
                (EXIT_SCHEDULE_COUNT, F::ONE),
            ],
            [(TURN_SCHEDULE_COUNT_INV, F::ONE)],
            // If this is 1, the total count must have an inverse, so it is nonzero.
            [(COL_TURN_BOUNDARY, F::ONE)],
        );

        // Event index: the ROM key component walking the template, from the
        // template's first event row.
        b.push_row(
            [(HOST_CALL_ACTIVE, F::ONE)],
            [(EVENT_INDEX_AFTER, F::ONE), (EVENT_BASE, -F::ONE)],
            [],
        );
        b.push_row(
            [(GATHER_WORD_POSITION[7], F::ONE), (GK_ENTER, F::ONE)],
            [
                (EVENT_INDEX_AFTER, F::ONE),
                (EVENT_INDEX_BEFORE, -F::ONE),
                (COL_ONE, -F::ONE),
            ],
            [],
        );
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (HOST_CALL_ACTIVE, -F::ONE),
                (GATHER_WORD_POSITION[7], -F::ONE),
                (GK_ENTER, -F::ONE),
                (COL_HOST_EVENT_EXIT_LATCH, -F::ONE),
                (COL_TURN_BOUNDARY, -F::ONE),
                (COL_HOST_RESUME, -F::ONE),
            ],
            [(EVENT_INDEX_AFTER, F::ONE), (EVENT_INDEX_BEFORE, -F::ONE)],
            [],
        );
        // Turn boundary: entry events of the next turn are numbered from the
        // first event row of the entered export's template.
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE)],
            [(EVENT_INDEX_AFTER, F::ONE), (EVENT_BASE, -F::ONE)],
            [],
        );

        // Argument-region base: latched on the host-call row from bound
        // quantities (sp, the indirect-index pop, the ROM-bound arity).
        b.push_row(
            [(HOST_CALL_ACTIVE, F::ONE)],
            [
                (ARGS_BASE_AFTER, F::ONE),
                (COL_SP_BEFORE, -F::ONE),
                (ci_sel, F::ONE),
                (PARAM_COUNT, F::ONE),
            ],
            [],
        );
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (HOST_CALL_ACTIVE, -F::ONE),
                (COL_HOST_RESUME, -F::ONE),
            ],
            [(ARGS_BASE_AFTER, F::ONE), (ARGS_BASE_BEFORE, -F::ONE)],
            [],
        );

        // Slot cursor + block-word one-hot lockstep.
        b.push_linear_zero(
            (0..8)
                .map(|k| (GATHER_WORD_POSITION[k], F::ONE))
                .chain([(COL_GATHER_ACTIVE, -F::ONE)]),
        );
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE)],
            (0..8)
                .map(|k| (GATHER_WORD_POSITION[k], F::from_u64(k as u64)))
                .chain([(SLOT_CURSOR_BEFORE, -F::ONE)]),
            [],
        );
        // Normally advance one slot (wrapping at eight). Enter instead
        // consumes the prefix's final instruction and resets directly to zero.
        b.push_row(
            [(GK_ENTER, F::ONE)],
            [(SLOT_CURSOR_BEFORE, F::ONE), (COL_ONE, F::ONE)],
            [
                (SLOT_CURSOR_BEFORE, F::ONE),
                (COL_GATHER_ACTIVE, F::ONE),
                (GATHER_WORD_POSITION[7], -F::from_u64(8)),
                (SLOT_CURSOR_AFTER, -F::ONE),
            ],
        );
        b.push_row([(COL_IS_PROGRAM_ROW, F::ONE)], [(SLOT_CURSOR_BEFORE, F::ONE)], []);

        // Advice uses the next code range above the raw slot kinds.
        b.push_linear_zero(
            (0..GKINDS)
                .map(|j| (GATHER_KIND[j], F::ONE))
                .chain([(COL_GATHER_ACTIVE, -F::ONE)]),
        );
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE)],
            (0..GKINDS)
                .map(|j| (GATHER_KIND[j], F::from_u64(j as u64)))
                .chain([
                    (G_ADVICE, F::from_u64(WasmHostEventSlotKind::COUNT as u64)),
                    (SLOT_KIND, -F::ONE),
                ]),
            [],
        );

        // The staged word lands in the buffer slot the cursor points at,
        // unless the row absorbs the block it fills.
        for k in 0..8 {
            let gate = if k == 7 {
                vec![(GATHER_WORD_POSITION[k], F::ONE), (ABSORB, -F::ONE)]
            } else {
                vec![(GATHER_WORD_POSITION[k], F::ONE)]
            };
            b.push_row(gate, [(COL_EVBUF_AFTER[k], F::ONE), (GSLOT_VALUE, -F::ONE)], []);
        }

        // Const slots: the word is the ROM constant (u32 limb pair).
        b.push_row(
            [(GATHER_KIND[0], F::ONE)],
            [
                (GSLOT_VALUE, F::ONE),
                (SLOT_IMMEDIATE0, -F::ONE),
                (SLOT_IMMEDIATE1, -F::from_u64(1 << 32)),
            ],
            [],
        );

        // Result-slot limb split: each lane is written by the slot that
        // absorbs it (the stack twin of the kind-4 locals pattern). Boolean
        // by the ROM's 0/1 limb content.
        b.push_row([(GK_RESULT, F::ONE)], [(SLOT_VARIANT, F::ONE)], [(GK2_HI, F::ONE)]);
        // A host result can enter the VM only when the verifier-owned static
        // function type has the complete i32/i64 scalar encoding. This bit
        // is zero for references, vectors, floats, resultless functions, and
        // multi-result shapes.
        b.push_row(
            [(GK_RESULT, F::ONE)],
            [(SLOT_IMMEDIATE0, F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            [(GK2_HI, F::ONE)],
            [(COL_WIDE_VALUES_ENABLED, F::ONE), (SLOT_IMMEDIATE1, -F::ONE)],
            [],
        );

        // Arg slots: an addressed stack read at the table offset from the
        // argument base, limb-selected into the word.
        b.push_row(
            [(GK_ARG, F::ONE)],
            [
                (COL_STACK_READ_ADDR_LO[0], F::ONE),
                (ARGS_BASE_BEFORE, -F::from_u64(2)),
                (SLOT_ARG, -F::from_u64(2)),
            ],
            [],
        );
        b.push_row(
            [(SLOT_VARIANT, F::ONE)],
            [
                (COL_STACK_READ_VALUE_HI[0], F::ONE),
                (COL_STACK_READ_VALUE_LO[0], -F::ONE),
            ],
            [(GARG_VAL, F::ONE), (COL_STACK_READ_VALUE_LO[0], -F::ONE)],
        );
        b.push_row([(GK_ARG, F::ONE)], [(GSLOT_VALUE, F::ONE), (GARG_VAL, -F::ONE)], []);

        // Result Lo slots (kind 2 with the hi flag low): the gather row
        // WRITES the staged word onto the operand stack — the host result's
        // push. The write ports
        // make the sp identity move by +1; the address is the post-pop
        // stack top (= the argument base, so arg-0 slots must be gathered
        // earlier — validated template-side). The write is a narrow TOTAL
        // write: the hi lane is pinned to zero, never advice — an i64
        // result's hi limb arrives through its own Hi slot write below.
        b.push_row(
            [(GK_RESULT, F::ONE), (GK2_HI, -F::ONE)],
            [
                (super::super::layout::COL_STACK_WRITE0_ADDR_LO, F::ONE),
                (COL_SP_BEFORE, -F::from_u64(2)),
            ],
            [],
        );
        b.push_row(
            [(GK_RESULT, F::ONE), (GK2_HI, -F::ONE)],
            [(GSLOT_VALUE, F::ONE), (COL_STACK_WRITE0_VALUE_LO, -F::ONE)],
            [],
        );
        b.push_row(
            [(GK_RESULT, F::ONE), (GK2_HI, -F::ONE)],
            [(COL_STACK_WRITE0_VALUE_HI, F::ONE)],
            [],
        );

        // Result Hi slots: write ONLY the pushed cell's hi word (the
        // hi-word port fires without the counted lo port, so sp is
        // untouched). The unconditional `addr_hi = addr_lo + 1` pair rule
        // routes the port to 2·args_base + 1.
        b.push_row(
            [(GK2_HI, F::ONE)],
            [
                (super::super::layout::COL_STACK_WRITE0_ADDR_LO, F::ONE),
                (ARGS_BASE_BEFORE, -F::from_u64(2)),
            ],
            [],
        );
        b.push_row(
            [(GK2_HI, F::ONE)],
            [(GSLOT_VALUE, F::ONE), (COL_STACK_WRITE0_VALUE_HI, -F::ONE)],
            [],
        );
        // The hi-word port gate: raw write0 activity, plus these rows.
        b.push_linear_zero([
            (super::super::layout::COL_STACK_WRITE0_HI_ACTIVE, F::ONE),
            (super::super::layout::COL_STACK_WRITE0_ACTIVE, -F::ONE),
            (GK2_HI, -F::ONE),
        ]);

        // Input-local slots: the staged input word is written
        // into one 32-bit lane of the entry frame's locals at the
        // table-pinned index (ROM limb select: 0 lo, 1 hi). Routing the word
        // through the U32-checked locals value columns range-proves it. Lo
        // rows also write the hi lane to zero, so a lone Lo write is total;
        // a Hi row (validated to follow its local's Lo row) overwrites the
        // hi lane with the input word. Advice words remain existential;
        // absorbing slots bind them through the final transcript.
        b.push_linear_zero([(COL_GATHER_LOCAL_WRITE, F::ONE), (GK_INPUT_LOCAL, -F::ONE)]);
        b.push_row(
            [(GK_INPUT_LOCAL, F::ONE)],
            [(COL_WIDE_VALUES_ENABLED, F::ONE), (SLOT_IMMEDIATE1, -F::ONE)],
            [],
        );
        b.push_row(
            [(GK_INPUT_LOCAL, F::ONE)],
            [(SLOT_IMMEDIATE0, F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            [(GK_INPUT_LOCAL, F::ONE)],
            [(COL_ONE, F::ONE), (SLOT_VARIANT, -F::ONE)],
            [(COL_GATHER_LOCAL_WRITE_LO, F::ONE)],
        );
        b.push_row(
            [(GK_INPUT_LOCAL, F::ONE)],
            [(COL_LOCAL_INDEX, F::ONE), (SLOT_ARG, -F::ONE)],
            [],
        );
        // Lo rows: locals lane = word, hi lane = 0.
        b.push_row(
            [(COL_GATHER_LOCAL_WRITE_LO, F::ONE)],
            [(GSLOT_VALUE, F::ONE), (COL_LOCAL_VALUE, -F::ONE)],
            [],
        );
        b.push_row(
            [(COL_GATHER_LOCAL_WRITE_LO, F::ONE)],
            [(COL_LOCAL_VALUE_HI, F::ONE)],
            [],
        );
        // Hi rows (gather_local_write - gather_local_write_lo): hi lane = word.
        b.push_row(
            [(COL_GATHER_LOCAL_WRITE, F::ONE), (COL_GATHER_LOCAL_WRITE_LO, -F::ONE)],
            [(COL_LOCAL_VALUE_HI, F::ONE), (GSLOT_VALUE, -F::ONE)],
            [],
        );

        // Memory-slot ROM variants encode one base (0 argument, 1 export
        // local, 8 captured output) plus one width (0 word, 2 byte, 4 half).
        b.push_row(
            [(GK_MEMORY_READ, F::ONE), (GK_MEMORY_WRITE, F::ONE)],
            [(SLOT_VARIANT, F::ONE)],
            [
                (GMEM_LOCAL, F::ONE),
                (
                    GMEM_BYTE,
                    F::from_u64(u64::from(
                        crate::ir::WasmHostEventRomVariant::MEMORY_BYTE_ENCODING_FACTOR,
                    )),
                ),
                (
                    GMEM_HALF,
                    F::from_u64(u64::from(
                        crate::ir::WasmHostEventRomVariant::MEMORY_HALF_ENCODING_FACTOR,
                    )),
                ),
                (
                    GMEM_OUTPUT,
                    F::from_u64(u64::from(
                        crate::ir::WasmHostEventRomVariant::MEMORY_OUTPUT_ENCODING_FACTOR,
                    )),
                ),
            ],
        );
        // Word slots have no intra-word byte offset. Subword slots bind it
        // through the shared width/offset selector families below.
        b.push_row(
            [
                (GK_MEMORY_READ, F::ONE),
                (GK_MEMORY_WRITE, F::ONE),
                (GMEM_BYTE, -F::ONE),
                (GMEM_HALF, -F::ONE),
            ],
            [(COL_LINEAR_MEM_BYTE_OFFSET, F::ONE)],
            [],
        );
        // Canonical-ABI half words are naturally aligned, so their offset
        // within a 32-bit memory word is 0 or 2.
        b.push_row(
            [(GMEM_HALF, F::ONE)],
            [
                (COL_LINEAR_MEM_OFFSET_IS_1, F::ONE),
                (COL_LINEAR_MEM_OFFSET_IS_3, F::ONE),
            ],
            [],
        );
        // Argument-base memory rows read the pointer from the call's
        // non-popping argument area.
        b.push_row(
            [
                (GK_MEMORY_READ, F::ONE),
                (GK_MEMORY_WRITE, F::ONE),
                (GMEM_LOCAL, -F::ONE),
                (GMEM_OUTPUT, -F::ONE),
            ],
            [
                (COL_STACK_READ_ADDR_LO[0], F::ONE),
                (ARGS_BASE_BEFORE, -F::from_u64(2)),
                (SLOT_ARG, -F::from_u64(2)),
            ],
            [],
        );
        b.push_row(
            [
                (GK_MEMORY_READ, F::ONE),
                (GK_MEMORY_WRITE, F::ONE),
                (GMEM_LOCAL, -F::ONE),
                (GMEM_OUTPUT, -F::ONE),
            ],
            [
                (COL_LINEAR_MEM_LANE_ADDR[0], F::from_u64(4)),
                (COL_LINEAR_MEM_BYTE_OFFSET, F::ONE),
                (COL_STACK_READ_VALUE_LO[0], -F::ONE),
                (SLOT_IMMEDIATE0, -F::ONE),
            ],
            [],
        );
        // Argument pointers are wasm32 values. Keep the authenticated high
        // stack lane from being silently truncated by the address identity.
        b.push_row(
            [
                (GK_MEMORY_READ, F::ONE),
                (GK_MEMORY_WRITE, F::ONE),
                (GMEM_LOCAL, -F::ONE),
                (GMEM_OUTPUT, -F::ONE),
            ],
            [(COL_STACK_READ_VALUE_HI[0], F::ONE)],
            [],
        );
        // Local-base memory rows bind the pointer through the local's stack cell.
        b.push_row(
            [(GMEM_LOCAL, F::ONE)],
            [(COL_LOCAL_INDEX, F::ONE), (SLOT_ARG, -F::ONE)],
            [],
        );
        // Exit memory reads use the captured single-result value as their
        // wasm32 pointer, independent of the terminal call frame.
        b.push_row(
            [(GMEM_OUTPUT, F::ONE)],
            [(COL_OUTPUT_ENABLED_BEFORE, F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            [(GMEM_OUTPUT, F::ONE)],
            [
                (COL_LINEAR_MEM_LANE_ADDR[0], F::from_u64(4)),
                (COL_LINEAR_MEM_BYTE_OFFSET, F::ONE),
                // this is a ptr here
                //
                // so read_memory_addr = output_ptr + read_memory_offset
                (COL_OUTPUT_VALUE_LO_BEFORE, -F::ONE),
                (SLOT_IMMEDIATE0, -F::ONE),
            ],
            [],
        );
        b.push_row([(GMEM_OUTPUT, F::ONE)], [(COL_OUTPUT_VALUE_HI_BEFORE, F::ONE)], []);
        b.push_row(
            [(GMEM_LOCAL, F::ONE)],
            [
                (COL_LINEAR_MEM_LANE_ADDR[0], F::from_u64(4)),
                (COL_LINEAR_MEM_BYTE_OFFSET, F::ONE),
                (COL_LOCAL_VALUE, -F::ONE),
                (SLOT_IMMEDIATE0, -F::ONE),
            ],
            [],
        );
        // A local-base host memory access is a wasm32 pointer. Authenticate
        // and reject a nonzero high local lane instead of silently using
        // only its low word as the address.
        push_gated_linear_zero(b, GMEM_LOCAL, [(COL_LOCAL_VALUE_HI, F::ONE)]);
        b.push_row(
            [
                (GK_MEMORY_READ, F::ONE),
                (GK_MEMORY_WRITE, F::ONE),
                (GMEM_BYTE, -F::ONE),
                (GMEM_HALF, -F::ONE),
            ],
            [(GSLOT_VALUE, F::ONE), (COL_LINEAR_MEM_LANE_VALUE[0], -F::ONE)],
            [],
        );
        b.push_row(
            [(GMEM_BYTE, F::ONE)],
            [(GSLOT_VALUE, F::ONE), (COL_LINEAR_MEM_ACCESS_BYTE0, -F::ONE)],
            [],
        );
        b.push_row(
            [(GMEM_HALF, F::ONE)],
            [
                (GSLOT_VALUE, F::ONE),
                (COL_LINEAR_MEM_ACCESS_BYTE0, -F::ONE),
                (COL_LINEAR_MEM_ACCESS_BYTE1, -F::from_u64(1 << 8)),
            ],
            [],
        );
        b.push_row(
            [(GK_MEMORY_READ, F::ONE), (GK_MEMORY_WRITE, F::ONE)],
            [(COL_MEM_OOB, F::ONE)],
            [],
        );

        // Export output slots (kind 5): the carried simple-output value,
        // limb-selected (bound by the output-capture machinery).
        b.push_row(
            [(SLOT_VARIANT, F::ONE)],
            [
                (COL_OUTPUT_VALUE_HI_BEFORE, F::ONE),
                (COL_OUTPUT_VALUE_LO_BEFORE, -F::ONE),
            ],
            [(GOUT_VAL, F::ONE), (COL_OUTPUT_VALUE_LO_BEFORE, -F::ONE)],
        );
        b.push_row([(GK_OUTPUT, F::ONE)], [(GSLOT_VALUE, F::ONE), (GOUT_VAL, -F::ONE)], []);
        b.push_row(
            [(GK_OUTPUT, F::ONE)],
            [(COL_ONE, F::ONE), (COL_OUTPUT_ENABLED_BEFORE, -F::ONE)],
            [],
        );

        // A clean activation return loads the owning export's exit schedule and
        // repoints gather attribution from the last import to that export.
        // This is independent of result capture, so constant-only exit
        // templates also cover resultless exports.
        b.push_linear_zero([
            (COL_ACTIVATION_FINISHED_AFTER, F::ONE),
            (COL_ACTIVATION_FINISHED_BEFORE, -F::ONE),
            (COL_TRAPPED_AFTER, -F::ONE),
            (COL_TRAPPED_BEFORE, F::ONE),
            (COL_HOST_CLEAR_ACTIVATION_FINISHED, F::ONE),
            (COL_HOST_EVENT_EXIT_LATCH, -F::ONE),
        ]);
        b.push_row(
            [(COL_HOST_EVENT_EXIT_LATCH, F::ONE)],
            [(EVENTS_REMAINING_AFTER, F::ONE), (EXIT_SCHEDULE_COUNT, -F::ONE)],
            [],
        );
        // The entry-count re-read carries the presence bias: event_index continues
        // at cell - 1 = the export's true entry count.
        b.push_row(
            [(COL_HOST_EVENT_EXIT_LATCH, F::ONE)],
            [
                (EVENT_INDEX_AFTER, F::ONE),
                (EVENT_BASE, -F::ONE),
                (INITIAL_SCHEDULE_COUNT, -F::ONE),
                (COL_ONE, F::ONE),
            ],
            [],
        );
        b.push_row(
            [(COL_HOST_EVENT_EXIT_LATCH, F::ONE)],
            [
                (COL_HOST_CALLEE_FREF_AFTER, F::ONE),
                (COL_TURN_EXPORT_FREF_BEFORE, -F::ONE),
            ],
            [],
        );
    });
}

/// Opaque objects suspend only the outer prefix and chain. Both contexts use
/// the same gather buffer and absorb gadget.
fn push_opaque_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    use crate::layout::COL_HOST_EVENT_SLOT_ARG;
    b.with_tag(host_event("opaque value context"), |b| {
        b.push_linear_zero([
            (COL_OBJECT_ACTIVE_AFTER, F::ONE),
            (COL_OBJECT_ACTIVE_BEFORE, -F::ONE),
            (GK_ENTER, -F::ONE),
            (OBJECT_ROOT_LANE[3], F::ONE),
        ]);
        b.push_row([(GK_ENTER, F::ONE)], [(COL_OBJECT_ACTIVE_BEFORE, F::ONE)], []);
        // only can use GK_ROOT (copy the digest to the outer evbuf) or GK_SAVED
        // (restore the evbuf) if we are in opaque object mode
        b.push_row(
            [(GK_ROOT, F::ONE), (GK_SAVED, F::ONE)],
            [(COL_OBJECT_ACTIVE_BEFORE, F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        for (kind, selectors, source) in [
            (GK_ROOT, OBJECT_ROOT_LANE, COL_COMM_CHAIN_BEFORE),
            (GK_SAVED, OBJECT_SAVED_LANE, COL_OUTER_PREFIX_BEFORE),
        ] {
            b.push_linear_zero(
                selectors
                    .into_iter()
                    .map(|col| (col, F::ONE))
                    .chain([(kind, -F::ONE)]),
            );
            b.push_row(
                [(kind, F::ONE)],
                selectors
                    .into_iter()
                    .enumerate()
                    .map(|(lane, col)| (col, F::from_u64(lane as u64)))
                    .chain([(COL_HOST_EVENT_SLOT_ARG, -F::ONE)]),
                [],
            );
            for lane in 0..4 {
                b.push_row(
                    [(selectors[lane], F::ONE)],
                    [(GSLOT_VALUE, F::ONE), (source[lane], -F::ONE)],
                    [],
                );
            }
        }
        for lane in 0..4 {
            b.push_row(
                [(GK_ENTER, F::ONE)],
                [
                    (COL_OUTER_CHAIN_AFTER[lane], F::ONE),
                    (COL_COMM_CHAIN_BEFORE[lane], -F::ONE),
                ],
                [],
            );
            b.push_row(
                [(COL_ONE, F::ONE), (GK_ENTER, -F::ONE)],
                [
                    (COL_OUTER_CHAIN_AFTER[lane], F::ONE),
                    (COL_OUTER_CHAIN_BEFORE[lane], -F::ONE),
                ],
                [],
            );
            b.push_row(
                [(GK_ENTER, F::ONE)],
                [(COL_OUTER_PREFIX_AFTER[lane], F::ONE), (COL_EVBUF_AFTER[lane], -F::ONE)],
                [],
            );
            b.push_row(
                [(COL_ONE, F::ONE), (GK_ENTER, -F::ONE)],
                [
                    (COL_OUTER_PREFIX_AFTER[lane], F::ONE),
                    (COL_OUTER_PREFIX_BEFORE[lane], -F::ONE),
                ],
                [],
            );
        }
    });
}

/// The absorb row: a non-advice gather at slot seven, which never enters an
/// object, folds `[evbuf_before[0..7] | staged word]` into the chain that the
/// row's object rules leave.
fn push_absorb_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let exit = OBJECT_ROOT_LANE[3];
    b.with_tag(host_event("host event absorb"), |b| {
        b.push_row(
            [(GATHER_WORD_POSITION[7], F::ONE)],
            [(COL_ONE, F::ONE), (G_ADVICE, -F::ONE)],
            [(ABSORB, F::ONE)],
        );
        b.push_row([(ABSORB, F::ONE)], [(GK_ENTER, F::ONE)], []);
        for lane in 0..4 {
            b.push_row(
                [(exit, F::ONE)],
                [
                    (COL_OUTER_CHAIN_BEFORE[lane], F::ONE),
                    (COL_COMM_CHAIN_BEFORE[lane], -F::ONE),
                ],
                [(ABSORB_PREVIOUS[lane], F::ONE), (COL_COMM_CHAIN_BEFORE[lane], -F::ONE)],
            );
        }
        absorb_commitment().push_constraints(b);
    });
}

/// Absorb-buffer writes: gather rows stage one ROM-described word, the
/// absorb row clears the buffer, and every untouched slot is carried.
fn push_buffer_write_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.with_tag(host_event("host event buffer write"), |b| {
        // The absorb row consumes the block: the buffer resets to zero so
        // the next block's unwritten slots are the zero padding.
        for j in 0..8 {
            b.push_row([(ABSORB, F::ONE)], [(COL_EVBUF_AFTER[j], F::ONE)], []);
        }

        // Untouched slots carry: gate out the absorb reset and the gather
        // position that stages this word. `ABSORB` implies slot seven.
        for j in 0..8 {
            let mut gate = vec![(COL_ONE, F::ONE), (GATHER_WORD_POSITION[j], -F::ONE)];
            if j < 7 {
                gate.push((ABSORB, -F::ONE));
            }
            b.push_row(gate, [(COL_EVBUF_AFTER[j], F::ONE), (COL_EVBUF_BEFORE[j], -F::ONE)], []);
        }
    });
}

/// One active chain for every absorb. Enter zeros it after saving the outer
/// chain; the absorb row lands on the commitment; every other row carries the
/// previous chain, which the final root-copy sets to the saved outer chain.
fn push_chain_update_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.with_tag(host_event("host event chain update"), |b| {
        for limb in 0..4 {
            b.push_row([(GK_ENTER, F::ONE)], [(COL_COMM_CHAIN_AFTER[limb], F::ONE)], []);
            b.push_row(
                [(ABSORB, F::ONE)],
                [(COL_COMM_CHAIN_AFTER[limb], F::ONE), (ABSORB_OUTPUT[limb], -F::ONE)],
                [],
            );
            b.push_row(
                [(COL_ONE, F::ONE), (ABSORB, -F::ONE), (GK_ENTER, -F::ONE)],
                [(COL_COMM_CHAIN_AFTER[limb], F::ONE), (ABSORB_PREVIOUS[limb], -F::ONE)],
                [],
            );
        }
    });
}

/// Recompute the turn-boundary schedule-guard inverse from the named columns.
/// Derived-only (like the range-check bits), so witness-tampering helpers can
/// keep it consistent with caller-mutated declared columns.
pub fn write_turn_schedule_guard_witness(wit: &mut [F]) -> Result<(), crate::WasmBuildError> {
    let last = [
        super::super::layout::COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT,
        super::super::layout::COL_HOST_EVENT_EXIT_SCHEDULE_COUNT,
        super::super::layout::COL_TURN_BOUNDARY,
        TURN_SCHEDULE_COUNT_INV,
    ]
    .into_iter()
    .max()
    .expect("schedule columns");
    if wit.len() <= last {
        return Err(crate::WasmBuildError::Trace(
            "turn schedule witness is too short".into(),
        ));
    }
    let delta = wit[super::super::layout::COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT] - F::ONE
        + wit[super::super::layout::COL_HOST_EVENT_EXIT_SCHEDULE_COUNT];
    wit[TURN_SCHEDULE_COUNT_INV] = if wit[super::super::layout::COL_TURN_BOUNDARY] == F::ONE {
        delta.try_inverse().unwrap_or(F::ZERO)
    } else {
        F::ZERO
    };
    Ok(())
}

/// Fill the gadget-internal columns for one row.
pub(crate) fn fill_witness(wit: &mut [F], trace: &WasmVmStep) {
    let bool_f = |flag: bool| if flag { F::ONE } else { F::ZERO };

    // Host-event gather one-hots and staged value.
    if trace.row_kind.is_host_event_gather() {
        let cursor = usize::from(trace.state_before.host_events.slot_cursor);
        wit[GATHER_WORD_POSITION[cursor]] = F::ONE;
        wit[GSLOT_VALUE] = F::from_u64(crate::comm_chain::staged_word(trace));
        if let Some(rom) = trace.host_event_rom_slot {
            wit[GATHER_KIND[rom.kind.index()]] = F::ONE;
            wit[GK2_HI] = bool_f(rom.kind == WasmHostEventSlotKind::Result && rom.variant.is_high_limb());
            wit[G_ADVICE] = bool_f(rom.advice);
            wit[GMEM_LOCAL] = bool_f(rom.variant.uses_local_memory_base());
            wit[GMEM_OUTPUT] = bool_f(rom.variant.uses_output_memory_base());
            wit[GMEM_BYTE] = bool_f(rom.variant.uses_byte_memory_width());
            wit[GMEM_HALF] = bool_f(rom.variant.uses_half_memory_width());
            if usize::from(rom.arg) < 4 {
                if rom.kind == WasmHostEventSlotKind::OpaqueRoot {
                    wit[OBJECT_ROOT_LANE[usize::from(rom.arg)]] = F::ONE;
                }
                if rom.kind == WasmHostEventSlotKind::OpaqueSaved {
                    wit[OBJECT_SAVED_LANE[usize::from(rom.arg)]] = F::ONE;
                }
            }
        }
    }
    // The absorb runs on every row; only `ABSORB` rows read its output.
    wit[ABSORB] = wit[GATHER_WORD_POSITION[7]] * (F::ONE - wit[G_ADVICE]);
    let exit = wit[OBJECT_ROOT_LANE[3]];
    let previous: [F; 4] = core::array::from_fn(|lane| {
        let chain = F::from_u64(trace.state_before.comm_chain[lane]);
        let outer = F::from_u64(trace.state_before.event_absorb.outer_chain[lane]);
        chain + exit * (outer - chain)
    });
    let block: [F; 8] = core::array::from_fn(|word| {
        if word < 7 {
            F::from_u64(trace.state_before.event_absorb.evbuf[word])
        } else {
            wit[GSLOT_VALUE]
        }
    });
    let output = commit_block(previous, block);
    for lane in 0..4 {
        wit[ABSORB_PREVIOUS[lane]] = previous[lane];
        wit[ABSORB_OUTPUT[lane]] = output[lane];
    }
    absorb_commitment().assign_auxiliaries(wit);
    // Atomic host-call arg pops; split calls defer them until resume.
    wit[GHC_PARAMS] = (wit[super::super::layout::COL_HOST_CALL_ACTIVE] - wit[COL_HOST_SPLIT_CALL])
        * wit[super::super::layout::COL_CALL_PARAM_COUNT];
    write_turn_schedule_guard_witness(wit).expect("the complete witness contains the schedule columns");
    // Limb-selected values: filled on every row so the unconditional select
    // rows hold (the limb column is zero off gather rows).
    let read_lo = wit[super::super::layout::COL_STACK_READ_VALUE_LO[0]];
    let read_hi = wit[COL_STACK_READ_VALUE_HI[0]];
    let variant = wit[super::super::layout::COL_HOST_EVENT_SLOT_VARIANT];
    wit[GARG_VAL] = read_lo + variant * (read_hi - read_lo);
    let out_lo = wit[super::super::layout::COL_OUTPUT_VALUE_LO_BEFORE];
    let out_hi = wit[super::super::layout::COL_OUTPUT_VALUE_HI_BEFORE];
    wit[GOUT_VAL] = out_lo + variant * (out_hi - out_lo);
}
