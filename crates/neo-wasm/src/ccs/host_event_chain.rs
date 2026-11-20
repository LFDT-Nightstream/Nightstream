//! In-circuit host-event chain gadget: binds the absorb buffer to host-event
//! gather rows and folds each filled block into the chain on the gather row
//! that stages its last word.

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
use crate::layout::{COL_HOST_CLEAR_ACTIVATION_FINISHED, COL_HOST_RESUME, COL_HOST_SPLIT_CALL, COL_TRAP_UNWIND};
use neo_application::event_commitment::{commit_block, EventCommitment, EVENT_COMMITMENT_AUX_COLUMNS};
use neo_application::{define_column_region, ZeroTest};
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

pub(crate) const HOST_EVENTS_REMAINING_ZERO_TEST: ZeroTest = ZeroTest::column(
    COL_HOST_EVENTS_REMAINING_BEFORE,
    COL_HOST_EVENTS_REMAINING_BEFORE_INV,
    COL_HOST_EVENTS_REMAINING_BEFORE_IS_ZERO,
);

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

const GK_CONST: usize = GATHER_KIND[WasmHostEventSlotKind::Const.index()];
const GK_ARG: usize = GATHER_KIND[WasmHostEventSlotKind::Arg.index()];
const GK_RESULT: usize = GATHER_KIND[WasmHostEventSlotKind::Result.index()];
const GK_INPUT_LOCAL: usize = GATHER_KIND[WasmHostEventSlotKind::InputLocal.index()];
const GK_OUTPUT: usize = GATHER_KIND[WasmHostEventSlotKind::Output.index()];
const GK_MEMORY_READ: usize = GATHER_KIND[WasmHostEventSlotKind::MemoryRead.index()];
const GK_MEMORY_WRITE: usize = GATHER_KIND[WasmHostEventSlotKind::MemoryWrite.index()];
const GK_ENTER: usize = GATHER_KIND[WasmHostEventSlotKind::EnterOpaque.index()];
const GK_ROOT: usize = GATHER_KIND[WasmHostEventSlotKind::OpaqueRoot.index()];
const GK_SAVED: usize = GATHER_KIND[WasmHostEventSlotKind::OpaqueSaved.index()];

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

pub(super) const fn host_call_params_col() -> usize {
    GHC_PARAMS
}

pub(crate) fn absorb_commitment() -> EventCommitment {
    EventCommitment {
        previous: ABSORB_PREVIOUS,
        block: core::array::from_fn(|word| if word < 7 { COL_EVBUF_BEFORE[word] } else { GSLOT_VALUE }),
        output: ABSORB_OUTPUT,
        auxiliary_start: ABSORB_PERMUTATION[0],
    }
}

pub(crate) fn absorb_flag_column() -> usize {
    ABSORB
}

pub(super) fn push_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    push_interface_constraints(b);
    push_host_event_gather_constraints(b);
    push_opaque_constraints(b);
    push_absorb_constraints(b);
    push_buffer_write_constraints(b);
    push_chain_update_constraints(b);
}

fn push_interface_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.with_tag(host_event("host event interface"), |b| {
        b.push_linear_zero(
            [(super::super::layout::COL_HOST_CALL_ACTIVE, F::ONE)]
                .into_iter()
                .chain(host_call_gate_terms().map(|(column, coefficient)| (column, -coefficient))),
        );
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
        b.push_row(
            [(HOST_CALL_ACTIVE, F::ONE), (COL_HOST_SPLIT_CALL, -F::ONE)],
            [(PARAM_COUNT, F::ONE)],
            [(GHC_PARAMS, F::ONE)],
        );

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
            [
                (INITIAL_SCHEDULE_COUNT, F::ONE),
                (COL_ONE, -F::ONE),
                (EXIT_SCHEDULE_COUNT, F::ONE),
            ],
            [(TURN_SCHEDULE_COUNT_INV, F::ONE)],
            [(COL_TURN_BOUNDARY, F::ONE)],
        );

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
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE)],
            [(EVENT_INDEX_AFTER, F::ONE), (EVENT_BASE, -F::ONE)],
            [],
        );

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
                (COL_TRAP_UNWIND, -F::ONE),
            ],
            [(ARGS_BASE_AFTER, F::ONE), (ARGS_BASE_BEFORE, -F::ONE)],
            [],
        );

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

        for k in 0..8 {
            let gate = if k == 7 {
                vec![(GATHER_WORD_POSITION[k], F::ONE), (ABSORB, -F::ONE)]
            } else {
                vec![(GATHER_WORD_POSITION[k], F::ONE)]
            };
            b.push_row(gate, [(COL_EVBUF_AFTER[k], F::ONE), (GSLOT_VALUE, -F::ONE)], []);
        }

        b.push_row(
            [(GK_CONST, F::ONE)],
            [
                (GSLOT_VALUE, F::ONE),
                (SLOT_IMMEDIATE0, -F::ONE),
                (SLOT_IMMEDIATE1, -F::from_u64(1 << 32)),
            ],
            [],
        );

        b.push_row([(GK_RESULT, F::ONE)], [(SLOT_VARIANT, F::ONE)], [(GK2_HI, F::ONE)]);
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
        b.push_linear_zero([
            (super::super::layout::COL_STACK_WRITE0_HI_ACTIVE, F::ONE),
            (super::super::layout::COL_STACK_WRITE0_ACTIVE, -F::ONE),
            (GK2_HI, -F::ONE),
        ]);

        // Input-local slots: the staged input word is written
        // into one 32-bit lane of the entry frame's locals at the
        // table-pinned index (ROM limb select: 0 lo, 1 hi). Routing the word
        // through the locals memory writes supplies its U32 bound (SPEC.md section 7.6). Lo
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
        b.push_row(
            [(COL_GATHER_LOCAL_WRITE, F::ONE), (COL_GATHER_LOCAL_WRITE_LO, -F::ONE)],
            [(COL_LOCAL_VALUE_HI, F::ONE), (GSLOT_VALUE, -F::ONE)],
            [],
        );

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
        b.push_row(
            [(GMEM_LOCAL, F::ONE)],
            [(COL_LOCAL_INDEX, F::ONE), (SLOT_ARG, -F::ONE)],
            [],
        );
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

        b.push_linear_zero([
            (COL_ACTIVATION_FINISHED_AFTER, F::ONE),
            (COL_ACTIVATION_FINISHED_BEFORE, -F::ONE),
            (COL_TRAPPED_AFTER, -F::ONE),
            (COL_TRAPPED_BEFORE, F::ONE),
            (COL_HOST_CLEAR_ACTIVATION_FINISHED, F::ONE),
            (COL_HOST_EVENT_EXIT_LATCH, -F::ONE),
            (COL_TRAP_UNWIND, -F::ONE),
        ]);
        b.push_row(
            [(COL_HOST_EVENT_EXIT_LATCH, F::ONE)],
            [(EVENTS_REMAINING_AFTER, F::ONE), (EXIT_SCHEDULE_COUNT, -F::ONE)],
            [],
        );
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

fn push_buffer_write_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.with_tag(host_event("host event buffer write"), |b| {
        for j in 0..8 {
            b.push_row([(ABSORB, F::ONE)], [(COL_EVBUF_AFTER[j], F::ONE)], []);
        }

        for j in 0..8 {
            let mut gate = vec![(COL_ONE, F::ONE), (GATHER_WORD_POSITION[j], -F::ONE)];
            if j < 7 {
                gate.push((ABSORB, -F::ONE));
            }
            b.push_row(gate, [(COL_EVBUF_AFTER[j], F::ONE), (COL_EVBUF_BEFORE[j], -F::ONE)], []);
        }
    });
}

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

pub(crate) fn fill_witness(wit: &mut [F], trace: &WasmVmStep) {
    let bool_f = |flag: bool| if flag { F::ONE } else { F::ZERO };

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
    wit[GHC_PARAMS] = (wit[super::super::layout::COL_HOST_CALL_ACTIVE] - wit[COL_HOST_SPLIT_CALL])
        * wit[super::super::layout::COL_CALL_PARAM_COUNT];
    write_turn_schedule_guard_witness(wit).expect("the complete witness contains the schedule columns");
    let read_lo = wit[super::super::layout::COL_STACK_READ_VALUE_LO[0]];
    let read_hi = wit[COL_STACK_READ_VALUE_HI[0]];
    let variant = wit[super::super::layout::COL_HOST_EVENT_SLOT_VARIANT];
    wit[GARG_VAL] = read_lo + variant * (read_hi - read_lo);
    let out_lo = wit[super::super::layout::COL_OUTPUT_VALUE_LO_BEFORE];
    let out_hi = wit[super::super::layout::COL_OUTPUT_VALUE_HI_BEFORE];
    wit[GOUT_VAL] = out_lo + variant * (out_hi - out_lo);
}
