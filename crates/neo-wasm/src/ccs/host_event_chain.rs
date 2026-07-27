//! In-circuit host-event chain gadget: constrains `HostEventPerm` rows to
//! advance the width-12 Poseidon2 block absorb one round-row at a time, and
//! binds the absorb buffer to the host-call rows that stream event words
//! into it. The protocol constants and the native round decomposition live
//! in [`crate::comm_chain`]; every linear map here is probed from those
//! functions, so the circuit cannot drift from the native chain.
//!
//! Row schedule per absorbed block (position one-hot `COL_PERM_POS*`):
//! positions 0-3 initial full rounds (0 also absorbs `[chain | evbuf]`
//! premixed by the initial external layer), 4-14 partial pairs (2 internal
//! rounds each), 15-18 terminal full rounds (18 also feeds the chain
//! forward and is the only row on which `comm_chain` may move).
//!
//! Scheduling soundness: `perm_pending` (raised only by the gated write-row
//! update below) forces the next row to be position 0, the round counter
//! walks 0→18 in lockstep with the one-hot, and the row-kind one-hot in
//! `ccs/call.rs` counts `pending + (round ≠ 0)` as the perm row kind — so
//! `pending = 1 ∧ round ≠ 0` (which would let two positions fire at once)
//! is unreachable: every row that can raise `pending` provably has
//! `round_after = 0`, and perm rows never raise it.
//!
//! S-box shape: each x^7 is 4 unconditional mult rows over dedicated power
//! columns (witness-filled with the powers of whatever the linear input
//! expression evaluates to on non-perm rows, where the round-constant
//! selectors are all zero), and only the linear round-output rows are gated
//! by the position one-hot.
//!
//! Column ownership: only the gadget's *interface* lives in the named wasm
//! layout — the carried absorb state (buffer, slot cursor, pending flag,
//! round counter + its zero-test, permutation lanes) that continuity links,
//! the semantic digest, and `ccs/call.rs` refer to. The gadget-internal
//! witness columns (position one-hot, S-box powers, write masks, event-end
//! products) are allocated here in a private block right after
//! `NAMED_COLUMN_COUNT`, mirroring how the range-check pass owns its bit
//! columns; nothing outside this module may address them.

use super::super::gadgets::push_zero_test_gadget;
use super::super::layout::{
    COL_CALL_PARAM_COUNT, COL_CALL_RESULT_COUNT, COL_COMM_CHAIN0_AFTER, COL_COMM_CHAIN0_BEFORE, COL_EVBUF0_AFTER,
    COL_EVBUF0_BEFORE, COL_EVBUF_SLOT0_AFTER, COL_EVBUF_SLOT0_BEFORE, COL_FUNCTION_REF, COL_GATHER_ACTIVE,
    COL_GRAMMAR_MODE_AFTER, COL_GRAMMAR_MODE_BEFORE, COL_HOST_ARGS_ACTIVE_BEFORE, COL_HOST_ARGS_REMAINING_AFTER,
    COL_HOST_ARGS_REMAINING_AFTER_IS_ZERO, COL_HOST_ARGS_REMAINING_BEFORE, COL_HOST_RESULT_ACTIVE,
    COL_HOST_RESULT_PENDING_AFTER, COL_HOST_RESULT_PENDING_BEFORE, COL_ONE, COL_PERM_PENDING_AFTER,
    COL_PERM_PENDING_BEFORE, COL_PERM_ROUND_AFTER, COL_PERM_ROUND_BEFORE, COL_PERM_ROUND_BEFORE_INV,
    COL_PERM_ROUND_BEFORE_IS_ZERO, COL_PERM_STATE0_AFTER, COL_PERM_STATE0_BEFORE, COL_RAW_ARGS_ACTIVE,
    COL_RAW_HOST_CALL, COL_RAW_RESULT_ACTIVE, COL_STACK_READ0_VALUE_HI, COL_STACK_READ0_VALUE_LO, COL_STACK_READS,
    COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO, COL_STACK_WRITES, COL_TURN_BOUNDARY,
};
use super::super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use super::always;
use super::call::host_call_gate_terms;
use crate::comm_chain::{
    perm_external_linear, perm_full_round_constants, perm_internal_linear, perm_partial_round_constants,
    perm_row_is_full_round, COMM_CHAIN_PERM_ROWS, HOST_CALL_EVENT_TAG, PERM_PARTIAL_FIRST_ROW, PERM_TERMINAL_FIRST_ROW,
};
use crate::ir::WasmVmStep;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

type R1csBuilder = WasmTaggedR1csBuilder;

// Gadget-internal column block, allocated right after the named layout (the
// range-check bit columns follow it). Indices are private: the interface
// columns everything else uses are the named carried-state columns above.
const POS0: usize = crate::witness_layout::POSEIDON_AUX_START; // 19 position one-hot flags
const FULL_T0: usize = POS0 + COMM_CHAIN_PERM_ROWS; // 48 full-round S-box powers
const PARTIAL_U0: usize = FULL_T0 + 48; // 8 partial-pair S-box powers
const WSA0: usize = PARTIAL_U0 + 8; // 4 arg-row write masks
const WSR0: usize = WSA0 + 4; // 4 result-row write masks
const STREAM_DONE: usize = WSR0 + 4; // event stream complete after this row
const EVENT_END: usize = STREAM_DONE + 1; // this row streams the final word pair
const EVENT_END_OR: usize = EVENT_END + 1; // (block filled)·(event end) product
const GW0: usize = EVENT_END_OR + 1; // 8 gather block-word one-hot flags
const GK0: usize = GW0 + 8; // 6 gather slot-kind one-hot flags (const/arg/result/claim/claim-local/output)
const GKINDS: usize = 6;
const GARG_VAL: usize = GK0 + GKINDS; // limb-selected stack-read value
const GOUT_VAL: usize = GARG_VAL + 1; // limb-selected output-carry value (export result)
const GSLOT_VALUE: usize = GOUT_VAL + 1; // the block word this gather row stages
const GK2_HI: usize = GSLOT_VALUE + 1; // result-slot hi-lane write: GK2 · slot_limb
const GHC_PARAMS: usize = GK2_HI + 1; // grammar host-call arg pops: GHC · call_param_count
const G_ADVICE: usize = GHC_PARAMS + 1; // advice-event slot flag (ROM kind cell = kind + 8)

/// Width of the gadget-internal column block.
pub const AUX_WIDTH: usize = G_ADVICE + 1 - POS0;

/// Declared bit-widths of the gadget-internal columns, in block order (for
/// the F' norm decomposition): booleans for the one-hot/masks/products,
/// full field elements for the S-box powers and gather values.
pub(crate) fn auxiliary_column_widths() -> impl Iterator<Item = usize> {
    core::iter::repeat_n(1, COMM_CHAIN_PERM_ROWS)
        .chain(core::iter::repeat_n(64, 48 + 8))
        .chain(core::iter::repeat_n(1, 4 + 4 + 3))
        .chain(core::iter::repeat_n(1, 8 + GKINDS))
        .chain(core::iter::repeat_n(64, 3))
        .chain([1, 64, 1])
}

/// The gather column whose flag pins a non-popping stack read (arg slots).
/// The `sp' = sp - reads + writes` identity in `ccs.rs` exempts these reads
/// from popping. (Result slots WRITE: the Lo slot is a genuine push moving
/// sp through the counted port; the Hi slot uses only the un-counted
/// hi-word port and leaves sp alone.)
pub(super) const fn gather_arg_read_kind_col() -> usize {
    GK0 + 1
}

/// Product column carrying `grammar_host_call · call_param_count`: the sp
/// identity in `ccs.rs` pops all host-call args on the call row itself in
/// grammar mode (there are no HostCallArg aux rows there).
pub(super) const fn grammar_host_call_params_col() -> usize {
    GHC_PARAMS
}

/// The position one-hot flag of the group's last row (position 18), needed
/// by `ccs/call.rs` to hand arg mode back after a perm group.
pub(super) const fn perm_last_pos_col() -> usize {
    POS0 + COMM_CHAIN_PERM_ROWS - 1
}

/// Dense 12×12 matrix of the external (`mds_light`) linear layer, probed
/// from the native implementation.
pub(crate) fn external_matrix() -> [[F; 12]; 12] {
    matrix_of(perm_external_linear)
}

/// Dense 12×12 matrix of the internal (`1 + diag(v)`) linear layer, probed
/// from the native implementation.
pub(crate) fn internal_matrix() -> [[F; 12]; 12] {
    matrix_of(perm_internal_linear)
}

fn matrix_of(apply: fn(&mut [F; 12])) -> [[F; 12]; 12] {
    let mut m = [[F::ZERO; 12]; 12];
    for col in 0..12 {
        let mut basis = [F::ZERO; 12];
        basis[col] = F::ONE;
        apply(&mut basis);
        for (row, value) in basis.iter().enumerate() {
            m[row][col] = *value;
        }
    }
    m
}

/// Gate terms that are 1 exactly on `HostEventPerm` rows: `perm_pending`
/// (position 0) plus "round counter is nonzero" (positions 1..18).
pub(crate) fn perm_row_gate_terms() -> [(usize, F); 3] {
    [
        (COL_PERM_PENDING_BEFORE, F::ONE),
        (COL_ONE, F::ONE),
        (COL_PERM_ROUND_BEFORE_IS_ZERO, -F::ONE),
    ]
}

/// Gate terms that are 1 exactly on the rows streaming RAW event words into
/// the absorb buffer: host-call program rows, host-arg rows, host-result
/// rows — each masked by `1 - grammar_mode` through its product column, so
/// the whole raw machinery is inert when the chain absorbs grammar events.
fn event_write_gate_terms() -> [(usize, F); 3] {
    [
        (COL_RAW_HOST_CALL, F::ONE),
        (COL_RAW_ARGS_ACTIVE, F::ONE),
        (COL_RAW_RESULT_ACTIVE, F::ONE),
    ]
}

pub(super) fn push_constraints(b: &mut R1csBuilder) {
    push_grammar_mode_constraints(b);
    push_grammar_gather_constraints(b);
    push_position_onehot_constraints(b);
    push_pending_update_constraints(b);
    push_buffer_write_constraints(b);
    push_slot_cursor_constraints(b);
    push_absorb_constraints(b);
    push_full_round_constraints(b);
    push_partial_pair_constraints(b);
    push_chain_update_constraints(b);
    push_perm_row_shape_constraints(b);
}

/// Grammar-mode plumbing: the per-program constant flag, the raw-machinery
/// mask products, and the `HostEventGather` row rules. Gather rows stage one
/// expanded event block (their buffer writes are free until the stage-C
/// slot-fill rows bind them to the grammar ROM) and raise `perm_pending`;
/// they exist only in grammar mode, so raw-mode enforcement is unaffected.
fn push_grammar_mode_constraints(b: &mut R1csBuilder) {
    b.with_tag(always("host event grammar mode"), |b| {
        // Per-program constant: preserved on every row; the initial value is
        // verifier-pinned through the semantic-state digest.
        b.push_linear_zero([(COL_GRAMMAR_MODE_AFTER, F::ONE), (COL_GRAMMAR_MODE_BEFORE, -F::ONE)]);

        // Raw-machinery masks: raw_x = x · (1 - mode).
        let not_mode = [(COL_ONE, F::ONE), (COL_GRAMMAR_MODE_BEFORE, -F::ONE)];
        b.push_row(host_call_gate_terms(), not_mode, [(COL_RAW_HOST_CALL, F::ONE)]);
        b.push_row(
            [(COL_HOST_ARGS_ACTIVE_BEFORE, F::ONE)],
            not_mode,
            [(COL_RAW_ARGS_ACTIVE, F::ONE)],
        );
        b.push_row(
            [(COL_HOST_RESULT_ACTIVE, F::ONE)],
            not_mode,
            [(COL_RAW_RESULT_ACTIVE, F::ONE)],
        );

        // Gather rows exist only in grammar mode.
        b.push_row([(COL_GATHER_ACTIVE, F::ONE)], not_mode, []);
        // Disable the pc-to-function lookup on gather, permutation, turn
        // boundary, and padding rows, which do not execute a program
        // instruction.
        b.push_linear_zero([
            (super::super::layout::COL_PC_FREF_ACTIVE, F::ONE),
            (COL_PERM_ROUND_BEFORE_IS_ZERO, -F::ONE),
            (COL_GATHER_ACTIVE, F::ONE),
            (COL_PERM_PENDING_BEFORE, F::ONE),
            (COL_TURN_BOUNDARY, F::ONE),
            (super::super::layout::COL_PADDING_ACTIVE, F::ONE),
        ]);
        // Turn boundaries only exist in grammar mode.
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE)],
            [(COL_ONE, F::ONE), (COL_GRAMMAR_MODE_BEFORE, -F::ONE)],
            [],
        );
        // On the last gather row, pending_after = 1 - advice.
        b.push_row(
            [(GW0 + 7, F::ONE)],
            [(COL_PERM_PENDING_AFTER, F::ONE), (COL_ONE, -F::ONE), (G_ADVICE, F::ONE)],
            [],
        );
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE), (GW0 + 7, -F::ONE)],
            [(COL_PERM_PENDING_AFTER, F::ONE)],
            [],
        );
        // Gather rows read the stack exactly on arg slots; result slots
        // WRITE it — the lo slot through the counted port pair (the push),
        // the hi slot through the hi-word port alone (no sp effect). Both
        // suspend the host-call countdown state like perm rows do.
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE)],
            [(COL_STACK_READS, F::ONE), (GK0 + 1, -F::ONE)],
            [],
        );
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE)],
            [(COL_STACK_WRITES, F::ONE), (GK0 + 2, -F::ONE), (GK2_HI, F::ONE)],
            [],
        );
        for (after, before) in [
            (COL_HOST_ARGS_REMAINING_AFTER, COL_HOST_ARGS_REMAINING_BEFORE),
            (COL_HOST_RESULT_PENDING_AFTER, COL_HOST_RESULT_PENDING_BEFORE),
        ] {
            b.push_row([(COL_GATHER_ACTIVE, F::ONE)], [(after, F::ONE), (before, -F::ONE)], []);
        }
    });
}

/// Grammar gather binding: each gather row stages exactly one block word,
/// whose value is pinned by the grammar ROM entry at
/// `(fref, event_index, slot_cursor)` — a constant, an addressed stack read
/// of an arg/result limb, or a free claim word — and the per-call
/// event schedule is forced by ROM-loaded countdowns. This closes the
/// stage-B gap: with these rows, a grammar chain commits exactly the event
/// sequence obtained by applying the committed tables to the values at the
/// call site.
fn push_grammar_gather_constraints(b: &mut R1csBuilder) {
    use super::super::layout::{
        COL_CALL_PARAM_COUNT as PARAM_COUNT, COL_GATHER_LOCAL_WRITE, COL_GATHER_LOCAL_WRITE_LO,
        COL_GRAMMAR_ARGS_BASE_AFTER as AB_A, COL_GRAMMAR_ARGS_BASE_BEFORE as AB_B, COL_GRAMMAR_EVIDX_AFTER as EVIDX_A,
        COL_GRAMMAR_EVIDX_BEFORE as EVIDX_B, COL_GRAMMAR_EVREM_AFTER as EVREM_A, COL_GRAMMAR_EVREM_BEFORE as EVREM_B,
        COL_GRAMMAR_EVREM_BEFORE_INV as EVREM_INV, COL_GRAMMAR_EVREM_BEFORE_IS_ZERO as EVREM_ISZERO,
        COL_GRAMMAR_EXIT_LATCH, COL_GRAMMAR_HOST_CALL as GHC, COL_GRAMMAR_POST_COUNT as POST_COUNT,
        COL_GRAMMAR_PRE_COUNT as PRE_COUNT, COL_GRAMMAR_SLOT_ARG as SLOT_ARG, COL_GRAMMAR_SLOT_CONST_HI as CONST_HI,
        COL_GRAMMAR_SLOT_CONST_LO as CONST_LO, COL_GRAMMAR_SLOT_CURSOR_AFTER as S_A,
        COL_GRAMMAR_SLOT_CURSOR_BEFORE as S_B, COL_GRAMMAR_SLOT_KIND as SLOT_KIND, COL_GRAMMAR_SLOT_LIMB as SLOT_LIMB,
        COL_IS_PROGRAM_ROW, COL_LOCAL_INDEX, COL_LOCAL_VALUE, COL_LOCAL_VALUE_HI, COL_OUTPUT_CAPTURED,
        COL_OUTPUT_VALUE_HI_BEFORE, COL_OUTPUT_VALUE_LO_BEFORE, COL_SP_BEFORE, COL_STACK_READ0_ADDR_LO,
        COL_TURN_BOUNDARY,
    };
    let ci_sel = super::super::layout::selector_col(crate::isa::WasmOpcode::CallIndirect).expect("ci selector");

    b.with_tag(always("grammar gather binding"), |b| {
        // Grammar-mode row mask: grammar_host_call = gate - raw = gate · mode.
        let [call, ci_not_trap, guest] = host_call_gate_terms();
        b.push_linear_zero([
            (GHC, F::ONE),
            (call.0, -call.1),
            (ci_not_trap.0, -ci_not_trap.1),
            (guest.0, -guest.1),
            (COL_RAW_HOST_CALL, F::ONE),
        ]);
        // Grammar host calls pop their args on the call row itself (no
        // HostCallArg aux rows in grammar mode): the sp identity consumes
        // this product of the mode-masked gate and the ROM-bound arity.
        b.push_row([(GHC, F::ONE)], [(PARAM_COUNT, F::ONE)], [(GHC_PARAMS, F::ONE)]);

        // Event schedule countdown: loaded from the event-count ROMs on the
        // grammar call row (the whole call, args and result, is one atomic
        // event sequence), decremented by each block's last slot row,
        // preserved elsewhere; program rows require it to be spent, and
        // gather rows require it to be live.
        push_zero_test_gadget(b, EVREM_B, EVREM_INV, EVREM_ISZERO);
        b.push_row([(COL_GATHER_ACTIVE, F::ONE)], [(EVREM_ISZERO, F::ONE)], []);
        b.push_row([(COL_IS_PROGRAM_ROW, F::ONE)], [(EVREM_B, F::ONE)], []);
        // Pre-count cells store count + 1 (presence bias): an undeclared
        // import's zero-filled cell loads the poisoned EVREM = -1 = p-1.
        // See the count-family relation-layout comment for the full
        // ROM-address non-termination argument.
        b.push_row(
            [(GHC, F::ONE)],
            [(EVREM_A, F::ONE), (PRE_COUNT, -F::ONE), (COL_ONE, F::ONE)],
            [],
        );
        b.push_row(
            [(GW0 + 7, F::ONE)],
            [(EVREM_A, F::ONE), (EVREM_B, -F::ONE), (COL_ONE, F::ONE)],
            [],
        );
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (GHC, -F::ONE),
                (GW0 + 7, -F::ONE),
                (COL_GRAMMAR_EXIT_LATCH, -F::ONE),
                (COL_TURN_BOUNDARY, -F::ONE),
            ],
            [(EVREM_A, F::ONE), (EVREM_B, -F::ONE)],
            [],
        );
        // Turn boundary: the previous turn's schedules must be spent, and
        // the next export's entry schedule loads from the count ROM (keyed
        // by the repointed attribution, like the exit latch). The presence
        // bias (+1) binds the target to a DECLARED export template: internal
        // functions and imports read the export family's zero-filled 0 and
        // load the poisoned EVREM = p-1 described above.
        b.push_row([(COL_TURN_BOUNDARY, F::ONE)], [(EVREM_B, F::ONE)], []);
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE)],
            [(EVREM_A, F::ONE), (PRE_COUNT, -F::ONE), (COL_ONE, F::ONE)],
            [],
        );

        // Event index: the ROM key component walking the template.
        b.push_row([(GHC, F::ONE)], [(EVIDX_A, F::ONE)], []);
        b.push_row(
            [(GW0 + 7, F::ONE)],
            [(EVIDX_A, F::ONE), (EVIDX_B, -F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (GHC, -F::ONE),
                (GW0 + 7, -F::ONE),
                (COL_GRAMMAR_EXIT_LATCH, -F::ONE),
                (COL_TURN_BOUNDARY, -F::ONE),
            ],
            [(EVIDX_A, F::ONE), (EVIDX_B, -F::ONE)],
            [],
        );
        // Turn boundary: entry events of the next turn are numbered from 0.
        b.push_row([(COL_TURN_BOUNDARY, F::ONE)], [(EVIDX_A, F::ONE)], []);

        // Argument-region base: latched on the grammar call row from bound
        // quantities (sp, the indirect-index pop, the ROM-bound arity).
        b.push_row(
            [(GHC, F::ONE)],
            [
                (AB_A, F::ONE),
                (COL_SP_BEFORE, -F::ONE),
                (ci_sel, F::ONE),
                (PARAM_COUNT, F::ONE),
            ],
            [],
        );
        b.push_row(
            [(COL_ONE, F::ONE), (GHC, -F::ONE)],
            [(AB_A, F::ONE), (AB_B, -F::ONE)],
            [],
        );

        // Slot cursor + block-word one-hot lockstep (the same pattern as the
        // perm position one-hot).
        for k in 0..8 {
            b.push_boolean(GW0 + k);
        }
        b.push_linear_zero(
            (0..8)
                .map(|k| (GW0 + k, F::ONE))
                .chain([(COL_GATHER_ACTIVE, -F::ONE)]),
        );
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE)],
            (0..8)
                .map(|k| (GW0 + k, F::from_u64(k as u64)))
                .chain([(S_B, -F::ONE)]),
            [],
        );
        b.push_linear_zero([
            (S_A, F::ONE),
            (S_B, -F::ONE),
            (COL_GATHER_ACTIVE, -F::ONE),
            (GW0 + 7, F::from_u64(8)),
        ]);
        b.push_row([(COL_IS_PROGRAM_ROW, F::ONE)], [(S_B, F::ONE)], []);

        // SLOT_KIND = raw_kind + 8 * advice.
        for j in 0..GKINDS {
            b.push_boolean(GK0 + j);
        }
        b.push_boolean(G_ADVICE);
        b.push_linear_zero(
            (0..GKINDS)
                .map(|j| (GK0 + j, F::ONE))
                .chain([(COL_GATHER_ACTIVE, -F::ONE)]),
        );
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE)],
            (0..GKINDS)
                .map(|j| (GK0 + j, F::from_u64(j as u64)))
                .chain([(G_ADVICE, F::from_u64(8)), (SLOT_KIND, -F::ONE)]),
            [],
        );

        // The staged word lands in the buffer slot the cursor points at.
        for k in 0..8 {
            b.push_row(
                [(GW0 + k, F::ONE)],
                [(COL_EVBUF0_AFTER + k, F::ONE), (GSLOT_VALUE, -F::ONE)],
                [],
            );
        }

        // Const slots: the word is the ROM constant (u32 limb pair).
        b.push_row(
            [(GK0, F::ONE)],
            [
                (GSLOT_VALUE, F::ONE),
                (CONST_LO, -F::ONE),
                (CONST_HI, -F::from_u64(1 << 32)),
            ],
            [],
        );

        // Result-slot limb split: each lane is written by the slot that
        // absorbs it (the stack twin of the kind-4 locals pattern). Boolean
        // by the ROM's 0/1 limb content; the booleanity row backs the
        // declared 1-bit width.
        b.push_row([(GK0 + 2, F::ONE)], [(SLOT_LIMB, F::ONE)], [(GK2_HI, F::ONE)]);
        b.push_boolean(GK2_HI);

        // Arg slots: an addressed stack read at the table offset from the
        // argument base, limb-selected into the word.
        b.push_row(
            [(GK0 + 1, F::ONE)],
            [
                (COL_STACK_READ0_ADDR_LO, F::ONE),
                (AB_B, -F::from_u64(2)),
                (SLOT_ARG, -F::from_u64(2)),
            ],
            [],
        );
        b.push_row(
            [(SLOT_LIMB, F::ONE)],
            [(COL_STACK_READ0_VALUE_HI, F::ONE), (COL_STACK_READ0_VALUE_LO, -F::ONE)],
            [(GARG_VAL, F::ONE), (COL_STACK_READ0_VALUE_LO, -F::ONE)],
        );
        b.push_row([(GK0 + 1, F::ONE)], [(GSLOT_VALUE, F::ONE), (GARG_VAL, -F::ONE)], []);

        // Result Lo slots (kind 2 with the hi flag low): the gather row
        // WRITES the staged word onto the operand stack — the host result's
        // push, replacing the raw mode's HostCallResult row. The write ports
        // make the sp identity move by +1; the address is the post-pop
        // stack top (= the argument base, so arg-0 slots must be gathered
        // earlier — validated template-side). The write is a narrow TOTAL
        // write: the hi lane is pinned to zero, never advice — an i64
        // result's hi limb arrives through its own Hi slot write below.
        b.push_row(
            [(GK0 + 2, F::ONE), (GK2_HI, -F::ONE)],
            [
                (super::super::layout::COL_STACK_WRITE0_ADDR_LO, F::ONE),
                (COL_SP_BEFORE, -F::from_u64(2)),
            ],
            [],
        );
        b.push_row(
            [(GK0 + 2, F::ONE), (GK2_HI, -F::ONE)],
            [(GSLOT_VALUE, F::ONE), (COL_STACK_WRITE0_VALUE_LO, -F::ONE)],
            [],
        );
        b.push_row(
            [(GK0 + 2, F::ONE), (GK2_HI, -F::ONE)],
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
                (AB_B, -F::from_u64(2)),
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

        // Claim slots (kind 3): free absorbed claim words. Their values —
        // and the identity of slots sharing a claim index — are claim-side
        // structure: expansion resolves every `Claim{idx}` from one claim
        // entry, and the transcript check (native fold or the interleaving
        // proof) binds the absorbed words to that claim.

        // Input-local slots (kind 4): the staged claim-input word is written
        // into one 32-bit lane of the entry frame's locals at the
        // table-pinned index (ROM limb select: 0 lo, 1 hi). Routing the word
        // through the U32-checked locals value columns range-proves it. Lo
        // rows also write the hi lane to zero, so a lone Lo write is total;
        // a Hi row (validated to follow its local's Lo row) overwrites the
        // hi lane with the claim word. The word itself is free at the row
        // level — the final-chain transcript check binds it globally.
        b.push_linear_zero([(COL_GATHER_LOCAL_WRITE, F::ONE), (GK0 + 4, -F::ONE)]);
        b.push_row(
            [(GK0 + 4, F::ONE)],
            [(COL_ONE, F::ONE), (SLOT_LIMB, -F::ONE)],
            [(COL_GATHER_LOCAL_WRITE_LO, F::ONE)],
        );
        b.push_row(
            [(GK0 + 4, F::ONE)],
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

        // Input slots (kind 6): absorb-only claim-input words; free at the
        // row level, bound globally by the final-chain transcript check.

        // Export output slots (kind 5): the carried simple-output value,
        // limb-selected (bound by the output-capture machinery).
        b.push_row(
            [(SLOT_LIMB, F::ONE)],
            [
                (COL_OUTPUT_VALUE_HI_BEFORE, F::ONE),
                (COL_OUTPUT_VALUE_LO_BEFORE, -F::ONE),
            ],
            [(GOUT_VAL, F::ONE), (COL_OUTPUT_VALUE_LO_BEFORE, -F::ONE)],
        );
        b.push_row([(GK0 + 5, F::ONE)], [(GSLOT_VALUE, F::ONE), (GOUT_VAL, -F::ONE)], []);

        // Export exit latch: the output-capture row in grammar mode loads
        // the exit schedule — the exit-event count, the event index
        // continuing after the entry events, and the event attribution
        // repointed at the halting export's fref (all ROM/carried-bound).
        b.push_row(
            [(COL_OUTPUT_CAPTURED, F::ONE)],
            [(COL_GRAMMAR_MODE_BEFORE, F::ONE)],
            [(COL_GRAMMAR_EXIT_LATCH, F::ONE)],
        );
        b.push_row(
            [(COL_GRAMMAR_EXIT_LATCH, F::ONE)],
            [(EVREM_A, F::ONE), (POST_COUNT, -F::ONE)],
            [],
        );
        // The entry-count re-read carries the presence bias: EVIDX continues
        // at cell - 1 = the export's true entry count.
        b.push_row(
            [(COL_GRAMMAR_EXIT_LATCH, F::ONE)],
            [(EVIDX_A, F::ONE), (PRE_COUNT, -F::ONE), (COL_ONE, F::ONE)],
            [],
        );
        b.push_row(
            [(COL_GRAMMAR_EXIT_LATCH, F::ONE)],
            [
                (super::super::layout::COL_HOST_CALLEE_FREF_AFTER, F::ONE),
                (super::super::layout::COL_CURRENT_FUNCTION_REF, -F::ONE),
            ],
            [],
        );
    });
}

/// Position one-hot ↔ round-counter lockstep. The position columns are
/// gadget-internal, so their booleanity rows are pushed here (the
/// range-check pass only covers named columns).
fn push_position_onehot_constraints(b: &mut R1csBuilder) {
    b.with_tag(always("host event perm position"), |b| {
        push_zero_test_gadget(
            b,
            COL_PERM_ROUND_BEFORE,
            COL_PERM_ROUND_BEFORE_INV,
            COL_PERM_ROUND_BEFORE_IS_ZERO,
        );
        for pos in 0..COMM_CHAIN_PERM_ROWS {
            b.push_boolean(POS0 + pos);
        }

        // sum(pos) = pending + (1 - round_is_zero): exactly one position on
        // perm rows, none elsewhere.
        b.push_linear_zero(
            (0..COMM_CHAIN_PERM_ROWS)
                .map(|pos| (POS0 + pos, F::ONE))
                .chain([
                    (COL_PERM_PENDING_BEFORE, -F::ONE),
                    (COL_ONE, -F::ONE),
                    (COL_PERM_ROUND_BEFORE_IS_ZERO, F::ONE),
                ]),
        );
        // sum(pos * P_pos) = round_before: the one-hot points at the counter.
        b.push_linear_zero(
            (0..COMM_CHAIN_PERM_ROWS)
                .map(|pos| (POS0 + pos, F::from_u64(pos as u64)))
                .chain([(COL_PERM_ROUND_BEFORE, -F::ONE)]),
        );
        // round_after = round_before + perm_row_gate - 19 * P_last: advance
        // through the group, wrap to 0 on the last row, preserve elsewhere.
        b.push_linear_zero([
            (COL_PERM_ROUND_AFTER, F::ONE),
            (COL_PERM_ROUND_BEFORE, -F::ONE),
            (COL_PERM_PENDING_BEFORE, -F::ONE),
            (COL_ONE, -F::ONE),
            (COL_PERM_ROUND_BEFORE_IS_ZERO, F::ONE),
            (
                POS0 + COMM_CHAIN_PERM_ROWS - 1,
                F::from_u64(COMM_CHAIN_PERM_ROWS as u64),
            ),
        ]);
        // pending forces the absorb row now, and only pending rows absorb.
        b.push_row(
            [(COL_PERM_PENDING_BEFORE, F::ONE)],
            [(COL_ONE, F::ONE), (POS0, -F::ONE)],
            [],
        );
        b.push_row(
            [(POS0, F::ONE)],
            [(COL_PERM_PENDING_BEFORE, F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
    });
}

/// `perm_pending` lifecycle: raised by the gated write-row formula
/// `WSA3 + WSR3 + E - (WSA3 + WSR3)·E` (block filled or event stream done),
/// cleared on the absorb row, preserved everywhere else.
fn push_pending_update_constraints(b: &mut R1csBuilder) {
    b.with_tag(always("host event pending update"), |b| {
        // The product flags are booleans; the rows also let the F' width
        // audit prove their declared 1-bit width.
        for col in [STREAM_DONE, EVENT_END, EVENT_END_OR] {
            b.push_boolean(col);
        }
        // stream_done = remaining_is_zero · ¬result_pending (valid on every
        // row; only write rows consume it).
        b.push_row(
            [(COL_HOST_ARGS_REMAINING_AFTER_IS_ZERO, F::ONE)],
            [(COL_ONE, F::ONE), (COL_HOST_RESULT_PENDING_AFTER, -F::ONE)],
            [(STREAM_DONE, F::ONE)],
        );
        // end = write_row · stream_done.
        b.push_row(event_write_gate_terms(), [(STREAM_DONE, F::ONE)], [(EVENT_END, F::ONE)]);
        // end_or = (WSA3 + WSR3) · end, so pending can OR both triggers.
        b.push_row(
            [(WSA0 + 3, F::ONE), (WSR0 + 3, F::ONE)],
            [(EVENT_END, F::ONE)],
            [(EVENT_END_OR, F::ONE)],
        );
        // Write rows: pending' = WSA3 + WSR3 + end - end_or.
        b.push_row(
            event_write_gate_terms(),
            [
                (COL_PERM_PENDING_AFTER, F::ONE),
                (WSA0 + 3, -F::ONE),
                (WSR0 + 3, -F::ONE),
                (EVENT_END, -F::ONE),
                (EVENT_END_OR, F::ONE),
            ],
            [],
        );
        // Absorb row consumes the flag.
        b.push_row([(POS0, F::ONE)], [(COL_PERM_PENDING_AFTER, F::ONE)], []);
        // Everything else preserves it (gather rows set it themselves).
        let mut preserve_gate = vec![(COL_ONE, F::ONE), (POS0, -F::ONE), (COL_GATHER_ACTIVE, -F::ONE)];
        preserve_gate.extend(event_write_gate_terms().map(|(col, coeff)| (col, -coeff)));
        b.push_row(
            preserve_gate,
            [(COL_PERM_PENDING_AFTER, F::ONE), (COL_PERM_PENDING_BEFORE, -F::ONE)],
            [],
        );
    });
}

/// Absorb-buffer writes: the call row stamps the 4-word event header (tag,
/// ROM-bound callee fref, ROM-bound arity), arg/result rows land their
/// word pair at the slot cursor, the absorb row clears the buffer, and
/// every untouched slot is carried.
fn push_buffer_write_constraints(b: &mut R1csBuilder) {
    b.with_tag(always("host event buffer write"), |b| {
        // The write masks are booleans; the rows also let the F' width
        // audit prove their declared 1-bit width.
        for k in 0..4 {
            b.push_boolean(WSA0 + k);
            b.push_boolean(WSR0 + k);
        }
        // WSA_k / WSR_k: slot-cursor one-hot masked by the raw writing row
        // kind (inert in grammar mode).
        for k in 0..4 {
            b.push_row(
                [(COL_EVBUF_SLOT0_BEFORE + k, F::ONE)],
                [(COL_RAW_ARGS_ACTIVE, F::ONE)],
                [(WSA0 + k, F::ONE)],
            );
            b.push_row(
                [(COL_EVBUF_SLOT0_BEFORE + k, F::ONE)],
                [(COL_RAW_RESULT_ACTIVE, F::ONE)],
                [(WSR0 + k, F::ONE)],
            );
        }

        // Call-row header: [TAG, fref, param_count, result_count, 0, 0, 0, 0].
        let header = [
            vec![(COL_EVBUF0_AFTER, F::ONE), (COL_ONE, -F::from_u64(HOST_CALL_EVENT_TAG))],
            vec![(COL_EVBUF0_AFTER + 1, F::ONE), (COL_FUNCTION_REF, -F::ONE)],
            vec![(COL_EVBUF0_AFTER + 2, F::ONE), (COL_CALL_PARAM_COUNT, -F::ONE)],
            vec![(COL_EVBUF0_AFTER + 3, F::ONE), (COL_CALL_RESULT_COUNT, -F::ONE)],
            vec![(COL_EVBUF0_AFTER + 4, F::ONE)],
            vec![(COL_EVBUF0_AFTER + 5, F::ONE)],
            vec![(COL_EVBUF0_AFTER + 6, F::ONE)],
            vec![(COL_EVBUF0_AFTER + 7, F::ONE)],
        ];
        for terms in header {
            b.push_row([(COL_RAW_HOST_CALL, F::ONE)], terms, []);
        }

        // Arg/result rows write the popped/pushed value's limbs at the slot
        // cursor; the value columns are bound by the stack argument.
        for k in 0..4 {
            b.push_row(
                [(WSA0 + k, F::ONE)],
                [(COL_EVBUF0_AFTER + 2 * k, F::ONE), (COL_STACK_READ0_VALUE_LO, -F::ONE)],
                [],
            );
            b.push_row(
                [(WSA0 + k, F::ONE)],
                [
                    (COL_EVBUF0_AFTER + 2 * k + 1, F::ONE),
                    (COL_STACK_READ0_VALUE_HI, -F::ONE),
                ],
                [],
            );
            b.push_row(
                [(WSR0 + k, F::ONE)],
                [(COL_EVBUF0_AFTER + 2 * k, F::ONE), (COL_STACK_WRITE0_VALUE_LO, -F::ONE)],
                [],
            );
            b.push_row(
                [(WSR0 + k, F::ONE)],
                [
                    (COL_EVBUF0_AFTER + 2 * k + 1, F::ONE),
                    (COL_STACK_WRITE0_VALUE_HI, -F::ONE),
                ],
                [],
            );
        }

        // The absorb row consumes the block: the buffer resets to zero so
        // the next block's unwritten slots are the zero padding.
        for j in 0..8 {
            b.push_row([(POS0, F::ONE)], [(COL_EVBUF0_AFTER + j, F::ONE)], []);
        }

        // Untouched slots carry: gate out the raw call row (rewrites all 8),
        // this pair's write flags, the absorb reset, and grammar gather rows
        // (which stage a whole block).
        for j in 0..8 {
            let pair = j / 2;
            let gate = vec![
                (COL_ONE, F::ONE),
                (WSA0 + pair, -F::ONE),
                (WSR0 + pair, -F::ONE),
                (POS0, -F::ONE),
                (COL_RAW_HOST_CALL, -F::ONE),
                (GW0 + j, -F::ONE),
            ];
            b.push_row(
                gate,
                [(COL_EVBUF0_AFTER + j, F::ONE), (COL_EVBUF0_BEFORE + j, -F::ONE)],
                [],
            );
        }
    });
}

/// Slot-cursor one-hot: the call row points it at pair 2 (past the header),
/// each word-pair write rotates it, the absorb row resets it to pair 0, and
/// every other row carries it.
fn push_slot_cursor_constraints(b: &mut R1csBuilder) {
    b.with_tag(always("host event slot cursor"), |b| {
        for k in 0..4 {
            let call_target = if k == 2 {
                vec![(COL_EVBUF_SLOT0_AFTER + k, F::ONE), (COL_ONE, -F::ONE)]
            } else {
                vec![(COL_EVBUF_SLOT0_AFTER + k, F::ONE)]
            };
            b.push_row([(COL_RAW_HOST_CALL, F::ONE)], call_target, []);

            b.push_row(
                [(COL_RAW_ARGS_ACTIVE, F::ONE), (COL_RAW_RESULT_ACTIVE, F::ONE)],
                [
                    (COL_EVBUF_SLOT0_AFTER + k, F::ONE),
                    (COL_EVBUF_SLOT0_BEFORE + (k + 3) % 4, -F::ONE),
                ],
                [],
            );

            let absorb_target = if k == 0 {
                vec![(COL_EVBUF_SLOT0_AFTER + k, F::ONE), (COL_ONE, -F::ONE)]
            } else {
                vec![(COL_EVBUF_SLOT0_AFTER + k, F::ONE)]
            };
            b.push_row([(POS0, F::ONE)], absorb_target, []);

            let gate = vec![
                (COL_ONE, F::ONE),
                (COL_RAW_ARGS_ACTIVE, -F::ONE),
                (COL_RAW_RESULT_ACTIVE, -F::ONE),
                (POS0, -F::ONE),
                (COL_RAW_HOST_CALL, -F::ONE),
            ];
            b.push_row(
                gate,
                [
                    (COL_EVBUF_SLOT0_AFTER + k, F::ONE),
                    (COL_EVBUF_SLOT0_BEFORE + k, -F::ONE),
                ],
                [],
            );
        }
    });
}

/// The absorb row's entry state is the premixed block input:
/// `state_before = M_ext · [chain_before | evbuf_before]`.
fn push_absorb_constraints(b: &mut R1csBuilder) {
    let me = external_matrix();
    b.with_tag(always("host event absorb"), |b| {
        for lane in 0..12 {
            let mut terms = vec![(COL_PERM_STATE0_BEFORE + lane, F::ONE)];
            for (k, coeff) in me[lane].iter().enumerate() {
                let input = if k < 4 {
                    COL_COMM_CHAIN0_BEFORE + k
                } else {
                    COL_EVBUF0_BEFORE + (k - 4)
                };
                terms.push((input, -*coeff));
            }
            b.push_row([(POS0, F::ONE)], terms, []);
        }
    });
}

/// Full-round rows: `state_after = M_ext · sbox(state_before + RC[pos])`,
/// with the S-box powers in unconditional mult rows over `COL_PERM_FULL_T*`
/// and the round constants blended in through the position one-hot.
fn push_full_round_constraints(b: &mut R1csBuilder) {
    let me = external_matrix();
    let full_positions: Vec<usize> = (0..COMM_CHAIN_PERM_ROWS)
        .filter(|&p| perm_row_is_full_round(p))
        .collect();

    b.with_tag(always("host event perm full round"), |b| {
        for lane in 0..12 {
            // x = state_before[lane] + sum_pos P_pos * RC[pos][lane]
            let x_terms: Vec<(usize, F)> = core::iter::once((COL_PERM_STATE0_BEFORE + lane, F::ONE))
                .chain(
                    full_positions
                        .iter()
                        .map(|&pos| (POS0 + pos, perm_full_round_constants(pos)[lane])),
                )
                .collect();
            let t = |i: usize| FULL_T0 + 4 * lane + i;
            b.push_row(x_terms.clone(), x_terms.clone(), [(t(0), F::ONE)]);
            b.push_row([(t(0), F::ONE)], [(t(0), F::ONE)], [(t(1), F::ONE)]);
            b.push_row([(t(1), F::ONE)], [(t(0), F::ONE)], [(t(2), F::ONE)]);
            b.push_row([(t(2), F::ONE)], x_terms, [(t(3), F::ONE)]);
        }
        // Gated round output: state_after = M_ext · [t3 per lane].
        let gate: Vec<(usize, F)> = full_positions
            .iter()
            .map(|&pos| (POS0 + pos, F::ONE))
            .collect();
        for lane in 0..12 {
            let mut terms = vec![(COL_PERM_STATE0_AFTER + lane, F::ONE)];
            for (k, coeff) in me[lane].iter().enumerate() {
                terms.push((FULL_T0 + 4 * k + 3, -*coeff));
            }
            b.push_row(gate.clone(), terms, []);
        }
    });
}

/// Partial-pair rows: two internal rounds. Round a S-boxes lane 0 into
/// `U3`, round b S-boxes the mixed lane 0 into `U7`, and the gated output
/// rows apply the composed internal linear layers.
fn push_partial_pair_constraints(b: &mut R1csBuilder) {
    let mi = internal_matrix();
    let partial_positions: Vec<usize> = (PERM_PARTIAL_FIRST_ROW..PERM_TERMINAL_FIRST_ROW).collect();

    // Linear forms over [U3, state_before[1..12]] for the state after round
    // a: t'_i = MI[i][0]·U3 + sum_{j>=1} MI[i][j]·SB_j.
    let u = |i: usize| PARTIAL_U0 + i;

    b.with_tag(always("host event perm partial pair"), |b| {
        // Round a S-box input: x_a = SB_0 + selected RC.
        let x_a: Vec<(usize, F)> = core::iter::once((COL_PERM_STATE0_BEFORE, F::ONE))
            .chain(
                partial_positions
                    .iter()
                    .map(|&pos| (POS0 + pos, perm_partial_round_constants(pos).0)),
            )
            .collect();
        b.push_row(x_a.clone(), x_a.clone(), [(u(0), F::ONE)]);
        b.push_row([(u(0), F::ONE)], [(u(0), F::ONE)], [(u(1), F::ONE)]);
        b.push_row([(u(1), F::ONE)], [(u(0), F::ONE)], [(u(2), F::ONE)]);
        b.push_row([(u(2), F::ONE)], x_a, [(u(3), F::ONE)]);

        // Round b S-box input: x_b = t'_0 + selected RC.
        let mut x_b: Vec<(usize, F)> = vec![(u(3), mi[0][0])];
        for j in 1..12 {
            x_b.push((COL_PERM_STATE0_BEFORE + j, mi[0][j]));
        }
        x_b.extend(
            partial_positions
                .iter()
                .map(|&pos| (POS0 + pos, perm_partial_round_constants(pos).1)),
        );
        b.push_row(x_b.clone(), x_b.clone(), [(u(4), F::ONE)]);
        b.push_row([(u(4), F::ONE)], [(u(4), F::ONE)], [(u(5), F::ONE)]);
        b.push_row([(u(5), F::ONE)], [(u(4), F::ONE)], [(u(6), F::ONE)]);
        b.push_row([(u(6), F::ONE)], x_b, [(u(7), F::ONE)]);

        // Gated output: state_after = MI · [U7 | t'_1..11], with t' expanded
        // over [U3, SB_1..11].
        let gate: Vec<(usize, F)> = partial_positions
            .iter()
            .map(|&pos| (POS0 + pos, F::ONE))
            .collect();
        for lane in 0..12 {
            let mut coeff_u3 = F::ZERO;
            let mut coeff_sb = [F::ZERO; 12];
            for j in 1..12 {
                coeff_u3 += mi[lane][j] * mi[j][0];
                for k in 1..12 {
                    coeff_sb[k] += mi[lane][j] * mi[j][k];
                }
            }
            let mut terms = vec![
                (COL_PERM_STATE0_AFTER + lane, F::ONE),
                (u(7), -mi[lane][0]),
                (u(3), -coeff_u3),
            ];
            for (k, coeff) in coeff_sb.iter().enumerate().skip(1) {
                terms.push((COL_PERM_STATE0_BEFORE + k, -*coeff));
            }
            b.push_row(gate.clone(), terms, []);
        }
    });
}

/// Chain movement: only the group's last row updates `comm_chain`, adding
/// the raw input lanes (feed-forward) to the permutation output; every
/// other row in the trace carries the chain unchanged.
fn push_chain_update_constraints(b: &mut R1csBuilder) {
    let last = POS0 + COMM_CHAIN_PERM_ROWS - 1;
    b.with_tag(always("host event chain update"), |b| {
        for limb in 0..4 {
            b.push_row(
                [(last, F::ONE)],
                [
                    (COL_COMM_CHAIN0_AFTER + limb, F::ONE),
                    (COL_PERM_STATE0_AFTER + limb, -F::ONE),
                    (COL_COMM_CHAIN0_BEFORE + limb, -F::ONE),
                ],
                [],
            );
            b.push_row(
                [(COL_ONE, F::ONE), (last, -F::ONE)],
                [
                    (COL_COMM_CHAIN0_AFTER + limb, F::ONE),
                    (COL_COMM_CHAIN0_BEFORE + limb, -F::ONE),
                ],
                [],
            );
        }
        // Rows that neither run the permutation nor raise `pending` (whose
        // successor's absorb constraint pins the premix through the
        // continuity link) carry the permutation state unchanged.
        for lane in 0..12 {
            b.push_row(
                [
                    (COL_PERM_ROUND_BEFORE_IS_ZERO, F::ONE),
                    (COL_PERM_PENDING_BEFORE, -F::ONE),
                    (COL_PERM_PENDING_AFTER, -F::ONE),
                ],
                [
                    (COL_PERM_STATE0_AFTER + lane, F::ONE),
                    (COL_PERM_STATE0_BEFORE + lane, -F::ONE),
                ],
                [],
            );
        }
    });
}

/// Perm rows are aux rows: no stack traffic, and the host-call countdown
/// state suspends across the group (pc/param-init handling lives with the
/// other aux-row shape rows in `ccs/call.rs`).
fn push_perm_row_shape_constraints(b: &mut R1csBuilder) {
    b.with_tag(always("host event perm row shape"), |b| {
        b.push_row(perm_row_gate_terms(), [(COL_STACK_READS, F::ONE)], []);
        b.push_row(perm_row_gate_terms(), [(COL_STACK_WRITES, F::ONE)], []);
        for (after, before) in [
            (COL_HOST_ARGS_REMAINING_AFTER, COL_HOST_ARGS_REMAINING_BEFORE),
            (COL_HOST_RESULT_PENDING_AFTER, COL_HOST_RESULT_PENDING_BEFORE),
        ] {
            b.push_row(perm_row_gate_terms(), [(after, F::ONE), (before, -F::ONE)], []);
        }
    });
}

/// Witness fill of the gadget-internal column block for one row.
///
/// Reads the named interface columns (which must already be filled) for its
/// gates, and the carried absorb state from the trace row. The S-box power
/// columns are filled on *every* row with the powers of their linear input
/// expression — on non-perm rows the round-constant selectors are zero, so
/// the inputs degenerate to the carried permutation lanes.
pub(crate) fn fill_witness(wit: &mut [F], trace: &WasmVmStep) {
    let bool_f = |flag: bool| if flag { F::ONE } else { F::ZERO };
    let before = trace.state_before.event_absorb;
    let sb: [F; 12] = before.perm_state.map(F::from_u64);

    let pos = trace
        .row_kind
        .is_host_event_perm()
        .then_some(usize::from(before.perm_round));
    if let Some(pos) = pos {
        wit[POS0 + pos] = F::ONE;
    }

    // Buffer-write masks and pending-update products, from the raw-masked
    // flags (filled by `fill_event_absorb`; inert in grammar mode).
    let args_active = wit[super::super::layout::COL_RAW_ARGS_ACTIVE];
    let result_active = wit[super::super::layout::COL_RAW_RESULT_ACTIVE];
    for k in 0..4 {
        wit[WSA0 + k] = wit[COL_EVBUF_SLOT0_BEFORE + k] * args_active;
        wit[WSR0 + k] = wit[COL_EVBUF_SLOT0_BEFORE + k] * result_active;
    }
    let stream_done = bool_f(trace.state_after.host_args.remaining == 0 && !trace.state_after.host_result_pending);
    wit[STREAM_DONE] = stream_done;
    // write row = raw host-call program row + raw arg row + raw result row
    // (each {0,1}, mutually exclusive), mirroring `event_write_gate_terms`.
    let write_row = wit[super::super::layout::COL_RAW_HOST_CALL] + args_active + result_active;
    let end = write_row * stream_done;
    wit[EVENT_END] = end;
    wit[EVENT_END_OR] = (wit[WSA0 + 3] + wit[WSR0 + 3]) * end;

    // Full-round S-box powers: x = state_before[lane] + selected RC.
    for lane in 0..12 {
        let rc = pos
            .filter(|&p| perm_row_is_full_round(p))
            .map(|p| perm_full_round_constants(p)[lane])
            .unwrap_or(F::ZERO);
        let x = sb[lane] + rc;
        let t = FULL_T0 + 4 * lane;
        wit[t] = x * x;
        wit[t + 1] = wit[t] * wit[t];
        wit[t + 2] = wit[t + 1] * wit[t];
        wit[t + 3] = wit[t + 2] * x;
    }

    // Partial-pair S-box powers: round a on lane 0, internal mix, round b.
    let (rc_a, rc_b) = pos
        .filter(|&p| !perm_row_is_full_round(p))
        .map(perm_partial_round_constants)
        .unwrap_or((F::ZERO, F::ZERO));
    let u = PARTIAL_U0;
    let x_a = sb[0] + rc_a;
    wit[u] = x_a * x_a;
    wit[u + 1] = wit[u] * wit[u];
    wit[u + 2] = wit[u + 1] * wit[u];
    wit[u + 3] = wit[u + 2] * x_a;
    let mut mixed = sb;
    mixed[0] = wit[u + 3];
    perm_internal_linear(&mut mixed);
    let x_b = mixed[0] + rc_b;
    wit[u + 4] = x_b * x_b;
    wit[u + 5] = wit[u + 4] * wit[u + 4];
    wit[u + 6] = wit[u + 5] * wit[u + 4];
    wit[u + 7] = wit[u + 6] * x_b;

    // Grammar gather one-hots and staged value.
    if trace.row_kind.is_host_event_gather() {
        let cursor = usize::from(trace.state_before.grammar.slot_cursor);
        wit[GW0 + cursor] = F::ONE;
        wit[GSLOT_VALUE] = F::from_u64(trace.state_after.event_absorb.evbuf[cursor]);
        if let Some(rom) = trace.grammar_rom_slot {
            wit[GK0 + usize::from(rom.kind)] = F::ONE;
            wit[GK2_HI] = bool_f(rom.kind == 2 && rom.limb == 1);
            wit[G_ADVICE] = bool_f(rom.advice);
        }
    }
    // Grammar host-call arg pops: GHC · ROM-bound param count.
    wit[GHC_PARAMS] =
        wit[super::super::layout::COL_GRAMMAR_HOST_CALL] * wit[super::super::layout::COL_CALL_PARAM_COUNT];
    // Limb-selected values: filled on every row so the unconditional select
    // rows hold (the limb column is zero off gather rows).
    let read_lo = wit[super::super::layout::COL_STACK_READ0_VALUE_LO];
    let read_hi = wit[COL_STACK_READ0_VALUE_HI];
    let limb = wit[super::super::layout::COL_GRAMMAR_SLOT_LIMB];
    wit[GARG_VAL] = read_lo + limb * (read_hi - read_lo);
    let out_lo = wit[super::super::layout::COL_OUTPUT_VALUE_LO_BEFORE];
    let out_hi = wit[super::super::layout::COL_OUTPUT_VALUE_HI_BEFORE];
    wit[GOUT_VAL] = out_lo + limb * (out_hi - out_lo);
}
