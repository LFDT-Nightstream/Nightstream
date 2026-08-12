//! In-circuit host-event chain gadget: constrains `HostEventPerm` rows to
//! advance the width-12 Poseidon2 block absorb one round-row at a time, and
//! binds the absorb buffer to grammar gather rows. The protocol constants
//! and the native round decomposition live
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
//! Column ownership: the gadget's shared interface lives in
//! `host_event_layout` — the carried absorb state (buffer, pending flag,
//! round counter + its zero-test, permutation lanes) that continuity links,
//! the semantic digest, and `ccs/call.rs` refer to. The gadget-internal
//! witness columns (position one-hot, S-box powers, and gather decoding) are
//! allocated here in a private block right after
//! `NAMED_COLUMN_COUNT`, mirroring how the range-check pass owns its bit
//! columns; nothing outside this module may address them.

use super::super::gadgets::push_zero_test_gadget;
use super::super::layout::{
    COL_COMM_CHAIN_AFTER, COL_COMM_CHAIN_BEFORE, COL_EVBUF_AFTER, COL_EVBUF_BEFORE, COL_EVENT_BINDING_ACTIVE_AFTER,
    COL_EVENT_BINDING_ACTIVE_BEFORE, COL_GATHER_ACTIVE, COL_HOST_CALLEE_FREF_AFTER, COL_ONE, COL_PERM_PENDING_AFTER,
    COL_PERM_PENDING_BEFORE, COL_PERM_ROUND_AFTER, COL_PERM_ROUND_BEFORE, COL_PERM_ROUND_BEFORE_INV,
    COL_PERM_ROUND_BEFORE_IS_ZERO, COL_PERM_STATE_AFTER, COL_PERM_STATE_BEFORE, COL_STACK_READS,
    COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO,
    COL_STACK_WRITES, COL_TURN_BOUNDARY, COL_TURN_EXPORT_FREF_AFTER, COL_TURN_EXPORT_FREF_BEFORE,
};
use super::super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use super::call::host_call_gate_terms;
use super::host_event;
use crate::column_registry::define_column_region;
use crate::comm_chain::{
    perm_external_linear, perm_full_round_constants, perm_internal_linear, perm_partial_round_constants,
    perm_row_is_full_round, COMM_CHAIN_PERM_ROWS, PERM_PARTIAL_FIRST_ROW, PERM_TERMINAL_FIRST_ROW,
};
use crate::ir::{WasmGrammarSlotKind, WasmVmStep};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

type R1csBuilder = WasmTaggedR1csBuilder;

// Gadget-internal column block, allocated right after the named layout (the
// range-check bit columns follow it). Indices are private: the interface
// columns everything else uses are the named carried-state columns above.
const GKINDS: usize = WasmGrammarSlotKind::COUNT;
define_column_region! {
    region: "host_event_chain_aux",
    start: crate::witness_layout::HOST_EVENT_AUX_START,
    width: pub AUX_WIDTH,
    specs: pub AUX_COLUMN_SPECS,
    indices: pub(self),
    columns: [
        PERM_POSITION: [Boolean; COMM_CHAIN_PERM_ROWS] => "permutation row-position one-hot flags",
        FULL_ROUND_POWERS: [Field; 48] => "full-round S-box power witnesses",
        PARTIAL_ROUND_POWERS: [Field; 8] => "partial-round S-box power witnesses",
        GATHER_WORD_POSITION: [Boolean; 8] => "gather block-word one-hot flags",
        GATHER_KIND: [Boolean; GKINDS] => "gather slot-kind one-hot flags",
        GARG_VAL: Field => "limb-selected stack argument value",
        GOUT_VAL: Field => "limb-selected export output value",
        GSLOT_VALUE: Field => "word staged by the current gather row",
        GK2_HI: Boolean => "result-slot high-limb write flag",
        GHC_PARAMS: Field => "grammar host-call and parameter-count product",
        G_ADVICE: Boolean => "advice-event slot flag",
        GMEM_LOCAL: Boolean => "memory pointer comes from an export local",
        GMEM_BYTE: Boolean => "byte-width grammar memory slot",
        GMEM_HALF: Boolean => "half-width grammar memory slot",
    ]
}

const GK_ARG: usize = GATHER_KIND[WasmGrammarSlotKind::Arg.index()];
const GK_RESULT: usize = GATHER_KIND[WasmGrammarSlotKind::Result.index()];
const GK_CLAIM_LOCAL: usize = GATHER_KIND[WasmGrammarSlotKind::ClaimLocal.index()];
const GK_OUTPUT: usize = GATHER_KIND[WasmGrammarSlotKind::Output.index()];
const GK_MEMORY_READ: usize = GATHER_KIND[WasmGrammarSlotKind::MemoryRead.index()];
const GK_MEMORY_WRITE: usize = GATHER_KIND[WasmGrammarSlotKind::MemoryWrite.index()];

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

pub(super) fn push_constraints(b: &mut R1csBuilder) {
    push_interface_constraints(b);
    push_grammar_gather_constraints(b);
    push_position_onehot_constraints(b);
    push_pending_update_constraints(b);
    push_buffer_write_constraints(b);
    push_absorb_constraints(b);
    push_full_round_constraints(b);
    push_partial_pair_constraints(b);
    push_chain_update_constraints(b);
    push_perm_row_shape_constraints(b);
}

/// Shared host-event row shape and carried-state transitions.
fn push_interface_constraints(b: &mut R1csBuilder) {
    b.with_tag(host_event("host event interface"), |b| {
        b.push_linear_zero(
            [(super::super::layout::COL_HOST_CALL_ACTIVE, F::ONE)]
                .into_iter()
                .chain(host_call_gate_terms().map(|(column, coefficient)| (column, -coefficient))),
        );
        b.push_linear_zero([
            (COL_EVENT_BINDING_ACTIVE_AFTER, F::ONE),
            (COL_EVENT_BINDING_ACTIVE_BEFORE, -F::ONE),
        ]);
        for gate in [
            COL_GATHER_ACTIVE,
            super::super::layout::COL_HOST_CALL_ACTIVE,
            COL_TURN_BOUNDARY,
        ] {
            b.push_row(
                [(gate, F::ONE)],
                [(COL_ONE, F::ONE), (COL_EVENT_BINDING_ACTIVE_BEFORE, -F::ONE)],
                [],
            );
        }

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
        b.push_row(
            [(COL_ONE, F::ONE), (COL_TURN_BOUNDARY, -F::ONE)],
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
        // On the last gather row, pending_after = 1 - advice.
        b.push_row(
            [(GATHER_WORD_POSITION[7], F::ONE)],
            [(COL_PERM_PENDING_AFTER, F::ONE), (COL_ONE, -F::ONE), (G_ADVICE, F::ONE)],
            [],
        );
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE), (GATHER_WORD_POSITION[7], -F::ONE)],
            [(COL_PERM_PENDING_AFTER, F::ONE)],
            [],
        );
        // Gather rows read the stack on arg slots and on memory slots whose
        // pointer base is an import argument; result slots
        // WRITE it — the lo slot through the counted port pair (the push),
        // the hi slot through the hi-word port alone (no sp effect). Both
        // preserve the rest of the VM state like permutation rows do.
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE)],
            [
                (COL_STACK_READS, F::ONE),
                (GK_ARG, -F::ONE),
                (GK_MEMORY_READ, -F::ONE),
                (GK_MEMORY_WRITE, -F::ONE),
                (GMEM_LOCAL, F::ONE),
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

/// Grammar gather binding: each gather row stages exactly one block word,
/// whose value is pinned by the grammar ROM entry at
/// `(fref, event_index, slot_cursor)` — a flat value, claim, or aligned
/// linear-memory access — and the per-call
/// event schedule is forced by ROM-loaded countdowns. This closes the
/// stage-B gap: with these rows, a grammar chain commits exactly the event
/// sequence obtained by applying the committed tables to the values at the
/// call site.
fn push_grammar_gather_constraints(b: &mut R1csBuilder) {
    use super::super::layout::{
        COL_CALL_PARAM_COUNT as PARAM_COUNT, COL_GATHER_LOCAL_WRITE, COL_GATHER_LOCAL_WRITE_LO,
        COL_GRAMMAR_ARGS_BASE_AFTER as ARGS_BASE_AFTER, COL_GRAMMAR_ARGS_BASE_BEFORE as ARGS_BASE_BEFORE,
        COL_GRAMMAR_EVIDX_AFTER as EVIDX_A, COL_GRAMMAR_EVIDX_BEFORE as EVIDX_B, COL_GRAMMAR_EVREM_AFTER as EVREM_A,
        COL_GRAMMAR_EVREM_BEFORE as EVREM_B, COL_GRAMMAR_EVREM_BEFORE_INV as EVREM_INV,
        COL_GRAMMAR_EVREM_BEFORE_IS_ZERO as EVREM_ISZERO, COL_GRAMMAR_EXIT_LATCH, COL_GRAMMAR_POST_COUNT as POST_COUNT,
        COL_GRAMMAR_PRE_COUNT as PRE_COUNT, COL_GRAMMAR_SLOT_ARG as SLOT_ARG, COL_GRAMMAR_SLOT_CONST_HI as CONST_HI,
        COL_GRAMMAR_SLOT_CONST_LO as CONST_LO, COL_GRAMMAR_SLOT_CURSOR_AFTER as S_A,
        COL_GRAMMAR_SLOT_CURSOR_BEFORE as S_B, COL_GRAMMAR_SLOT_KIND as SLOT_KIND,
        COL_GRAMMAR_SLOT_VARIANT as SLOT_VARIANT, COL_HALTED, COL_HALTED_BEFORE, COL_HOST_CALL_ACTIVE as GHC,
        COL_IS_PROGRAM_ROW, COL_LINEAR_MEM_ACCESS_BYTE0, COL_LINEAR_MEM_ACCESS_BYTE1, COL_LINEAR_MEM_BYTE_OFFSET,
        COL_LINEAR_MEM_LANE_ADDR, COL_LINEAR_MEM_LANE_VALUE, COL_LINEAR_MEM_OFFSET_IS_1, COL_LINEAR_MEM_OFFSET_IS_3,
        COL_LOCAL_INDEX, COL_LOCAL_VALUE, COL_LOCAL_VALUE_HI, COL_MEM_OOB, COL_OUTPUT_ENABLED_BEFORE,
        COL_OUTPUT_VALUE_HI_BEFORE, COL_OUTPUT_VALUE_LO_BEFORE, COL_SP_BEFORE, COL_STACK_READ_ADDR_LO,
        COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE, COL_TURN_BOUNDARY,
    };
    let ci_sel = super::super::layout::selector_col(crate::isa::WasmOpcode::CallIndirect).expect("ci selector");

    b.with_tag(host_event("grammar gather binding"), |b| {
        // Host calls pop their args on the call row itself. The sp identity
        // consumes this product of the host-call gate and ROM-bound arity.
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
            [(GATHER_WORD_POSITION[7], F::ONE)],
            [(EVREM_A, F::ONE), (EVREM_B, -F::ONE), (COL_ONE, F::ONE)],
            [],
        );
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (GHC, -F::ONE),
                (GATHER_WORD_POSITION[7], -F::ONE),
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
            [(GATHER_WORD_POSITION[7], F::ONE)],
            [(EVIDX_A, F::ONE), (EVIDX_B, -F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (GHC, -F::ONE),
                (GATHER_WORD_POSITION[7], -F::ONE),
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
                (ARGS_BASE_AFTER, F::ONE),
                (COL_SP_BEFORE, -F::ONE),
                (ci_sel, F::ONE),
                (PARAM_COUNT, F::ONE),
            ],
            [],
        );
        b.push_row(
            [(COL_ONE, F::ONE), (GHC, -F::ONE)],
            [(ARGS_BASE_AFTER, F::ONE), (ARGS_BASE_BEFORE, -F::ONE)],
            [],
        );

        // Slot cursor + block-word one-hot lockstep (the same pattern as the
        // perm position one-hot).
        for k in 0..8 {
            b.push_boolean(GATHER_WORD_POSITION[k]);
        }
        b.push_linear_zero(
            (0..8)
                .map(|k| (GATHER_WORD_POSITION[k], F::ONE))
                .chain([(COL_GATHER_ACTIVE, -F::ONE)]),
        );
        b.push_row(
            [(COL_GATHER_ACTIVE, F::ONE)],
            (0..8)
                .map(|k| (GATHER_WORD_POSITION[k], F::from_u64(k as u64)))
                .chain([(S_B, -F::ONE)]),
            [],
        );
        b.push_linear_zero([
            (S_A, F::ONE),
            (S_B, -F::ONE),
            (COL_GATHER_ACTIVE, -F::ONE),
            (GATHER_WORD_POSITION[7], F::from_u64(8)),
        ]);
        b.push_row([(COL_IS_PROGRAM_ROW, F::ONE)], [(S_B, F::ONE)], []);

        // Advice uses the next code range above the raw slot kinds.
        for j in 0..GKINDS {
            b.push_boolean(GATHER_KIND[j]);
        }
        b.push_boolean(G_ADVICE);
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
                    (G_ADVICE, F::from_u64(WasmGrammarSlotKind::COUNT as u64)),
                    (SLOT_KIND, -F::ONE),
                ]),
            [],
        );

        // The staged word lands in the buffer slot the cursor points at.
        for k in 0..8 {
            b.push_row(
                [(GATHER_WORD_POSITION[k], F::ONE)],
                [(COL_EVBUF_AFTER[k], F::ONE), (GSLOT_VALUE, -F::ONE)],
                [],
            );
        }

        // Const slots: the word is the ROM constant (u32 limb pair).
        b.push_row(
            [(GATHER_KIND[0], F::ONE)],
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
        b.push_row([(GK_RESULT, F::ONE)], [(SLOT_VARIANT, F::ONE)], [(GK2_HI, F::ONE)]);
        b.push_boolean(GK2_HI);

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
        b.push_linear_zero([(COL_GATHER_LOCAL_WRITE, F::ONE), (GK_CLAIM_LOCAL, -F::ONE)]);
        b.push_row(
            [(GK_CLAIM_LOCAL, F::ONE)],
            [(COL_ONE, F::ONE), (SLOT_VARIANT, -F::ONE)],
            [(COL_GATHER_LOCAL_WRITE_LO, F::ONE)],
        );
        b.push_row(
            [(GK_CLAIM_LOCAL, F::ONE)],
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

        // Memory slots (kinds 6-7): ROM variant bit 0 selects an
        // import-argument or export-local pointer base; bits 1 and 2 select
        // byte and half width respectively.
        b.push_row(
            [(GK_MEMORY_READ, F::ONE), (GK_MEMORY_WRITE, F::ONE)],
            [(SLOT_VARIANT, F::ONE)],
            [
                (GMEM_LOCAL, F::ONE),
                (
                    GMEM_BYTE,
                    F::from_u64(u64::from(crate::ir::WasmGrammarRomVariant::MEMORY_BYTE_ENCODING_FACTOR)),
                ),
                (
                    GMEM_HALF,
                    F::from_u64(u64::from(crate::ir::WasmGrammarRomVariant::MEMORY_HALF_ENCODING_FACTOR)),
                ),
            ],
        );
        b.push_boolean(GMEM_LOCAL);
        b.push_boolean(GMEM_BYTE);
        b.push_boolean(GMEM_HALF);
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
            ],
            [
                (COL_LINEAR_MEM_LANE_ADDR[0], F::from_u64(4)),
                (COL_LINEAR_MEM_BYTE_OFFSET, F::ONE),
                (COL_STACK_READ_VALUE_LO[0], -F::ONE),
                (CONST_LO, -F::ONE),
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
            ],
            [(COL_STACK_READ_VALUE_HI[0], F::ONE)],
            [],
        );
        // Local-base memory rows bind the pointer through the locals RAM.
        b.push_row(
            [(GMEM_LOCAL, F::ONE)],
            [(COL_LOCAL_INDEX, F::ONE), (SLOT_ARG, -F::ONE)],
            [],
        );
        b.push_row(
            [(GMEM_LOCAL, F::ONE)],
            [
                (COL_LINEAR_MEM_LANE_ADDR[0], F::from_u64(4)),
                (COL_LINEAR_MEM_BYTE_OFFSET, F::ONE),
                (COL_LOCAL_VALUE, -F::ONE),
                (CONST_LO, -F::ONE),
            ],
            [],
        );
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

        // With event binding enabled, a clean halt transition loads
        // the owning turn's exit schedule and repoints gather attribution
        // from the last import to that export. This is independent of result
        // capture, so constant-only exit templates also cover resultless
        // exports.
        b.push_row(
            [(COL_EVENT_BINDING_ACTIVE_BEFORE, F::ONE)],
            [
                (COL_HALTED, F::ONE),
                (COL_HALTED_BEFORE, -F::ONE),
                (COL_TRAPPED_AFTER, -F::ONE),
                (COL_TRAPPED_BEFORE, F::ONE),
                (COL_TURN_BOUNDARY, F::ONE),
            ],
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
                (COL_HOST_CALLEE_FREF_AFTER, F::ONE),
                (COL_TURN_EXPORT_FREF_BEFORE, -F::ONE),
            ],
            [],
        );
    });
}

/// Position one-hot ↔ round-counter lockstep. The position columns are
/// gadget-internal, so their booleanity rows are pushed here (the
/// range-check pass only covers named columns).
fn push_position_onehot_constraints(b: &mut R1csBuilder) {
    b.with_tag(host_event("host event perm position"), |b| {
        push_zero_test_gadget(
            b,
            COL_PERM_ROUND_BEFORE,
            COL_PERM_ROUND_BEFORE_INV,
            COL_PERM_ROUND_BEFORE_IS_ZERO,
        );
        for pos in 0..COMM_CHAIN_PERM_ROWS {
            b.push_boolean(PERM_POSITION[pos]);
        }

        // sum(pos) = pending + (1 - round_is_zero): exactly one position on
        // perm rows, none elsewhere.
        b.push_linear_zero(
            (0..COMM_CHAIN_PERM_ROWS)
                .map(|pos| (PERM_POSITION[pos], F::ONE))
                .chain([
                    (COL_PERM_PENDING_BEFORE, -F::ONE),
                    (COL_ONE, -F::ONE),
                    (COL_PERM_ROUND_BEFORE_IS_ZERO, F::ONE),
                ]),
        );
        // sum(pos * P_pos) = round_before: the one-hot points at the counter.
        b.push_linear_zero(
            (0..COMM_CHAIN_PERM_ROWS)
                .map(|pos| (PERM_POSITION[pos], F::from_u64(pos as u64)))
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
                PERM_POSITION[COMM_CHAIN_PERM_ROWS - 1],
                F::from_u64(COMM_CHAIN_PERM_ROWS as u64),
            ),
        ]);
        // pending forces the absorb row now, and only pending rows absorb.
        b.push_row(
            [(COL_PERM_PENDING_BEFORE, F::ONE)],
            [(COL_ONE, F::ONE), (PERM_POSITION[0], -F::ONE)],
            [],
        );
        b.push_row(
            [(PERM_POSITION[0], F::ONE)],
            [(COL_PERM_PENDING_BEFORE, F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
    });
}

/// `perm_pending` is raised by the final gather row, cleared on the absorb
/// row, and preserved everywhere else.
fn push_pending_update_constraints(b: &mut R1csBuilder) {
    b.with_tag(host_event("host event pending update"), |b| {
        // Absorb row consumes the flag.
        b.push_row([(PERM_POSITION[0], F::ONE)], [(COL_PERM_PENDING_AFTER, F::ONE)], []);
        // Everything else preserves it; gather rows set it above.
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (PERM_POSITION[0], -F::ONE),
                (COL_GATHER_ACTIVE, -F::ONE),
            ],
            [(COL_PERM_PENDING_AFTER, F::ONE), (COL_PERM_PENDING_BEFORE, -F::ONE)],
            [],
        );
    });
}

/// Absorb-buffer writes: gather rows stage one ROM-described word, the
/// absorb row clears the buffer, and every untouched slot is carried.
fn push_buffer_write_constraints(b: &mut R1csBuilder) {
    b.with_tag(host_event("host event buffer write"), |b| {
        // The absorb row consumes the block: the buffer resets to zero so
        // the next block's unwritten slots are the zero padding.
        for j in 0..8 {
            b.push_row([(PERM_POSITION[0], F::ONE)], [(COL_EVBUF_AFTER[j], F::ONE)], []);
        }

        // Untouched slots carry: gate out the absorb reset and the gather
        // position that stages this word.
        for j in 0..8 {
            let gate = [
                (COL_ONE, F::ONE),
                (PERM_POSITION[0], -F::ONE),
                (GATHER_WORD_POSITION[j], -F::ONE),
            ];
            b.push_row(gate, [(COL_EVBUF_AFTER[j], F::ONE), (COL_EVBUF_BEFORE[j], -F::ONE)], []);
        }
    });
}

/// The absorb row's entry state is the premixed block input:
/// `state_before = M_ext · [chain_before | evbuf_before]`.
fn push_absorb_constraints(b: &mut R1csBuilder) {
    let me = external_matrix();
    b.with_tag(host_event("host event absorb"), |b| {
        for lane in 0..12 {
            let mut terms = vec![(COL_PERM_STATE_BEFORE[lane], F::ONE)];
            for (k, coeff) in me[lane].iter().enumerate() {
                let input = if k < 4 {
                    COL_COMM_CHAIN_BEFORE[k]
                } else {
                    COL_EVBUF_BEFORE[k - 4]
                };
                terms.push((input, -*coeff));
            }
            b.push_row([(PERM_POSITION[0], F::ONE)], terms, []);
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

    b.with_tag(host_event("host event perm full round"), |b| {
        for lane in 0..12 {
            // x = state_before[lane] + sum_pos P_pos * RC[pos][lane]
            let x_terms: Vec<(usize, F)> = core::iter::once((COL_PERM_STATE_BEFORE[lane], F::ONE))
                .chain(
                    full_positions
                        .iter()
                        .map(|&pos| (PERM_POSITION[pos], perm_full_round_constants(pos)[lane])),
                )
                .collect();
            let t = |i: usize| FULL_ROUND_POWERS[4 * lane + i];
            b.push_row(x_terms.clone(), x_terms.clone(), [(t(0), F::ONE)]);
            b.push_row([(t(0), F::ONE)], [(t(0), F::ONE)], [(t(1), F::ONE)]);
            b.push_row([(t(1), F::ONE)], [(t(0), F::ONE)], [(t(2), F::ONE)]);
            b.push_row([(t(2), F::ONE)], x_terms, [(t(3), F::ONE)]);
        }
        // Gated round output: state_after = M_ext · [t3 per lane].
        let gate: Vec<(usize, F)> = full_positions
            .iter()
            .map(|&pos| (PERM_POSITION[pos], F::ONE))
            .collect();
        for lane in 0..12 {
            let mut terms = vec![(COL_PERM_STATE_AFTER[lane], F::ONE)];
            for (k, coeff) in me[lane].iter().enumerate() {
                terms.push((FULL_ROUND_POWERS[4 * k + 3], -*coeff));
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
    let u = |i: usize| PARTIAL_ROUND_POWERS[i];

    b.with_tag(host_event("host event perm partial pair"), |b| {
        // Round a S-box input: x_a = SB_0 + selected RC.
        let x_a: Vec<(usize, F)> = core::iter::once((COL_PERM_STATE_BEFORE[0], F::ONE))
            .chain(
                partial_positions
                    .iter()
                    .map(|&pos| (PERM_POSITION[pos], perm_partial_round_constants(pos).0)),
            )
            .collect();
        b.push_row(x_a.clone(), x_a.clone(), [(u(0), F::ONE)]);
        b.push_row([(u(0), F::ONE)], [(u(0), F::ONE)], [(u(1), F::ONE)]);
        b.push_row([(u(1), F::ONE)], [(u(0), F::ONE)], [(u(2), F::ONE)]);
        b.push_row([(u(2), F::ONE)], x_a, [(u(3), F::ONE)]);

        // Round b S-box input: x_b = t'_0 + selected RC.
        let mut x_b: Vec<(usize, F)> = vec![(u(3), mi[0][0])];
        for j in 1..12 {
            x_b.push((COL_PERM_STATE_BEFORE[j], mi[0][j]));
        }
        x_b.extend(
            partial_positions
                .iter()
                .map(|&pos| (PERM_POSITION[pos], perm_partial_round_constants(pos).1)),
        );
        b.push_row(x_b.clone(), x_b.clone(), [(u(4), F::ONE)]);
        b.push_row([(u(4), F::ONE)], [(u(4), F::ONE)], [(u(5), F::ONE)]);
        b.push_row([(u(5), F::ONE)], [(u(4), F::ONE)], [(u(6), F::ONE)]);
        b.push_row([(u(6), F::ONE)], x_b, [(u(7), F::ONE)]);

        // Gated output: state_after = MI · [U7 | t'_1..11], with t' expanded
        // over [U3, SB_1..11].
        let gate: Vec<(usize, F)> = partial_positions
            .iter()
            .map(|&pos| (PERM_POSITION[pos], F::ONE))
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
                (COL_PERM_STATE_AFTER[lane], F::ONE),
                (u(7), -mi[lane][0]),
                (u(3), -coeff_u3),
            ];
            for (k, coeff) in coeff_sb.iter().enumerate().skip(1) {
                terms.push((COL_PERM_STATE_BEFORE[k], -*coeff));
            }
            b.push_row(gate.clone(), terms, []);
        }
    });
}

/// Chain movement: only the group's last row updates `comm_chain`, adding
/// the raw input lanes (feed-forward) to the permutation output; every
/// other row in the trace carries the chain unchanged.
fn push_chain_update_constraints(b: &mut R1csBuilder) {
    let last = PERM_POSITION[COMM_CHAIN_PERM_ROWS - 1];
    b.with_tag(host_event("host event chain update"), |b| {
        for limb in 0..4 {
            b.push_row(
                [(last, F::ONE)],
                [
                    (COL_COMM_CHAIN_AFTER[limb], F::ONE),
                    (COL_PERM_STATE_AFTER[limb], -F::ONE),
                    (COL_COMM_CHAIN_BEFORE[limb], -F::ONE),
                ],
                [],
            );
            b.push_row(
                [(COL_ONE, F::ONE), (last, -F::ONE)],
                [
                    (COL_COMM_CHAIN_AFTER[limb], F::ONE),
                    (COL_COMM_CHAIN_BEFORE[limb], -F::ONE),
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
                    (COL_PERM_STATE_AFTER[lane], F::ONE),
                    (COL_PERM_STATE_BEFORE[lane], -F::ONE),
                ],
                [],
            );
        }
    });
}

/// Perm rows are aux rows with no stack traffic (pc/param-init handling
/// lives with the other aux-row shape rows in `ccs/call.rs`).
fn push_perm_row_shape_constraints(b: &mut R1csBuilder) {
    b.with_tag(host_event("host event perm row shape"), |b| {
        b.push_row(perm_row_gate_terms(), [(COL_STACK_READS, F::ONE)], []);
        b.push_row(perm_row_gate_terms(), [(COL_STACK_WRITES, F::ONE)], []);
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
        wit[PERM_POSITION[pos]] = F::ONE;
    }

    // Full-round S-box powers: x = state_before[lane] + selected RC.
    for lane in 0..12 {
        let rc = pos
            .filter(|&p| perm_row_is_full_round(p))
            .map(|p| perm_full_round_constants(p)[lane])
            .unwrap_or(F::ZERO);
        let x = sb[lane] + rc;
        let t = |power| FULL_ROUND_POWERS[4 * lane + power];
        wit[t(0)] = x * x;
        wit[t(1)] = wit[t(0)] * wit[t(0)];
        wit[t(2)] = wit[t(1)] * wit[t(0)];
        wit[t(3)] = wit[t(2)] * x;
    }

    // Partial-pair S-box powers: round a on lane 0, internal mix, round b.
    let (rc_a, rc_b) = pos
        .filter(|&p| !perm_row_is_full_round(p))
        .map(perm_partial_round_constants)
        .unwrap_or((F::ZERO, F::ZERO));
    let x_a = sb[0] + rc_a;
    wit[PARTIAL_ROUND_POWERS[0]] = x_a * x_a;
    wit[PARTIAL_ROUND_POWERS[1]] = wit[PARTIAL_ROUND_POWERS[0]] * wit[PARTIAL_ROUND_POWERS[0]];
    wit[PARTIAL_ROUND_POWERS[2]] = wit[PARTIAL_ROUND_POWERS[1]] * wit[PARTIAL_ROUND_POWERS[0]];
    wit[PARTIAL_ROUND_POWERS[3]] = wit[PARTIAL_ROUND_POWERS[2]] * x_a;
    let mut mixed = sb;
    mixed[0] = wit[PARTIAL_ROUND_POWERS[3]];
    perm_internal_linear(&mut mixed);
    let x_b = mixed[0] + rc_b;
    wit[PARTIAL_ROUND_POWERS[4]] = x_b * x_b;
    wit[PARTIAL_ROUND_POWERS[5]] = wit[PARTIAL_ROUND_POWERS[4]] * wit[PARTIAL_ROUND_POWERS[4]];
    wit[PARTIAL_ROUND_POWERS[6]] = wit[PARTIAL_ROUND_POWERS[5]] * wit[PARTIAL_ROUND_POWERS[4]];
    wit[PARTIAL_ROUND_POWERS[7]] = wit[PARTIAL_ROUND_POWERS[6]] * x_b;

    // Grammar gather one-hots and staged value.
    if trace.row_kind.is_host_event_gather() {
        let cursor = usize::from(trace.state_before.grammar.slot_cursor);
        wit[GATHER_WORD_POSITION[cursor]] = F::ONE;
        wit[GSLOT_VALUE] = F::from_u64(trace.state_after.event_absorb.evbuf[cursor]);
        if let Some(rom) = trace.grammar_rom_slot {
            wit[GATHER_KIND[rom.kind.index()]] = F::ONE;
            wit[GK2_HI] = bool_f(rom.kind == WasmGrammarSlotKind::Result && rom.variant.is_high_limb());
            wit[G_ADVICE] = bool_f(rom.advice);
            wit[GMEM_LOCAL] = bool_f(rom.variant.uses_local_memory_base());
            wit[GMEM_BYTE] = bool_f(rom.variant.uses_byte_memory_width());
            wit[GMEM_HALF] = bool_f(rom.variant.uses_half_memory_width());
        }
    }
    // Host-call arg pops: GHC · ROM-bound param count.
    wit[GHC_PARAMS] = wit[super::super::layout::COL_HOST_CALL_ACTIVE] * wit[super::super::layout::COL_CALL_PARAM_COUNT];
    // Limb-selected values: filled on every row so the unconditional select
    // rows hold (the limb column is zero off gather rows).
    let read_lo = wit[super::super::layout::COL_STACK_READ_VALUE_LO[0]];
    let read_hi = wit[COL_STACK_READ_VALUE_HI[0]];
    let variant = wit[super::super::layout::COL_GRAMMAR_SLOT_VARIANT];
    wit[GARG_VAL] = read_lo + variant * (read_hi - read_lo);
    let out_lo = wit[super::super::layout::COL_OUTPUT_VALUE_LO_BEFORE];
    let out_hi = wit[super::super::layout::COL_OUTPUT_VALUE_HI_BEFORE];
    wit[GOUT_VAL] = out_lo + variant * (out_hi - out_lo);
}
