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
    COL_EVBUF0_BEFORE, COL_EVBUF_SLOT0_AFTER, COL_EVBUF_SLOT0_BEFORE, COL_FUNCTION_REF, COL_HOST_ARGS_ACTIVE_BEFORE,
    COL_HOST_ARGS_REMAINING_AFTER, COL_HOST_ARGS_REMAINING_AFTER_IS_ZERO, COL_HOST_ARGS_REMAINING_BEFORE,
    COL_HOST_RESULT_ACTIVE, COL_HOST_RESULT_PENDING_AFTER, COL_HOST_RESULT_PENDING_BEFORE, COL_ONE,
    COL_PERM_PENDING_AFTER, COL_PERM_PENDING_BEFORE, COL_PERM_ROUND_AFTER, COL_PERM_ROUND_BEFORE,
    COL_PERM_ROUND_BEFORE_INV, COL_PERM_ROUND_BEFORE_IS_ZERO, COL_PERM_STATE0_AFTER, COL_PERM_STATE0_BEFORE,
    COL_STACK_READ0_VALUE_HI, COL_STACK_READ0_VALUE_LO, COL_STACK_READS, COL_STACK_WRITE0_VALUE_HI,
    COL_STACK_WRITE0_VALUE_LO, COL_STACK_WRITES, NAMED_COLUMN_COUNT,
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
const POS0: usize = NAMED_COLUMN_COUNT; // 19 position one-hot flags
const FULL_T0: usize = POS0 + COMM_CHAIN_PERM_ROWS; // 48 full-round S-box powers
const PARTIAL_U0: usize = FULL_T0 + 48; // 8 partial-pair S-box powers
const WSA0: usize = PARTIAL_U0 + 8; // 4 arg-row write masks
const WSR0: usize = WSA0 + 4; // 4 result-row write masks
const STREAM_DONE: usize = WSR0 + 4; // event stream complete after this row
const EVENT_END: usize = STREAM_DONE + 1; // this row streams the final word pair
const EVENT_END_OR: usize = EVENT_END + 1; // (block filled)·(event end) product

/// Width of the gadget-internal column block.
pub const PERM_GADGET_AUX_WIDTH: usize = EVENT_END_OR + 1 - NAMED_COLUMN_COUNT;

/// Declared bit-widths of the gadget-internal columns, in block order (for
/// the F' norm decomposition): booleans for the one-hot/masks/products,
/// full field elements for the S-box powers.
pub(crate) fn perm_gadget_col_widths() -> impl Iterator<Item = usize> {
    core::iter::repeat_n(1, COMM_CHAIN_PERM_ROWS)
        .chain(core::iter::repeat_n(64, 48 + 8))
        .chain(core::iter::repeat_n(1, 4 + 4 + 3))
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

/// Gate terms that are 1 exactly on the rows streaming event words into the
/// absorb buffer: host-call program rows, host-arg rows, host-result rows.
fn event_write_gate_terms() -> [(usize, F); 5] {
    let [call, ci_not_trap, guest] = host_call_gate_terms();
    [
        call,
        ci_not_trap,
        guest,
        (COL_HOST_ARGS_ACTIVE_BEFORE, F::ONE),
        (COL_HOST_RESULT_ACTIVE, F::ONE),
    ]
}

pub(super) fn push_host_event_perm_constraints(b: &mut R1csBuilder) {
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
        // Everything else preserves it.
        let mut preserve_gate = vec![(COL_ONE, F::ONE), (POS0, -F::ONE)];
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
        // WSA_k / WSR_k: slot-cursor one-hot masked by the writing row kind.
        for k in 0..4 {
            b.push_row(
                [(COL_EVBUF_SLOT0_BEFORE + k, F::ONE)],
                [(COL_HOST_ARGS_ACTIVE_BEFORE, F::ONE)],
                [(WSA0 + k, F::ONE)],
            );
            b.push_row(
                [(COL_EVBUF_SLOT0_BEFORE + k, F::ONE)],
                [(COL_HOST_RESULT_ACTIVE, F::ONE)],
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
            b.push_row(host_call_gate_terms(), terms, []);
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

        // Untouched slots carry: gate out the call row (rewrites all 8),
        // this pair's write flags, and the absorb reset.
        for j in 0..8 {
            let pair = j / 2;
            let mut gate = vec![
                (COL_ONE, F::ONE),
                (WSA0 + pair, -F::ONE),
                (WSR0 + pair, -F::ONE),
                (POS0, -F::ONE),
            ];
            gate.extend(host_call_gate_terms().map(|(col, coeff)| (col, -coeff)));
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
            b.push_row(host_call_gate_terms(), call_target, []);

            b.push_row(
                [(COL_HOST_ARGS_ACTIVE_BEFORE, F::ONE), (COL_HOST_RESULT_ACTIVE, F::ONE)],
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

            let mut gate = vec![
                (COL_ONE, F::ONE),
                (COL_HOST_ARGS_ACTIVE_BEFORE, -F::ONE),
                (COL_HOST_RESULT_ACTIVE, -F::ONE),
                (POS0, -F::ONE),
            ];
            gate.extend(host_call_gate_terms().map(|(col, coeff)| (col, -coeff)));
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
pub(crate) fn fill_perm_gadget_witness(wit: &mut [F], trace: &WasmVmStep) {
    use super::super::layout::selector_col;

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

    // Buffer-write masks and pending-update products, from the named flags.
    let args_active = wit[COL_HOST_ARGS_ACTIVE_BEFORE];
    let result_active = wit[COL_HOST_RESULT_ACTIVE];
    for k in 0..4 {
        wit[WSA0 + k] = wit[COL_EVBUF_SLOT0_BEFORE + k] * args_active;
        wit[WSR0 + k] = wit[COL_EVBUF_SLOT0_BEFORE + k] * result_active;
    }
    let stream_done = bool_f(trace.state_after.host_args.remaining == 0 && !trace.state_after.host_result_pending);
    wit[STREAM_DONE] = stream_done;
    // write row = host-call program row + arg row + result row (each {0,1},
    // mutually exclusive), mirroring `event_write_gate_terms`.
    let write_row = wit[selector_col(crate::isa::WasmOpcode::Call).expect("call selector")]
        + wit[super::super::layout::COL_CALL_INDIRECT_IS_NOT_TRAP]
        - wit[super::super::layout::COL_GUEST_CALL_ACTIVE]
        + args_active
        + result_active;
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
}
