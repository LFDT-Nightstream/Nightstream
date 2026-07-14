//! Host-event commitment chain: the hash that binds host-call events into an
//! incrementally carried digest, shared bit-for-bit with external interaction
//! verifiers (the Starstream interleaving proof's `LedgerEffectsCommitment`).
//!
//! Owns the chain-update permutation and its protocol constants. Does not own
//! the event grammar (which host import maps to which discriminant/arg slots)
//! or the circuit gadget enforcing the update in CCS rows.
//!
//! Protocol constants (must match `starstream-interleaving-proof`):
//! - Poseidon2 over Goldilocks, width 12, S-box x^7, 4+4 full / 22 partial
//!   rounds, as instantiated by p3-goldilocks 0.5.3
//!   `default_goldilocks_poseidon2_12()` (Grain LFSR round constants:
//!   field_type=1, alpha=7, n=64, t=12, R_F=8, R_P=22).
//! - Chain update = compression: permute `[prev_4 | discriminant | args_7]`,
//!   truncate to 4 lanes, feed-forward add the matching input lanes.

use crate::ir::{WasmBuildError, WasmVmStep};
use crate::isa::WasmOpcode;
use once_cell::sync::Lazy;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{
    default_goldilocks_poseidon2_12, Goldilocks, Poseidon2Goldilocks, GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL,
    GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL, GOLDILOCKS_POSEIDON2_RC_12_INTERNAL, MATRIX_DIAG_12_GOLDILOCKS,
};
use p3_poseidon2::{matmul_internal, mds_light_permutation, MDSMat4};
use p3_symmetric::Permutation;

/// Field elements carried as the chain state (and emitted as the digest).
pub const COMM_CHAIN_STATE_LEN: usize = 4;
/// Fixed argument slots absorbed per event, after the discriminant.
pub const COMM_CHAIN_EVENT_ARGS: usize = 7;

static PERM12: Lazy<Poseidon2Goldilocks<12>> = Lazy::new(default_goldilocks_poseidon2_12);

/// Fold a claimed event transcript (absorb blocks in emission order) from a
/// zero chain. The verifier-side half of transcript binding: a proof's
/// final carried `comm_chain` (authenticated by [`crate::verify`] through
/// the final semantic digest) equals this fold iff the execution absorbed
/// exactly these blocks — entry inputs, import events, and exit events
/// alike, at any arity.
pub fn fold_event_blocks(blocks: &[[Goldilocks; COMM_CHAIN_BLOCK_WORDS]]) -> [Goldilocks; 4] {
    let mut chain = [Goldilocks::ZERO; 4];
    for block in blocks {
        chain = commit_event(chain, block[0], core::array::from_fn(|i| block[1 + i]));
    }
    chain
}

/// Absorb one host event into the chain: `H([prev | discriminant | args])`.
pub fn commit_event(
    prev: [Goldilocks; COMM_CHAIN_STATE_LEN],
    discriminant: Goldilocks,
    args: [Goldilocks; COMM_CHAIN_EVENT_ARGS],
) -> [Goldilocks; COMM_CHAIN_STATE_LEN] {
    let mut state = [Goldilocks::ZERO; 12];
    state[..COMM_CHAIN_STATE_LEN].copy_from_slice(&prev);
    state[COMM_CHAIN_STATE_LEN] = discriminant;
    state[COMM_CHAIN_STATE_LEN + 1..].copy_from_slice(&args);

    let permuted = PERM12.permute(state);
    core::array::from_fn(|i| permuted[i] + state[i])
}

/// Domain tag opening every raw host-call event.
pub const HOST_CALL_EVENT_TAG: u64 = 1;

/// Words absorbed per chain block (discriminant slot + arg slots).
pub const COMM_CHAIN_BLOCK_WORDS: usize = 1 + COMM_CHAIN_EVENT_ARGS;

/// Serialize one raw host-call event into its absorb stream: the canonical,
/// embedder-agnostic record of "this import was called with these arguments
/// and returned this result".
///
/// ```text
/// [HOST_CALL_EVENT_TAG, callee_fref, param_count, result_count,
///  arg{n-1}_lo, arg{n-1}_hi, ..., arg0_lo, arg0_hi,
///  result_lo, result_hi]              // present iff result_count = 1
/// ```
///
/// Args are two 32-bit limbs each regardless of wasm type, in operand-stack
/// pop order (last declared parameter first) — the order the trace's
/// `HostCallArg` rows stream them into the in-circuit absorb buffer. The
/// declared order is recoverable since `param_count` is part of the header,
/// and the block count is a static function of the callee's arity.
pub fn host_call_event_stream(
    callee_fref: u32,
    param_count: u8,
    result_count: u8,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
) -> Vec<Goldilocks> {
    assert_eq!(args.len(), usize::from(param_count), "arg limbs must match param_count");
    assert_eq!(
        result.is_some(),
        result_count == 1,
        "result limbs must match result_count"
    );

    let f = Goldilocks::from_u64;
    let mut stream = Vec::with_capacity(4 + 2 * args.len() + 2);
    stream.extend([
        f(HOST_CALL_EVENT_TAG),
        f(u64::from(callee_fref)),
        f(u64::from(param_count)),
        f(u64::from(result_count)),
    ]);
    for &(lo, hi) in args.iter().rev() {
        stream.push(f(u64::from(lo)));
        stream.push(f(u64::from(hi)));
    }
    if let Some((lo, hi)) = result {
        stream.push(f(u64::from(lo)));
        stream.push(f(u64::from(hi)));
    }
    stream
}

/// Absorb one raw host-call event (see [`host_call_event_stream`] for the
/// serialization): the stream is absorbed in blocks of
/// [`COMM_CHAIN_BLOCK_WORDS`] words via [`commit_event`], zero-padding the
/// final block — unambiguous because the two counts fix the stream length.
///
/// `args` are in declared parameter order; the stream absorbs them in pop
/// order internally.
pub fn commit_host_call_event(
    prev: [Goldilocks; COMM_CHAIN_STATE_LEN],
    callee_fref: u32,
    param_count: u8,
    result_count: u8,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
) -> [Goldilocks; COMM_CHAIN_STATE_LEN] {
    let stream = host_call_event_stream(callee_fref, param_count, result_count, args, result);
    let mut state = prev;
    for block in stream.chunks(COMM_CHAIN_BLOCK_WORDS) {
        let mut words = [Goldilocks::ZERO; COMM_CHAIN_BLOCK_WORDS];
        words[..block.len()].copy_from_slice(block);
        state = commit_event(state, words[0], words[1..].try_into().unwrap());
    }
    state
}

/// [`commit_host_call_event`] over canonical-u64 chain limbs, as carried in
/// [`crate::ir::WasmStepState::comm_chain`].
pub fn commit_host_call_event_u64(
    prev: [u64; COMM_CHAIN_STATE_LEN],
    callee_fref: u32,
    param_count: u8,
    result_count: u8,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
) -> [u64; COMM_CHAIN_STATE_LEN] {
    commit_host_call_event(
        prev.map(Goldilocks::from_u64),
        callee_fref,
        param_count,
        result_count,
        args,
        result,
    )
    .map(|limb| limb.as_canonical_u64())
}

/// Circuit rows per absorbed block: 4 initial full rounds, 11 partial-pair
/// rows (2 internal rounds each), 4 terminal full rounds.
pub const COMM_CHAIN_PERM_ROWS: usize = 19;
/// Row positions `0..PERM_PARTIAL_FIRST_ROW` are the initial full rounds.
pub const PERM_PARTIAL_FIRST_ROW: usize = 4;
/// Row positions `PERM_TERMINAL_FIRST_ROW..COMM_CHAIN_PERM_ROWS` are the
/// terminal full rounds.
pub const PERM_TERMINAL_FIRST_ROW: usize = 15;

/// Is circuit row position `pos` a full (external) round row?
pub fn perm_row_is_full_round(pos: usize) -> bool {
    pos < PERM_PARTIAL_FIRST_ROW || pos >= PERM_TERMINAL_FIRST_ROW
}

/// External round constants for full-round row position `pos`.
pub fn perm_full_round_constants(pos: usize) -> &'static [Goldilocks; 12] {
    if pos < PERM_PARTIAL_FIRST_ROW {
        &GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL[pos]
    } else {
        &GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL[pos - PERM_TERMINAL_FIRST_ROW]
    }
}

/// Internal round constants `(first, second)` for partial-pair row position `pos`.
pub fn perm_partial_round_constants(pos: usize) -> (Goldilocks, Goldilocks) {
    let pair = pos - PERM_PARTIAL_FIRST_ROW;
    (
        GOLDILOCKS_POSEIDON2_RC_12_INTERNAL[2 * pair],
        GOLDILOCKS_POSEIDON2_RC_12_INTERNAL[2 * pair + 1],
    )
}

/// The external (`mds_light`) linear layer of the chain permutation.
pub fn perm_external_linear(state: &mut [Goldilocks; 12]) {
    mds_light_permutation(state, &MDSMat4);
}

/// The internal (`1 + diag(v)`) linear layer of the chain permutation.
pub fn perm_internal_linear(state: &mut [Goldilocks; 12]) {
    matmul_internal(state, MATRIX_DIAG_12_GOLDILOCKS);
}

fn sbox(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x2 * x
}

/// Apply the circuit row at position `pos` to a permutation state.
pub fn perm_row_transition(pos: usize, state: &mut [Goldilocks; 12]) {
    if perm_row_is_full_round(pos) {
        let rc = perm_full_round_constants(pos);
        for (lane, rc) in state.iter_mut().zip(rc) {
            *lane = sbox(*lane + *rc);
        }
        perm_external_linear(state);
    } else {
        let (rc_a, rc_b) = perm_partial_round_constants(pos);
        for rc in [rc_a, rc_b] {
            state[0] = sbox(state[0] + rc);
            perm_internal_linear(state);
        }
    }
}

/// Row-level checkpoints of one absorbed block: `checkpoints[pos]` is the
/// permutation state entering circuit row `pos`, and
/// `checkpoints[COMM_CHAIN_PERM_ROWS]` is the permutation output.
///
/// `checkpoints[0]` is the *pre-mixed* absorb state — the initial external
/// linear layer applied to `[prev | words]` — so every full-round row shares
/// the same add-RC/S-box/linear shape. The chain update is
/// `checkpoints[19][0..4] + prev` (feed-forward over the raw input lanes),
/// exactly [`commit_event`]; a test pins that equality.
pub fn perm_row_checkpoints(
    prev: [Goldilocks; COMM_CHAIN_STATE_LEN],
    words: [Goldilocks; COMM_CHAIN_BLOCK_WORDS],
) -> [[Goldilocks; 12]; COMM_CHAIN_PERM_ROWS + 1] {
    let mut state = [Goldilocks::ZERO; 12];
    state[..COMM_CHAIN_STATE_LEN].copy_from_slice(&prev);
    state[COMM_CHAIN_STATE_LEN..].copy_from_slice(&words);
    perm_external_linear(&mut state);

    let mut checkpoints = [[Goldilocks::ZERO; 12]; COMM_CHAIN_PERM_ROWS + 1];
    for pos in 0..COMM_CHAIN_PERM_ROWS {
        checkpoints[pos] = state;
        perm_row_transition(pos, &mut state);
    }
    checkpoints[COMM_CHAIN_PERM_ROWS] = state;
    checkpoints
}

/// Recompute the host-event commitment chain from the trace's host-call
/// events and validate every carried `comm_chain` state against it.
///
/// Debug-side stand-in mirroring how `memory_semantics` stands in for the
/// committed memory argument: events are reconstructed from the call/arg/
/// result rows' own data, re-serialized, chunked into blocks, and each
/// perm-group is checked to absorb the right block words and land the chain
/// on the right value (on the group's last row; every other row must carry
/// the chain unchanged).
pub fn sanity_check_comm_chain(trace: &[WasmVmStep]) -> Result<(), WasmBuildError> {
    use std::collections::VecDeque;

    let err = |msg: String| Err(WasmBuildError::StateMismatch(msg));
    let mut expected = match trace.first() {
        Some(row) => row.state_before.comm_chain,
        None => return Ok(()),
    };
    // Blocks owed by events already seen but not yet absorbed, in absorb
    // order: the block's words and the chain value its perm group lands on.
    let mut owed_blocks: VecDeque<([u64; COMM_CHAIN_BLOCK_WORDS], [u64; COMM_CHAIN_STATE_LEN])> = VecDeque::new();
    let mut owed_chain = expected;

    for (i, row) in trace.iter().enumerate() {
        if row.state_before.comm_chain != expected {
            return err(format!(
                "row {i}: comm_chain before {:?} does not match expected {:?}",
                row.state_before.comm_chain, expected
            ));
        }

        // Grammar mode: each event block is staged by 8 gather rows; the one
        // that completes the block raises `perm_pending`, and the chain must
        // fold exactly the staged blocks in order. The binding of block
        // contents to the grammar tables is checked against the grammar ROM
        // (see `memory_semantics::preload_grammar_tables`), not here.
        if row.row_kind.is_host_event_gather()
            && row.state_after.event_absorb.perm_pending
            && !row.state_before.event_absorb.perm_pending
        {
            let words = row.state_after.event_absorb.evbuf;
            let updated = commit_event(
                owed_chain.map(Goldilocks::from_u64),
                Goldilocks::from_u64(words[0]),
                core::array::from_fn(|i| Goldilocks::from_u64(words[1 + i])),
            );
            owed_chain = updated.map(|limb| limb.as_canonical_u64());
            owed_blocks.push_back((words, owed_chain));
        }

        let is_host_call = !row.state_before.grammar_mode
            && row.row_kind.is_program()
            && matches!(row.opcode, WasmOpcode::Call | WasmOpcode::CallIndirect)
            && !row.target_function_is_guest
            && !row.state_after.trapped;
        if is_host_call {
            // Reconstruct the event from its own rows: arg rows stream pop
            // order (which is also serialization order), then the result.
            let param_count = row.state_after.host_args.remaining;
            let result_count = row.state_after.host_result_pending;
            let mut pop_args: Vec<(u32, u32)> = Vec::new();
            let mut result: Option<(u32, u32)> = None;
            for later in &trace[i + 1..] {
                if later.row_kind.is_host_call_arg() {
                    let read = later.stack_read0.ok_or_else(|| {
                        WasmBuildError::StateMismatch(format!("row {i}: host arg row without a stack read"))
                    })?;
                    pop_args.push((read.value_lo, read.value_hi.unwrap_or(0)));
                } else if later.row_kind.is_host_call_result() {
                    let write = later.stack_write0.ok_or_else(|| {
                        WasmBuildError::StateMismatch(format!("row {i}: host result row without a stack write"))
                    })?;
                    result = Some((write.value_lo, write.value_hi.unwrap_or(0)));
                    break;
                } else if !later.row_kind.is_host_event_perm() {
                    break;
                }
                if pop_args.len() == param_count as usize && !result_count {
                    break;
                }
            }
            if pop_args.len() != param_count as usize || result.is_some() != result_count {
                return err(format!("row {i}: host-call event rows do not match its declared arity"));
            }
            let declared_args: Vec<(u32, u32)> = pop_args.iter().rev().copied().collect();
            let stream = host_call_event_stream(
                row.state_after.host_callee_fref,
                u8::try_from(param_count)
                    .map_err(|_| WasmBuildError::StateMismatch(format!("row {i}: host call arity exceeds u8")))?,
                u8::from(result_count),
                &declared_args,
                result,
            );
            for chunk in stream.chunks(COMM_CHAIN_BLOCK_WORDS) {
                let mut words = [Goldilocks::ZERO; COMM_CHAIN_BLOCK_WORDS];
                words[..chunk.len()].copy_from_slice(chunk);
                let updated = commit_event(
                    owed_chain.map(Goldilocks::from_u64),
                    words[0],
                    words[1..].try_into().unwrap(),
                );
                owed_chain = updated.map(|limb| limb.as_canonical_u64());
                owed_blocks.push_back((words.map(|w| w.as_canonical_u64()), owed_chain));
            }
        }

        let mut want_after = expected;
        if row.row_kind.is_host_event_perm() {
            let round = row.state_before.event_absorb.perm_round;
            if round == 0 {
                let Some(&(words, _)) = owed_blocks.front() else {
                    return err(format!("row {i}: perm group without an owed event block"));
                };
                if row.state_before.event_absorb.evbuf != words {
                    return err(format!(
                        "row {i}: perm group absorbs buffer {:?} but the event owes block {:?}",
                        row.state_before.event_absorb.evbuf, words
                    ));
                }
            }
            if usize::from(round) + 1 == COMM_CHAIN_PERM_ROWS {
                let Some((_, updated)) = owed_blocks.pop_front() else {
                    return err(format!("row {i}: perm group tail without an owed event block"));
                };
                want_after = updated;
            }
        }
        if row.state_after.comm_chain != want_after {
            return err(format!(
                "row {i}: comm_chain after {:?} does not match recomputed chain {:?}",
                row.state_after.comm_chain, want_after
            ));
        }
        expected = want_after;
    }
    if !owed_blocks.is_empty() {
        return err(format!(
            "trace ended with {} unabsorbed event blocks",
            owed_blocks.len()
        ));
    }
    Ok(())
}
