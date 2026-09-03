//! Wasm host-event extraction and diagnostic replay for the shared event
//! commitment protocol.
//!
//! The exact Poseidon2 parameters and compression live in `neo-application`;
//! this module owns only wasm's event-block interpretation and trace checks.

use crate::ir::{WasmBuildError, WasmVmStep};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use neo_application::event_commitment::{self, EVENT_COMMITMENT_BLOCK_WIDTH, EVENT_COMMITMENT_STATE_WIDTH};
use neo_application::poseidon2::{
    self, POSEIDON2_GROUPED_ROUNDS, POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_PARTIAL_PAIRS, POSEIDON2_WIDTH,
};

/// Field elements carried as the chain state (and emitted as the digest).
pub const COMM_CHAIN_STATE_LEN: usize = EVENT_COMMITMENT_STATE_WIDTH;
/// Fixed argument slots absorbed per event, after the discriminant.
pub const COMM_CHAIN_EVENT_ARGS: usize = EVENT_COMMITMENT_BLOCK_WIDTH - 1;

/// Initial state of the host-event commitment chain.
pub use neo_application::event_commitment::EventCommitmentState as CommChainState;

/// Fold event blocks from the supplied initial commitment state.
pub fn fold_event_blocks(
    initial_state: CommChainState,
    blocks: &[[Goldilocks; COMM_CHAIN_BLOCK_WORDS]],
) -> CommChainState {
    event_commitment::fold_blocks(initial_state, blocks)
}

/// Absorb one host event into the chain: `H([prev | discriminant | args])`.
pub fn commit_event(
    prev: [Goldilocks; COMM_CHAIN_STATE_LEN],
    discriminant: Goldilocks,
    args: [Goldilocks; COMM_CHAIN_EVENT_ARGS],
) -> [Goldilocks; COMM_CHAIN_STATE_LEN] {
    let mut block = [Goldilocks::ZERO; COMM_CHAIN_BLOCK_WORDS];
    block[0] = discriminant;
    block[1..].copy_from_slice(&args);
    event_commitment::commit_block(prev, block)
}

/// Words absorbed per chain block (discriminant slot + arg slots).
pub const COMM_CHAIN_BLOCK_WORDS: usize = EVENT_COMMITMENT_BLOCK_WIDTH;

/// Trace attribution accompanying an absorbed block.
///
/// In a valid wasm trace these fields are circuit-constrained, but they are
/// not words in the event-chain commitment. Consumers must not authenticate
/// them from the commitment alone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AbsorbedEventMetadata {
    /// Import or export template that supplied the event.
    pub attributed_fref: u32,
    /// Export fref owning the turn that emitted the event.
    pub turn_export_fref: u32,
}

/// One committed event block, extracted from the gather row that
/// completed it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AbsorbedEventBlock {
    /// Exact words absorbed by the event chain. Their interpretation belongs
    /// to the embedder bindings; position zero is not necessarily a discriminant.
    pub words: [u64; COMM_CHAIN_BLOCK_WORDS],
    /// Trace-derived attribution, excluded from the event-chain commitment.
    pub metadata: AbsorbedEventMetadata,
}

/// Committed event blocks of a trace, in absorb order: the
/// exact stream the carried chain folds, for external consumers rebuilding
/// the event sequence (e.g. the Starstream interleaving buffer). A block
/// commits on the gather row that raises `perm_pending`; advice events stage
/// `evbuf` without raising it and are excluded by construction. This extracts
/// witness data; it does not validate the supplied trace.
pub fn absorbed_event_blocks(trace: &[WasmVmStep]) -> Vec<AbsorbedEventBlock> {
    trace
        .iter()
        .filter(|row| {
            row.row_kind.is_host_event_gather()
                && row.state_after.event_absorb.perm_pending
                && !row.state_before.event_absorb.perm_pending
        })
        .map(|row| AbsorbedEventBlock {
            words: row.state_after.event_absorb.evbuf,
            metadata: AbsorbedEventMetadata {
                attributed_fref: row.state_before.host_callee_fref,
                turn_export_fref: row.state_before.host_events.turn_export_fref,
            },
        })
        .collect()
}

/// Circuit rows per absorbed block: 4 initial full rounds, 11 partial-pair
/// rows (2 internal rounds each), 4 terminal full rounds.
pub const COMM_CHAIN_PERM_ROWS: usize = POSEIDON2_GROUPED_ROUNDS;
/// Row positions `0..PERM_PARTIAL_FIRST_ROW` are the initial full rounds.
pub const PERM_PARTIAL_FIRST_ROW: usize = POSEIDON2_HALF_FULL_ROUNDS;
/// Row positions `PERM_TERMINAL_FIRST_ROW..COMM_CHAIN_PERM_ROWS` are the
/// terminal full rounds.
pub const PERM_TERMINAL_FIRST_ROW: usize = POSEIDON2_HALF_FULL_ROUNDS + POSEIDON2_PARTIAL_PAIRS;

/// Is circuit row position `pos` a full (external) round row?
fn perm_row_is_full_round(pos: usize) -> bool {
    pos < PERM_PARTIAL_FIRST_ROW || pos >= PERM_TERMINAL_FIRST_ROW
}

/// The external (`mds_light`) linear layer of the chain permutation.
pub fn perm_external_linear(state: &mut [Goldilocks; POSEIDON2_WIDTH]) {
    poseidon2::external_linear(state);
}

/// Apply the circuit row at position `pos` to a permutation state.
pub fn perm_row_transition(pos: usize, state: &mut [Goldilocks; POSEIDON2_WIDTH]) {
    if perm_row_is_full_round(pos) {
        let round = if pos < PERM_PARTIAL_FIRST_ROW {
            pos
        } else {
            pos - POSEIDON2_PARTIAL_PAIRS
        };
        poseidon2::apply_full_round(round, state);
    } else {
        poseidon2::apply_partial_pair(pos - PERM_PARTIAL_FIRST_ROW, state);
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
) -> [[Goldilocks; POSEIDON2_WIDTH]; COMM_CHAIN_PERM_ROWS + 1] {
    let mut state = [Goldilocks::ZERO; POSEIDON2_WIDTH];
    state[..COMM_CHAIN_STATE_LEN].copy_from_slice(&prev);
    state[COMM_CHAIN_STATE_LEN..].copy_from_slice(&words);
    poseidon2::apply_initial_linear(&mut state);

    let mut checkpoints = [[Goldilocks::ZERO; POSEIDON2_WIDTH]; COMM_CHAIN_PERM_ROWS + 1];
    for (position, checkpoint) in checkpoints[..COMM_CHAIN_PERM_ROWS].iter_mut().enumerate() {
        *checkpoint = state;
        perm_row_transition(position, &mut state);
    }
    checkpoints[COMM_CHAIN_PERM_ROWS] = state;
    checkpoints
}

/// Recompute the host-event commitment chain from gathered event blocks and
/// validate every carried `comm_chain` state against it.
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

        // Each event block is staged by 8 gather rows; the one
        // that completes the block raises `perm_pending`, and the chain must
        // fold exactly the staged blocks in order. The binding of block
        // contents to the host-event tables is checked against the host-event ROM
        // (see `memory_semantics::preload_host_event_tables`), not here.
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
