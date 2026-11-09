//! Wasm host-event extraction and the block absorb of the shared event
//! commitment protocol.
//!
//! The exact Poseidon2 parameters and compression live in `neo-application`;
//! this module owns only wasm's event-block interpretation and row transitions.

use crate::ir::WasmVmStep;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use neo_application::event_commitment::{self, EVENT_COMMITMENT_BLOCK_WIDTH, EVENT_COMMITMENT_STATE_WIDTH};

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

/// Fold one filled event buffer into the chain (canonical u64 limbs): the
/// absorb that the gather row filling a block performs.
pub fn absorb_block(
    chain: [u64; COMM_CHAIN_STATE_LEN],
    evbuf: [u64; COMM_CHAIN_BLOCK_WORDS],
) -> [u64; COMM_CHAIN_STATE_LEN] {
    event_commitment::commit_block(chain.map(Goldilocks::from_u64), evbuf.map(Goldilocks::from_u64))
        .map(|lane| lane.as_canonical_u64())
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
/// commits on the gather row that stages its last word, unless the slot is
/// advice; advice events stage `evbuf` without hashing and are excluded by
/// construction. Object-internal blocks are excluded; only the outer block
/// containing their root is emitted. This extracts witness data; it does not
/// validate the supplied trace.
pub fn absorbed_event_blocks(trace: &[WasmVmStep]) -> Vec<AbsorbedEventBlock> {
    trace
        .iter()
        .filter_map(|row| {
            let rom = row
                .host_event_rom_slot
                .filter(|_| row.row_kind.is_host_event_gather())?;
            let last = usize::from(row.state_before.host_events.slot_cursor) == COMM_CHAIN_BLOCK_WORDS - 1;
            if !last || rom.advice || row.state_after.event_absorb.object_active {
                return None;
            }
            let mut words = row.state_before.event_absorb.evbuf;
            words[COMM_CHAIN_BLOCK_WORDS - 1] = row.host_event_word?;
            Some(AbsorbedEventBlock {
                words,
                metadata: AbsorbedEventMetadata {
                    attributed_fref: row.state_before.host_callee_fref,
                    turn_export_fref: row.state_before.host_events.turn_export_fref,
                },
            })
        })
        .collect()
}
