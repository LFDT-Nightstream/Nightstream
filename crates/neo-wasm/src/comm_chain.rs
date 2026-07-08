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

use once_cell::sync::Lazy;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::{default_goldilocks_poseidon2_12, Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;

/// Field elements carried as the chain state (and emitted as the digest).
pub const COMM_CHAIN_STATE_LEN: usize = 4;
/// Fixed argument slots absorbed per event, after the discriminant.
pub const COMM_CHAIN_EVENT_ARGS: usize = 7;

static PERM12: Lazy<Poseidon2Goldilocks<12>> = Lazy::new(default_goldilocks_poseidon2_12);

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
