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

/// Domain tag opening every raw host-call event.
pub const HOST_CALL_EVENT_TAG: u64 = 1;

/// Absorb one raw host-call event: the canonical, embedder-agnostic record of
/// "this import was called with these arguments and returned this result".
///
/// Serialization (then absorbed in blocks of 8 words via [`commit_event`],
/// zero-padding the final block — unambiguous because the two counts fix the
/// stream length):
///
/// ```text
/// [HOST_CALL_EVENT_TAG, callee_fref, param_count, result_count,
///  arg0_lo, arg0_hi, ..., arg{n-1}_lo, arg{n-1}_hi,
///  result_lo, result_hi]              // present iff result_count = 1
/// ```
///
/// Args are in declared parameter order, two 32-bit limbs each regardless of
/// wasm type, so the block count is a static function of the callee's arity.
pub fn commit_host_call_event(
    prev: [Goldilocks; COMM_CHAIN_STATE_LEN],
    callee_fref: u32,
    param_count: u8,
    result_count: u8,
    args: &[(u32, u32)],
    result: Option<(u32, u32)>,
) -> [Goldilocks; COMM_CHAIN_STATE_LEN] {
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
    for &(lo, hi) in args {
        stream.push(f(u64::from(lo)));
        stream.push(f(u64::from(hi)));
    }
    if let Some((lo, hi)) = result {
        stream.push(f(u64::from(lo)));
        stream.push(f(u64::from(hi)));
    }

    let mut state = prev;
    for block in stream.chunks(1 + COMM_CHAIN_EVENT_ARGS) {
        let mut words = [Goldilocks::ZERO; 1 + COMM_CHAIN_EVENT_ARGS];
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

/// Recompute the host-event commitment chain from the trace's host-call
/// events and validate every carried `comm_chain` state against it.
///
/// Debug-side stand-in for the (not yet built) in-circuit chain-update
/// gadget, mirroring how `memory_semantics` stands in for the committed
/// memory argument.
pub fn sanity_check_comm_chain(trace: &[WasmVmStep]) -> Result<(), WasmBuildError> {
    let err = |msg: String| Err(WasmBuildError::StateMismatch(msg));
    let mut expected = match trace.first() {
        Some(row) => row.state_before.comm_chain,
        None => return Ok(()),
    };
    let mut i = 0;
    while i < trace.len() {
        let row = &trace[i];
        if row.state_before.comm_chain != expected {
            return err(format!(
                "row {i}: comm_chain before {:?} does not match expected {:?}",
                row.state_before.comm_chain, expected
            ));
        }
        let is_host_call = row.row_kind.is_program()
            && matches!(row.opcode, WasmOpcode::Call | WasmOpcode::CallIndirect)
            && !row.target_function_is_guest
            && !row.state_after.trapped;
        if !is_host_call {
            if row.state_after.comm_chain != expected {
                return err(format!(
                    "row {i}: comm_chain after {:?} changed outside a host-call event",
                    row.state_after.comm_chain
                ));
            }
            i += 1;
            continue;
        }

        // Collect the event: arg rows pop in reverse parameter order, then
        // an optional result row.
        let mut j = i + 1;
        let mut args: Vec<(u32, u32)> = Vec::new();
        while j < trace.len() && trace[j].row_kind.is_host_call_arg() {
            let read = trace[j]
                .stack_read0
                .ok_or_else(|| WasmBuildError::StateMismatch(format!("row {j}: host arg row without a stack read")))?;
            args.push((read.value_lo, read.value_hi.unwrap_or(0)));
            j += 1;
        }
        args.reverse();
        let result = if j < trace.len() && trace[j].row_kind.is_host_call_result() {
            let write = trace[j].stack_write0.ok_or_else(|| {
                WasmBuildError::StateMismatch(format!("row {j}: host result row without a stack write"))
            })?;
            j += 1;
            Some((write.value_lo, write.value_hi.unwrap_or(0)))
        } else {
            None
        };
        let param_count = u8::try_from(args.len())
            .map_err(|_| WasmBuildError::StateMismatch(format!("row {i}: host call arity exceeds u8")))?;
        let updated = commit_host_call_event_u64(
            expected,
            row.state_after.host_callee_fref,
            param_count,
            u8::from(result.is_some()),
            &args,
            result,
        );

        let end = j - 1;
        for k in i..=end {
            let event_row = &trace[k];
            if k > i && event_row.state_before.comm_chain != expected {
                return err(format!("row {k}: comm_chain changed before the event's last row"));
            }
            let want_after = if k == end { updated } else { expected };
            if event_row.state_after.comm_chain != want_after {
                return err(format!(
                    "row {k}: comm_chain after {:?} does not match recomputed event chain",
                    event_row.state_after.comm_chain
                ));
            }
        }
        expected = updated;
        i = j;
    }
    Ok(())
}
