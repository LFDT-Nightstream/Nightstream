//! Cross-repo parity fixtures for the host-event commitment chain.
//!
//! The expected digests below are protocol constants shared bit-for-bit with
//! `starstream-interleaving-proof` (its `LedgerEffectsCommitment` chain); the
//! same vectors are pinned by a test there. If either side changes the
//! permutation instantiation or the compression layout, both tests must be
//! updated together.

mod common;

use neo_wasm::comm_chain::{
    self, commit_event, fold_event_blocks, CommChainState, COMM_CHAIN_EVENT_ARGS, COMM_CHAIN_STATE_LEN,
};
use neo_wasm::layout::{COL_COMM_CHAIN_AFTER, COL_EVBUF_BEFORE, COL_PERM_PENDING_AFTER, COL_PERM_STATE_AFTER};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::WasmVmStep;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

fn f(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

#[test]
fn comm_chain_fixture_vectors() {
    // Vector 1: genesis state, discriminant 1, args 1..=7.
    let prev = [Goldilocks::ZERO; COMM_CHAIN_STATE_LEN];
    let args: [Goldilocks; COMM_CHAIN_EVENT_ARGS] = core::array::from_fn(|i| f(i as u64 + 1));
    let state1 = commit_event(prev, f(1), args);
    assert_eq!(
        state1,
        [
            f(16060384774117980274),
            f(6217562501851223455),
            f(9809238410420041413),
            f(4191298748431046296),
        ]
    );

    // Vector 2: chained on vector 1, discriminant 16, distinctive args.
    let args2: [Goldilocks; COMM_CHAIN_EVENT_ARGS] =
        [f(0xffff_ffff), f(0xffff_ffff_0000_0000), f(0), f(42), f(7), f(0), f(1)];
    let state2 = commit_event(state1, f(16), args2);
    assert_eq!(
        state2,
        [
            f(2581777910110991851),
            f(4248944502313846729),
            f(3337412769805346927),
            f(12455009736376722043),
        ]
    );
}

#[test]
fn event_blocks_fold_from_explicit_initial_state() {
    let initial_lanes = [f(11), f(22), f(33), f(44)];
    let initial = CommChainState::new(initial_lanes);
    let block: [Goldilocks; 8] = core::array::from_fn(|i| f(i as u64 + 1));

    let expected = commit_event(initial_lanes, block[0], block[1..].try_into().expect("event args"));
    let folded = fold_event_blocks(initial, &[block]);

    assert_eq!(folded.into_lanes(), expected);
    assert_eq!(CommChainState::default().canonical_u64(), [0; 4]);
}

/// The row-level round decomposition (what the in-circuit gadget enforces)
/// must reproduce the whole-permutation compression exactly.
#[test]
fn perm_row_checkpoints_match_commit_event() {
    let prev = [f(11), f(22), f(33), f(44)];
    let disc = f(5);
    let args: [Goldilocks; COMM_CHAIN_EVENT_ARGS] = core::array::from_fn(|i| f(1000 + i as u64));

    let mut words = [Goldilocks::ZERO; comm_chain::COMM_CHAIN_BLOCK_WORDS];
    words[0] = disc;
    words[1..].copy_from_slice(&args);
    let checkpoints = comm_chain::perm_row_checkpoints(prev, words);

    let expected = commit_event(prev, disc, args);
    let fed_forward: [Goldilocks; COMM_CHAIN_STATE_LEN] =
        core::array::from_fn(|i| checkpoints[comm_chain::COMM_CHAIN_PERM_ROWS][i] + prev[i]);
    assert_eq!(fed_forward, expected);

    // Every intermediate checkpoint is reachable from its predecessor via the
    // row transition the CCS gadget encodes.
    for pos in 0..comm_chain::COMM_CHAIN_PERM_ROWS {
        let mut state = checkpoints[pos];
        comm_chain::perm_row_transition(pos, &mut state);
        assert_eq!(state, checkpoints[pos + 1], "row {pos} transition mismatch");
    }
}

/// A host-event trace with committed event blocks. Every row is CCS-checked, so
/// the permutation rows themselves are exercised against the gadget.
fn two_event_trace() -> Vec<WasmVmStep> {
    let trace = common::host_event_fixture::host_event_lifecycle_setup().trace;
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);
    trace
}

fn perm_rows(trace: &[WasmVmStep]) -> Vec<&WasmVmStep> {
    trace
        .iter()
        .filter(|row| row.row_kind.is_host_event_perm())
        .collect()
}

/// The chain may only move on a perm group's last row, and there it must be
/// the feed-forward of the enforced permutation: forging the landed chain
/// limb is CCS-rejected.
#[test]
fn ccs_rejects_forged_chain_update() {
    let trace = two_event_trace();
    let last_row = perm_rows(&trace)
        .into_iter()
        .find(|row| usize::from(row.state_before.event_absorb.perm_round) == comm_chain::COMM_CHAIN_PERM_ROWS - 1)
        .expect("perm group last row");
    let mut witness = build_witness_vector(last_row);
    common::assert_satisfied(&witness, "untampered chain-update row");
    witness[COL_COMM_CHAIN_AFTER[0]] += neo_math::F::ONE;
    common::assert_rejected(&witness, "chain-update row landing a forged chain limb");
}

/// Every perm row's output state is pinned to its round function: forging a
/// lane is CCS-rejected.
#[test]
fn ccs_rejects_forged_perm_round_output() {
    let trace = two_event_trace();
    for row in perm_rows(&trace).into_iter().take(2) {
        let mut witness = build_witness_vector(row);
        common::assert_satisfied(&witness, "untampered perm row");
        witness[COL_PERM_STATE_AFTER[0]] += neo_math::F::ONE;
        common::assert_rejected(&witness, "perm row with a forged round output lane");
    }
}

/// Gadget auxiliary assignment must not repair a bad semantic state supplied
/// by the trace normalizer.
#[test]
fn ccs_rejects_trace_with_forged_perm_state_after() {
    let trace = two_event_trace();
    let original = perm_rows(&trace)
        .into_iter()
        .next()
        .expect("permutation row");
    let mut corrupted = (*original).clone();
    corrupted.state_after.event_absorb.perm_state[0] ^= 1;

    let witness = build_witness_vector(&corrupted);
    common::assert_rejected(&witness, "trace row with a forged permutation state");
}

/// The absorb row's entry state is pinned to `[chain | evbuf]`: forging a
/// buffer word out from under the absorb is CCS-rejected.
#[test]
fn ccs_rejects_forged_absorb_buffer() {
    let trace = two_event_trace();
    let absorb_row = perm_rows(&trace)
        .into_iter()
        .find(|row| row.state_before.event_absorb.perm_pending)
        .expect("perm group absorb row");
    let mut witness = build_witness_vector(absorb_row);
    common::assert_satisfied(&witness, "untampered absorb row");
    witness[COL_EVBUF_BEFORE[4]] += neo_math::F::ONE;
    common::assert_rejected(&witness, "absorb row with a forged buffer word");
}

/// A row that fills the block buffer (or ends the event) must raise
/// `perm_pending`; suppressing the flag to skip the absorb is CCS-rejected.
#[test]
fn ccs_rejects_suppressed_absorb_schedule() {
    let trace = two_event_trace();
    let filling_row = trace
        .iter()
        .find(|row| !row.state_before.event_absorb.perm_pending && row.state_after.event_absorb.perm_pending)
        .expect("buffer-filling row");
    let mut witness = build_witness_vector(filling_row);
    common::assert_satisfied(&witness, "untampered buffer-filling row");
    witness[COL_PERM_PENDING_AFTER] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "buffer-filling row suppressing the pending absorb");
}

/// `absorbed_event_blocks` must reproduce the verifier-expected transcript,
/// re-fold to the carried chain, and retain separate trace attribution.
#[test]
fn absorbed_event_blocks_reconstruct_the_host_event_transcript() {
    use common::host_event_fixture::{expected_transcript, host_event_lifecycle_setup, mul_fref, sink_fref};

    let setup = host_event_lifecycle_setup();
    let events = comm_chain::absorbed_event_blocks(&setup.trace);

    let expected = expected_transcript(&setup.bindings, setup.run_fref);
    assert_eq!(events.len(), expected.len());
    for (event, expected) in events.iter().zip(&expected) {
        assert_eq!(event.words.map(f), *expected);
    }

    let first = setup.trace.first().expect("rows");
    let last = setup.trace.last().expect("rows");
    let mut chain = first.state_before.comm_chain.map(f);
    for event in &events {
        chain = commit_event(
            chain,
            f(event.words[0]),
            core::array::from_fn(|i| f(event.words[1 + i])),
        );
    }
    assert_eq!(
        chain.map(|limb| p3_field::PrimeField64::as_canonical_u64(&limb)),
        last.state_after.comm_chain
    );

    // Entry pair → export, mul's two events → mul, sink's one → sink, exit
    // → back to the export; the whole stream belongs to the export's turn.
    let mul = mul_fref(&setup.bindings);
    let sink = sink_fref(&setup.bindings);
    let attributed: Vec<u32> = events
        .iter()
        .map(|event| event.metadata.attributed_fref)
        .collect();
    assert_eq!(
        attributed,
        [setup.run_fref, setup.run_fref, mul, mul, sink, setup.run_fref]
    );
    assert!(events
        .iter()
        .all(|event| event.metadata.turn_export_fref == setup.run_fref));
}

/// The debug checker must reject a forged carried chain state.
#[test]
fn comm_chain_checker_rejects_forged_state() {
    let wasm = wat::parse_str(r#"(module (func (export "main") (result i32) i32.const 20 i32.const 22 i32.add))"#)
        .expect("wat");
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect("trace");
    let mut trace = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("normalize");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("untampered chain");
    let mid = trace.len() / 2;
    trace[mid].state_after.comm_chain[0] ^= 1;
    assert!(
        neo_wasm::comm_chain::sanity_check_comm_chain(&trace).is_err(),
        "checker must reject a forged chain state"
    );
}
