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
use neo_wasm::layout::{COL_COMM_CHAIN_AFTER, COL_EVBUF_AFTER, COL_EVBUF_BEFORE};
use neo_wasm::operations::r1cs::operation_circuit;
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

/// The absorb that a gather row performs is the event compression.
#[test]
fn absorb_block_matches_commit_event() {
    let prev = [11, 22, 33, 44];
    let words: [u64; comm_chain::COMM_CHAIN_BLOCK_WORDS] = core::array::from_fn(|i| 1000 + i as u64);
    let expected = commit_event(prev.map(f), f(words[0]), core::array::from_fn(|i| f(words[1 + i])));
    assert_eq!(
        comm_chain::absorb_block(prev, words),
        expected.map(|limb| p3_field::PrimeField64::as_canonical_u64(&limb))
    );
}

/// A host-event trace with committed event blocks. Every row is CCS-checked, so
/// the absorb rows themselves are exercised against the gadget.
fn two_event_trace() -> Vec<WasmVmStep> {
    let trace = common::host_event_fixture::host_event_lifecycle_setup().trace;
    common::check_native_event_hashes(&trace).expect("native event hashes");
    common::ccs_check_trace(&trace);
    trace
}

/// Gather rows that fill a non-advice block and fold it into the chain.
fn absorb_rows(trace: &[WasmVmStep]) -> Vec<&WasmVmStep> {
    trace
        .iter()
        .filter(|row| {
            row.row_kind.is_host_event_gather()
                && row.state_before.host_events.slot_cursor == 7
                && row.host_event_rom_slot.is_some_and(|rom| !rom.advice)
        })
        .collect()
}

/// The chain moves on the absorb row to the commitment of the filled block:
/// forging the landed chain limb is CCS-rejected.
#[test]
fn ccs_rejects_forged_chain_update() {
    let trace = two_event_trace();
    let rows = absorb_rows(&trace);
    assert!(!rows.is_empty(), "the fixture absorbs blocks");
    for row in rows {
        let mut witness = build_witness_vector(row).unwrap();
        common::assert_satisfied(&witness, "untampered absorb row");
        witness[COL_COMM_CHAIN_AFTER[0]] += neo_math::F::ONE;
        common::assert_rejected(&witness, "absorb row landing a forged chain limb");
    }
}

/// Gadget auxiliary assignment must not repair a bad semantic state supplied
/// by the trace normalizer.
#[test]
fn ccs_rejects_trace_with_forged_absorb_output() {
    let trace = two_event_trace();
    let mut corrupted = (*absorb_rows(&trace)[0]).clone();
    corrupted.state_after.comm_chain[0] ^= 1;
    let witness = build_witness_vector(&corrupted).unwrap();
    common::assert_rejected(&witness, "trace row with a forged absorb output");
}

/// The absorb reads `[chain | evbuf_before[0..7] | staged word]`: forging a
/// buffer word out from under the absorb is CCS-rejected.
#[test]
fn ccs_rejects_forged_absorb_buffer() {
    let trace = two_event_trace();
    let mut witness = build_witness_vector(absorb_rows(&trace)[0]).unwrap();
    common::assert_satisfied(&witness, "untampered absorb row");
    witness[COL_EVBUF_BEFORE[4]] += neo_math::F::ONE;
    common::assert_rejected(&witness, "absorb row with a forged buffer word");
}

/// A row that fills a non-advice block must absorb it: carrying the chain and
/// keeping the filled buffer instead is CCS-rejected.
#[test]
fn ccs_rejects_suppressed_absorb() {
    let trace = two_event_trace();
    let mut skipped = (*absorb_rows(&trace)[0]).clone();
    skipped.state_after.comm_chain = skipped.state_before.comm_chain;
    let mut block = skipped.state_before.event_absorb.evbuf;
    block[7] = skipped.host_event_word.expect("staged word");
    skipped.state_after.event_absorb.evbuf = block;
    let witness = build_witness_vector(&skipped).unwrap();
    common::assert_rejected(&witness, "buffer-filling row skipping the absorb");
}

/// The absorb row clears the buffer, so the next block's unwritten slots are
/// zero: keeping a word is CCS-rejected.
#[test]
fn ccs_rejects_absorb_row_keeping_a_buffer_word() {
    let trace = two_event_trace();
    let mut witness = build_witness_vector(absorb_rows(&trace)[0]).unwrap();
    witness[COL_EVBUF_AFTER[3]] += neo_math::F::ONE;
    common::assert_rejected(&witness, "absorb row keeping a buffer word");
}

/// The complete one-row witness of `row` with the shared absorb block
/// `block` in place of its own (Lean
/// `Lifecycle.Nebula.SharedAbsorb.groupRows_iff_forall`: a group holds
/// exactly when each of its rows holds on the group's block).
fn with_block(row: &WasmVmStep, block: &[neo_math::F]) -> Vec<neo_math::F> {
    let shared = operation_circuit().unwrap().shared().columns.clone();
    let mut witness = common::complete_witness(&build_witness_vector(row).unwrap()).unwrap();
    witness[shared].copy_from_slice(block);
    witness
}

/// The shared absorb block of `row`'s own witness: its input lanes are the
/// row's absorb input, its output lanes the commitment of that input.
fn own_block(row: &WasmVmStep) -> Vec<neo_math::F> {
    let shared = operation_circuit().unwrap().shared().columns.clone();
    common::complete_witness(&build_witness_vector(row).unwrap()).unwrap()[shared].to_vec()
}

/// An absorb row lands on the output lanes of its block, which hold the
/// commitment of its own input. A changed output lane, or the block of a
/// row with another input, is rejected. A row that does not absorb reads no
/// word of the block, so it holds with any block whose rows hold.
#[test]
fn an_absorb_row_reads_the_output_of_its_own_input_in_the_shared_block() {
    let trace = two_event_trace();
    let absorb = absorb_rows(&trace)[0];
    let block = own_block(absorb);
    let output = &block[block.len() - COMM_CHAIN_STATE_LEN..];
    assert_eq!(output, absorb.state_after.comm_chain.map(f));
    assert_eq!(common::complete_holds(&with_block(absorb, &block)), Ok(()));
    for lane in block.len() - COMM_CHAIN_STATE_LEN..block.len() {
        let mut wrong = block.clone();
        wrong[lane] += neo_math::F::ONE;
        assert!(common::complete_holds(&with_block(absorb, &wrong)).is_err());
    }
    let index = trace
        .iter()
        .position(|row| std::ptr::eq(row, absorb))
        .unwrap();
    let other = trace[..index]
        .iter()
        .rev()
        .find(|row| own_block(row)[..12] != block[..12])
        .expect("an earlier row with another absorb input");
    assert!(absorb_rows(std::slice::from_ref(other)).is_empty());
    assert!(common::complete_holds(&with_block(absorb, &own_block(other))).is_err());
    assert_eq!(common::complete_holds(&with_block(other, &block)), Ok(()));
}

/// Two absorb rows with different inputs have no common block: with either
/// row's block the other row fails, as the gated input rows require the
/// block's input lanes to equal each row's input (Lean
/// `Lifecycle.Nebula.SharedAbsorb.eq_of_gated`). An honest group never holds
/// two absorbs (`SharedAbsorb.absorbs_unique`).
#[test]
fn two_absorbs_with_different_inputs_share_no_block() {
    let trace = two_event_trace();
    let rows = absorb_rows(&trace);
    let (first, second) = (rows[0], rows[1]);
    let blocks = [own_block(first), own_block(second)];
    assert_ne!(blocks[0][..12], blocks[1][..12], "the two inputs differ");
    for block in &blocks {
        let holds = [first, second].map(|row| common::complete_holds(&with_block(row, block)).is_ok());
        assert_eq!(holds.iter().filter(|holds| **holds).count(), 1);
    }
}

/// Only absorb rows move the chain: a gather row before slot seven that
/// changes it is CCS-rejected.
#[test]
fn ccs_rejects_chain_change_off_the_absorb_row() {
    let trace = two_event_trace();
    let row = trace
        .iter()
        .find(|row| row.row_kind.is_host_event_gather() && row.state_before.host_events.slot_cursor == 3)
        .expect("a mid-block gather row");
    let mut witness = build_witness_vector(row).unwrap();
    common::assert_satisfied(&witness, "untampered gather row");
    witness[COL_COMM_CHAIN_AFTER[2]] += neo_math::F::ONE;
    common::assert_rejected(&witness, "mid-block gather row changing the chain");
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

#[test]
fn shared_transcript_fixture_excludes_opaque_internal_blocks() {
    use common::host_event_fixture::{expected_transcript, host_event_lifecycle_setup};
    use neo_wasm::host_event_bindings::{opaque_value_root, SlotBinding};

    let mut setup = host_event_lifecycle_setup();
    let mut expected = expected_transcript(&setup.bindings, setup.run_fref);
    let template = setup.bindings.exports.get_mut(&setup.run_fref).unwrap();
    let last = expected.len() - 1;
    for (events, index, value) in [(&mut template.entry, 0, 17), (&mut template.exit, last, 23)] {
        let schema = [1, 2, 3, 4];
        let lowered = events[0]
            .clone()
            .with_opaque(2, schema, vec![SlotBinding::Const(value)])
            .unwrap();
        events.splice(0..1, lowered);
        expected[index][2..6].copy_from_slice(&opaque_value_root(schema, &[value]).unwrap().map(f));
    }
    assert_eq!(expected_transcript(&setup.bindings, setup.run_fref), expected);
}

/// `absorbed_event_blocks` reads the staged word the way the witness does, so a
/// trace without `host_event_word` on an absorb row that the circuit accepts
/// still yields every block, folding to the circuit's chain.
#[test]
fn absorbed_event_blocks_follow_the_circuit_without_the_staged_word_field() {
    let mut trace = two_event_trace();
    let index = trace
        .iter()
        .position(|row| absorb_rows(std::slice::from_ref(row)).len() == 1 && row.host_event_word == Some(0))
        .expect("an absorb row staging a zero word");
    trace[index].host_event_word = None;
    common::ccs_check_trace(&trace);
    let mut chain = trace[0].state_before.comm_chain;
    let blocks = comm_chain::absorbed_event_blocks(&trace);
    assert_eq!(blocks.len(), absorb_rows(&trace).len());
    for block in &blocks {
        chain = comm_chain::absorb_block(chain, block.words);
    }
    assert_eq!(chain, trace.last().unwrap().state_after.comm_chain);
}

/// The native test oracle compares each absorb row's output with the native
/// compression, independently of the circuit.
#[test]
fn native_event_hash_check_rejects_forged_output() {
    let mut trace = two_event_trace();
    let index = trace
        .iter()
        .position(|row| {
            row.row_kind.is_host_event_gather()
                && row.state_before.host_events.slot_cursor == 7
                && row.host_event_rom_slot.is_some_and(|rom| !rom.advice)
        })
        .unwrap();
    trace[index].state_after.comm_chain[0] ^= 1;
    assert!(
        common::check_native_event_hashes(&trace).is_err(),
        "native compression must disagree with a forged absorb output"
    );
}
