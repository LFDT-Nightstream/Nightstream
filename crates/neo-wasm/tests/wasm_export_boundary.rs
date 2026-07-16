//! Export-boundary grammar templates: entry events (receiver-side
//! `Enter`/`Activation`/payload publication) absorb before the export's
//! first instruction — with `ClaimLocal` slots bootstrapping the
//! zero-initialized entry frame's locals from the claim inputs — and exit
//! events (`Return` with the captured result) absorb after the halting
//! row. Event values are bound by the final-chain transcript check: the
//! verifier folds the claimed transcript natively (`fold_event_blocks`)
//! and compares it with the proof-carried final `comm_chain`. Single-turn
//! V1. Discriminants below are example embedder data.

mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::event_grammar::{ExportTemplate, GrammarEvent, HostEventGrammar, Limb, SlotSource};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::WasmVmStep;
use p3_field::PrimeCharacteristicRing;
use wasmtime::component::Val as ComponentVal;

const ZERO: SlotSource = SlotSource::Const(0);

fn slots(entries: &[(usize, SlotSource)]) -> [SlotSource; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
}

/// `run(x: s32, y: s32) -> s32 { x + y }`: a pure export, no host calls.
fn add_component_wat() -> &'static str {
    r#"
    (component
      (type $run-type (func (param "x" s32) (param "y" s32) (result s32)))
      (core module $m
        (func (export "run") (param i32 i32) (result i32)
          local.get 0
          local.get 1
          i32.add))
      (core instance $i (instantiate $m))
      (alias core export $i "run" (core func $run))
      (func (export "run") (type $run-type)
        (canon lift (core func $run))))
    "#
}

/// Entry: Enter(f_id) + one spec-shaped Activation whose payload slots
/// carry the claim inputs — two absorb-only words and the two param words,
/// the latter annotated `ClaimLocal` so they also bootstrap the param
/// locals. The absorbed Activation block is bit-identical to one with
/// plain `Claim` slots: the write behavior is table data, never
/// transcript. Exit: Return-ish event carrying the output. Claim-input
/// words are consumed in slot order:
/// [activation val, activation caller, param 0, param 1].
fn export_template() -> ExportTemplate {
    let write = |idx, local| SlotSource::ClaimLocal {
        idx,
        local,
        limb: Limb::Lo,
    };
    ExportTemplate {
        entry: vec![
            GrammarEvent::op(20, slots(&[(0, SlotSource::Const(55))])), // Enter(f_id)
            GrammarEvent::op(
                8,
                slots(&[
                    (1, SlotSource::Claim { idx: 0 }),
                    (3, SlotSource::Claim { idx: 1 }),
                    (4, write(2, 0)),
                    (5, write(3, 1)),
                ]),
            ), // Activation(val, caller, payload = params)
        ],
        exit: vec![GrammarEvent::op(
            17,
            slots(&[(1, SlotSource::OutputElem { limb: Limb::Lo })]),
        )],
        entry_claim_count: 4,
        exit_claim_count: 0,
    }
}

fn export_fref(trace: &[WasmVmStep]) -> u32 {
    trace
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row")
        .current_function_ref
}

fn boundary_trace() -> (Vec<WasmVmStep>, HostEventGrammar) {
    let component_bytes = wat::parse_str(add_component_wat()).expect("component wat");
    let args = [ComponentVal::S32(7), ComponentVal::S32(35)];
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(&component_bytes, "run", &args, |_| Ok(()))
        .expect("component run");

    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let fref = export_fref(&raw);
    let mut grammar = HostEventGrammar::default();
    grammar.exports.insert(fref, export_template());

    let turns = [neo_wasm::event_grammar::TurnClaims {
        entry: vec![500, 501, 7, 35],
        exit: vec![],
    }];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &turns).expect("grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    // Claim bootstrap: the RAM model's entry locals start all-zero; the
    // entry gather rows write the claim inputs into them.
    let mut preload =
        neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &vec![0; run.initial_locals.len()]);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &grammar);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("grammar ROM + locals reads match");

    (trace, grammar)
}

#[test]
fn export_boundary_folds_entry_and_exit_events() {
    let (trace, grammar) = boundary_trace();

    // The trace opens with the entry gather rows (before any program row).
    assert!(trace[0].row_kind.is_host_event_gather());
    assert_eq!(trace[0].state_before.grammar.events_remaining, 2);

    let staged: Vec<[u64; 8]> = trace
        .iter()
        .filter(|row| {
            row.row_kind.is_host_event_gather()
                && row.state_after.event_absorb.perm_pending
                && !row.state_before.event_absorb.perm_pending
        })
        .map(|row| row.state_after.event_absorb.evbuf)
        .collect();
    assert_eq!(
        staged,
        vec![
            [20, 55, 0, 0, 0, 0, 0, 0],    // Enter
            [8, 0, 500, 0, 501, 7, 35, 0], // Activation(val, caller, payload = params)
            [17, 0, 42, 0, 0, 0, 0, 0],    // Return(output)
        ],
    );

    // The verifier's transcript check: expand the claimed transcript from
    // the template + claims (entry inputs, captured output) and fold it
    // natively; the proof-carried final chain must equal that fold — and a
    // different input claim must not.
    let template = grammar.exports.values().next().expect("template");
    let mut expected = neo_wasm::event_grammar::expand_export_entry(template, &[500, 501, 7, 35]).expect("entry");
    expected.extend(neo_wasm::event_grammar::expand_export_exit(template, Some((42, 0)), &[]).expect("exit"));
    assert_eq!(expected, staged, "claimed transcript must match the staged blocks");
    let lift = |blocks: &[[u64; 8]]| -> Vec<[p3_goldilocks::Goldilocks; 8]> {
        blocks
            .iter()
            .map(|block| block.map(p3_goldilocks::Goldilocks::from_u64))
            .collect()
    };
    let canonical =
        |chain: [p3_goldilocks::Goldilocks; 4]| chain.map(|limb| p3_field::PrimeField64::as_canonical_u64(&limb));
    let final_chain = trace.last().expect("rows").state_after.comm_chain;
    assert_eq!(
        final_chain,
        canonical(neo_wasm::comm_chain::fold_event_blocks(&lift(&expected)))
    );
    let wrong_inputs = neo_wasm::event_grammar::expand_export_entry(template, &[500, 501, 7, 36]).expect("wrong entry");
    assert_ne!(
        final_chain,
        canonical(neo_wasm::comm_chain::fold_event_blocks(&lift(
            &[wrong_inputs, expected[2..].to_vec()].concat()
        ))),
        "a different input claim must fold to a different chain"
    );
    assert!(trace[0].state_before.grammar_mode);
}

/// Forging the exit event's output word is CCS-rejected: the word is bound
/// to the carried output-capture value.
#[test]
fn ccs_rejects_forged_exit_output() {
    let (trace, _) = boundary_trace();
    let output_slot_row = trace
        .iter()
        .find(|row| row.row_kind.is_host_event_gather() && row.grammar_rom_slot.is_some_and(|rom| rom.kind == 5))
        .expect("output slot row");
    let mut witness = build_witness_vector(output_slot_row);
    common::assert_satisfied(&witness, "untampered output slot row");
    let cursor = usize::from(output_slot_row.state_before.grammar.slot_cursor);
    witness[neo_wasm::layout::COL_EVBUF0_AFTER + cursor] += neo_math::F::ONE;
    common::assert_rejected(&witness, "exit gather row staging a forged output word");
}

/// An input-local gather row must write the very word it stages: forging
/// the staged word away from the locals write is CCS-rejected.
#[test]
fn ccs_rejects_forged_input_word() {
    let (trace, _) = boundary_trace();
    let input_slot_row = trace
        .iter()
        .find(|row| row.row_kind.is_host_event_gather() && row.grammar_rom_slot.is_some_and(|rom| rom.kind == 4))
        .expect("input-local slot row");
    let mut witness = build_witness_vector(input_slot_row);
    common::assert_satisfied(&witness, "untampered input-local slot row");
    let cursor = usize::from(input_slot_row.state_before.grammar.slot_cursor);
    witness[neo_wasm::layout::COL_EVBUF0_AFTER + cursor] += neo_math::F::ONE;
    common::assert_rejected(&witness, "entry gather row staging a word other than the locals write");
}

/// The exit latch is forced: suppressing the exit schedule on the capture
/// row is CCS-rejected.
#[test]
fn ccs_rejects_suppressed_exit_schedule() {
    let (trace, _) = boundary_trace();
    let capture_row = trace
        .iter()
        .find(|row| row.output_captured)
        .expect("capture row");
    let mut witness = build_witness_vector(capture_row);
    common::assert_satisfied(&witness, "untampered capture row");
    witness[neo_wasm::layout::COL_GRAMMAR_EVREM_AFTER] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "capture row suppressing the exit schedule");
}

/// `run(x: s64) -> s64 { x }`: an i64 export param bootstrapped lane by
/// lane (lo slot then hi slot), read back through the exit output limbs.
fn identity64_component_wat() -> &'static str {
    r#"
    (component
      (type $run-type (func (param "x" s64) (result s64)))
      (core module $m
        (func (export "run") (param i64) (result i64)
          local.get 0))
      (core instance $i (instantiate $m))
      (alias core export $i "run" (core func $run))
      (func (export "run") (type $run-type)
        (canon lift (core func $run))))
    "#
}

#[test]
fn i64_param_bootstraps_both_lanes() {
    let component_bytes = wat::parse_str(identity64_component_wat()).expect("component wat");
    // x = 3·2^32 + 7.
    let args = [ComponentVal::S64((3i64 << 32) | 7)];
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(&component_bytes, "run", &args, |_| Ok(()))
        .expect("component run");

    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let fref = export_fref(&raw);
    let mut grammar = HostEventGrammar::default();
    grammar.exports.insert(
        fref,
        ExportTemplate {
            entry: vec![GrammarEvent::op(
                21,
                slots(&[
                    (
                        0,
                        SlotSource::ClaimLocal {
                            idx: 0,
                            local: 0,
                            limb: Limb::Lo,
                        },
                    ),
                    (
                        1,
                        SlotSource::ClaimLocal {
                            idx: 1,
                            local: 0,
                            limb: Limb::Hi,
                        },
                    ),
                ]),
            )],
            exit: vec![GrammarEvent::op(
                18,
                slots(&[
                    (0, SlotSource::OutputElem { limb: Limb::Lo }),
                    (1, SlotSource::OutputElem { limb: Limb::Hi }),
                ]),
            )],
            entry_claim_count: 2,
            exit_claim_count: 0,
        },
    );

    let turns = [neo_wasm::event_grammar::TurnClaims {
        entry: vec![7, 3],
        exit: vec![],
    }];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &turns).expect("grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

    let staged: Vec<[u64; 8]> = trace
        .iter()
        .filter(|row| {
            row.row_kind.is_host_event_gather()
                && row.state_after.event_absorb.perm_pending
                && !row.state_before.event_absorb.perm_pending
        })
        .map(|row| row.state_after.event_absorb.evbuf)
        .collect();
    assert_eq!(
        staged,
        vec![
            [21, 7, 3, 0, 0, 0, 0, 0], // entry: both param lanes
            [18, 7, 3, 0, 0, 0, 0, 0], // exit: both output lanes
        ],
    );

    // The locals RAM sees the zero-init + lane writes and the guest's i64
    // read back out of them.
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload =
        neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &vec![0; run.initial_locals.len()]);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &grammar);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("grammar ROM + lane writes match");

    // The captured output is the full i64 round-tripped through the locals.
    let last = trace.last().expect("rows").state_after;
    assert_eq!((last.output.value_lo, last.output.value_hi), (7, 3));
}
