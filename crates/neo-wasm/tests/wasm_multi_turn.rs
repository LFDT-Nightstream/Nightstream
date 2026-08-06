//! Multi-turn export traces with persistent module state, per-turn input
//! bootstrapping, boundary constraints, and transcript binding.

mod common;

use common::audit::{prove_batched, verify_with_transcript, AuditProveError};
use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::event_grammar::{ExportTemplate, GrammarEvent, HostEventGrammar, Limb, SlotSource, TurnClaims};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{grammar_top_level_initial_state_digest, preprocess_seeded_batched, WasmVmStep};
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

/// Stateful export used to test cross-turn global persistence.
fn counter_component_wat() -> &'static str {
    r#"
    (component
      (type $add-type (func (param "x" s32) (result s32)))
      (core module $m
        (global $acc (mut i32) (i32.const 0))
        (func (export "add") (param i32) (result i32)
          global.get $acc
          local.get 0
          i32.add
          global.set $acc
          global.get $acc))
      (core instance $i (instantiate $m))
      (alias core export $i "add" (core func $add))
      (func (export "add") (type $add-type)
        (canon lift (core func $add))))
    "#
}

/// Entry absorbs caller attribution and initializes local 0; exit absorbs the
/// captured output.
fn add_template() -> ExportTemplate {
    ExportTemplate {
        entry: vec![GrammarEvent::op(
            8,
            slots(&[
                (0, SlotSource::Claim { idx: 0 }),
                (
                    1,
                    SlotSource::ClaimLocal {
                        idx: 1,
                        local: 0,
                        limb: Limb::Lo,
                    },
                ),
            ]),
        )],
        exit: vec![GrammarEvent::op(
            17,
            slots(&[(0, SlotSource::OutputElem { limb: Limb::Lo })]),
        )],
        entry_claim_count: 2,
        exit_claim_count: 0,
    }
}

fn turn_claims() -> [TurnClaims; 2] {
    [
        TurnClaims {
            entry: vec![901, 7],
            exit: vec![],
            ..Default::default()
        },
        TurnClaims {
            entry: vec![902, 35],
            exit: vec![],
            ..Default::default()
        },
    ]
}

struct MultiTurnSetup {
    trace: Vec<WasmVmStep>,
    grammar: HostEventGrammar,
    add_fref: u32,
    component_bytes: Vec<u8>,
}

fn multi_turn_setup() -> MultiTurnSetup {
    let component_bytes = wat::parse_str(counter_component_wat()).expect("component wat");
    let calls = [
        ("add", vec![ComponentVal::S32(7)]),
        ("add", vec![ComponentVal::S32(35)]),
    ];
    let run = neo_wasm::collect_wasmtime_component_run_calls(&component_bytes, &calls, |_| Ok(()))
        .expect("two-turn component run");
    assert_eq!(run.results, ["7", "42"], "the global must accumulate across turns");

    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps);
    assert!(raw.is_err(), "a multi-turn trace must not normalize in raw mode");

    let component_first = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &HostEventGrammar::default(),
        &turn_claims(),
        Default::default(),
    );
    assert!(component_first.is_err(), "missing export template must be rejected");

    // Resolve the export fref from a single-call raw run.
    let single = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(
        &component_bytes,
        "add",
        &[ComponentVal::S32(1)],
        |_| Ok(()),
    )
    .expect("single run");
    let add_fref = neo_wasm::traces_from_wasmtime_steps(&single.steps)
        .expect("raw single-turn trace")
        .first()
        .expect("rows")
        .current_function_ref;

    let mut grammar = HostEventGrammar::default();
    grammar.exports.insert(add_fref, add_template());

    let trace =
        neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &turn_claims(), Default::default())
            .expect("multi-turn grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

    MultiTurnSetup {
        trace,
        grammar,
        add_fref,
        component_bytes,
    }
}

fn expected_transcript(
    grammar: &HostEventGrammar,
    add_fref: u32,
    turns: &[TurnClaims],
    outputs: &[u32],
) -> Vec<[p3_goldilocks::Goldilocks; 8]> {
    let template = grammar.exports.get(&add_fref).expect("template");
    let mut blocks = Vec::new();
    for (turn, &output) in turns.iter().zip(outputs) {
        blocks.extend(neo_wasm::event_grammar::expand_export_entry(template, &turn.entry).expect("entry"));
        blocks.extend(
            neo_wasm::event_grammar::expand_export_exit(
                template,
                Some((output, 0)),
                &turn.exit,
                &turn.exit_memory_reads,
            )
            .expect("exit"),
        );
    }
    blocks
        .into_iter()
        .map(|block| block.map(p3_goldilocks::Goldilocks::from_u64))
        .collect()
}

#[test]
fn turn_boundary_row_bridges_the_turns() {
    let setup = multi_turn_setup();
    let boundaries: Vec<&WasmVmStep> = setup
        .trace
        .iter()
        .filter(|row| row.row_kind.is_turn_boundary())
        .collect();
    assert_eq!(boundaries.len(), 1, "two turns, one boundary");
    let tb = boundaries[0];

    // Completed turn.
    assert_eq!(tb.state_before.sp, 0);
    assert!(tb.state_before.output.enabled);
    assert_eq!(tb.state_before.output.value_lo, 7);
    assert_eq!(tb.state_before.grammar.events_remaining, 0);
    assert_eq!(tb.state_before.call_stack_depth, 0);

    // Fresh turn.
    assert_eq!(tb.state_after.sp, 0);
    assert!(!tb.state_after.output.enabled);
    assert_eq!(tb.state_after.grammar.events_remaining, 1);
    assert_eq!(tb.state_after.grammar.event_index, 0);
    assert_eq!(tb.state_after.host_callee_fref, setup.add_fref);
    assert_ne!(
        tb.state_before.pc, tb.state_after.pc,
        "the boundary bridges the pc jump"
    );

    // Cross-turn state.
    assert_eq!(tb.state_before.comm_chain, tb.state_after.comm_chain);
    assert_eq!(tb.state_before.event_absorb, tb.state_after.event_absorb);

    // Entry events drain before program execution resumes.
    let tb_idx = setup
        .trace
        .iter()
        .position(|row| row.row_kind.is_turn_boundary())
        .expect("boundary");
    let next_program = setup.trace[tb_idx + 1..]
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("turn-2 program row");
    assert_eq!(next_program.state_before.grammar.events_remaining, 0);
    assert_eq!(next_program.state_before.pc, tb.state_after.pc);
}

#[test]
fn multi_turn_proof_binds_both_turns_inputs() {
    let setup = multi_turn_setup();
    let artifacts =
        neo_wasm::extract_first_component_core_program_artifacts(&setup.component_bytes).expect("artifacts");
    let entry_pc = common::entry_pc_for_function_ref(&artifacts, u64::from(setup.add_fref));
    let digest = grammar_top_level_initial_state_digest(
        &artifacts.tables,
        entry_pc,
        &setup.grammar,
        setup.add_fref,
        Default::default(),
    );
    assert_eq!(
        digest,
        neo_wasm::semantic_state_digest(setup.trace[0].state_before),
        "verifier initial state must match the trace's first before-state"
    );

    let batch_size = 8;
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &setup.trace, batch_size).expect("prove");
    let final_state = common::final_state(&setup.trace);
    assert_eq!((final_state.output.value_lo, final_state.output.value_hi), (42, 0));

    let transcript = expected_transcript(&setup.grammar, setup.add_fref, &turn_claims(), &[7, 42]);
    verify_with_transcript(&prep, &proof, final_state, Default::default(), &transcript)
        .expect("verify with the two-turn transcript");

    let mut wrong_turns = turn_claims();
    wrong_turns[1].entry[1] = 34;
    let wrong = expected_transcript(&setup.grammar, setup.add_fref, &wrong_turns, &[7, 42]);
    assert!(
        matches!(
            verify_with_transcript(&prep, &proof, final_state, Default::default(), &wrong),
            Err(AuditProveError::TranscriptMismatch)
        ),
        "a transcript claiming a different turn-2 input must be rejected"
    );
}

#[test]
fn memory_model_carries_state_across_turns() {
    let setup = multi_turn_setup();
    let calls = [
        ("add", vec![ComponentVal::S32(7)]),
        ("add", vec![ComponentVal::S32(35)]),
    ];
    let run = neo_wasm::collect_wasmtime_component_run_calls(&setup.component_bytes, &calls, |_| Ok(()))
        .expect("two-turn component run");
    let artifacts =
        neo_wasm::extract_first_component_core_program_artifacts(&setup.component_bytes).expect("artifacts");
    let mut preload =
        neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &vec![0; run.initial_locals.len()]);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &setup.grammar);
    let witness_rows: Vec<Vec<neo_math::F>> = setup.trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("locals overwrite + global persistence check out");
}

#[test]
fn ccs_rejects_forged_turn_boundary() {
    let setup = multi_turn_setup();
    let tb = setup
        .trace
        .iter()
        .find(|row| row.row_kind.is_turn_boundary())
        .expect("boundary row");

    let witness = build_witness_vector(tb);
    common::assert_satisfied(&witness, "untampered turn boundary");

    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_OUTPUT_ENABLED_AFTER] = neo_math::F::ONE;
    common::assert_rejected(&forged, "boundary keeping the previous turn's output armed");

    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_GRAMMAR_EVREM_AFTER] = neo_math::F::ZERO;
    common::assert_rejected(&forged, "boundary skipping the next turn's entry schedule");

    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_SP_BEFORE] = neo_math::F::ONE;
    forged[neo_wasm::layout::COL_SP_AFTER] = neo_math::F::ONE;
    common::assert_rejected(&forged, "boundary firing with a live operand stack");

    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_TURN_EXPORT_FREF_AFTER] += neo_math::F::ONE;
    common::assert_rejected(&forged, "boundary carrying a different export from its target");

    // A boundary can't fire while the previous turn still owes events.
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_GRAMMAR_EVREM_BEFORE] = neo_math::F::ONE;
    common::assert_rejected(&forged, "boundary before the previous schedule is spent");

    // Presence binding: a boundary pointed at a fref with no export
    // template reads the zero-filled count cell (the memory model pins the
    // claim; see memory_model_rejects_boundary_into_undeclared_fref), and
    // under the biased load no normal schedule satisfies the row.
    let undeclared_fref = neo_math::F::from_u64(u64::from(setup.add_fref) + 7);
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_HOST_CALLEE_FREF_AFTER] = undeclared_fref;
    forged[neo_wasm::layout::COL_TURN_EXPORT_FREF_AFTER] = undeclared_fref;
    forged[neo_wasm::layout::COL_GRAMMAR_PRE_COUNT] = neo_math::F::ZERO;
    common::assert_rejected(&forged, "boundary entering an undeclared fref with a normal schedule");

    // The only row-locally satisfiable assignment loads the poisoned
    // schedule EVREM = -1 = p-1 ...
    let mut poisoned = forged.clone();
    poisoned[neo_wasm::layout::COL_GRAMMAR_EVREM_AFTER] = -neo_math::F::ONE;
    common::assert_satisfied(&poisoned, "undeclared boundary target loads the poisoned schedule");

    // ... which the composed circuit's grammar-ROM address bound prevents
    // from draining before another program row can run.
    let program_row = setup
        .trace
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row");
    let mut wedged = build_witness_vector(program_row);
    wedged[neo_wasm::layout::COL_GRAMMAR_EVREM_BEFORE] = -neo_math::F::ONE;
    common::assert_rejected(&wedged, "program row under the poisoned schedule");
}

/// The claim side of the presence binding: an undeclared boundary target
/// cannot fake a declared export's biased count cell — the export
/// entry-count family has no cell for it, so the ROM read mismatches.
#[test]
fn memory_model_rejects_boundary_into_undeclared_fref() {
    let setup = multi_turn_setup();
    let calls = [
        ("add", vec![ComponentVal::S32(7)]),
        ("add", vec![ComponentVal::S32(35)]),
    ];
    let run = neo_wasm::collect_wasmtime_component_run_calls(&setup.component_bytes, &calls, |_| Ok(()))
        .expect("two-turn component run");
    let artifacts =
        neo_wasm::extract_first_component_core_program_artifacts(&setup.component_bytes).expect("artifacts");
    let mut preload =
        neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &vec![0; run.initial_locals.len()]);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &setup.grammar);
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();

    let tb_index = setup
        .trace
        .iter()
        .position(|row| row.row_kind.is_turn_boundary())
        .expect("boundary row");
    let mut witness_rows: Vec<Vec<neo_math::F>> = setup.trace.iter().map(build_witness_vector).collect();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("the honest trace must pass");
    witness_rows[tb_index][neo_wasm::layout::COL_HOST_CALLEE_FREF_AFTER] =
        neo_math::F::from_u64(u64::from(setup.add_fref) + 7);
    assert!(
        neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload).is_err(),
        "an undeclared boundary target must not read a declared export's count cell"
    );
}

#[test]
fn ccs_rejects_execution_after_halt() {
    let setup = multi_turn_setup();

    // A program row claiming the turn already finished is rejected.
    let program_row = setup
        .trace
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row");
    let mut witness = build_witness_vector(program_row);
    common::assert_satisfied(&witness, "untampered program row");
    witness[neo_wasm::layout::COL_HALTED_BEFORE] = neo_math::F::ONE;
    witness[neo_wasm::layout::COL_HALTED] = neo_math::F::ONE;
    common::assert_rejected(&witness, "program row executing after a halt");

    // The halting row cannot suppress the latch (with or without capture).
    let halting_row = setup
        .trace
        .iter()
        .find(|row| row.row_kind.is_program() && row.state_after.halted)
        .expect("halting row");
    let mut witness = build_witness_vector(halting_row);
    common::assert_satisfied(&witness, "untampered halting row");
    witness[neo_wasm::layout::COL_HALTED] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "halting row pretending the turn is not done");
}

#[test]
fn resultless_turn_can_precede_another_turn() {
    let component_bytes = wat::parse_str(
        r#"
        (component
          (type $poke-type (func (param "x" s32)))
          (type $read-type (func (result s32)))
          (core module $m
            (global $acc (mut i32) (i32.const 0))
            (func (export "poke") (param i32)
              local.get 0
              global.set $acc)
            (func (export "read") (result i32)
              global.get $acc))
          (core instance $i (instantiate $m))
          (alias core export $i "poke" (core func $poke))
          (alias core export $i "read" (core func $read))
          (func (export "poke") (type $poke-type)
            (canon lift (core func $poke)))
          (func (export "read") (type $read-type)
            (canon lift (core func $read))))
        "#,
    )
    .expect("component wat");
    let calls = [("poke", vec![ComponentVal::S32(41)]), ("read", vec![])];
    let run =
        neo_wasm::collect_wasmtime_component_run_calls(&component_bytes, &calls, |_| Ok(())).expect("two-turn run");
    assert_eq!(run.results, ["41"]);

    // Resolve both frefs from single-call raw runs.
    let fref_of = |export: &str, args: &[ComponentVal]| {
        let single =
            neo_wasm::collect_wasmtime_component_run_with_linker_and_args(&component_bytes, export, args, |_| Ok(()))
                .expect("single run");
        neo_wasm::traces_from_wasmtime_steps(&single.steps)
            .expect("raw trace")
            .first()
            .expect("rows")
            .current_function_ref
    };
    let poke_fref = fref_of("poke", &[ComponentVal::S32(1)]);
    let read_fref = fref_of("read", &[]);

    let mut grammar = HostEventGrammar::default();
    grammar.exports.insert(
        poke_fref,
        ExportTemplate {
            entry: vec![GrammarEvent::op(
                8,
                slots(&[(
                    0,
                    SlotSource::ClaimLocal {
                        idx: 0,
                        local: 0,
                        limb: Limb::Lo,
                    },
                )]),
            )],
            exit: vec![GrammarEvent::op(16, slots(&[]))],
            entry_claim_count: 1,
            exit_claim_count: 0,
        },
    );
    grammar.exports.insert(
        read_fref,
        ExportTemplate {
            entry: vec![GrammarEvent::op(9, slots(&[]))],
            exit: vec![GrammarEvent::op(
                17,
                slots(&[(0, SlotSource::OutputElem { limb: Limb::Lo })]),
            )],
            entry_claim_count: 0,
            exit_claim_count: 0,
        },
    );
    let turns = [
        TurnClaims {
            entry: vec![41],
            exit: vec![],
            ..Default::default()
        },
        TurnClaims::default(),
    ];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &turns, Default::default())
        .expect("resultless-then-value trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

    let tb = trace
        .iter()
        .find(|row| row.row_kind.is_turn_boundary())
        .expect("boundary row");
    assert!(!tb.state_before.output.enabled);
    assert!(!tb.state_after.output.enabled);
    assert!(tb.state_before.halted);
    assert!(!tb.state_after.halted);
    assert_eq!(tb.state_before.grammar.turn_export_fref, poke_fref);
    assert_eq!(tb.state_after.grammar.turn_export_fref, read_fref);

    let event_metadata: Vec<_> = neo_wasm::comm_chain::absorbed_event_blocks(&trace)
        .into_iter()
        .map(|event| (event.metadata.attributed_fref, event.metadata.turn_export_fref))
        .collect();
    assert_eq!(
        event_metadata,
        [
            (poke_fref, poke_fref),
            (poke_fref, poke_fref),
            (read_fref, read_fref),
            (read_fref, read_fref),
        ]
    );

    let mut blocks =
        neo_wasm::event_grammar::expand_export_entry(&grammar.exports[&poke_fref], &[41]).expect("poke entry");
    blocks.extend(
        neo_wasm::event_grammar::expand_export_exit(&grammar.exports[&poke_fref], None, &[], &[])
            .expect("resultless poke exit"),
    );
    blocks.extend(neo_wasm::event_grammar::expand_export_entry(&grammar.exports[&read_fref], &[]).expect("read entry"));
    blocks.extend(
        neo_wasm::event_grammar::expand_export_exit(&grammar.exports[&read_fref], Some((41, 0)), &[], &[])
            .expect("exit"),
    );
    let lifted: Vec<[p3_goldilocks::Goldilocks; 8]> = blocks
        .into_iter()
        .map(|block| block.map(p3_goldilocks::Goldilocks::from_u64))
        .collect();
    let final_state = common::final_state(&trace);
    assert_eq!((final_state.output.value_lo, final_state.output.enabled), (41, true));
    assert_eq!(
        final_state.comm_chain,
        neo_wasm::comm_chain::fold_event_blocks(Default::default(), &lifted).canonical_u64()
    );

    // Resultless exits may not reference a captured output.
    let mut bad_grammar = grammar.clone();
    bad_grammar.exports.get_mut(&poke_fref).expect("poke").exit = vec![GrammarEvent::op(
        17,
        slots(&[(0, SlotSource::OutputElem { limb: Limb::Lo })]),
    )];
    assert!(
        neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &bad_grammar, &turns, Default::default())
            .is_err(),
        "output-dependent exit events on a resultless turn must be rejected"
    );
}
