//! Grammar-mode traces: the chain absorbs embedder grammar events staged by
//! `HostEventGather` slot rows (8 per block, one word each) instead of raw
//! host-call records. Every row is CCS-checked, the grammar ROM content is
//! checked by the native memory-rows pass, and the rejection tests cover
//! mode gating, gather forgery, and the event schedule.

mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::event_grammar::{GrammarEvent, HostEventGrammar, ImportTemplate, Limb, MemoryBase, SlotSource};
use neo_wasm::layout::{COL_GATHER_ACTIVE, COL_GRAMMAR_MODE_AFTER, COL_RAW_HOST_CALL};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{WasmGrammarSlotKind, WasmOpcode, WasmRowKind, WasmVmStep};
use p3_field::PrimeCharacteristicRing;

const ZERO: SlotSource = SlotSource::Const(0);

fn slots(entries: &[(usize, SlotSource)]) -> [SlotSource; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
}

/// Example embedder grammar for the mul/sink component: `mul(x, y) -> r`
/// expands to a two-event template (args event + result event referencing a
/// shared claim word), `sink(x)` to a single event.
fn test_grammar(mul_fref: u32, sink_fref: u32) -> HostEventGrammar {
    let arg = |arg, limb| SlotSource::ArgElem { arg, limb };
    let mut grammar = HostEventGrammar::default();
    grammar.imports.insert(
        mul_fref,
        ImportTemplate {
            events: vec![
                GrammarEvent::op(
                    10,
                    slots(&[
                        (0, SlotSource::Claim { idx: 0 }),
                        (1, arg(0, Limb::Lo)),
                        (2, arg(1, Limb::Lo)),
                        (3, SlotSource::Const(5)),
                    ]),
                ),
                // The ResultElem Lo slot is the gather row that pushes the
                // host result onto the operand stack; the Hi slot binds the
                // pushed hi lane (0 for the i32 result).
                GrammarEvent::op(
                    12,
                    slots(&[
                        (0, SlotSource::ResultElem { limb: Limb::Lo }),
                        (1, SlotSource::Claim { idx: 0 }),
                        (2, SlotSource::ResultElem { limb: Limb::Hi }),
                    ]),
                ),
            ],
            claim_count: 1,
        },
    );
    grammar.imports.insert(
        sink_fref,
        ImportTemplate {
            events: vec![GrammarEvent::op(7, slots(&[(0, arg(0, Limb::Lo))]))],
            claim_count: 0,
        },
    );
    grammar
}

fn mul_sink_component_wat() -> &'static str {
    r#"
    (component
      (type $host-mul (func (param "x" s32) (param "y" s32) (result s32)))
      (type $host-sink (func (param "x" s32)))
      (type $run-type (func (result s32)))
      (import "host-mul" (func $host-mul (type $host-mul)))
      (import "host-sink" (func $host-sink (type $host-sink)))
      (core module $m
        (import "" "0" (func $mul (param i32 i32) (result i32)))
        (import "" "1" (func $sink (param i32)))
        (func (export "run") (result i32)
          (local i32)
          i32.const 7
          i32.const 6
          call $mul
          local.tee 0
          call $sink
          local.get 0))
      (core func $lowered-mul (canon lower (func $host-mul)))
      (core func $lowered-sink (canon lower (func $host-sink)))
      (core instance $lowered-host
        (export "0" (func $lowered-mul))
        (export "1" (func $lowered-sink)))
      (core instance $i
        (instantiate $m
          (with "" (instance $lowered-host))))
      (alias core export $i "run" (core func $run))
      (func (export "run") (type $run-type)
        (canon lift (core func $run))))
    "#
}

/// Run the two-call component; the mul host function records `mul_claims`
/// for its in-flight call (the grammar hand-off path), sink records nothing.
fn run_component_with_mul_claims(mul_claims: &'static [u64]) -> neo_wasm::WasmtimeTraceRun {
    let component_bytes = wat::parse_str(mul_sink_component_wat()).expect("component wat");
    neo_wasm::collect_wasmtime_component_run_with_linker(&component_bytes, "run", |linker| {
        linker
            .root()
            .func_wrap("host-mul", move |mut store, (x, y): (i32, i32)| {
                store.data_mut().record_call_claims(mul_claims)?;
                Ok((x * y,))
            })
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-mul: {err}")))?;
        linker
            .root()
            .func_wrap("host-sink", |_store, (_x,): (i32,)| Ok(()))
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-sink: {err}")))
    })
    .expect("component trace run")
}

fn run_component() -> neo_wasm::WasmtimeTraceRun {
    run_component_with_mul_claims(&[100])
}

fn host_call_frefs(trace: &[WasmVmStep]) -> Vec<u32> {
    trace
        .iter()
        .filter(|row| {
            row.row_kind.is_program()
                && matches!(row.opcode, neo_wasm::WasmOpcode::Call)
                && !row.target_function_is_guest
        })
        .map(|row| row.state_after.host_callee_fref)
        .collect()
}

/// Grammar trace for the two-call component, with claim words `[100]` for mul
/// and `[]` for sink. The invoked export gets an empty boundary template
/// (required in grammar mode; no boundary events for this test).
fn grammar_trace() -> Vec<WasmVmStep> {
    grammar_trace_from(Default::default())
}

fn grammar_trace_from(initial_comm_chain: neo_wasm::CommChainState) -> Vec<WasmVmStep> {
    let run = run_component();
    // Resolve frefs from a raw normalization of the same run.
    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let frefs = host_call_frefs(&raw);
    assert_eq!(frefs.len(), 2);
    let mut grammar = test_grammar(frefs[0], frefs[1]);
    let export_fref = raw
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row")
        .current_function_ref;
    grammar
        .exports
        .insert(export_fref, neo_wasm::event_grammar::ExportTemplate::default());
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &grammar,
        &[Default::default()],
        initial_comm_chain,
    )
    .expect("grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

    // The claimed grammar-ROM entries must match the embedder tables.
    let component_bytes = wat::parse_str(mul_sink_component_wat()).expect("component wat");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &run.initial_locals);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &grammar);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("grammar ROM contents match");
    trace
}

/// An i64-returning import: each result lane is written by the slot that
/// absorbs it — the Lo gather row pushes (a narrow total write, hi lane
/// pinned to zero), the Hi row writes only the pushed cell's hi word.
/// Forging either write's address or value, suppressing either, or
/// smuggling advice into the Lo row's hi lane is CCS-rejected.
#[test]
fn i64_result_lane_writes() {
    let component_bytes = wat::parse_str(
        r#"
        (component
          (type $host-add64 (func (param "x" s64) (param "y" s64) (result s64)))
          (type $run-type (func (result s64)))
          (import "host-add64" (func $host-add64 (type $host-add64)))
          (core module $m
            (type $host-ty (func (param i64 i64) (result i64)))
            (import "" "0" (func $host-add64-core (type $host-ty)))
            (func (export "run") (result i64)
              i64.const 4294967296
              i64.const 8589934592
              call $host-add64-core))
          (core func $lowered (canon lower (func $host-add64)))
          (core instance $lowered-host
            (export "0" (func $lowered)))
          (core instance $i
            (instantiate $m
              (with "" (instance $lowered-host))))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type)
            (canon lift (core func $run))))
        "#,
    )
    .expect("component wat");
    let run = neo_wasm::collect_wasmtime_component_run_with_linker(&component_bytes, "run", |linker| {
        linker
            .root()
            .func_wrap("host-add64", |_store, (x, y): (i64, i64)| Ok((x + y,)))
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-add64: {err}")))
    })
    .expect("component run");
    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let frefs = host_call_frefs(&raw);
    assert_eq!(frefs.len(), 1);
    let export_fref = raw
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row")
        .current_function_ref;
    let arg = |arg, limb| SlotSource::ArgElem { arg, limb };
    let mut grammar = HostEventGrammar::default();
    grammar.imports.insert(
        frefs[0],
        ImportTemplate {
            events: vec![
                GrammarEvent::op(
                    3,
                    slots(&[
                        (0, arg(0, Limb::Lo)),
                        (1, arg(0, Limb::Hi)),
                        (2, arg(1, Limb::Lo)),
                        (3, arg(1, Limb::Hi)),
                    ]),
                ),
                GrammarEvent::op(
                    4,
                    slots(&[
                        (0, SlotSource::ResultElem { limb: Limb::Lo }),
                        (1, SlotSource::ResultElem { limb: Limb::Hi }),
                    ]),
                ),
            ],
            claim_count: 0,
        },
    );
    grammar
        .exports
        .insert(export_fref, neo_wasm::event_grammar::ExportTemplate::default());
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &grammar,
        &[Default::default()],
        Default::default(),
    )
    .expect("grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &run.initial_locals);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &grammar);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("grammar ROM contents match");

    // 2^32 + 2^33 = 3·2^32: the result lives entirely in the hi limb.
    let lo_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .grammar_rom_slot
                    .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::Result && rom.variant.is_low_limb())
        })
        .expect("result lo slot row");
    let write = lo_row.stack_write0.expect("result push");
    // The lo slot is a narrow TOTAL write: hi lane pinned to zero.
    assert_eq!((write.value_lo, write.value_hi), (0, Some(0)));
    assert_eq!(lo_row.state_after.sp, lo_row.state_before.sp + 1, "the lo slot pushes");

    let witness = build_witness_vector(lo_row);
    common::assert_satisfied(&witness, "untampered result-lo gather row");
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_STACK_WRITE0_ADDR_LO] += neo_math::F::from_u64(2);
    forged[neo_wasm::layout::COL_STACK_WRITE0_ADDR_HI] += neo_math::F::from_u64(2);
    common::assert_rejected(&forged, "result push redirected to a different stack slot");
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_STACK_WRITE0_VALUE_LO] += neo_math::F::ONE;
    common::assert_rejected(&forged, "pushed value diverging from the absorbed word");
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_STACK_WRITES] = neo_math::F::ZERO;
    forged[neo_wasm::layout::COL_STACK_WRITE0_ACTIVE] = neo_math::F::ZERO;
    forged[neo_wasm::layout::COL_STACK_WRITE0_HI_ACTIVE] = neo_math::F::ZERO;
    common::assert_rejected(&forged, "result-lo row suppressing the push");
    // The hi lane can never carry advice on the lo row (the old P1 shape).
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_WIDE_VALUES_ENABLED] = neo_math::F::ONE;
    forged[neo_wasm::layout::COL_STACK_WRITE0_VALUE_HI] = neo_math::F::from_u64(0xdead);
    common::assert_rejected(&forged, "result-lo row smuggling advice into the hi lane");

    let hi_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .grammar_rom_slot
                    .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::Result && rom.variant.is_high_limb())
        })
        .expect("result hi slot row");
    assert_eq!(
        hi_row.state_after.sp, hi_row.state_before.sp,
        "the hi slot writes without pushing"
    );
    let hi_write = hi_row.stack_write0.expect("hi lane write");
    assert_eq!(hi_write.value_hi, Some(3));
    let witness = build_witness_vector(hi_row);
    common::assert_satisfied(&witness, "untampered result-hi gather row");
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_STACK_WRITE0_ADDR_LO] += neo_math::F::from_u64(2);
    forged[neo_wasm::layout::COL_STACK_WRITE0_ADDR_HI] += neo_math::F::from_u64(2);
    common::assert_rejected(&forged, "hi-lane write redirected to a different stack slot");
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_STACK_WRITE0_VALUE_HI] += neo_math::F::ONE;
    common::assert_rejected(&forged, "hi-lane value diverging from the absorbed word");
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_STACK_WRITE0_HI_ACTIVE] = neo_math::F::ZERO;
    common::assert_rejected(&forged, "result-hi row suppressing its lane write");
}

/// Advice events keep their VM effects without changing the transcript.
#[test]
fn advice_import_pushes_without_absorbing() {
    let run = run_component_with_mul_claims(&[]);
    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let frefs = host_call_frefs(&raw);
    let export_fref = raw
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row")
        .current_function_ref;
    // `mul` is advice; `sink` remains transcript-bound.
    let arg = |arg, limb| SlotSource::ArgElem { arg, limb };
    let mut grammar = HostEventGrammar::default();
    let mut advice_block = [SlotSource::Const(0); 8];
    advice_block[0] = SlotSource::ResultElem { limb: Limb::Lo };
    advice_block[1] = SlotSource::ResultElem { limb: Limb::Hi };
    grammar.imports.insert(
        frefs[0],
        ImportTemplate {
            events: vec![GrammarEvent::advice(advice_block)],
            claim_count: 0,
        },
    );
    grammar.imports.insert(
        frefs[1],
        ImportTemplate {
            events: vec![GrammarEvent::op(7, slots(&[(0, arg(0, Limb::Lo))]))],
            claim_count: 0,
        },
    );
    grammar
        .exports
        .insert(export_fref, neo_wasm::event_grammar::ExportTemplate::default());
    let turns = [neo_wasm::event_grammar::TurnClaims::default()];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &grammar,
        &turns,
        Default::default(),
    )
    .expect("grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

    let component_bytes = wat::parse_str(mul_sink_component_wat()).expect("component wat");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &run.initial_locals);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &grammar);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("grammar ROM contents match");

    // Only `sink` contributes to the chain.
    let f = p3_goldilocks::Goldilocks::from_u64;
    let expected = neo_wasm::comm_chain::commit_event(
        [p3_goldilocks::Goldilocks::ZERO; 4],
        f(7),
        core::array::from_fn(|i| if i == 0 { f(42) } else { f(0) }),
    );
    assert_eq!(
        trace.last().expect("rows").state_after.comm_chain,
        expected.map(|limb| p3_field::PrimeField64::as_canonical_u64(&limb))
    );

    // Extraction sees exactly the committed stream: sink's block, no advice.
    let events = neo_wasm::comm_chain::absorbed_event_blocks(&trace);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].words, [7, 42, 0, 0, 0, 0, 0, 0]);
    assert_eq!(events[0].metadata.attributed_fref, frefs[1]);

    let advice_rows: Vec<&neo_wasm::WasmVmStep> = trace
        .iter()
        .filter(|row| row.row_kind.is_host_event_gather() && row.grammar_rom_slot.is_some_and(|rom| rom.advice))
        .collect();
    assert_eq!(advice_rows.len(), 8, "one advice event = 8 gather rows");
    assert!(
        advice_rows
            .iter()
            .all(|row| !row.state_after.event_absorb.perm_pending),
        "advice rows never raise pending"
    );
    let lo_row = advice_rows
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::Result && rom.variant.is_low_limb())
        })
        .expect("advice result-lo row");
    assert_eq!(lo_row.stack_write0.expect("push").value_lo, 42);
    assert_eq!(lo_row.state_after.sp, lo_row.state_before.sp + 1);

    // The final advice row cannot start a permutation or shed its ROM flag.
    let word7 = advice_rows.last().expect("word 7");
    let witness = build_witness_vector(word7);
    common::assert_satisfied(&witness, "untampered advice word-7 row");
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_PERM_PENDING_AFTER] = neo_math::F::ONE;
    common::assert_rejected(&forged, "advice row absorbing its block");
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_GRAMMAR_SLOT_KIND] -= neo_math::F::from_u64(8);
    common::assert_rejected(&forged, "advice row shedding the advice flag");
}

#[test]
fn grammar_trace_folds_expanded_blocks() {
    let trace = grammar_trace();
    assert!(trace.iter().all(|row| row.state_before.grammar_mode));

    // Three grammar events → three completed blocks (each staged by 8 slot
    // rows; the one raising `pending` holds the full block).
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
            [10, 100, 7, 6, 5, 0, 0, 0],  // mul pre-result event
            [12, 42, 100, 0, 0, 0, 0, 0], // mul post-result event
            [7, 42, 0, 0, 0, 0, 0, 0],    // sink event
        ],
    );

    // The carried chain equals the fold of exactly those blocks.
    let f = p3_goldilocks::Goldilocks::from_u64;
    let mut chain = [p3_goldilocks::Goldilocks::ZERO; 4];
    for block in &staged {
        chain = neo_wasm::comm_chain::commit_event(chain, f(block[0]), core::array::from_fn(|i| f(block[1 + i])));
    }
    let final_chain = trace.last().expect("rows").state_after.comm_chain;
    assert_eq!(
        final_chain,
        chain.map(|limb| p3_field::PrimeField64::as_canonical_u64(&limb))
    );
}

#[test]
fn grammar_trace_folds_from_explicit_initial_state() {
    let f = p3_goldilocks::Goldilocks::from_u64;
    let initial = neo_wasm::CommChainState::new([f(11), f(22), f(33), f(44)]);
    let trace = grammar_trace_from(initial);
    let staged: Vec<[p3_goldilocks::Goldilocks; 8]> = trace
        .iter()
        .filter(|row| {
            row.row_kind.is_host_event_gather()
                && row.state_after.event_absorb.perm_pending
                && !row.state_before.event_absorb.perm_pending
        })
        .map(|row| row.state_after.event_absorb.evbuf.map(f))
        .collect();

    assert_eq!(trace[0].state_before.comm_chain, initial.canonical_u64());
    assert_eq!(
        trace.last().expect("rows").state_after.comm_chain,
        neo_wasm::comm_chain::fold_event_blocks(initial, &staged).canonical_u64()
    );
}

#[test]
fn missing_template_is_rejected() {
    let run = run_component();
    let grammar = HostEventGrammar::default();
    assert!(neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &grammar,
        &[Default::default()],
        Default::default(),
    )
    .is_err());
}

/// A host call recording more claim words than its template consumes
/// indicates a misaligned hand-off and is rejected.
#[test]
fn surplus_claim_words_are_rejected() {
    let run = run_component_with_mul_claims(&[100, 7]);
    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let frefs = host_call_frefs(&raw);
    let mut grammar = test_grammar(frefs[0], frefs[1]);
    let export_fref = raw
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row")
        .current_function_ref;
    grammar
        .exports
        .insert(export_fref, neo_wasm::event_grammar::ExportTemplate::default());
    assert!(neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &grammar,
        &[Default::default()],
        Default::default(),
    )
    .is_err());
}

/// The raw absorb machinery must stay de-gated in grammar mode: grammar
/// traces have no host arg/result aux rows at all (the call row pops the
/// args, a gather row pushes the result), and forging the raw host-call
/// mask back on is CCS-rejected.
#[test]
fn ccs_rejects_raw_machinery_in_grammar_mode() {
    let trace = grammar_trace();
    assert!(
        !trace.iter().any(|row| matches!(
            row.row_kind,
            WasmRowKind::Aux(neo_wasm::WasmAuxOpcode::HostCallArg | neo_wasm::WasmAuxOpcode::HostCallResult)
        )),
        "grammar traces must not contain raw host aux rows"
    );
    let call_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_program()
                && matches!(row.opcode, neo_wasm::WasmOpcode::Call)
                && !row.target_function_is_guest
        })
        .expect("host-call row");
    let mut witness = build_witness_vector(call_row);
    common::assert_satisfied(&witness, "untampered grammar host-call row");
    witness[COL_RAW_HOST_CALL] = neo_math::F::ONE;
    common::assert_rejected(&witness, "grammar host-call row with the raw machinery forged on");
}

/// Gather rows only exist in grammar mode: claiming one on a raw trace row
/// is CCS-rejected.
#[test]
fn ccs_rejects_gather_row_in_raw_mode() {
    let run = run_component();
    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let arg_row = raw
        .iter()
        .find(|row| row.row_kind == WasmRowKind::Aux(neo_wasm::WasmAuxOpcode::HostCallArg))
        .expect("arg row");
    let mut witness = build_witness_vector(arg_row);
    common::assert_satisfied(&witness, "untampered raw arg row");
    witness[COL_GATHER_ACTIVE] = neo_math::F::ONE;
    common::assert_rejected(&witness, "raw row claiming the gather kind");
}

/// The mode flag is a carried constant: flipping it mid-trace is rejected.
#[test]
fn ccs_rejects_mode_flip() {
    let trace = grammar_trace();
    let mut witness = build_witness_vector(&trace[0]);
    common::assert_satisfied(&witness, "untampered grammar row");
    witness[COL_GRAMMAR_MODE_AFTER] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "row flipping the per-program mode constant");
}

/// A gather row staging a word that contradicts its (honest) ROM entry is
/// CCS-rejected: here the constant discriminant word is forged.
#[test]
fn ccs_rejects_forged_gather_word() {
    let trace = grammar_trace();
    let disc_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .grammar_rom_slot
                    .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::Const)
                && row.state_before.grammar.slot_cursor == 0
        })
        .expect("discriminant slot row");
    let mut witness = build_witness_vector(disc_row);
    common::assert_satisfied(&witness, "untampered discriminant slot row");
    witness[neo_wasm::layout::COL_EVBUF0_AFTER] += neo_math::F::ONE;
    common::assert_rejected(&witness, "gather row staging a forged discriminant");
}

/// An arg-slot gather row must read the table-pinned stack address.
#[test]
fn ccs_rejects_redirected_gather_read() {
    let trace = grammar_trace();
    let arg_slot_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .grammar_rom_slot
                    .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::Arg)
        })
        .expect("arg slot row");
    let mut witness = build_witness_vector(arg_slot_row);
    common::assert_satisfied(&witness, "untampered arg slot row");
    witness[neo_wasm::layout::COL_STACK_READ0_ADDR_LO] += neo_math::F::from_u64(2);
    common::assert_rejected(&witness, "arg slot row reading a different stack slot");
}

/// Forging the claimed ROM entry itself is caught by the grammar-ROM
/// content check (the native stand-in for the lookup argument).
#[test]
fn memory_rows_reject_forged_rom_claim() {
    let run = run_component();
    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let frefs = host_call_frefs(&raw);
    let mut grammar = test_grammar(frefs[0], frefs[1]);
    let export_fref = raw
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row")
        .current_function_ref;
    grammar
        .exports
        .insert(export_fref, neo_wasm::event_grammar::ExportTemplate::default());
    let mut trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &grammar,
        &[Default::default()],
        Default::default(),
    )
    .expect("grammar trace");

    let idx = trace
        .iter()
        .position(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .grammar_rom_slot
                    .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::Const)
        })
        .expect("const slot row");
    if let Some(rom) = &mut trace[idx].grammar_rom_slot {
        rom.const_lo ^= 1;
    }

    let component_bytes = wat::parse_str(mul_sink_component_wat()).expect("component wat");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &run.initial_locals);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &grammar);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    assert!(
        neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload).is_err(),
        "a forged grammar-ROM claim must fail the content check"
    );
}

/// The event schedule is forced: a program row cannot leave grammar events
/// unabsorbed, and a gather row cannot run with none owed.
#[test]
fn ccs_rejects_broken_event_schedule() {
    let trace = grammar_trace();

    let program_row = trace
        .iter()
        .find(|row| row.row_kind.is_program() && !row.state_before.grammar_mode == false)
        .expect("program row");
    let mut witness = build_witness_vector(program_row);
    common::assert_satisfied(&witness, "untampered program row");
    witness[neo_wasm::layout::COL_GRAMMAR_EVREM_BEFORE] = neo_math::F::ONE;
    witness[neo_wasm::layout::COL_GRAMMAR_EVREM_BEFORE_IS_ZERO] = neo_math::F::ZERO;
    witness[neo_wasm::layout::COL_GRAMMAR_EVREM_BEFORE_INV] = neo_math::F::ONE;
    common::assert_rejected(&witness, "program row with grammar events still owed");

    let gather_row = trace
        .iter()
        .find(|row| row.row_kind.is_host_event_gather())
        .expect("gather row");
    let mut witness = build_witness_vector(gather_row);
    common::assert_satisfied(&witness, "untampered gather row");
    witness[neo_wasm::layout::COL_GRAMMAR_EVREM_BEFORE] = neo_math::F::ZERO;
    witness[neo_wasm::layout::COL_GRAMMAR_EVREM_BEFORE_IS_ZERO] = neo_math::F::ONE;
    witness[neo_wasm::layout::COL_GRAMMAR_EVREM_BEFORE_INV] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "gather row with no grammar events owed");
}

/// An import with no template reads the zero-filled biased count cell, so the
/// only row-locally satisfiable assignment loads the poisoned EVREM = -1.
#[test]
fn ccs_forces_untemplated_import_into_poisoned_schedule() {
    let trace = grammar_trace();
    let call_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_program()
                && matches!(row.opcode, neo_wasm::WasmOpcode::Call)
                && !row.target_function_is_guest
        })
        .expect("host-call row");
    let witness = build_witness_vector(call_row);
    common::assert_satisfied(&witness, "untampered host-call row");

    // An undeclared import's cell is 0; a normal schedule can't load from it.
    let mut forged = witness.clone();
    forged[neo_wasm::layout::COL_GRAMMAR_PRE_COUNT] = neo_math::F::ZERO;
    common::assert_rejected(&forged, "untemplated import call claiming a normal schedule");

    // The poisoned schedule satisfies the row itself. The composed circuit's
    // grammar-ROM address bound prevents enough blocks from draining it; see
    // the count-family relation-layout comment for the full argument.
    let mut poisoned = forged.clone();
    poisoned[neo_wasm::layout::COL_GRAMMAR_EVREM_AFTER] = -neo_math::F::ONE;
    common::assert_satisfied(&poisoned, "untemplated import call loads the poisoned schedule");
}

/// Import pre-counts and export entry-counts are separate ROM families
/// (with the +1 presence bias), so a turn boundary or exit latch can never
/// read an import's cell and vice versa.
#[test]
fn count_families_are_split_and_biased() {
    let run = run_component();
    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let frefs = host_call_frefs(&raw);
    let mut grammar = test_grammar(frefs[0], frefs[1]);
    let export_fref = raw
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row")
        .current_function_ref;
    grammar
        .exports
        .insert(export_fref, neo_wasm::event_grammar::ExportTemplate::default());

    let mut preload = neo_wasm::memory_semantics::WasmMemoryPreload::default();
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &grammar);
    let cell = |family: &str, fref: u32| {
        preload
            .entries()
            .into_iter()
            .find(|(memory, address, _)| *memory == family && address == &[fref])
            .map(|(_, _, value)| value)
    };
    for (&fref, template) in &grammar.imports {
        assert_eq!(
            cell("grammar_import_pre_counts", fref),
            Some(template.events.len() as u32 + 1),
            "import cells live in the import family, biased"
        );
        assert_eq!(
            cell("grammar_export_entry_counts", fref),
            None,
            "imports must have no export entry-count cell"
        );
    }
    assert_eq!(
        cell("grammar_export_entry_counts", export_fref),
        Some(1),
        "the export's zero-event entry template is the biased 1, distinct from the zero-filled 0"
    );
    assert_eq!(
        cell("grammar_import_pre_counts", export_fref),
        None,
        "exports must have no import pre-count cell"
    );
}

/// Claim slots are free absorbed words: staging a different value
/// satisfies the per-row CCS (there is deliberately no local binding) but
/// diverges the chain, so the transcript check rejects the claim. The
/// same-index identity lives in claim construction: expansion resolves
/// every `Claim{idx}` from one claim entry.
#[test]
fn claim_words_are_row_free_and_transcript_bound() {
    let trace = grammar_trace();
    let claim_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .grammar_rom_slot
                    .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::Claim)
        })
        .expect("claim slot row");
    let mut witness = build_witness_vector(claim_row);
    common::assert_satisfied(&witness, "untampered claim slot row");
    // Forge the claim word consistently in the staged buffer and the
    // gadget's slot value: the row still satisfies (free word) ...
    let cursor = usize::from(claim_row.state_before.grammar.slot_cursor);
    witness[neo_wasm::layout::COL_EVBUF0_AFTER + cursor] += neo_math::F::ONE;
    common::assert_rejected(&witness, "buffer word diverging from the staged slot value");
    // ... but any divergence between the absorbed words and the claimed
    // transcript is caught by the final-chain fold (see
    // wasm_grammar_lifecycle's verify_with_transcript rejection).
}

#[test]
fn import_memory_accesses_use_argument_based_addresses() {
    let component_bytes = wat::parse_str(
        r#"
        (component
          (type $host-touch (func (param "ptr" s32)))
          (type $run-type (func))
          (import "host-touch" (func $host-touch (type $host-touch)))
          (core module $m
            (import "" "0" (func $touch (param i32)))
            (memory 1)
            (data (i32.const 24) "\7b\00\00\00")
            (func (export "run")
              i32.const 16
              i32.const 99
              i32.store
              i32.const 19
              i32.const 0x1234
              i32.store16
              i32.const 16
              call $touch))
          (core func $lowered (canon lower (func $host-touch)))
          (core instance $host (export "0" (func $lowered)))
          (core instance $i (instantiate $m (with "" (instance $host))))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type)
            (canon lift (core func $run))))
        "#,
    )
    .expect("component wat");
    let run = neo_wasm::collect_wasmtime_component_run_with_linker(&component_bytes, "run", |linker| {
        linker
            .root()
            .func_wrap("host-touch", |mut store, (_ptr,): (i32,)| {
                store.data_mut().record_call_claims(&[77])?;
                Ok(())
            })
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-touch: {err}")))
    })
    .expect("component run");
    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let host_fref = host_call_frefs(&raw)[0];
    let export_fref = raw
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row")
        .current_function_ref;
    let mut grammar = HostEventGrammar::default();
    grammar.imports.insert(
        host_fref,
        ImportTemplate {
            events: vec![
                GrammarEvent::op(
                    40,
                    slots(&[
                        (
                            0,
                            SlotSource::MemoryRead32 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 0,
                            },
                        ),
                        (
                            1,
                            SlotSource::MemoryRead32 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 4,
                            },
                        ),
                        (
                            2,
                            SlotSource::MemoryWrite32 {
                                claim: 0,
                                base: MemoryBase::Arg(0),
                                byte_offset: 4,
                            },
                        ),
                        (
                            3,
                            SlotSource::MemoryRead32 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 4,
                            },
                        ),
                        (
                            4,
                            SlotSource::MemoryRead32 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 8,
                            },
                        ),
                    ]),
                ),
                GrammarEvent::op(
                    41,
                    slots(&[
                        (
                            0,
                            SlotSource::MemoryRead8 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 0,
                            },
                        ),
                        (
                            1,
                            SlotSource::MemoryRead8 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 1,
                            },
                        ),
                        (
                            2,
                            SlotSource::MemoryRead8 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 2,
                            },
                        ),
                        (
                            3,
                            SlotSource::MemoryRead8 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 3,
                            },
                        ),
                        (
                            4,
                            SlotSource::MemoryWrite8 {
                                claim: 0,
                                base: MemoryBase::Arg(0),
                                byte_offset: 2,
                            },
                        ),
                        (
                            5,
                            SlotSource::MemoryRead8 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 2,
                            },
                        ),
                    ]),
                ),
                GrammarEvent::op(
                    42,
                    slots(&[
                        (
                            0,
                            SlotSource::MemoryRead16 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 0,
                            },
                        ),
                        (
                            1,
                            SlotSource::MemoryRead16 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 2,
                            },
                        ),
                        (
                            2,
                            SlotSource::MemoryWrite16 {
                                claim: 0,
                                base: MemoryBase::Arg(0),
                                byte_offset: 2,
                            },
                        ),
                        (
                            3,
                            SlotSource::MemoryRead16 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 2,
                            },
                        ),
                        (
                            4,
                            SlotSource::MemoryRead16 {
                                base: MemoryBase::Arg(0),
                                byte_offset: 10,
                            },
                        ),
                    ]),
                ),
            ],
            claim_count: 1,
        },
    );
    grammar
        .exports
        .insert(export_fref, neo_wasm::event_grammar::ExportTemplate::default());

    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &grammar,
        &[Default::default()],
        Default::default(),
    )
    .expect("grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &run.initial_locals);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &grammar);
    let witnesses: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect("grammar argument base and linear-memory accesses match");

    let observed_word_reads: Vec<u32> = trace
        .iter()
        .filter_map(|row| {
            (row.grammar_rom_slot?.kind == WasmGrammarSlotKind::MemoryRead && row.linear_memory?.width_bytes == 4)
                .then_some(row.linear_memory?.lane0.value_before)
        })
        .collect();
    assert_eq!(observed_word_reads, [0x3400_0063, 0x12, 77, 123]);

    let observed_byte_reads: Vec<u8> = trace
        .iter()
        .filter_map(|row| {
            let access = row.linear_memory?;
            (row.grammar_rom_slot?.kind == WasmGrammarSlotKind::MemoryRead && access.width_bytes == 1)
                .then_some(access.lane0.value_before.to_le_bytes()[usize::from(access.byte_offset)])
        })
        .collect();
    assert_eq!(observed_byte_reads, [99, 0, 0, 0x34, 77]);

    let observed_half_reads: Vec<u16> = trace
        .iter()
        .filter_map(|row| {
            let access = row.linear_memory?;
            if row.grammar_rom_slot?.kind != WasmGrammarSlotKind::MemoryRead || access.width_bytes != 2 {
                return None;
            }
            let bytes = access.lane0.value_before.to_le_bytes();
            let offset = usize::from(access.byte_offset);
            Some(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
        })
        .collect();
    assert_eq!(observed_half_reads, [99, 0x344d, 77, 0]);

    let mut misaligned_grammar = grammar.clone();
    let half_read = misaligned_grammar
        .imports
        .get_mut(&host_fref)
        .expect("host template")
        .events
        .iter_mut()
        .flat_map(|event| &mut event.block)
        .find_map(|slot| match slot {
            SlotSource::MemoryRead16 { byte_offset, .. } => Some(byte_offset),
            _ => None,
        })
        .expect("half-word read");
    *half_read = 1;
    let err = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &misaligned_grammar,
        &[Default::default()],
        Default::default(),
    )
    .expect_err("misaligned grammar half-word access must be rejected");
    assert!(err.to_string().contains("is not naturally aligned"));

    let mut high_pointer_steps = run.steps.clone();
    let host_call = high_pointer_steps
        .iter_mut()
        .find(|step| step.opcode_decoded == Some(WasmOpcode::Call) && !step.target_function_is_guest)
        .expect("host call step");
    host_call
        .operand_stack_words_hi
        .resize(host_call.operand_stack_words.len(), 0);
    *host_call
        .operand_stack_words_hi
        .last_mut()
        .expect("pointer argument high limb") = 1;
    let err = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &high_pointer_steps,
        &run.program_tables,
        &grammar,
        &[Default::default()],
        Default::default(),
    )
    .expect_err("wasm32 grammar pointer with a high limb must be rejected");
    assert!(err.to_string().contains("not a wasm32 pointer"));

    let read = trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryRead)
        })
        .expect("memory read gather");
    let mut forged = build_witness_vector(read);
    common::assert_satisfied(&forged, "untampered argument-base memory read");
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE0_ADDR] += neo_math::F::ONE;
    common::assert_rejected(&forged, "grammar memory read redirected to another word");

    let mut high_pointer = read.clone();
    high_pointer.wide_values_enabled = true;
    high_pointer
        .stack_read0
        .as_mut()
        .expect("pointer argument read")
        .value_hi = Some(1);
    let forged = build_witness_vector(&high_pointer);
    common::assert_rejected(&forged, "grammar memory read with a nonzero pointer high limb");

    // Move both the authenticated pointer value and its derived word address
    // to the first lane beyond memory. The witness builder recomputes the
    // comparison columns, so this exercises the grammar's no-OOB semantics
    // rather than failing the pointer/address identity above.
    let mut oob_read = read.clone();
    let first_oob_word = u64::from(oob_read.state_before.memory_pages.expect("memory pages")) * 16384;
    oob_read
        .stack_read0
        .as_mut()
        .expect("pointer argument read")
        .value_lo = u32::try_from(first_oob_word * 4).expect("wasm32 byte address");
    oob_read
        .linear_memory
        .as_mut()
        .expect("grammar memory access")
        .lane0
        .word_addr = first_oob_word;
    let forged = build_witness_vector(&oob_read);
    assert_eq!(forged[neo_wasm::layout::COL_CMP_GE], neo_math::F::ONE);
    assert_eq!(forged[neo_wasm::layout::COL_MEM_OOB], neo_math::F::ONE);
    common::assert_rejected(&forged, "aligned OOB grammar memory read");

    let write = trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryWrite)
        })
        .expect("memory write gather");
    let mut forged = build_witness_vector(write);
    common::assert_satisfied(&forged, "untampered grammar memory write");
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE0_VALUE] += neo_math::F::ONE;
    common::assert_rejected(&forged, "grammar memory write diverging from the staged claim");

    let byte_read = trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryRead)
                && row
                    .linear_memory
                    .is_some_and(|access| access.width_bytes == 1 && access.byte_offset == 3)
        })
        .expect("byte memory read gather");
    let mut redirected = byte_read.clone();
    redirected
        .linear_memory
        .as_mut()
        .expect("byte access")
        .byte_offset = 2;
    common::assert_rejected(
        &build_witness_vector(&redirected),
        "grammar byte read with a forged intra-word offset",
    );

    let equal_neighbor_read = trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryRead)
                && row
                    .linear_memory
                    .is_some_and(|access| access.width_bytes == 1 && access.byte_offset == 1)
        })
        .expect("byte read beside an equal-valued byte");
    let mut forged = build_witness_vector(equal_neighbor_read);
    common::assert_satisfied(&forged, "untampered grammar byte offset selector");
    forged[neo_wasm::layout::COL_LINEAR_MEM_OFFSET_IS_1] = neo_math::F::ZERO;
    forged[neo_wasm::layout::COL_LINEAR_MEM_OFFSET_IS_2] = neo_math::F::ONE;
    forged[neo_wasm::layout::COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1] = neo_math::F::ZERO;
    forged[neo_wasm::layout::COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2] = neo_math::F::ONE;
    common::assert_rejected(
        &forged,
        "grammar byte routing selector diverging from the effective address",
    );

    let byte_write_index = trace
        .iter()
        .position(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryWrite)
                && row
                    .linear_memory
                    .is_some_and(|access| access.width_bytes == 1)
        })
        .expect("byte memory write gather");
    let byte_write = &trace[byte_write_index];
    let mut forged = build_witness_vector(byte_write);
    common::assert_satisfied(&forged, "untampered grammar byte write");
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE0_VALUE] += neo_math::F::ONE;
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE0_BYTE0] += neo_math::F::ONE;
    common::assert_rejected(&forged, "grammar byte write changing an unselected byte");

    let half_read = trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryRead)
                && row.linear_memory.is_some_and(|access| {
                    access.width_bytes == 2 && access.byte_offset == 2 && access.lane0.word_addr == 6
                })
        })
        .expect("zero-valued aligned half-word read");
    let mut misaligned = half_read.clone();
    misaligned
        .stack_read0
        .as_mut()
        .expect("pointer argument read")
        .value_lo = 15;
    misaligned
        .linear_memory
        .as_mut()
        .expect("half-word access")
        .byte_offset = 1;
    common::assert_rejected(
        &build_witness_vector(&misaligned),
        "grammar half-word read with an odd effective address",
    );

    let half_write = trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryWrite)
                && row
                    .linear_memory
                    .is_some_and(|access| access.width_bytes == 2)
        })
        .expect("half-word memory write gather");
    let mut forged = build_witness_vector(half_write);
    common::assert_satisfied(&forged, "untampered grammar half-word write");
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE0_VALUE] += neo_math::F::ONE;
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE0_BYTE0] += neo_math::F::ONE;
    common::assert_rejected(&forged, "grammar half-word write changing an unselected byte");

    let mut forged_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    forged_rows[byte_write_index][neo_wasm::layout::COL_LINEAR_MEM_LANE0_VALUE_BEFORE] +=
        neo_math::F::from_u64(1 << 16);
    forged_rows[byte_write_index][neo_wasm::layout::COL_LINEAR_MEM_LANE0_BYTE2_BEFORE] += neo_math::F::ONE;
    neo_wasm::memory_semantics::sanity_check_memory_rows(layout, &forged_rows, &preload)
        .expect_err("grammar byte write must authenticate its prior word");
}
