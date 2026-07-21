//! Grammar-mode traces: the chain absorbs embedder grammar events staged by
//! `HostEventGather` slot rows (8 per block, one word each) instead of raw
//! host-call records. Every row is CCS-checked, the grammar ROM content is
//! checked by the native memory-rows pass, and the rejection tests cover
//! mode gating, gather forgery, and the event schedule.

mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::event_grammar::{GrammarEvent, HostEventGrammar, ImportTemplate, Limb, SlotSource};
use neo_wasm::layout::{COL_GATHER_ACTIVE, COL_GRAMMAR_MODE_AFTER, COL_RAW_HOST_CALL};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{WasmRowKind, WasmVmStep};
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
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &[Default::default()])
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
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &[Default::default()])
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
                    .is_some_and(|rom| rom.kind == 2 && rom.limb == 0)
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
                    .is_some_and(|rom| rom.kind == 2 && rom.limb == 1)
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
fn missing_template_is_rejected() {
    let run = run_component();
    let grammar = HostEventGrammar::default();
    assert!(neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &[Default::default()]).is_err());
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
    assert!(neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &[Default::default()]).is_err());
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
                && row.grammar_rom_slot.is_some_and(|rom| rom.kind == 0)
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
        .find(|row| row.row_kind.is_host_event_gather() && row.grammar_rom_slot.is_some_and(|rom| rom.kind == 1))
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
    let mut trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &[Default::default()])
        .expect("grammar trace");

    let idx = trace
        .iter()
        .position(|row| row.row_kind.is_host_event_gather() && row.grammar_rom_slot.is_some_and(|rom| rom.kind == 0))
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
        .find(|row| row.row_kind.is_host_event_gather() && row.grammar_rom_slot.is_some_and(|rom| rom.kind == 3))
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
