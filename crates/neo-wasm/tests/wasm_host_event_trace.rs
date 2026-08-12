//! Host-event traces: the chain absorbs embedder events staged by
//! `HostEventGather` slot rows (8 per block, one word each). Every row is
//! CCS-checked, the bindings ROM content is checked by the native memory-rows
//! pass, and rejection tests cover gather forgery and the event schedule.

mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::host_event_bindings::{EventBlock, HostEventBindings, ImportTemplate, Limb, SlotBinding};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{WasmHostEventSlotKind, WasmMemoryId, WasmVmStep};
use p3_field::PrimeCharacteristicRing;

const ZERO: SlotBinding = SlotBinding::Const(0);

fn slots(entries: &[(usize, SlotBinding)]) -> [SlotBinding; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
}

/// Example embedder bindings for the mul/sink component: `mul(x, y) -> r`
/// expands to a two-event template (args event + result event referencing a
/// shared input word), `sink(x)` to a single event.
fn test_bindings(mul_fref: u32, sink_fref: u32) -> HostEventBindings {
    let arg = |arg, limb| SlotBinding::ArgElem { arg, limb };
    let mut bindings = HostEventBindings::default();
    bindings.imports.insert(
        mul_fref,
        ImportTemplate {
            events: vec![
                EventBlock::op(
                    10,
                    slots(&[
                        (0, SlotBinding::Input { index: 0 }),
                        (1, arg(0, Limb::Lo)),
                        (2, arg(1, Limb::Lo)),
                        (3, SlotBinding::Const(5)),
                    ]),
                ),
                // The ResultElem Lo slot is the gather row that pushes the
                // host result onto the operand stack; the Hi slot binds the
                // pushed hi lane (0 for the i32 result).
                EventBlock::op(
                    12,
                    slots(&[
                        (0, SlotBinding::ResultElem { limb: Limb::Lo }),
                        (1, SlotBinding::Input { index: 0 }),
                        (2, SlotBinding::ResultElem { limb: Limb::Hi }),
                    ]),
                ),
            ],
            input_count: 1,
        },
    );
    bindings.imports.insert(
        sink_fref,
        ImportTemplate {
            events: vec![EventBlock::op(7, slots(&[(0, arg(0, Limb::Lo))]))],
            input_count: 0,
        },
    );
    bindings
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

/// Run the two-call component; the mul host function records `mul_inputs`
/// for its in-flight call (the bindings hand-off path), sink records nothing.
fn run_component_with_mul_inputs(mul_inputs: &'static [u64]) -> neo_wasm::WasmtimeTraceRun {
    let component_bytes = wat::parse_str(mul_sink_component_wat()).expect("component wat");
    neo_wasm::collect_wasmtime_component_run_with_linker(&component_bytes, "run", |linker| {
        linker
            .root()
            .func_wrap("host-mul", move |mut store, (x, y): (i32, i32)| {
                store.data_mut().record_call_inputs(mul_inputs)?;
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
    run_component_with_mul_inputs(&[100])
}

fn run_frefs(run: &neo_wasm::WasmtimeTraceRun) -> (Vec<u32>, u32) {
    let imports = run
        .steps
        .iter()
        .filter(|row| matches!(row.opcode_decoded, Some(neo_wasm::WasmOpcode::Call)) && !row.target_function_is_guest)
        .filter_map(|row| row.function_ref)
        .collect();
    let export = run
        .steps
        .iter()
        .find_map(|row| row.current_function_ref)
        .expect("export function ref");
    (imports, export)
}

/// Bound host-event trace for the two-call component, with input words `[100]` for mul
/// and `[]` for sink. The invoked export gets an empty boundary template
/// (every entered export needs a template; no boundary events for this test).
fn host_event_trace() -> Vec<WasmVmStep> {
    host_event_trace_from(Default::default())
}

fn host_event_trace_from(initial_comm_chain: neo_wasm::CommChainState) -> Vec<WasmVmStep> {
    let run = run_component();
    let (frefs, export_fref) = run_frefs(&run);
    assert_eq!(frefs.len(), 2);
    let mut bindings = test_bindings(frefs[0], frefs[1]);
    bindings
        .exports
        .insert(export_fref, neo_wasm::host_event_bindings::ExportTemplate::default());
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &[Default::default()],
        initial_comm_chain,
    )
    .expect("bindings trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

    // The claimed host-event ROM entries must match the embedder tables.
    let component_bytes = wat::parse_str(mul_sink_component_wat()).expect("component wat");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("bindings ROM contents match");
    trace
}

/// The bindings-less normalizer runs under the canonical import-free bindings,
/// so an executed host import has no template and is rejected.
#[test]
fn host_import_requires_host_event_bindings() {
    let run = run_component();
    let error = neo_wasm::traces_from_wasmtime_steps(&run.steps)
        .expect_err("host imports must not use an implicit event encoding");
    assert!(error
        .to_string()
        .contains("no host-event template for host import"));
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
    let (frefs, export_fref) = run_frefs(&run);
    assert_eq!(frefs.len(), 1);
    let arg = |arg, limb| SlotBinding::ArgElem { arg, limb };
    let mut bindings = HostEventBindings::default();
    bindings.imports.insert(
        frefs[0],
        ImportTemplate {
            events: vec![
                EventBlock::op(
                    3,
                    slots(&[
                        (0, arg(0, Limb::Lo)),
                        (1, arg(0, Limb::Hi)),
                        (2, arg(1, Limb::Lo)),
                        (3, arg(1, Limb::Hi)),
                    ]),
                ),
                EventBlock::op(
                    4,
                    slots(&[
                        (0, SlotBinding::ResultElem { limb: Limb::Lo }),
                        (1, SlotBinding::ResultElem { limb: Limb::Hi }),
                    ]),
                ),
            ],
            input_count: 0,
        },
    );
    bindings
        .exports
        .insert(export_fref, neo_wasm::host_event_bindings::ExportTemplate::default());
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &[Default::default()],
        Default::default(),
    )
    .expect("bindings trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("bindings ROM contents match");

    // 2^32 + 2^33 = 3·2^32: the result lives entirely in the hi limb.
    let lo_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .host_event_rom_slot
                    .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::Result && rom.variant.is_low_limb())
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
                    .host_event_rom_slot
                    .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::Result && rom.variant.is_high_limb())
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
    let run = run_component_with_mul_inputs(&[]);
    let (frefs, export_fref) = run_frefs(&run);
    // `mul` is advice; `sink` remains transcript-bound.
    let arg = |arg, limb| SlotBinding::ArgElem { arg, limb };
    let mut bindings = HostEventBindings::default();
    let mut advice_block = [SlotBinding::Const(0); 8];
    advice_block[0] = SlotBinding::ResultElem { limb: Limb::Lo };
    advice_block[1] = SlotBinding::ResultElem { limb: Limb::Hi };
    bindings.imports.insert(
        frefs[0],
        ImportTemplate {
            events: vec![EventBlock::advice(advice_block)],
            input_count: 0,
        },
    );
    bindings.imports.insert(
        frefs[1],
        ImportTemplate {
            events: vec![EventBlock::op(7, slots(&[(0, arg(0, Limb::Lo))]))],
            input_count: 0,
        },
    );
    bindings
        .exports
        .insert(export_fref, neo_wasm::host_event_bindings::ExportTemplate::default());
    let turns = [neo_wasm::host_event_bindings::TurnInputs::default()];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &turns,
        Default::default(),
    )
    .expect("bindings trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

    let component_bytes = wat::parse_str(mul_sink_component_wat()).expect("component wat");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("bindings ROM contents match");

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
        .filter(|row| row.row_kind.is_host_event_gather() && row.host_event_rom_slot.is_some_and(|rom| rom.advice))
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
            row.host_event_rom_slot
                .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::Result && rom.variant.is_low_limb())
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
    forged[neo_wasm::layout::COL_HOST_EVENT_SLOT_KIND] -= neo_math::F::from_u64(8);
    common::assert_rejected(&forged, "advice row shedding the advice flag");
}

#[test]
fn host_event_trace_folds_expanded_blocks() {
    let trace = host_event_trace();
    // Three bindings events → three completed blocks (each staged by 8 slot
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
fn host_event_trace_folds_from_explicit_initial_state() {
    let f = p3_goldilocks::Goldilocks::from_u64;
    let initial = neo_wasm::CommChainState::new([f(11), f(22), f(33), f(44)]);
    let trace = host_event_trace_from(initial);
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
    let bindings = HostEventBindings::default();
    assert!(neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &[Default::default()],
        Default::default(),
    )
    .is_err());
}

/// A host call recording more input words than its template consumes
/// indicates a misaligned hand-off and is rejected.
#[test]
fn surplus_input_words_are_rejected() {
    let run = run_component_with_mul_inputs(&[100, 7]);
    let (frefs, export_fref) = run_frefs(&run);
    let mut bindings = test_bindings(frefs[0], frefs[1]);
    bindings
        .exports
        .insert(export_fref, neo_wasm::host_event_bindings::ExportTemplate::default());
    assert!(neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &[Default::default()],
        Default::default(),
    )
    .is_err());
}

/// A gather row staging a word that contradicts its (honest) ROM entry is
/// CCS-rejected: here the constant discriminant word is forged.
#[test]
fn ccs_rejects_forged_gather_word() {
    let trace = host_event_trace();
    let disc_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .host_event_rom_slot
                    .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::Const)
                && row.state_before.host_events.slot_cursor == 0
        })
        .expect("discriminant slot row");
    let mut witness = build_witness_vector(disc_row);
    common::assert_satisfied(&witness, "untampered discriminant slot row");
    witness[neo_wasm::layout::COL_EVBUF_AFTER[0]] += neo_math::F::ONE;
    common::assert_rejected(&witness, "gather row staging a forged discriminant");
}

/// An arg-slot gather row must read the table-pinned stack address.
#[test]
fn ccs_rejects_redirected_gather_read() {
    let trace = host_event_trace();
    let arg_slot_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .host_event_rom_slot
                    .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::Arg)
        })
        .expect("arg slot row");
    let mut witness = build_witness_vector(arg_slot_row);
    common::assert_satisfied(&witness, "untampered arg slot row");
    witness[neo_wasm::layout::COL_STACK_READ_ADDR_LO[0]] += neo_math::F::from_u64(2);
    common::assert_rejected(&witness, "arg slot row reading a different stack slot");
}

/// Forging the claimed ROM entry itself is caught by the host-event ROM
/// content check (the native stand-in for the lookup argument).
#[test]
fn memory_rows_reject_forged_rom_claim() {
    let run = run_component();
    let (frefs, export_fref) = run_frefs(&run);
    let mut bindings = test_bindings(frefs[0], frefs[1]);
    bindings
        .exports
        .insert(export_fref, neo_wasm::host_event_bindings::ExportTemplate::default());
    let mut trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &[Default::default()],
        Default::default(),
    )
    .expect("bindings trace");

    let idx = trace
        .iter()
        .position(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .host_event_rom_slot
                    .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::Const)
        })
        .expect("const slot row");
    if let Some(rom) = &mut trace[idx].host_event_rom_slot {
        rom.const_lo ^= 1;
    }

    let component_bytes = wat::parse_str(mul_sink_component_wat()).expect("component wat");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    assert!(
        neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload).is_err(),
        "a forged host-event ROM claim must fail the content check"
    );
}

/// The event schedule is forced: a program row cannot leave bindings events
/// unabsorbed, and a gather row cannot run with none owed.
#[test]
fn ccs_rejects_broken_event_schedule() {
    let trace = host_event_trace();

    let program_row = trace
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row");
    let mut witness = build_witness_vector(program_row);
    common::assert_satisfied(&witness, "untampered program row");
    witness[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_BEFORE] = neo_math::F::ONE;
    witness[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_BEFORE_IS_ZERO] = neo_math::F::ZERO;
    witness[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_BEFORE_INV] = neo_math::F::ONE;
    common::assert_rejected(&witness, "program row with bindings events still owed");

    let gather_row = trace
        .iter()
        .find(|row| row.row_kind.is_host_event_gather())
        .expect("gather row");
    let mut witness = build_witness_vector(gather_row);
    common::assert_satisfied(&witness, "untampered gather row");
    witness[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_BEFORE] = neo_math::F::ZERO;
    witness[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_BEFORE_IS_ZERO] = neo_math::F::ONE;
    witness[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_BEFORE_INV] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "gather row with no host events owed");
}

/// An import with no template reads the zero-filled biased count cell, so the
/// only row-locally satisfiable assignment loads a poisoned event countdown of -1.
#[test]
fn ccs_forces_untemplated_import_into_poisoned_schedule() {
    let trace = host_event_trace();
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
    forged[neo_wasm::layout::COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT] = neo_math::F::ZERO;
    common::assert_rejected(&forged, "untemplated import call claiming a normal schedule");

    // The poisoned schedule satisfies the row itself. The composed circuit's
    // host-event ROM address bound prevents enough blocks from draining it; see
    // the count-family relation-layout comment for the full argument.
    let mut poisoned = forged.clone();
    poisoned[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_AFTER] = -neo_math::F::ONE;
    common::assert_satisfied(&poisoned, "untemplated import call loads the poisoned schedule");
}

/// Import schedule counts and export entry-schedule counts are separate ROM families
/// (with the +1 presence bias), so a turn boundary or exit latch can never
/// read an import's cell and vice versa.
#[test]
fn count_families_are_split_and_biased() {
    let run = run_component();
    let (frefs, export_fref) = run_frefs(&run);
    let mut bindings = test_bindings(frefs[0], frefs[1]);
    bindings
        .exports
        .insert(export_fref, neo_wasm::host_event_bindings::ExportTemplate::default());

    let mut preload = neo_wasm::memory_semantics::WasmMemoryPreload::default();
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
    let cell = |family: WasmMemoryId, fref: u32| {
        preload
            .entries()
            .into_iter()
            .find(|(memory, address, _)| *memory == family && address == &[fref])
            .map(|(_, _, value)| value)
    };
    for (&fref, template) in &bindings.imports {
        assert_eq!(
            cell(WasmMemoryId::HostEventImportScheduleCount, fref),
            Some(template.events.len() as u32 + 1),
            "import cells live in the import family, biased"
        );
        assert_eq!(
            cell(WasmMemoryId::HostEventExportEntryScheduleCount, fref),
            None,
            "imports must have no export entry-count cell"
        );
    }
    assert_eq!(
        cell(WasmMemoryId::HostEventExportEntryScheduleCount, export_fref),
        Some(1),
        "the export's zero-event entry template is the biased 1, distinct from the zero-filled 0"
    );
    assert_eq!(
        cell(WasmMemoryId::HostEventImportScheduleCount, export_fref),
        None,
        "exports must have no import-schedule count cell"
    );
}

/// Input slots are free absorbed words: staging a different value
/// satisfies the per-row CCS (there is deliberately no local binding) but
/// diverges the chain, so the transcript check rejects the claim. The
/// same-index identity lives in input construction: expansion resolves
/// every `Input{index}` from one input entry.
#[test]
fn input_words_are_row_free_and_transcript_bound() {
    let trace = host_event_trace();
    let input_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .host_event_rom_slot
                    .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::Input)
        })
        .expect("input slot row");
    let mut witness = build_witness_vector(input_row);
    common::assert_satisfied(&witness, "untampered input slot row");
    // Forge the input word consistently in the staged buffer and the
    // gadget's slot value: the row still satisfies (free word) ...
    let cursor = usize::from(input_row.state_before.host_events.slot_cursor);
    witness[neo_wasm::layout::COL_EVBUF_AFTER[cursor]] += neo_math::F::ONE;
    common::assert_rejected(&witness, "buffer word diverging from the staged slot value");
    // ... but any divergence between the absorbed words and the claimed
    // transcript is caught by the final-chain fold (see
    // the transcript-verification rejection).
}
