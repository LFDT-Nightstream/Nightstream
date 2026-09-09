//! Static object roots: native framing, shared-gadget execution, and rejection
//! of forged roots, context transitions, and verifier-owned recipes.

mod common;

use neo_math::F;
use neo_wasm::comm_chain::{absorbed_event_blocks, fold_event_blocks, COMM_CHAIN_PERM_ROWS};
use neo_wasm::host_event_bindings::{
    absorbed_blocks, expand_export_entry, expand_export_exit, expand_import_events, opaque_value_root, EventBlock,
    EventSequenceBuilder, EventSources, ExportTemplate, HostEventBindings, HostEventBindingsBuilder, ImportTemplate,
    Limb, MemoryBase, SlotBinding, TurnInputs,
};
use neo_wasm::layout::*;
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{WasmHostEventSlotKind, WasmOpcode, WasmVmStep};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

const ZERO: SlotBinding = SlotBinding::Const(0);
const SCHEMA: [u64; 4] = [11, 12, 13, 14];

fn outer(word: usize) -> EventBlock {
    let mut block = core::array::from_fn(|i| SlotBinding::Const(100 + i as u64));
    block[word..word + 4].fill(ZERO);
    EventBlock { block, absorb: true }
}

#[test]
fn native_roots_cover_empty_partial_and_multiple_blocks_at_every_position() {
    for len in [0, 1, 7, 8, 9, 16, 21] {
        let words: Vec<u64> = (0..len).map(|i| i as u64 + 1).collect();
        let root = opaque_value_root(SCHEMA, &words).unwrap();
        for word in 0..=4 {
            let template = ImportTemplate {
                events: outer(word)
                    .with_opaque(word, SCHEMA, words.iter().copied().map(SlotBinding::Const).collect())
                    .unwrap(),
                input_count: 0,
            };
            template.validate(0, 0).unwrap();
            let expanded = expand_import_events(&template, &[], None, &[], &[]).unwrap();
            let mut expected = core::array::from_fn(|i| 100 + i as u64);
            expected[word..word + 4].copy_from_slice(&root);
            assert_eq!(absorbed_blocks(&template.events, &expanded).unwrap(), vec![expected]);
            let mut forged = expanded.clone();
            forged.last_mut().unwrap()[word] ^= 1;
            assert!(absorbed_blocks(&template.events, &forged).is_err());
        }
    }
    let root = opaque_value_root(SCHEMA, &[1, 2]).unwrap();
    assert_ne!(root, opaque_value_root(SCHEMA, &[1, 2, 0]).unwrap());
    assert_ne!(root, opaque_value_root(SCHEMA, &[2, 1]).unwrap());
    assert_ne!(root, opaque_value_root([12, 12, 13, 14], &[1, 2]).unwrap());
    assert_ne!(opaque_value_root(SCHEMA, &[]).unwrap(), [0; 4]);
    assert!(opaque_value_root(SCHEMA, &[Goldilocks::ORDER_U64]).is_err());
    assert!(opaque_value_root([Goldilocks::ORDER_U64; 4], &[]).is_err());
}

#[test]
fn malformed_or_nested_schedules_are_rejected() {
    // A four-word root cannot fit starting at slot five.
    assert!(outer(4).with_opaque(5, SCHEMA, vec![]).is_err());
    // An opaque root must belong to an absorbing outer block, not advice.
    assert!(EventBlock::advice([ZERO; 8])
        .with_opaque(0, SCHEMA, vec![])
        .is_err());
    // The root destination must contain only zero placeholders.
    assert!(outer(2).with_opaque(1, SCHEMA, vec![]).is_err());
    // Object sources cannot contain a nested Enter.
    assert!(outer(2)
        .with_opaque(2, SCHEMA, vec![SlotBinding::EnterOpaque])
        .is_err());
    let events = outer(2)
        .with_opaque(2, SCHEMA, vec![SlotBinding::Const(9)])
        .unwrap();
    let assert_rejected = |events, case: &str| {
        assert!(
            ImportTemplate { events, input_count: 0 }
                .validate(0, 0)
                .is_err(),
            "accepted {case}"
        );
    };

    // The object schedule must include its copy-back block.
    let mut broken = events.clone();
    broken.pop();
    assert_rejected(broken, "missing copy-back block");

    // Moving Enter changes the prefix length without updating copy-back.
    let mut broken = events.clone();
    broken[0].block.swap(2, 3);
    assert_rejected(broken, "mismatched prefix length");

    // Unexecuted slots after Enter must remain zero placeholders.
    let mut broken = events.clone();
    broken[0].block[6] = SlotBinding::Const(1);
    assert_rejected(broken, "nonzero placeholder after Enter");

    // The object header must carry the expected domain tag.
    let mut broken = events.clone();
    broken[1].block[0] = SlotBinding::Const(0);
    assert_rejected(broken, "wrong object domain tag");

    // The declared payload length cannot require missing blocks.
    let mut broken = events.clone();
    broken[1].block[6] = SlotBinding::Const(100);
    assert_rejected(broken, "missing declared payload blocks");

    // Payload padding beyond the declared length must be zero.
    let mut broken = events.clone();
    broken[2].block[7] = SlotBinding::Const(1);
    assert_rejected(broken, "nonzero payload padding");

    // Root copy-back instructions cannot appear inside the payload.
    let mut broken = events.clone();
    broken[2].block[0] = SlotBinding::OpaqueRoot { lane: 0 };
    assert_rejected(broken, "root copy-back inside payload");

    // Copy-back must contain root lanes 0, 1, 2, 3 in order.
    let mut broken = events.clone();
    broken[3].block[5] = SlotBinding::OpaqueRoot { lane: 2 };
    assert_rejected(broken, "repeated root lane");

    // Object payload blocks must be hashed, not treated as advice.
    let mut broken = events.clone();
    broken[2].absorb = false;
    assert_rejected(broken, "advice payload block");

    // A lowered Enter cannot leave fewer than four slots for the root.
    let mut broken = events.clone();
    broken[0].block.swap(2, 7);
    assert_rejected(broken, "Enter beyond the maximum prefix length");

    // Skipped slots after Enter cannot contain argument reads.
    let mut broken = events.clone();
    broken[0].block[3] = SlotBinding::ArgElem { arg: 0, limb: Limb::Lo };
    assert_rejected(broken, "argument read after Enter");

    // A prefix group cannot contain a second Enter.
    let mut broken = events.clone();
    broken[0].block[3] = SlotBinding::EnterOpaque;
    assert_rejected(broken, "second Enter in prefix group");
}

#[test]
fn early_enter_skips_only_prefix_padding_at_every_root_position() {
    let bytes = wat::parse_str(
        r#"
        (component
          (type $f (func))
          (import "host" (func $host (type $f)))
          (core module $m
            (import "" "host" (func $host))
            (func (export "run") call $host))
          (core func $lowered (canon lower (func $host)))
          (core instance $host (export "host" (func $lowered)))
          (core instance $i (instantiate $m (with "" (instance $host))))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $f) (canon lift (core func $run))))
    "#,
    )
    .unwrap();

    let run = neo_wasm::collect_wasmtime_component_run_with_linker(&bytes, "run", |linker| {
        linker
            .root()
            .func_wrap("host", |_store, (): ()| Ok(()))
            .map_err(|err| neo_wasm::WasmBuildError::Trace(err.to_string()))
    })
    .unwrap();

    let export = run
        .steps
        .iter()
        .find_map(|row| row.current_function_ref)
        .unwrap();

    let import = run
        .steps
        .iter()
        .find(|row| row.opcode_decoded == Some(WasmOpcode::Call) && !row.target_function_is_guest)
        .unwrap()
        .function_ref
        .unwrap();

    let mut events = Vec::new();
    let mut expected = Vec::new();
    let mut saved_rows = 0;
    let mut permutations = 0;
    for prefix in 0..=4 {
        for len in [0usize, 9] {
            // Leave a dirty shared buffer before Enter, including slots that
            // the shortened prefix group will no longer overwrite with zeros.
            events.push(EventBlock::advice([SlotBinding::Const(77); 8]));
            let words: Vec<_> = (1..=len as u64).collect();
            events.extend(
                outer(prefix)
                    .with_opaque(prefix, SCHEMA, words.iter().copied().map(SlotBinding::Const).collect())
                    .unwrap(),
            );
            let mut block = core::array::from_fn(|lane| 100 + lane as u64);
            block[prefix..prefix + 4].copy_from_slice(&opaque_value_root(SCHEMA, &words).unwrap());
            expected.push(block);
            saved_rows += 7 - prefix;
            permutations += 2 + len.div_ceil(8); // header, payload, outer block
        }
    }
    let old_gather_count = events.len() * 8;
    let mut builder = HostEventBindingsBuilder::new(&run.program_tables);
    builder.export(export, vec![], vec![]).unwrap();
    builder.import(import, events).unwrap();
    let bindings = builder.finish().unwrap();
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &[TurnInputs::default()],
        Default::default(),
    )
    .unwrap();

    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&bytes).unwrap();
    common::sanity_check_trace_with_bindings(&trace, &artifacts, &bindings);
    common::ccs_check_trace(&trace);

    assert_eq!(
        trace
            .iter()
            .filter(|row| row.row_kind.is_host_event_gather())
            .count(),
        old_gather_count - saved_rows
    );
    assert_eq!(
        trace
            .iter()
            .filter(|row| row.row_kind.is_host_event_perm())
            .count(),
        permutations * COMM_CHAIN_PERM_ROWS
    );
    assert_eq!(
        absorbed_event_blocks(&trace)
            .iter()
            .map(|block| block.words)
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        trace.last().unwrap().state_after.comm_chain,
        fold_event_blocks(
            Default::default(),
            &expected
                .iter()
                .map(|block| block.map(Goldilocks::from_u64))
                .collect::<Vec<_>>()
        )
        .canonical_u64()
    );

    for row in trace.iter().filter(|row| {
        row.host_event_rom_slot
            .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::EnterOpaque)
    }) {
        for column in [
            COL_HOST_EVENT_SLOT_CURSOR_AFTER,
            COL_HOST_EVENT_INDEX_AFTER,
            COL_HOST_EVENTS_REMAINING_AFTER,
        ] {
            let mut witness = build_witness_vector(row);
            witness[column] += F::ONE;
            common::assert_rejected(&witness, "forged early-Enter cursor or schedule advance");
        }
    }
}

struct Setup {
    trace: Vec<WasmVmStep>,
    bindings: HostEventBindings,
    artifacts: neo_wasm::WasmProgramArtifacts,
    expected: Vec<[u64; 8]>,
}

fn setup() -> Setup {
    let bytes = wat::parse_str(
        r#"
      (component
        (type $mul-type (func (param "x" s32) (param "y" s32) (result s32)))
        (type $run-type (func (param "x" s32) (param "y" s32) (result s32)))
        (import "mul" (func $mul (type $mul-type)))
        (core module $m
          (import "" "mul" (func $mul (param i32 i32) (result i32)))
          (memory 1)
          (data (i32.const 16) "\7b\00\00\00")
          (data (i32.const 48) "\2d\00\00\00")
          (func (export "run") (param i32 i32) (result i32)
            local.get 0
            local.get 1
            call $mul))
        (core func $lowered (canon lower (func $mul)))
        (core instance $host (export "mul" (func $lowered)))
        (core instance $i (instantiate $m (with "" (instance $host))))
        (alias core export $i "run" (core func $run))
        (func (export "run") (type $run-type) (canon lift (core func $run))))
    "#,
    )
    .unwrap();
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(
        &bytes,
        "run",
        &[wasmtime::component::Val::S32(16), wasmtime::component::Val::S32(3)],
        |linker| {
            linker
                .root()
                .func_wrap("mul", |_store, (x, y): (i32, i32)| Ok((x * y,)))
                .map_err(|err| neo_wasm::WasmBuildError::Trace(err.to_string()))
        },
    )
    .unwrap();
    let export = run
        .steps
        .iter()
        .find_map(|row| row.current_function_ref)
        .unwrap();
    let import = run
        .steps
        .iter()
        .find(|row| row.opcode_decoded == Some(WasmOpcode::Call) && !row.target_function_is_guest)
        .unwrap()
        .function_ref
        .unwrap();
    let entry = EventSequenceBuilder::op(1)
        .input_local_i32(0, 0)
        .unwrap()
        .constant_i32(0)
        .unwrap()
        .constant_i32(0)
        .unwrap()
        .opaque(
            SCHEMA,
            EventSources::new()
                .input_local_i32(1, 1)
                .memory_write_i32(2, MemoryBase::Local(0), 0),
        )
        .unwrap()
        .finish()
        .unwrap();
    let exit = EventSequenceBuilder::absorbing()
        .opaque(
            SCHEMA,
            EventSources::new()
                .output_i32()
                .memory_read_i32(MemoryBase::Output, 0),
        )
        .unwrap()
        .constant_i32(104)
        .unwrap()
        .constant_i32(105)
        .unwrap()
        .constant_i32(106)
        .unwrap()
        .constant_i32(107)
        .unwrap()
        .finish()
        .unwrap();
    let mut payload = EventSources::new()
        .arg_i32(1)
        .memory_read_i32(MemoryBase::Arg(0), 0);
    for value in 50..57 {
        payload = payload.constant_i32(value);
    }
    let call = EventSequenceBuilder::op(2)
        .arg_i32(0)
        .unwrap()
        .opaque(SCHEMA, payload)
        .unwrap()
        .result()
        .unwrap();
    // A second object in the same import must reset the previous inner chain.
    let call = call
        .constant_i32(100)
        .unwrap()
        .constant_i32(101)
        .unwrap()
        .constant_i32(102)
        .unwrap()
        .constant_i32(103)
        .unwrap()
        .opaque(SCHEMA, EventSources::new())
        .unwrap()
        .finish()
        .unwrap();
    let mut builder = HostEventBindingsBuilder::new(&run.program_tables);
    builder.export(export, entry, exit).unwrap();
    builder.import(import, call).unwrap();
    let bindings = builder.finish().unwrap();
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &[TurnInputs {
            entry: vec![16, 3, 123],
        }],
        Default::default(),
    )
    .unwrap();
    let boundary = &bindings.exports[&export];
    let call = &bindings.imports[&import];
    let mut expected =
        absorbed_blocks(&boundary.entry, &expand_export_entry(boundary, &[16, 3, 123]).unwrap()).unwrap();
    expected.extend(
        absorbed_blocks(
            &call.events,
            &expand_import_events(call, &[(16, 0), (3, 0)], Some((48, 0)), &[], &[123]).unwrap(),
        )
        .unwrap(),
    );
    expected.extend(
        absorbed_blocks(
            &boundary.exit,
            &expand_export_exit(boundary, Some((48, 0)), &[45]).unwrap(),
        )
        .unwrap(),
    );
    Setup {
        trace,
        bindings,
        artifacts: neo_wasm::extract_first_component_core_program_artifacts(&bytes).unwrap(),
        expected,
    }
}

fn checked() -> &'static Setup {
    static SETUP: std::sync::OnceLock<Setup> = std::sync::OnceLock::new();
    SETUP.get_or_init(|| {
        let setup = setup();
        common::sanity_check_trace_with_bindings(&setup.trace, &setup.artifacts, &setup.bindings);
        common::ccs_check_trace(&setup.trace);
        setup
    })
}

#[test]
fn opaque_import_and_export_effects_share_one_permutation_and_match_native_roots() {
    let setup = checked();
    let trace = &setup.trace;
    let nonzero_entries = trace
        .iter()
        .filter(|row| {
            row.host_event_rom_slot
                .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::EnterOpaque)
                && row.state_before.comm_chain != [0; 4]
        })
        .count();
    assert!(
        nonzero_entries >= 2,
        "exercise save/restore of non-genesis outer chains"
    );
    let blocks: Vec<_> = absorbed_event_blocks(trace)
        .into_iter()
        .map(|block| block.words)
        .collect();
    assert_eq!(blocks, setup.expected);
    assert_eq!(blocks.len(), 4);
    assert_eq!(&blocks[0][4..], &opaque_value_root(SCHEMA, &[3, 123]).unwrap());
    assert_eq!(
        &blocks[1][2..6],
        &opaque_value_root(SCHEMA, &[3, 123, 50, 51, 52, 53, 54, 55, 56]).unwrap()
    );
    assert_eq!(&blocks[2][4..], &opaque_value_root(SCHEMA, &[]).unwrap());
    assert_eq!(&blocks[3][..4], &opaque_value_root(SCHEMA, &[48, 45]).unwrap());
    let expected_chain = fold_event_blocks(
        Default::default(),
        &blocks
            .iter()
            .map(|b| b.map(Goldilocks::from_u64))
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        trace.last().unwrap().state_after.comm_chain,
        expected_chain.canonical_u64()
    );
    assert_eq!(trace.last().unwrap().state_after.output.value_lo, 48);
    assert!(!trace.last().unwrap().state_after.event_absorb.object_active);
    assert_eq!(
        trace
            .iter()
            .filter(|r| r.row_kind.is_host_event_perm())
            .count(),
        12 * COMM_CHAIN_PERM_ROWS
    );
}

#[test]
fn ccs_rejects_forged_context_roots_prefix_and_chain_updates() {
    let trace = &checked().trace;
    for kind in [WasmHostEventSlotKind::OpaqueSaved, WasmHostEventSlotKind::OpaqueRoot] {
        for row in trace
            .iter()
            .filter(|r| r.host_event_rom_slot.is_some_and(|rom| rom.kind == kind))
        {
            let cursor = usize::from(row.state_before.host_events.slot_cursor);
            let mut forged = row.clone();
            forged.state_after.event_absorb.evbuf[cursor] ^= 1;
            common::assert_rejected(
                &build_witness_vector(&forged),
                "forged copy-back value with regenerated advice",
            );
        }
    }
    let enter = trace
        .iter()
        .find(|r| {
            r.host_event_rom_slot
                .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::EnterOpaque)
        })
        .unwrap();
    for column in COL_OUTER_CHAIN_AFTER
        .into_iter()
        .chain(COL_COMM_CHAIN_AFTER)
        .chain(COL_OUTER_PREFIX_AFTER)
        .chain([COL_OBJECT_ACTIVE_AFTER, COL_PERM_PENDING_AFTER])
    {
        let mut witness = build_witness_vector(enter);
        witness[column] += F::ONE;
        common::assert_rejected(&witness, "forged enter/reset/save");
    }
    let last = trace
        .iter()
        .find(|r| {
            r.state_before.event_absorb.object_active
                && usize::from(r.state_before.event_absorb.perm_round) + 1 == COMM_CHAIN_PERM_ROWS
        })
        .unwrap();
    for column in COL_OUTER_CHAIN_AFTER
        .into_iter()
        .chain(COL_COMM_CHAIN_AFTER)
        .chain([COL_OBJECT_ACTIVE_AFTER])
    {
        let mut witness = build_witness_vector(last);
        witness[column] += F::ONE;
        common::assert_rejected(&witness, "forged object permutation target or result");
    }
    let root_last = trace
        .iter()
        .find(|r| {
            r.host_event_rom_slot
                .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::OpaqueRoot && rom.arg == 3)
        })
        .unwrap();
    let mut forged = root_last.clone();
    forged.state_after.event_absorb.object_active = true;
    common::assert_rejected(&build_witness_vector(&forged), "missing exit with regenerated advice");
    for column in COL_COMM_CHAIN_AFTER
        .into_iter()
        .chain(COL_OUTER_CHAIN_AFTER)
    {
        let mut witness = build_witness_vector(root_last);
        witness[column] += F::ONE;
        common::assert_rejected(&witness, "forged outer-chain restoration or saved chain");
    }

    let layout = neo_wasm::build_wasm_relation_layout();
    let before: Vec<_> = trace.iter().map(build_witness_vector).collect();
    for column in COL_OUTER_CHAIN_BEFORE
        .into_iter()
        .chain(COL_OUTER_PREFIX_BEFORE)
        .chain([COL_OBJECT_ACTIVE_BEFORE])
    {
        let mut rows = before.clone();
        rows[1][column] += F::ONE;
        assert!(neo_application::check_continuity_rows(&layout.auxiliary.continuity, &rows).is_err());
    }
}

#[test]
fn halted_export_is_not_terminal_until_opaque_exit_finishes() {
    let trace = &checked().trace;
    assert!(trace
        .iter()
        .any(|row| row.state_after.halted && row.state_after.event_absorb.object_active));
    for (index, row) in trace.iter().enumerate() {
        assert_eq!(row.state_after.is_terminal(), index + 1 == trace.len());
    }
    let final_state = trace.last().unwrap().state_after;
    for mutation in 0..6 {
        let mut incomplete = final_state;
        match mutation {
            0 => incomplete.halted = false,
            1 => incomplete.event_absorb.object_active = true,
            2 => incomplete.event_absorb.perm_pending = true,
            3 => incomplete.event_absorb.perm_round = 1,
            4 => incomplete.host_events.events_remaining = 1,
            5 => incomplete.host_events.slot_cursor = 1,
            _ => unreachable!(),
        }
        assert!(
            !incomplete.is_terminal(),
            "incomplete terminal-state component {mutation}"
        );
    }
}

#[test]
fn internally_consistent_trace_fails_a_different_schema_rom() {
    let original = checked();
    let mut bindings = original.bindings.clone();
    let export = original.trace[0].state_before.host_events.turn_export_fref;
    let header = &mut bindings.exports.get_mut(&export).unwrap().entry[1].block;
    assert_eq!(header[2], SlotBinding::Const(SCHEMA[0]));
    header[2] = SlotBinding::Const(99);
    bindings
        .validate_against_program(&original.artifacts.tables)
        .unwrap();
    // Both chains are already checked; only the verifier's schema differs.
    let mut preload = neo_wasm::preload_from_program_artifacts(&original.artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
    let rows: Vec<_> = original.trace.iter().map(build_witness_vector).collect();
    assert!(neo_wasm::sanity_check_memory_rows(neo_wasm::build_wasm_relation_layout(), &rows, &preload).is_err());
}

#[test]
fn object_lowering_preserves_import_result_order_and_export_validation() {
    let events = outer(0)
        .with_opaque(
            0,
            SCHEMA,
            vec![
                SlotBinding::ResultElem { limb: Limb::Lo },
                SlotBinding::ArgElem { arg: 0, limb: Limb::Lo },
                SlotBinding::ResultElem { limb: Limb::Hi },
            ],
        )
        .unwrap();
    assert!(ImportTemplate { events, input_count: 0 }
        .validate(1, 1)
        .is_err());
    let events = outer(0)
        .with_opaque(0, SCHEMA, vec![SlotBinding::OutputElem { limb: Limb::Lo }])
        .unwrap();
    assert!(ExportTemplate {
        entry: events,
        exit: vec![],
        entry_input_count: 0
    }
    .validate(0, 1)
    .is_err());
}

#[test]
fn opaque_state_is_bound_across_batches_and_in_the_initial_anchor() {
    let setup = checked();
    let first = setup.trace[0].state_before;
    let export = first.host_events.turn_export_fref;
    let anchor = neo_wasm::host_event_top_level_initial_state_digest(
        &setup.artifacts.tables,
        first.pc,
        &setup.bindings,
        export,
        Default::default(),
    )
    .unwrap();
    assert_eq!(anchor, neo_wasm::semantic_state_digest(first));
    for mutation in 0..9 {
        let mut forged = first;
        match mutation {
            0..=3 => forged.event_absorb.outer_chain[mutation] = 1,
            4..=7 => forged.event_absorb.outer_prefix[mutation - 4] = 1,
            _ => forged.event_absorb.object_active = true,
        }
        assert_ne!(anchor, neo_wasm::semantic_state_digest(forged));
    }
    let batch_size = 8;
    assert!((batch_size..setup.trace.len())
        .step_by(batch_size)
        .any(|i| setup.trace[i].state_before.event_absorb.object_active));
    let batched = neo_wasm::batch::build_batched_wasm_ccs(batch_size).unwrap();
    for index in 0..neo_wasm::batch::batch_count(setup.trace.len(), batch_size) {
        let witness = neo_wasm::batch::build_batched_witness(&setup.trace, batch_size, index);
        batched.sparse_r1cs.is_satisfied_by(&witness).unwrap();
    }
}
