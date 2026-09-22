//! Automatic export-input recovery through real Wasmtime capture and replay.

mod common;

use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, Limb, MemoryBase, SlotBinding};
use neo_wasm::{
    WasmBuildError, WasmProgramArtifacts, WasmVmStep, WasmtimeTraceHandler, WasmtimeTraceRegistry, WasmtimeTraceStep,
};
use wasmtime::{Config, Engine, Instance, Linker, Module, Store, Val};

struct Runtime {
    engine: Engine,
    store: Store<WasmtimeTraceRegistry>,
}

impl Runtime {
    fn new() -> Self {
        let mut config = Config::new();
        config.guest_debug(true);
        config.wasm_reference_types(true);
        config.wasm_function_references(true);
        config.wasm_tail_call(true);
        let engine = Engine::new(&config).unwrap();
        let mut store = Store::new(&engine, WasmtimeTraceRegistry::default());
        store.set_debug_handler(WasmtimeTraceHandler::new());
        store.edit_breakpoints().unwrap().single_step(true).unwrap();
        Self { engine, store }
    }

    fn instantiate(&mut self, wat: &str, bindings: &HostEventBindings) -> (Instance, WasmProgramArtifacts) {
        let wasm = wat::parse_str(wat).unwrap();
        let mut artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).unwrap();
        artifacts.host_event_bindings = bindings.clone();
        bindings
            .validate_against_program(&artifacts.tables)
            .unwrap();
        self.store
            .data_mut()
            .register_module(&wasm, bindings.clone())
            .unwrap();
        let module = Module::new(&self.engine, &wasm).unwrap();
        let instance =
            futures::executor::block_on(Linker::new(&self.engine).instantiate_async(&mut self.store, &module)).unwrap();
        (instance, artifacts)
    }

    fn call(&mut self, instance: Instance, args: &[Val]) {
        let function = instance.get_func(&mut self.store, "run").unwrap();
        futures::executor::block_on(function.call_async(&mut self.store, args, &mut [])).unwrap();
    }

    fn steps(&self, instance: Instance) -> &[WasmtimeTraceStep] {
        self.store
            .data()
            .instance(instance.debug_index_in_store())
            .unwrap()
            .steps()
    }

    fn write(&mut self, instance: Instance, address: usize, bytes: &[u8]) {
        instance
            .get_memory(&mut self.store, "memory")
            .unwrap()
            .write(&mut self.store, address, bytes)
            .unwrap();
    }
}

fn template(inputs: u8, slots: &[SlotBinding]) -> ExportTemplate {
    let mut block = [SlotBinding::Const(0); 8];
    block[..slots.len()].copy_from_slice(slots);
    ExportTemplate {
        entry: vec![EventBlock { block, absorb: true }],
        exit: vec![],
        entry_input_count: inputs,
    }
}

fn local(input: u8, local: u8, limb: Limb) -> SlotBinding {
    SlotBinding::InputLocal { input, local, limb }
}

fn normalize(steps: &[WasmtimeTraceStep], artifacts: &WasmProgramArtifacts) -> Result<Vec<WasmVmStep>, WasmBuildError> {
    neo_wasm::traces_from_wasmtime_steps_with_host_events(steps, artifacts, Default::default())
}

fn check_entry_values(
    steps: &[WasmtimeTraceStep],
    artifacts: &WasmProgramArtifacts,
    inputs: &[Vec<u64>],
) -> Vec<WasmVmStep> {
    let bindings = &artifacts.host_event_bindings;
    let trace = normalize(steps, artifacts).unwrap();
    let mut turn = 0;
    for row in &trace {
        if row.row_kind == neo_wasm::WasmRowKind::Aux(neo_wasm::WasmAuxOpcode::TurnBoundary) {
            turn += 1;
        }
        let Some(rom) = row.host_event_rom_slot else { continue };
        if !matches!(
            rom.kind,
            neo_wasm::WasmHostEventSlotKind::InputLocal | neo_wasm::WasmHostEventSlotKind::MemoryWrite
        ) {
            continue;
        }
        let state = row.state_before;
        assert_eq!(state.host_callee_fref, state.host_events.turn_export_fref);
        let template = &bindings.exports[&state.host_events.turn_export_fref];
        let slot = usize::from(state.host_events.slot_cursor);
        let input = match template.entry[state.host_events.event_index as usize].block[slot] {
            SlotBinding::InputLocal { input, .. }
            | SlotBinding::MemoryWrite8 { input, .. }
            | SlotBinding::MemoryWrite16 { input, .. }
            | SlotBinding::MemoryWrite32 { input, .. } => input,
            _ => panic!("unexpected entry source"),
        };
        assert_eq!(
            row.state_after.event_absorb.evbuf[slot],
            inputs[turn][usize::from(input)]
        );
    }
    assert_eq!(turn + 1, inputs.len());
    common::sanity_check_trace_with_bindings(&trace, artifacts, bindings);
    common::ccs_check_trace(&trace);
    trace
}

#[test]
fn entry_locals_shared_inputs_nested_calls_and_tail_calls_across_turns() {
    for indirect in [false, true] {
        for top_level_tail in [false, true] {
            let tail = if indirect {
                "i32.const 0 return_call_indirect (type $unary)"
            } else {
                "return_call $leaf"
            };
            let call = if top_level_tail {
                "return_call $middle"
            } else {
                "call $middle"
            };
            let wat = format!(
                r#"(module
                (type $unary (func (param i32)))
                (table 1 funcref) (elem (i32.const 0) $leaf)
                (func $leaf (export "leaf") (param i32) local.get 0 drop)
                (func $middle (param i32) local.get 0 {tail})
                (func (export "run") (param i32 i64 i32) local.get 0 {call}))"#
            );
            let mut bindings = HostEventBindings::default();
            bindings
                .exports
                .insert(1, template(1, &[local(0, 0, Limb::Lo)]));
            bindings.exports.insert(
                3,
                template(
                    3,
                    &[
                        local(0, 0, Limb::Lo),
                        local(1, 1, Limb::Lo),
                        local(2, 1, Limb::Hi),
                        local(0, 2, Limb::Lo),
                    ],
                ),
            );
            let mut runtime = Runtime::new();
            let (instance, artifacts) = runtime.instantiate(&wat, &bindings);
            runtime.call(instance, &[Val::I32(-1), Val::I64(0x1234_5678_8765_4321), Val::I32(-1)]);
            runtime.call(instance, &[Val::I32(9), Val::I64(-2), Val::I32(9)]);
            check_entry_values(
                runtime.steps(instance),
                &artifacts,
                &[
                    vec![u32::MAX.into(), 0x8765_4321, 0x1234_5678],
                    vec![9, u64::from(u32::MAX - 1), u32::MAX.into()],
                ],
            );
            let mut conflicting = runtime.steps(instance).to_vec();
            conflicting[0].locals_words[2].0 = 7;
            assert!(normalize(&conflicting, &artifacts)
                .unwrap_err()
                .to_string()
                .contains("conflicting mappings"));
            let mut missing = runtime.steps(instance).to_vec();
            missing[0].locals_words.pop();
            assert!(normalize(&missing, &artifacts)
                .unwrap_err()
                .to_string()
                .contains("locals snapshot length"));
        }
    }
}

fn memory_bindings(fref: u32) -> HostEventBindings {
    let base = MemoryBase::Local(0);
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        fref,
        template(
            4,
            &[
                local(0, 0, Limb::Lo),
                local(1, 1, Limb::Lo),
                SlotBinding::MemoryWrite32 {
                    input: 1,
                    base,
                    byte_offset: 0,
                },
                SlotBinding::MemoryWrite8 {
                    input: 2,
                    base,
                    byte_offset: 4,
                },
                SlotBinding::MemoryWrite16 {
                    input: 3,
                    base,
                    byte_offset: 6,
                },
            ],
        ),
    );
    bindings
}

const MEMORY_WAT: &str = r#"(module
    (memory (export "memory") 1)
    (data (i32.const 16) "\44\33\22\11\00\aa\00\00")
    (func $clobber i32.const 16 i32.const 0 i32.store)
    (func (export "run") (param i32 i32)
        call $clobber
        local.get 0 i32.load drop
        local.get 0 i32.load8_u offset=4 drop
        local.get 0 i32.load16_u offset=6 drop))"#;

#[test]
fn entry_memory_uses_pre_instruction_bytes_and_preserves_history_across_instances_and_turns() {
    let mut runtime = Runtime::new();
    let bindings = memory_bindings(2);
    let (a, artifacts) = runtime.instantiate(MEMORY_WAT, &bindings);
    let (b, _) = runtime.instantiate(MEMORY_WAT, &bindings);
    // Interleave two instances of the same module: identical frefs and addresses.
    for (instance, word, byte, half) in [
        (a, 0x8765_4321u32, 7u8, 0xfedcu16),
        (b, 19, 0, 21),
        (a, 23, 24, 25),
        (b, 0, 28, 29),
    ] {
        let mut bytes = word.to_le_bytes().to_vec();
        bytes.extend([byte, 0xaa]);
        bytes.extend(half.to_le_bytes());
        runtime.write(instance, 16, &bytes);
        runtime.call(instance, &[Val::I32(16), Val::I32(word as i32)]);
    }
    for (instance, inputs) in [
        (a, [[16, 0x8765_4321, 7, 0xfedc], [16, 23, 24, 25]]),
        (b, [[16, 19, 0, 21], [16, 0, 28, 29]]),
    ] {
        let steps = runtime.steps(instance);
        let captures: Vec<_> = steps
            .iter()
            .filter_map(|step| step.entry_memory.as_ref())
            .collect();
        assert_eq!(captures.len(), 2);
        for capture in captures {
            let bytes = capture.as_ref().unwrap();
            assert_eq!(bytes.len(), 7, "capture only the declared bytes");
            assert!(!bytes.contains_key(&21), "unselected byte is not captured");
        }
        let inputs = inputs.map(|entry| entry.to_vec());
        let trace = check_entry_values(steps, &artifacts, &inputs);
        let writes: Vec<_> = trace
            .iter()
            .filter(|row| {
                row.host_event_rom_slot
                    .is_some_and(|slot| slot.kind == neo_wasm::WasmHostEventSlotKind::MemoryWrite)
            })
            .collect();
        assert_eq!(writes.len(), 6);
        assert_eq!(writes[0].linear_memory.unwrap().lane0.value_before, 0x1122_3344);
        assert_eq!(
            writes[3].linear_memory.unwrap().lane0.value_before,
            0,
            "second turn continues the guest's memory history"
        );
    }
}

#[test]
fn cross_instance_call_captures_the_callee_memory_with_a_live_caller_frame() {
    let mut runtime = Runtime::new();
    let bindings_a = memory_bindings(2);
    let (a, artifacts_a) = runtime.instantiate(MEMORY_WAT, &bindings_a);
    let wasm_b = wat::parse_str(
        r#"(module
        (import "a" "run" (func $a (param i32 i32)))
        (memory (export "memory") 1)
        (data (i32.const 16) "\44\33\22\11\00\aa\00\00")
        (func (export "run") (param i32 i32)
            i32.const 16 i32.const 11 call $a
            local.get 0 i32.load drop
            local.get 0 i32.load8_u offset=4 drop
            local.get 0 i32.load16_u offset=6 drop))"#,
    )
    .unwrap();
    let mut artifacts_b = neo_wasm::extract_wasm_program_artifacts(&wasm_b).unwrap();
    let mut bindings_b = memory_bindings(2);
    bindings_b.imports.insert(1, Default::default());
    runtime
        .store
        .data_mut()
        .register_module(&wasm_b, bindings_b.clone())
        .unwrap();
    let module = Module::new(&runtime.engine, wasm_b).unwrap();
    let mut linker = Linker::new(&runtime.engine);
    let callee = a.get_func(&mut runtime.store, "run").unwrap();
    linker.define(&runtime.store, "a", "run", callee).unwrap();
    let b = futures::executor::block_on(linker.instantiate_async(&mut runtime.store, &module)).unwrap();
    artifacts_b.host_event_bindings = bindings_b;
    runtime.write(a, 16, &[11, 0, 0, 0, 12, 0xaa, 13, 0]);
    runtime.write(b, 16, &[22, 0, 0, 0, 23, 0xaa, 24, 0]);
    runtime.call(b, &[Val::I32(16), Val::I32(22)]);
    check_entry_values(runtime.steps(a), &artifacts_a, &[vec![16, 11, 12, 13]]);
    check_entry_values(runtime.steps(b), &artifacts_b, &[vec![16, 22, 23, 24]]);
}

#[test]
fn entry_memory_rejects_missing_capture_and_conflicts() {
    let mut runtime = Runtime::new();
    let mut bindings = memory_bindings(2);
    // TODO: consider forbidding shared inputs statically in the template.
    let template = bindings.exports.get_mut(&2).unwrap();
    template.entry_input_count = 3;
    template.entry[0].block[4] = SlotBinding::MemoryWrite16 {
        input: 2,
        base: MemoryBase::Local(0),
        byte_offset: 6,
    };
    let (instance, artifacts) = runtime.instantiate(MEMORY_WAT, &bindings);
    runtime.write(instance, 16, &[42, 0, 0, 0, 7, 0xaa, 7, 0]);
    runtime.call(instance, &[Val::I32(16), Val::I32(42)]);
    let steps = runtime.steps(instance);
    check_entry_values(steps, &artifacts, &[vec![16, 42, 7]]);
    for absent in [true, false] {
        let mut bad = steps.to_vec();
        if absent {
            bad[0].entry_memory = None;
        } else {
            bad[0]
                .entry_memory
                .as_mut()
                .unwrap()
                .as_mut()
                .unwrap()
                .remove(&16);
        }
        assert!(normalize(&bad, &artifacts)
            .unwrap_err()
            .to_string()
            .contains("missing entry memory"));
    }
    // Input 1 agrees between a local and memory; input 2 agrees between two addresses.
    for (address, value) in [(16, 43), (22, 8)] {
        let mut bad = steps.to_vec();
        bad[0]
            .entry_memory
            .as_mut()
            .unwrap()
            .as_mut()
            .unwrap()
            .insert(address, value);
        assert!(normalize(&bad, &artifacts)
            .unwrap_err()
            .to_string()
            .contains("conflicting mappings"));
    }
    let mut high_pointer = steps.to_vec();
    high_pointer[0].locals_words[0].1 = 1;
    assert!(normalize(&high_pointer, &artifacts)
        .unwrap_err()
        .to_string()
        .contains("not a wasm32 pointer"));
    let mut failed_read = steps.to_vec();
    failed_read[0].entry_memory = Some(Err("debug memory unavailable".into()));
    assert!(normalize(&failed_read, &artifacts)
        .unwrap_err()
        .to_string()
        .contains("debug memory unavailable"));
}

#[test]
fn entry_memory_requires_capture_configuration() {
    let mut runtime = Runtime::new();
    let bindings = memory_bindings(2);
    let (instance, mut artifacts) = runtime.instantiate(MEMORY_WAT, &HostEventBindings::default());
    runtime.write(instance, 16, &[42, 0, 0, 0, 7, 0xaa, 8, 0]);
    runtime.call(instance, &[Val::I32(16), Val::I32(42)]);
    let steps = runtime.steps(instance);
    assert!(steps.iter().all(|step| step.entry_memory.is_none()));
    // Changing artifacts after capture cannot recover bytes that were never recorded.
    artifacts.host_event_bindings = bindings;
    assert!(normalize(steps, &artifacts)
        .unwrap_err()
        .to_string()
        .contains("missing entry memory capture"));
}

#[test]
fn recovery_requires_the_actual_entry_row_of_each_turn() {
    for memory_input in [false, true] {
        let wat = r#"(module (memory 1)
            (func (export "run") (param i32)
                i32.const 99 local.set 0 local.get 0 drop))"#;
        let mut slots = vec![local(0, 0, Limb::Lo)];
        if memory_input {
            slots.push(SlotBinding::MemoryWrite8 {
                input: 1,
                base: MemoryBase::Local(0),
                byte_offset: 0,
            });
        }
        let mut bindings = HostEventBindings::default();
        bindings
            .exports
            .insert(1, template(slots.len() as u8, &slots));
        let mut runtime = Runtime::new();
        let (instance, artifacts) = runtime.instantiate(wat, &bindings);
        runtime.call(instance, &[Val::I32(16)]);
        let second_turn = runtime.steps(instance).len();
        runtime.call(instance, &[Val::I32(16)]);
        let steps = runtime.steps(instance);
        for start in [0, second_turn] {
            let mut skipped = steps.to_vec();
            skipped[start].opcode_decoded = None;
            assert!(normalize(&skipped, &artifacts)
                .unwrap_err()
                .to_string()
                .contains("expected entry pc"));
            let mut missing = steps.to_vec();
            missing.drain(start..start + 2);
            // The next captured local has already been changed to 99. Recovery
            // must reject its PC instead of treating 99 as the entry argument.
            assert!(normalize(&missing, &artifacts)
                .unwrap_err()
                .to_string()
                .contains("expected entry pc"));
        }
    }
}

#[test]
fn entry_memory_capture_rejects_overlapping_aliases() {
    let wat = r#"(module (memory 1) (func (export "run") (param i32 i32) nop))"#;
    let mut bindings = memory_bindings(1);
    let block = &mut bindings.exports.get_mut(&1).unwrap().entry[0].block;
    // Different locals alias the same address: overlap detection must use addresses.
    block[3] = SlotBinding::MemoryWrite8 {
        input: 2,
        base: MemoryBase::Local(1),
        byte_offset: 0,
    };
    let mut runtime = Runtime::new();
    let (instance, artifacts) = runtime.instantiate(wat, &bindings);
    runtime.call(instance, &[Val::I32(16), Val::I32(16)]);
    let steps = runtime.steps(instance);
    assert!(steps[0]
        .entry_memory
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap_err()
        .contains("overlapping"));
    assert!(normalize(steps, &artifacts)
        .unwrap_err()
        .to_string()
        .contains("overlapping"));
}

#[test]
fn capture_and_replay_share_host_event_address_rules() {
    for width in [1u32, 2, 4] {
        for (base, byte_offset, has_memory, error) in [
            (16, 4, true, None),
            (65536 - width, 0, true, None),
            (65536, 0, true, Some("out of bounds")),
            (u32::MAX, 1, true, Some("overflows wasm32")),
            (16, 0, false, Some("requires default linear memory")),
            (16, 1, true, (width > 1).then_some("not naturally aligned")),
        ] {
            let memory = if has_memory { "(memory 1)" } else { "" };
            let wat = format!("(module {memory} (func (export \"run\") (param i32) (result i32) local.get 0))");
            let base_local = MemoryBase::Local(0);
            let (write, read) = match width {
                1 => (
                    SlotBinding::MemoryWrite8 {
                        input: 1,
                        base: base_local,
                        byte_offset,
                    },
                    SlotBinding::MemoryRead8 {
                        base: MemoryBase::Output,
                        byte_offset,
                    },
                ),
                2 => (
                    SlotBinding::MemoryWrite16 {
                        input: 1,
                        base: base_local,
                        byte_offset,
                    },
                    SlotBinding::MemoryRead16 {
                        base: MemoryBase::Output,
                        byte_offset,
                    },
                ),
                4 => (
                    SlotBinding::MemoryWrite32 {
                        input: 1,
                        base: base_local,
                        byte_offset,
                    },
                    SlotBinding::MemoryRead32 {
                        base: MemoryBase::Output,
                        byte_offset,
                    },
                ),
                _ => unreachable!(),
            };
            let mut write_bindings = HostEventBindings::default();
            write_bindings
                .exports
                .insert(1, template(2, &[local(0, 0, Limb::Lo), write]));
            let wasm = wat::parse_str(&wat).unwrap();
            let run = neo_wasm::collect_wasmtime_steps(&wasm, &write_bindings, "run", &[base as i32]).unwrap();
            let artifacts = run.artifacts().clone();
            let steps = &run.steps;
            let captured = steps[0].entry_memory.as_ref().unwrap();

            // Resolve the same address from the export's output, so an invalid
            // access reaches replay without entry recovery rejecting it first.
            let mut read_bindings = HostEventBindings::default();
            let mut read_template = template(1, &[local(0, 0, Limb::Lo)]);
            read_template.exit = template(0, &[read]).entry;
            read_bindings.exports.insert(1, read_template);
            let mut read_artifacts = artifacts.clone();
            read_artifacts.host_event_bindings = read_bindings.clone();
            let replay = normalize(steps, &read_artifacts);
            if let Some(expected) = error {
                let capture_error = captured.as_ref().unwrap_err();
                assert!(capture_error.contains(expected), "{capture_error}");
                let replay_error = replay.unwrap_err().to_string();
                assert!(replay_error.contains(expected), "{replay_error}");
                assert!(normalize(steps, &artifacts)
                    .unwrap_err()
                    .to_string()
                    .contains(expected));
            } else {
                assert_eq!(captured.as_ref().unwrap().len(), width as usize);
                for (trace, bindings) in [
                    (normalize(steps, &artifacts).unwrap(), &write_bindings),
                    (replay.unwrap(), &read_bindings),
                ] {
                    common::sanity_check_trace_with_bindings(&trace, &artifacts, bindings);
                    common::ccs_check_trace(&trace);
                }
            }
        }
    }
}

#[test]
fn nested_export_candidate_capture_errors_do_not_start_a_turn() {
    let wat = r#"(module (memory 1)
        (func $leaf (export "leaf") (param i32) nop)
        (func (export "run") i32.const -1 call $leaf))"#;
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        1,
        template(
            2,
            &[
                local(0, 0, Limb::Lo),
                SlotBinding::MemoryWrite8 {
                    input: 1,
                    base: MemoryBase::Local(0),
                    byte_offset: 0,
                },
            ],
        ),
    );
    bindings.exports.insert(2, template(0, &[]));
    let mut runtime = Runtime::new();
    let (instance, artifacts) = runtime.instantiate(wat, &bindings);
    runtime.call(instance, &[]);
    runtime.call(instance, &[]);
    assert!(runtime
        .steps(instance)
        .iter()
        .any(|step| step.entry_memory.as_ref().is_some_and(Result::is_err)));
    check_entry_values(
        runtime.steps(instance),
        &artifacts,
        &[Vec::<u64>::new(), Vec::<u64>::new()],
    );
}

#[test]
fn component_entry_memory_is_captured_after_canonical_argument_lowering() {
    use wasmtime::component::{Component, Linker as ComponentLinker, Val as ComponentVal};

    let wasm = wat::parse_str(
        r#"(component
        (core module $m
            (memory (export "memory") 1)
            (func (export "realloc") (param i32 i32 i32 i32) (result i32) i32.const 16)
            (func (export "run") (param i32 i32)
                local.get 0 i32.load drop
                local.get 0 i32.load8_u offset=4 drop
                local.get 0 i32.load16_u offset=6 drop))
        (core instance $i (instantiate $m))
        (alias core export $i "memory" (core memory $memory))
        (alias core export $i "realloc" (core func $realloc))
        (alias core export $i "run" (core func $run))
        (func (export "run") (param "bytes" (list u8))
            (canon lift (core func $run) (memory $memory) (realloc $realloc))))"#,
    )
    .unwrap();
    let mut artifacts = neo_wasm::extract_first_component_core_program_artifacts(&wasm).unwrap();
    let mut bindings = memory_bindings(2);
    let run = bindings.exports.get_mut(&2).unwrap();
    run.entry_input_count = 5;
    // Local 1 is the list length, not the word stored at the list pointer.
    for (slot, input) in run.entry[0].block[2..5].iter_mut().zip(2..5) {
        match slot {
            SlotBinding::MemoryWrite8 { input: index, .. }
            | SlotBinding::MemoryWrite16 { input: index, .. }
            | SlotBinding::MemoryWrite32 { input: index, .. } => *index = input,
            _ => unreachable!(),
        }
    }
    let mut realloc = template(
        3,
        &[
            local(0, 0, Limb::Lo),
            local(0, 1, Limb::Lo),
            local(1, 2, Limb::Lo),
            local(2, 3, Limb::Lo),
        ],
    );
    realloc.exit = vec![template(0, &[SlotBinding::OutputElem { limb: Limb::Lo }])
        .entry
        .remove(0)];
    bindings.exports.insert(1, realloc);
    bindings
        .validate_against_program(&artifacts.tables)
        .unwrap();
    let mut config = Config::new();
    config.guest_debug(true);
    config.wasm_component_model(true);
    let engine = Engine::new(&config).unwrap();
    artifacts.host_event_bindings = bindings;
    let mut state = WasmtimeTraceRegistry::default();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm) {
        if let wasmparser::Payload::ModuleSection { unchecked_range, .. } = payload.unwrap() {
            state
                .register_module(&wasm[unchecked_range], artifacts.host_event_bindings.clone())
                .unwrap();
        }
    }
    let mut store = Store::new(&engine, state);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    let component = Component::new(&engine, wasm).unwrap();
    let instance =
        futures::executor::block_on(ComponentLinker::new(&engine).instantiate_async(&mut store, &component)).unwrap();
    let function = instance.get_func(&mut store, "run").unwrap();
    for bytes in [[42, 0, 0, 0, 7, 0, 8, 0], [99, 0, 0, 0, 10, 0, 11, 0]] {
        let args = [ComponentVal::List(bytes.into_iter().map(ComponentVal::U8).collect())];
        futures::executor::block_on(function.call_async(&mut store, &args, &mut [])).unwrap();
    }
    check_entry_values(
        store.data().single_instance().unwrap().steps(),
        &artifacts,
        &[
            vec![0, 1, 8],
            vec![16, 8, 42, 7, 8],
            vec![0, 1, 8],
            vec![16, 8, 99, 10, 11],
        ],
    );
}
