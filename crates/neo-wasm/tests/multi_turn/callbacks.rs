use futures::executor::block_on;
use neo_wasm::comm_chain::{absorb_block, absorbed_event_blocks, AbsorbedEventBlock, AbsorbedEventMetadata};
use neo_wasm::host_event_bindings::{
    EventBlock, ExportTemplate, HostEventBindings, ImportTemplate, Limb, MemoryBase, SlotBinding,
};
use neo_wasm::{
    WasmHostEventSlotKind, WasmOpcode, WasmOutputState, WasmProgramArtifacts, WasmVmStep, WasmtimeTraceHandler,
    WasmtimeTraceRegistry,
};
use wasmtime::{Caller, Config, Engine, Linker, Module, Store, Val};

use super::super::common;

const BASE: usize = 16;
const MEMORY_BYTES: usize = 8;
const EFFECT: u32 = 1;
const RUN: u32 = 2;
const CALLBACK: u32 = 3;
const STORE: u32 = 4;
const READ: u32 = 5;

const MODULE: &str = r#"
    (module
      (import "host" "effect" (func $effect (param i32 i32 i32) (result i32)))
      (memory (export "memory") 1)
      (func (export "run") (param $ptr i32) (param $seed i32) (param $count i32) (result i32)
        local.get $ptr local.get $seed local.get $count call $effect
        local.get $ptr i32.load8_u i32.add)
      (func (export "callback") (param $ptr i32) (param $increment i32) (result i32) (local $sum i32)
        local.get $ptr local.get $ptr i32.load8_u local.get $increment i32.add
        local.tee $sum i32.store8 local.get $sum)
      (func (export "store") (param $ptr i32) (param $value i32)
        local.get $ptr local.get $value i32.store8)
      (func (export "read") (param $ptr i32) (result i32)
        local.get $ptr i32.load8_u))
"#;

#[derive(Clone, Copy, Debug)]
enum Call {
    Effect(usize, u8, u8),
    Store(usize, u8),
    Read(usize),
}

impl Call {
    fn export(self) -> (&'static str, u32, Vec<u32>) {
        match self {
            Self::Effect(cell, seed, count) => ("run", RUN, vec![(BASE + cell) as u32, seed.into(), count.into()]),
            Self::Store(cell, value) => ("store", STORE, vec![(BASE + cell) as u32, value.into()]),
            Self::Read(cell) => ("read", READ, vec![(BASE + cell) as u32]),
        }
    }

    fn vary(self, seed: u8) -> Self {
        let shift_cell = |cell| cell + usize::from(seed & 2);
        match self {
            Self::Effect(cell, value, count) => Self::Effect(shift_cell(cell), value.wrapping_add(seed), count),
            Self::Store(cell, value) => Self::Store(shift_cell(cell), value.wrapping_add(seed)),
            Self::Read(cell) => Self::Read(shift_cell(cell)),
        }
    }
}

fn history(seed: u8, groups: usize) -> Vec<Call> {
    use Call::*;
    let calls = [
        Store(2, 13),
        Effect(1, 9, 0),
        Read(2),
        Effect(3, 91, 2),
        Store(3, 80),
        Effect(3, 201, 1),
        Read(3),
        Effect(1, 54, 2),
        Store(1, 240),
        Effect(1, 129, 1),
        Effect(3, 169, 0),
        Read(1),
    ];
    let mut history = Vec::with_capacity(groups * calls.len());
    for group in 0..groups {
        let variant = usize::from(seed) + group;
        let stride = [1, 5, 7, 11][variant % 4];
        for turn in 0..calls.len() {
            let call = calls[(variant + stride * turn) % calls.len()];
            history.push(call.vary(variant as u8));
        }
    }
    history
}

fn op(tag: u64, slots: impl IntoIterator<Item = SlotBinding>) -> EventBlock {
    let mut block = [SlotBinding::Const(0); 8];
    block[0] = SlotBinding::Const(tag);
    for (word, binding) in block[1..].iter_mut().zip(slots) {
        *word = binding;
    }
    EventBlock { block, absorb: true }
}

fn bindings() -> HostEventBindings {
    use SlotBinding::*;
    let mut bindings = HostEventBindings::default();
    for (fref, params) in [(RUN, 3), (CALLBACK, 2), (STORE, 2), (READ, 1)] {
        bindings.exports.insert(
            fref,
            ExportTemplate {
                entry: vec![op(
                    u64::from(fref) * 10,
                    (0..params).map(|local| InputLocal {
                        input: local,
                        local,
                        limb: Limb::Lo,
                    }),
                )],
                exit: if fref == STORE {
                    vec![]
                } else {
                    vec![op(u64::from(fref) * 10 + 1, [OutputElem { limb: Limb::Lo }])]
                },
                entry_input_count: params,
            },
        );
    }
    bindings.imports.insert(
        EFFECT,
        ImportTemplate {
            events: vec![op(10, (0..3).map(|arg| ArgElem { arg, limb: Limb::Lo }))],
            return_events: Some(vec![op(
                11,
                [
                    MemoryWrite8 {
                        input: 0,
                        base: MemoryBase::Arg(0),
                        byte_offset: 0,
                    },
                    MemoryWrite8 {
                        input: 1,
                        base: MemoryBase::Arg(0),
                        byte_offset: 1,
                    },
                    ResultElem { limb: Limb::Lo },
                    ResultElem { limb: Limb::Hi },
                ],
            )]),
            input_count: 2,
        },
    );
    bindings
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AccessKind {
    GuestLoad,
    GuestStore,
    HostWrite,
}

#[derive(Debug, PartialEq, Eq)]
struct Access {
    kind: AccessKind,
    address: usize,
    before: u32,
    after: u32,
}

#[derive(Default)]
struct ExpectedTurn {
    output: Option<u32>,
    memory: [u8; MEMORY_BYTES],
    accesses: Vec<Access>,
    events: Vec<AbsorbedEventBlock>,
}

impl ExpectedTurn {
    fn event(&mut self, attributed_fref: u32, turn_export_fref: u32, tag: u32, values: &[u32]) {
        let mut words = [0; 8];
        words[0] = u64::from(tag);
        for (word, &value) in words[1..].iter_mut().zip(values) {
            *word = u64::from(value);
        }
        self.events.push(AbsorbedEventBlock {
            words,
            metadata: AbsorbedEventMetadata {
                attributed_fref,
                turn_export_fref,
            },
        });
    }

    fn access(&mut self, kind: AccessKind, cell: usize, write: Option<u8>) -> u8 {
        let word = cell / 4 * 4;
        let word_value = |memory: &[u8; MEMORY_BYTES]| u32::from_le_bytes(memory[word..word + 4].try_into().unwrap());
        let before = word_value(&self.memory);
        if let Some(byte) = write {
            self.memory[cell] = byte;
        }
        self.accesses.push(Access {
            kind,
            address: BASE + cell,
            before,
            after: word_value(&self.memory),
        });
        self.memory[cell]
    }

    fn model(call: Call, memory: [u8; MEMORY_BYTES]) -> Self {
        use AccessKind::*;
        let (_, fref, args) = call.export();
        let mut turn = Self {
            memory,
            ..Self::default()
        };
        turn.event(fref, fref, fref * 10, &args);
        turn.output = match call {
            Call::Effect(cell, seed, callbacks) => {
                turn.event(EFFECT, RUN, 10, &args);
                let mut sum = 0;
                for index in 0..u32::from(callbacks) {
                    let increment = u32::from(seed) + index;
                    turn.event(CALLBACK, CALLBACK, 30, &[args[0], increment]);
                    let value = u32::from(turn.access(GuestLoad, cell, None)) + increment;
                    turn.access(GuestStore, cell, Some(value as u8));
                    turn.event(CALLBACK, CALLBACK, 31, &[value]);
                    sum += value;
                }
                let first = seed.wrapping_add(3);
                let second = seed ^ 0x5a;
                turn.access(HostWrite, cell, Some(first));
                turn.access(HostWrite, cell + 1, Some(second));
                turn.event(EFFECT, RUN, 11, &[first.into(), second.into(), sum]);
                Some(sum + u32::from(turn.access(GuestLoad, cell, None)))
            }
            Call::Store(cell, value) => {
                turn.access(GuestStore, cell, Some(value));
                None
            }
            Call::Read(cell) => Some(u32::from(turn.access(GuestLoad, cell, None))),
        };
        if let Some(output) = turn.output {
            turn.event(fref, fref, fref * 10 + 1, &[output]);
        }
        turn
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Effects {
    split_imports: [usize; 3],
    callbacks: usize,
    sequential_callbacks: usize,
    guest_loads: usize,
    guest_stores: usize,
    host_writes: usize,
}

fn replay(calls: &[Call]) -> (Vec<WasmVmStep>, WasmProgramArtifacts, Effects) {
    eprintln!("Callback history {calls:?}");
    let bytes = wat::parse_str(MODULE).unwrap();
    let bindings = bindings();
    let mut config = Config::new();
    config.guest_debug(true);
    let engine = Engine::new(&config).unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry.register_module(&bytes, bindings.clone()).unwrap();
    let mut store = Store::new(&engine, registry);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    let mut linker = Linker::new(&engine);
    linker
        .func_wrap_async(
            "host",
            "effect",
            |mut caller: Caller<'_, WasmtimeTraceRegistry>, (ptr, seed, count): (i32, i32, i32)| {
                Box::new(async move {
                    let callback = caller.get_export("callback").unwrap().into_func().unwrap();
                    let mut sum = 0;
                    for index in 0..count {
                        let mut result = [Val::I32(0)];
                        callback
                            .call_async(&mut caller, &[Val::I32(ptr), Val::I32(seed + index)], &mut result)
                            .await?;
                        sum += result[0].i32().unwrap();
                    }
                    let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                    memory.write(&mut caller, ptr as usize, &[(seed + 3) as u8, (seed ^ 0x5a) as u8])?;
                    Ok((sum,))
                })
            },
        )
        .unwrap();
    let module = Module::new(&engine, &bytes).unwrap();
    let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    let mut expected = Vec::new();
    let mut image = [0; MEMORY_BYTES];
    for (index, &call) in calls.iter().enumerate() {
        let turn = ExpectedTurn::model(call, image);
        let (name, _, args) = call.export();
        let func = instance.get_func(&mut store, name).unwrap();
        let args: Vec<_> = args.into_iter().map(|arg| Val::I32(arg as i32)).collect();
        let mut result = [Val::I32(0)];
        let result = &mut result[..usize::from(turn.output.is_some())];
        block_on(func.call_async(&mut store, &args, result))
            .unwrap_or_else(|error| panic!("turn {index} {call:?}: {error}"));
        assert_eq!(
            result.first().map(|value| value.i32().unwrap() as u32),
            turn.output,
            "Wasmtime output at turn {index} {call:?}",
        );
        assert_eq!(
            &memory.data(&store)[BASE..BASE + MEMORY_BYTES],
            &turn.memory,
            "Wasmtime memory at turn {index} {call:?}"
        );
        image = turn.memory;
        expected.push(turn);
    }

    let captured = store.data().single_instance().unwrap();
    let artifacts = captured.artifacts();
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(captured.steps(), artifacts, Default::default())
        .expect("normalize callback history");
    let first_fref = calls[0].export().1;
    let pc = common::entry_pc_for_function_ref(artifacts, first_fref.into());
    let anchor =
        neo_wasm::host_event_top_level_initial_state(&artifacts.tables, pc, &bindings, first_fref, Default::default())
            .unwrap();
    assert_eq!(trace[0].state_before, anchor, "initial state");
    assert_eq!(common::observe_turns(&trace), calls.len() - 1, "top-level boundaries");
    let turns: Vec<_> = trace
        .split_inclusive(|row| row.state_after.is_terminal())
        .collect();
    assert_eq!(turns.len(), calls.len(), "completed turns");
    let mut effects = Effects::default();
    let mut chain = anchor.comm_chain;
    for (index, ((rows, turn), &call)) in turns.into_iter().zip(expected).zip(calls).enumerate() {
        let label = format!("turn {index} {call:?}");
        let end = common::final_state(rows);
        assert!(end.is_terminal() && !end.trapped, "{label}");
        assert_eq!(end.host_events.turn_export_fref, call.export().1, "{label}");
        let output = turn
            .output
            .map_or(WasmOutputState::ZERO, |value_lo| WasmOutputState {
                enabled: true,
                value_lo,
                value_hi: 0,
            });
        assert_eq!(end.output, output, "{label}");
        assert_eq!(absorbed_event_blocks(rows), turn.events, "events at {label}");
        for event in &turn.events {
            chain = absorb_block(chain, event.words);
        }
        assert_eq!(end.comm_chain, chain, "transcript at {label}");
        check_accesses(rows, &turn.accesses, &mut effects, &label);
        let nested = rows
            .iter()
            .filter(|row| row.row_kind.is_turn_boundary() && row.state_before.host_events.host_frame_depth != 0);
        let callbacks = nested.clone().count();
        let sequential = nested
            .filter(|row| row.state_before.activation_finished)
            .count();
        let imports = rows
            .iter()
            .filter(|row| row.row_kind.is_host_suspend())
            .count();
        let resumes = rows
            .iter()
            .filter(|row| row.row_kind.is_host_resume())
            .count();
        if let Call::Effect(_, _, count) = call {
            assert_eq!(imports, 1, "split imports at {label}");
            assert_eq!(callbacks, usize::from(count), "callbacks at {label}");
            assert_eq!(
                sequential,
                usize::from(count.saturating_sub(1)),
                "sequential callbacks at {label}"
            );
            effects.split_imports[callbacks] += imports;
        } else {
            assert_eq!((imports, callbacks, sequential), (0, 0, 0), "{label}");
        }
        assert_eq!(resumes, imports, "resumes at {label}");
        effects.callbacks += callbacks;
        effects.sequential_callbacks += sequential;
    }
    common::sanity_check_trace_with_bindings(&trace, artifacts, &bindings);
    common::ccs_check_trace(&trace);
    (trace, artifacts.clone(), effects)
}

fn check_accesses(rows: &[WasmVmStep], expected: &[Access], effects: &mut Effects, label: &str) {
    let actual: Vec<_> = rows
        .iter()
        .filter_map(|row| {
            let memory = row.linear_memory?;
            assert_eq!(
                (memory.width_bytes, memory.lane1, memory.lane2),
                (1, None, None),
                "{label}"
            );
            let kind = match row.opcode {
                WasmOpcode::I32Load8U => {
                    effects.guest_loads += 1;
                    AccessKind::GuestLoad
                }
                WasmOpcode::I32Store8 => {
                    effects.guest_stores += 1;
                    AccessKind::GuestStore
                }
                _ => {
                    assert!(
                        row.host_event_rom_slot
                            .is_some_and(|slot| slot.kind == WasmHostEventSlotKind::MemoryWrite),
                        "{label}"
                    );
                    effects.host_writes += 1;
                    AccessKind::HostWrite
                }
            };
            Some(Access {
                kind,
                address: 4 * memory.lane0.word_addr as usize + usize::from(memory.byte_offset),
                before: memory.lane0.value_before,
                after: memory.lane0.value_after,
            })
        })
        .collect();
    assert_eq!(actual, expected, "memory accesses at {label}");
}

#[test]
fn long_callback_histories() {
    for (seed, groups) in [(0, 1), (19, 2), (42, 4), (253, 3)] {
        let (_, _, effects) = replay(&history(seed, groups));
        assert_eq!(
            effects,
            Effects {
                split_imports: [2 * groups; 3],
                callbacks: 6 * groups,
                sequential_callbacks: 2 * groups,
                guest_loads: 15 * groups,
                guest_stores: 9 * groups,
                host_writes: 12 * groups,
            },
            "seed {seed}"
        );
    }
}

#[test]
#[ignore = "extended callback and memory conformance corpus"]
fn every_three_call_schedule_in_a_long_history() {
    use Call::*;
    let blocks = [
        Read(1),
        Store(1, 129),
        Effect(1, 9, 0),
        Effect(1, 91, 1),
        Effect(1, 201, 2),
    ];
    for (i, first) in blocks.iter().enumerate() {
        for (j, second) in blocks.iter().enumerate() {
            let mut calls = Vec::new();
            for (k, third) in blocks.iter().enumerate() {
                let seed = (i * 25 + j * 5 + k) as u8;
                calls.extend([*first, *second, *third].map(|call| call.vary(seed)));
            }
            replay(&calls);
        }
    }
}

#[test]
#[ignore = "runs the public memory prover and verifier"]
fn public_callback_memory_replay() {
    let (trace, artifacts, _) = replay(&[Call::Effect(3, 9, 2), Call::Read(4)]);
    common::shared_host::prove(&trace, &artifacts);
}
