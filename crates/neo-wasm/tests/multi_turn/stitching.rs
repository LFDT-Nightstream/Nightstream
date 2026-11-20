use std::any::Any;
use std::panic::{catch_unwind, resume_unwind};

use neo_math::F;
use neo_wasm::comm_chain::{absorbed_event_blocks, fold_event_blocks, AbsorbedEventBlock, AbsorbedEventMetadata};
use neo_wasm::{WasmOutputState, WasmProgramArtifacts};
use p3_field::PrimeCharacteristicRing;

use super::{common, slots, ComponentVal, EventBlock, ExportTemplate, HostEventBindings, Limb, SlotBinding};
use super::{host_event_top_level_initial_state, TracedTestComponent, WasmVmStep};

const COMPONENT: &str = r#"
    (component
      (core module $counter
        (global $value (mut i32) (i32.const 0))
        (func (export "add") (param i32) (result i32)
          global.get $value local.get 0 i32.add global.set $value global.get $value)
        (func (export "poke") (param i32)
          local.get 0 global.set $value)
        (func (export "read") (result i32)
          global.get $value))
      (core instance $counter (instantiate $counter))
      (alias core export $counter "add" (core func $add))
      (alias core export $counter "poke" (core func $poke))
      (alias core export $counter "read" (core func $read))
      (func (export "add") (param "n" s32) (result s32) (canon lift (core func $add)))
      (func (export "poke") (param "n" s32) (canon lift (core func $poke)))
      (func (export "read") (result s32) (canon lift (core func $read))))
"#;

#[derive(Clone, Copy, Debug)]
enum Call {
    Add(i32),
    Poke(i32),
    Read,
}

const CALLS: [Call; 3] = [Call::Add(7), Call::Poke(-23), Call::Read];

impl Call {
    fn export(self) -> (&'static str, u32) {
        match self {
            Self::Add(_) => ("add", 1),
            Self::Poke(_) => ("poke", 2),
            Self::Read => ("read", 3),
        }
    }

    fn with_offset(self, offset: i32) -> Self {
        match self {
            Self::Add(n) => Self::Add(n + offset),
            Self::Poke(n) => Self::Poke(n - offset),
            Self::Read => Self::Read,
        }
    }

    fn expected(self, counter: &mut i32, transcript: &mut Vec<AbsorbedEventBlock>) -> WasmOutputState {
        let (_, fref) = self.export();
        let mut append = |op, value: i32| {
            transcript.push(AbsorbedEventBlock {
                words: [op, u64::from(value as u32), 0, 0, 0, 0, 0, 0],
                metadata: AbsorbedEventMetadata {
                    attributed_fref: fref,
                    turn_export_fref: fref,
                },
            });
        };
        match self {
            Self::Add(n) => {
                *counter = counter.wrapping_add(n);
                append(8, n);
                append(17, *counter);
            }
            Self::Poke(n) => {
                *counter = n;
                append(9, n);
                return WasmOutputState::ZERO;
            }
            Self::Read => append(18, *counter),
        }
        WasmOutputState {
            enabled: true,
            value_lo: *counter as u32,
            value_hi: 0,
        }
    }

    fn invoke(self, runtime: &mut TracedTestComponent) -> Option<ComponentVal> {
        let (export, _) = self.export();
        let mut result = [ComponentVal::S32(0)];
        match self {
            Self::Add(n) => runtime.call(export, &[ComponentVal::S32(n)], &mut result),
            Self::Poke(n) => {
                runtime.call(export, &[ComponentVal::S32(n)], &mut []);
                return None;
            }
            Self::Read => runtime.call(export, &[], &mut result),
        }
        let [value] = result;
        Some(value)
    }
}

fn bindings() -> HostEventBindings {
    let input = |op| {
        vec![EventBlock::op(
            op,
            slots(&[(
                0,
                SlotBinding::InputLocal {
                    input: 0,
                    local: 0,
                    limb: Limb::Lo,
                },
            )]),
        )]
    };
    let output = |op| {
        vec![EventBlock::op(
            op,
            slots(&[(0, SlotBinding::OutputElem { limb: Limb::Lo })]),
        )]
    };
    let mut bindings = HostEventBindings::default();
    for (call, template) in CALLS.into_iter().zip([
        ExportTemplate {
            entry: input(8),
            exit: output(17),
            entry_input_count: 1,
        },
        ExportTemplate {
            entry: input(9),
            exit: vec![],
            entry_input_count: 1,
        },
        ExportTemplate {
            entry: vec![],
            exit: output(18),
            entry_input_count: 0,
        },
    ]) {
        bindings.exports.insert(call.export().1, template);
    }
    bindings
}

fn check(calls: &[Call]) -> (Vec<WasmVmStep>, WasmProgramArtifacts) {
    let component = wat::parse_str(COMPONENT).unwrap();
    let bindings = bindings();
    let mut runtime = TracedTestComponent::new(&component, &bindings);
    let mut counter = 0;
    let mut expected_events = Vec::new();
    let mut expected_turns = Vec::new();
    for &call in calls {
        let output = call.expected(&mut counter, &mut expected_events);
        let wasmtime_output = output
            .enabled
            .then(|| ComponentVal::S32(output.value_lo as i32));
        assert_eq!(call.invoke(&mut runtime), wasmtime_output, "Wasmtime result");
        expected_turns.push((call.export().1, output, counter as u32));
    }

    let run = runtime.finish();
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &run.artifacts, Default::default())
        .expect("normalize stitched trace");
    let (_, first_fref) = calls[0].export();
    let entry = common::entry_pc_for_function_ref(&run.artifacts, u64::from(first_fref));
    let anchor =
        host_event_top_level_initial_state(&run.artifacts.tables, entry, &bindings, first_fref, Default::default())
            .unwrap();
    assert_eq!(trace[0].state_before, anchor, "initial state");

    let segments: Vec<_> = trace.split(|row| row.row_kind.is_turn_boundary()).collect();
    assert_eq!(segments.len(), expected_turns.len(), "turn count");
    let mut actual_counter = 0;
    for (rows, (fref, output, counter)) in segments.into_iter().zip(expected_turns) {
        let end = common::final_state(rows);
        assert!(end.is_terminal() && !end.trapped, "turn completion");
        assert_eq!(end.host_events.turn_export_fref, fref, "turn attribution");
        assert_eq!(end.output, output, "turn output");
        actual_counter = rows
            .iter()
            .filter_map(|row| row.global_write_value)
            .last()
            .unwrap_or(actual_counter);
        assert_eq!(actual_counter, counter, "counter state");
    }

    assert_eq!(absorbed_event_blocks(&trace), expected_events, "event stream");
    let blocks: Vec<_> = expected_events
        .iter()
        .map(|event| event.words.map(F::from_u64))
        .collect();
    assert_eq!(
        common::final_state(&trace).comm_chain,
        fold_event_blocks(Default::default(), &blocks).canonical_u64(),
        "final transcript",
    );
    common::sanity_check_trace_with_bindings(&trace, &run.artifacts, &bindings);
    common::ccs_check_trace(&trace);
    (trace, run.artifacts)
}

fn failure_kind(error: &(dyn Any + Send)) -> Option<&str> {
    error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())?
        .lines()
        .next()
}

fn replay(calls: &[Call]) -> (Vec<WasmVmStep>, WasmProgramArtifacts) {
    match catch_unwind(|| check(calls)) {
        Ok(run) => {
            super::observe_turns(&run.0);
            run
        }
        Err(failure) => {
            eprintln!("Stitching replay: {calls:?}");
            if let Some(message) = failure_kind(failure.as_ref()) {
                let mut reduced = calls.to_vec();
                let mut index = 0;
                while reduced.len() > 1 && index < reduced.len() {
                    let mut candidate = reduced.clone();
                    candidate.remove(index);
                    if catch_unwind(|| check(&candidate))
                        .is_err_and(|error| failure_kind(error.as_ref()) == Some(message))
                    {
                        reduced = candidate;
                        index = 0;
                    } else {
                        index += 1;
                    }
                }
                eprintln!("Reduced stitching replay: {reduced:?}");
            }
            resume_unwind(failure);
        }
    }
}

#[test]
fn all_three_turn_histories() {
    for first in CALLS {
        for second in CALLS {
            for third in CALLS {
                replay(&[first, second.with_offset(1), third.with_offset(2)]);
            }
        }
    }
}

#[test]
fn swapped_additions_bind_order() {
    let first = [Call::Add(7), Call::Add(35), Call::Read];
    let second = [first[1], first[0], first[2]];
    let (trace, _) = replay(&first);
    let (swapped, _) = replay(&second);
    let end = common::final_state(&trace);
    let swapped_end = common::final_state(&swapped);
    assert_eq!(end.output, swapped_end.output, "replays {first:?} and {second:?}");
    assert_ne!(
        end.comm_chain, swapped_end.comm_chain,
        "replays {first:?} and {second:?}"
    );
}

#[test]
#[ignore = "runs the public memory prover and verifier"]
fn public_memory_replay() {
    let (trace, artifacts) = replay(&[Call::Read, Call::Add(7)]);
    common::shared_host::prove(&trace, &artifacts);
}
