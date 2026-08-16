//! Native expansion of host-event templates: the worked resource
//! method from `docs/host-event-grammar-tables.md` §3.2, pinned against
//! hand-built absorb blocks and `commit_event` folds.
//!
//! The discriminants and slot indices below are EXAMPLE embedder data
//! (mirroring `starstream-interleaving-spec`'s `EffectDiscriminant` and
//! `ArgName::idx`); neo-wasm itself never interprets them.

use neo_wasm::comm_chain::{commit_event, COMM_CHAIN_EVENT_ARGS};
use neo_wasm::host_event_bindings::{
    expand_export_entry, expand_import_events, EventBlock, EventBlockBuilder, ExportTemplate, HostEventBindings,
    HostEventBindingsBuilder, ImportTemplate, Limb, MemoryBase, SlotBinding, TurnInputs,
};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

const ZERO: SlotBinding = SlotBinding::Const(0);

#[test]
fn public_builder_pads_blocks_derives_inputs_and_validates_functions() -> Result<(), Box<dyn std::error::Error>> {
    let wasm = wat::parse_str(
        r#"(module
            (import "host" "f" (func $f (param i32) (result i32)))
            (func (export "run") (param i32) (result i32)
                local.get 0
                call $f))"#,
    )?;
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm)?;
    let import_fref = u32::try_from(artifacts.tables.call_targets[0].1).expect("import fref");
    let export_fref = u32::try_from(artifacts.tables.function_entries[0].0).expect("export fref");

    let import_event = EventBlockBuilder::op(10)
        .input_i32(0, 0)?
        .input_i32(1, 1)?
        .input_i32(2, 2)?
        .arg_i32(3, 0)?
        .result(4)?
        .finish();
    let entry_event = EventBlockBuilder::op(20)
        .input_i32(0, 0)?
        .input_local_i32(1, 1, 0)?
        .finish();
    let exit_event = EventBlockBuilder::op(17).output_i32(0)?.finish();

    let mut builder = HostEventBindingsBuilder::new(&artifacts.tables);
    builder.import(import_fref, vec![import_event])?;
    builder.export(export_fref, vec![entry_event], vec![exit_event])?;
    let bindings = builder.finish()?;

    assert_eq!(bindings.imports[&import_fref].input_count, 3);
    assert_eq!(bindings.exports[&export_fref].entry_input_count, 2);
    assert_eq!(bindings.exports[&export_fref].exit_input_count, 0);
    assert_eq!(
        bindings.imports[&import_fref].events[0].block[0],
        SlotBinding::Const(10)
    );
    assert_eq!(bindings.imports[&import_fref].events[0].block[7], SlotBinding::Const(0));

    let duplicate = EventBlockBuilder::op(1)
        .word(0, SlotBinding::Const(2))
        .expect_err("the discriminant already owns word zero");
    assert!(format!("{duplicate:?}").contains("assigned more than once"));
    let duplicate = EventBlockBuilder::op(1)
        .constant_i32(0, 2)?
        .constant_i32(0, 3)
        .expect_err("slot zero was already assigned");
    assert!(duplicate.to_string().contains("block slot 0"));

    let gap = EventBlockBuilder::op(1).input_i32(0, 1)?.finish();
    let mut builder = HostEventBindingsBuilder::new(&artifacts.tables);
    let err = builder
        .import(import_fref, vec![gap])
        .err()
        .expect("builder inputs must form a dense tuple");
    assert!(err.to_string().contains("input 0 is unreferenced"));
    Ok(())
}

#[test]
fn scalar_helpers_address_tagged_and_continuation_blocks() -> Result<(), Box<dyn std::error::Error>> {
    let tagged = EventBlockBuilder::op(9)
        .constant_i64(0, 0x1122_3344_5566_7788)?
        .arg_i64(2, 3)?
        .input_i64(4, 7)?
        .finish();

    assert_eq!(
        tagged.block,
        [
            SlotBinding::Const(9),
            SlotBinding::Const(0x5566_7788),
            SlotBinding::Const(0x1122_3344),
            SlotBinding::ArgElem { arg: 3, limb: Limb::Lo },
            SlotBinding::ArgElem { arg: 3, limb: Limb::Hi },
            SlotBinding::Input { index: 7 },
            SlotBinding::Input { index: 8 },
            ZERO,
        ]
    );

    let continuation = EventBlockBuilder::absorbing()
        .output_i64(0)?
        .memory_read_i64(2, MemoryBase::Arg(1), 12)?
        .memory_write_i64(4, 10, MemoryBase::Arg(2), 20)?
        .constant_i64(6, 0x0000_0002_0000_0001)?
        .finish();

    assert_eq!(
        continuation.block,
        [
            SlotBinding::OutputElem { limb: Limb::Lo },
            SlotBinding::OutputElem { limb: Limb::Hi },
            SlotBinding::MemoryRead32 {
                base: MemoryBase::Arg(1),
                byte_offset: 12,
            },
            SlotBinding::MemoryRead32 {
                base: MemoryBase::Arg(1),
                byte_offset: 16,
            },
            SlotBinding::MemoryWrite32 {
                input: 10,
                base: MemoryBase::Arg(2),
                byte_offset: 20,
            },
            SlotBinding::MemoryWrite32 {
                input: 11,
                base: MemoryBase::Arg(2),
                byte_offset: 24,
            },
            SlotBinding::Const(1),
            SlotBinding::Const(2),
        ]
    );

    assert!(EventBlockBuilder::op(0).arg_i64(6, 0).is_err());
    assert!(EventBlockBuilder::absorbing().arg_i64(7, 0).is_err());

    Ok(())
}

fn slots(entries: &[(usize, SlotBinding)]) -> [SlotBinding; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
}

/// `foo(handle, a: u32, b: u64) -> u32` from the design note: payload tuple
/// `(a, b)` encodes to one ref word `[a, b.lo, b.hi, 0]`.
fn method_template() -> ImportTemplate {
    let arg = |arg, limb| SlotBinding::ArgElem { arg, limb };
    let input = |idx| SlotBinding::Input { index: idx };
    ImportTemplate {
        events: vec![
            // NewRef(size=1) -> payload ref (input 0). Ret=slot2, Size=slot3.
            EventBlock::op(10, slots(&[(2, input(0)), (3, SlotBinding::Const(1))])),
            // RefPush([a, b.lo, b.hi, 0]). PackedRef0..3 = slots 0..3.
            EventBlock::op(
                11,
                slots(&[(0, arg(1, Limb::Lo)), (1, arg(2, Limb::Lo)), (2, arg(2, Limb::Hi))]),
            ),
            // Resume(target, f_id, val_ref) -> (ret_ref, caller).
            // Target=slot0, Val=slot1, Ret=slot2, Caller=slot3, FunctionId1=slot4.
            EventBlock::op(
                0,
                slots(&[
                    (0, input(3)),
                    (1, input(0)),
                    (2, input(1)),
                    (3, input(2)),
                    (4, SlotBinding::Const(77)), // dummy f_id
                ]),
            ),
            // RefGet(ret_ref, 0) -> [r, 0, 0, 0]: the ResultElem Lo slot is
            // also the row that pushes the host result onto the stack.
            // Val(=ref)=slot1, Offset=slot3, PackedRef0/2/4/5 = slots 0/2/4/5.
            EventBlock::op(
                12,
                slots(&[
                    (1, input(1)),
                    (0, SlotBinding::ResultElem { limb: Limb::Lo }),
                    (2, SlotBinding::ResultElem { limb: Limb::Hi }),
                ]),
            ),
        ],
        input_count: 4,
    }
}

#[test]
fn method_template_expands_to_pinned_blocks_and_chain() {
    let template = method_template();
    template
        .validate(3, 1)
        .expect("template validates against arity (3, 1)");

    // handle = 0xAA, a = 5, b = 3·2^32 + 7, result = 42.
    let args = [(0xAA, 0), (5, 0), (7, 3)];
    let result = Some((42, 0));
    let inputs = [100u64, 101, 102, 103]; // payload ref, ret ref, caller, target

    let blocks = expand_import_events(&template, &args, result, &inputs, &[]).expect("expansion");

    assert_eq!(
        blocks,
        vec![
            [10, 0, 0, 100, 1, 0, 0, 0],       // NewRef
            [11, 5, 7, 3, 0, 0, 0, 0],         // RefPush
            [0, 103, 100, 101, 102, 77, 0, 0], // Resume
            [12, 42, 101, 0, 0, 0, 0, 0],      // RefGet (result push)
        ],
    );

    // The chain fold over the blocks equals the manual commit_event sequence.
    let f = Goldilocks::from_u64;
    let mut chain = [Goldilocks::ZERO; 4];
    for block in &blocks {
        let words: [Goldilocks; COMM_CHAIN_EVENT_ARGS] = core::array::from_fn(|i| f(block[1 + i]));
        chain = commit_event(chain, f(block[0]), words);
    }
    let expected = {
        let mut c = [Goldilocks::ZERO; 4];
        c = commit_event(c, f(10), [f(0), f(0), f(100), f(1), f(0), f(0), f(0)]);
        c = commit_event(c, f(11), [f(5), f(7), f(3), f(0), f(0), f(0), f(0)]);
        c = commit_event(c, f(0), [f(103), f(100), f(101), f(102), f(77), f(0), f(0)]);
        commit_event(c, f(12), [f(42), f(101), f(0), f(0), f(0), f(0), f(0)])
    };
    assert_eq!(chain, expected);
}

#[test]
fn zero_arg_import_expands_to_single_const_event() {
    // `burn()`: one event, all slots constant.
    let template = ImportTemplate {
        events: vec![EventBlock::op(7, [ZERO; COMM_CHAIN_EVENT_ARGS])],
        input_count: 0,
    };
    template.validate(0, 0).expect("burn validates");
    let blocks = expand_import_events(&template, &[], None, &[], &[]).expect("expansion");
    assert_eq!(blocks, vec![[7, 0, 0, 0, 0, 0, 0, 0]]);
}

#[test]
fn validation_rejects_unresolvable_templates() {
    let event = |slot: SlotBinding| EventBlock::op(0, slots(&[(0, slot)]));

    let result_lo = SlotBinding::ResultElem { limb: Limb::Lo };
    let result_hi = SlotBinding::ResultElem { limb: Limb::Hi };

    // Arg index beyond the import's arity.
    let template = ImportTemplate {
        events: vec![event(SlotBinding::ArgElem { arg: 2, limb: Limb::Lo })],
        ..Default::default()
    };
    assert!(template.validate(2, 0).is_err());

    // Result reference on a resultless import.
    let template = ImportTemplate {
        events: vec![event(result_lo)],
        ..Default::default()
    };
    assert!(template.validate(0, 0).is_err());

    // A returning import MUST push: the ResultElem Lo slot is the push.
    let template = ImportTemplate {
        events: vec![event(SlotBinding::Const(1))],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_err());

    // ... and must push exactly once.
    let template = ImportTemplate {
        events: vec![event(result_lo), event(result_lo)],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_err());

    // The Hi slot writes the pushed cell's hi lane, so it must follow the
    // Lo slot.
    let template = ImportTemplate {
        events: vec![event(result_hi), event(result_lo)],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_err());
    let template = ImportTemplate {
        events: vec![event(result_lo), event(result_hi)],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_ok());

    // ... and is REQUIRED: a Lo-only template leaves the pushed hi lane as
    // unbound advice (an i32 result absorbs 0).
    let template = ImportTemplate {
        events: vec![event(result_lo)],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_err());

    // Input index beyond the declared count.
    let template = ImportTemplate {
        events: vec![event(SlotBinding::Input { index: 1 })],
        input_count: 1,
    };
    assert!(template.validate(0, 0).is_err());

    // Non-canonical constant.
    let template = ImportTemplate {
        events: vec![event(SlotBinding::Const(u64::MAX))],
        ..Default::default()
    };
    assert!(template.validate(0, 0).is_err());

    // Advice events allow only VM effects and padding.
    let advice = |slot: SlotBinding| {
        let mut block = [ZERO; 8];
        block[0] = slot;
        EventBlock::advice(block)
    };
    let template = ImportTemplate {
        events: vec![advice(result_lo), advice(result_hi)],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_ok());
    let template = ImportTemplate {
        events: vec![advice(SlotBinding::ArgElem { arg: 0, limb: Limb::Lo })],
        ..Default::default()
    };
    assert!(template.validate(1, 0).is_err());
    let template = ImportTemplate {
        events: vec![advice(SlotBinding::Input { index: 0 })],
        input_count: 1,
    };
    assert!(template.validate(0, 0).is_err());
    let template = ImportTemplate {
        events: vec![advice(result_lo), advice(result_hi)],
        input_count: 1,
    };
    assert!(template.validate(0, 1).is_err(), "input words need an absorbing event");
    let template = ExportTemplate {
        entry: vec![EventBlock::advice([ZERO; 8])],
        ..Default::default()
    };
    assert!(template.validate(1, 0).is_err(), "export events must absorb");

    // Argument 0 after the result push (its stack slot holds the result).
    let template = ImportTemplate {
        events: vec![event(result_lo), event(SlotBinding::ArgElem { arg: 0, limb: Limb::Lo })],
        ..Default::default()
    };
    assert!(template.validate(1, 1).is_err());
    // Later arguments stay addressable after the push.
    let template = ImportTemplate {
        events: vec![
            event(result_lo),
            event(result_hi),
            event(SlotBinding::ArgElem { arg: 1, limb: Limb::Lo }),
        ],
        ..Default::default()
    };
    assert!(template.validate(2, 1).is_ok());
}

/// Export entry-phase rules: each local lane written at most once, lo
/// before hi, indices inside the declared input counts, and every
/// `InputLocal` word must fit the 32-bit locals lane.
#[test]
fn export_entry_validation_and_expansion_rules() {
    let event = |slot: SlotBinding| EventBlock::op(0, slots(&[(0, slot)]));

    let output = ExportTemplate {
        exit: vec![event(SlotBinding::OutputElem { limb: Limb::Lo })],
        ..Default::default()
    };
    assert!(output.validate(1, 0).is_err());
    output
        .validate(1, 1)
        .expect("single-result export may bind its output");

    // Input index beyond the phase's declared count.
    let template = ExportTemplate {
        entry: vec![event(SlotBinding::Input { index: 1 })],
        entry_input_count: 1,
        ..Default::default()
    };
    assert!(template.validate(1, 0).is_err());

    // Locals bootstrap is entry-phase only.
    let template = ExportTemplate {
        exit: vec![event(SlotBinding::InputLocal {
            input: 0,
            local: 0,
            limb: Limb::Lo,
        })],
        exit_input_count: 1,
        ..Default::default()
    };
    assert!(template.validate(1, 0).is_err());

    // A local lane written twice is rejected.
    let lo = |local| SlotBinding::InputLocal {
        input: 0,
        local,
        limb: Limb::Lo,
    };
    let hi = |local| SlotBinding::InputLocal {
        input: 1,
        local,
        limb: Limb::Hi,
    };
    let template = ExportTemplate {
        entry: vec![event(lo(0)), event(lo(0))],
        entry_input_count: 2,
        ..Default::default()
    };
    assert!(template.validate(1, 0).is_err());

    // Local index out of range.
    let template = ExportTemplate {
        entry: vec![event(lo(1))],
        entry_input_count: 2,
        ..Default::default()
    };
    assert!(template.validate(1, 0).is_err());

    // A hi-lane write requires (and must follow) its local's lo-lane write,
    // because the lo write zeroes the hi lane.
    let template = ExportTemplate {
        entry: vec![event(hi(0))],
        entry_input_count: 2,
        ..Default::default()
    };
    assert!(template.validate(1, 0).is_err());
    let template = ExportTemplate {
        entry: vec![event(hi(0)), event(lo(0))],
        entry_input_count: 2,
        ..Default::default()
    };
    assert!(template.validate(1, 0).is_err());
    let template = ExportTemplate {
        entry: vec![event(lo(0)), event(hi(0))],
        entry_input_count: 2,
        ..Default::default()
    };
    template.validate(1, 0).expect("lo-then-hi validates");

    // Entry expansion resolves indexed input words — the same index may
    // feed several slots — and rejects a wrong array length or a
    // locals-bound word that does not fit the lane.
    let template = ExportTemplate {
        entry: vec![EventBlock::op(
            9,
            slots(&[
                (0, SlotBinding::Input { index: 1 }),
                (1, lo(0)),
                (2, SlotBinding::Input { index: 0 }),
            ]),
        )],
        entry_input_count: 2,
        ..Default::default()
    };
    template.validate(1, 0).expect("entry template validates");
    let blocks = expand_export_entry(&template, &[7, 500]).expect("entry expansion");
    assert_eq!(blocks, vec![[9, 500, 7, 7, 0, 0, 0, 0]]);
    assert!(expand_export_entry(&template, &[7]).is_err());
    assert!(expand_export_entry(&template, &[1 << 32, 500]).is_err());
}

#[test]
fn program_validation_rejects_output_on_a_resultless_export() {
    let wasm = wat::parse_str("(module (func (export \"run\")))").expect("valid wasm");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let &(fref, entry_pc) = artifacts
        .tables
        .function_entries
        .first()
        .expect("export function entry");
    let fref = u32::try_from(fref).expect("function ref");
    let mut bindings = neo_wasm::host_event_bindings::HostEventBindings::default();
    bindings.exports.insert(
        fref,
        ExportTemplate {
            exit: vec![EventBlockBuilder::absorbing()
                .output_i32(0)
                .expect("valid block")
                .finish()],
            ..Default::default()
        },
    );

    let err =
        neo_wasm::host_event_top_level_initial_state(&artifacts.tables, entry_pc, &bindings, fref, Default::default())
            .expect_err("authoritative initial-state construction must validate bindings");
    assert!(err.to_string().contains("single-result export"));
}

#[test]
fn expansion_rejects_wrong_input_count() {
    let template = method_template();
    let args = [(0, 0), (0, 0), (0, 0)];
    assert!(expand_import_events(&template, &args, Some((0, 0)), &[1, 2, 3], &[]).is_err());
}

#[test]
fn expansion_rejects_non_canonical_input() {
    let template = method_template();
    let args = [(0, 0), (0, 0), (0, 0)];
    assert!(expand_import_events(&template, &args, Some((0, 0)), &[1, 2, 3, u64::MAX], &[]).is_err());
}

#[test]
fn memory_slots_validate_phase_base_and_input_source() {
    let event = |source| EventBlock::op(1, slots(&[(0, source)]));
    let import = ImportTemplate {
        events: vec![event(SlotBinding::MemoryRead32 {
            base: MemoryBase::Local(0),
            byte_offset: 0,
        })],
        input_count: 0,
    };
    assert!(import.validate(1, 0).is_err());

    let import = ImportTemplate {
        events: vec![event(SlotBinding::MemoryWrite32 {
            input: 0,
            base: MemoryBase::Arg(0),
            byte_offset: 0,
        })],
        input_count: 0,
    };
    assert!(import.validate(1, 0).is_err());

    let import = ImportTemplate {
        events: vec![EventBlock::op(
            1,
            slots(&[
                (0, SlotBinding::ResultElem { limb: Limb::Lo }),
                (1, SlotBinding::ResultElem { limb: Limb::Hi }),
                (
                    2,
                    SlotBinding::MemoryRead32 {
                        base: MemoryBase::Arg(0),
                        byte_offset: 0,
                    },
                ),
            ]),
        )],
        input_count: 0,
    };
    assert!(import.validate(1, 1).is_err());

    let export = ExportTemplate {
        entry: vec![event(SlotBinding::MemoryRead32 {
            base: MemoryBase::Local(0),
            byte_offset: 0,
        })],
        ..Default::default()
    };
    assert!(export.validate(1, 0).is_err());

    let export = ExportTemplate {
        exit: vec![event(SlotBinding::MemoryRead32 {
            base: MemoryBase::Local(0),
            byte_offset: 0,
        })],
        ..Default::default()
    };
    assert!(export.validate(1, 1).is_err());

    let export = ExportTemplate {
        exit: vec![event(SlotBinding::MemoryRead32 {
            base: MemoryBase::Output,
            byte_offset: 0,
        })],
        ..Default::default()
    };
    assert!(export.validate(1, 0).is_err());
    export
        .validate(1, 1)
        .expect("single-result export memory may use its captured output pointer");

    let export = ExportTemplate {
        exit: vec![event(SlotBinding::MemoryWrite32 {
            input: 0,
            base: MemoryBase::Local(0),
            byte_offset: 0,
        })],
        exit_input_count: 1,
        ..Default::default()
    };
    assert!(export.validate(1, 0).is_err());

    let pointer = SlotBinding::InputLocal {
        input: 0,
        local: 0,
        limb: Limb::Lo,
    };
    let write = SlotBinding::MemoryWrite32 {
        input: 1,
        base: MemoryBase::Local(0),
        byte_offset: 0,
    };
    let missing_pointer = ExportTemplate {
        entry: vec![event(write)],
        entry_input_count: 2,
        ..Default::default()
    };
    assert!(missing_pointer.validate(1, 0).is_err());

    let late_pointer = ExportTemplate {
        entry: vec![event(write), event(pointer)],
        entry_input_count: 2,
        ..Default::default()
    };
    assert!(late_pointer.validate(1, 0).is_err());

    let ordered = ExportTemplate {
        entry: vec![event(pointer), event(write)],
        entry_input_count: 2,
        ..Default::default()
    };
    ordered
        .validate(1, 0)
        .expect("pointer bootstrap precedes memory write");

    let byte_write = ImportTemplate {
        events: vec![event(SlotBinding::MemoryWrite8 {
            input: 0,
            base: MemoryBase::Arg(0),
            byte_offset: 0,
        })],
        input_count: 1,
    };
    assert!(expand_import_events(&byte_write, &[(0, 0)], None, &[256], &[]).is_err());

    let half_write = ImportTemplate {
        events: vec![event(SlotBinding::MemoryWrite16 {
            input: 0,
            base: MemoryBase::Arg(0),
            byte_offset: 0,
        })],
        input_count: 1,
    };
    assert!(expand_import_events(&half_write, &[(0, 0)], None, &[1 << 16], &[]).is_err());
}

#[test]
fn mismatched_runtime_locals_return_an_error() {
    let runtime_wasm =
        wat::parse_str("(module (func (export \"run\") (result i32) i32.const 0))").expect("runtime wasm");
    let run = neo_wasm::collect_wasmtime_steps(&runtime_wasm, "run", &[]).expect("runtime trace");
    let table_wasm =
        wat::parse_str("(module (func (export \"run\") (param i32) (result i32) local.get 0))").expect("table wasm");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&table_wasm).expect("program artifacts");
    let runtime_fref = run
        .steps
        .first()
        .and_then(|row| row.current_function_ref)
        .expect("runtime export fref");
    let table_fref = u32::try_from(artifacts.tables.function_entries[0].0).expect("table export fref");
    assert_eq!(runtime_fref, table_fref, "fixture requires matching function refs");

    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        runtime_fref,
        ExportTemplate {
            entry: vec![EventBlock::op(
                1,
                slots(&[(
                    0,
                    SlotBinding::InputLocal {
                        input: 0,
                        local: 0,
                        limb: Limb::Lo,
                    },
                )]),
            )],
            entry_input_count: 1,
            ..Default::default()
        },
    );
    let err = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &artifacts.tables,
        &bindings,
        &[TurnInputs {
            entry: vec![0],
            ..Default::default()
        }],
        Default::default(),
    )
    .expect_err("mismatched runtime locals must not panic");
    assert!(err.to_string().contains("runtime locals snapshot"));
}
