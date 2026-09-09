//! Native expansion and validation of host-event templates.
//!
//! The tags and slot indices below are arbitrary embedder data; neo-wasm
//! never interprets them.

mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::host_event_bindings::{
    absorbed_blocks, expand_export_entry, expand_export_exit, expand_import_events, opaque_value_root, EventBlock,
    EventSequenceBuilder, EventSources, ExportTemplate, HostEventBindings, HostEventBindingsBuilder, ImportTemplate,
    Limb, MemoryBase, SlotBinding, TurnInputs,
};
use neo_wasm::CommChainState;

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

    let import_event = EventSequenceBuilder::op(10)
        .memory_write_i32(0, MemoryBase::Arg(0), 0)?
        .memory_write_i32(1, MemoryBase::Arg(0), 4)?
        .memory_write_i32(2, MemoryBase::Arg(0), 8)?
        .arg_i32(0)?
        .result()?
        .finish()?;
    let entry_event = EventSequenceBuilder::op(20)
        .input_local_i32(0, 0)?
        .memory_write_i32(1, MemoryBase::Local(0), 0)?
        .finish()?;
    let exit_event = EventSequenceBuilder::op(17).output_i32()?.finish()?;

    let mut builder = HostEventBindingsBuilder::new(&artifacts.tables);
    builder.import(import_fref, import_event)?;
    builder.export(export_fref, entry_event, exit_event)?;
    let bindings = builder.finish()?;

    assert_eq!(bindings.imports[&import_fref].input_count, 3);
    assert_eq!(bindings.exports[&export_fref].entry_input_count, 2);
    assert_eq!(
        bindings.imports[&import_fref].events[0].block[0],
        SlotBinding::Const(10)
    );
    assert_eq!(bindings.imports[&import_fref].events[0].block[7], SlotBinding::Const(0));

    let gap = EventSequenceBuilder::op(1)
        .memory_write_i32(1, MemoryBase::Arg(0), 0)?
        .finish()?;
    let mut builder = HostEventBindingsBuilder::new(&artifacts.tables);
    let err = builder
        .import(import_fref, gap)
        .err()
        .expect("builder inputs must form a dense tuple");
    assert!(err.to_string().contains("input 0 is unreferenced"));
    Ok(())
}

#[test]
fn core_export_input_local_bootstraps_parameter() -> Result<(), Box<dyn std::error::Error>> {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (param i32) (result i32)
                local.get 0))"#,
    )?;
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm)?;
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "run", &[37])?;
    let export_fref = run
        .steps
        .iter()
        .find_map(|row| row.current_function_ref)
        .expect("export function ref");

    let schema = [1, 2, 3, 4];
    let entry = EventSequenceBuilder::op(1)
        .constant_i64(2)?
        .opaque(schema, EventSources::new().input_local_i32(0, 0))?
        .constant_i64(0x1122_3344_5566_7788)?
        .opaque(schema, EventSources::new().constant_i32(9))?
        .opaque(schema, EventSources::new())?
        .finish()?;

    let exit = EventSequenceBuilder::absorbing().output_i32()?.finish()?;

    let mut builder = HostEventBindingsBuilder::new(&run.program_tables);

    builder.export(export_fref, entry, exit)?;

    let bindings = builder.finish()?;
    let inputs = [TurnInputs { entry: vec![37] }];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &inputs,
        CommChainState::default(),
    )?;

    common::sanity_check_trace_with_bindings(&trace, &artifacts, &bindings);

    common::ccs_check_trace(&trace);

    // The suffix scalar straddles blocks; the third root forces zero padding.
    let first_root = opaque_value_root(schema, &[37])?;
    let second_root = opaque_value_root(schema, &[9])?;
    let third_root = opaque_value_root(schema, &[])?;
    let mut first = [1, 2, 0, 0, 0, 0, 0, 0x5566_7788];
    first[3..7].copy_from_slice(&first_root);
    let mut second = [0x1122_3344, 0, 0, 0, 0, 0, 0, 0];
    second[1..5].copy_from_slice(&second_root);
    let mut third = [0; 8];
    third[..4].copy_from_slice(&third_root);
    assert_eq!(
        neo_wasm::comm_chain::absorbed_event_blocks(&trace)
            .into_iter()
            .map(|event| event.words)
            .collect::<Vec<_>>(),
        vec![first, second, third, [37, 0, 0, 0, 0, 0, 0, 0]],
    );

    let output = trace.last().expect("final row").state_after.output;
    assert!(output.enabled);
    assert_eq!(output.value_lo, 37);

    Ok(())
}

#[test]
fn scalar_helpers_address_tagged_and_continuation_blocks() -> Result<(), Box<dyn std::error::Error>> {
    let tagged = EventSequenceBuilder::op(9)
        .constant_i64(0x1122_3344_5566_7788)?
        .arg_i64(3)?
        .finish()?;

    assert_eq!(
        tagged[0].block,
        [
            SlotBinding::Const(9),
            SlotBinding::Const(0x5566_7788),
            SlotBinding::Const(0x1122_3344),
            SlotBinding::ArgElem { arg: 3, limb: Limb::Lo },
            SlotBinding::ArgElem { arg: 3, limb: Limb::Hi },
            ZERO,
            ZERO,
            ZERO,
        ]
    );

    let continuation = EventSequenceBuilder::absorbing()
        .output_i64()?
        .memory_read_i64(MemoryBase::Arg(1), 12)?
        .memory_write_i64(10, MemoryBase::Arg(2), 20)?
        .constant_i64(0x0000_0002_0000_0001)?
        .finish()?;

    assert_eq!(
        continuation[0].block,
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

    Ok(())
}

#[test]
fn sequence_builder_chunks_across_scalar_boundaries() -> Result<(), Box<dyn std::error::Error>> {
    assert!(EventSequenceBuilder::absorbing().finish()?.is_empty());
    let prefix = EventSequenceBuilder::absorbing()
        .constant_i64(1)?
        .constant_i64(2)?
        .constant_i64(3)?
        .constant_i32(4)?;
    let full = prefix.clone().constant_i32(5)?.finish()?;
    assert_eq!(full.len(), 1, "an exact block must not gain a padding block");
    assert!(full[0].absorb);

    // The argument's low limb ends block zero; its high limb starts block one.
    let template = ImportTemplate {
        events: prefix.clone().arg_i64(0)?.finish()?,
        input_count: 0,
    };
    template.validate(1, 0)?;
    assert!(template.events.iter().all(|event| event.absorb));
    assert_eq!(
        expand_import_events(&template, &[(11, 22)], None, &[], &[])?,
        vec![[1, 0, 2, 0, 3, 0, 4, 11], [22, 0, 0, 0, 0, 0, 0, 0]],
    );

    // The result pair can cross the same boundary without being reordered.
    let template = ImportTemplate {
        events: prefix.clone().result()?.finish()?,
        input_count: 0,
    };
    template.validate(0, 1)?;
    assert_eq!(
        expand_import_events(&template, &[], Some((33, 44)), &[], &[])?,
        vec![[1, 0, 2, 0, 3, 0, 4, 33], [44, 0, 0, 0, 0, 0, 0, 0]],
    );

    // Chunking must not bypass the ban on reading argument zero after the result push.
    let invalid = ImportTemplate {
        events: prefix.clone().result()?.arg_i64(0)?.finish()?,
        input_count: 0,
    };
    assert!(invalid.validate(1, 1).is_err());

    // Export input pairs also preserve their input/local limb mapping across blocks.
    let template = ExportTemplate {
        entry: prefix.input_local_i64(0, 0)?.finish()?,
        entry_input_count: 2,
        ..Default::default()
    };
    template.validate(1, 0)?;
    assert_eq!(
        expand_export_entry(&template, &[55, 66])?,
        vec![[1, 0, 2, 0, 3, 0, 4, 55], [66, 0, 0, 0, 0, 0, 0, 0]],
    );
    Ok(())
}

#[test]
fn sequence_builder_pads_roots_without_splitting_them() -> Result<(), Box<dyn std::error::Error>> {
    let schema = [1, 2, 3, 4];
    // A root requested at word five starts the next block, not across the boundary.
    let events = EventSequenceBuilder::op(7)
        .arg_i64(0)?
        .arg_i64(1)?
        .opaque(schema, EventSources::new().arg_i64(2))?
        // A second root cannot share an outer block, even when four words remain.
        .opaque(schema, EventSources::new())?
        .constant_i32(99)?
        .finish()?;
    let template = ImportTemplate { events, input_count: 0 };
    template.validate(3, 0)?;
    let expanded = expand_import_events(&template, &[(10, 11), (20, 21), (30, 31)], None, &[], &[])?;
    let root = opaque_value_root(schema, &[30, 31])?;
    let empty_root = opaque_value_root(schema, &[])?;
    assert_eq!(
        absorbed_blocks(&template.events, &expanded)?,
        vec![
            [7, 10, 11, 20, 21, 0, 0, 0],
            [root[0], root[1], root[2], root[3], 0, 0, 0, 0],
            [empty_root[0], empty_root[1], empty_root[2], empty_root[3], 99, 0, 0, 0],
        ],
    );
    Ok(())
}

#[test]
fn append_builder_places_opaque_roots_and_preserves_suffixes() -> Result<(), Box<dyn std::error::Error>> {
    let schema = [1, 2, 3, 4];
    let events = EventSequenceBuilder::op(7)
        .arg_i32(0)?
        .arg_i32(1)?
        .opaque(
            schema,
            EventSources::new()
                .arg_i32(2)
                .arg_i64(3)
                .constant_i64(0x1122_3344_5566_7788)
                .constant_i32(50)
                .constant_i32(51)
                .constant_i32(52)
                .constant_i32(53),
        )?
        .constant_i32(99)?
        .finish()?;
    let template = ImportTemplate { events, input_count: 0 };
    template.validate(4, 0)?;
    assert_eq!(
        template.events.len(),
        5,
        "prefix, header, two payload blocks, copy-back"
    );
    assert_eq!(
        template.events[1].block[6],
        SlotBinding::Const(9),
        "unpadded word count"
    );
    let expanded = expand_import_events(&template, &[(10, 0), (20, 0), (30, 0), (40, 41)], None, &[], &[])?;
    let root = opaque_value_root(schema, &[30, 40, 41, 0x5566_7788, 0x1122_3344, 50, 51, 52, 53])?;
    assert_eq!(
        absorbed_blocks(&template.events, &expanded)?,
        vec![[7, 10, 20, root[0], root[1], root[2], root[3], 99]],
    );

    Ok(())
}

#[test]
fn opaque_builder_preserves_wide_export_inputs_and_outputs() -> Result<(), Box<dyn std::error::Error>> {
    let wasm = wat::parse_str(r#"(module (func (export "run") (param i64) (result i64) local.get 0))"#)?;
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm)?;
    let fref = u32::try_from(artifacts.tables.function_entries[0].0)?;
    let schema = [1, 2, 3, 4];
    let entry = EventSequenceBuilder::op(1)
        .opaque(schema, EventSources::new().input_local_i64(0, 0)?)?
        .finish()?;
    let exit = EventSequenceBuilder::op(2)
        .opaque(schema, EventSources::new().output_i64())?
        .finish()?;
    let mut builder = HostEventBindingsBuilder::new(&artifacts.tables);
    builder.export(fref, entry, exit)?;
    let bindings = builder.finish()?;
    let template = &bindings.exports[&fref];
    assert_eq!(template.entry_input_count, 2);
    let root = opaque_value_root(schema, &[11, 22])?;
    assert_eq!(
        absorbed_blocks(&template.entry, &expand_export_entry(template, &[11, 22])?)?,
        vec![[1, root[0], root[1], root[2], root[3], 0, 0, 0]],
    );
    assert_eq!(
        absorbed_blocks(&template.exit, &expand_export_exit(template, Some((11, 22)), &[])?)?,
        vec![[2, root[0], root[1], root[2], root[3], 0, 0, 0]],
    );
    Ok(())
}

#[test]
fn sequence_builder_rejects_invalid_opaque_sources() -> Result<(), Box<dyn std::error::Error>> {
    let schema = [1, 2, 3, 4];
    // Advice does not commit an outer root.
    assert!(EventSequenceBuilder::advice()
        .opaque(schema, EventSources::new())
        .is_err());
    // Low-level slot access cannot smuggle nested opaque instructions into a value.
    assert!(EventSequenceBuilder::absorbing()
        .opaque(schema, EventSources::new().push(SlotBinding::EnterOpaque))?
        .finish()
        .is_err());
    // Deferred lowering still validates the schema's field representation.
    assert!(EventSequenceBuilder::absorbing()
        .opaque([u64::MAX; 4], EventSources::new())?
        .finish()
        .is_err());
    // Wide sources cannot wrap their second input index or memory offset.
    assert!(EventSources::new().input_local_i64(u8::MAX, 0).is_err());
    assert!(EventSources::new()
        .memory_write_i64(u8::MAX, MemoryBase::Arg(0), 0)
        .is_err());
    assert!(EventSources::new()
        .memory_read_i64(MemoryBase::Arg(0), u32::MAX)
        .is_err());
    assert!(EventSources::new()
        .memory_write_i64(0, MemoryBase::Arg(0), u32::MAX)
        .is_err());
    Ok(())
}

fn slots(entries: &[(usize, SlotBinding)]) -> [SlotBinding; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
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
fn direct_templates_reject_unreferenced_declared_inputs() {
    let import = ImportTemplate {
        events: vec![EventBlock::op(
            1,
            slots(&[(
                0,
                SlotBinding::MemoryWrite32 {
                    input: 0,
                    base: MemoryBase::Arg(0),
                    byte_offset: 0,
                },
            )]),
        )],
        input_count: 2,
    };

    let err = import
        .validate(1, 0)
        .expect_err("declared import inputs must all be referenced");

    assert!(dbg!(err.to_string()).contains("unreferenced inputs"));

    let export = ExportTemplate {
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
        entry_input_count: 2,
        ..Default::default()
    };

    let err = export
        .validate(1, 0)
        .expect_err("declared export inputs must all be referenced");

    assert!(err.to_string().contains("unreferenced inputs"));
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

    // Memory-write input index beyond the declared count.
    let template = ImportTemplate {
        events: vec![event(SlotBinding::MemoryWrite32 {
            input: 1,
            base: MemoryBase::Arg(0),
            byte_offset: 0,
        })],
        input_count: 1,
    };
    assert!(template.validate(1, 0).is_err());

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
        events: vec![advice(SlotBinding::MemoryWrite32 {
            input: 0,
            base: MemoryBase::Arg(0),
            byte_offset: 0,
        })],
        input_count: 1,
    };
    assert!(template.validate(1, 0).is_err());
    let template = ImportTemplate {
        events: vec![advice(result_lo), advice(result_hi)],
        input_count: 1,
    };
    assert!(
        template.validate(0, 1).is_err(),
        "recorded input words need an absorbing event"
    );
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

    // Input-local index beyond the declared count.
    let template = ExportTemplate {
        entry: vec![event(SlotBinding::InputLocal {
            input: 1,
            local: 0,
            limb: Limb::Lo,
        })],
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

    // Entry expansion rejects a wrong array length or a locals-bound word
    // that does not fit the lane.
    let template = ExportTemplate {
        entry: vec![EventBlock::op(9, slots(&[(0, lo(0))]))],
        entry_input_count: 1,
        ..Default::default()
    };
    template.validate(1, 0).expect("entry template validates");
    let blocks = expand_export_entry(&template, &[7]).expect("entry expansion");
    assert_eq!(blocks, vec![[9, 7, 0, 0, 0, 0, 0, 0]]);
    assert!(expand_export_entry(&template, &[]).is_err());
    assert!(expand_export_entry(&template, &[1 << 32]).is_err());
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
            exit: EventSequenceBuilder::absorbing()
                .output_i32()
                .expect("valid block")
                .finish()
                .expect("valid schedule"),
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
    let template = ImportTemplate {
        events: vec![EventBlock::op(
            1,
            slots(&[(
                0,
                SlotBinding::MemoryWrite32 {
                    input: 0,
                    base: MemoryBase::Arg(0),
                    byte_offset: 0,
                },
            )]),
        )],
        input_count: 1,
    };
    assert!(expand_import_events(&template, &[(0, 0)], None, &[], &[]).is_err());
}

#[test]
fn expansion_rejects_non_canonical_input() {
    let template = ImportTemplate {
        events: vec![EventBlock::op(
            1,
            slots(&[(
                0,
                SlotBinding::MemoryWrite32 {
                    input: 0,
                    base: MemoryBase::Arg(0),
                    byte_offset: 0,
                },
            )]),
        )],
        input_count: 1,
    };
    assert!(expand_import_events(&template, &[(0, 0)], None, &[u64::MAX], &[]).is_err());
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
