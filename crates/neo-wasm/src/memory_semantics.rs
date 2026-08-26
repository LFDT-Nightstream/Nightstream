//! Witness-driven debug memory checker over shared logical memory declarations.

use super::adapters::wasmtime::WasmProgramArtifacts;
use super::relation_layout::WasmRelationLayout;
use super::WasmMemoryId;
use neo_application::{check_memory_rows, MemoryCheckPolicy, MemoryKind, MemoryPreload, RamInitialization};
use neo_math::F;
use std::collections::BTreeMap;

/// Initial cells for the WASM application's 32-bit logical memories.
pub type WasmMemoryPreload = MemoryPreload<WasmMemoryId>;

/// The preload is program-derived only: the locals RAM starts all-zero
/// ([`RamInitialization::Zero`]), so entry-frame inputs must arrive through the export
/// template's `InputLocal` bootstrap; callee params are written by
/// CallParamInit rows before use.
pub fn preload_from_program_artifacts(artifacts: &WasmProgramArtifacts) -> WasmMemoryPreload {
    let tables = &artifacts.tables;
    let mut preload = WasmMemoryPreload::default();
    // Program-table fields are typed as u64 in the parse output even
    // though every wasm-VM column they feed is width-pinned to u32; the
    // helper centralises the narrowing so an out-of-range static table
    // entry fails loudly with the source field name.
    fn narrow(value: u64, field: &str) -> u32 {
        u32::try_from(value)
            .unwrap_or_else(|_| panic!("program table field `{field}` value {value} does not fit in u32"))
    }
    for entry in &tables.program_decode {
        let pc = narrow(entry.pc, "program_decode.pc");
        preload.insert(WasmMemoryId::ProgramOpcode, vec![pc], entry.opcode_code);
        preload.insert(WasmMemoryId::ProgramLocalIndex, vec![pc], entry.local_index);
        preload.insert(WasmMemoryId::ProgramGlobalIndex, vec![pc], entry.global_index);
        preload.insert(WasmMemoryId::ProgramTableId, vec![pc], entry.table_id);
        preload.insert(WasmMemoryId::ProgramMemoryOffset, vec![pc], entry.memory_offset);
        preload.insert(
            WasmMemoryId::ProgramCallIndirectTypeIndex,
            vec![pc],
            entry.call_indirect_type_index,
        );
        preload.insert(
            WasmMemoryId::ProgramCallIndirectExpectedTypeId,
            vec![pc],
            entry.call_indirect_expected_type_id,
        );
        preload.insert(WasmMemoryId::ProgramI32ConstValue, vec![pc], entry.i32_const_value);
        preload.insert(WasmMemoryId::ProgramI64ConstValueLo, vec![pc], entry.i64_const_value_lo);
        preload.insert(WasmMemoryId::ProgramI64ConstValueHi, vec![pc], entry.i64_const_value_hi);
        preload.insert(WasmMemoryId::ProgramRefFuncRef, vec![pc], entry.ref_func_ref);
    }
    for &(index, lo, hi) in &tables.globals_init {
        preload.insert(WasmMemoryId::GlobalLo, vec![index], lo);
        preload.insert(WasmMemoryId::GlobalHi, vec![index], hi);
    }
    for &(table_id, size) in &tables.table_sizes_init {
        preload.insert(WasmMemoryId::TableSize, vec![table_id], size);
    }
    for &(table_id, index, funcref) in &tables.tables_init {
        preload.insert(WasmMemoryId::TableElement, vec![table_id, index], funcref);
    }
    for &(pc_before, control_choice, pc_after) in &tables.pc_rom {
        preload.insert(
            WasmMemoryId::PcRom,
            vec![
                narrow(pc_before, "pc_rom.state_before.pc"),
                narrow(control_choice, "pc_rom.control_choice"),
            ],
            narrow(pc_after, "pc_rom.state_after.pc"),
        );
    }
    for &(pc_before, edge_kind) in &tables.pc_edge_kinds {
        preload.insert(
            WasmMemoryId::PcEdgeKind,
            vec![narrow(pc_before, "pc_edge_kinds.state_before.pc")],
            narrow(edge_kind, "pc_edge_kinds.edge_kind"),
        );
    }
    for &(pc_before, function_ref) in &tables.pc_function_refs {
        preload.insert(
            WasmMemoryId::PcFunctionRef,
            vec![narrow(pc_before, "pc_function_refs.state_before.pc")],
            narrow(function_ref, "pc_function_refs.function_ref"),
        );
    }
    for &(function_ref, entry_pc) in &tables.function_entries {
        preload.insert(
            WasmMemoryId::FunctionEntry,
            vec![narrow(function_ref, "function_entries.function_ref")],
            narrow(entry_pc, "function_entries.entry_pc"),
        );
    }
    for &(function_ref, type_id) in &tables.function_types {
        preload.insert(
            WasmMemoryId::FunctionType,
            vec![narrow(function_ref, "function_types.function_ref")],
            narrow(type_id, "function_types.type_id"),
        );
    }
    for &(function_ref, metadata) in &tables.function_call_metadata {
        preload.insert(
            WasmMemoryId::FunctionCallMetadata,
            vec![narrow(function_ref, "function_call_metadata.function_ref")],
            narrow(metadata, "function_call_metadata.metadata"),
        );
    }
    for &(function_ref, local_count) in &tables.function_local_counts {
        preload.insert(
            WasmMemoryId::FunctionLocalCount,
            vec![narrow(function_ref, "function_local_counts.function_ref")],
            narrow(local_count, "function_local_counts.local_count"),
        );
    }
    for &(pc_before, function_ref) in &tables.call_targets {
        preload.insert(
            WasmMemoryId::CallTarget,
            vec![narrow(pc_before, "call_targets.state_before.pc")],
            narrow(function_ref, "call_targets.function_ref"),
        );
    }
    for &(raw_type_index, expected_type_id) in &tables.module_types {
        preload.insert(
            WasmMemoryId::ModuleType,
            vec![narrow(raw_type_index, "module_types.raw_type_index")],
            narrow(expected_type_id, "module_types.expected_type_id"),
        );
    }
    // Pack data-section bytes into the word-addressed linear_memory cells.
    // Bytes outside any data segment stay absent from the preload, so
    // ZeroReadDefault catches first reads at those addresses (the wasm spec
    // guarantees them zero at instantiation, and a malicious prover claiming
    // non-zero `value_before` for an uninitialized byte will fail the check).
    if !tables.linear_memory_init.is_empty() {
        let mut packed: BTreeMap<u32, u32> = BTreeMap::new();
        for &(byte_addr, byte_value) in &tables.linear_memory_init {
            let word_addr = narrow(byte_addr / 4, "linear_memory_init.word_addr");
            let byte_index = (byte_addr % 4) as u32;
            let word = packed.entry(word_addr).or_insert(0);
            // Clear any prior occupant at this byte position (later segments
            // override earlier ones in spec order) and OR the new byte in.
            *word &= !(0xffu32 << (byte_index * 8));
            *word |= u32::from(byte_value) << (byte_index * 8);
        }
        for (word_addr, word_value) in packed {
            preload.insert(WasmMemoryId::LinearMemory, vec![word_addr], word_value);
        }
    }
    preload
}

/// Preload the event-template ROM families from verifier-authored bindings:
/// per-slot source descriptors keyed by `(fref, event_index, slot_cursor)`
/// (exports number entry events then exit events) and the per-fref event
/// counts. Call after [`preload_from_program_artifacts`] when checking a
/// event-bound trace.
pub fn preload_host_event_tables(
    preload: &mut WasmMemoryPreload,
    bindings: &crate::host_event_bindings::HostEventBindings,
) {
    use crate::host_event_bindings::{memory_rom_arg_variant, EventBlock, Limb, SlotBinding};
    use crate::ir::{WasmHostEventMemoryWidth, WasmHostEventRomVariant, WasmHostEventSlotKind};
    let limb_variant = |limb| match limb {
        Limb::Lo => WasmHostEventRomVariant::LowLimb,
        Limb::Hi => WasmHostEventRomVariant::HighLimb,
    };
    let encode = |source: &SlotBinding| match *source {
        SlotBinding::Const(value) => (
            u32::from(WasmHostEventSlotKind::Const.code()),
            0,
            0,
            value as u32,
            (value >> 32) as u32,
        ),
        SlotBinding::ArgElem { arg, limb } => (
            u32::from(WasmHostEventSlotKind::Arg.code()),
            u32::from(arg),
            u32::from(limb_variant(limb).encoded()),
            0,
            0,
        ),
        SlotBinding::ResultElem { limb } => (
            u32::from(WasmHostEventSlotKind::Result.code()),
            0,
            u32::from(limb_variant(limb).encoded()),
            0,
            0,
        ),
        SlotBinding::InputLocal { local, limb, .. } => (
            u32::from(WasmHostEventSlotKind::InputLocal.code()),
            u32::from(local),
            u32::from(limb_variant(limb).encoded()),
            0,
            0,
        ),
        SlotBinding::OutputElem { limb } => (
            u32::from(WasmHostEventSlotKind::Output.code()),
            0,
            u32::from(limb_variant(limb).encoded()),
            0,
            0,
        ),
        SlotBinding::MemoryRead32 { base, byte_offset } => {
            let (arg, variant) = memory_rom_arg_variant(base, WasmHostEventMemoryWidth::Word);
            (
                u32::from(WasmHostEventSlotKind::MemoryRead.code()),
                u32::from(arg),
                u32::from(variant.encoded()),
                byte_offset,
                0,
            )
        }
        SlotBinding::MemoryRead8 { base, byte_offset } => {
            let (arg, variant) = memory_rom_arg_variant(base, WasmHostEventMemoryWidth::Byte);
            (
                u32::from(WasmHostEventSlotKind::MemoryRead.code()),
                u32::from(arg),
                u32::from(variant.encoded()),
                byte_offset,
                0,
            )
        }
        SlotBinding::MemoryRead16 { base, byte_offset } => {
            let (arg, variant) = memory_rom_arg_variant(base, WasmHostEventMemoryWidth::Half);
            (
                u32::from(WasmHostEventSlotKind::MemoryRead.code()),
                u32::from(arg),
                u32::from(variant.encoded()),
                byte_offset,
                0,
            )
        }
        SlotBinding::MemoryWrite32 {
            input,
            base,
            byte_offset,
        } => {
            let (arg, variant) = memory_rom_arg_variant(base, WasmHostEventMemoryWidth::Word);
            (
                u32::from(WasmHostEventSlotKind::MemoryWrite.code()),
                u32::from(arg),
                u32::from(variant.encoded()),
                byte_offset,
                u32::from(input),
            )
        }
        SlotBinding::MemoryWrite8 {
            input,
            base,
            byte_offset,
        } => {
            let (arg, variant) = memory_rom_arg_variant(base, WasmHostEventMemoryWidth::Byte);
            (
                u32::from(WasmHostEventSlotKind::MemoryWrite.code()),
                u32::from(arg),
                u32::from(variant.encoded()),
                byte_offset,
                u32::from(input),
            )
        }
        SlotBinding::MemoryWrite16 {
            input,
            base,
            byte_offset,
        } => {
            let (arg, variant) = memory_rom_arg_variant(base, WasmHostEventMemoryWidth::Half);
            (
                u32::from(WasmHostEventSlotKind::MemoryWrite.code()),
                u32::from(arg),
                u32::from(variant.encoded()),
                byte_offset,
                u32::from(input),
            )
        }
    };
    let insert_slots = |preload: &mut WasmMemoryPreload, fref: u32, events: Vec<&EventBlock>| {
        for (event_index, event) in events.into_iter().enumerate() {
            for (slot_index, source) in event.block.iter().enumerate() {
                let key = vec![fref, event_index as u32, slot_index as u32];
                let (kind, arg, variant, immediate0, immediate1) = encode(source);
                // Bit 3 carries the per-event advice flag.
                let kind = kind + WasmHostEventSlotKind::COUNT as u32 * u32::from(!event.absorb);
                preload.insert(WasmMemoryId::HostEventSlotKind, key.clone(), kind);
                preload.insert(WasmMemoryId::HostEventSlotArg, key.clone(), arg);
                preload.insert(WasmMemoryId::HostEventSlotVariant, key.clone(), variant);
                preload.insert(WasmMemoryId::HostEventSlotImmediate0, key.clone(), immediate0);
                preload.insert(WasmMemoryId::HostEventSlotImmediate1, key, immediate1);
            }
        }
    };
    // Count cells in the fref-keyed-from-free-state families store
    // count + 1 (presence bias): an undeclared fref reads the zero-filled 0
    // and the CCS load rows subtract 1, poisoning the schedule to
    // events_remaining = p-1. See the relation-layout family comment for the full
    // non-termination argument. Export exit counts stay raw: their read key
    // is bound within an already-entered turn.
    for (&fref, template) in &bindings.imports {
        preload.insert(
            WasmMemoryId::HostEventImportScheduleCount,
            vec![fref],
            template.events.len() as u32 + 1,
        );
        insert_slots(preload, fref, template.events.iter().collect());
    }
    for (&fref, template) in &bindings.exports {
        preload.insert(
            WasmMemoryId::HostEventExportEntryScheduleCount,
            vec![fref],
            template.entry.len() as u32 + 1,
        );
        preload.insert(
            WasmMemoryId::HostEventExportExitScheduleCount,
            vec![fref],
            template.exit.len() as u32,
        );
        insert_slots(preload, fref, template.entry.iter().chain(&template.exit).collect());
    }
}

pub fn sanity_check_memory_rows(
    layout: &WasmRelationLayout,
    witness_rows: &[Vec<F>],
    preload: &WasmMemoryPreload,
) -> Result<(), String> {
    let columns = crate::witness_layout::range_checked_column_registry();
    let policy = wasm_memory_check_policy(layout)?;
    check_memory_rows(&layout.auxiliary.memory, &columns, witness_rows, preload, &policy)
        .map_err(|error| error.to_string())
}

fn wasm_memory_check_policy(layout: &WasmRelationLayout) -> Result<MemoryCheckPolicy<WasmMemoryId>, String> {
    let ram_initialization = layout
        .auxiliary
        .memory
        .entries()
        .iter()
        .filter(|memory| memory.kind == MemoryKind::Ram)
        .map(|memory| {
            memory_init_mode(memory.id)
                .map(|initialization| (memory.id, initialization))
                .ok_or_else(|| format!("memory semantics missing init-mode coverage for `{}`", memory.id))
        })
        .collect::<Result<Vec<_>, _>>()?;

    MemoryCheckPolicy::new(&layout.auxiliary.memory, ram_initialization)
        .map_err(|error| format!("invalid WASM memory check policy: {error}"))
}

fn memory_init_mode(memory: WasmMemoryId) -> Option<RamInitialization> {
    match memory {
        WasmMemoryId::Stack
        | WasmMemoryId::CallStackReturnPc
        | WasmMemoryId::CallStackCallerFbp
        | WasmMemoryId::CallStackCallerSpBase
        | WasmMemoryId::TableSize => Some(RamInitialization::Explicit),
        WasmMemoryId::LinearMemory
        | WasmMemoryId::LocalLo
        | WasmMemoryId::LocalHi
        | WasmMemoryId::GlobalLo
        | WasmMemoryId::GlobalHi
        | WasmMemoryId::TableElement => Some(RamInitialization::Zero),
        _ => None,
    }
}
