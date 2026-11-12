//! Witness-driven debug memory checker over shared logical memory declarations.

use super::adapters::wasmtime::WasmProgramArtifacts;
use super::relation_layout::WasmRelationLayout;
use super::WasmMemoryId;
use crate::isa::opcode_info_from_code;
use neo_application::{check_memory_rows, MemoryCheckPolicy, MemoryKind, MemoryPreload, RamInitialization};
use neo_math::F;
use std::collections::BTreeMap;

/// Initial cells for the WASM application's 32-bit logical memories.
pub type WasmMemoryPreload = MemoryPreload<WasmMemoryId>;

/// The preload is program-derived only. Stack RAM starts at zero, including the
/// first export's declared locals. Entry inputs arrive through `InputLocal`, call
/// arguments are already in place, and later frames use LocalZero rows for the rest.
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
    let edge_kinds: BTreeMap<_, _> = tables.pc_edge_kinds.iter().copied().collect();
    for entry in &tables.program_decode {
        let pc = narrow(entry.pc, "program_decode.pc");
        let kind = edge_kinds[&entry.pc];
        assert!(kind <= 3 && entry.opcode_code <= u16::MAX.into());
        preload.insert(
            WasmMemoryId::ProgramHeader,
            vec![pc],
            entry.opcode_code + ((kind as u32) << 16),
        );
        // An immediate cell exists only at the pc of an instruction that reads
        // it, so the memories of one table do not own the same cell.
        let opcode = opcode_info_from_code(entry.opcode_code as u16).opcode;
        use crate::isa::WasmOpcode as Op;
        for (reads, memory, value) in [
            (
                opcode == Op::BrTable,
                WasmMemoryId::ProgramBrTableLen,
                entry.br_table_len,
            ),
            (
                opcode.uses_local_index_immediate(),
                WasmMemoryId::ProgramLocalIndex,
                entry.local_index,
            ),
            (
                opcode.uses_global_index_immediate(),
                WasmMemoryId::ProgramGlobalIndex,
                entry.global_index,
            ),
            (
                opcode.uses_table_id_immediate(),
                WasmMemoryId::ProgramTableId,
                entry.table_id,
            ),
            (
                opcode.uses_linear_memory(),
                WasmMemoryId::ProgramMemoryOffset,
                entry.memory_offset,
            ),
            (
                opcode.uses_call_indirect_immediates(),
                WasmMemoryId::ProgramCallIndirectExpectedTypeId,
                entry.call_indirect_expected_type_id,
            ),
            (
                opcode == Op::I32Const,
                WasmMemoryId::ProgramI32ConstValue,
                entry.i32_const_value,
            ),
            (
                opcode == Op::I64Const,
                WasmMemoryId::ProgramI64ConstValueLo,
                entry.i64_const_value_lo,
            ),
            (
                opcode == Op::I64Const,
                WasmMemoryId::ProgramI64ConstValueHi,
                entry.i64_const_value_hi,
            ),
            (
                opcode == Op::RefFunc,
                WasmMemoryId::ProgramRefFuncRef,
                entry.ref_func_ref,
            ),
        ] {
            if reads {
                preload.insert(memory, vec![pc], value);
            }
        }
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
            vec![narrow(pc_before + control_choice, "pc_rom.key")],
            narrow(pc_after, "pc_rom.state_after.pc"),
        );
    }
    for &(pc_before, control_choice, remaining, distance) in &tables.frame_slides {
        let key = vec![narrow(pc_before + control_choice, "frame_slides.key")];
        preload.insert(WasmMemoryId::FrameSlideRemaining, key.clone(), remaining);
        preload.insert(WasmMemoryId::FrameSlideDistance, key, distance);
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

/// Preload the event-template ROM families from verifier-authored bindings
/// and program-derived parameter, host-scalar, and value-width flags:
/// per-slot source descriptors keyed by `(event_base(fref) + event_index, slot_cursor)`
/// (exports number entry events then exit events) and the per-fref event
/// counts. Call after [`preload_from_program_artifacts`] when checking a
/// event-bound trace.
pub fn preload_host_event_tables(
    preload: &mut WasmMemoryPreload,
    bindings: &crate::host_event_bindings::HostEventBindings,
    artifacts: &WasmProgramArtifacts,
) {
    use crate::host_event_bindings::{memory_rom_arg_variant, opaque_control_encoding, EventBlock, Limb, SlotBinding};
    use crate::ir::{WasmHostEventMemoryWidth, WasmHostEventRomVariant, WasmHostEventSlotKind};
    let limb_variant = |limb| match limb {
        Limb::Lo => WasmHostEventRomVariant::LowLimb,
        Limb::Hi => WasmHostEventRomVariant::HighLimb,
    };
    let result_is_i64 = |fref: u32| {
        artifacts
            .tables
            .function_result_is_i64
            .iter()
            .find_map(|&(function_ref, is_i64)| (function_ref == u64::from(fref)).then_some(is_i64))
            .map(|is_i64| u32::try_from(is_i64).expect("static result type flag fits u32"))
            .filter(|is_i64| *is_i64 <= 1)
            .unwrap_or_else(|| panic!("missing or invalid static result type for function {fref}"))
    };
    let result_is_host_scalar = |fref: u32| {
        artifacts
            .tables
            .function_result_is_host_scalar
            .iter()
            .find_map(|&(function_ref, is_host_scalar)| (function_ref == u64::from(fref)).then_some(is_host_scalar))
            .map(|flag| u32::try_from(flag).expect("static result host-scalar flag fits u32"))
            .filter(|flag| *flag <= 1)
            .unwrap_or_else(|| panic!("missing or invalid static result host-scalar flag for function {fref}"))
    };
    let local_is_i64 = |fref: u32, local: u8| {
        artifacts
            .tables
            .function_local_is_i64
            .iter()
            .find_map(|&(function_ref, local_index, is_i64)| {
                (function_ref == u64::from(fref) && local_index == u64::from(local)).then_some(is_i64)
            })
            .map(|is_i64| u32::try_from(is_i64).expect("static local type flag fits u32"))
            .filter(|is_i64| *is_i64 <= 1)
            .unwrap_or_else(|| panic!("missing or invalid static type for function {fref}, local {local}"))
    };
    let local_is_host_scalar = |fref: u32, local: u8| {
        artifacts
            .tables
            .function_local_is_host_scalar
            .iter()
            .find_map(|&(function_ref, local_index, flag)| {
                (function_ref == u64::from(fref) && local_index == u64::from(local)).then_some(flag)
            })
            .map(|flag| u32::try_from(flag).expect("static local host-scalar flag fits u32"))
            .filter(|flag| *flag <= 1)
            .unwrap_or_else(|| panic!("missing or invalid static host-scalar flag for function {fref}, local {local}"))
    };
    let parameter_count = |fref: u32| {
        artifacts
            .tables
            .function_call_metadata
            .iter()
            .find_map(|&(function_ref, metadata)| {
                (function_ref == u64::from(fref)).then_some(crate::ir::function_call_metadata_shape(metadata).0)
            })
            .unwrap_or_else(|| panic!("missing static call metadata for function {fref}"))
    };
    let encode = |fref: u32, source: &SlotBinding| match *source {
        SlotBinding::EnterOpaque | SlotBinding::OpaqueSaved { .. } | SlotBinding::OpaqueRoot { .. } => {
            let (kind, arg) = opaque_control_encoding(source).expect("control slot");
            (kind.code().into(), arg.into(), 0, 0, 0)
        }
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
            result_is_host_scalar(fref),
            u32::from(matches!(limb, Limb::Hi)) * result_is_i64(fref),
        ),
        SlotBinding::InputLocal { local, limb, .. } => (
            u32::from(WasmHostEventSlotKind::InputLocal.code()),
            u32::from(local),
            u32::from(limb_variant(limb).encoded()),
            u32::from(local < parameter_count(fref)) * local_is_host_scalar(fref, local),
            local_is_i64(fref, local),
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
                let key = vec![bindings.event_base(fref) + event_index as u32, slot_index as u32];
                let (kind, arg, variant, immediate0, immediate1) = encode(fref, source);
                // Advice uses the next code range above the raw slot kinds.
                let kind = kind + WasmHostEventSlotKind::COUNT as u32 * u32::from(!event.absorb);
                assert!(kind < 32 && arg < 256 && variant < 16);
                preload.insert(
                    WasmMemoryId::HostEventSlotDescriptor,
                    key.clone(),
                    kind + 32 * arg + 8192 * variant,
                );
                preload.insert(WasmMemoryId::HostEventSlotImmediate0, key.clone(), immediate0);
                preload.insert(WasmMemoryId::HostEventSlotImmediate1, key, immediate1);
            }
        }
    };
    // Import cells pack the biased call count in the low half and the biased
    // return count (zero for atomic) in the high half. Export entry counts
    // also use count + 1: an undeclared fref reads the zero-filled 0
    // and the CCS load rows subtract 1, poisoning the schedule to
    // events_remaining = p-1. See the relation-layout family comment for the full
    // non-termination argument. Export exit counts stay raw: their read key
    // is bound within an already-entered turn.
    for &fref in bindings.imports.keys().chain(bindings.exports.keys()) {
        preload.insert(WasmMemoryId::HostEventBase, vec![fref], bindings.event_base(fref));
    }
    for (&fref, template) in &bindings.imports {
        preload.insert(
            WasmMemoryId::HostEventImportScheduleCount,
            vec![fref],
            template.packed_schedule_counts(),
        );
        insert_slots(
            preload,
            fref,
            template
                .events
                .iter()
                .chain(template.return_events.iter().flatten())
                .collect(),
        );
    }
    for (&fref, template) in &bindings.exports {
        preload.insert(
            WasmMemoryId::HostEventExportScheduleCount,
            vec![fref],
            template.packed_schedule_counts(),
        );
        insert_slots(preload, fref, template.entry.iter().chain(&template.exit).collect());
    }
}

/// The preload of the tables of the catalog: each cell of `preload` under the
/// table of its memory (`WasmMemoryId::table`). Refuses a preload in which two
/// memories of one table own the same cell, because a read of that cell would
/// not have one meaning.
pub fn table_preload(preload: &WasmMemoryPreload) -> Result<WasmMemoryPreload, String> {
    let mut tables = WasmMemoryPreload::default();
    let mut owners = BTreeMap::new();
    for (memory, address, value) in preload.entries() {
        let table = memory.table();
        if let Some(owner) = owners.insert((table, address.clone()), memory) {
            return Err(format!(
                "memories `{owner}` and `{memory}` own cell {address:?} of table `{table}`"
            ));
        }
        tables.insert(table, address, value);
    }
    Ok(tables)
}

pub fn sanity_check_memory_rows(
    layout: &WasmRelationLayout,
    witness_rows: &[Vec<F>],
    preload: &WasmMemoryPreload,
) -> Result<(), String> {
    let columns = crate::witness_layout::range_checked_column_registry();
    let policy = wasm_memory_check_policy(layout)?;
    let preload = &table_preload(preload)?;
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
        WasmMemoryId::HostContextExport
        | WasmMemoryId::HostContextImport
        | WasmMemoryId::HostContextArgsBase
        | WasmMemoryId::HostContextParentDepth
        | WasmMemoryId::HostContextSp
        | WasmMemoryId::CallStackReturnPc
        | WasmMemoryId::CallStackCallerFbp
        | WasmMemoryId::CallStackCallerSpBase
        | WasmMemoryId::TableSize => Some(RamInitialization::Explicit),
        WasmMemoryId::Stack
        | WasmMemoryId::LinearMemory
        | WasmMemoryId::GlobalLo
        | WasmMemoryId::GlobalHi
        | WasmMemoryId::TableElement => Some(RamInitialization::Zero),
        _ => None,
    }
}
