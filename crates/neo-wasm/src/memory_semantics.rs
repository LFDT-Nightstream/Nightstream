//! Witness-driven debug memory checker over `WasmMemorySpec`.

use super::adapters::wasmtime::WasmProgramArtifacts;
use super::layout::column_spec;
use super::relation_layout::{WasmMemoryActivation, WasmMemoryColumnKind, WasmMemorySpec, WasmRelationLayout};
use super::WasmMemoryId;
use neo_math::F;
use p3_field::PrimeField64;
use std::collections::BTreeMap;

/// Per-column-limb width for the cells log. Every column the wasm VM
/// stores or compares through the memory checker carries one Goldilocks
/// limb that is supposed to be width-pinned to u32 by the bit-decomp /
/// booleanity rows; narrowing here surfaces any column where that
/// pinning is missing or broken — the cells log itself would otherwise
/// happily round-trip a witness value up to `q - 1 ≈ 2^64`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WasmMemoryPreload {
    cells: BTreeMap<WasmMemoryId, BTreeMap<Vec<u32>, u32>>,
}

impl WasmMemoryPreload {
    pub fn insert(&mut self, memory: WasmMemoryId, address: Vec<u32>, value: u32) {
        self.cells.entry(memory).or_default().insert(address, value);
    }

    pub fn remove(&mut self, memory: WasmMemoryId, address: &[u32]) -> Option<u32> {
        self.cells.get_mut(&memory)?.remove(address)
    }

    fn clone_cells(&self) -> BTreeMap<WasmMemoryId, BTreeMap<Vec<u32>, u32>> {
        self.cells.clone()
    }

    pub fn entries(&self) -> Vec<(WasmMemoryId, Vec<u32>, u32)> {
        self.cells
            .iter()
            .flat_map(|(&memory, cells)| {
                cells
                    .iter()
                    .map(move |(address, &value)| (memory, address.clone(), value))
            })
            .collect()
    }
}

/// Read a witness column and narrow to u32, returning a descriptive
/// error if the field-element representative does not fit. A failure
/// here means the column carried a value the witness layer was supposed
/// to pin to 32 bits but did not — i.e., a missing or broken bit-width
/// constraint upstream. Naming the column and row makes the offender
/// trivial to find from the test failure message.
fn read_u32_column(witness: &[F], col: usize, row_index: usize, role: &str) -> Result<u32, String> {
    let canonical = witness[col].as_canonical_u64();
    u32::try_from(canonical).map_err(|_| {
        let name = column_spec(col).map(|spec| spec.name).unwrap_or("?");
        format!(
            "{role} column `{name}` (col {col}) carried value {canonical} that does not fit in u32 on row {row_index} \
             — missing or broken bit-width constraint upstream",
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DebugInitMode {
    Strict,
    ZeroReadDefault,
}

/// The preload is program-derived only: the locals RAM starts all-zero
/// (`ZeroReadDefault`), so entry-frame inputs must arrive through the export
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
        SlotBinding::Input { index: idx } => (u32::from(WasmHostEventSlotKind::Input.code()), u32::from(idx), 0, 0, 0),
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
    assert_all_memory_specs_have_init_modes(layout)?;
    let mut state = preload.clone_cells();
    for (row_index, witness) in witness_rows.iter().enumerate() {
        let expected = crate::RANGE_CHECKED_WITNESS_WIDTH;
        if witness.len() != expected {
            return Err(format!(
                "memory sanity check expected witness width {}, got {} on row {}",
                expected,
                witness.len(),
                row_index
            ));
        }
        for memory in &layout.auxiliary.memories {
            apply_memory_row(memory, witness, row_index, &mut state)?;
        }
    }
    Ok(())
}

fn apply_memory_row(
    memory: &WasmMemorySpec,
    witness: &[F],
    row_index: usize,
    state: &mut BTreeMap<WasmMemoryId, BTreeMap<Vec<u32>, u32>>,
) -> Result<(), String> {
    let cells = state.entry(memory.id).or_default();
    for column in &memory.columns {
        let active = activation_active(column.activation, witness, row_index, memory.id)?;
        if !active {
            continue;
        }
        let address = column
            .address_columns
            .iter()
            .map(|column| read_u32_column(witness, column.0, row_index, "address"))
            .collect::<Result<Vec<_>, _>>()?;
        let value = read_u32_column(witness, column.value_column.0, row_index, "value")?;
        if memory.id.is_rom() {
            match cells.get(&address).copied() {
                Some(expected) if expected != value => {
                    return Err(format!(
                        "memory `{}` ROM mismatch at {:?} on row {}: expected {}, got {}",
                        memory.id, address, row_index, expected, value
                    ));
                }
                Some(_) => {}
                None => {
                    return Err(format!(
                        "memory `{}` ROM read before initialization at {:?} on row {}",
                        memory.id, address, row_index
                    ));
                }
            }
            continue;
        }
        match column.kind {
            WasmMemoryColumnKind::Read => match cells.get(&address).copied() {
                Some(expected) if expected != value => {
                    return Err(format!(
                        "memory `{}` read mismatch at {:?} on row {}: expected {}, got {}",
                        memory.id, address, row_index, expected, value
                    ));
                }
                Some(_) => {}
                None => match init_mode(memory.id) {
                    DebugInitMode::Strict => {
                        return Err(format!(
                            "memory `{}` read before initialization at {:?} on row {}",
                            memory.id, address, row_index
                        ));
                    }
                    DebugInitMode::ZeroReadDefault => {
                        if value != 0 {
                            return Err(format!(
                                "memory `{}` expected zero-default read at {:?} on row {}, got {}",
                                memory.id, address, row_index, value
                            ));
                        }
                        cells.insert(address, 0);
                    }
                },
            },
            WasmMemoryColumnKind::Write { value_before_column } => {
                // Nebula-style RMW: if `value_before_column` is named, this
                // row's read tuple must match the prior write at this address
                // (or the documented init mode). Catches a malicious prover
                // who writes a word whose unmodified bytes don't preserve the
                // prior state — see `i32_store8_row_rejects_tampered_...`.
                if let Some(before_col) = value_before_column {
                    let before_value = read_u32_column(witness, before_col.0, row_index, "value_before")?;
                    match cells.get(&address).copied() {
                        Some(expected) if expected != before_value => {
                            return Err(format!(
                                "memory `{}` RMW read mismatch at {:?} on row {}: \
                                 prior write was {}, witness claims {}",
                                memory.id, address, row_index, expected, before_value
                            ));
                        }
                        Some(_) => {}
                        None => match init_mode(memory.id) {
                            DebugInitMode::Strict => {
                                return Err(format!(
                                    "memory `{}` RMW read before initialization at {:?} on row {}",
                                    memory.id, address, row_index
                                ));
                            }
                            DebugInitMode::ZeroReadDefault => {
                                if before_value != 0 {
                                    return Err(format!(
                                        "memory `{}` expected zero-default RMW read at {:?} on row {}, got {}",
                                        memory.id, address, row_index, before_value
                                    ));
                                }
                                cells.insert(address.clone(), 0);
                            }
                        },
                    }
                }
                cells.insert(address, value);
            }
        }
    }
    Ok(())
}

fn activation_active(
    activation: WasmMemoryActivation,
    witness: &[F],
    row_index: usize,
    memory: WasmMemoryId,
) -> Result<bool, String> {
    match activation {
        WasmMemoryActivation::Always => Ok(true),
        WasmMemoryActivation::BooleanGate(gate) => match witness[gate.0].as_canonical_u64() {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(format!(
                "memory `{}` has non-boolean gate {} on row {}",
                memory, other, row_index
            )),
        },
    }
}

fn init_mode(memory: WasmMemoryId) -> DebugInitMode {
    memory_init_mode(memory)
        .unwrap_or_else(|| panic!("memory semantics missing init-mode coverage for non-ROM memory `{memory}`"))
}

fn memory_init_mode(memory: WasmMemoryId) -> Option<DebugInitMode> {
    match memory {
        WasmMemoryId::Stack
        | WasmMemoryId::CallStackReturnPc
        | WasmMemoryId::CallStackCallerFbp
        | WasmMemoryId::CallStackCallerSpBase
        | WasmMemoryId::TableSize => Some(DebugInitMode::Strict),
        WasmMemoryId::LinearMemory
        | WasmMemoryId::LocalLo
        | WasmMemoryId::LocalHi
        | WasmMemoryId::GlobalLo
        | WasmMemoryId::GlobalHi
        | WasmMemoryId::TableElement => Some(DebugInitMode::ZeroReadDefault),
        _ => None,
    }
}

fn assert_all_memory_specs_have_init_modes(layout: &WasmRelationLayout) -> Result<(), String> {
    for memory in &layout.auxiliary.memories {
        if !memory.id.is_rom() && memory_init_mode(memory.id).is_none() {
            return Err(format!(
                "memory semantics missing init-mode coverage for `{}`",
                memory.id
            ));
        }
    }
    Ok(())
}
