//! Witness-driven debug memory checker over `WasmMemorySpec`.

use super::adapters::wasmtime::WasmProgramArtifacts;
use super::layout::COLUMN_SPECS;
use super::relation_layout::{WasmMemoryActivation, WasmMemoryColumnKind, WasmMemorySpec, WasmRelationLayout};
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
    cells: BTreeMap<&'static str, BTreeMap<Vec<u32>, u32>>,
}

impl WasmMemoryPreload {
    pub fn insert(&mut self, memory: &'static str, address: Vec<u32>, value: u32) {
        self.cells.entry(memory).or_default().insert(address, value);
    }

    pub fn remove(&mut self, memory: &'static str, address: &[u32]) -> Option<u32> {
        self.cells.get_mut(memory)?.remove(address)
    }

    fn clone_cells(&self) -> BTreeMap<&'static str, BTreeMap<Vec<u32>, u32>> {
        self.cells.clone()
    }

    pub fn entries(&self) -> Vec<(&'static str, Vec<u32>, u32)> {
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
        let name = COLUMN_SPECS.get(col).map(|spec| spec.name).unwrap_or("?");
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

pub fn preload_from_program_artifacts(artifacts: &WasmProgramArtifacts, initial_locals: &[u32]) -> WasmMemoryPreload {
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
    // Entry-frame locals use `fbp = 0`; callee params are written by
    // CallParamInit rows before use.
    for (idx, &value) in initial_locals.iter().enumerate() {
        let address = vec![0u32, idx as u32];
        preload.insert("locals", address.clone(), value);
        preload.insert("locals_hi", address, 0);
    }
    for entry in &tables.program_decode {
        let pc = narrow(entry.pc, "program_decode.pc");
        preload.insert("program_opcodes", vec![pc], entry.opcode_code);
        preload.insert("program_local_indices", vec![pc], entry.local_index);
        preload.insert("program_global_indices", vec![pc], entry.global_index);
        preload.insert("program_table_ids", vec![pc], entry.table_id);
        preload.insert("program_memory_offsets", vec![pc], entry.memory_offset);
        preload.insert(
            "program_call_indirect_type_indices",
            vec![pc],
            entry.call_indirect_type_index,
        );
        preload.insert(
            "program_call_indirect_expected_type_ids",
            vec![pc],
            entry.call_indirect_expected_type_id,
        );
        preload.insert("program_i32_const_values", vec![pc], entry.i32_const_value);
        preload.insert("program_i64_const_values_lo", vec![pc], entry.i64_const_value_lo);
        preload.insert("program_i64_const_values_hi", vec![pc], entry.i64_const_value_hi);
        preload.insert("program_ref_func_refs", vec![pc], entry.ref_func_ref);
    }
    for &(index, lo, hi) in &tables.globals_init {
        preload.insert("globals", vec![index], lo);
        preload.insert("globals_hi", vec![index], hi);
    }
    for &(table_id, size) in &tables.table_sizes_init {
        preload.insert("table_sizes", vec![table_id], size);
    }
    for &(table_id, index, funcref) in &tables.tables_init {
        preload.insert("tables", vec![table_id, index], funcref);
    }
    for &(pc_before, control_choice, pc_after) in &tables.pc_rom {
        preload.insert(
            "pc_rom",
            vec![
                narrow(pc_before, "pc_rom.state_before.pc"),
                narrow(control_choice, "pc_rom.control_choice"),
            ],
            narrow(pc_after, "pc_rom.state_after.pc"),
        );
    }
    for &(pc_before, edge_kind) in &tables.pc_edge_kinds {
        preload.insert(
            "pc_edge_kinds",
            vec![narrow(pc_before, "pc_edge_kinds.state_before.pc")],
            narrow(edge_kind, "pc_edge_kinds.edge_kind"),
        );
    }
    for &(pc_before, function_ref) in &tables.pc_function_refs {
        preload.insert(
            "pc_function_refs",
            vec![narrow(pc_before, "pc_function_refs.state_before.pc")],
            narrow(function_ref, "pc_function_refs.function_ref"),
        );
    }
    for &(function_ref, entry_pc) in &tables.function_entries {
        preload.insert(
            "function_entries",
            vec![narrow(function_ref, "function_entries.function_ref")],
            narrow(entry_pc, "function_entries.entry_pc"),
        );
    }
    for &(function_ref, type_id) in &tables.function_types {
        preload.insert(
            "function_types",
            vec![narrow(function_ref, "function_types.function_ref")],
            narrow(type_id, "function_types.type_id"),
        );
    }
    for &(function_ref, metadata) in &tables.function_call_metadata {
        preload.insert(
            "function_call_metadata",
            vec![narrow(function_ref, "function_call_metadata.function_ref")],
            narrow(metadata, "function_call_metadata.metadata"),
        );
    }
    for &(function_ref, local_count) in &tables.function_local_counts {
        preload.insert(
            "function_local_counts",
            vec![narrow(function_ref, "function_local_counts.function_ref")],
            narrow(local_count, "function_local_counts.local_count"),
        );
    }
    for &(pc_before, function_ref) in &tables.call_targets {
        preload.insert(
            "call_targets",
            vec![narrow(pc_before, "call_targets.state_before.pc")],
            narrow(function_ref, "call_targets.function_ref"),
        );
    }
    for &(raw_type_index, expected_type_id) in &tables.module_types {
        preload.insert(
            "module_types",
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
            preload.insert("linear_memory", vec![word_addr], word_value);
        }
    }
    preload
}

/// Preload the grammar-mode ROM families from an embedder grammar: the
/// per-slot source descriptors keyed by `(fref, event_index, slot_cursor)`
/// (exports number entry events then exit events) and the per-fref event
/// counts. Call after [`preload_from_program_artifacts`] when checking a
/// grammar-mode trace.
pub fn preload_grammar_tables(preload: &mut WasmMemoryPreload, grammar: &crate::event_grammar::HostEventGrammar) {
    use crate::event_grammar::{GrammarEvent, Limb, MemoryBase, SlotSource};
    use crate::ir::WasmGrammarSlotKind;
    let limb_bit = |limb| match limb {
        Limb::Lo => 0,
        Limb::Hi => 1,
    };
    let encode = |source: &SlotSource| match *source {
        SlotSource::Const(value) => (
            u32::from(WasmGrammarSlotKind::Const.code()),
            0,
            0,
            value as u32,
            (value >> 32) as u32,
        ),
        SlotSource::ArgElem { arg, limb } => (
            u32::from(WasmGrammarSlotKind::Arg.code()),
            u32::from(arg),
            limb_bit(limb),
            0,
            0,
        ),
        SlotSource::ResultElem { limb } => (u32::from(WasmGrammarSlotKind::Result.code()), 0, limb_bit(limb), 0, 0),
        SlotSource::Claim { idx } => (u32::from(WasmGrammarSlotKind::Claim.code()), u32::from(idx), 0, 0, 0),
        SlotSource::ClaimLocal { local, limb, .. } => (
            u32::from(WasmGrammarSlotKind::ClaimLocal.code()),
            u32::from(local),
            limb_bit(limb),
            0,
            0,
        ),
        SlotSource::OutputElem { limb } => (u32::from(WasmGrammarSlotKind::Output.code()), 0, limb_bit(limb), 0, 0),
        SlotSource::MemoryRead32 { base, byte_offset } => {
            let (arg, base_kind) = match base {
                MemoryBase::Arg(arg) => (arg, 0),
                MemoryBase::Local(local) => (local, 1),
            };
            (
                u32::from(WasmGrammarSlotKind::MemoryRead.code()),
                u32::from(arg),
                base_kind,
                byte_offset,
                0,
            )
        }
        SlotSource::MemoryWrite32 {
            claim,
            base,
            byte_offset,
        } => {
            let (arg, base_kind) = match base {
                MemoryBase::Arg(arg) => (arg, 0),
                MemoryBase::Local(local) => (local, 1),
            };
            (
                u32::from(WasmGrammarSlotKind::MemoryWrite.code()),
                u32::from(arg),
                base_kind,
                byte_offset,
                u32::from(claim),
            )
        }
    };
    let insert_slots = |preload: &mut WasmMemoryPreload, fref: u32, events: Vec<&GrammarEvent>| {
        for (event_index, event) in events.into_iter().enumerate() {
            for (slot_index, source) in event.block.iter().enumerate() {
                let key = vec![fref, event_index as u32, slot_index as u32];
                let (kind, arg, variant, const_lo, const_hi) = encode(source);
                // Bit 3 carries the per-event advice flag.
                let kind = kind + WasmGrammarSlotKind::COUNT as u32 * u32::from(!event.absorb);
                preload.insert("grammar_slot_kind", key.clone(), kind);
                preload.insert("grammar_slot_arg", key.clone(), arg);
                preload.insert("grammar_slot_variant", key.clone(), variant);
                preload.insert("grammar_slot_const_lo", key.clone(), const_lo);
                preload.insert("grammar_slot_const_hi", key, const_hi);
            }
        }
    };
    // Count cells in the fref-keyed-from-free-state families store
    // count + 1 (presence bias): an undeclared fref reads the zero-filled 0
    // and the CCS load rows subtract 1, poisoning the schedule to
    // EVREM = p-1. See the relation-layout family comment for the full
    // non-termination argument. Export exit counts stay raw: their read key
    // is bound within an already-entered turn.
    for (&fref, template) in &grammar.imports {
        preload.insert(
            "grammar_import_pre_counts",
            vec![fref],
            template.events.len() as u32 + 1,
        );
        insert_slots(preload, fref, template.events.iter().collect());
    }
    for (&fref, template) in &grammar.exports {
        preload.insert(
            "grammar_export_entry_counts",
            vec![fref],
            template.entry.len() as u32 + 1,
        );
        preload.insert("grammar_export_exit_counts", vec![fref], template.exit.len() as u32);
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
    state: &mut BTreeMap<&'static str, BTreeMap<Vec<u32>, u32>>,
) -> Result<(), String> {
    let cells = state.entry(memory.name).or_default();
    for column in &memory.columns {
        let active = activation_active(column.activation, witness, row_index, memory.name)?;
        if !active {
            continue;
        }
        let address = column
            .address_columns
            .iter()
            .map(|column| read_u32_column(witness, column.0, row_index, "address"))
            .collect::<Result<Vec<_>, _>>()?;
        let value = read_u32_column(witness, column.value_column.0, row_index, "value")?;
        if memory.is_rom {
            match cells.get(&address).copied() {
                Some(expected) if expected != value => {
                    return Err(format!(
                        "memory `{}` ROM mismatch at {:?} on row {}: expected {}, got {}",
                        memory.name, address, row_index, expected, value
                    ));
                }
                Some(_) => {}
                None => {
                    return Err(format!(
                        "memory `{}` ROM read before initialization at {:?} on row {}",
                        memory.name, address, row_index
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
                        memory.name, address, row_index, expected, value
                    ));
                }
                Some(_) => {}
                None => match init_mode(memory.name) {
                    DebugInitMode::Strict => {
                        return Err(format!(
                            "memory `{}` read before initialization at {:?} on row {}",
                            memory.name, address, row_index
                        ));
                    }
                    DebugInitMode::ZeroReadDefault => {
                        if value != 0 {
                            return Err(format!(
                                "memory `{}` expected zero-default read at {:?} on row {}, got {}",
                                memory.name, address, row_index, value
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
                                memory.name, address, row_index, expected, before_value
                            ));
                        }
                        Some(_) => {}
                        None => match init_mode(memory.name) {
                            DebugInitMode::Strict => {
                                return Err(format!(
                                    "memory `{}` RMW read before initialization at {:?} on row {}",
                                    memory.name, address, row_index
                                ));
                            }
                            DebugInitMode::ZeroReadDefault => {
                                if before_value != 0 {
                                    return Err(format!(
                                        "memory `{}` expected zero-default RMW read at {:?} on row {}, got {}",
                                        memory.name, address, row_index, before_value
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
    memory_name: &str,
) -> Result<bool, String> {
    match activation {
        WasmMemoryActivation::Always => Ok(true),
        WasmMemoryActivation::BooleanGate(gate) => match witness[gate.0].as_canonical_u64() {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(format!(
                "memory `{}` has non-boolean gate {} on row {}",
                memory_name, other, row_index
            )),
        },
    }
}

fn init_mode(memory_name: &str) -> DebugInitMode {
    memory_init_mode(memory_name)
        .unwrap_or_else(|| panic!("memory semantics missing init-mode coverage for non-ROM memory `{memory_name}`"))
}

fn memory_init_mode(memory_name: &str) -> Option<DebugInitMode> {
    MEMORY_INIT_MODES
        .iter()
        .find_map(|(name, mode)| (*name == memory_name).then_some(*mode))
}

fn assert_all_memory_specs_have_init_modes(layout: &WasmRelationLayout) -> Result<(), String> {
    for memory in &layout.auxiliary.memories {
        if !memory.is_rom && memory_init_mode(memory.name).is_none() {
            return Err(format!(
                "memory semantics missing init-mode coverage for `{}`",
                memory.name
            ));
        }
    }
    Ok(())
}

const MEMORY_INIT_MODES: &[(&str, DebugInitMode)] = &[
    ("stack", DebugInitMode::Strict),
    ("call_stack_return_pcs", DebugInitMode::Strict),
    ("call_stack_caller_fbps", DebugInitMode::Strict),
    ("call_stack_caller_sp_bases", DebugInitMode::Strict),
    // linear_memory: ZeroReadDefault. Bytes initialized by active `(data ...)`
    // segments are preloaded into the cells in `preload_from_program_artifacts`
    // (via `artifacts.tables.linear_memory_init`), so the RMW Read at data-initialized
    // addresses sees the actual prior word. Bytes outside any data segment
    // stay absent from the preload, and the wasm spec guarantees them zero
    // at instantiation — so a malicious prover claiming a non-zero
    // `value_before` at an uninitialized byte fails this check.
    ("linear_memory", DebugInitMode::ZeroReadDefault),
    // Locals are either preloaded entry-frame slots, call-param writes, or
    // wasm-zero initialized slots.
    ("locals", DebugInitMode::ZeroReadDefault),
    ("locals_hi", DebugInitMode::ZeroReadDefault),
    // Declared globals are preloaded from their wasm initializer.
    ("globals", DebugInitMode::ZeroReadDefault),
    ("globals_hi", DebugInitMode::ZeroReadDefault),
    // tables: entries covered by an active element segment are preloaded from
    // `tables_init`; every other in-bounds entry is a null funcref at
    // instantiation, which normalizes to 0.
    ("tables", DebugInitMode::ZeroReadDefault),
    // table_sizes: every declared table's size is preloaded from
    // `table_sizes_init` and `table.grow` is unsupported, so an unpreloaded
    // read is always a bug.
    ("table_sizes", DebugInitMode::Strict),
];
