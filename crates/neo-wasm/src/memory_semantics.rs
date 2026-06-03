//! Witness-driven debug memory checker over `WasmMemorySpec`.

use super::adapters::wasmtime::WasmtimeTraceRun;
use super::lookup_binding_builder::{
    WasmLookupBindingLayout, WasmMemoryActivation, WasmMemoryColumnKind, WasmMemorySpec,
};
use neo_math::F;
use p3_field::PrimeField64;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WasmMemoryPreload {
    cells: BTreeMap<&'static str, BTreeMap<Vec<u64>, u64>>,
}

impl WasmMemoryPreload {
    pub fn insert(&mut self, memory: &'static str, address: Vec<u64>, value: u64) {
        self.cells.entry(memory).or_default().insert(address, value);
    }

    pub fn remove(&mut self, memory: &'static str, address: &[u64]) -> Option<u64> {
        self.cells.get_mut(memory)?.remove(address)
    }

    fn clone_cells(&self) -> BTreeMap<&'static str, BTreeMap<Vec<u64>, u64>> {
        self.cells.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DebugInitMode {
    Strict,
    ZeroReadDefault,
    FirstReadDefines,
}

pub fn preload_from_wasmtime_run(run: &WasmtimeTraceRun, initial_locals: &[u32]) -> WasmMemoryPreload {
    let mut preload = WasmMemoryPreload::default();
    // Entry-frame locals use `fbp = 0`; callee params are written by
    // CallParamInit rows before use.
    for (idx, &value) in initial_locals.iter().enumerate() {
        let address = vec![0u64, idx as u64];
        preload.insert("locals", address.clone(), u64::from(value));
        preload.insert("locals_hi", address, 0);
    }
    for &(index, lo, hi) in &run.globals_init {
        preload.insert("globals", vec![u64::from(index)], u64::from(lo));
        preload.insert("globals_hi", vec![u64::from(index)], u64::from(hi));
    }
    for &(pc_before, control_choice, pc_after) in &run.pc_rom {
        preload.insert("pc_rom", vec![pc_before, control_choice], pc_after);
    }
    for &(pc_before, edge_kind) in &run.pc_edge_kinds {
        preload.insert("pc_edge_kinds", vec![pc_before], edge_kind);
    }
    for &(pc_before, function_ref) in &run.pc_function_refs {
        preload.insert("pc_function_refs", vec![pc_before], function_ref);
    }
    for &(function_ref, entry_pc) in &run.function_entries {
        preload.insert("function_entries", vec![function_ref], entry_pc);
    }
    for &(function_ref, type_id) in &run.function_types {
        preload.insert("function_types", vec![function_ref], type_id);
    }
    for &(function_ref, param_count) in &run.function_param_counts {
        preload.insert("function_param_counts", vec![function_ref], param_count);
    }
    for &(function_ref, result_count) in &run.function_result_counts {
        preload.insert("function_result_counts", vec![function_ref], result_count);
    }
    for &(function_ref, local_count) in &run.function_local_counts {
        preload.insert("function_local_counts", vec![function_ref], local_count);
    }
    for &(function_ref, is_guest) in &run.function_guest_flags {
        preload.insert("function_guest_flags", vec![function_ref], is_guest);
    }
    for &(pc_before, function_ref) in &run.call_targets {
        preload.insert("call_targets", vec![pc_before], function_ref);
    }
    for &(raw_type_index, expected_type_id) in &run.module_types {
        preload.insert("module_types", vec![raw_type_index], expected_type_id);
    }
    // Pack data-section bytes into the word-addressed linear_memory cells.
    // Bytes outside any data segment stay absent from the preload, so
    // ZeroReadDefault catches first reads at those addresses (the wasm spec
    // guarantees them zero at instantiation, and a malicious prover claiming
    // non-zero `value_before` for an uninitialized byte will fail the check).
    if !run.linear_memory_init.is_empty() {
        let mut packed: BTreeMap<u64, u64> = BTreeMap::new();
        for &(byte_addr, byte_value) in &run.linear_memory_init {
            let word_addr = byte_addr / 4;
            let byte_index = (byte_addr % 4) as u32;
            let word = packed.entry(word_addr).or_insert(0);
            // Clear any prior occupant at this byte position (later segments
            // override earlier ones in spec order) and OR the new byte in.
            *word &= !(0xffu64 << (byte_index * 8));
            *word |= u64::from(byte_value) << (byte_index * 8);
        }
        for (word_addr, word_value) in packed {
            preload.insert("linear_memory", vec![word_addr], word_value);
        }
    }
    preload
}

pub fn sanity_check_memory_rows(
    layout: &WasmLookupBindingLayout,
    witness_rows: &[Vec<F>],
    preload: &WasmMemoryPreload,
) -> Result<(), String> {
    assert_all_memory_specs_have_init_modes(layout)?;
    sanity_check_cross_step_links(layout, witness_rows)?;
    let mut state = preload.clone_cells();
    for (row_index, witness) in witness_rows.iter().enumerate() {
        if witness.len() != layout.witness_width {
            return Err(format!(
                "memory sanity check expected witness width {}, got {} on row {}",
                layout.witness_width,
                witness.len(),
                row_index
            ));
        }
        for memory in &layout.memories {
            apply_memory_row(memory, witness, row_index, &mut state)?;
        }
    }
    Ok(())
}

fn sanity_check_cross_step_links(layout: &WasmLookupBindingLayout, witness_rows: &[Vec<F>]) -> Result<(), String> {
    for (row_index, pair) in witness_rows.windows(2).enumerate() {
        let prev = &pair[0];
        let next = &pair[1];
        for link in &layout.cross_step_links {
            for column_pair in &link.column_pairs {
                let prev_value = prev[column_pair.prev_after.0].as_canonical_u64();
                let next_value = next[column_pair.next_before.0].as_canonical_u64();
                if prev_value != next_value {
                    return Err(format!(
                        "cross-step link `{}` failed at rows {} -> {}: column {} value {} != column {} value {}",
                        link.name,
                        row_index,
                        row_index + 1,
                        column_pair.prev_after.0,
                        prev_value,
                        column_pair.next_before.0,
                        next_value
                    ));
                }
            }
        }
    }
    Ok(())
}

fn apply_memory_row(
    memory: &WasmMemorySpec,
    witness: &[F],
    row_index: usize,
    state: &mut BTreeMap<&'static str, BTreeMap<Vec<u64>, u64>>,
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
            .map(|column| witness[column.0].as_canonical_u64())
            .collect::<Vec<_>>();
        let value = witness[column.value_column.0].as_canonical_u64();
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
                    DebugInitMode::FirstReadDefines => {
                        cells.insert(address, value);
                    }
                },
            },
            WasmMemoryColumnKind::Write => {
                // Nebula-style RMW: if `value_before_column` is named, this
                // row's read tuple must match the prior write at this address
                // (or the documented init mode). Catches a malicious prover
                // who writes a word whose unmodified bytes don't preserve the
                // prior state — see `i32_store8_row_rejects_tampered_...`.
                if let Some(before_col) = column.value_before_column {
                    let before_value = witness[before_col.0].as_canonical_u64();
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
                            DebugInitMode::FirstReadDefines => {
                                cells.insert(address.clone(), before_value);
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
        WasmMemoryActivation::ColumnEquals { column, value } => Ok(witness[column.0].as_canonical_u64() == value),
    }
}

fn init_mode(memory_name: &str) -> DebugInitMode {
    memory_init_mode(memory_name).unwrap_or(DebugInitMode::FirstReadDefines)
}

fn memory_init_mode(memory_name: &str) -> Option<DebugInitMode> {
    MEMORY_INIT_MODES
        .iter()
        .find_map(|(name, mode)| (*name == memory_name).then_some(*mode))
}

fn assert_all_memory_specs_have_init_modes(layout: &WasmLookupBindingLayout) -> Result<(), String> {
    for memory in &layout.memories {
        if memory_init_mode(memory.name).is_none() {
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
    // linear_memory: ZeroReadDefault. Bytes initialized by active `(data ...)`
    // segments are preloaded into the cells in `preload_from_wasmtime_run`
    // (via `run.linear_memory_init`), so the RMW Read at data-initialized
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
    ("tables", DebugInitMode::FirstReadDefines),
    ("table_sizes", DebugInitMode::FirstReadDefines),
    ("function_types", DebugInitMode::FirstReadDefines),
    ("function_param_counts", DebugInitMode::FirstReadDefines),
    ("function_result_counts", DebugInitMode::FirstReadDefines),
    ("function_local_counts", DebugInitMode::FirstReadDefines),
    ("function_guest_flags", DebugInitMode::FirstReadDefines),
    ("call_targets", DebugInitMode::FirstReadDefines),
    ("module_types", DebugInitMode::FirstReadDefines),
    ("function_entries", DebugInitMode::FirstReadDefines),
    ("pc_edge_kinds", DebugInitMode::FirstReadDefines),
    ("pc_function_refs", DebugInitMode::FirstReadDefines),
    ("pc_rom", DebugInitMode::FirstReadDefines),
];
