//! Compiles logical WASM memory declarations into verifier-owned Nebula slots.

use std::collections::{BTreeMap, BTreeSet};

use neo_fold_clean::frontends::nebula::application::{MemoryOpSlot, MemoryPort, MemoryPortActivation, MemoryPortKind};

use crate::isa::WasmOpcode;
use crate::layout::{
    selector_col, COL_LOCAL_WRITE_ENABLED, COL_PADDING_ACTIVE, COL_TABLE_READ_ENABLED, COL_TABLE_SIZE_READ_ENABLED,
    SELECTOR_COLS,
};
use crate::relation_layout::{WasmMemoryActivation, WasmMemoryColumnKind, WasmRelationLayout};

fn program_activation_supports() -> BTreeMap<usize, BTreeSet<usize>> {
    let mut supports = SELECTOR_COLS
        .into_iter()
        .map(|selector| (selector, BTreeSet::from([selector])))
        .collect();
    insert_derived_activation_support(
        &mut supports,
        "local writes",
        COL_LOCAL_WRITE_ENABLED,
        [
            selector_col(WasmOpcode::LocalSet).unwrap(),
            selector_col(WasmOpcode::LocalTee).unwrap(),
        ],
    );
    insert_derived_activation_support(
        &mut supports,
        "table reads",
        COL_TABLE_READ_ENABLED,
        [
            selector_col(WasmOpcode::TableGet).unwrap(),
            selector_col(WasmOpcode::CallIndirect).unwrap(),
            selector_col(WasmOpcode::ReturnCallIndirect).unwrap(),
        ],
    );
    insert_derived_activation_support(
        &mut supports,
        "table-size reads",
        COL_TABLE_SIZE_READ_ENABLED,
        [
            selector_col(WasmOpcode::TableSize).unwrap(),
            selector_col(WasmOpcode::CallIndirect).unwrap(),
            selector_col(WasmOpcode::ReturnCallIndirect).unwrap(),
        ],
    );
    supports
}

/// Record the verifier-owned implication `gate = 1 => one of selectors = 1`.
/// These support bounds come from the WASM CCS, not sampled traces.
fn insert_derived_activation_support(
    supports: &mut BTreeMap<usize, BTreeSet<usize>>,
    name: &'static str,
    gate: usize,
    possible_selectors: impl IntoIterator<Item = usize>,
) {
    assert!(
        !SELECTOR_COLS.contains(&gate),
        "program-gate support {name:?} uses opcode selector {gate} as a derived gate",
    );
    let mut unique = BTreeSet::new();
    for selector in possible_selectors {
        assert!(
            SELECTOR_COLS.contains(&selector),
            "program-gate support {name:?} contains non-selector column {selector}",
        );
        assert!(
            unique.insert(selector),
            "program-gate support {name:?} repeats selector {selector}",
        );
    }
    assert!(
        !unique.is_empty(),
        "program-gate support {name:?} needs at least one possible selector",
    );
    assert!(
        supports.insert(gate, unique).is_none(),
        "program-gate support {name:?} repeats gate column {gate}",
    )
}

pub(crate) fn build_batched_memory_slots(
    relation: &WasmRelationLayout,
    batch_size: usize,
    single_step_columns: usize,
) -> Vec<MemoryOpSlot> {
    let single_step = build_single_step_memory_slots(relation);
    let mut batched = Vec::with_capacity(single_step.len() * batch_size);
    for block in 0..batch_size {
        let offset = block * single_step_columns;
        batched.extend(single_step.iter().map(|slot| offset_slot(slot, offset)));
    }
    batched
}

pub(crate) fn build_single_step_memory_slots(relation: &WasmRelationLayout) -> Vec<MemoryOpSlot> {
    let activation_supports = program_activation_supports();
    let mut singleton_slots = Vec::new();
    let mut shared_slots: Vec<Vec<MemoryPort>> = Vec::new();

    for (region, memory) in relation.auxiliary.memories.iter().enumerate() {
        for column in &memory.columns {
            let port = MemoryPort::new(
                region,
                column
                    .address_columns
                    .iter()
                    .map(|column| column.0)
                    .collect(),
                column.value_column.0,
                match column.kind {
                    WasmMemoryColumnKind::Read => MemoryPortKind::Read,
                    WasmMemoryColumnKind::Write { value_before_column } => MemoryPortKind::Write {
                        value_before_column: value_before_column.map(|column| column.0),
                    },
                },
                match column.activation {
                    WasmMemoryActivation::Always => MemoryPortActivation::UnlessColumn(COL_PADDING_ACTIVE),
                    WasmMemoryActivation::BooleanGate(column) => MemoryPortActivation::Column(column.0),
                },
            );

            let Some(activation) = activation_column(port.activation()) else {
                singleton_slots.push(MemoryOpSlot::new(vec![port]));
                continue;
            };
            if !activation_supports.contains_key(&activation) {
                singleton_slots.push(MemoryOpSlot::new(vec![port]));
                continue;
            }

            let target = shared_slots.iter().position(|slot| {
                slot.iter().all(|candidate| {
                    let candidate_activation = activation_column(candidate.activation())
                        .expect("shared slots contain only column-activated ports");
                    activations_are_disjoint(activation, candidate_activation, &activation_supports)
                })
            });
            match target {
                Some(index) => shared_slots[index].push(port),
                None => shared_slots.push(vec![port]),
            }
        }
    }

    singleton_slots.extend(shared_slots.into_iter().map(MemoryOpSlot::new));
    singleton_slots
}

fn activation_column(activation: MemoryPortActivation) -> Option<usize> {
    match activation {
        MemoryPortActivation::Column(column) => Some(column),
        _ => None,
    }
}

fn activations_are_disjoint(left: usize, right: usize, activation_supports: &BTreeMap<usize, BTreeSet<usize>>) -> bool {
    let (Some(left), Some(right)) = (activation_supports.get(&left), activation_supports.get(&right)) else {
        return false;
    };
    left.is_disjoint(right)
}

fn offset_slot(slot: &MemoryOpSlot, offset: usize) -> MemoryOpSlot {
    MemoryOpSlot::new(
        slot.candidates()
            .iter()
            .map(|port| {
                MemoryPort::new(
                    port.region(),
                    port.address_columns()
                        .iter()
                        .map(|column| column + offset)
                        .collect(),
                    port.value_column() + offset,
                    match port.kind() {
                        MemoryPortKind::Read => MemoryPortKind::Read,
                        MemoryPortKind::Write { value_before_column } => MemoryPortKind::Write {
                            value_before_column: value_before_column.map(|column| column + offset),
                        },
                    },
                    match port.activation() {
                        MemoryPortActivation::Always => MemoryPortActivation::Always,
                        MemoryPortActivation::Column(column) => MemoryPortActivation::Column(column + offset),
                        MemoryPortActivation::UnlessColumn(column) => {
                            MemoryPortActivation::UnlessColumn(column + offset)
                        }
                    },
                )
            })
            .collect(),
    )
}

#[cfg(test)]
#[path = "../tests/nebula/memory_routing.rs"]
mod tests;
