//! Compiles logical WASM memory declarations into verifier-owned Nebula slots.

use std::collections::BTreeMap;

use neo_fold_clean::frontends::nebula::application::{MemoryOpSlot, MemoryPort, MemoryPortActivation, MemoryPortKind};

use crate::layout::{COL_PADDING_ACTIVE, SELECTOR_COLS};
use crate::relation_layout::{WasmMemoryActivation, WasmMemoryColumnKind, WasmRelationLayout};

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
    let mut singleton_slots = Vec::new();
    let mut selector_slots: Vec<Vec<MemoryPort>> = Vec::new();

    // used to assign slot "ids" (index)
    let mut selector_port_count = BTreeMap::new();

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

            let Some(selector) = opcode_selector(port.activation()) else {
                singleton_slots.push(MemoryOpSlot::new(vec![port]));
                continue;
            };

            // Different opcode selectors are circuit-proven disjoint. The nth
            // port used by each opcode therefore shares selector slot n, while
            // repeated ports for one opcode necessarily occupy separate slots.
            let next_slot_index_for_selector = selector_port_count.entry(selector).or_insert(0);
            if selector_slots.len() == *next_slot_index_for_selector {
                selector_slots.push(Vec::new());
            }
            selector_slots[*next_slot_index_for_selector].push(port);
            *next_slot_index_for_selector += 1;
        }
    }

    singleton_slots.extend(selector_slots.into_iter().map(MemoryOpSlot::new));
    singleton_slots
}

fn opcode_selector(activation: MemoryPortActivation) -> Option<usize> {
    match activation {
        MemoryPortActivation::Column(column) if SELECTOR_COLS.contains(&column) => Some(column),
        _ => None,
    }
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
