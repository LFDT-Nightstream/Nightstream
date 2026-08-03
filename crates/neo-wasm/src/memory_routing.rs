//! Compiles logical WASM memory declarations into verifier-owned Nebula slots.

use std::collections::BTreeSet;

use neo_fold_clean::frontends::nebula::application::{MemoryOpSlot, MemoryPort, MemoryPortActivation, MemoryPortKind};

use crate::layout::{COL_PADDING_ACTIVE, SELECTOR_COLS};
use crate::relation_layout::{WasmMemoryActivation, WasmMemoryColumnKind, WasmRelationLayout};

#[derive(Debug)]
struct ExclusiveActivationFamily {
    columns: BTreeSet<usize>,
}

impl ExclusiveActivationFamily {
    fn new(name: &'static str, columns: impl IntoIterator<Item = usize>) -> Self {
        let mut unique = BTreeSet::new();
        for column in columns {
            assert!(
                unique.insert(column),
                "exclusive activation family {name:?} repeats column {column}",
            );
        }
        assert!(
            unique.len() >= 2,
            "exclusive activation family {name:?} needs at least two columns"
        );
        Self { columns: unique }
    }

    fn contains(&self, column: usize) -> bool {
        self.columns.contains(&column)
    }

    fn contains_pair(&self, left: usize, right: usize) -> bool {
        self.contains(left) && self.contains(right)
    }
}

fn exclusive_activation_families() -> Vec<ExclusiveActivationFamily> {
    vec![ExclusiveActivationFamily::new("opcode selectors", SELECTOR_COLS)]
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
    let exclusive_families = exclusive_activation_families();
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
            if !exclusive_families
                .iter()
                .any(|family| family.contains(activation))
            {
                singleton_slots.push(MemoryOpSlot::new(vec![port]));
                continue;
            }

            let target = shared_slots.iter().position(|slot| {
                slot.iter().all(|candidate| {
                    let candidate_activation = activation_column(candidate.activation())
                        .expect("shared slots contain only column-activated ports");
                    activations_are_disjoint(activation, candidate_activation, &exclusive_families)
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

fn activations_are_disjoint(left: usize, right: usize, exclusive_families: &[ExclusiveActivationFamily]) -> bool {
    left != right
        && exclusive_families
            .iter()
            .any(|family| family.contains_pair(left, right))
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
