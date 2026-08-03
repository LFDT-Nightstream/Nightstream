//! Fast structural checks for the private WASM-to-Nebula routing compiler.

use super::{build_batched_memory_slots, build_single_step_memory_slots};
use crate::layout::SELECTOR_COLS;
use crate::nebula::WasmNebulaProfile;
use crate::{build_wasm_relation_layout, RANGE_CHECKED_WITNESS_WIDTH};
use neo_fold_clean::frontends::nebula::application::{MemoryPortActivation, MemoryPortKind};
use std::collections::BTreeSet;

#[test]
fn routing_is_deterministic_complete_and_selector_disjoint() {
    let relation = build_wasm_relation_layout();
    let first = build_single_step_memory_slots(relation);
    let second = build_single_step_memory_slots(relation);

    assert_eq!(first, second);
    assert_eq!(first.len(), 62, "current selector-only physical slot census");
    assert_eq!(
        first
            .iter()
            .map(|slot| slot.candidates().len())
            .sum::<usize>(),
        79,
        "current logical port census"
    );
    let shared = first.iter().filter(|slot| slot.candidates().len() > 1);
    assert_eq!(shared.clone().count(), 4, "one shared lane per selector ordinal");
    assert_eq!(
        shared.map(|slot| slot.candidates().len()).sum::<usize>(),
        21,
        "current opcode-selector-gated logical port census"
    );

    let mut expected = relation
        .auxiliary
        .memories
        .iter()
        .enumerate()
        .flat_map(|(region, memory)| {
            memory.columns.iter().map(move |column| {
                (
                    region,
                    column.address_columns.clone(),
                    column.value_column,
                    column.kind,
                    column.activation,
                )
            })
        })
        .collect::<Vec<_>>();

    for slot in &first {
        let mut selectors = BTreeSet::new();
        for port in slot.candidates() {
            let position = expected
                .iter()
                .position(|(region, address, value, kind, activation)| {
                    *region == port.region()
                        && address
                            .iter()
                            .map(|column| column.0)
                            .eq(port.address_columns().iter().copied())
                        && value.0 == port.value_column()
                        && memory_kinds_match(*kind, port.kind())
                        && activations_match(*activation, port.activation())
                })
                .expect("routed candidate must correspond to one logical declaration");
            expected.remove(position);

            if slot.candidates().len() > 1 {
                let MemoryPortActivation::Column(selector) = port.activation() else {
                    panic!("only opcode-selector ports may share a slot");
                };
                assert!(SELECTOR_COLS.contains(&selector));
                assert!(
                    selectors.insert(selector),
                    "one opcode cannot use a physical slot twice"
                );
            }
        }
    }
    assert!(
        expected.is_empty(),
        "every logical declaration must be routed exactly once"
    );
}

#[test]
fn batching_offsets_every_candidate_without_changing_the_route() {
    let relation = build_wasm_relation_layout();
    let single = build_single_step_memory_slots(relation);
    let batched = build_batched_memory_slots(relation, 2, RANGE_CHECKED_WITNESS_WIDTH);

    assert_eq!(batched.len(), single.len() * 2);
    for (original, offset) in single.iter().zip(&batched[single.len()..]) {
        assert_eq!(original.candidates().len(), offset.candidates().len());
        for (original, offset) in original.candidates().iter().zip(offset.candidates()) {
            assert_eq!(original.region(), offset.region());
            assert_eq!(
                offset.address_columns(),
                original
                    .address_columns()
                    .iter()
                    .map(|column| column + RANGE_CHECKED_WITNESS_WIDTH)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                offset.value_column(),
                original.value_column() + RANGE_CHECKED_WITNESS_WIDTH
            );
            assert_eq!(
                offset.kind(),
                match original.kind() {
                    MemoryPortKind::Read => MemoryPortKind::Read,
                    MemoryPortKind::Write { value_before_column } => MemoryPortKind::Write {
                        value_before_column: value_before_column.map(|column| column + RANGE_CHECKED_WITNESS_WIDTH),
                    },
                }
            );
            assert_eq!(
                offset.activation(),
                match original.activation() {
                    MemoryPortActivation::Always => MemoryPortActivation::Always,
                    MemoryPortActivation::Column(column) => {
                        MemoryPortActivation::Column(column + RANGE_CHECKED_WITNESS_WIDTH)
                    }
                    MemoryPortActivation::UnlessColumn(column) => {
                        MemoryPortActivation::UnlessColumn(column + RANGE_CHECKED_WITNESS_WIDTH)
                    }
                }
            );
        }
    }
}

#[test]
fn nebula_geometry_uses_the_physical_slot_count() {
    let relation = build_wasm_relation_layout();
    let physical_slots = build_single_step_memory_slots(relation).len();

    for profile in [WasmNebulaProfile::test_profile(), WasmNebulaProfile::production()] {
        assert_eq!(profile.memory().b_ops, physical_slots * profile.batch_size());
    }
    assert_eq!(physical_slots, 62);
}

fn memory_kinds_match(declared: crate::WasmMemoryColumnKind, routed: MemoryPortKind) -> bool {
    match (declared, routed) {
        (crate::WasmMemoryColumnKind::Read, MemoryPortKind::Read) => true,
        (
            crate::WasmMemoryColumnKind::Write { value_before_column },
            MemoryPortKind::Write {
                value_before_column: routed,
            },
        ) => value_before_column.map(|column| column.0) == routed,
        _ => false,
    }
}

fn activations_match(declared: crate::WasmMemoryActivation, routed: MemoryPortActivation) -> bool {
    match (declared, routed) {
        (crate::WasmMemoryActivation::Always, MemoryPortActivation::UnlessColumn(column)) => {
            column == crate::layout::COL_PADDING_ACTIVE
        }
        (crate::WasmMemoryActivation::BooleanGate(column), MemoryPortActivation::Column(routed)) => column.0 == routed,
        _ => false,
    }
}
