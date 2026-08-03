//! Fast structural checks for the private WASM-to-Nebula routing compiler.

use super::{
    activation_column, activations_are_disjoint, build_batched_memory_slots, build_single_step_memory_slots,
    program_activation_supports,
};
use crate::layout::{
    selector_col, COL_LOCAL_WRITE_ENABLED, COL_STACK_READ0_ACTIVE, COL_TABLE_READ_ENABLED, COL_TABLE_SIZE_READ_ENABLED,
};
use crate::nebula::WasmNebulaProfile;
use crate::{build_wasm_relation_layout, WasmOpcode, RANGE_CHECKED_WITNESS_WIDTH};
use neo_fold_clean::frontends::nebula::application::{MemoryPortActivation, MemoryPortKind};
use neo_fold_clean::frontends::nebula::circuit::SMemCircuit;
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use std::collections::BTreeSet;

#[test]
fn program_support_derives_only_known_disjointness() {
    let supports = program_activation_supports();

    assert!(activations_are_disjoint(
        selector_col(WasmOpcode::GlobalGet).unwrap(),
        selector_col(WasmOpcode::GlobalSet).unwrap(),
        &supports,
    ));
    assert!(!activations_are_disjoint(
        selector_col(WasmOpcode::GlobalGet).unwrap(),
        selector_col(WasmOpcode::GlobalGet).unwrap(),
        &supports,
    ));
    assert!(activations_are_disjoint(
        COL_LOCAL_WRITE_ENABLED,
        COL_TABLE_READ_ENABLED,
        &supports,
    ));
    assert!(!activations_are_disjoint(
        COL_TABLE_READ_ENABLED,
        COL_TABLE_SIZE_READ_ENABLED,
        &supports,
    ));
    assert!(activations_are_disjoint(
        COL_LOCAL_WRITE_ENABLED,
        selector_col(WasmOpcode::GlobalGet).unwrap(),
        &supports,
    ));
    assert!(!activations_are_disjoint(
        COL_LOCAL_WRITE_ENABLED,
        selector_col(WasmOpcode::LocalSet).unwrap(),
        &supports,
    ));
    assert!(!activations_are_disjoint(
        COL_STACK_READ0_ACTIVE,
        selector_col(WasmOpcode::GlobalGet).unwrap(),
        &supports,
    ));
}

#[test]
fn routing_is_deterministic_complete_and_pairwise_disjoint() {
    let relation = build_wasm_relation_layout();
    let first = build_single_step_memory_slots(relation);
    let second = build_single_step_memory_slots(relation);
    let activation_supports = program_activation_supports();

    assert_eq!(first, second);
    assert_eq!(first.len(), 58, "current physical slot census");
    assert_eq!(
        first
            .iter()
            .map(|slot| slot.candidates().len())
            .sum::<usize>(),
        79,
        "current logical port census"
    );
    let shared = first.iter().filter(|slot| slot.candidates().len() > 1);
    assert_eq!(shared.clone().count(), 4, "current shared-slot census");
    assert_eq!(
        shared.map(|slot| slot.candidates().len()).sum::<usize>(),
        25,
        "current shared logical-port census"
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
        let mut activations = BTreeSet::new();
        for (candidate_index, port) in slot.candidates().iter().enumerate() {
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
                let MemoryPortActivation::Column(activation) = port.activation() else {
                    panic!("only column-activated ports may share a slot");
                };
                assert!(
                    activations.insert(activation),
                    "one activation cannot use a physical slot twice"
                );
                for other in &slot.candidates()[..candidate_index] {
                    let other_activation =
                        activation_column(other.activation()).expect("shared slot candidates use activation columns");
                    assert!(activations_are_disjoint(
                        activation,
                        other_activation,
                        &activation_supports,
                    ));
                }
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
    assert_eq!(physical_slots, 58);
}

#[test]
fn s_mem_structure_census() {
    let profile = WasmNebulaProfile::production();
    let circuit = SMemCircuit::new(*profile.memory());
    let previous_params = NebulaParams::new(
        profile.memory().r,
        profile.memory().mu,
        79 * profile.batch_size(),
        profile.memory().b_scan,
        profile.memory().seg_max,
    )
    .expect("previous logical-port geometry");
    let previous = SMemCircuit::new(previous_params);
    let public_bits = circuit.m_in() - 1;
    let private_bits = circuit.cols() - circuit.m_in();

    println!("WASM Nebula S_mem structural census");
    println!(
        "  physical slots    {} -> {}",
        previous_params.b_ops,
        profile.memory().b_ops
    );
    println!(
        "  constraints       {} -> {} (-{})",
        previous.rows(),
        circuit.rows(),
        previous.rows() - circuit.rows()
    );
    println!(
        "  assignment bits   {} -> {} (-{})",
        previous.cols() - 1,
        circuit.cols() - 1,
        previous.cols() - circuit.cols()
    );
    println!("    public bits     {public_bits}");
    println!("    private bits    {private_bits}");
    println!(
        "  nonzero entries   {} -> {} (-{})",
        previous.nnz(),
        circuit.nnz(),
        previous.nnz() - circuit.nnz()
    );

    assert_eq!(public_bits + private_bits, circuit.cols() - 1);
    assert_eq!(profile.memory().b_ops, 58 * profile.batch_size());
    assert_eq!(
        (
            circuit.rows(),
            circuit.cols() - 1,
            public_bits,
            private_bits,
            circuit.nnz(),
        ),
        (105_344, 103_347, 1_400, 101_947, 688_295),
        "production S_mem structure changed; review the constraint and committed-bit census",
    );
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
