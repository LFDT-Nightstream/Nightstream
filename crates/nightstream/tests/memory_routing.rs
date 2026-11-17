//! Exercise the slot planner through the public API.

use std::collections::{BTreeMap, BTreeSet};

use nightstream::relation::{
    plan_ordered_memory_slots, ColumnFamilySpec, ColumnRegistry, ColumnWidth, MemoryActivationFacts, MemoryCatalog,
    MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec,
};
use MemoryPortActivation::{Always, Unless, When};

#[derive(PartialEq, Eq)]
enum Id {
    Stack,
    Heap,
}

fn catalog(groups: Vec<Vec<(MemoryPortActivation, bool)>>) -> MemoryCatalog<Id> {
    let columns = ColumnRegistry::new([ColumnFamilySpec {
        region: "test",
        start: 0,
        len: 8,
        name: "bits".into(),
        role: "test",
        width: ColumnWidth::Boolean,
    }])
    .unwrap();
    MemoryCatalog::new(
        groups
            .into_iter()
            .zip([Id::Stack, Id::Heap])
            .map(|(ports, id)| MemorySpec {
                id,
                kind: MemoryKind::Ram,
                ports: ports
                    .into_iter()
                    .map(|(activation, writes)| MemoryPortSpec {
                        address_columns: vec![6],
                        value_column: 7,
                        kind: if writes {
                            MemoryPortKind::Write {
                                value_before_column: Some(7),
                            }
                        } else {
                            MemoryPortKind::Read
                        },
                        activation,
                    })
                    .collect(),
            }),
        &columns,
    )
    .unwrap()
}

#[test]
fn generic_plan_preserves_all_allowed_activations_and_dependent_order() {
    let catalog = catalog(vec![
        vec![(When(0), false), (When(2), true), (When(1), false), (When(3), true)],
        vec![(When(1), true), (When(4), false), (Always, false), (Unless(0), true)],
    ]);
    let facts = MemoryActivationFacts {
        supports: BTreeMap::from([
            (0, BTreeSet::from([0])),
            (1, BTreeSet::from([1])),
            (2, BTreeSet::from([0, 1])),
            (3, BTreeSet::from([0, 1])),
        ]),
        gate_exclusions: vec![(2, 3)],
    };
    let plan = plan_ordered_memory_slots(&catalog, &facts);
    let mut covered: Vec<_> = plan.slots.iter().flatten().copied().collect();
    covered.sort_unstable();
    assert_eq!(covered, (0..8).collect::<Vec<_>>());
    assert!(plan.slots.len() < 8);
    let mut slot_of = vec![0; 8];
    for (slot, ports) in plan.slots.iter().enumerate() {
        assert!(!ports.is_empty());
        for &port in ports {
            slot_of[port] = slot;
        }
    }
    let ports: Vec<_> = catalog
        .entries()
        .iter()
        .enumerate()
        .flat_map(|(memory, spec)| spec.ports.iter().map(move |port| (memory, port)))
        .collect();
    for assignment in 0..32 {
        let bit = |c| assignment & (1 << c) != 0;
        if bit(0) && bit(1)
            || facts
                .supports
                .iter()
                .any(|(&gate, atoms)| bit(gate) && !atoms.iter().any(|&atom| bit(atom)))
            || facts.gate_exclusions.iter().any(|&(a, b)| bit(a) && bit(b))
        {
            continue;
        }
        let active = |p: usize| match ports[p].1.activation {
            Always => true,
            When(c) => bit(c),
            Unless(c) => !bit(c),
        };
        for b in 0..ports.len() {
            for a in 0..b {
                if !active(a) || !active(b) {
                    continue;
                }
                assert_ne!(slot_of[a], slot_of[b]);
                if ports[a].0 == ports[b].0
                    && [a, b]
                        .iter()
                        .any(|&p| matches!(ports[p].1.kind, MemoryPortKind::Write { .. }))
                {
                    assert!(slot_of[a] < slot_of[b]);
                }
            }
        }
    }
}

#[test]
fn unknown_activations_are_conservative_and_exclusions_work_without_supports() {
    let catalog = catalog(vec![vec![
        (When(0), false),
        (When(1), false),
        (Always, false),
        (Unless(0), false),
    ]]);
    let mut facts = MemoryActivationFacts::default();
    let plan = plan_ordered_memory_slots(&catalog, &facts);
    assert_eq!(plan.slots.len(), 4);
    assert_eq!(plan.ports_without_support, 4);
    facts.gate_exclusions.push((1, 0));
    let plan = plan_ordered_memory_slots(&catalog, &facts);
    assert_eq!(plan.slots, vec![vec![0, 1], vec![2], vec![3]]);
}

/// The conflicts are the path 0 - 2 - 3 - 1. First fit gives ports 0 and 1
/// slot 0, port 2 slot 1 and port 3 slot 2. Two slots are sufficient.
#[test]
fn bounded_search_reduces_the_first_fit_plan() {
    let catalog = catalog(vec![(0..4).map(|gate| (When(gate), false)).collect()]);
    let facts = MemoryActivationFacts {
        supports: BTreeMap::new(),
        gate_exclusions: vec![(0, 1), (0, 3), (1, 2)],
    };
    let plan = plan_ordered_memory_slots(&catalog, &facts);
    assert_eq!(plan.slots, vec![vec![0, 3], vec![1, 2]]);
}

#[test]
fn empty_catalog_and_large_conservative_plan_are_supported() {
    let facts = MemoryActivationFacts::default();
    assert!(plan_ordered_memory_slots(&catalog(vec![]), &facts)
        .slots
        .is_empty());
    let catalog = catalog(vec![vec![(Always, true); 65]]);
    let plan = plan_ordered_memory_slots(&catalog, &facts);
    assert_eq!(plan.slots, (0..65).map(|p| vec![p]).collect::<Vec<_>>());
}
