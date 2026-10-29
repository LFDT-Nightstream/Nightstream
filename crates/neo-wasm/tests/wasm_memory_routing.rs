use std::collections::{BTreeMap, BTreeSet};

use neo_application::{MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec};
use neo_wasm::{
    build_wasm_relation, build_wasm_relation_layout, layout::*, plan_ordered_wasm_memory_slots, plan_wasm_memory_slots,
    WasmMemoryId, WasmOpcode,
};

#[test]
fn slot_plan_is_deterministic_and_covers_every_logical_port_once() {
    let layout = build_wasm_relation_layout();
    let memories = layout.auxiliary.memory.entries();
    let plan = plan_wasm_memory_slots(memories);
    assert_eq!(plan, plan_wasm_memory_slots(memories));
    let mut ports: Vec<_> = plan.slots.iter().flatten().copied().collect();
    ports.sort_unstable();
    assert_eq!(ports, (0..plan.logical_ports).collect::<Vec<_>>());
    assert!(plan.slots.iter().all(|slot| !slot.is_empty()));
    let max_load = plan.support_loads.values().copied().max().unwrap();
    assert_eq!((plan.logical_ports, plan.slots.len(), max_load), (98, 21, 21));
    assert!(plan.slots.len() >= max_load + plan.conservative_singletons);
    assert!(plan.slots.len() < plan.logical_ports);
    assert_eq!(
        plan.conservative_singletons, 0,
        "every current port needs a support bound"
    );
}

#[test]
fn sharing_requires_disjoint_supports_and_unknown_gates_stay_dedicated() {
    use MemoryPortActivation::{Always, Unless, When};
    let get = selector_col(WasmOpcode::GlobalGet).unwrap();
    let set = selector_col(WasmOpcode::LocalSet).unwrap();
    let activations = [
        When(get),
        When(set),
        When(COL_LOCAL_WRITE_ENABLED), // overlaps LocalSet, but not GlobalGet
        When(get),                     // same selector cannot share
        Always,
        Unless(get),
        When(COL_ACTIVATION_FINISHED_AFTER), // no positive support bound
    ];
    let memories = [MemorySpec {
        id: WasmMemoryId::Stack,
        kind: MemoryKind::Ram,
        ports: activations
            .into_iter()
            .map(|activation| MemoryPortSpec {
                address_columns: vec![COL_ONE],
                value_column: COL_ONE,
                kind: MemoryPortKind::Read,
                activation,
            })
            .collect(),
    }];
    let plan = plan_wasm_memory_slots(&memories);
    assert_eq!(plan.slots, vec![vec![4], vec![5], vec![6], vec![0, 1], vec![2, 3]]);
    assert_eq!(plan.logical_ports, 7);
    assert_eq!(plan.conservative_singletons, 3);
    assert_eq!(plan.support_loads[&get], 2);
    assert_eq!(plan.support_loads[&set], 2);
}

/// The activation supports that the WASM relation enforces: each row tagged
/// "memory activation support" is `gate · (1 − Σ atoms) = 0`, and each atom
/// is its own support. The row-kind and opcode one-hots let at most one atom
/// be 1 on a row, so two gates are 1 together only when their supports meet.
fn enforced_supports() -> BTreeMap<usize, BTreeSet<usize>> {
    let relation = build_wasm_relation().expect("the WASM relation");
    let mut supports: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for tagged in relation.r1cs().catalog().rows() {
        if tagged.tag().label() != "memory activation support" {
            continue;
        }
        let row = tagged.row();
        let [(gate, _)] = row.a_terms() else {
            panic!("a support row gates on one column");
        };
        let atoms = row
            .b_terms()
            .iter()
            .map(|(column, _)| *column)
            .filter(|column| *column != COL_ONE)
            .collect();
        supports.insert(*gate, atoms);
    }
    let atoms: BTreeSet<usize> = supports.values().flatten().copied().collect();
    supports.extend(atoms.into_iter().map(|atom| (atom, BTreeSet::from([atom]))));
    supports
}

/// Spec §10.1 I1 and I2 accept a row whose active ports sit in distinct
/// slots, with each pair of one memory and a write in catalog order. So two
/// ports that the relation lets be active together must sit in distinct
/// slots, and in increasing slots when they are such a pair. The ports whose
/// supports share an atom may all be active together, so a plan from these
/// supports needs at least the largest support load of slots.
#[test]
fn ordered_slot_plan_keeps_dependent_pairs_in_catalog_order_wherever_the_relation_allows() {
    use MemoryPortActivation::When;
    let layout = build_wasm_relation_layout();
    let memories = layout.auxiliary.memory.entries();
    let plan = plan_ordered_wasm_memory_slots(memories);
    let supports = enforced_supports();
    let ports: Vec<(WasmMemoryId, &MemoryPortSpec)> = memories
        .iter()
        .flat_map(|memory| memory.ports.iter().map(|port| (memory.id, port)))
        .collect();
    let support = |port: usize| match ports[port].1.activation {
        When(gate) => supports.get(&gate),
        _ => None,
    };
    let together = |p: usize, q: usize| match (support(p), support(q)) {
        (Some(a), Some(b)) => !a.is_disjoint(b),
        _ => true,
    };
    let writes = |port: usize| matches!(ports[port].1.kind, MemoryPortKind::Write { .. });
    let dependent = |p: usize, q: usize| ports[p].0 == ports[q].0 && (writes(p) || writes(q));

    let mut slot_of = vec![None; ports.len()];
    for (slot, candidates) in plan.slots.iter().enumerate() {
        for &port in candidates {
            assert_eq!(slot_of[port].replace(slot), None, "port {port} has one slot");
        }
    }
    let slot_of: Vec<usize> = slot_of
        .into_iter()
        .map(|slot| slot.expect("every port has a slot"))
        .collect();
    let mut reordered = 0;
    for q in 0..ports.len() {
        for p in (0..q).filter(|p| together(*p, q)) {
            assert_ne!(slot_of[p], slot_of[q], "ports {p} and {q} may be active together");
            if dependent(p, q) {
                assert!(slot_of[p] < slot_of[q], "ports {p} and {q} keep catalog order");
            } else if slot_of[p] > slot_of[q] {
                reordered += 1;
            }
        }
    }
    let mut loads: BTreeMap<usize, usize> = BTreeMap::new();
    for atom in (0..ports.len()).filter_map(support).flatten() {
        *loads.entry(*atom).or_insert(0) += 1;
    }
    let largest = loads.values().copied().max().unwrap();
    assert_eq!(
        plan.slots.len(),
        largest,
        "the plan has the fewest slots its supports allow"
    );
    assert_eq!(plan.logical_ports, ports.len());

    eprintln!(
        "ordered slots={} (plan_wasm_memory_slots: {}), {reordered} pairs that may be active together swap",
        plan.slots.len(),
        plan_wasm_memory_slots(memories).slots.len()
    );
}
