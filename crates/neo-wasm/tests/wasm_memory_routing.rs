use std::collections::{BTreeMap, BTreeSet};

use neo_application::{MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec};
use neo_wasm::{
    build_wasm_relation, build_wasm_relation_layout, layout::*, named_column_family, plan_ordered_wasm_memory_slots,
    plan_wasm_memory_slots, WasmMemoryId, WasmOpcode,
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
/// slots in catalog order. So two ports that the relation lets be active
/// together must sit in increasing slots. A chain of such ports needs one
/// slot per port, so a plan as long as the longest chain is the shortest.
#[test]
fn ordered_slot_plan_keeps_catalog_order_wherever_the_relation_allows() {
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
    let mut chain_to: Vec<Vec<usize>> = Vec::with_capacity(ports.len());
    for q in 0..ports.len() {
        let mut chain = Vec::new();
        for p in (0..q).filter(|p| together(*p, q)) {
            assert!(slot_of[p] < slot_of[q], "ports {p} and {q} may be active together");
            if chain_to[p].len() > chain.len() {
                chain = chain_to[p].clone();
            }
        }
        chain.push(q);
        chain_to.push(chain);
    }
    let longest = chain_to.into_iter().max_by_key(Vec::len).unwrap();
    assert_eq!(
        plan.slots.len(),
        longest.len(),
        "the plan is as long as the longest chain"
    );
    assert_eq!(plan.logical_ports, ports.len());

    eprintln!(
        "ordered slots={} (plan_wasm_memory_slots: {}), longest chain:",
        plan.slots.len(),
        plan_wasm_memory_slots(memories).slots.len()
    );
    for port in longest {
        let (id, spec) = ports[port];
        let gate = match spec.activation {
            When(gate) => named_column_family(gate).map_or("unknown", |family| family.name),
            _ => "none",
        };
        eprintln!("  port {port}: {id} {:?} when {gate}", spec.kind);
    }
}
