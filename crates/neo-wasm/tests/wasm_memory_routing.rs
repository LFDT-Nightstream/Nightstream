use neo_application::{MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec};
use neo_wasm::{build_wasm_relation_layout, layout::*, plan_wasm_memory_slots, WasmMemoryId, WasmOpcode};

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
