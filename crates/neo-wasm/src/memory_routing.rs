//! Packs logical memory ports using CCS-enforced activation supports.
//! The plan is backend-independent; consumers must bind every candidate's
//! address, value, memory identity and activation to the selected physical slot.

use std::collections::{BTreeMap, BTreeSet};

use neo_application::{MemoryPortActivation, MemoryPortKind, MemorySpec};

use crate::{memory_activation::activation_supports, WasmMemoryId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmMemorySlotPlan {
    /// Candidates per physical slot, indexed in flattened memory/port declaration order.
    pub slots: Vec<Vec<usize>>,
    pub logical_ports: usize,
    /// Ports without a known positive support receive dedicated slots.
    pub conservative_singletons: usize,
    /// Logical-port load per support atom. This is conservative, not observed execution.
    pub support_loads: BTreeMap<usize, usize>,
}

/// A slot plan that keeps the order a binding needs: two ports that may be
/// active on one row get distinct slots, and when they also share a memory
/// and one of them writes, the earlier port gets the earlier slot. Every other
/// pair may take either order, because their events swap. So a binding that
/// keeps each such pair in catalog order accepts every row the relation
/// accepts. Two `When` ports may be active together only when their supports
/// intersect; every other port may be active with any port. Each port takes
/// the first free slot after the slots of the earlier such pairs.
pub fn plan_ordered_wasm_memory_slots(memories: &[MemorySpec<WasmMemoryId>]) -> WasmMemorySlotPlan {
    struct Port<'a> {
        memory: usize,
        writes: bool,
        support: Option<&'a BTreeSet<usize>>,
    }
    let supports = activation_supports();
    let ports: Vec<Port> = memories
        .iter()
        .enumerate()
        .flat_map(|(memory, spec)| spec.ports.iter().map(move |port| (memory, port)))
        .map(|(memory, port)| Port {
            memory,
            writes: matches!(port.kind, MemoryPortKind::Write { .. }),
            support: match port.activation {
                MemoryPortActivation::When(gate) => supports.get(&gate),
                MemoryPortActivation::Always | MemoryPortActivation::Unless(_) => None,
            },
        })
        .collect();
    let mut support_loads = BTreeMap::new();
    for &atom in ports.iter().filter_map(|port| port.support).flatten() {
        *support_loads.entry(atom).or_insert(0) += 1;
    }
    let mut slot_of: Vec<usize> = Vec::with_capacity(ports.len());
    for (index, port) in ports.iter().enumerate() {
        let together: Vec<usize> = (0..index)
            .filter(|&earlier| match (ports[earlier].support, port.support) {
                (Some(earlier), Some(support)) => !earlier.is_disjoint(support),
                _ => true,
            })
            .collect();
        let after = together
            .iter()
            .filter(|&&earlier| ports[earlier].memory == port.memory && (ports[earlier].writes || port.writes))
            .map(|&earlier| slot_of[earlier] + 1)
            .max()
            .unwrap_or(0);
        let used: BTreeSet<usize> = together.iter().map(|&earlier| slot_of[earlier]).collect();
        slot_of.push(
            (after..)
                .find(|slot| !used.contains(slot))
                .expect("a free slot"),
        );
    }
    let mut slots = vec![Vec::new(); slot_of.iter().max().map_or(0, |last| last + 1)];
    for (index, slot) in slot_of.into_iter().enumerate() {
        slots[slot].push(index);
    }
    WasmMemorySlotPlan {
        slots,
        logical_ports: ports.len(),
        conservative_singletons: ports.iter().filter(|port| port.support.is_none()).count(),
        support_loads,
    }
}
