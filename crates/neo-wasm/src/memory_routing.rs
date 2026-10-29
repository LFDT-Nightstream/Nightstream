//! Packs logical memory ports using CCS-enforced activation supports.
//! The plan is backend-independent; consumers must bind every candidate's
//! address, value, memory identity and activation to the selected physical slot.

use std::collections::{BTreeMap, BTreeSet};

use neo_application::{MemoryPortActivation, MemorySpec};

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

/// Deterministic first-fit coloring of the support-intersection graph.
/// Ports share a slot only when their supports are disjoint; this does not
/// deduplicate accesses or promise a minimum coloring. Uses the support bounds
/// enforced by the WASM relation, never evidence sampled from traces.
pub fn plan_wasm_memory_slots(memories: &[MemorySpec<WasmMemoryId>]) -> WasmMemorySlotPlan {
    let supports = activation_supports();
    let mut singletons = Vec::new();
    let mut shared: Vec<Vec<usize>> = Vec::new();
    let mut slot_supports: Vec<BTreeSet<usize>> = Vec::new();
    let mut support_loads = BTreeMap::new();
    let mut logical_ports = 0;
    for (index, port) in memories.iter().flat_map(|memory| &memory.ports).enumerate() {
        logical_ports += 1;
        let support = match port.activation {
            MemoryPortActivation::When(gate) => supports.get(&gate),
            MemoryPortActivation::Always | MemoryPortActivation::Unless(_) => None,
        };
        let Some(support) = support else {
            singletons.push(vec![index]);
            continue;
        };
        for &atom in support {
            *support_loads.entry(atom).or_insert(0) += 1;
        }
        match slot_supports
            .iter()
            .position(|used| used.is_disjoint(support))
        {
            Some(slot) => {
                shared[slot].push(index);
                slot_supports[slot].extend(support);
            }
            None => {
                shared.push(vec![index]);
                slot_supports.push(support.clone());
            }
        }
    }
    let conservative_singletons = singletons.len();
    singletons.extend(shared);
    WasmMemorySlotPlan {
        slots: singletons,
        logical_ports,
        conservative_singletons,
        support_loads,
    }
}

/// A slot plan that keeps declaration order: when two ports may be active on
/// one row, the earlier port gets the earlier slot. So a binding that needs a
/// row's active slots in catalog order accepts every row the relation accepts.
/// Two `When` ports may be active together only when their supports intersect;
/// every other port may be active with any port. Each port goes one slot after
/// the last slot of an earlier port that may be active with it, so the slot
/// count is the longest chain of such ports, the fewest such a plan can have.
pub fn plan_ordered_wasm_memory_slots(memories: &[MemorySpec<WasmMemoryId>]) -> WasmMemorySlotPlan {
    let supports = activation_supports();
    let ports: Vec<Option<&BTreeSet<usize>>> = memories
        .iter()
        .flat_map(|memory| &memory.ports)
        .map(|port| match port.activation {
            MemoryPortActivation::When(gate) => supports.get(&gate),
            MemoryPortActivation::Always | MemoryPortActivation::Unless(_) => None,
        })
        .collect();
    let mut support_loads = BTreeMap::new();
    for &atom in ports.iter().flatten().copied().flatten() {
        *support_loads.entry(atom).or_insert(0) += 1;
    }
    let mut slot_of: Vec<usize> = Vec::with_capacity(ports.len());
    for (index, support) in ports.iter().enumerate() {
        let slot = ports[..index]
            .iter()
            .zip(&slot_of)
            .filter(|(earlier, _)| match (earlier, support) {
                (Some(earlier), Some(support)) => !earlier.is_disjoint(support),
                _ => true,
            })
            .map(|(_, slot)| slot + 1)
            .max()
            .unwrap_or(0);
        slot_of.push(slot);
    }
    let mut slots = vec![Vec::new(); slot_of.iter().max().map_or(0, |last| last + 1)];
    for (index, slot) in slot_of.into_iter().enumerate() {
        slots[slot].push(index);
    }
    WasmMemorySlotPlan {
        slots,
        logical_ports: ports.len(),
        conservative_singletons: ports.iter().filter(|support| support.is_none()).count(),
        support_loads,
    }
}
