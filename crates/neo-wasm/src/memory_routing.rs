//! Packs logical memory ports using CCS-enforced activation supports.
//! The plan is backend-independent; consumers must bind every candidate's
//! address, value, memory identity and activation to the selected physical slot.

use std::collections::{BTreeMap, BTreeSet};

use neo_application::{MemoryPortActivation, MemoryPortKind, MemorySpec};

use crate::{
    memory_activation::{activation_supports, gate_exclusions},
    WasmMemoryId,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmMemorySlotPlan {
    /// Candidates per physical slot, indexed in flattened memory/port declaration order.
    pub slots: Vec<Vec<usize>>,
    pub logical_ports: usize,
    /// Additional disjoint gates, exported and proved in Rocq.
    pub gate_exclusions: Vec<(usize, usize)>,
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
/// intersect and no exported gate exclusion separates them. Every other port
/// may be active with any port. Each port takes
/// the first free slot after the slots of the earlier such pairs. A bounded
/// search then tries fewer slots with the same conflict and order conditions.
/// The result is an upper bound, not a claim that the slot count is minimal.
pub fn plan_ordered_wasm_memory_slots(memories: &[MemorySpec<WasmMemoryId>]) -> WasmMemorySlotPlan {
    struct Port<'a> {
        memory: usize,
        gate: Option<usize>,
        writes: bool,
        support: Option<&'a BTreeSet<usize>>,
    }
    let supports = activation_supports();
    let gate_exclusions = gate_exclusions();
    let ports: Vec<Port> = memories
        .iter()
        .enumerate()
        .flat_map(|(memory, spec)| spec.ports.iter().map(move |port| (memory, port)))
        .map(|(memory, port)| Port {
            memory,
            gate: match port.activation {
                MemoryPortActivation::When(gate) => Some(gate),
                _ => None,
            },
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
    let mut conflicts = Vec::new();
    let mut slot_of: Vec<usize> = Vec::with_capacity(ports.len());
    for (index, port) in ports.iter().enumerate() {
        let together = (0..index)
            .filter(|&earlier| {
                !gate_exclusions.iter().any(|&(a, b)| {
                    (ports[earlier].gate == Some(a) && port.gate == Some(b))
                        || (ports[earlier].gate == Some(b) && port.gate == Some(a))
                })
            })
            .filter(|&earlier| match (ports[earlier].support, port.support) {
                (Some(earlier), Some(support)) => !earlier.is_disjoint(support),
                _ => true,
            });

        let mut after = 0;
        let mut used = BTreeSet::new();
        for earlier in together {
            let ordered = ports[earlier].memory == port.memory && (ports[earlier].writes || port.writes);
            conflicts.push((earlier, index, ordered));
            if ordered {
                after = after.max(slot_of[earlier] + 1);
            }
            used.insert(slot_of[earlier]);
        }

        slot_of.push(
            (after..)
                .find(|slot| !used.contains(slot))
                .expect("a free slot"),
        );
    }
    compact_slots(&mut slot_of, &conflicts);
    let mut slots = vec![Vec::new(); slot_of.iter().max().map_or(0, |last| last + 1)];
    for (index, slot) in slot_of.into_iter().enumerate() {
        slots[slot].push(index);
    }
    WasmMemorySlotPlan {
        slots,
        logical_ports: ports.len(),
        gate_exclusions,
        conservative_singletons: ports.iter().filter(|port| port.support.is_none()).count(),
        support_loads,
    }
}

/// SPEC.md section 7.6: a smaller plan must preserve every conflict and ordered
/// pair. Search exhaustion retains the best complete plan found so far.
///
/// TODO: maybe formalize the invariant validity of this with aeneas, or extract
// from lean (or rocq), or at least write a non computational verification
// function for the result that describes te contract more clearly (maybe not
// for soundness but for completeness)
fn compact_slots(slots: &mut Vec<usize>, conflicts: &[(usize, usize, bool)]) {
    let mut budget = 10_000;
    let mut degree = vec![0; slots.len()];
    for &(a, b, _) in conflicts {
        degree[a] += 1;
        degree[b] += 1;
    }
    while let Some(max_slot) = slots.iter().copied().max() {
        let current_count = max_slot + 1;
        let target_count = current_count - 1;
        if target_count == 0 || target_count >= u64::BITS as usize {
            break;
        }
        let domains = vec![(1_u64 << target_count) - 1; slots.len()];
        let Some(smaller) = color_slots(domains, conflicts, &degree, slots, &mut budget) else {
            break;
        };
        *slots = smaller
            .iter()
            .map(|bits| bits.trailing_zeros() as usize)
            .collect();
    }
}

fn color_slots(
    mut domains: Vec<u64>,
    conflicts: &[(usize, usize, bool)],
    degree: &[usize],
    preferred: &[usize],
    budget: &mut usize,
) -> Option<Vec<u64>> {
    *budget = budget.checked_sub(1)?;
    loop {
        let mut changed = false;
        for &(a, b, ordered) in conflicts {
            let (mut da, mut db) = (domains[a], domains[b]);
            if ordered {
                // a < b: remove values without a possible later/earlier partner.
                da &= (1_u64 << (63 - db.leading_zeros())) - 1;
                db &= !((1_u64 << (domains[a].trailing_zeros() + 1)) - 1);
            } else {
                if db.is_power_of_two() {
                    da &= !db;
                }
                if domains[a].is_power_of_two() {
                    db &= !domains[a];
                }
            }
            if da == 0 || db == 0 {
                return None;
            }
            changed |= da != domains[a] || db != domains[b];
            domains[a] = da;
            domains[b] = db;
        }
        // the loop terminates because the logic above only removes bits/colors,
        // never adds them, so either we hit the no solution early return, or we
        // found our sets
        if !changed {
            break;
        }
    }

    let next = (0..domains.len())
        .filter(|&i| !domains[i].is_power_of_two())
        .min_by_key(|&i| (domains[i].count_ones(), std::cmp::Reverse(degree[i]), i));

    let Some(next) = next else { return Some(domains) };

    let mut remaining = domains[next];

    while remaining != 0 {
        let preferred_bit = 1_u64.checked_shl(preferred[next] as u32).unwrap_or(0);
        let bit = if remaining & preferred_bit != 0 {
            preferred_bit
        } else {
            1_u64 << remaining.trailing_zeros()
        };
        remaining &= !bit;
        let mut candidate = domains.clone();
        candidate[next] = bit;
        if let Some(solution) = color_slots(candidate, conflicts, degree, preferred, budget) {
            return Some(solution);
        }
        if *budget == 0 {
            break;
        }
    }

    None
}
