//! Packs logical memory ports using relation-enforced activation facts.
//! The plan is backend-independent; consumers must bind every candidate's
//! address, value, memory identity and activation to the selected physical slot.

use std::collections::{BTreeMap, BTreeSet};

use crate::{MemoryCatalog, MemoryPortActivation, MemoryPortKind};

/// Activation facts of the source relation.
/// These are caller claims, not checked proofs. The relation must enforce
/// Boolean gates, pairwise exclusive support atoms, and every implication
/// and exclusion below. Never derive these facts from observed traces.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryActivationFacts {
    /// `gate = 1` implies that an atom in this set is 1. All atoms across
    /// all sets must be pairwise exclusive. An empty set means the gate is
    /// always zero. Include singleton supports for atoms used as gates.
    pub supports: BTreeMap<usize, BTreeSet<usize>>,
    /// Gate pairs that cannot both be 1 under the source relation.
    pub gate_exclusions: Vec<(usize, usize)>,
}

/// A proposed packing in flattened catalog order.
/// Consumers must check coverage and enforce the slot, order, address and
/// value bindings. This plan and its activation facts are not authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemorySlotPlan {
    /// Candidates per physical slot, indexed in flattened memory/port declaration order.
    pub slots: Vec<Vec<usize>>,
    pub logical_ports: usize,
    /// Caller-supplied disjoint gates used by the planner.
    pub gate_exclusions: Vec<(usize, usize)>,
    /// Ports without a known positive support. Exclusions can still permit sharing.
    pub ports_without_support: usize,
    /// Logical-port load per support atom. This is conservative, not observed execution.
    pub support_loads: BTreeMap<usize, usize>,
}

/// A slot plan that keeps the order a binding needs: two ports that may be
/// active on one row get distinct slots, and when they also share a memory
/// and one of them writes, the earlier port gets the earlier slot. Every other
/// pair may take either order, because their events swap. So a binding that
/// keeps each such pair in catalog order accepts every row the relation
/// accepts, provided the supplied facts hold in that relation. Two `When`
/// ports can share when their known supports are disjoint or a supplied
/// exclusion separates them. Other pairs are treated as potentially active
/// together. Each port takes the first free slot after the slots of the
/// earlier such pairs. A bounded search then tries fewer slots with the same
/// conflict and order conditions. The search runs only below 65 slots, and
/// each of its steps visits every pair that can be active together, so its
/// cost grows with the square of the port count. The result is an upper
/// bound, not a claim that the slot count is minimal.
pub fn plan_ordered_memory_slots<Id>(catalog: &MemoryCatalog<Id>, facts: &MemoryActivationFacts) -> MemorySlotPlan {
    struct Port<'a> {
        memory: usize,
        gate: Option<usize>,
        writes: bool,
        support: Option<&'a BTreeSet<usize>>,
    }
    let supports = &facts.supports;
    let gate_exclusions = &facts.gate_exclusions;
    let memories = catalog.entries();
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
    MemorySlotPlan {
        slots,
        logical_ports: ports.len(),
        gate_exclusions: gate_exclusions.clone(),
        ports_without_support: ports.iter().filter(|port| port.support.is_none()).count(),
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
