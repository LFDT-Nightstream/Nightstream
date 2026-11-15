//! SPEC.md section 7.6; Lean `Lifecycle.Nebula.ContinuityRanges.sound`.

use std::collections::{BTreeMap, BTreeSet};

use neo_application::{MemoryCatalog, MemoryPortActivation, MemoryPortKind};
use p3_field::PrimeCharacteristicRing;

use super::{RowRelation, StepLayout, F};

struct Check {
    bits: usize,
    rows: Vec<usize>,
}

pub(super) fn internal_output_checks<Id>(
    relation: &RowRelation,
    links: &[(usize, usize)],
    layout: &StepLayout,
    catalog: &MemoryCatalog<Id>,
) -> BTreeSet<usize> {
    if layout.rows.get() == 1 {
        return BTreeSet::new();
    }
    let mut protected = (0..relation.width)
        .map(|column| column == relation.one || layout.is_shared(column))
        .collect::<Vec<_>>();
    for &(after, before) in links {
        protected[after] = true;
        protected[before] = true;
    }
    for port in catalog.entries().iter().flat_map(|memory| &memory.ports) {
        for &column in &port.address_columns {
            protected[column] = true;
        }
        protected[port.value_column] = true;
        if let MemoryPortKind::Write {
            value_before_column: Some(column),
        } = port.kind
        {
            protected[column] = true;
        }
        if let MemoryPortActivation::When(column) | MemoryPortActivation::Unless(column) = port.activation {
            protected[column] = true;
        }
    }
    let mut uses = vec![BTreeSet::new(); relation.width];
    let mut booleans = BTreeMap::new();
    for (index, [a, b, c]) in relation.rows.iter().enumerate() {
        for &(column, _) in a.iter().chain(b).chain(c) {
            uses[column].insert(index);
        }
        if let [(bit, coefficient)] = a.as_slice() {
            if *coefficient == F::ONE
                && *bit != relation.one
                && c.is_empty()
                && b.len() == 2
                && b.contains(&(*bit, F::ONE))
                && b.contains(&(relation.one, -F::ONE))
            {
                booleans.insert(*bit, index);
            }
        }
    }
    let mut checks = BTreeMap::new();
    for (index, [a, b, c]) in relation.rows.iter().enumerate() {
        let [(value, coefficient)] = c.as_slice() else { continue };
        if *coefficient != F::ONE || b.as_slice() != [(relation.one, F::ONE)] || a.is_empty() || a.len() > 63 {
            continue;
        }
        let mut rows = vec![index];
        let mut weight = F::ONE;
        let mut bits = BTreeSet::new();
        let valid = a.iter().all(|&(bit, coefficient)| {
            let Some(&boolean) = booleans.get(&bit) else {
                return false;
            };
            let valid = !protected[bit]
                && bits.insert(bit)
                && coefficient == weight
                && uses[bit] == BTreeSet::from([index, boolean]);
            weight += weight;
            rows.push(boolean);
            valid
        });
        if valid {
            checks.insert(*value, Check { bits: a.len(), rows });
        }
    }
    let inputs = links
        .iter()
        .map(|&(_, before)| before)
        .collect::<BTreeSet<_>>();
    let mut omitted = BTreeSet::new();
    for &(after, before) in links {
        if inputs.contains(&after) {
            continue;
        }
        let (Some(output), Some(input)) = (checks.get(&after), checks.get(&before)) else {
            continue;
        };
        if output.bits == input.bits {
            omitted.extend(output.rows.iter().copied());
        }
    }
    omitted
}
