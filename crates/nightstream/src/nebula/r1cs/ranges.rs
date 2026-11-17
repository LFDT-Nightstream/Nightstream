//! SPEC.md section 7.6; Lean `Lifecycle.Nebula.ContinuityRanges.sound`.

use std::collections::{BTreeMap, BTreeSet};

use neo_application::{MemoryCatalog, MemoryPortActivation, MemoryPortKind};
use p3_field::PrimeCharacteristicRing;

use super::{RowRelation, StepLayout, F};

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
    for (index, row) in relation.rows.iter().enumerate() {
        for &(column, _) in row.iter().flatten() {
            uses[column].insert(index);
        }
    }
    let mut checks = BTreeMap::new();
    for check in &relation.range_checks {
        // SPEC.md section 7.6: public RowRelation rows and metadata can change independently
        // after construction. Assert that each recorded check still matches the final relation.
        let bit_count = check.bits.len();
        assert!((1..=63).contains(&bit_count), "invalid range-check width");
        assert!(check.bits.end <= relation.width, "range-check bits outside relation");
        assert!(check.column < relation.width, "range-check column outside relation");
        assert!(
            check.rows.end <= relation.rows.len(),
            "range-check rows outside relation"
        );
        assert_eq!(check.rows.len(), bit_count + 1, "invalid range-check row count");
        let recomposition = check.rows.end - 1;
        let [a, b, c] = &relation.rows[recomposition];
        assert!(
            b.as_slice() == [(relation.one, F::ONE)]
                && c.as_slice() == [(check.column, F::ONE)]
                && a.iter().copied().eq(check
                    .bits
                    .clone()
                    .enumerate()
                    .map(|(i, bit)| (bit, F::from_u64(1 << i)))),
            "range-check metadata does not match recomposition row"
        );

        let mut valid = true;

        for (bit, boolean) in check.bits.clone().zip(check.rows.clone()) {
            let [a, b, c] = &relation.rows[boolean];
            assert!(
                a.as_slice() == [(bit, F::ONE)]
                    && b.len() == 2
                    && b.contains(&(bit, F::ONE))
                    && b.contains(&(relation.one, -F::ONE))
                    && c.is_empty(),
                "range-check metadata does not match Boolean row"
            );
            valid &= !protected[bit] && uses[bit] == BTreeSet::from([boolean, recomposition]);
        }

        if valid {
            checks.insert(check.column, check);
        }
    }

    let mut omitted = BTreeSet::new();

    for &(after, before) in links {
        let (Some(output), Some(input)) = (checks.get(&after), checks.get(&before)) else {
            continue;
        };

        if output.bits.len() == input.bits.len() {
            omitted.extend(output.rows.clone());
        }
    }

    omitted
}
