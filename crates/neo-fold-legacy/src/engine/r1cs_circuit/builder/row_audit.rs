//! Exact checks for recently emitted builder rows.
//!
//! Owns: comparison of a bounded suffix of A/B/C triplets with pure row
//! formulas. Does not own relation semantics or trace-family selection.

use std::collections::BTreeMap;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::R1csBuilder;
use crate::engine::r1cs_circuit::row_formula::{canonical_sparse_row, ConstraintRow};

impl R1csBuilder {
    /// Fail if the current bounded row suffix differs from the given formulas.
    pub(crate) fn assert_recent_rows_equal(&self, start: usize, expected: &[ConstraintRow]) {
        if !self.record_structure {
            return;
        }
        let end = start
            .checked_add(expected.len())
            .expect("recent row range overflow");
        assert_eq!(end, self.rows, "recent row audit must end at the row frontier");
        assert!(
            self.seeded_phi81_a_blocks
                .iter()
                .all(|block| block.row_end() <= start || block.row_start() >= end),
            "recent row audit does not accept compact seeded A rows"
        );

        let actual = [
            bounded_rows(&self.a_trips, start, end),
            bounded_rows(&self.b_trips, start, end),
            bounded_rows(&self.c_trips, start, end),
        ];
        for (offset, formula) in expected.iter().enumerate() {
            let canonical = canonical_sparse_row(formula);
            for (port, expected) in [canonical.a, canonical.b, canonical.c]
                .into_iter()
                .enumerate()
            {
                assert_eq!(
                    actual[port][offset],
                    expected,
                    "recent row audit mismatch at row {} port {port}",
                    start + offset
                );
            }
        }
    }
}

fn bounded_rows(triplets: &[(usize, usize, F)], start: usize, end: usize) -> Vec<Vec<(usize, F)>> {
    let mut rows = (start..end)
        .map(|_| BTreeMap::<usize, F>::new())
        .collect::<Vec<_>>();
    let first = triplets.partition_point(|&(row, _, _)| row < start);
    for &(row, column, coefficient) in &triplets[first..] {
        if row >= end {
            break;
        }
        *rows[row - start].entry(column).or_insert(F::ZERO) += coefficient;
    }
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .filter(|(_, coefficient)| *coefficient != F::ZERO)
                .collect()
        })
        .collect()
}
