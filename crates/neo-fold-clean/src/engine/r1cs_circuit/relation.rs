//! Immutable R1CS relation snapshots used by fixed-size F' lowering.
//!
//! The builder remains the compact synthesis owner. This module expands its
//! seeded Phi81 A-matrix blocks only when a consumer explicitly requests a
//! row-addressable snapshot.

use std::sync::Arc;

use neo_ccs::SeededPhi81LinearBlock;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::builder::Var;

/// Immutable R1CS relation plus one satisfying-candidate witness.
#[derive(Clone, Debug)]
pub struct R1csSnapshot {
    relation: Arc<R1csRelation>,
    witness: Vec<F>,
}

/// Immutable, witness-independent R1CS language.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csRelation {
    a_rows: Vec<Vec<(usize, F)>>,
    b_rows: Vec<Vec<(usize, F)>>,
    c_rows: Vec<Vec<(usize, F)>>,
    cols: usize,
}

impl R1csSnapshot {
    pub(crate) fn from_builder_parts(
        a_trips: &[(usize, usize, F)],
        b_trips: &[(usize, usize, F)],
        c_trips: &[(usize, usize, F)],
        seeded_a_blocks: &[SeededPhi81LinearBlock],
        rows: usize,
        witness: Vec<F>,
    ) -> Self {
        Self {
            relation: Arc::new(R1csRelation {
                a_rows: normalized_a_rows(a_trips, seeded_a_blocks, rows),
                b_rows: normalized_rows(b_trips, rows),
                c_rows: normalized_rows(c_trips, rows),
                cols: witness.len(),
            }),
            witness,
        }
    }

    pub fn rows(&self) -> usize {
        self.relation.a_rows.len()
    }

    pub fn cols(&self) -> usize {
        self.relation.cols
    }

    pub fn witness(&self) -> &[F] {
        &self.witness
    }

    pub fn a_row(&self, row: usize) -> &[(usize, F)] {
        &self.relation.a_rows[row]
    }

    pub fn b_row(&self, row: usize) -> &[(usize, F)] {
        &self.relation.b_rows[row]
    }

    pub fn c_row(&self, row: usize) -> &[(usize, F)] {
        &self.relation.c_rows[row]
    }

    /// Columns explicitly constrained by `v * (v - 1) = 0`.
    pub fn explicitly_boolean_columns(&self) -> Vec<bool> {
        let mut out = vec![false; self.cols()];
        for row in 0..self.rows() {
            if !self.relation.c_rows[row].is_empty() {
                continue;
            }
            if let Some(col) = explicit_bit_row(&self.relation.a_rows[row], &self.relation.b_rows[row])
                .or_else(|| explicit_bit_row(&self.relation.b_rows[row], &self.relation.a_rows[row]))
            {
                out[col] = true;
            }
        }
        out
    }

    /// Source columns that do not occur in any relation row.
    pub fn unconstrained_columns(&self) -> Vec<usize> {
        let mut used = vec![false; self.cols()];
        if !used.is_empty() {
            used[Var::ONE.col()] = true;
        }
        for row in 0..self.rows() {
            for &(column, _) in self.relation.a_rows[row]
                .iter()
                .chain(self.relation.b_rows[row].iter())
                .chain(self.relation.c_rows[row].iter())
            {
                used[column] = true;
            }
        }
        used.into_iter()
            .enumerate()
            .filter_map(|(column, is_used)| (!is_used).then_some(column))
            .collect()
    }

    pub fn has_same_relation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.relation, &other.relation) || self.relation == other.relation
    }

    pub(crate) fn relation_arc(&self) -> Arc<R1csRelation> {
        Arc::clone(&self.relation)
    }

    pub(crate) fn from_shared_relation(relation: Arc<R1csRelation>, witness: Vec<F>) -> Self {
        assert_eq!(relation.cols, witness.len(), "shared R1CS witness width");
        Self { relation, witness }
    }

    pub fn first_unsatisfied_row(&self, witness: &[F]) -> Option<usize> {
        if witness.len() != self.cols() {
            return Some(0);
        }
        (0..self.rows()).find(|&row| {
            eval_row(&self.relation.a_rows[row], witness) * eval_row(&self.relation.b_rows[row], witness)
                != eval_row(&self.relation.c_rows[row], witness)
        })
    }

    pub fn is_satisfied(&self, witness: &[F]) -> bool {
        self.first_unsatisfied_row(witness).is_none()
    }
}

fn normalized_rows(trips: &[(usize, usize, F)], row_count: usize) -> Vec<Vec<(usize, F)>> {
    let mut rows = vec![Vec::new(); row_count];
    for &(row, col, coeff) in trips {
        rows[row].push((col, coeff));
    }
    normalize_rows(&mut rows);
    rows
}

fn normalized_a_rows(
    trips: &[(usize, usize, F)],
    seeded_blocks: &[SeededPhi81LinearBlock],
    row_count: usize,
) -> Vec<Vec<(usize, F)>> {
    let mut rows = vec![Vec::new(); row_count];
    for &(row, col, coeff) in trips {
        rows[row].push((col, coeff));
    }
    for block in seeded_blocks {
        block.for_each_term::<F, _>(|row, col, coeff| rows[row].push((col, coeff)));
    }
    normalize_rows(&mut rows);
    rows
}

fn normalize_rows(rows: &mut [Vec<(usize, F)>]) {
    for row in rows {
        row.sort_unstable_by_key(|&(col, _)| col);
        let mut write = 0usize;
        for read in 0..row.len() {
            let current = row[read];
            if write > 0 && row[write - 1].0 == current.0 {
                row[write - 1].1 += current.1;
            } else {
                row[write] = current;
                write += 1;
            }
        }
        row.truncate(write);
        row.retain(|&(_, coeff)| coeff != F::ZERO);
    }
}

fn explicit_bit_row(variable: &[(usize, F)], variable_minus_one: &[(usize, F)]) -> Option<usize> {
    if variable.len() != 1 || variable[0].0 == Var::ONE.col() || variable[0].1 != F::ONE {
        return None;
    }
    let col = variable[0].0;
    (variable_minus_one == [(Var::ONE.col(), -F::ONE), (col, F::ONE)]).then_some(col)
}

fn eval_row(row: &[(usize, F)], witness: &[F]) -> F {
    row.iter()
        .fold(F::ZERO, |acc, &(col, coeff)| acc + coeff * witness[col])
}
