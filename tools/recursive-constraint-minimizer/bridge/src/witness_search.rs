//! Solver-free removal-counterexample construction.
//!
//! An exclusive column is read by exactly one family. Mutating it cannot
//! change another family's rows. A mutation that breaks an owning row can
//! therefore give a removal counterexample. This module replays each result
//! against every exported source row. Lean remains the proof authority.

use std::collections::BTreeSet;

use recursive_constraint_minimizer::{row_is_satisfied, FieldModel, Problem, GOLDILOCKS_MODULUS};

use super::ExportError;

/// One replayed exclusive-column removal counterexample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExclusiveColumnWitness {
    family: String,
    column: usize,
    delta: u64,
    violated_rows: Vec<usize>,
    model: FieldModel,
}

impl ExclusiveColumnWitness {
    pub fn family(&self) -> &str {
        &self.family
    }

    pub fn column(&self) -> usize {
        self.column
    }

    pub fn delta(&self) -> u64 {
        self.delta
    }

    /// Source rows of the removed family that the witness violates.
    pub fn violated_rows(&self) -> &[usize] {
        &self.violated_rows
    }

    pub fn model(&self) -> &FieldModel {
        &self.model
    }
}

/// Search one complete source problem for an exclusive-column witness.
///
/// `background` must be a normalized assignment that satisfies every source
/// row. `Ok(None)` is only a search miss. It is not a necessity or redundancy
/// result.
pub fn find_exclusive_column_witness(
    problem: &Problem,
    background: &[u64],
    family: &str,
) -> Result<Option<ExclusiveColumnWitness>, ExportError> {
    validate_complete_input(problem, background)?;
    if !problem.complete_families.iter().any(|name| name == family) {
        return Err(ExportError::new(format!(
            "family {family:?} is not a complete family of this problem"
        )));
    }

    let modulus = GOLDILOCKS_MODULUS
        .parse::<u64>()
        .expect("fixed Goldilocks modulus fits in u64");
    let mut owner = vec![Ownership::Unused; problem.column_count];
    for row in &problem.rows {
        for term in row.a.iter().chain(&row.b).chain(&row.c) {
            owner[term.column] = match &owner[term.column] {
                Ownership::Unused => Ownership::One(row.family.clone()),
                Ownership::One(existing) if *existing == row.family => Ownership::One(existing.clone()),
                _ => Ownership::Shared,
            };
        }
    }

    let background_model = FieldModel::from_canonical_values(background.to_vec())
        .map_err(|error| ExportError::new(format!("invalid background assignment: {error}")))?;
    for row in &problem.rows {
        let holds = row_is_satisfied(row, &background_model)
            .map_err(|error| ExportError::new(format!("background replay failed: {error}")))?;
        if !holds {
            return Err(ExportError::new(format!(
                "background assignment violates source row {}",
                row.source_index
            )));
        }
    }

    for column in 0..problem.column_count {
        if column == problem.constant_one_column {
            continue;
        }
        if !matches!(&owner[column], Ownership::One(name) if name == family) {
            continue;
        }

        // With one changed column, an R1CS residual has degree at most two.
        // These three distinct nonzero shifts cover every nonzero residual.
        for delta in [1u64, modulus - 1, 2] {
            let mut values = background.to_vec();
            values[column] = ((u128::from(values[column]) + u128::from(delta)) % u128::from(modulus)) as u64;
            let model = FieldModel::from_canonical_values(values)
                .map_err(|error| ExportError::new(format!("mutated model is invalid: {error}")))?;
            let mut violated_rows = Vec::new();
            let mut violates_other_family = false;
            for row in &problem.rows {
                let holds = row_is_satisfied(row, &model)
                    .map_err(|error| ExportError::new(format!("witness replay failed: {error}")))?;
                if holds {
                    continue;
                }
                if row.family == family {
                    violated_rows.push(row.source_index);
                } else {
                    violates_other_family = true;
                    break;
                }
            }
            if !violates_other_family && !violated_rows.is_empty() {
                return Ok(Some(ExclusiveColumnWitness {
                    family: family.to_owned(),
                    column,
                    delta,
                    violated_rows,
                    model,
                }));
            }
        }
    }
    Ok(None)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Ownership {
    Unused,
    One(String),
    Shared,
}

fn validate_complete_input(problem: &Problem, background: &[u64]) -> Result<(), ExportError> {
    problem
        .validate()
        .map_err(|error| ExportError::new(format!("invalid problem: {error}")))?;
    let rows_are_complete = problem.rows.len() == problem.source.total_rows
        && problem
            .rows
            .iter()
            .enumerate()
            .all(|(index, row)| row.source_index == index);
    let row_families = problem
        .rows
        .iter()
        .map(|row| row.family.as_str())
        .collect::<BTreeSet<_>>();
    let complete_families = problem
        .complete_families
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if !rows_are_complete || row_families != complete_families {
        return Err(ExportError::new(
            "exclusive-column search requires every source row and its exact family ledger",
        ));
    }
    if background.len() != problem.column_count {
        return Err(ExportError::new(format!(
            "background assignment has {} columns; the relation has {}",
            background.len(),
            problem.column_count
        )));
    }
    if background[problem.constant_one_column] != 1 {
        return Err(ExportError::new(
            "background assignment does not set the constant-one column to one",
        ));
    }
    Ok(())
}
