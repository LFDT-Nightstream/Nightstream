//! Compact operation-table constraints and witness audits.
//! Owns lookup synthesis and advice checks; it does not prove memory consistency.

mod builder;
mod compact;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::layout::COL_ONE;
use builder::LookupR1csRow;

#[doc(hidden)]
pub fn audit_compact_lookup_witness(base_assignment: &[F]) -> Result<usize, LookupCircuitError> {
    let (rows, fixed_auxiliary) = fixed_rows(base_assignment.len())?;
    let (_, auxiliary_assignment) = compact::synthesize(base_assignment)?;
    if auxiliary_assignment.len() != fixed_auxiliary.len() {
        return Err(LookupCircuitError::AuxiliaryShapeDrift {
            actual: auxiliary_assignment.len(),
            expected: fixed_auxiliary.len(),
        });
    }
    let mut assignment = base_assignment.to_vec();
    assignment.extend_from_slice(&auxiliary_assignment);
    for (row_index, row) in rows.iter().enumerate() {
        let left = evaluate(&row.a_terms, &assignment);
        let right = evaluate(&row.b_terms, &assignment);
        let output = evaluate(&row.c_terms, &assignment);
        if left * right != output {
            return Err(LookupCircuitError::Unsatisfied { row: row_index });
        }
    }
    Ok(auxiliary_assignment.len())
}

#[doc(hidden)]
pub fn audit_compact_lookup_auxiliary_load_bearing(base_assignment: &[F]) -> Result<usize, LookupCircuitError> {
    let (rows, fixed_auxiliary) = fixed_rows(base_assignment.len())?;
    let (_, auxiliary_assignment) = compact::synthesize(base_assignment)?;
    if auxiliary_assignment.len() != fixed_auxiliary.len() {
        return Err(LookupCircuitError::AuxiliaryShapeDrift {
            actual: auxiliary_assignment.len(),
            expected: fixed_auxiliary.len(),
        });
    }
    let mut assignment = base_assignment.to_vec();
    assignment.extend_from_slice(&auxiliary_assignment);
    let auxiliary_start = base_assignment.len();
    for column in auxiliary_start..assignment.len() {
        assignment[column] = F::ONE - assignment[column];
        let rejected = rows.iter().any(|row| {
            evaluate(&row.a_terms, &assignment) * evaluate(&row.b_terms, &assignment)
                != evaluate(&row.c_terms, &assignment)
        });
        assignment[column] = F::ONE - assignment[column];
        if !rejected {
            return Err(LookupCircuitError::UnconstrainedAuxiliary { column });
        }
    }
    Ok(auxiliary_assignment.len())
}

fn fixed_rows(base_columns: usize) -> Result<(Vec<LookupR1csRow>, Vec<F>), LookupCircuitError> {
    if COL_ONE >= base_columns {
        return Err(LookupCircuitError::MissingConstantColumn {
            columns: base_columns,
            constant: COL_ONE,
        });
    }
    let mut zero_assignment = vec![F::ZERO; base_columns];
    zero_assignment[COL_ONE] = F::ONE;
    compact::synthesize(&zero_assignment).map_err(Into::into)
}

fn evaluate(terms: &[(usize, F)], assignment: &[F]) -> F {
    terms.iter().fold(F::ZERO, |sum, &(column, coefficient)| {
        sum + assignment[column] * coefficient
    })
}

#[derive(Debug, Error)]
pub enum LookupCircuitError {
    #[error("lookup relation synthesis failed: {0}")]
    Synthesis(String),
    #[error("lookup relation row {row} is unsatisfied")]
    Unsatisfied { row: usize },
    #[error("lookup relation auxiliary column {column} can be flipped without violating a row")]
    UnconstrainedAuxiliary { column: usize },
    #[error("lookup relation has {actual} witness auxiliary columns, but its fixed structure has {expected}")]
    AuxiliaryShapeDrift { actual: usize, expected: usize },
    #[error("lookup relation has {columns} base columns and cannot address constant column {constant}")]
    MissingConstantColumn { columns: usize, constant: usize },
}

impl From<String> for LookupCircuitError {
    fn from(value: String) -> Self {
        Self::Synthesis(value)
    }
}
