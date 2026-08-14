//! Folding-native closure for Enzo's operation-table bindings.
//!
//! The base WASM relation owns selectors, operands, results, and range bits.
//! This module appends a compact R1CS relation plus deterministic Boolean
//! advice; it does not replace or reinterpret the authoritative VM layout.

mod builder;
mod compact;

use neo_ccs::{CcsMatrix, CscMat};
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::layout::COL_ONE;
use crate::tagged_r1cs_builder::WasmR1csRow;

pub(crate) struct CompactLookupShape {
    pub(crate) relation: SparseR1cs,
    pub(crate) widths: Vec<usize>,
    pub(crate) auxiliary_column_count: usize,
}

pub(crate) fn extend_relation(
    base: &SparseR1cs,
    mut widths: Vec<usize>,
) -> Result<CompactLookupShape, LookupCircuitError> {
    if widths.len() != base.m {
        return Err(LookupCircuitError::WidthCount {
            actual: widths.len(),
            expected: base.m,
        });
    }
    let (lookup_rows, auxiliary_assignment) = fixed_rows(base.m)?;
    let columns = base.m + auxiliary_assignment.len();
    let rows = base.n + lookup_rows.len();
    let mut a = matrix_triplets(&base.a)?;
    let mut b = matrix_triplets(&base.b)?;
    let mut c = matrix_triplets(&base.c)?;
    for (offset, row) in lookup_rows.iter().enumerate() {
        let target = base.n + offset;
        a.extend(
            row.a_terms
                .iter()
                .map(|&(column, value)| (target, column, value)),
        );
        b.extend(
            row.b_terms
                .iter()
                .map(|&(column, value)| (target, column, value)),
        );
        c.extend(
            row.c_terms
                .iter()
                .map(|&(column, value)| (target, column, value)),
        );
    }
    let relation = SparseR1cs::new(
        CcsMatrix::Csc(CscMat::from_triplets(a, rows, columns)),
        CcsMatrix::Csc(CscMat::from_triplets(b, rows, columns)),
        CcsMatrix::Csc(CscMat::from_triplets(c, rows, columns)),
        rows,
        columns,
        base.m_in,
    )?;
    widths.resize(columns, 1);
    Ok(CompactLookupShape {
        relation,
        widths,
        auxiliary_column_count: auxiliary_assignment.len(),
    })
}

pub(crate) fn extend_witness(mut base_assignment: Vec<F>) -> Result<Vec<F>, LookupCircuitError> {
    let (_, auxiliary_assignment) = compact::synthesize(&base_assignment)?;
    base_assignment.extend(auxiliary_assignment);
    Ok(base_assignment)
}

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

fn fixed_rows(base_columns: usize) -> Result<(Vec<WasmR1csRow>, Vec<F>), LookupCircuitError> {
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

fn matrix_triplets(matrix: &CcsMatrix<F>) -> Result<Vec<(usize, usize, F)>, LookupCircuitError> {
    match matrix {
        CcsMatrix::Identity { n } => Ok((0..*n).map(|index| (index, index, F::ONE)).collect()),
        CcsMatrix::Csc(csc) => {
            let mut out = Vec::with_capacity(csc.vals.len());
            for column in 0..csc.ncols {
                for index in csc.column_range(column) {
                    out.push((csc.row_index(index), column, csc.vals[index]));
                }
            }
            Ok(out)
        }
        CcsMatrix::CscWithSeededPhi81 { .. } => Err(LookupCircuitError::CompactBaseMatrix),
        CcsMatrix::VerifierArtifact { .. } => Err(LookupCircuitError::VerifierArtifactBaseMatrix),
    }
}

#[derive(Debug, Error)]
pub enum LookupCircuitError {
    #[error("lookup relation has {actual} width declarations for {expected} base columns")]
    WidthCount { actual: usize, expected: usize },
    #[error("lookup relation cannot extend a compact seeded base matrix")]
    CompactBaseMatrix,
    #[error("lookup relation cannot extend a verifier-artifact matrix header")]
    VerifierArtifactBaseMatrix,
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
    #[error(transparent)]
    Frontend(#[from] neo_fold_clean::frontends::direct_ccs::FrontendError),
}

impl From<String> for LookupCircuitError {
    fn from(value: String) -> Self {
        Self::Synthesis(value)
    }
}
