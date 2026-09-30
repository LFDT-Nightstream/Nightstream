//! Literal expansion of verifier-owned CCS matrix descriptions.
//!
//! Compact descriptors are public storage formats. This module expands them
//! from their scalar parameters. It does not call their production entry
//! evaluators.

use neo_ccs::CcsMatrix;
use p3_field::{Field, PrimeCharacteristicRing};

pub(super) fn matrix_entry<Ff>(matrix: &CcsMatrix<Ff>, row: usize, column: usize) -> Ff
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if row >= matrix.rows() || column >= matrix.cols() {
        return Ff::ZERO;
    }
    match matrix {
        CcsMatrix::Identity { .. } => {
            if row == column {
                Ff::ONE
            } else {
                Ff::ZERO
            }
        }
        CcsMatrix::Csc(csc) => csc_entry(csc, row, column),
        CcsMatrix::CscWithGeometricRuns { csc, geometric_runs } => {
            let mut value = csc_entry(csc, row, column);
            for run in geometric_runs {
                if row == run.row() && column >= run.column_start() && column < run.column_start() + run.len() {
                    let mut coefficient = *run.initial();
                    for _ in run.column_start()..column {
                        coefficient *= *run.ratio();
                    }
                    value += coefficient;
                }
            }
            value
        }
        CcsMatrix::VerifierArtifact { .. } => {
            panic!("paper-exact matrix access is unavailable for verifier-artifact matrices")
        }
    }
}

fn csc_entry<Ff>(matrix: &neo_ccs::CscMat<Ff>, row: usize, column: usize) -> Ff
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut value = Ff::ZERO;
    for entry in matrix.column_range(column) {
        if matrix.row_index(entry) == row {
            value += matrix.vals[entry];
        }
    }
    value
}
