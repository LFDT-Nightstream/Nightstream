//! Data transforms and diagnostics used by low-norm R1CS lowering.
//!
//! Owns: assignment/column normalization, encoded matrix-term expansion, and
//! relation-satisfaction diagnostics.
//!
//! Does not own: encoding policy, CCS polynomial construction, source semantics,
//! or proof authority.
//!
//! Emits constraints: no. It returns normalized values and row terms to the
//! parent lowering.
//!
//! Authority boundary: public-column order and slot maps are supplied by the
//! parent; diagnostics and transformed terms are not independent authority.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Assignment order | [`normalized_field_assignment`] | no | Verifier-selected public outputs |
//! | Matrix expansion | `encoded_matrix_rows` | no | Source matrix and constrained slot map |
//! | Satisfaction diagnostics | `first_unsatisfied_structure_row` | no | Completed structure and assignment |

use neo_ccs::CcsMatrix;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::FieldR1csLoweringError;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::paper::relations::Structure;

pub(crate) fn normalized_field_assignment(
    builder: &R1csBuilder,
    public_outputs: &[Var],
) -> Result<Vec<F>, FieldR1csLoweringError> {
    let witness = builder.witness();
    let cols = witness.len();
    let mut selected = vec![false; cols];
    selected[Var::ONE.col()] = true;
    let mut assignment = Vec::with_capacity(cols);
    assignment.push(witness[Var::ONE.col()]);
    for output in public_outputs {
        let col = output.col();
        if col == Var::ONE.col() {
            return Err(FieldR1csLoweringError::ConstantOneIsImplicit);
        }
        if col >= cols {
            return Err(FieldR1csLoweringError::PublicOutputOutOfRange { col, cols });
        }
        if selected[col] {
            return Err(FieldR1csLoweringError::DuplicatePublicOutput { col });
        }
        selected[col] = true;
        assignment.push(witness[col]);
    }
    assignment.extend(
        (1..cols)
            .filter(|&col| !selected[col])
            .map(|col| witness[col]),
    );
    Ok(assignment)
}

/// Recover a normalized column's source only when an error needs it.
pub(crate) fn normalized_source_column(cols: usize, public_outputs: &[Var], normalized_col: usize) -> Option<usize> {
    if normalized_col >= cols {
        return None;
    }
    if normalized_col == 0 {
        return Some(Var::ONE.col());
    }
    if normalized_col <= public_outputs.len() {
        return Some(public_outputs[normalized_col - 1].col());
    }

    let target = normalized_col - 1 - public_outputs.len();
    let mut excluded = public_outputs
        .iter()
        .map(|output| output.col())
        .collect::<Vec<_>>();
    excluded.sort_unstable();
    let mut remaining = 0usize;
    for source_col in 1..cols {
        if excluded.binary_search(&source_col).is_ok() {
            continue;
        }
        if remaining == target {
            return Some(source_col);
        }
        remaining += 1;
    }
    None
}

pub(super) fn eval_source_lc(lc: &Lc, assignment: &[F]) -> F {
    lc.terms
        .iter()
        .fold(lc.constant, |sum, &(column, coefficient)| {
            sum + coefficient * assignment[column]
        })
}

pub(super) fn encoded_matrix_rows(
    matrix: &CcsMatrix<F>,
    slots: &[Option<(usize, usize)>],
    rows: usize,
) -> Vec<Vec<(usize, F)>> {
    let mut out = vec![Vec::new(); rows];
    match matrix {
        CcsMatrix::Identity { n } => {
            for row in 0..(*n).min(rows).min(slots.len()) {
                extend_encoded_terms(&mut out[row], row, F::ONE, slots);
            }
        }
        CcsMatrix::Csc(csc) => {
            for col in 0..csc.ncols.min(slots.len()) {
                for idx in csc.column_range(col) {
                    let row = csc.row_index(idx);
                    if row < rows {
                        extend_encoded_terms(&mut out[row], col, csc.vals[idx], slots);
                    }
                }
            }
        }
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            for col in 0..csc.ncols.min(slots.len()) {
                for idx in csc.column_range(col) {
                    let row = csc.row_index(idx);
                    if row < rows {
                        extend_encoded_terms(&mut out[row], col, csc.vals[idx], slots);
                    }
                }
            }
            for block in blocks {
                block.for_each_term::<F, _>(|row, col, coefficient| {
                    if row < rows {
                        extend_encoded_terms(&mut out[row], col, coefficient, slots);
                    }
                });
            }
            for run in geometric_runs {
                run.for_each_term(|row, col, coefficient| {
                    if row < rows {
                        extend_encoded_terms(&mut out[row], col, coefficient, slots);
                    }
                });
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            panic!("encoded source rows require materialized matrix content")
        }
    }
    out
}

fn extend_encoded_terms(out: &mut Vec<(usize, F)>, field_col: usize, coefficient: F, slots: &[Option<(usize, usize)>]) {
    if coefficient == F::ZERO {
        return;
    }
    if field_col == 0 {
        out.push((0, coefficient));
        return;
    }
    let (start, width) = slots[field_col].expect("every non-constant R1CS column has a bit slot");
    let mut power = coefficient;
    for bit in 0..width {
        out.push((start + bit, power));
        power += power;
    }
}

pub(super) fn is_structure_satisfied(structure: &Structure, assignment: &[F]) -> bool {
    first_unsatisfied_structure_row(structure, assignment).is_none()
}

pub(super) fn first_unsatisfied_structure_row(structure: &Structure, assignment: &[F]) -> Option<usize> {
    if assignment.len() != structure.m {
        return Some(structure.n);
    }
    let mut matrix_z = vec![vec![F::ZERO; structure.n]; structure.matrices.len()];
    for (matrix, values) in structure.matrices.iter().zip(matrix_z.iter_mut()) {
        matrix.add_mul_into(assignment, values, structure.n);
    }
    (0..structure.n).find(|&row| {
        let point: Vec<F> = matrix_z.iter().map(|values| values[row]).collect();
        structure.f.eval(&point) != F::ZERO
    })
}
