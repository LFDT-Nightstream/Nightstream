//! Exact linear-definition discovery for selective lowering.
//!
//! The source matrices remain authoritative. This module checks each exact
//! linear row before it removes the source value.

use std::collections::HashMap;

use neo_ccs::{CcsMatrix, CscMat};
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

use super::{trace_error, Lc, LowNormR1csError, SparseR1cs};

pub(super) struct LinearDefinition {
    pub(super) row: Option<usize>,
    pub(super) target: usize,
    pub(super) rhs: Lc,
}

pub(super) struct LinearDefinitions {
    pub(super) by_column: Vec<Option<usize>>,
    pub(super) entries: Vec<LinearDefinition>,
}

impl LinearDefinitions {
    pub(super) fn get(&self, column: usize) -> Option<&Lc> {
        self.by_column[column].map(|index| &self.entries[index].rhs)
    }
}

pub(super) fn find_linear_definitions(
    arm: &SparseR1cs,
    shared_end: usize,
    directly_eliminated: &[bool],
    skipped: &[bool],
) -> Result<LinearDefinitions, LowNormR1csError> {
    let mut protected = directly_eliminated.to_vec();
    protected[..shared_end].fill(true);
    for decomposition in arm.canonical_u64_decompositions() {
        protected[decomposition.field_col] = true;
        for &column in &decomposition.bit_cols {
            protected[column] = true;
        }
    }
    for decomposition in arm.balanced_ternary_decompositions() {
        protected[decomposition.field_col] = true;
        for &column in &decomposition.digit_cols {
            protected[column] = true;
        }
    }
    for trace in arm.poseidon2_traces() {
        for sbox in &trace.sboxes {
            protected[sbox.output_col] = true;
        }
    }
    for trace in arm.polynomial_evaluation_traces() {
        for &column in &trace.output_cols {
            protected[column] = true;
        }
    }
    for trace in arm.product_sum_batch_traces() {
        for &column in &trace.retained_columns {
            protected[column] = true;
        }
    }
    if let CcsMatrix::CscWithSeededPhi81 { blocks, .. } = &arm.a {
        for block in blocks {
            for &start in block.word_starts() {
                protected[start..start + block.word_width()].fill(true);
            }
        }
    }

    let mut by_column = vec![None; arm.m];
    let mut entries = Vec::<LinearDefinition>::new();
    for trace in arm.poseidon2_traces() {
        for (&target, rhs) in trace.output_cols.iter().zip(&trace.output_linear_forms) {
            if target == 0 || target >= arm.m {
                return Err(trace_error("Poseidon2 output column is outside the source arm"));
            }
            if target < shared_end || protected[target] {
                protected[target] = true;
                continue;
            }
            if by_column[target].is_some() {
                return Err(trace_error("Poseidon2 output column has multiple linear definitions"));
            }
            let index = entries.len();
            by_column[target] = Some(index);
            entries.push(LinearDefinition {
                row: None,
                target,
                rhs: rhs.clone(),
            });
        }
    }

    let mut b_state = vec![0u8; arm.n];
    for_each_explicit_term(&arm.b, |row, column, coefficient| {
        b_state[row] = if b_state[row] == 0 && column == 0 && coefficient == F::ONE {
            1
        } else {
            2
        };
    });
    let mut c_nonzero = vec![false; arm.n];
    for_each_explicit_term(&arm.c, |row, _, _| c_nonzero[row] = true);

    let mut candidates = HashMap::<usize, (usize, F)>::new();
    for_each_explicit_term(&arm.a, |row, column, coefficient| {
        if skipped[row] || b_state[row] != 1 || c_nonzero[row] || column == 0 {
            return;
        }
        let candidate = candidates.entry(row).or_insert((column, coefficient));
        if column > candidate.0 {
            *candidate = (column, coefficient);
        }
    });
    let mut candidates = candidates
        .into_iter()
        .filter(|(_, (target, _))| !protected[*target])
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|(row, _)| *row);

    let mut row_to_definition = HashMap::<usize, usize>::new();
    let mut target_coefficients = HashMap::<usize, (usize, F)>::new();
    for (row, (target, coefficient)) in &candidates {
        if by_column[*target].is_some() {
            continue;
        }
        let index = entries.len();
        by_column[*target] = Some(index);
        row_to_definition.insert(*row, index);
        target_coefficients.insert(*row, (*target, *coefficient));
        entries.push(LinearDefinition {
            row: Some(*row),
            target: *target,
            rhs: Lc::zero(),
        });
    }

    for_each_explicit_term(&arm.a, |row, column, coefficient| {
        let Some(&definition_index) = row_to_definition.get(&row) else {
            return;
        };
        let definition = &mut entries[definition_index];
        if column == definition.target {
            return;
        }
        let scale = -target_coefficients[&row].1.inverse();
        if column == 0 {
            definition.rhs.constant += coefficient * scale;
        } else {
            definition.rhs.terms.push((column, coefficient * scale));
        }
    });
    if let Some((target, dependency)) = entries.iter().find_map(|definition| {
        definition.rhs.terms.iter().find_map(|&(column, _)| {
            let invalid =
                column >= definition.target || column >= directly_eliminated.len() || directly_eliminated[column];
            invalid.then_some((definition.target, column))
        })
    }) {
        return Err(trace_error(&format!(
            "linear definition for column {target} is not acyclic over retained dependency {dependency}"
        )));
    }
    Ok(LinearDefinitions { by_column, entries })
}

fn for_each_explicit_term(matrix: &CcsMatrix<F>, mut visit: impl FnMut(usize, usize, F)) {
    let mut visit_csc = |csc: &CscMat<F>| {
        for column in 0..csc.ncols {
            for index in csc.column_range(column) {
                visit(csc.row_index(index), column, csc.vals[index]);
            }
        }
    };
    match matrix {
        CcsMatrix::Identity { n } => {
            for row in 0..*n {
                visit(row, row, F::ONE);
            }
        }
        CcsMatrix::Csc(csc) => visit_csc(csc),
        CcsMatrix::CscWithSeededPhi81 { csc, .. } => visit_csc(csc),
        CcsMatrix::VerifierArtifact { .. } => panic!("selective planning requires materialized source matrices"),
    }
}
