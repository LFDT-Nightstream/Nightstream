//! Exact linear-definition discovery for selective lowering.
//!
//! The source matrices remain authoritative. PiDEC radix-four metadata only
//! identifies candidate rows; this module checks every row before it removes
//! the source value and substitutes two retained signed-unit limbs.

use std::collections::{BTreeSet, HashMap};

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

    let radix_four = collect_radix_four_definitions(arm, shared_end, skipped, &mut protected)?;
    let radix_four_targets = radix_four
        .iter()
        .map(|trace| trace.value_col)
        .collect::<BTreeSet<_>>();
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
    for trace in &radix_four {
        if b_state[trace.row] != 1 || c_nonzero[trace.row] {
            return Err(trace_error("radix-four decomposition is not one exact linear R1CS row"));
        }
        let index = entries.len();
        if by_column[trace.value_col].replace(index).is_some() || row_to_definition.insert(trace.row, index).is_some() {
            return Err(trace_error(
                "radix-four decomposition has duplicate row or value ownership",
            ));
        }
        target_coefficients.insert(trace.row, (trace.value_col, F::ONE));
        entries.push(LinearDefinition {
            row: Some(trace.row),
            target: trace.value_col,
            rhs: Lc::zero(),
        });
    }
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

    let mut actual_target_coefficients = HashMap::<usize, F>::new();
    for_each_explicit_term(&arm.a, |row, column, coefficient| {
        let Some(&definition_index) = row_to_definition.get(&row) else {
            return;
        };
        let definition = &mut entries[definition_index];
        if column == definition.target {
            *actual_target_coefficients.entry(row).or_insert(F::ZERO) += coefficient;
            return;
        }
        let scale = -target_coefficients[&row].1.inverse();
        if column == 0 {
            definition.rhs.constant += coefficient * scale;
        } else {
            definition.rhs.terms.push((column, coefficient * scale));
        }
    });
    validate_radix_four_rows(&radix_four, &entries, &row_to_definition, &actual_target_coefficients)?;

    if let Some((target, dependency)) = entries.iter().find_map(|definition| {
        definition.rhs.terms.iter().find_map(|&(column, _)| {
            let invalid = if radix_four_targets.contains(&definition.target) {
                column >= directly_eliminated.len() || directly_eliminated[column] || by_column[column].is_some()
            } else {
                column >= definition.target || column >= directly_eliminated.len() || directly_eliminated[column]
            };
            invalid.then_some((definition.target, column))
        })
    }) {
        return Err(trace_error(&format!(
            "linear definition for column {target} is not acyclic over retained dependency {dependency}"
        )));
    }
    Ok(LinearDefinitions { by_column, entries })
}

fn collect_radix_four_definitions(
    arm: &SparseR1cs,
    shared_end: usize,
    skipped: &[bool],
    protected: &mut [bool],
) -> Result<Vec<crate::engine::r1cs_circuit::builder::PiDecRadixFourDecompositionAudit>, LowNormR1csError> {
    let mut out = Vec::new();
    let mut rows = BTreeSet::new();
    let mut targets = BTreeSet::new();
    for audit in arm.pi_dec_strict_audits() {
        if audit.radix != 4 && !audit.x_radix_four_decompositions.is_empty() {
            return Err(trace_error(
                "non-radix-four PiDEC audit carries radix-four decompositions",
            ));
        }
        if audit.radix == 4 && audit.x_radix_four_decompositions.is_empty() {
            return Err(trace_error("radix-four PiDEC audit omits its exact decompositions"));
        }
        for &trace in &audit.x_radix_four_decompositions {
            let [low, high] = trace.limb_cols;
            if trace.row >= arm.n
                || skipped[trace.row]
                || trace.value_col < shared_end
                || trace.value_col >= arm.m
                || low == 0
                || high == 0
                || low >= arm.m
                || high >= arm.m
                || trace.value_col == low
                || trace.value_col == high
                || low == high
                || !rows.insert(trace.row)
                || !targets.insert(trace.value_col)
            {
                return Err(trace_error("radix-four PiDEC decomposition has invalid geometry"));
            }
            protected[trace.value_col] = true;
            protected[low] = true;
            protected[high] = true;
            out.push(trace);
        }
    }
    Ok(out)
}

fn validate_radix_four_rows(
    traces: &[crate::engine::r1cs_circuit::builder::PiDecRadixFourDecompositionAudit],
    entries: &[LinearDefinition],
    row_to_definition: &HashMap<usize, usize>,
    actual_target_coefficients: &HashMap<usize, F>,
) -> Result<(), LowNormR1csError> {
    for trace in traces {
        let definition = &entries[row_to_definition[&trace.row]];
        let expected = vec![(trace.limb_cols[0], F::ONE), (trace.limb_cols[1], F::from_u64(2))];
        if actual_target_coefficients.get(&trace.row) != Some(&F::ONE)
            || definition.rhs.constant != F::ZERO
            || definition.rhs.terms != expected
        {
            return Err(trace_error(
                "radix-four PiDEC decomposition row differs from d = low + 2*high",
            ));
        }
    }
    Ok(())
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
