//! Streaming evaluation of independent logical rows against an assignment.

use super::assignment::LogicalAssignment;
use super::matrix::MatrixProgram;
use super::relation::Relation;
use super::source::SourcePackage;
use super::{empty_row, Field, Form, Result, RowForms, MATRIX_COUNT};

pub const ACTIVE_ROWS: usize = 6_377_559;
pub const PADDED_ROWS: usize = 1 << 28;
pub const LOGICAL_WIDTH: usize = 256_532_147;
pub const CARRIER_WIDTH: usize = 256_532_184;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Evaluation {
    pub active_rows: usize,
    pub relation_terms: usize,
    pub carrier_padding_columns: usize,
    pub assignment_block_mutations: usize,
    pub matrix_slot_mutations: usize,
    pub zero_slot_mutation_rejected: bool,
    pub public_digest_bit_mutations: usize,
    pub public_digest_word_mutations: usize,
}

pub fn verify_satisfaction_with(
    program: &MatrixProgram,
    sources: &SourcePackage,
    relation: &Relation,
    mut value_at: impl FnMut(usize) -> Field,
) -> Result<()> {
    let mut next = 0usize;
    program.visit_rows(0, ACTIVE_ROWS, sources, |ordinal, row| {
        if ordinal != next {
            return Err(format!("logical row order changed: got {ordinal}, expected {next}"));
        }
        let matrix_values = evaluate_row_with(&row, &mut value_at);
        validate_zero_slot(&matrix_values, ordinal)?;
        let residual = relation.evaluate(&matrix_values);
        if residual != Field::ZERO {
            return Err(format!(
                "logical relation failed at row {ordinal}: {}",
                residual.canonical()
            ));
        }
        next += 1;
        Ok(())
    })?;
    if next != ACTIVE_ROWS {
        return Err(format!("logical row coverage ended at {next}, expected {ACTIVE_ROWS}"));
    }

    let implicit_padding = empty_row();
    if relation.evaluate(&evaluate_row_with(&implicit_padding, &mut value_at)) != Field::ZERO {
        return Err("implicit zero row does not satisfy the CCS polynomial".into());
    }
    for ordinal in [ACTIVE_ROWS, PADDED_ROWS - 1, PADDED_ROWS] {
        if program.row(ordinal, sources).is_ok() {
            return Err(format!("logical padding row {ordinal} decoded as active data"));
        }
    }
    for column in LOGICAL_WIDTH..CARRIER_WIDTH {
        if value_at(column) != Field::ZERO {
            return Err(format!("carrier padding column {column} is nonzero"));
        }
    }
    Ok(())
}

/// Evaluate one exact logical-row range through a fallible assignment view.
/// This is the phase-local form of the full relation check: unavailable
/// assignment values remain errors and cannot be replaced with zero.
pub fn verify_satisfaction_range_with(
    program: &MatrixProgram,
    sources: &SourcePackage,
    relation: &Relation,
    start: usize,
    end: usize,
    mut value_at: impl FnMut(usize) -> Result<Field>,
) -> Result<usize> {
    let mut next = start;
    program.visit_rows(start, end, sources, |ordinal, row| {
        if ordinal != next {
            return Err(format!("logical row order changed: got {ordinal}, expected {next}"));
        }
        let matrix_values = evaluate_row_with_result(&row, &mut value_at)?;
        validate_zero_slot(&matrix_values, ordinal)?;
        let residual = relation.evaluate(&matrix_values);
        if residual != Field::ZERO {
            return Err(format!(
                "logical relation failed at row {ordinal}: {}",
                residual.canonical()
            ));
        }
        next += 1;
        Ok(())
    })?;
    if next != end {
        return Err(format!("logical row coverage ended at {next}, expected {end}"));
    }
    Ok(next - start)
}

pub fn evaluate(
    program: &MatrixProgram,
    sources: &SourcePackage,
    relation: &Relation,
    assignment: &LogicalAssignment,
) -> Result<Evaluation> {
    if assignment.len() != LOGICAL_WIDTH {
        return Err(format!(
            "logical assignment has width {}, expected {LOGICAL_WIDTH}",
            assignment.len()
        ));
    }
    let mut next = 0usize;
    let mut assignment_mutations = [None; 38];
    let mut matrix_mutations = [None; MATRIX_COUNT - 1];
    let mut public_bit_mutations = [None; 256];
    let mut zero_slot_mutation_rejected = false;
    let mut candidate_columns = Vec::new();
    program.visit_rows(0, ACTIVE_ROWS, sources, |ordinal, row| {
        if ordinal != next {
            return Err(format!("logical row order changed: got {ordinal}, expected {next}"));
        }
        let matrix_values = evaluate_row(&row, assignment)?;
        validate_zero_slot(&matrix_values, ordinal)?;
        let residual = relation.evaluate(&matrix_values);
        if residual != Field::ZERO {
            return Err(format!(
                "logical relation failed at row {ordinal}: {}",
                residual.canonical()
            ));
        }

        for (slot, detected) in matrix_mutations.iter_mut().enumerate() {
            if detected.is_none() {
                let mut changed = matrix_values;
                changed[slot] += Field::ONE;
                if relation.evaluate(&changed) != Field::ZERO {
                    *detected = Some(ordinal);
                }
            }
        }
        if !zero_slot_mutation_rejected {
            let mut changed = matrix_values;
            changed[MATRIX_COUNT - 1] = Field::ONE;
            zero_slot_mutation_rejected = validate_zero_slot(&changed, ordinal).is_err();
        }

        if assignment_mutations
            .iter()
            .enumerate()
            .any(|(block, row)| row.is_none() && assignment.block_is_nonempty(block))
            || public_bit_mutations.iter().any(Option::is_none)
        {
            candidate_columns.clear();
            for form in &row {
                for entry in form.entries() {
                    let pending_block = assignment
                        .block_for_column(entry.column)
                        .is_some_and(|block| assignment_mutations[block].is_none());
                    let pending_public_bit =
                        (1..257).contains(&entry.column) && public_bit_mutations[entry.column - 1].is_none();
                    if pending_block || pending_public_bit {
                        candidate_columns.push(entry.column);
                    }
                }
            }
            candidate_columns.sort_unstable();
            candidate_columns.dedup();
            for &column in &candidate_columns {
                let delta = assignment.mutation_delta(column)?;
                let changed = evaluate_row_with_delta(&row, assignment, matrix_values, column, delta)?;
                if relation.evaluate(&changed) == Field::ZERO {
                    continue;
                }
                if let Some(block) = assignment.block_for_column(column) {
                    assignment_mutations[block].get_or_insert(ordinal);
                }
                if (1..257).contains(&column) {
                    public_bit_mutations[column - 1].get_or_insert(ordinal);
                }
            }
        }
        next += 1;
        Ok(())
    })?;
    if next != ACTIVE_ROWS {
        return Err(format!("logical row coverage ended at {next}, expected {ACTIVE_ROWS}"));
    }

    let implicit_padding = empty_row();
    if relation.evaluate(&evaluate_row(&implicit_padding, assignment)?) != Field::ZERO {
        return Err("implicit zero row does not satisfy the CCS polynomial".into());
    }
    for ordinal in [ACTIVE_ROWS, PADDED_ROWS - 1, PADDED_ROWS] {
        if program.row(ordinal, sources).is_ok() {
            return Err(format!("logical padding row {ordinal} decoded as active data"));
        }
    }
    for column in LOGICAL_WIDTH..CARRIER_WIDTH {
        if assignment.carrier_value(column)? != Field::ZERO {
            return Err(format!("carrier padding column {column} is nonzero"));
        }
    }
    let expected_assignment_mutations = assignment.nonempty_block_count();
    let assignment_block_mutations = assignment_mutations.iter().flatten().count();
    if assignment_block_mutations != expected_assignment_mutations {
        let missing = assignment_mutations
            .iter()
            .enumerate()
            .filter_map(|(block, row)| (row.is_none() && assignment.block_is_nonempty(block)).then_some(block))
            .collect::<Vec<_>>();
        return Err(format!(
            "no effective logical-coordinate mutation found for assignment blocks {missing:?}"
        ));
    }
    let matrix_slot_mutations = matrix_mutations.iter().flatten().count();
    if matrix_slot_mutations != MATRIX_COUNT - 1 {
        let missing = matrix_mutations
            .iter()
            .enumerate()
            .filter_map(|(slot, row)| row.is_none().then_some(slot))
            .collect::<Vec<_>>();
        return Err(format!(
            "no effective coefficient mutation found for matrix slots {missing:?}"
        ));
    }
    if !zero_slot_mutation_rejected {
        return Err("a nonzero insertion into logical matrix slot 13 was not rejected".into());
    }
    let public_digest_bit_mutations = public_bit_mutations.iter().flatten().count();
    if public_digest_bit_mutations != public_bit_mutations.len() {
        let missing = public_bit_mutations
            .iter()
            .enumerate()
            .filter_map(|(bit, row)| row.is_none().then_some(bit))
            .collect::<Vec<_>>();
        return Err(format!(
            "no effective logical-public mutation found for digest bits {missing:?}"
        ));
    }
    let public_digest_word_mutations = public_bit_mutations
        .chunks_exact(64)
        .filter(|word| word.iter().all(Option::is_some))
        .count();

    Ok(Evaluation {
        active_rows: next,
        relation_terms: relation.term_count(),
        carrier_padding_columns: CARRIER_WIDTH - LOGICAL_WIDTH,
        assignment_block_mutations,
        matrix_slot_mutations,
        zero_slot_mutation_rejected,
        public_digest_bit_mutations,
        public_digest_word_mutations,
    })
}

pub fn first_failure(
    program: &MatrixProgram,
    sources: &SourcePackage,
    relation: &Relation,
    assignment: &LogicalAssignment,
) -> Result<Option<usize>> {
    if assignment.len() != LOGICAL_WIDTH {
        return Err("mutated logical assignment has the wrong width".into());
    }
    const STOP: &str = "independent logical mutation detected";
    let mut failure = None;
    let result = program.visit_rows(0, ACTIVE_ROWS, sources, |ordinal, row| {
        let values = evaluate_row(&row, assignment)?;
        validate_zero_slot(&values, ordinal)?;
        if relation.evaluate(&values) != Field::ZERO {
            failure = Some(ordinal);
            return Err(STOP.into());
        }
        Ok(())
    });
    match (failure, result) {
        (Some(row), Err(error)) if error == STOP => Ok(Some(row)),
        (None, Ok(())) => Ok(None),
        (_, Err(error)) => Err(error),
        (Some(_), Ok(())) => Err("logical mutation stop was lost".into()),
    }
}

fn validate_zero_slot(values: &[Field; MATRIX_COUNT], ordinal: usize) -> Result<()> {
    if values[MATRIX_COUNT - 1] == Field::ZERO {
        Ok(())
    } else {
        Err(format!("logical zero matrix is nonzero at row {ordinal}"))
    }
}

fn evaluate_row(row: &RowForms, assignment: &LogicalAssignment) -> Result<[Field; MATRIX_COUNT]> {
    let mut values = [Field::ZERO; MATRIX_COUNT];
    for (value, form) in values.iter_mut().zip(row) {
        *value = evaluate_form(form, assignment)?;
    }
    Ok(values)
}

fn evaluate_form(form: &Form, assignment: &LogicalAssignment) -> Result<Field> {
    form.entries().iter().try_fold(Field::ZERO, |sum, entry| {
        Ok(sum + entry.coefficient * assignment.value(entry.column)?)
    })
}

fn evaluate_row_with(row: &RowForms, value_at: &mut impl FnMut(usize) -> Field) -> [Field; MATRIX_COUNT] {
    let mut values = [Field::ZERO; MATRIX_COUNT];
    for (value, form) in values.iter_mut().zip(row) {
        *value = form.entries().iter().fold(Field::ZERO, |sum, entry| {
            sum + entry.coefficient * value_at(entry.column)
        });
    }
    values
}

pub fn evaluate_row_with_result(
    row: &RowForms,
    value_at: &mut impl FnMut(usize) -> Result<Field>,
) -> Result<[Field; MATRIX_COUNT]> {
    let mut values = [Field::ZERO; MATRIX_COUNT];
    for (value, form) in values.iter_mut().zip(row) {
        *value = form
            .entries()
            .iter()
            .try_fold(Field::ZERO, |sum, entry| -> Result<Field> {
                Ok(sum + entry.coefficient * value_at(entry.column)?)
            })?;
    }
    Ok(values)
}

fn evaluate_row_with_delta(
    row: &RowForms,
    assignment: &LogicalAssignment,
    mut values: [Field; MATRIX_COUNT],
    column: usize,
    delta: Field,
) -> Result<[Field; MATRIX_COUNT]> {
    assignment.value(column)?;
    for (value, form) in values.iter_mut().zip(row) {
        if let Ok(position) = form
            .entries()
            .binary_search_by_key(&column, |entry| entry.column)
        {
            *value += form.entries()[position].coefficient * delta;
        }
    }
    Ok(values)
}
