//! One-shot strict family audit with complete Rust replay.

use std::error::Error;
use std::fmt;
use std::fmt::Write as _;

use crate::{
    parse_model, render_complete_typed_query, row_is_satisfied, typed_target_row_is_satisfied, Conclusion, FieldModel,
    Problem, Query, Selection, SolverConfig, SolverRun, SolverStatus, TypedTarget,
};

/// Bounded evidence for one complete typed family query.
///
/// Every conclusion remains a candidate. This report cannot authorize a row
/// removal without the required Lean certificate or counterexample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompleteTypedAuditReport {
    pub conclusion: Conclusion,
    pub query: Query,
    pub solver_run: SolverRun,
    pub model: Option<FieldModel>,
    pub retained_rows_replayed: Vec<usize>,
    pub violated_target_rows: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompleteTypedAuditError(String);

impl CompleteTypedAuditError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for CompleteTypedAuditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CompleteTypedAuditError {}

/// Run one strict family query and replay a SAT model in Rust.
///
/// The query includes every exact source row outside the selected family, the
/// constant-one equation, all canonical columns, and the negation of an
/// independent complete typed target. A parse or replay failure is an error
/// and therefore inconclusive for constraint removal.
pub fn audit_complete_typed_family(
    problem: &Problem,
    selection: &Selection,
    target: &TypedTarget,
    solver: &SolverConfig,
) -> Result<CompleteTypedAuditReport, CompleteTypedAuditError> {
    require_family(selection)?;
    let query = render_complete_typed_query(problem, selection, target)
        .map_err(|error| CompleteTypedAuditError::new(format!("query generation failed: {error}")))?;
    run_and_replay(problem, target, query, solver, None)
}

/// Ask cvc5 to check one complete Rust-replayed candidate assignment.
///
/// The solver remains untrusted. Its returned full model must equal the
/// pinned candidate and must pass retained-row and typed-target replay.
pub fn audit_complete_typed_candidate(
    problem: &Problem,
    selection: &Selection,
    target: &TypedTarget,
    candidate: &FieldModel,
    solver: &SolverConfig,
) -> Result<CompleteTypedAuditReport, CompleteTypedAuditError> {
    require_family(selection)?;
    if candidate.values().len() != problem.column_count {
        return Err(CompleteTypedAuditError::new(format!(
            "candidate assignment has {} columns; expected {}",
            candidate.values().len(),
            problem.column_count
        )));
    }
    if candidate.values()[problem.constant_one_column] != 1 {
        return Err(CompleteTypedAuditError::new(
            "candidate assignment does not set the constant-one column to one",
        ));
    }
    let mut query = render_complete_typed_query(problem, selection, target)
        .map_err(|error| CompleteTypedAuditError::new(format!("query generation failed: {error}")))?;
    pin_candidate(&mut query, candidate)?;
    run_and_replay(problem, target, query, solver, Some(candidate))
}

fn require_family(selection: &Selection) -> Result<(), CompleteTypedAuditError> {
    if matches!(selection, Selection::Family(_)) {
        Ok(())
    } else {
        Err(CompleteTypedAuditError::new(
            "strict typed audit requires a complete family selection",
        ))
    }
}

fn pin_candidate(query: &mut Query, candidate: &FieldModel) -> Result<(), CompleteTypedAuditError> {
    const CHECK_SAT: &str = "(check-sat)\n";
    if !query.smt2.ends_with(CHECK_SAT) {
        return Err(CompleteTypedAuditError::new(
            "strict typed query has no final check-sat command",
        ));
    }
    query.smt2.truncate(query.smt2.len() - CHECK_SAT.len());
    for (column, value) in candidate.values().iter().enumerate() {
        writeln!(query.smt2, "(assert (= x_{column} (as ff{value} F)))").expect("writing to String cannot fail");
    }
    query.smt2.push_str(CHECK_SAT);
    Ok(())
}

fn run_and_replay(
    problem: &Problem,
    target: &TypedTarget,
    query: Query,
    solver: &SolverConfig,
    expected_model: Option<&FieldModel>,
) -> Result<CompleteTypedAuditReport, CompleteTypedAuditError> {
    let solver_run = crate::run_cvc5(&query, solver)
        .map_err(|error| CompleteTypedAuditError::new(format!("cvc5 failed: {error}")))?;

    if solver_run.status != SolverStatus::Sat {
        return Ok(CompleteTypedAuditReport {
            conclusion: solver_run.conclusion,
            query,
            solver_run,
            model: None,
            retained_rows_replayed: Vec::new(),
            violated_target_rows: Vec::new(),
        });
    }

    let model = parse_model(&solver_run.stdout, problem.column_count)
        .map_err(|error| CompleteTypedAuditError::new(format!("cvc5 model parse failed: {error}")))?;
    if expected_model.is_some_and(|expected| expected != &model) {
        return Err(CompleteTypedAuditError::new(
            "cvc5 model differs from the complete pinned candidate assignment",
        ));
    }
    if model.values()[problem.constant_one_column] != 1 {
        return Err(CompleteTypedAuditError::new(
            "cvc5 model does not set the constant-one column to one",
        ));
    }

    let mut retained_rows_replayed = Vec::with_capacity(query.retained_rows.len());
    for reference in &query.retained_rows {
        let row = problem.rows.get(reference.problem_index).ok_or_else(|| {
            CompleteTypedAuditError::new(format!(
                "retained row reference {} is out of range",
                reference.problem_index
            ))
        })?;
        let holds = row_is_satisfied(row, &model)
            .map_err(|error| CompleteTypedAuditError::new(format!("retained row replay failed: {error}")))?;
        if !holds {
            return Err(CompleteTypedAuditError::new(format!(
                "cvc5 model violates asserted retained source row {}",
                row.source_index
            )));
        }
        retained_rows_replayed.push(row.source_index);
    }

    let mut violated_target_rows = Vec::new();
    for (index, row) in target.rows.iter().enumerate() {
        let holds = typed_target_row_is_satisfied(row, &model)
            .map_err(|error| CompleteTypedAuditError::new(format!("typed target replay failed: {error}")))?;
        if !holds {
            violated_target_rows.push(index);
        }
    }
    if violated_target_rows.is_empty() {
        return Err(CompleteTypedAuditError::new(
            "cvc5 model does not violate the complete typed target relation",
        ));
    }

    Ok(CompleteTypedAuditReport {
        conclusion: Conclusion::CounterexampleCandidate,
        query,
        solver_run,
        model: Some(model),
        retained_rows_replayed,
        violated_target_rows,
    })
}
