//! Bounded counterexample-guided query refinement with full Rust replay.

use std::error::Error;
use std::fmt;

use neo_fold_clean::engine::r1cs_circuit::builder::RowFamilyRange;
use neo_fold_clean::engine::r1cs_circuit::R1csSnapshot;
use p3_field::PrimeField64;
use recursive_constraint_minimizer::{
    parse_model, render_query, Conclusion, FieldModel, Problem, Query, Selection, SolverConfig, SolverRun,
    SolverStatus, GOLDILOCKS_MODULUS,
};

use crate::{export_problem, ExportRequest};

pub const MAX_REFINEMENT_ITERATIONS: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefinementReport {
    pub conclusion: Conclusion,
    pub iterations: usize,
    pub problem: Problem,
    pub query: Query,
    pub solver_run: SolverRun,
    pub model: Option<FieldModel>,
    pub violated_candidate_rows: Vec<usize>,
    /// A retained source row that must enter a later query after the iteration cap.
    pub pending_retained_row: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefinementError(String);

impl RefinementError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for RefinementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for RefinementError {}

/// Run cvc5 and replay each `sat` assignment against all retained source rows.
///
/// A replay failure adds one violated retained row to the next query. `unsat`
/// remains non-authoritative solver evidence. An iteration-cap result is
/// inconclusive and keeps the candidate constraint.
pub fn refine_with_cvc5(
    snapshot: &R1csSnapshot,
    ranges: &[RowFamilyRange],
    mut request: ExportRequest,
    selection: &Selection,
    solver: &SolverConfig,
    max_iterations: usize,
) -> Result<RefinementReport, RefinementError> {
    if max_iterations == 0 || max_iterations > MAX_REFINEMENT_ITERATIONS {
        return Err(RefinementError::new(format!(
            "max_iterations must be in 1..={MAX_REFINEMENT_ITERATIONS}"
        )));
    }

    for iteration in 1..=max_iterations {
        let problem = export_problem(snapshot, ranges, request.clone())
            .map_err(|error| RefinementError::new(format!("source export failed: {error}")))?;
        let query = render_query(&problem, selection)
            .map_err(|error| RefinementError::new(format!("query generation failed: {error}")))?;
        let solver_run = recursive_constraint_minimizer::run_cvc5(&query, solver)
            .map_err(|error| RefinementError::new(format!("cvc5 failed: {error}")))?;

        match solver_run.status {
            SolverStatus::Unsat => {
                return Ok(report(
                    Conclusion::RedundancyCandidate,
                    iteration,
                    problem,
                    query,
                    solver_run,
                    None,
                    Vec::new(),
                    None,
                ));
            }
            SolverStatus::Unknown => {
                return Ok(report(
                    Conclusion::Inconclusive,
                    iteration,
                    problem,
                    query,
                    solver_run,
                    None,
                    Vec::new(),
                    None,
                ));
            }
            SolverStatus::Sat => {}
        }

        let model = parse_model(&solver_run.stdout, snapshot.cols())
            .map_err(|error| RefinementError::new(format!("cvc5 model parse failed: {error}")))?;
        if model.values()[0] != 1 {
            return Err(RefinementError::new(
                "cvc5 model does not set the constant-one column to one",
            ));
        }
        let candidate_rows = candidate_source_rows(&problem, selection)?;
        let violated_candidate_rows = candidate_rows
            .iter()
            .copied()
            .filter(|&row| !snapshot_row_holds(snapshot, row, &model))
            .collect::<Vec<_>>();
        if violated_candidate_rows.is_empty() {
            return Err(RefinementError::new(
                "cvc5 model does not violate a selected candidate row",
            ));
        }

        let first_violated_retained = (0..snapshot.rows())
            .find(|row| candidate_rows.binary_search(row).is_err() && !snapshot_row_holds(snapshot, *row, &model));
        let Some(violated_row) = first_violated_retained else {
            return Ok(report(
                Conclusion::CounterexampleCandidate,
                iteration,
                problem,
                query,
                solver_run,
                Some(model),
                violated_candidate_rows,
                None,
            ));
        };
        if request.source_rows.binary_search(&violated_row).is_ok() {
            return Err(RefinementError::new(format!(
                "cvc5 model violates asserted retained source row {violated_row}"
            )));
        }
        if iteration == max_iterations {
            return Ok(report(
                Conclusion::Inconclusive,
                iteration,
                problem,
                query,
                solver_run,
                Some(model),
                violated_candidate_rows,
                Some(violated_row),
            ));
        }
        let insertion = request
            .source_rows
            .binary_search(&violated_row)
            .expect_err("the violated row was checked as absent");
        request.source_rows.insert(insertion, violated_row);
    }
    unreachable!("the bounded refinement loop always returns")
}

#[allow(clippy::too_many_arguments)]
fn report(
    conclusion: Conclusion,
    iterations: usize,
    problem: Problem,
    query: Query,
    solver_run: SolverRun,
    model: Option<FieldModel>,
    violated_candidate_rows: Vec<usize>,
    pending_retained_row: Option<usize>,
) -> RefinementReport {
    RefinementReport {
        conclusion,
        iterations,
        problem,
        query,
        solver_run,
        model,
        violated_candidate_rows,
        pending_retained_row,
    }
}

fn candidate_source_rows(problem: &Problem, selection: &Selection) -> Result<Vec<usize>, RefinementError> {
    let rows: Vec<usize> = match selection {
        Selection::Row(id) => problem
            .rows
            .iter()
            .filter(|row| row.id == *id)
            .map(|row| row.source_index)
            .collect(),
        Selection::Family(family) => problem
            .rows
            .iter()
            .filter(|row| row.family == *family)
            .map(|row| row.source_index)
            .collect(),
    };
    if rows.is_empty() {
        return Err(RefinementError::new(
            "selection does not identify a source candidate row",
        ));
    }
    Ok(rows)
}

fn snapshot_row_holds(snapshot: &R1csSnapshot, row: usize, model: &FieldModel) -> bool {
    let a = evaluate(snapshot.a_row(row), model);
    let b = evaluate(snapshot.b_row(row), model);
    let c = evaluate(snapshot.c_row(row), model);
    multiply(a, b) == c
}

fn evaluate(terms: &[(usize, neo_math::F)], model: &FieldModel) -> u64 {
    let modulus = GOLDILOCKS_MODULUS
        .parse::<u128>()
        .expect("fixed Goldilocks modulus fits in u128");
    terms.iter().fold(0u64, |sum, &(column, coefficient)| {
        let product = multiply(coefficient.as_canonical_u64(), model.values()[column]);
        ((u128::from(sum) + u128::from(product)) % modulus) as u64
    })
}

fn multiply(left: u64, right: u64) -> u64 {
    let modulus = GOLDILOCKS_MODULUS
        .parse::<u128>()
        .expect("fixed Goldilocks modulus fits in u128");
    (u128::from(left) * u128::from(right) % modulus) as u64
}
