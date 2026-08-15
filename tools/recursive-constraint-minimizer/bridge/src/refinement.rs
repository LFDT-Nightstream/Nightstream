//! Bounded counterexample-guided query refinement with full Rust replay.

use std::error::Error;
use std::fmt;

use neo_fold_clean::engine::r1cs_circuit::builder::RowFamilyRange;
use neo_fold_clean::engine::r1cs_circuit::R1csSnapshot;
use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeBranch, NebulaFPrimeConstraintSourceAudit};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvcBranch, R1csIvcConstraintSourceAudit};
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::TerminalR1csConstraintAudit;
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use recursive_constraint_minimizer::{
    parse_model_with_defaults, render_query, Conclusion, FieldModel, Problem, Query, Selection, SolverConfig,
    SolverRun, SolverStatus, GOLDILOCKS_MODULUS,
};

use crate::{
    ExportRequest, FixedPointProblemExport, SnapshotProblemExporter, SparseProblemExporter, TerminalProblemExport,
};

pub const MAX_REFINEMENT_ITERATIONS: usize = 256;

/// Upper bound on violated retained rows added to the slice per iteration.
/// Batching keeps real-arm necessity searches convergent while every query
/// stays a bounded slice.
const REFINEMENT_ROW_BATCH: usize = 512;

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

/// Production fixed-point export plus its bounded solver report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedPointRefinementReport {
    source_export: FixedPointProblemExport,
    refinement: RefinementReport,
}

/// Exact terminal export plus its bounded solver report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalRefinementReport {
    source_export: TerminalProblemExport,
    refinement: RefinementReport,
}

impl TerminalRefinementReport {
    pub fn source_export(&self) -> &TerminalProblemExport {
        &self.source_export
    }

    pub fn refinement(&self) -> &RefinementReport {
        &self.refinement
    }
}

impl FixedPointRefinementReport {
    pub fn source_export(&self) -> &FixedPointProblemExport {
        &self.source_export
    }

    pub fn refinement(&self) -> &RefinementReport {
        &self.refinement
    }
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
/// A replay failure adds a bounded batch of violated retained rows to the next query. `unsat`
/// remains non-authoritative solver evidence. An iteration-cap result is
/// inconclusive and keeps the candidate constraint.
pub fn refine_with_cvc5(
    snapshot: &R1csSnapshot,
    ranges: &[RowFamilyRange],
    request: ExportRequest,
    selection: &Selection,
    solver: &SolverConfig,
    max_iterations: usize,
) -> Result<RefinementReport, RefinementError> {
    if !snapshot.is_satisfied(snapshot.witness()) {
        return Err(RefinementError::new(
            "snapshot background witness does not satisfy the complete source relation",
        ));
    }
    let defaults = snapshot
        .witness()
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect::<Vec<_>>();
    let exporter = SnapshotProblemExporter::new(snapshot, ranges, request.public_input_count)
        .map_err(|error| RefinementError::new(format!("source export setup failed: {error}")))?;
    refine(
        request,
        selection,
        solver,
        max_iterations,
        &defaults,
        |request| {
            exporter
                .export(request.clone())
                .map_err(|error| RefinementError::new(format!("source export failed: {error}")))
        },
        |model, candidate_rows| {
            let violated_candidate_rows = candidate_rows
                .iter()
                .copied()
                .filter(|&row| !snapshot_row_holds(snapshot, row, model))
                .collect::<Vec<_>>();
            let violated_retained_rows = (0..snapshot.rows())
                .filter(|row| candidate_rows.binary_search(row).is_err() && !snapshot_row_holds(snapshot, *row, model))
                .take(REFINEMENT_ROW_BATCH)
                .collect::<Vec<_>>();
            Ok((violated_candidate_rows, violated_retained_rows))
        },
    )
}

/// Run bounded cvc5 refinement against an exact sparse source relation.
///
/// Every `sat` assignment is evaluated through all three complete sparse Rust
/// matrices. The result is a counterexample candidate only when every row
/// outside the selected candidate set holds.
pub fn refine_sparse_with_cvc5(
    arm: &SparseR1cs,
    background_assignment: &[F],
    request: ExportRequest,
    selection: &Selection,
    solver: &SolverConfig,
    max_iterations: usize,
) -> Result<RefinementReport, RefinementError> {
    let exporter = SparseProblemExporter::new(arm)
        .map_err(|error| RefinementError::new(format!("source export setup failed: {error}")))?;
    arm.is_satisfied_by(background_assignment)
        .map_err(|error| {
            RefinementError::new(format!(
                "sparse background assignment does not satisfy the complete source relation: {error}"
            ))
        })?;
    let defaults = background_assignment
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect::<Vec<_>>();
    refine(
        request,
        selection,
        solver,
        max_iterations,
        &defaults,
        |request| {
            exporter
                .export(request.clone())
                .map_err(|error| RefinementError::new(format!("source export failed: {error}")))
        },
        |model, candidate_rows| replay_sparse_arm(arm, model, candidate_rows),
    )
}

/// Run sparse refinement for one reviewed fixed-point arm and retain its exact
/// source-to-final selective binding.
pub fn refine_fixed_point_with_cvc5(
    audit: &R1csIvcConstraintSourceAudit,
    branch: R1csIvcBranch,
    background_assignment: &[F],
    request: ExportRequest,
    selection: &Selection,
    solver: &SolverConfig,
    max_iterations: usize,
) -> Result<FixedPointRefinementReport, RefinementError> {
    crate::validate_fixed_point_stage_vocabulary(audit.arm(branch), branch)
        .map_err(|error| RefinementError::new(format!("fixed-point source validation failed: {error}")))?;
    let refinement = refine_sparse_with_cvc5(
        audit.arm(branch),
        background_assignment,
        request,
        selection,
        solver,
        max_iterations,
    )?;
    let source_export = crate::selective_binding::bind_fixed_point_problem(audit, branch, refinement.problem.clone())
        .map_err(|error| RefinementError::new(format!("fixed-point result binding failed: {error}")))?;
    Ok(FixedPointRefinementReport {
        source_export,
        refinement,
    })
}

/// Run sparse refinement for one reviewed Nebula F-prime arm and retain its
/// exact source-to-final selective binding.
pub fn refine_nebula_with_cvc5(
    audit: &NebulaFPrimeConstraintSourceAudit,
    branch: NebulaFPrimeBranch,
    background_assignment: &[F],
    request: ExportRequest,
    selection: &Selection,
    solver: &SolverConfig,
    max_iterations: usize,
) -> Result<FixedPointRefinementReport, RefinementError> {
    crate::validate_nebula_stage_vocabulary(audit, branch)
        .map_err(|error| RefinementError::new(format!("Nebula source validation failed: {error}")))?;
    let refinement = refine_sparse_with_cvc5(
        audit.arm(branch),
        background_assignment,
        request,
        selection,
        solver,
        max_iterations,
    )?;
    let source_export = crate::selective_binding::bind_nebula_problem(audit, branch, refinement.problem.clone())
        .map_err(|error| RefinementError::new(format!("Nebula result binding failed: {error}")))?;
    Ok(FixedPointRefinementReport {
        source_export,
        refinement,
    })
}

/// Run bounded cvc5 refinement against the exact unpadded terminal R1CS and
/// retain its exact map into the padded Spartan relation.
pub fn refine_terminal_with_cvc5(
    audit: &TerminalR1csConstraintAudit,
    request: ExportRequest,
    selection: &Selection,
    solver: &SolverConfig,
    max_iterations: usize,
) -> Result<TerminalRefinementReport, RefinementError> {
    let refinement = refine_with_cvc5(
        audit.source(),
        audit.row_families(),
        request,
        selection,
        solver,
        max_iterations,
    )?;
    let source_export = crate::terminal_binding::bind_terminal_problem(audit, refinement.problem.clone())
        .map_err(|error| RefinementError::new(format!("terminal result binding failed: {error}")))?;
    Ok(TerminalRefinementReport {
        source_export,
        refinement,
    })
}

#[allow(clippy::too_many_arguments)]
fn refine<Export, Replay>(
    mut request: ExportRequest,
    selection: &Selection,
    solver: &SolverConfig,
    max_iterations: usize,
    defaults: &[u64],
    export: Export,
    replay: Replay,
) -> Result<RefinementReport, RefinementError>
where
    Export: Fn(&ExportRequest) -> Result<Problem, RefinementError>,
    Replay: Fn(&FieldModel, &[usize]) -> Result<(Vec<usize>, Vec<usize>), RefinementError>,
{
    if max_iterations == 0 || max_iterations > MAX_REFINEMENT_ITERATIONS {
        return Err(RefinementError::new(format!(
            "max_iterations must be in 1..={MAX_REFINEMENT_ITERATIONS}"
        )));
    }

    for iteration in 1..=max_iterations {
        let problem = export(&request)?;
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

        let model = parse_model_with_defaults(&solver_run.stdout, defaults, &query.model_columns)
            .map_err(|error| RefinementError::new(format!("cvc5 model parse failed: {error}")))?;
        if model.values()[problem.constant_one_column] != 1 {
            return Err(RefinementError::new(
                "cvc5 model does not set the constant-one column to one",
            ));
        }
        let candidate_rows = candidate_source_rows(&problem, selection)?;
        let (violated_candidate_rows, violated_retained_rows) = replay(&model, &candidate_rows)?;
        if violated_candidate_rows.is_empty() {
            return Err(RefinementError::new(
                "cvc5 model does not violate a selected candidate row",
            ));
        }

        if violated_retained_rows.is_empty() {
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
        }
        if let Some(&asserted) = violated_retained_rows
            .iter()
            .find(|row| request.source_rows.binary_search(row).is_ok())
        {
            return Err(RefinementError::new(format!(
                "cvc5 model violates asserted retained source row {asserted}"
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
                Some(violated_retained_rows[0]),
            ));
        }
        for &violated_row in &violated_retained_rows {
            let insertion = request
                .source_rows
                .binary_search(&violated_row)
                .expect_err("the violated row was checked as absent");
            request.source_rows.insert(insertion, violated_row);
        }
    }
    unreachable!("the bounded refinement loop always returns")
}

fn replay_sparse_arm(
    arm: &SparseR1cs,
    model: &FieldModel,
    candidate_rows: &[usize],
) -> Result<(Vec<usize>, Vec<usize>), RefinementError> {
    if model.values().len() != arm.m {
        return Err(RefinementError::new(format!(
            "cvc5 model has {} columns; sparse source requires {}",
            model.values().len(),
            arm.m
        )));
    }
    let assignment = model
        .values()
        .iter()
        .copied()
        .map(F::from_u64)
        .collect::<Vec<_>>();
    let mut az = vec![F::ZERO; arm.n];
    let mut bz = vec![F::ZERO; arm.n];
    let mut cz = vec![F::ZERO; arm.n];
    arm.a.add_mul_into(&assignment, &mut az, arm.n);
    arm.b.add_mul_into(&assignment, &mut bz, arm.n);
    arm.c.add_mul_into(&assignment, &mut cz, arm.n);

    let row_holds = |row: usize| az[row] * bz[row] == cz[row];
    let violated_candidate_rows = candidate_rows
        .iter()
        .copied()
        .filter(|&row| !row_holds(row))
        .collect::<Vec<_>>();
    let violated_retained_rows = (0..arm.n)
        .filter(|row| candidate_rows.binary_search(row).is_err() && !row_holds(*row))
        .take(REFINEMENT_ROW_BATCH)
        .collect::<Vec<_>>();
    Ok((violated_candidate_rows, violated_retained_rows))
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
