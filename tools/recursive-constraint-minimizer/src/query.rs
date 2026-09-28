//! Exact SMT-LIB encoding of one R1CS row or row-family implication query.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::problem::{LinearCombination, Problem, ProblemError, Row, Selection};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RowReference {
    pub problem_index: usize,
    pub source_index: usize,
    pub id: String,
    pub family: String,
    pub assertion: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    pub smt2: String,
    /// Strictly ordered source columns declared in this bounded query.
    pub model_columns: Vec<usize>,
    pub retained_rows: Vec<RowReference>,
    pub removed_rows: Vec<RowReference>,
    pub target_rows: Vec<RowReference>,
}

/// One independently defined row in the complete typed target relation.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct TypedTargetRow {
    pub id: String,
    pub a: LinearCombination,
    pub b: LinearCombination,
    pub c: LinearCombination,
}

/// Complete typed relation that a strict family-removal query must violate.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct TypedTarget {
    pub id: String,
    pub column_count: usize,
    pub rows: Vec<TypedTargetRow>,
}

pub fn render_query(problem: &Problem, selection: &Selection) -> Result<Query, ProblemError> {
    let partition = problem.partition(selection)?;
    let mut smt2 = String::new();
    writeln!(smt2, "(set-logic QF_FF)").unwrap();
    writeln!(smt2, "(set-option :produce-models true)").unwrap();
    writeln!(smt2, "(set-option :produce-unsat-cores true)").unwrap();
    writeln!(smt2, "(define-sort F () (_ FiniteField {}))", problem.field_modulus).unwrap();
    let mut model_columns = BTreeSet::from([problem.constant_one_column]);
    for row in &problem.rows {
        for term in row.a.iter().chain(&row.b).chain(&row.c) {
            model_columns.insert(term.column);
        }
    }
    let model_columns = model_columns.into_iter().collect::<Vec<_>>();
    for &column in &model_columns {
        writeln!(smt2, "(declare-const x_{column} F)").unwrap();
    }
    writeln!(
        smt2,
        "(assert (! (= x_{} (as ff1 F)) :named constant_one))",
        problem.constant_one_column
    )
    .unwrap();

    let retained_rows = partition
        .retained
        .iter()
        .map(|&(index, row)| {
            let assertion = format!("keep_{index}");
            writeln!(smt2, "(assert (! {} :named {assertion}))", row_equality(row)).unwrap();
            row_reference(index, row, assertion)
        })
        .collect::<Vec<_>>();

    let violations = partition
        .removed
        .iter()
        .map(|(_, row)| format!("(not {})", row_equality(row)))
        .collect::<Vec<_>>();
    let violation = if violations.len() == 1 {
        violations[0].clone()
    } else {
        format!("(or {})", violations.join(" "))
    };
    writeln!(smt2, "(assert (! {violation} :named candidate_violation))").unwrap();
    writeln!(smt2, "(check-sat)").unwrap();

    let removed_rows = partition
        .removed
        .iter()
        .map(|&(index, row)| row_reference(index, row, "candidate_violation".to_owned()))
        .collect();
    Ok(Query {
        smt2,
        model_columns,
        retained_rows,
        removed_rows,
        target_rows: Vec::new(),
    })
}

/// Render the strict lifecycle-family query required for removal review.
///
/// The source problem must contain every exact source row and every source
/// family. The selected family is absent, every other row is asserted, the
/// constant-one column is one, and the independent complete typed target is
/// violated.
pub fn render_complete_typed_query(
    problem: &Problem,
    selection: &Selection,
    target: &TypedTarget,
) -> Result<Query, ProblemError> {
    let partition = problem.partition(selection)?;
    validate_complete_source(problem)?;
    validate_typed_target(problem, target)?;

    let model_columns = (0..problem.column_count).collect::<Vec<_>>();

    let mut smt2 = String::new();
    writeln!(smt2, "(set-logic QF_FF)").unwrap();
    writeln!(smt2, "(set-option :produce-models true)").unwrap();
    writeln!(smt2, "(set-option :produce-unsat-cores true)").unwrap();
    writeln!(smt2, "(define-sort F () (_ FiniteField {}))", problem.field_modulus).unwrap();
    for &column in &model_columns {
        writeln!(smt2, "(declare-const x_{column} F)").unwrap();
    }
    writeln!(
        smt2,
        "(assert (! (= x_{} (as ff1 F)) :named constant_one))",
        problem.constant_one_column
    )
    .unwrap();

    let retained_rows = partition
        .retained
        .iter()
        .map(|&(index, row)| {
            let assertion = format!("keep_{index}");
            writeln!(smt2, "(assert (! {} :named {assertion}))", row_equality(row)).unwrap();
            row_reference(index, row, assertion)
        })
        .collect::<Vec<_>>();
    let removed_rows = partition
        .removed
        .iter()
        .map(|&(index, row)| row_reference(index, row, "family_absent".to_owned()))
        .collect::<Vec<_>>();

    let target_violations = target
        .rows
        .iter()
        .map(|row| format!("(not {})", target_row_equality(row)))
        .collect::<Vec<_>>();
    let target_violation = if target_violations.len() == 1 {
        target_violations[0].clone()
    } else {
        format!("(or {})", target_violations.join(" "))
    };
    writeln!(smt2, "(assert (! {target_violation} :named typed_target_violation))").unwrap();
    writeln!(smt2, "(check-sat)").unwrap();

    let target_rows = target
        .rows
        .iter()
        .enumerate()
        .map(|(index, row)| RowReference {
            problem_index: index,
            source_index: index,
            id: row.id.clone(),
            family: target.id.clone(),
            assertion: "typed_target_violation".to_owned(),
        })
        .collect();
    Ok(Query {
        smt2,
        model_columns,
        retained_rows,
        removed_rows,
        target_rows,
    })
}

fn validate_complete_source(problem: &Problem) -> Result<(), ProblemError> {
    if problem.rows.len() != problem.source.total_rows
        || problem
            .rows
            .iter()
            .enumerate()
            .any(|(index, row)| row.source_index != index)
    {
        return Err(ProblemError::new(
            "strict typed query requires every source row in source order",
        ));
    }
    let row_families = problem
        .rows
        .iter()
        .map(|row| row.family.as_str())
        .collect::<BTreeSet<_>>();
    let complete_families = problem
        .complete_families
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if row_families != complete_families {
        return Err(ProblemError::new(
            "strict typed query requires the exact complete source-family ledger",
        ));
    }
    Ok(())
}

fn validate_typed_target(problem: &Problem, target: &TypedTarget) -> Result<(), ProblemError> {
    if target.id.trim().is_empty() || target.column_count != problem.column_count || target.rows.is_empty() {
        return Err(ProblemError::new(
            "typed target must have an identity, the source column width, and at least one row",
        ));
    }
    let mut ids = BTreeSet::new();
    for row in &target.rows {
        if row.id.trim().is_empty() || !ids.insert(row.id.as_str()) {
            return Err(ProblemError::new(
                "typed target row identities must be nonempty and unique",
            ));
        }
        for terms in [&row.a, &row.b, &row.c] {
            validate_target_terms(problem, terms)?;
        }
    }
    Ok(())
}

fn validate_target_terms(problem: &Problem, terms: &LinearCombination) -> Result<(), ProblemError> {
    let modulus = problem
        .field_modulus
        .parse::<u64>()
        .expect("validated Goldilocks modulus fits in u64");
    let mut prior = None;
    for term in terms {
        if term.column >= problem.column_count || prior.is_some_and(|column| term.column <= column) {
            return Err(ProblemError::new(
                "typed target terms must use strictly ordered in-range columns",
            ));
        }
        let coefficient = term
            .coefficient
            .parse::<u64>()
            .map_err(|_| ProblemError::new("typed target coefficient is not a canonical decimal residue"))?;
        if coefficient == 0 || coefficient >= modulus || term.coefficient != coefficient.to_string() {
            return Err(ProblemError::new(
                "typed target coefficient is not a nonzero canonical Goldilocks residue",
            ));
        }
        prior = Some(term.column);
    }
    Ok(())
}

fn row_reference(index: usize, row: &Row, assertion: String) -> RowReference {
    RowReference {
        problem_index: index,
        source_index: row.source_index,
        id: row.id.clone(),
        family: row.family.clone(),
        assertion,
    }
}

fn row_equality(row: &Row) -> String {
    format!(
        "(= (ff.mul {} {}) {})",
        linear_combination(&row.a),
        linear_combination(&row.b),
        linear_combination(&row.c)
    )
}

fn target_row_equality(row: &TypedTargetRow) -> String {
    format!(
        "(= (ff.mul {} {}) {})",
        linear_combination(&row.a),
        linear_combination(&row.b),
        linear_combination(&row.c)
    )
}

fn linear_combination(terms: &LinearCombination) -> String {
    if terms.is_empty() {
        return "(as ff0 F)".to_owned();
    }
    let rendered = terms
        .iter()
        .map(|term| {
            if term.coefficient == "1" {
                format!("x_{}", term.column)
            } else {
                format!("(ff.mul (as ff{} F) x_{})", term.coefficient, term.column)
            }
        })
        .collect::<Vec<_>>();
    if rendered.len() == 1 {
        rendered[0].clone()
    } else {
        format!("(ff.add {})", rendered.join(" "))
    }
}
