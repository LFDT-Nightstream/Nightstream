//! Exact SMT-LIB encoding of one R1CS row or row-family implication query.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::Serialize;

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
    })
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
