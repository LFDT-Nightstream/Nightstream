//! Serializable record of the exact query and raw cvc5 response.

use serde::Serialize;

use crate::{Conclusion, Problem, Query, Selection, SolverConfig, SolverRun, SolverStatus, Source};

pub const EVIDENCE_SCHEMA: &str = "nightstream/cvc5-redundancy-evidence/v3";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SolverRecord {
    pub executable: String,
    pub mode: String,
    pub timeout_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Evidence {
    pub schema: String,
    pub authority: String,
    pub source: Source,
    pub field_modulus: String,
    pub column_count: usize,
    pub public_input_count: usize,
    pub complete_families: Vec<String>,
    pub selection: Selection,
    pub retained_rows: Vec<crate::RowReference>,
    pub removed_rows: Vec<crate::RowReference>,
    pub solver: SolverRecord,
    pub solver_status: SolverStatus,
    pub conclusion: Conclusion,
    pub elapsed_ms: u64,
    pub exit_code: Option<i32>,
    pub query_smt2: String,
    pub solver_stdout: String,
    pub solver_stderr: String,
}

impl Evidence {
    pub fn new(problem: &Problem, selection: Selection, query: Query, config: &SolverConfig, run: SolverRun) -> Self {
        Self {
            schema: EVIDENCE_SCHEMA.to_owned(),
            authority: "non_authoritative_solver_evidence".to_owned(),
            source: problem.source.clone(),
            field_modulus: problem.field_modulus.clone(),
            column_count: problem.column_count,
            public_input_count: problem.public_input_count,
            complete_families: problem.complete_families.clone(),
            selection,
            retained_rows: query.retained_rows,
            removed_rows: query.removed_rows,
            solver: SolverRecord {
                executable: config.executable.display().to_string(),
                mode: config.mode.as_str().to_owned(),
                timeout_ms: config.timeout_ms,
            },
            solver_status: run.status,
            conclusion: run.conclusion,
            elapsed_ms: run.elapsed_ms,
            exit_code: run.exit_code,
            query_smt2: query.smt2,
            solver_stdout: run.stdout,
            solver_stderr: run.stderr,
        }
    }
}
