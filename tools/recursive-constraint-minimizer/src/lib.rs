//! cvc5 query generation for bounded recursive-verifier constraint slices.
//!
//! This crate does not authorize circuit changes. It emits an exact finite-field
//! implication query and records non-authoritative solver evidence for later
//! Rust replay and Lean checking.

mod certificate;
mod evidence;
mod model;
mod problem;
mod query;
mod solver;

pub use certificate::{
    derive_scalar_certificate, validate_scalar_certificate, CertificateError, ScalarCertificate, ScalarRowCertificate,
    ScalarSupport, SCALAR_CERTIFICATE_SCHEMA,
};
pub use evidence::{Evidence, EVIDENCE_SCHEMA};
pub use model::{parse_model, parse_model_with_defaults, row_is_satisfied, FieldModel, ModelError};
pub use problem::{
    LinearCombination, Problem, ProblemError, Row, Scope, Selection, Source, Term, GOLDILOCKS_MODULUS, PROBLEM_SCHEMA,
};
pub use query::{render_query, Query, RowReference};
pub use solver::{run_cvc5, Conclusion, SolverConfig, SolverError, SolverMode, SolverRun, SolverStatus};
