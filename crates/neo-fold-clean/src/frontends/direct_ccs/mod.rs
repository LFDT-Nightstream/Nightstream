//! Direct-CCS frontend — R1CS in, foldable `CcsInstance` out.
//!
//! The minimal frontend: the user provides an R1CS shape `(A, B, C)` and
//! a satisfying assignment `z = [x | w]` per step. This module:
//!
//! 1. Translates the R1CS into a `paper::relations::Structure` via the
//!    standard `r1cs_to_ccs` embedding (Az ∘ Bz = Cz, polynomial
//!    `f(X0, X1, X2) = X0·X1 - X2`).
//! 2. Validates the assignment satisfies the R1CS row-by-row, returning
//!    a friendly error at the offending row instead of failing later
//!    inside Π_CCS sumcheck.
//! 3. Packs `z` into the Ajtai-shaped `Z` matrix and commits.
//! 4. Returns a `paper::relations::CcsInstance` ready to fold.
//!
//! ## Soundness boundary
//!
//! See the parent module's `Soundness boundary` section. Briefly: this
//! frontend proves that each supplied instance satisfies its R1CS and
//! that the IVC chain folded them correctly. It does **not** prove the
//! instance is the F' encoding of a previous recursive step — that
//! requires the decider's in-circuit F' R1CS (PR5).
//!
//! ## API surface
//!
//! - [`R1cs`] — the user-facing R1CS shape (A, B, C, m_in).
//! - [`R1cs::is_satisfied_by`] — check an assignment without committing.
//! - [`preprocess`] — primary `Preprocessing` entry; caller supplies
//!   protocol params and the Ajtai setup is read from the canonical global
//!   registry for this shape.
//! - [`preprocess_seeded`] — test/demo helper that uses the production
//!   R1CS profile and derives the Ajtai module deterministically from a
//!   caller-supplied seed.
//! - [`build_instance`] — turn a satisfying assignment into a
//!   `CcsInstance`.
//! - [`mixers::ajtai_rlc_mixer`], [`mixers::ajtai_dec_mixer`] — the
//!   standard commitment-action closures Π_RLC and Π_DEC need.

pub mod ajtai;
pub mod instance;
pub mod mixers;
pub mod r1cs;

pub use instance::build_instance;
pub use mixers::{ajtai_dec_mixer, ajtai_rlc_mixer};
pub use r1cs::R1cs;

use thiserror::Error;

use crate::lifecycle::{preprocess as lifecycle_preprocess, preprocess_with_test_log, Preprocessing};
use crate::paper::params::Params;

#[derive(Debug, Error)]
pub enum FrontendError {
    #[error(
        "R1CS shape: A is {a_rows}x{a_cols}, B is {b_rows}x{b_cols}, C is {c_rows}x{c_cols} — all three must agree"
    )]
    ShapeMismatch {
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
        c_rows: usize,
        c_cols: usize,
    },
    #[error("R1CS shape: m_in ({m_in}) > number of variables ({m})")]
    PublicInputTooLarge { m_in: usize, m: usize },
    #[error("R1CS shape must contain at least one constraint and one assignment column")]
    EmptyShape,
    #[error("R1CS unsatisfied at row {row}: (A·z)·(B·z) - (C·z) \u{2260} 0")]
    Unsatisfied { row: usize },
    #[error("witness assignment length {got} \u{2260} R1CS column count {expected}")]
    AssignmentLength { got: usize, expected: usize },
    #[error("direct-CCS preprocessing public-input mismatch: R1CS m_in={r1cs_m_in}, preprocessing public_input_len={prep_m_in:?}")]
    PreprocessingPublicInputMismatch {
        r1cs_m_in: usize,
        prep_m_in: Option<usize>,
    },
    #[error("direct-CCS preprocessing structure does not match the R1CS relation")]
    PreprocessingStructureMismatch,
    #[error(transparent)]
    Params(#[from] neo_params::ParamsError),
    #[error(transparent)]
    Relations(#[from] crate::paper::relations::RelationError),
    #[error(transparent)]
    Lifecycle(#[from] crate::lifecycle::Error),
}

/// Build a `Preprocessing` from a user R1CS shape and caller-supplied
/// protocol params.
///
/// The Ajtai setup is verifier-owned global protocol configuration. This
/// function reads the already-registered canonical setup for the R1CS
/// witness shape; it does not accept a proof/prover-supplied setup.
pub fn preprocess(params: Params, r1cs: &R1cs) -> Result<Preprocessing, FrontendError> {
    r1cs.validate_shape()?;
    let structure = r1cs.to_structure();
    Ok(lifecycle_preprocess(params, structure, Some(r1cs.m_in))?)
}

/// Test/demo helper: use the production R1CS profile, derive the Ajtai
/// module deterministically from `seed`, and build preprocessing.
/// The profile keeps Appendix B.2 core parameters and uses the explicit
/// shape-effective λ from [`crate::config::r1cs_params`].
///
/// Production callers should still install the canonical Ajtai setup out of
/// band and then call [`preprocess`]. This helper is for deterministic tests
/// and examples.
pub fn preprocess_seeded(r1cs: &R1cs, seed: u64) -> Result<Preprocessing, FrontendError> {
    r1cs.validate_shape()?;
    let structure = r1cs.to_structure();
    let params = crate::config::r1cs_params(structure.n, structure.m)?;
    let log = ajtai::setup_seeded(&params, &structure, seed);
    Ok(preprocess_with_test_log(params, structure, log, Some(r1cs.m_in))?)
}
