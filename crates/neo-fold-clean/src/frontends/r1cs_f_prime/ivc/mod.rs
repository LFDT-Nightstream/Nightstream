//! Generic HyperNova-style IVC over an arbitrary verifier-owned R1CS app.
//!
//! This is the authoritative recursive path. Each fresh relation instance
//! contains the app rows and, after the base step, the constrained SuperNeo
//! `NIFS.V` execution that authenticates the preceding fold. The terminal
//! verifier can therefore check only the running accumulator and latest fresh
//! instance, exactly as HyperNova Construction 2 requires.

mod chain;
mod compilation_audit;
mod pi_ccs_output_digest_audit;
mod relation;
pub(crate) mod shape;

pub use chain::{
    validate_raw_old_block_execution, R1csIvc, R1csIvcCombinedNcExecutionAudit, R1csIvcCombinedNcRoundAudit,
    R1csIvcCombinedNcTerminalAudit, R1csIvcFullZChildAudit, R1csIvcGeneratedKBindingAudit, R1csIvcGeneratedKSlot,
    R1csIvcPiDecCanonicalXCoordinateAudit, R1csIvcPiDecPaperShapeExecutionAudit, R1csIvcPiDecPaperShapeProfile,
    R1csIvcPiDecPaperTraceColumnAudit, R1csIvcPiDecPaperXOwner, R1csIvcPiDecPaperXPinAudit,
    R1csIvcPostPiDecExecutionAudit, R1csIvcPreprocessing, R1csIvcPublicWriteAudit, R1csIvcPublicWriteSource,
    R1csIvcRawAssignmentAuthority, R1csIvcRawChildAssignmentAudit, R1csIvcRawOldBlockChildAudit,
    R1csIvcRawOldBlockExecutionAudit, R1csIvcRawOldBlockFieldDecoding, R1csIvcRawOldBlockProfile,
    R1csIvcSelectorWriteAudit, PI_DEC_PAPER_ACTIVE_X_COLUMNS, PI_DEC_PAPER_CANONICALITY_ROWS_PER_COORDINATE,
    PI_DEC_PAPER_CHILD_COUNT, PI_DEC_PAPER_EVALUATION_ARITY, PI_DEC_PAPER_PUBLIC_COORDINATES,
    RAW_OLD_BLOCK_ACTIVE_LANES, RAW_OLD_BLOCK_CHILD_COUNT, RAW_OLD_BLOCK_PADDED_LANES,
    RAW_OLD_BLOCK_ZERO_PADDING_LANES,
};
pub use compilation_audit::{
    ArmShapeAudit, FixedPointRoundAudit, R1csIvcCompilationAudit, R1csIvcFixedPointShapeAudit, RelationHeaderAudit,
};
pub use pi_ccs_output_digest_audit::{
    CanonicalOpeningAudit, CanonicalOpeningPlacement, PiCcsOutputDigestAudit, PiCcsOutputDigestProfileAudit,
    PiCcsOutputEnvelopePrefixAudit, PiCcsOutputSisPhysicalAudit, PiCcsOutputYZcolProducerEntryAudit,
    PiCcsOutputYZcolProjectionAudit, PiCcsOutputYZcolProjectionInputAudit, PiRlcYZcolKMulAudit,
    PiRlcYZcolLinearCombinationAudit, PiRlcYZcolPolynomialEvaluationAudit, PiRlcYZcolProductFactorAudit,
    PiRlcYZcolProductIdentityAudit, PiRlcYZcolProjectionIdentityAudit, PiRlcYZcolProjectionLeafRowMappingAudit,
    PiRlcYZcolProjectionLimbAudit, PiRlcYZcolProjectionLoweredFragmentAudit, PiRlcYZcolProjectionLoweringDisposition,
    PiRlcYZcolProjectionProfileAudit, PiRlcYZcolProjectionRowAudit, PiRlcYZcolProjectionRowMappingAudit,
    PiRlcYZcolProjectionSharedAudit, SeededPhi81BlockAudit,
};
pub use relation::{
    R1csIvcBlockLaneNcSelectiveRowsAudit, R1csIvcBlockLaneNcSourceRowAudit, R1csIvcBranch,
    R1csIvcPiDecCanonicalXSelectiveRowsAudit, R1csIvcPiDecSelectiveRowsAudit, R1csIvcPiDecSourceRowAudit,
    R1csIvcPiDecSourceRowsAudit, R1csIvcRawRunningAssignmentAudit, R1csIvcRawRunningEncodingAudit, R1csIvcRelation,
    R1csIvcYZcolSelectiveRowsAudit, R1CS_IVC_COMMITTED_COORDINATE_BUDGET,
};

use thiserror::Error;

use super::R1csCompilerError;
use crate::frontends::direct_ccs::FrontendError;
use crate::frontends::r1cs_f_prime::{FieldR1csLoweringError, LowNormR1csError};
use crate::lifecycle;
use crate::paper::f_prime::r1cs;
use crate::paper::relations::RelationError;

#[derive(Debug, Error)]
pub enum R1csIvcError {
    #[error(transparent)]
    Plan(#[from] super::Error),
    #[error(transparent)]
    App(#[from] FrontendError),
    #[error(transparent)]
    Lowering(#[from] FieldR1csLoweringError),
    #[error(transparent)]
    LowNorm(#[from] LowNormR1csError),
    #[error(transparent)]
    Composition(#[from] r1cs::Error),
    #[error(transparent)]
    Compiler(#[from] R1csCompilerError),
    #[error(transparent)]
    Lifecycle(#[from] lifecycle::Error),
    #[error(transparent)]
    Instance(#[from] RelationError),
    #[error("R1CS IVC fixed point did not stabilize after {rounds} rounds (last input {input_rows}x{input_columns}, output {output_rows}x{output_columns})")]
    NoFixedPoint {
        rounds: usize,
        input_rows: usize,
        input_columns: usize,
        output_rows: usize,
        output_columns: usize,
    },
    #[error("R1CS IVC relation needs {required} committed coordinates, exceeding the {budget} coordinate budget")]
    BudgetExceeded { required: usize, budget: usize },
    #[error("invalid stabilized PiCCS output-digest audit: {detail}")]
    InvalidPiCcsOutputDigestAudit { detail: String },
    #[error("R1CS IVC branch {branch:?} synthesized {rows}x{columns} with {public_columns} public columns; expected {expected_rows}x{expected_columns} with {expected_public_columns} public columns")]
    ArmShapeMismatch {
        branch: R1csIvcBranch,
        rows: usize,
        columns: usize,
        public_columns: usize,
        expected_rows: usize,
        expected_columns: usize,
        expected_public_columns: usize,
    },
    #[error("R1CS IVC preprocessing does not match the compiled relation")]
    PreprocessingMismatch,
    #[error("R1CS IVC branch {branch:?} encoded an unsatisfied CCS row at index {row} ({owner})")]
    UnsatisfiedEncodedRelation {
        branch: R1csIvcBranch,
        row: usize,
        owner: String,
    },
    #[error("R1CS IVC expected an active lifecycle state")]
    ExpectedActiveState,
    #[error("R1CS IVC expected a recursive fold proof")]
    ExpectedRecursiveFold,
    #[error("R1CS IVC chain has no appended steps")]
    EmptyChain,
    #[error("R1CS IVC semantic input does not match the carried state")]
    SemanticInputMismatch,
}
