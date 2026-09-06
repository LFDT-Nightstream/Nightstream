//! Error types for the neo-fold crate

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PiCcsError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Sumcheck error: {0}")]
    SumcheckError(String),

    #[error("Extension policy failed: {0}")]
    ExtensionPolicyFailed(String),

    #[error("Transcript error: {0}")]
    TranscriptError(String),

    #[error("prover backend `{backend}` failed: {reason}")]
    BackendFailure {
        backend: &'static str,
        reason: String,
    },

    #[error("Protocol error: {0}")]
    ProtocolError(String),
}
