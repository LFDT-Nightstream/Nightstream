//! Error types for the Spartan bridge

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpartanBridgeError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Circuit synthesis failed: {0}")]
    SynthesisError(String),

    #[error("Spartan proving failed: {0}")]
    ProvingError(String),

    #[error("Spartan verification failed: {0}")]
    VerificationError(String),

    #[error("Neo folding error: {0}")]
    NeoError(#[from] neo_fold::PiCcsError),

    #[error("Field conversion error: {0}")]
    FieldError(String),

    #[error("Bellpepper synthesis error: {0:?}")]
    BellpepperError(#[from] bellpepper_core::SynthesisError),
}

pub type Result<T> = std::result::Result<T, SpartanBridgeError>;
