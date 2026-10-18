use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AjtaiError {
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Range assertion failed: |{value}| >= {bound}")]
    RangeViolation { value: i128, bound: u32 },

    #[error("Array size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("Empty input not allowed")]
    EmptyInput,

    #[error("Commitment verification failed")]
    VerificationFailed,

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type AjtaiResult<T> = Result<T, AjtaiError>;
