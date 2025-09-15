//! Error types for neo-ajtai commitment scheme

use thiserror::Error;

/// Errors that can occur during Ajtai commitment operations
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AjtaiError {
    /// Invalid dimensions or parameters
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
    
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Range constraint violation
    #[error("Range assertion failed: |{value}| >= {bound}")]
    RangeViolation { value: i128, bound: u32 },
    
    /// Mismatched array sizes
    #[error("Array size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },
    
    /// Empty input where non-empty expected
    #[error("Empty input not allowed")]
    EmptyInput,
    
    /// Commitment verification failed
    #[error("Commitment verification failed")]
    VerificationFailed,
    
    /// Internal error (e.g., lock poisoning, system failures)
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for Ajtai operations
pub type AjtaiResult<T> = Result<T, AjtaiError>;
