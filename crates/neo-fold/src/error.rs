//! Error types for the Neo folding protocol

use thiserror::Error;

/// Main folding protocol error
#[derive(Debug, Error)]
pub enum FoldingError {
    /// Invalid input to the folding pipeline
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Π_CCS protocol error
    #[error("Π_CCS error: {0}")]
    PiCcs(#[from] PiCcsError),
    
    /// Π_RLC protocol error  
    #[error("Π_RLC error: {0}")]
    PiRlc(#[from] PiRlcError),
    
    /// Π_DEC protocol error
    #[error("Π_DEC error: {0}")]
    PiDec(#[from] PiDecError),
}

/// Π_CCS sum-check protocol error
#[derive(Debug, Error)]
pub enum PiCcsError {
    /// Invalid structure or parameters
    #[error("Invalid structure: {0}")]
    InvalidStructure(String),
    
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Transcript error
    #[error("Transcript error: {0}")]
    TranscriptError(String),
    
    /// Sum-check protocol error
    #[error("Sum-check error: {0}")]
    SumcheckError(String),
    
    /// Extension policy validation error
    #[error("Extension policy failed: {0}")]
    ExtensionPolicyFailed(String),
}

/// Π_RLC random linear combination error
#[derive(Debug, Error)]
pub enum PiRlcError {
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Strong sampling failed
    #[error("Sampling failed: {0}")]
    SamplingFailed(String),
    
    /// Guard constraint violation
    #[error("Guard violation: {0}")]
    GuardViolation(String),
    
    /// S-action computation error
    #[error("S-action error: {0}")]
    SActionError(String),
}

/// Π_DEC verified split opening error
#[derive(Debug, Error)]
pub enum PiDecError {
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Commitment scheme error
    #[error("Commitment error: {0}")]
    CommitmentError(String),
    
    /// Range constraint violation
    #[error("Range violation: {0}")]
    RangeViolation(String),
    
    /// Opening verification failed
    #[error("Opening verification failed: {0}")]
    OpeningFailed(String),
}
