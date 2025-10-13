//! Error types for NIVC operations
//!
//! Currently using anyhow::Error for simplicity.
//! Can be expanded to use thiserror if more structured error handling is needed.

/// NIVC-specific error type (currently just anyhow::Error)
pub type NivcError = anyhow::Error;
