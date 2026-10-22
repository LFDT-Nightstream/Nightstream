//! Native Metal prover backend for SuperNeo.
//!
//! The protocol crates remain authoritative for transcript order, proof
//! assembly, and verification. This crate owns Metal device state and the
//! row evaluator.

pub mod poseidon2;

#[cfg(all(feature = "metal", not(target_vendor = "apple")))]
compile_error!("the `metal` feature requires an Apple target");

use thiserror::Error;

#[cfg(feature = "metal")]
mod rows;
#[cfg(feature = "metal")]
mod session;
#[cfg(feature = "metal")]
pub use rows::MetalRowProver;
#[cfg(feature = "metal")]
pub use session::{MetalGkrLayer, MetalSession};

/// Owner's Metal memory ceiling, chosen so a prover can run on a current
/// iPhone. Device buffer requests stay below it; CPU memory is measured
/// separately.
pub const BUFFER_LIMIT_BYTES: usize = 24_000_000_000;

pub type PoseidonDigest = [u64; 4];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalActivity {
    pub command_buffers: u64,
    pub dispatches: u64,
    pub host_waits: u64,
    pub allocated_bytes: u64,
    pub uploaded_bytes: u64,
    pub downloaded_bytes: u64,
}

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Metal device creation failed")]
    Device,
    #[error("Metal command queue creation failed")]
    Queue,
    #[error("Metal shader library load failed: {0}")]
    Library(String),
    #[error("Metal shader function {0} is missing")]
    Function(&'static str),
    #[error("Metal compute pipeline creation failed: {0}")]
    Pipeline(String),
    #[error("Metal buffer allocation failed for {bytes} bytes")]
    Buffer { bytes: usize },
    #[error(
        "Metal buffer request for {requested} bytes with {allocated} bytes allocated exceeds the {limit}-byte limit"
    )]
    MemoryLimit {
        requested: usize,
        allocated: usize,
        limit: usize,
    },
    #[error("Metal command buffer creation failed")]
    CommandBuffer,
    #[error("Metal compute encoder creation failed")]
    Encoder,
    #[error("Metal command execution failed: {0}")]
    Execution(String),
    #[error("Metal input shape mismatch: {0}")]
    Shape(&'static str),
    #[error("Metal fixed-key commitment input: {0}")]
    Commitment(#[from] neo_ajtai::AjtaiError),
}

#[cfg(feature = "metal")]
pub(crate) fn oracle_error(error: MetalError) -> neo_reductions::PiCcsError {
    match error {
        MetalError::Shape(reason) => neo_reductions::PiCcsError::InvalidInput(reason.into()),
        MetalError::Commitment(error) => neo_reductions::PiCcsError::InvalidInput(error.to_string()),
        error => neo_reductions::PiCcsError::BackendFailure {
            backend: "metal",
            reason: error.to_string(),
        },
    }
}

#[cfg(all(test, feature = "metal"))]
#[path = "../tests/unit/oracle_errors.rs"]
mod oracle_errors;
