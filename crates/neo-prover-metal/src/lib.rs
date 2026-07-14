//! Native Metal prover substrate for Apple GPUs.
//!
//! Owns the Metal device boundary and byte-exact mirrors of proof-critical
//! kernels. It does not own protocol semantics: `neo-fold-clean`, `neo-ccs`,
//! and `neo-reductions` remain canonical. macOS and iOS compile the same MSL
//! sources into SDK-specific Metal libraries.

pub mod poseidon2;

#[cfg(all(feature = "metal", not(target_vendor = "apple")))]
compile_error!("the `metal` feature requires an Apple target");

#[cfg(all(feature = "metal", target_vendor = "apple"))]
mod device;

#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub use device::{MetalDevice, MetalError, MetalTranscriptOp, MetalTranscriptOutput};
