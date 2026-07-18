//! This module provides implementations of polynomial commitment schemes (PCS).

// helper code for polynomial commitment schemes
#[cfg(not(target_arch = "wasm32"))]
pub mod ipa;

// implementations of polynomial commitment schemes
#[cfg(not(target_arch = "wasm32"))]
pub mod hyrax_pc;
pub mod merkle_mle_pc;

// backend interface for hash-mle
mod hash_mle_backend;

// p3 poseidon2 implementation
#[cfg(feature = "p3_backend")]
pub mod merkle_mle_pc_p3;

// production-security WHIR implementation over SuperNeo's Goldilocks profile
#[cfg(feature = "p3_backend")]
pub mod whir_pc;
