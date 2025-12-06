//! Cryptographic gadgets for CCS/R1CS circuits
//!
//! This module contains production-ready implementations of cryptographic
//! primitives expressed as CCS constraints for the public-ρ embedded verifier architecture.
//!
//! ## Available Gadgets
//!
//! - `public_equality`: Enforces equality of public vectors
//! - `commitment_opening`: Homomorphic commitment operations and placeholder Ajtai opening
//!
//! All gadgets are designed to work with the public-ρ EV where Fiat-Shamir challenges
//! are computed off-circuit and passed as public inputs.

pub mod commitment_opening;
pub mod public_equality;
