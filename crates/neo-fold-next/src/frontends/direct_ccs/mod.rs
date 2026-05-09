//! Owns the generic non-VM direct CCS/R1CS IVC compression path.
//!
//! Start here for the direct frontend public API. Implementation ownership is
//! split below by adapter, native state, F', terminal proof, and recursive proof.

mod adapter;
mod api;
mod f_prime;
mod public_image;
mod recursive;
mod snark;
mod state;
mod terminal;
mod verify;

pub use api::*;
