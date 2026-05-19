//! Owns CE-claim circuit variables, allocation surfaces, Poseidon2 digests, and equality checks.
//!
//! Callers should use this module as the single CE-claim boundary for SuperNeo
//! circuits. It does not own Π_CCS, Π_RLC, or Π_DEC protocol logic; those modules
//! consume the allocation, digest, and equality surfaces defined here.

mod alloc;
mod digest;
mod encoding;
mod equality;
mod fields;
mod types;

use crate::spartan_backend::SpartanF;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError, Variable};
use ff::Field;
use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField64;

use super::k_field::{alloc_k, enforce_k_eq, KNum, KNumVar};
use super::transcript::hash_field_linear_combinations_raw;

pub use alloc::*;
pub use digest::*;
pub use equality::*;
pub use fields::packed_bytes_field_values;
pub use types::*;
