//! Shared package-input and Stage 2 shape helpers.
//!
//! This compatibility namespace does not define or construct a Stage 1
//! relation. Production Stage 1 uses `r1cs_f_prime::production` and the
//! verifier-owned Lean package.

mod package_v1_1;
pub(crate) mod shape;

pub use package_v1_1::{
    encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage,
    PiCcsV1_1PackageBridgeError, PiCcsV1_1ProofInputs,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum R1csIvcError {
    #[error("R1CS F' packed public-input variable z[{index}] is not Boolean (got {value:?})")]
    PackedPublicInputNotBit { index: usize, value: neo_math::F },
}
