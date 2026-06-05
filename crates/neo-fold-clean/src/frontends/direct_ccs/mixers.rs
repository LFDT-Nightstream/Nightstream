//! Standard Ajtai-action commitment mixers for direct-CCS users.
//!
//! Both Π_RLC and Π_DEC need a closure that combines commitments under
//! the protocol's commitment-scheme action. For the audited Ajtai
//! homomorphism, the actions are:
//!
//! - **Π_RLC**: `Σ ρ_i · c_i` where ρ_i is the *polynomial* a rotation
//!   matrix represents (not a scalar). Use `cf_inv` to recover the
//!   polynomial coefficients from the matrix's first column, then
//!   `s_mul_add` for the polynomial-times-commitment multiplication.
//! - **Π_DEC**: `Σ b^{i-1} · c_i` for `i = 1..k`. The base `b` is a
//!   small integer, so this is just per-lane scalar multiplication via
//!   `scale_commitment_add_inplace`.
//!
//! These functions match `RlcMixer`/`DecMixer` `fn(...)` types exactly.

pub use crate::paper::relations::{ajtai_dec_mixer, ajtai_rlc_mixer};
