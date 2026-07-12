//! Device kernels, one module per protocol phase.
//!
//! `goldilocks` holds plain-Rust field/ring arithmetic shared by all kernel
//! modules; it compiles for both host (unit-testable, parity-checkable) and
//! device (reached from `#[cuda_module]` bodies).

pub mod goldilocks;

#[cfg(feature = "cuda")]
pub mod ajtai;
#[cfg(feature = "cuda")]
pub mod csr;
pub mod pi_ccs_fe;
#[cfg(feature = "cuda")]
pub mod pi_ccs_nc;
#[cfg(feature = "cuda")]
pub mod pi_ccs_output;
#[cfg(feature = "cuda")]
pub mod pi_ccs_tail;
#[cfg(feature = "cuda")]
pub mod pi_dec;
#[cfg(feature = "cuda")]
pub mod pi_rlc;
#[cfg(feature = "cuda")]
pub mod poseidon2;
#[cfg(feature = "cuda")]
pub mod probe;
#[cfg(feature = "cuda")]
pub mod sis;
#[cfg(feature = "cuda")]
pub mod sumcheck_common;
