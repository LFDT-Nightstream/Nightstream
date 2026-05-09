//! Owns the direct-CCS terminal F' circuit and committed proof.
//!
//! This layer builds and verifies the terminal Construction-2 step that binds
//! the latest SuperNeo carry/output surface to the compact public image. It
//! does not own frontend lowering or live IVC state updates.

pub(crate) mod ce_bundle;
pub(crate) mod circuit;
pub(crate) mod committed;
pub(crate) mod construction2_fold;
pub(crate) mod final_ce;
pub(crate) mod gadgets;
pub(crate) mod initial_carry;
pub(crate) mod measure;
mod prove;
pub(crate) mod public_io;
mod verify;

pub use committed::DirectCcsTerminalCommittedConstraintBreakdown;
pub(crate) use prove::prove_direct_ccs_f_prime_circuit;
pub use verify::verify_direct_ccs_terminal_snark_against_state;
