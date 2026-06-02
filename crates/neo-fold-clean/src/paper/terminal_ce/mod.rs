//! Compact terminal-CE proof boundary.
//!
//! Owns only the backend-neutral public statement shape and fail-closed circuit
//! entrypoint for a future compact terminal CE proof. The proof must establish
//! the full SuperNeo terminal CE relation for the NIFS-produced children:
//! commitment opening, public-input projection, low norm, `y_ring = M*z(r)`,
//! and the implementation-level `ct = lane0(y_ring)` invariant.
//!
//! This module does not contain an accepting proof verifier. A matching public
//! digest is binding material, not authority. The current production decider
//! still uses `paper::decider_ce_relation` as the sound direct verifier.

pub mod circuit;
pub mod merkle;
mod proof;
mod public;

pub use proof::{TerminalCeProof, TerminalCeVerifyError};
pub use public::{TerminalCePublic, TerminalCePublicError};
