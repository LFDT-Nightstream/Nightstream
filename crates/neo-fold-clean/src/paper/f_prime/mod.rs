//! F' — the augmented function from Hypernova §6.3 Construction 2.
//!
//! ## What this module owns
//!
//! - **Native** F' execution (`native.rs`): one F' transition on the
//!   prover and verifier sides, including the NIFS step, state advance,
//!   and x_out hash.
//!
//! ## What lands in PR5
//!
//! - **R1CS encoding** (`encoding.rs`): the R1CS shape that encodes one
//!   F' execution as constraints. Spartan terminal compression proves
//!   satisfiability of this shape.
//! - **R1CS witness synthesis** (`witness.rs`): the satisfying assignment
//!   for `encoding.rs` derived from each step's native execution data.

pub mod native;

// Public surface — paper-named entry points kept stable so call sites
// (mostly `paper::construction2`) don't churn when PR5 adds the R1CS
// siblings.
pub use native::{prove, verify, Error};
