//! Public entry surface for the direct CCS/R1CS frontend.
//!
//! The API is grouped by audience: program construction/lowering, F' authority
//! helpers, terminal compression, and recursive compression.

mod f_prime;
mod program;
mod recursive;
mod terminal;

pub use f_prime::*;
pub use program::*;
pub use recursive::*;
pub use terminal::*;
