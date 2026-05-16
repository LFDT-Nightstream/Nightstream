//! Public entry surface for the RV32IM frontend.
//!
//! The submodules below group the flattened `rv32im::*` API by audience:
//! program construction, proof production/verification, recursion/F',
//! stage summaries, and diagnostics.

pub mod diagnostics;
pub mod program;
pub mod proof;
pub mod recursion;
pub mod stages;

pub use diagnostics::*;
pub use program::*;
pub use proof::*;
pub use recursion::*;
pub use stages::*;
