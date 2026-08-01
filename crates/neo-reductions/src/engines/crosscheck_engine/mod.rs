//! Exact optimized-versus-PaperExact proof comparison.

mod crosscheck;
pub use crosscheck::{crosscheck_prove, crosscheck_verify, CrossCheckEngine, CrosscheckCfg};
