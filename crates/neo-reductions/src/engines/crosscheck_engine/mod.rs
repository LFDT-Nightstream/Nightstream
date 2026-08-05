//! Exact optimized-versus-PaperExact proof comparison.

mod crosscheck;
pub use crosscheck::{
    crosscheck_prove, crosscheck_prove_with_binding, crosscheck_verify, crosscheck_verify_with_binding,
    CrossCheckEngine,
};
