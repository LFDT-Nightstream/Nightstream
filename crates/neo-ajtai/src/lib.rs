//! Ajtai matrix commitment (Neo).
//!
//! MUST: Ajtai L: F_q^{d×m} → C via c = cf(M·cf^{-1}(Z)) (Def. 9, 11, 12, 13). S-homomorphic,
//! (d,m,B)-binding, with pay-per-bit embedding (Sec. 3.2–3.3). Provides decomp_b/split_b,
//! verified openings, and range checks used by Π_DEC and Π_RLC pipelines.
//!
//! SHOULD: Docs for Goldilocks parameters (Sec. 6.2) + estimator pointers (App. B.12).
//!
//! ## API Guard: verify_linear
//!
//! This crate intentionally does NOT export `verify_linear`. Linear relation verification
//! must be performed at the folding layer via `neo_fold::verify_linear` (Π_RLC).
//!
//! ```compile_fail
//! // This must fail to compile - verify_linear is not provided by neo-ajtai
//! use neo_ajtai::verify_linear;
//! ```

#![forbid(unsafe_code)]

mod error;
mod types;
pub mod util;
mod decomp;
mod commit;

pub mod s_module;

pub use error::{AjtaiError, AjtaiResult};
pub use types::{Commitment, PP};
pub use decomp::{decomp_b, split_b, assert_range_b, DecompStyle};
pub use commit::{setup, commit, try_commit, verify_open, verify_split_open, s_mul, s_lincomb, commit_masked_ct, commit_precomp_ct};

// Testing-only exports (open_linear for differential testing only)
#[cfg(any(test, feature = "testing"))]
pub use commit::{LinearOpeningProof, open_linear, rot_step};

// NOTE: verify_linear is intentionally NOT PROVIDED in the Ajtai commitment layer.
// Generic linear openings y = Z·v cannot be verified from a single Ajtai commitment.
// Use neo_fold::verify_linear (Π_RLC) for linear relation verification instead.

// Test-only differential testing function
#[cfg(any(test, feature = "testing"))]
pub use commit::commit_spec;

pub use s_module::{AjtaiSModule, set_global_pp, get_global_pp};
