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

mod commit;
mod decomp;
mod error;
pub mod prg;
mod types;
pub mod util;

pub mod s_module;

pub use commit::{
    commit, commit_masked_ct, commit_precomp_ct, s_lincomb, s_mul, setup, try_commit, verify_open, verify_split_open,
};
pub use decomp::{assert_range_b, decomp_b, split_b, DecompStyle};
pub use error::{AjtaiError, AjtaiResult};
pub use types::{Commitment, PP};

#[cfg(feature = "testing")]
pub use commit::rot_step;

pub use s_module::{
    get_global_pp, get_global_pp_for_dims, get_global_pp_for_z_len, has_global_pp_for_dims, set_global_pp, AjtaiSModule,
};
