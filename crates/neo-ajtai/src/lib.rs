//! Ajtai matrix commitment (Neo).
//!
//! MUST: Ajtai L: F_q^{d×m} → C via c = cf(M·cf^{-1}(Z)) (Def. 9, 11, 12, 13). S-homomorphic,
//! (d,m,B)-binding, with pay-per-bit embedding (Sec. 3.2–3.3). Provides decomp_b/split_b,
//! verified openings, and range checks used by Π_DEC and Π_RLC pipelines.
//!
//! SHOULD: Docs for Goldilocks parameters (Sec. 6.2) + estimator pointers (App. B.12).

mod types;
pub mod util;
mod decomp;
mod commit;

pub use types::{Commitment, PP};
pub use decomp::{decomp_b, split_b, assert_range_b, DecompStyle};
pub use commit::{setup, commit, verify_open, verify_split_open, s_mul, s_lincomb};
