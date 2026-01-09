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
pub mod opening_weights;
pub mod pp_id;
pub mod prg;
mod types;
pub mod util;

pub mod s_module;

pub use commit::{
    commit, commit_masked_ct, commit_precomp_ct, commit_row_major, precompute_rot_columns, s_lincomb, s_mul, setup,
    setup_par, try_commit, try_commit_row_major, verify_open,
    verify_split_open,
};
pub use decomp::{assert_range_b, decomp_b, decomp_b_row_major, split_b, DecompStyle};
pub use error::{AjtaiError, AjtaiResult};
pub use types::{Commitment, PP};

pub use opening_weights::compute_opening_weights_for_u_seeded;
pub use pp_id::compute_pp_id_digest_v1;

#[cfg(feature = "testing")]
pub use commit::rot_step;

pub use s_module::{
    get_global_pp, get_global_pp_for_dims, get_global_pp_for_z_len, has_global_pp_for_dims, has_seed_for_dims,
    get_global_pp_seeded_params_for_dims, set_global_pp, set_global_pp_seeded, try_get_loaded_global_pp_for_dims,
    unload_global_pp_for_dims, AjtaiSModule,
};

#[doc(hidden)]
pub use commit::{commit_row_major_seeded, sample_uniform_rq, seeded_pp_chunk_seeds};
