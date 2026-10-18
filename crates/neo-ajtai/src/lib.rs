//! Ajtai matrix commitment.

#![forbid(unsafe_code)]

mod commit;
mod error;
pub mod nightstream_fprime_setup;
mod types;

pub mod s_module;

pub use commit::{
    commit, commit_masked_ct, commit_precomp_ct, commit_row_major, precompute_rot_columns, s_lincomb, s_mul, s_mul_add,
    s_mul_add_from_rot_col, scale_commitment, scale_commitment_add_inplace, setup, try_commit, try_commit_row_major,
    verify_open,
};
pub use error::{AjtaiError, AjtaiResult};
pub use types::{Commitment, PP};

#[cfg(feature = "testing")]
pub use commit::rot_step;

pub use s_module::{
    get_global_pp, get_global_pp_for_dims, get_global_pp_for_z_len, has_global_pp_for_dims, set_global_pp, AjtaiSModule,
};

#[doc(hidden)]
pub use commit::sample_uniform_rq;
