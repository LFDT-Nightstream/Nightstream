//! Independent direct references for corrected SuperNeo reductions.
//!
//! `paper_joint` owns the direct one-polynomial evaluator on the paper's
//! zero-padded rectangular row domain. This module does not import optimized
//! or cached evaluator code.

#![allow(non_snake_case)]

pub mod paper_joint;
mod paper_matrix;
mod paper_ring;
pub mod prove;
mod rlc_dec;
mod transcript;
pub mod verify;

pub use rlc_dec::{
    dec_reduction_paper_exact_with_commit_check, rlc_claim_paper_exact_with_commit_mix,
    rlc_reduction_paper_exact_with_commit_mix, verify_dec_public_paper_exact,
};

pub use prove::paper_exact_prove;
pub use transcript::encode_proof;
pub(crate) use transcript::PaperTranscriptBinding;
pub use verify::{paper_exact_verify, paper_exact_verify_with_trace};
