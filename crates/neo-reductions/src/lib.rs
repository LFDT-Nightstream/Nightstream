//! SuperNeo reductions with one canonical rectangular-paper PiCCS protocol.
//!
//! The optimized engine and independent PaperExact engine share only neutral
//! protocol messages and transcript flow. Their polynomial evaluators are
//! separate so tests can require exact proof-byte equality.

#![allow(non_snake_case)]

pub mod api;
pub mod common;
pub mod engines;
pub mod error;
pub mod sumcheck;
pub mod superneo_eval;

pub use engines::optimized_engine;
#[cfg(feature = "paper-exact")]
pub use engines::paper_exact_engine;

pub use engines::optimized_engine::{pi_ccs_prove, pi_ccs_verify, Challenges, PiCcsProof};

pub use error::PiCcsError;

pub use common::{
    rot_rhos_from_mats, rot_rhos_to_mats, sample_rot_rhos_n_typed, split_b_matrix_k,
    split_b_matrix_k_with_nonzero_flags, RotRho, RotRing, SplitDigits,
};
