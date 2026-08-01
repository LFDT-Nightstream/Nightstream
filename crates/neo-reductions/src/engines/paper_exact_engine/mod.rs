//! Independent direct references for corrected SuperNeo reductions.
//!
//! `paper_rectangular` owns the two-domain derivation from the paper's joint
//! polynomial. The only protocol change is the second SumCheck required when
//! row and column dimensions differ. This module does not import optimized
//! or cached evaluator code.

#![allow(non_snake_case)]

pub mod paper_rectangular;
pub mod prove;
mod rlc_dec;
pub mod verify;

pub use rlc_dec::{
    dec_reduction_paper_exact, dec_reduction_paper_exact_with_commit_check, rlc_reduction_paper_exact,
    rlc_reduction_paper_exact_with_commit_mix,
};

pub use prove::{paper_exact_prove, paper_joint_square_prove_phase};
pub use verify::paper_exact_verify;
