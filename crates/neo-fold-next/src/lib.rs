//! Owns the active Rust proving path for `neo-fold-next`.
//!
//! Ownership:
//! - `prover`, `verifier`: generic `Π_CCS -> Π_RLC -> Π_DEC`
//! - `construction2`: relation-neutral recursive public-image primitives
//! - `run`: session orchestration
//! - `ivc`: generic native SuperNeo IVC/NIFS accumulator carrier
//! - `proof`: generic session proof boundary
//! - `opening`: shared opening-claim and time-opening summary boundary
//! - `step_build`: frontend-produced step packaging and extension records
//! - `time_opening`, `finalize`: final opening and packaged-proof boundaries
//! - `witness_layout`: shared local packed witness layout helpers
//! - `vm`: static VM contracts
//! - `chip8`: current VM frontend and staged kernel

pub mod chip8;
pub mod chunk_relation;
pub mod construction2;
pub mod decider;
pub mod finalize;
pub mod ivc;
pub mod nightstream;
pub mod opening;
pub mod proof;
pub mod prover;
pub mod run;
pub mod rv64im;
pub mod step_build;
pub mod time_opening;
pub mod verifier;
pub mod vm;
pub mod witness_layout;

pub use rv64im::{
    prove_direct_ccs_f_prime_snark_with_perf, DirectCcsFPrimeSnarkError, DirectCcsFPrimeSnarkPerf,
    DirectCcsFPrimeSnarkProof, DirectCcsIvcState, DirectCcsLatestFPrimeSummary,
};
