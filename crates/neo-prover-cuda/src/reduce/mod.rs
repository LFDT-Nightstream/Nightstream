//! The three SuperNeo reductions, one module per protocol step:
//! `ccs` (Π_CCS sumcheck), `rlc` (Π_RLC combination), `dec` (Π_DEC
//! decomposition). Each owns its device flow; shared session state lives
//! in `crate::session`.

pub mod ccs;
pub mod dec;
pub mod rlc;
