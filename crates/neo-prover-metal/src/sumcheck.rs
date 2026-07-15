//! Metal sumcheck backends, split by the FE row phase and NC column phase.

mod encoding;
mod fe;
mod mask_residency;
mod nc;

pub(crate) use fe::{FeSumcheckProfile, MetalFeBackend};
pub(crate) use nc::{MetalNcBackend, NcSumcheckProfile};
