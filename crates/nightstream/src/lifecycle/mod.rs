//! The step lifecycle: proving, extending and verifying steps over the light
//! and refresh circuits, and the final proof (`option_b`).

mod encoding;
pub(crate) mod option_b;

pub(crate) use encoding::ProofCodecError;

#[cfg(test)]
#[path = "../../tests/option_b_lifecycle/ir_log.rs"]
mod ir_log;
#[cfg(test)]
#[path = "../../tests/option_b_lifecycle/runs_file.rs"]
mod runs_file;
