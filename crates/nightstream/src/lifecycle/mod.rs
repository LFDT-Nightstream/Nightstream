//! The step lifecycle: proving, extending and verifying steps over the light
//! and refresh circuits, and the final proof (`bounded_depth`).

pub(crate) mod bounded_depth;
mod encoding;

pub use encoding::ProofCodecError;

#[cfg(test)]
#[path = "../../tests/bounded_depth_lifecycle/ir_log.rs"]
mod ir_log;
#[cfg(test)]
#[path = "../../tests/bounded_depth_lifecycle/runs_file.rs"]
mod runs_file;
