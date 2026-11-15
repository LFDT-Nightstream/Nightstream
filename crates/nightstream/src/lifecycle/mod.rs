//! The step lifecycle: proving, extending and verifying steps over the light
//! and refresh circuits (`bounded_depth`).

pub(crate) mod bounded_depth;
mod encoding;

pub use encoding::ProofCodecError;
