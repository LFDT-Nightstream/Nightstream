//! In-circuit Π_RLC verifier algebra (paper §7.4, steps 1–2).
//!
//! Reduction: `CE(b, L)^(K+k) -> CE(B, L)`, where `B = b^k`.
//!
//! Owns: public arithmetic leaf modules for Π_RLC combinations, projection
//! identities, padding, and shared authority consistency.
//!
//! Does not own: transcript sampling, NIFS orchestration, or cost checkpoints.
//!
//! Emits constraints: no; child leaves do.
//!
//! Authority boundary: child functions consume already allocated Π_CCS input
//! and Π_DEC parent wires; they never accept a digest in place of those wires.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | `commitment` | Paper CE commitment combination and projection | yes | `commitment` | `NifsPaper.PiRlc` paper-public leaf |
//! | adv | Nebula product-commitment extension; not a SuperNeo CE field | yes | `commitment` | separate Nebula refinement open |
//! | `x` | Packed 270-coefficient combination plus inactive zero encoding | yes | `x` | packed arithmetic only; 257-field paper bridge open |
//! | `y_ring` | Identity-first paper CE ring evaluations | yes | `padded_k` | ring-action refinement |
//! | padded `y_ring` | Active paper evaluation plus canonical zero tail | yes | `padded_k` | active leaf plus encoding refinement |
//! | `consistency` | Fold-digest continuity | yes | `consistency` | authority replay |
//!
//! Transcript sampling is owned by `engine::r1cs_circuit::alphabet_sampling`.
//! NIFS composition and diagnostic stage boundaries are owned by
//! `paper::nifs::circuit`. [`stage`] only names that cost tree. The remaining
//! modules allocate Π_RLC algebra witnesses and emit the rows named above.

mod commitment;
mod consistency;
mod padded_k;
pub mod stage;
mod x;
mod y_ring;

pub use commitment::*;
pub use consistency::*;
pub use padded_k::*;
pub use x::*;
pub use y_ring::*;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("\u{03A0}_RLC.V: empty input set")]
    Empty,
    #[error("\u{03A0}_RLC.V: |rhos| ({rhos}) \u{2260} |inputs| ({inputs})")]
    PairCountMismatch { rhos: usize, inputs: usize },
    #[error("\u{03A0}_RLC.V: shape mismatch — {what}: expected {expected}, got {got}")]
    ShapeMismatch {
        what: &'static str,
        expected: String,
        got: String,
    },
}
