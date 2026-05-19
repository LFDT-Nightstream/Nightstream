//! Owns Stage 3 continuity and bridge semantics for the RV32IM parity slice.

mod proof;
mod semantics;

pub use proof::{build_stage3_proof_bundle, PcAdjacentBridge, Stage3LinkageProof, Stage3ProofBundle};
pub use proof::{build_stage3_summary, continuity_event_word_width, ContinuityEvent, Stage3Summary};
pub(crate) use proof::{continuity_event_digest, continuity_event_words};
pub(crate) use semantics::verify_stage3_semantics;
pub use semantics::Stage3SemanticsProof;
