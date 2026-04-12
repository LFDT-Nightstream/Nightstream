//! Owns Stage 2 register/RAM/Twist summaries for the RV64IM parity slice.

mod proof;
mod semantics;

pub use proof::{
    build_stage2_proof_bundle, build_stage2_summary, ram_event_word_width, register_read_word_width,
    register_write_word_width, twist_link_word_width, RamAccessKind, RamEvent, RamTwistProof, RegisterReadEvent,
    RegisterReadRole, RegisterTwistProof, RegisterWriteEvent, Stage2LinkageProof, Stage2ProofBundle, Stage2Summary,
    Stage2TemporalContext, TwistLinkEvent,
};
pub(crate) use proof::{
    ram_event_digest, ram_event_words, ram_timeline_digest, ram_timeline_words, register_read_event_digest,
    register_read_timeline_words, register_read_words, register_timeline_digest, register_write_event_digest,
    register_write_timeline_words, register_write_words, twist_link_event_digest, twist_link_timeline_words,
    twist_link_words, twist_links_timeline_digest,
};
pub use semantics::Stage2SemanticsProof;
pub(crate) use semantics::{
    ram_events_family_digest, register_reads_family_digest, register_writes_family_digest, twist_links_family_digest,
    verify_stage2_semantics_from_events,
};
