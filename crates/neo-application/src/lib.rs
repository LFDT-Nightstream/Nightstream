//! Application-relation construction for Nightstream folding frontends.
//!
//! This crate owns the domain-neutral description and compilation of application
//! constraints, shared algebraic gadgets, state continuity, memory ports, and
//! diagnostic metadata. It does not own application state-machine semantics,
//! top-level witness orchestration, or the folding backend.

mod audit;
#[cfg(feature = "audit-html")]
mod audit_html;
mod columns;
mod continuity;
pub mod event_commitment;
mod gadgets;
mod memory;
mod memory_check;
pub mod poseidon2;
mod r1cs;
mod relation;

pub use audit::{
    continuity_column_occurrences, memory_column_occurrences, ColumnConstraintIndex, ContinuityColumnOccurrence,
    ContinuityColumnRole, GadgetColumnOccurrence, GadgetColumnRole, MemoryColumnOccurrence, MemoryColumnRole,
    R1csColumnOccurrence,
};
#[cfg(feature = "audit-html")]
pub use audit_html::render_column_audit_html;
pub use columns::{ColumnFamilySpec, ColumnRegistry, ColumnRegistryError, ColumnWidth};
pub use continuity::{
    check_continuity_rows, ContinuityCatalog, ContinuityCatalogError, ContinuityCheckError, ContinuityGroup,
    ContinuityLink,
};
pub use event_commitment::{EventCommitment, EVENT_COMMITMENT_AUX_COLUMNS};
pub use gadgets::{ConditionalSelect, GadgetDescriptor, GadgetOccurrence, Pow7, ZeroTest};
pub use memory::{
    MemoryCatalog, MemoryCatalogError, MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec,
};
pub use memory_check::{
    check_memory_rows, MemoryCheckError, MemoryCheckPolicy, MemoryCheckPolicyError, MemoryPreload, RamInitialization,
};
pub use poseidon2::{
    Poseidon2FullRound12, Poseidon2FullRoundChoice, Poseidon2PartialPair12, Poseidon2PartialPairChoice,
    Poseidon2Permutation12, POSEIDON2_PERMUTATION_AUX_COLUMNS,
};
pub use r1cs::{
    ConstraintCatalog, ConstraintTag, R1csBuildError, R1csBuilder, R1csRelation, R1csRow, R1csSide, TaggedR1csBuilder,
    TaggedR1csRow,
};
pub use relation::{ApplicationRelation, ApplicationRelationError};
