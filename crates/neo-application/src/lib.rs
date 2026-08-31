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
mod gadgets;
mod memory;
mod memory_check;
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
pub use continuity::{ContinuityCatalog, ContinuityCatalogError, ContinuityGroup, ContinuityLink};
pub use gadgets::{ConditionalSelect, GadgetDescriptor, GadgetOccurrence, ZeroTest};
pub use memory::{
    MemoryCatalog, MemoryCatalogError, MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec,
};
pub use memory_check::{
    check_memory_rows, MemoryCheckError, MemoryCheckPolicy, MemoryCheckPolicyError, MemoryPreload, RamInitialization,
};
pub use r1cs::{
    ConstraintCatalog, ConstraintTag, R1csBuildError, R1csBuilder, R1csRelation, R1csRow, R1csSide, TaggedR1csBuilder,
    TaggedR1csRow,
};
pub use relation::{ApplicationRelation, ApplicationRelationError};
