//! Application-relation construction for Nightstream folding frontends.
//!
//! This crate owns the domain-neutral description and compilation of application
//! constraints, state continuity, memory ports, and diagnostic metadata. It does
//! not own application semantics, witness generation, or the folding backend.

mod columns;
mod continuity;
mod memory;
mod memory_check;
mod r1cs;
mod relation;

pub use columns::{ColumnFamilySpec, ColumnRegistry, ColumnRegistryError, ColumnWidth};
pub use continuity::{ContinuityCatalog, ContinuityCatalogError, ContinuityGroup, ContinuityLink};
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
