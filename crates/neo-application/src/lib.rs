//! Application-relation construction for Nightstream folding frontends.
//!
//! This crate owns the domain-neutral description and compilation of application
//! constraints, state continuity, memory ports, and diagnostic metadata. It does
//! not own application semantics, witness generation, or the folding backend.

mod columns;
mod r1cs;
mod relation;

pub use columns::{ColumnRegistry, ColumnRegistryError, ColumnSpec, ColumnWidth};
pub use r1cs::{
    ConstraintCatalog, ConstraintTag, R1csBuildError, R1csBuilder, R1csRelation, R1csRow, R1csSide, TaggedR1csBuilder,
    TaggedR1csRow,
};
pub use relation::{ApplicationRelation, ApplicationRelationError};
