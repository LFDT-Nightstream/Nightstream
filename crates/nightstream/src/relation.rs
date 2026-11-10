//! The relation builders of an application (`neo-application`): R1CS rows
//! with their tags, column registries, range checks, continuity links,
//! memory catalogs with their host checks, and the event commitment. A
//! frontend builds the relation of one row with them and proves it with
//! `nebula::R1csApplication` (SPEC.md section 7.6).

pub use neo_application::event_commitment;
pub use neo_application::{
    check_continuity_rows, check_memory_rows, define_column_region, range_checked_variable_widths, ApplicationRelation,
    ApplicationRelationError, ColumnFamilySpec, ColumnId, ColumnRegistry, ColumnRegistryError, ColumnWidth,
    ConstraintCatalog, ConstraintTag, ContinuityCatalog, ContinuityCatalogError, ContinuityCheckError, ContinuityGroup,
    ContinuityLink, EventCommitment, MemoryCatalog, MemoryCatalogError, MemoryCheckError, MemoryCheckPolicy,
    MemoryCheckPolicyError, MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemoryPreload,
    MemorySpec, R1csBuildError, R1csBuilder, R1csCheckError, R1csRelation, R1csRow, RamInitialization,
    RangeCheckAssignmentError, RangeCheckBitFamily, RangeCheckLayout, RangeCheckLayoutError, TaggedR1csBuilder,
    TaggedR1csRow, ZeroTest, EVENT_COMMITMENT_AUX_COLUMNS,
};

pub use neo_application::{AuxiliaryAssignmentError, AuxiliaryBuilder, AuxiliaryLayout, LinearExpression};
