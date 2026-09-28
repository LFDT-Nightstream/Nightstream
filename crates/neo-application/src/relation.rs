//! Assembly and validation of a complete compiled application relation.

use crate::{ColumnRegistry, R1csRelation};

/// An application R1CS paired with metadata for every physical witness column.
#[derive(Clone, Debug)]
pub struct ApplicationRelation<Owner> {
    r1cs: R1csRelation<Owner>,
    columns: ColumnRegistry,
}

impl<Owner> ApplicationRelation<Owner> {
    pub fn new(r1cs: R1csRelation<Owner>, columns: ColumnRegistry) -> Result<Self, ApplicationRelationError> {
        if r1cs.column_count() != columns.column_count() {
            return Err(ApplicationRelationError::ColumnCountMismatch {
                r1cs_column_count: r1cs.column_count(),
                registry_column_count: columns.column_count(),
            });
        }

        Ok(Self { r1cs, columns })
    }

    pub const fn r1cs(&self) -> &R1csRelation<Owner> {
        &self.r1cs
    }

    pub const fn columns(&self) -> &ColumnRegistry {
        &self.columns
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ApplicationRelationError {
    #[error("R1CS has {r1cs_column_count} columns but the registry describes {registry_column_count}")]
    ColumnCountMismatch {
        r1cs_column_count: usize,
        registry_column_count: usize,
    },
}
