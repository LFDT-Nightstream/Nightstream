//! Named auxiliary allocation and fixed assignment schedules.

use crate::{ColumnFamilySpec, ColumnRegistry, ColumnRegistryError, ColumnWidth};
use neo_math::F;
use std::ops::Range;

type Assign = Box<dyn Fn(&[F], &mut [F]) -> Result<(), String> + Send + Sync>;

struct Assignment {
    columns: Range<usize>,
    assign: Assign,
}

/// Construction has no witness values. Width declarations do not emit constraints.
pub struct AuxiliaryBuilder {
    base_columns: usize,
    next_column: usize,
    families: Vec<ColumnFamilySpec>,
    assignments: Vec<Assignment>,
}

impl AuxiliaryBuilder {
    pub fn new(base: &ColumnRegistry) -> Self {
        Self {
            base_columns: base.column_count(),
            next_column: base.column_count(),
            families: base.families().to_vec(),
            assignments: Vec::new(),
        }
    }

    /// The callback can read earlier columns and write only its own family.
    /// The caller must establish the declared width through relation constraints.
    pub fn allocate(
        &mut self,
        region: &'static str,
        name: impl Into<std::borrow::Cow<'static, str>>,
        role: &'static str,
        width: ColumnWidth,
        len: usize,
        assign: impl Fn(&[F], &mut [F]) -> Result<(), String> + Send + Sync + 'static,
    ) -> Result<Range<usize>, ColumnRegistryError> {
        let name = name.into();
        let end = self
            .next_column
            .checked_add(len)
            .ok_or_else(|| ColumnRegistryError::IndexOverflow { name: name.clone() })?;
        let columns = self.next_column..end;
        self.families.push(ColumnFamilySpec {
            region,
            name,
            role,
            width,
            start: columns.start,
            len,
        });
        self.assignments.push(Assignment {
            columns: columns.clone(),
            assign: Box::new(assign),
        });
        self.next_column = end;
        Ok(columns)
    }

    pub fn column_count(&self) -> usize {
        self.next_column
    }

    pub fn finish(self) -> Result<AuxiliaryLayout, ColumnRegistryError> {
        Ok(AuxiliaryLayout {
            base_columns: self.base_columns,
            columns: ColumnRegistry::new(self.families)?,
            assignments: self.assignments,
        })
    }
}

/// Frozen allocation shared by constraints, assignment, and circuit export.
pub struct AuxiliaryLayout {
    base_columns: usize,
    columns: ColumnRegistry,
    assignments: Vec<Assignment>,
}

impl AuxiliaryLayout {
    pub fn columns(&self) -> &ColumnRegistry {
        &self.columns
    }
    pub fn base_columns(&self) -> usize {
        self.base_columns
    }

    pub fn assign(&self, witness: &mut [F]) -> Result<(), AuxiliaryAssignmentError> {
        if witness.len() != self.columns.column_count() {
            return Err(AuxiliaryAssignmentError::Width {
                expected: self.columns.column_count(),
                actual: witness.len(),
            });
        }
        for assignment in &self.assignments {
            let (input, tail) = witness.split_at_mut(assignment.columns.start);
            let output = &mut tail[..assignment.columns.len()];
            (assignment.assign)(input, output).map_err(|message| AuxiliaryAssignmentError::Family {
                name: self
                    .columns
                    .family_for_column(assignment.columns.start)
                    .expect("registered allocation")
                    .name
                    .to_string(),
                message,
            })?;
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuxiliaryAssignmentError {
    #[error("auxiliary assignment has {actual} columns, expected {expected}")]
    Width { expected: usize, actual: usize },
    #[error("auxiliary family {name}: {message}")]
    Family { name: String, message: String },
}
