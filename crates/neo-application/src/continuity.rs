//! Ordered declarations and diagnostic replay of application state carried
//! between consecutive steps.

use std::collections::BTreeMap;

use neo_math::F;

use crate::ColumnRegistry;

/// One equality enforced across a step boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContinuityLink {
    pub previous_step_column: usize,
    pub next_step_column: usize,
}

/// A named family of related cross-step equalities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinuityGroup {
    pub name: &'static str,
    pub role: &'static str,
    pub links: Vec<ContinuityLink>,
}

/// Validated continuity declarations in verifier-facing order.
///
/// Group and link order is preserved because consumers may use the flattened
/// sequence to define public digests and proof shapes. The links form a
/// partial bijection: a carried column has at most one source and destination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinuityCatalog {
    groups: Vec<ContinuityGroup>,
    link_count: usize,
}

impl ContinuityCatalog {
    pub fn new(
        groups: impl IntoIterator<Item = ContinuityGroup>,
        columns: &ColumnRegistry,
    ) -> Result<Self, ContinuityCatalogError> {
        let groups: Vec<_> = groups.into_iter().collect();
        let mut names = BTreeMap::new();
        let mut previous_step_columns = BTreeMap::new();
        let mut next_step_columns = BTreeMap::new();
        let mut link_count = 0usize;

        for (group_index, group) in groups.iter().enumerate() {
            if group.name.is_empty() {
                return Err(ContinuityCatalogError::EmptyName { group: group_index });
            }
            if let Some(first_group) = names.insert(group.name, group_index) {
                return Err(ContinuityCatalogError::DuplicateName {
                    name: group.name,
                    first_group,
                    second_group: group_index,
                });
            }
            if group.links.is_empty() {
                return Err(ContinuityCatalogError::EmptyGroup {
                    group: group_index,
                    name: group.name,
                });
            }

            for (link_index, link) in group.links.iter().enumerate() {
                validate_column(
                    columns,
                    group_index,
                    link_index,
                    "previous-step",
                    link.previous_step_column,
                )?;
                validate_column(columns, group_index, link_index, "next-step", link.next_step_column)?;

                if let Some(&(first_group, first_link)) = previous_step_columns.get(&link.previous_step_column) {
                    return Err(ContinuityCatalogError::RepeatedPreviousStepColumn {
                        column: link.previous_step_column,
                        first_group,
                        first_link,
                        second_group: group_index,
                        second_link: link_index,
                    });
                }
                if let Some(&(first_group, first_link)) = next_step_columns.get(&link.next_step_column) {
                    return Err(ContinuityCatalogError::RepeatedNextStepColumn {
                        column: link.next_step_column,
                        first_group,
                        first_link,
                        second_group: group_index,
                        second_link: link_index,
                    });
                }
                previous_step_columns.insert(link.previous_step_column, (group_index, link_index));
                next_step_columns.insert(link.next_step_column, (group_index, link_index));
                link_count += 1;
            }
        }

        Ok(Self { groups, link_count })
    }

    pub fn groups(&self) -> &[ContinuityGroup] {
        &self.groups
    }

    /// Iterate over all links without changing their declared order.
    pub fn links(&self) -> impl Iterator<Item = &ContinuityLink> {
        self.groups.iter().flat_map(|group| group.links.iter())
    }

    pub const fn link_count(&self) -> usize {
        self.link_count
    }
}

/// Check every declared continuity link between adjacent witness rows.
///
/// This is diagnostic replay, not a substitute for emitting the corresponding
/// relation or backend constraints.
pub fn check_continuity_rows(catalog: &ContinuityCatalog, witness_rows: &[Vec<F>]) -> Result<(), ContinuityCheckError> {
    for (boundary, rows) in witness_rows.windows(2).enumerate() {
        let previous = &rows[0];
        let next = &rows[1];
        for (group_index, group) in catalog.groups().iter().enumerate() {
            for (link_index, link) in group.links.iter().enumerate() {
                let previous_value = read_column(
                    previous,
                    boundary,
                    group_index,
                    group.name,
                    link_index,
                    link.previous_step_column,
                )?;
                let next_value = read_column(
                    next,
                    boundary + 1,
                    group_index,
                    group.name,
                    link_index,
                    link.next_step_column,
                )?;
                if previous_value != next_value {
                    return Err(ContinuityCheckError::Mismatch {
                        boundary,
                        group: group_index,
                        group_name: group.name,
                        link: link_index,
                        previous_step_column: link.previous_step_column,
                        next_step_column: link.next_step_column,
                        previous_value,
                        next_value,
                    });
                }
            }
        }
    }
    Ok(())
}

fn read_column(
    witness: &[F],
    row: usize,
    group: usize,
    group_name: &'static str,
    link: usize,
    column: usize,
) -> Result<F, ContinuityCheckError> {
    witness
        .get(column)
        .copied()
        .ok_or(ContinuityCheckError::MissingColumn {
            row,
            group,
            group_name,
            link,
            column,
            actual_width: witness.len(),
        })
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ContinuityCheckError {
    #[error(
        "continuity group {group} ({group_name:?}) link {link} references column {column}, but witness row {row} has width {actual_width}"
    )]
    MissingColumn {
        row: usize,
        group: usize,
        group_name: &'static str,
        link: usize,
        column: usize,
        actual_width: usize,
    },
    #[error(
        "continuity mismatch across boundary {boundary} in group {group} ({group_name:?}) link {link}: previous column {previous_step_column} is {previous_value}, next column {next_step_column} is {next_value}"
    )]
    Mismatch {
        boundary: usize,
        group: usize,
        group_name: &'static str,
        link: usize,
        previous_step_column: usize,
        next_step_column: usize,
        previous_value: F,
        next_value: F,
    },
}

fn validate_column(
    columns: &ColumnRegistry,
    group: usize,
    link: usize,
    endpoint: &'static str,
    column: usize,
) -> Result<(), ContinuityCatalogError> {
    if column >= columns.column_count() {
        return Err(ContinuityCatalogError::ColumnOutOfRange {
            group,
            link,
            endpoint,
            column,
            column_count: columns.column_count(),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ContinuityCatalogError {
    #[error("continuity group {group} has an empty name")]
    EmptyName { group: usize },
    #[error("continuity group {group} ({name:?}) has no links")]
    EmptyGroup { group: usize, name: &'static str },
    #[error("continuity group name {name:?} is repeated at indices {first_group} and {second_group}")]
    DuplicateName {
        name: &'static str,
        first_group: usize,
        second_group: usize,
    },
    #[error(
        "continuity group {group} link {link} references {endpoint} column {column}, but the registry has {column_count} columns"
    )]
    ColumnOutOfRange {
        group: usize,
        link: usize,
        endpoint: &'static str,
        column: usize,
        column_count: usize,
    },
    #[error(
        "previous-step continuity column {column} is repeated at group/link {first_group}/{first_link} and {second_group}/{second_link}"
    )]
    RepeatedPreviousStepColumn {
        column: usize,
        first_group: usize,
        first_link: usize,
        second_group: usize,
        second_link: usize,
    },
    #[error(
        "next-step continuity column {column} is repeated at group/link {first_group}/{first_link} and {second_group}/{second_link}"
    )]
    RepeatedNextStepColumn {
        column: usize,
        first_group: usize,
        first_link: usize,
        second_group: usize,
        second_link: usize,
    },
}
