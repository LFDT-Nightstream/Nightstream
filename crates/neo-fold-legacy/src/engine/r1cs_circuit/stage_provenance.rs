//! Row and allocation intervals from sequential R1CS emission-stage markers.
//!
//! Owns: lightweight row checkpoints and their validated conversion into one
//! ordered physical-stage partition when instrumentation is available.
//!
//! Does not own: per-column semantic provenance, nested semantic row families,
//! constraint validity, or permission to remove rows.
//!
//! Emits constraints: no.
//!
//! | Surface | Mathematical content | Authority boundary |
//! |---|---|---|
//! | `PhysicalStageCheckpoint` | Start row and allocation cursor of one sequential emission stage | Builder diagnostic only |
//! | `PhysicalStageRange` | Exact half-open source row/allocation intervals labeled by one caller marker | Coordinate provenance, not semantic ownership |
//! | `finalize_physical_stages` | Ordered row partition plus monotone allocation spans when markers are nonempty | Does not validate an expected path tree or normalized column meaning |

use thiserror::Error;

const COMPLETE: &str = "complete";

/// Lightweight start marker recorded independently of detailed gadget traces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PhysicalStageCheckpoint {
    path: &'static str,
    row: usize,
    column: usize,
}

impl PhysicalStageCheckpoint {
    pub(crate) fn new(path: &'static str, row: usize, column: usize) -> Self {
        Self { path, row, column }
    }

    pub(crate) fn row(self) -> usize {
        self.row
    }
}

/// One exact interval in the sequential physical source-row schedule.
///
/// Fields stay private so callers cannot fabricate provenance-bearing ranges.
/// Its path remains a caller assertion: consumers that claim semantic
/// ownership must separately validate the expected root and path universe.
/// Repeated paths and empty ranges are intentional.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalStageRange {
    path: &'static str,
    row_start: usize,
    row_end: usize,
    column_start: usize,
    column_end: usize,
}

impl PhysicalStageRange {
    pub fn path(&self) -> &'static str {
        self.path
    }

    pub fn row_start(&self) -> usize {
        self.row_start
    }

    pub fn row_end(&self) -> usize {
        self.row_end
    }

    pub fn rows(&self) -> core::ops::Range<usize> {
        self.row_start..self.row_end
    }

    pub fn contains_row(&self, row: usize) -> bool {
        self.row_start <= row && row < self.row_end
    }

    /// Start of the normalized private-allocation interval owned by this
    /// checkpoint. Public outputs are moved ahead of every such interval by
    /// field-R1CS lowering.
    pub fn column_start(&self) -> usize {
        self.column_start
    }

    pub fn column_end(&self) -> usize {
        self.column_end
    }

    pub fn columns(&self) -> core::ops::Range<usize> {
        self.column_start..self.column_end
    }

    pub fn contains_column(&self, column: usize) -> bool {
        self.column_start <= column && column < self.column_end
    }

    pub(crate) fn with_columns(self, column_start: usize, column_end: usize) -> Self {
        debug_assert!(column_start <= column_end);
        Self {
            column_start,
            column_end,
            ..self
        }
    }
}

/// A lightweight stage schedule did not define one complete source-row
/// partition. No fallback owner is invented when this validation fails.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PhysicalStageError {
    #[error("physical R1CS stage {index} has an empty path")]
    EmptyPath { index: usize },
    #[error("physical R1CS stages start at row {row}, expected row 0")]
    FirstRow { row: usize },
    #[error("physical R1CS stages start at column {column}, expected the constant-one boundary 1")]
    FirstColumn { column: usize },
    #[error("physical R1CS stage {index} starts at row {row}, before prior row {prior}")]
    NonMonotone {
        index: usize,
        prior: usize,
        row: usize,
    },
    #[error("physical R1CS stage {index} starts at row {row}, beyond relation row count {rows}")]
    OutOfBounds {
        index: usize,
        row: usize,
        rows: usize,
    },
    #[error("physical R1CS stage {index} starts at column {column}, before prior column {prior}")]
    NonMonotoneColumn {
        index: usize,
        prior: usize,
        column: usize,
    },
    #[error("physical R1CS stage {index} starts at column {column}, beyond relation column count {columns}")]
    ColumnOutOfBounds {
        index: usize,
        column: usize,
        columns: usize,
    },
    #[error(
        "physical R1CS stage {index} uses reserved terminator `complete` at row {row}; it must occur once, last, at row {rows}"
    )]
    ReservedComplete {
        index: usize,
        row: usize,
        rows: usize,
    },
    #[error("physical R1CS terminator at index {index} starts at column {column}, expected final column {columns}")]
    ReservedCompleteColumn {
        index: usize,
        column: usize,
        columns: usize,
    },
}

/// Close a source-row schedule at the synthesized relation boundary.
///
/// An exact `complete` checkpoint at `rows` is a terminator rather than an
/// owner. With no explicit terminator, the final named stage closes at
/// `rows`. An empty checkpoint list remains empty: lack of provenance is not
/// silently relabeled as an unclassified stage.
pub(crate) fn finalize_physical_stages(
    checkpoints: &[PhysicalStageCheckpoint],
    rows: usize,
    columns: usize,
) -> Result<Vec<PhysicalStageRange>, PhysicalStageError> {
    if checkpoints.is_empty() {
        return Ok(Vec::new());
    }
    if checkpoints[0].row != 0 {
        return Err(PhysicalStageError::FirstRow {
            row: checkpoints[0].row,
        });
    }
    if checkpoints[0].column != 1 {
        return Err(PhysicalStageError::FirstColumn {
            column: checkpoints[0].column,
        });
    }
    let mut prior = 0;
    let mut prior_column = 0;
    for (index, checkpoint) in checkpoints.iter().enumerate() {
        if checkpoint.path.is_empty() {
            return Err(PhysicalStageError::EmptyPath { index });
        }
        if checkpoint.row < prior {
            return Err(PhysicalStageError::NonMonotone {
                index,
                prior,
                row: checkpoint.row,
            });
        }
        if checkpoint.row > rows {
            return Err(PhysicalStageError::OutOfBounds {
                index,
                row: checkpoint.row,
                rows,
            });
        }
        if checkpoint.column < prior_column {
            return Err(PhysicalStageError::NonMonotoneColumn {
                index,
                prior: prior_column,
                column: checkpoint.column,
            });
        }
        if checkpoint.column > columns {
            return Err(PhysicalStageError::ColumnOutOfBounds {
                index,
                column: checkpoint.column,
                columns,
            });
        }
        prior = checkpoint.row;
        prior_column = checkpoint.column;
    }

    let last_index = checkpoints.len() - 1;
    for (index, checkpoint) in checkpoints.iter().enumerate() {
        if checkpoint.path == COMPLETE && (index != last_index || checkpoint.row != rows) {
            return Err(PhysicalStageError::ReservedComplete {
                index,
                row: checkpoint.row,
                rows,
            });
        }
    }
    if checkpoints[last_index].path == COMPLETE && checkpoints[last_index].column != columns {
        return Err(PhysicalStageError::ReservedCompleteColumn {
            index: last_index,
            column: checkpoints[last_index].column,
            columns,
        });
    }
    let terminal = checkpoints[last_index].path == COMPLETE;
    let owner_count = checkpoints.len() - usize::from(terminal);
    let mut ranges = Vec::with_capacity(owner_count);
    for index in 0..owner_count {
        let start = checkpoints[index];
        let row_end = checkpoints.get(index + 1).map_or(rows, |next| next.row);
        let column_end = checkpoints
            .get(index + 1)
            .map_or(columns, |next| next.column);
        ranges.push(PhysicalStageRange {
            path: start.path,
            row_start: start.row,
            row_end,
            column_start: start.column,
            column_end,
        });
    }
    Ok(ranges)
}
