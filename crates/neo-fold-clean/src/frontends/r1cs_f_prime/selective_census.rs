//! Borrowed storage census for one production-shaped selective CCS relation.
//!
//! Owns: exact Rust matrix tags and compact-storage byte accounting.
//!
//! Does not own: relation semantics, constraint necessity, artifact rendering,
//! serialization compatibility, or permission to remove rows.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the census borrows an already compiled [`Structure`].
//! It reports storage facts only and rejects the identity representation, which
//! is not emitted by the rectangular selective compiler.
//!
//! | Census leaf | Exact source | Reported quantity |
//! |---|---|---|
//! | CSC | `CscMat` slices | dimensions, pointer length, and stored nonzeros |
//! | Seeded Phi81 | compact block metadata | block count and metadata bytes |
//! | Geometric rows | compact run descriptors | run count and descriptor bytes |
//! | Port | `CcsMatrix` enum | exact Rust tag and conservative wire bytes |

use neo_ccs::{CcsMatrix, CscMat};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::paper::relations::Structure;

const SELECTIVE_PORT_COUNT: usize = 13;

// The audit envelope deliberately uses fixed-width words for every tag,
// length, position, and scalar. It therefore does not depend on `usize` width
// or any particular serde codec.
const WIRE_WORD_BYTES: u128 = 8;
const WIRE_CSC_INDEX_BYTES: u128 = 4;
const WIRE_FIELD_BYTES: u128 = 8;
const WIRE_SEED_BYTES: u128 = 32;
const WIRE_GEOMETRIC_RUN_BYTES: u128 = 5 * WIRE_WORD_BYTES;

/// Exact in-memory Rust representation of one CCS matrix port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectiveMatrixTag {
    Identity,
    Csc,
    CscWithSeededPhi81,
}

impl SelectiveMatrixTag {
    /// Read the enum tag without expanding or cloning matrix contents.
    pub fn from_matrix(matrix: &CcsMatrix<neo_math::F>) -> Self {
        match matrix {
            CcsMatrix::Identity { .. } => Self::Identity,
            CcsMatrix::Csc(_) => Self::Csc,
            CcsMatrix::CscWithSeededPhi81 { .. } => Self::CscWithSeededPhi81,
        }
    }
}

/// Compact storage facts for one selective polynomial port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectivePortCensus {
    port: usize,
    tag: SelectiveMatrixTag,
    rows: usize,
    columns: usize,
    col_ptr_len: usize,
    nnz: usize,
    seeded_block_count: usize,
    seeded_metadata_bytes: u128,
    geometric_run_count: usize,
    conservative_raw_wire_bytes: u128,
}

impl SelectivePortCensus {
    pub fn port(&self) -> usize {
        self.port
    }

    pub fn tag(&self) -> SelectiveMatrixTag {
        self.tag
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn col_ptr_len(&self) -> usize {
        self.col_ptr_len
    }

    /// Number of explicitly stored CSC coefficients. Compact seeded and
    /// geometric coefficients are intentionally excluded.
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    pub fn seeded_block_count(&self) -> usize {
        self.seeded_block_count
    }

    /// Fixed-width metadata payload for all seeded blocks, excluding the
    /// enclosing block-list length word.
    pub fn seeded_metadata_bytes(&self) -> u128 {
        self.seeded_metadata_bytes
    }

    pub fn geometric_run_count(&self) -> usize {
        self.geometric_run_count
    }

    /// Conservative fixed-width envelope for this port. It counts eight-byte
    /// tags, lengths, positions, and field scalars; four-byte CSC indices;
    /// raw 32-byte seeds; and both compact-list length words. It is an audit
    /// estimate, not a claim about serde or bincode bytes.
    pub fn conservative_raw_wire_bytes(&self) -> u128 {
        self.conservative_raw_wire_bytes
    }
}

/// Invalid storage found at the production selective-census boundary.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SelectiveStructureCensusError {
    #[error("selective census requires 13 matrix/polynomial ports, got {matrices}/{polynomial_arity}")]
    PortCount {
        matrices: usize,
        polynomial_arity: usize,
    },
    #[error("selective census requires positive dimensions, got {rows}x{columns}")]
    EmptyDimensions { rows: usize, columns: usize },
    #[error("selective census rejects identity matrix representation at port {port}")]
    IdentityPort { port: usize },
    #[error(
        "selective census port {port} has dimensions {rows}x{columns}, expected {expected_rows}x{expected_columns}"
    )]
    MatrixDimensions {
        port: usize,
        rows: usize,
        columns: usize,
        expected_rows: usize,
        expected_columns: usize,
    },
    #[error("selective census port {port} CSC pointer length {got} != columns + 1 ({expected})")]
    ColumnPointerLength {
        port: usize,
        got: usize,
        expected: usize,
    },
    #[error("selective census port {port} CSC row/value lengths differ: {rows}/{values}")]
    ParallelEntryLength {
        port: usize,
        rows: usize,
        values: usize,
    },
    #[error("selective census port {port} CSC pointer endpoints are {head:?}/{tail:?}, expected 0/{nnz}")]
    ColumnPointerEndpoints {
        port: usize,
        head: Option<u32>,
        tail: Option<u32>,
        nnz: usize,
    },
    #[error("selective census port {port} column {column} has invalid CSC range {start}..{end} for {nnz} entries")]
    ColumnPointerRange {
        port: usize,
        column: usize,
        start: usize,
        end: usize,
        nnz: usize,
    },
    #[error("selective census port {port} column {column} entry {entry} has row {row}, outside 0..{rows}")]
    RowIndexOutOfBounds {
        port: usize,
        column: usize,
        entry: usize,
        row: usize,
        rows: usize,
    },
    #[error(
        "selective census port {port} column {column} rows are not strictly increasing at entry {entry}: {previous} then {row}"
    )]
    RowIndexOrder {
        port: usize,
        column: usize,
        entry: usize,
        previous: usize,
        row: usize,
    },
    #[error("selective census port {port} column {column} entry {entry} stores an explicit zero")]
    ExplicitZero {
        port: usize,
        column: usize,
        entry: usize,
    },
    #[error("selective census port {port} seeded block {block} lies outside the matrix")]
    SeededBlockShape { port: usize, block: usize },
    #[error("selective census port {port} geometric run {run} lies outside the matrix")]
    GeometricRunShape { port: usize, run: usize },
}

/// Checked zero-copy view of one production-shaped selective structure.
#[derive(Clone, Copy, Debug)]
pub struct SelectiveStructureCensus<'a> {
    structure: &'a Structure,
}

impl<'a> SelectiveStructureCensus<'a> {
    /// Validate the compact storage envelope without cloning any matrix data.
    pub fn new(structure: &'a Structure) -> Result<Self, SelectiveStructureCensusError> {
        if structure.matrices.len() != SELECTIVE_PORT_COUNT || structure.f.arity() != SELECTIVE_PORT_COUNT {
            return Err(SelectiveStructureCensusError::PortCount {
                matrices: structure.matrices.len(),
                polynomial_arity: structure.f.arity(),
            });
        }
        if structure.n == 0 || structure.m == 0 {
            return Err(SelectiveStructureCensusError::EmptyDimensions {
                rows: structure.n,
                columns: structure.m,
            });
        }
        for (port, matrix) in structure.matrices.iter().enumerate() {
            validate_port(structure, port, matrix)?;
        }
        Ok(Self { structure })
    }

    pub fn structure(&self) -> &'a Structure {
        self.structure
    }

    pub fn port_count(&self) -> usize {
        self.structure.matrices.len()
    }

    pub fn port(&self, port: usize) -> Option<SelectivePortCensus> {
        self.structure
            .matrices
            .get(port)
            .map(|matrix| census_port(port, matrix))
    }

    pub fn ports(&self) -> impl ExactSizeIterator<Item = SelectivePortCensus> + '_ {
        self.structure
            .matrices
            .iter()
            .enumerate()
            .map(|(port, matrix)| census_port(port, matrix))
    }

    pub fn conservative_raw_wire_bytes(&self) -> u128 {
        // Enclosing rows, columns, and matrix-count words.
        3 * WIRE_WORD_BYTES
            + self
                .ports()
                .map(|port| port.conservative_raw_wire_bytes())
                .sum::<u128>()
    }
}

fn validate_port(
    structure: &Structure,
    port: usize,
    matrix: &CcsMatrix<neo_math::F>,
) -> Result<(), SelectiveStructureCensusError> {
    let (csc, blocks, runs) = match matrix {
        CcsMatrix::Identity { .. } => return Err(SelectiveStructureCensusError::IdentityPort { port }),
        CcsMatrix::Csc(csc) => (csc, &[][..], &[][..]),
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => (csc, blocks.as_slice(), geometric_runs.as_slice()),
    };
    validate_csc(structure, port, csc)?;
    for (block, value) in blocks.iter().enumerate() {
        if value.validate_matrix_shape(csc.nrows, csc.ncols).is_err() {
            return Err(SelectiveStructureCensusError::SeededBlockShape { port, block });
        }
    }
    for (run, value) in runs.iter().enumerate() {
        if !value.validate_shape(csc.nrows, csc.ncols) {
            return Err(SelectiveStructureCensusError::GeometricRunShape { port, run });
        }
    }
    Ok(())
}

fn validate_csc(
    structure: &Structure,
    port: usize,
    csc: &CscMat<neo_math::F>,
) -> Result<(), SelectiveStructureCensusError> {
    if csc.nrows != structure.n || csc.ncols != structure.m {
        return Err(SelectiveStructureCensusError::MatrixDimensions {
            port,
            rows: csc.nrows,
            columns: csc.ncols,
            expected_rows: structure.n,
            expected_columns: structure.m,
        });
    }
    let expected_pointers = csc.ncols + 1;
    if csc.col_ptr.len() != expected_pointers {
        return Err(SelectiveStructureCensusError::ColumnPointerLength {
            port,
            got: csc.col_ptr.len(),
            expected: expected_pointers,
        });
    }
    if csc.row_idx.len() != csc.vals.len() {
        return Err(SelectiveStructureCensusError::ParallelEntryLength {
            port,
            rows: csc.row_idx.len(),
            values: csc.vals.len(),
        });
    }
    let nnz = csc.vals.len();
    let tail = u32::try_from(nnz).ok();
    if csc.col_ptr.first().copied() != Some(0) || csc.col_ptr.last().copied() != tail {
        return Err(SelectiveStructureCensusError::ColumnPointerEndpoints {
            port,
            head: csc.col_ptr.first().copied(),
            tail: csc.col_ptr.last().copied(),
            nnz,
        });
    }
    for column in 0..csc.ncols {
        let start = csc.col_ptr[column] as usize;
        let end = csc.col_ptr[column + 1] as usize;
        if start > end || end > nnz {
            return Err(SelectiveStructureCensusError::ColumnPointerRange {
                port,
                column,
                start,
                end,
                nnz,
            });
        }
        let mut previous = None;
        for entry in start..end {
            let row = csc.row_idx[entry] as usize;
            if row >= csc.nrows {
                return Err(SelectiveStructureCensusError::RowIndexOutOfBounds {
                    port,
                    column,
                    entry,
                    row,
                    rows: csc.nrows,
                });
            }
            if let Some(previous) = previous {
                if previous >= row {
                    return Err(SelectiveStructureCensusError::RowIndexOrder {
                        port,
                        column,
                        entry,
                        previous,
                        row,
                    });
                }
            }
            if csc.vals[entry] == neo_math::F::ZERO {
                return Err(SelectiveStructureCensusError::ExplicitZero { port, column, entry });
            }
            previous = Some(row);
        }
    }
    Ok(())
}

fn census_port(port: usize, matrix: &CcsMatrix<neo_math::F>) -> SelectivePortCensus {
    let (tag, csc, blocks, runs) = match matrix {
        CcsMatrix::Identity { .. } => unreachable!("validated selective census excludes identity ports"),
        CcsMatrix::Csc(csc) => (SelectiveMatrixTag::Csc, csc, &[][..], &[][..]),
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => (
            SelectiveMatrixTag::CscWithSeededPhi81,
            csc,
            blocks.as_slice(),
            geometric_runs.as_slice(),
        ),
    };
    let seeded_metadata_bytes = blocks.iter().map(seeded_block_metadata_bytes).sum::<u128>();
    let csc_wire_bytes = 6 * WIRE_WORD_BYTES
        + csc.col_ptr.len() as u128 * WIRE_CSC_INDEX_BYTES
        + csc.row_idx.len() as u128 * WIRE_CSC_INDEX_BYTES
        + csc.vals.len() as u128 * WIRE_FIELD_BYTES;
    let conservative_raw_wire_bytes =
        csc_wire_bytes + 2 * WIRE_WORD_BYTES + seeded_metadata_bytes + runs.len() as u128 * WIRE_GEOMETRIC_RUN_BYTES;
    SelectivePortCensus {
        port,
        tag,
        rows: csc.nrows,
        columns: csc.ncols,
        col_ptr_len: csc.col_ptr.len(),
        nnz: csc.vals.len(),
        seeded_block_count: blocks.len(),
        seeded_metadata_bytes,
        geometric_run_count: runs.len(),
        conservative_raw_wire_bytes,
    }
}

fn seeded_block_metadata_bytes(block: &neo_ccs::SeededPhi81LinearBlock) -> u128 {
    // Six fixed words: row start, word width, kappa, message columns, chunk
    // size, and transformed-columns flag. Two more words prefix word starts
    // and seed rows; every seed row has its own length word.
    8 * WIRE_WORD_BYTES
        + block.word_starts().len() as u128 * WIRE_WORD_BYTES
        + block.chunk_seeds_by_row().len() as u128 * WIRE_WORD_BYTES
        + block
            .chunk_seeds_by_row()
            .iter()
            .map(|seeds| seeds.len() as u128 * WIRE_SEED_BYTES)
            .sum::<u128>()
}
