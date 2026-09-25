//! Exact materialization of one row from the final selective CCS structure.
//!
//! Owns: additive expansion of one physical row across ordinary CSC terms,
//! seeded Phi81 blocks, and geometric runs; canonical per-port sparse terms;
//! and the join to the exclusive emitted-row ledger.
//!
//! Does not own: emitter intent, row-family semantics, witness acceptance,
//! constraint necessity, artifact rendering, or permission to remove rows.
//!
//! Emits constraints: no.
//!
//! Authority boundary: every coefficient is read from the already compiled
//! [`Structure`](crate::paper::relations::Structure). Family metadata is
//! attached only after the final row is found in exactly one nonempty emitted
//! run. A family label never substitutes for inspecting all thirteen matrices.
//!
//! | Stage path | Mathematical object | Exact source | Result |
//! |---|---|---|---|
//! | `f_prime.selective_ccs.artifact.row.csc` | one sparse row contribution | final `CscMat` arrays | accumulated terms |
//! | `f_prime.selective_ccs.artifact.row.seeded_phi81` | one expanded seeded row | final block seed/geometry | accumulated terms |
//! | `f_prime.selective_ccs.artifact.row.geometric` | one geometric row contribution | final compact run | accumulated terms |
//! | `f_prime.selective_ccs.artifact.row.canonical` | sorted unique nonzero row | field addition by column | [`SelectiveMatrixRow`] |
//! | `f_prime.selective_ccs.artifact.row.owner` | unique emitted family/run | compiler row ledger | [`SelectiveRowArtifact`] |

use neo_math::F;
use thiserror::Error;

use super::lowering::SelectiveLowNormSnapshot;
use super::selective_audit::SelectiveEmittedRowFamily;
use super::selective_census::{SelectiveStructureCensus, SelectiveStructureCensusError};

pub const SELECTIVE_ROW_ARTIFACT_SCHEMA_VERSION: u64 = 1;
const SELECTIVE_PORT_COUNT: usize = 13;

/// One nonzero coefficient in a canonical materialized matrix row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectiveRowTerm {
    column: usize,
    coefficient: F,
}

impl SelectiveRowTerm {
    pub fn column(&self) -> usize {
        self.column
    }

    pub fn coefficient(&self) -> F {
        self.coefficient
    }
}

/// Exact thirteen-port sparse row decoded from a final compact structure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveMatrixRow {
    rows: usize,
    columns: usize,
    emitted_row: usize,
    ports: [Vec<SelectiveRowTerm>; SELECTIVE_PORT_COUNT],
}

impl SelectiveMatrixRow {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn emitted_row(&self) -> usize {
        self.emitted_row
    }

    pub fn port(&self, port: usize) -> Option<&[SelectiveRowTerm]> {
        self.ports.get(port).map(Vec::as_slice)
    }

    pub fn ports(&self) -> &[Vec<SelectiveRowTerm>; SELECTIVE_PORT_COUNT] {
        &self.ports
    }
}

/// One materialized row joined to its exclusive compiler owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveRowArtifact {
    matrix_row: SelectiveMatrixRow,
    run_index: usize,
    family: SelectiveEmittedRowFamily,
    arm: Option<usize>,
}

impl SelectiveRowArtifact {
    pub fn schema_version(&self) -> u64 {
        SELECTIVE_ROW_ARTIFACT_SCHEMA_VERSION
    }

    pub fn matrix_row(&self) -> &SelectiveMatrixRow {
        &self.matrix_row
    }

    pub fn run_index(&self) -> usize {
        self.run_index
    }

    pub fn family(&self) -> SelectiveEmittedRowFamily {
        self.family
    }

    pub fn arm(&self) -> Option<usize> {
        self.arm
    }
}

/// Failure to derive a canonical row from final structure data and its row
/// ledger. Every mismatch fails closed instead of repairing metadata.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SelectiveRowArtifactError {
    #[error(transparent)]
    InvalidStructure(#[from] SelectiveStructureCensusError),
    #[error("selective row {row} is outside final structure rows 0..{rows}")]
    RowOutOfBounds { row: usize, rows: usize },
    #[error("selective row ledger has {ledger_rows} rows, final structure has {structure_rows}")]
    LedgerRowCount {
        ledger_rows: usize,
        structure_rows: usize,
    },
    #[error("selective row {row} has no nonempty emitted-run owner")]
    MissingOwner { row: usize },
    #[error("selective row {row} belongs to more than one emitted run ({first} and {second})")]
    MultipleOwners {
        row: usize,
        first: usize,
        second: usize,
    },
}

impl<'a> SelectiveLowNormSnapshot<'a> {
    /// Materialize one final structure row and attach the unique emitted-run
    /// owner from the same checked compiler snapshot.
    pub fn materialize_row(&self, row: usize) -> Result<SelectiveRowArtifact, SelectiveRowArtifactError> {
        let census = SelectiveStructureCensus::new(self.structure())?;
        let rows = self.compiler_audit().rows();
        if rows.total_rows() != self.structure().n {
            return Err(SelectiveRowArtifactError::LedgerRowCount {
                ledger_rows: rows.total_rows(),
                structure_rows: self.structure().n,
            });
        }
        if row >= self.structure().n {
            return Err(SelectiveRowArtifactError::RowOutOfBounds {
                row,
                rows: self.structure().n,
            });
        }
        let mut owner = None;
        for (run_index, run) in rows.emitted_runs().iter().enumerate() {
            if !run.emitted_rows().contains(&row) {
                continue;
            }
            if let Some((first, _)) = owner {
                return Err(SelectiveRowArtifactError::MultipleOwners {
                    row,
                    first,
                    second: run_index,
                });
            }
            owner = Some((run_index, run));
        }
        let (run_index, owner) = owner.ok_or(SelectiveRowArtifactError::MissingOwner { row })?;
        let ports = std::array::from_fn(|port| {
            census.structure().matrices[port]
                .materialize_row(row)
                .expect("census-validated matrix shares the structure row count")
                .into_iter()
                .map(|(column, coefficient)| SelectiveRowTerm { column, coefficient })
                .collect()
        });
        Ok(SelectiveRowArtifact {
            matrix_row: SelectiveMatrixRow {
                rows: self.structure().n,
                columns: self.structure().m,
                emitted_row: row,
                ports,
            },
            run_index,
            family: owner.family(),
            arm: owner.arm(),
        })
    }
}
