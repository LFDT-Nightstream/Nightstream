//! Phase-local loading of one Lean-authored circuit-and-matrix envelope.
//!
//! This boundary checks the structural package identity. It is not the final
//! production verifier path, which must also bind the concrete application,
//! verifier context, commitment key, and verification key.

use p3_field::PrimeField64;
use serde::Deserialize;
use serde_json::Value;

use super::matrix_program::{MatrixProgram, MEANINGFUL_PORTS};
use super::{relation_identifier, validate_package_schema, LoadedPackage, PackageError, RawPackage};

const SEALED_PACKAGE_SCHEMA: u64 = 1;
const INNER_PACKAGE_SCHEMA: u64 = 8;
const MATRIX_COUNT: usize = 14;

#[derive(Debug, Deserialize)]
struct RawSealedPackage(u64, RawPackage, Value);

/// One canonical sparse entry in a Lean-derived SuperNeo matrix row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogicalMatrixEntry {
    column: usize,
    coefficient: u64,
}

impl LogicalMatrixEntry {
    pub fn column(&self) -> usize {
        self.column
    }

    pub fn coefficient(&self) -> u64 {
        self.coefficient
    }
}

/// The fourteen matrix forms at one Boolean-row ordinal. Slot 13 is always
/// the canonical zero form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalMatrixRow {
    matrices: [Vec<LogicalMatrixEntry>; MATRIX_COUNT],
}

impl LogicalMatrixRow {
    pub fn matrix(&self, slot: usize) -> Option<&[LogicalMatrixEntry]> {
        self.matrices.get(slot).map(Vec::as_slice)
    }
}

/// A structurally identity-bound per-application package. Final production
/// acceptance remains unavailable until the concrete key binding is checked.
#[derive(Clone, Debug)]
pub struct LoadedPerApplicationPackage {
    circuit: LoadedPackage,
    matrix_program: MatrixProgram,
    structural_identifier: [u64; 4],
}

impl LoadedPerApplicationPackage {
    pub fn structural_identifier(&self) -> [u64; 4] {
        self.structural_identifier
    }

    pub fn row_count(&self) -> usize {
        self.circuit.relation.row_count()
    }

    pub fn logical_column_count(&self) -> usize {
        self.circuit.relation.column_count()
    }

    /// Decode one live Lean matrix-program row. Boolean rows after
    /// `row_count()` are the zero padding defined by Lean and are not stored.
    pub fn matrix_row(&self, ordinal: usize) -> Result<LogicalMatrixRow, PackageError> {
        if ordinal >= self.row_count() {
            return Err(PackageError::Invalid("logical matrix row ordinal"));
        }
        let forms = self
            .matrix_program
            .row(self.logical_column_count(), ordinal, &|source| {
                self.circuit.source_row(source)
            })?;
        let mut matrices = forms
            .into_iter()
            .map(|form| {
                form.entries
                    .into_iter()
                    .map(|entry| LogicalMatrixEntry {
                        column: entry.column,
                        coefficient: entry.coefficient.as_canonical_u64(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(matrices.len(), MEANINGFUL_PORTS);
        matrices.push(Vec::new());
        Ok(LogicalMatrixRow {
            matrices: matrices
                .try_into()
                .map_err(|_| PackageError::Invalid("logical matrix port count"))?,
        })
    }
}

/// Strictly decode one canonical Lean sealed value and pin its complete
/// circuit-and-matrix structural identity. This is a phase-local conformance
/// boundary, not the final verifier-key binding.
pub fn load_per_application_package(
    bytes: &[u8],
    expected_structural_identifier: [u64; 4],
) -> Result<LoadedPerApplicationPackage, PackageError> {
    let value: Value = serde_json::from_slice(bytes)?;
    let mut canonical = serde_json::to_vec(&value)?;
    canonical.push(b'\n');
    if bytes != canonical {
        return Err(PackageError::NonCanonicalBytes);
    }

    let computed = relation_identifier(&value)?;
    if computed != expected_structural_identifier {
        return Err(PackageError::ExpectedIdentityMismatch {
            expected: expected_structural_identifier,
            computed,
        });
    }

    let RawSealedPackage(schema, raw_circuit, raw_matrix): RawSealedPackage = serde_json::from_value(value)?;
    if schema != SEALED_PACKAGE_SCHEMA {
        return Err(PackageError::Invalid("sealed package schema version"));
    }
    let circuit = validate_package_schema(raw_circuit, computed, INNER_PACKAGE_SCHEMA)?;
    let matrix_program = MatrixProgram::decode(&raw_matrix)?;
    matrix_program.validate(circuit.layout.row_count)?;
    if matrix_program.row_count()? != circuit.relation.row_count() {
        return Err(PackageError::Invalid("matrix program relation row count"));
    }

    Ok(LoadedPerApplicationPackage {
        circuit,
        matrix_program,
        structural_identifier: computed,
    })
}
