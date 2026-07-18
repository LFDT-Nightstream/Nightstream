//! Verifier-owned prefix of the stabilized output-digest envelope.
//!
//! Owns: the ten exact constant-pin equations and the complete column identity
//! `prefix[10] ++ compressed_sis[54] = poseidon2_inputs[64]`.
//!
//! Does not own: source authority, either SIS map's binding security,
//! Poseidon2 permutation semantics, transcript placement, costs, or row
//! removal.
//!
//! Emits constraints: no.
//!
//! Authority boundary: prefix values are recomputed from the canonical native
//! envelope definition. Compression columns remain derived wires whose
//! equations are audited separately.
//!
//! | Leaf | Mathematical/physical obligation |
//! |---|---|
//! | `prefix.domain` | exact packed `accumulator/sis/digest/v4` fields |
//! | `prefix.map_domain` | verifier-owned Π_CCS output-map domain |
//! | `prefix.field_count` | exact profile field count |
//! | `prefix.rank` | rank two for the primary binding map |
//! | `preimage` | prefix followed by every rank-one compression output |

use std::ops::Range;

use neo_ccs::CcsMatrix;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::Poseidon2HashAudit;
use crate::frontends::r1cs_f_prime::SparseR1cs;
use crate::paper::reductions::accumulator_sis_circuit::{
    accumulator_digest_envelope_prefix, PI_CCS_OUTPUTS_SIS_CONFIG,
};
use crate::paper::reductions::pi_ccs_output_message::Profile;

use super::invalid;
use crate::frontends::r1cs_f_prime::ivc::R1csIvcError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsOutputEnvelopePrefixAudit {
    columns: Vec<usize>,
    values: Vec<F>,
    rows: Range<usize>,
}

impl PiCcsOutputEnvelopePrefixAudit {
    pub fn columns(&self) -> &[usize] {
        &self.columns
    }

    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }
}

pub(super) fn recover(
    arm: &SparseR1cs,
    profile: Profile,
    compression_row_end: usize,
    compression_outputs: &[usize],
    hash: &Poseidon2HashAudit,
) -> Result<PiCcsOutputEnvelopePrefixAudit, R1csIvcError> {
    let values = accumulator_digest_envelope_prefix(PI_CCS_OUTPUTS_SIS_CONFIG, profile.field_count());
    let prefix_len = hash
        .input_cols
        .len()
        .checked_sub(compression_outputs.len())
        .ok_or_else(|| invalid("PiCCS output Poseidon2 input is shorter than its compressed SIS payload"))?;
    if prefix_len != values.len() {
        return Err(invalid(format!(
            "PiCCS output Poseidon2 envelope has {prefix_len} prefix fields, expected {}",
            values.len()
        )));
    }
    let columns = hash.input_cols[..prefix_len].to_vec();
    let mut expected_inputs = columns.clone();
    expected_inputs.extend_from_slice(compression_outputs);
    if hash.input_cols != expected_inputs {
        return Err(invalid(
            "PiCCS output Poseidon2 inputs are not the exact prefix-plus-compression sequence",
        ));
    }

    let row_start = hash
        .row_start
        .checked_sub(values.len())
        .ok_or_else(|| invalid("PiCCS output Poseidon2 starts before its prefix rows can fit"))?;
    let rows = row_start..hash.row_start;
    if rows.start != compression_row_end {
        return Err(invalid(format!(
            "PiCCS output envelope prefix rows start at {}, but compression ends at {compression_row_end}",
            rows.start
        )));
    }

    validate_pins(&arm.a, &arm.b, &arm.c, &columns, &values, rows.clone())?;
    Ok(PiCcsOutputEnvelopePrefixAudit { columns, values, rows })
}

fn validate_pins(
    a: &CcsMatrix<F>,
    b: &CcsMatrix<F>,
    c: &CcsMatrix<F>,
    columns: &[usize],
    values: &[F],
    rows: Range<usize>,
) -> Result<(), R1csIvcError> {
    reject_compact_overlap(a, &rows, "A")?;
    reject_compact_overlap(b, &rows, "B")?;
    reject_compact_overlap(c, &rows, "C")?;
    let a_rows = scan_rows(a, &rows, "A")?;
    let b_rows = scan_rows(b, &rows, "B")?;
    let c_rows = scan_rows(c, &rows, "C")?;
    for (index, ((&column, &value), ((a_row, b_row), c_row))) in columns
        .iter()
        .zip(values)
        .zip(a_rows.iter().zip(&b_rows).zip(&c_rows))
        .enumerate()
    {
        let mut expected_a = Vec::with_capacity(2);
        if value != F::ZERO {
            expected_a.push((0, -value));
        }
        expected_a.push((column, F::ONE));
        expected_a.sort_unstable_by_key(|term| term.0);
        if a_row != &expected_a || b_row.as_slice() != [(0, F::ONE)] || !c_row.is_empty() {
            return Err(invalid(format!(
                "PiCCS output envelope prefix pin {index} is not the exact constant equation"
            )));
        }
    }
    Ok(())
}

fn scan_rows(matrix: &CcsMatrix<F>, rows: &Range<usize>, label: &str) -> Result<Vec<Vec<(usize, F)>>, R1csIvcError> {
    let csc = matrix
        .sparse_component()
        .ok_or_else(|| invalid(format!("PiCCS output {label} matrix has no sparse component")))?;
    let mut terms = vec![Vec::new(); rows.len()];
    for column in 0..csc.ncols {
        for entry in csc.column_range(column) {
            let row = csc.row_index(entry);
            if rows.contains(&row) {
                terms[row - rows.start].push((column, csc.vals[entry]));
            }
        }
    }
    Ok(terms)
}

fn reject_compact_overlap(matrix: &CcsMatrix<F>, rows: &Range<usize>, label: &str) -> Result<(), R1csIvcError> {
    let overlaps = |start: usize, end: usize| start < rows.end && rows.start < end;
    if matrix
        .seeded_phi81_blocks()
        .iter()
        .any(|block| overlaps(block.row_start(), block.row_end()))
        || matrix
            .geometric_runs()
            .iter()
            .any(|run| overlaps(run.row(), run.row() + 1))
    {
        return Err(invalid(format!(
            "PiCCS output {label} matrix contains a compact contribution in envelope-prefix rows"
        )));
    }
    Ok(())
}
