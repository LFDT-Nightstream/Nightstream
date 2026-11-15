//! One Poseidon2 permutation's rows and its external layer from the Lean
//! formula library, over caller forms. The bounded-depth packages use it for each
//! packed block (`CircuitRelation.sboxInterface`, `CircuitRelation.bindingRows`).

use std::ops::ControlFlow;

use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;

use super::form::entries;
use super::template::RowScratch;
use super::{external_layer, Entry, Form, MatrixRun, PackageError, MEANINGFUL_PORTS, POSEIDON_WIDTH};
use crate::package::SBOX_ROWS;

/// A sparse form as `(column, coefficient)` pairs.
pub(crate) type Entries = Vec<(usize, Fq)>;

fn form(pairs: &[(usize, Fq)]) -> Form {
    Form::from_entries(
        pairs
            .iter()
            .map(|&(column, coefficient)| Entry { column, coefficient })
            .collect(),
    )
}

fn pairs(runs: &[MatrixRun]) -> Entries {
    entries(runs)
        .into_iter()
        .map(|entry| (entry.column, entry.coefficient))
        .collect()
}

/// The 150 rows of `poseidon2-permutation-v1` with constant column 0, input
/// lane forms `lanes`, and S-box outputs in the 150 columns from `sbox`.
pub(crate) fn permutation_rows(
    lanes: &[Entries],
    sbox: usize,
    logical_width: usize,
    mut visit: impl FnMut([Entries; MEANINGFUL_PORTS]) -> Result<(), PackageError>,
) -> Result<(), PackageError> {
    if lanes.len() != POSEIDON_WIDTH {
        return Err(PackageError::Malformed("Poseidon2 permutation lanes"));
    }
    let forms: Vec<Form> = std::iter::once(Form::singleton(0, Fq::ONE))
        .chain(lanes.iter().map(|lane| form(lane)))
        .chain((0..SBOX_ROWS).map(|row| Form::singleton(sbox + row, Fq::ONE)))
        .collect();
    let _ = RowScratch::default().visit_rows_until(
        "poseidon2-permutation-v1",
        0,
        &forms,
        logical_width,
        0..SBOX_ROWS,
        |row| {
            visit(row.map(pairs))?;
            Ok(ControlFlow::Continue(()))
        },
    )?;
    Ok(())
}

/// `poseidon2-external-v1` on the 16 single columns `columns`.
pub(crate) fn external_outputs(columns: &[usize], logical_width: usize) -> Result<Vec<Entries>, PackageError> {
    let state: Vec<Form> = columns
        .iter()
        .map(|&column| Form::singleton(column, Fq::ONE))
        .collect();
    Ok(external_layer(&state, logical_width)?
        .iter()
        .map(|output| {
            output
                .entries()
                .into_iter()
                .map(|entry| (entry.column, entry.coefficient))
                .collect()
        })
        .collect())
}
