//! Lean `CircuitRelation.sboxInterface`.

use std::ops::ControlFlow;

use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;

use super::form::entries;
use super::template::RowScratch;
use super::{external_layer, Entry, Form, MatrixRun, PackageError, MEANINGFUL_PORTS, POSEIDON_WIDTH};
use crate::package::SBOX_ROWS;

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
