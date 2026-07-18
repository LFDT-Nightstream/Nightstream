//! Exact sparse-row replay for the active `y_zcol` projection audit.
//!
//! Owns: one bounded index of selected source-R1CS rows and normalized A/B/C
//! equality against caller-supplied linear combinations.
//!
//! Does not own: stage selection, trace semantics, protocol authority,
//! constraint replacement, or row removal.
//!
//! Emits constraints: no.
//!
//! | Selected row family | Mathematical obligation | Physical check |
//! |---|---|---|
//! | selected source row | `(A_r z)(B_r z) = C_r z` | exact normalized A/B/C terms |
//! | compact-row exclusion | selected arithmetic is ordinary sparse R1CS | no seeded/geometric owner overlaps |

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use neo_ccs::{CcsMatrix, CscMat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::Lc;
use crate::frontends::r1cs_f_prime::ivc::R1csIvcError;
use crate::frontends::r1cs_f_prime::SparseR1cs;

use super::super::invalid;

type SparseRow = Vec<(usize, F)>;

/// One indexed, normalized source-R1CS equation retained after replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolProjectionRowAudit {
    index: usize,
    ports: [SparseRow; 3],
}

impl PiRlcYZcolProjectionRowAudit {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn a(&self) -> &[(usize, F)] {
        &self.ports[0]
    }

    pub fn b(&self) -> &[(usize, F)] {
        &self.ports[1]
    }

    pub fn c(&self) -> &[(usize, F)] {
        &self.ports[2]
    }
}

/// A bounded row-major view constructed by scanning each sparse matrix once.
pub(super) struct SelectedR1csRows {
    row_indices: HashMap<usize, usize>,
    row_numbers: Vec<usize>,
    rows: Vec<[SparseRow; 3]>,
}

impl SelectedR1csRows {
    pub(super) fn recover(arm: &SparseR1cs, ranges: &[Range<usize>]) -> Result<Self, R1csIvcError> {
        if ranges.is_empty() || ranges.iter().any(Range::is_empty) {
            return Err(invalid(
                "PiRLC y_zcol projection row selection contains no complete interval",
            ));
        }
        let mut selected = ranges
            .iter()
            .flat_map(|range| range.clone())
            .collect::<Vec<_>>();
        selected.sort_unstable();
        if selected.last().is_some_and(|&row| row >= arm.n) {
            return Err(invalid("PiRLC y_zcol projection row lies outside the source R1CS"));
        }
        if selected.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid(
                "PiRLC y_zcol projection row families overlap instead of forming disjoint intervals",
            ));
        }
        let row_indices = selected
            .iter()
            .enumerate()
            .map(|(index, &row)| (row, index))
            .collect::<HashMap<_, _>>();
        let mut rows = (0..selected.len())
            .map(|_| std::array::from_fn(|_| Vec::new()))
            .collect::<Vec<[SparseRow; 3]>>();

        for (port, matrix) in [&arm.a, &arm.b, &arm.c].into_iter().enumerate() {
            reject_compact_overlap(matrix, &selected, port)?;
            match matrix {
                CcsMatrix::Identity { n } => {
                    for (&row, &index) in &row_indices {
                        if row >= *n {
                            return Err(invalid(format!(
                                "PiRLC y_zcol selected row {row} exceeds identity port {port} dimension {n}"
                            )));
                        }
                        rows[index][port].push((row, F::ONE));
                    }
                }
                CcsMatrix::Csc(csc) | CcsMatrix::CscWithSeededPhi81 { csc, .. } => {
                    index_csc(csc, port, &row_indices, &mut rows);
                }
            }
        }
        for ports in &mut rows {
            for row in ports {
                *row = normalize_sparse_row(std::mem::take(row));
            }
        }

        Ok(Self {
            row_indices,
            row_numbers: selected,
            rows,
        })
    }

    pub(super) fn expect(&self, row: usize, a: &Lc, b: &Lc, c: &Lc, owner: &str) -> Result<(), R1csIvcError> {
        let index = self
            .row_indices
            .get(&row)
            .copied()
            .ok_or_else(|| invalid(format!("{owner} row {row} was not selected for sparse replay")))?;
        let expected = [normalized_lc(a), normalized_lc(b), normalized_lc(c)];
        for port in 0..3 {
            if self.rows[index][port] != expected[port] {
                return Err(invalid(format!(
                    "{owner} row {row} port {} differs from its exact emitted equation",
                    ["A", "B", "C"][port]
                )));
            }
        }
        Ok(())
    }

    pub(super) fn into_indexed_rows(self) -> Vec<PiRlcYZcolProjectionRowAudit> {
        self.row_numbers
            .into_iter()
            .zip(self.rows)
            .map(|(index, ports)| PiRlcYZcolProjectionRowAudit { index, ports })
            .collect()
    }
}

pub(super) fn normalized_lc(lc: &Lc) -> SparseRow {
    let mut terms = lc.terms.clone();
    if lc.constant != F::ZERO {
        terms.push((0, lc.constant));
    }
    normalize_sparse_row(terms)
}

fn normalize_sparse_row(row: SparseRow) -> SparseRow {
    let mut terms = BTreeMap::<usize, F>::new();
    for (column, coefficient) in row {
        if coefficient != F::ZERO {
            *terms.entry(column).or_insert(F::ZERO) += coefficient;
        }
    }
    terms
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect()
}

pub(super) fn same_lc(left: &Lc, right: &Lc) -> bool {
    normalized_lc(left) == normalized_lc(right)
}

fn index_csc(csc: &CscMat<F>, port: usize, row_indices: &HashMap<usize, usize>, rows: &mut [[SparseRow; 3]]) {
    for column in 0..csc.ncols {
        for entry in csc.column_range(column) {
            let row = csc.row_index(entry);
            let Some(&index) = row_indices.get(&row) else {
                continue;
            };
            let coefficient = csc.vals[entry];
            if coefficient != F::ZERO {
                rows[index][port].push((column, coefficient));
            }
        }
    }
}

fn reject_compact_overlap(matrix: &CcsMatrix<F>, selected: &[usize], port: usize) -> Result<(), R1csIvcError> {
    for block in matrix.seeded_phi81_blocks() {
        if selected
            .iter()
            .any(|&row| (block.row_start()..block.row_end()).contains(&row))
        {
            return Err(invalid(format!(
                "PiRLC y_zcol arithmetic overlaps a compact seeded-Phi81 block in port {}",
                ["A", "B", "C"][port]
            )));
        }
    }
    for run in matrix.geometric_runs() {
        if selected.binary_search(&run.row()).is_ok() {
            return Err(invalid(format!(
                "PiRLC y_zcol arithmetic overlaps a compact geometric row in port {}",
                ["A", "B", "C"][port]
            )));
        }
    }
    Ok(())
}
