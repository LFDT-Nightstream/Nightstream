//! The parts of the shrink CCS that `neo-verified` does not own: the
//! serialization and transcript words of a shape, the slots of each part,
//! and the block part of the matrix MLEs in closed form.
//!
//! Owns: the sum of the block template entries over all blocks at a point
//! pair. Does not own the positions (`neo_verified::layout`), the diagonal
//! sums (`neo_verified::diagonal`), the templates (`circuit/block.rs`), the
//! recording or the sum-checks.

use serde::{Deserialize, Serialize};

use crate::circuit::block::entries;
use crate::field::{ext_from_verified, exts_to_verified, verified_eq_table, Ext, Gl};
use neo_verified::circuit::{Kind, KINDS, MATRICES};
use neo_verified::diagonal::{diagonal, Track};
use neo_verified::layout::{cell_variables, place, row_variables, Part, Shape};
use p3_field_v08::PrimeCharacteristicRing;

/// The serde form of `Shape`, for `#[serde(with = "ShapeDef")]`.
#[derive(Serialize, Deserialize)]
#[serde(remote = "Shape")]
pub struct ShapeDef {
    blocks: [usize; KINDS.len()],
    glue_rows: usize,
    glue_cells: usize,
    publics: usize,
}

/// The words of a shape that the shrink transcript binds.
pub(crate) fn words(shape: &Shape) -> Vec<Gl> {
    let mut words: Vec<Gl> = shape
        .blocks
        .iter()
        .map(|&count| Gl::from_usize(count))
        .collect();
    words.extend([shape.glue_rows, shape.glue_cells, shape.publics].map(Gl::from_usize));
    words
}

/// The parts of `kind`'s rows and of its cells.
fn parts(shape: &Shape, kind: Kind) -> ([Part; 2], [Part; 2]) {
    let count = shape.blocks[kind.index()];
    (place(kind.rows(), count).0, place(kind.cells(), count).0)
}

/// The row slots of `kind` per part: the part's start, its stride (one
/// slot per block) and the template row at the start of a slot.
pub(crate) fn row_parts(shape: &Shape, kind: Kind) -> impl Iterator<Item = (usize, usize, usize)> {
    slots(parts(shape, kind).0)
}

/// The cell slots of `kind` per part, as `row_parts`.
pub(crate) fn cell_parts(shape: &Shape, kind: Kind) -> impl Iterator<Item = (usize, usize, usize)> {
    slots(parts(shape, kind).1)
}

fn slots([wide, narrow]: [Part; 2]) -> impl Iterator<Item = (usize, usize, usize)> {
    [(wide.start, wide.stride, 0), (narrow.start, narrow.stride, wide.stride)]
        .into_iter()
        .filter(|&(_, stride, _)| stride > 0)
}

/// `Σ_j ρ^j·M̃_j(rx, ry)` over the block rows. `rx` has `row_variables`
/// coordinates and `ry` has `cell_variables + 1`, low bit first.
pub(crate) fn block_part(shape: &Shape, rx: &[Ext], ry: &[Ext], rho: &[Ext; MATRICES]) -> Ext {
    let n = cell_variables(shape);
    assert_eq!(rx.len(), row_variables(shape));
    assert_eq!(ry.len(), n + 1);
    let w_half = Ext::ONE - ry[n];
    let one_column: Ext = ry[..n].iter().map(|&y| Ext::ONE - y).product::<Ext>() * ry[n];
    // An empty part (stride 0) holds no entry; give it no variables.
    let bits = |stride: usize| {
        if stride == 0 {
            0
        } else {
            stride.trailing_zeros() as usize
        }
    };
    let mut total = Ext::ZERO;
    for kind in KINDS {
        let count = shape.blocks[kind.index()];
        if count == 0 {
            continue;
        }
        let (rows, cells) = parts(shape, kind);
        let row_low = rows.map(|part| verified_eq_table(&rx[..bits(part.stride)]));
        let cell_low = cells.map(|part| verified_eq_table(&ry[..bits(part.stride)]));
        // sums[row part][cell part]; cell part 2 is the constant one.
        let mut sums = [[Ext::ZERO; 3]; 2];
        for entry in entries(kind) {
            let row_part = usize::from(entry.row >= rows[0].stride);
            let local_row = entry.row - row_part * rows[0].stride;
            let (cell_part, cell_value) = match entry.cell {
                None => (2, Ext::ONE),
                Some(cell) => {
                    let part = usize::from(cell >= cells[0].stride);
                    (part, cell_low[part][cell - part * cells[0].stride])
                }
            };
            sums[row_part][cell_part] +=
                rho[entry.matrix] * row_low[row_part][local_row] * cell_value * entry.coefficient;
        }
        for (row_part, row_sums) in sums.iter().enumerate() {
            let row = rows[row_part];
            if row.stride == 0 {
                continue;
            }
            let x = Track {
                point: exts_to_verified(&rx[bits(row.stride)..]),
                offset: row.start / row.stride,
            };
            for (cell_part, &sum) in row_sums.iter().enumerate() {
                if sum == Ext::ZERO {
                    continue;
                }
                let diagonal = if cell_part == 2 {
                    one_column * ext_from_verified(diagonal(std::slice::from_ref(&x), count))
                } else {
                    let cell = cells[cell_part];
                    let y = Track {
                        point: exts_to_verified(&ry[bits(cell.stride)..n]),
                        offset: cell.start / cell.stride,
                    };
                    w_half * ext_from_verified(diagonal(&[x.clone(), y], count))
                };
                total += sum * diagonal;
            }
        }
    }
    total
}
