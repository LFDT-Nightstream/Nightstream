//! The parts of the shrink CCS that `neo-verified` does not own: the
//! serialization and transcript words of a shape, the slots of each part,
//! and the block part of the matrix MLEs in closed form.
//!
//! Owns: the sum of the block template entries over all blocks at a point
//! pair. Does not own the positions (`neo_verified::layout`), the templates
//! (`circuit/block.rs`), the recording or the sum-checks.

use serde::{Deserialize, Serialize};

use crate::circuit::block::entries;
use crate::field::{eq_table, Ext, Gl};
use neo_verified::circuit::{Kind, KINDS, MATRICES};
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
        let row_low = rows.map(|part| eq_table(&rx[..bits(part.stride)]));
        let cell_low = cells.map(|part| eq_table(&ry[..bits(part.stride)]));
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
            let x = (&rx[bits(row.stride)..], (row.start / row.stride) as u64);
            for (cell_part, &sum) in row_sums.iter().enumerate() {
                if sum == Ext::ZERO {
                    continue;
                }
                let diagonal = if cell_part == 2 {
                    one_column * diagonal(&[x], count as u64)
                } else {
                    let cell = cells[cell_part];
                    let y = (&ry[bits(cell.stride)..n], (cell.start / cell.stride) as u64);
                    w_half * diagonal(&[x, y], count as u64)
                };
                total += sum * diagonal;
            }
        }
    }
    total
}

/// `Σ_{b < count} Π_tracks eq(point, offset + b)`, by a carry walk over the
/// bits of `b`, low bit first. The state is each track's carry and whether
/// the low bits of `b` are below those of `count`. A track's value must fit
/// in its point's coordinates.
pub(crate) fn diagonal(tracks: &[(&[Ext], u64)], count: u64) -> Ext {
    let bits = tracks
        .iter()
        .map(|(point, offset)| point.len().max(64 - offset.leading_zeros() as usize))
        .chain(std::iter::once(64 - count.leading_zeros() as usize))
        .max()
        .unwrap_or(0);
    let carries = 1usize << tracks.len();
    // states[carry bits][below]
    let mut states = vec![[Ext::ZERO; 2]; carries];
    states[0][0] = Ext::ONE;
    for t in 0..bits {
        let count_bit = (count >> t) & 1;
        let mut next = vec![[Ext::ZERO; 2]; carries];
        for (carry, values) in states.iter().enumerate() {
            for (below, &value) in values.iter().enumerate() {
                if value == Ext::ZERO {
                    continue;
                }
                'bit: for b in 0..2u64 {
                    let next_below = match b.cmp(&count_bit) {
                        std::cmp::Ordering::Less => 1,
                        std::cmp::Ordering::Greater => 0,
                        std::cmp::Ordering::Equal => below,
                    };
                    let mut factor = value;
                    let mut next_carry = 0;
                    for (i, &(point, offset)) in tracks.iter().enumerate() {
                        let sum = ((offset >> t) & 1) + b + ((carry >> i) & 1) as u64;
                        let bit = sum & 1;
                        next_carry |= ((sum >> 1) as usize) << i;
                        factor *= match point.get(t) {
                            Some(&x) if bit == 1 => x,
                            Some(&x) => Ext::ONE - x,
                            None if bit == 1 => continue 'bit,
                            None => Ext::ONE,
                        };
                    }
                    next[next_carry][next_below] += factor;
                }
            }
        }
        states = next;
    }
    states[0][1]
}
