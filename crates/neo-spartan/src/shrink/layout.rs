//! Where the rows and cells of a recorded program sit in the shrink CCS, and
//! the block part of the matrix MLEs in closed form.
//!
//! Owns: the row and column of every recorded row and wire, and the sum of
//! the block template entries over all blocks at a point pair. Does not own
//! the templates (`circuit/block.rs`), the recording or the sum-checks.
//!
//! - A block kind with `s` rows (or cells) has two parts: a wide one of
//!   `2^⌊log2 s⌋` slots and a narrow one for the rest, rounded up to a power
//!   of two. Template index `k` of block `b` sits at `start + b·stride + k'`
//!   in its part. Parts are placed by decreasing stride, so every part start
//!   is a multiple of its stride.
//! - Glue rows and glue cells follow the blocks.
//! - `z = (w, 1, x)`: `w` on `2^n` cells, then the constant one and the
//!   statement words in the upper half.

use serde::{Deserialize, Serialize};

use crate::circuit::block::{Kind, KINDS, MATRICES};
use crate::circuit::record::Wire;
use crate::field::{eq_table, Ext, Gl};
use p3_field_v08::PrimeCharacteristicRing;

/// The counts of one program's recording. Authority: derived from the
/// program by a shape run; the rows of a run depend only on these and the
/// program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Shape {
    /// Blocks per kind, in `KINDS` order.
    pub(crate) blocks: [usize; KINDS.len()],
    pub(crate) glue_rows: usize,
    pub(crate) glue_cells: usize,
    /// Statement words, the constant one excluded.
    pub(crate) publics: usize,
}

impl Shape {
    pub(crate) fn words(&self) -> Vec<Gl> {
        let mut words: Vec<Gl> = self
            .blocks
            .iter()
            .map(|&count| Gl::from_usize(count))
            .collect();
        words.extend([self.glue_rows, self.glue_cells, self.publics].map(Gl::from_usize));
        words
    }
}

/// One part of one kind: one slot of `stride` per block from `start`.
#[derive(Clone, Copy, Debug, Default)]
struct Part {
    start: usize,
    stride: usize,
}

/// The wide and narrow sizes of `size` template indices.
fn split(size: usize) -> [usize; 2] {
    let wide = 1 << (usize::BITS - 1 - size.leading_zeros());
    let rest = size - wide;
    [wide, if rest == 0 { 0 } else { rest.next_power_of_two() }]
}

/// Place the parts of every kind; returns the parts and their end.
fn place(size: impl Fn(Kind) -> usize, counts: &[usize; KINDS.len()]) -> ([[Part; 2]; KINDS.len()], usize) {
    let mut slots: Vec<(usize, Kind, usize)> = Vec::new();
    for kind in KINDS {
        for (part, stride) in split(size(kind)).into_iter().enumerate() {
            if stride > 0 {
                slots.push((stride, kind, part));
            }
        }
    }
    slots.sort_by(|a, b| b.0.cmp(&a.0));
    let mut parts = [[Part::default(); 2]; KINDS.len()];
    let mut start = 0;
    for (stride, kind, part) in slots {
        parts[kind.index()][part] = Part { start, stride };
        start += stride * counts[kind.index()];
    }
    (parts, start)
}

fn variables(count: usize) -> usize {
    (count.next_power_of_two().trailing_zeros() as usize).max(1)
}

/// The positions of one shape.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub(crate) shape: Shape,
    rows: [[Part; 2]; KINDS.len()],
    cells: [[Part; 2]; KINDS.len()],
    glue_rows: usize,
    glue_cells: usize,
    /// Variables of the row cube.
    pub(crate) row_variables: usize,
    /// Variables of `w`; `z` has one more.
    pub(crate) cell_variables: usize,
}

impl Layout {
    pub(crate) fn new(shape: Shape) -> Self {
        let (rows, glue_rows) = place(Kind::rows, &shape.blocks);
        let (cells, glue_cells) = place(Kind::cells, &shape.blocks);
        Self {
            shape,
            rows,
            cells,
            glue_rows,
            glue_cells,
            row_variables: variables(glue_rows + shape.glue_rows),
            cell_variables: variables((glue_cells + shape.glue_cells).max(shape.publics + 1)),
        }
    }

    fn index(parts: &[Part; 2], b: usize, k: usize) -> usize {
        let wide = parts[0].stride;
        let (part, local) = if k < wide { (parts[0], k) } else { (parts[1], k - wide) };
        part.start + b * part.stride + local
    }

    /// The row of template row `k` of block `b` of `kind`.
    pub(crate) fn block_row(&self, kind: Kind, b: usize, k: usize) -> usize {
        Self::index(&self.rows[kind.index()], b, k)
    }

    /// The column of template cell `k` of block `b` of `kind`.
    pub(crate) fn block_cell(&self, kind: Kind, b: usize, k: usize) -> usize {
        Self::index(&self.cells[kind.index()], b, k)
    }

    pub(crate) fn glue_row(&self, g: usize) -> usize {
        self.glue_rows + g
    }

    /// The column of the constant one.
    pub(crate) fn one(&self) -> usize {
        1 << self.cell_variables
    }

    /// The column of `wire` in `z`.
    pub(crate) fn column(&self, wire: Wire) -> usize {
        match wire {
            Wire::Glue(cell) => self.glue_cells + cell as usize,
            Wire::Block(kind, block, cell) => self.block_cell(kind, block as usize, cell as usize),
            Wire::Public(index) => self.one() + index as usize,
        }
    }

    /// The column of a template entry's cell in block `b`.
    pub(crate) fn entry_column(&self, kind: Kind, b: usize, cell: Option<usize>) -> usize {
        cell.map_or(self.one(), |cell| self.block_cell(kind, b, cell))
    }

    /// `Σ_j ρ^j·M̃_j(rx, ry)` over the block rows. `rx` has `row_variables`
    /// coordinates and `ry` has `cell_variables + 1`, low bit first.
    pub(crate) fn block_part(&self, rx: &[Ext], ry: &[Ext], rho: &[Ext; MATRICES]) -> Ext {
        let n = self.cell_variables;
        assert_eq!(rx.len(), self.row_variables);
        assert_eq!(ry.len(), n + 1);
        let w_half = Ext::ONE - ry[n];
        let one_column: Ext = ry[..n].iter().map(|&y| Ext::ONE - y).product::<Ext>() * ry[n];
        let bits = |stride: usize| stride.trailing_zeros() as usize;
        let mut total = Ext::ZERO;
        for kind in KINDS {
            let count = self.shape.blocks[kind.index()];
            if count == 0 {
                continue;
            }
            let (rows, cells) = (&self.rows[kind.index()], &self.cells[kind.index()]);
            let row_low = rows.map(|part| eq_table(&rx[..bits(part.stride)]));
            let cell_low = cells.map(|part| eq_table(&ry[..bits(part.stride)]));
            // sums[row part][cell part]; cell part 2 is the constant one.
            let mut sums = [[Ext::ZERO; 3]; 2];
            for entry in kind.entries() {
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
