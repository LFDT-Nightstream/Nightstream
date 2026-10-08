//! Where the rows and cells of a recorded program sit in the shrink CCS, and
//! the permutation-block part of the matrix MLEs in closed form.
//!
//! Owns: the row and column of every recorded row and wire, the block
//! template as entries of the eight CCS matrices, and the sum of those
//! entries over all blocks at a point pair. Does not own the recording or the
//! sum-checks.
//!
//! - Matrices: `X, A_0, B_0, A_1, B_1, A_2, B_2, C`; row `r` holds when
//!   `(X·z)_r^7 + Σ_i (A_i·z)_r·(B_i·z)_r − (C·z)_r = 0`.
//! - Block `b < P`, template row or cell `k`: `128·b + k` for `k < 128`
//!   (region A), else `128·P + 64·b + (k − 128)` (region B). A block has 166
//!   rows and 182 cells, so both regions fit.
//! - Glue row `g` sits at `192·P + g`, glue cell `i` at `192·P + i`.
//! - `z = (w, 1, x)`: `w` on `2^n` cells, then the constant one and the
//!   statement words in the upper half.

use serde::{Deserialize, Serialize};

use crate::circuit::poseidon2::{self, OUTPUT, SBOXES};
use crate::circuit::record::Wire;
use crate::field::{eq_table, Ext, Gl};
use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use p3_field_v08::PrimeCharacteristicRing;

/// `X, A_0, B_0, A_1, B_1, A_2, B_2, C`.
pub(crate) const MATRICES: usize = 8;
pub(crate) const X: usize = 0;
pub(crate) const C: usize = 7;
/// Template rows: the S-boxes, then the outputs.
const BLOCK_ROWS: usize = SBOXES + WIDTH;
const WIDE: usize = 128;
const NARROW: usize = 64;
const STRIDE: usize = WIDE + NARROW;

/// The counts of one program's recording. Authority: derived from the
/// program by a shape run; the rows of a run depend only on these and the
/// program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Shape {
    pub(crate) blocks: usize,
    pub(crate) glue_rows: usize,
    pub(crate) glue_cells: usize,
    /// Statement words, the constant one excluded.
    pub(crate) publics: usize,
}

impl Shape {
    pub(crate) fn words(&self) -> Vec<Gl> {
        [self.blocks, self.glue_rows, self.glue_cells, self.publics]
            .map(Gl::from_usize)
            .to_vec()
    }
}

/// One nonzero of a block row: the matrix, the template row, the block cell
/// (`None` for the constant one) and the coefficient.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    pub(crate) matrix: usize,
    pub(crate) row: usize,
    pub(crate) cell: Option<usize>,
    pub(crate) coefficient: Gl,
}

/// Every nonzero of the block template. S-box row `r`: `X` is the S-box input
/// form, `C` the S-box output cell. Output row `150 + i`: `C` is the output
/// cell minus its form.
pub(crate) fn entries() -> &'static [Entry] {
    static ENTRIES: std::sync::OnceLock<Vec<Entry>> = std::sync::OnceLock::new();
    ENTRIES.get_or_init(|| {
        let template = poseidon2::template();
        let mut entries = Vec::new();
        let mut push = |matrix, row, form: &poseidon2::Linear, sign: Gl| {
            for &(cell, coefficient) in &form.terms {
                entries.push(Entry {
                    matrix,
                    row,
                    cell: Some(cell),
                    coefficient: sign * coefficient,
                });
            }
            if form.constant != Gl::ZERO {
                entries.push(Entry {
                    matrix,
                    row,
                    cell: None,
                    coefficient: sign * form.constant,
                });
            }
        };
        for (row, form) in template.sbox_inputs.iter().enumerate() {
            push(X, row, form, Gl::ONE);
        }
        for (lane, form) in template.outputs.iter().enumerate() {
            push(C, SBOXES + lane, form, -Gl::ONE);
        }
        for row in 0..BLOCK_ROWS {
            let cell = if row < SBOXES {
                WIDTH + row
            } else {
                OUTPUT + row - SBOXES
            };
            entries.push(Entry {
                matrix: C,
                row,
                cell: Some(cell),
                coefficient: Gl::ONE,
            });
        }
        entries
    })
}

/// The positions of one shape.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub(crate) shape: Shape,
    /// Variables of the row cube.
    pub(crate) row_variables: usize,
    /// Variables of `w`; `z` has one more.
    pub(crate) cell_variables: usize,
}

fn variables(count: usize) -> usize {
    (count.next_power_of_two().trailing_zeros() as usize).max(1)
}

impl Layout {
    pub(crate) fn new(shape: Shape) -> Self {
        let blocks = STRIDE * shape.blocks;
        Self {
            shape,
            row_variables: variables(blocks + shape.glue_rows),
            cell_variables: variables((blocks + shape.glue_cells).max(shape.publics + 1)),
        }
    }

    /// The row or cell of template index `k` of block `b`.
    pub(crate) fn block_index(&self, b: usize, k: usize) -> usize {
        if k < WIDE {
            WIDE * b + k
        } else {
            WIDE * self.shape.blocks + NARROW * b + (k - WIDE)
        }
    }

    pub(crate) fn glue_row(&self, g: usize) -> usize {
        STRIDE * self.shape.blocks + g
    }

    /// The column of `wire` in `z`.
    pub(crate) fn column(&self, wire: Wire) -> usize {
        match wire {
            Wire::Glue(cell) => STRIDE * self.shape.blocks + cell as usize,
            Wire::Block(block, cell) => self.block_index(block as usize, cell as usize),
            Wire::Public(index) => (1 << self.cell_variables) + index as usize,
        }
    }

    /// The column of a template entry's cell in block `b`.
    pub(crate) fn entry_column(&self, b: usize, cell: Option<usize>) -> usize {
        cell.map_or(1 << self.cell_variables, |cell| self.block_index(b, cell))
    }

    /// `Σ_j ρ^j·M̃_j(rx, ry)` over the block rows. `rx` has `row_variables`
    /// coordinates and `ry` has `cell_variables + 1`, low bit first.
    pub(crate) fn block_part(&self, rx: &[Ext], ry: &[Ext], rho: &[Ext; MATRICES]) -> Ext {
        let n = self.cell_variables;
        assert_eq!(rx.len(), self.row_variables);
        assert_eq!(ry.len(), n + 1);
        let blocks = self.shape.blocks as u64;
        if blocks == 0 {
            return Ext::ZERO;
        }
        // Region A of rows or cells: low 7 bits, block offset 0; region B: low
        // 6 bits, block offset 2P (128·P / 64).
        let split = |region: usize| if region == 0 { (7, 0) } else { (6, 2 * blocks) };
        let row_low = [eq_table(&rx[..7]), eq_table(&rx[..6])];
        let cell_low = [eq_table(&ry[..7]), eq_table(&ry[..6])];
        let w_half = Ext::ONE - ry[n];
        let one_column: Ext = ry[..n].iter().map(|&y| Ext::ONE - y).product::<Ext>() * ry[n];
        // sums[row region][cell region]; cell region 2 is the constant one.
        let mut sums = [[Ext::ZERO; 3]; 2];
        for entry in entries() {
            let row_region = usize::from(entry.row >= WIDE);
            let local_row = entry.row - WIDE * row_region;
            let (cell_region, cell_value) = match entry.cell {
                None => (2, Ext::ONE),
                Some(cell) => {
                    let region = usize::from(cell >= WIDE);
                    (region, cell_low[region][cell - WIDE * region])
                }
            };
            sums[row_region][cell_region] +=
                rho[entry.matrix] * row_low[row_region][local_row] * cell_value * entry.coefficient;
        }
        let mut total = Ext::ZERO;
        for (row_region, row_sums) in sums.iter().enumerate() {
            let (row_bits, row_offset) = split(row_region);
            let x = (&rx[row_bits..], row_offset);
            for (cell_region, &sum) in row_sums.iter().enumerate() {
                let diagonal = if cell_region == 2 {
                    one_column * diagonal(&[x], blocks)
                } else {
                    let (cell_bits, cell_offset) = split(cell_region);
                    w_half * diagonal(&[x, (&ry[cell_bits..n], cell_offset)], blocks)
                };
                total += sum * diagonal;
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
