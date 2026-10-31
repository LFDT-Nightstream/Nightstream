//! Where the rows and cells of a recorded program sit in the shrink CCS.
//!
//! - A block kind with `s` rows (or cells) has two parts: a wide one of
//!   `2^⌊log2 s⌋` slots and a narrow one for the rest, rounded up to a power
//!   of two. Template index `k` of block `b` sits at `start + b·stride + k'`
//!   in its part. Every part start is a multiple of its stride.
//! - Glue rows and glue cells follow the blocks.
//! - `z = (w, 1, x)`: `w` on `2^n` cells, then the constant one and the
//!   statement words in the upper half.

use crate::circuit::{Kind, Wire, KINDS};

/// The counts of one program's recording. Authority: derived from the
/// program by a shape run; the rows of a run depend only on these and the
/// program.
#[derive(Clone, Copy, PartialEq)]
pub struct Shape {
    /// Blocks per kind, in `KINDS` order.
    pub blocks: [usize; KINDS.len()],
    pub glue_rows: usize,
    pub glue_cells: usize,
    /// ClaimWords words, the constant one excluded.
    pub publics: usize,
}

/// One part of one kind: one slot of `stride` per block from `start`.
#[derive(Clone, Copy)]
pub struct Part {
    pub start: usize,
    pub stride: usize,
}

pub fn split(size: usize) -> [usize; 2] {
    let wide = two_pow(floor_log2(size));
    let rest = size.checked_sub(wide).unwrap();
    if rest == 0 {
        [wide, 0]
    } else {
        [wide, two_pow(ceil_log2(rest))]
    }
}

pub fn variables(count: usize) -> usize {
    if count <= 2 {
        1
    } else {
        ceil_log2(count) as usize
    }
}

fn floor_log2(n: usize) -> u32 {
    usize::BITS
        .checked_sub(1)
        .unwrap()
        .checked_sub(n.leading_zeros())
        .unwrap()
}

fn ceil_log2(n: usize) -> u32 {
    usize::BITS
        .checked_sub(n.checked_sub(1).unwrap().leading_zeros())
        .unwrap()
}

fn two_pow(k: u32) -> usize {
    // `wrapping_shl` masks the shift, so a shift of `usize::BITS` or more panics here.
    let _fits = usize::BITS.checked_sub(1).unwrap().checked_sub(k).unwrap();
    1usize.wrapping_shl(k)
}

/// The parts of `count` blocks of `size` rows or cells, wide then narrow,
/// and their end. With one block kind, this is the order of decreasing
/// stride that keeps every part start a multiple of its stride.
pub fn place(size: usize, count: usize) -> ([Part; 2], usize) {
    let [wide, narrow] = split(size);
    let start = wide.checked_mul(count).unwrap();
    let end = start
        .checked_add(narrow.checked_mul(count).unwrap())
        .unwrap();
    ([Part { start: 0, stride: wide }, Part { start, stride: narrow }], end)
}

/// The slot of template index `k` of block `b` in `place(size, count)`.
pub fn slot(size: usize, count: usize, b: usize, k: usize) -> usize {
    let ([wide, narrow], _) = place(size, count);
    let (part, local) = if k < wide.stride {
        (wide, k)
    } else {
        (narrow, k.checked_sub(wide.stride).unwrap())
    };
    part.start
        .checked_add(b.checked_mul(part.stride).unwrap())
        .unwrap()
        .checked_add(local)
        .unwrap()
}

/// The row of template row `k` of block `b` of `kind`.
pub fn block_row(shape: &Shape, kind: Kind, b: usize, k: usize) -> usize {
    let [count] = shape.blocks;
    slot(kind.rows(), count, b, k)
}

/// The column of template cell `k` of block `b` of `kind`.
pub fn block_cell(shape: &Shape, kind: Kind, b: usize, k: usize) -> usize {
    let [count] = shape.blocks;
    slot(kind.cells(), count, b, k)
}

/// The first glue row: with one block kind, the end of its row parts.
fn rows_end(shape: &Shape) -> usize {
    let [count] = shape.blocks;
    let (_, end) = place(Kind::Permutation.rows(), count);
    end
}

/// The first glue cell: with one block kind, the end of its cell parts.
fn cells_end(shape: &Shape) -> usize {
    let [count] = shape.blocks;
    let (_, end) = place(Kind::Permutation.cells(), count);
    end
}

pub fn glue_row(shape: &Shape, g: usize) -> usize {
    rows_end(shape).checked_add(g).unwrap()
}

/// Variables of the row cube.
pub fn row_variables(shape: &Shape) -> usize {
    variables(rows_end(shape).checked_add(shape.glue_rows).unwrap())
}

/// Variables of `w`; `z` has one more.
pub fn cell_variables(shape: &Shape) -> usize {
    let cells = cells_end(shape).checked_add(shape.glue_cells).unwrap();
    variables(cells.max(shape.publics.checked_add(1).unwrap()))
}

/// The column of the constant one.
pub fn one(shape: &Shape) -> usize {
    two_pow(cell_variables(shape) as u32)
}

/// The column of `wire` in `z`.
pub fn column(shape: &Shape, wire: Wire) -> usize {
    match wire {
        Wire::Glue(cell) => cells_end(shape).checked_add(cell as usize).unwrap(),
        Wire::Block(kind, block, cell) => block_cell(shape, kind, block as usize, cell as usize),
        Wire::Public(index) => one(shape).checked_add(index as usize).unwrap(),
    }
}

/// The column of a template entry's cell in block `b`; `None` is the
/// constant one.
pub fn entry_column(shape: &Shape, kind: Kind, b: usize, cell: Option<usize>) -> usize {
    match cell {
        None => one(shape),
        Some(cell) => block_cell(shape, kind, b, cell),
    }
}
