//! Uniform blocks: a fixed set of rows over the block's own cells.
//!
//! Owns: the inputs and outputs of each block kind
//! (`neo_verified::circuit::Kind`), and its rows as entries of the eight CCS
//! matrices `X, A_0, B_0, A_1, B_1, A_2, B_2, C`. A row holds when
//! `(X·z)^7 + Σ_i (A_i·z)·(B_i·z) − C·z = 0`. The first `inputs` cells of a
//! block are its inputs; its outputs are the cells in `outputs`.

use std::ops::Range;
use std::sync::OnceLock;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_verified::circuit::{Kind, CELLS};

use super::poseidon2;
use crate::field::Gl;

pub(crate) const X: usize = 0;
pub(crate) const C: usize = 7;

/// One nonzero of a block row: the matrix, the template row, the block cell
/// (`None` for the constant one) and the coefficient.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    pub(crate) matrix: usize,
    pub(crate) row: usize,
    pub(crate) cell: Option<usize>,
    pub(crate) coefficient: Gl,
}

pub(crate) fn inputs(kind: Kind) -> usize {
    match kind {
        Kind::Permutation => WIDTH,
    }
}

pub(crate) fn outputs(kind: Kind) -> Range<usize> {
    match kind {
        Kind::Permutation => poseidon2::OUTPUT..CELLS,
    }
}

pub(crate) fn entries(kind: Kind) -> &'static [Entry] {
    static ENTRIES: OnceLock<[Vec<Entry>; 1]> = OnceLock::new();
    &ENTRIES.get_or_init(|| [poseidon2::entries()])[kind.index()]
}

/// The cell values of one block with these inputs.
pub(crate) fn trace(kind: Kind, inputs: &[Gl]) -> Vec<Gl> {
    assert_eq!(inputs.len(), self::inputs(kind));
    match kind {
        Kind::Permutation => poseidon2::trace(&std::array::from_fn(|lane| inputs[lane])),
    }
}

/// The first row of `cells` that fails, if any.
#[cfg(test)]
pub(crate) fn failure(kind: Kind, cells: &[Gl]) -> Option<usize> {
    use neo_verified::circuit::MATRICES;
    use p3_field_v08::PrimeCharacteristicRing;
    let mut sides = vec![[Gl::ZERO; MATRICES]; kind.rows()];
    for entry in entries(kind) {
        let value = entry.cell.map_or(Gl::ONE, |cell| cells[cell]);
        sides[entry.row][entry.matrix] += entry.coefficient * value;
    }
    sides.iter().position(|side| {
        side[X].exp_const_u64::<7>() + side[1] * side[2] + side[3] * side[4] + side[5] * side[6] != side[C]
    })
}
