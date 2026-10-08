//! Uniform blocks: a fixed set of rows over the block's own cells.
//!
//! Owns: the block kinds (a Poseidon2 permutation, a ring product), their
//! sizes, inputs and outputs, and their rows as entries of the eight CCS
//! matrices `X, A_0, B_0, A_1, B_1, A_2, B_2, C`. A row holds when
//! `(X·z)^7 + Σ_i (A_i·z)·(B_i·z) − C·z = 0`. The first `inputs` cells of a
//! block are its inputs; its outputs are the cells in `outputs`.

use std::ops::Range;
use std::sync::OnceLock;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;

use super::{poseidon2, ring_mul};
use crate::field::Gl;

pub(crate) const MATRICES: usize = 8;
pub(crate) const X: usize = 0;
pub(crate) const A0: usize = 1;
pub(crate) const B0: usize = 2;
pub(crate) const C: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kind {
    Permutation,
    RingProduct,
}

pub(crate) const KINDS: [Kind; 2] = [Kind::Permutation, Kind::RingProduct];

/// One nonzero of a block row: the matrix, the template row, the block cell
/// (`None` for the constant one) and the coefficient.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    pub(crate) matrix: usize,
    pub(crate) row: usize,
    pub(crate) cell: Option<usize>,
    pub(crate) coefficient: Gl,
}

impl Kind {
    pub(crate) fn index(self) -> usize {
        self as usize
    }

    pub(crate) fn cells(self) -> usize {
        match self {
            Kind::Permutation => poseidon2::CELLS,
            Kind::RingProduct => ring_mul::CELLS,
        }
    }

    pub(crate) fn rows(self) -> usize {
        match self {
            Kind::Permutation => poseidon2::ROWS,
            Kind::RingProduct => ring_mul::ROWS,
        }
    }

    pub(crate) fn inputs(self) -> usize {
        match self {
            Kind::Permutation => WIDTH,
            Kind::RingProduct => 2 * neo_math::D,
        }
    }

    pub(crate) fn outputs(self) -> Range<usize> {
        match self {
            Kind::Permutation => poseidon2::OUTPUT..poseidon2::CELLS,
            Kind::RingProduct => ring_mul::OUTPUT..ring_mul::CELLS,
        }
    }

    pub(crate) fn entries(self) -> &'static [Entry] {
        static ENTRIES: OnceLock<[Vec<Entry>; 2]> = OnceLock::new();
        &ENTRIES.get_or_init(|| [poseidon2::entries(), ring_mul::entries()])[self.index()]
    }

    /// The cell values of one block with these inputs.
    pub(crate) fn trace(self, inputs: &[Gl]) -> Vec<Gl> {
        assert_eq!(inputs.len(), self.inputs());
        match self {
            Kind::Permutation => poseidon2::trace(&std::array::from_fn(|lane| inputs[lane])),
            Kind::RingProduct => ring_mul::trace(&inputs[..neo_math::D], &inputs[neo_math::D..]),
        }
    }

    /// The first row of `cells` that fails, if any.
    #[cfg(test)]
    pub(crate) fn failure(self, cells: &[Gl]) -> Option<usize> {
        use p3_field_v08::PrimeCharacteristicRing;
        let mut sides = vec![[Gl::ZERO; MATRICES]; self.rows()];
        for entry in self.entries() {
            let value = entry.cell.map_or(Gl::ONE, |cell| cells[cell]);
            sides[entry.row][entry.matrix] += entry.coefficient * value;
        }
        sides.iter().position(|side| {
            side[X].exp_const_u64::<7>() + side[1] * side[2] + side[3] * side[4] + side[5] * side[6] != side[C]
        })
    }
}
