//! The prover's view of uniform blocks (`neo_verified::block`): each kind's entries with
//! Plonky3 0.8 coefficients, built once, and the cell values of a block.

use std::sync::OnceLock;

pub(crate) use neo_verified::block::{C, X};
use neo_verified::circuit::{Kind, KINDS};

use super::poseidon2;
use crate::field::{from_verified, Gl};

/// One nonzero of a block row: the matrix, the template row, the block cell
/// (`None` for the constant one) and the coefficient.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    pub(crate) matrix: usize,
    pub(crate) row: usize,
    pub(crate) cell: Option<usize>,
    pub(crate) coefficient: Gl,
}

pub(crate) fn entries(kind: Kind) -> &'static [Entry] {
    static ENTRIES: OnceLock<[Vec<Entry>; KINDS.len()]> = OnceLock::new();
    &ENTRIES.get_or_init(|| {
        KINDS.map(|kind| {
            neo_verified::block::entries(kind)
                .into_iter()
                .map(|entry| Entry {
                    matrix: entry.matrix,
                    row: entry.row,
                    cell: entry.cell,
                    coefficient: from_verified(entry.coefficient),
                })
                .collect()
        })
    })[kind.index()]
}

/// The cell values of one block with these inputs.
pub(crate) fn trace(kind: Kind, inputs: &[Gl]) -> Vec<Gl> {
    assert_eq!(inputs.len(), neo_verified::block::inputs(kind));
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
