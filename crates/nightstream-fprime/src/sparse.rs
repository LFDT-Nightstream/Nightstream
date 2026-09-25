//! Runtime values for the Lean-owned sparse row and witness-program IR.

use p3_goldilocks::Goldilocks;

#[derive(Clone, Copy, Debug)]
pub(super) struct SparseTerm {
    pub(super) column: usize,
    pub(super) coefficient: Goldilocks,
}

#[derive(Clone, Debug)]
pub(super) struct SparseCombination {
    pub(super) constant: Goldilocks,
    pub(super) terms: Vec<SparseTerm>,
}

#[derive(Clone, Debug)]
pub(super) struct SparseRow {
    pub(super) row_index: usize,
    pub(super) a: SparseCombination,
    pub(super) b: SparseCombination,
    pub(super) c: SparseCombination,
}

#[derive(Clone, Debug)]
pub(super) struct WitnessInstruction {
    pub(super) row_index: usize,
    pub(super) target: usize,
    pub(super) a: SparseCombination,
    pub(super) b: SparseCombination,
}

pub(super) fn eval_sparse_combination(combination: &SparseCombination, assignment: &[Goldilocks]) -> Goldilocks {
    combination
        .terms
        .iter()
        .fold(combination.constant, |sum, term| {
            sum + term.coefficient * assignment[term.column]
        })
}
