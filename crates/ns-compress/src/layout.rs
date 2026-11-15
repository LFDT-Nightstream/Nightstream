//! Where each claim's cells are in the two polynomials that a proof commits
//! (SPEC.md section 6). `W` holds every witness and every table's counts. `V`
//! holds the lookup inverses of the witnesses that have a table.

use neo_math::{D, K};
use p3_field::PrimeCharacteristicRing;

use crate::akita::{Schedules, Shape};
use crate::key::ROW_VARS;
use crate::{Error, Norm};

/// The variables of the lanes of one ring block.
pub(crate) const LANE_VARS: usize = D.trailing_zeros() as usize;

/// The lookup runs once for each of this many base-field challenges. Four
/// fill the columns of `V`, and with `(N + T)² ≤ q` they put each lookup
/// term at most `1/|K|` (`error_bits`).
pub(crate) const LOOKUPS: usize = 4;
/// `V` has one column for each value of this many variables.
pub(crate) const LOOKUP_VARS: usize = 2;
const _: () = assert!(LOOKUPS <= 1 << LOOKUP_VARS);

/// `2^vars` cells of a polynomial from `offset`, a multiple of `2^vars`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slice {
    pub(crate) offset: usize,
    pub(crate) vars: usize,
}

impl Slice {
    pub(crate) fn len(self) -> usize {
        1 << self.vars
    }

    pub(crate) fn range(self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.len()
    }

    /// The multilinear polynomial that is 1 on this slice and 0 on each other
    /// slice of its size, at `point`: the coordinates from `vars` select the
    /// bits of `offset`.
    pub(crate) fn selector(self, point: &[K]) -> K {
        point
            .iter()
            .enumerate()
            .skip(self.vars)
            .map(|(bit, &coordinate)| {
                if self.offset >> bit & 1 == 1 {
                    coordinate
                } else {
                    K::ONE - coordinate
                }
            })
            .product()
    }
}

/// The cells of one claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cells {
    /// Block `j`, lane `h` of the witness is cell `64j + h` of this slice.
    pub(crate) witness: Slice,
    pub(crate) blocks: usize,
    /// The table `[-bound, bound]` and the slice of its counts: the count of
    /// value `t - bound` is cell `t`.
    pub(crate) table: Option<(u64, Slice)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub(crate) cells: Vec<Cells>,
    /// The variables of the key polynomial that the claims are proved with.
    pub(crate) key_vars: usize,
    /// The variables of `W`, and the length of its prefix that holds cells.
    pub(crate) witness_vars: usize,
    pub(crate) witness_len: usize,
    /// The witnesses with a table are the first `span_len` cells of `W`,
    /// below `2^span_vars`. Cell `x` of them has its inverse for challenge
    /// `k` at cell `k · 2^span_vars + x` of `V`.
    pub(crate) span_vars: usize,
    pub(crate) span_len: usize,
    pub(crate) inverse_vars: usize,
}

fn vars_for(cells: usize) -> usize {
    cells.next_power_of_two().trailing_zeros() as usize
}

impl Layout {
    /// The layout of claims with `blocks` ring blocks and the norm proof
    /// `norm` each, under a key polynomial of `key_blocks` blocks. The
    /// witnesses with a table come first, the larger first, then the other
    /// witnesses, then the counts. One claim or more has a table, so `V` has
    /// cells. `W` and `V` get the variables of the first schedule row of the
    /// key that holds them.
    pub fn new(claims: &[(usize, Norm)], key_blocks: usize) -> Result<Self, Error> {
        let schedules =
            Schedules::of_blocks(key_blocks).ok_or(Error::Layout("no Akita schedule row has a key of this size"))?;
        if claims.iter().any(|&(blocks, _)| blocks == 0) {
            return Err(Error::Layout("a claim has no block"));
        }
        if claims.iter().any(|&(blocks, _)| blocks > key_blocks) {
            return Err(Error::Layout("a claim has more blocks than the key"));
        }
        if !claims
            .iter()
            .any(|&(_, norm)| matches!(norm, Norm::Table { .. }))
        {
            return Err(Error::Layout("no claim has a table"));
        }
        let mut order: Vec<usize> = (0..claims.len()).collect();
        order.sort_by_key(|&claim| {
            let (blocks, norm) = claims[claim];
            (matches!(norm, Norm::Caller), std::cmp::Reverse(vars_for(blocks)))
        });
        let mut cursor = 0usize;
        let mut place = |vars: usize| {
            let offset = cursor.next_multiple_of(1 << vars);
            cursor = offset + (1 << vars);
            Slice { offset, vars }
        };
        let mut witnesses = vec![None; claims.len()];
        let mut span_len = 0;
        for claim in order {
            let (blocks, norm) = claims[claim];
            let slice = place(vars_for(blocks) + LANE_VARS);
            witnesses[claim] = Some(slice);
            if matches!(norm, Norm::Table { .. }) {
                span_len = slice.range().end;
            }
        }
        let cells = claims
            .iter()
            .zip(witnesses)
            .map(|(&(blocks, norm), witness)| {
                let table = match norm {
                    Norm::Table { bound } => {
                        let values = usize::try_from(bound)
                            .ok()
                            .and_then(|bound| bound.checked_mul(2)?.checked_add(1))
                            .filter(|&values| values <= usize::from(u16::MAX))
                            .ok_or(Error::Layout("a table exceeds the code width"))?;
                        Some((bound, place(vars_for(values))))
                    }
                    Norm::Caller => None,
                };
                Ok(Cells {
                    witness: witness.expect("each claim has a slice"),
                    blocks,
                    table,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let span_vars = vars_for(span_len);
        let &(witness_vars, inverse_vars) = schedules
            .rows
            .iter()
            .find(|&&(witness, inverse)| witness >= vars_for(cursor) && inverse >= span_vars + LOOKUP_VARS)
            .ok_or(Error::Layout("no Akita schedule row holds the cells"))?;
        Ok(Self {
            cells,
            key_vars: schedules.key_vars,
            witness_vars,
            witness_len: cursor,
            span_vars,
            span_len,
            inverse_vars,
        })
    }

    /// The schedule row of Akita that a proof of this layout uses.
    pub(crate) fn shape(&self) -> Shape {
        Shape {
            key: self.key_vars,
            witness: self.witness_vars,
            inverse: self.inverse_vars,
        }
    }

    /// The variables of the blocks of the key polynomial.
    pub(crate) fn key_block_vars(&self) -> usize {
        self.key_vars - LANE_VARS - ROW_VARS
    }

    /// The length of the prefix of `V` that holds inverses.
    pub(crate) fn inverse_len(&self) -> usize {
        ((LOOKUPS - 1) << self.span_vars) + self.span_len
    }

    /// The claims with a table: the claim, its witness slice, its bound and
    /// the slice of its counts.
    pub(crate) fn tables(&self) -> impl Iterator<Item = (usize, Slice, u64, Slice)> + '_ {
        self.cells.iter().enumerate().filter_map(|(claim, cells)| {
            cells
                .table
                .map(|(bound, counts)| (claim, cells.witness, bound, counts))
        })
    }

    pub fn witness_vars(&self) -> usize {
        self.witness_vars
    }

    pub fn inverse_vars(&self) -> usize {
        self.inverse_vars
    }

    /// The cells of `W` that hold a witness coefficient.
    pub fn witness_cells(&self) -> usize {
        self.cells.iter().map(|cells| cells.blocks * D).sum()
    }
}

#[cfg(test)]
#[path = "../tests/unit/layout.rs"]
mod tests;
