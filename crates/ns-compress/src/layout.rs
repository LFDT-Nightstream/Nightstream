//! Where each claim's cells are in the two polynomials that a proof commits (SPEC.md section 6).

use neo_math::{D, K};
use p3_field::PrimeCharacteristicRing;

use crate::akita::{Schedules, Shape};
use crate::key::ROW_VARS;
use crate::{Error, Norm};

pub(crate) const LANE_VARS: usize = D.trailing_zeros() as usize;

pub(crate) const LOOKUPS: usize = 4;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cells {
    pub(crate) witness: Slice,
    pub(crate) blocks: usize,
    pub(crate) table: Option<(u64, Slice)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub(crate) cells: Vec<Cells>,
    pub(crate) key_vars: usize,
    pub(crate) witness_vars: usize,
    pub(crate) witness_len: usize,
    pub(crate) table_witness_vars: usize,
    pub(crate) table_witness_len: usize,
    pub(crate) inverse_vars: usize,
}

fn vars_for(cells: usize) -> usize {
    cells.next_power_of_two().trailing_zeros() as usize
}

impl Layout {
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
            (matches!(norm, Norm::BoundByCaller), std::cmp::Reverse(vars_for(blocks)))
        });
        let mut cursor = 0usize;
        let mut place = |vars: usize| {
            let offset = cursor.next_multiple_of(1 << vars);
            cursor = offset + (1 << vars);
            Slice { offset, vars }
        };
        let mut witnesses = vec![None; claims.len()];
        let mut table_witness_len = 0;
        for claim in order {
            let (blocks, norm) = claims[claim];
            let slice = place(vars_for(blocks) + LANE_VARS);
            witnesses[claim] = Some(slice);
            if matches!(norm, Norm::Table { .. }) {
                table_witness_len = slice.range().end;
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
                    Norm::BoundByCaller => None,
                };
                Ok(Cells {
                    witness: witness.expect("each claim has a slice"),
                    blocks,
                    table,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let table_witness_vars = vars_for(table_witness_len);
        let &(witness_vars, inverse_vars) = schedules
            .rows
            .iter()
            .find(|&&(witness, inverse)| witness >= vars_for(cursor) && inverse >= table_witness_vars)
            .ok_or(Error::Layout("no Akita schedule row holds the cells"))?;
        Ok(Self {
            cells,
            key_vars: schedules.key_vars,
            witness_vars,
            witness_len: cursor,
            table_witness_vars,
            table_witness_len,
            inverse_vars,
        })
    }

    pub(crate) fn shape(&self) -> Shape {
        Shape {
            key: self.key_vars,
            witness: self.witness_vars,
            inverse: self.inverse_vars,
        }
    }

    pub(crate) fn key_block_vars(&self) -> usize {
        self.key_vars - LANE_VARS - ROW_VARS
    }

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

    pub fn witness_cells(&self) -> usize {
        self.cells.iter().map(|cells| cells.blocks * D).sum()
    }
}

#[cfg(test)]
#[path = "../tests/unit/layout.rs"]
mod tests;
