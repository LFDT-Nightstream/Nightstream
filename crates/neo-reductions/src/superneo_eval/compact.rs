//! Compact explicit-matrix export for accelerators.

use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use super::{DenseBlockStore, RealBlockStorage, SuperneoMatrixCache, SuperneoZBlocks};

#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum SuperneoCompactRowOffsets<'a> {
    Empty,
    U16Chunked {
        chunk_offsets: &'a [u32],
        local_offsets: &'a [u16],
        chunk_rows: usize,
    },
    U24(&'a [u8]),
    U32(&'a [u32]),
}

#[doc(hidden)]
pub struct SuperneoCompactDeviceParts<'a> {
    pub row_offsets: SuperneoCompactRowOffsets<'a>,
    pub row_blocks: &'a [u32],
    pub dense_row_blocks: &'a [[u32; 2]],
    pub dense_offsets: &'a [u32],
    pub dense_locals: &'a [u8],
    pub dense_coefficients: &'a [F],
    pub geometric_row_offsets: SuperneoCompactRowOffsets<'a>,
    pub geometric_runs: &'a [[u64; 3]],
    pub identity: bool,
}

impl SuperneoZBlocks {
    pub fn signed_unit_masks(&self) -> Option<(&[u64], &[u64])> {
        match &self.re {
            RealBlockStorage::SignedUnit { positive, negative } => Some((positive, negative)),
            _ => None,
        }
    }

    #[doc(hidden)]
    pub fn signed_digit_masks(&self, base: u32) -> Option<Vec<u64>> {
        if !(2..=4).contains(&base) || !self.imag_all_zero() {
            return None;
        }
        let magnitudes = (base - 1) as usize;
        let mut masks = vec![0; self.block_len() * 2 * magnitudes];
        if let Some((positive, negative)) = self.signed_unit_masks() {
            for (block, target) in masks.chunks_exact_mut(2 * magnitudes).enumerate() {
                target[0] = positive[block];
                target[1] = negative[block];
            }
            return Some(masks);
        }
        for block in 0..self.block_len() {
            for lane in 0..D {
                let value = self.real_coefficient(block, lane);
                if value == F::ZERO {
                    continue;
                }
                let mut matched = false;
                for magnitude in 1..=magnitudes {
                    let positive = F::from_u64(magnitude as u64);
                    let negative = F::ZERO - positive;
                    let sign = if value == positive {
                        Some(0)
                    } else if value == negative {
                        Some(1)
                    } else {
                        None
                    };
                    if let Some(sign) = sign {
                        let index = (block * magnitudes + magnitude - 1) * 2 + sign;
                        masks[index] |= 1u64 << lane;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    return None;
                }
            }
        }
        Some(masks)
    }
}

impl SuperneoMatrixCache {
    #[doc(hidden)]
    pub fn compact_device_parts(&self) -> Option<SuperneoCompactDeviceParts<'_>> {
        let row_offsets = match &self.row_offsets {
            super::RowOffsetStore::Empty => SuperneoCompactRowOffsets::Empty,
            super::RowOffsetStore::U16Chunked {
                chunk_offsets,
                local_offsets,
            } => SuperneoCompactRowOffsets::U16Chunked {
                chunk_offsets,
                local_offsets,
                chunk_rows: super::RowOffsetStore::CHUNK_ROWS,
            },
            super::RowOffsetStore::U24(bytes) => SuperneoCompactRowOffsets::U24(bytes),
            super::RowOffsetStore::U32(offsets) => SuperneoCompactRowOffsets::U32(offsets),
        };
        // SAFETY: `CompactRowBlock` is `repr(transparent)` over `u32`, with
        // the size and alignment of `u32` (asserted in `row_block.rs`). So
        // the `row_blocks.len()` values of `self.row_blocks` are that many
        // `u32` values in the same memory, borrowed from `self`.
        let row_blocks =
            unsafe { core::slice::from_raw_parts(self.row_blocks.as_ptr().cast::<u32>(), self.row_blocks.len()) };
        // SAFETY: `DenseRowBlock` is `repr(C)` with two `u32` fields, `block`
        // then `pattern`, so it has no padding, and it has the size and
        // alignment of `[u32; 2]` (asserted in `row_block.rs`). So the
        // `dense_row_blocks.len()` values of `self.dense_row_blocks` are that
        // many `[block, pattern]` pairs in the same memory, borrowed from
        // `self`.
        let dense_row_blocks = unsafe {
            core::slice::from_raw_parts(
                self.dense_row_blocks.as_ptr().cast::<[u32; 2]>(),
                self.dense_row_blocks.len(),
            )
        };
        let DenseBlockStore::Compact {
            offsets,
            locals,
            coefficients,
        } = &self.dense_orig
        else {
            return None;
        };
        let geometric_row_offsets = match &self.geometric_row_offsets {
            super::RowOffsetStore::Empty => SuperneoCompactRowOffsets::Empty,
            super::RowOffsetStore::U16Chunked {
                chunk_offsets,
                local_offsets,
            } => SuperneoCompactRowOffsets::U16Chunked {
                chunk_offsets,
                local_offsets,
                chunk_rows: super::RowOffsetStore::CHUNK_ROWS,
            },
            super::RowOffsetStore::U24(bytes) => SuperneoCompactRowOffsets::U24(bytes),
            super::RowOffsetStore::U32(offsets) => SuperneoCompactRowOffsets::U32(offsets),
        };
        Some(SuperneoCompactDeviceParts {
            row_offsets,
            row_blocks,
            dense_row_blocks,
            dense_offsets: offsets,
            dense_locals: locals,
            dense_coefficients: coefficients,
            geometric_row_offsets,
            geometric_runs: &self.geometric_runs,
            identity: self.identity,
        })
    }

    pub fn compact_explicit_shape(&self) -> (usize, usize, bool) {
        (self.rows, self.cols, self.identity)
    }

    pub fn compact_explicit_coefficient_count(&self) -> usize {
        if self.identity {
            return 0;
        }
        self.row_blocks
            .iter()
            .map(|block| {
                block
                    .single_parts()
                    .map_or_else(|| self.dense_coefficient_count(self.dense_pattern_index(*block)), |_| 1)
            })
            .sum()
    }

    pub fn has_compact_geometric_row(&self, row: usize) -> bool {
        row < self.rows && self.geometric_row_offsets.get(row) != self.geometric_row_offsets.get(row + 1)
    }

    pub fn compact_geometric_run_count(&self) -> usize {
        self.geometric_runs.len()
    }

    pub fn for_each_compact_geometric_run(&self, mut visit: impl FnMut(usize, usize, usize, usize, F, F)) {
        for row in 0..self.rows {
            for index in self.geometric_row_offsets.range(row) {
                let run = self.geometric_runs[index];
                visit(
                    index,
                    row,
                    run[0] as u32 as usize,
                    (run[0] >> 32) as u32 as usize,
                    F::from_u64(run[1]),
                    F::from_u64(run[2]),
                );
            }
        }
    }

    pub fn for_each_compact_explicit_row_coefficient(&self, row: usize, mut visit: impl FnMut(u32, u8, F)) {
        if self.identity || row >= self.rows {
            return;
        }
        for block in self.row_blocks_for(row).iter().copied() {
            if let Some((block, local, coefficient)) = block.single_parts() {
                visit(block as u32, local as u8, coefficient);
                continue;
            }
            let dense_row = self.dense_row_blocks[block.dense_index().expect("dense compact block")];
            let dense = dense_row.pattern();
            match &self.dense_orig {
                DenseBlockStore::Building(blocks) => {
                    for (local, &coefficient) in blocks[dense].0.iter().enumerate() {
                        if coefficient != F::ZERO {
                            visit(dense_row.block() as u32, local as u8, coefficient);
                        }
                    }
                }
                DenseBlockStore::Compact {
                    offsets,
                    locals,
                    coefficients,
                } => {
                    for entry in offsets[dense] as usize..offsets[dense + 1] as usize {
                        visit(dense_row.block() as u32, locals[entry], coefficients[entry]);
                    }
                }
            }
        }
    }

    fn dense_coefficient_count(&self, dense: usize) -> usize {
        match &self.dense_orig {
            DenseBlockStore::Building(blocks) => blocks[dense]
                .0
                .iter()
                .filter(|&&coefficient| coefficient != F::ZERO)
                .count(),
            DenseBlockStore::Compact { offsets, .. } => (offsets[dense + 1] - offsets[dense]) as usize,
        }
    }
}
