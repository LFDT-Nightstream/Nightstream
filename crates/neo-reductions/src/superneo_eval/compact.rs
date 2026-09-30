//! Compact explicit-matrix export for accelerators.

use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use super::{DenseBlockStore, RealBlockStorage, SuperneoMatrixCache, SuperneoRingLinearForm, SuperneoZBlocks};

/// Packed row-offset storage borrowed by an accelerator backend.
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

/// Borrowed, structure-static parts of one compact SuperNeo matrix cache.
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
    /// Packed positive/negative masks when the real plane is signed-unit.
    pub fn signed_unit_masks(&self) -> Option<(&[u64], &[u64])> {
        match &self.re {
            RealBlockStorage::SignedUnit { positive, negative } => Some((positive, negative)),
            _ => None,
        }
    }

    /// Pack each real coefficient into positive/negative magnitude masks.
    ///
    /// The per-block layout is `[+1, -1, +2, -2, ...]`. This is the compact
    /// accelerator form for the exact SuperNeo radix alphabet.
    #[doc(hidden)]
    pub fn signed_digit_masks(&self, base: u32) -> Option<Vec<u64>> {
        if !(2..=4).contains(&base) || !self.imag_all_zero() {
            return None;
        }
        let magnitudes = (base - 1) as usize;
        let mut masks = vec![0; self.block_len() * 2 * magnitudes];
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
    /// Borrow the finished compact storage without expanding scalar entries.
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
        let row_blocks =
            unsafe { core::slice::from_raw_parts(self.row_blocks.as_ptr().cast::<u32>(), self.row_blocks.len()) };
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

    /// `(rows, scalar columns, identity)` for the explicit matrix component.
    pub fn compact_explicit_shape(&self) -> (usize, usize, bool) {
        (self.rows, self.cols, self.identity)
    }

    /// Number of scalar coefficients in the compact explicit matrix.
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

    /// Whether one row contains at least one compact geometric run.
    pub fn has_compact_geometric_row(&self, row: usize) -> bool {
        row < self.rows && self.geometric_row_offsets.get(row) != self.geometric_row_offsets.get(row + 1)
    }

    /// Number of compact geometric runs retained outside the explicit CSR.
    pub fn compact_geometric_run_count(&self) -> usize {
        self.geometric_runs.len()
    }

    /// Visit compact geometric runs in canonical row order.
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

    /// Visit the nonzero original coefficients in one explicit row.
    ///
    /// Identity matrices are reported by [`Self::compact_explicit_shape`] and
    /// intentionally have no stored coefficients.
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

impl SuperneoRingLinearForm {
    /// Sparse `(block, real coefficients, imaginary coefficients)` export.
    pub fn to_sparse_block_coeffs(&self) -> Vec<(usize, [F; D], [F; D])> {
        self.entries
            .iter()
            .map(|entry| (entry.blk, entry.re_form.0, entry.im_form.0))
            .collect()
    }
}
