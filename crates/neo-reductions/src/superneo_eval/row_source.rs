//! Build the existing compact cache from complete, ordered sparse matrix rows.

use super::row_block::{COMPACT_DENSE_INDEX_MASK, COMPACT_SINGLE_BLOCK_MASK};
use super::{CompactRowBlock, DenseBlockStore, DenseRowBlock, RowOffsetStore, SuperneoEvalCache, SuperneoMatrixCache};
use crate::PiCcsError;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

/// Incremental construction from original coefficients, with columns grouped
/// into consecutive degree-54 blocks. No scalar matrix or dense block table
/// is retained. Rows must arrive in `(row, matrix)` order, including empty rows.
pub struct SuperneoEvalCacheBuilder {
    mats: Vec<SuperneoMatrixCache>,
    logical_columns: usize,
    next_row: usize,
    next_matrix: usize,
    failed: bool,
    explicit_matrix_masks: Option<Vec<u16>>,
}

impl SuperneoEvalCacheBuilder {
    pub fn new(rows: usize, logical_columns: usize, matrices: usize) -> Result<Self, PiCcsError> {
        let blocks = logical_columns.div_ceil(D);
        if logical_columns == 0 || matrices == 0 || blocks > COMPACT_SINGLE_BLOCK_MASK as usize + 1 {
            return Err(invalid("cache row-source shape"));
        }
        let cols = blocks
            .checked_mul(D)
            .ok_or_else(|| invalid("cache column padding overflow"))?;
        rows.checked_add(1)
            .ok_or_else(|| invalid("cache row-offset length overflow"))?;
        let mats = (0..matrices)
            .map(|_| SuperneoMatrixCache {
                rows,
                cols,
                row_offsets: RowOffsetStore::Empty,
                row_blocks: Vec::new(),
                dense_row_blocks: Vec::new(),
                dense_orig: DenseBlockStore::Compact {
                    offsets: vec![0],
                    locals: Vec::new(),
                    coefficients: Vec::new(),
                },
                geometric_row_offsets: RowOffsetStore::Empty,
                geometric_runs: Vec::new(),
                identity: false,
                seeded_phi81_blocks: Vec::new(),
            })
            .collect();
        let explicit_matrix_masks = if matrices <= u16::BITS as usize {
            let mut masks = Vec::new();
            reserve_exact(&mut masks, rows)?;
            masks.resize(rows, 0);
            Some(masks)
        } else {
            None
        };
        Ok(Self {
            mats,
            logical_columns,
            next_row: 0,
            next_matrix: 0,
            failed: false,
            explicit_matrix_masks,
        })
    }

    /// Reserve storage from a prior row count. These values only control
    /// allocation; every supplied row still passes the same value checks.
    /// Underestimated hints grow normally and do not change the output.
    pub fn reserve_matrix(
        &mut self,
        matrix: usize,
        row_blocks: usize,
        dense_blocks: usize,
        coefficients: usize,
    ) -> Result<(), PiCcsError> {
        if self.failed {
            return Err(invalid("cache row-source builder has failed"));
        }
        self.failed = true;
        if row_blocks > u32::MAX as usize
            || dense_blocks > COMPACT_DENSE_INDEX_MASK as usize + 1
            || coefficients > u32::MAX as usize
        {
            return Err(invalid("cache reservation exceeds compact index widths"));
        }
        let cache = self
            .mats
            .get_mut(matrix)
            .ok_or_else(|| invalid("cache reservation matrix slot"))?;
        reserve_exact(&mut cache.row_blocks, row_blocks)?;
        reserve_exact(&mut cache.dense_row_blocks, dense_blocks)?;
        let DenseBlockStore::Compact {
            offsets,
            locals,
            coefficients: stored_coefficients,
        } = &mut cache.dense_orig
        else {
            unreachable!("row-source construction always emits compact patterns");
        };
        reserve_exact(offsets, dense_blocks + 1)?;
        reserve_exact(locals, coefficients)?;
        reserve_exact(stored_coefficients, coefficients)?;
        self.failed = false;
        Ok(())
    }

    /// Append one row. Columns must be strictly increasing, in range, and
    /// have nonzero coefficients. After any error, `finish` rejects the builder.
    pub fn push_row(
        &mut self,
        matrix: usize,
        row: usize,
        entries: impl IntoIterator<Item = (usize, F)>,
    ) -> Result<(), PiCcsError> {
        if self.failed {
            return Err(invalid("cache row-source builder has failed"));
        }
        self.failed = true;
        if row != self.next_row || matrix != self.next_matrix || row >= self.mats[0].rows {
            return Err(invalid("cache row-source order or row bound"));
        }
        let cache = &mut self.mats[matrix];
        let previous_length = cache.row_blocks.len();
        let mut previous_column = None;
        let mut block = 0;
        let mut locals = [0u8; D];
        let mut coefficients = [F::ZERO; D];
        let mut used = 0;
        for (column, coefficient) in entries {
            if column >= self.logical_columns
                || previous_column.is_some_and(|previous| column <= previous)
                || coefficient == F::ZERO
            {
                return Err(invalid("cache row-source noncanonical entry"));
            }
            let next_block = column / D;
            if used != 0 && next_block != block {
                append_block(cache, block, &locals[..used], &coefficients[..used])?;
                used = 0;
            }
            block = next_block;
            locals[used] = (column % D) as u8;
            coefficients[used] = coefficient;
            used += 1;
            previous_column = Some(column);
        }
        if used != 0 {
            append_block(cache, block, &locals[..used], &coefficients[..used])?;
        }
        let end = u32::try_from(cache.row_blocks.len()).map_err(|_| invalid("cache row-block count exceeds u32"))?;
        if previous_length != cache.row_blocks.len() {
            if let Some(masks) = &mut self.explicit_matrix_masks {
                masks[row] |= 1u16 << matrix;
            }
            if matches!(cache.row_offsets, RowOffsetStore::Empty) {
                let mut offsets = Vec::new();
                reserve_exact(&mut offsets, cache.rows + 1)?;
                offsets.resize(row + 1, 0);
                cache.row_offsets = RowOffsetStore::U32(offsets);
            }
        }
        if let RowOffsetStore::U32(offsets) = &mut cache.row_offsets {
            offsets.push(end);
        }
        self.next_matrix += 1;
        if self.next_matrix == self.mats.len() {
            self.next_matrix = 0;
            self.next_row += 1;
        }
        self.failed = false;
        Ok(())
    }

    /// Finish only after every matrix row has been supplied. Empty matrices
    /// keep implicit zero offsets; all other offsets use the existing packing.
    pub fn finish(mut self) -> Result<SuperneoEvalCache, PiCcsError> {
        if self.failed || self.next_row != self.mats[0].rows || self.next_matrix != 0 {
            return Err(invalid("incomplete or failed cache row source"));
        }
        for matrix in &mut self.mats {
            matrix.compact_row_offsets();
        }
        Ok(SuperneoEvalCache {
            mats: self.mats,
            explicit_matrix_masks: self.explicit_matrix_masks,
        })
    }
}

fn reserve_exact<T>(values: &mut Vec<T>, capacity: usize) -> Result<(), PiCcsError> {
    values
        .try_reserve_exact(capacity.saturating_sub(values.len()))
        .map_err(|error| invalid(&format!("cache allocation failed: {error}")))
}

fn append_block(
    cache: &mut SuperneoMatrixCache,
    block: usize,
    locals: &[u8],
    coefficients: &[F],
) -> Result<(), PiCcsError> {
    if coefficients.len() == 1 && (coefficients[0] == F::ONE || coefficients[0] == -F::ONE) {
        cache
            .row_blocks
            .push(CompactRowBlock::single(block, usize::from(locals[0]), coefficients[0]));
        return Ok(());
    }
    let dense_index = cache.dense_row_blocks.len();
    if dense_index > COMPACT_DENSE_INDEX_MASK as usize {
        return Err(invalid("cache dense row-block index exceeds u31"));
    }
    let DenseBlockStore::Compact {
        offsets,
        locals: stored_locals,
        coefficients: stored_coefficients,
    } = &mut cache.dense_orig
    else {
        unreachable!("row-source construction always emits compact patterns");
    };
    let end = stored_locals
        .len()
        .checked_add(locals.len())
        .and_then(|count| u32::try_from(count).ok())
        .ok_or_else(|| invalid("cache dense coefficient count exceeds u32"))?;
    let pattern = offsets.len() - 1;
    stored_locals.extend_from_slice(locals);
    stored_coefficients.extend_from_slice(coefficients);
    offsets.push(end);
    cache
        .dense_row_blocks
        .push(DenseRowBlock::new(block, pattern));
    cache.row_blocks.push(CompactRowBlock::dense(dense_index));
    Ok(())
}

fn invalid(message: &str) -> PiCcsError {
    PiCcsError::InvalidInput(message.to_owned())
}
