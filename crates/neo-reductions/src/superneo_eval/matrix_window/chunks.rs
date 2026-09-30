//! Builds one complete matrix window from row chunks that are counted and
//! filled concurrently. It is used only when the complete requested range
//! fits the workspace; the serial loader owns longest-prefix selection.

use std::{mem::size_of, ops::ControlFlow, ops::Range};

use neo_ccs::GeometricRowRun;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;

use super::{
    cache_storage_bytes, count_rows, filled, invalid, product, require_workspace, reserved, row_offsets, storage_size,
    sum, CompactRowBlock, Counts, Coverage, DenseBlockStore, DenseRowBlock, MatrixRowSink, MatrixRows, MatrixWindow,
    RowOffsetStore, SuperneoEvalCache, SuperneoMatrixCache,
};
use crate::PiCcsError;

/// Rows per chunk for the production loader.
pub(super) const CHUNK_ROWS: usize = 1 << 16;

/// Return the complete window over `requested`, or `None` when it does not fit
/// the workspace. The result equals the serial loader's complete window.
pub(super) fn load_complete(
    source: &dyn MatrixRows,
    requested: Range<usize>,
    workspace_bytes: usize,
    payload_bytes_per_row: usize,
    chunk_rows: usize,
) -> Result<Option<MatrixWindow>, PiCcsError> {
    let shape = source.shape();
    shape.validate(&requested)?;
    if requested.is_empty() || chunk_rows == 0 {
        return Err(invalid("matrix window requires a nonempty row range"));
    }
    let chunks = requested
        .clone()
        .step_by(chunk_rows)
        .map(|start| start..(start + chunk_rows).min(requested.end))
        .collect::<Vec<_>>();
    let counts = chunks
        .par_iter()
        .map(|rows| {
            let count = count_rows(source, rows.clone(), usize::MAX, 0)?;
            if count.accepted_end != rows.end {
                return Err(invalid("matrix source stopped inside a counted chunk"));
            }
            Ok(count.totals)
        })
        .collect::<Result<Vec<_>, PiCcsError>>()?;
    let mut totals = filled(shape.matrices, Counts::default())?;
    for chunk in &counts {
        for (total, &count) in totals.iter_mut().zip(chunk) {
            *total = total.add(count)?;
        }
    }
    let rows = requested.len();
    let count_bytes = product(sum(counts.len(), 1)?, product(shape.matrices, size_of::<Counts>())?)?;
    let payload_bytes = product(rows, payload_bytes_per_row)?;
    if sum(sum(storage_size(shape, rows, &totals)?, count_bytes)?, payload_bytes)? > workspace_bytes {
        return Ok(None);
    }

    let mut matrices = reserved(shape.matrices)?;
    for total in &totals {
        matrices.push(SuperneoMatrixCache {
            rows,
            cols: shape.columns,
            row_offsets: row_offsets(rows, total.explicit)?,
            row_blocks: filled(total.explicit, CompactRowBlock::default())?,
            dense_row_blocks: filled(total.dense, DenseRowBlock::default())?,
            dense_orig: DenseBlockStore::Compact {
                offsets: filled(sum(total.dense, 1)?, 0)?,
                locals: filled(total.dense, 0)?,
                coefficients: filled(total.dense, F::ZERO)?,
            },
            geometric_row_offsets: row_offsets(rows, total.geometric)?,
            geometric_runs: filled(total.geometric, [0; 3])?,
            identity: false,
        });
    }
    let mut cache = SuperneoEvalCache {
        mats: matrices,
        explicit_matrix_masks: None,
    };
    let storage_bytes = cache_storage_bytes(&cache)?;
    let workspace_peak_bytes = sum(sum(storage_bytes, count_bytes)?, payload_bytes)?;
    require_workspace(workspace_peak_bytes, workspace_bytes)?;

    let mut fills = chunks
        .iter()
        .map(|rows| ChunkFill {
            coverage: Coverage::new(shape, rows.clone()),
            matrices: Vec::with_capacity(shape.matrices),
        })
        .collect::<Vec<_>>();
    for (matrix, cache) in cache.mats.iter_mut().enumerate() {
        let mut rest = MatrixSlices::whole(cache);
        for (fill, chunk) in fills.iter_mut().zip(&counts) {
            let (head, tail) = rest.split(fill.coverage.requested.len(), chunk[matrix])?;
            fill.matrices.push(head);
            rest = tail;
        }
    }
    fills
        .par_iter_mut()
        .zip(chunks.par_iter())
        .try_for_each(|(fill, rows)| {
            source.visit_rows(rows.clone(), fill)?;
            fill.finish()
        })?;
    drop(fills);
    Ok(Some(MatrixWindow {
        rows: requested,
        cache,
        storage_bytes,
        workspace_peak_bytes,
    }))
}

/// One matrix's storage for a contiguous row chunk. Offsets exclude the
/// initial zero; `base` is the global index of the chunk's first entry.
struct MatrixSlices<'a> {
    row_offsets: Option<&'a mut [u32]>,
    row_blocks: &'a mut [CompactRowBlock],
    dense_row_blocks: &'a mut [DenseRowBlock],
    dense_offsets: &'a mut [u32],
    locals: &'a mut [u8],
    coefficients: &'a mut [F],
    geometric_offsets: Option<&'a mut [u32]>,
    geometric_runs: &'a mut [[u64; 3]],
    base: Counts,
    used: Counts,
}

impl<'a> MatrixSlices<'a> {
    fn whole(cache: &'a mut SuperneoMatrixCache) -> Self {
        let DenseBlockStore::Compact {
            offsets,
            locals,
            coefficients,
        } = &mut cache.dense_orig
        else {
            unreachable!("matrix windows have compact original patterns");
        };
        Self {
            row_offsets: after_initial_offset(&mut cache.row_offsets),
            row_blocks: &mut cache.row_blocks,
            dense_row_blocks: &mut cache.dense_row_blocks,
            dense_offsets: &mut offsets[1..],
            locals,
            coefficients,
            geometric_offsets: after_initial_offset(&mut cache.geometric_row_offsets),
            geometric_runs: &mut cache.geometric_runs,
            base: Counts::default(),
            used: Counts::default(),
        }
    }

    /// Split off the first `rows` rows holding `count` entries.
    fn split(self, rows: usize, count: Counts) -> Result<(Self, Self), PiCcsError> {
        let (row_offsets, rest_row_offsets) = split_offsets(self.row_offsets, rows);
        let (row_blocks, rest_row_blocks) = self.row_blocks.split_at_mut(count.explicit);
        let (dense_row_blocks, rest_dense_row_blocks) = self.dense_row_blocks.split_at_mut(count.dense);
        let (dense_offsets, rest_dense_offsets) = self.dense_offsets.split_at_mut(count.dense);
        let (locals, rest_locals) = self.locals.split_at_mut(count.dense);
        let (coefficients, rest_coefficients) = self.coefficients.split_at_mut(count.dense);
        let (geometric_offsets, rest_geometric_offsets) = split_offsets(self.geometric_offsets, rows);
        let (geometric_runs, rest_geometric_runs) = self.geometric_runs.split_at_mut(count.geometric);
        let next = self.base.add(count)?;
        Ok((
            Self {
                row_offsets,
                row_blocks,
                dense_row_blocks,
                dense_offsets,
                locals,
                coefficients,
                geometric_offsets,
                geometric_runs,
                base: self.base,
                used: Counts::default(),
            },
            Self {
                row_offsets: rest_row_offsets,
                row_blocks: rest_row_blocks,
                dense_row_blocks: rest_dense_row_blocks,
                dense_offsets: rest_dense_offsets,
                locals: rest_locals,
                coefficients: rest_coefficients,
                geometric_offsets: rest_geometric_offsets,
                geometric_runs: rest_geometric_runs,
                base: next,
                used: Counts::default(),
            },
        ))
    }
}

fn after_initial_offset(offsets: &mut RowOffsetStore) -> Option<&mut [u32]> {
    match offsets {
        RowOffsetStore::U32(offsets) => Some(&mut offsets[1..]),
        _ => None,
    }
}

type OffsetSplit<'a> = (Option<&'a mut [u32]>, Option<&'a mut [u32]>);

fn split_offsets(offsets: Option<&mut [u32]>, rows: usize) -> OffsetSplit<'_> {
    match offsets {
        Some(offsets) => {
            let (head, tail) = offsets.split_at_mut(rows);
            (Some(head), Some(tail))
        }
        None => (None, None),
    }
}

struct ChunkFill<'a> {
    coverage: Coverage,
    matrices: Vec<MatrixSlices<'a>>,
}

impl ChunkFill<'_> {
    fn finish(&self) -> Result<(), PiCcsError> {
        self.coverage.complete()?;
        for matrix in &self.matrices {
            if matrix.used.explicit != matrix.row_blocks.len()
                || matrix.used.dense != matrix.dense_row_blocks.len()
                || matrix.used.geometric != matrix.geometric_runs.len()
            {
                return Err(invalid("matrix source changed its run count between visits"));
            }
        }
        Ok(())
    }
}

impl MatrixRowSink for ChunkFill<'_> {
    fn push_run(&mut self, row: usize, matrix: usize, run: GeometricRowRun<F>) -> Result<(), PiCcsError> {
        self.coverage.run(row, matrix, &run)?;
        let destination = &mut self.matrices[matrix];
        let scalar = run.len() == 1;
        let coefficient = *run.initial();
        let unit = coefficient == F::ONE || coefficient == -F::ONE;
        if (scalar && destination.used.explicit == destination.row_blocks.len())
            || (scalar && !unit && destination.used.dense == destination.dense_row_blocks.len())
            || (!scalar && destination.used.geometric == destination.geometric_runs.len())
        {
            self.coverage.failed = true;
            return Err(invalid("matrix source changed its run count between visits"));
        }
        if scalar {
            let block = run.column_start() / D;
            let local = run.column_start() % D;
            let reference = if unit {
                CompactRowBlock::single(block, local, coefficient)
            } else {
                let slot = destination.used.dense;
                let index = destination.base.dense + slot;
                destination.locals[slot] = local as u8;
                destination.coefficients[slot] = coefficient;
                destination.dense_offsets[slot] = (index + 1) as u32;
                destination.dense_row_blocks[slot] = DenseRowBlock::new(block, index);
                destination.used.dense += 1;
                CompactRowBlock::dense(index)
            };
            destination.row_blocks[destination.used.explicit] = reference;
            destination.used.explicit += 1;
            return Ok(());
        }
        destination.geometric_runs[destination.used.geometric] = [
            (run.column_start() as u64) | ((run.len() as u64) << 32),
            run.initial().as_canonical_u64(),
            run.ratio().as_canonical_u64(),
        ];
        destination.used.geometric += 1;
        Ok(())
    }

    fn finish_matrix_row(&mut self, row: usize, matrix: usize) -> Result<ControlFlow<()>, PiCcsError> {
        self.coverage.advance(row, matrix)?;
        let slot = row - self.coverage.requested.start;
        let destination = &mut self.matrices[matrix];
        if let Some(offsets) = destination.row_offsets.as_deref_mut() {
            offsets[slot] = (destination.base.explicit + destination.used.explicit) as u32;
        }
        if let Some(offsets) = destination.geometric_offsets.as_deref_mut() {
            offsets[slot] = (destination.base.geometric + destination.used.geometric) as u32;
        }
        Ok(ControlFlow::Continue(()))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/matrix_window_chunks.rs"]
mod tests;
