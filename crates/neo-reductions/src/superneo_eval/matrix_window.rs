//! Exact local caches for a bounded prefix of requested global rows.
//! Counting and filling use the same borrowed source; no full-domain cache is built.

use std::{mem::size_of, ops::ControlFlow, ops::Range};

use neo_ccs::GeometricRowRun;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{
    matrix_rows::invalid,
    row_block::{CompactRowBlock, DenseRowBlock, COMPACT_DENSE_INDEX_MASK},
    DenseBlockStore, MatrixRowSink, MatrixRows, MatrixShape, RowOffsetStore, SuperneoEvalCache, SuperneoMatrixCache,
};
use crate::PiCcsError;

/// A complete global row range with local cache row indices and global columns.
/// Storage accounting includes owned vector capacities and construction counts,
/// but excludes the borrowed source, allocator overhead, and device copies.
pub struct MatrixWindow {
    rows: Range<usize>,
    cache: SuperneoEvalCache,
    storage_bytes: usize,
    workspace_peak_bytes: usize,
}

impl MatrixWindow {
    /// Count the full requested range without allocating cache storage. This
    /// derives the workspace for callers that intentionally retain all rows.
    pub fn required_workspace(
        source: &dyn MatrixRows,
        requested: Range<usize>,
        payload_bytes_per_row: usize,
    ) -> Result<usize, PiCcsError> {
        let count = count_rows(source, requested, usize::MAX, payload_bytes_per_row)?;
        count.required_bytes()
    }

    /// Read the largest complete prefix that fits the supplied host workspace.
    /// The counting pass can inspect one additional row before it stops. The
    /// filling pass reads only the returned range, and never grows its buffers.
    pub fn load_next(
        source: &dyn MatrixRows,
        requested: Range<usize>,
        workspace_bytes: usize,
    ) -> Result<Self, PiCcsError> {
        Self::load_next_with_payload(source, requested, workspace_bytes, 0)
    }

    /// Reserve caller-owned row values alongside the matrix window. Storage
    /// reports cache bytes only; peak includes the supplied per-row payload.
    pub fn load_next_with_payload(
        source: &dyn MatrixRows,
        requested: Range<usize>,
        workspace_bytes: usize,
        payload_bytes_per_row: usize,
    ) -> Result<Self, PiCcsError> {
        let count = count_rows(source, requested, workspace_bytes, payload_bytes_per_row)?;
        let shape = count.coverage.shape;
        let count_bytes = count.count_bytes;
        let rows = count.coverage.requested.start..count.accepted_end;
        if rows.is_empty() {
            return Err(invalid("matrix source supplied no complete row"));
        }
        let expected_bytes = storage_size(shape, rows.len(), &count.totals)?;
        let payload_bytes = product(rows.len(), payload_bytes_per_row)?;
        let expected_peak = sum(sum(expected_bytes, count_bytes)?, payload_bytes)?;
        require_workspace(expected_peak, workspace_bytes)?;

        let mut matrices = reserved(shape.matrices)?;
        for counts in &count.totals {
            matrices.push(SuperneoMatrixCache {
                rows: rows.len(),
                cols: shape.columns,
                row_offsets: row_offsets(rows.len(), counts.explicit)?,
                row_blocks: reserved(counts.explicit)?,
                dense_row_blocks: reserved(counts.dense)?,
                dense_orig: DenseBlockStore::Compact {
                    offsets: filled(sum(counts.dense, 1)?, 0)?,
                    locals: reserved(counts.dense)?,
                    coefficients: reserved(counts.dense)?,
                },
                geometric_row_offsets: row_offsets(rows.len(), counts.geometric)?,
                geometric_runs: reserved(counts.geometric)?,
                identity: false,
                seeded_phi81_blocks: Vec::new(),
            });
        }
        let mut cache = SuperneoEvalCache {
            mats: matrices,
            explicit_matrix_masks: None,
        };
        let storage_bytes = cache_storage_bytes(&cache)?;
        let count_bytes = product(
            sum(count.totals.capacity(), count.current.capacity())?,
            size_of::<Counts>(),
        )?;
        let workspace_peak_bytes = sum(sum(storage_bytes, count_bytes)?, payload_bytes)?;
        require_workspace(workspace_peak_bytes, workspace_bytes)?;
        let mut fill = FillRows {
            coverage: Coverage::new(shape, rows.clone()),
            cache: &mut cache,
            counts: &count.totals,
        };
        source.visit_rows(rows.clone(), &mut fill)?;
        fill.coverage.complete()?;
        for (matrix, &expected) in fill.cache.mats.iter().zip(fill.counts) {
            if matrix.row_blocks.len() != expected.explicit
                || matrix.dense_row_blocks.len() != expected.dense
                || matrix.geometric_runs.len() != expected.geometric
            {
                return Err(invalid("matrix source changed its run count between visits"));
            }
        }
        Ok(Self {
            rows,
            cache,
            storage_bytes,
            workspace_peak_bytes,
        })
    }

    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }

    pub fn cache(&self) -> &SuperneoEvalCache {
        &self.cache
    }

    pub fn storage_bytes(&self) -> usize {
        self.storage_bytes
    }

    pub fn workspace_peak_bytes(&self) -> usize {
        self.workspace_peak_bytes
    }

    pub fn into_cache(self) -> SuperneoEvalCache {
        self.cache
    }
}

struct Coverage {
    shape: MatrixShape,
    requested: Range<usize>,
    row: usize,
    matrix: usize,
    stopped: bool,
    failed: bool,
}

impl Coverage {
    fn new(shape: MatrixShape, requested: Range<usize>) -> Self {
        Self {
            shape,
            row: requested.start,
            requested,
            matrix: 0,
            stopped: false,
            failed: false,
        }
    }

    fn slot(&mut self, row: usize, matrix: usize) -> Result<(), PiCcsError> {
        if self.failed || self.stopped || row != self.row || matrix != self.matrix || row >= self.requested.end {
            self.failed = true;
            return Err(invalid("matrix source row/matrix order or coverage"));
        }
        Ok(())
    }

    fn run(&mut self, row: usize, matrix: usize, run: &GeometricRowRun<F>) -> Result<(), PiCcsError> {
        self.slot(row, matrix)?;
        if run.len() == 0
            || run.row() != row
            || !run.validate_shape(self.shape.rows, self.shape.columns)
            || u32::try_from(run.column_start()).is_err()
            || u32::try_from(run.len()).is_err()
        {
            self.failed = true;
            return Err(invalid("matrix source run range or compact index width"));
        }
        Ok(())
    }

    fn advance(&mut self, row: usize, matrix: usize) -> Result<bool, PiCcsError> {
        self.slot(row, matrix)?;
        self.matrix += 1;
        if self.matrix == self.shape.matrices {
            self.matrix = 0;
            self.row += 1;
            return Ok(true);
        }
        Ok(false)
    }

    fn complete(&self) -> Result<(), PiCcsError> {
        if self.failed || (!self.stopped && (self.row != self.requested.end || self.matrix != 0)) {
            return Err(invalid("incomplete or failed matrix source visit"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
struct Counts {
    explicit: usize,
    dense: usize,
    geometric: usize,
}

impl Counts {
    fn add(self, other: Self) -> Result<Self, PiCcsError> {
        let next = Self {
            explicit: sum(self.explicit, other.explicit)?,
            dense: sum(self.dense, other.dense)?,
            geometric: sum(self.geometric, other.geometric)?,
        };
        if u32::try_from(next.explicit).is_err()
            || u32::try_from(next.geometric).is_err()
            || next.dense > COMPACT_DENSE_INDEX_MASK as usize + 1
        {
            return Err(invalid("matrix window entry count exceeds compact indices"));
        }
        Ok(next)
    }

    fn payload_bytes(self, rows: usize) -> Result<usize, PiCcsError> {
        let families = usize::from(self.explicit != 0) + usize::from(self.geometric != 0);
        let offsets = product(product(sum(rows, 1)?, families)?, size_of::<u32>())?;
        let explicit = product(self.explicit, size_of::<CompactRowBlock>())?;
        let dense = product(
            self.dense,
            size_of::<DenseRowBlock>() + size_of::<u32>() + size_of::<u8>() + size_of::<F>(),
        )?;
        sum(
            sum(offsets, explicit)?,
            sum(dense, product(self.geometric, size_of::<[u64; 3]>())?)?,
        )
    }
}

struct CountRows {
    coverage: Coverage,
    // Accepted totals remain unchanged while current includes the next row.
    totals: Vec<Counts>,
    current: Vec<Counts>,
    run_storage_bytes: usize,
    offset_families: usize,
    fixed_storage_bytes: usize,
    accepted_end: usize,
    workspace_bytes: usize,
    count_bytes: usize,
    payload_bytes_per_row: usize,
}

impl CountRows {
    fn required_bytes(&self) -> Result<usize, PiCcsError> {
        let rows = self.accepted_end - self.coverage.requested.start;
        sum(
            sum(storage_size(self.coverage.shape, rows, &self.totals)?, self.count_bytes)?,
            product(rows, self.payload_bytes_per_row)?,
        )
    }
}

fn count_rows(
    source: &dyn MatrixRows,
    requested: Range<usize>,
    workspace_bytes: usize,
    payload_bytes_per_row: usize,
) -> Result<CountRows, PiCcsError> {
    let shape = source.shape();
    shape.validate(&requested)?;
    if requested.is_empty() {
        return Err(invalid("matrix window requires a nonempty row range"));
    }
    let count_bytes = product(shape.matrices, 2 * size_of::<Counts>())?;
    let minimum = sum(sum(storage_size(shape, 1, &[])?, count_bytes)?, payload_bytes_per_row)?;
    require_workspace(minimum, workspace_bytes)?;
    let mut count = CountRows {
        coverage: Coverage::new(shape, requested.clone()),
        totals: filled(shape.matrices, Counts::default())?,
        current: filled(shape.matrices, Counts::default())?,
        run_storage_bytes: 0,
        offset_families: 0,
        fixed_storage_bytes: storage_size(shape, 1, &[])?,
        accepted_end: requested.start,
        workspace_bytes,
        count_bytes,
        payload_bytes_per_row,
    };
    source.visit_rows(requested, &mut count)?;
    count.coverage.complete()?;
    Ok(count)
}

impl MatrixRowSink for CountRows {
    fn push_run(&mut self, row: usize, matrix: usize, run: GeometricRowRun<F>) -> Result<(), PiCcsError> {
        self.coverage.run(row, matrix, &run)?;
        let current = self.current[matrix];
        let (entry, bytes, new_family) = if run.len() == 1 {
            let dense = usize::from(*run.initial() != F::ONE && *run.initial() != -F::ONE);
            (
                Counts {
                    explicit: 1,
                    dense,
                    geometric: 0,
                },
                size_of::<CompactRowBlock>()
                    + dense * (size_of::<DenseRowBlock>() + size_of::<u32>() + size_of::<u8>() + size_of::<F>()),
                current.explicit == 0,
            )
        } else {
            (
                Counts {
                    geometric: 1,
                    ..Counts::default()
                },
                size_of::<[u64; 3]>(),
                current.geometric == 0,
            )
        };
        let next = current.add(entry)?;
        let run_storage_bytes = sum(self.run_storage_bytes, bytes)?;
        let offset_families = sum(self.offset_families, usize::from(new_family))?;
        self.current[matrix] = next;
        self.run_storage_bytes = run_storage_bytes;
        self.offset_families = offset_families;
        Ok(())
    }

    fn finish_matrix_row(&mut self, row: usize, matrix: usize) -> Result<ControlFlow<()>, PiCcsError> {
        if !self.coverage.advance(row, matrix)? {
            return Ok(ControlFlow::Continue(()));
        }
        let rows = self.coverage.row - self.coverage.requested.start;
        // An offset family starts at its first entry, but includes every
        // preceding empty row and the initial zero offset as well.
        let offsets = product(product(sum(rows, 1)?, self.offset_families)?, size_of::<u32>())?;
        let storage = sum(sum(self.fixed_storage_bytes, self.run_storage_bytes)?, offsets)?;
        let required = sum(
            sum(storage, self.count_bytes)?,
            product(rows, self.payload_bytes_per_row)?,
        )?;
        if required > self.workspace_bytes {
            if self.accepted_end == self.coverage.requested.start {
                self.coverage.failed = true;
                return Err(PiCcsError::MatrixWorkspace {
                    required,
                    available: self.workspace_bytes,
                });
            }
            self.coverage.stopped = true;
            return Ok(ControlFlow::Break(()));
        }
        self.totals.copy_from_slice(&self.current);
        self.accepted_end = self.coverage.row;
        Ok(ControlFlow::Continue(()))
    }
}

struct FillRows<'a> {
    coverage: Coverage,
    cache: &'a mut SuperneoEvalCache,
    counts: &'a [Counts],
}

impl MatrixRowSink for FillRows<'_> {
    fn push_run(&mut self, row: usize, matrix: usize, run: GeometricRowRun<F>) -> Result<(), PiCcsError> {
        self.coverage.run(row, matrix, &run)?;
        let destination = &mut self.cache.mats[matrix];
        let expected = self.counts[matrix];
        let scalar = run.len() == 1;
        let coefficient = *run.initial();
        let unit = coefficient == F::ONE || coefficient == -F::ONE;
        if (scalar && destination.row_blocks.len() == expected.explicit)
            || (scalar && !unit && destination.dense_row_blocks.len() == expected.dense)
            || (!scalar && destination.geometric_runs.len() == expected.geometric)
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
                let index = destination.dense_row_blocks.len();
                let DenseBlockStore::Compact {
                    offsets,
                    locals,
                    coefficients,
                } = &mut destination.dense_orig
                else {
                    unreachable!("matrix windows have compact original patterns");
                };
                locals.push(local as u8);
                coefficients.push(coefficient);
                offsets[index + 1] = coefficients.len() as u32;
                destination
                    .dense_row_blocks
                    .push(DenseRowBlock::new(block, index));
                CompactRowBlock::dense(index)
            };
            destination.row_blocks.push(reference);
            return Ok(());
        }
        destination.geometric_runs.push([
            (run.column_start() as u64) | ((run.len() as u64) << 32),
            run.initial().as_canonical_u64(),
            run.ratio().as_canonical_u64(),
        ]);
        Ok(())
    }

    fn finish_matrix_row(&mut self, row: usize, matrix: usize) -> Result<ControlFlow<()>, PiCcsError> {
        self.coverage.advance(row, matrix)?;
        let destination = &mut self.cache.mats[matrix];
        let local = row - self.coverage.requested.start + 1;
        for (offsets, end) in [
            (&mut destination.row_offsets, destination.row_blocks.len()),
            (&mut destination.geometric_row_offsets, destination.geometric_runs.len()),
        ] {
            if let RowOffsetStore::U32(offsets) = offsets {
                offsets[local] = end as u32;
            }
        }
        Ok(ControlFlow::Continue(()))
    }
}

fn row_offsets(rows: usize, entries: usize) -> Result<RowOffsetStore, PiCcsError> {
    Ok(if entries == 0 {
        RowOffsetStore::Empty
    } else {
        RowOffsetStore::U32(filled(sum(rows, 1)?, 0)?)
    })
}

fn storage_size(shape: MatrixShape, rows: usize, counts: &[Counts]) -> Result<usize, PiCcsError> {
    let fixed = product(shape.matrices, size_of::<SuperneoMatrixCache>() + size_of::<u32>())?;
    counts
        .iter()
        .try_fold(fixed, |bytes, &count| sum(bytes, count.payload_bytes(rows)?))
}

fn cache_storage_bytes(cache: &SuperneoEvalCache) -> Result<usize, PiCcsError> {
    let mut bytes = product(cache.mats.capacity(), size_of::<SuperneoMatrixCache>())?;
    for matrix in &cache.mats {
        for offsets in [&matrix.row_offsets, &matrix.geometric_row_offsets] {
            if let RowOffsetStore::U32(offsets) = offsets {
                bytes = sum(bytes, product(offsets.capacity(), size_of::<u32>())?)?;
            }
        }
        let DenseBlockStore::Compact {
            offsets,
            locals,
            coefficients,
        } = &matrix.dense_orig
        else {
            unreachable!("matrix windows have compact original patterns");
        };
        bytes = sum(bytes, product(offsets.capacity(), size_of::<u32>())?)?;
        bytes = sum(
            bytes,
            product(matrix.row_blocks.capacity(), size_of::<CompactRowBlock>())?,
        )?;
        bytes = sum(
            bytes,
            product(matrix.dense_row_blocks.capacity(), size_of::<DenseRowBlock>())?,
        )?;
        bytes = sum(bytes, product(locals.capacity(), size_of::<u8>())?)?;
        bytes = sum(bytes, product(coefficients.capacity(), size_of::<F>())?)?;
        bytes = sum(bytes, product(matrix.geometric_runs.capacity(), size_of::<[u64; 3]>())?)?;
    }
    Ok(bytes)
}

fn reserved<T>(capacity: usize) -> Result<Vec<T>, PiCcsError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|error| PiCcsError::BackendFailure {
            backend: "matrix-rows",
            reason: format!("matrix window allocation failed: {error}"),
        })?;
    Ok(values)
}

fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, PiCcsError> {
    let mut values = reserved(len)?;
    values.resize(len, value);
    Ok(values)
}

fn product(left: usize, right: usize) -> Result<usize, PiCcsError> {
    left.checked_mul(right)
        .ok_or_else(|| invalid("matrix window storage size overflow"))
}

fn sum(left: usize, right: usize) -> Result<usize, PiCcsError> {
    left.checked_add(right)
        .ok_or_else(|| invalid("matrix window storage size overflow"))
}

fn require_workspace(required: usize, available: usize) -> Result<(), PiCcsError> {
    if required > available {
        return Err(PiCcsError::MatrixWorkspace { required, available });
    }
    Ok(())
}
