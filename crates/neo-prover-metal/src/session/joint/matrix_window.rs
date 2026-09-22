//! One original-row cache and its local device metadata. Columns stay global.

use std::ops::Range;

use neo_reductions::superneo_eval::{MatrixWindow, SuperneoCompactRowOffsets, SuperneoMatrixCache};

use super::*;

pub(super) struct MetalMatrixWindow {
    pub(super) rows: Range<usize>,
    pub(super) cache: SuperneoEvalCache,
    pub(super) matrices: Vec<MetalCompactMatrix>,
    pub(super) workspace_peak_bytes: usize,
    pub(super) upload_bytes: usize,
}

fn add_bytes(total: usize, bytes: usize) -> Result<usize, MetalError> {
    total
        .checked_add(bytes)
        .ok_or(MetalError::Shape("matrix window size overflow"))
}

fn offset_bytes(offsets: SuperneoCompactRowOffsets<'_>) -> Result<(usize, usize), MetalError> {
    match offsets {
        SuperneoCompactRowOffsets::Empty => Ok((0, 0)),
        SuperneoCompactRowOffsets::U16Chunked { local_offsets, .. } => {
            let bytes = local_offsets
                .len()
                .checked_mul(size_of::<u32>())
                .ok_or(MetalError::Shape("matrix window offset size overflow"))?;
            Ok((bytes, bytes))
        }
        SuperneoCompactRowOffsets::U24(values) => Ok((size_of_val(values), 0)),
        SuperneoCompactRowOffsets::U32(values) => Ok((size_of_val(values), 0)),
    }
}

pub(super) fn upload_size(matrices: &[SuperneoMatrixCache]) -> Result<(usize, usize), MetalError> {
    // All absent arrays share one bindable word. Expanded u16 row offsets
    // use one temporary u32 array at a time during upload.
    let mut bytes = size_of::<u64>();
    let mut staging = 0;
    for matrix in matrices {
        let parts = matrix
            .compact_device_parts()
            .ok_or(MetalError::Shape("matrix window cache is not finished"))?;
        for offsets in [parts.row_offsets, parts.geometric_row_offsets] {
            let (upload, temporary) = offset_bytes(offsets)?;
            bytes = add_bytes(bytes, upload)?;
            staging = staging.max(temporary);
        }
        for payload in [
            size_of_val(parts.row_blocks),
            size_of_val(parts.dense_row_blocks),
            size_of_val(parts.dense_offsets),
            size_of_val(parts.dense_locals),
            size_of_val(parts.dense_coefficients),
            size_of_val(parts.geometric_runs),
        ] {
            bytes = add_bytes(bytes, payload)?;
        }
    }
    Ok((bytes, staging))
}

pub(super) fn smaller_window(rows: usize, required: usize, available: usize) -> Result<usize, MetalError> {
    if rows <= 1 {
        return Err(MetalError::MemoryLimit {
            requested: required,
            allocated: 0,
            limit: available,
        });
    }
    let estimate = (rows as u128 * available as u128 / required as u128) as usize;
    // Fixed column costs need not fall with the row count. Halving bounds
    // the total rows rebuilt by retries even when those costs dominate.
    Ok(estimate.min(rows / 2).max(1))
}

impl MetalSession {
    pub(super) fn load_matrix_window(
        &self,
        plan: &MetalJointMatrixPlan<'_>,
        requested: Range<usize>,
        reserved_bytes: usize,
    ) -> Result<MetalMatrixWindow, MetalError> {
        let budget = application::available_workspace(self, reserved_bytes).min(plan.workspace_bytes);
        let mut requested = requested;
        loop {
            let window =
                MatrixWindow::load_next(plan.source, requested.clone(), budget).map_err(|error| match error {
                    neo_reductions::PiCcsError::MatrixWorkspace { required, available } => MetalError::MemoryLimit {
                        requested: required,
                        allocated: 0,
                        limit: available,
                    },
                    error => MetalError::Execution(error.to_string()),
                })?;
            let rows = window.rows();
            let matrices = window.cache().matrix_caches();
            let (upload_bytes, staging_bytes) = upload_size(matrices)?;
            let descriptors = matrices
                .len()
                .checked_mul(size_of::<MetalCompactMatrix>())
                .ok_or(MetalError::Shape("matrix window descriptor size overflow"))?;
            let live_bytes = add_bytes(
                add_bytes(add_bytes(window.storage_bytes(), descriptors)?, upload_bytes)?,
                staging_bytes,
            )?;
            let peak = window.workspace_peak_bytes().max(live_bytes);
            if peak > budget {
                let count = rows.end - rows.start;
                requested.end = rows.start + smaller_window(count, peak, budget)?;
                continue;
            }
            let empty = self.buffer(size_of::<u64>())?;
            let mut uploaded = Vec::with_capacity(matrices.len());
            for matrix in matrices {
                uploaded.push(self.upload_matrix_metadata(matrix, &empty)?);
            }
            return Ok(MetalMatrixWindow {
                rows,
                cache: window.into_cache(),
                matrices: uploaded,
                workspace_peak_bytes: peak,
                upload_bytes,
            });
        }
    }

    pub(super) fn upload_matrix_metadata(
        &self,
        matrix: &SuperneoMatrixCache,
        empty: &Buffer,
    ) -> Result<MetalCompactMatrix, MetalError> {
        let parts = matrix
            .compact_device_parts()
            .ok_or(MetalError::Shape("matrix window cache is not finished"))?;
        let (row_offsets, row_offset_width) = self.upload_matrix_offsets(parts.row_offsets, empty)?;
        let (geometric_row_offsets, geometric_row_offset_width) =
            self.upload_matrix_offsets(parts.geometric_row_offsets, empty)?;
        Ok(MetalCompactMatrix {
            row_offsets,
            row_offset_width,
            row_blocks: self.upload_matrix_array(parts.row_blocks, empty)?,
            dense_row_blocks: self.upload_matrix_array(parts.dense_row_blocks, empty)?,
            dense_offsets: self.upload_matrix_array(parts.dense_offsets, empty)?,
            dense_locals: self.upload_matrix_array(parts.dense_locals, empty)?,
            dense_coefficients: self.upload_matrix_array(parts.dense_coefficients, empty)?,
            geometric_row_offsets,
            geometric_row_offset_width,
            geometric_runs: self.upload_matrix_array(parts.geometric_runs, empty)?,
            identity: parts.identity,
        })
    }

    fn upload_matrix_array<T: Copy>(&self, values: &[T], empty: &Buffer) -> Result<Buffer, MetalError> {
        if values.is_empty() {
            Ok(empty.clone())
        } else {
            self.buffer_from_slice(values)
        }
    }

    fn upload_matrix_offsets(
        &self,
        offsets: SuperneoCompactRowOffsets<'_>,
        empty: &Buffer,
    ) -> Result<(Buffer, u64), MetalError> {
        match offsets {
            SuperneoCompactRowOffsets::Empty => Ok((empty.clone(), 0)),
            SuperneoCompactRowOffsets::U16Chunked {
                chunk_offsets,
                local_offsets,
                chunk_rows,
            } => {
                let expanded: Vec<_> = local_offsets
                    .iter()
                    .enumerate()
                    .map(|(index, &local)| chunk_offsets[index / chunk_rows] + u32::from(local))
                    .collect();
                Ok((self.buffer_from_slice(&expanded)?, 4))
            }
            SuperneoCompactRowOffsets::U24(values) => Ok((self.buffer_from_slice(values)?, 3)),
            SuperneoCompactRowOffsets::U32(values) => Ok((self.buffer_from_slice(values)?, 4)),
        }
    }
}
