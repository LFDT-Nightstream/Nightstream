//! One original-row cache and its local device metadata. Columns stay global.

use std::ops::Range;
use std::sync::{Arc, Weak};

use neo_reductions::superneo_eval::{MatrixWindow, SuperneoCompactRowOffsets, SuperneoMatrixCache};

use super::*;

pub(super) struct MetalMatrixWindow {
    pub(super) rows: Range<usize>,
    pub(super) cache: Arc<SuperneoEvalCache>,
    pub(super) matrices: Arc<Vec<MetalCompactMatrix>>,
    pub(super) workspace_peak_bytes: usize,
    pub(super) upload_bytes: usize,
}

const RESIDENT_LOCK: MetalError = MetalError::Shape("resident device data lock was poisoned");

/// Device data that depends only on a host matrix cache and a key. An entry
/// lives while the host holds the cache, which a relation's retained window
/// does between folds.
pub(in crate::session) struct Resident<K, V>(std::sync::Mutex<Vec<(Weak<SuperneoEvalCache>, K, Arc<V>)>>);

impl<K, V> Default for Resident<K, V> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<K: PartialEq, V> Resident<K, V> {
    pub(super) fn get(&self, cache: &Arc<SuperneoEvalCache>, key: &K) -> Result<Option<Arc<V>>, MetalError> {
        let mut entries = self.0.lock().map_err(|_| RESIDENT_LOCK)?;
        entries.retain(|(host, ..)| host.strong_count() != 0);
        Ok(entries
            .iter()
            .find(|(host, kept, _)| std::ptr::eq(host.as_ptr(), Arc::as_ptr(cache)) && kept == key)
            .map(|(.., value)| Arc::clone(value)))
    }

    /// Keeps `value` if another owner also holds `cache`; a cache that only
    /// its window holds dies with the window.
    pub(super) fn keep(&self, cache: &Arc<SuperneoEvalCache>, key: K, value: &Arc<V>) -> Result<(), MetalError> {
        if Arc::strong_count(cache) > 1 {
            self.0
                .lock()
                .map_err(|_| RESIDENT_LOCK)?
                .push((Arc::downgrade(cache), key, Arc::clone(value)));
        }
        Ok(())
    }
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
    // All absent arrays share one bindable word.
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
    Ok(estimate.min(rows / 2).max(1))
}

impl MetalSession {
    pub(super) fn load_matrix_window(
        &self,
        plan: &MetalJointMatrixPlan<'_>,
        requested: Range<usize>,
        reserved_bytes: usize,
        upload: bool,
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
            let (storage_bytes, workspace_peak) = (window.storage_bytes(), window.workspace_peak_bytes());
            let cache = window.into_cache();
            let matrices = cache.matrix_caches();
            let (upload_bytes, staging_bytes) = if upload { upload_size(matrices)? } else { (0, 0) };
            let descriptors = if upload { matrices.len() } else { 0 }
                .checked_mul(size_of::<MetalCompactMatrix>())
                .ok_or(MetalError::Shape("matrix window descriptor size overflow"))?;
            let live_bytes = add_bytes(
                add_bytes(add_bytes(storage_bytes, descriptors)?, upload_bytes)?,
                staging_bytes,
            )?;
            let peak = workspace_peak.max(live_bytes);
            let resident = if upload {
                self.resident_matrices.get(&cache, &())?
            } else {
                None
            };
            // Resident metadata is already in the allocated size.
            let budget = if resident.is_some() {
                application::available_workspace(self, reserved_bytes)
                    .saturating_add(upload_bytes)
                    .min(plan.workspace_bytes)
            } else {
                budget
            };
            if peak > budget {
                let count = rows.end - rows.start;
                requested.end = rows.start + smaller_window(count, peak, budget)?;
                continue;
            }
            let uploaded = match resident {
                Some(resident) => resident,
                None if !upload => Arc::default(),
                None => {
                    let empty = self.buffer(size_of::<u64>())?;
                    let uploaded = Arc::new(
                        matrices
                            .iter()
                            .map(|matrix| self.upload_matrix_metadata(matrix, &empty))
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                    self.resident_matrices.keep(&cache, (), &uploaded)?;
                    uploaded
                }
            };
            return Ok(MetalMatrixWindow {
                rows,
                cache,
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
