//! Local matrix transposes and device ring openings with global equality weights.

use std::mem::size_of;

use neo_ccs::V1_2Evaluations;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::SuperneoMatrixCache;
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use super::{matrix_window::smaller_window, Buffer, CudaJointMatrixPlan, CudaSession, Grid, JointSources};
use crate::CudaError;

mod spans;
mod transpose;
use spans::SpanLayer;
use transpose::BlockTranspose;

#[cfg(test)]
#[path = "../../../tests/unit/joint_openings.rs"]
pub(super) mod tests;

const FORM_REDUCTION_THREADS: usize = 256;
const PARALLEL_FORM_LIST_THRESHOLD: usize = 128;
const FORM_TILE_ENTRIES: usize = 16 * 1024;
const CHUNK_BLOCKS: usize = 512;
const PRODUCT_COEFFICIENTS: usize = 2 * D - 1;

pub(super) struct CudaJointOpeningPlan {
    active_offsets: Buffer,
    active_local_masks: Buffer,
    active_blocks: Buffer,
    active_chunk_bases: Buffer,
    matrix_active_offsets: Buffer,
    matrix_active_offsets_host: Vec<u32>,
    matrix_chunk_offsets: Vec<u32>,
    matrix_identity: Buffer,
    entries: Buffer,
    pattern_offsets: Buffer,
    pattern_locals: Buffer,
    pattern_coefficients: Buffer,
    parallel_form_lists: Buffer,
    tiled_form_lists: Buffer,
    tiled_form_tile_offsets: Buffer,
    tiled_form_tiles: Buffer,
    tiled_form_partials: Buffer,
    geometric: Vec<DeviceGeometricOpeningPlan>,
    parallel_form_list_count: usize,
    tiled_form_list_count: usize,
    tiled_form_tile_count: usize,
    matrix_count: usize,
    rows: usize,
    row_start: usize,
    blocks: usize,
}

struct DeviceGeometricOpeningPlan {
    matrix: usize,
    starts: Buffer,
    ranks: Buffer,
    shapes: Buffer,
    offsets: Buffer,
    rows: Buffer,
    coefficients: Buffer,
    longest: u64,
    span_count: usize,
}

#[derive(Default)]
struct OpeningLayout {
    active: usize,
    chunks: usize,
    entries: usize,
    patterns: usize,
    coefficients: usize,
    parallel: usize,
    tiled: usize,
    tiles: usize,
    max_blocks: usize,
    max_chunks: usize,
    max_spans: usize,
    metadata_bytes: usize,
    spans: Vec<Vec<SpanLayer>>,
    census_bytes: usize,
}

impl OpeningLayout {
    fn measure(
        matrices: &[SuperneoMatrixCache],
        blocks: usize,
        include_pad: bool,
        host_bytes: usize,
    ) -> Result<Option<Self>, CudaError> {
        let pad = if include_pad { blocks } else { 0 };
        let mut layout = Self {
            active: pad,
            chunks: pad.div_ceil(CHUNK_BLOCKS),
            max_blocks: pad,
            max_chunks: pad.div_ceil(CHUNK_BLOCKS),
            ..Self::default()
        };
        let Some(census_bytes) = host_bytes.checked_sub(blocks * size_of::<bool>()) else {
            return Ok(None);
        };
        let budget = spans::CensusBudget::new(census_bytes);
        let census: Option<Vec<_>> = matrices
            .par_iter()
            .map(|matrix| {
                if matrix.compact_geometric_run_count() == 0 {
                    Ok(Some(Vec::new()))
                } else {
                    spans::span_layers(matrix, blocks * D, &budget)
                }
            })
            .collect::<Result<_, CudaError>>()?;
        let Some(census) = census else {
            return Ok(None);
        };
        layout.spans = census;
        layout.census_bytes = census_bytes - budget.left();
        let mut active = vec![false; blocks];
        let mut transpose_scratch = 0usize;
        for (matrix, layers) in matrices.iter().zip(&layout.spans) {
            active.fill(false);
            let parts = matrix
                .compact_device_parts()
                .ok_or(CudaError::Shape("unfinished opening matrix"))?;
            let (rows, _, identity) = matrix.compact_explicit_shape();
            if identity {
                active[..rows.min(blocks * D).div_ceil(D)].fill(true);
            }
            for &reference in parts.row_blocks {
                let block = if reference & (1 << 31) == 0 {
                    reference & ((1 << 24) - 1)
                } else {
                    parts.dense_row_blocks[(reference & !(1 << 31)) as usize][0]
                } as usize;
                if block >= blocks {
                    return Err(CudaError::Shape("opening block exceeds the carrier"));
                }
                active[block] = true;
            }
            spans::mark_blocks(layers, &mut active);
            let count = active.iter().filter(|&&value| value).count();
            layout.active = checked_sum(&[layout.active, count])?;
            layout.chunks = checked_sum(&[layout.chunks, count.div_ceil(CHUNK_BLOCKS)])?;
            layout.max_blocks = layout.max_blocks.max(count);
            layout.max_chunks = layout.max_chunks.max(count.div_ceil(CHUNK_BLOCKS));
            let entries = parts.row_blocks.len();
            let patterns = parts.dense_offsets.len().saturating_sub(1);
            layout.entries = checked_sum(&[layout.entries, entries])?;
            layout.patterns = checked_sum(&[layout.patterns, patterns])?;
            layout.coefficients = checked_sum(&[layout.coefficients, parts.dense_locals.len()])?;
            let parallel = blocks.min(entries / PARALLEL_FORM_LIST_THRESHOLD);
            let tiled = blocks.min(entries / (FORM_TILE_ENTRIES + 1));
            layout.parallel = checked_sum(&[
                layout.parallel,
                checked_product(&[D, parallel], "opening lists overflow")?,
            ])?;
            layout.tiled = checked_sum(&[layout.tiled, checked_product(&[D, tiled], "opening lists overflow")?])?;
            if tiled != 0 {
                layout.tiles = checked_sum(&[
                    layout.tiles,
                    checked_product(
                        &[D, entries.div_ceil(FORM_TILE_ENTRIES) + tiled],
                        "opening tiles overflow",
                    )?,
                ])?;
            }
            transpose_scratch = transpose_scratch.max(checked_sum(&[
                checked_product(&[blocks, 17], "opening transpose size overflow")?,
                size_of::<u32>(),
                checked_product(&[entries + patterns, 8], "opening transpose size overflow")?,
            ])?);
        }
        let matrices = matrices.len() + 1;
        let arrays = [
            (layout.active + 1, 8),
            (layout.active, 8),
            (layout.active, 4),
            (matrices + 1, 4),
            (matrices, 4),
            (layout.entries, 8),
            (layout.patterns + 1, 4),
            (layout.coefficients, 1),
            (layout.coefficients, 8),
            (layout.parallel, 4),
            (layout.tiled, 4),
            (layout.tiled + 1, 4),
            (layout.tiles, 12),
            (layout.chunks, 4),
        ];
        let mut host = checked_product(&[matrices + 1, 4], "opening index size overflow")?;
        let mut device = checked_product(&[layout.tiles.max(1), 16], "opening tile scratch overflow")?;
        for (count, width) in arrays {
            host = checked_sum(&[host, checked_product(&[count, width], "opening index size overflow")?])?;
            device = checked_sum(&[
                device,
                checked_product(&[count.max(1), width], "opening index size overflow")?,
            ])?;
        }
        let mut geometric = layout.census_bytes;
        for layer in layout.spans.iter().flatten() {
            layout.max_spans = layout.max_spans.max(layer.span_count());
            geometric = checked_sum(&[
                geometric,
                layer.device_bytes(),
                size_of::<u64>() + size_of::<DeviceGeometricOpeningPlan>(),
            ])?;
        }
        layout.metadata_bytes = checked_sum(&[host, device, geometric, transpose_scratch.max(blocks)])?;
        Ok(Some(layout))
    }

    fn evaluation_bytes(&self, rows: usize, point_len: usize, active_witnesses: usize) -> Result<usize, CudaError> {
        let low_bits = point_len / 2;
        let tensor_values = (1usize << low_bits) + (1usize << (point_len - low_bits));
        checked_sum(&[
            checked_product(&[tensor_values + rows, size_of::<K>()], "opening weights overflow")?,
            checked_product(&[point_len, 3, size_of::<u64>()], "opening tensor metadata overflow")?,
            checked_product(
                &[active_witnesses, size_of::<u32>()],
                "opening witness indices overflow",
            )?,
            checked_product(&[self.max_blocks, D, size_of::<K>()], "opening forms overflow")?,
            checked_product(
                &[active_witnesses, self.max_chunks, PRODUCT_COEFFICIENTS, size_of::<K>()],
                "opening partials overflow",
            )?,
            checked_product(
                &[active_witnesses, PRODUCT_COEFFICIENTS + 2 * D, size_of::<K>()],
                "opening outputs overflow",
            )?,
            checked_product(&[self.max_chunks, size_of::<u32>()], "opening chunk indices overflow")?,
            checked_product(
                &[self.max_spans.max(1), size_of::<K>()],
                "opening span weights overflow",
            )?,
            26 * size_of::<u64>() + 2 * size_of::<u32>(),
        ])
    }
}

impl CudaSession {
    pub(super) fn eval_streamed_joint_openings<'a>(
        &self,
        plan: &CudaJointMatrixPlan<'_>,
        sources: impl Into<JointSources<'a>>,
        point: &[K],
        witness_count: usize,
        assignment_width: usize,
    ) -> Result<Vec<V1_2Evaluations<K>>, CudaError> {
        Ok(self
            .eval_streamed_rows(plan, sources, point, witness_count, assignment_width, None)?
            .openings)
    }

    pub(super) fn eval_streamed_rows<'a>(
        &self,
        plan: &CudaJointMatrixPlan<'_>,
        sources: impl Into<JointSources<'a>>,
        point: &[K],
        witness_count: usize,
        assignment_width: usize,
        terminal: Option<&super::relation::TerminalRowCheck<'_>>,
    ) -> Result<neo_reductions::superneo_eval::TerminalEvaluations, CudaError> {
        let sources = sources.into();
        let domain = 1usize
            .checked_shl(u32::try_from(point.len()).map_err(|_| CudaError::Shape("opening point is too long"))?)
            .ok_or(CudaError::Shape("opening point domain overflow"))?;
        if point.is_empty()
            || witness_count == 0
            || assignment_width > plan.blocks * D
            || domain < plan.rows.max(plan.blocks * D)
            || !sources.matches_joint(witness_count, plan.blocks)
        {
            return Err(CudaError::Shape("streamed opening dimensions are invalid"));
        }
        let mut result = to_evaluations(vec![vec![vec![K::ZERO; D]; plan.matrix_count + 1]; witness_count]);
        if sources.active_count() == 0 && terminal.is_none() {
            return Ok(neo_reductions::superneo_eval::TerminalEvaluations {
                openings: result,
                first_unsatisfied_row: None,
            });
        }
        let mut next_row = 0;
        while next_row < plan.rows {
            let include_pad = next_row == 0;
            let pad = if include_pad { plan.blocks } else { 0 };
            let floor = OpeningLayout {
                max_blocks: pad,
                max_chunks: pad.div_ceil(CHUNK_BLOCKS),
                ..OpeningLayout::default()
            }
            .evaluation_bytes(0, point.len(), sources.active_count())?;
            let reserved = checked_sum(&[floor, plan.blocks])?;
            let mut requested_end = plan.rows;
            loop {
                let window = self.load_matrix_window(plan, next_row..requested_end, reserved, terminal.is_some())?;
                let count = window.rows.end - window.rows.start;
                let count_peak = checked_sum(&[window.workspace_peak_bytes, plan.blocks])?;
                if count_peak > plan.workspace_bytes {
                    requested_end = next_row + smaller_window(count, count_peak, plan.workspace_bytes)?;
                    continue;
                }
                let host_bytes = plan
                    .workspace_bytes
                    .saturating_sub(window.workspace_peak_bytes);
                let Some(layout) =
                    OpeningLayout::measure(window.cache.matrix_caches(), plan.blocks, include_pad, host_bytes)?
                else {
                    requested_end =
                        next_row + smaller_window(count, plan.workspace_bytes.saturating_add(1), plan.workspace_bytes)?;
                    continue;
                };
                let evaluation = layout
                    .evaluation_bytes(count, point.len(), sources.active_count())?
                    .max(
                        terminal
                            .map(|check| check.workspace_bytes(count))
                            .transpose()?
                            .unwrap_or(0),
                    );
                let available = checked_sum(&[super::application::available_workspace(self, 0), window.upload_bytes])?;
                let budget = plan
                    .workspace_bytes
                    .min(available.saturating_sub(evaluation));
                let metadata = checked_sum(&[window.workspace_peak_bytes, layout.metadata_bytes])?;
                if metadata > budget {
                    requested_end = next_row + smaller_window(count, metadata, budget)?;
                    continue;
                }
                if sources.active_count() != 0 {
                    let opening = self.prepare_joint_opening_plan(
                        window.cache.matrix_caches(),
                        plan.blocks * D,
                        window.rows.start,
                        include_pad,
                        &layout,
                    )?;
                    let values = self.eval_joint_openings(&opening, sources, point, witness_count, assignment_width)?;
                    for (result, value) in result.iter_mut().zip(values) {
                        if include_pad {
                            result.eval_k = value.eval_k;
                        }
                        for (matrix, contribution) in result.eval_a.iter_mut().zip(value.eval_a) {
                            for (coefficient, value) in matrix.iter_mut().zip(contribution) {
                                *coefficient += value;
                            }
                        }
                    }
                    drop(opening);
                }
                if let Some(check) = terminal {
                    if let Some(row) = check.check_window(self, plan, &window)? {
                        return Ok(neo_reductions::superneo_eval::TerminalEvaluations {
                            openings: result,
                            first_unsatisfied_row: Some(row),
                        });
                    }
                }
                next_row = window.rows.end;
                drop(window);
                break;
            }
        }
        Ok(neo_reductions::superneo_eval::TerminalEvaluations {
            openings: result,
            first_unsatisfied_row: None,
        })
    }

    fn prepare_joint_opening_plan(
        &self,
        matrices: &[SuperneoMatrixCache],
        scalar_columns: usize,
        row_start: usize,
        include_pad: bool,
        layout: &OpeningLayout,
    ) -> Result<CudaJointOpeningPlan, CudaError> {
        let first = matrices
            .first()
            .ok_or(CudaError::Shape("openings need application matrices"))?;
        let (rows, columns, _) = first.compact_explicit_shape();
        let blocks = scalar_columns.div_ceil(D);
        if rows == 0 || columns != scalar_columns || blocks == 0 {
            return Err(CudaError::Shape("opening matrix shape is invalid"));
        }
        let matrix_count = matrices.len() + 1;
        let pad_blocks = if include_pad { blocks } else { 0 };
        let mut active_offsets = Vec::with_capacity(layout.active + 1);
        active_offsets.resize(pad_blocks + 1, 0u64);
        let mut active_local_masks = Vec::with_capacity(layout.active);
        active_local_masks.resize(pad_blocks, 0u64);
        let mut active_blocks_host = Vec::with_capacity(layout.active);
        for block in 0..pad_blocks {
            active_blocks_host.push(encoded_block(0, blocks, block)?);
        }
        let mut matrix_active_offsets_host = Vec::with_capacity(matrix_count + 1);
        matrix_active_offsets_host.extend([0u32, active_count_u32(&active_blocks_host)?]);
        let mut matrix_identity = Vec::with_capacity(matrix_count);
        matrix_identity.push(1u32);
        let mut entries = Vec::<[u32; 2]>::with_capacity(layout.entries);
        let mut pattern_offsets = Vec::with_capacity(layout.patterns + 1);
        pattern_offsets.push(0u32);
        let mut pattern_locals = Vec::with_capacity(layout.coefficients);
        let mut pattern_coefficients = Vec::with_capacity(layout.coefficients);
        let mut parallel_form_lists = Vec::with_capacity(layout.parallel);
        let mut tiled_form_lists = Vec::with_capacity(layout.tiled);
        let mut tiled_form_tile_offsets = Vec::with_capacity(layout.tiled + 1);
        tiled_form_tile_offsets.push(0u32);
        let mut tiled_form_tiles = Vec::with_capacity(layout.tiles * 3);
        let mut pattern_bases = Vec::with_capacity(matrices.len());
        let mut pattern_count = 0usize;
        for matrix in matrices {
            pattern_bases
                .push(u32::try_from(pattern_count).map_err(|_| CudaError::Shape("opening pattern count exceeds u32"))?);
            let parts = matrix
                .compact_device_parts()
                .expect("validated compact matrix");
            pattern_count += parts.dense_offsets.len().saturating_sub(1);
        }
        let transposes = matrices
            .par_iter()
            .zip(pattern_bases)
            .zip(&layout.spans)
            .map(|((matrix, pattern_base), layers)| BlockTranspose::new(matrix, layers, rows, columns, pattern_base))
            .collect::<Result<Vec<_>, _>>()?;
        for (application, (matrix, transpose)) in matrices.iter().zip(transposes).enumerate() {
            let matrix_index = application + 1;
            let parts = matrix
                .compact_device_parts()
                .expect("validated compact matrix");
            let coefficient_base = u32::try_from(pattern_locals.len())
                .map_err(|_| CudaError::Shape("opening coefficient count exceeds u32"))?;
            for &end in parts.dense_offsets.iter().skip(1) {
                pattern_offsets.push(
                    coefficient_base
                        .checked_add(end)
                        .ok_or(CudaError::Shape("opening coefficient count exceeds u32"))?,
                );
            }
            pattern_locals.extend_from_slice(parts.dense_locals);
            pattern_coefficients.extend_from_slice(parts.dense_coefficients);
            matrix_identity.push(u32::from(transpose.identity));
            let entry_base = entries.len() as u64;
            for (block, &active) in transpose.active.iter().enumerate() {
                if !active {
                    continue;
                }
                let active_index = active_blocks_host.len();
                active_blocks_host.push(encoded_block(matrix_index, blocks, block)?);
                active_local_masks.push(transpose.local_masks[block]);
                active_offsets.push(entry_base + u64::from(transpose.offsets[block + 1]));
                let count = (transpose.offsets[block + 1] - transpose.offsets[block]) as usize;
                if count < PARALLEL_FORM_LIST_THRESHOLD || transpose.identity {
                    continue;
                }
                let mut locals = transpose.local_masks[block];
                while locals != 0 {
                    let local = locals.trailing_zeros() as usize;
                    locals &= locals - 1;
                    let encoded = u32::try_from(active_index * D + local)
                        .map_err(|_| CudaError::Shape("opening list index exceeds u32"))?;
                    if count <= FORM_TILE_ENTRIES {
                        parallel_form_lists.push(encoded);
                    } else {
                        tiled_form_lists.push(encoded);
                        for start in (0..count).step_by(FORM_TILE_ENTRIES) {
                            tiled_form_tiles.extend([
                                encoded,
                                start as u32,
                                (count - start).min(FORM_TILE_ENTRIES) as u32,
                            ]);
                        }
                        tiled_form_tile_offsets.push(
                            u32::try_from(tiled_form_tiles.len() / 3)
                                .map_err(|_| CudaError::Shape("opening tile count exceeds u32"))?,
                        );
                    }
                }
            }
            entries.extend(transpose.entries);
            matrix_active_offsets_host.push(active_count_u32(&active_blocks_host)?);
        }
        let parallel_form_list_count = parallel_form_lists.len();
        let tiled_form_list_count = tiled_form_lists.len();
        let tiled_form_tile_count = tiled_form_tiles.len() / 3;
        let mut active_chunk_bases = Vec::with_capacity(layout.chunks);
        let mut matrix_chunk_offsets = Vec::with_capacity(matrix_count + 1);
        matrix_chunk_offsets.push(0u32);
        for matrix in 0..matrix_count {
            let start = matrix_active_offsets_host[matrix] as usize;
            let end = matrix_active_offsets_host[matrix + 1] as usize;
            for base in (start..end).step_by(CHUNK_BLOCKS) {
                active_chunk_bases.push(base as u32);
            }
            matrix_chunk_offsets.push(
                u32::try_from(active_chunk_bases.len())
                    .map_err(|_| CudaError::Shape("opening chunk count exceeds u32"))?,
            );
        }
        let geometric = self.prepare_geometric_opening_plans(layout)?;
        Ok(CudaJointOpeningPlan {
            active_offsets: self.buffer_from_slice(&active_offsets)?,
            active_local_masks: self.buffer_from_slice(super::nonempty(&active_local_masks))?,
            active_blocks: self.buffer_from_slice(if active_blocks_host.is_empty() {
                &[0u32]
            } else {
                &active_blocks_host
            })?,
            active_chunk_bases: self.buffer_from_slice(if active_chunk_bases.is_empty() {
                &[0u32]
            } else {
                &active_chunk_bases
            })?,
            matrix_active_offsets: self.buffer_from_slice(&matrix_active_offsets_host)?,
            matrix_active_offsets_host,
            matrix_chunk_offsets,
            matrix_identity: self.buffer_from_slice(&matrix_identity)?,
            entries: self.buffer_from_slice(if entries.is_empty() { &[[0u32; 2]] } else { &entries })?,
            pattern_offsets: self.buffer_from_slice(&pattern_offsets)?,
            pattern_locals: self.buffer_from_slice(if pattern_locals.is_empty() {
                &[0u8]
            } else {
                &pattern_locals
            })?,
            pattern_coefficients: self.buffer_from_slice(if pattern_coefficients.is_empty() {
                &[F::ZERO]
            } else {
                &pattern_coefficients
            })?,
            parallel_form_lists: self.buffer_from_slice(if parallel_form_lists.is_empty() {
                &[0u32]
            } else {
                &parallel_form_lists
            })?,
            tiled_form_lists: self.buffer_from_slice(if tiled_form_lists.is_empty() {
                &[0u32]
            } else {
                &tiled_form_lists
            })?,
            tiled_form_tile_offsets: self.buffer_from_slice(&tiled_form_tile_offsets)?,
            tiled_form_tiles: self.buffer_from_slice(if tiled_form_tiles.is_empty() {
                &[0u32]
            } else {
                &tiled_form_tiles
            })?,
            tiled_form_partials: self.buffer(tiled_form_tile_count.max(1) * 2 * size_of::<u64>())?,
            geometric,
            parallel_form_list_count,
            tiled_form_list_count,
            tiled_form_tile_count,
            matrix_count,
            rows,
            row_start,
            blocks,
        })
    }

    fn prepare_geometric_opening_plans(
        &self,
        layout: &OpeningLayout,
    ) -> Result<Vec<DeviceGeometricOpeningPlan>, CudaError> {
        let mut plans = Vec::new();
        for (application, layers) in layout.spans.iter().enumerate() {
            for layer in layers {
                plans.push(DeviceGeometricOpeningPlan {
                    matrix: application + 1,
                    starts: self.buffer_from_slice(&layer.starts)?,
                    ranks: self.buffer_from_slice(&layer.ranks)?,
                    shapes: self.buffer_from_slice(&layer.shapes)?,
                    offsets: self.buffer_from_slice(&layer.offsets)?,
                    rows: self.buffer_from_slice(&layer.rows)?,
                    coefficients: self.buffer_from_slice(&layer.coefficients)?,
                    longest: layer.longest,
                    span_count: layer.span_count(),
                });
            }
        }
        Ok(plans)
    }

    pub(super) fn eval_joint_openings<'a>(
        &self,
        plan: &CudaJointOpeningPlan,
        sources: impl Into<JointSources<'a>>,
        point: &[K],
        witness_count: usize,
        assignment_width: usize,
    ) -> Result<Vec<V1_2Evaluations<K>>, CudaError> {
        let sources = sources.into();
        let masks = sources.masks;
        let chi_len = 1usize
            .checked_shl(u32::try_from(point.len()).map_err(|_| CudaError::Shape("opening point is too long"))?)
            .ok_or(CudaError::Shape("one-joint opening tensor length overflow"))?;
        let carrier_width = plan
            .blocks
            .checked_mul(D)
            .ok_or(CudaError::Shape("one-joint opening carrier width overflow"))?;
        if point.is_empty()
            || witness_count == 0
            || assignment_width > carrier_width
            || carrier_width > chi_len
            || plan
                .row_start
                .checked_add(plan.rows)
                .is_none_or(|end| end > chi_len)
            || !sources.matches_joint(witness_count, plan.blocks)
        {
            return Err(CudaError::Shape("one-joint opening dimensions are invalid"));
        }
        let mut openings = vec![vec![vec![K::ZERO; D]; plan.matrix_count]; witness_count];
        let active_witness_ids = masks.active_witnesses();
        let masked_count = active_witness_ids.len();
        let dense_ids = sources.dense.map_or(&[][..], |dense| dense.active());
        let active_count = sources.active_count();
        if active_count == 0 {
            return Ok(to_evaluations(openings));
        }
        let max_blocks = plan
            .matrix_active_offsets_host
            .windows(2)
            .map(|range| (range[1] - range[0]) as usize)
            .max()
            .unwrap_or(0);
        let max_chunks = plan
            .matrix_chunk_offsets
            .windows(2)
            .map(|range| (range[1] - range[0]) as usize)
            .max()
            .unwrap_or(0);
        if max_blocks == 0 {
            return Ok(to_evaluations(openings));
        }
        let active_witnesses = self.buffer_from_slice(super::nonempty_u32(active_witness_ids))?;
        let dense_sources = self.buffer_from_slice(super::nonempty_u32(dense_ids))?;
        let low_bits = point.len() / 2;
        let chi_low = if low_bits == 0 {
            self.buffer_from_slice(&[1u64, 0])?
        } else {
            self.buffer((1usize << low_bits) * 2 * size_of::<u64>())?
        };
        let chi_high = self.buffer((1usize << (point.len() - low_bits)) * 2 * size_of::<u64>())?;
        let chi = self.buffer(checked_product(
            &[plan.rows, 2, size_of::<u64>()],
            "opening row weights overflow",
        )?)?;
        let shape = [
            plan.matrix_count as u64,
            plan.blocks as u64,
            plan.rows as u64,
            plan.rows as u64,
            carrier_width as u64,
            0,
            0,
            low_bits as u64,
            plan.row_start as u64,
        ];
        let mut tensor_resources = Vec::new();
        if low_bits != 0 {
            self.encode_joint_tensor_point(&chi_low, 0, &point[..low_bits], &mut tensor_resources)?;
        }
        self.encode_joint_tensor_point(&chi_high, 0, &point[low_bits..], &mut tensor_resources)?;
        let forms_kernels = &self.kernels.dec_forms;
        self.launch(Grid::Threads(plan.rows), |stream, config| {
            forms_kernels.dec_build_row_weights(
                stream,
                config,
                chi_low.ptr(),
                chi_high.ptr(),
                shape,
                chi.ptr(),
                plan.rows as u64,
            )
        })?;
        self.finish()?;
        drop(tensor_resources);

        let forms = self.buffer(checked_product(
            &[max_blocks, 2, D, size_of::<u64>()],
            "opening form size overflow",
        )?)?;
        let partials = self.buffer(checked_product(
            &[active_count, max_chunks, 2, PRODUCT_COEFFICIENTS, size_of::<u64>()],
            "opening partial size overflow",
        )?)?;
        let sums = self.buffer(active_count * 2 * PRODUCT_COEFFICIENTS * size_of::<u64>())?;
        let output_words = active_count * 2 * D;
        let output = self.buffer(output_words * size_of::<u64>())?;
        let chunk_matrices = self.buffer_from_slice(&vec![0u32; max_chunks])?;
        let max_spans = plan
            .geometric
            .iter()
            .map(|layer| layer.span_count)
            .max()
            .unwrap_or(0);
        let span_weights = self.buffer(max_spans.max(1) * 2 * size_of::<u64>())?;
        for matrix in 0..plan.matrix_count {
            let active_start = plan.matrix_active_offsets_host[matrix] as usize;
            let active_end = plan.matrix_active_offsets_host[matrix + 1] as usize;
            if active_start == active_end {
                continue;
            }
            let chunk_start = plan.matrix_chunk_offsets[matrix] as usize;
            let chunk_count = (plan.matrix_chunk_offsets[matrix + 1] as usize) - chunk_start;
            let chunk_offsets = self.buffer_from_slice(&[0u32, chunk_count as u32])?;
            let form_words = (active_end - active_start) * 2 * D;
            let masked_partial_words = masked_count * chunk_count * 2 * PRODUCT_COEFFICIENTS;
            let dense_partial_words = dense_ids.len() * chunk_count * 2 * PRODUCT_COEFFICIENTS;
            let sum_words = active_count * 2 * PRODUCT_COEFFICIENTS;
            let mut form_shape = shape;
            form_shape[5] = active_start as u64;
            form_shape[6] = active_end as u64;
            let tail_shape = [
                (active_end - active_start) as u64,
                active_count as u64,
                2,
                plan.blocks as u64,
                chunk_count as u64,
                0,
                masks.magnitudes() as u64,
                active_start as u64,
                masked_count as u64,
                dense_ids.len() as u64,
            ];
            let split = &self.kernels.decomposition;
            self.zero_words(&forms, form_words)?;
            self.launch(Grid::Threads(form_words), |stream, config| {
                forms_kernels.dec_build_ring_forms(
                    stream,
                    config,
                    plan.active_offsets.ptr(),
                    plan.active_local_masks.ptr(),
                    plan.matrix_identity.ptr(),
                    plan.entries.ptr(),
                    plan.pattern_offsets.ptr(),
                    chi.ptr(),
                    form_shape,
                    forms.ptr(),
                    plan.active_blocks.ptr(),
                    plan.pattern_locals.ptr(),
                    plan.pattern_coefficients.ptr(),
                    chi_low.ptr(),
                    chi_high.ptr(),
                    form_words as u64,
                )
            })?;
            if plan.parallel_form_list_count != 0 {
                let grid = Grid::groups(2 * plan.parallel_form_list_count, FORM_REDUCTION_THREADS);
                self.launch(grid, |stream, config| {
                    forms_kernels.dec_build_parallel_original_forms(
                        stream,
                        config,
                        plan.active_offsets.ptr(),
                        plan.entries.ptr(),
                        plan.pattern_offsets.ptr(),
                        chi.ptr(),
                        form_shape,
                        forms.ptr(),
                        plan.parallel_form_lists.ptr(),
                        plan.pattern_locals.ptr(),
                        plan.pattern_coefficients.ptr(),
                    )
                })?;
            }
            if plan.tiled_form_tile_count != 0 {
                let grid = Grid::groups(2 * plan.tiled_form_tile_count, FORM_REDUCTION_THREADS);
                self.launch(grid, |stream, config| {
                    forms_kernels.dec_build_parallel_original_form_tiles(
                        stream,
                        config,
                        plan.active_offsets.ptr(),
                        plan.entries.ptr(),
                        plan.pattern_offsets.ptr(),
                        chi.ptr(),
                        form_shape,
                        plan.tiled_form_tiles.ptr(),
                        plan.tiled_form_partials.ptr(),
                        plan.pattern_locals.ptr(),
                        plan.pattern_coefficients.ptr(),
                    )
                })?;
                let grid = Grid::groups(2 * plan.tiled_form_list_count, FORM_REDUCTION_THREADS);
                self.launch(grid, |stream, config| {
                    forms_kernels.dec_reduce_parallel_original_form_tiles(
                        stream,
                        config,
                        forms.ptr(),
                        plan.tiled_form_lists.ptr(),
                        plan.tiled_form_tile_offsets.ptr(),
                        plan.tiled_form_partials.ptr(),
                        form_shape,
                    )
                })?;
            }

            for layer in plan.geometric.iter().filter(|layer| layer.matrix == matrix) {
                self.launch(Grid::Threads(layer.span_count), |stream, config| {
                    forms_kernels.dec_geometric_span_weights(
                        stream,
                        config,
                        layer.offsets.ptr(),
                        layer.rows.ptr(),
                        layer.coefficients.ptr(),
                        chi.ptr(),
                        form_shape,
                        span_weights.ptr(),
                        layer.span_count as u64,
                    )
                })?;
                let lanes = (active_end - active_start) * D;
                self.launch(Grid::Threads(lanes), |stream, config| {
                    forms_kernels.dec_add_geometric_span_forms(
                        stream,
                        config,
                        layer.starts.ptr(),
                        layer.ranks.ptr(),
                        layer.shapes.ptr(),
                        span_weights.ptr(),
                        form_shape,
                        forms.ptr(),
                        plan.active_blocks.ptr(),
                        layer.longest,
                        lanes as u64,
                    )
                })?;
            }

            let slots = (active_end - active_start) * 2 * 14;
            self.launch(Grid::Threads(slots), |stream, config| {
                forms_kernels.dec_bar_ring_forms_in_place(stream, config, forms.ptr(), slots as u64)
            })?;
            let chunk_bases = plan
                .active_chunk_bases
                .at::<u32>(chunk_start * size_of::<u32>());
            let matrix_offsets = plan
                .matrix_active_offsets
                .at::<u32>(matrix * size_of::<u32>());

            if masked_partial_words != 0 {
                self.launch(Grid::Threads(masked_partial_words), |stream, config| {
                    split.dec_sparse_ring_partials(
                        stream,
                        config,
                        forms.ptr(),
                        masks.words().ptr(),
                        tail_shape,
                        partials.ptr(),
                        active_witnesses.ptr(),
                        plan.active_blocks.ptr(),
                        chunk_bases,
                        chunk_matrices.ptr(),
                        matrix_offsets,
                        active_witnesses.ptr(),
                        masked_partial_words as u64,
                    )
                })?;
            }
            if let Some(dense) = sources.dense.filter(|_| dense_partial_words != 0) {
                self.launch(Grid::Threads(dense_partial_words), |stream, config| {
                    split.dec_dense_ring_partials(
                        stream,
                        config,
                        forms.ptr(),
                        dense.words().ptr(),
                        tail_shape,
                        partials.ptr(),
                        dense_sources.ptr(),
                        plan.active_blocks.ptr(),
                        chunk_bases,
                        chunk_matrices.ptr(),
                        matrix_offsets,
                        dense_partial_words as u64,
                    )
                })?;
            }
            self.launch(Grid::Threads(sum_words), |stream, config| {
                split.dec_sparse_ring_sum_chunks(
                    stream,
                    config,
                    partials.ptr(),
                    tail_shape,
                    sums.ptr(),
                    chunk_offsets.ptr(),
                    sum_words as u64,
                )
            })?;
            self.launch(Grid::Threads(output_words), |stream, config| {
                split.dec_ring_reduce_phi81(stream, config, sums.ptr(), tail_shape, output.ptr())
            })?;
            self.finish()?;

            let words = self.read_buffer::<u64>(&output, output_words)?;
            let dense_witnesses = dense_ids
                .iter()
                .map(|&source| masks.witness_count() as u32 + source);
            for (active, witness) in active_witness_ids
                .iter()
                .copied()
                .chain(dense_witnesses)
                .enumerate()
            {
                let coefficients = &mut openings[witness as usize][matrix];
                let real = active * 2 * D;
                let imaginary = real + D;
                for coefficient in 0..D {
                    coefficients[coefficient] = K::from_coeffs([
                        F::from_u64(words[real + coefficient]),
                        F::from_u64(words[imaginary + coefficient]),
                    ]);
                }
            }
        }
        Ok(to_evaluations(openings))
    }
}

fn checked_product(values: &[usize], message: &'static str) -> Result<usize, CudaError> {
    values
        .iter()
        .try_fold(1usize, |product, &value| product.checked_mul(value))
        .ok_or(CudaError::Shape(message))
}

fn checked_sum(values: &[usize]) -> Result<usize, CudaError> {
    values
        .iter()
        .try_fold(0usize, |sum, &value| sum.checked_add(value))
        .ok_or(CudaError::Shape("opening workspace size overflow"))
}

fn encoded_block(matrix: usize, blocks: usize, block: usize) -> Result<u32, CudaError> {
    u32::try_from(
        matrix
            .checked_mul(blocks)
            .and_then(|base| base.checked_add(block))
            .ok_or(CudaError::Shape("one-joint opening block index overflow"))?,
    )
    .map_err(|_| CudaError::Shape("one-joint opening block index exceeds u32"))
}

fn active_count_u32(active: &[u32]) -> Result<u32, CudaError> {
    u32::try_from(active.len()).map_err(|_| CudaError::Shape("one-joint active opening count exceeds u32"))
}

fn to_evaluations(openings: Vec<Vec<Vec<K>>>) -> Vec<V1_2Evaluations<K>> {
    openings
        .into_iter()
        .map(|families| {
            let mut families = families.into_iter();
            V1_2Evaluations {
                eval_k: families.next().expect("the opening plan includes Pad"),
                eval_a: families.collect(),
            }
        })
        .collect()
}
