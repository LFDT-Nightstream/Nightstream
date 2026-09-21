//! Structure-static transpose and point-specific SuperNeo ring openings.

use std::mem::size_of;

use neo_ccs::V1_1Evaluations;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::SuperneoMatrixCache;
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use p3_field::PrimeCharacteristicRing;

use super::{Buffer, DeviceSeededRows, MetalCompactMatrix, MetalSession, MetalWitnessMasks};
use crate::MetalError;

#[path = "transpose.rs"]
mod transpose;
use transpose::BlockTranspose;

#[cfg(test)]
#[path = "../../../tests/unit/joint_openings.rs"]
mod tests;

const FORM_REDUCTION_THREADS: usize = 256;
const PARALLEL_FORM_LIST_THRESHOLD: usize = 128;
const FORM_TILE_ENTRIES: usize = 16 * 1024;
const CHUNK_BLOCKS: usize = 512;
const PRODUCT_COEFFICIENTS: usize = 2 * D - 1;

pub(super) struct MetalJointOpeningPlan {
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
    seeded: Option<DeviceSeededOpeningPlan>,
    geometric: Vec<DeviceGeometricOpeningPlan>,
    parallel_form_list_count: usize,
    tiled_form_list_count: usize,
    tiled_form_tile_count: usize,
    matrix_count: usize,
    rows: usize,
    blocks: usize,
}

struct DeviceSeededOpeningPlan {
    output_headers: Buffer,
    word_starts: Buffer,
    rotations: Buffer,
    active_indices: Buffer,
    segment_offsets: Buffer,
    segments: Buffer,
    group_count: usize,
}

struct DeviceGeometricOpeningPlan {
    groups: Buffer,
    segments: Buffer,
    runs: Buffer,
    group_count: usize,
}

impl MetalSession {
    pub(super) fn prepare_joint_opening_plan(
        &self,
        matrices: &[SuperneoMatrixCache],
        device_matrices: &[MetalCompactMatrix],
        scalar_columns: usize,
        seeded_rows: Option<&DeviceSeededRows>,
    ) -> Result<MetalJointOpeningPlan, MetalError> {
        let first = matrices
            .first()
            .ok_or(MetalError::Shape("openings need application matrices"))?;
        let (rows, columns, _) = first.compact_explicit_shape();
        let blocks = scalar_columns.div_ceil(D);
        if rows == 0 || columns != scalar_columns || blocks == 0 {
            return Err(MetalError::Shape("opening matrix shape is invalid"));
        }
        let matrix_count = matrices.len() + 1;
        let mut active_offsets = vec![0u64; blocks + 1];
        let mut active_local_masks = vec![0u64; blocks];
        let mut active_blocks_host = (0..blocks)
            .map(|block| encoded_block(0, blocks, block))
            .collect::<Result<Vec<_>, _>>()?;
        let mut matrix_active_offsets_host = vec![0u32, active_count_u32(&active_blocks_host)?];
        let mut matrix_identity = vec![1u32];
        let mut entries = Vec::<[u32; 2]>::new();
        let mut pattern_offsets = vec![0u32];
        let mut pattern_locals = Vec::new();
        let mut pattern_coefficients = Vec::new();
        let mut parallel_form_lists = Vec::new();
        let mut tiled_form_lists = Vec::new();
        let mut tiled_form_tile_offsets = vec![0u32];
        let mut tiled_form_tiles = Vec::new();
        for (application, matrix) in matrices.iter().enumerate() {
            let matrix_index = application + 1;
            let pattern_base = u32::try_from(pattern_offsets.len() - 1)
                .map_err(|_| MetalError::Shape("opening pattern count exceeds u32"))?;
            let transpose = BlockTranspose::new(matrix, rows, columns, pattern_base)?;
            let parts = matrix
                .compact_device_parts()
                .expect("validated compact matrix");
            let coefficient_base = u32::try_from(pattern_locals.len())
                .map_err(|_| MetalError::Shape("opening coefficient count exceeds u32"))?;
            for &end in parts.dense_offsets.iter().skip(1) {
                pattern_offsets.push(
                    coefficient_base
                        .checked_add(end)
                        .ok_or(MetalError::Shape("opening coefficient count exceeds u32"))?,
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
                        .map_err(|_| MetalError::Shape("opening list index exceeds u32"))?;
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
                                .map_err(|_| MetalError::Shape("opening tile count exceeds u32"))?,
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
        let mut active_chunk_bases = Vec::new();
        let mut matrix_chunk_offsets = vec![0u32];
        for matrix in 0..matrix_count {
            let start = matrix_active_offsets_host[matrix] as usize;
            let end = matrix_active_offsets_host[matrix + 1] as usize;
            for base in (start..end).step_by(CHUNK_BLOCKS) {
                active_chunk_bases.push(base as u32);
            }
            matrix_chunk_offsets.push(
                u32::try_from(active_chunk_bases.len())
                    .map_err(|_| MetalError::Shape("opening chunk count exceeds u32"))?,
            );
        }
        let seeded = self.prepare_seeded_opening_plan(matrices, seeded_rows, &active_blocks_host, blocks)?;
        let geometric = self.prepare_geometric_opening_plans(matrices, device_matrices, &active_blocks_host, blocks)?;
        Ok(MetalJointOpeningPlan {
            active_offsets: self.buffer_from_slice(&active_offsets)?,
            active_local_masks: self.buffer_from_slice(&active_local_masks)?,
            active_blocks: self.buffer_from_slice(&active_blocks_host)?,
            active_chunk_bases: self.buffer_from_slice(&active_chunk_bases)?,
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
            seeded,
            geometric,
            parallel_form_list_count,
            tiled_form_list_count,
            tiled_form_tile_count,
            matrix_count,
            rows,
            blocks,
        })
    }

    fn prepare_geometric_opening_plans(
        &self,
        matrices: &[SuperneoMatrixCache],
        device_matrices: &[MetalCompactMatrix],
        active_blocks: &[u32],
        blocks: usize,
    ) -> Result<Vec<DeviceGeometricOpeningPlan>, MetalError> {
        if matrices.len() != device_matrices.len() {
            return Err(MetalError::Shape("one-joint geometric device matrix count mismatch"));
        }
        let mut plans = Vec::new();
        for (application, (matrix, device)) in matrices.iter().zip(device_matrices).enumerate() {
            if matrix.compact_geometric_run_count() == 0 {
                continue;
            }
            let mut counts = vec![0u32; blocks];
            let mut invalid = false;
            matrix.for_each_compact_geometric_run(|index, row, start, len, _, _| {
                if u32::try_from(index).is_err() || u32::try_from(row).is_err() {
                    invalid = true;
                    return;
                }
                let Some(end) = start.checked_add(len) else {
                    invalid = true;
                    return;
                };
                if end > blocks * D {
                    invalid = true;
                    return;
                }
                for block in start / D..end.div_ceil(D) {
                    counts[block] = match counts[block].checked_add(1) {
                        Some(count) => count,
                        None => {
                            invalid = true;
                            return;
                        }
                    };
                }
            });
            if invalid {
                return Err(MetalError::Shape(
                    "one-joint geometric opening metadata exceeds device limits",
                ));
            }

            let mut offsets = Vec::with_capacity(blocks + 1);
            offsets.push(0u32);
            for &count in &counts {
                offsets.push(
                    offsets
                        .last()
                        .copied()
                        .expect("geometric opening offset")
                        .checked_add(count)
                        .ok_or(MetalError::Shape(
                            "one-joint geometric opening segment count exceeds u32",
                        ))?,
                );
            }
            let mut segments = vec![[0u32; 2]; offsets[blocks] as usize];
            let mut cursor = offsets[..blocks].to_vec();
            matrix.for_each_compact_geometric_run(|index, row, start, len, _, _| {
                let end = start + len;
                for block in start / D..end.div_ceil(D) {
                    let destination = cursor[block] as usize;
                    cursor[block] += 1;
                    segments[destination] = [row as u32, index as u32];
                }
            });

            let mut groups = Vec::<[u32; 4]>::new();
            for block in 0..blocks {
                if offsets[block] == offsets[block + 1] {
                    continue;
                }
                let encoded = encoded_block(application + 1, blocks, block)?;
                let active = active_blocks
                    .binary_search(&encoded)
                    .map_err(|_| MetalError::Shape("one-joint geometric block is absent from the transpose"))?;
                groups.push([
                    u32::try_from(active)
                        .map_err(|_| MetalError::Shape("one-joint geometric active block exceeds u32"))?,
                    u32::try_from(block)
                        .map_err(|_| MetalError::Shape("one-joint geometric column block exceeds u32"))?,
                    offsets[block],
                    offsets[block + 1],
                ]);
            }
            let group_count = groups.len();
            plans.push(DeviceGeometricOpeningPlan {
                groups: self.buffer_from_slice(&groups)?,
                segments: self.buffer_from_slice(&segments)?,
                runs: device.geometric_runs.clone(),
                group_count,
            });
        }
        Ok(plans)
    }

    fn prepare_seeded_opening_plan(
        &self,
        matrices: &[SuperneoMatrixCache],
        device: Option<&DeviceSeededRows>,
        active_blocks: &[u32],
        blocks: usize,
    ) -> Result<Option<DeviceSeededOpeningPlan>, MetalError> {
        let has_seeded = matrices
            .iter()
            .any(SuperneoMatrixCache::has_compact_seeded_phi81_blocks);
        if !has_seeded {
            return Ok(None);
        }
        let Some(device) = device else {
            return Err(MetalError::Shape(
                "one-joint seeded openings are unsupported for this matrix form",
            ));
        };
        let mut by_active = vec![Vec::<[u32; 2]>::new(); active_blocks.len()];
        let mut output_index = 0usize;
        let mut word_base = 0usize;
        for (application, matrix) in matrices.iter().enumerate() {
            for seeded in matrix.compact_seeded_phi81_blocks() {
                for _ in 0..seeded.kappa() {
                    let output = u32::try_from(output_index)
                        .map_err(|_| MetalError::Shape("one-joint seeded opening output exceeds u32"))?;
                    for (word, &start) in seeded.word_starts().iter().enumerate() {
                        let global_word = u32::try_from(word_base + word)
                            .map_err(|_| MetalError::Shape("one-joint seeded opening word exceeds u32"))?;
                        let end = start
                            .checked_add(seeded.word_width())
                            .ok_or(MetalError::Shape("one-joint seeded opening range overflow"))?;
                        for column_block in start / D..end.div_ceil(D) {
                            let encoded = encoded_block(application + 1, blocks, column_block)?;
                            let active = active_blocks.binary_search(&encoded).map_err(|_| {
                                MetalError::Shape("one-joint seeded opening block is absent from the transpose")
                            })?;
                            by_active[active].push([output, global_word]);
                        }
                    }
                    output_index += 1;
                }
                word_base += seeded.word_starts().len();
            }
        }
        let expected_outputs = device.output_headers.length() as usize / (9 * size_of::<u64>());
        if output_index != expected_outputs {
            return Err(MetalError::Shape("one-joint seeded opening metadata is inconsistent"));
        }

        let mut active_indices = Vec::new();
        let mut offsets = vec![0u32];
        let mut segments = Vec::new();
        for (active, entries) in by_active.into_iter().enumerate() {
            if entries.is_empty() {
                continue;
            }
            active_indices.push(
                u32::try_from(active).map_err(|_| MetalError::Shape("one-joint seeded active block exceeds u32"))?,
            );
            segments.extend(entries.into_iter().flatten());
            offsets.push(
                u32::try_from(segments.len() / 2)
                    .map_err(|_| MetalError::Shape("one-joint seeded segment count exceeds u32"))?,
            );
        }
        if active_indices.is_empty() || segments.is_empty() {
            return Err(MetalError::Shape("one-joint seeded opening plan is empty"));
        }
        Ok(Some(DeviceSeededOpeningPlan {
            output_headers: device.output_headers.clone(),
            word_starts: device.word_starts.clone(),
            rotations: device.rotations.clone(),
            active_indices: self.buffer_from_slice(&active_indices)?,
            segment_offsets: self.buffer_from_slice(&offsets)?,
            segments: self.buffer_from_slice(&segments)?,
            group_count: active_indices.len(),
        }))
    }

    pub(super) fn eval_joint_openings(
        &self,
        plan: &MetalJointOpeningPlan,
        _seeded_rows: Option<&DeviceSeededRows>,
        masks: &MetalWitnessMasks,
        point: &[K],
        witness_count: usize,
        assignment_width: usize,
    ) -> Result<Vec<V1_1Evaluations<K>>, MetalError> {
        let chi_len = 1usize
            .checked_shl(u32::try_from(point.len()).map_err(|_| MetalError::Shape("opening point is too long"))?)
            .ok_or(MetalError::Shape("one-joint opening tensor length overflow"))?;
        let carrier_width = plan
            .blocks
            .checked_mul(D)
            .ok_or(MetalError::Shape("one-joint opening carrier width overflow"))?;
        if point.is_empty()
            || witness_count == 0
            || assignment_width > carrier_width
            || carrier_width > chi_len
            || plan.rows > chi_len
            || !masks.matches_joint(witness_count, plan.blocks)
        {
            return Err(MetalError::Shape("one-joint opening dimensions are invalid"));
        }
        let mut openings = vec![vec![vec![K::ZERO; D]; plan.matrix_count]; witness_count];
        let active_witness_ids = masks.active_witnesses();
        let active_count = active_witness_ids.len();
        if active_count == 0 {
            return Ok(to_evaluations(openings));
        }
        let active_witnesses = self.buffer_from_slice(active_witness_ids)?;
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
        ];
        let weight_shape = self.buffer_from_slice(&shape)?;
        let command = self.command_buffer("nightstream.pi_ccs.opening.weights")?;
        let mut tensor_resources = Vec::new();
        if low_bits != 0 {
            self.encode_joint_tensor_point(&command, &chi_low, 0, &point[..low_bits], &mut tensor_resources)?;
        }
        self.encode_joint_tensor_point(&command, &chi_high, 0, &point[low_bits..], &mut tensor_resources)?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.dec_build_row_weights);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&chi_low), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&chi_high), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&weight_shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&chi), 0, 3);
        }
        self.dispatch(&encoder, &self.dec_build_row_weights, plan.rows);
        encoder.endEncoding();
        self.finish(&command)?;
        drop(tensor_resources);

        // A matrix opening is an independent sum. Reuse one matrix's storage
        // instead of keeping every matrix form live at the same point.
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
            let partial_words = active_count * chunk_count * 2 * PRODUCT_COEFFICIENTS;
            let sum_words = active_count * 2 * PRODUCT_COEFFICIENTS;
            let mut shape = shape;
            shape[5] = active_start as u64;
            shape[6] = active_end as u64;
            let form_shape = self.buffer_from_slice(&shape)?;
            let tail_shape = self.buffer_from_slice(&[
                (active_end - active_start) as u64,
                active_count as u64,
                2,
                plan.blocks as u64,
                chunk_count as u64,
                0,
                masks.magnitudes() as u64,
                active_start as u64,
            ])?;
            let command = self.command_buffer("nightstream.pi_ccs.joint.openings")?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.opening.forms")));
            encoder.setComputePipelineState(&self.dec_build_ring_forms);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&plan.pattern_locals), 0, 11);
                encoder.setBuffer_offset_atIndex(Some(&plan.pattern_coefficients), 0, 12);
                encoder.setBuffer_offset_atIndex(Some(&chi_low), 0, 13);
                encoder.setBuffer_offset_atIndex(Some(&chi_high), 0, 14);
                encoder.setBuffer_offset_atIndex(Some(&plan.active_offsets), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.active_local_masks), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.matrix_identity), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&plan.entries), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&plan.pattern_offsets), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&chi), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&form_shape), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(&forms), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(&plan.active_blocks), 0, 8);
            }
            self.dispatch(&encoder, &self.dec_build_ring_forms, form_words);
            encoder.endEncoding();

            if plan.parallel_form_list_count != 0 {
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.dec_build_parallel_original_forms);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&plan.pattern_locals), 0, 11);
                    encoder.setBuffer_offset_atIndex(Some(&plan.pattern_coefficients), 0, 12);
                    encoder.setBuffer_offset_atIndex(Some(&plan.active_offsets), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&plan.active_local_masks), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&plan.entries), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&plan.pattern_offsets), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&chi), 0, 5);
                    encoder.setBuffer_offset_atIndex(Some(&form_shape), 0, 6);
                    encoder.setBuffer_offset_atIndex(Some(&forms), 0, 7);
                    encoder.setBuffer_offset_atIndex(Some(&plan.parallel_form_lists), 0, 9);
                }
                self.dispatch_threadgroups(
                    &encoder,
                    &self.dec_build_parallel_original_forms,
                    2 * plan.parallel_form_list_count,
                    FORM_REDUCTION_THREADS,
                );
                encoder.endEncoding();
            }
            if plan.tiled_form_tile_count != 0 {
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.dec_build_parallel_original_form_tiles);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&plan.pattern_locals), 0, 11);
                    encoder.setBuffer_offset_atIndex(Some(&plan.pattern_coefficients), 0, 12);
                    encoder.setBuffer_offset_atIndex(Some(&plan.active_offsets), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&plan.active_local_masks), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&plan.entries), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&plan.pattern_offsets), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&chi), 0, 5);
                    encoder.setBuffer_offset_atIndex(Some(&form_shape), 0, 6);
                    encoder.setBuffer_offset_atIndex(Some(&plan.tiled_form_tiles), 0, 9);
                    encoder.setBuffer_offset_atIndex(Some(&plan.tiled_form_partials), 0, 10);
                }
                self.dispatch_threadgroups(
                    &encoder,
                    &self.dec_build_parallel_original_form_tiles,
                    2 * plan.tiled_form_tile_count,
                    FORM_REDUCTION_THREADS,
                );
                encoder.endEncoding();

                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.dec_reduce_parallel_original_form_tiles);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&forms), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&plan.tiled_form_lists), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&plan.tiled_form_tile_offsets), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&plan.tiled_form_partials), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&form_shape), 0, 4);
                }
                self.dispatch_threadgroups(
                    &encoder,
                    &self.dec_reduce_parallel_original_form_tiles,
                    2 * plan.tiled_form_list_count,
                    FORM_REDUCTION_THREADS,
                );
                encoder.endEncoding();
            }

            for geometric in &plan.geometric {
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.dec_add_geometric_ring_forms);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&geometric.groups), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&geometric.segments), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&geometric.runs), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&chi), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&form_shape), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&forms), 0, 5);
                }
                self.dispatch(
                    &encoder,
                    &self.dec_add_geometric_ring_forms,
                    geometric.group_count * 2 * D,
                );
                encoder.endEncoding();
            }

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.dec_bar_ring_forms_in_place);
            unsafe { encoder.setBuffer_offset_atIndex(Some(&forms), 0, 0) };
            self.dispatch(
                &encoder,
                &self.dec_bar_ring_forms_in_place,
                (active_end - active_start) * 2 * 14,
            );
            encoder.endEncoding();

            let seeded_scratch = if let Some(seeded) = &plan.seeded {
                let words = checked_product(&[seeded.group_count, 2, D], "seeded opening size overflow")?;
                let scratch = self.buffer(words * size_of::<u64>())?;
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.dec_build_seeded_ring_forms);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&seeded.output_headers), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&seeded.word_starts), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&seeded.rotations), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&seeded.segment_offsets), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&seeded.segments), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&chi), 0, 5);
                    encoder.setBuffer_offset_atIndex(Some(&form_shape), 0, 6);
                    encoder.setBuffer_offset_atIndex(Some(&scratch), 0, 7);
                    encoder.setBuffer_offset_atIndex(Some(&plan.active_blocks), 0, 8);
                    encoder.setBuffer_offset_atIndex(Some(&seeded.active_indices), 0, 9);
                }
                self.dispatch(&encoder, &self.dec_build_seeded_ring_forms, words);
                encoder.endEncoding();

                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.dec_add_bar_seeded_ring_forms);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&scratch), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&forms), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&seeded.active_indices), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&form_shape), 0, 3);
                }
                self.dispatch(&encoder, &self.dec_add_bar_seeded_ring_forms, words);
                encoder.endEncoding();
                Some(scratch)
            } else {
                None
            };

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.dec_sparse_ring_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&forms), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(masks.words()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&tail_shape), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&active_witnesses), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&plan.active_blocks), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&plan.active_chunk_bases), chunk_start * size_of::<u32>(), 6);
                encoder.setBuffer_offset_atIndex(Some(&chunk_matrices), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(&plan.matrix_active_offsets), matrix * size_of::<u32>(), 8);
                encoder.setBuffer_offset_atIndex(Some(&active_witnesses), 0, 9);
            }
            self.dispatch(&encoder, &self.dec_sparse_ring_partials, partial_words);
            encoder.endEncoding();

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.dec_sparse_ring_sum_chunks);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&tail_shape), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&sums), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&chunk_offsets), 0, 3);
            }
            self.dispatch(&encoder, &self.dec_sparse_ring_sum_chunks, sum_words);
            encoder.endEncoding();

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.dec_ring_reduce_phi81);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&sums), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&tail_shape), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
            }
            self.dispatch(&encoder, &self.dec_ring_reduce_phi81, output_words);
            encoder.endEncoding();
            self.finish(&command)?;
            drop(seeded_scratch);

            let words = self.read_buffer::<u64>(&output, output_words);
            for (active, &witness) in active_witness_ids.iter().enumerate() {
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

fn checked_product(values: &[usize], message: &'static str) -> Result<usize, MetalError> {
    values
        .iter()
        .try_fold(1usize, |product, &value| product.checked_mul(value))
        .ok_or(MetalError::Shape(message))
}

fn encoded_block(matrix: usize, blocks: usize, block: usize) -> Result<u32, MetalError> {
    u32::try_from(
        matrix
            .checked_mul(blocks)
            .and_then(|base| base.checked_add(block))
            .ok_or(MetalError::Shape("one-joint opening block index overflow"))?,
    )
    .map_err(|_| MetalError::Shape("one-joint opening block index exceeds u32"))
}

fn active_count_u32(active: &[u32]) -> Result<u32, MetalError> {
    u32::try_from(active.len()).map_err(|_| MetalError::Shape("one-joint active opening count exceeds u32"))
}

fn to_evaluations(openings: Vec<Vec<Vec<K>>>) -> Vec<V1_1Evaluations<K>> {
    openings
        .into_iter()
        .map(|families| {
            let mut families = families.into_iter();
            V1_1Evaluations {
                eval_k: families.next().expect("the opening plan includes Pad"),
                eval_a: families.collect(),
            }
        })
        .collect()
}
