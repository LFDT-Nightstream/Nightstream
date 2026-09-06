//! Structure-static transpose and point-specific SuperNeo ring openings.

use std::mem::size_of;

use neo_ccs::V1_1Evaluations;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::SuperneoMatrixCache;
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;

use super::{Buffer, DeviceSeededRows, MetalCompactMatrix, MetalSession, MetalWitnessMasks};
use crate::MetalError;

const FORM_REDUCTION_THREADS: usize = 256;
const PARALLEL_FORM_LIST_THRESHOLD: usize = 128;
const FORM_TILE_ENTRIES: usize = 16 * 1024;
const CHUNK_BLOCKS: usize = 512;
const PRODUCT_COEFFICIENTS: usize = 2 * D - 1;

pub(super) struct MetalJointOpeningPlan {
    active_local_offsets: Buffer,
    active_entry_bases: Buffer,
    active_blocks: Buffer,
    active_chunk_bases: Buffer,
    active_chunk_matrices: Buffer,
    matrix_active_offsets: Buffer,
    matrix_chunk_offsets: Buffer,
    matrix_identity: Buffer,
    entry_rows: Buffer,
    entry_coefficients: Buffer,
    parallel_form_lists: Buffer,
    tiled_form_lists: Buffer,
    tiled_form_tile_offsets: Buffer,
    tiled_form_tiles: Buffer,
    tiled_form_partials: Buffer,
    seeded: Option<DeviceSeededOpeningPlan>,
    geometric: Vec<DeviceGeometricOpeningPlan>,
    active_block_count: usize,
    active_chunk_count: usize,
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

struct MatrixLayout {
    offsets: Vec<u32>,
    active: Vec<bool>,
    identity: bool,
    entry_count: usize,
}

impl MetalSession {
    pub(super) fn prepare_joint_opening_plan(
        &self,
        matrices: &[SuperneoMatrixCache],
        device_matrices: &[MetalCompactMatrix],
        scalar_columns: usize,
        seeded_rows: Option<&DeviceSeededRows>,
    ) -> Result<MetalJointOpeningPlan, MetalError> {
        let Some(first) = matrices.first() else {
            return Err(MetalError::Shape("one-joint openings require application matrices"));
        };
        let (scalar_rows, expected_columns, _) = first.compact_explicit_shape();
        let blocks = scalar_columns.div_ceil(D);
        if scalar_rows == 0 || expected_columns != scalar_columns || blocks == 0 {
            return Err(MetalError::Shape("one-joint opening matrix shape is invalid"));
        }
        let local_slots = checked_product(&[blocks, D], "one-joint opening slot count overflow")?;
        let matrix_stride = local_slots
            .checked_add(1)
            .ok_or(MetalError::Shape("one-joint opening slot count overflow"))?;
        let total_coefficients = matrices
            .iter()
            .map(SuperneoMatrixCache::compact_explicit_coefficient_count)
            .try_fold(0usize, |total, count| total.checked_add(count))
            .ok_or(MetalError::Shape("one-joint opening coefficient count overflow"))?;

        let layouts = matrices
            .par_iter()
            .map(|matrix| {
                let (rows, columns, identity) = matrix.compact_explicit_shape();
                if rows != scalar_rows || columns != scalar_columns || u32::try_from(rows - 1).is_err() {
                    return Err(MetalError::Shape("one-joint opening matrices have inconsistent shapes"));
                }
                let mut active = vec![false; blocks];
                for block in matrix.compact_seeded_column_blocks() {
                    if block >= blocks {
                        return Err(MetalError::Shape("one-joint seeded opening block is out of range"));
                    }
                    active[block] = true;
                }
                let mut geometric_invalid = false;
                matrix.for_each_compact_geometric_run(|_, _, start, len, _, _| {
                    let Some(end) = start.checked_add(len) else {
                        geometric_invalid = true;
                        return;
                    };
                    if end > scalar_columns {
                        geometric_invalid = true;
                        return;
                    }
                    active[start / D..end.div_ceil(D)].fill(true);
                });
                if geometric_invalid {
                    return Err(MetalError::Shape("one-joint geometric opening range is invalid"));
                }
                if identity {
                    active[..rows.div_ceil(D)].fill(true);
                    return Ok(MatrixLayout {
                        offsets: Vec::new(),
                        active,
                        identity,
                        entry_count: 0,
                    });
                }

                let entry_count = matrix.compact_explicit_coefficient_count();
                if u32::try_from(entry_count).is_err() {
                    return Err(MetalError::Shape("one-joint matrix opening count exceeds u32"));
                }
                let mut offsets = vec![0u32; matrix_stride];
                let mut invalid = false;
                for row in 0..rows {
                    matrix.for_each_compact_explicit_row_coefficient(row, |block, local, _| {
                        let block = block as usize;
                        let local = local as usize;
                        if block >= blocks || local >= D {
                            invalid = true;
                            return;
                        }
                        active[block] = true;
                        let slot = block * D + local + 1;
                        match offsets[slot].checked_add(1) {
                            Some(next) => offsets[slot] = next,
                            None => invalid = true,
                        }
                    });
                }
                if invalid {
                    return Err(MetalError::Shape("one-joint opening coordinates are invalid"));
                }
                for slot in 0..local_slots {
                    offsets[slot + 1] = offsets[slot + 1]
                        .checked_add(offsets[slot])
                        .ok_or(MetalError::Shape("one-joint opening offset overflow"))?;
                }
                if offsets[local_slots] as usize != entry_count {
                    return Err(MetalError::Shape("one-joint opening coefficient count changed"));
                }
                Ok(MatrixLayout {
                    offsets,
                    active,
                    identity,
                    entry_count,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut entry_bases = Vec::with_capacity(layouts.len());
        let mut entry_base = 0usize;
        for layout in &layouts {
            entry_bases.push(entry_base);
            entry_base = entry_base
                .checked_add(layout.entry_count)
                .ok_or(MetalError::Shape("one-joint opening coefficient count overflow"))?;
        }
        if entry_base != total_coefficients {
            return Err(MetalError::Shape("one-joint opening transpose is incomplete"));
        }

        // The first device batch evaluates Pad; the remaining batches evaluate
        // the genuine CCS matrices. Pad is not part of the CCS matrix list.
        // Application matrices follow in their canonical structure order.
        let matrix_count = matrices.len() + 1;
        let mut active_local_offsets = Vec::new();
        let mut active_entry_bases = Vec::new();
        let mut active_blocks_host = Vec::new();
        let mut matrix_active_offsets_host = vec![0u32];
        let mut matrix_identity = vec![1u32];
        for block in 0..blocks {
            active_blocks_host
                .push(u32::try_from(block).map_err(|_| MetalError::Shape("one-joint identity block exceeds u32"))?);
            active_entry_bases.push(0);
            active_local_offsets.resize(active_local_offsets.len() + D + 1, 0);
        }
        matrix_active_offsets_host.push(
            u32::try_from(active_blocks_host.len())
                .map_err(|_| MetalError::Shape("one-joint active opening count exceeds u32"))?,
        );

        let mut parallel_form_lists = Vec::new();
        let mut tiled_form_lists = Vec::new();
        let mut tiled_form_tile_offsets = vec![0u32];
        let mut tiled_form_tiles = Vec::new();
        for (application, layout) in layouts.iter().enumerate() {
            let matrix = application + 1;
            matrix_identity.push(u32::from(layout.identity));
            if layout.identity {
                for (block, _) in layout
                    .active
                    .iter()
                    .enumerate()
                    .filter(|(_, active)| **active)
                {
                    active_blocks_host.push(encoded_block(matrix, blocks, block)?);
                    active_entry_bases.push(0);
                    active_local_offsets.resize(active_local_offsets.len() + D + 1, 0);
                }
                matrix_active_offsets_host.push(active_count_u32(&active_blocks_host)?);
                continue;
            }

            let offsets = &layout.offsets;
            let matrix_entry_base = entry_bases[application];
            for (block, _) in layout
                .active
                .iter()
                .enumerate()
                .filter(|(_, active)| **active)
            {
                let active = active_blocks_host.len();
                active_blocks_host.push(encoded_block(matrix, blocks, block)?);
                let block_start = offsets[block * D];
                active_entry_bases.push(
                    u64::try_from(matrix_entry_base + block_start as usize)
                        .map_err(|_| MetalError::Shape("one-joint opening entry base exceeds u64"))?,
                );
                for local in 0..D {
                    let entries = (offsets[block * D + local + 1] - offsets[block * D + local]) as usize;
                    if entries >= PARALLEL_FORM_LIST_THRESHOLD {
                        let encoded = u32::try_from(active * D + local)
                            .map_err(|_| MetalError::Shape("one-joint opening list exceeds u32"))?;
                        if entries <= FORM_TILE_ENTRIES {
                            parallel_form_lists.push(encoded);
                        } else {
                            tiled_form_lists.push(encoded);
                            for relative_start in (0..entries).step_by(FORM_TILE_ENTRIES) {
                                let tile_entries = (entries - relative_start).min(FORM_TILE_ENTRIES);
                                tiled_form_tiles.extend([
                                    encoded,
                                    u32::try_from(relative_start)
                                        .map_err(|_| MetalError::Shape("one-joint opening tile start exceeds u32"))?,
                                    u32::try_from(tile_entries)
                                        .map_err(|_| MetalError::Shape("one-joint opening tile size exceeds u32"))?,
                                ]);
                            }
                            tiled_form_tile_offsets.push(
                                u32::try_from(tiled_form_tiles.len() / 3)
                                    .map_err(|_| MetalError::Shape("one-joint opening tile count exceeds u32"))?,
                            );
                        }
                    }
                }
                active_local_offsets.extend(
                    offsets[block * D..=block * D + D]
                        .iter()
                        .map(|&offset| offset - block_start),
                );
            }
            matrix_active_offsets_host.push(active_count_u32(&active_blocks_host)?);
        }

        let entry_rows = self.buffer(total_coefficients.max(1) * size_of::<u32>())?;
        let entry_coefficients = self.buffer(total_coefficients.max(1) * size_of::<u64>())?;
        let rows = unsafe {
            std::slice::from_raw_parts_mut(entry_rows.contents().as_ptr().cast::<u32>(), total_coefficients.max(1))
        };
        let coefficients = unsafe {
            std::slice::from_raw_parts_mut(
                entry_coefficients.contents().as_ptr().cast::<u64>(),
                total_coefficients.max(1),
            )
        };
        let mut row_tail = &mut rows[..total_coefficients];
        let mut coefficient_tail = &mut coefficients[..total_coefficients];
        let mut fill_parts = Vec::with_capacity(matrices.len());
        for (matrix, layout) in matrices.iter().zip(layouts) {
            let (matrix_rows, remaining_rows) = row_tail.split_at_mut(layout.entry_count);
            let (matrix_coefficients, remaining_coefficients) = coefficient_tail.split_at_mut(layout.entry_count);
            fill_parts.push((matrix, layout, matrix_rows, matrix_coefficients));
            row_tail = remaining_rows;
            coefficient_tail = remaining_coefficients;
        }
        fill_parts
            .into_par_iter()
            .map(|(matrix, mut layout, rows, coefficients)| {
                if layout.identity {
                    return Ok(());
                }
                let mut invalid = false;
                let mut filled = 0usize;
                for row in 0..scalar_rows {
                    matrix.for_each_compact_explicit_row_coefficient(row, |block, local, coefficient| {
                        let slot = block as usize * D + local as usize;
                        let destination = layout.offsets[slot] as usize;
                        if destination >= rows.len() {
                            invalid = true;
                            return;
                        }
                        layout.offsets[slot] += 1;
                        rows[destination] = row as u32;
                        coefficients[destination] = coefficient.as_canonical_u64();
                        filled += 1;
                    });
                }
                if invalid || filled != rows.len() {
                    return Err(MetalError::Shape("one-joint opening transpose changed during fill"));
                }
                Ok(())
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.record_host_write(total_coefficients * (size_of::<u32>() + size_of::<u64>()));

        let parallel_form_list_count = parallel_form_lists.len();
        let tiled_form_list_count = tiled_form_lists.len();
        let tiled_form_tile_count = tiled_form_tiles.len() / 3;
        if parallel_form_lists.is_empty() {
            parallel_form_lists.push(0);
        }
        if tiled_form_lists.is_empty() {
            tiled_form_lists.push(0);
        }
        if tiled_form_tiles.is_empty() {
            tiled_form_tiles.extend([0, 0, 0]);
        }

        let active_block_count = active_blocks_host.len();
        let mut active_chunk_bases = Vec::new();
        let mut active_chunk_matrices = Vec::new();
        let mut matrix_chunk_offsets = vec![0u32];
        for matrix in 0..matrix_count {
            let start = matrix_active_offsets_host[matrix] as usize;
            let end = matrix_active_offsets_host[matrix + 1] as usize;
            for base in (start..end).step_by(CHUNK_BLOCKS) {
                active_chunk_bases.push(
                    u32::try_from(base).map_err(|_| MetalError::Shape("one-joint opening chunk base exceeds u32"))?,
                );
                active_chunk_matrices.push(
                    u32::try_from(matrix).map_err(|_| MetalError::Shape("one-joint opening matrix exceeds u32"))?,
                );
            }
            matrix_chunk_offsets.push(
                u32::try_from(active_chunk_bases.len())
                    .map_err(|_| MetalError::Shape("one-joint opening chunk count exceeds u32"))?,
            );
        }
        let active_chunk_count = active_chunk_bases.len();
        let seeded = self.prepare_seeded_opening_plan(matrices, seeded_rows, &active_blocks_host, blocks)?;
        let geometric = self.prepare_geometric_opening_plans(matrices, device_matrices, &active_blocks_host, blocks)?;

        Ok(MetalJointOpeningPlan {
            active_local_offsets: self.buffer_from_slice(&active_local_offsets)?,
            active_entry_bases: self.buffer_from_slice(&active_entry_bases)?,
            active_blocks: self.buffer_from_slice(&active_blocks_host)?,
            active_chunk_bases: self.buffer_from_slice(&active_chunk_bases)?,
            active_chunk_matrices: self.buffer_from_slice(&active_chunk_matrices)?,
            matrix_active_offsets: self.buffer_from_slice(&matrix_active_offsets_host)?,
            matrix_chunk_offsets: self.buffer_from_slice(&matrix_chunk_offsets)?,
            matrix_identity: self.buffer_from_slice(&matrix_identity)?,
            entry_rows,
            entry_coefficients,
            parallel_form_lists: self.buffer_from_slice(&parallel_form_lists)?,
            tiled_form_lists: self.buffer_from_slice(&tiled_form_lists)?,
            tiled_form_tile_offsets: self.buffer_from_slice(&tiled_form_tile_offsets)?,
            tiled_form_tiles: self.buffer_from_slice(&tiled_form_tiles)?,
            tiled_form_partials: self.buffer(tiled_form_tile_count.max(1) * 2 * size_of::<u64>())?,
            seeded,
            geometric,
            active_block_count,
            active_chunk_count,
            parallel_form_list_count,
            tiled_form_list_count,
            tiled_form_tile_count,
            matrix_count,
            rows: scalar_rows,
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
            || !masks.matches_joint(witness_count, plan.blocks)
        {
            return Err(MetalError::Shape("one-joint opening dimensions are invalid"));
        }
        let chi = self.buffer(checked_product(
            &[chi_len, 2, size_of::<u64>()],
            "opening tensor size overflow",
        )?)?;
        let form_words = checked_product(&[plan.active_block_count, 2, D], "one-joint opening form size overflow")?;
        let forms = self.buffer(form_words * size_of::<u64>())?;
        let form_shape = self.buffer_from_slice(&[
            plan.matrix_count as u64,
            plan.blocks as u64,
            plan.rows as u64,
            chi_len as u64,
            carrier_width as u64,
        ])?;
        let form_rows = 2 * plan.matrix_count;
        let partial_words = checked_product(
            &[witness_count, plan.active_chunk_count, 2, PRODUCT_COEFFICIENTS],
            "one-joint opening partial size overflow",
        )?;
        let sum_words = checked_product(
            &[witness_count, form_rows, PRODUCT_COEFFICIENTS],
            "one-joint opening sum size overflow",
        )?;
        let output_words = checked_product(&[witness_count, form_rows, D], "one-joint opening output size overflow")?;
        let partials = self.buffer(partial_words * size_of::<u64>())?;
        let sums = self.buffer(sum_words * size_of::<u64>())?;
        let output = self.buffer(output_words * size_of::<u64>())?;
        let active_witnesses = (0..witness_count)
            .map(|witness| {
                u32::try_from(witness).map_err(|_| MetalError::Shape("one-joint opening witness exceeds u32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let active_witnesses = self.buffer_from_slice(&active_witnesses)?;
        let tail_shape = self.buffer_from_slice(&[
            plan.active_block_count as u64,
            witness_count as u64,
            form_rows as u64,
            plan.blocks as u64,
            plan.active_chunk_count as u64,
            0,
            masks.magnitudes() as u64,
        ])?;

        let command = self.command_buffer("nightstream.pi_ccs.joint.openings")?;
        let mut tensor_resources = Vec::new();
        self.encode_joint_tensor_point(&command, &chi, 0, point, &mut tensor_resources)?;

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.opening.forms")));
        encoder.setComputePipelineState(&self.dec_build_ring_forms);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.active_local_offsets), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.active_entry_bases), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_identity), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&plan.entry_rows), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&plan.entry_coefficients), 0, 4);
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
                encoder.setBuffer_offset_atIndex(Some(&plan.active_local_offsets), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.active_entry_bases), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.entry_rows), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&plan.entry_coefficients), 0, 4);
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
                encoder.setBuffer_offset_atIndex(Some(&plan.active_local_offsets), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.active_entry_bases), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.entry_rows), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&plan.entry_coefficients), 0, 4);
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
            plan.active_block_count * 2 * 14,
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
            encoder.setBuffer_offset_atIndex(Some(&plan.active_chunk_bases), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&plan.active_chunk_matrices), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_active_offsets), 0, 8);
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
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_chunk_offsets), 0, 3);
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
        drop(tensor_resources);
        drop(seeded_scratch);

        let words = self.read_buffer::<u64>(&output, output_words);
        let mut openings = vec![vec![vec![K::ZERO; D]; plan.matrix_count]; witness_count];
        for (witness, witness_openings) in openings.iter_mut().enumerate() {
            for (matrix, coefficients) in witness_openings.iter_mut().enumerate() {
                let real = (witness * form_rows + 2 * matrix) * D;
                let imaginary = real + D;
                for coefficient in 0..D {
                    coefficients[coefficient] = K::from_coeffs([
                        F::from_u64(words[real + coefficient]),
                        F::from_u64(words[imaginary + coefficient]),
                    ]);
                }
            }
        }
        Ok(openings
            .into_iter()
            .map(|families| {
                let mut families = families.into_iter();
                V1_1Evaluations {
                    eval_k: families.next().expect("the opening plan includes Pad"),
                    eval_a: families.collect(),
                }
            })
            .collect())
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
