//! Fixed production-key commitments. Validate on the host, generate key rows
//! and sum signed ring products on the device. A streaming session expands the
//! rows it needs in every call; a prover session keeps the complete key rows
//! after their first use.

use std::{borrow::Borrow, sync::MutexGuard};

use neo_ajtai::{
    nightstream_fprime_setup::{
        element_input, signed_unit_prefix_blocks, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS, SETUP_ID,
    },
    Commitment,
};
use neo_ccs::Mat;
use neo_math::{D, F};
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputePipelineState, MTLDevice};
use p3_field::PrimeCharacteristicRing;

use super::{Arg, Buffer, Grid, MetalSession};
use crate::MetalError;

/// Measured on an M5 Max with 16 dense witnesses at production width: 128,
/// 512 and 2,048 columns took 5.01 s, 4.91 s and 5.12 s.
const COLUMNS_PER_GROUP: usize = 512;

/// The production key rows over columns `0..columns`, kept on the device by a
/// prover across its commitments. A wider witness regenerates them.
#[derive(Default)]
pub(super) struct ResidentKeyRows {
    columns: usize,
    rows: Vec<Buffer>,
}

/// Where one commitment call reads its key rows.
enum KeyRows<'a> {
    /// A prover's kept rows, read through the occupied column indices.
    Kept(MutexGuard<'a, ResidentKeyRows>),
    /// One row at a time over the occupied columns, read in position order.
    Streamed { prefix: Buffer, slab: Buffer },
}

impl MetalSession {
    /// SHAKE128 input bytes 0..69 (setup ID and seed) as nine little-endian
    /// lanes; the kernel adds each row and column.
    fn production_key_prefix(&self) -> Result<Buffer, MetalError> {
        let fixed = SETUP_ID.len() + PRODUCTION_SEED.len();
        let input = element_input(&PRODUCTION_SEED, 0, 0);
        let prefix: [u64; 9] = std::array::from_fn(|lane| {
            u64::from_le_bytes(std::array::from_fn(|byte| {
                let position = 8 * lane + byte;
                if position < fixed {
                    input[position]
                } else {
                    0
                }
            }))
        });
        self.buffer_from_slice(&prefix)
    }

    /// Expand every key row over columns `0..columns` unless the kept rows
    /// already cover them. The old rows are released first.
    fn cover_production_key(&self, key: &mut ResidentKeyRows, columns: usize) -> Result<(), MetalError> {
        if key.columns >= columns {
            return Ok(());
        }
        *key = ResidentKeyRows::default();
        let row_words = columns
            .checked_mul(D)
            .ok_or(MetalError::Shape("resident production key row overflows"))?;
        let rows = (0..PRODUCTION_VERIFIER_ROWS)
            .map(|_| self.buffer(row_words * size_of::<u64>()))
            .collect::<Result<Vec<_>, _>>()?;
        let prefix = self.production_key_prefix()?;
        let all_columns = self.buffer_from_slice(&(0..columns as u32).collect::<Vec<_>>())?;
        let shape_words: Vec<u64> = (0..PRODUCTION_VERIFIER_ROWS)
            .flat_map(|row| [columns as u64, 0, 0, 0, row, 0])
            .collect();
        let shapes = self.buffer_from_slice(&shape_words)?;
        let command = self.command_buffer("nightstream.production_key")?;
        for (row, slab) in rows.iter().enumerate() {
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("production_key_row")));
            self.launch(
                &encoder,
                &self.production_key_row,
                &[
                    Arg::Buffer(0, &prefix),
                    Arg::Buffer(1, &all_columns),
                    Arg::BufferAt(2, &shapes, row * 6 * size_of::<u64>()),
                    Arg::Buffer(3, slab),
                ],
                Grid::Threads(columns),
            );
            encoder.endEncoding();
        }
        self.finish(&command)?;
        *key = ResidentKeyRows { columns, rows };
        Ok(())
    }

    /// `Σ_i λ_i·a_{i,e}` over key rows `i < powers.len()` for every key
    /// coefficient `e < columns·D` (column-major, `D` per column), with each
    /// `λ_i` and result a cubic-extension value as three base words. Each row
    /// is expanded into one reused buffer and added in, so the weights hold
    /// one key row on the device, not the key.
    pub(crate) fn production_key_weights(
        &self,
        powers: &[[u64; 3]],
        columns: usize,
    ) -> Result<Vec<[u64; 3]>, MetalError> {
        if powers.is_empty() || powers.len() > PRODUCTION_VERIFIER_ROWS as usize {
            return Err(MetalError::Shape("key weights take one power per key row"));
        }
        let elements = columns
            .checked_mul(D)
            .ok_or(MetalError::Shape("key weights overflow"))?;
        let row = self.buffer(elements * size_of::<u64>())?;
        let out = self.buffer(elements * 3 * size_of::<u64>())?;
        let prefix = self.production_key_prefix()?;
        let all_columns = self.buffer_from_slice(&(0..columns as u32).collect::<Vec<_>>())?;
        let shape_words: Vec<u64> = (0..powers.len() as u64)
            .flat_map(|row| [columns as u64, 0, 0, 0, row, 0])
            .collect();
        let shapes = self.buffer_from_slice(&shape_words)?;
        let inputs = powers
            .iter()
            .enumerate()
            .map(|(row, power)| {
                Ok((
                    self.buffer_from_slice(power)?,
                    self.buffer_from_slice(&[elements as u64, u64::from(row == 0)])?,
                ))
            })
            .collect::<Result<Vec<_>, MetalError>>()?;
        // Encoders run in order: each row's expansion waits for the previous
        // row's sum to finish reading the buffer.
        let command = self.command_buffer("nightstream.production_key_weights")?;
        for (index, (power, shape)) in inputs.iter().enumerate() {
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            self.launch(
                &encoder,
                &self.production_key_row,
                &[
                    Arg::Buffer(0, &prefix),
                    Arg::Buffer(1, &all_columns),
                    Arg::BufferAt(2, &shapes, index * 6 * size_of::<u64>()),
                    Arg::Buffer(3, &row),
                ],
                Grid::Threads(columns),
            );
            encoder.endEncoding();
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            self.launch(
                &encoder,
                &self.production_key_weights,
                &[
                    Arg::Buffer(0, &row),
                    Arg::Buffer(1, &out),
                    Arg::Buffer(2, power),
                    Arg::Buffer(3, shape),
                ],
                Grid::Threads(elements),
            );
            encoder.endEncoding();
        }
        self.finish(&command)?;
        Ok(self.read_buffer::<[u64; 3]>(&out, elements))
    }

    /// Free the kept key rows. A kept key expands them again at its next
    /// commitment.
    pub(crate) fn release_production_key_rows(&self) {
        if let Some(key) = &self.production_key {
            *key.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = ResidentKeyRows::default();
        }
    }

    pub(crate) fn commit_production_prefixes<W: Borrow<Mat<F>>>(
        &self,
        witnesses: &[W],
    ) -> Result<Vec<Commitment>, MetalError> {
        let blocks = witnesses
            .iter()
            .map(|witness| signed_unit_prefix_blocks(witness.borrow()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut output = vec![Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize); witnesses.len()];
        let active = blocks
            .iter()
            .enumerate()
            .filter_map(|(index, blocks)| (!blocks.is_empty()).then_some(index))
            .collect::<Vec<_>>();
        if active.is_empty() {
            return Ok(output);
        }
        let width = witnesses
            .iter()
            .map(|witness| witness.borrow().cols())
            .max()
            .unwrap();
        let mut occupied = vec![false; width];
        for blocks in &blocks {
            for block in blocks {
                occupied[block.index() as usize] = true;
            }
        }
        let columns = occupied
            .into_iter()
            .enumerate()
            .filter_map(|(index, nonzero)| nonzero.then_some(index as u32))
            .collect::<Vec<_>>();
        let count = columns.len();
        let mut positions = vec![0usize; width];
        for (position, &column) in columns.iter().enumerate() {
            positions[column as usize] = position;
        }
        let mask_count = count
            .checked_mul(active.len())
            .ok_or(MetalError::Shape("production commitment mask dimensions overflow"))?;
        let mut masks = vec![[0u64; 2]; mask_count];
        for (local, &index) in active.iter().enumerate() {
            for block in &blocks[index] {
                masks[local * count + positions[block.index() as usize]] = [block.positive(), block.negative()];
            }
        }
        drop((blocks, positions));
        let device_masks = self.buffer_from_slice(&masks)?;
        let device_columns = self.buffer_from_slice(&columns)?;
        drop((masks, columns));

        let key_row = &self.production_key_row;
        let accumulate = &self.production_ajtai_accumulate;
        let witnesses_per_group = active
            .len()
            .min(accumulate.maxTotalThreadsPerThreadgroup() / 64);
        let scratch_bytes = (witnesses_per_group * (2 * D - 1) * size_of::<u64>()).next_multiple_of(16);
        if witnesses_per_group == 0
            || scratch_bytes + accumulate.staticThreadgroupMemoryLength() > self.device.maxThreadgroupMemoryLength()
        {
            return Err(MetalError::Shape("device cannot hold production commitment scratch"));
        }
        let groups = count.div_ceil(COLUMNS_PER_GROUP);
        let witness_blocks = active.len().div_ceil(witnesses_per_group);
        let slab_words = count
            .checked_mul(D)
            .ok_or(MetalError::Shape("production commitment key row overflows"))?;
        let partial_words = active
            .len()
            .checked_mul(groups)
            .and_then(|n| n.checked_mul(D))
            .ok_or(MetalError::Shape("production commitment partial dimensions overflow"))?;
        let rows = PRODUCTION_VERIFIER_ROWS as usize;
        let output_words = rows * active.len() * D;
        // Kept rows are indexed by column. A streamed row holds the occupied
        // columns in order, so its column table is the identity.
        let (keys, key_columns) = match &self.production_key {
            Some(kept) => {
                let mut kept = kept
                    .lock()
                    .map_err(|_| MetalError::Execution("production key lock was poisoned".into()))?;
                self.cover_production_key(&mut kept, width)?;
                (KeyRows::Kept(kept), device_columns.clone())
            }
            None => (
                KeyRows::Streamed {
                    prefix: self.production_key_prefix()?,
                    slab: self.buffer(slab_words * size_of::<u64>())?,
                },
                self.buffer_from_slice(&(0..count as u32).collect::<Vec<_>>())?,
            ),
        };
        let partials = self.buffer(partial_words * size_of::<u64>())?;
        let sums = self.buffer(output_words * size_of::<u64>())?;
        let shape_words: Vec<u64> = (0..PRODUCTION_VERIFIER_ROWS)
            .flat_map(|row| {
                [
                    count as u64,
                    active.len() as u64,
                    groups as u64,
                    COLUMNS_PER_GROUP as u64,
                    row,
                    witnesses_per_group as u64,
                ]
            })
            .collect();
        let shapes = self.buffer_from_slice(&shape_words)?;

        // All rows in one command buffer: the slab and the partials are reused
        // row after row, and buffer hazard tracking orders the encoders.
        let command = self.command_buffer("nightstream.production_commitment")?;
        for row in 0..rows {
            let shape_offset = row * 6 * size_of::<u64>();
            let key = match &keys {
                KeyRows::Kept(kept) => &kept.rows[row],
                KeyRows::Streamed { prefix, slab } => {
                    let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                    encoder.setLabel(Some(&NSString::from_str("production_key_row")));
                    self.launch(
                        &encoder,
                        key_row,
                        &[
                            Arg::Buffer(0, prefix),
                            Arg::Buffer(1, &device_columns),
                            Arg::BufferAt(2, &shapes, shape_offset),
                            Arg::Buffer(3, slab),
                        ],
                        Grid::Threads(count),
                    );
                    encoder.endEncoding();
                    slab
                }
            };

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("production_ajtai_accumulate")));
            self.launch(
                &encoder,
                accumulate,
                &[
                    Arg::Buffer(0, key),
                    Arg::Buffer(1, &device_masks),
                    Arg::BufferAt(2, &shapes, shape_offset),
                    Arg::Buffer(3, &partials),
                    Arg::Buffer(4, &key_columns),
                    Arg::Threadgroup(0, scratch_bytes),
                ],
                Grid::Groups {
                    groups: groups * witness_blocks,
                    threads: 64 * witnesses_per_group,
                },
            );
            encoder.endEncoding();

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("production_ajtai_sum_groups")));
            self.launch(
                &encoder,
                &self.production_ajtai_sum_groups,
                &[
                    Arg::Buffer(0, &partials),
                    Arg::BufferAt(1, &shapes, shape_offset),
                    Arg::BufferAt(2, &sums, row * active.len() * D * size_of::<u64>()),
                ],
                Grid::Threads(active.len() * D),
            );
            encoder.endEncoding();
        }
        self.finish(&command)?;
        let words = self.read_buffer::<u64>(&sums, output_words);
        for (row, row_words) in words.chunks_exact(active.len() * D).enumerate() {
            for (&index, coefficients) in active.iter().zip(row_words.chunks_exact(D)) {
                for (target, &value) in output[index].col_mut(row).iter_mut().zip(coefficients) {
                    *target = F::from_u64(value);
                }
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/production_commitment.rs"]
mod tests;
