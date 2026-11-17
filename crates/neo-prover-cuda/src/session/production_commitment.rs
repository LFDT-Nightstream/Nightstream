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
use p3_field::PrimeCharacteristicRing;

use super::{Buffer, CudaSession, Grid};
use crate::CudaError;

/// The Metal group width (`neo-prover-metal`), measured on an M5 Max.
const COLUMNS_PER_GROUP: usize = 512;
/// Witnesses per block of the accumulate kernel: 64 threads each, at most
/// 1,024 threads per block.
const MAX_WITNESSES_PER_GROUP: usize = 16;

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

impl CudaSession {
    /// SHAKE128 input bytes 0..69 (setup ID and seed) as nine little-endian
    /// lanes; the kernel adds each row and column.
    fn production_key_prefix(&self) -> Result<Buffer, CudaError> {
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

    /// Expand key row `row` at the `count` columns of `columns` into `slab`.
    fn launch_key_row(
        &self,
        prefix: &Buffer,
        columns: &Buffer,
        slab: &Buffer,
        count: usize,
        row: u64,
    ) -> Result<(), CudaError> {
        let kernels = &self.kernels.production_commitment;
        self.launch(Grid::Threads(count), |stream, config| {
            kernels.production_key_row(
                stream,
                config,
                prefix.ptr(),
                columns.ptr(),
                slab.ptr(),
                count as u64,
                row,
            )
        })
    }

    /// Expand every key row over columns `0..columns` unless the kept rows
    /// already cover them. The old rows are released first.
    fn cover_production_key(&self, key: &mut ResidentKeyRows, columns: usize) -> Result<(), CudaError> {
        if key.columns >= columns {
            return Ok(());
        }
        *key = ResidentKeyRows::default();
        let row_words = columns
            .checked_mul(D)
            .ok_or(CudaError::Shape("resident production key row overflows"))?;
        let rows = (0..PRODUCTION_VERIFIER_ROWS)
            .map(|_| self.buffer(row_words * size_of::<u64>()))
            .collect::<Result<Vec<_>, _>>()?;
        let prefix = self.production_key_prefix()?;
        let all_columns = self.buffer_from_slice(&(0..columns as u32).collect::<Vec<_>>())?;
        for (row, slab) in rows.iter().enumerate() {
            self.launch_key_row(&prefix, &all_columns, slab, columns, row as u64)?;
        }
        self.finish()?;
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
    ) -> Result<Vec<[u64; 3]>, CudaError> {
        if powers.is_empty() || powers.len() > PRODUCTION_VERIFIER_ROWS as usize {
            return Err(CudaError::Shape("key weights take one power per key row"));
        }
        let elements = columns
            .checked_mul(D)
            .ok_or(CudaError::Shape("key weights overflow"))?;
        let row = self.buffer(elements * size_of::<u64>())?;
        let out = self.buffer(elements * 3 * size_of::<u64>())?;
        let prefix = self.production_key_prefix()?;
        let all_columns = self.buffer_from_slice(&(0..columns as u32).collect::<Vec<_>>())?;
        let kernels = &self.kernels.production_commitment;
        // Launches run in order: each row's expansion waits for the previous
        // row's sum to finish reading the buffer.
        for (index, power) in powers.iter().enumerate() {
            self.launch_key_row(&prefix, &all_columns, &row, columns, index as u64)?;
            self.launch(Grid::Threads(elements), |stream, config| {
                kernels.production_key_weights(
                    stream,
                    config,
                    row.ptr(),
                    out.ptr(),
                    *power,
                    u64::from(index == 0),
                    elements as u64,
                )
            })?;
        }
        self.finish()?;
        self.read_buffer::<[u64; 3]>(&out, elements)
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
    ) -> Result<Vec<Commitment>, CudaError> {
        let blocks = witnesses
            .iter()
            .map(|witness| signed_unit_prefix_blocks(witness.borrow()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut output = vec![Commitment::zeros(PRODUCTION_VERIFIER_ROWS as usize); witnesses.len()];
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
            .ok_or(CudaError::Shape("production commitment mask dimensions overflow"))?;
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

        let witnesses_per_group = active.len().min(MAX_WITNESSES_PER_GROUP);
        let scratch_bytes = witnesses_per_group * (2 * D - 1) * size_of::<u64>();
        let groups = count.div_ceil(COLUMNS_PER_GROUP);
        let witness_blocks = active.len().div_ceil(witnesses_per_group);
        let slab_words = count
            .checked_mul(D)
            .ok_or(CudaError::Shape("production commitment key row overflows"))?;
        let partial_words = active
            .len()
            .checked_mul(groups)
            .and_then(|n| n.checked_mul(D))
            .ok_or(CudaError::Shape("production commitment partial dimensions overflow"))?;
        let rows = PRODUCTION_VERIFIER_ROWS as usize;
        let output_words = rows * active.len() * D;
        // Kept rows are indexed by column. A streamed row holds the occupied
        // columns in order, so its column table is the identity.
        let (keys, key_columns) = match &self.production_key {
            Some(kept) => {
                let mut kept = kept
                    .lock()
                    .map_err(|_| CudaError::Execution("production key lock was poisoned".into()))?;
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
        let kernels = &self.kernels.production_commitment;

        // All rows on one stream: the slab and the partials are reused row
        // after row, and the launches run in order.
        for row in 0..rows {
            let key = match &keys {
                KeyRows::Kept(kept) => &kept.rows[row],
                KeyRows::Streamed { prefix, slab } => {
                    self.launch_key_row(prefix, &device_columns, slab, count, row as u64)?;
                    slab
                }
            };
            let shape = [
                count as u64,
                active.len() as u64,
                groups as u64,
                COLUMNS_PER_GROUP as u64,
                row as u64,
                witnesses_per_group as u64,
            ];
            self.launch(
                Grid::Groups {
                    groups: groups * witness_blocks,
                    threads: 64 * witnesses_per_group,
                    shared: scratch_bytes,
                },
                |stream, config| {
                    kernels.production_ajtai_accumulate(
                        stream,
                        config,
                        key.ptr(),
                        device_masks.ptr(),
                        partials.ptr(),
                        key_columns.ptr(),
                        shape,
                    )
                },
            )?;
            self.launch(Grid::Threads(active.len() * D), |stream, config| {
                kernels.production_ajtai_sum_groups(
                    stream,
                    config,
                    partials.ptr(),
                    sums.at(row * active.len() * D * size_of::<u64>()),
                    groups as u64,
                    (active.len() * D) as u64,
                )
            })?;
        }
        self.finish()?;
        let words = self.read_buffer::<u64>(&sums, output_words)?;
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
