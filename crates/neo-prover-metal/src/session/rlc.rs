//! Device Π_RLC witness mix `Σ ρ_i·Z_i`, with or without its base-2 digit
//! split.
//!
//! Inputs are signed-unit column masks and rotation matrices with small
//! entries. `split_rlc_witnesses` keeps the parent on the device and returns
//! only its digit planes; the result equals the host
//! `split_b_matrix_k_with_nonzero_flags(rlc_mix_witnesses(..), k, 2)`, except
//! that an out-of-range parent is an error without the entry index.
//! `mix_rlc_witnesses` returns the parent itself, equal to the host
//! `rlc_mix_witnesses`; it also takes a source outside {-1, 0, 1} with weight
//! 1 (option B's light accumulator, Lean `lightChallenge`), which the host
//! adds.

use std::{mem::size_of, sync::atomic::Ordering};

use neo_ccs::Mat;
use neo_math::{D, F};
use neo_reductions::common::SplitDigits;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{host_view, Arg, Buffer, Grid, MetalSession};
use crate::MetalError;

type Command = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>>;

const CHILDREN_PER_THREAD: usize = 14;

impl MetalSession {
    /// `Σ ρ_i·Z_i` as a dense matrix.
    pub(crate) fn mix_rlc_witnesses(&self, rhos: &[Mat<F>], witnesses: &[&Mat<F>]) -> Result<Mat<F>, MetalError> {
        let columns = witnesses
            .first()
            .ok_or(MetalError::Shape("device PiRLC needs witnesses"))?
            .cols();
        if rhos.len() != witnesses.len() || columns == 0 {
            return Err(MetalError::Shape("device PiRLC mix shape"));
        }
        let mut rho_entries = Vec::new();
        let mut sources = Vec::new();
        let mut added = Vec::new();
        for (rho, witness) in rhos.iter().zip(witnesses) {
            if rho.rows() != D || rho.cols() != D || witness.rows() != D || witness.cols() != columns {
                return Err(MetalError::Shape("device PiRLC input shape"));
            }
            if witness
                .virtual_constant_value()
                .is_some_and(|value| *value == F::ZERO)
            {
                continue;
            }
            let masks = match witness.packed_signed_unit_column_masks() {
                Some((positive, negative)) => Some((positive.to_vec(), negative.to_vec())),
                None => signed_unit_masks(witness),
            };
            let Some(masks) = masks else {
                if !rho.is_identity() {
                    return Err(MetalError::Shape(
                        "device PiRLC adds a dense witness only with weight 1",
                    ));
                }
                added.push(*witness);
                continue;
            };
            sources.push(masks);
            for &entry in rho.as_slice() {
                rho_entries.push(small_signed(entry).ok_or(MetalError::Shape("device PiRLC rho entry exceeds i8"))?);
            }
        }
        let mut parent = if sources.is_empty() {
            vec![F::ZERO; D * columns]
        } else {
            let masks: Vec<_> = sources
                .iter()
                .map(|(positive, negative)| (&positive[..], &negative[..]))
                .collect();
            let (command, parent, resources) = self.encode_rlc_mix(&rho_entries, &masks, columns)?;
            self.finish(&command)?;
            drop(resources);
            self.read_buffer::<u64>(&parent, D * columns)
                .into_iter()
                .map(F::from_u64)
                .collect()
        };
        for witness in added {
            for (row, values) in parent.chunks_exact_mut(columns).enumerate() {
                for (column, value) in values.iter_mut().enumerate() {
                    *value += witness[(row, column)];
                }
            }
        }
        Ok(Mat::from_row_major(D, columns, parent))
    }

    /// Encode the mix of `masks` (planar positive and negative column masks
    /// of each source) under `rho_entries` into a new command buffer; the
    /// caller finishes it. Returns the command, the parent buffer and the
    /// inputs that must live until the command finishes.
    fn encode_rlc_mix(
        &self,
        rho_entries: &[i8],
        masks: &[(&[u64], &[u64])],
        columns: usize,
    ) -> Result<(Command, Buffer, [Buffer; 3]), MetalError> {
        let masks = self.filled_buffer(masks.len() * 2 * columns, |destination| {
            for ((positive, negative), target) in masks.iter().zip(destination.chunks_exact_mut(2 * columns)) {
                target[..columns].copy_from_slice(positive);
                target[columns..].copy_from_slice(negative);
            }
            Ok(())
        })?;
        let rhos = self.buffer_from_slice(rho_entries)?;
        let parent = self.buffer(D * columns * size_of::<u64>())?;
        let shape = self.buffer_from_slice(&[(rho_entries.len() / (D * D)) as u64, columns as u64])?;
        let command = self.command_buffer("nightstream.pi_rlc.mix")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        self.launch(
            &encoder,
            &self.rlc_witness_mix_signed_masks,
            &[
                Arg::Buffer(0, &rhos),
                Arg::Buffer(1, &masks),
                Arg::Buffer(2, &shape),
                Arg::Buffer(3, &parent),
            ],
            Grid::Threads(D * columns),
        );
        encoder.endEncoding();
        Ok((command, parent, [masks, rhos, shape]))
    }

    pub(crate) fn split_rlc_witnesses(
        &self,
        rhos: &[Mat<F>],
        witnesses: &[&Mat<F>],
        digits: usize,
        base: u32,
    ) -> Result<SplitDigits, MetalError> {
        let columns = witnesses
            .first()
            .ok_or(MetalError::Shape("device PiRLC needs witnesses"))?
            .cols();
        if base != 2 || !(1..63).contains(&digits) || rhos.len() != witnesses.len() || columns == 0 {
            return Err(MetalError::Shape("device PiRLC split shape"));
        }
        let mut rho_entries = Vec::new();
        let mut sources = Vec::new();
        for (rho, witness) in rhos.iter().zip(witnesses) {
            if rho.rows() != D || rho.cols() != D || witness.rows() != D || witness.cols() != columns {
                return Err(MetalError::Shape("device PiRLC input shape"));
            }
            if witness
                .virtual_constant_value()
                .is_some_and(|value| *value == F::ZERO)
            {
                continue;
            }
            sources.push(
                witness
                    .packed_signed_unit_column_masks()
                    .ok_or(MetalError::Shape("device PiRLC needs signed-unit column masks"))?,
            );
            for &entry in rho.as_slice() {
                rho_entries.push(small_signed(entry).ok_or(MetalError::Shape("device PiRLC rho entry exceeds i8"))?);
            }
        }
        let zero = || Mat::virtual_constant(D, columns, F::ZERO);
        if sources.is_empty() {
            return Ok(SplitDigits::from_device_split(
                (0..digits).map(|_| zero()).collect(),
                vec![false; digits],
            ));
        }

        let planes = self.buffer(digits * 2 * columns * size_of::<u64>())?;
        let nonzero = self.buffer_from_slice(&vec![0u32; digits])?;
        let status = self.buffer_from_slice(&[0u32])?;
        let split_shape = self.buffer_from_slice(&[0, digits as u64, 0, columns as u64])?;
        let (command, parent, resources) = self.encode_rlc_mix(&rho_entries, &sources, columns)?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        self.launch(
            &encoder,
            &self.dec_split_base2_masks,
            &[
                Arg::Buffer(0, &parent),
                Arg::Buffer(1, &split_shape),
                Arg::Buffer(2, &planes),
                Arg::Buffer(3, &nonzero),
                Arg::Buffer(4, &status),
            ],
            Grid::Threads(digits.div_ceil(CHILDREN_PER_THREAD) * columns),
        );
        encoder.endEncoding();
        self.finish(&command)?;
        drop(encoder);
        drop(command);
        drop((resources, parent));

        if self.read_buffer::<u32>(&status, 1)[0] != 0 {
            return Err(MetalError::Shape("PiRLC parent coefficient exceeds the digit range"));
        }
        let flags: Vec<bool> = self
            .read_buffer::<u32>(&nonzero, digits)
            .into_iter()
            .map(|word| word != 0)
            .collect();
        self.activity
            .downloaded_bytes
            .fetch_add((digits * 2 * columns * size_of::<u64>()) as u64, Ordering::Relaxed);
        let words = host_view::<u64>(&planes, digits * 2 * columns);
        let children = words
            .chunks_exact(2 * columns)
            .zip(&flags)
            .map(|(plane, &used)| {
                if used {
                    Mat::compact_signed_unit_from_column_masks(D, columns, &plane[..columns], &plane[columns..])
                        .map_err(MetalError::Shape)
                } else {
                    Ok(zero())
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(SplitDigits::from_device_split(children, flags))
    }
}

/// The planar positive and negative lane masks of each column of a witness
/// with entries in {-1, 0, 1}, or `None` for any other entry.
fn signed_unit_masks(witness: &Mat<F>) -> Option<(Vec<u64>, Vec<u64>)> {
    let mut positive = vec![0u64; witness.cols()];
    let mut negative = vec![0u64; witness.cols()];
    for row in 0..witness.rows() {
        for column in 0..witness.cols() {
            let value = witness[(row, column)];
            if value == F::ONE {
                positive[column] |= 1 << row;
            } else if value == -F::ONE {
                negative[column] |= 1 << row;
            } else if value != F::ZERO {
                return None;
            }
        }
    }
    Some((positive, negative))
}

fn small_signed(value: F) -> Option<i8> {
    let canonical = value.as_canonical_u64();
    if canonical <= i8::MAX as u64 {
        Some(canonical as i8)
    } else {
        let negative = F::ORDER_U64 - canonical;
        (negative <= 128).then(|| (-(negative as i64)) as i8)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/rlc_split.rs"]
mod tests;
