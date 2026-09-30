//! Device Π_RLC witness mix and its base-2 digit split.
//!
//! Inputs are signed-unit column masks and rotation matrices with small
//! entries. The parent `Σ ρ_i·Z_i` stays on the device; only its digit
//! planes return. The result equals the host
//! `split_b_matrix_k_with_nonzero_flags(rlc_mix_witnesses(..), k, 2)`,
//! except that an out-of-range parent is an error without the entry index.

use std::{mem::size_of, sync::atomic::Ordering};

use neo_ccs::Mat;
use neo_math::{D, F};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::MetalSession;
use crate::MetalError;

/// Children that one split thread writes; it matches the split kernel.
const CHILDREN_PER_THREAD: usize = 14;

impl MetalSession {
    /// Mix the Π_RLC witnesses and split the parent into `digits` balanced
    /// base-2 planes. Returns the planes and whether each one is nonzero.
    pub(crate) fn split_rlc_witnesses(
        &self,
        rhos: &[Mat<F>],
        witnesses: &[&Mat<F>],
        digits: usize,
        base: u32,
    ) -> Result<(Vec<Mat<F>>, Vec<bool>), MetalError> {
        let columns = witnesses
            .first()
            .ok_or(MetalError::Shape("device PiRLC needs witnesses"))?
            .cols();
        if base != 2 || !(1..63).contains(&digits) || rhos.len() != witnesses.len() || columns == 0 {
            return Err(MetalError::Shape("device PiRLC split shape"));
        }
        // Zero witnesses add nothing; every other witness must be signed-unit.
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
            return Ok(((0..digits).map(|_| zero()).collect(), vec![false; digits]));
        }

        let mask_words = sources.len() * 2 * columns;
        let masks = self.buffer(mask_words * size_of::<u64>())?;
        // This new shared buffer has no device readers yet.
        let destination =
            unsafe { std::slice::from_raw_parts_mut(masks.contents().as_ptr().cast::<u64>(), mask_words) };
        for ((positive, negative), target) in sources
            .iter()
            .zip(destination.chunks_exact_mut(2 * columns))
        {
            target[..columns].copy_from_slice(positive);
            target[columns..].copy_from_slice(negative);
        }
        self.activity
            .uploaded_bytes
            .fetch_add((mask_words * size_of::<u64>()) as u64, Ordering::Relaxed);
        let rhos = self.buffer_from_slice(&rho_entries)?;
        let parent = self.buffer(D * columns * size_of::<u64>())?;
        let planes = self.buffer(digits * 2 * columns * size_of::<u64>())?;
        let nonzero = self.buffer_from_slice(&vec![0u32; digits])?;
        let status = self.buffer_from_slice(&[0u32])?;
        let mix_shape = self.buffer_from_slice(&[sources.len() as u64, columns as u64])?;
        let split_shape = self.buffer_from_slice(&[0, digits as u64, 0, columns as u64])?;

        let command = self.command_buffer("nightstream.pi_rlc.split")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.rlc_witness_mix_signed_masks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&rhos), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&masks), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&mix_shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&parent), 0, 3);
        }
        self.dispatch(&encoder, &self.rlc_witness_mix_signed_masks, D * columns);
        encoder.endEncoding();
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.dec_split_base2_masks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&parent), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&split_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&planes), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&nonzero), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&status), 0, 4);
        }
        self.dispatch(
            &encoder,
            &self.dec_split_base2_masks,
            digits.div_ceil(CHILDREN_PER_THREAD) * columns,
        );
        encoder.endEncoding();
        self.finish(&command)?;
        drop(encoder);
        drop(command);
        drop((masks, parent));

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
        let words =
            unsafe { std::slice::from_raw_parts(planes.contents().as_ptr().cast::<u64>(), digits * 2 * columns) };
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
        Ok((children, flags))
    }
}

/// The signed byte of a small field element.
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
