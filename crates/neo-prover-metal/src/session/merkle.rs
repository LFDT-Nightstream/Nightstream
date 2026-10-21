//! The Poseidon2 Merkle tree of a row-major matrix in one command buffer,
//! or its leaf digests alone: the layer-1 prover's commitments.

use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::MetalSession;
use crate::{poseidon2, MetalError, PoseidonDigest};

impl MetalSession {
    /// Every digest layer of the binary Merkle tree over `rows` rows of
    /// `width` words, leaves first and the root last. A leaf is Plonky3's
    /// overwrite sponge (`PaddingFreeSponge`, width 16, rate 12, four
    /// outputs) of its row; a node is the same sponge of its two children's
    /// eight words, which is Plonky3's `TruncatedPermutation` compression.
    /// `fill` writes the rows back to back into the device buffer; words
    /// may be non-canonical, digests are canonical. One command buffer
    /// builds all layers.
    pub fn poseidon2_merkle_layers(
        &self,
        rows: usize,
        width: usize,
        fill: impl FnOnce(&mut [u64]),
    ) -> Result<Vec<Vec<PoseidonDigest>>, MetalError> {
        if width == 0 || !rows.is_power_of_two() {
            return Err(MetalError::Shape(
                "a Merkle tree needs a positive width and a power-of-two height",
            ));
        }
        let input = self.buffer(rows * width * size_of::<u64>())?;
        fill(unsafe { std::slice::from_raw_parts_mut(input.contents().as_ptr().cast::<u64>(), rows * width) });
        let sizes: Vec<usize> = std::iter::successors(Some(rows), |&n| (n > 1).then_some(n / 2)).collect();
        let layers = sizes
            .iter()
            .map(|&n| self.buffer(n * poseidon2::DIGEST_LEN * size_of::<u64>()))
            .collect::<Result<Vec<_>, _>>()?;
        let leaf_width = self.buffer_from_slice(&[width as u64])?;
        let node_width = self.buffer_from_slice(&[2 * poseidon2::DIGEST_LEN as u64])?;
        let command = self.command_buffer("nightstream.poseidon2.merkle_layers")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.poseidon2_hash_rows_overwrite);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.poseidon2_constants), 0, 2);
        }
        // Serial dispatches: each layer reads the one written before it.
        let mut source = (&input, &leaf_width);
        for (layer, &n) in layers.iter().zip(&sizes) {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(source.0), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(layer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(source.1), 0, 3);
            }
            self.dispatch(&encoder, &self.poseidon2_hash_rows_overwrite, n);
            source = (layer, &node_width);
        }
        encoder.endEncoding();
        self.finish(&command)?;
        drop(input);
        Ok(layers
            .iter()
            .zip(&sizes)
            .map(|(layer, &n)| self.read_buffer_parallel::<PoseidonDigest>(layer, n))
            .collect())
    }

    /// The leaf digests of the rows of `width` words in `values`: the leaves
    /// of `poseidon2_merkle_layers`, and no layer above them. The rows are
    /// read in place when `values` spans whole pages, otherwise copied once;
    /// words may be non-canonical, digests are canonical.
    pub fn poseidon2_merkle_leaves(&self, values: &[u64], width: usize) -> Result<Vec<PoseidonDigest>, MetalError> {
        if width == 0 || values.is_empty() || !values.len().is_multiple_of(width) {
            return Err(MetalError::Shape("Merkle leaves need whole rows of a positive width"));
        }
        let rows = values.len() / width;
        let wrapped = self.wrap_read_pages(values);
        let copied = match wrapped {
            Some(_) => None,
            None => Some(self.buffer_from_slice(values)?),
        };
        let input = wrapped
            .as_ref()
            .or(copied.as_ref())
            .expect("one of the two buffers");
        let output = self.buffer(rows * poseidon2::DIGEST_LEN * size_of::<u64>())?;
        let leaf_width = self.buffer_from_slice(&[width as u64])?;
        let command = self.command_buffer("nightstream.poseidon2.merkle_leaves")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.poseidon2_hash_rows_overwrite);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.poseidon2_constants), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&leaf_width), 0, 3);
        }
        self.dispatch(&encoder, &self.poseidon2_hash_rows_overwrite, rows);
        encoder.endEncoding();
        self.finish(&command)?;
        drop(wrapped);
        Ok(self.read_buffer_parallel::<PoseidonDigest>(&output, rows))
    }
}
