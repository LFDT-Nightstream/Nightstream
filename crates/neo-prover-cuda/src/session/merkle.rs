//! The Poseidon2 Merkle tree of a row-major matrix in one stream of
//! launches, or its leaf digests alone: the layer-1 prover's commitments.

use super::{Buffer, CudaSession, Grid};
use crate::{poseidon2, CudaError, PoseidonDigest};

impl CudaSession {
    /// Every digest layer of the binary Merkle tree over `rows` rows of
    /// `width` words, leaves first and the root last. A leaf is Plonky3's
    /// overwrite sponge (`PaddingFreeSponge`, width 16, rate 12, four
    /// outputs) of its row; a node is the same sponge of its two children's
    /// eight words, which is Plonky3's `TruncatedPermutation` compression.
    /// `fill` writes the rows back to back into the upload; words may be
    /// non-canonical, digests are canonical.
    pub fn poseidon2_merkle_layers(
        &self,
        rows: usize,
        width: usize,
        fill: impl FnOnce(&mut [u64]),
    ) -> Result<Vec<Vec<PoseidonDigest>>, CudaError> {
        if width == 0 || !rows.is_power_of_two() {
            return Err(CudaError::Shape(
                "a Merkle tree needs a positive width and a power-of-two height",
            ));
        }
        let words = rows
            .checked_mul(width)
            .filter(|words| words.checked_mul(size_of::<u64>()).is_some())
            .ok_or(CudaError::Shape("a Merkle tree's rows overflow"))?;
        let input = self.filled_buffer(words, |words| {
            fill(words);
            Ok(())
        })?;
        let sizes: Vec<usize> = std::iter::successors(Some(rows), |&n| (n > 1).then_some(n / 2)).collect();
        let layers = sizes
            .iter()
            .map(|&n| self.buffer(n * poseidon2::DIGEST_LEN * size_of::<u64>()))
            .collect::<Result<Vec<_>, _>>()?;
        // Launches in order: each layer reads the one written before it.
        let mut source = (&input, width);
        for (layer, &n) in layers.iter().zip(&sizes) {
            self.hash_rows(source.0, layer, source.1, n)?;
            source = (layer, 2 * poseidon2::DIGEST_LEN);
        }
        self.finish()?;
        drop(input);
        layers
            .iter()
            .zip(&sizes)
            .map(|(layer, &n)| self.read_buffer::<PoseidonDigest>(layer, n))
            .collect()
    }

    /// The leaf digests of the rows of `width` words in `values`: the leaves
    /// of `poseidon2_merkle_layers`, and no layer above them. Words may be
    /// non-canonical, digests are canonical.
    pub fn poseidon2_merkle_leaves(&self, values: &[u64], width: usize) -> Result<Vec<PoseidonDigest>, CudaError> {
        if width == 0 || values.is_empty() || !values.len().is_multiple_of(width) {
            return Err(CudaError::Shape("Merkle leaves need whole rows of a positive width"));
        }
        let rows = values.len() / width;
        let input = self.buffer_from_slice(values)?;
        let output = self.buffer(rows * poseidon2::DIGEST_LEN * size_of::<u64>())?;
        self.hash_rows(&input, &output, width, rows)?;
        self.finish()?;
        self.read_buffer::<PoseidonDigest>(&output, rows)
    }

    fn hash_rows(&self, input: &Buffer, output: &Buffer, width: usize, rows: usize) -> Result<(), CudaError> {
        let kernels = &self.kernels.nightstream;
        let constants = self.poseidon2_constants.cu_deviceptr() as *const u64;
        self.launch(Grid::Threads(rows), |stream, config| {
            kernels.poseidon2_hash_rows_overwrite(
                stream,
                config,
                input.ptr(),
                output.ptr(),
                constants,
                width as u64,
                rows as u64,
            )
        })
    }
}
