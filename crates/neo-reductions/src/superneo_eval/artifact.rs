//! Bounded binary storage for the verifier-owned compact matrix cache.
//!
//! The artifact stores only evaluator data. It does not replace the CCS
//! relation, the online witness encoder, or the verifier key. A verifier must
//! keep the returned receipt outside the artifact and supply it during load.

use std::io::{Read, Write};

use neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN;
use neo_ccs::SeededPhi81LinearBlock;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use super::{
    CompactRowBlock, DenseBlockStore, DenseRowBlock, RowOffsetStore, SuperneoEvalCache, SuperneoMatrixCache,
    COMPACT_SINGLE_BLOCK_MASK,
};

const MAGIC: [u8; 8] = *b"NSCEV001";
const SCHEMA_VERSION: u32 = 2;
const HEADER_BYTES: u64 = 8 + 4 + 8 + 4 + 4 * 8 + 4 * 8;
const DIGEST_CHUNK_ITEMS: usize = 16_384;

const OFFSET_EMPTY: u8 = 0;
const OFFSET_U16_CHUNKED: u8 = 1;
const OFFSET_U24: u8 = 2;
const OFFSET_U32: u8 = 3;

const SECTION_ROW_CHUNK_BASES: u64 = 1;
const SECTION_ROW_LOCALS: u64 = 2;
const SECTION_ROW_U24: u64 = 3;
const SECTION_ROW_U32: u64 = 4;
const SECTION_ROW_BLOCKS: u64 = 5;
const SECTION_DENSE_OFFSETS: u64 = 6;
const SECTION_DENSE_LOCALS: u64 = 7;
const SECTION_DENSE_COEFFICIENTS: u64 = 8;
const SECTION_GEOMETRIC_CHUNK_BASES: u64 = 9;
const SECTION_GEOMETRIC_LOCALS: u64 = 10;
const SECTION_GEOMETRIC_U24: u64 = 11;
const SECTION_GEOMETRIC_U32: u64 = 12;
const SECTION_GEOMETRIC_RUNS: u64 = 13;
const SECTION_EXPLICIT_MASKS: u64 = 14;
const SECTION_DENSE_ROW_BLOCKS: u64 = 15;

mod digest;

use digest::cache_digest;

/// Verifier-owned values that bind one compact cache artifact.
///
/// Store this receipt in trusted profile metadata. Do not read it from the
/// same untrusted file that contains the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuperneoCacheArtifactReceipt {
    artifact_bytes: u64,
    matrix_count: u32,
    matrix_digest: [u64; DIGEST_LEN],
    cache_digest: [u64; DIGEST_LEN],
}

impl SuperneoCacheArtifactReceipt {
    /// Rebuild a receipt from trusted profile metadata.
    pub fn from_parts(
        artifact_bytes: u64,
        matrix_count: usize,
        matrix_digest: [u64; DIGEST_LEN],
        cache_digest: [u64; DIGEST_LEN],
    ) -> Result<Self, SuperneoCacheArtifactError> {
        if artifact_bytes < HEADER_BYTES {
            return Err(invalid("receipt artifact size is smaller than the header"));
        }
        let matrix_count = u32::try_from(matrix_count).map_err(|_| invalid("receipt matrix count exceeds u32"))?;
        validate_digest_words(&matrix_digest, "receipt matrix digest")?;
        validate_digest_words(&cache_digest, "receipt cache digest")?;
        Ok(Self {
            artifact_bytes,
            matrix_count,
            matrix_digest,
            cache_digest,
        })
    }

    /// Exact encoded artifact size.
    pub fn artifact_bytes(&self) -> u64 {
        self.artifact_bytes
    }

    /// Number of cached matrices.
    pub fn matrix_count(&self) -> usize {
        self.matrix_count as usize
    }

    /// Verifier-bound CCS matrix digest.
    pub fn matrix_digest(&self) -> [u64; DIGEST_LEN] {
        self.matrix_digest
    }

    /// Poseidon2 digest of the compact cache representation.
    pub fn cache_digest(&self) -> [u64; DIGEST_LEN] {
        self.cache_digest
    }
}

/// Hard load bounds selected by the deployment profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuperneoCacheArtifactLimits {
    max_bytes: u64,
    max_rows: usize,
    max_cols: usize,
    max_matrices: usize,
}

/// A compact cache that passed bounded decoding and its verifier-owned
/// receipt check. Keeping the receipt with the cache prevents later setup
/// code from pairing checked bytes with a different matrix identity.
pub struct VerifiedSuperneoCacheArtifact {
    cache: SuperneoEvalCache,
    receipt: SuperneoCacheArtifactReceipt,
}

impl VerifiedSuperneoCacheArtifact {
    /// The checked compact evaluator cache.
    pub fn cache(&self) -> &SuperneoEvalCache {
        &self.cache
    }

    /// The verifier-owned receipt used during bounded decoding.
    pub fn receipt(&self) -> &SuperneoCacheArtifactReceipt {
        &self.receipt
    }

    /// Discard the verified pairing and keep only the evaluator cache.
    pub fn into_cache(self) -> SuperneoEvalCache {
        self.cache
    }

    pub(crate) fn into_parts(self) -> (SuperneoEvalCache, SuperneoCacheArtifactReceipt) {
        (self.cache, self.receipt)
    }
}

impl SuperneoCacheArtifactLimits {
    /// Set the maximum file size and matrix shape accepted by the loader.
    pub fn new(max_bytes: u64, max_rows: usize, max_cols: usize, max_matrices: usize) -> Self {
        Self {
            max_bytes,
            max_rows,
            max_cols,
            max_matrices,
        }
    }
}

/// Failure while encoding or loading a compact cache artifact.
#[derive(Debug, Error)]
pub enum SuperneoCacheArtifactError {
    /// The backing reader or writer failed.
    #[error("compact cache artifact I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// The artifact is malformed or does not match its receipt.
    #[error("invalid compact cache artifact: {0}")]
    Invalid(String),
    /// A deployment bound was exceeded before allocation.
    #[error("compact cache artifact {item} {actual} exceeds limit {maximum}")]
    Limit {
        /// Bounded item.
        item: &'static str,
        /// Encoded value.
        actual: u64,
        /// Accepted maximum.
        maximum: u64,
    },
}

fn invalid(message: impl Into<String>) -> SuperneoCacheArtifactError {
    SuperneoCacheArtifactError::Invalid(message.into())
}

fn limit(item: &'static str, actual: u64, maximum: u64) -> SuperneoCacheArtifactError {
    SuperneoCacheArtifactError::Limit { item, actual, maximum }
}

impl SuperneoEvalCache {
    /// Encode this cache and return the verifier-owned receipt for the bytes.
    pub fn write_artifact<W: Write>(
        &self,
        writer: W,
        matrix_digest: [F; DIGEST_LEN],
    ) -> Result<SuperneoCacheArtifactReceipt, SuperneoCacheArtifactError> {
        validate_cache(self)?;
        let artifact_bytes = encoded_size(self)?;
        let cache_digest = cache_digest(self, &matrix_digest, artifact_bytes);
        let receipt = SuperneoCacheArtifactReceipt {
            artifact_bytes,
            matrix_count: u32::try_from(self.mats.len()).map_err(|_| invalid("matrix count exceeds u32"))?,
            matrix_digest: matrix_digest.map(|value| value.as_canonical_u64()),
            cache_digest: cache_digest.map(|value| value.as_canonical_u64()),
        };

        let mut encoder = Encoder::new(writer);
        encoder.bytes(&MAGIC)?;
        encoder.u32(SCHEMA_VERSION)?;
        encoder.u64(receipt.artifact_bytes)?;
        encoder.u32(receipt.matrix_count)?;
        encoder.digest_words(&receipt.matrix_digest)?;
        encoder.digest_words(&receipt.cache_digest)?;
        for matrix in &self.mats {
            encode_matrix(&mut encoder, matrix)?;
        }
        encode_masks(&mut encoder, self.explicit_matrix_masks.as_deref())?;
        if encoder.written != artifact_bytes {
            return Err(invalid(format!(
                "encoder wrote {} bytes, expected {artifact_bytes}",
                encoder.written
            )));
        }
        Ok(receipt)
    }

    /// Load a cache under exact verifier-owned receipt values and hard bounds.
    pub fn read_artifact<R: Read>(
        reader: R,
        receipt: &SuperneoCacheArtifactReceipt,
        limits: SuperneoCacheArtifactLimits,
    ) -> Result<Self, SuperneoCacheArtifactError> {
        Self::read_verified_artifact(reader, receipt, limits).map(VerifiedSuperneoCacheArtifact::into_cache)
    }

    /// Load and retain proof that bounded decoding used this exact receipt.
    pub fn read_verified_artifact<R: Read>(
        reader: R,
        receipt: &SuperneoCacheArtifactReceipt,
        limits: SuperneoCacheArtifactLimits,
    ) -> Result<VerifiedSuperneoCacheArtifact, SuperneoCacheArtifactError> {
        if receipt.artifact_bytes > limits.max_bytes {
            return Err(limit("byte size", receipt.artifact_bytes, limits.max_bytes));
        }
        if receipt.matrix_count as usize > limits.max_matrices {
            return Err(limit(
                "matrix count",
                u64::from(receipt.matrix_count),
                usize_to_u64(limits.max_matrices, "matrix limit")?,
            ));
        }

        let mut decoder = Decoder::new(reader, limits.max_bytes);
        let magic = decoder.array::<8>()?;
        if magic != MAGIC {
            return Err(invalid("magic does not match"));
        }
        if decoder.u32()? != SCHEMA_VERSION {
            return Err(invalid("schema version does not match"));
        }
        let artifact_bytes = decoder.u64()?;
        if artifact_bytes != receipt.artifact_bytes {
            return Err(invalid("artifact size does not match the verifier receipt"));
        }
        decoder.set_exact_limit(artifact_bytes)?;
        let matrix_count = decoder.u32()?;
        if matrix_count != receipt.matrix_count {
            return Err(invalid("matrix count does not match the verifier receipt"));
        }
        let matrix_digest_words = decoder.digest_words()?;
        validate_digest_words(&matrix_digest_words, "matrix digest")?;
        if matrix_digest_words != receipt.matrix_digest {
            return Err(invalid("matrix digest does not match the verifier receipt"));
        }
        let encoded_cache_digest = decoder.digest_words()?;
        validate_digest_words(&encoded_cache_digest, "cache digest")?;
        if encoded_cache_digest != receipt.cache_digest {
            return Err(invalid("cache digest does not match the verifier receipt"));
        }

        let mut mats = Vec::new();
        mats.try_reserve_exact(matrix_count as usize)
            .map_err(|_| invalid("cannot reserve matrix cache storage"))?;
        for _ in 0..matrix_count {
            mats.push(decode_matrix(&mut decoder, limits)?);
        }
        let explicit_matrix_masks = decode_masks(&mut decoder)?;
        decoder.finish_exact()?;

        let cache = Self {
            mats,
            explicit_matrix_masks,
        };
        validate_cache(&cache)?;
        if encoded_size(&cache)? != artifact_bytes {
            return Err(invalid("decoded cache does not have the encoded byte size"));
        }
        let matrix_digest = matrix_digest_words.map(F::from_u64);
        let computed = cache_digest(&cache, &matrix_digest, artifact_bytes).map(|value| value.as_canonical_u64());
        if computed != receipt.cache_digest {
            return Err(invalid("cache content does not match the verifier receipt"));
        }
        Ok(VerifiedSuperneoCacheArtifact {
            cache,
            receipt: *receipt,
        })
    }
}

fn validate_digest_words(words: &[u64; DIGEST_LEN], label: &str) -> Result<(), SuperneoCacheArtifactError> {
    if words.iter().any(|&word| word >= F::ORDER_U64) {
        return Err(invalid(format!("{label} has a non-canonical Goldilocks word")));
    }
    Ok(())
}

fn validate_cache(cache: &SuperneoEvalCache) -> Result<(), SuperneoCacheArtifactError> {
    let rows = cache.mats.first().map_or(0, |matrix| matrix.rows);
    for (index, matrix) in cache.mats.iter().enumerate() {
        if matrix.rows != rows {
            return Err(invalid(format!("matrix {index} has a different row count")));
        }
        validate_matrix(index, matrix)?;
    }
    validate_masks(cache, rows)
}

fn validate_matrix(index: usize, matrix: &SuperneoMatrixCache) -> Result<(), SuperneoCacheArtifactError> {
    if matrix.cols == 0 || matrix.cols % D != 0 {
        return Err(invalid(format!(
            "matrix {index} column count is not a nonzero multiple of D"
        )));
    }
    if matrix.cols / D > COMPACT_SINGLE_BLOCK_MASK as usize + 1 {
        return Err(invalid(format!(
            "matrix {index} column-block count exceeds the packed row-reference limit"
        )));
    }
    let DenseBlockStore::Compact {
        offsets: dense_offsets,
        locals: dense_locals,
        coefficients: dense_coefficients,
    } = &matrix.dense_orig
    else {
        return Err(invalid(format!("matrix {index} dense dictionary is unfinished")));
    };
    validate_dense(index, dense_offsets, dense_locals, dense_coefficients)?;
    let dense_count = dense_offsets.len() - 1;

    if matrix.identity {
        if matrix.rows > matrix.cols
            || !matches!(matrix.row_offsets, RowOffsetStore::Empty)
            || !matrix.row_blocks.is_empty()
            || !matrix.dense_row_blocks.is_empty()
            || dense_count != 0
            || !matches!(matrix.geometric_row_offsets, RowOffsetStore::Empty)
            || !matrix.geometric_runs.is_empty()
            || !matrix.seeded_phi81_blocks.is_empty()
        {
            return Err(invalid(format!("matrix {index} has invalid identity cache data")));
        }
        return Ok(());
    }

    let offset_len = matrix
        .rows
        .checked_add(1)
        .ok_or_else(|| invalid(format!("matrix {index} row count overflows")))?;
    validate_offsets(
        index,
        "row",
        &matrix.row_offsets,
        offset_len,
        matrix.row_blocks.len(),
        false,
    )?;
    validate_row_blocks(index, matrix, dense_count)?;
    validate_offsets(
        index,
        "geometric row",
        &matrix.geometric_row_offsets,
        offset_len,
        matrix.geometric_runs.len(),
        matrix.geometric_runs.is_empty(),
    )?;
    validate_geometric_runs(index, matrix)?;
    for (block_index, block) in matrix.seeded_phi81_blocks.iter().enumerate() {
        block
            .validate_matrix_shape(matrix.rows, matrix.cols)
            .map_err(|error| {
                invalid(format!(
                    "matrix {index} seeded block {block_index} has invalid shape: {error}"
                ))
            })?;
    }
    Ok(())
}

fn validate_dense(
    matrix: usize,
    offsets: &[u32],
    locals: &[u8],
    coefficients: &[F],
) -> Result<(), SuperneoCacheArtifactError> {
    if offsets.first().copied() != Some(0) {
        return Err(invalid(format!("matrix {matrix} dense offsets do not start at zero")));
    }
    if locals.len() != coefficients.len() || offsets.last().copied().map(|value| value as usize) != Some(locals.len()) {
        return Err(invalid(format!(
            "matrix {matrix} dense dictionary lengths do not match"
        )));
    }
    for (pattern, pair) in offsets.windows(2).enumerate() {
        let start = pair[0] as usize;
        let end = pair[1] as usize;
        if start >= end || end > locals.len() {
            return Err(invalid(format!(
                "matrix {matrix} dense pattern {pattern} is empty or out of range"
            )));
        }
        let local_slice = &locals[start..end];
        if local_slice.iter().any(|&local| local as usize >= D)
            || local_slice.windows(2).any(|pair| pair[0] >= pair[1])
            || coefficients[start..end]
                .iter()
                .any(|&value| value == F::ZERO)
        {
            return Err(invalid(format!(
                "matrix {matrix} dense pattern {pattern} is not canonical"
            )));
        }
    }
    Ok(())
}

fn validate_offsets(
    matrix: usize,
    label: &str,
    store: &RowOffsetStore,
    expected_len: usize,
    terminal: usize,
    allow_empty: bool,
) -> Result<(), SuperneoCacheArtifactError> {
    match store {
        RowOffsetStore::Empty => {
            if allow_empty && terminal == 0 {
                return Ok(());
            }
            return Err(invalid(format!("matrix {matrix} {label} offsets are missing")));
        }
        RowOffsetStore::U16Chunked {
            chunk_offsets,
            local_offsets,
        } => {
            if chunk_offsets.len() != expected_len.div_ceil(RowOffsetStore::CHUNK_ROWS)
                || local_offsets.len() != expected_len
            {
                return Err(invalid(format!(
                    "matrix {matrix} {label} u16 offset lengths do not match"
                )));
            }
            for chunk in 0..chunk_offsets.len() {
                if local_offsets[chunk * RowOffsetStore::CHUNK_ROWS] != 0 {
                    return Err(invalid(format!("matrix {matrix} {label} chunk does not start at zero")));
                }
            }
        }
        RowOffsetStore::U24(bytes) => {
            if bytes.len()
                != expected_len
                    .checked_mul(3)
                    .ok_or_else(|| invalid("u24 offset length overflows"))?
            {
                return Err(invalid(format!(
                    "matrix {matrix} {label} u24 offset length does not match"
                )));
            }
        }
        RowOffsetStore::U32(offsets) => {
            if offsets.len() != expected_len {
                return Err(invalid(format!(
                    "matrix {matrix} {label} u32 offset length does not match"
                )));
            }
        }
    }
    let terminal =
        u32::try_from(terminal).map_err(|_| invalid(format!("matrix {matrix} {label} terminal exceeds u32")))?;
    if store.get(0) != 0 || store.get(expected_len - 1) != terminal {
        return Err(invalid(format!(
            "matrix {matrix} {label} offset terminals do not match"
        )));
    }
    let mut previous = 0u32;
    for position in 0..expected_len {
        let value = store.get(position);
        if value < previous || value > terminal {
            return Err(invalid(format!("matrix {matrix} {label} offsets are not monotone")));
        }
        previous = value;
    }
    Ok(())
}

fn validate_row_blocks(
    matrix_index: usize,
    matrix: &SuperneoMatrixCache,
    dense_count: usize,
) -> Result<(), SuperneoCacheArtifactError> {
    let column_blocks = matrix.cols / D;
    let mut seen_dense = vec![false; matrix.dense_row_blocks.len()];
    for row in 0..matrix.rows {
        let mut previous = None;
        for block in &matrix.row_blocks[matrix.row_offsets.range(row)] {
            let block_index = if let Some((block_index, local, _)) = block.single_parts() {
                if local >= D {
                    return Err(invalid(format!(
                        "matrix {matrix_index} row {row} single block local is invalid"
                    )));
                }
                block_index
            } else {
                let dense = block.dense_index().expect("packed row reference kind");
                let Some(dense_row) = matrix.dense_row_blocks.get(dense) else {
                    return Err(invalid(format!(
                        "matrix {matrix_index} row {row} dense row-block index is invalid"
                    )));
                };
                if seen_dense[dense] || dense_row.pattern() >= dense_count {
                    return Err(invalid(format!(
                        "matrix {matrix_index} row {row} dense row-block is not canonical"
                    )));
                }
                seen_dense[dense] = true;
                dense_row.block()
            };
            if block_index >= column_blocks || previous.is_some_and(|value| value >= block_index) {
                return Err(invalid(format!(
                    "matrix {matrix_index} row {row} block order is invalid"
                )));
            }
            previous = Some(block_index);
        }
    }
    if seen_dense.iter().any(|seen| !seen) {
        return Err(invalid(format!(
            "matrix {matrix_index} has an unreferenced dense row block"
        )));
    }
    Ok(())
}

fn validate_geometric_runs(
    matrix_index: usize,
    matrix: &SuperneoMatrixCache,
) -> Result<(), SuperneoCacheArtifactError> {
    for row in 0..matrix.rows {
        let mut previous = None;
        for run in &matrix.geometric_runs[matrix.geometric_row_offsets.range(row)] {
            let start = run[0] as u32 as usize;
            let len = (run[0] >> 32) as u32 as usize;
            let end = start.checked_add(len);
            if len == 0 || end.is_none_or(|value| value > matrix.cols) {
                return Err(invalid(format!(
                    "matrix {matrix_index} row {row} geometric run is out of range"
                )));
            }
            let key = (start, len);
            if previous.is_some_and(|value| value > key) {
                return Err(invalid(format!(
                    "matrix {matrix_index} row {row} geometric runs are not sorted"
                )));
            }
            previous = Some(key);
            if run[1] >= F::ORDER_U64 || run[2] >= F::ORDER_U64 {
                return Err(invalid(format!(
                    "matrix {matrix_index} row {row} geometric coefficient is not canonical"
                )));
            }
        }
    }
    Ok(())
}

fn validate_masks(cache: &SuperneoEvalCache, rows: usize) -> Result<(), SuperneoCacheArtifactError> {
    if cache.mats.len() > u16::BITS as usize {
        if cache.explicit_matrix_masks.is_some() {
            return Err(invalid("explicit matrix masks exist for more than 16 matrices"));
        }
        return Ok(());
    }
    let Some(masks) = &cache.explicit_matrix_masks else {
        return Err(invalid("explicit matrix masks are missing"));
    };
    if masks.len() != rows {
        return Err(invalid("explicit matrix mask row count does not match"));
    }
    for (row, &actual) in masks.iter().enumerate() {
        let expected = cache
            .mats
            .iter()
            .enumerate()
            .fold(0u16, |mask, (matrix, value)| {
                if !value.identity
                    && (value.row_offsets.get(row) != value.row_offsets.get(row + 1)
                        || value.geometric_row_offsets.get(row) != value.geometric_row_offsets.get(row + 1))
                {
                    mask | (1 << matrix)
                } else {
                    mask
                }
            });
        if actual != expected {
            return Err(invalid(format!("explicit matrix mask does not match row {row}")));
        }
    }
    Ok(())
}

fn offset_encoded_size(store: &RowOffsetStore) -> Result<u64, SuperneoCacheArtifactError> {
    let bytes = match store {
        RowOffsetStore::Empty => 1,
        RowOffsetStore::U16Chunked {
            chunk_offsets,
            local_offsets,
        } => checked_sum(&[
            1,
            checked_product(chunk_offsets.len(), 4, "u16 chunk bases")?,
            checked_product(local_offsets.len(), 2, "u16 local offsets")?,
        ])?,
        RowOffsetStore::U24(bytes) => checked_sum(&[1, usize_to_u64(bytes.len(), "u24 offsets")?])?,
        RowOffsetStore::U32(offsets) => checked_sum(&[1, checked_product(offsets.len(), 4, "u32 offsets")?])?,
    };
    Ok(bytes)
}

fn encoded_size(cache: &SuperneoEvalCache) -> Result<u64, SuperneoCacheArtifactError> {
    let mut size = HEADER_BYTES;
    for matrix in &cache.mats {
        size = checked_sum(&[
            size,
            8 + 8 + 1,
            offset_encoded_size(&matrix.row_offsets)?,
            8,
            checked_product(matrix.row_blocks.len(), 4, "row references")?,
            8,
            checked_product(matrix.dense_row_blocks.len(), 8, "dense row blocks")?,
        ])?;
        let DenseBlockStore::Compact {
            offsets,
            locals,
            coefficients,
        } = &matrix.dense_orig
        else {
            return Err(invalid("cannot size unfinished dense dictionary"));
        };
        size = checked_sum(&[
            size,
            8,
            checked_product(offsets.len(), 4, "dense offsets")?,
            8,
            usize_to_u64(locals.len(), "dense locals")?,
            8,
            checked_product(coefficients.len(), 8, "dense coefficients")?,
            offset_encoded_size(&matrix.geometric_row_offsets)?,
            8,
            checked_product(matrix.geometric_runs.len(), 24, "geometric runs")?,
            8,
        ])?;
        for block in &matrix.seeded_phi81_blocks {
            let seed_count = block
                .chunk_seeds_by_row()
                .iter()
                .try_fold(0usize, |sum, seeds| sum.checked_add(seeds.len()))
                .ok_or_else(|| invalid("seed count overflows"))?;
            size = checked_sum(&[
                size,
                5 * 8 + 1 + 8,
                checked_product(block.word_starts().len(), 8, "seeded word starts")?,
                checked_product(seed_count, 32, "seeded chunk seeds")?,
            ])?;
        }
    }
    size = checked_sum(&[size, 1])?;
    if let Some(masks) = &cache.explicit_matrix_masks {
        size = checked_sum(&[size, 8, checked_product(masks.len(), 2, "explicit masks")?])?;
    }
    Ok(size)
}

fn checked_sum(values: &[u64]) -> Result<u64, SuperneoCacheArtifactError> {
    values
        .iter()
        .try_fold(0u64, |sum, &value| sum.checked_add(value))
        .ok_or_else(|| invalid("artifact byte size overflows u64"))
}

fn checked_product(count: usize, width: u64, label: &str) -> Result<u64, SuperneoCacheArtifactError> {
    usize_to_u64(count, label)?
        .checked_mul(width)
        .ok_or_else(|| invalid(format!("{label} byte size overflows u64")))
}

fn usize_to_u64(value: usize, label: &str) -> Result<u64, SuperneoCacheArtifactError> {
    u64::try_from(value).map_err(|_| invalid(format!("{label} exceeds u64")))
}

struct Encoder<W> {
    inner: W,
    written: u64,
}

impl<W: Write> Encoder<W> {
    fn new(inner: W) -> Self {
        Self { inner, written: 0 }
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), SuperneoCacheArtifactError> {
        self.inner.write_all(bytes)?;
        self.written = self
            .written
            .checked_add(usize_to_u64(bytes.len(), "written byte count")?)
            .ok_or_else(|| invalid("written byte count overflows"))?;
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<(), SuperneoCacheArtifactError> {
        self.bytes(&[value])
    }

    fn u16(&mut self, value: u16) -> Result<(), SuperneoCacheArtifactError> {
        self.bytes(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> Result<(), SuperneoCacheArtifactError> {
        self.bytes(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> Result<(), SuperneoCacheArtifactError> {
        self.bytes(&value.to_le_bytes())
    }

    fn usize(&mut self, value: usize, label: &str) -> Result<(), SuperneoCacheArtifactError> {
        self.u64(usize_to_u64(value, label)?)
    }

    fn digest_words(&mut self, words: &[u64; DIGEST_LEN]) -> Result<(), SuperneoCacheArtifactError> {
        for &word in words {
            self.u64(word)?;
        }
        Ok(())
    }
}

fn encode_matrix<W: Write>(
    encoder: &mut Encoder<W>,
    matrix: &SuperneoMatrixCache,
) -> Result<(), SuperneoCacheArtifactError> {
    encoder.usize(matrix.rows, "matrix rows")?;
    encoder.usize(matrix.cols, "matrix columns")?;
    encoder.u8(u8::from(matrix.identity))?;
    encode_offsets(encoder, &matrix.row_offsets)?;
    encoder.usize(matrix.row_blocks.len(), "row block count")?;
    for block in &matrix.row_blocks {
        encoder.u32(block.word())?;
    }
    encoder.usize(matrix.dense_row_blocks.len(), "dense row block count")?;
    for block in &matrix.dense_row_blocks {
        let [column_block, pattern] = block.words();
        encoder.u32(column_block)?;
        encoder.u32(pattern)?;
    }
    let DenseBlockStore::Compact {
        offsets,
        locals,
        coefficients,
    } = &matrix.dense_orig
    else {
        return Err(invalid("cannot encode unfinished dense dictionary"));
    };
    encoder.usize(offsets.len(), "dense offset count")?;
    for &offset in offsets {
        encoder.u32(offset)?;
    }
    encoder.usize(locals.len(), "dense local count")?;
    encoder.bytes(locals)?;
    encoder.usize(coefficients.len(), "dense coefficient count")?;
    for coefficient in coefficients {
        encoder.u64(coefficient.as_canonical_u64())?;
    }
    encode_offsets(encoder, &matrix.geometric_row_offsets)?;
    encoder.usize(matrix.geometric_runs.len(), "geometric run count")?;
    for run in &matrix.geometric_runs {
        for &word in run {
            encoder.u64(word)?;
        }
    }
    encoder.usize(matrix.seeded_phi81_blocks.len(), "seeded block count")?;
    for block in &matrix.seeded_phi81_blocks {
        encode_seeded_block(encoder, block)?;
    }
    Ok(())
}

fn encode_offsets<W: Write>(
    encoder: &mut Encoder<W>,
    offsets: &RowOffsetStore,
) -> Result<(), SuperneoCacheArtifactError> {
    match offsets {
        RowOffsetStore::Empty => encoder.u8(OFFSET_EMPTY),
        RowOffsetStore::U16Chunked {
            chunk_offsets,
            local_offsets,
        } => {
            encoder.u8(OFFSET_U16_CHUNKED)?;
            for &offset in chunk_offsets {
                encoder.u32(offset)?;
            }
            for &offset in local_offsets {
                encoder.u16(offset)?;
            }
            Ok(())
        }
        RowOffsetStore::U24(bytes) => {
            encoder.u8(OFFSET_U24)?;
            encoder.bytes(bytes)
        }
        RowOffsetStore::U32(offsets) => {
            encoder.u8(OFFSET_U32)?;
            for &offset in offsets {
                encoder.u32(offset)?;
            }
            Ok(())
        }
    }
}

fn encode_seeded_block<W: Write>(
    encoder: &mut Encoder<W>,
    block: &SeededPhi81LinearBlock,
) -> Result<(), SuperneoCacheArtifactError> {
    encoder.usize(block.row_start(), "seeded row start")?;
    encoder.usize(block.word_width(), "seeded word width")?;
    encoder.usize(block.kappa(), "seeded kappa")?;
    encoder.usize(block.message_cols(), "seeded message columns")?;
    encoder.usize(block.chunk_size(), "seeded chunk size")?;
    encoder.u8(u8::from(block.has_superneo_transformed_columns()))?;
    encoder.usize(block.word_starts().len(), "seeded word count")?;
    for &start in block.word_starts() {
        encoder.usize(start, "seeded word start")?;
    }
    for seeds in block.chunk_seeds_by_row() {
        for seed in seeds {
            encoder.bytes(seed)?;
        }
    }
    Ok(())
}

fn encode_masks<W: Write>(encoder: &mut Encoder<W>, masks: Option<&[u16]>) -> Result<(), SuperneoCacheArtifactError> {
    let Some(masks) = masks else {
        return encoder.u8(0);
    };
    encoder.u8(1)?;
    encoder.usize(masks.len(), "explicit mask count")?;
    for &mask in masks {
        encoder.u16(mask)?;
    }
    Ok(())
}

struct Decoder<R> {
    inner: R,
    consumed: u64,
    limit: u64,
}

impl<R: Read> Decoder<R> {
    fn new(inner: R, limit: u64) -> Self {
        Self {
            inner,
            consumed: 0,
            limit,
        }
    }

    fn set_exact_limit(&mut self, limit: u64) -> Result<(), SuperneoCacheArtifactError> {
        if limit > self.limit || limit < self.consumed {
            return Err(invalid("encoded byte limit is invalid"));
        }
        self.limit = limit;
        Ok(())
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], SuperneoCacheArtifactError> {
        let next = self
            .consumed
            .checked_add(N as u64)
            .ok_or_else(|| invalid("decoded byte count overflows"))?;
        if next > self.limit {
            return Err(invalid("encoded data exceeds its declared byte size"));
        }
        let mut out = [0u8; N];
        self.inner.read_exact(&mut out)?;
        self.consumed = next;
        Ok(out)
    }

    fn bytes(&mut self, len: usize, label: &str) -> Result<Vec<u8>, SuperneoCacheArtifactError> {
        let len_u64 = usize_to_u64(len, label)?;
        if len_u64 > self.remaining() {
            return Err(invalid(format!("{label} exceeds remaining artifact bytes")));
        }
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| invalid(format!("cannot reserve {label}")))?;
        out.resize(len, 0);
        self.inner.read_exact(&mut out)?;
        self.consumed += len_u64;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, SuperneoCacheArtifactError> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, SuperneoCacheArtifactError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, SuperneoCacheArtifactError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, SuperneoCacheArtifactError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn usize(&mut self, label: &str) -> Result<usize, SuperneoCacheArtifactError> {
        usize::try_from(self.u64()?).map_err(|_| invalid(format!("{label} exceeds usize")))
    }

    fn len(&mut self, width: u64, label: &str) -> Result<usize, SuperneoCacheArtifactError> {
        let count = self.usize(label)?;
        let bytes = checked_product(count, width, label)?;
        if bytes > self.remaining() {
            return Err(invalid(format!("{label} exceeds remaining artifact bytes")));
        }
        Ok(count)
    }

    fn digest_words(&mut self) -> Result<[u64; DIGEST_LEN], SuperneoCacheArtifactError> {
        let mut words = [0u64; DIGEST_LEN];
        for word in &mut words {
            *word = self.u64()?;
        }
        Ok(words)
    }

    fn remaining(&self) -> u64 {
        self.limit - self.consumed
    }

    fn finish_exact(mut self) -> Result<(), SuperneoCacheArtifactError> {
        if self.consumed != self.limit {
            return Err(invalid(format!(
                "decoded {} bytes, expected {}",
                self.consumed, self.limit
            )));
        }
        let mut trailing = [0u8; 1];
        if self.inner.read(&mut trailing)? != 0 {
            return Err(invalid("artifact has trailing bytes"));
        }
        Ok(())
    }
}

fn decode_matrix<R: Read>(
    decoder: &mut Decoder<R>,
    limits: SuperneoCacheArtifactLimits,
) -> Result<SuperneoMatrixCache, SuperneoCacheArtifactError> {
    let rows = decoder.usize("matrix rows")?;
    if rows > limits.max_rows {
        return Err(limit(
            "row count",
            usize_to_u64(rows, "matrix rows")?,
            usize_to_u64(limits.max_rows, "row limit")?,
        ));
    }
    let cols = decoder.usize("matrix columns")?;
    if cols > limits.max_cols {
        return Err(limit(
            "column count",
            usize_to_u64(cols, "matrix columns")?,
            usize_to_u64(limits.max_cols, "column limit")?,
        ));
    }
    let identity = decode_bool(decoder.u8()?, "identity flag")?;
    let offset_len = rows
        .checked_add(1)
        .ok_or_else(|| invalid("matrix row count overflows"))?;
    let row_offsets = decode_offsets(decoder, offset_len, "row offsets")?;
    let row_block_count = decoder.len(4, "row block count")?;
    let mut row_blocks = Vec::new();
    row_blocks
        .try_reserve_exact(row_block_count)
        .map_err(|_| invalid("cannot reserve row blocks"))?;
    for _ in 0..row_block_count {
        row_blocks.push(CompactRowBlock::from_word(decoder.u32()?));
    }
    let dense_row_block_count = decoder.len(8, "dense row block count")?;
    let mut dense_row_blocks = Vec::new();
    dense_row_blocks
        .try_reserve_exact(dense_row_block_count)
        .map_err(|_| invalid("cannot reserve dense row blocks"))?;
    for _ in 0..dense_row_block_count {
        dense_row_blocks.push(DenseRowBlock::from_words(decoder.u32()?, decoder.u32()?));
    }

    let dense_offset_count = decoder.len(4, "dense offset count")?;
    let mut dense_offsets = Vec::new();
    dense_offsets
        .try_reserve_exact(dense_offset_count)
        .map_err(|_| invalid("cannot reserve dense offsets"))?;
    for _ in 0..dense_offset_count {
        dense_offsets.push(decoder.u32()?);
    }
    let dense_local_count = decoder.len(1, "dense local count")?;
    let dense_locals = decoder.bytes(dense_local_count, "dense locals")?;
    let dense_coefficient_count = decoder.len(8, "dense coefficient count")?;
    let mut dense_coefficients = Vec::new();
    dense_coefficients
        .try_reserve_exact(dense_coefficient_count)
        .map_err(|_| invalid("cannot reserve dense coefficients"))?;
    for _ in 0..dense_coefficient_count {
        let word = decoder.u64()?;
        if word >= F::ORDER_U64 {
            return Err(invalid("dense coefficient is not a canonical Goldilocks word"));
        }
        dense_coefficients.push(F::from_u64(word));
    }

    let geometric_row_offsets = decode_offsets(decoder, offset_len, "geometric row offsets")?;
    let geometric_run_count = decoder.len(24, "geometric run count")?;
    let mut geometric_runs = Vec::new();
    geometric_runs
        .try_reserve_exact(geometric_run_count)
        .map_err(|_| invalid("cannot reserve geometric runs"))?;
    for _ in 0..geometric_run_count {
        geometric_runs.push([decoder.u64()?, decoder.u64()?, decoder.u64()?]);
    }

    let seeded_block_count = decoder.len(41, "seeded block count")?;
    let mut seeded_phi81_blocks = Vec::new();
    seeded_phi81_blocks
        .try_reserve_exact(seeded_block_count)
        .map_err(|_| invalid("cannot reserve seeded blocks"))?;
    for _ in 0..seeded_block_count {
        seeded_phi81_blocks.push(decode_seeded_block(decoder, rows, cols)?);
    }

    Ok(SuperneoMatrixCache {
        rows,
        cols,
        row_offsets,
        row_blocks,
        dense_row_blocks,
        dense_orig: DenseBlockStore::Compact {
            offsets: dense_offsets,
            locals: dense_locals,
            coefficients: dense_coefficients,
        },
        geometric_row_offsets,
        geometric_runs,
        identity,
        seeded_phi81_blocks,
    })
}

fn decode_offsets<R: Read>(
    decoder: &mut Decoder<R>,
    expected_len: usize,
    label: &str,
) -> Result<RowOffsetStore, SuperneoCacheArtifactError> {
    match decoder.u8()? {
        OFFSET_EMPTY => Ok(RowOffsetStore::Empty),
        OFFSET_U16_CHUNKED => {
            let chunk_count = expected_len.div_ceil(RowOffsetStore::CHUNK_ROWS);
            ensure_remaining(decoder, chunk_count, 4, label)?;
            ensure_remaining(decoder, expected_len, 2, label)?;
            let mut chunk_offsets = Vec::new();
            chunk_offsets
                .try_reserve_exact(chunk_count)
                .map_err(|_| invalid(format!("cannot reserve {label} chunk bases")))?;
            for _ in 0..chunk_count {
                chunk_offsets.push(decoder.u32()?);
            }
            let mut local_offsets = Vec::new();
            local_offsets
                .try_reserve_exact(expected_len)
                .map_err(|_| invalid(format!("cannot reserve {label} locals")))?;
            for _ in 0..expected_len {
                local_offsets.push(decoder.u16()?);
            }
            Ok(RowOffsetStore::U16Chunked {
                chunk_offsets,
                local_offsets,
            })
        }
        OFFSET_U24 => {
            let len = expected_len
                .checked_mul(3)
                .ok_or_else(|| invalid(format!("{label} byte count overflows")))?;
            Ok(RowOffsetStore::U24(decoder.bytes(len, label)?))
        }
        OFFSET_U32 => {
            ensure_remaining(decoder, expected_len, 4, label)?;
            let mut offsets = Vec::new();
            offsets
                .try_reserve_exact(expected_len)
                .map_err(|_| invalid(format!("cannot reserve {label}")))?;
            for _ in 0..expected_len {
                offsets.push(decoder.u32()?);
            }
            Ok(RowOffsetStore::U32(offsets))
        }
        _ => Err(invalid(format!("{label} storage tag is invalid"))),
    }
}

fn ensure_remaining<R: Read>(
    decoder: &Decoder<R>,
    count: usize,
    width: u64,
    label: &str,
) -> Result<(), SuperneoCacheArtifactError> {
    let bytes = checked_product(count, width, label)?;
    if bytes > decoder.remaining() {
        return Err(invalid(format!("{label} exceeds remaining artifact bytes")));
    }
    Ok(())
}

fn decode_seeded_block<R: Read>(
    decoder: &mut Decoder<R>,
    rows: usize,
    cols: usize,
) -> Result<SeededPhi81LinearBlock, SuperneoCacheArtifactError> {
    let row_start = decoder.usize("seeded row start")?;
    let word_width = decoder.usize("seeded word width")?;
    let kappa = decoder.usize("seeded kappa")?;
    let message_cols = decoder.usize("seeded message columns")?;
    let chunk_size = decoder.usize("seeded chunk size")?;
    let transformed = decode_bool(decoder.u8()?, "seeded transform flag")?;
    let word_count = decoder.len(8, "seeded word count")?;
    if word_count == 0 || word_width == 0 || kappa == 0 || message_cols == 0 || chunk_size == 0 {
        return Err(invalid("seeded block has a zero required dimension"));
    }
    let bit_count = word_count
        .checked_mul(word_width)
        .ok_or_else(|| invalid("seeded input width overflows"))?;
    if bit_count.div_ceil(D) != message_cols {
        return Err(invalid("seeded message column count does not match its input width"));
    }
    let row_end = kappa
        .checked_mul(D)
        .and_then(|height| row_start.checked_add(height))
        .ok_or_else(|| invalid("seeded row range overflows"))?;
    if row_end > rows {
        return Err(invalid("seeded row range exceeds the matrix"));
    }
    let mut word_starts = Vec::new();
    word_starts
        .try_reserve_exact(word_count)
        .map_err(|_| invalid("cannot reserve seeded word starts"))?;
    for _ in 0..word_count {
        let start = decoder.usize("seeded word start")?;
        if start.checked_add(word_width).is_none_or(|end| end > cols) {
            return Err(invalid("seeded word range exceeds the matrix"));
        }
        word_starts.push(start);
    }
    let chunks_per_row = message_cols.div_ceil(chunk_size);
    let seed_count = kappa
        .checked_mul(chunks_per_row)
        .ok_or_else(|| invalid("seeded chunk count overflows"))?;
    ensure_remaining(decoder, seed_count, 32, "seeded chunk seeds")?;
    let mut chunk_seeds_by_row = Vec::new();
    chunk_seeds_by_row
        .try_reserve_exact(kappa)
        .map_err(|_| invalid("cannot reserve seeded rows"))?;
    for _ in 0..kappa {
        let mut seeds = Vec::new();
        seeds
            .try_reserve_exact(chunks_per_row)
            .map_err(|_| invalid("cannot reserve seeded row chunks"))?;
        for _ in 0..chunks_per_row {
            seeds.push(decoder.array::<32>()?);
        }
        chunk_seeds_by_row.push(seeds);
    }
    let mut block = SeededPhi81LinearBlock::new_with_word_width(
        row_start,
        word_starts,
        word_width,
        kappa,
        message_cols,
        chunk_size,
        chunk_seeds_by_row,
    )
    .map_err(|error| invalid(format!("seeded block metadata is invalid: {error}")))?;
    if transformed {
        block = block.with_superneo_transformed_columns();
    }
    block
        .validate_matrix_shape(rows, cols)
        .map_err(|error| invalid(format!("seeded block shape is invalid: {error}")))?;
    Ok(block)
}

fn decode_masks<R: Read>(decoder: &mut Decoder<R>) -> Result<Option<Vec<u16>>, SuperneoCacheArtifactError> {
    match decoder.u8()? {
        0 => Ok(None),
        1 => {
            let count = decoder.len(2, "explicit mask count")?;
            let mut masks = Vec::new();
            masks
                .try_reserve_exact(count)
                .map_err(|_| invalid("cannot reserve explicit masks"))?;
            for _ in 0..count {
                masks.push(decoder.u16()?);
            }
            Ok(Some(masks))
        }
        _ => Err(invalid("explicit mask option tag is invalid")),
    }
}

fn decode_bool(value: u8, label: &str) -> Result<bool, SuperneoCacheArtifactError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid(format!("{label} is not Boolean"))),
    }
}

#[derive(Clone, Copy)]
enum DigestLeaf<'a> {
    MatrixMeta {
        matrix: usize,
        value: &'a SuperneoMatrixCache,
    },
    U32 {
        matrix: usize,
        section: u64,
        start: usize,
        values: &'a [u32],
    },
    U16 {
        matrix: usize,
        section: u64,
        start: usize,
        values: &'a [u16],
    },
    Bytes {
        matrix: usize,
        section: u64,
        start: usize,
        values: &'a [u8],
    },
    RowBlocks {
        matrix: usize,
        start: usize,
        values: &'a [CompactRowBlock],
    },
    DenseRowBlocks {
        matrix: usize,
        start: usize,
        values: &'a [DenseRowBlock],
    },
    Fields {
        matrix: usize,
        start: usize,
        values: &'a [F],
    },
    Runs {
        matrix: usize,
        start: usize,
        values: &'a [[u64; 3]],
    },
    Seeded {
        matrix: usize,
        block: usize,
        value: &'a SeededPhi81LinearBlock,
    },
    MasksMeta {
        present: bool,
        len: usize,
    },
    Masks {
        start: usize,
        values: &'a [u16],
    },
}

fn digest_leaves(cache: &SuperneoEvalCache) -> Vec<DigestLeaf<'_>> {
    let mut leaves = Vec::new();
    for (matrix_index, matrix) in cache.mats.iter().enumerate() {
        leaves.push(DigestLeaf::MatrixMeta {
            matrix: matrix_index,
            value: matrix,
        });
        push_offset_leaves(
            &mut leaves,
            matrix_index,
            &matrix.row_offsets,
            [
                SECTION_ROW_CHUNK_BASES,
                SECTION_ROW_LOCALS,
                SECTION_ROW_U24,
                SECTION_ROW_U32,
            ],
        );
        for (chunk, values) in matrix.row_blocks.chunks(DIGEST_CHUNK_ITEMS).enumerate() {
            leaves.push(DigestLeaf::RowBlocks {
                matrix: matrix_index,
                start: chunk * DIGEST_CHUNK_ITEMS,
                values,
            });
        }
        for (chunk, values) in matrix
            .dense_row_blocks
            .chunks(DIGEST_CHUNK_ITEMS)
            .enumerate()
        {
            leaves.push(DigestLeaf::DenseRowBlocks {
                matrix: matrix_index,
                start: chunk * DIGEST_CHUNK_ITEMS,
                values,
            });
        }
        let DenseBlockStore::Compact {
            offsets,
            locals,
            coefficients,
        } = &matrix.dense_orig
        else {
            unreachable!("cache validation rejects unfinished dense dictionaries")
        };
        push_u32_leaves(&mut leaves, matrix_index, SECTION_DENSE_OFFSETS, offsets);
        push_byte_leaves(&mut leaves, matrix_index, SECTION_DENSE_LOCALS, locals);
        for (chunk, values) in coefficients.chunks(DIGEST_CHUNK_ITEMS).enumerate() {
            leaves.push(DigestLeaf::Fields {
                matrix: matrix_index,
                start: chunk * DIGEST_CHUNK_ITEMS,
                values,
            });
        }
        push_offset_leaves(
            &mut leaves,
            matrix_index,
            &matrix.geometric_row_offsets,
            [
                SECTION_GEOMETRIC_CHUNK_BASES,
                SECTION_GEOMETRIC_LOCALS,
                SECTION_GEOMETRIC_U24,
                SECTION_GEOMETRIC_U32,
            ],
        );
        for (chunk, values) in matrix.geometric_runs.chunks(DIGEST_CHUNK_ITEMS).enumerate() {
            leaves.push(DigestLeaf::Runs {
                matrix: matrix_index,
                start: chunk * DIGEST_CHUNK_ITEMS,
                values,
            });
        }
        for (block, value) in matrix.seeded_phi81_blocks.iter().enumerate() {
            leaves.push(DigestLeaf::Seeded {
                matrix: matrix_index,
                block,
                value,
            });
        }
    }
    leaves.push(DigestLeaf::MasksMeta {
        present: cache.explicit_matrix_masks.is_some(),
        len: cache.explicit_matrix_masks.as_ref().map_or(0, Vec::len),
    });
    if let Some(masks) = &cache.explicit_matrix_masks {
        for (chunk, values) in masks.chunks(DIGEST_CHUNK_ITEMS).enumerate() {
            leaves.push(DigestLeaf::Masks {
                start: chunk * DIGEST_CHUNK_ITEMS,
                values,
            });
        }
    }
    leaves
}

fn push_offset_leaves<'a>(
    leaves: &mut Vec<DigestLeaf<'a>>,
    matrix: usize,
    offsets: &'a RowOffsetStore,
    sections: [u64; 4],
) {
    match offsets {
        RowOffsetStore::Empty => {}
        RowOffsetStore::U16Chunked {
            chunk_offsets,
            local_offsets,
        } => {
            push_u32_leaves(leaves, matrix, sections[0], chunk_offsets);
            for (chunk, values) in local_offsets.chunks(DIGEST_CHUNK_ITEMS).enumerate() {
                leaves.push(DigestLeaf::U16 {
                    matrix,
                    section: sections[1],
                    start: chunk * DIGEST_CHUNK_ITEMS,
                    values,
                });
            }
        }
        RowOffsetStore::U24(bytes) => push_byte_leaves(leaves, matrix, sections[2], bytes),
        RowOffsetStore::U32(values) => push_u32_leaves(leaves, matrix, sections[3], values),
    }
}

fn push_u32_leaves<'a>(leaves: &mut Vec<DigestLeaf<'a>>, matrix: usize, section: u64, values: &'a [u32]) {
    for (chunk, values) in values.chunks(DIGEST_CHUNK_ITEMS).enumerate() {
        leaves.push(DigestLeaf::U32 {
            matrix,
            section,
            start: chunk * DIGEST_CHUNK_ITEMS,
            values,
        });
    }
}

fn push_byte_leaves<'a>(leaves: &mut Vec<DigestLeaf<'a>>, matrix: usize, section: u64, values: &'a [u8]) {
    let byte_chunk = DIGEST_CHUNK_ITEMS * 4;
    for (chunk, values) in values.chunks(byte_chunk).enumerate() {
        leaves.push(DigestLeaf::Bytes {
            matrix,
            section,
            start: chunk * byte_chunk,
            values,
        });
    }
}
