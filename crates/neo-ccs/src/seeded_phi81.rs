//! Compact seeded linear blocks over `F[X] / (Phi_81)`.
//!
//! A block names the exact linear map used by a seeded Ajtai commitment
//! without materializing every rotated coefficient in CSC form. The public
//! chunk seeds are the coefficient source; forward and transpose products
//! expand one ring element at a time.

use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use neo_math::D;

/// Canonical Goldilocks word width consumed by each input entry.
pub const CANONICAL_FIELD_BITS: usize = 64;

/// Invalid compact-block geometry or coefficient-source metadata.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SeededPhi81Error {
    /// No input words were supplied.
    #[error("seeded Phi81 block requires at least one input word")]
    EmptyInput,
    /// One input word has no coordinates.
    #[error("seeded Phi81 block input-word width must be nonzero")]
    ZeroWordWidth,
    /// The output commitment has no ring columns.
    #[error("seeded Phi81 block requires at least one output column")]
    ZeroKappa,
    /// The declared row-major message width disagrees with the inputs.
    #[error("seeded Phi81 block message columns {actual} != ceil({bits}/{dimension}) = {expected}")]
    MessageColumns {
        /// Declared message columns.
        actual: usize,
        /// Number of input bits.
        bits: usize,
        /// Ring dimension used for packing.
        dimension: usize,
        /// Required message columns.
        expected: usize,
    },
    /// The seeded-PP chunk width was zero.
    #[error("seeded Phi81 block chunk size must be nonzero")]
    ZeroChunkSize,
    /// The number of seed rows disagrees with `kappa`.
    #[error("seeded Phi81 block has {actual} seed rows, expected {expected}")]
    SeedRowCount {
        /// Supplied seed rows.
        actual: usize,
        /// Required seed rows.
        expected: usize,
    },
    /// One seed row has the wrong number of deterministic chunks.
    #[error("seeded Phi81 block seed row {row} has {actual} chunks, expected {expected}")]
    SeedChunkCount {
        /// Seed-row index.
        row: usize,
        /// Supplied chunk count.
        actual: usize,
        /// Required chunk count.
        expected: usize,
    },
    /// The output rows exceed the enclosing matrix.
    #[error("seeded Phi81 block rows [{start}, {end}) exceed matrix row count {rows}")]
    RowRange {
        /// First block row.
        start: usize,
        /// Exclusive block-row end.
        end: usize,
        /// Enclosing matrix row count.
        rows: usize,
    },
    /// One canonical input word exceeds the enclosing matrix.
    #[error("seeded Phi81 input word {word} columns [{start}, {end}) exceed matrix column count {cols}")]
    ColumnRange {
        /// Input-word index.
        word: usize,
        /// First word column.
        start: usize,
        /// Exclusive word-column end.
        end: usize,
        /// Enclosing matrix column count.
        cols: usize,
    },
}

/// One seeded Ajtai-style map from fixed-width low-norm words to `D*kappa`
/// coefficient rows. Input words may repeat, but each word's coordinates are
/// contiguous in the enclosing matrix.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeededPhi81LinearBlock {
    row_start: usize,
    word_starts: Vec<usize>,
    word_width: usize,
    kappa: usize,
    message_cols: usize,
    chunk_size: usize,
    chunk_seeds_by_row: Vec<Vec<[u8; 32]>>,
    superneo_transformed_columns: bool,
}

impl SeededPhi81LinearBlock {
    /// Construct and validate one compact block.
    pub fn new(
        row_start: usize,
        word_starts: Vec<usize>,
        kappa: usize,
        message_cols: usize,
        chunk_size: usize,
        chunk_seeds_by_row: Vec<Vec<[u8; 32]>>,
    ) -> Result<Self, SeededPhi81Error> {
        Self::new_with_word_width(
            row_start,
            word_starts,
            CANONICAL_FIELD_BITS,
            kappa,
            message_cols,
            chunk_size,
            chunk_seeds_by_row,
        )
    }

    /// Construct one compact block over fixed-width low-norm input words.
    pub fn new_with_word_width(
        row_start: usize,
        word_starts: Vec<usize>,
        word_width: usize,
        kappa: usize,
        message_cols: usize,
        chunk_size: usize,
        chunk_seeds_by_row: Vec<Vec<[u8; 32]>>,
    ) -> Result<Self, SeededPhi81Error> {
        if word_starts.is_empty() {
            return Err(SeededPhi81Error::EmptyInput);
        }
        if word_width == 0 {
            return Err(SeededPhi81Error::ZeroWordWidth);
        }
        if kappa == 0 {
            return Err(SeededPhi81Error::ZeroKappa);
        }
        let bits = word_starts
            .len()
            .checked_mul(word_width)
            .expect("seeded Phi81 input bit count overflow");
        let expected_message_cols = bits.div_ceil(D);
        if message_cols != expected_message_cols {
            return Err(SeededPhi81Error::MessageColumns {
                actual: message_cols,
                bits,
                dimension: D,
                expected: expected_message_cols,
            });
        }
        if chunk_size == 0 {
            return Err(SeededPhi81Error::ZeroChunkSize);
        }
        if chunk_seeds_by_row.len() != kappa {
            return Err(SeededPhi81Error::SeedRowCount {
                actual: chunk_seeds_by_row.len(),
                expected: kappa,
            });
        }
        let expected_chunks = message_cols.div_ceil(chunk_size);
        for (row, seeds) in chunk_seeds_by_row.iter().enumerate() {
            if seeds.len() != expected_chunks {
                return Err(SeededPhi81Error::SeedChunkCount {
                    row,
                    actual: seeds.len(),
                    expected: expected_chunks,
                });
            }
        }
        Ok(Self {
            row_start,
            word_starts,
            word_width,
            kappa,
            message_cols,
            chunk_size,
            chunk_seeds_by_row,
            superneo_transformed_columns: false,
        })
    }

    /// Validate that all block rows and input words fit an enclosing matrix.
    pub fn validate_matrix_shape(&self, rows: usize, cols: usize) -> Result<(), SeededPhi81Error> {
        let row_end = self.row_end();
        if row_end > rows {
            return Err(SeededPhi81Error::RowRange {
                start: self.row_start,
                end: row_end,
                rows,
            });
        }
        for (word, &start) in self.word_starts.iter().enumerate() {
            let end = start
                .checked_add(self.word_width)
                .expect("seeded Phi81 input column range overflow");
            if end > cols {
                return Err(SeededPhi81Error::ColumnRange { word, start, end, cols });
            }
        }
        Ok(())
    }

    /// First matrix row owned by the block.
    pub fn row_start(&self) -> usize {
        self.row_start
    }

    /// Exclusive end of the block's matrix rows.
    pub fn row_end(&self) -> usize {
        self.row_start + D * self.kappa
    }

    /// Start column of each low-norm input word.
    pub fn word_starts(&self) -> &[usize] {
        &self.word_starts
    }

    /// Number of low-norm coordinates in each input word.
    pub fn word_width(&self) -> usize {
        self.word_width
    }

    /// Number of output ring columns.
    pub fn kappa(&self) -> usize {
        self.kappa
    }

    /// Number of row-major message columns committed by the map.
    pub fn message_cols(&self) -> usize {
        self.message_cols
    }

    /// Deterministic seeded-PP chunk width.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Public coefficient-source seed for every output row and chunk.
    pub fn chunk_seeds_by_row(&self) -> &[Vec<[u8; 32]>] {
        &self.chunk_seeds_by_row
    }

    /// Whether matrix columns carry SuperNeo's blockwise bar transform.
    pub fn has_superneo_transformed_columns(&self) -> bool {
        self.superneo_transformed_columns
    }

    /// Shift the block into a larger block-diagonal matrix.
    pub fn shifted(&self, row_offset: usize, col_offset: usize) -> Self {
        let mut out = self.clone();
        out.row_start += row_offset;
        for start in &mut out.word_starts {
            *start += col_offset;
        }
        out
    }

    /// Return the same map after SuperNeo's per-column-block bar transform.
    pub fn with_superneo_transformed_columns(&self) -> Self {
        let mut out = self.clone();
        out.superneo_transformed_columns = true;
        out
    }

    /// Reuse the coefficient source with different row and input-word geometry.
    pub fn with_geometry(&self, row_start: usize, word_starts: Vec<usize>) -> Result<Self, SeededPhi81Error> {
        let mut out = Self::new_with_word_width(
            row_start,
            word_starts,
            self.word_width,
            self.kappa,
            self.message_cols,
            self.chunk_size,
            self.chunk_seeds_by_row.clone(),
        )?;
        out.superneo_transformed_columns = self.superneo_transformed_columns;
        Ok(out)
    }

    /// Visit every nonzero `(row, column, coefficient)` in the block.
    /// This is intended for parity tests and compatibility fallbacks; hot
    /// proving paths should use the streaming products below.
    pub fn for_each_term<Ff, Visit>(&self, mut visit: Visit)
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        Visit: FnMut(usize, usize, Ff),
    {
        if self.superneo_transformed_columns {
            self.for_each_superneo_transformed_term(&mut visit);
            return;
        }
        self.for_each_original_term(&mut visit);
    }

    /// Visit every nonzero `(column, coefficient)` in one block row.
    pub fn for_each_row_term<Ff, Visit>(&self, row: usize, mut visit: Visit)
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        Visit: FnMut(usize, Ff),
    {
        if row < self.row_start || row >= self.row_end() {
            return;
        }
        if self.superneo_transformed_columns {
            self.for_each_term::<Ff, _>(|candidate_row, column, coefficient| {
                if candidate_row == row {
                    visit(column, coefficient);
                }
            });
            return;
        }

        let local_row = row - self.row_start;
        let output = local_row / D;
        let coordinate = local_row % D;
        let seeds = &self.chunk_seeds_by_row[output];
        for (chunk, &seed) in seeds.iter().enumerate() {
            let start = chunk * self.chunk_size;
            let end = core::cmp::min(self.message_cols, start + self.chunk_size);
            let mut rng = ChaCha8Rng::from_seed(seed);
            for message_col in start..end {
                let mut rotation = sample_uniform_coefficients::<Ff>(&mut rng);
                for message_row in 0..D {
                    let bit_index = message_row * self.message_cols + message_col;
                    if let Some(column) = self.bit_column(bit_index) {
                        let coefficient = rotation[coordinate];
                        if coefficient != Ff::ZERO {
                            visit(column, coefficient);
                        }
                    }
                    rotation = rotate_phi81(&rotation);
                }
            }
        }
    }

    fn for_each_original_term<Ff, Visit>(&self, visit: &mut Visit)
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        Visit: FnMut(usize, usize, Ff),
    {
        self.for_each_original_column_rotation::<Ff, _>(|output, column, rotation| {
            for (coordinate, &coefficient) in rotation.iter().enumerate() {
                if coefficient != Ff::ZERO {
                    visit(self.row_start + output * D + coordinate, column, coefficient);
                }
            }
        });
    }

    /// Visit each original seeded column contribution before the optional
    /// SuperNeo column transform.
    ///
    /// One callback represents all `D` output coordinates for a single
    /// logical input column. Consumers that need a matrix-vector or
    /// transpose product should use this surface instead of calling
    /// [`Self::for_each_row_term`] once per coordinate, which would regenerate
    /// the same deterministic coefficient stream `D` times.
    #[doc(hidden)]
    pub fn for_each_original_column_rotation<Ff, Visit>(&self, mut visit: Visit)
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        Visit: FnMut(usize, usize, [Ff; D]),
    {
        self.for_each_seeded_column::<Ff, _>(|output, message_col, mut rotation| {
            for message_row in 0..D {
                let bit_index = message_row * self.message_cols + message_col;
                if let Some(column) = self.bit_column(bit_index) {
                    visit(output, column, rotation);
                }
                rotation = rotate_phi81(&rotation);
            }
        });
    }

    /// Visit one independently seeded coefficient chunk for one output.
    ///
    /// Chunk seeds are independent public parameters, so callers may evaluate
    /// different `(output, chunk)` pairs in parallel and add their results.
    #[doc(hidden)]
    pub fn for_each_original_chunk_column_rotation<Ff, Visit>(&self, output: usize, chunk: usize, visit: Visit)
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        Visit: FnMut(usize, [Ff; D]),
    {
        let chunk_len = self.original_chunk_len(chunk);
        self.for_each_original_chunk_range_column_rotation(output, chunk, 0, chunk_len, visit);
    }

    /// Number of message columns generated by one protocol seed chunk.
    #[doc(hidden)]
    pub fn original_chunk_len(&self, chunk: usize) -> usize {
        let start = chunk * self.chunk_size;
        core::cmp::min(self.message_cols, start + self.chunk_size) - start
    }

    /// Visit a subrange of one independently seeded coefficient chunk.
    /// Random access preserves the exact ChaCha8 stream used by the full walk.
    #[doc(hidden)]
    pub fn for_each_original_chunk_range_column_rotation<Ff, Visit>(
        &self,
        output: usize,
        chunk: usize,
        local_start: usize,
        local_end: usize,
        mut visit: Visit,
    ) where
        Ff: Field + PrimeCharacteristicRing + Copy,
        Visit: FnMut(usize, [Ff; D]),
    {
        let chunk_len = self.original_chunk_len(chunk);
        assert!(
            local_start <= local_end && local_end <= chunk_len,
            "seeded chunk subrange"
        );
        let seed = self.chunk_seeds_by_row[output][chunk];
        let chunk_start = chunk * self.chunk_size;
        let mut rng = ChaCha8Rng::from_seed(seed);
        rng.set_word_pos((local_start * D * 2) as u128);
        for message_col in chunk_start + local_start..chunk_start + local_end {
            let mut rotation = sample_uniform_coefficients::<Ff>(&mut rng);
            for message_row in 0..D {
                let bit_index = message_row * self.message_cols + message_col;
                if let Some(column) = self.bit_column(bit_index) {
                    visit(column, rotation);
                }
                rotation = rotate_phi81(&rotation);
            }
        }
    }

    fn for_each_superneo_transformed_term<Ff, Visit>(&self, visit: &mut Visit)
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        Visit: FnMut(usize, usize, Ff),
    {
        let bar = neo_math::superneo_bar_matrix();
        for local_row in 0..D * self.kappa {
            let row = self.row_start + local_row;
            let mut blocks = std::collections::BTreeMap::<usize, [Ff; D]>::new();
            self.for_each_original_term::<Ff, _>(&mut |candidate_row, column, coefficient| {
                if candidate_row == row {
                    blocks.entry(column / D).or_insert([Ff::ZERO; D])[column % D] += coefficient;
                }
            });
            for (block, original) in blocks {
                for output_local in 0..D {
                    let mut coefficient = Ff::ZERO;
                    for input_local in 0..D {
                        coefficient +=
                            original[input_local] * Ff::from_u64(bar[output_local][input_local].as_canonical_u64());
                    }
                    if coefficient != Ff::ZERO {
                        visit(row, block * D + output_local, coefficient);
                    }
                }
            }
        }
    }

    /// Accumulate the block's contribution to `y += M*x`.
    pub fn add_mul_into<Ff, Kf>(&self, x: &[Kf], y: &mut [Kf], n_eff: usize)
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        Kf: Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + From<Ff>,
    {
        if self.superneo_transformed_columns {
            self.for_each_term::<Ff, _>(|row, column, coefficient| {
                if row < n_eff && row < y.len() {
                    y[row] += Kf::from(coefficient) * x[column];
                }
            });
            return;
        }
        self.for_each_seeded_column::<Ff, _>(|output, message_col, mut rotation| {
            for message_row in 0..D {
                let bit_index = message_row * self.message_cols + message_col;
                if let Some(column) = self.bit_column(bit_index) {
                    let value = x[column];
                    for (coordinate, &coefficient) in rotation.iter().enumerate() {
                        let row = self.row_start + output * D + coordinate;
                        if row < n_eff && row < y.len() && coefficient != Ff::ZERO {
                            y[row] += Kf::from(coefficient) * value;
                        }
                    }
                }
                rotation = rotate_phi81(&rotation);
            }
        });
    }

    /// Accumulate the block's contribution to `y += M^T*x`.
    pub fn add_mul_transpose_into<Ff, Kf>(&self, x: &[Kf], y: &mut [Kf], n_eff: usize)
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        Kf: Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + From<Ff>,
    {
        if self.superneo_transformed_columns {
            self.for_each_term::<Ff, _>(|row, column, coefficient| {
                if row < n_eff && row < x.len() {
                    y[column] += Kf::from(coefficient) * x[row];
                }
            });
            return;
        }
        self.for_each_seeded_column::<Ff, _>(|output, message_col, mut rotation| {
            for message_row in 0..D {
                let bit_index = message_row * self.message_cols + message_col;
                if let Some(column) = self.bit_column(bit_index) {
                    for (coordinate, &coefficient) in rotation.iter().enumerate() {
                        let row = self.row_start + output * D + coordinate;
                        if row < n_eff && row < x.len() && coefficient != Ff::ZERO {
                            y[column] += Kf::from(coefficient) * x[row];
                        }
                    }
                }
                rotation = rotate_phi81(&rotation);
            }
        });
    }

    /// Read one matrix entry, including duplicate input-word occurrences.
    pub fn entry<Ff>(&self, row: usize, col: usize) -> Ff
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
    {
        if row < self.row_start || row >= self.row_end() {
            return Ff::ZERO;
        }
        let mut value = Ff::ZERO;
        self.for_each_term::<Ff, _>(|candidate_row, candidate_col, coefficient| {
            if candidate_row == row && candidate_col == col {
                value += coefficient;
            }
        });
        value
    }

    fn bit_column(&self, bit_index: usize) -> Option<usize> {
        if bit_index >= self.word_starts.len() * self.word_width {
            return None;
        }
        Some(self.word_starts[bit_index / self.word_width] + bit_index % self.word_width)
    }

    fn for_each_seeded_column<Ff, Visit>(&self, mut visit: Visit)
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        Visit: FnMut(usize, usize, [Ff; D]),
    {
        for (output, seeds) in self.chunk_seeds_by_row.iter().enumerate() {
            for (chunk, &seed) in seeds.iter().enumerate() {
                let start = chunk * self.chunk_size;
                let end = core::cmp::min(self.message_cols, start + self.chunk_size);
                let mut rng = ChaCha8Rng::from_seed(seed);
                for message_col in start..end {
                    visit(output, message_col, sample_uniform_coefficients(&mut rng));
                }
            }
        }
    }
}

fn sample_uniform_coefficients<Ff>(rng: &mut ChaCha8Rng) -> [Ff; D]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut bytes = [0u8; D * 8];
    rng.fill_bytes(&mut bytes);
    core::array::from_fn(|index| {
        let start = index * 8;
        let sampled = u64::from_le_bytes(
            bytes[start..start + 8]
                .try_into()
                .expect("eight-byte coefficient"),
        );
        Ff::from_u64(if sampled < Goldilocks::ORDER_U64 {
            sampled
        } else {
            sample_uniform_goldilocks(rng)
        })
    })
}

fn sample_uniform_goldilocks(rng: &mut ChaCha8Rng) -> u64 {
    loop {
        let sampled = rng.next_u64();
        if sampled < Goldilocks::ORDER_U64 {
            return sampled;
        }
    }
}

fn rotate_phi81<Ff>(current: &[Ff; D]) -> [Ff; D]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let last = current[D - 1];
    let mut next = [Ff::ZERO; D];
    next[0] = -last;
    next[1..D].copy_from_slice(&current[..D - 1]);
    next[27] -= last;
    next
}
