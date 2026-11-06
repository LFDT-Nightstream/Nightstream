//! The DFT of the columns of a row-major matrix: the layer-1 prover's
//! Reed-Solomon encodings.

use super::{CudaSession, Grid};
use crate::kernels::dft::{DFT_COLUMNS, DFT_PASS_BITS};
use crate::CudaError;

/// Threads per pass block; the Metal threadgroup width.
const THREADS: usize = 256;

impl CudaSession {
    /// The DFT of every column of the row-major matrix `values` of `width`
    /// columns, in place: row `k` becomes `Σ_i row_i·ω^{ik}` for `ω` =
    /// `generator`, of order the row count (a power of two). Inputs may be
    /// non-canonical; outputs are canonical. A width above 16 must be a
    /// multiple of 16. Only the copy back writes `values`, so an error
    /// before it leaves `values` unchanged.
    pub fn dft_columns(&self, values: &mut [u64], width: usize, generator: u64) -> Result<(), CudaError> {
        if width == 0
            || !values.len().is_multiple_of(width)
            || (width > DFT_COLUMNS && !width.is_multiple_of(DFT_COLUMNS))
        {
            return Err(CudaError::Shape("a DFT needs whole rows of a supported width"));
        }
        let rows = values.len() / width;
        if !rows.is_power_of_two() || rows > 1 << 32 {
            return Err(CudaError::Shape("a DFT needs a power-of-two row count"));
        }
        if rows == 1 {
            return Ok(());
        }
        let log_rows = rows.trailing_zeros();
        let data = self.buffer_from_slice(values)?;
        let twiddles = self.buffer(rows / 2 * size_of::<u64>())?;
        let kernels = &self.kernels.dft;
        self.launch(Grid::Threads(rows / 2), |stream, config| {
            kernels.dft_twiddles(stream, config, twiddles.ptr(), generator, (rows / 2) as u64)
        })?;
        // The fewest passes of at most DFT_PASS_BITS stages, split evenly: a
        // short pass would launch one block per two rows. Launches in order:
        // each pass reads what the one before wrote.
        let count = log_rows.div_ceil(DFT_PASS_BITS);
        let chunks = width / width.min(DFT_COLUMNS);
        let mut first = 0;
        for pass in 0..count {
            let stages = log_rows / count + u32::from(pass < log_rows % count);
            let shape = [u64::from(log_rows), width as u64, u64::from(first), u64::from(stages)];
            first += stages;
            self.launch(Grid::groups((rows >> stages) * chunks, THREADS), |stream, config| {
                kernels.dft_dif_pass(stream, config, data.ptr(), twiddles.ptr(), shape)
            })?;
        }
        self.launch(Grid::Threads(rows), |stream, config| {
            kernels.dft_bit_reverse(stream, config, data.ptr(), u64::from(log_rows), width as u64)
        })?;
        self.finish()?;
        self.read_into(&data, 0, values)
    }
}
