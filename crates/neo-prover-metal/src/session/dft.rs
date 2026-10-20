//! The DFT of the columns of a row-major matrix, in place when the memory
//! spans whole pages: the layer-1 prover's Reed-Solomon encodings.

use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions,
};

use super::{Buffer, MetalSession};
use crate::MetalError;

impl MetalSession {
    /// The DFT of every column of the row-major matrix `values` of `width`
    /// columns, in place: row `k` becomes `Σ_i row_i·ω^{ik}` for `ω` =
    /// `generator`, of order the row count (a power of two). Inputs may be
    /// non-canonical; outputs are canonical. A width above 16 must be a
    /// multiple of 16.
    pub fn dft_columns(&self, values: &mut [u64], width: usize, generator: u64) -> Result<(), MetalError> {
        use rayon::prelude::*;
        const PASS_BITS: u32 = 8;
        const COLUMNS: usize = 16;
        const THREADS: usize = 256;
        if width == 0 || !values.len().is_multiple_of(width) || (width > COLUMNS && !width.is_multiple_of(COLUMNS)) {
            return Err(MetalError::Shape("a DFT needs whole rows of a supported width"));
        }
        let rows = values.len() / width;
        if !rows.is_power_of_two() || rows > 1 << 32 {
            return Err(MetalError::Shape("a DFT needs a power-of-two row count"));
        }
        if rows == 1 {
            return Ok(());
        }
        let log_rows = rows.trailing_zeros();
        // Page-aligned memory is used in place; otherwise it is copied.
        let wrapped = self.wrap_pages(values);
        let copied = match wrapped {
            Some(_) => None,
            None => {
                let data = self.buffer(size_of_val(values))?;
                let words =
                    unsafe { std::slice::from_raw_parts_mut(data.contents().as_ptr().cast::<u64>(), values.len()) };
                words
                    .par_chunks_mut(width)
                    .zip(values.par_chunks(width))
                    .for_each(|(to, from)| to.copy_from_slice(from));
                Some(data)
            }
        };
        let data = wrapped
            .as_ref()
            .or(copied.as_ref())
            .expect("one of the two buffers");
        let twiddles = self.buffer(rows / 2 * size_of::<u64>())?;
        let twiddle_shape = self.buffer_from_slice(&[generator])?;
        let reverse_shape = self.buffer_from_slice(&[u64::from(log_rows), width as u64])?;
        // The fewest passes of at most PASS_BITS stages, split evenly: a
        // short pass would launch one threadgroup per two rows.
        let count = log_rows.div_ceil(PASS_BITS);
        let mut first = 0;
        let passes = (0..count)
            .map(|pass| {
                let stages = log_rows / count + u32::from(pass < log_rows % count);
                let shape = [u64::from(log_rows), width as u64, u64::from(first), u64::from(stages)];
                first += stages;
                Ok((self.buffer_from_slice(&shape)?, stages))
            })
            .collect::<Result<Vec<_>, MetalError>>()?;
        let command = self.command_buffer("nightstream.dft_columns")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.dft_twiddles);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&twiddles), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&twiddle_shape), 0, 1);
        }
        self.dispatch(&encoder, &self.dft_twiddles, rows / 2);
        // Serial dispatches: each pass reads what the one before wrote.
        encoder.setComputePipelineState(&self.dft_dif_pass);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(data), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&twiddles), 0, 1);
        }
        let chunks = width / width.min(COLUMNS);
        for (shape, stages) in &passes {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(shape), 0, 2);
            }
            self.dispatch_threadgroups(&encoder, &self.dft_dif_pass, (rows >> stages) * chunks, THREADS);
        }
        encoder.setComputePipelineState(&self.dft_bit_reverse);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(data), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&reverse_shape), 0, 1);
        }
        self.dispatch(&encoder, &self.dft_bit_reverse, rows);
        encoder.endEncoding();
        self.finish(&command)?;
        if let Some(data) = &copied {
            let words = unsafe { std::slice::from_raw_parts(data.contents().as_ptr().cast::<u64>(), values.len()) };
            values
                .par_chunks_mut(width)
                .zip(words.par_chunks(width))
                .for_each(|(to, from)| to.copy_from_slice(from));
        }
        Ok(())
    }

    /// `values` as a shared buffer over the same memory, when it starts on a
    /// page and spans whole pages (16 KiB, the Apple silicon VM page). The
    /// buffer must be dropped before `values` is used again.
    pub(super) fn wrap_pages(&self, values: &mut [u64]) -> Option<Buffer> {
        self.wrap(values.as_mut_ptr(), size_of_val(values))
    }

    /// `wrap_pages` for memory that kernels only read.
    pub(super) fn wrap_read_pages(&self, values: &[u64]) -> Option<Buffer> {
        self.wrap(values.as_ptr().cast_mut(), size_of_val(values))
    }

    fn wrap(&self, words: *mut u64, bytes: usize) -> Option<Buffer> {
        const PAGE: usize = 16 * 1024;
        let pointer = std::ptr::NonNull::new(words.cast::<std::ffi::c_void>())?;
        if !(pointer.as_ptr() as usize).is_multiple_of(PAGE) || !bytes.is_multiple_of(PAGE) {
            return None;
        }
        unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pointer,
                    bytes,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
    }
}
