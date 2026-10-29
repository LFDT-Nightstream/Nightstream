//! The DFT of the columns of a row-major matrix, in place when the memory
//! spans whole pages: the layer-1 prover's Reed-Solomon encodings.

use std::marker::PhantomData;

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLResourceOptions};

use super::{host_view, host_view_mut, Arg, Buffer, Grid, MetalSession};
use crate::MetalError;

/// A no-copy buffer over caller memory. It holds the borrow `B` of that
/// memory (`&mut [u64]`, or `&[u64]` when kernels only read it) for as long
/// as the buffer lives. Never retain the buffer: a retained handle would
/// outlive the borrow.
pub(super) struct WrappedPages<B> {
    buffer: Buffer,
    borrow: PhantomData<B>,
}

impl<B> std::ops::Deref for WrappedPages<B> {
    type Target = ProtocolObject<dyn MTLBuffer>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

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
        let (wrapped, copied) = match self.wrap_pages(values) {
            Some(pages) => (Some(pages), None),
            None => {
                let mut data = self.buffer(size_of_val(values))?;
                host_view_mut(&mut data, values.len())
                    .par_chunks_mut(width)
                    .zip(values.par_chunks(width))
                    .for_each(|(to, from)| to.copy_from_slice(from));
                (None, Some(data))
            }
        };
        let data = wrapped
            .as_deref()
            .or(copied.as_deref())
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
        self.launch(
            &encoder,
            &self.dft_twiddles,
            &[Arg::Buffer(0, &twiddles), Arg::Buffer(1, &twiddle_shape)],
            Grid::Threads(rows / 2),
        );
        // Serial dispatches: each pass reads what the one before wrote.
        let chunks = width / width.min(COLUMNS);
        for (shape, stages) in &passes {
            self.launch(
                &encoder,
                &self.dft_dif_pass,
                &[Arg::Buffer(0, data), Arg::Buffer(1, &twiddles), Arg::Buffer(2, shape)],
                Grid::Groups {
                    groups: (rows >> stages) * chunks,
                    threads: THREADS,
                },
            );
        }
        self.launch(
            &encoder,
            &self.dft_bit_reverse,
            &[Arg::Buffer(0, data), Arg::Buffer(1, &reverse_shape)],
            Grid::Threads(rows),
        );
        encoder.endEncoding();
        self.finish(&command)?;
        drop(wrapped);
        if let Some(data) = &copied {
            let words = host_view::<u64>(data, values.len());
            values
                .par_chunks_mut(width)
                .zip(words.par_chunks(width))
                .for_each(|(to, from)| to.copy_from_slice(from));
        }
        Ok(())
    }

    /// `values` as a shared buffer over the same memory, when it starts on a
    /// page and spans whole pages (16 KiB, the Apple silicon VM page).
    pub(super) fn wrap_pages<'a>(&self, values: &'a mut [u64]) -> Option<WrappedPages<&'a mut [u64]>> {
        self.wrap(values.as_mut_ptr(), size_of_val(values))
    }

    /// `wrap_pages` for memory that kernels only read.
    pub(super) fn wrap_read_pages<'a>(&self, values: &'a [u64]) -> Option<WrappedPages<&'a [u64]>> {
        self.wrap(values.as_ptr().cast_mut(), size_of_val(values))
    }

    fn wrap<B>(&self, words: *mut u64, bytes: usize) -> Option<WrappedPages<B>> {
        const PAGE: usize = 16 * 1024;
        let pointer = std::ptr::NonNull::new(words.cast::<std::ffi::c_void>())?;
        if !(pointer.as_ptr() as usize).is_multiple_of(PAGE) || !bytes.is_multiple_of(PAGE) {
            return None;
        }
        // SAFETY: `pointer` and `bytes` come from one live Rust slice (the
        // two callers above), start on a page and span whole pages, as Metal
        // requires. With no deallocator, Metal never frees that memory. The
        // returned guard holds the slice's borrow for as long as Rust holds
        // the buffer, and no Rust code retains the buffer past the guard. A
        // command buffer that retains it completes inside the call that
        // encodes it (see `launch`), so no kernel touches the memory after
        // the borrow ends. A buffer from `wrap_read_pages` is bound only to
        // kernel arguments that the shaders declare `device const`, so no
        // kernel writes memory that Rust lends as shared.
        let buffer = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pointer,
                    bytes,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }?;
        Some(WrappedPages {
            buffer,
            borrow: PhantomData,
        })
    }
}
