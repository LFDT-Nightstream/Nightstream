//! Device context and stream ownership. Nothing protocol-aware lives here.

use std::sync::Arc;

use cuda_core::{memory, CudaContext, CudaStream, DeviceBuffer, DriverError};

/// One opened GPU: a context plus the streams the prover owns.
pub struct Device {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl Device {
    /// Open GPU 0.
    pub fn open() -> Result<Self, DriverError> {
        Self::from_context(Self::open_context()?)
    }

    /// Open GPU 0's CUDA context without choosing a stream yet.
    pub fn open_context() -> Result<Arc<CudaContext>, DriverError> {
        CudaContext::new(0)
    }

    /// Create an independent stream in an existing context.
    pub fn from_context(ctx: Arc<CudaContext>) -> Result<Self, DriverError> {
        let stream = ctx.new_stream()?;
        Ok(Self { ctx, stream })
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Block the host until all queued work on the prover stream finished.
    pub fn sync(&self) -> Result<(), DriverError> {
        self.stream.synchronize()
    }
}

pub(crate) fn copy_host_to_device<T>(stream: &CudaStream, dst: &DeviceBuffer<T>, src: &[T]) -> Result<(), DriverError> {
    assert!(
        src.len() <= dst.len(),
        "destination device buffer too small: {} < {}",
        dst.len(),
        src.len()
    );
    if src.is_empty() {
        return Ok(());
    }
    stream.context().bind_to_thread()?;
    unsafe {
        memory::memcpy_htod_async(
            dst.cu_deviceptr(),
            src.as_ptr(),
            std::mem::size_of_val(src),
            stream.cu_stream(),
        )
    }
}

/// Allocate a `u64` device buffer without initializing it.
///
/// Use only when the next queued kernels write every element read by later
/// kernels. This avoids turning resident scratch reuse into repeated device
/// memsets in hot prover paths.
pub(crate) fn uninit_u64_device_buffer(stream: &Arc<CudaStream>, len: usize) -> Result<DeviceBuffer<u64>, DriverError> {
    // SAFETY: callers uphold this helper's full-write-before-read contract.
    // The buffer retains `stream`, so Drop releases the allocation in stream
    // order instead of serializing all concurrent prover streams in
    // `cuMemFree`.
    unsafe { DeviceBuffer::uninitialized_async(stream, len) }
}

/// Upload borrowed host words into a stream-ordered allocation.
///
/// The copy still synchronizes this stream so the borrowed slice can be
/// released safely. Unlike `DeviceBuffer::from_host`, allocation and drop do
/// not serialize every independent prover stream through `cuMemAlloc` /
/// `cuMemFree`.
pub(crate) fn upload_u64_device_buffer(
    stream: &Arc<CudaStream>,
    words: &[u64],
) -> Result<DeviceBuffer<u64>, DriverError> {
    let mut buffer = uninit_u64_device_buffer(stream, words.len())?;
    buffer.copy_from_host(stream, words)?;
    Ok(buffer)
}

/// Allocate zeroed words without using the context-wide synchronous
/// allocator. The memset is ordered before later work on `stream`.
pub(crate) fn zeroed_u64_device_buffer(stream: &Arc<CudaStream>, len: usize) -> Result<DeviceBuffer<u64>, DriverError> {
    let buffer = uninit_u64_device_buffer(stream, len)?;
    if buffer.num_bytes() != 0 {
        unsafe {
            memory::memset_d8_async(buffer.cu_deviceptr(), 0, buffer.num_bytes(), stream.cu_stream())?;
        }
    }
    Ok(buffer)
}
