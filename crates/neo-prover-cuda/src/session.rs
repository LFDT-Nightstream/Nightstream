//! CUDA context, stream, kernel modules, and device-buffer ownership; the
//! CUDA port of `neo-prover-metal`'s `session.rs`.
//!
//! Protocol phase ordering stays in the lifecycle. This layer owns kernel
//! launches and accounts for online-path CPU reads, writes, and waits. All
//! work goes to one stream, so launches run in order, as the encoders of
//! one Metal command buffer do; `finish` waits for them.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use cuda_core::{memory, CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};

use crate::kernels::Kernels;
use crate::{poseidon2, CudaActivity, CudaError, BUFFER_LIMIT_BYTES};

mod dft;
mod gkr;
mod joint;
mod masks;
mod merkle;
mod production_commitment;
mod rlc;
pub use gkr::CudaGkrLayer;
pub(crate) use joint::CudaPaperJointOracle;
pub(crate) use masks::CudaWitnessMasks;

/// Bytes that the buffers of every session of the process hold, and the
/// most that they held since the last `reset_peak_bytes`.
static RESIDENT: Mutex<(usize, usize)> = Mutex::new((0, 0));

/// Threads per block of a `Grid::Threads` launch: Metal's dispatch width.
const THREADS_PER_BLOCK: usize = 256;

/// The threads of one launch.
#[derive(Clone, Copy)]
pub(crate) enum Grid {
    /// Exactly this many threads, in blocks of `THREADS_PER_BLOCK`; the
    /// kernel ignores the threads past the count.
    Threads(usize),
    /// Whole blocks, with `shared` bytes of dynamic shared memory each.
    Groups {
        groups: usize,
        threads: usize,
        shared: usize,
    },
}

impl Grid {
    pub(crate) fn groups(groups: usize, threads: usize) -> Self {
        Self::Groups {
            groups,
            threads,
            shared: 0,
        }
    }

    fn config(self) -> Option<LaunchConfig> {
        let (blocks, threads, shared) = match self {
            Self::Threads(count) => (count.div_ceil(THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0),
            Self::Groups {
                groups,
                threads,
                shared,
            } => (groups, threads, shared),
        };
        (blocks != 0 && threads != 0).then(|| LaunchConfig {
            grid_dim: (u32::try_from(blocks).expect("CUDA grid fits u32"), 1, 1),
            block_dim: (u32::try_from(threads).expect("CUDA block fits u32"), 1, 1),
            shared_mem_bytes: u32::try_from(shared).expect("CUDA shared memory fits u32"),
        })
    }
}

/// A device allocation. It is zero-filled, as a new Metal buffer is, and
/// it leaves the process count when its last handle drops.
struct Allocation {
    memory: DeviceBuffer<u8>,
    bytes: usize,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        RESIDENT.lock().unwrap_or_else(PoisonError::into_inner).0 -= self.bytes;
    }
}

/// A shared handle to a device allocation, which kernels reach by address.
#[derive(Clone)]
pub(crate) struct Buffer(Arc<Allocation>);

impl Buffer {
    pub(crate) fn length(&self) -> usize {
        self.0.bytes
    }

    /// The device address of the first byte, as a kernel pointer.
    pub(crate) fn ptr<T>(&self) -> *mut T {
        self.0.memory.cu_deviceptr() as *mut T
    }

    /// The device address `offset` bytes into the buffer.
    pub(crate) fn at<T>(&self, offset: usize) -> *mut T {
        assert!(
            offset < self.0.bytes,
            "CUDA buffer offset {offset} is outside the buffer"
        );
        (self.0.memory.cu_deviceptr() + offset as u64) as *mut T
    }
}

#[derive(Default)]
struct ActivityCounters {
    launches: AtomicU64,
    host_waits: AtomicU64,
    allocated_bytes: AtomicU64,
    uploaded_bytes: AtomicU64,
    downloaded_bytes: AtomicU64,
}

pub struct CudaSession {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: Kernels,
    poseidon2_constants: Buffer,
    // A prover keeps the production key rows after their first commitment;
    // `None` expands the needed rows in every call.
    production_key: Option<Mutex<production_commitment::ResidentKeyRows>>,
    activity: ActivityCounters,
}

impl CudaSession {
    pub fn new() -> Result<Self, CudaError> {
        let device = |error: DriverError| CudaError::Device(error.to_string());
        let context = CudaContext::new(0).map_err(device)?;
        let stream = context.new_stream().map_err(device)?;
        let kernels = Kernels::load(&context).map_err(|error| CudaError::Module(error.to_string()))?;
        let mut session = Self {
            poseidon2_constants: allocate(&context, &stream, 0)?,
            context,
            stream,
            kernels,
            production_key: None,
            activity: ActivityCounters::default(),
        };
        session.poseidon2_constants = session.buffer_from_slice(&poseidon2::round_constant_words())?;
        Ok(session)
    }

    /// Keep the production key rows on the device after their first
    /// commitment. A prover commits at every step; a verifier commits once,
    /// so it streams the rows and holds no complete key.
    pub(crate) fn keep_production_key(&mut self) {
        self.production_key = Some(Mutex::default());
    }

    fn bind(&self) -> Result<(), CudaError> {
        self.context
            .bind_to_thread()
            .map_err(|error| CudaError::Execution(error.to_string()))
    }

    /// A zero-filled buffer of `bytes` bytes, inside the process ceiling.
    pub(crate) fn buffer(&self, bytes: usize) -> Result<Buffer, CudaError> {
        let buffer = allocate(&self.context, &self.stream, bytes)?;
        self.activity
            .allocated_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(buffer)
    }

    pub(crate) fn buffer_from_slice<T: Copy>(&self, values: &[T]) -> Result<Buffer, CudaError> {
        let buffer = self.buffer(size_of_val(values))?;
        self.write_shared(&buffer, values)?;
        Ok(buffer)
    }

    /// A buffer of `words` words that `fill` writes on the host.
    pub(crate) fn filled_buffer(
        &self,
        words: usize,
        fill: impl FnOnce(&mut [u64]) -> Result<(), CudaError>,
    ) -> Result<Buffer, CudaError> {
        words
            .checked_mul(size_of::<u64>())
            .ok_or(CudaError::Shape("CUDA buffer size overflows"))?;
        let mut host = vec![0u64; words];
        fill(&mut host)?;
        self.buffer_from_slice(&host)
    }

    /// Run `launch` (a generated kernel method) on the session stream over
    /// `grid`. An empty grid launches nothing.
    pub(crate) fn launch(
        &self,
        grid: Grid,
        launch: impl FnOnce(&CudaStream, LaunchConfig) -> Result<(), DriverError>,
    ) -> Result<(), CudaError> {
        let Some(config) = grid.config() else {
            return Ok(());
        };
        self.activity.launches.fetch_add(1, Ordering::Relaxed);
        launch(&self.stream, config).map_err(|error| CudaError::Execution(error.to_string()))
    }

    /// Copy `values` to the start of `buffer`.
    pub(crate) fn write_shared<T: Copy>(&self, buffer: &Buffer, values: &[T]) -> Result<(), CudaError> {
        self.upload(buffer.ptr::<u8>() as u64, buffer.length(), values)
    }

    /// Copy `values` to `address`, the start of `capacity` device bytes.
    fn upload<T: Copy>(&self, address: u64, capacity: usize, values: &[T]) -> Result<(), CudaError> {
        let bytes = size_of_val(values);
        if bytes > capacity {
            return Err(CudaError::Shape("CUDA buffer write exceeds capacity"));
        }
        if bytes == 0 {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: the source is a live slice of `bytes` bytes, and the
        // destination holds at least `bytes` bytes (checked above). The
        // stream is synchronized before the borrow of `values` ends.
        unsafe { memory::memcpy_htod_async(address, values.as_ptr(), bytes, self.stream.cu_stream()) }
            .map_err(|error| CudaError::Execution(error.to_string()))?;
        self.stream
            .synchronize()
            .map_err(|error| CudaError::Execution(error.to_string()))?;
        self.activity
            .uploaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Wait for every launch on the stream; a kernel fault is an
    /// execution error.
    pub(crate) fn finish(&self) -> Result<(), CudaError> {
        self.activity.host_waits.fetch_add(1, Ordering::Relaxed);
        self.stream
            .synchronize()
            .map_err(|error| CudaError::Execution(error.to_string()))
    }

    /// The first `len` values of `buffer`, after the stream finishes.
    pub(crate) fn read_buffer<T: Pod>(&self, buffer: &Buffer, len: usize) -> Result<Vec<T>, CudaError> {
        self.read_at(buffer, 0, len)
    }

    /// `len` values of `buffer` from byte `offset`.
    pub(crate) fn read_at<T: Pod>(&self, buffer: &Buffer, offset: usize, len: usize) -> Result<Vec<T>, CudaError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .filter(|&bytes| {
                offset
                    .checked_add(bytes)
                    .is_some_and(|end| end <= buffer.length())
            })
            .expect("a host read fits its CUDA buffer");
        let mut values = Vec::<T>::with_capacity(len);
        if bytes != 0 {
            self.bind()?;
            // SAFETY: the destination has capacity for `len` values, and the
            // source range lies inside the buffer (checked above). The copy
            // finishes before `set_len`, and every bit pattern is a `T`.
            unsafe {
                memory::memcpy_dtoh_async(
                    values.as_mut_ptr(),
                    buffer.ptr::<u8>() as u64 + offset as u64,
                    bytes,
                    self.stream.cu_stream(),
                )
            }
            .map_err(|error| CudaError::Execution(error.to_string()))?;
            self.finish()?;
        }
        // SAFETY: the copy above wrote all `len` values.
        unsafe { values.set_len(len) };
        self.activity
            .downloaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(values)
    }

    /// Copy `len` values of `buffer` from byte `offset` into `values`.
    pub(crate) fn read_into<T: Pod>(&self, buffer: &Buffer, offset: usize, values: &mut [T]) -> Result<(), CudaError> {
        let bytes = size_of_val(values);
        assert!(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= buffer.length()),
            "a host read fits its CUDA buffer"
        );
        if bytes == 0 {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: the destination is a live slice of `bytes` bytes, the
        // source range lies inside the buffer (checked above), and the
        // stream finishes before the borrow ends.
        unsafe {
            memory::memcpy_dtoh_async(
                values.as_mut_ptr(),
                buffer.ptr::<u8>() as u64 + offset as u64,
                bytes,
                self.stream.cu_stream(),
            )
        }
        .map_err(|error| CudaError::Execution(error.to_string()))?;
        self.finish()?;
        self.activity
            .downloaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    pub fn activity(&self) -> CudaActivity {
        CudaActivity {
            launches: self.activity.launches.load(Ordering::Relaxed),
            host_waits: self.activity.host_waits.load(Ordering::Relaxed),
            allocated_bytes: self.activity.allocated_bytes.load(Ordering::Relaxed),
            uploaded_bytes: self.activity.uploaded_bytes.load(Ordering::Relaxed),
            downloaded_bytes: self.activity.downloaded_bytes.load(Ordering::Relaxed),
        }
    }

    /// Bytes that the buffers of every session of the process hold now.
    pub fn current_allocated_bytes() -> usize {
        RESIDENT.lock().unwrap_or_else(PoisonError::into_inner).0
    }

    /// The most bytes that the buffers of every session held at one time
    /// since the last reset.
    pub fn peak_allocated_bytes() -> usize {
        RESIDENT.lock().unwrap_or_else(PoisonError::into_inner).1
    }

    /// Start a new peak measurement from the bytes held now.
    pub fn reset_peak_allocated_bytes() {
        let mut resident = RESIDENT.lock().unwrap_or_else(PoisonError::into_inner);
        resident.1 = resident.0;
    }
}

/// A zero-filled device buffer of `bytes` bytes, counted against
/// `BUFFER_LIMIT_BYTES` before it is allocated: every device allocation of
/// the crate comes from here.
fn allocate(context: &CudaContext, stream: &CudaStream, bytes: usize) -> Result<Buffer, CudaError> {
    {
        let mut resident = RESIDENT.lock().unwrap_or_else(PoisonError::into_inner);
        let allocated = resident.0;
        match allocated.checked_add(bytes) {
            Some(total) if total <= BUFFER_LIMIT_BYTES => {
                resident.0 = total;
                resident.1 = resident.1.max(total);
            }
            _ => {
                return Err(CudaError::MemoryLimit {
                    requested: bytes,
                    allocated,
                    limit: BUFFER_LIMIT_BYTES,
                })
            }
        }
    }
    let release = || RESIDENT.lock().unwrap_or_else(PoisonError::into_inner).0 -= bytes;
    if let Err(error) = context.bind_to_thread() {
        release();
        return Err(CudaError::Execution(error.to_string()));
    }
    match DeviceBuffer::<u8>::zeroed(stream, bytes) {
        Ok(memory) => Ok(Buffer(Arc::new(Allocation { memory, bytes }))),
        Err(_) => {
            release();
            Err(CudaError::Buffer { bytes })
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/buffer_limit.rs"]
mod buffer_limit_tests;

/// A type that device bytes can be read as: it has no padding, and every bit
/// pattern is a value.
///
/// # Safety
///
/// Implement it only for such types.
pub(crate) unsafe trait Pod: Copy + Send + Sync {}

// SAFETY: an integer has no padding, and every bit pattern is a value.
unsafe impl Pod for u8 {}
// SAFETY: an integer has no padding, and every bit pattern is a value.
unsafe impl Pod for u16 {}
// SAFETY: an integer has no padding, and every bit pattern is a value.
unsafe impl Pod for u32 {}
// SAFETY: an integer has no padding, and every bit pattern is a value.
unsafe impl Pod for u64 {}
// SAFETY: an array has no padding between its elements, so an array of
// `Pod` values has no padding, and every bit pattern is a value.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}
