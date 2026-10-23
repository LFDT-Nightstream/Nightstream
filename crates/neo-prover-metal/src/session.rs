//! Metal device, pipeline, queue, and shared-buffer ownership.
//!
//! Protocol phase ordering stays in the lifecycle. This layer owns command
//! encoding and accounts for online-path CPU reads, writes, and waits.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::{poseidon2, MetalActivity, MetalError, BUFFER_LIMIT_BYTES};

mod dft;
mod gkr;
mod joint;
mod merkle;
pub use gkr::MetalGkrLayer;
mod masks;
mod production_commitment;
mod rlc;
pub(crate) use joint::MetalPaperJointOracle;
pub(crate) use masks::MetalWitnessMasks;

static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nightstream-metal.metallib"));
type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

static BUFFER_ALLOCATION: Mutex<()> = Mutex::new(());

/// Entries in a Metal kernel's buffer table, and in its threadgroup memory
/// table (Apple's Metal feature set tables).
const ARGUMENT_SLOTS: usize = 31;

/// One kernel argument, at the index that the shader signature gives it.
#[derive(Clone, Copy)]
enum Arg<'a> {
    /// `[[buffer(index)]]` from the buffer's first byte.
    Buffer(usize, &'a ProtocolObject<dyn MTLBuffer>),
    /// `[[buffer(index)]]` from a byte offset.
    BufferAt(usize, &'a ProtocolObject<dyn MTLBuffer>, usize),
    /// `[[buffer(index)]]` as words that Metal copies into the command.
    Words(usize, &'a [u64]),
    /// `[[threadgroup(index)]]` memory of a byte length.
    Threadgroup(usize, usize),
}

/// The threads of one dispatch.
#[derive(Clone, Copy)]
enum Grid {
    /// Exactly this many threads, in threadgroups of the pipeline's width
    /// (at most 256).
    Threads(usize),
    /// Whole threadgroups.
    Groups { groups: usize, threads: usize },
}

#[derive(Default)]
struct ActivityCounters {
    command_buffers: AtomicU64,
    dispatches: AtomicU64,
    host_waits: AtomicU64,
    allocated_bytes: AtomicU64,
    uploaded_bytes: AtomicU64,
    downloaded_bytes: AtomicU64,
}

pub struct MetalSession {
    device: Device,
    queue: Queue,
    poseidon2_hash_rows_overwrite: Pipeline,
    poseidon2_constants: Buffer,
    production_key_row: Pipeline,
    production_key_weights: Pipeline,
    dft_twiddles: Pipeline,
    dft_dif_pass: Pipeline,
    dft_bit_reverse: Pipeline,
    gkr_message: Pipeline,
    gkr_fold: Pipeline,
    production_ajtai_accumulate: Pipeline,
    production_ajtai_sum_groups: Pipeline,
    tensor_point_expand_k: Pipeline,
    sumcheck_reduce_partials: Pipeline,
    joint_build_application_tables: Pipeline,
    ccs_first_unsatisfied_row: Pipeline,
    joint_zero_words: Pipeline,
    joint_fold_base_tables: Pipeline,
    joint_fold_k_tables: Pipeline,
    joint_accumulate_application: Pipeline,
    joint_fold_assignments: Pipeline,
    joint_fold_assignment_values: Pipeline,
    joint_round_partials: Pipeline,
    dec_build_row_weights: Pipeline,
    dec_build_ring_forms: Pipeline,
    dec_build_parallel_original_forms: Pipeline,
    dec_build_parallel_original_form_tiles: Pipeline,
    dec_reduce_parallel_original_form_tiles: Pipeline,
    dec_geometric_span_weights: Pipeline,
    dec_add_geometric_span_forms: Pipeline,
    dec_bar_ring_forms_in_place: Pipeline,
    rlc_witness_mix_signed_masks: Pipeline,
    dec_split_base2_masks: Pipeline,
    dec_sparse_ring_partials: Pipeline,
    dec_sparse_ring_sum_chunks: Pipeline,
    joint_carried_projection: Pipeline,
    fe_weighted_row_table: Pipeline,
    dec_ring_reduce_phi81: Pipeline,
    // A prover keeps the production key rows after their first commitment;
    // `None` expands the needed rows in every call.
    production_key: Option<Mutex<production_commitment::ResidentKeyRows>>,
    activity: ActivityCounters,
}

// SAFETY: the session owns a device, command queues, pipelines and buffers,
// which Apple documents as safe to retain, release and use from any thread;
// its other state is atomics and a mutex. Every command buffer and encoder
// is created, used and dropped inside one call. So the session may move
// between threads; callers that share it still serialize calls (a mutex).
unsafe impl Send for MetalSession {}

impl MetalSession {
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::Device)?;
        let queue = device.newCommandQueue().ok_or(MetalError::Queue)?;
        queue.setLabel(Some(&NSString::from_str("nightstream.compute")));
        let data = DispatchData::from_static_bytes(METALLIB);
        let library = device
            .newLibraryWithData_error(&data)
            .map_err(|error| MetalError::Library(format!("{error:?}")))?;
        let poseidon2_hash_rows_overwrite = pipeline(&device, &library, "poseidon2_hash_rows_overwrite")?;
        let poseidon2_constants = pre_session_buffer(&device, &poseidon2::round_constant_words())?;
        let production_key_row = pipeline(&device, &library, "production_key_row")?;
        let production_key_weights = pipeline(&device, &library, "production_key_weights")?;
        let dft_twiddles = pipeline(&device, &library, "dft_twiddles")?;
        let dft_dif_pass = pipeline(&device, &library, "dft_dif_pass")?;
        let dft_bit_reverse = pipeline(&device, &library, "dft_bit_reverse")?;
        let gkr_message = pipeline(&device, &library, "gkr_message")?;
        let gkr_fold = pipeline(&device, &library, "gkr_fold")?;
        let production_ajtai_accumulate = pipeline(&device, &library, "production_ajtai_accumulate")?;
        let production_ajtai_sum_groups = pipeline(&device, &library, "production_ajtai_sum_groups")?;
        let tensor_point_expand_k = pipeline(&device, &library, "tensor_point_expand_k")?;
        let sumcheck_reduce_partials = pipeline(&device, &library, "sumcheck_reduce_partials")?;
        let joint_build_application_tables = pipeline(&device, &library, "joint_build_application_tables")?;
        let ccs_first_unsatisfied_row = pipeline(&device, &library, "ccs_first_unsatisfied_row")?;
        let joint_zero_words = pipeline(&device, &library, "joint_zero_words")?;
        let joint_fold_base_tables = pipeline(&device, &library, "joint_fold_base_tables")?;
        let joint_fold_k_tables = pipeline(&device, &library, "joint_fold_k_tables")?;
        let joint_accumulate_application = pipeline(&device, &library, "joint_accumulate_application")?;
        let joint_fold_assignments = pipeline(&device, &library, "joint_fold_assignments")?;
        let joint_fold_assignment_values = pipeline(&device, &library, "joint_fold_assignment_values")?;
        let joint_round_partials = pipeline(&device, &library, "joint_round_partials")?;
        let dec_build_row_weights = pipeline(&device, &library, "dec_build_row_weights")?;
        let dec_build_ring_forms = pipeline(&device, &library, "dec_build_ring_forms")?;
        let dec_build_parallel_original_forms = pipeline(&device, &library, "dec_build_parallel_original_forms")?;
        let dec_build_parallel_original_form_tiles =
            pipeline(&device, &library, "dec_build_parallel_original_form_tiles")?;
        let dec_reduce_parallel_original_form_tiles =
            pipeline(&device, &library, "dec_reduce_parallel_original_form_tiles")?;
        let dec_geometric_span_weights = pipeline(&device, &library, "dec_geometric_span_weights")?;
        let dec_add_geometric_span_forms = pipeline(&device, &library, "dec_add_geometric_span_forms")?;
        let dec_bar_ring_forms_in_place = pipeline(&device, &library, "dec_bar_ring_forms_in_place")?;
        let rlc_witness_mix_signed_masks = pipeline(&device, &library, "rlc_witness_mix_signed_masks")?;
        let dec_split_base2_masks = pipeline(&device, &library, "dec_split_base2_masks")?;
        let dec_sparse_ring_partials = pipeline(&device, &library, "dec_sparse_ring_partials")?;
        let dec_sparse_ring_sum_chunks = pipeline(&device, &library, "dec_sparse_ring_sum_chunks")?;
        let joint_carried_projection = pipeline(&device, &library, "joint_carried_projection")?;
        let fe_weighted_row_table = pipeline(&device, &library, "fe_weighted_row_table")?;
        let dec_ring_reduce_phi81 = pipeline(&device, &library, "dec_ring_reduce_phi81")?;
        Ok(Self {
            device,
            queue,
            poseidon2_hash_rows_overwrite,
            poseidon2_constants,
            production_key_row,
            production_key_weights,
            dft_twiddles,
            dft_dif_pass,
            dft_bit_reverse,
            gkr_message,
            gkr_fold,
            production_ajtai_accumulate,
            production_ajtai_sum_groups,
            tensor_point_expand_k,
            sumcheck_reduce_partials,
            joint_build_application_tables,
            ccs_first_unsatisfied_row,
            joint_zero_words,
            joint_fold_base_tables,
            joint_fold_k_tables,
            joint_accumulate_application,
            joint_fold_assignments,
            joint_fold_assignment_values,
            joint_round_partials,
            dec_build_row_weights,
            dec_build_ring_forms,
            dec_build_parallel_original_forms,
            dec_build_parallel_original_form_tiles,
            dec_reduce_parallel_original_form_tiles,
            dec_geometric_span_weights,
            dec_add_geometric_span_forms,
            dec_bar_ring_forms_in_place,
            rlc_witness_mix_signed_masks,
            dec_split_base2_masks,
            dec_sparse_ring_partials,
            dec_sparse_ring_sum_chunks,
            joint_carried_projection,
            fe_weighted_row_table,
            dec_ring_reduce_phi81,
            production_key: None,
            activity: ActivityCounters::default(),
        })
    }

    /// Keep the production key rows on the device after their first
    /// commitment. A prover commits at every step; a verifier commits once,
    /// so it streams the rows and holds no complete key.
    pub(crate) fn keep_production_key(&mut self) {
        self.production_key = Some(Mutex::default());
    }

    fn buffer(&self, bytes: usize) -> Result<Buffer, MetalError> {
        let _allocation = BUFFER_ALLOCATION
            .lock()
            .map_err(|_| MetalError::Execution("buffer allocation lock was poisoned".into()))?;
        let allocated = self.device.currentAllocatedSize();
        if allocated
            .checked_add(bytes)
            .is_none_or(|total| total > BUFFER_LIMIT_BYTES)
        {
            return Err(MetalError::MemoryLimit {
                requested: bytes,
                allocated,
                limit: BUFFER_LIMIT_BYTES,
            });
        }
        // Shared storage is required for explicit CPU boundaries on Apple unified memory.
        let buffer = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::Buffer { bytes })?;
        self.activity
            .allocated_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(buffer)
    }

    fn buffer_from_slice<T: Copy>(&self, values: &[T]) -> Result<Buffer, MetalError> {
        let buffer = self.buffer(size_of_val(values))?;
        self.write_shared(&buffer, values)?;
        Ok(buffer)
    }

    fn filled_buffer(
        &self,
        words: usize,
        fill: impl FnOnce(&mut [u64]) -> Result<(), MetalError>,
    ) -> Result<Buffer, MetalError> {
        let bytes = words
            .checked_mul(size_of::<u64>())
            .ok_or(MetalError::Shape("Metal buffer size overflows"))?;
        let mut buffer = self.buffer(bytes)?;
        fill(host_view_mut(&mut buffer, words))?;
        self.activity
            .uploaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(buffer)
    }

    fn command_buffer(&self, label: &str) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, MetalError> {
        let command = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandBuffer)?;
        command.setLabel(Some(&NSString::from_str(label)));
        self.activity
            .command_buffers
            .fetch_add(1, Ordering::Relaxed);
        Ok(command)
    }

    /// Bind `args` and dispatch `pipeline` over `grid`. This is the only
    /// place that binds kernel arguments.
    fn launch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        args: &[Arg<'_>],
        grid: Grid,
    ) {
        let slot = |index: usize| {
            assert!(
                index < ARGUMENT_SLOTS,
                "Metal argument index {index} is outside the table"
            );
            index
        };
        encoder.setComputePipelineState(pipeline);
        for &arg in args {
            // SAFETY: `slot` asserts that each index is inside the kernel's
            // argument table, and a `BufferAt` offset is asserted to be inside
            // its buffer. `words` is a live slice of `size_of_val(words)`
            // bytes, and Metal copies it before `setBytes` returns. A bound
            // buffer lives until its command completes: the command buffer
            // (from `commandBuffer`) retains it, and every caller commits and
            // waits (`finish`) before it returns. The session is not `Sync`,
            // so no host code reads or writes `contents()` while a kernel
            // runs, and Metal's default hazard tracking orders the dispatches
            // that share a buffer. Each caller owns the rest. For `grid` and
            // the shape words that it binds, the kernel reads and writes only
            // inside the bound buffers (from their offsets) and inside the
            // buffers that the caller made resident with `useResource`. The
            // kernel reads each slot as the element type that the caller
            // stored there, at an offset aligned to that type. A `Threadgroup`
            // length is a multiple of 16 bytes that fits beside the
            // pipeline's static threadgroup memory.
            unsafe {
                match arg {
                    Arg::Buffer(index, buffer) => encoder.setBuffer_offset_atIndex(Some(buffer), 0, slot(index)),
                    Arg::BufferAt(index, buffer, offset) => {
                        assert!(
                            offset < buffer.length(),
                            "Metal buffer offset {offset} is outside the buffer"
                        );
                        encoder.setBuffer_offset_atIndex(Some(buffer), offset, slot(index));
                    }
                    Arg::Words(index, words) => encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::from(words).cast(),
                        size_of_val(words),
                        slot(index),
                    ),
                    Arg::Threadgroup(index, bytes) => encoder.setThreadgroupMemoryLength_atIndex(bytes, slot(index)),
                }
            }
        }
        self.activity.dispatches.fetch_add(1, Ordering::Relaxed);
        let line = |width| MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        match grid {
            Grid::Threads(threads) => encoder.dispatchThreads_threadsPerThreadgroup(
                line(threads),
                line(pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256)),
            ),
            Grid::Groups { groups, threads } => {
                encoder.dispatchThreadgroups_threadsPerThreadgroup(line(groups), line(threads))
            }
        }
    }

    fn write_shared<T: Copy>(&self, buffer: &ProtocolObject<dyn MTLBuffer>, values: &[T]) -> Result<(), MetalError> {
        upload(buffer, values)?;
        self.activity
            .uploaded_bytes
            .fetch_add(size_of_val(values) as u64, Ordering::Relaxed);
        Ok(())
    }

    fn finish(&self, command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
        self.activity.host_waits.fetch_add(1, Ordering::Relaxed);
        finish(command)
    }

    /// `read_buffer` with the copy split over the CPU cores.
    fn read_buffer_parallel<T: Pod>(&self, buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<T> {
        use rayon::prelude::*;
        let values = host_view::<T>(buffer, len);
        self.activity
            .downloaded_bytes
            .fetch_add(size_of_val(values) as u64, Ordering::Relaxed);
        values.par_iter().copied().collect()
    }

    fn read_buffer<T: Pod>(&self, buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<T> {
        let values = host_view::<T>(buffer, len);
        self.activity
            .downloaded_bytes
            .fetch_add(size_of_val(values) as u64, Ordering::Relaxed);
        values.to_vec()
    }

    pub fn activity(&self) -> MetalActivity {
        MetalActivity {
            command_buffers: self.activity.command_buffers.load(Ordering::Relaxed),
            dispatches: self.activity.dispatches.load(Ordering::Relaxed),
            host_waits: self.activity.host_waits.load(Ordering::Relaxed),
            allocated_bytes: self.activity.allocated_bytes.load(Ordering::Relaxed),
            uploaded_bytes: self.activity.uploaded_bytes.load(Ordering::Relaxed),
            downloaded_bytes: self.activity.downloaded_bytes.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn current_allocated_bytes(&self) -> usize {
        self.device.currentAllocatedSize()
    }
}

#[cfg(test)]
#[path = "../tests/unit/buffer_limit.rs"]
mod buffer_limit_tests;

/// Uploads `values` before the session exists, so outside its memory limit
/// and activity counters.
fn pre_session_buffer<T: Copy>(device: &ProtocolObject<dyn MTLDevice>, values: &[T]) -> Result<Buffer, MetalError> {
    let bytes = size_of_val(values);
    let buffer = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::Buffer { bytes })?;
    upload(&buffer, values)?;
    Ok(buffer)
}

/// Copy `values` to the start of `buffer`, a buffer that Metal allocated
/// and that no command is using.
fn upload<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, values: &[T]) -> Result<(), MetalError> {
    let bytes = size_of_val(values);
    if bytes > buffer.length() {
        return Err(MetalError::Shape("Metal shared-buffer write exceeds capacity"));
    }
    // SAFETY: the source is a live slice of `bytes` bytes, and the
    // destination is the start of the buffer's shared memory, which holds
    // at least `bytes` bytes (checked above) and lives as long as `buffer`.
    // The copy is untyped, so it needs no alignment. The ranges do not
    // overlap, because callers upload only into buffers that Metal
    // allocated, never into a wrapped one. No command uses the buffer during
    // the copy: every command finishes inside the call that encodes it (see
    // `launch`).
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>(),
            bytes,
        );
    }
    Ok(())
}

fn pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &'static str,
) -> Result<Pipeline, MetalError> {
    let name_string = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&name_string)
        .ok_or(MetalError::Function(name))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| MetalError::Pipeline(format!("{error:?}")))
}

fn finish(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    command.commit();
    wait(command)
}

fn wait(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    command.waitUntilCompleted();
    if let Some(error) = command.error() {
        return Err(MetalError::Execution(format!("{error:?}")));
    }
    Ok(())
}

/// A type that device bytes can be read as: it has no padding, and every bit
/// pattern is a value.
///
/// # Safety
///
/// Implement it only for such types.
unsafe trait Pod: Copy + Send + Sync {}

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

/// The start of `buffer`'s shared memory as `len` values of `T`. It panics
/// unless they fit in the buffer from an aligned start: the length comes
/// from the same shape that sized the buffer, so a mismatch is a bug here,
/// and the panic stops an out-of-bounds read.
fn host_contents<T>(buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> *mut T {
    let contents = buffer.contents().as_ptr().cast::<T>();
    assert!(
        len.checked_mul(size_of::<T>())
            .is_some_and(|bytes| bytes <= buffer.length())
            && contents.is_aligned(),
        "a host view of {len} values does not fit its Metal buffer"
    );
    contents
}

/// The first `len` values of a buffer that Metal allocated, after its
/// commands finished.
fn host_view<T: Pod>(buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> &[T] {
    let contents = host_contents::<T>(buffer, len);
    // SAFETY: `host_contents` checked that `len` values of `T` fit in the
    // buffer's shared memory from an aligned, non-null start. That memory
    // lives as long as the `buffer` borrow. Metal zero-fills a new buffer,
    // so every byte is initialized, and every bit pattern is a `T` (`Pod`).
    // No Rust reference aliases it, because callers view only buffers that
    // Metal allocated, never a wrapped one. Every command finishes inside
    // the call that encodes it (see `launch`), and callers take a view only
    // after `finish`, so no kernel writes the memory while the view lives.
    unsafe { std::slice::from_raw_parts(contents, len) }
}

/// The first `len` words of a buffer that the caller has just allocated.
fn host_view_mut(buffer: &mut Buffer, len: usize) -> &mut [u64] {
    let contents = host_contents::<u64>(buffer, len);
    // SAFETY: `host_contents` checked that `len` words fit in the buffer's
    // shared memory from an aligned, non-null start, and Metal zero-fills a
    // new buffer. The caller has just allocated `buffer`, so no other handle
    // and no command uses it, and the `&mut` borrow keeps it from being
    // bound or viewed again while this view lives.
    unsafe { std::slice::from_raw_parts_mut(contents, len) }
}
