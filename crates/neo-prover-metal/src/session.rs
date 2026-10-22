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

use crate::{MetalActivity, MetalError, BUFFER_LIMIT_BYTES};

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
static POSEIDON2_CONSTANT_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/poseidon2.constants"));
type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

static BUFFER_ALLOCATION: Mutex<()> = Mutex::new(());

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
        let poseidon2_constants = buffer_from_slice(&device, &poseidon2_round_constants())?;
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
        let bytes = size_of_val(values);
        let buffer = self.buffer(bytes)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.contents().as_ptr().cast::<u8>(),
                bytes,
            );
        }
        self.activity
            .uploaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
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
        let buffer = self.buffer(bytes)?;
        // SAFETY: Metal gives a shared-storage buffer a CPU-visible, aligned `contents()`
        // pointer that stays valid while the buffer lives; this new buffer has no device readers.
        let destination = unsafe { std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u64>(), words) };
        fill(destination)?;
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

    fn dispatch(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        elements: usize,
    ) {
        self.activity.dispatches.fetch_add(1, Ordering::Relaxed);
        dispatch(encoder, pipeline, elements);
    }

    fn dispatch_threadgroups(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        _pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        groups: usize,
        threads: usize,
    ) {
        self.activity.dispatches.fetch_add(1, Ordering::Relaxed);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
    }

    fn write_shared<T: Copy>(&self, buffer: &ProtocolObject<dyn MTLBuffer>, values: &[T]) -> Result<(), MetalError> {
        let bytes = size_of_val(values);
        if bytes > buffer.length() as usize {
            return Err(MetalError::Shape("Metal shared-buffer write exceeds capacity"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.contents().as_ptr().cast::<u8>(),
                bytes,
            );
        }
        self.activity
            .uploaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    fn finish(&self, command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
        self.activity.host_waits.fetch_add(1, Ordering::Relaxed);
        finish(command)
    }

    /// `read_buffer` with the copy split over the CPU cores.
    fn read_buffer_parallel<T: Copy + Send + Sync>(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        len: usize,
    ) -> Vec<T> {
        use rayon::prelude::*;
        self.activity
            .downloaded_bytes
            .fetch_add((len * size_of::<T>()) as u64, Ordering::Relaxed);
        let source = unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<T>(), len) };
        source.par_iter().copied().collect()
    }

    fn read_buffer<T: Copy>(&self, buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<T> {
        self.activity
            .downloaded_bytes
            .fetch_add((len * size_of::<T>()) as u64, Ordering::Relaxed);
        read_buffer(buffer, len)
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

fn dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    elements: usize,
) {
    let width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: elements,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

fn poseidon2_round_constants() -> Vec<u64> {
    POSEIDON2_CONSTANT_BYTES
        .chunks_exact(size_of::<u64>())
        .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("chunks_exact has width 8")))
        .collect()
}

fn buffer_from_slice<T: Copy>(device: &ProtocolObject<dyn MTLDevice>, values: &[T]) -> Result<Buffer, MetalError> {
    let bytes = size_of_val(values);
    let buffer = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::Buffer { bytes })?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>(),
            bytes,
        );
    }
    Ok(buffer)
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

fn read_buffer<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<T> {
    let mut out = Vec::<T>::with_capacity(len);
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.contents().as_ptr().cast::<T>(), out.as_mut_ptr(), len);
        out.set_len(len);
    }
    out
}
