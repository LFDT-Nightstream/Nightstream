//! Metal device ownership and synchronous parity primitives.
//!
//! These calls deliberately synchronize at their result boundary so the
//! initial Apple parity gate can compare ordinary host words. The production
//! prover loop will retain these buffers behind a Metal fold-output carrier
//! and synchronize only at explicit proof egress.

use std::ffi::c_void;
use std::ptr::NonNull;

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::poseidon2::{DIGEST_LEN, RC_WORDS, WIDTH};

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

type BufferObject = ProtocolObject<dyn MTLBuffer>;
type PipelineObject = ProtocolObject<dyn MTLComputePipelineState>;

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("no Metal device is available")]
    NoDevice,
    #[error("the Metal device did not create a command queue")]
    NoCommandQueue,
    #[error("invalid Metal input: {0}")]
    InvalidInput(&'static str),
    #[error("Metal buffer size overflow")]
    BufferSizeOverflow,
    #[error("Metal failed to allocate a {bytes}-byte buffer")]
    BufferAllocation { bytes: usize },
    #[error("failed to load the Metal library: {0}")]
    Library(String),
    #[error("Metal library does not contain kernel `{0}`")]
    MissingKernel(&'static str),
    #[error("failed to create Metal pipeline `{kernel}`: {message}")]
    Pipeline {
        kernel: &'static str,
        message: String,
    },
    #[error("Metal did not create a command buffer")]
    NoCommandBuffer,
    #[error("Metal did not create a compute command encoder")]
    NoComputeEncoder,
    #[error("Metal command execution failed: {0}")]
    Execution(String),
}

struct Pipelines {
    add: Retained<PipelineObject>,
    sub: Retained<PipelineObject>,
    mul: Retained<PipelineObject>,
    mul_low_norm: Retained<PipelineObject>,
    extension_mul: Retained<PipelineObject>,
    poseidon2_permute: Retained<PipelineObject>,
    poseidon2_hash: Retained<PipelineObject>,
    poseidon2_transcript: Retained<PipelineObject>,
}

/// One raw operation against the canonical Poseidon2 transcript sponge.
/// Callers own protocol framing such as labels and field-count prefixes.
pub enum MetalTranscriptOp {
    AbsorbRaw(Vec<u64>),
    Challenge(usize),
}

pub struct MetalTranscriptOutput {
    pub state: [u64; WIDTH],
    pub absorbed: usize,
    pub challenges: Vec<u64>,
}

/// One Metal device, command queue, and the proof-critical pipelines landed
/// so far. The same type is used on macOS and iOS.
pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: Pipelines,
}

impl MetalDevice {
    /// Create the device from an SDK-specific, precompiled `.metallib`.
    ///
    /// # Safety
    ///
    /// `metallib` must be compiled from this crate's shader sources. Metal
    /// kernel signatures are not a sufficient memory-safety boundary for an
    /// arbitrary library with matching function names.
    pub unsafe fn from_metallib(metallib: &[u8]) -> Result<Self, MetalError> {
        if metallib.is_empty() {
            return Err(MetalError::InvalidInput("metallib is empty"));
        }
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;
        let data = DispatchData::from_bytes(metallib);
        let library = device
            .newLibraryWithData_error(&data)
            .map_err(|error| MetalError::Library(error.to_string()))?;
        let pipelines = Pipelines {
            add: make_pipeline(&device, &library, "goldilocks_add")?,
            sub: make_pipeline(&device, &library, "goldilocks_sub")?,
            mul: make_pipeline(&device, &library, "goldilocks_mul")?,
            mul_low_norm: make_pipeline(&device, &library, "goldilocks_mul_low_norm")?,
            extension_mul: make_pipeline(&device, &library, "goldilocks_extension_mul")?,
            poseidon2_permute: make_pipeline(&device, &library, "poseidon2_permute_states")?,
            poseidon2_hash: make_pipeline(&device, &library, "poseidon2_hash_fields")?,
            poseidon2_transcript: make_pipeline(&device, &library, "poseidon2_transcript_ops")?,
        };
        Ok(Self {
            device,
            queue,
            pipelines,
        })
    }

    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn goldilocks_add(&self, lhs: &[u64], rhs: &[u64]) -> Result<Vec<u64>, MetalError> {
        self.binary_words(&self.pipelines.add, lhs, rhs)
    }

    pub fn goldilocks_sub(&self, lhs: &[u64], rhs: &[u64]) -> Result<Vec<u64>, MetalError> {
        self.binary_words(&self.pipelines.sub, lhs, rhs)
    }

    pub fn goldilocks_mul(&self, lhs: &[u64], rhs: &[u64]) -> Result<Vec<u64>, MetalError> {
        self.binary_words(&self.pipelines.mul, lhs, rhs)
    }

    pub fn goldilocks_mul_low_norm(&self, lhs: &[u64], rhs: &[u64]) -> Result<Vec<u64>, MetalError> {
        self.binary_words(&self.pipelines.mul_low_norm, lhs, rhs)
    }

    pub fn extension_mul(&self, lhs: &[[u64; 2]], rhs: &[[u64; 2]]) -> Result<Vec<[u64; 2]>, MetalError> {
        if lhs.len() != rhs.len() {
            return Err(MetalError::InvalidInput("extension operand lengths differ"));
        }
        if lhs.is_empty() {
            return Ok(Vec::new());
        }
        let lhs_words: Vec<u64> = lhs.iter().flatten().copied().collect();
        let rhs_words: Vec<u64> = rhs.iter().flatten().copied().collect();
        let lhs_buffer = DeviceWords::from_slice(&self.device, &lhs_words)?;
        let rhs_buffer = DeviceWords::from_slice(&self.device, &rhs_words)?;
        let out_buffer = DeviceWords::zeroed(&self.device, lhs_words.len())?;
        self.dispatch(
            &self.pipelines.extension_mul,
            &[&lhs_buffer, &rhs_buffer, &out_buffer],
            lhs.len(),
        )?;
        Ok(out_buffer
            .to_vec()
            .chunks_exact(2)
            .map(|words| [words[0], words[1]])
            .collect())
    }

    pub fn poseidon2_permute(
        &self,
        states: &[[u64; WIDTH]],
        round_constants: &[u64],
    ) -> Result<Vec<[u64; WIDTH]>, MetalError> {
        validate_round_constants(round_constants)?;
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let state_words: Vec<u64> = states.iter().flatten().copied().collect();
        let state_buffer = DeviceWords::from_slice(&self.device, &state_words)?;
        let constants_buffer = DeviceWords::from_slice(&self.device, round_constants)?;
        self.dispatch(
            &self.pipelines.poseidon2_permute,
            &[&state_buffer, &constants_buffer],
            states.len(),
        )?;
        Ok(state_buffer
            .to_vec()
            .chunks_exact(WIDTH)
            .map(|words| core::array::from_fn(|lane| words[lane]))
            .collect())
    }

    pub fn poseidon2_hash(
        &self,
        inputs: &[&[u64]],
        round_constants: &[u64],
    ) -> Result<Vec<[u64; DIGEST_LEN]>, MetalError> {
        validate_round_constants(round_constants)?;
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let mut fields = Vec::new();
        let mut offsets = Vec::with_capacity(inputs.len());
        let mut lengths = Vec::with_capacity(inputs.len());
        for input in inputs {
            offsets.push(fields.len() as u64);
            lengths.push(input.len() as u64);
            fields.extend_from_slice(input);
        }
        if fields.is_empty() {
            fields.push(0);
        }
        let fields_buffer = DeviceWords::from_slice(&self.device, &fields)?;
        let offsets_buffer = DeviceWords::from_slice(&self.device, &offsets)?;
        let lengths_buffer = DeviceWords::from_slice(&self.device, &lengths)?;
        let out_buffer = DeviceWords::zeroed(&self.device, inputs.len() * DIGEST_LEN)?;
        let constants_buffer = DeviceWords::from_slice(&self.device, round_constants)?;
        self.dispatch(
            &self.pipelines.poseidon2_hash,
            &[
                &fields_buffer,
                &offsets_buffer,
                &lengths_buffer,
                &out_buffer,
                &constants_buffer,
            ],
            inputs.len(),
        )?;
        Ok(out_buffer
            .to_vec()
            .chunks_exact(DIGEST_LEN)
            .map(|words| core::array::from_fn(|lane| words[lane]))
            .collect())
    }

    pub fn poseidon2_transcript(
        &self,
        state: [u64; WIDTH],
        absorbed: usize,
        ops: &[MetalTranscriptOp],
        round_constants: &[u64],
    ) -> Result<MetalTranscriptOutput, MetalError> {
        validate_round_constants(round_constants)?;
        if absorbed > crate::poseidon2::RATE {
            return Err(MetalError::InvalidInput("Poseidon2 transcript cursor exceeds the rate"));
        }
        if ops.is_empty() {
            return Ok(MetalTranscriptOutput {
                state,
                absorbed,
                challenges: Vec::new(),
            });
        }

        let mut op_words = Vec::with_capacity(2 * ops.len());
        let mut payload = Vec::new();
        let mut challenge_count = 0usize;
        for op in ops {
            match op {
                MetalTranscriptOp::AbsorbRaw(words) => {
                    op_words.extend([0, words.len() as u64]);
                    payload.extend_from_slice(words);
                }
                MetalTranscriptOp::Challenge(count) => {
                    op_words.extend([1, *count as u64]);
                    challenge_count = challenge_count
                        .checked_add(*count)
                        .ok_or(MetalError::BufferSizeOverflow)?;
                }
            }
        }
        if payload.is_empty() {
            payload.push(0);
        }

        let mut state_words = state.to_vec();
        state_words.push(absorbed as u64);
        let state_buffer = DeviceWords::from_slice(&self.device, &state_words)?;
        let ops_buffer = DeviceWords::from_slice(&self.device, &op_words)?;
        let payload_buffer = DeviceWords::from_slice(&self.device, &payload)?;
        let out_buffer = DeviceWords::zeroed(&self.device, challenge_count.max(1))?;
        let meta_buffer = DeviceWords::from_slice(&self.device, &[ops.len() as u64])?;
        let constants_buffer = DeviceWords::from_slice(&self.device, round_constants)?;
        self.dispatch(
            &self.pipelines.poseidon2_transcript,
            &[
                &state_buffer,
                &ops_buffer,
                &payload_buffer,
                &out_buffer,
                &meta_buffer,
                &constants_buffer,
            ],
            1,
        )?;
        let final_state = state_buffer.to_vec();
        Ok(MetalTranscriptOutput {
            state: core::array::from_fn(|lane| final_state[lane]),
            absorbed: final_state[WIDTH] as usize,
            challenges: out_buffer.to_vec()[..challenge_count].to_vec(),
        })
    }

    fn binary_words(&self, pipeline: &PipelineObject, lhs: &[u64], rhs: &[u64]) -> Result<Vec<u64>, MetalError> {
        if lhs.len() != rhs.len() {
            return Err(MetalError::InvalidInput("operand lengths differ"));
        }
        if lhs.is_empty() {
            return Ok(Vec::new());
        }
        let lhs_buffer = DeviceWords::from_slice(&self.device, lhs)?;
        let rhs_buffer = DeviceWords::from_slice(&self.device, rhs)?;
        let out_buffer = DeviceWords::zeroed(&self.device, lhs.len())?;
        self.dispatch(pipeline, &[&lhs_buffer, &rhs_buffer, &out_buffer], lhs.len())?;
        Ok(out_buffer.to_vec())
    }

    fn dispatch(
        &self,
        pipeline: &PipelineObject,
        buffers: &[&DeviceWords],
        thread_count: usize,
    ) -> Result<(), MetalError> {
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::NoCommandBuffer)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::NoComputeEncoder)?;
        encoder.setComputePipelineState(pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&buffer.buffer), 0, index);
            }
        }
        let execution_width = pipeline.threadExecutionWidth();
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        let group_width = execution_width.min(max_threads).min(thread_count).max(1);
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: thread_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            let message = command_buffer
                .error()
                .map(|error| error.to_string())
                .unwrap_or_else(|| "unknown command-buffer error".to_owned());
            return Err(MetalError::Execution(message));
        }
        Ok(())
    }
}

fn make_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    kernel: &'static str,
) -> Result<Retained<PipelineObject>, MetalError> {
    let name = NSString::from_str(kernel);
    let function = library
        .newFunctionWithName(&name)
        .ok_or(MetalError::MissingKernel(kernel))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| MetalError::Pipeline {
            kernel,
            message: error.to_string(),
        })
}

struct DeviceWords {
    buffer: Retained<BufferObject>,
    len: usize,
}

impl DeviceWords {
    fn from_slice(device: &ProtocolObject<dyn MTLDevice>, words: &[u64]) -> Result<Self, MetalError> {
        if words.is_empty() {
            return Err(MetalError::InvalidInput("cannot allocate an empty device buffer"));
        }
        let bytes = byte_len(words.len())?;
        let pointer = NonNull::new(words.as_ptr().cast_mut().cast::<c_void>())
            .ok_or(MetalError::InvalidInput("host slice pointer is null"))?;
        let buffer =
            unsafe { device.newBufferWithBytes_length_options(pointer, bytes, MTLResourceOptions::StorageModeShared) }
                .ok_or(MetalError::BufferAllocation { bytes })?;
        Ok(Self {
            buffer,
            len: words.len(),
        })
    }

    fn zeroed(device: &ProtocolObject<dyn MTLDevice>, len: usize) -> Result<Self, MetalError> {
        if len == 0 {
            return Err(MetalError::InvalidInput("cannot allocate an empty device buffer"));
        }
        let bytes = byte_len(len)?;
        let buffer = device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocation { bytes })?;
        unsafe {
            std::ptr::write_bytes(buffer.contents().cast::<u8>().as_ptr(), 0, bytes);
        }
        Ok(Self { buffer, len })
    }

    fn to_vec(&self) -> Vec<u64> {
        unsafe { std::slice::from_raw_parts(self.buffer.contents().cast::<u64>().as_ptr(), self.len) }.to_vec()
    }
}

fn byte_len(words: usize) -> Result<usize, MetalError> {
    words
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(MetalError::BufferSizeOverflow)
}

fn validate_round_constants(words: &[u64]) -> Result<(), MetalError> {
    if words.len() != RC_WORDS {
        return Err(MetalError::InvalidInput("Poseidon2 round-constant length differs"));
    }
    Ok(())
}
