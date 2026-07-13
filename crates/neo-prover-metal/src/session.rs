//! Metal device, pipeline, and shared-buffer ownership for the arithmetic gate.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::{
    GoldilocksMulVariant, GoldilocksOps, KWords, MetalActivity, MetalDeviceInfo, MetalError, MetalRunStats,
    PoseidonDigest, PoseidonHashVariant, PoseidonState,
};

mod carrier;
mod dec;
mod resident;
pub(crate) use carrier::MetalResidentWitness;
pub(crate) use dec::MetalDecFormPlan;
pub(crate) use resident::{
    MetalFeSumcheckInputs, MetalFeSumcheckPlan, MetalNcFinalState, MetalNcSumcheckInputs, MetalNcSumcheckPlan,
    MetalNcSumcheckTrace, MetalSumcheckTrace,
};

static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nightstream-metal.metallib"));
static POSEIDON2_CONSTANT_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/poseidon2.constants"));

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

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
    goldilocks_ops: Pipeline,
    goldilocks_ops_native: Pipeline,
    copy_k_words: Pipeline,
    kx_mul_add: Pipeline,
    poseidon2_permute: Pipeline,
    poseidon2_hash: Pipeline,
    poseidon2_hash_simd: Pipeline,
    poseidon2_hash_uniform: Pipeline,
    poseidon2_hash_uniform_simd: Pipeline,
    transcript_absorb_challenge2: Pipeline,
    poseidon2_constants: Buffer,
    ajtai_mat_vec: Pipeline,
    ajtai_low_norm_products: Pipeline,
    ajtai_reduce_columns: Pipeline,
    fold_k_table: Pipeline,
    fe_round_partials: Pipeline,
    nc_round_partials: Pipeline,
    sumcheck_reduce_partials: Pipeline,
    nc_fold_compact: Pipeline,
    rlc_witness_mix: Pipeline,
    rlc_witness_mix_resident_tail: Pipeline,
    dec_split_base2: Pipeline,
    dec_validate_split: Pipeline,
    dec_build_ring_forms: Pipeline,
    dec_binary_masks: Pipeline,
    dec_ring_partials: Pipeline,
    dec_ring_sum_chunks: Pipeline,
    dec_ring_reduce_phi81: Pipeline,
    resident_running: RefCell<Option<(u64, carrier::MetalResidentChildren)>>,
    next_resident_id: Cell<u64>,
    activity: ActivityCounters,
}

pub struct MetalAjtaiLowNormPlan {
    matrix: Buffer,
    shapes: Buffer,
    first: Buffer,
    second: Buffer,
    rows: usize,
    cols: usize,
    product_words: usize,
}

pub struct MetalKxChainPlan {
    initial: Buffer,
    multipliers: Buffer,
    first: Buffer,
    second: Buffer,
    elements: usize,
}

pub struct MetalPoseidonUniformPlan {
    fields: Buffer,
    output: Buffer,
    shape: Buffer,
    hashes: usize,
}

impl MetalSession {
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::Device)?;
        let queue = device.newCommandQueue().ok_or(MetalError::Queue)?;
        let data = DispatchData::from_static_bytes(METALLIB);
        let library = device
            .newLibraryWithData_error(&data)
            .map_err(|error| MetalError::Library(format!("{error:?}")))?;
        let goldilocks_ops = pipeline(&device, &library, "goldilocks_ops")?;
        let goldilocks_ops_native = pipeline(&device, &library, "goldilocks_ops_native")?;
        let copy_k_words = pipeline(&device, &library, "copy_k_words")?;
        let kx_mul_add = pipeline(&device, &library, "kx_mul_add")?;
        let poseidon2_permute = pipeline(&device, &library, "poseidon2_permute_states")?;
        let poseidon2_hash = pipeline(&device, &library, "poseidon2_hash_fields")?;
        let poseidon2_hash_simd = pipeline(&device, &library, "poseidon2_hash_fields_simd")?;
        let poseidon2_hash_uniform = pipeline(&device, &library, "poseidon2_hash_uniform")?;
        let poseidon2_hash_uniform_simd = pipeline(&device, &library, "poseidon2_hash_uniform_simd")?;
        let transcript_absorb_challenge2 = pipeline(&device, &library, "transcript_absorb_challenge2")?;
        let poseidon2_constants = buffer_from_slice(&device, &poseidon2_round_constants())?;
        let ajtai_mat_vec = pipeline(&device, &library, "ajtai_mat_vec")?;
        let ajtai_low_norm_products = pipeline(&device, &library, "ajtai_low_norm_products")?;
        let ajtai_reduce_columns = pipeline(&device, &library, "ajtai_reduce_columns")?;
        let fold_k_table = pipeline(&device, &library, "fold_k_table")?;
        let fe_round_partials = pipeline(&device, &library, "fe_round_partials")?;
        let nc_round_partials = pipeline(&device, &library, "nc_round_partials")?;
        let sumcheck_reduce_partials = pipeline(&device, &library, "sumcheck_reduce_partials")?;
        let nc_fold_compact = pipeline(&device, &library, "nc_fold_compact")?;
        let rlc_witness_mix = pipeline(&device, &library, "rlc_witness_mix")?;
        let rlc_witness_mix_resident_tail = pipeline(&device, &library, "rlc_witness_mix_resident_tail")?;
        let dec_split_base2 = pipeline(&device, &library, "dec_split_base2")?;
        let dec_validate_split = pipeline(&device, &library, "dec_validate_split")?;
        let dec_build_ring_forms = pipeline(&device, &library, "dec_build_ring_forms")?;
        let dec_binary_masks = pipeline(&device, &library, "dec_binary_masks")?;
        let dec_ring_partials = pipeline(&device, &library, "dec_ring_partials")?;
        let dec_ring_sum_chunks = pipeline(&device, &library, "dec_ring_sum_chunks")?;
        let dec_ring_reduce_phi81 = pipeline(&device, &library, "dec_ring_reduce_phi81")?;
        Ok(Self {
            device,
            queue,
            goldilocks_ops,
            goldilocks_ops_native,
            copy_k_words,
            kx_mul_add,
            poseidon2_permute,
            poseidon2_hash,
            poseidon2_hash_simd,
            poseidon2_hash_uniform,
            poseidon2_hash_uniform_simd,
            transcript_absorb_challenge2,
            poseidon2_constants,
            ajtai_mat_vec,
            ajtai_low_norm_products,
            ajtai_reduce_columns,
            fold_k_table,
            fe_round_partials,
            nc_round_partials,
            sumcheck_reduce_partials,
            nc_fold_compact,
            rlc_witness_mix,
            rlc_witness_mix_resident_tail,
            dec_split_base2,
            dec_validate_split,
            dec_build_ring_forms,
            dec_binary_masks,
            dec_ring_partials,
            dec_ring_sum_chunks,
            dec_ring_reduce_phi81,
            resident_running: RefCell::new(None),
            next_resident_id: Cell::new(1),
            activity: ActivityCounters::default(),
        })
    }

    pub fn goldilocks_ops(&self, lhs: &[u64], rhs: &[u64]) -> Result<Vec<GoldilocksOps>, MetalError> {
        self.goldilocks_ops_variant(lhs, rhs, GoldilocksMulVariant::Limb32)
    }

    pub fn goldilocks_ops_variant(
        &self,
        lhs: &[u64],
        rhs: &[u64],
        variant: GoldilocksMulVariant,
    ) -> Result<Vec<GoldilocksOps>, MetalError> {
        if lhs.len() != rhs.len() {
            return Err(MetalError::Shape("Goldilocks operands have different lengths"));
        }
        if lhs.is_empty() {
            return Ok(Vec::new());
        }
        let lhs_buffer = self.buffer_from_slice(lhs)?;
        let rhs_buffer = self.buffer_from_slice(rhs)?;
        let output = self.buffer(lhs.len() * 3 * size_of::<u64>())?;
        let command = self.command_buffer()?;
        let pipeline = match variant {
            GoldilocksMulVariant::Limb32 => &self.goldilocks_ops,
            GoldilocksMulVariant::Native64 => &self.goldilocks_ops_native,
        };
        self.encode(&command, pipeline, &lhs_buffer, &rhs_buffer, &output, lhs.len())?;
        self.finish(&command)?;
        let words = self.read_buffer::<u64>(&output, lhs.len() * 3);
        Ok(words
            .chunks_exact(3)
            .map(|values| GoldilocksOps {
                add: values[0],
                sub: values[1],
                mul: values[2],
            })
            .collect())
    }

    pub fn kx_mul_add_chain(
        &self,
        initial: &[KWords],
        multipliers: &[KWords],
        rounds: usize,
    ) -> Result<(Vec<KWords>, MetalRunStats), MetalError> {
        if initial.len() != multipliers.len() {
            return Err(MetalError::Shape("extension-field operands have different lengths"));
        }
        if initial.is_empty() || rounds == 0 {
            return Ok((
                initial.to_vec(),
                MetalRunStats {
                    elements: initial.len(),
                    dispatches: 0,
                    elapsed: std::time::Duration::ZERO,
                },
            ));
        }
        let plan = self.prepare_kx_chain(initial, multipliers)?;
        self.kx_mul_add_chain_with_plan(&plan, rounds)
    }

    pub fn prepare_kx_chain(&self, initial: &[KWords], multipliers: &[KWords]) -> Result<MetalKxChainPlan, MetalError> {
        if initial.is_empty() || initial.len() != multipliers.len() {
            return Err(MetalError::Shape(
                "resident extension-field operands must have the same positive length",
            ));
        }
        let initial_words = flatten_k_words(initial);
        let multiplier_words = flatten_k_words(multipliers);
        let initial = self.buffer_from_slice(&initial_words)?;
        let multipliers = self.buffer_from_slice(&multiplier_words)?;
        let first = self.buffer(initial_words.len() * size_of::<u64>())?;
        let second = self.buffer(initial_words.len() * size_of::<u64>())?;
        Ok(MetalKxChainPlan {
            initial,
            multipliers,
            first,
            second,
            elements: initial_words.len() / 2,
        })
    }

    pub fn kx_mul_add_chain_with_plan(
        &self,
        plan: &MetalKxChainPlan,
        rounds: usize,
    ) -> Result<(Vec<KWords>, MetalRunStats), MetalError> {
        if rounds == 0 {
            let words = self.read_buffer::<u64>(&plan.initial, plan.elements * 2);
            return Ok((
                words
                    .chunks_exact(2)
                    .map(|words| KWords::new(words[0], words[1]))
                    .collect(),
                MetalRunStats {
                    elements: plan.elements,
                    dispatches: 0,
                    elapsed: std::time::Duration::ZERO,
                },
            ));
        }
        let command = self.command_buffer()?;
        let started = Instant::now();
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.copy_k_words);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.initial), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.first), 0, 1);
        }
        self.dispatch(&encoder, &self.copy_k_words, plan.elements);
        encoder.endEncoding();
        for round in 0..rounds {
            let (input, output) = if round % 2 == 0 {
                (&plan.first, &plan.second)
            } else {
                (&plan.second, &plan.first)
            };
            self.encode(
                &command,
                &self.kx_mul_add,
                input,
                &plan.multipliers,
                output,
                plan.elements,
            )?;
        }
        self.finish(&command)?;
        let elapsed = started.elapsed();
        let output = if rounds.is_multiple_of(2) {
            &plan.first
        } else {
            &plan.second
        };
        let words = self.read_buffer::<u64>(output, plan.elements * 2);
        let values = words
            .chunks_exact(2)
            .map(|words| KWords::new(words[0], words[1]))
            .collect();
        Ok((
            values,
            MetalRunStats {
                elements: plan.elements,
                dispatches: rounds + 1,
                elapsed,
            },
        ))
    }

    pub fn poseidon2_permute(&self, states: &[PoseidonState]) -> Result<Vec<PoseidonState>, MetalError> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let words = states.iter().flatten().copied().collect::<Vec<_>>();
        let states_buffer = self.buffer_from_slice(&words)?;
        let command = self.command_buffer()?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.poseidon2_permute);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&states_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.poseidon2_constants), 0, 1);
        }
        self.dispatch(&encoder, &self.poseidon2_permute, states.len());
        encoder.endEncoding();
        self.finish(&command)?;
        let output = self.read_buffer::<u64>(&states_buffer, words.len());
        Ok(output
            .chunks_exact(8)
            .map(|words| words.try_into().expect("chunks_exact has width 8"))
            .collect())
    }

    pub fn poseidon2_hash(&self, inputs: &[Vec<u64>]) -> Result<Vec<PoseidonDigest>, MetalError> {
        self.poseidon2_hash_variant(inputs, PoseidonHashVariant::Scalar)
    }

    pub fn poseidon2_hash_variant(
        &self,
        inputs: &[Vec<u64>],
        variant: PoseidonHashVariant,
    ) -> Result<Vec<PoseidonDigest>, MetalError> {
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

        let fields = self.buffer_from_slice(&fields)?;
        let offsets = self.buffer_from_slice(&offsets)?;
        let lengths = self.buffer_from_slice(&lengths)?;
        let output = self.buffer(inputs.len() * 4 * size_of::<u64>())?;
        let command = self.command_buffer()?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        let (pipeline, threads) = match variant {
            PoseidonHashVariant::Scalar => (&self.poseidon2_hash, inputs.len()),
            PoseidonHashVariant::SimdGroup => (&self.poseidon2_hash_simd, inputs.len() * 8),
        };
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&fields), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&offsets), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&lengths), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&self.poseidon2_constants), 0, 4);
        }
        self.dispatch(&encoder, pipeline, threads);
        encoder.endEncoding();
        self.finish(&command)?;
        let output = self.read_buffer::<u64>(&output, inputs.len() * 4);
        Ok(output
            .chunks_exact(4)
            .map(|words| words.try_into().expect("chunks_exact has width 4"))
            .collect())
    }

    pub fn poseidon2_hash_uniform(
        &self,
        fields: &[u64],
        fields_per_hash: usize,
        variant: PoseidonHashVariant,
    ) -> Result<Vec<PoseidonDigest>, MetalError> {
        let plan = self.prepare_poseidon2_uniform(fields, fields_per_hash)?;
        self.poseidon2_hash_uniform_with_plan(&plan, variant)
    }

    pub fn prepare_poseidon2_uniform(
        &self,
        fields: &[u64],
        fields_per_hash: usize,
    ) -> Result<MetalPoseidonUniformPlan, MetalError> {
        if fields_per_hash == 0 || fields.is_empty() || !fields.len().is_multiple_of(fields_per_hash) {
            return Err(MetalError::Shape(
                "uniform Poseidon batch must contain a positive integral number of hashes",
            ));
        }
        let hashes = fields.len() / fields_per_hash;
        let fields = self.buffer_from_slice(fields)?;
        let output = self.buffer(hashes * 4 * size_of::<u64>())?;
        let shape = self.buffer_from_slice(&[fields_per_hash as u64])?;
        Ok(MetalPoseidonUniformPlan {
            fields,
            output,
            shape,
            hashes,
        })
    }

    pub fn poseidon2_hash_uniform_with_plan(
        &self,
        plan: &MetalPoseidonUniformPlan,
        variant: PoseidonHashVariant,
    ) -> Result<Vec<PoseidonDigest>, MetalError> {
        let command = self.command_buffer()?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        let (pipeline, threads) = match variant {
            PoseidonHashVariant::Scalar => (&self.poseidon2_hash_uniform, plan.hashes),
            PoseidonHashVariant::SimdGroup => (&self.poseidon2_hash_uniform_simd, plan.hashes * 8),
        };
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.fields), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.output), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.poseidon2_constants), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&plan.shape), 0, 3);
        }
        self.dispatch(&encoder, pipeline, threads);
        encoder.endEncoding();
        self.finish(&command)?;
        let output = self.read_buffer::<u64>(&plan.output, plan.hashes * 4);
        Ok(output
            .chunks_exact(4)
            .map(|words| words.try_into().expect("chunks_exact has width 4"))
            .collect())
    }

    pub fn ajtai_mat_vec(
        &self,
        matrix: &[u64],
        rows: usize,
        cols: usize,
        message: &[u64],
    ) -> Result<Vec<u64>, MetalError> {
        const RING_DEGREE: usize = 54;
        let matrix_words = rows
            .checked_mul(cols)
            .and_then(|words| words.checked_mul(RING_DEGREE))
            .ok_or(MetalError::Shape("Ajtai matrix dimensions overflow"))?;
        let message_words = cols
            .checked_mul(RING_DEGREE)
            .ok_or(MetalError::Shape("Ajtai message dimensions overflow"))?;
        if matrix.len() != matrix_words || message.len() != message_words {
            return Err(MetalError::Shape(
                "Ajtai matrix or message length does not match its dimensions",
            ));
        }
        if rows == 0 || cols == 0 {
            return Err(MetalError::Shape("Ajtai matrix dimensions must be nonzero"));
        }

        let matrix = self.buffer_from_slice(matrix)?;
        let message = self.buffer_from_slice(message)?;
        let shape = self.buffer_from_slice(&[rows as u64, cols as u64])?;
        let output_words = rows * RING_DEGREE;
        let output = self.buffer(output_words * size_of::<u64>())?;
        let command = self.command_buffer()?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.ajtai_mat_vec);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&matrix), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&message), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 3);
        }
        self.dispatch(&encoder, &self.ajtai_mat_vec, output_words);
        encoder.endEncoding();
        self.finish(&command)?;
        Ok(self.read_buffer::<u64>(&output, output_words))
    }

    pub fn ajtai_low_norm_mat_vec(
        &self,
        matrix: &[u64],
        rows: usize,
        cols: usize,
        message: &[i8],
    ) -> Result<Vec<u64>, MetalError> {
        let plan = self.prepare_ajtai_low_norm(matrix, rows, cols)?;
        self.ajtai_low_norm_with_plan(&plan, message)
    }

    pub fn prepare_ajtai_low_norm(
        &self,
        matrix: &[u64],
        rows: usize,
        cols: usize,
    ) -> Result<MetalAjtaiLowNormPlan, MetalError> {
        const RING_DEGREE: usize = 54;
        let matrix_words = rows
            .checked_mul(cols)
            .and_then(|words| words.checked_mul(RING_DEGREE))
            .ok_or(MetalError::Shape("Ajtai matrix dimensions overflow"))?;
        if matrix.len() != matrix_words {
            return Err(MetalError::Shape("Ajtai matrix length does not match its dimensions"));
        }
        if rows == 0 || cols == 0 {
            return Err(MetalError::Shape("Ajtai dimensions must be nonzero"));
        }

        let matrix = self.buffer_from_slice(matrix)?;
        let mut shape_words = Vec::new();
        let mut current_cols = cols;
        loop {
            shape_words.extend_from_slice(&[rows as u64, current_cols as u64]);
            if current_cols == 1 {
                break;
            }
            current_cols = current_cols.div_ceil(2);
        }
        let shapes = self.buffer_from_slice(&shape_words)?;
        let product_words = rows * cols * RING_DEGREE;
        let first = self.buffer(product_words * size_of::<u64>())?;
        let second = self.buffer(product_words * size_of::<u64>())?;
        Ok(MetalAjtaiLowNormPlan {
            matrix,
            shapes,
            first,
            second,
            rows,
            cols,
            product_words,
        })
    }

    pub fn ajtai_low_norm_with_plan(
        &self,
        plan: &MetalAjtaiLowNormPlan,
        message: &[i8],
    ) -> Result<Vec<u64>, MetalError> {
        const RING_DEGREE: usize = 54;
        let message_digits = plan
            .cols
            .checked_mul(RING_DEGREE)
            .ok_or(MetalError::Shape("Ajtai message dimensions overflow"))?;
        if message.len() != message_digits || message.iter().any(|digit| !(-1..=1).contains(digit)) {
            return Err(MetalError::Shape(
                "Ajtai low-norm message length or digit range is invalid",
            ));
        }
        let message = self.buffer_from_slice(message)?;
        let command = self.command_buffer()?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.ajtai_low_norm_products);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&message), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.first), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&plan.shapes), 0, 3);
        }
        self.dispatch(&encoder, &self.ajtai_low_norm_products, plan.product_words);
        encoder.endEncoding();

        let mut current_cols = plan.cols;
        let mut reduction_round = 0;
        while current_cols > 1 {
            let next_cols = current_cols.div_ceil(2);
            let (input, output) = if reduction_round % 2 == 0 {
                (&plan.first, &plan.second)
            } else {
                (&plan.second, &plan.first)
            };
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.ajtai_reduce_columns);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.shapes), reduction_round * 2 * size_of::<u64>(), 2);
            }
            self.dispatch(
                &encoder,
                &self.ajtai_reduce_columns,
                plan.rows * next_cols * RING_DEGREE,
            );
            encoder.endEncoding();
            current_cols = next_cols;
            reduction_round += 1;
        }
        self.finish(&command)?;
        let output = if reduction_round.is_multiple_of(2) {
            &plan.first
        } else {
            &plan.second
        };
        Ok(self.read_buffer::<u64>(output, plan.rows * RING_DEGREE))
    }

    pub fn fold_k_table(&self, table: &[KWords], challenge: KWords) -> Result<Vec<KWords>, MetalError> {
        if table.is_empty() || !table.len().is_multiple_of(2) {
            return Err(MetalError::Shape(
                "extension-field fold table must have positive even length",
            ));
        }
        let table_words = flatten_k_words(table);
        let table = self.buffer_from_slice(&table_words)?;
        let challenge = self.buffer_from_slice(&[challenge.c0, challenge.c1])?;
        let elements = table_words.len() / 4;
        let output = self.buffer(elements * 2 * size_of::<u64>())?;
        let command = self.command_buffer()?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.fold_k_table);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&table), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&challenge), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
        }
        self.dispatch(&encoder, &self.fold_k_table, elements);
        encoder.endEncoding();
        self.finish(&command)?;
        Ok(self
            .read_buffer::<u64>(&output, elements * 2)
            .chunks_exact(2)
            .map(|words| KWords::new(words[0], words[1]))
            .collect())
    }

    pub fn fold_k_tables(&self, tables: &[Vec<KWords>], challenge: KWords) -> Result<Vec<Vec<KWords>>, MetalError> {
        let Some(table_len) = tables.first().map(Vec::len) else {
            return Ok(Vec::new());
        };
        if table_len == 0 || !table_len.is_multiple_of(2) || tables.iter().any(|table| table.len() != table_len) {
            return Err(MetalError::Shape(
                "batched extension-field fold tables must have the same positive even length",
            ));
        }
        let mut values = Vec::with_capacity(tables.len() * table_len);
        for table in tables {
            values.extend_from_slice(table);
        }
        let folded = self.fold_k_table(&values, challenge)?;
        let folded_len = table_len / 2;
        Ok(folded
            .chunks_exact(folded_len)
            .map(<[KWords]>::to_vec)
            .collect())
    }

    pub fn fold_k_table_full(
        &self,
        table: &[KWords],
        challenges: &[KWords],
    ) -> Result<(KWords, MetalRunStats), MetalError> {
        if table.len() < 2 || !table.len().is_power_of_two() || challenges.len() != table.len().ilog2() as usize {
            return Err(MetalError::Shape(
                "full FE reduction needs a power-of-two table and one challenge per round",
            ));
        }
        let rounds = challenges.len();
        let table_words = flatten_k_words(table);
        let challenge_words = flatten_k_words(challenges);
        let first = self.buffer_from_slice(&table_words)?;
        let second = self.buffer(table_words.len() * size_of::<u64>())?;
        let challenges = self.buffer_from_slice(&challenge_words)?;
        let command = self.command_buffer()?;
        let started = Instant::now();
        let mut current = table.len();
        for round in 0..rounds {
            let (input, output) = if round % 2 == 0 {
                (&first, &second)
            } else {
                (&second, &first)
            };
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.fold_k_table);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&challenges), round * 2 * size_of::<u64>(), 1);
                encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
            }
            current /= 2;
            self.dispatch(&encoder, &self.fold_k_table, current);
            encoder.endEncoding();
        }
        self.finish(&command)?;
        let elapsed = started.elapsed();
        let output = if rounds.is_multiple_of(2) { &first } else { &second };
        let words = self.read_buffer::<u64>(output, 2);
        Ok((
            KWords::new(words[0], words[1]),
            MetalRunStats {
                elements: table.len(),
                dispatches: rounds,
                elapsed,
            },
        ))
    }

    fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        first: &ProtocolObject<dyn MTLBuffer>,
        second: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        elements: usize,
    ) -> Result<(), MetalError> {
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(first), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(second), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        }
        self.dispatch(&encoder, pipeline, elements);
        encoder.endEncoding();
        Ok(())
    }

    fn buffer(&self, bytes: usize) -> Result<Buffer, MetalError> {
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

    fn command_buffer(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, MetalError> {
        let command = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandBuffer)?;
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

    fn finish(&self, command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
        self.activity.host_waits.fetch_add(1, Ordering::Relaxed);
        finish(command)
    }

    fn read_buffer<T: Copy>(&self, buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<T> {
        self.activity
            .downloaded_bytes
            .fetch_add((len * size_of::<T>()) as u64, Ordering::Relaxed);
        read_buffer(buffer, len)
    }

    pub fn device_info(&self) -> Result<MetalDeviceInfo, MetalError> {
        Ok(MetalDeviceInfo {
            name: self.device.name().to_string(),
            unified_memory: self.device.hasUnifiedMemory(),
            recommended_working_set_bytes: self.device.recommendedMaxWorkingSetSize(),
        })
    }

    pub fn activity(&self) -> MetalActivity {
        MetalActivity {
            command_buffers: self.activity.command_buffers.load(Ordering::Relaxed),
            dispatches: self.activity.dispatches.load(Ordering::Relaxed),
            host_waits: self.activity.host_waits.load(Ordering::Relaxed),
            allocated_bytes: self.activity.allocated_bytes.load(Ordering::Relaxed),
            uploaded_bytes: self.activity.uploaded_bytes.load(Ordering::Relaxed),
            downloaded_bytes: self.activity.downloaded_bytes.load(Ordering::Relaxed),
            current_allocated_bytes: self.device.currentAllocatedSize() as u64,
        }
    }

    pub fn reset_activity(&self) {
        self.activity.command_buffers.store(0, Ordering::Relaxed);
        self.activity.dispatches.store(0, Ordering::Relaxed);
        self.activity.host_waits.store(0, Ordering::Relaxed);
        self.activity.allocated_bytes.store(0, Ordering::Relaxed);
        self.activity.uploaded_bytes.store(0, Ordering::Relaxed);
        self.activity.downloaded_bytes.store(0, Ordering::Relaxed);
    }
}

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
    command.waitUntilCompleted();
    if let Some(error) = command.error() {
        return Err(MetalError::Execution(format!("{error:?}")));
    }
    Ok(())
}

fn nonempty(values: &[u64]) -> &[u64] {
    static ZERO: [u64; 1] = [0];
    if values.is_empty() {
        &ZERO
    } else {
        values
    }
}

fn flatten_k_words(values: &[KWords]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|value| [value.c0, value.c1])
        .collect()
}

fn read_buffer<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<T> {
    let mut out = Vec::<T>::with_capacity(len);
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.contents().as_ptr().cast::<T>(), out.as_mut_ptr(), len);
        out.set_len(len);
    }
    out
}
