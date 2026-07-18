//! Metal device, pipeline, queue, and shared-buffer ownership.
//!
//! Protocol phase ordering stays in the adapter. This layer owns command
//! encoding and accounts for online-path CPU reads, writes, and waits.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

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

mod ajtai_batch;
mod carrier;
mod dec;
mod dec_public;
mod dec_seeded;
mod fe_streaming;
mod oracle;
mod resident;
mod sis;
pub(crate) use carrier::{MetalResidentWitness, MetalResidentWitnessSnapshot};
pub(crate) use dec::{MetalAjtaiRingForms, MetalDecFormPlan};
pub(crate) use dec_public::MetalDecPublicProjection;
pub(crate) use oracle::{MetalDeferredEvalTable, MetalDeferredMcsRowTables, MetalFeOraclePlan};
pub(crate) use resident::{
    MetalFeSumcheckInputs, MetalFeSumcheckPlan, MetalFeTableInput, MetalNcDigitInput, MetalNcFinalState,
    MetalNcSumcheckInputs, MetalNcSumcheckPlan, MetalNcSumcheckTrace, MetalSumcheckTrace, MetalWitnessMasks,
};
use sis::MetalSisMap;

static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nightstream-metal.metallib"));
static POSEIDON2_CONSTANT_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/poseidon2.constants"));
static NEXT_SESSION_OWNERSHIP_ID: AtomicU64 = AtomicU64::new(1);

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

/// Long-lived owner of one Metal device and its reusable prover resources.
///
/// The primary queue preserves dependencies within the proof pipeline. The
/// independent queue is reserved for work that can overlap that pipeline.
pub struct MetalSession {
    ownership_id: u64,
    device: Device,
    queue: Queue,
    independent_queue: Queue,
    // Primitive arithmetic, hashing, and transcript pipelines.
    goldilocks_ops: Pipeline,
    goldilocks_ops_native: Pipeline,
    copy_k_words: Pipeline,
    copy_base_to_k: Pipeline,
    kx_mul_add: Pipeline,
    poseidon2_permute: Pipeline,
    poseidon2_hash: Pipeline,
    poseidon2_hash_simd: Pipeline,
    poseidon2_hash_uniform: Pipeline,
    poseidon2_hash_uniform_simd: Pipeline,
    transcript_absorb_challenge2: Pipeline,
    poseidon2_constants: Buffer,
    // Ajtai and SIS commitment pipelines.
    ajtai_mat_vec: Pipeline,
    ajtai_low_norm_products: Pipeline,
    ajtai_reduce_columns: Pipeline,
    seeded_ajtai_matrix: Pipeline,
    sis_balanced_ternary_message: Pipeline,
    sis_pack_signed_masks: Pipeline,
    // Shared table construction and sumcheck pipelines.
    fold_k_table: Pipeline,
    tensor_point_expand_k: Pipeline,
    fe_carried_plane_lin_comb: Pipeline,
    fe_weighted_basis_dots: Pipeline,
    fe_weighted_row_table: Pipeline,
    fe_build_mcs_row_tables: Pipeline,
    fe_add_sparse_base_rows: Pipeline,
    fe_seeded_k_partials: Pipeline,
    fe_seeded_k_reduce: Pipeline,
    fe_stream_mcs_round_partials: Pipeline,
    fe_stream_mcs_factored_round_partials: Pipeline,
    fe_stream_eval_round_partials: Pipeline,
    fe_stream_constant_round_partials: Pipeline,
    fe_fold_base_tables_in_place: Pipeline,
    fe_round_partials: Pipeline,
    nc_round_mask_partials: Pipeline,
    nc_fold_signed_masks: Pipeline,
    nc_expand_mask_basis: Pipeline,
    nc_materialize_mask_dense: Pipeline,
    nc_round_partials: Pipeline,
    sumcheck_reduce_partials: Pipeline,
    nc_fold_compact: Pipeline,
    // Cross-phase witness mixing and Pi_DEC pipelines.
    rlc_witness_mix: Pipeline,
    rlc_witness_mix_resident_tail: Pipeline,
    rlc_witness_mix_signed_masks: Pipeline,
    rlc_witness_mix_signed_masks_resident_tail: Pipeline,
    dec_split_base2: Pipeline,
    dec_validate_split: Pipeline,
    dec_build_ring_forms: Pipeline,
    dec_build_parallel_original_forms: Pipeline,
    dec_bar_ring_forms_in_place: Pipeline,
    dec_build_seeded_ring_forms: Pipeline,
    dec_add_bar_seeded_ring_forms: Pipeline,
    dec_add_sparse_ring_forms: Pipeline,
    dec_binary_masks: Pipeline,
    dec_ring_partials: Pipeline,
    dec_ring_sum_chunks: Pipeline,
    dec_sparse_ring_partials: Pipeline,
    dec_sparse_ring_sum_chunks: Pipeline,
    dec_ring_reduce_phi81: Pipeline,
    dec_y_zcol_partials: Pipeline,
    dec_y_zcol_reduce: Pipeline,
    ajtai_lane_ring_partials: Pipeline,
    ajtai_lane_ring_sum_chunks: Pipeline,
    ajtai_lane_ring_reduce_phi81: Pipeline,
    // Structure-static caches, the current running generation, and bounded
    // recycling slots. Protocol code sees opaque plans rather than buffers.
    sis_maps: RefCell<Vec<MetalSisMap>>,
    resident_running: RefCell<Option<(u64, carrier::MetalResidentChildren)>>,
    recycled_nc_plan: RefCell<Option<MetalNcSumcheckPlan>>,
    recycled_ajtai_forms: RefCell<Option<Buffer>>,
    recycled_seeded_forms: RefCell<Option<Buffer>>,
    recycled_dec_partials: RefCell<Option<Buffer>>,
    recycled_dec_children: RefCell<Option<carrier::MetalResidentChildren>>,
    next_resident_id: Cell<u64>,
    fe_sumcheck_duration: Cell<Duration>,
    nc_sumcheck_duration: Cell<Duration>,
    activity: ActivityCounters,
}

/// Resident Ajtai matrix and reduction scratch reused across commitments.
pub struct MetalAjtaiLowNormPlan {
    matrix: Buffer,
    shapes: Buffer,
    first: Buffer,
    second: Buffer,
    rows: usize,
    cols: usize,
    product_words: usize,
}

/// Ping-pong buffers for repeated quadratic-extension multiply-add rounds.
pub struct MetalKxChainPlan {
    initial: Buffer,
    multipliers: Buffer,
    first: Buffer,
    second: Buffer,
    elements: usize,
}

/// Resident fixed-width Poseidon2 batch with reusable output storage.
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
        queue.setLabel(Some(&NSString::from_str("nightstream.compute")));
        let independent_queue = device.newCommandQueue().ok_or(MetalError::Queue)?;
        independent_queue.setLabel(Some(&NSString::from_str("nightstream.independent")));
        let data = DispatchData::from_static_bytes(METALLIB);
        let library = device
            .newLibraryWithData_error(&data)
            .map_err(|error| MetalError::Library(format!("{error:?}")))?;
        let goldilocks_ops = pipeline(&device, &library, "goldilocks_ops")?;
        let goldilocks_ops_native = pipeline(&device, &library, "goldilocks_ops_native")?;
        let copy_k_words = pipeline(&device, &library, "copy_k_words")?;
        let copy_base_to_k = pipeline(&device, &library, "copy_base_to_k")?;
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
        let seeded_ajtai_matrix = pipeline(&device, &library, "seeded_ajtai_matrix")?;
        let sis_balanced_ternary_message = pipeline(&device, &library, "sis_balanced_ternary_message")?;
        let sis_pack_signed_masks = pipeline(&device, &library, "sis_pack_signed_masks")?;
        let fold_k_table = pipeline(&device, &library, "fold_k_table")?;
        let tensor_point_expand_k = pipeline(&device, &library, "tensor_point_expand_k")?;
        let fe_carried_plane_lin_comb = pipeline(&device, &library, "fe_carried_plane_lin_comb")?;
        let fe_weighted_basis_dots = pipeline(&device, &library, "fe_weighted_basis_dots")?;
        let fe_weighted_row_table = pipeline(&device, &library, "fe_weighted_row_table")?;
        let fe_build_mcs_row_tables = pipeline(&device, &library, "fe_build_mcs_row_tables")?;
        let fe_add_sparse_base_rows = pipeline(&device, &library, "fe_add_sparse_base_rows")?;
        let fe_seeded_k_partials = pipeline(&device, &library, "fe_seeded_k_partials")?;
        let fe_seeded_k_reduce = pipeline(&device, &library, "fe_seeded_k_reduce")?;
        let fe_stream_mcs_round_partials = pipeline(&device, &library, "fe_stream_mcs_round_partials")?;
        let fe_stream_mcs_factored_round_partials =
            pipeline(&device, &library, "fe_stream_mcs_factored_round_partials")?;
        let fe_stream_eval_round_partials = pipeline(&device, &library, "fe_stream_eval_round_partials")?;
        let fe_stream_constant_round_partials = pipeline(&device, &library, "fe_stream_constant_round_partials")?;
        let fe_fold_base_tables_in_place = pipeline(&device, &library, "fe_fold_base_tables_in_place")?;
        let fe_round_partials = pipeline(&device, &library, "fe_round_partials")?;
        let nc_round_mask_partials = pipeline(&device, &library, "nc_round_mask_partials")?;
        let nc_fold_signed_masks = pipeline(&device, &library, "nc_fold_signed_masks")?;
        let nc_expand_mask_basis = pipeline(&device, &library, "nc_expand_mask_basis")?;
        let nc_materialize_mask_dense = pipeline(&device, &library, "nc_materialize_mask_dense")?;
        let nc_round_partials = pipeline(&device, &library, "nc_round_partials")?;
        let sumcheck_reduce_partials = pipeline(&device, &library, "sumcheck_reduce_partials")?;
        let nc_fold_compact = pipeline(&device, &library, "nc_fold_compact")?;
        let rlc_witness_mix = pipeline(&device, &library, "rlc_witness_mix")?;
        let rlc_witness_mix_resident_tail = pipeline(&device, &library, "rlc_witness_mix_resident_tail")?;
        let rlc_witness_mix_signed_masks = pipeline(&device, &library, "rlc_witness_mix_signed_masks")?;
        let rlc_witness_mix_signed_masks_resident_tail =
            pipeline(&device, &library, "rlc_witness_mix_signed_masks_resident_tail")?;
        let dec_split_base2 = pipeline(&device, &library, "dec_split_base2")?;
        let dec_validate_split = pipeline(&device, &library, "dec_validate_split")?;
        let dec_build_ring_forms = pipeline(&device, &library, "dec_build_ring_forms")?;
        let dec_build_parallel_original_forms = pipeline(&device, &library, "dec_build_parallel_original_forms")?;
        let dec_bar_ring_forms_in_place = pipeline(&device, &library, "dec_bar_ring_forms_in_place")?;
        let dec_build_seeded_ring_forms = pipeline(&device, &library, "dec_build_seeded_ring_forms")?;
        let dec_add_bar_seeded_ring_forms = pipeline(&device, &library, "dec_add_bar_seeded_ring_forms")?;
        let dec_add_sparse_ring_forms = pipeline(&device, &library, "dec_add_sparse_ring_forms")?;
        let dec_binary_masks = pipeline(&device, &library, "dec_binary_masks")?;
        let dec_ring_partials = pipeline(&device, &library, "dec_ring_partials")?;
        let dec_ring_sum_chunks = pipeline(&device, &library, "dec_ring_sum_chunks")?;
        let dec_sparse_ring_partials = pipeline(&device, &library, "dec_sparse_ring_partials")?;
        let dec_sparse_ring_sum_chunks = pipeline(&device, &library, "dec_sparse_ring_sum_chunks")?;
        let dec_ring_reduce_phi81 = pipeline(&device, &library, "dec_ring_reduce_phi81")?;
        let dec_y_zcol_partials = pipeline(&device, &library, "dec_y_zcol_partials")?;
        let dec_y_zcol_reduce = pipeline(&device, &library, "dec_y_zcol_reduce")?;
        let ajtai_lane_ring_partials = pipeline(&device, &library, "ajtai_lane_ring_partials")?;
        let ajtai_lane_ring_sum_chunks = pipeline(&device, &library, "ajtai_lane_ring_sum_chunks")?;
        let ajtai_lane_ring_reduce_phi81 = pipeline(&device, &library, "ajtai_lane_ring_reduce_phi81")?;
        Ok(Self {
            ownership_id: NEXT_SESSION_OWNERSHIP_ID.fetch_add(1, Ordering::Relaxed),
            device,
            queue,
            independent_queue,
            goldilocks_ops,
            goldilocks_ops_native,
            copy_k_words,
            copy_base_to_k,
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
            seeded_ajtai_matrix,
            sis_balanced_ternary_message,
            sis_pack_signed_masks,
            fold_k_table,
            tensor_point_expand_k,
            fe_carried_plane_lin_comb,
            fe_weighted_basis_dots,
            fe_weighted_row_table,
            fe_build_mcs_row_tables,
            fe_add_sparse_base_rows,
            fe_seeded_k_partials,
            fe_seeded_k_reduce,
            fe_stream_mcs_round_partials,
            fe_stream_mcs_factored_round_partials,
            fe_stream_eval_round_partials,
            fe_stream_constant_round_partials,
            fe_fold_base_tables_in_place,
            fe_round_partials,
            nc_round_mask_partials,
            nc_fold_signed_masks,
            nc_expand_mask_basis,
            nc_materialize_mask_dense,
            nc_round_partials,
            sumcheck_reduce_partials,
            nc_fold_compact,
            rlc_witness_mix,
            rlc_witness_mix_resident_tail,
            rlc_witness_mix_signed_masks,
            rlc_witness_mix_signed_masks_resident_tail,
            dec_split_base2,
            dec_validate_split,
            dec_build_ring_forms,
            dec_build_parallel_original_forms,
            dec_bar_ring_forms_in_place,
            dec_build_seeded_ring_forms,
            dec_add_bar_seeded_ring_forms,
            dec_add_sparse_ring_forms,
            dec_binary_masks,
            dec_ring_partials,
            dec_ring_sum_chunks,
            dec_sparse_ring_partials,
            dec_sparse_ring_sum_chunks,
            dec_ring_reduce_phi81,
            dec_y_zcol_partials,
            dec_y_zcol_reduce,
            ajtai_lane_ring_partials,
            ajtai_lane_ring_sum_chunks,
            ajtai_lane_ring_reduce_phi81,
            sis_maps: RefCell::new(Vec::new()),
            resident_running: RefCell::new(None),
            recycled_nc_plan: RefCell::new(None),
            recycled_ajtai_forms: RefCell::new(None),
            recycled_seeded_forms: RefCell::new(None),
            recycled_dec_partials: RefCell::new(None),
            recycled_dec_children: RefCell::new(None),
            next_resident_id: Cell::new(1),
            fe_sumcheck_duration: Cell::new(Duration::ZERO),
            nc_sumcheck_duration: Cell::new(Duration::ZERO),
            activity: ActivityCounters::default(),
        })
    }

    pub(crate) fn ownership_id(&self) -> u64 {
        self.ownership_id
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
        let command = self.command_buffer("nightstream.primitive.goldilocks")?;
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
        let command = self.command_buffer("nightstream.primitive.kx_chain")?;
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
        let command = self.command_buffer("nightstream.poseidon2.permute")?;
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
        let command = self.command_buffer("nightstream.poseidon2.hash")?;
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
        let command = self.command_buffer("nightstream.poseidon2.hash_uniform")?;
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
        let command = self.command_buffer("nightstream.ajtai.mat_vec")?;
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
        self.prepare_ajtai_low_norm_from_buffer(matrix, rows, cols)
    }

    /// Expands the canonical chunked ChaCha matrix directly on Metal, falling
    /// back to the canonical host expansion only if rejection sampling flags it.
    pub(crate) fn prepare_ajtai_low_norm_seeded(
        &self,
        seed: [u8; 32],
        rows: usize,
        cols: usize,
    ) -> Result<MetalAjtaiLowNormPlan, MetalError> {
        const RING_DEGREE: usize = 54;
        const CHACHA_U64S_PER_BLOCK: usize = 8;
        if rows == 0 || cols == 0 {
            return Err(MetalError::Shape("Ajtai dimensions must be nonzero"));
        }
        let matrix_words = rows
            .checked_mul(cols)
            .and_then(|words| words.checked_mul(RING_DEGREE))
            .ok_or(MetalError::Shape("Ajtai matrix dimensions overflow"))?;
        let (chunk_size, chunk_seeds) = neo_ajtai::seeded_pp_chunk_seeds(seed, rows, cols);
        let chunks_per_row = cols.div_ceil(chunk_size);
        if chunk_seeds.len() != rows
            || chunk_seeds
                .iter()
                .any(|seeds| seeds.len() != chunks_per_row)
            || (chunks_per_row > 1 && !(chunk_size * RING_DEGREE).is_multiple_of(CHACHA_U64S_PER_BLOCK))
        {
            return Err(MetalError::Shape("seeded Ajtai chunk geometry is inconsistent"));
        }
        let mut seed_words = Vec::with_capacity(rows * chunks_per_row * 8);
        for chunk in chunk_seeds.iter().flatten() {
            for word in chunk.chunks_exact(4) {
                seed_words.push(u32::from_le_bytes(word.try_into().expect("four-byte seed word")));
            }
        }
        let groups_per_row = (cols * RING_DEGREE).div_ceil(CHACHA_U64S_PER_BLOCK);
        let seeds = self.buffer_from_slice(&seed_words)?;
        let shape = self.buffer_from_slice(&[
            rows as u64,
            cols as u64,
            chunk_size as u64,
            chunks_per_row as u64,
            groups_per_row as u64,
        ])?;
        let rejected = self.buffer_from_slice(&[0u32])?;
        let mut matrix = self.buffer(matrix_words * size_of::<u64>())?;
        let command = self.command_buffer("nightstream.ajtai.seeded_matrix")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.seeded_ajtai_matrix);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&seeds), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&matrix), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&rejected), 0, 3);
        }
        self.dispatch(&encoder, &self.seeded_ajtai_matrix, rows * groups_per_row);
        encoder.endEncoding();
        self.finish(&command)?;
        // The shader flags a rejection-sampling corner case instead of
        // silently choosing different field elements. Materialize the same
        // canonical seeded matrix on the host when that rare case occurs.
        if self.read_buffer::<u32>(&rejected, 1)[0] != 0 {
            let pp = neo_ajtai::materialize_seeded_pp(seed, RING_DEGREE, rows, cols)
                .map_err(|_| MetalError::Shape("materialize rejected seeded Ajtai matrix"))?;
            let words = pp
                .m_rows
                .iter()
                .flat_map(|row| {
                    row.iter()
                        .flat_map(|value| value.0.iter().map(p3_field::PrimeField64::as_canonical_u64))
                })
                .collect::<Vec<_>>();
            matrix = self.buffer_from_slice(&words)?;
        }
        self.prepare_ajtai_low_norm_from_buffer(matrix, rows, cols)
    }

    /// Turns an owned matrix buffer into a reusable reduction plan, including
    /// every halving shape and both ping-pong workspaces.
    fn prepare_ajtai_low_norm_from_buffer(
        &self,
        matrix: Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<MetalAjtaiLowNormPlan, MetalError> {
        const RING_DEGREE: usize = 54;
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
        self.ajtai_low_norm_with_plan_on_queue(plan, message, &self.queue)
    }

    pub(super) fn ajtai_low_norm_with_plan_independent(
        &self,
        plan: &MetalAjtaiLowNormPlan,
        message: &[i8],
    ) -> Result<Vec<u64>, MetalError> {
        self.ajtai_low_norm_with_plan_on_queue(plan, message, &self.independent_queue)
    }

    /// Packs the signed-unit message and executes the prepared commitment on the
    /// selected queue; the independent queue permits overlap with the fold path.
    fn ajtai_low_norm_with_plan_on_queue(
        &self,
        plan: &MetalAjtaiLowNormPlan,
        message: &[i8],
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Vec<u64>, MetalError> {
        const RING_DEGREE: usize = 54;
        let message_digits = plan
            .cols
            .checked_mul(RING_DEGREE)
            .ok_or(MetalError::Shape("Ajtai message dimensions overflow"))?;
        if message.len() != message_digits {
            return Err(MetalError::Shape(
                "Ajtai low-norm message length or digit range is invalid",
            ));
        }
        let mut message_masks = Vec::with_capacity(plan.cols * 2);
        for digits in message.chunks_exact(RING_DEGREE) {
            let mut positive = 0u64;
            let mut negative = 0u64;
            for (coefficient, digit) in digits.iter().enumerate() {
                match digit {
                    1 => positive |= 1 << coefficient,
                    -1 => negative |= 1 << coefficient,
                    0 => {}
                    _ => {
                        return Err(MetalError::Shape(
                            "Ajtai low-norm message length or digit range is invalid",
                        ));
                    }
                }
            }
            message_masks.extend_from_slice(&[positive, negative]);
        }
        let message_masks = self.buffer_from_slice(&message_masks)?;
        let command = self.command_buffer_on(queue, "nightstream.ajtai.low_norm")?;
        let output = self.encode_ajtai_low_norm_masks(&command, plan, &message_masks)?;
        self.finish(&command)?;
        Ok(self.read_buffer::<u64>(output, plan.rows * RING_DEGREE))
    }

    /// Encodes products and the complete column-reduction tree into one command,
    /// returning whichever ping-pong buffer owns the final row sums.
    fn encode_ajtai_low_norm_masks<'a>(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        plan: &'a MetalAjtaiLowNormPlan,
        message_masks: &ProtocolObject<dyn MTLBuffer>,
    ) -> Result<&'a Buffer, MetalError> {
        const RING_DEGREE: usize = 54;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.ajtai_low_norm_products);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(message_masks), 0, 1);
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
        Ok(if reduction_round.is_multiple_of(2) {
            &plan.first
        } else {
            &plan.second
        })
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
        let command = self.command_buffer("nightstream.sumcheck.fold")?;
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
        let command = self.command_buffer("nightstream.sumcheck.fe_full")?;
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
        // Shared storage is required for explicit CPU boundaries on Apple
        // unified memory. It does not make those boundaries free, so reads and
        // writes are still counted separately below.
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

    pub(super) fn record_host_write(&self, bytes: usize) {
        self.activity
            .uploaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    fn command_buffer(&self, label: &str) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, MetalError> {
        self.command_buffer_on(&self.queue, label)
    }

    fn independent_command_buffer(
        &self,
        label: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, MetalError> {
        self.command_buffer_on(&self.independent_queue, label)
    }

    fn command_buffer_on(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        label: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, MetalError> {
        let command = queue.commandBuffer().ok_or(MetalError::CommandBuffer)?;
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

    fn finish(&self, command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
        self.activity.host_waits.fetch_add(1, Ordering::Relaxed);
        finish(command)
    }

    // Commit without a host wait. Queue ordering keeps later commands correct
    // while allowing the CPU to continue encoding independent work.
    fn submit(&self, command: &ProtocolObject<dyn MTLCommandBuffer>) {
        command.commit();
    }

    // Waiting is reserved for a real CPU dependency or an explicit readback.
    fn wait(&self, command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
        self.activity.host_waits.fetch_add(1, Ordering::Relaxed);
        wait(command)
    }

    fn read_buffer<T: Copy>(&self, buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<T> {
        self.activity
            .downloaded_bytes
            .fetch_add((len * size_of::<T>()) as u64, Ordering::Relaxed);
        read_buffer(buffer, len)
    }

    fn read_buffer_range<T: Copy>(&self, buffer: &ProtocolObject<dyn MTLBuffer>, start: usize, len: usize) -> Vec<T> {
        self.activity
            .downloaded_bytes
            .fetch_add((len * size_of::<T>()) as u64, Ordering::Relaxed);
        let mut out = Vec::<T>::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(buffer.contents().as_ptr().cast::<T>().add(start), out.as_mut_ptr(), len);
            out.set_len(len);
        }
        out
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

    pub(crate) fn reset_sumcheck_durations(&self) {
        self.fe_sumcheck_duration.set(Duration::ZERO);
        self.nc_sumcheck_duration.set(Duration::ZERO);
    }

    pub(crate) fn sumcheck_durations(&self) -> (Duration, Duration) {
        (self.fe_sumcheck_duration.get(), self.nc_sumcheck_duration.get())
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
    wait(command)
}

fn wait(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    command.waitUntilCompleted();
    if let Some(error) = command.error() {
        return Err(MetalError::Execution(format!("{error:?}")));
    }
    Ok(())
}

fn nonempty(values: &[u64]) -> &[u64] {
    // Metal does not provide a useful zero-length binding. Kernels use shape
    // metadata to ignore this sentinel whenever the logical input is empty.
    static ZERO: [u64; 1] = [0];
    if values.is_empty() {
        &ZERO
    } else {
        values
    }
}

pub(super) fn command_gpu_duration(command: &ProtocolObject<dyn MTLCommandBuffer>) -> std::time::Duration {
    // Metal timestamps are defined after completion; every caller waits on the
    // command before requesting this duration.
    let start: f64 = unsafe { objc2::msg_send![command, GPUStartTime] };
    let end: f64 = unsafe { objc2::msg_send![command, GPUEndTime] };
    std::time::Duration::from_secs_f64((end - start).max(0.0))
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
