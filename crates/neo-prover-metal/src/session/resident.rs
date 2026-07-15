//! Resident FE and NC sumcheck buffers.
//!
//! Tables are uploaded once, folded through ping-pong buffers, and only the
//! round polynomial or final NC state crosses back to the host.

use std::sync::atomic::Ordering;

use neo_math::{KExtensions, K};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{Buffer, MetalDeferredEvalTable, MetalDeferredMcsRowTables, MetalSession};
use crate::{KWords, MetalError};

mod nc_mask;

const NC_THREADS: usize = 64;
const NC_DENSE_PAIRS_PER_GROUP: usize = 2;
const NC_MASK_DENSE_CROSSOVER: usize = 64;

fn nc_partial_groups(rows: usize, dense: bool) -> usize {
    let pairs_per_group = if dense { NC_DENSE_PAIRS_PER_GROUP } else { NC_THREADS };
    (rows / 2).div_ceil(pairs_per_group).max(1)
}

pub(crate) struct MetalFeSumcheckInputs<'a> {
    pub tables: &'a [MetalFeTableInput<'a>],
    pub shape: &'a [u64],
    pub mcs_headers: &'a [u64],
    pub mcs_table_indices: &'a [u64],
    pub gammas: &'a [u64],
    pub term_headers: &'a [u64],
    pub term_variables: &'a [u64],
    pub table_count: usize,
    pub coefficient_count: usize,
}

pub(crate) enum MetalFeTableInput<'a> {
    Host(&'a [K]),
    TensorPoint(&'a [K]),
    DeferredMcs {
        tables: &'a MetalDeferredMcsRowTables,
        table: usize,
    },
    DeferredEval(&'a MetalDeferredEvalTable),
}

pub(crate) enum MetalFeSumcheckPlan {
    Packed(MetalPackedFeSumcheckPlan),
    Streaming(super::fe_streaming::MetalStreamingFePlan),
}

pub(crate) struct MetalPackedFeSumcheckPlan {
    tables: [Buffer; 2],
    shape: Buffer,
    mcs_headers: Buffer,
    mcs_table_indices: Buffer,
    gammas: Buffer,
    term_headers: Buffer,
    term_variables: Buffer,
    challenge: Buffer,
    partials: Buffer,
    reduction_shape: Buffer,
    output: Buffer,
    round_shapes: Buffer,
    round_reduction_shapes: Buffer,
    challenge_log: Buffer,
    transcript_state: Buffer,
    transcript_shape: Buffer,
    table_count: usize,
    coefficient_count: usize,
    max_rounds: usize,
    current_len: usize,
    current_slot: usize,
}

pub(crate) struct MetalNcSumcheckInputs<'a> {
    pub eq_point: &'a [u64],
    pub digits: MetalNcDigitInput<'a>,
    pub resident_masks: Option<&'a MetalWitnessMasks>,
    pub weights: &'a [u64],
    pub witness_count: usize,
    pub rows: usize,
    pub width: usize,
    pub dense: bool,
}

#[derive(Clone, Copy)]
pub(crate) enum MetalNcDigitInput<'a> {
    Table(&'a [u64]),
    SignedMasks {
        words: &'a [u64],
        blocks: usize,
        active_rows: usize,
    },
}

struct MetalNcMaskSource {
    masks: Buffer,
    shape: Buffer,
    active_witnesses: Buffer,
    basis: [Buffer; 2],
    active_witnesses_host: Vec<u32>,
    source_witness_count: usize,
    basis_slot: usize,
    direct_compact: bool,
    round_encoded: bool,
    folded: bool,
}

#[derive(Clone)]
pub(crate) struct MetalWitnessMasks {
    words: Buffer,
    pub(super) witness_count: usize,
    blocks: usize,
    active_rows: usize,
    pub(super) active_witnesses: Vec<u32>,
}

impl MetalWitnessMasks {
    pub(super) fn from_buffer(
        words: Buffer,
        witness_count: usize,
        blocks: usize,
        active_rows: usize,
    ) -> Result<Self, MetalError> {
        let expected_bytes = witness_count
            .checked_mul(blocks)
            .and_then(|values| values.checked_mul(2))
            .and_then(|values| values.checked_mul(size_of::<u64>()))
            .ok_or(MetalError::Shape("witness mask dimensions overflow"))?;
        let scalar_columns = blocks
            .checked_mul(54)
            .ok_or(MetalError::Shape("witness mask column count overflow"))?;
        if witness_count == 0
            || blocks == 0
            || active_rows == 0
            || active_rows > scalar_columns
            || words.length() as usize != expected_bytes
        {
            return Err(MetalError::Shape("witness masks have inconsistent dimensions"));
        }
        Ok(Self {
            words,
            witness_count,
            blocks,
            active_rows,
            active_witnesses: (0..witness_count as u32).collect(),
        })
    }

    pub(crate) fn matches(&self, witness_count: usize, blocks: usize) -> bool {
        self.witness_count == witness_count && self.blocks == blocks
    }

    pub(super) fn contains(&self, witness: usize, blocks: usize) -> bool {
        witness < self.witness_count && self.blocks == blocks
    }

    pub(crate) fn matches_nc(&self, witness_count: usize, blocks: usize, active_rows: usize) -> bool {
        self.matches(witness_count, blocks) && self.active_rows == active_rows
    }

    pub(super) fn words(&self) -> &Buffer {
        &self.words
    }
}

pub(crate) struct MetalNcSumcheckPlan {
    eq_tables: [Buffer; 2],
    digit_values: [Buffer; 2],
    mask_source: Option<MetalNcMaskSource>,
    weights: Buffer,
    shape: Buffer,
    fold_shape: Buffer,
    challenge: Buffer,
    partials: Buffer,
    reduction_shape: Buffer,
    output: Buffer,
    round_shapes: Buffer,
    round_fold_shapes: Buffer,
    round_reduction_shapes: Buffer,
    challenge_log: Buffer,
    transcript_state: Buffer,
    transcript_shape: Buffer,
    witness_count: usize,
    active_witness_count: usize,
    max_rounds: usize,
    rows: usize,
    width: usize,
    dense: bool,
    current_slot: usize,
}

impl MetalNcSumcheckPlan {
    pub(super) fn signed_mask_buffer(&self, witness_count: usize, blocks: usize) -> Option<Buffer> {
        let expected_bytes = witness_count
            .checked_mul(blocks)?
            .checked_mul(2)?
            .checked_mul(std::mem::size_of::<u64>())?;
        let source = self.mask_source.as_ref()?;
        (source.source_witness_count == witness_count && source.masks.length() as usize == expected_bytes)
            .then(|| source.masks.clone())
    }

    pub(crate) fn active_witness_count(&self) -> usize {
        self.active_witness_count
    }

    fn can_reset_from_signed_masks(
        &self,
        inputs: &MetalNcSumcheckInputs<'_>,
        active_witness_count: usize,
        eq_bytes: usize,
        digit_bytes: usize,
        mask_bytes: usize,
    ) -> bool {
        let Some(source) = self.mask_source.as_ref() else {
            return false;
        };
        self.witness_count == inputs.witness_count
            && self.active_witness_count == active_witness_count
            && self.max_rounds == inputs.rows.ilog2() as usize
            && self
                .eq_tables
                .iter()
                .all(|buffer| buffer.length() as usize == eq_bytes)
            && self
                .digit_values
                .iter()
                .all(|buffer| buffer.length() as usize == digit_bytes)
            && self.weights.length() as usize == active_witness_count * 54 * 2 * size_of::<u64>()
            && source.masks.length() as usize == mask_bytes
            && source
                .basis
                .iter()
                .all(|buffer| buffer.length() as usize == NC_MASK_DENSE_CROSSOVER * 2 * size_of::<u64>())
            && source.active_witnesses.length() as usize == active_witness_count * size_of::<u32>()
    }

    fn digit_workspace_bytes(&self) -> usize {
        self.digit_values
            .iter()
            .map(|buffer| buffer.length() as usize)
            .sum()
    }
}

pub(crate) struct MetalNcFinalState {
    pub eq_beta: KWords,
    pub digit_words: Vec<u64>,
    pub width: usize,
    pub dense: bool,
}

pub(crate) struct MetalSumcheckTrace {
    pub coeffs: Vec<Vec<KWords>>,
    pub challenges: Vec<KWords>,
    pub transcript_state: [u64; 8],
    pub transcript_absorbed: usize,
}

pub(crate) struct MetalNcSumcheckTrace {
    pub rounds: MetalSumcheckTrace,
    pub final_state: MetalNcFinalState,
}

impl MetalSession {
    pub(crate) fn prepare_witness_masks(
        &self,
        words: &[u64],
        witness_count: usize,
        blocks: usize,
        active_rows: usize,
    ) -> Result<MetalWitnessMasks, MetalError> {
        let expected_words = witness_count
            .checked_mul(blocks)
            .and_then(|values| values.checked_mul(2))
            .ok_or(MetalError::Shape("witness mask dimensions overflow"))?;
        if words.len() != expected_words {
            return Err(MetalError::Shape("witness masks have inconsistent dimensions"));
        }
        let active_witnesses = words
            .chunks_exact(2 * blocks)
            .enumerate()
            .filter(|(_, masks)| masks.iter().any(|&mask| mask != 0))
            .map(|(witness, _)| witness as u32)
            .collect();
        MetalWitnessMasks::from_buffer(self.buffer_from_slice(words)?, witness_count, blocks, active_rows)?
            .with_active_witnesses(active_witnesses)
    }

    pub(crate) fn prepare_fe_sumcheck(
        &self,
        inputs: MetalFeSumcheckInputs<'_>,
    ) -> Result<MetalFeSumcheckPlan, MetalError> {
        if inputs
            .tables
            .iter()
            .any(|source| matches!(source, MetalFeTableInput::DeferredMcs { .. }))
        {
            return self
                .prepare_streaming_fe_sumcheck(inputs)
                .map(MetalFeSumcheckPlan::Streaming);
        }
        if inputs.shape.len() < 13 || inputs.table_count == 0 || inputs.coefficient_count == 0 || inputs.shape[3] >= 10
        {
            return Err(MetalError::Shape("resident FE sumcheck metadata is invalid"));
        }
        let current_len = inputs.shape[0] as usize;
        let active_len = inputs.shape[1] as usize;
        let expected_words = inputs
            .table_count
            .checked_mul(current_len)
            .and_then(|values| values.checked_mul(2))
            .ok_or(MetalError::Shape("resident FE table dimensions overflow"))?;
        if current_len < 2
            || !current_len.is_power_of_two()
            || active_len == 0
            || active_len > current_len
            || inputs.tables.len() != inputs.table_count
        {
            return Err(MetalError::Shape("resident FE table dimensions are invalid"));
        }

        for source in inputs.tables {
            let valid = match source {
                MetalFeTableInput::Host(values) => values.len() == current_len,
                MetalFeTableInput::TensorPoint(point) => {
                    point.len() < usize::BITS as usize && 1usize << point.len() == current_len
                }
                MetalFeTableInput::DeferredMcs { tables, table } => {
                    tables.n_pad() == current_len && *table < tables.table_count()
                }
                MetalFeTableInput::DeferredEval(table) => table.matches(current_len),
            };
            if !valid {
                return Err(MetalError::Shape("resident FE table source is invalid"));
            }
        }

        let table_bytes = 2 * current_len * size_of::<u64>();
        let tables_a = self.buffer(expected_words * size_of::<u64>())?;
        let tables_b = self.buffer(expected_words * size_of::<u64>() / 2)?;
        let command = self.command_buffer("nightstream.pi_ccs.fe_install_tables")?;
        for (slot, source) in inputs.tables.iter().enumerate() {
            let destination_offset = slot * table_bytes;
            match source {
                MetalFeTableInput::Host(values) => {
                    self.write_k_table_at(&tables_a, destination_offset, values)?;
                }
                MetalFeTableInput::TensorPoint(point) => {
                    let words = point
                        .iter()
                        .flat_map(|value| {
                            let (real, imaginary) = value.to_limbs_u64();
                            [real, imaginary]
                        })
                        .collect::<Vec<_>>();
                    let challenges = self.buffer_from_slice(&words)?;
                    let stages = self.buffer_from_slice(&(0..point.len() as u64).collect::<Vec<_>>())?;
                    for stage in 0..point.len() {
                        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                        encoder.setComputePipelineState(&self.tensor_point_expand_k);
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(&challenges), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(&stages), stage * size_of::<u64>(), 1);
                            encoder.setBuffer_offset_atIndex(Some(&tables_a), destination_offset, 2);
                        }
                        self.dispatch(&encoder, &self.tensor_point_expand_k, 1usize << stage);
                        encoder.endEncoding();
                    }
                }
                MetalFeTableInput::DeferredMcs { tables, table } => {
                    let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                    encoder.setComputePipelineState(&self.copy_base_to_k);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(
                            Some(tables.words()),
                            table * current_len * size_of::<u64>(),
                            0,
                        );
                        encoder.setBuffer_offset_atIndex(Some(&tables_a), destination_offset, 1);
                    }
                    self.dispatch(&encoder, &self.copy_base_to_k, current_len);
                    encoder.endEncoding();
                }
                MetalFeTableInput::DeferredEval(table) => {
                    let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                    encoder.setComputePipelineState(&self.copy_k_words);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(table.words()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(&tables_a), destination_offset, 1);
                    }
                    self.dispatch(&encoder, &self.copy_k_words, current_len);
                    encoder.endEncoding();
                }
            }
        }
        self.submit(&command);
        let groups = active_len.div_ceil(2).div_ceil(64).max(1);
        let max_rounds = current_len.ilog2() as usize;
        Ok(MetalFeSumcheckPlan::Packed(MetalPackedFeSumcheckPlan {
            tables: [tables_a, tables_b],
            shape: self.buffer_from_slice(inputs.shape)?,
            mcs_headers: self.buffer_from_slice(super::nonempty(inputs.mcs_headers))?,
            mcs_table_indices: self.buffer_from_slice(super::nonempty(inputs.mcs_table_indices))?,
            gammas: self.buffer_from_slice(super::nonempty(inputs.gammas))?,
            term_headers: self.buffer_from_slice(super::nonempty(inputs.term_headers))?,
            term_variables: self.buffer_from_slice(super::nonempty(inputs.term_variables))?,
            challenge: self.buffer_from_slice(&[0u64; 2])?,
            partials: self.buffer(groups * inputs.coefficient_count * 2 * size_of::<u64>())?,
            reduction_shape: self.buffer_from_slice(&[groups as u64, inputs.coefficient_count as u64])?,
            output: self.buffer(max_rounds * inputs.coefficient_count * 2 * size_of::<u64>())?,
            round_shapes: self.buffer(max_rounds * 13 * size_of::<u64>())?,
            round_reduction_shapes: self.buffer(max_rounds * 2 * size_of::<u64>())?,
            challenge_log: self.buffer(max_rounds * 2 * size_of::<u64>())?,
            transcript_state: self.buffer(9 * size_of::<u64>())?,
            transcript_shape: self.buffer_from_slice(&[(inputs.coefficient_count * 2) as u64])?,
            table_count: inputs.table_count,
            coefficient_count: inputs.coefficient_count,
            max_rounds,
            current_len,
            current_slot: 0,
        }))
    }

    pub(crate) fn fe_sumcheck_round(
        &self,
        plan: &mut MetalFeSumcheckPlan,
        shape: &[u64],
        fold_challenge: Option<KWords>,
    ) -> Result<Vec<KWords>, MetalError> {
        match plan {
            MetalFeSumcheckPlan::Packed(plan) => self.packed_fe_sumcheck_round(plan, shape, fold_challenge),
            MetalFeSumcheckPlan::Streaming(plan) => self.streaming_fe_sumcheck_round(plan, shape, fold_challenge),
        }
    }

    fn packed_fe_sumcheck_round(
        &self,
        plan: &mut MetalPackedFeSumcheckPlan,
        shape: &[u64],
        fold_challenge: Option<KWords>,
    ) -> Result<Vec<KWords>, MetalError> {
        if shape.len() < 13 {
            return Err(MetalError::Shape("resident FE round shape is invalid"));
        }
        let command = self.command_buffer("nightstream.pi_ccs.fe_round")?;
        if let Some(challenge) = fold_challenge {
            self.encode_fe_fold(&command, plan, challenge)?;
        }
        if shape[0] as usize != plan.current_len {
            return Err(MetalError::Shape("resident FE round length is out of sequence"));
        }
        let active_len = shape[1] as usize;
        if active_len == 0 || active_len > plan.current_len {
            return Err(MetalError::Shape("resident FE active length is invalid"));
        }
        let groups = active_len.div_ceil(2).div_ceil(64).max(1);
        self.write_shared(&plan.shape, shape)?;
        self.write_shared(&plan.reduction_shape, &[groups as u64, plan.coefficient_count as u64])?;

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.fe_round_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.tables[plan.current_slot]), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.mcs_headers), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&plan.mcs_table_indices), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&plan.gammas), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&plan.term_headers), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&plan.term_variables), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 7);
        }
        self.dispatch_threadgroups(&encoder, &self.fe_round_partials, groups, 64);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.sumcheck_reduce_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.reduction_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.output), 0, 2);
        }
        self.dispatch(&encoder, &self.sumcheck_reduce_partials, plan.coefficient_count);
        encoder.endEncoding();
        self.finish(&command)?;

        Ok(self
            .read_buffer::<u64>(&plan.output, plan.coefficient_count * 2)
            .chunks_exact(2)
            .map(|words| KWords::new(words[0], words[1]))
            .collect())
    }

    pub(crate) fn fe_sumcheck_trace(
        &self,
        plan: &mut MetalFeSumcheckPlan,
        base_shape: &[u64],
        transcript_state: [u64; 8],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Result<MetalSumcheckTrace, MetalError> {
        let started = std::time::Instant::now();
        let result = match plan {
            MetalFeSumcheckPlan::Packed(plan) => {
                self.packed_fe_sumcheck_trace(plan, base_shape, transcript_state, transcript_absorbed, rounds)
            }
            MetalFeSumcheckPlan::Streaming(plan) => {
                self.streaming_fe_sumcheck_trace(plan, base_shape, transcript_state, transcript_absorbed, rounds)
            }
        };
        self.fe_sumcheck_duration
            .set(self.fe_sumcheck_duration.get() + started.elapsed());
        result
    }

    fn packed_fe_sumcheck_trace(
        &self,
        plan: &mut MetalPackedFeSumcheckPlan,
        base_shape: &[u64],
        transcript_state: [u64; 8],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Result<MetalSumcheckTrace, MetalError> {
        if base_shape.len() < 13
            || rounds == 0
            || rounds > plan.max_rounds
            || transcript_absorbed > 4
            || plan.current_len >> rounds != 1
        {
            return Err(MetalError::Shape("resident FE trace dimensions are invalid"));
        }
        let mut shapes = Vec::with_capacity(rounds * 13);
        let mut reductions = Vec::with_capacity(rounds * 2);
        let mut current_len = plan.current_len;
        let mut active_len = base_shape[1] as usize;
        for _ in 0..rounds {
            let mut shape = base_shape[..13].to_vec();
            shape[0] = current_len as u64;
            shape[1] = active_len as u64;
            shapes.extend_from_slice(&shape);
            let groups = active_len.div_ceil(2).div_ceil(64).max(1);
            reductions.extend_from_slice(&[groups as u64, plan.coefficient_count as u64]);
            current_len /= 2;
            active_len = active_len.div_ceil(2).max(1);
        }
        self.write_shared(&plan.round_shapes, &shapes)?;
        self.write_shared(&plan.round_reduction_shapes, &reductions)?;
        let mut state_words = transcript_state.to_vec();
        state_words.push(transcript_absorbed as u64);
        self.write_shared(&plan.transcript_state, &state_words)?;

        let command = self.command_buffer("nightstream.pi_ccs.fe_trace")?;
        for round in 0..rounds {
            let groups = reductions[2 * round] as usize;
            let coeff_offset = round * plan.coefficient_count * 2 * size_of::<u64>();
            let challenge_offset = round * 2 * size_of::<u64>();

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.fe_round_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&plan.tables[plan.current_slot]), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.round_shapes), round * 13 * size_of::<u64>(), 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.mcs_headers), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&plan.mcs_table_indices), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&plan.gammas), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&plan.term_headers), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&plan.term_variables), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 7);
            }
            self.dispatch_threadgroups(&encoder, &self.fe_round_partials, groups, 64);
            encoder.endEncoding();

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.sumcheck_reduce_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.round_reduction_shapes), round * 2 * size_of::<u64>(), 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.output), coeff_offset, 2);
            }
            self.dispatch(&encoder, &self.sumcheck_reduce_partials, plan.coefficient_count);
            encoder.endEncoding();

            self.encode_transcript_challenge(
                &command,
                &plan.transcript_state,
                &plan.output,
                coeff_offset,
                &plan.challenge_log,
                challenge_offset,
                &plan.transcript_shape,
            )?;

            let next_slot = plan.current_slot ^ 1;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.fold_k_table);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&plan.tables[plan.current_slot]), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.challenge_log), challenge_offset, 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.tables[next_slot]), 0, 2);
            }
            self.dispatch(&encoder, &self.fold_k_table, plan.table_count * (plan.current_len / 2));
            encoder.endEncoding();
            plan.current_slot = next_slot;
            plan.current_len /= 2;
        }
        self.finish(&command)?;

        self.read_sumcheck_trace(
            &plan.output,
            &plan.challenge_log,
            &plan.transcript_state,
            plan.coefficient_count,
            rounds,
        )
    }

    fn encode_fe_fold(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        plan: &mut MetalPackedFeSumcheckPlan,
        challenge: KWords,
    ) -> Result<(), MetalError> {
        if plan.current_len < 2 {
            return Err(MetalError::Shape("resident FE fold exhausted its table"));
        }
        self.write_shared(&plan.challenge, &[challenge.c0, challenge.c1])?;
        let next_slot = plan.current_slot ^ 1;
        let elements = plan.table_count * (plan.current_len / 2);
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.fold_k_table);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.tables[plan.current_slot]), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.challenge), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.tables[next_slot]), 0, 2);
        }
        self.dispatch(&encoder, &self.fold_k_table, elements);
        encoder.endEncoding();
        plan.current_slot = next_slot;
        plan.current_len /= 2;
        Ok(())
    }

    fn install_nc_eq_tensor(&self, output: &Buffer, point_words: &[u64], rows: usize) -> Result<(), MetalError> {
        let rounds = rows.ilog2() as usize;
        let expected_bytes = rows
            .checked_mul(2)
            .and_then(|words| words.checked_mul(size_of::<u64>()))
            .ok_or(MetalError::Shape("resident NC equality table byte size overflow"))?;
        if point_words.len() != 2 * rounds || output.length() as usize != expected_bytes {
            return Err(MetalError::Shape("resident NC equality point has invalid dimensions"));
        }
        let challenges = self.buffer_from_slice(point_words)?;
        let stages = self.buffer_from_slice(&(0..rounds as u64).collect::<Vec<_>>())?;
        let command = self.command_buffer("nightstream.pi_ccs.nc.eq_tensor")?;
        for stage in 0..rounds {
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.tensor_point_expand_k);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&challenges), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&stages), stage * size_of::<u64>(), 1);
                encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
            }
            self.dispatch(&encoder, &self.tensor_point_expand_k, 1usize << stage);
            encoder.endEncoding();
        }
        self.submit(&command);
        Ok(())
    }

    pub(crate) fn prepare_nc_sumcheck(
        &self,
        inputs: MetalNcSumcheckInputs<'_>,
    ) -> Result<MetalNcSumcheckPlan, MetalError> {
        if inputs.witness_count == 0 || inputs.rows < 2 || !inputs.rows.is_power_of_two() || inputs.width == 0 {
            return Err(MetalError::Shape("resident NC sumcheck dimensions are invalid"));
        }
        let max_rounds = inputs.rows.ilog2() as usize;
        let eq_words = inputs
            .rows
            .checked_mul(2)
            .ok_or(MetalError::Shape("resident NC equality table dimensions overflow"))?;
        let eq_bytes = eq_words
            .checked_mul(size_of::<u64>())
            .ok_or(MetalError::Shape("resident NC equality table byte size overflow"))?;
        let values_per_witness = if inputs.dense {
            inputs
                .rows
                .checked_mul(54)
                .ok_or(MetalError::Shape("resident NC table dimensions overflow"))?
        } else {
            inputs
                .rows
                .checked_mul(inputs.width)
                .ok_or(MetalError::Shape("resident NC table dimensions overflow"))?
        };
        let source_digit_words = inputs
            .witness_count
            .checked_mul(values_per_witness)
            .and_then(|values| values.checked_mul(2))
            .ok_or(MetalError::Shape("resident NC table dimensions overflow"))?;
        let weight_words = inputs
            .witness_count
            .checked_mul(54)
            .and_then(|values| values.checked_mul(2))
            .ok_or(MetalError::Shape("resident NC weight dimensions overflow"))?;
        if inputs.eq_point.len() != max_rounds * 2 || inputs.weights.len() != weight_words {
            return Err(MetalError::Shape("resident NC input lengths are invalid"));
        }
        let mask_input = match inputs.digits {
            MetalNcDigitInput::Table(words) => {
                if words.len() != source_digit_words {
                    return Err(MetalError::Shape("resident NC digit table length is invalid"));
                }
                None
            }
            MetalNcDigitInput::SignedMasks {
                words,
                blocks,
                active_rows,
            } => {
                let expected_masks = inputs
                    .witness_count
                    .checked_mul(blocks)
                    .and_then(|values| values.checked_mul(2))
                    .ok_or(MetalError::Shape("resident NC mask dimensions overflow"))?;
                if inputs.dense
                    || inputs.width != 1
                    || active_rows == 0
                    || active_rows > inputs.rows
                    || blocks != active_rows.div_ceil(54)
                    || (words.len() != expected_masks && !(words.is_empty() && inputs.resident_masks.is_some()))
                {
                    return Err(MetalError::Shape("resident NC signed masks are invalid"));
                }
                Some((words, blocks, active_rows))
            }
        };
        if let (Some(resident), Some((_, blocks, active_rows))) = (inputs.resident_masks, mask_input) {
            if !resident.matches_nc(inputs.witness_count, blocks, active_rows) {
                return Err(MetalError::Shape("resident NC masks do not match the host source"));
            }
        } else if inputs.resident_masks.is_some() {
            return Err(MetalError::Shape("resident NC masks require a signed-mask source"));
        }

        let mut active_witnesses = if let Some(resident) = inputs.resident_masks {
            resident.active_witnesses().to_vec()
        } else {
            match mask_input {
                Some((words, blocks, _)) => words
                    .chunks_exact(2 * blocks)
                    .enumerate()
                    .filter(|(_, masks)| masks.iter().any(|&mask| mask != 0))
                    .map(|(witness, _)| {
                        u32::try_from(witness).map_err(|_| MetalError::Shape("resident NC witness index exceeds u32"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                None => (0..inputs.witness_count)
                    .map(|witness| {
                        u32::try_from(witness).map_err(|_| MetalError::Shape("resident NC witness index exceeds u32"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            }
        };
        if active_witnesses.is_empty() {
            active_witnesses.push(0);
        }
        let active_witness_count = active_witnesses.len();
        let direct_compact = mask_input.is_some() && inputs.rows >= NC_MASK_DENSE_CROSSOVER;
        let workspace_values_per_witness = if direct_compact {
            (inputs.rows / NC_MASK_DENSE_CROSSOVER) * 54
        } else {
            values_per_witness
        };
        let digit_words = active_witness_count
            .checked_mul(workspace_values_per_witness)
            .and_then(|values| values.checked_mul(2))
            .ok_or(MetalError::Shape("resident NC active table dimensions overflow"))?;
        let digit_bytes = digit_words
            .checked_mul(size_of::<u64>())
            .ok_or(MetalError::Shape("resident NC table byte size overflow"))?;
        let active_weights = active_witnesses
            .iter()
            .flat_map(|&witness| {
                let start = witness as usize * 54 * 2;
                inputs.weights[start..start + 54 * 2].iter().copied()
            })
            .collect::<Vec<_>>();

        if let Some((words, blocks, active_rows)) = mask_input {
            let mask_bytes = inputs.resident_masks.map_or_else(
                || std::mem::size_of_val(words),
                |resident| resident.words.length() as usize,
            );
            let mut recycled = {
                let mut slot = self.recycled_nc_plan.borrow_mut();
                slot.as_ref()
                    .is_some_and(|plan| {
                        plan.can_reset_from_signed_masks(
                            &inputs,
                            active_witness_count,
                            eq_bytes,
                            digit_bytes,
                            mask_bytes,
                        )
                    })
                    .then(|| slot.take().expect("recycled NC plan exists above"))
            };
            if let Some(plan) = recycled.as_mut() {
                self.install_nc_eq_tensor(&plan.eq_tables[0], inputs.eq_point, inputs.rows)?;
                self.write_shared(&plan.weights, &active_weights)?;
                let source = plan
                    .mask_source
                    .as_mut()
                    .expect("recycled signed-mask NC plan has a mask source");
                if let Some(resident) = inputs.resident_masks {
                    source.masks = resident.words.clone();
                } else {
                    self.write_shared(&source.masks, words)?;
                }
                self.write_shared(
                    &source.shape,
                    &[
                        inputs.rows as u64,
                        active_witness_count as u64,
                        blocks as u64,
                        active_rows as u64,
                    ],
                )?;
                self.write_shared(&source.active_witnesses, &active_witnesses)?;
                self.write_shared(&source.basis[0], &[1u64, 0])?;
                source.active_witnesses_host.clone_from(&active_witnesses);
                source.source_witness_count = inputs.witness_count;
                source.basis_slot = 0;
                source.direct_compact = direct_compact;
                source.round_encoded = false;
                source.folded = false;
                plan.active_witness_count = active_witness_count;
                plan.rows = inputs.rows;
                plan.width = inputs.width;
                plan.dense = inputs.dense;
                plan.current_slot = 0;
                return Ok(recycled.expect("recycled NC plan was reset above"));
            }
        }

        let (initial_digits, mask_source) = match inputs.digits {
            MetalNcDigitInput::Table(words) => (self.buffer_from_slice(words)?, None),
            MetalNcDigitInput::SignedMasks {
                words,
                blocks,
                active_rows,
            } => {
                let basis = [
                    self.buffer(NC_MASK_DENSE_CROSSOVER * 2 * size_of::<u64>())?,
                    self.buffer(NC_MASK_DENSE_CROSSOVER * 2 * size_of::<u64>())?,
                ];
                self.write_shared(&basis[0], &[1u64, 0])?;
                (
                    self.buffer(digit_bytes)?,
                    Some(MetalNcMaskSource {
                        masks: match inputs.resident_masks {
                            Some(resident) => resident.words.clone(),
                            None => self.buffer_from_slice(words)?,
                        },
                        shape: self.buffer_from_slice(&[
                            inputs.rows as u64,
                            active_witness_count as u64,
                            blocks as u64,
                            active_rows as u64,
                        ])?,
                        active_witnesses: self.buffer_from_slice(&active_witnesses)?,
                        basis,
                        active_witnesses_host: active_witnesses.clone(),
                        source_witness_count: inputs.witness_count,
                        basis_slot: 0,
                        direct_compact,
                        round_encoded: false,
                        folded: false,
                    }),
                )
            }
        };
        let groups = nc_partial_groups(inputs.rows, inputs.dense);
        let shape = [
            inputs.rows as u64,
            active_witness_count as u64,
            inputs.width as u64,
            u64::from(inputs.dense),
            values_per_witness as u64,
        ];
        let plan = MetalNcSumcheckPlan {
            eq_tables: [self.buffer(eq_bytes)?, self.buffer(eq_bytes)?],
            digit_values: [initial_digits, self.buffer(digit_bytes)?],
            mask_source,
            weights: self.buffer_from_slice(&active_weights)?,
            shape: self.buffer_from_slice(&shape)?,
            fold_shape: self.buffer_from_slice(&[0u64; 4])?,
            challenge: self.buffer_from_slice(&[0u64; 2])?,
            partials: self.buffer(groups * 5 * 2 * size_of::<u64>())?,
            reduction_shape: self.buffer_from_slice(&[groups as u64, 5])?,
            output: self.buffer(max_rounds * 10 * size_of::<u64>())?,
            round_shapes: self.buffer(max_rounds * 5 * size_of::<u64>())?,
            round_fold_shapes: self.buffer(max_rounds * 4 * size_of::<u64>())?,
            round_reduction_shapes: self.buffer(max_rounds * 2 * size_of::<u64>())?,
            challenge_log: self.buffer(max_rounds * 2 * size_of::<u64>())?,
            transcript_state: self.buffer(9 * size_of::<u64>())?,
            transcript_shape: self.buffer_from_slice(&[10u64])?,
            witness_count: inputs.witness_count,
            active_witness_count,
            max_rounds,
            rows: inputs.rows,
            width: inputs.width,
            dense: inputs.dense,
            current_slot: 0,
        };
        self.install_nc_eq_tensor(&plan.eq_tables[0], inputs.eq_point, inputs.rows)?;
        Ok(plan)
    }

    pub(crate) fn recycle_nc_sumcheck(&self, plan: MetalNcSumcheckPlan) {
        if plan.mask_source.is_none() {
            return;
        }
        let workspace_bytes = plan.digit_workspace_bytes();
        let mut slot = self.recycled_nc_plan.borrow_mut();
        if slot
            .as_ref()
            .is_none_or(|cached| cached.digit_workspace_bytes() <= workspace_bytes)
        {
            *slot = Some(plan);
        }
    }

    pub(crate) fn nc_sumcheck_round(
        &self,
        plan: &mut MetalNcSumcheckPlan,
        shape: &[u64],
        fold_challenge: Option<KWords>,
    ) -> Result<Vec<KWords>, MetalError> {
        if shape.len() < 5 {
            return Err(MetalError::Shape("resident NC round shape is invalid"));
        }
        if fold_challenge.is_none()
            && plan
                .mask_source
                .as_ref()
                .is_some_and(|source| source.round_encoded && !source.folded)
        {
            return Err(MetalError::Shape("resident NC mask round challenge was not consumed"));
        }
        let command = self.command_buffer("nightstream.pi_ccs.nc_round")?;
        if let Some(challenge) = fold_challenge {
            if plan
                .mask_source
                .as_ref()
                .is_some_and(|source| !source.folded)
            {
                self.encode_nc_mask_fold(&command, plan, challenge)?;
            } else {
                self.encode_nc_fold(&command, plan, challenge)?;
            }
        }
        let values_per_witness = if plan.dense {
            plan.rows * 54
        } else {
            plan.rows * plan.width
        };
        let expected = [
            plan.rows as u64,
            plan.witness_count as u64,
            plan.width as u64,
            u64::from(plan.dense),
            values_per_witness as u64,
        ];
        if shape[..5] != expected {
            return Err(MetalError::Shape("resident NC round shape is out of sequence"));
        }
        let device_shape = [
            plan.rows as u64,
            plan.active_witness_count as u64,
            plan.width as u64,
            u64::from(plan.dense),
            values_per_witness as u64,
        ];
        let groups = nc_partial_groups(plan.rows, plan.dense);
        self.write_shared(&plan.shape, &device_shape)?;
        self.write_shared(&plan.reduction_shape, &[groups as u64, 5])?;

        if plan
            .mask_source
            .as_ref()
            .is_some_and(|source| !source.folded)
        {
            self.encode_nc_mask_round(&command, plan, None)?;
        } else {
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.nc_round_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&plan.eq_tables[plan.current_slot]), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.shape), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[plan.current_slot]), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&plan.weights), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 4);
            }
            self.dispatch_threadgroups(&encoder, &self.nc_round_partials, groups, NC_THREADS);
            encoder.endEncoding();
        }

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.sumcheck_reduce_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.reduction_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.output), 0, 2);
        }
        self.dispatch(&encoder, &self.sumcheck_reduce_partials, 5);
        encoder.endEncoding();
        self.finish(&command)?;

        Ok(self
            .read_buffer::<u64>(&plan.output, 10)
            .chunks_exact(2)
            .map(|words| KWords::new(words[0], words[1]))
            .collect())
    }

    pub(crate) fn nc_sumcheck_trace(
        &self,
        plan: &mut MetalNcSumcheckPlan,
        transcript_state: [u64; 8],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Result<MetalNcSumcheckTrace, MetalError> {
        let started = std::time::Instant::now();
        let result = self.nc_sumcheck_trace_inner(plan, transcript_state, transcript_absorbed, rounds);
        self.nc_sumcheck_duration
            .set(self.nc_sumcheck_duration.get() + started.elapsed());
        result
    }

    fn nc_sumcheck_trace_inner(
        &self,
        plan: &mut MetalNcSumcheckPlan,
        transcript_state: [u64; 8],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Result<MetalNcSumcheckTrace, MetalError> {
        if rounds == 0 || rounds > plan.max_rounds || transcript_absorbed > 4 || plan.rows >> rounds != 1 {
            return Err(MetalError::Shape("resident NC trace dimensions are invalid"));
        }
        let mut shapes = Vec::with_capacity(rounds * 5);
        let mut fold_shapes = Vec::with_capacity(rounds * 4);
        let mut reductions = Vec::with_capacity(rounds * 2);
        let mut rows = plan.rows;
        let mut width = plan.width;
        let mut dense = plan.dense;
        for _ in 0..rounds {
            let values_per_witness = if dense { rows * 54 } else { rows * width };
            shapes.extend_from_slice(&[
                rows as u64,
                plan.active_witness_count as u64,
                width as u64,
                u64::from(dense),
                values_per_witness as u64,
            ]);
            fold_shapes.extend_from_slice(&[
                plan.active_witness_count as u64,
                rows as u64,
                width as u64,
                u64::from(dense),
            ]);
            reductions.extend_from_slice(&[nc_partial_groups(rows, dense) as u64, 5]);
            rows /= 2;
            dense = dense || 2 * width > 54;
            width = if dense { 54 } else { 2 * width };
        }
        self.write_shared(&plan.round_shapes, &shapes)?;
        self.write_shared(&plan.round_fold_shapes, &fold_shapes)?;
        self.write_shared(&plan.round_reduction_shapes, &reductions)?;
        let mut state_words = transcript_state.to_vec();
        state_words.push(transcript_absorbed as u64);
        self.write_shared(&plan.transcript_state, &state_words)?;

        let command = self.command_buffer("nightstream.pi_ccs.nc_trace")?;
        for round in 0..rounds {
            let coeff_offset = round * 10 * size_of::<u64>();
            let challenge_offset = round * 2 * size_of::<u64>();
            let groups = reductions[2 * round] as usize;

            if plan
                .mask_source
                .as_ref()
                .is_some_and(|source| !source.folded)
            {
                self.encode_nc_mask_round(&command, plan, Some(round))?;
            } else {
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.nc_round_partials);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&plan.eq_tables[plan.current_slot]), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&plan.round_shapes), round * 5 * size_of::<u64>(), 1);
                    encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[plan.current_slot]), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&plan.weights), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 4);
                }
                self.dispatch_threadgroups(&encoder, &self.nc_round_partials, groups, NC_THREADS);
                encoder.endEncoding();
            }

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.sumcheck_reduce_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.round_reduction_shapes), round * 2 * size_of::<u64>(), 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.output), coeff_offset, 2);
            }
            self.dispatch(&encoder, &self.sumcheck_reduce_partials, 5);
            encoder.endEncoding();

            self.encode_transcript_challenge(
                &command,
                &plan.transcript_state,
                &plan.output,
                coeff_offset,
                &plan.challenge_log,
                challenge_offset,
                &plan.transcript_shape,
            )?;

            if plan
                .mask_source
                .as_ref()
                .is_some_and(|source| !source.folded)
            {
                self.encode_nc_mask_trace_fold(&command, plan, round, challenge_offset)?;
            } else {
                let next_slot = plan.current_slot ^ 1;
                let next_rows = plan.rows / 2;
                let next_dense = plan.dense || 2 * plan.width > 54;
                let next_width = if next_dense { 54 } else { 2 * plan.width };
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.fold_k_table);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&plan.eq_tables[plan.current_slot]), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&plan.challenge_log), challenge_offset, 1);
                    encoder.setBuffer_offset_atIndex(Some(&plan.eq_tables[next_slot]), 0, 2);
                }
                self.dispatch(&encoder, &self.fold_k_table, next_rows);
                encoder.endEncoding();

                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.nc_fold_compact);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[plan.current_slot]), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&plan.challenge_log), challenge_offset, 1);
                    encoder.setBuffer_offset_atIndex(Some(&plan.round_fold_shapes), round * 4 * size_of::<u64>(), 2);
                    encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[next_slot]), 0, 3);
                }
                self.dispatch(
                    &encoder,
                    &self.nc_fold_compact,
                    plan.active_witness_count * next_rows * next_width,
                );
                encoder.endEncoding();

                plan.current_slot = next_slot;
                plan.rows = next_rows;
                plan.width = next_width;
                plan.dense = next_dense;
            }
        }
        self.finish(&command)?;

        let rounds = self.read_sumcheck_trace(&plan.output, &plan.challenge_log, &plan.transcript_state, 5, rounds)?;
        let eq = self.read_buffer::<u64>(&plan.eq_tables[plan.current_slot], 2);
        let values_per_witness = if plan.dense { 54 } else { plan.width };
        Ok(MetalNcSumcheckTrace {
            rounds,
            final_state: MetalNcFinalState {
                eq_beta: KWords::new(eq[0], eq[1]),
                digit_words: self.read_nc_final_digits(plan, values_per_witness),
                width: plan.width,
                dense: plan.dense,
            },
        })
    }

    fn read_nc_final_digits(&self, plan: &MetalNcSumcheckPlan, values_per_witness: usize) -> Vec<u64> {
        let words_per_witness = values_per_witness * 2;
        let active = self.read_buffer::<u64>(
            &plan.digit_values[plan.current_slot],
            plan.active_witness_count * words_per_witness,
        );
        if plan.active_witness_count == plan.witness_count {
            return active;
        }
        let indices = &plan
            .mask_source
            .as_ref()
            .expect("compacted NC witnesses require a signed-mask source")
            .active_witnesses_host;
        let mut full = vec![0u64; plan.witness_count * words_per_witness];
        for (active_witness, &source_witness) in indices.iter().enumerate() {
            let source = active_witness * words_per_witness;
            let destination = source_witness as usize * words_per_witness;
            full[destination..destination + words_per_witness]
                .copy_from_slice(&active[source..source + words_per_witness]);
        }
        full
    }

    pub(crate) fn finalize_nc_sumcheck(
        &self,
        plan: &mut MetalNcSumcheckPlan,
        fold_challenge: Option<KWords>,
    ) -> Result<MetalNcFinalState, MetalError> {
        if let Some(challenge) = fold_challenge {
            let command = self.command_buffer("nightstream.pi_ccs.nc_finalize")?;
            if plan
                .mask_source
                .as_ref()
                .is_some_and(|source| !source.folded)
            {
                self.encode_nc_mask_fold(&command, plan, challenge)?;
            } else {
                self.encode_nc_fold(&command, plan, challenge)?;
            }
            self.finish(&command)?;
        }
        if plan.rows != 1 {
            return Err(MetalError::Shape("resident NC sumcheck finalized before one row"));
        }
        let eq = self.read_buffer::<u64>(&plan.eq_tables[plan.current_slot], 2);
        let values_per_witness = if plan.dense { 54 } else { plan.width };
        Ok(MetalNcFinalState {
            eq_beta: KWords::new(eq[0], eq[1]),
            digit_words: self.read_nc_final_digits(plan, values_per_witness),
            width: plan.width,
            dense: plan.dense,
        })
    }

    fn encode_nc_fold(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        plan: &mut MetalNcSumcheckPlan,
        challenge: KWords,
    ) -> Result<(), MetalError> {
        if plan.rows < 2 {
            return Err(MetalError::Shape("resident NC fold exhausted its table"));
        }
        self.write_shared(&plan.challenge, &[challenge.c0, challenge.c1])?;
        let next_slot = plan.current_slot ^ 1;
        let next_rows = plan.rows.div_ceil(2);
        let next_dense = plan.dense || 2 * plan.width > 54;
        let next_width = if next_dense { 54 } else { 2 * plan.width };

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.fold_k_table);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.eq_tables[plan.current_slot]), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.challenge), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.eq_tables[next_slot]), 0, 2);
        }
        self.dispatch(&encoder, &self.fold_k_table, next_rows);
        encoder.endEncoding();

        let fold_shape = [
            plan.active_witness_count as u64,
            plan.rows as u64,
            plan.width as u64,
            u64::from(plan.dense),
        ];
        self.write_shared(&plan.fold_shape, &fold_shape)?;
        let output_elements = plan.active_witness_count * next_rows * next_width;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.nc_fold_compact);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[plan.current_slot]), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.challenge), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.fold_shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[next_slot]), 0, 3);
        }
        self.dispatch(&encoder, &self.nc_fold_compact, output_elements);
        encoder.endEncoding();

        plan.current_slot = next_slot;
        plan.rows = next_rows;
        plan.width = next_width;
        plan.dense = next_dense;
        Ok(())
    }

    pub(super) fn write_shared<T: Copy>(&self, buffer: &Buffer, values: &[T]) -> Result<(), MetalError> {
        let bytes = size_of_val(values);
        if bytes > buffer.length() as usize {
            return Err(MetalError::Shape("resident Metal metadata buffer is too small"));
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

    pub(super) fn write_k_table_at(&self, buffer: &Buffer, byte_offset: usize, values: &[K]) -> Result<(), MetalError> {
        let bytes = values
            .len()
            .checked_mul(2 * size_of::<u64>())
            .ok_or(MetalError::Shape("resident K table byte size overflow"))?;
        if byte_offset
            .checked_add(bytes)
            .is_none_or(|end| end > buffer.length() as usize)
        {
            return Err(MetalError::Shape("resident K table destination is too small"));
        }
        let destination = unsafe {
            buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(byte_offset)
                .cast::<u64>()
        };
        for (index, value) in values.iter().enumerate() {
            let (real, imaginary) = value.to_limbs_u64();
            unsafe {
                destination.add(2 * index).write(real);
                destination.add(2 * index + 1).write(imaginary);
            }
        }
        self.activity
            .uploaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_transcript_challenge(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        transcript_state: &Buffer,
        fields: &Buffer,
        fields_offset: usize,
        challenge: &Buffer,
        challenge_offset: usize,
        transcript_shape: &Buffer,
    ) -> Result<(), MetalError> {
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.transcript_absorb_challenge2);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(transcript_state), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(fields), fields_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(challenge), challenge_offset, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.poseidon2_constants), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(transcript_shape), 0, 4);
        }
        self.dispatch(&encoder, &self.transcript_absorb_challenge2, 1);
        encoder.endEncoding();
        Ok(())
    }

    pub(super) fn read_sumcheck_trace(
        &self,
        coefficient_buffer: &Buffer,
        challenge_buffer: &Buffer,
        transcript_buffer: &Buffer,
        coefficient_count: usize,
        rounds: usize,
    ) -> Result<MetalSumcheckTrace, MetalError> {
        let coefficient_words = self.read_buffer::<u64>(coefficient_buffer, rounds * coefficient_count * 2);
        let coeffs = coefficient_words
            .chunks_exact(coefficient_count * 2)
            .map(|round| {
                round
                    .chunks_exact(2)
                    .map(|words| KWords::new(words[0], words[1]))
                    .collect()
            })
            .collect();
        let challenges = self
            .read_buffer::<u64>(challenge_buffer, rounds * 2)
            .chunks_exact(2)
            .map(|words| KWords::new(words[0], words[1]))
            .collect();
        let transcript = self.read_buffer::<u64>(transcript_buffer, 9);
        Ok(MetalSumcheckTrace {
            coeffs,
            challenges,
            transcript_state: transcript[..8]
                .try_into()
                .map_err(|_| MetalError::Shape("resident transcript state has invalid width"))?,
            transcript_absorbed: transcript[8] as usize,
        })
    }
}
