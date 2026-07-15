//! Resident buffers and command encoding for the FE row sumcheck.

use std::mem::size_of;

use neo_math::{KExtensions, K};
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::MetalSumcheckTrace;
use crate::session::{Buffer, MetalDeferredEvalTable, MetalDeferredMcsRowTables, MetalSession};
use crate::{KWords, MetalError};

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
    Streaming(crate::session::fe_streaming::MetalStreamingFePlan),
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

impl MetalSession {
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
            mcs_headers: self.buffer_from_slice(crate::session::nonempty(inputs.mcs_headers))?,
            mcs_table_indices: self.buffer_from_slice(crate::session::nonempty(inputs.mcs_table_indices))?,
            gammas: self.buffer_from_slice(crate::session::nonempty(inputs.gammas))?,
            term_headers: self.buffer_from_slice(crate::session::nonempty(inputs.term_headers))?,
            term_variables: self.buffer_from_slice(crate::session::nonempty(inputs.term_variables))?,
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
}
