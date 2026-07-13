//! Resident FE and NC sumcheck buffers.
//!
//! Tables are uploaded once, folded through ping-pong buffers, and only the
//! round polynomial or final NC state crosses back to the host.

use std::sync::atomic::Ordering;

use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{Buffer, MetalSession};
use crate::{KWords, MetalError};

pub(crate) struct MetalFeSumcheckInputs<'a> {
    pub tables: &'a [u64],
    pub shape: &'a [u64],
    pub mcs_headers: &'a [u64],
    pub mcs_table_indices: &'a [u64],
    pub gammas: &'a [u64],
    pub term_headers: &'a [u64],
    pub term_variables: &'a [u64],
    pub table_count: usize,
    pub coefficient_count: usize,
}

pub(crate) struct MetalFeSumcheckPlan {
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
    pub eq_table: &'a [u64],
    pub digit_values: &'a [u64],
    pub weights: &'a [u64],
    pub witness_count: usize,
    pub rows: usize,
    pub width: usize,
    pub dense: bool,
}

pub(crate) struct MetalNcSumcheckPlan {
    eq_tables: [Buffer; 2],
    digit_values: [Buffer; 2],
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
    max_rounds: usize,
    rows: usize,
    width: usize,
    dense: bool,
    current_slot: usize,
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
    pub(crate) fn prepare_fe_sumcheck(
        &self,
        inputs: MetalFeSumcheckInputs<'_>,
    ) -> Result<MetalFeSumcheckPlan, MetalError> {
        if inputs.shape.len() < 13
            || inputs.table_count == 0
            || inputs.coefficient_count == 0
            || inputs.coefficient_count > 9
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
            || inputs.tables.len() != expected_words
        {
            return Err(MetalError::Shape("resident FE table dimensions are invalid"));
        }

        let tables_a = self.buffer_from_slice(inputs.tables)?;
        let tables_b = self.buffer(std::mem::size_of_val(inputs.tables))?;
        let groups = active_len.div_ceil(2).div_ceil(64).max(1);
        let max_rounds = current_len.ilog2() as usize;
        Ok(MetalFeSumcheckPlan {
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
        })
    }

    pub(crate) fn fe_sumcheck_round(
        &self,
        plan: &mut MetalFeSumcheckPlan,
        shape: &[u64],
        fold_challenge: Option<KWords>,
    ) -> Result<Vec<KWords>, MetalError> {
        if shape.len() < 13 {
            return Err(MetalError::Shape("resident FE round shape is invalid"));
        }
        let command = self.command_buffer()?;
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

        let command = self.command_buffer()?;
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
        plan: &mut MetalFeSumcheckPlan,
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

    pub(crate) fn prepare_nc_sumcheck(
        &self,
        inputs: MetalNcSumcheckInputs<'_>,
    ) -> Result<MetalNcSumcheckPlan, MetalError> {
        if inputs.witness_count == 0 || inputs.rows < 2 || !inputs.rows.is_power_of_two() || inputs.width == 0 {
            return Err(MetalError::Shape("resident NC sumcheck dimensions are invalid"));
        }
        let values_per_witness = if inputs.dense {
            inputs.rows * 54
        } else {
            inputs.rows * inputs.width
        };
        if inputs.eq_table.len() != inputs.rows * 2
            || inputs.digit_values.len() != inputs.witness_count * values_per_witness * 2
            || inputs.weights.len() != inputs.witness_count * 54 * 2
        {
            return Err(MetalError::Shape("resident NC input lengths are invalid"));
        }
        let groups = (inputs.rows / 2).div_ceil(64).max(1);
        let max_rounds = inputs.rows.ilog2() as usize;
        let shape = [
            inputs.rows as u64,
            inputs.witness_count as u64,
            inputs.width as u64,
            u64::from(inputs.dense),
            values_per_witness as u64,
        ];
        Ok(MetalNcSumcheckPlan {
            eq_tables: [
                self.buffer_from_slice(inputs.eq_table)?,
                self.buffer(std::mem::size_of_val(inputs.eq_table))?,
            ],
            digit_values: [
                self.buffer_from_slice(inputs.digit_values)?,
                self.buffer(std::mem::size_of_val(inputs.digit_values))?,
            ],
            weights: self.buffer_from_slice(inputs.weights)?,
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
            max_rounds,
            rows: inputs.rows,
            width: inputs.width,
            dense: inputs.dense,
            current_slot: 0,
        })
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
        let command = self.command_buffer()?;
        if let Some(challenge) = fold_challenge {
            self.encode_nc_fold(&command, plan, challenge)?;
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
        let groups = (plan.rows / 2).div_ceil(64).max(1);
        self.write_shared(&plan.shape, &expected)?;
        self.write_shared(&plan.reduction_shape, &[groups as u64, 5])?;

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.nc_round_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.eq_tables[plan.current_slot]), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[plan.current_slot]), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&plan.weights), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 4);
        }
        self.dispatch_threadgroups(&encoder, &self.nc_round_partials, groups, 64);
        encoder.endEncoding();

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
                plan.witness_count as u64,
                width as u64,
                u64::from(dense),
                values_per_witness as u64,
            ]);
            fold_shapes.extend_from_slice(&[plan.witness_count as u64, rows as u64, width as u64, u64::from(dense)]);
            reductions.extend_from_slice(&[(rows / 2).div_ceil(64).max(1) as u64, 5]);
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

        let command = self.command_buffer()?;
        for round in 0..rounds {
            let coeff_offset = round * 10 * size_of::<u64>();
            let challenge_offset = round * 2 * size_of::<u64>();
            let groups = reductions[2 * round] as usize;

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.nc_round_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&plan.eq_tables[plan.current_slot]), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.round_shapes), round * 5 * size_of::<u64>(), 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[plan.current_slot]), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&plan.weights), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 4);
            }
            self.dispatch_threadgroups(&encoder, &self.nc_round_partials, groups, 64);
            encoder.endEncoding();

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
                plan.witness_count * next_rows * next_width,
            );
            encoder.endEncoding();

            plan.current_slot = next_slot;
            plan.rows = next_rows;
            plan.width = next_width;
            plan.dense = next_dense;
        }
        self.finish(&command)?;

        let rounds = self.read_sumcheck_trace(&plan.output, &plan.challenge_log, &plan.transcript_state, 5, rounds)?;
        let eq = self.read_buffer::<u64>(&plan.eq_tables[plan.current_slot], 2);
        let values_per_witness = if plan.dense { 54 } else { plan.width };
        Ok(MetalNcSumcheckTrace {
            rounds,
            final_state: MetalNcFinalState {
                eq_beta: KWords::new(eq[0], eq[1]),
                digit_words: self.read_buffer::<u64>(
                    &plan.digit_values[plan.current_slot],
                    plan.witness_count * values_per_witness * 2,
                ),
                width: plan.width,
                dense: plan.dense,
            },
        })
    }

    pub(crate) fn finalize_nc_sumcheck(
        &self,
        plan: &mut MetalNcSumcheckPlan,
        fold_challenge: Option<KWords>,
    ) -> Result<MetalNcFinalState, MetalError> {
        if let Some(challenge) = fold_challenge {
            let command = self.command_buffer()?;
            self.encode_nc_fold(&command, plan, challenge)?;
            self.finish(&command)?;
        }
        if plan.rows != 1 {
            return Err(MetalError::Shape("resident NC sumcheck finalized before one row"));
        }
        let eq = self.read_buffer::<u64>(&plan.eq_tables[plan.current_slot], 2);
        let values_per_witness = if plan.dense { 54 } else { plan.width };
        Ok(MetalNcFinalState {
            eq_beta: KWords::new(eq[0], eq[1]),
            digit_words: self.read_buffer::<u64>(
                &plan.digit_values[plan.current_slot],
                plan.witness_count * values_per_witness * 2,
            ),
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
            plan.witness_count as u64,
            plan.rows as u64,
            plan.width as u64,
            u64::from(plan.dense),
        ];
        self.write_shared(&plan.fold_shape, &fold_shape)?;
        let output_elements = plan.witness_count * next_rows * next_width;
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

    fn write_shared<T: Copy>(&self, buffer: &Buffer, values: &[T]) -> Result<(), MetalError> {
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

    #[allow(clippy::too_many_arguments)]
    fn encode_transcript_challenge(
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

    fn read_sumcheck_trace(
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
