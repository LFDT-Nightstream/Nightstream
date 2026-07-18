//! FE row sumcheck over independent resident MCS table sets.
//!
//! Each non-zero MCS witness keeps its base-field row tables in the buffer
//! that produced them. Round zero folds those tables in place into extension
//! values; later rounds reuse one scratch buffer across MCS sets. This avoids
//! a second, monolithic copy of every table and keeps independent witnesses
//! independently schedulable.

use std::{collections::BTreeMap, mem::size_of};

use neo_math::{KExtensions, F, K};
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use p3_field::PrimeCharacteristicRing;

use super::resident::{MetalFeSumcheckInputs, MetalFeTableInput, MetalSumcheckTrace};
use super::{Buffer, MetalDeferredMcsRowTables, MetalSession};
use crate::{KWords, MetalError};

const ROUND_SHAPE_WORDS: usize = 7;

/// Optional algebra-preserving program that extracts a shared selector per group.
struct FactoredTerms {
    group_headers: Vec<u64>,
    term_headers: Vec<u64>,
    term_variables: Vec<u64>,
}

fn factor_streaming_terms(
    term_headers: &[u64],
    term_variables: &[u64],
    term_count: usize,
    table_count: usize,
) -> Option<FactoredTerms> {
    if term_headers.len() != 4 * term_count || term_variables.len() % 2 != 0 {
        return None;
    }
    let variable_count = term_variables.len() / 2;
    for term in 0..term_count {
        let start = term_headers[4 * term + 2] as usize;
        let count = term_headers[4 * term + 3] as usize;
        if start
            .checked_add(count)
            .is_none_or(|end| end > variable_count)
        {
            return None;
        }
    }

    // Failure to find a useful factor is not an error; the caller selects the
    // unfactored kernel with the original term program.
    let mut assigned = vec![false; term_count];
    let mut group_headers = Vec::new();
    let mut factored_headers = Vec::with_capacity(term_headers.len());
    let mut factored_variables = Vec::with_capacity(term_variables.len());
    while assigned.iter().any(|&value| !value) {
        let mut counts = BTreeMap::<u64, usize>::new();
        for term in 0..term_count {
            if assigned[term] {
                continue;
            }
            let start = term_headers[4 * term + 2] as usize;
            let count = term_headers[4 * term + 3] as usize;
            for variable in start..start + count {
                let position = term_variables[2 * variable];
                let exponent = term_variables[2 * variable + 1];
                if position < table_count as u64 && exponent == 1 {
                    *counts.entry(position).or_default() += 1;
                }
            }
        }
        let mut selector = None;
        let mut selector_count = 0;
        for (position, count) in counts {
            if count > selector_count {
                selector = Some(position);
                selector_count = count;
            }
        }
        let selector = selector.filter(|_| selector_count >= 2)?;
        let first_term = factored_headers.len() / 4;
        let mut group_term_count = 0;
        for term in 0..term_count {
            if assigned[term] {
                continue;
            }
            let header = 4 * term;
            let start = term_headers[header + 2] as usize;
            let count = term_headers[header + 3] as usize;
            let contains_selector = (start..start + count)
                .any(|variable| term_variables[2 * variable] == selector && term_variables[2 * variable + 1] == 1);
            if !contains_selector {
                continue;
            }
            assigned[term] = true;
            group_term_count += 1;
            let factored_start = factored_variables.len() / 2;
            let mut removed_selector = false;
            for variable in start..start + count {
                let position = term_variables[2 * variable];
                let exponent = term_variables[2 * variable + 1];
                if !removed_selector && position == selector && exponent == 1 {
                    removed_selector = true;
                } else {
                    factored_variables.extend_from_slice(&[position, exponent]);
                }
            }
            let factored_count = factored_variables.len() / 2 - factored_start;
            factored_headers.extend_from_slice(&[
                term_headers[header],
                term_headers[header + 1],
                factored_start as u64,
                factored_count as u64,
            ]);
        }
        debug_assert_eq!(group_term_count, selector_count);
        group_headers.extend_from_slice(&[selector, first_term as u64, group_term_count as u64]);
    }
    Some(FactoredTerms {
        group_headers,
        term_headers: factored_headers,
        term_variables: factored_variables,
    })
}

/// Per-MCS buffers, shared scratch, and transcript state for streaming FE.
pub(crate) struct MetalStreamingFePlan {
    mcs_tables: Vec<Buffer>,
    mcs_scratch: Buffer,
    special_tables: [Buffer; 2],
    gammas: Buffer,
    zero_constant: Buffer,
    factor_groups: Option<Buffer>,
    term_headers: Buffer,
    term_variables: Buffer,
    partials: Buffer,
    round_shapes: Buffer,
    round_eval_shapes: Buffer,
    round_fold_shapes: Buffer,
    round_reduction_shapes: Buffer,
    output: Buffer,
    challenge_log: Buffer,
    transcript_state: Buffer,
    transcript_shape: Buffer,
    mcs_table_count: usize,
    special_table_count: usize,
    input_slot: Option<usize>,
    eval_slot: Option<usize>,
    coefficient_count: usize,
    row_degree: usize,
    program_count: usize,
    contribution_count: usize,
    has_zero_constant: bool,
    max_rounds: usize,
    current_len: usize,
    active_len: usize,
    base_mcs: bool,
    mcs_slot: usize,
    special_slot: usize,
    copyless_mcs: bool,
}

impl MetalSession {
    pub(super) fn prepare_streaming_fe_sumcheck(
        &self,
        inputs: MetalFeSumcheckInputs<'_>,
    ) -> Result<MetalStreamingFePlan, MetalError> {
        if inputs.shape.len() < 13
            || inputs.table_count == 0
            || inputs.tables.len() != inputs.table_count
            || inputs.coefficient_count == 0
            || inputs.shape[3] >= 10
        {
            return Err(MetalError::Shape("streaming FE metadata is invalid"));
        }
        let current_len = inputs.shape[0] as usize;
        let active_len = inputs.shape[1] as usize;
        let row_degree = inputs.shape[3] as usize;
        let mcs_count = inputs.shape[7] as usize;
        let term_count = inputs.shape[8] as usize;
        if current_len < 2
            || !current_len.is_power_of_two()
            || active_len == 0
            || active_len > current_len
            || row_degree >= inputs.coefficient_count
            || inputs.mcs_headers.len() != 3 * mcs_count
            || inputs.gammas.len() < 2 * mcs_count
        {
            return Err(MetalError::Shape("streaming FE dimensions are invalid"));
        }

        let mut mcs_tables = Vec::new();
        let mut gamma_words = Vec::new();
        let mut zero_gamma = K::ZERO;
        let mut mcs_table_count = None;
        for mcs in 0..mcs_count {
            let header = 3 * mcs;
            let is_zero = inputs.mcs_headers[header] != 0;
            let table_start = inputs.mcs_headers[header + 1] as usize;
            let table_count = inputs.mcs_headers[header + 2] as usize;
            let gamma = k_from_words(inputs.gammas[2 * mcs], inputs.gammas[2 * mcs + 1]);
            if is_zero {
                if table_count != 0 {
                    return Err(MetalError::Shape("zero MCS has resident FE tables"));
                }
                zero_gamma += gamma;
                continue;
            }
            if table_count == 0
                || table_start
                    .checked_add(table_count)
                    .is_none_or(|end| end > inputs.mcs_table_indices.len())
                || mcs_table_count.is_some_and(|expected| expected != table_count)
            {
                return Err(MetalError::Shape("streaming MCS table metadata is inconsistent"));
            }
            mcs_table_count = Some(table_count);
            let mut deferred: Option<&MetalDeferredMcsRowTables> = None;
            for table in 0..table_count {
                let logical = inputs.mcs_table_indices[table_start + table] as usize;
                let Some(MetalFeTableInput::DeferredMcs {
                    tables,
                    table: source_table,
                }) = inputs.tables.get(logical)
                else {
                    return Err(MetalError::Shape("streaming MCS table is not device-owned"));
                };
                if *source_table != table || !tables.matches(mcs, active_len, current_len, table_count) {
                    return Err(MetalError::Shape("streaming MCS table source is stale"));
                }
                if deferred.is_some_and(|first| !std::ptr::eq(first, *tables)) {
                    return Err(MetalError::Shape("streaming MCS tables span multiple buffers"));
                }
                deferred = Some(*tables);
            }
            let deferred = deferred.ok_or(MetalError::Shape("streaming MCS source is missing"))?;
            mcs_tables.push(deferred.words().clone());
            gamma_words.extend_from_slice(&inputs.gammas[2 * mcs..2 * mcs + 2]);
        }
        let mcs_table_count = mcs_table_count.unwrap_or(0);
        let factored_terms =
            factor_streaming_terms(inputs.term_headers, inputs.term_variables, term_count, mcs_table_count);
        let (factor_groups, term_headers, term_variables, program_count) = if let Some(factored) = factored_terms {
            let group_count = factored.group_headers.len() / 3;
            (
                Some(self.buffer_from_slice(&factored.group_headers)?),
                self.buffer_from_slice(super::nonempty(&factored.term_headers))?,
                self.buffer_from_slice(super::nonempty(&factored.term_variables))?,
                group_count,
            )
        } else {
            (
                None,
                self.buffer_from_slice(super::nonempty(inputs.term_headers))?,
                self.buffer_from_slice(super::nonempty(inputs.term_variables))?,
                term_count,
            )
        };

        let eq_logical = inputs.shape[4] as usize;
        let input_logical = inputs.shape[5].checked_sub(1).map(|value| value as usize);
        let eval_logical = inputs.shape[6].checked_sub(1).map(|value| value as usize);
        if eq_logical >= inputs.tables.len() || input_logical.is_some() != eval_logical.is_some() {
            return Err(MetalError::Shape("streaming FE special table metadata is invalid"));
        }
        let mut special_logical = vec![eq_logical];
        let input_slot = input_logical.map(|logical| {
            special_logical.push(logical);
            special_logical.len() - 1
        });
        let eval_slot = eval_logical.map(|logical| {
            special_logical.push(logical);
            special_logical.len() - 1
        });
        let special_table_count = special_logical.len();
        let table_bytes = 2 * current_len * size_of::<u64>();
        let special_tables = self.buffer(special_table_count * table_bytes)?;
        let install = self.command_buffer("nightstream.pi_ccs.fe_stream.install")?;
        for (slot, &logical) in special_logical.iter().enumerate() {
            let destination_offset = slot * table_bytes;
            match &inputs.tables[logical] {
                MetalFeTableInput::Host(values) => {
                    self.write_k_table_at(&special_tables, destination_offset, values)?;
                }
                MetalFeTableInput::TensorPoint(point) => {
                    if point.len() >= usize::BITS as usize || 1usize << point.len() != current_len {
                        return Err(MetalError::Shape("streaming FE tensor point has the wrong length"));
                    }
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
                        let encoder = install.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                        encoder.setComputePipelineState(&self.tensor_point_expand_k);
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(&challenges), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(&stages), stage * size_of::<u64>(), 1);
                            encoder.setBuffer_offset_atIndex(Some(&special_tables), destination_offset, 2);
                        }
                        self.dispatch(&encoder, &self.tensor_point_expand_k, 1usize << stage);
                        encoder.endEncoding();
                    }
                }
                MetalFeTableInput::DeferredEval(table) => {
                    if !table.matches(current_len) {
                        return Err(MetalError::Shape("streaming deferred Eval table is stale"));
                    }
                    let encoder = install.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                    encoder.setComputePipelineState(&self.copy_k_words);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(table.words()), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(&special_tables), destination_offset, 1);
                    }
                    self.dispatch(&encoder, &self.copy_k_words, current_len);
                    encoder.endEncoding();
                }
                MetalFeTableInput::DeferredMcs { .. } => {
                    return Err(MetalError::Shape("MCS table used as a special FE table"));
                }
            }
        }
        self.submit(&install);

        let f_at_zero = k_from_words(inputs.shape[9], inputs.shape[10]);
        let zero_constant = f_at_zero * zero_gamma;
        let has_zero_constant = zero_constant != K::ZERO;
        let (zero_real, zero_imaginary) = zero_constant.to_limbs_u64();
        let zero_words = [zero_real, zero_imaginary];
        let has_eval = eval_slot.is_some();
        let contribution_count = mcs_tables.len() + usize::from(has_eval) + usize::from(has_zero_constant);
        if contribution_count == 0 {
            return Err(MetalError::Shape("streaming FE phase has no contribution channels"));
        }

        let max_groups = active_len.div_ceil(2).div_ceil(64).max(1);
        let max_rounds = current_len.ilog2() as usize;
        let mcs_scratch_words = (mcs_table_count * current_len.div_ceil(2)).max(2);
        let special_scratch_words = (special_table_count * current_len).max(2);
        let copyless_mcs = mcs_tables.len() == 1;
        Ok(MetalStreamingFePlan {
            mcs_tables,
            mcs_scratch: self.buffer(mcs_scratch_words * size_of::<u64>())?,
            special_tables: [special_tables, self.buffer(special_scratch_words * size_of::<u64>())?],
            gammas: self.buffer_from_slice(super::nonempty(&gamma_words))?,
            zero_constant: self.buffer_from_slice(&zero_words)?,
            factor_groups,
            term_headers,
            term_variables,
            partials: self.buffer(contribution_count * max_groups * inputs.coefficient_count * 2 * size_of::<u64>())?,
            round_shapes: self.buffer(max_rounds * ROUND_SHAPE_WORDS * size_of::<u64>())?,
            round_eval_shapes: self.buffer(max_rounds * ROUND_SHAPE_WORDS * size_of::<u64>())?,
            round_fold_shapes: self.buffer(max_rounds * 3 * size_of::<u64>())?,
            round_reduction_shapes: self.buffer(max_rounds * 2 * size_of::<u64>())?,
            output: self.buffer(max_rounds * inputs.coefficient_count * 2 * size_of::<u64>())?,
            challenge_log: self.buffer(max_rounds * 2 * size_of::<u64>())?,
            transcript_state: self.buffer(9 * size_of::<u64>())?,
            transcript_shape: self.buffer_from_slice(&[(inputs.coefficient_count * 2) as u64])?,
            mcs_table_count,
            special_table_count,
            input_slot,
            eval_slot,
            coefficient_count: inputs.coefficient_count,
            row_degree,
            program_count,
            contribution_count,
            has_zero_constant,
            max_rounds,
            current_len,
            active_len,
            base_mcs: true,
            mcs_slot: 0,
            special_slot: 0,
            copyless_mcs,
        })
    }

    pub(super) fn streaming_fe_sumcheck_round(
        &self,
        _plan: &mut MetalStreamingFePlan,
        _shape: &[u64],
        _fold_challenge: Option<KWords>,
    ) -> Result<Vec<KWords>, MetalError> {
        Err(MetalError::Shape("streaming FE plan requires the bulk transcript path"))
    }

    pub(super) fn streaming_fe_sumcheck_trace(
        &self,
        plan: &mut MetalStreamingFePlan,
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
            || base_shape[0] as usize != plan.current_len
            || base_shape[1] as usize != plan.active_len
        {
            return Err(MetalError::Shape("streaming FE trace dimensions are invalid"));
        }
        let gamma_to_k = [base_shape[11], base_shape[12]];
        let mut round_shapes = Vec::with_capacity(rounds * ROUND_SHAPE_WORDS);
        let mut eval_shapes = Vec::with_capacity(rounds * ROUND_SHAPE_WORDS);
        let mut fold_shapes = Vec::with_capacity(rounds * 3);
        let mut reduction_shapes = Vec::with_capacity(rounds * 2);
        let mut current_len = plan.current_len;
        let mut active_len = plan.active_len;
        let mut base_mode = plan.base_mcs;
        for _ in 0..rounds {
            let groups = active_len.div_ceil(2).div_ceil(64).max(1);
            round_shapes.extend_from_slice(&[
                current_len as u64,
                active_len as u64,
                plan.coefficient_count as u64,
                plan.row_degree as u64,
                plan.mcs_table_count as u64,
                u64::from(base_mode),
                plan.program_count as u64,
            ]);
            eval_shapes.extend_from_slice(&[
                current_len as u64,
                active_len as u64,
                plan.coefficient_count as u64,
                plan.input_slot.unwrap_or(0) as u64,
                plan.eval_slot.unwrap_or(0) as u64,
                gamma_to_k[0],
                gamma_to_k[1],
            ]);
            fold_shapes.extend_from_slice(&[plan.mcs_table_count as u64, current_len as u64, active_len as u64]);
            reduction_shapes
                .extend_from_slice(&[(groups * plan.contribution_count) as u64, plan.coefficient_count as u64]);
            current_len /= 2;
            active_len = active_len.div_ceil(2).max(1);
            base_mode = false;
        }
        self.write_shared(&plan.round_shapes, &round_shapes)?;
        self.write_shared(&plan.round_eval_shapes, &eval_shapes)?;
        self.write_shared(&plan.round_fold_shapes, &fold_shapes)?;
        self.write_shared(&plan.round_reduction_shapes, &reduction_shapes)?;
        let mut transcript_words = transcript_state.to_vec();
        transcript_words.push(transcript_absorbed as u64);
        self.write_shared(&plan.transcript_state, &transcript_words)?;

        let command = self.command_buffer("nightstream.pi_ccs.fe_stream.trace")?;
        current_len = plan.current_len;
        active_len = plan.active_len;
        base_mode = plan.base_mcs;
        let mut mcs_slot = plan.mcs_slot;
        let mut special_slot = plan.special_slot;
        let mcs_round_pipeline = if plan.factor_groups.is_some() {
            &self.fe_stream_mcs_factored_round_partials
        } else {
            &self.fe_stream_mcs_round_partials
        };
        for round in 0..rounds {
            let groups = active_len.div_ceil(2).div_ceil(64).max(1);
            let coefficient_offset = round * plan.coefficient_count * 2 * size_of::<u64>();
            let challenge_offset = round * 2 * size_of::<u64>();
            let mut channel = 0usize;
            for (mcs, tables) in plan.mcs_tables.iter().enumerate() {
                let tables = if plan.copyless_mcs && mcs_slot == 1 {
                    &plan.mcs_scratch
                } else {
                    tables
                };
                let partial_offset = channel * groups * plan.coefficient_count * 2 * size_of::<u64>();
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(mcs_round_pipeline);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(tables), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&plan.special_tables[special_slot]), 0, 1);
                    encoder.setBuffer_offset_atIndex(
                        Some(&plan.round_shapes),
                        round * ROUND_SHAPE_WORDS * size_of::<u64>(),
                        2,
                    );
                    encoder.setBuffer_offset_atIndex(Some(&plan.gammas), mcs * 2 * size_of::<u64>(), 3);
                    if let Some(groups) = &plan.factor_groups {
                        encoder.setBuffer_offset_atIndex(Some(groups), 0, 4);
                        encoder.setBuffer_offset_atIndex(Some(&plan.term_headers), 0, 5);
                        encoder.setBuffer_offset_atIndex(Some(&plan.term_variables), 0, 6);
                        encoder.setBuffer_offset_atIndex(Some(&plan.partials), partial_offset, 7);
                    } else {
                        encoder.setBuffer_offset_atIndex(Some(&plan.term_headers), 0, 4);
                        encoder.setBuffer_offset_atIndex(Some(&plan.term_variables), 0, 5);
                        encoder.setBuffer_offset_atIndex(Some(&plan.partials), partial_offset, 6);
                    }
                }
                self.dispatch_threadgroups(&encoder, mcs_round_pipeline, groups, 64);
                encoder.endEncoding();
                channel += 1;
            }
            if plan.eval_slot.is_some() {
                let partial_offset = channel * groups * plan.coefficient_count * 2 * size_of::<u64>();
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.fe_stream_eval_round_partials);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&plan.special_tables[special_slot]), 0, 0);
                    encoder.setBuffer_offset_atIndex(
                        Some(&plan.round_eval_shapes),
                        round * ROUND_SHAPE_WORDS * size_of::<u64>(),
                        1,
                    );
                    encoder.setBuffer_offset_atIndex(Some(&plan.partials), partial_offset, 2);
                }
                self.dispatch_threadgroups(&encoder, &self.fe_stream_eval_round_partials, groups, 64);
                encoder.endEncoding();
                channel += 1;
            }
            if plan.has_zero_constant {
                let partial_offset = channel * groups * plan.coefficient_count * 2 * size_of::<u64>();
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.fe_stream_constant_round_partials);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&plan.special_tables[special_slot]), 0, 0);
                    encoder.setBuffer_offset_atIndex(
                        Some(&plan.round_eval_shapes),
                        round * ROUND_SHAPE_WORDS * size_of::<u64>(),
                        1,
                    );
                    encoder.setBuffer_offset_atIndex(Some(&plan.zero_constant), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&plan.partials), partial_offset, 3);
                }
                self.dispatch_threadgroups(&encoder, &self.fe_stream_constant_round_partials, groups, 64);
                encoder.endEncoding();
                channel += 1;
            }
            debug_assert_eq!(channel, plan.contribution_count);

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.sumcheck_reduce_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.round_reduction_shapes), round * 2 * size_of::<u64>(), 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.output), coefficient_offset, 2);
            }
            self.dispatch(&encoder, &self.sumcheck_reduce_partials, plan.coefficient_count);
            encoder.endEncoding();

            self.encode_transcript_challenge(
                &command,
                &plan.transcript_state,
                &plan.output,
                coefficient_offset,
                &plan.challenge_log,
                challenge_offset,
                &plan.transcript_shape,
            )?;

            for tables in &plan.mcs_tables {
                let elements = plan.mcs_table_count * active_len.div_ceil(2);
                let fold_shape_offset = round * 3 * size_of::<u64>();
                if base_mode {
                    let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                    encoder.setComputePipelineState(&self.fe_fold_base_tables_in_place);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(tables), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(&plan.challenge_log), challenge_offset, 1);
                        encoder.setBuffer_offset_atIndex(Some(&plan.round_fold_shapes), fold_shape_offset, 2);
                    }
                    self.dispatch(&encoder, &self.fe_fold_base_tables_in_place, elements);
                    encoder.endEncoding();
                } else if plan.copyless_mcs {
                    let output_slot = 1 - mcs_slot;
                    let (input, output) = if mcs_slot == 0 {
                        (tables, &plan.mcs_scratch)
                    } else {
                        (&plan.mcs_scratch, tables)
                    };
                    self.encode_streaming_mcs_k_fold(
                        &command,
                        input,
                        output,
                        &plan.challenge_log,
                        challenge_offset,
                        &plan.round_fold_shapes,
                        fold_shape_offset,
                        elements,
                    )?;
                    mcs_slot = output_slot;
                } else {
                    self.encode_streaming_mcs_k_fold(
                        &command,
                        tables,
                        &plan.mcs_scratch,
                        &plan.challenge_log,
                        challenge_offset,
                        &plan.round_fold_shapes,
                        fold_shape_offset,
                        elements,
                    )?;
                    self.encode_streaming_mcs_k_copy(
                        &command,
                        &plan.mcs_scratch,
                        tables,
                        &plan.round_fold_shapes,
                        fold_shape_offset,
                        elements,
                    )?;
                }
            }
            let special_elements = plan.special_table_count * (current_len / 2);
            let special_output_slot = 1 - special_slot;
            self.encode_streaming_k_fold(
                &command,
                &plan.special_tables[special_slot],
                &plan.special_tables[special_output_slot],
                &plan.challenge_log,
                challenge_offset,
                special_elements,
            )?;
            special_slot = special_output_slot;
            current_len /= 2;
            active_len = active_len.div_ceil(2).max(1);
            base_mode = false;
        }
        self.finish(&command)?;
        plan.current_len = current_len;
        plan.active_len = active_len;
        plan.base_mcs = base_mode;
        plan.mcs_slot = mcs_slot;
        plan.special_slot = special_slot;
        self.read_sumcheck_trace(
            &plan.output,
            &plan.challenge_log,
            &plan.transcript_state,
            plan.coefficient_count,
            rounds,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_streaming_mcs_k_fold(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        tables: &Buffer,
        output: &Buffer,
        challenges: &Buffer,
        challenge_offset: usize,
        shapes: &Buffer,
        shape_offset: usize,
        elements: usize,
    ) -> Result<(), MetalError> {
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.fe_fold_k_tables_live);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(tables), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(challenges), challenge_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(shapes), shape_offset, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
        }
        self.dispatch(&encoder, &self.fe_fold_k_tables_live, elements);
        encoder.endEncoding();
        Ok(())
    }

    fn encode_streaming_mcs_k_copy(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        tables: &Buffer,
        output: &Buffer,
        shapes: &Buffer,
        shape_offset: usize,
        elements: usize,
    ) -> Result<(), MetalError> {
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.fe_copy_k_tables_live);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(tables), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(shapes), shape_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        }
        self.dispatch(&encoder, &self.fe_copy_k_tables_live, elements);
        encoder.endEncoding();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_streaming_k_fold(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        tables: &Buffer,
        output: &Buffer,
        challenges: &Buffer,
        challenge_offset: usize,
        elements: usize,
    ) -> Result<(), MetalError> {
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.fold_k_table);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(tables), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(challenges), challenge_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
        }
        self.dispatch(&encoder, &self.fold_k_table, elements);
        encoder.endEncoding();
        Ok(())
    }
}

fn k_from_words(real: u64, imaginary: u64) -> K {
    K::from_coeffs([F::from_u64(real), F::from_u64(imaginary)])
}
