//! Resident buffers and command encoding for the NC column sumcheck.
//!
//! A compact row stores only its cyclic nonzero window. Folding doubles that
//! window until it overlaps itself, at which point the plan becomes a dense
//! 54-lane row and remains dense. Digit work may crop to its live prefix, but
//! equality tables always fold across their full padded domain.

mod mask;

use std::mem::size_of;

use neo_math::D;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{MetalSumcheckTrace, MetalWitnessMasks};
use crate::session::{Buffer, MetalSession};
use crate::{KWords, MetalError};

const NC_THREADS: usize = 64;
const NC_DENSE_THREADS: usize = 256;
const NC_DENSE_PAIRS_PER_GROUP: usize = 8;
const NC_MASK_DENSE_CROSSOVER: usize = 128;

fn nc_partial_groups(live_rows: usize, dense: bool) -> usize {
    let pairs_per_group = if dense { NC_DENSE_PAIRS_PER_GROUP } else { NC_THREADS };
    live_rows.div_ceil(2).div_ceil(pairs_per_group).max(1)
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

/// Initial digit representation at the host/device boundary.
#[derive(Clone, Copy)]
pub(crate) enum MetalNcDigitInput<'a> {
    /// Interleaved extension-field words in compact or dense row layout.
    Table(&'a [u64]),
    /// Signed-unit values packed as positive/negative ring masks.
    SignedMasks {
        words: &'a [u64],
        blocks: usize,
        active_rows: usize,
    },
}

/// Original masks and the folded basis used before dense materialization.
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

/// Ping-pong NC state, transcript storage, and optional mask-native source.
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
    // Buffers stay power-of-two padded; rows after this prefix are structurally zero.
    live_rows: usize,
    width: usize,
    dense: bool,
    current_slot: usize,
}

impl MetalNcSumcheckPlan {
    pub(in crate::session) fn signed_mask_buffer(&self, witness_count: usize, blocks: usize) -> Option<Buffer> {
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
            && self.weights.length() as usize == active_witness_count * D * 2 * size_of::<u64>()
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

pub(crate) struct MetalNcSumcheckTrace {
    pub rounds: MetalSumcheckTrace,
    pub final_state: MetalNcFinalState,
}

impl MetalSession {
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

    /// Installs dense rows or an immutable signed-mask source, compacts zero
    /// witnesses, and reuses a compatible mask-native workspace when available.
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
                .checked_mul(D)
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
            .checked_mul(D)
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
                    || blocks != active_rows.div_ceil(D)
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
        // Large domains stay as immutable masks plus a folded basis through
        // width 64, then materialize width 128 into dense 54-lane rows.
        let direct_compact = mask_input.is_some() && inputs.rows >= NC_MASK_DENSE_CROSSOVER;
        let workspace_values_per_witness = if direct_compact {
            (inputs.rows / NC_MASK_DENSE_CROSSOVER) * D
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
                let start = witness as usize * D * 2;
                inputs.weights[start..start + D * 2].iter().copied()
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
                plan.live_rows = active_rows;
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
        let live_rows = mask_input.map_or(inputs.rows, |(_, _, active_rows)| active_rows);
        let shape = [
            inputs.rows as u64,
            active_witness_count as u64,
            inputs.width as u64,
            u64::from(inputs.dense),
            values_per_witness as u64,
            live_rows as u64,
        ];
        let plan = MetalNcSumcheckPlan {
            eq_tables: [self.buffer(eq_bytes)?, self.buffer(eq_bytes)?],
            digit_values: [initial_digits, self.buffer(digit_bytes)?],
            mask_source,
            weights: self.buffer_from_slice(&active_weights)?,
            shape: self.buffer_from_slice(&shape)?,
            fold_shape: self.buffer_from_slice(&[0u64; 5])?,
            challenge: self.buffer_from_slice(&[0u64; 2])?,
            partials: self.buffer(groups * 5 * 2 * size_of::<u64>())?,
            reduction_shape: self.buffer_from_slice(&[groups as u64, 5])?,
            output: self.buffer(max_rounds * 10 * size_of::<u64>())?,
            round_shapes: self.buffer(max_rounds * 6 * size_of::<u64>())?,
            round_fold_shapes: self.buffer(max_rounds * 5 * size_of::<u64>())?,
            round_reduction_shapes: self.buffer(max_rounds * 2 * size_of::<u64>())?,
            challenge_log: self.buffer(max_rounds * 2 * size_of::<u64>())?,
            transcript_state: self.buffer(9 * size_of::<u64>())?,
            transcript_shape: self.buffer_from_slice(&[10u64])?,
            witness_count: inputs.witness_count,
            active_witness_count,
            max_rounds,
            rows: inputs.rows,
            live_rows,
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
            plan.rows * D
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
            plan.live_rows as u64,
        ];
        let groups = nc_partial_groups(plan.live_rows, plan.dense);
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
            let threads = if plan.dense { NC_DENSE_THREADS } else { NC_THREADS };
            self.dispatch_threadgroups(&encoder, &self.nc_round_partials, groups, threads);
            encoder.endEncoding();
        }

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.nc_reduce_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.reduction_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.output), 0, 2);
        }
        self.dispatch_threadgroups(&encoder, &self.nc_reduce_partials, 5, 256);
        encoder.endEncoding();
        self.finish(&command)?;

        Ok(self
            .read_buffer::<u64>(&plan.output, 10)
            .chunks_exact(2)
            .map(|words| KWords::new(words[0], words[1]))
            .collect())
    }

    /// Executes every NC round, transcript challenge, and compact-to-dense state
    /// transition as one resident trace before decoding the final column state.
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
        let mut shapes = Vec::with_capacity(rounds * 6);
        let mut fold_shapes = Vec::with_capacity(rounds * 5);
        let mut reductions = Vec::with_capacity(rounds * 2);
        let mut rows = plan.rows;
        let mut live_rows = plan.live_rows;
        let mut width = plan.width;
        let mut dense = plan.dense;
        let mut mask_native = plan
            .mask_source
            .as_ref()
            .is_some_and(|source| !source.folded);
        let direct_compact = plan
            .mask_source
            .as_ref()
            .is_some_and(|source| source.direct_compact);
        for _ in 0..rounds {
            let values_per_witness = if dense { rows * D } else { rows * width };
            shapes.extend_from_slice(&[
                rows as u64,
                plan.active_witness_count as u64,
                width as u64,
                u64::from(dense),
                values_per_witness as u64,
                live_rows as u64,
            ]);
            fold_shapes.extend_from_slice(&[
                plan.active_witness_count as u64,
                rows as u64,
                width as u64,
                u64::from(dense),
                live_rows as u64,
            ]);
            reductions.extend_from_slice(&[nc_partial_groups(live_rows, dense) as u64, 5]);
            rows /= 2;
            live_rows = live_rows.div_ceil(2);
            if mask_native {
                let materialized = !direct_compact || 2 * width == NC_MASK_DENSE_CROSSOVER;
                if materialized {
                    mask_native = false;
                    dense = direct_compact;
                    width = if direct_compact { D } else { 2 };
                } else {
                    width *= 2;
                }
            } else {
                dense = dense || 2 * width > D;
                width = if dense { D } else { 2 * width };
            }
        }
        self.write_shared(&plan.round_shapes, &shapes)?;
        self.write_shared(&plan.round_fold_shapes, &fold_shapes)?;
        self.write_shared(&plan.round_reduction_shapes, &reductions)?;
        let mut state_words = transcript_state.to_vec();
        state_words.push(transcript_absorbed as u64);
        self.write_shared(&plan.transcript_state, &state_words)?;

        // The device derives challenges and folds both equality and digit state
        // in one ordered command buffer; the CPU receives only the final trace.
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
                    encoder.setBuffer_offset_atIndex(Some(&plan.round_shapes), round * 6 * size_of::<u64>(), 1);
                    encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[plan.current_slot]), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&plan.weights), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 4);
                }
                let threads = if plan.dense { NC_DENSE_THREADS } else { NC_THREADS };
                self.dispatch_threadgroups(&encoder, &self.nc_round_partials, groups, threads);
                encoder.endEncoding();
            }

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.nc_reduce_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&plan.partials), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.round_reduction_shapes), round * 2 * size_of::<u64>(), 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.output), coeff_offset, 2);
            }
            self.dispatch_threadgroups(&encoder, &self.nc_reduce_partials, 5, 256);
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
                let next_live_rows = plan.live_rows.div_ceil(2);
                let next_dense = plan.dense || 2 * plan.width > D;
                let next_width = if next_dense { D } else { 2 * plan.width };
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
                    encoder.setBuffer_offset_atIndex(Some(&plan.round_fold_shapes), round * 5 * size_of::<u64>(), 2);
                    encoder.setBuffer_offset_atIndex(Some(&plan.digit_values[next_slot]), 0, 3);
                }
                self.dispatch(
                    &encoder,
                    &self.nc_fold_compact,
                    plan.active_witness_count * next_live_rows * next_width,
                );
                encoder.endEncoding();

                plan.current_slot = next_slot;
                plan.rows = next_rows;
                plan.live_rows = next_live_rows;
                plan.width = next_width;
                plan.dense = next_dense;
            }
        }
        self.finish(&command)?;

        let rounds = self.read_sumcheck_trace(&plan.output, &plan.challenge_log, &plan.transcript_state, 5, rounds)?;
        let eq = self.read_buffer::<u64>(&plan.eq_tables[plan.current_slot], 2);
        let values_per_witness = if plan.dense { D } else { plan.width };
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
        // All-zero witnesses are compacted from kernel work (with one sentinel
        // for an entirely zero batch). Restore every logical slot at egress.
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
        let values_per_witness = if plan.dense { D } else { plan.width };
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
        let next_dense = plan.dense || 2 * plan.width > D;
        let next_width = if next_dense { D } else { 2 * plan.width };

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
            plan.live_rows as u64,
        ];
        self.write_shared(&plan.fold_shape, &fold_shape)?;
        let next_live_rows = plan.live_rows.div_ceil(2);
        let output_elements = plan.active_witness_count * next_live_rows * next_width;
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
        plan.live_rows = next_live_rows;
        plan.width = next_width;
        plan.dense = next_dense;
        Ok(())
    }
}
