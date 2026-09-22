//! Application matrix values: resident prefixes or bounded replay of original rows.
//! Prior challenges define each replayed value. No transcript or circuit rule is changed.

use super::*;

pub(super) struct ApplicationWindow {
    pub(super) values: Buffer,
    pub(super) stride: usize,
}

pub(super) struct ApplicationTables {
    resident: Option<ApplicationWindow>,
    rows: usize,
    source_rows: usize,
    fresh_count: usize,
    table_count: usize,
    challenges: Vec<K>,
    workspace_limit: Option<usize>,
    budget: usize,
    peak: usize,
}

fn table_bytes(tables: usize, rows: usize, word_bytes: usize) -> Result<usize, MetalError> {
    tables
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(word_bytes))
        .ok_or(MetalError::Shape("application workspace size overflow"))
}

pub(super) fn available_workspace(session: &MetalSession, reserved: usize) -> usize {
    super::super::BUFFER_LIMIT_BYTES
        .saturating_sub(session.activity().current_allocated_bytes as usize)
        .saturating_sub(reserved)
}

impl ApplicationTables {
    pub(super) fn new(
        session: &MetalSession,
        plan: &MetalJointMatrixPlan<'_>,
        masks: &MetalWitnessMasks,
        fresh_count: usize,
        workspace_limit: Option<usize>,
        reserved: usize,
    ) -> Result<Self, MetalError> {
        let table_count = fresh_count
            .checked_mul(plan.matrix_count)
            .filter(|&n| n != 0)
            .ok_or(MetalError::Shape("empty application table family"))?;
        let mut tables = Self {
            resident: None,
            rows: plan.rows,
            source_rows: plan.rows,
            fresh_count,
            table_count,
            challenges: Vec::new(),
            workspace_limit,
            budget: 0,
            peak: 0,
        };
        tables.update_budget(session, reserved);
        let bytes = table_bytes(table_count, plan.rows, size_of::<F>())?;
        let next = table_bytes(table_count, plan.rows.div_ceil(2), size_of::<K>())?;
        if bytes
            .checked_add(next)
            .is_some_and(|peak| peak <= tables.budget)
        {
            let values = session.build_joint_application_tables(plan, masks, fresh_count, plan.rows)?;
            tables.record(values.length())?;
            tables.resident = Some(ApplicationWindow {
                values,
                stride: plan.rows,
            });
        } else {
            tables.replay_capacity()?;
        }
        Ok(tables)
    }

    pub(super) fn scratch_bytes(plan: &MetalJointMatrixPlan<'_>, fresh_count: usize) -> Result<usize, MetalError> {
        let metadata = table_bytes(fresh_count, plan.matrix_count, 12 * size_of::<u64>())?;
        plan.workspace_bytes
            .checked_add(metadata)
            .and_then(|bytes| bytes.checked_add(7 * size_of::<u64>()))
            .ok_or(MetalError::Shape("application metadata size overflow"))
    }

    fn update_budget(&mut self, session: &MetalSession, reserved: usize) {
        let owned = self.resident.as_ref().map_or(0, |w| w.values.length());
        self.budget = super::super::BUFFER_LIMIT_BYTES
            .saturating_sub((session.activity().current_allocated_bytes as usize).saturating_sub(owned))
            .saturating_sub(reserved)
            .min(self.workspace_limit.unwrap_or(usize::MAX));
    }

    fn record(&mut self, bytes: usize) -> Result<(), MetalError> {
        if bytes > self.budget {
            return Err(MetalError::MemoryLimit {
                requested: bytes,
                allocated: 0,
                limit: self.budget,
            });
        }
        self.peak = self.peak.max(bytes);
        Ok(())
    }

    pub(super) fn peak_bytes(&self) -> usize {
        self.peak
    }

    pub(super) fn resident(&self) -> Option<&ApplicationWindow> {
        self.resident.as_ref()
    }

    // Two final K rows are needed by a SumCheck pair. A power-of-two source
    // window has an in-place first (F-to-K) fold. Later folds need half as
    // much new storage as the preceding window, at most W/2 base words.
    fn replay_capacity(&self) -> Result<usize, MetalError> {
        let destination = table_bytes(self.table_count, 2, size_of::<K>())?;
        let mut rows = self
            .source_rows
            .checked_next_power_of_two()
            .ok_or(MetalError::Shape("application row domain overflow"))?
            .max(2);
        loop {
            let base = table_bytes(self.table_count, rows, size_of::<F>())?;
            let peak = base
                .checked_add(if rows > 2 { base / 2 } else { 0 })
                .and_then(|n| n.checked_add(destination));
            if peak.is_some_and(|n| n <= self.budget) {
                return Ok(rows);
            }
            if rows == 2 {
                return Err(MetalError::MemoryLimit {
                    requested: base.saturating_add(destination),
                    allocated: 0,
                    limit: self.budget,
                });
            }
            rows /= 2;
        }
    }

    pub(super) fn window_rows(&self) -> Result<usize, MetalError> {
        Ok((self.replay_capacity()? >> self.challenges.len()).max(2))
    }

    pub(super) fn prepare_round(
        &mut self,
        session: &MetalSession,
        plan: &MetalJointMatrixPlan<'_>,
        masks: &MetalWitnessMasks,
        reserved: usize,
    ) -> Result<(), MetalError> {
        self.update_budget(session, reserved);
        if self
            .resident
            .as_ref()
            .is_some_and(|w| w.values.length() > self.budget)
        {
            self.resident = None;
        }
        if self.resident.is_none() && self.rows <= self.window_rows()? {
            let window = self.replay_window(session, plan, masks, 0)?;
            self.resident = Some(window);
        }
        Ok(())
    }

    fn fold_window(
        &mut self,
        session: &MetalSession,
        mut values: Buffer,
        mut rows: usize,
        stages: usize,
        external_bytes: usize,
    ) -> Result<ApplicationWindow, MetalError> {
        self.record(external_bytes + values.length())?;
        for stage in 0..stages {
            let next_rows = rows.div_ceil(2);
            let next = if stage == 0 {
                // Source windows are even. Each F pair and its K result own
                // the same 16 bytes, including at matrix boundaries.
                values.clone()
            } else {
                let bytes = table_bytes(self.table_count, next_rows, size_of::<K>())?;
                self.record(external_bytes + values.length() + bytes)?;
                session.buffer(bytes)?
            };
            let challenge = session.buffer_from_slice(&k_words(&[self.challenges[stage]]))?;
            let shape = session.buffer_from_slice(&[rows as u64, self.table_count as u64])?;
            let command = session.command_buffer("nightstream.pi_ccs.application.replay.fold")?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            let pipeline = if stage == 0 {
                &session.joint_fold_base_tables
            } else {
                &session.joint_fold_k_tables
            };
            encoder.setComputePipelineState(pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&values), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&challenge), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&next), 0, 3);
            }
            session.dispatch(&encoder, pipeline, self.table_count * next_rows);
            encoder.endEncoding();
            session.finish(&command)?;
            drop(encoder);
            drop(command);
            values = next;
            rows = next_rows;
        }
        Ok(ApplicationWindow { values, stride: rows })
    }

    pub(super) fn replay_window(
        &mut self,
        session: &MetalSession,
        plan: &MetalJointMatrixPlan<'_>,
        masks: &MetalWitnessMasks,
        row_start: usize,
    ) -> Result<ApplicationWindow, MetalError> {
        let source_window = self.replay_capacity()?;
        let round = self.challenges.len();
        let stages = round.min(source_window.ilog2() as usize);
        let span = 1usize
            .checked_shl(round as u32)
            .ok_or(MetalError::Shape("application replay span overflow"))?;
        let source_start = row_start
            .checked_mul(span)
            .ok_or(MetalError::Shape("application replay offset overflow"))?;
        if round < source_window.ilog2() as usize {
            let bytes = table_bytes(self.table_count, source_window, size_of::<F>())?;
            self.record(bytes)?;
            let base =
                session.build_joint_application_window(plan, masks, self.fresh_count, source_window, source_start)?;
            let folded = self.fold_window(session, base, source_window, stages, 0)?;
            return Ok(folded);
        }
        let bytes = table_bytes(self.table_count, 2, size_of::<K>())?;
        self.record(bytes)?;
        let output = session.buffer(bytes)?;
        let command = session.command_buffer("nightstream.pi_ccs.application.replay.zero")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&session.joint_zero_words);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 0);
        }
        session.dispatch(&encoder, &session.joint_zero_words, bytes / size_of::<u64>());
        encoder.endEncoding();
        session.finish(&command)?;
        drop(encoder);
        drop(command);
        for column in 0..2 {
            let start = (row_start + column)
                .checked_mul(span)
                .ok_or(MetalError::Shape("application replay offset overflow"))?;
            let end = start.saturating_add(span).min(self.source_rows);
            for source in (start..end).step_by(source_window) {
                self.record(bytes + table_bytes(self.table_count, source_window, size_of::<F>())?)?;
                let base =
                    session.build_joint_application_window(plan, masks, self.fresh_count, source_window, source)?;
                let folded = self.fold_window(session, base, source_window, stages, bytes)?;
                let index = (source - start) >> stages;
                let weight = self.challenges[stages..]
                    .iter()
                    .enumerate()
                    .fold(K::ONE, |weight, (bit, &challenge)| {
                        weight
                            * if (index >> bit) & 1 == 0 {
                                K::ONE - challenge
                            } else {
                                challenge
                            }
                    });
                let (real, imaginary) = weight.to_limbs_u64();
                let shape = session.buffer_from_slice(&[self.table_count as u64, 2, column as u64, real, imaginary])?;
                let command = session.command_buffer("nightstream.pi_ccs.application.replay.accumulate")?;
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&session.joint_accumulate_application);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&folded.values), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
                }
                session.dispatch(&encoder, &session.joint_accumulate_application, self.table_count);
                encoder.endEncoding();
                session.finish(&command)?;
            }
        }
        Ok(ApplicationWindow {
            values: output,
            stride: 2,
        })
    }

    pub(super) fn fold(&mut self, session: &MetalSession, challenge: K, reserved: usize) -> Result<(), MetalError> {
        self.update_budget(session, reserved);
        if let Some(window) = self.resident.take() {
            let next_rows = window.stride.div_ceil(2);
            let bytes = table_bytes(self.table_count, next_rows, size_of::<K>())?;
            if window
                .values
                .length()
                .checked_add(bytes)
                .is_some_and(|peak| peak <= self.budget)
            {
                self.record(window.values.length() + bytes)?;
                let output = session.buffer(bytes)?;
                let challenge = session.buffer_from_slice(&k_words(&[challenge]))?;
                let shape = session.buffer_from_slice(&[window.stride as u64, self.table_count as u64])?;
                let command = session.command_buffer("nightstream.pi_ccs.application.fold")?;
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                let pipeline = if self.challenges.is_empty() {
                    &session.joint_fold_base_tables
                } else {
                    &session.joint_fold_k_tables
                };
                encoder.setComputePipelineState(pipeline);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&window.values), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&challenge), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&output), 0, 3);
                }
                session.dispatch(&encoder, pipeline, self.table_count * next_rows);
                encoder.endEncoding();
                session.finish(&command)?;
                self.resident = Some(ApplicationWindow {
                    values: output,
                    stride: next_rows,
                });
            }
        }
        self.challenges.push(challenge);
        self.rows = self.rows.div_ceil(2);
        Ok(())
    }
}

impl MetalSession {
    pub(super) fn build_joint_application_tables(
        &self,
        plan: &MetalJointMatrixPlan<'_>,
        masks: &MetalWitnessMasks,
        fresh_count: usize,
        n_eff: usize,
    ) -> Result<Buffer, MetalError> {
        if n_eff > plan.rows {
            return Err(MetalError::Shape("application row count exceeds its plan"));
        }
        self.build_joint_application_window(plan, masks, fresh_count, n_eff, 0)
    }

    pub(super) fn build_joint_application_window(
        &self,
        plan: &MetalJointMatrixPlan<'_>,
        masks: &MetalWitnessMasks,
        fresh_count: usize,
        n_eff: usize,
        row_start: usize,
    ) -> Result<Buffer, MetalError> {
        if n_eff == 0 || row_start >= plan.rows {
            return Err(MetalError::Shape("one-joint application table shape is invalid"));
        }
        let valid_end = row_start
            .checked_add(n_eff)
            .ok_or(MetalError::Shape("application row window overflow"))?
            .min(plan.rows);
        let table_count = fresh_count
            .checked_mul(plan.matrix_count)
            .ok_or(MetalError::Shape("one-joint application table count overflow"))?;
        let output = self.zero_application_table(table_count, n_eff)?;

        let reserved = table_bytes(table_count, 12, size_of::<u64>())?;
        let mut next_row = row_start;
        while next_row < valid_end {
            let window = self.load_matrix_window(plan, next_row..valid_end, reserved)?;
            self.fill_application_from_matrix_window(
                plan,
                masks,
                fresh_count,
                row_start..row_start + n_eff,
                &window,
                &output,
            )?;
            next_row = window.rows.end;
            drop(window);
        }
        Ok(output)
    }
    fn zero_application_table(&self, table_count: usize, n_eff: usize) -> Result<Buffer, MetalError> {
        let bytes = table_bytes(table_count, n_eff, size_of::<u64>())?;
        let output = self.buffer(bytes)?;
        let command = self.command_buffer("nightstream.pi_ccs.joint.application.zero")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.joint_zero_words);
        unsafe { encoder.setBuffer_offset_atIndex(Some(&output), 0, 0) };
        self.dispatch(&encoder, &self.joint_zero_words, bytes / size_of::<u64>());
        encoder.endEncoding();
        self.finish(&command)?;
        drop(encoder);
        drop(command);

        Ok(output)
    }

    pub(super) fn build_application_from_matrix_window(
        &self,
        plan: &MetalJointMatrixPlan<'_>,
        masks: &MetalWitnessMasks,
        window: &super::matrix_window::MetalMatrixWindow,
    ) -> Result<Buffer, MetalError> {
        let output = self.zero_application_table(plan.matrix_count, window.rows.len())?;
        self.fill_application_from_matrix_window(plan, masks, 1, window.rows.clone(), window, &output)?;
        Ok(output)
    }

    fn fill_application_from_matrix_window(
        &self,
        plan: &MetalJointMatrixPlan<'_>,
        masks: &MetalWitnessMasks,
        fresh_count: usize,
        output_rows: std::ops::Range<usize>,
        window: &super::matrix_window::MetalMatrixWindow,
        output: &Buffer,
    ) -> Result<(), MetalError> {
        let local_rows = window.rows.end - window.rows.start;
        let command = self.command_buffer("nightstream.pi_ccs.joint.application")?;
        let mut resources = Vec::new();
        for source in 0..fresh_count.min(masks.stored_witnesses()) {
            for (matrix_index, matrix) in window.matrices.iter().enumerate() {
                let shape = self.buffer_from_slice(&[
                    local_rows as u64,
                    plan.blocks as u64,
                    plan.rows as u64,
                    output_rows.len() as u64,
                    source as u64,
                    (source * plan.matrix_count + matrix_index) as u64,
                    matrix.row_offset_width,
                    u64::from(matrix.identity),
                    masks.magnitudes() as u64,
                    matrix.geometric_row_offset_width,
                    window.rows.start as u64,
                    (window.rows.start - output_rows.start) as u64,
                ])?;
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.joint_build_application_tables);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&matrix.row_offsets), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&matrix.row_blocks), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&matrix.dense_offsets), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&matrix.dense_locals), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&matrix.dense_coefficients), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&matrix.geometric_row_offsets), 0, 5);
                    encoder.setBuffer_offset_atIndex(Some(&matrix.geometric_runs), 0, 6);
                    encoder.setBuffer_offset_atIndex(Some(masks.words()), 0, 7);
                    encoder.setBuffer_offset_atIndex(Some(&shape), 0, 8);
                    encoder.setBuffer_offset_atIndex(Some(output), 0, 9);
                    encoder.setBuffer_offset_atIndex(Some(&matrix.dense_row_blocks), 0, 10);
                }
                self.dispatch(&encoder, &self.joint_build_application_tables, local_rows);
                encoder.endEncoding();
                resources.push(shape);
            }
        }
        self.finish(&command)?;
        Ok(())
    }
}
