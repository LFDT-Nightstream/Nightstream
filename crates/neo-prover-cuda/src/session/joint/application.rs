//! Application matrix values: resident prefixes or bounded replay of original rows.

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

fn table_bytes(tables: usize, rows: usize, word_bytes: usize) -> Result<usize, CudaError> {
    tables
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(word_bytes))
        .ok_or(CudaError::Shape("application workspace size overflow"))
}

pub(super) fn available_workspace(_session: &CudaSession, reserved: usize) -> usize {
    crate::BUFFER_LIMIT_BYTES
        .saturating_sub(CudaSession::current_allocated_bytes())
        .saturating_sub(reserved)
}

impl ApplicationTables {
    pub(super) fn new(
        session: &CudaSession,
        plan: &CudaJointMatrixPlan<'_>,
        masks: &CudaWitnessMasks,
        fresh_count: usize,
        workspace_limit: Option<usize>,
        reserved: usize,
    ) -> Result<Self, CudaError> {
        let table_count = fresh_count
            .checked_mul(plan.matrix_count)
            .filter(|&n| n != 0)
            .ok_or(CudaError::Shape("empty application table family"))?;
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

    pub(super) fn scratch_bytes(plan: &CudaJointMatrixPlan<'_>, fresh_count: usize) -> Result<usize, CudaError> {
        let metadata = table_bytes(fresh_count, plan.matrix_count, 12 * size_of::<u64>())?;
        plan.workspace_bytes
            .checked_add(metadata)
            .and_then(|bytes| bytes.checked_add(7 * size_of::<u64>()))
            .ok_or(CudaError::Shape("application metadata size overflow"))
    }

    fn update_budget(&mut self, session: &CudaSession, reserved: usize) {
        let owned = self.resident.as_ref().map_or(0, |w| w.values.length());
        let _ = session;
        self.budget = crate::BUFFER_LIMIT_BYTES
            .saturating_sub(CudaSession::current_allocated_bytes().saturating_sub(owned))
            .saturating_sub(reserved)
            .min(self.workspace_limit.unwrap_or(usize::MAX));
    }

    fn record(&mut self, bytes: usize) -> Result<(), CudaError> {
        if bytes > self.budget {
            return Err(CudaError::MemoryLimit {
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

    fn replay_capacity(&self) -> Result<usize, CudaError> {
        let destination = table_bytes(self.table_count, 2, size_of::<K>())?;
        let mut rows = self
            .source_rows
            .checked_next_power_of_two()
            .ok_or(CudaError::Shape("application row domain overflow"))?
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
                return Err(CudaError::MemoryLimit {
                    requested: base.saturating_add(destination),
                    allocated: 0,
                    limit: self.budget,
                });
            }
            rows /= 2;
        }
    }

    pub(super) fn window_rows(&self) -> Result<usize, CudaError> {
        Ok((self.replay_capacity()? >> self.challenges.len()).max(2))
    }

    pub(super) fn prepare_round(
        &mut self,
        session: &CudaSession,
        plan: &CudaJointMatrixPlan<'_>,
        masks: &CudaWitnessMasks,
        reserved: usize,
    ) -> Result<(), CudaError> {
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
        session: &CudaSession,
        mut values: Buffer,
        mut rows: usize,
        stages: usize,
        external_bytes: usize,
    ) -> Result<ApplicationWindow, CudaError> {
        self.record(external_bytes + values.length())?;
        for stage in 0..stages {
            let next_rows = rows.div_ceil(2);
            let next = if stage == 0 {
                values.clone()
            } else {
                let bytes = table_bytes(self.table_count, next_rows, size_of::<K>())?;
                self.record(external_bytes + values.length() + bytes)?;
                session.buffer(bytes)?
            };
            session.fold_tables(
                &values,
                &next,
                self.challenges[stage],
                rows,
                self.table_count,
                stage == 0,
            )?;
            session.finish()?;
            values = next;
            rows = next_rows;
        }
        Ok(ApplicationWindow { values, stride: rows })
    }

    pub(super) fn replay_window(
        &mut self,
        session: &CudaSession,
        plan: &CudaJointMatrixPlan<'_>,
        masks: &CudaWitnessMasks,
        row_start: usize,
    ) -> Result<ApplicationWindow, CudaError> {
        let source_window = self.replay_capacity()?;
        let round = self.challenges.len();
        let stages = round.min(source_window.ilog2() as usize);
        let span = 1usize
            .checked_shl(round as u32)
            .ok_or(CudaError::Shape("application replay span overflow"))?;
        let source_start = row_start
            .checked_mul(span)
            .ok_or(CudaError::Shape("application replay offset overflow"))?;
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
        session.zero_words(&output, bytes / size_of::<u64>())?;
        session.finish()?;
        for column in 0..2 {
            let start = (row_start + column)
                .checked_mul(span)
                .ok_or(CudaError::Shape("application replay offset overflow"))?;
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
                let shape = [self.table_count as u64, 2, column as u64, real, imaginary];
                let kernels = &session.kernels.joint;
                session.launch(Grid::Threads(self.table_count), |stream, config| {
                    kernels.joint_accumulate_application(stream, config, folded.values.ptr(), shape, output.ptr())
                })?;
                session.finish()?;
            }
        }
        Ok(ApplicationWindow {
            values: output,
            stride: 2,
        })
    }

    pub(super) fn fold(&mut self, session: &CudaSession, challenge: K, reserved: usize) -> Result<(), CudaError> {
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
                session.fold_tables(
                    &window.values,
                    &output,
                    challenge,
                    window.stride,
                    self.table_count,
                    self.challenges.is_empty(),
                )?;
                session.finish()?;
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

impl CudaSession {
    pub(super) fn build_joint_application_tables(
        &self,
        plan: &CudaJointMatrixPlan<'_>,
        masks: &CudaWitnessMasks,
        fresh_count: usize,
        n_eff: usize,
    ) -> Result<Buffer, CudaError> {
        if n_eff > plan.rows {
            return Err(CudaError::Shape("application row count exceeds its plan"));
        }
        self.build_joint_application_window(plan, masks, fresh_count, n_eff, 0)
    }

    pub(super) fn build_joint_application_window(
        &self,
        plan: &CudaJointMatrixPlan<'_>,
        masks: &CudaWitnessMasks,
        fresh_count: usize,
        n_eff: usize,
        row_start: usize,
    ) -> Result<Buffer, CudaError> {
        if n_eff == 0 || row_start >= plan.rows {
            return Err(CudaError::Shape("one-joint application table shape is invalid"));
        }
        let valid_end = row_start
            .checked_add(n_eff)
            .ok_or(CudaError::Shape("application row window overflow"))?
            .min(plan.rows);
        let table_count = fresh_count
            .checked_mul(plan.matrix_count)
            .ok_or(CudaError::Shape("one-joint application table count overflow"))?;
        let output = self.zero_application_table(table_count, n_eff)?;

        let reserved = table_bytes(table_count, 12, size_of::<u64>())?;
        let mut next_row = row_start;
        while next_row < valid_end {
            let window = self.load_matrix_window(plan, next_row..valid_end, reserved, true)?;
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
    fn zero_application_table(&self, table_count: usize, n_eff: usize) -> Result<Buffer, CudaError> {
        let bytes = table_bytes(table_count, n_eff, size_of::<u64>())?;
        let output = self.buffer(bytes)?;
        self.zero_words(&output, bytes / size_of::<u64>())?;
        self.finish()?;
        Ok(output)
    }

    /// Zero the first `words` words of `buffer`.
    pub(super) fn zero_words(&self, buffer: &Buffer, words: usize) -> Result<(), CudaError> {
        let kernels = &self.kernels.joint;
        self.launch(Grid::Threads(words), |stream, config| {
            kernels.joint_zero_words(stream, config, buffer.ptr(), words as u64)
        })
    }

    /// Fold `table_count` tables of `table_len` entries (base-field entries
    /// when `base`) at `challenge` into `output`.
    pub(super) fn fold_tables(
        &self,
        input: &Buffer,
        output: &Buffer,
        challenge: K,
        table_len: usize,
        table_count: usize,
        base: bool,
    ) -> Result<(), CudaError> {
        let (real, imaginary) = challenge.to_limbs_u64();
        let challenge = [real, imaginary];
        let shape = [table_len as u64, table_count as u64];
        let kernels = &self.kernels.joint;
        self.launch(Grid::Threads(table_count * table_len.div_ceil(2)), |stream, config| {
            if base {
                kernels.joint_fold_base_tables(stream, config, input.ptr(), challenge, shape, output.ptr())
            } else {
                kernels.joint_fold_k_tables(stream, config, input.ptr(), challenge, shape, output.ptr())
            }
        })
    }

    pub(super) fn build_application_from_matrix_window(
        &self,
        plan: &CudaJointMatrixPlan<'_>,
        masks: &CudaWitnessMasks,
        window: &super::matrix_window::CudaMatrixWindow,
    ) -> Result<Buffer, CudaError> {
        let output = self.zero_application_table(plan.matrix_count, window.rows.len())?;
        self.fill_application_from_matrix_window(plan, masks, 1, window.rows.clone(), window, &output)?;
        Ok(output)
    }

    fn fill_application_from_matrix_window(
        &self,
        plan: &CudaJointMatrixPlan<'_>,
        masks: &CudaWitnessMasks,
        fresh_count: usize,
        output_rows: std::ops::Range<usize>,
        window: &super::matrix_window::CudaMatrixWindow,
        output: &Buffer,
    ) -> Result<(), CudaError> {
        let local_rows = window.rows.end - window.rows.start;
        let kernels = &self.kernels.joint;
        for source in 0..fresh_count.min(masks.stored_witnesses()) {
            for (matrix_index, matrix) in window.matrices.iter().enumerate() {
                let shape = [
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
                ];
                self.launch(Grid::Threads(local_rows), |stream, config| {
                    kernels.joint_build_application_tables(
                        stream,
                        config,
                        matrix.row_offsets.ptr(),
                        matrix.row_blocks.ptr(),
                        matrix.dense_offsets.ptr(),
                        matrix.dense_locals.ptr(),
                        matrix.dense_coefficients.ptr(),
                        matrix.geometric_row_offsets.ptr(),
                        matrix.geometric_runs.ptr(),
                        masks.words().ptr(),
                        shape,
                        output.ptr(),
                        matrix.dense_row_blocks.ptr(),
                    )
                })?;
            }
        }
        self.finish()?;
        Ok(())
    }
}
