//! FE (row-phase) device oracle and its prove-driver backend.
//!
//! Table data comes from `OptimizedOracle::row_phase_snapshot`; every round
//! must stay field-identical to `RowStreamState::evals_row_phase` + fold.

use cuda_core::{DeviceBuffer, PinnedHostBuffer};
use neo_ccs::Mat;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::oracle::{RowPhaseSnapshot, RowTableSnapshot};
use neo_reductions::optimized_engine::{
    Challenges, FeEvalTable, FeMcsRowTables, FePhaseTraceRequest, FeRowRoundSummary, FeRowRoundTrace, FeSumcheckBackend,
};
use neo_reductions::superneo_eval::{SuperneoEvalCache, SuperneoZBlocks};
use p3_field::PrimeCharacteristicRing;

use crate::device::{copy_host_to_device, Device};
use crate::field::k_from_device_words;
use crate::graph::GraphAllocations;
use crate::kernels::ajtai::{launch_plane_copy, launch_plane_copy_slice, RingMatVecScratch};
use crate::kernels::csr::launch_tensor_point_k_at;
use crate::kernels::pi_ccs_fe::{
    launch_fe_cooperative_row_round, launch_fe_round_partials, EVAL_CHUNK_PAIRS, MAX_WIDTH, NO_TABLE,
};
use crate::kernels::sumcheck_common::{
    launch_sum_partials, launch_table_fold, launch_table_fold_from_challenge, SUM_BLOCKS,
};
use crate::reduce::ccs::{CcsDeviceError, DevicePiCcsKSurfaces, FeOracleWorkspace, FePhaseWorkspace, SumcheckKernels};
use crate::ring_forms::{DeviceBarMatrices, DeviceRowMatrices};
use crate::transcript::DeviceTranscript;

use super::fe_oracle_workspace::{store_buffer, store_pinned, take_buffer, take_pinned};
use super::oracle_plan::{DeferredEvalTable, DeferredMcsRowTables, DevicePiCcsOraclePlan};

mod tail;

pub(super) const TAIL_CHALLENGE_HEADER_WORDS: usize = 12;

/// Device-resident `Y_eval[witness][matrix][lane]` surface used by the
/// Ajtai tail. Layout matches the ring mat-vec output: for each witness,
/// matrix `j` stores real lanes then imaginary lanes. The matching ring
/// forms stay attached so Pi_DEC can consume the same `bar(M)^T * chi(r)`
/// value instead of rebuilding it later in the fold.
pub struct DeviceAjtaiYEval {
    pub(super) words: DeviceBuffer<u64>,
    pub(super) forms: Option<DeviceBuffer<u64>>,
    pub(super) witnesses: usize,
    pub(super) matrices: usize,
}

impl DeviceAjtaiYEval {
    pub fn words(&self) -> &DeviceBuffer<u64> {
        &self.words
    }

    pub(crate) fn into_parts(self) -> (DeviceBuffer<u64>, Option<DeviceBuffer<u64>>) {
        (self.words, self.forms)
    }

    pub fn witnesses(&self) -> usize {
        self.witnesses
    }

    pub fn matrices(&self) -> usize {
        self.matrices
    }
}

/// One Ajtai-tail round request after the row point `r'` is fixed.
pub struct DeviceFeTailRound<'a> {
    pub alpha: &'a [K],
    pub beta_a: &'a [K],
    pub prefix: &'a [K],
    pub gamma: K,
    pub eq_beta_r: K,
    pub eq_r_inputs: K,
    pub k_mcs: usize,
    pub has_inputs: bool,
}

#[derive(Clone, Copy)]
pub(super) struct FePhaseLogShape {
    pub(super) total_rounds: usize,
    pub(super) coeff_words_per_round: usize,
    pub(super) width: usize,
}

#[derive(Clone, Copy)]
enum HostTableSource<'a> {
    Extension(&'a [K]),
    Split(RowTableSnapshot<'a>),
}

enum FeTableSource<'a> {
    Host(HostTableSource<'a>),
    TensorPoint(&'a [K]),
    Deferred { mcs_idx: usize, var_pos: usize },
    DeferredEval,
}

fn tensor_point_len(point: &[K]) -> Result<usize, CcsDeviceError> {
    1usize
        .checked_shl(point.len() as u32)
        .ok_or(CcsDeviceError::Shape("row equality point is too wide"))
}

fn tensor_point_words(point: &[K]) -> Vec<u64> {
    let mut words = Vec::with_capacity(point.len() * 2);
    for value in point {
        let (c0, c1) = value.to_limbs_u64();
        words.extend([c0, c1]);
    }
    words
}

fn write_table_words(
    words: &mut [u64],
    slot: usize,
    table: HostTableSource<'_>,
    stride: usize,
) -> Result<(), CcsDeviceError> {
    let len = match table {
        HostTableSource::Extension(values) => values.len(),
        HostTableSource::Split(table) => table.real.len(),
    };
    if len != stride {
        return Err(CcsDeviceError::Shape("row tables must share cur_len"));
    }
    let base = slot * stride * 2;
    match table {
        HostTableSource::Extension(values) => {
            for (i, value) in values.iter().enumerate() {
                let (c0, c1) = value.to_limbs_u64();
                words[base + 2 * i] = c0;
                words[base + 2 * i + 1] = c1;
            }
        }
        HostTableSource::Split(table) => {
            for (i, &real) in table.real.iter().enumerate() {
                let imag = table.imag.map_or(F::ZERO, |imag| imag[i]);
                let (c0, c1) = K::from_coeffs([real, imag]).to_limbs_u64();
                words[base + 2 * i] = c0;
                words[base + 2 * i + 1] = c1;
            }
        }
    }
    Ok(())
}

fn deferred_table_slice(
    tables: &[DeferredMcsRowTables],
    mcs_idx: usize,
    var_pos: usize,
    stride: usize,
) -> Result<(&DeviceBuffer<u64>, usize), CcsDeviceError> {
    let mcs = tables
        .iter()
        .find(|tables| tables.mcs_idx == mcs_idx)
        .ok_or(CcsDeviceError::Shape("deferred MCS row table missing"))?;
    if mcs.n_pad != stride {
        return Err(CcsDeviceError::Shape("deferred MCS row table length mismatch"));
    }
    if var_pos >= mcs.table_count {
        return Err(CcsDeviceError::Shape("deferred MCS variable table missing"));
    }
    let offset = var_pos * stride * 2;
    if mcs.packed.len() < offset + stride * 2 {
        return Err(CcsDeviceError::Shape("deferred MCS row table word length mismatch"));
    }
    Ok((&mcs.packed, offset))
}

fn deferred_eval_table(table: Option<&DeferredEvalTable>, stride: usize) -> Result<&DeviceBuffer<u64>, CcsDeviceError> {
    let table = table.ok_or(CcsDeviceError::Shape("deferred carried eval table missing"))?;
    if table.n_pad != stride || table.table.len() != stride * 2 {
        return Err(CcsDeviceError::Shape("deferred carried eval table length mismatch"));
    }
    Ok(&table.table)
}

/// Device-resident FE (row-phase) sumcheck oracle.
pub struct DeviceFeOracle {
    /// Ping-pong table buffers; `front_is_a` marks the live one.
    pub(super) tables_a: DeviceBuffer<u64>,
    pub(super) tables_b: DeviceBuffer<u64>,
    pub(super) front_is_a: bool,
    pub(super) header: DeviceBuffer<u64>,
    pub(super) mcs_meta: DeviceBuffer<u64>,
    pub(super) term_meta: DeviceBuffer<u64>,
    pub(super) term_vars: DeviceBuffer<u64>,
    pub(super) num_tables: usize,
    /// K-elements reserved per table region (the initial `cur_len`).
    pub(super) stride: usize,
    pub(super) cur_len: usize,
    pub(super) active_len: usize,
    /// Canonical coefficient count used by the CPU proof/transcript path.
    pub(super) coeff_width: usize,
    deg_max: usize,
    /// Round-eval scratch, allocated once: per-round allocation/free pairs
    /// implicitly synchronize the device and dominate the round loop.
    pub(super) partials: DeviceBuffer<u64>,
    pub(super) sum_scratch: DeviceBuffer<u64>,
    pub(super) coeffs_out: DeviceBuffer<u64>,
    /// Pinned readback target: a pageable-destination D2H pays a staging
    /// cost even for a hundred bytes; pinned memory copies directly.
    coeffs_host: PinnedHostBuffer<u64>,
}

impl DeviceFeOracle {
    pub fn from_snapshot(
        device: &Device,
        kernels: &SumcheckKernels,
        snapshot: &RowPhaseSnapshot<'_>,
    ) -> Result<Self, CcsDeviceError> {
        Self::from_snapshot_with_deferred(device, kernels, snapshot, &[], None)
    }

    fn from_snapshot_with_deferred(
        device: &Device,
        kernels: &SumcheckKernels,
        snapshot: &RowPhaseSnapshot<'_>,
        deferred_tables: &[DeferredMcsRowTables],
        deferred_eval: Option<&DeferredEvalTable>,
    ) -> Result<Self, CcsDeviceError> {
        let mut workspace = FeOracleWorkspace::new();
        Self::from_snapshot_with_deferred_and_workspace(
            device,
            kernels,
            snapshot,
            deferred_tables,
            deferred_eval,
            &mut workspace,
        )
    }

    fn from_snapshot_with_deferred_and_workspace(
        device: &Device,
        kernels: &SumcheckKernels,
        snapshot: &RowPhaseSnapshot<'_>,
        deferred_tables: &[DeferredMcsRowTables],
        deferred_eval: Option<&DeferredEvalTable>,
        workspace: &mut FeOracleWorkspace,
    ) -> Result<Self, CcsDeviceError> {
        let coeff_width = snapshot.sumcheck_degree_bound + 1;
        if coeff_width > MAX_WIDTH {
            return Err(CcsDeviceError::Shape(
                "row-phase canonical sumcheck degree exceeds kernel MAX_WIDTH",
            ));
        }
        if snapshot.row_phase_deg_max + 1 > coeff_width {
            return Err(CcsDeviceError::Shape(
                "row-phase degree exceeds canonical sumcheck degree",
            ));
        }
        if snapshot.row_phase_deg_max + 1 > MAX_WIDTH {
            return Err(CcsDeviceError::Shape("row-phase degree exceeds kernel MAX_WIDTH"));
        }
        // The eval kernel only implements the fast term shapes
        // (`CompiledPolyTermKind`); anything else stays on the CPU.
        for (_, vars) in &snapshot.f_terms {
            let fast_shape = match vars.as_slice() {
                [] | [(_, 1)] | [(_, 1), (_, 1)] => true,
                [(_, exp)] => *exp <= 8,
                _ => false,
            };
            if !fast_shape {
                return Err(CcsDeviceError::Shape("f term without a per-coefficient closed form"));
            }
        }

        let stride = snapshot.cur_len;
        if tensor_point_len(snapshot.beta_r)? != stride
            || (snapshot.eq_beta_r_tbl.len() != stride && !snapshot.eq_beta_r_tbl.is_empty())
        {
            return Err(CcsDeviceError::Shape("row beta point/table length mismatch"));
        }

        // Table slots: eq_beta_r first, optionals next, then the var tables
        // of each nonzero MCS contiguously. Equality tables are generated on
        // device from their point instead of uploaded as host-built tables.
        let mut table_sources = vec![FeTableSource::TensorPoint(snapshot.beta_r)];
        let r_inputs_slot = if snapshot.eq_r_inputs_tbl.is_some() {
            let r_inputs = snapshot
                .r_inputs
                .ok_or(CcsDeviceError::Shape("row input point missing for equality table"))?;
            if tensor_point_len(r_inputs)? != stride {
                return Err(CcsDeviceError::Shape("row input point/table length mismatch"));
            }
            table_sources.push(FeTableSource::TensorPoint(r_inputs));
            Some(table_sources.len() as u64 - 1)
        } else {
            None
        };
        let eval_slot = if let Some(tbl) = snapshot.eval_tbl {
            table_sources.push(FeTableSource::Host(HostTableSource::Extension(tbl)));
            Some(table_sources.len() as u64 - 1)
        } else if snapshot.deferred_eval_tbl {
            deferred_eval_table(deferred_eval, stride)?;
            table_sources.push(FeTableSource::DeferredEval);
            Some(table_sources.len() as u64 - 1)
        } else {
            None
        };
        let mut mcs_meta_words = Vec::with_capacity(snapshot.gamma_pow_mcs.len() * 4);
        for (mcs, tables) in snapshot.f_var_tables_by_mcs.iter().enumerate() {
            let gamma = snapshot.gamma_pow_mcs.get(mcs).copied().unwrap_or(K::ONE);
            let (c0, c1) = gamma.to_limbs_u64();
            let zero = snapshot.zero_mcs[mcs];
            let var_slot_base = if zero {
                NO_TABLE
            } else {
                let base = table_sources.len() as u64;
                if snapshot.deferred_mcs.get(mcs).copied().unwrap_or(false) {
                    for var_pos in 0..snapshot.f_var_count {
                        table_sources.push(FeTableSource::Deferred { mcs_idx: mcs, var_pos });
                    }
                } else {
                    table_sources.extend(
                        tables
                            .iter()
                            .copied()
                            .map(HostTableSource::Split)
                            .map(FeTableSource::Host),
                    );
                }
                base
            };
            mcs_meta_words.extend([c0, c1, zero as u64, var_slot_base]);
        }

        let table_words_len = table_sources.len() * stride * 2;
        let needs_mixed_upload = table_sources
            .iter()
            .any(|source| !matches!(source, FeTableSource::Host(_)));

        let mut term_meta_words = Vec::with_capacity(snapshot.f_terms.len() * 4);
        let mut term_var_words = Vec::new();
        for (coeff, vars) in &snapshot.f_terms {
            let (c0, c1) = coeff.to_limbs_u64();
            term_meta_words.extend([c0, c1, (term_var_words.len() / 2) as u64, vars.len() as u64]);
            for &(var_pos, exp) in vars {
                term_var_words.extend([var_pos as u64, exp as u64]);
            }
        }

        let (g_c0, g_c1) = snapshot.gamma_to_k.to_limbs_u64();
        let (f0_c0, f0_c1) = snapshot.f_at_zero.to_limbs_u64();
        let header_words = [
            snapshot.row_phase_deg_max as u64,
            snapshot.f_var_tables_by_mcs.len() as u64,
            snapshot.f_terms.len() as u64,
            r_inputs_slot.unwrap_or(NO_TABLE),
            eval_slot.unwrap_or(NO_TABLE),
            g_c0,
            g_c1,
            f0_c0,
            f0_c1,
        ];

        let stream = device.stream();
        let (mut tables_a, tables_b);
        perf_timed!("fold.superneo.pi_ccs.oracle.upload", {
            tables_a = take_buffer(&mut workspace.tables_a, stream, table_words_len)?;
            if needs_mixed_upload {
                let mut slot = 0usize;
                while slot < table_sources.len() {
                    match &table_sources[slot] {
                        FeTableSource::Host(_) => {
                            let start = slot;
                            let mut end = slot;
                            while matches!(table_sources.get(end), Some(FeTableSource::Host(_))) {
                                end += 1;
                            }
                            let mut words = vec![0u64; (end - start) * stride * 2];
                            for (local_slot, source) in table_sources[start..end].iter().enumerate() {
                                let FeTableSource::Host(table) = source else {
                                    unreachable!("host run checked above");
                                };
                                write_table_words(&mut words, local_slot, *table, stride)?;
                            }
                            let staging = take_buffer(&mut workspace.upload_staging, stream, words.len())?;
                            copy_host_to_device(stream, &staging, &words)?;
                            launch_plane_copy(&kernels.ring, stream, &staging, start * stride * 2, &mut tables_a)?;
                            store_buffer(&mut workspace.upload_staging, staging);
                            slot = end;
                        }
                        FeTableSource::TensorPoint(point) => {
                            if tensor_point_len(point)? != stride {
                                return Err(CcsDeviceError::Shape("row equality point length mismatch"));
                            }
                            let point_words = tensor_point_words(point);
                            let point_dev = take_buffer(&mut workspace.upload_staging, stream, point_words.len())?;
                            copy_host_to_device(stream, &point_dev, &point_words)?;
                            launch_tensor_point_k_at(
                                &kernels.csr,
                                stream,
                                &point_dev,
                                point.len(),
                                slot * stride * 2,
                                &mut tables_a,
                            )?;
                            store_buffer(&mut workspace.upload_staging, point_dev);
                            slot += 1;
                        }
                        FeTableSource::Deferred { mcs_idx, var_pos } => {
                            let (table, src_offset) =
                                deferred_table_slice(deferred_tables, *mcs_idx, *var_pos, stride)?;
                            launch_plane_copy_slice(
                                &kernels.ring,
                                stream,
                                table,
                                src_offset,
                                slot * stride * 2,
                                stride * 2,
                                &mut tables_a,
                            )?;
                            slot += 1;
                        }
                        FeTableSource::DeferredEval => {
                            let table = deferred_eval_table(deferred_eval, stride)?;
                            launch_plane_copy(&kernels.ring, stream, table, slot * stride * 2, &mut tables_a)?;
                            slot += 1;
                        }
                    }
                }
            } else {
                let mut words = vec![0u64; table_words_len];
                for (slot, source) in table_sources.iter().enumerate() {
                    let FeTableSource::Host(table) = source else {
                        unreachable!("needs_mixed_upload checked above");
                    };
                    write_table_words(&mut words, slot, *table, stride)?;
                }
                copy_host_to_device(stream, &tables_a, &words)?;
            }
            tables_b = take_buffer(&mut workspace.tables_b, stream, table_words_len)?;
        });
        let max_groups = (snapshot.active_len.div_ceil(2))
            .div_ceil(EVAL_CHUNK_PAIRS)
            .max(1);
        let header = take_buffer(&mut workspace.header, stream, header_words.len())?;
        copy_host_to_device(stream, &header, &header_words)?;
        let mcs_meta = take_buffer(&mut workspace.mcs_meta, stream, mcs_meta_words.len())?;
        copy_host_to_device(stream, &mcs_meta, &mcs_meta_words)?;
        let term_meta = take_buffer(&mut workspace.term_meta, stream, term_meta_words.len())?;
        copy_host_to_device(stream, &term_meta, &term_meta_words)?;
        let term_vars = take_buffer(&mut workspace.term_vars, stream, term_var_words.len())?;
        copy_host_to_device(stream, &term_vars, &term_var_words)?;
        let partials = take_buffer(&mut workspace.partials, stream, max_groups * coeff_width * 2)?;
        let sum_scratch = take_buffer(&mut workspace.sum_scratch, stream, SUM_BLOCKS * coeff_width * 2)?;
        let coeffs_out = take_buffer(&mut workspace.coeffs_out, stream, coeff_width * 2)?;
        let coeffs_host = take_pinned(&mut workspace.coeffs_host, device, coeff_width * 2)?;
        Ok(Self {
            tables_a,
            tables_b,
            front_is_a: true,
            header,
            mcs_meta,
            term_meta,
            term_vars,
            num_tables: table_sources.len(),
            stride,
            cur_len: snapshot.cur_len,
            active_len: snapshot.active_len,
            coeff_width,
            deg_max: snapshot.row_phase_deg_max,
            partials,
            sum_scratch,
            coeffs_out,
            coeffs_host,
        })
    }

    fn return_to_workspace(self, workspace: &mut FeOracleWorkspace) {
        store_buffer(&mut workspace.tables_a, self.tables_a);
        store_buffer(&mut workspace.tables_b, self.tables_b);
        store_buffer(&mut workspace.header, self.header);
        store_buffer(&mut workspace.mcs_meta, self.mcs_meta);
        store_buffer(&mut workspace.term_meta, self.term_meta);
        store_buffer(&mut workspace.term_vars, self.term_vars);
        // `upload_staging` is returned immediately after each staged upload.
        store_buffer(&mut workspace.partials, self.partials);
        store_buffer(&mut workspace.sum_scratch, self.sum_scratch);
        store_buffer(&mut workspace.coeffs_out, self.coeffs_out);
        store_pinned(&mut workspace.coeffs_host, self.coeffs_host);
    }

    /// This round's canonical univariate coefficients (low→high), written
    /// to `coeffs_out` at the same width the CPU proof serializes.
    pub(super) fn write_round_coeffs(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
    ) -> Result<usize, CcsDeviceError> {
        let tail_len = self.active_len.div_ceil(2);
        let width = self.coeff_width;
        let groups = tail_len.div_ceil(EVAL_CHUNK_PAIRS).max(1);
        let front = if self.front_is_a {
            &self.tables_a
        } else {
            &self.tables_b
        };
        launch_fe_round_partials(
            &kernels.fe,
            device.stream(),
            front,
            &self.header,
            &self.mcs_meta,
            &self.term_meta,
            &self.term_vars,
            self.stride,
            tail_len,
            groups,
            width,
            &mut self.partials,
        )?;
        launch_sum_partials(
            &kernels.common,
            device.stream(),
            &self.partials,
            groups,
            width * 2,
            &mut self.sum_scratch,
            &mut self.coeffs_out,
        )?;
        Ok(width)
    }

    /// This round's canonical univariate coefficients (low→high), identical
    /// to the CPU proof/transcript encoding.
    pub fn round_coeffs(&mut self, device: &Device, kernels: &SumcheckKernels) -> Result<Vec<K>, CcsDeviceError> {
        let width = self.write_round_coeffs(device, kernels)?;
        self.coeffs_out
            .copy_to_pinned_host(device.stream(), &mut self.coeffs_host)?;
        let words = self.coeffs_host.as_slice();
        Ok((0..width)
            .map(|d| k_from_device_words(words[2 * d], words[2 * d + 1]))
            .collect())
    }

    /// Fold every table at the sampled challenge; lengths halve.
    pub fn fold(&mut self, device: &Device, kernels: &SumcheckKernels, r: K) -> Result<(), CcsDeviceError> {
        if self.cur_len < 2 {
            return Err(CcsDeviceError::Shape("fold below length 2"));
        }
        let (r_c0, r_c1) = r.to_limbs_u64();
        let (src, dst) = if self.front_is_a {
            (&self.tables_a, &mut self.tables_b)
        } else {
            (&self.tables_b, &mut self.tables_a)
        };
        launch_table_fold(
            &kernels.common,
            device.stream(),
            src,
            self.num_tables,
            self.stride,
            self.cur_len,
            r_c0,
            r_c1,
            dst,
        )?;
        self.front_is_a = !self.front_is_a;
        self.cur_len /= 2;
        self.active_len = self.active_len.div_ceil(2).max(1);
        Ok(())
    }

    pub(super) fn fold_from_challenge(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        challenges: &DeviceBuffer<u64>,
        challenge_offset: usize,
    ) -> Result<(), CcsDeviceError> {
        if self.cur_len < 2 {
            return Err(CcsDeviceError::Shape("fold below length 2"));
        }
        let (src, dst) = if self.front_is_a {
            (&self.tables_a, &mut self.tables_b)
        } else {
            (&self.tables_b, &mut self.tables_a)
        };
        launch_table_fold_from_challenge(
            &kernels.common,
            device.stream(),
            src,
            self.num_tables,
            self.stride,
            self.cur_len,
            challenges,
            challenge_offset,
            dst,
        )?;
        self.front_is_a = !self.front_is_a;
        self.cur_len /= 2;
        self.active_len = self.active_len.div_ceil(2).max(1);
        Ok(())
    }

    pub(super) fn mark_row_rounds_replayed(&mut self, rounds: usize) -> Result<(), CcsDeviceError> {
        for _ in 0..rounds {
            if self.cur_len < 2 {
                return Err(CcsDeviceError::Shape("replayed FE fold below length 2"));
            }
            self.front_is_a = !self.front_is_a;
            self.cur_len /= 2;
            self.active_len = self.active_len.div_ceil(2).max(1);
        }
        Ok(())
    }

    pub(super) fn record_graph_allocations(&self, allocations: &mut GraphAllocations) {
        allocations.push(&self.tables_a);
        allocations.push(&self.tables_b);
        allocations.push(&self.header);
        allocations.push(&self.mcs_meta);
        allocations.push(&self.term_meta);
        allocations.push(&self.term_vars);
        allocations.push(&self.partials);
        allocations.push(&self.sum_scratch);
        allocations.push(&self.coeffs_out);
    }

    fn row_round_trace_from_transcript_with_workspace(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        phase_workspace: &mut FePhaseWorkspace,
        transcript_state: [F; crate::kernels::poseidon2::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Result<FeRowRoundTrace, CcsDeviceError> {
        let width = self.coeff_width;
        let stream = device.stream();
        let coeff_words_per_round = width * 2;
        phase_workspace.prepare_logs(stream, rounds * coeff_words_per_round, rounds * 2)?;
        phase_workspace.reset_transcript(device, transcript_state, transcript_absorbed)?;

        crate::perf_timed!("fold.superneo.pi_ccs.sumcheck.fe.row_enqueue_loop", {
            for round in 0..rounds {
                let coeff_offset = round * coeff_words_per_round;
                let emitted_width = self.write_round_coeffs(device, kernels)?;
                if emitted_width != width {
                    return Err(CcsDeviceError::Shape("FE round width changed"));
                }
                launch_plane_copy(
                    &kernels.ring,
                    stream,
                    &self.coeffs_out,
                    coeff_offset,
                    phase_workspace.coeff_log_mut(),
                )?;
                phase_workspace.enqueue_coeff_challenge(
                    device,
                    &kernels.poseidon,
                    &kernels.poseidon_rc,
                    &self.coeffs_out,
                    coeff_words_per_round,
                    2 * round,
                )?;
                self.fold_from_challenge(device, kernels, phase_workspace.challenges(), 2 * round)?;
            }
        });

        let transcript_words;
        let coeff_words;
        let challenge_words;
        crate::perf_timed!("fold.superneo.pi_ccs.sumcheck.fe.row_download", {
            transcript_words = phase_workspace.transcript_state_words_to_host(device)?;
            coeff_words = phase_workspace.coeff_log().to_host_vec(stream)?;
            challenge_words = phase_workspace.challenges().to_host_vec(stream)?;
            device.sync()?;
        });

        let coeffs;
        let challenges;
        let transcript_after;
        crate::perf_timed!("fold.superneo.pi_ccs.sumcheck.fe.row_decode", {
            coeffs = (0..rounds)
                .map(|round| {
                    let base = round * coeff_words_per_round;
                    (0..width)
                        .map(|d| k_from_device_words(coeff_words[base + 2 * d], coeff_words[base + 2 * d + 1]))
                        .collect()
                })
                .collect();
            challenges = (0..rounds)
                .map(|round| k_from_device_words(challenge_words[2 * round], challenge_words[2 * round + 1]))
                .collect();
            transcript_after = Some(DeviceTranscript::decode_state_words(&transcript_words));
        });
        Ok(FeRowRoundTrace {
            coeffs,
            challenges,
            transcript_after,
            ajtai_y_eval: None,
        })
    }

    fn row_round_trace_from_transcript_cooperative_with_workspace(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        phase_workspace: &mut FePhaseWorkspace,
        transcript_state: [F; crate::kernels::poseidon2::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Result<FeRowRoundTrace, CcsDeviceError> {
        let width = self.coeff_width;
        let stream = device.stream();
        let coeff_words_per_round = width * 2;
        phase_workspace.prepare_logs(stream, rounds * coeff_words_per_round, rounds * 2)?;
        phase_workspace.reset_transcript(device, transcript_state, transcript_absorbed)?;

        for round in 0..rounds {
            if self.cur_len < 2 {
                return Err(CcsDeviceError::Shape("cooperative FE fold below length 2"));
            }
            let tail_len = self.active_len.div_ceil(2);
            let groups = tail_len.div_ceil(EVAL_CHUNK_PAIRS).max(1);
            let (src, dst) = if self.front_is_a {
                (&self.tables_a, &mut self.tables_b)
            } else {
                (&self.tables_b, &mut self.tables_a)
            };
            let (transcript_state, coeff_log, challenges) = phase_workspace.cooperative_row_round_buffers();
            launch_fe_cooperative_row_round(
                &kernels.fe,
                stream,
                src,
                &self.header,
                &self.mcs_meta,
                &self.term_meta,
                &self.term_vars,
                self.stride,
                tail_len,
                groups,
                width,
                self.num_tables,
                self.cur_len,
                &mut self.partials,
                &mut self.sum_scratch,
                &mut self.coeffs_out,
                transcript_state,
                coeff_log,
                round * coeff_words_per_round,
                challenges,
                2 * round,
                &kernels.poseidon_rc,
                dst,
            )?;
            self.front_is_a = !self.front_is_a;
            self.cur_len /= 2;
            self.active_len = self.active_len.div_ceil(2).max(1);
        }

        let transcript_words = phase_workspace.transcript_state_words_to_host(device)?;
        let coeff_words = phase_workspace.coeff_log().to_host_vec(stream)?;
        let challenge_words = phase_workspace.challenges().to_host_vec(stream)?;
        device.sync()?;

        let coeffs = (0..rounds)
            .map(|round| {
                let base = round * coeff_words_per_round;
                (0..width)
                    .map(|d| k_from_device_words(coeff_words[base + 2 * d], coeff_words[base + 2 * d + 1]))
                    .collect()
            })
            .collect();
        let challenges = (0..rounds)
            .map(|round| k_from_device_words(challenge_words[2 * round], challenge_words[2 * round + 1]))
            .collect();
        let transcript_after = Some(DeviceTranscript::decode_state_words(&transcript_words));
        Ok(FeRowRoundTrace {
            coeffs,
            challenges,
            transcript_after,
            ajtai_y_eval: None,
        })
    }

    pub fn cur_len(&self) -> usize {
        self.cur_len
    }

    pub fn deg_max(&self) -> usize {
        self.deg_max
    }
}

/// The `FeSumcheckBackend` the CPU prove driver calls: builds the device
/// oracle from the row-phase snapshot at `start` and drives it through the
/// row rounds. Device errors after `start` panic — the transcript has
/// already advanced, so falling back mid-prove would desync the proof.
pub struct DeviceFeBackend<'a> {
    pub(super) device: &'a Device,
    pub(super) kernels: &'a SumcheckKernels,
    pub(super) oracle: Option<DeviceFeOracle>,
    /// Device-owned preparation plan for the Π_CCS oracle inputs.
    pub(super) oracle_plan: DevicePiCcsOraclePlan<'a>,
    /// Running-child claim surfaces retained by the previous CUDA fold.
    running_surfaces: Option<&'a DevicePiCcsKSurfaces>,
    /// Reused ring mat-vec stage buffers for the Ajtai `Y_eval`.
    pub(super) ring_scratch: RingMatVecScratch,
    /// Reused whole-FE trace buffers, owned by `DeviceSession` between folds.
    pub(super) phase_workspace: Option<FePhaseWorkspace>,
    /// Reused FE row-oracle buffers, owned by `DeviceSession` between folds.
    oracle_workspace: Option<FeOracleWorkspace>,
    /// Last device `Y_eval` surface returned to the CPU oracle. The adapter
    /// may consume this immediately as the Pi_CCS -> Pi_RLC resident handoff.
    pub(super) last_y_eval: Option<DeviceAjtaiYEval>,
    /// Coefficient-log shape from the last compact phase summary. Proof
    /// assembly can export the resident log later without rerunning Pi_CCS.
    pub(super) last_phase_log_shape: Option<FePhaseLogShape>,
    /// Opt-in whole FE row+Ajtai-tail tracing.
    pub(super) whole_fe_trace_enabled: bool,
    /// Capture/replay the whole FE trace as a CUDA graph.
    pub(super) whole_fe_graph_enabled: bool,
    /// Diagnostic mode: capture the whole-FE graph every time instead of
    /// replaying a cached graph. This separates graph body/lifetime failures
    /// from cached graph replay failures under profiler instrumentation.
    pub(super) whole_fe_graph_recapture: bool,
}

impl<'a> DeviceFeBackend<'a> {
    pub fn new(device: &'a Device, kernels: &'a SumcheckKernels) -> Self {
        Self {
            device,
            kernels,
            oracle: None,
            oracle_plan: DevicePiCcsOraclePlan::new(),
            running_surfaces: None,
            ring_scratch: RingMatVecScratch::new(),
            phase_workspace: None,
            oracle_workspace: None,
            last_y_eval: None,
            last_phase_log_shape: None,
            whole_fe_trace_enabled: false,
            whole_fe_graph_enabled: false,
            whole_fe_graph_recapture: false,
        }
    }

    pub fn enable_whole_fe_trace(&mut self) {
        self.whole_fe_trace_enabled = true;
        self.whole_fe_graph_enabled = false;
        self.whole_fe_graph_recapture = false;
    }

    pub fn enable_whole_fe_trace_for_parity(&mut self) {
        self.enable_whole_fe_trace();
    }

    pub fn enable_whole_fe_graph_for_parity(&mut self) {
        self.whole_fe_trace_enabled = true;
        self.whole_fe_graph_enabled = true;
        self.whole_fe_graph_recapture = false;
    }

    pub fn enable_whole_fe_trace_recapture_for_parity(&mut self) {
        self.enable_whole_fe_graph_for_parity();
        self.whole_fe_graph_recapture = true;
    }

    pub(crate) fn set_phase_workspace(&mut self, workspace: Option<FePhaseWorkspace>) {
        self.phase_workspace = workspace;
    }

    pub(crate) fn take_phase_workspace(&mut self) -> Option<FePhaseWorkspace> {
        self.phase_workspace.take()
    }

    pub(crate) fn set_oracle_workspace(&mut self, workspace: Option<FeOracleWorkspace>) {
        self.oracle_workspace = workspace;
    }

    pub(crate) fn take_oracle_workspace(&mut self) -> Option<FeOracleWorkspace> {
        let mut workspace = self
            .oracle_workspace
            .take()
            .unwrap_or_else(FeOracleWorkspace::new);
        if let Some(oracle) = self.oracle.take() {
            oracle.return_to_workspace(&mut workspace);
        }
        Some(workspace)
    }

    pub(crate) fn set_ring_scratch(&mut self, scratch: Option<RingMatVecScratch>) {
        if let Some(scratch) = scratch {
            self.ring_scratch = scratch;
        }
    }

    pub(crate) fn take_ring_scratch(&mut self) -> Option<RingMatVecScratch> {
        Some(std::mem::take(&mut self.ring_scratch))
    }

    /// Move out the resident terminal `Y_eval` surface from the last prove.
    ///
    /// This is a diagnostic/integration handoff: callers must consume it
    /// before starting another prove on this backend.
    pub fn take_last_y_eval_surface(&mut self) -> Option<DeviceAjtaiYEval> {
        self.last_y_eval.take()
    }

    /// Share this fold's witness planes (from
    /// `pi_rlc::upload_witness_planes`) so the Ajtai `Y_eval` skips its own
    /// prep + upload. `count` is the witness count the buffer holds.
    pub fn set_witness_planes(&mut self, planes: &'a DeviceBuffer<u64>, count: usize) {
        self.oracle_plan.set_witness_planes(planes, count);
    }

    pub(crate) fn set_running_surfaces(&mut self, surfaces: Option<&'a DevicePiCcsKSurfaces>) {
        self.running_surfaces = surfaces;
    }

    /// Move the cross-fold static state out so a session can persist it
    /// while backends themselves are constructed per fold.
    pub fn take_statics(&mut self) -> (Option<DeviceBarMatrices>, Option<DeviceRowMatrices>) {
        self.oracle_plan.take_statics()
    }

    /// Seed this backend with static uploads persisted from an earlier
    /// fold. Stale uploads are re-checked against the cache before use.
    pub fn set_statics(&mut self, bar: Option<DeviceBarMatrices>, rows: Option<DeviceRowMatrices>) {
        self.oracle_plan.set_statics(bar, rows);
    }
}

impl DeviceFeBackend<'_> {
    pub(super) fn enqueue_full_fe_phase_body(
        &mut self,
        request: &FePhaseTraceRequest<'_>,
        width: usize,
        coeff_words_per_round: usize,
        tail_partial_count: usize,
        phase_workspace: &mut FePhaseWorkspace,
    ) -> Result<DeviceAjtaiYEval, super::CcsDeviceError> {
        let stream = self.device.stream();
        for round in 0..request.row_rounds {
            let oracle = self
                .oracle
                .as_mut()
                .ok_or(super::CcsDeviceError::Shape("FE backend used before start"))?;
            let coeff_offset = round * coeff_words_per_round;
            let emitted_width = oracle.write_round_coeffs(self.device, self.kernels)?;
            if emitted_width != width {
                return Err(super::CcsDeviceError::Shape("FE round width changed"));
            }
            launch_plane_copy(
                &self.kernels.ring,
                stream,
                &oracle.coeffs_out,
                coeff_offset,
                phase_workspace.coeff_log_mut(),
            )?;
            phase_workspace.enqueue_coeff_challenge(
                self.device,
                &self.kernels.poseidon,
                &self.kernels.poseidon_rc,
                &oracle.coeffs_out,
                coeff_words_per_round,
                2 * round,
            )?;
            oracle.fold_from_challenge(self.device, self.kernels, phase_workspace.challenges(), 2 * round)?;
        }

        let y_eval = self
            .device_ajtai_y_eval_surface_from_device_challenges_in_workspace(
                request.cache,
                request.row_rounds,
                request.n_eff,
                &request.witnesses,
                phase_workspace,
            )?
            .ok_or(super::CcsDeviceError::Shape("device Ajtai Y_eval not applicable"))?;

        let tail_module = &self.kernels.tail;
        for tail_round in 0..request.tail_rounds {
            let round = request.row_rounds + tail_round;
            let oracle = self
                .oracle
                .as_mut()
                .ok_or(super::CcsDeviceError::Shape("FE backend used before start"))?;
            let (points, tail_headers, challenges, tail_partials, tail_partial_scratch, tail_inner_sums) =
                phase_workspace.tail_round_buffers();
            oracle.write_tail_round_coeffs_from_challenges(
                self.device,
                self.kernels,
                tail_module,
                &y_eval,
                tail_headers,
                tail_round * TAIL_CHALLENGE_HEADER_WORDS,
                points,
                challenges,
                tail_partial_count,
                tail_partials,
                tail_partial_scratch,
                tail_inner_sums,
            )?;
            launch_plane_copy(
                &self.kernels.ring,
                stream,
                &oracle.coeffs_out,
                round * coeff_words_per_round,
                phase_workspace.coeff_log_mut(),
            )?;
            phase_workspace.enqueue_coeff_challenge(
                self.device,
                &self.kernels.poseidon,
                &self.kernels.poseidon_rc,
                &oracle.coeffs_out,
                coeff_words_per_round,
                2 * round,
            )?;
        }
        Ok(y_eval)
    }

    pub fn row_round_trace_from_transcript_cooperative(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> FeRowRoundTrace {
        let mut phase_workspace = self
            .phase_workspace
            .take()
            .unwrap_or_else(FePhaseWorkspace::new);
        let trace = self
            .oracle
            .as_mut()
            .expect("FE backend used before start")
            .row_round_trace_from_transcript_cooperative_with_workspace(
                self.device,
                self.kernels,
                &mut phase_workspace,
                transcript_state,
                transcript_absorbed,
                rounds,
            )
            .expect("cooperative device FE row trace failed mid-prove");
        self.phase_workspace = Some(phase_workspace);
        trace
    }
}

impl FeSumcheckBackend for DeviceFeBackend<'_> {
    fn defers_row_equality_tables(&self) -> bool {
        true
    }

    fn claimed_initial_sum(
        &mut self,
        challenges: &Challenges,
        k_mcs: usize,
        me_input_count: usize,
        matrix_count: usize,
    ) -> Option<K> {
        let surfaces = self.running_surfaces?;
        assert_eq!(
            surfaces.claims(),
            me_input_count,
            "resident running-claim count mismatch"
        );
        assert_eq!(
            surfaces.t_core(),
            matrix_count,
            "resident running matrix count mismatch"
        );
        Some(
            surfaces
                .claimed_initial_sum(self.device, self.kernels, &challenges.alpha, challenges.gamma, k_mcs)
                .expect("device running FE initial sum failed mid-prove"),
        )
    }

    fn start(&mut self, snapshot: &RowPhaseSnapshot<'_>) -> bool {
        self.last_y_eval = None;
        self.last_phase_log_shape = None;
        let mut workspace = self
            .oracle_workspace
            .take()
            .unwrap_or_else(FeOracleWorkspace::new);
        if let Some(oracle) = self.oracle.take() {
            oracle.return_to_workspace(&mut workspace);
        }
        let oracle_result = DeviceFeOracle::from_snapshot_with_deferred_and_workspace(
            self.device,
            self.kernels,
            snapshot,
            &self.oracle_plan.deferred_row_tables,
            self.oracle_plan.deferred_eval_table.as_ref(),
            &mut workspace,
        );
        match oracle_result {
            Ok(oracle) => {
                self.oracle_plan.reset_deferred();
                self.oracle = Some(oracle);
                self.oracle_workspace = Some(workspace);
                true
            }
            Err(_) => {
                self.oracle_plan.reset_deferred();
                self.oracle_workspace = Some(workspace);
                false
            }
        }
    }

    fn round_coeffs(&mut self) -> Vec<K> {
        self.oracle
            .as_mut()
            .expect("FE backend used before start")
            .round_coeffs(self.device, self.kernels)
            .expect("device FE round eval failed mid-prove")
    }

    fn fold(&mut self, r: K) {
        self.oracle
            .as_mut()
            .expect("FE backend used before start")
            .fold(self.device, self.kernels, r)
            .expect("device FE fold failed mid-prove")
    }

    fn row_round_trace_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Option<FeRowRoundTrace> {
        let mut phase_workspace = self
            .phase_workspace
            .take()
            .unwrap_or_else(FePhaseWorkspace::new);
        let trace = self
            .oracle
            .as_mut()
            .expect("FE backend used before start")
            .row_round_trace_from_transcript_with_workspace(
                self.device,
                self.kernels,
                &mut phase_workspace,
                transcript_state,
                transcript_absorbed,
                rounds,
            )
            .expect("device FE row trace failed mid-prove");
        self.phase_workspace = Some(phase_workspace);
        Some(trace)
    }

    fn row_round_summary_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
        initial_sum: K,
    ) -> Option<FeRowRoundSummary> {
        Some(
            self.row_round_summary_from_transcript_retained(transcript_state, transcript_absorbed, rounds, initial_sum)
                .expect("device FE row summary failed mid-prove"),
        )
    }

    fn export_row_rounds(&mut self) -> Option<Vec<Vec<K>>> {
        Some(
            self.export_retained_row_rounds()
                .expect("device FE row proof-log export failed mid-prove"),
        )
    }

    fn fe_phase_trace_from_transcript(&mut self, request: FePhaseTraceRequest<'_>) -> Option<FeRowRoundTrace> {
        if !self.whole_fe_trace_enabled {
            return None;
        }
        Some(
            self.full_fe_trace_from_transcript(request)
                .expect("device FE phase trace failed mid-prove"),
        )
    }

    fn mcs_row_tables(
        &mut self,
        cache: &SuperneoEvalCache,
        mcs_idx: usize,
        f_var_indices: &[usize],
        z_blocks: &SuperneoZBlocks,
        n_eff: usize,
        _crop: bool,
        n_pad: usize,
    ) -> Option<FeMcsRowTables> {
        let result;
        perf_timed!("fold.superneo.pi_ccs.oracle.F", {
            result = self
                .oracle_plan
                .mcs_row_tables(
                    self.device,
                    self.kernels,
                    cache,
                    mcs_idx,
                    f_var_indices,
                    z_blocks,
                    n_eff,
                    n_pad,
                )
                .expect("device f-var row tables failed mid-prove");
        });
        result
    }

    fn serves_carried_eval_table(&self) -> bool {
        self.oracle_plan.serves_carried_eval_table()
    }

    fn carried_eval_table(
        &mut self,
        cache: &SuperneoEvalCache,
        carried_coeffs: &[K],
        k_mcs: usize,
        weights: &[K; D],
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Option<FeEvalTable> {
        let result;
        perf_timed!("fold.superneo.pi_ccs.oracle.Eval", {
            result = self
                .oracle_plan
                .carried_eval_table(
                    self.device,
                    self.kernels,
                    cache,
                    carried_coeffs,
                    k_mcs,
                    weights,
                    mat_coeffs,
                    n_eff,
                    n_pad,
                )
                .expect("device carried eval table failed mid-prove");
        });
        result
    }

    fn eval_weighted_row_table(
        &mut self,
        cache: &SuperneoEvalCache,
        z_blocks: &SuperneoZBlocks,
        weights: &[K; D],
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Option<Vec<K>> {
        let result;
        perf_timed!("fold.superneo.pi_ccs.oracle.Eval", {
            result = self
                .oracle_plan
                .eval_weighted_row_table(
                    self.device,
                    self.kernels,
                    cache,
                    z_blocks,
                    weights,
                    mat_coeffs,
                    n_eff,
                    n_pad,
                )
                .expect("device eval table failed mid-prove");
        });
        result
    }

    fn ajtai_y_eval(
        &mut self,
        cache: &SuperneoEvalCache,
        chi_r: &[K],
        n_eff: usize,
        witnesses: &[&Mat<F>],
    ) -> Option<Vec<Vec<[K; D]>>> {
        let result;
        perf_timed!("fold.superneo.pi_ccs.output.y_prime", {
            result = self
                .device_ajtai_y_eval(cache, chi_r, n_eff, witnesses)
                .expect("device Ajtai Y_eval failed mid-prove");
        });
        result
    }
}
