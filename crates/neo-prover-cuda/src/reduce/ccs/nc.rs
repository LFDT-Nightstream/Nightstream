//! NC (column-phase) device oracle and its prove-driver backend.
//!
//! Table data comes from `NcOracle::col_phase_snapshot`; every column round
//! must stay field-identical to `col_phase_coeffs_b2` + the column fold,
//! and the whole-phase path uses the finalized rows for the device Ajtai tail.

use cuda_core::{DeviceBuffer, PinnedHostBuffer};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::engines::utils::{PI_CCS_SUMCHECK_INITIAL_RAW_TAG, PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG};
use neo_reductions::optimized_engine::oracle::{NcColSnapshot, NcDigitTableView};
use neo_reductions::optimized_engine::{
    NcColRoundTrace, NcColTraceRequest, NcFinalizedColState, NcPhaseRoundTrace, NcSumcheckBackend,
};
use neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG;
use p3_field::PrimeCharacteristicRing;

use crate::device::{copy_host_to_device, uninit_u64_device_buffer, upload_u64_device_buffer, Device};
use crate::field::k_from_device_words;
use crate::graph::GraphAllocations;
use crate::kernels::ajtai::{launch_plane_copy, launch_plane_copy_slice};
use crate::kernels::csr::launch_tensor_point_k;
use crate::kernels::pi_ccs_nc::{
    launch_nc_ajtai_tail_partials, launch_nc_col_partials, launch_nc_fold_dense, launch_nc_fold_dense_from_challenge,
    launch_nc_fold_strided, launch_nc_fold_strided_from_challenge, launch_nc_fold_strided_to_dense,
    launch_nc_fold_strided_to_dense_from_challenge, launch_nc_pack_final_state, launch_nc_widen_planes, NC_CHUNK_PAIRS,
    NC_COEFFS, NC_MODE_DENSE, NC_MODE_STRIDED, RING_LANES,
};
use crate::kernels::sumcheck_common::{
    launch_sum_partials, launch_table_fold, launch_table_fold_from_challenge, SUM_BLOCKS,
};
use crate::reduce::ccs::nc_workspace::{store_buffer, store_pinned, take_buffer, take_pinned, NcOracleWorkspace};
use crate::reduce::ccs::{CcsDeviceError, DevicePublicChallenges, NcPhaseSummary, NcPhaseWorkspace, SumcheckKernels};
use crate::transcript::{encode_transcript_io_ops, DeviceIoSlots, DeviceTranscript, TranscriptIoOp};

/// Fully folded NC column state packed on device.
///
/// Layout is `[witness][lane][c0,c1]` followed by one `eq_beta_m0` K word.
/// Pi_CCS output packing consumes the witness rows directly; parity/export
/// paths can download this same buffer into `NcFinalizedColState`.
pub struct DeviceNcFinalState {
    words: DeviceBuffer<u64>,
    witnesses: usize,
}

impl DeviceNcFinalState {
    pub fn words(&self) -> &DeviceBuffer<u64> {
        &self.words
    }

    pub fn witnesses(&self) -> usize {
        self.witnesses
    }
}

/// How the device digit buffer is currently laid out, mirroring
/// `NcDigitTable`'s representation evolution across folds.
enum NcDigitLayout {
    /// Flat windowed values; `len` live slots per witness.
    Strided {
        width: usize,
        len: usize,
        rows: usize,
    },
    /// `[K; 54]` rows; `rows` per witness.
    Dense { rows: usize },
}

/// Device-resident NC (column-phase) sumcheck oracle.
pub struct DeviceNcOracle {
    /// Ping-pong digit buffers, all witnesses at `wit_stride` K-elements.
    digits_a: DeviceBuffer<u64>,
    digits_b: DeviceBuffer<u64>,
    digits_front_a: bool,
    /// Ping-pong eq_beta_m buffers.
    eq_a: DeviceBuffer<u64>,
    eq_b: DeviceBuffer<u64>,
    eq_front_a: bool,
    weights: DeviceBuffer<u64>,
    layout: NcDigitLayout,
    wit_stride: usize,
    num_wits: usize,
    cur_len: usize,
    /// Round-eval scratch, allocated once (see `DeviceFeOracle::partials`).
    inner_partials: DeviceBuffer<u64>,
    partials: DeviceBuffer<u64>,
    sum_scratch: DeviceBuffer<u64>,
    coeffs_out: DeviceBuffer<u64>,
    /// Pinned readback target (see `DeviceFeOracle::coeffs_host`).
    coeffs_host: PinnedHostBuffer<u64>,
}

#[derive(Clone, Copy)]
pub(crate) struct PendingNcPhase {
    pub(super) col_rounds: usize,
    pub(super) tail_rounds: usize,
    pub(super) tail_coeff_count: usize,
    pub(super) col_coeff_words_per_round: usize,
    pub(super) tail_coeff_words_per_round: usize,
}

pub(crate) struct PreparedNcPhase {
    col_rounds: usize,
    tail_rounds: usize,
    tail_coeff_count: usize,
    col_coeff_words_per_round: usize,
    tail_coeff_words_per_round: usize,
}

fn tensor_point_len(point: &[K]) -> Result<usize, CcsDeviceError> {
    if point.len() >= usize::BITS as usize {
        return Err(CcsDeviceError::Shape("NC equality point is too wide"));
    }
    Ok(1usize << point.len())
}

fn point_words(point: &[K]) -> Vec<u64> {
    let mut words = Vec::with_capacity(point.len() * 2);
    for value in point {
        let (c0, c1) = value.to_limbs_u64();
        words.extend([c0, c1]);
    }
    words
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NcPhaseGraphKey {
    col_rounds: usize,
    tail_rounds: usize,
    tail_coeff_count: usize,
    col_coeff_words_per_round: usize,
    tail_coeff_words_per_round: usize,
    num_wits: usize,
    wit_stride: usize,
    cur_len: usize,
    layout_tag: u8,
    layout_width: usize,
    layout_len: usize,
    layout_rows: usize,
    allocations: GraphAllocations,
}

impl DeviceNcOracle {
    pub fn from_snapshot(
        device: &Device,
        kernels: &SumcheckKernels,
        snapshot: &NcColSnapshot<'_>,
        witness_planes: Option<(&DeviceBuffer<u64>, usize)>,
    ) -> Result<Self, CcsDeviceError> {
        let mut workspace = NcOracleWorkspace::new();
        Self::from_snapshot_with_workspace(device, kernels, snapshot, witness_planes, &mut workspace)
    }

    pub(crate) fn from_snapshot_with_workspace(
        device: &Device,
        kernels: &SumcheckKernels,
        snapshot: &NcColSnapshot<'_>,
        witness_planes: Option<(&DeviceBuffer<u64>, usize)>,
        workspace: &mut NcOracleWorkspace,
    ) -> Result<Self, CcsDeviceError> {
        assert_eq!(D, RING_LANES, "kernel lane count out of sync with neo_math::D");
        let stream = device.stream();
        let num_wits = snapshot.digit_tables.len();
        if num_wits == 0 || snapshot.cur_len < 2 {
            return Err(CcsDeviceError::Shape("NC snapshot needs witnesses and >= 2 columns"));
        }

        // Initial tables must be flat (Lane0 is value-identical to width-1
        // strided: nonzero entries only exist where the lane maps agree).
        let mut len = 0usize;
        for view in &snapshot.digit_tables {
            let table_len = match view {
                NcDigitTableView::Zero { len } => *len,
                NcDigitTableView::Lane0(values) => values.len(),
                NcDigitTableView::Strided { width: 1, values } => values.len(),
                NcDigitTableView::Deferred { len } => *len,
                _ => return Err(CcsDeviceError::Shape("NC snapshot must start unfolded (width 1)")),
            };
            if len == 0 {
                len = table_len;
            } else if len != table_len {
                return Err(CcsDeviceError::Shape("NC digit tables must share length"));
            }
        }
        // Capacity: ragged folds can grow the flat size by < width per fold;
        // one extra dense row of headroom covers every schedule.
        let wit_stride = len + 2 * RING_LANES;

        // Prefer widening resident fold planes on device; the unfolded
        // table is exactly the plane's K lift. Host K-word upload otherwise.
        let planes = witness_planes
            .filter(|(buf, count)| *count == num_wits && buf.len() % count == 0 && buf.len() / count >= len);
        let any_deferred = snapshot
            .digit_tables
            .iter()
            .any(|view| matches!(view, NcDigitTableView::Deferred { .. }));
        if any_deferred && planes.is_none() {
            // The host skipped the build expecting us to source the values
            // from resident planes; without them we must decline so the
            // host materializes and keeps the CPU path.
            return Err(CcsDeviceError::Shape(
                "deferred digit tables need shared witness planes",
            ));
        }
        let digit_words = if planes.is_some() {
            Vec::new()
        } else {
            let mut words = vec![0u64; num_wits * wit_stride * 2];
            for (wit, view) in snapshot.digit_tables.iter().enumerate() {
                let values = match view {
                    NcDigitTableView::Zero { .. } => continue,
                    NcDigitTableView::Lane0(values) | NcDigitTableView::Strided { values, .. } => *values,
                    NcDigitTableView::Dense(_) | NcDigitTableView::Deferred { .. } => {
                        unreachable!("rejected or planes-guarded above")
                    }
                };
                let base = wit * wit_stride * 2;
                for (i, value) in values.iter().enumerate() {
                    let (c0, c1) = value.to_limbs_u64();
                    words[base + 2 * i] = c0;
                    words[base + 2 * i + 1] = c1;
                }
            }
            words
        };

        let mut weight_words = vec![0u64; num_wits * RING_LANES * 2];
        for (wit, lanes) in snapshot.weights.iter().enumerate() {
            for (rho, w) in lanes.iter().enumerate() {
                let (c0, c1) = w.to_limbs_u64();
                weight_words[(wit * RING_LANES + rho) * 2] = c0;
                weight_words[(wit * RING_LANES + rho) * 2 + 1] = c1;
            }
        }

        if tensor_point_len(snapshot.beta_m)? != snapshot.cur_len
            || (snapshot.eq_beta_m_tbl.len() != snapshot.cur_len
                && !(any_deferred && snapshot.eq_beta_m_tbl.is_empty()))
        {
            return Err(CcsDeviceError::Shape("NC beta point/table length mismatch"));
        }

        let digit_words_len = num_wits * wit_stride * 2;
        let mut digits_a = take_buffer(&mut workspace.digits_a, stream, digit_words_len)?;
        let digits_b = take_buffer(&mut workspace.digits_b, stream, digit_words_len)?;
        perf_timed!("fold.superneo.pi_ccs.oracle.NC", {
            match planes {
                Some((buf, count)) => {
                    launch_nc_widen_planes(
                        &kernels.nc,
                        stream,
                        buf,
                        buf.len() / count,
                        len,
                        wit_stride,
                        num_wits,
                        &mut digits_a,
                    )?;
                }
                None => {
                    copy_host_to_device(stream, &digits_a, &digit_words)?;
                }
            }
        });
        let max_groups = num_wits * (snapshot.cur_len / 2).div_ceil(NC_CHUNK_PAIRS).max(1);
        let mut eq_a = take_buffer(&mut workspace.eq_a, stream, snapshot.cur_len * 2)?;
        let eq_point = upload_u64_device_buffer(stream, &point_words(snapshot.beta_m))?;
        launch_tensor_point_k(&kernels.csr, stream, &eq_point, snapshot.beta_m.len(), &mut eq_a)?;
        let eq_b = take_buffer(&mut workspace.eq_b, stream, snapshot.cur_len * 2)?;
        let weights = take_buffer(&mut workspace.weights, stream, weight_words.len())?;
        copy_host_to_device(stream, &weights, &weight_words)?;
        let inner_partials = take_buffer(
            &mut workspace.inner_partials,
            stream,
            num_wits * (snapshot.cur_len / 2) * 8,
        )?;
        let partials = take_buffer(&mut workspace.partials, stream, max_groups * NC_COEFFS * 2)?;
        let sum_scratch = take_buffer(&mut workspace.sum_scratch, stream, SUM_BLOCKS * NC_COEFFS * 2)?;
        let coeffs_out = take_buffer(&mut workspace.coeffs_out, stream, NC_COEFFS * 2)?;
        let coeffs_host = take_pinned(&mut workspace.coeffs_host, device, NC_COEFFS * 2)?;
        Ok(Self {
            digits_a,
            digits_b,
            digits_front_a: true,
            eq_a,
            eq_b,
            eq_front_a: true,
            weights,
            layout: NcDigitLayout::Strided {
                width: 1,
                len,
                rows: len,
            },
            wit_stride,
            num_wits,
            cur_len: snapshot.cur_len,
            inner_partials,
            partials,
            sum_scratch,
            coeffs_out,
            coeffs_host,
        })
    }

    pub(crate) fn return_to_workspace(self, workspace: &mut NcOracleWorkspace) {
        store_buffer(&mut workspace.digits_a, self.digits_a);
        store_buffer(&mut workspace.digits_b, self.digits_b);
        store_buffer(&mut workspace.eq_a, self.eq_a);
        store_buffer(&mut workspace.eq_b, self.eq_b);
        store_buffer(&mut workspace.weights, self.weights);
        store_buffer(&mut workspace.inner_partials, self.inner_partials);
        store_buffer(&mut workspace.partials, self.partials);
        store_buffer(&mut workspace.sum_scratch, self.sum_scratch);
        store_buffer(&mut workspace.coeffs_out, self.coeffs_out);
        store_pinned(&mut workspace.coeffs_host, self.coeffs_host);
    }

    pub(crate) fn record_graph_allocations(&self, allocations: &mut GraphAllocations) {
        allocations.push(&self.digits_a);
        allocations.push(&self.digits_b);
        allocations.push(&self.eq_a);
        allocations.push(&self.eq_b);
        allocations.push(&self.weights);
        allocations.push(&self.inner_partials);
        allocations.push(&self.partials);
        allocations.push(&self.sum_scratch);
        allocations.push(&self.coeffs_out);
    }

    fn layout_key(&self) -> (u8, usize, usize, usize) {
        match self.layout {
            NcDigitLayout::Strided { width, len, rows } => (0, width, len, rows),
            NcDigitLayout::Dense { rows } => (1, 0, 0, rows),
        }
    }

    fn mark_col_rounds_replayed(&mut self, rounds: usize) -> Result<(), CcsDeviceError> {
        for _ in 0..rounds {
            if self.cur_len < 2 {
                return Err(CcsDeviceError::Shape("replayed NC fold below length 2"));
            }
            self.eq_front_a = !self.eq_front_a;
            self.digits_front_a = !self.digits_front_a;
            self.layout = match self.layout {
                NcDigitLayout::Strided { width, len: _, rows } => {
                    let half = rows.div_ceil(2);
                    if 2 * width <= D {
                        let new_width = 2 * width;
                        NcDigitLayout::Strided {
                            width: new_width,
                            len: half * new_width,
                            rows: half,
                        }
                    } else {
                        NcDigitLayout::Dense { rows: half }
                    }
                }
                NcDigitLayout::Dense { rows } => NcDigitLayout::Dense { rows: rows.div_ceil(2) },
            };
            self.cur_len /= 2;
        }
        Ok(())
    }

    fn write_round_coeffs(&mut self, device: &Device, kernels: &SumcheckKernels) -> Result<(), CcsDeviceError> {
        let tail_len = self.cur_len / 2;
        let pair_groups = tail_len.div_ceil(NC_CHUNK_PAIRS).max(1);
        let groups = self.num_wits * pair_groups;
        let (mode, width, live_len) = match &self.layout {
            NcDigitLayout::Strided { width, len, .. } => (NC_MODE_STRIDED, *width, *len),
            NcDigitLayout::Dense { rows } => (NC_MODE_DENSE, 0, *rows),
        };
        let eq_front = if self.eq_front_a { &self.eq_a } else { &self.eq_b };
        let digits_front = if self.digits_front_a {
            &self.digits_a
        } else {
            &self.digits_b
        };
        launch_nc_col_partials(
            &kernels.nc,
            device.stream(),
            eq_front,
            digits_front,
            &self.weights,
            mode,
            width,
            live_len,
            self.wit_stride,
            self.num_wits,
            tail_len,
            pair_groups,
            &mut self.inner_partials,
            &mut self.partials,
        )?;
        launch_sum_partials(
            &kernels.common,
            device.stream(),
            &self.partials,
            groups,
            NC_COEFFS * 2,
            &mut self.sum_scratch,
            &mut self.coeffs_out,
        )?;
        Ok(())
    }

    /// This column round's 5 coefficients, identical to `col_phase_coeffs_b2`.
    pub fn round_coeffs(&mut self, device: &Device, kernels: &SumcheckKernels) -> Result<Vec<K>, CcsDeviceError> {
        self.write_round_coeffs(device, kernels)?;
        self.coeffs_out
            .copy_to_pinned_host(device.stream(), &mut self.coeffs_host)?;
        let words = self.coeffs_host.as_slice();
        Ok((0..NC_COEFFS)
            .map(|d| k_from_device_words(words[2 * d], words[2 * d + 1]))
            .collect())
    }

    /// Fold eq_beta_m and every digit table at the sampled challenge,
    /// evolving the digit layout exactly as `NcDigitTable::fold_inplace`.
    pub fn fold(&mut self, device: &Device, kernels: &SumcheckKernels, r: K) -> Result<(), CcsDeviceError> {
        if self.cur_len < 2 {
            return Err(CcsDeviceError::Shape("NC fold below length 2"));
        }
        let (r_c0, r_c1) = r.to_limbs_u64();
        let stream = device.stream();

        let (eq_src, eq_dst) = if self.eq_front_a {
            (&self.eq_a, &mut self.eq_b)
        } else {
            (&self.eq_b, &mut self.eq_a)
        };
        launch_table_fold(
            &kernels.common,
            stream,
            eq_src,
            1,
            self.cur_len,
            self.cur_len,
            r_c0,
            r_c1,
            eq_dst,
        )?;
        self.eq_front_a = !self.eq_front_a;

        let (src, dst) = if self.digits_front_a {
            (&self.digits_a, &mut self.digits_b)
        } else {
            (&self.digits_b, &mut self.digits_a)
        };
        self.layout = match self.layout {
            NcDigitLayout::Strided { width, len, rows } => {
                let half = rows.div_ceil(2);
                if 2 * width <= D {
                    let new_width = 2 * width;
                    let out_len = half * new_width;
                    launch_nc_fold_strided(
                        &kernels.nc,
                        stream,
                        src,
                        len,
                        width,
                        out_len,
                        self.wit_stride,
                        self.num_wits,
                        r_c0,
                        r_c1,
                        dst,
                    )?;
                    NcDigitLayout::Strided {
                        width: new_width,
                        len: out_len,
                        rows: half,
                    }
                } else {
                    launch_nc_fold_strided_to_dense(
                        &kernels.nc,
                        stream,
                        src,
                        len,
                        width,
                        rows,
                        half,
                        self.wit_stride,
                        self.wit_stride,
                        self.num_wits,
                        r_c0,
                        r_c1,
                        dst,
                    )?;
                    NcDigitLayout::Dense { rows: half }
                }
            }
            NcDigitLayout::Dense { rows } => {
                let half = rows.div_ceil(2);
                launch_nc_fold_dense(
                    &kernels.nc,
                    stream,
                    src,
                    rows,
                    half,
                    self.wit_stride,
                    self.num_wits,
                    r_c0,
                    r_c1,
                    dst,
                )?;
                NcDigitLayout::Dense { rows: half }
            }
        };
        self.digits_front_a = !self.digits_front_a;
        self.cur_len /= 2;
        Ok(())
    }

    fn fold_from_challenge(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        challenges: &DeviceBuffer<u64>,
        challenge_offset: usize,
    ) -> Result<(), CcsDeviceError> {
        if self.cur_len < 2 {
            return Err(CcsDeviceError::Shape("NC fold below length 2"));
        }
        let stream = device.stream();

        let (eq_src, eq_dst) = if self.eq_front_a {
            (&self.eq_a, &mut self.eq_b)
        } else {
            (&self.eq_b, &mut self.eq_a)
        };
        launch_table_fold_from_challenge(
            &kernels.common,
            stream,
            eq_src,
            1,
            self.cur_len,
            self.cur_len,
            challenges,
            challenge_offset,
            eq_dst,
        )?;
        self.eq_front_a = !self.eq_front_a;

        let (src, dst) = if self.digits_front_a {
            (&self.digits_a, &mut self.digits_b)
        } else {
            (&self.digits_b, &mut self.digits_a)
        };
        self.layout = match self.layout {
            NcDigitLayout::Strided { width, len, rows } => {
                let half = rows.div_ceil(2);
                if 2 * width <= D {
                    let new_width = 2 * width;
                    let out_len = half * new_width;
                    launch_nc_fold_strided_from_challenge(
                        &kernels.nc,
                        stream,
                        src,
                        len,
                        width,
                        out_len,
                        self.wit_stride,
                        self.num_wits,
                        challenges,
                        challenge_offset,
                        dst,
                    )?;
                    NcDigitLayout::Strided {
                        width: new_width,
                        len: out_len,
                        rows: half,
                    }
                } else {
                    launch_nc_fold_strided_to_dense_from_challenge(
                        &kernels.nc,
                        stream,
                        src,
                        len,
                        width,
                        rows,
                        half,
                        self.wit_stride,
                        self.wit_stride,
                        self.num_wits,
                        challenges,
                        challenge_offset,
                        dst,
                    )?;
                    NcDigitLayout::Dense { rows: half }
                }
            }
            NcDigitLayout::Dense { rows } => {
                let half = rows.div_ceil(2);
                launch_nc_fold_dense_from_challenge(
                    &kernels.nc,
                    stream,
                    src,
                    rows,
                    half,
                    self.wit_stride,
                    self.num_wits,
                    challenges,
                    challenge_offset,
                    dst,
                )?;
                NcDigitLayout::Dense { rows: half }
            }
        };
        self.digits_front_a = !self.digits_front_a;
        self.cur_len /= 2;
        Ok(())
    }

    pub fn col_round_trace_from_transcript(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        transcript_state: [F; crate::kernels::poseidon2::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Result<NcColRoundTrace, CcsDeviceError> {
        let stream = device.stream();
        let coeff_words_per_round = NC_COEFFS * 2;
        let mut coeff_log = uninit_u64_device_buffer(stream, rounds * coeff_words_per_round)?;
        let mut challenges_dev = uninit_u64_device_buffer(stream, rounds * 2)?;
        let mut transcript = DeviceTranscript::from_state_and_absorbed(device, transcript_state, transcript_absorbed)?;

        for round in 0..rounds {
            self.write_round_coeffs(device, kernels)?;
            launch_plane_copy(
                &kernels.ring,
                stream,
                &self.coeffs_out,
                round * coeff_words_per_round,
                &mut coeff_log,
            )?;
            transcript.enqueue_absorb_device_challenge(
                device,
                &kernels.poseidon,
                &kernels.poseidon_rc,
                coeff_words_per_round as u64,
                DeviceIoSlots {
                    payload: &self.coeffs_out,
                    payload_offset: 0,
                    payload_len: coeff_words_per_round,
                    out: &mut challenges_dev,
                    out_offset: 2 * round,
                },
            )?;
            self.fold_from_challenge(device, kernels, &challenges_dev, 2 * round)?;
        }

        let transcript_words = transcript.state_words_to_host(device)?;
        let coeff_words = coeff_log.to_host_vec(stream)?;
        let challenge_words = challenges_dev.to_host_vec(stream)?;
        // The host replays these logs into the canonical transcript
        // immediately; if downloads ever become async, this boundary must
        // stay paired with the log decode below.
        device.sync()?;

        let coeffs = (0..rounds)
            .map(|round| {
                let base = round * coeff_words_per_round;
                (0..NC_COEFFS)
                    .map(|d| k_from_device_words(coeff_words[base + 2 * d], coeff_words[base + 2 * d + 1]))
                    .collect()
            })
            .collect();
        let challenges = (0..rounds)
            .map(|round| k_from_device_words(challenge_words[2 * round], challenge_words[2 * round + 1]))
            .collect();
        let finalized = self.finalized_col_state(device, kernels)?;
        Ok(NcColRoundTrace {
            coeffs,
            challenges,
            transcript_after: Some(DeviceTranscript::decode_state_words(&transcript_words)),
            finalized,
        })
    }

    pub fn col_round_trace_with_prolog(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        request: NcColTraceRequest,
    ) -> Result<NcColRoundTrace, CcsDeviceError> {
        let mut transcript =
            DeviceTranscript::from_state_and_absorbed(device, request.transcript_state, request.transcript_absorbed)?;
        self.col_round_trace_with_prolog_from_device_transcript(
            device,
            kernels,
            &mut transcript,
            request.rounds,
            request.initial_sum,
        )
    }

    pub(crate) fn col_round_trace_with_prolog_from_device_transcript(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        transcript: &mut DeviceTranscript,
        rounds: usize,
        initial_sum: K,
    ) -> Result<NcColRoundTrace, CcsDeviceError> {
        let stream = device.stream();
        let coeff_words_per_round = NC_COEFFS * 2;
        let mut coeff_log = uninit_u64_device_buffer(stream, rounds * coeff_words_per_round)?;
        let mut challenges_dev = uninit_u64_device_buffer(stream, rounds * 2)?;

        let initial_coeffs = initial_sum.as_coeffs();
        let raw_append_words = |fields: &[F]| {
            let mut out = Vec::with_capacity(fields.len() + 1);
            out.push(F::from_u64(fields.len() as u64));
            out.extend_from_slice(fields);
            out
        };
        let prolog_ops = [
            TranscriptIoOp::AbsorbHost(raw_append_words(&[F::from_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)])),
            TranscriptIoOp::AbsorbHost(raw_append_words(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)])),
            TranscriptIoOp::AbsorbHost(raw_append_words(&initial_coeffs)),
            TranscriptIoOp::AbsorbHost(raw_append_words(&[F::from_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)])),
        ];
        let encoded = encode_transcript_io_ops(&prolog_ops);
        let ops_dev = upload_u64_device_buffer(stream, &encoded.op_words)?;
        let host_payload_dev = upload_u64_device_buffer(stream, &encoded.host_payload)?;
        let device_payload_dummy = uninit_u64_device_buffer(stream, 1)?;
        let mut host_out_dummy = uninit_u64_device_buffer(stream, 1)?;
        let mut device_out_dummy = uninit_u64_device_buffer(stream, 1)?;
        transcript.enqueue_io(
            device,
            &kernels.poseidon,
            &kernels.poseidon_rc,
            &ops_dev,
            &host_payload_dev,
            &device_payload_dummy,
            &mut host_out_dummy,
            &mut device_out_dummy,
        )?;

        for round in 0..rounds {
            self.write_round_coeffs(device, kernels)?;
            launch_plane_copy(
                &kernels.ring,
                stream,
                &self.coeffs_out,
                round * coeff_words_per_round,
                &mut coeff_log,
            )?;
            transcript.enqueue_absorb_device_challenge(
                device,
                &kernels.poseidon,
                &kernels.poseidon_rc,
                coeff_words_per_round as u64,
                DeviceIoSlots {
                    payload: &self.coeffs_out,
                    payload_offset: 0,
                    payload_len: coeff_words_per_round,
                    out: &mut challenges_dev,
                    out_offset: 2 * round,
                },
            )?;
            self.fold_from_challenge(device, kernels, &challenges_dev, 2 * round)?;
        }

        let transcript_words = transcript.state_words_to_host(device)?;
        let coeff_words = coeff_log.to_host_vec(stream)?;
        let challenge_words = challenges_dev.to_host_vec(stream)?;
        device.sync()?;

        let coeffs = (0..rounds)
            .map(|round| {
                let base = round * coeff_words_per_round;
                (0..NC_COEFFS)
                    .map(|d| k_from_device_words(coeff_words[base + 2 * d], coeff_words[base + 2 * d + 1]))
                    .collect()
            })
            .collect();
        let challenges = (0..rounds)
            .map(|round| k_from_device_words(challenge_words[2 * round], challenge_words[2 * round + 1]))
            .collect();
        let finalized = self.finalized_col_state(device, kernels)?;
        Ok(NcColRoundTrace {
            coeffs,
            challenges,
            transcript_after: Some(DeviceTranscript::decode_state_words(&transcript_words)),
            finalized,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_phase_with_prolog_and_tail_from_device_transcript(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        transcript: &mut DeviceTranscript,
        workspace: &mut NcPhaseWorkspace,
        col_rounds: usize,
        tail_rounds: usize,
        tail_coeff_count: usize,
        initial_sum: K,
        beta_a: &[K],
        gamma: K,
        public_challenges: Option<&DevicePublicChallenges>,
    ) -> Result<PendingNcPhase, CcsDeviceError> {
        let prepared = Self::prepare_phase_with_prolog_and_tail(
            device,
            kernels,
            workspace,
            col_rounds,
            tail_rounds,
            tail_coeff_count,
            initial_sum,
            beta_a,
            gamma,
            public_challenges,
        )?;
        self.enqueue_prepared_phase_with_prolog_and_tail(device, kernels, transcript, workspace, &prepared)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_phase_with_prolog_and_tail(
        device: &Device,
        kernels: &SumcheckKernels,
        workspace: &mut NcPhaseWorkspace,
        col_rounds: usize,
        tail_rounds: usize,
        tail_coeff_count: usize,
        initial_sum: K,
        beta_a: &[K],
        gamma: K,
        public_challenges: Option<&DevicePublicChallenges>,
    ) -> Result<PreparedNcPhase, CcsDeviceError> {
        let stream = device.stream();
        if tail_rounds > 0 && tail_coeff_count < NC_COEFFS {
            return Err(CcsDeviceError::Shape(
                "NC tail coefficient count below b=2 polynomial width",
            ));
        }
        let col_coeff_words_per_round = NC_COEFFS * 2;
        let tail_coeff_words_per_round = tail_coeff_count * 2;
        let total_rounds = col_rounds + tail_rounds;
        workspace.prepare_logs(
            stream,
            col_rounds * col_coeff_words_per_round,
            tail_rounds * tail_coeff_words_per_round,
            tail_coeff_words_per_round.max(1),
            total_rounds * 2,
        )?;

        let initial_coeffs = initial_sum.as_coeffs();
        let raw_append_words = |fields: &[F]| {
            let mut out = Vec::with_capacity(fields.len() + 1);
            out.push(F::from_u64(fields.len() as u64));
            out.extend_from_slice(fields);
            out
        };
        let prolog_ops = [
            TranscriptIoOp::AbsorbHost(raw_append_words(&[F::from_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)])),
            TranscriptIoOp::AbsorbHost(raw_append_words(&[F::from_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)])),
            TranscriptIoOp::AbsorbHost(raw_append_words(&initial_coeffs)),
            TranscriptIoOp::AbsorbHost(raw_append_words(&[F::from_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)])),
        ];
        let encoded = encode_transcript_io_ops(&prolog_ops);
        workspace.upload_prolog(stream, &encoded.op_words, &encoded.host_payload)?;

        let resident_shape = public_challenges
            .filter(|public| public.matches_shape(beta_a.len(), beta_a.len() + col_rounds, col_rounds));
        if let Some(public) = resident_shape {
            workspace.prepare_beta(stream, public.beta_a_words())?;
            launch_plane_copy_slice(
                &kernels.ring,
                stream,
                public.words(),
                public.beta_a_word_offset(),
                0,
                public.beta_a_words(),
                workspace.beta_mut(),
            )?;
            workspace.prepare_gamma(stream)?;
            launch_plane_copy_slice(
                &kernels.ring,
                stream,
                public.words(),
                public.gamma_word_offset(),
                0,
                2,
                workspace.gamma_mut(),
            )?;
        } else {
            let mut beta_words = Vec::with_capacity(beta_a.len() * 2);
            for value in beta_a {
                let (c0, c1) = value.to_limbs_u64();
                beta_words.extend([c0, c1]);
            }
            workspace.upload_beta(stream, &beta_words)?;
            let (gamma_c0, gamma_c1) = gamma.to_limbs_u64();
            workspace.upload_gamma(stream, &[gamma_c0, gamma_c1])?;
        }
        Ok(PreparedNcPhase {
            col_rounds,
            tail_rounds,
            tail_coeff_count,
            col_coeff_words_per_round,
            tail_coeff_words_per_round,
        })
    }

    pub(crate) fn enqueue_prepared_phase_with_prolog_and_tail(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        transcript: &mut DeviceTranscript,
        workspace: &mut NcPhaseWorkspace,
        prepared: &PreparedNcPhase,
    ) -> Result<PendingNcPhase, CcsDeviceError> {
        let stream = device.stream();
        let (ops_dev, host_payload_dev, device_payload_dummy, host_out_dummy, device_out_dummy) =
            workspace.prolog_io_buffers();
        transcript.enqueue_io(
            device,
            &kernels.poseidon,
            &kernels.poseidon_rc,
            ops_dev,
            host_payload_dev,
            device_payload_dummy,
            host_out_dummy,
            device_out_dummy,
        )?;

        for round in 0..prepared.col_rounds {
            self.write_round_coeffs(device, kernels)?;
            launch_plane_copy(
                &kernels.ring,
                stream,
                &self.coeffs_out,
                round * prepared.col_coeff_words_per_round,
                workspace.col_coeff_log_mut(),
            )?;
            transcript.enqueue_absorb_device_challenge(
                device,
                &kernels.poseidon,
                &kernels.poseidon_rc,
                prepared.col_coeff_words_per_round as u64,
                DeviceIoSlots {
                    payload: &self.coeffs_out,
                    payload_offset: 0,
                    payload_len: prepared.col_coeff_words_per_round,
                    out: workspace.challenges_mut(),
                    out_offset: 2 * round,
                },
            )?;
            self.fold_from_challenge(device, kernels, workspace.challenges(), 2 * round)?;
        }

        for tail_round in 0..prepared.tail_rounds {
            let round = prepared.col_rounds + tail_round;
            self.write_tail_round_coeffs_from_challenges(
                device,
                kernels,
                workspace.beta(),
                workspace.gamma(),
                workspace.challenges(),
                prepared.col_rounds,
                tail_round,
                prepared.tail_rounds,
            )?;
            launch_plane_copy(
                &kernels.ring,
                stream,
                &self.coeffs_out,
                0,
                workspace.tail_coeffs_out_mut(),
            )?;
            let (tail_coeffs_out, tail_coeff_log) = workspace.tail_coeffs_and_log_mut();
            launch_plane_copy(
                &kernels.ring,
                stream,
                tail_coeffs_out,
                tail_round * prepared.tail_coeff_words_per_round,
                tail_coeff_log,
            )?;
            let (tail_coeffs_out, challenges_dev) = workspace.tail_coeffs_and_challenges_mut();
            transcript.enqueue_absorb_device_challenge(
                device,
                &kernels.poseidon,
                &kernels.poseidon_rc,
                prepared.tail_coeff_words_per_round as u64,
                DeviceIoSlots {
                    payload: tail_coeffs_out,
                    payload_offset: 0,
                    payload_len: prepared.tail_coeff_words_per_round,
                    out: challenges_dev,
                    out_offset: 2 * round,
                },
            )?;
        }

        Ok(PendingNcPhase {
            col_rounds: prepared.col_rounds,
            tail_rounds: prepared.tail_rounds,
            tail_coeff_count: prepared.tail_coeff_count,
            col_coeff_words_per_round: prepared.col_coeff_words_per_round,
            tail_coeff_words_per_round: prepared.tail_coeff_words_per_round,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn write_tail_round_coeffs_from_challenges(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        beta_a: &DeviceBuffer<u64>,
        gamma: &DeviceBuffer<u64>,
        challenges: &DeviceBuffer<u64>,
        col_rounds: usize,
        tail_round: usize,
        tail_rounds: usize,
    ) -> Result<(), CcsDeviceError> {
        if self.cur_len != 1 {
            return Err(CcsDeviceError::Shape("NC Ajtai tail requires finalized column point"));
        }
        let digits_front = if self.digits_front_a {
            &self.digits_a
        } else {
            &self.digits_b
        };
        let eq_front = if self.eq_front_a { &self.eq_a } else { &self.eq_b };
        let (mode, width) = match &self.layout {
            NcDigitLayout::Strided { width, .. } => (NC_MODE_STRIDED, *width),
            NcDigitLayout::Dense { .. } => (NC_MODE_DENSE, 0),
        };
        launch_nc_ajtai_tail_partials(
            &kernels.nc,
            device.stream(),
            digits_front,
            eq_front,
            beta_a,
            gamma,
            challenges,
            mode,
            width,
            self.wit_stride,
            self.num_wits,
            col_rounds,
            tail_round,
            tail_rounds,
            &mut self.partials,
        )?;
        launch_sum_partials(
            &kernels.common,
            device.stream(),
            &self.partials,
            self.num_wits,
            NC_COEFFS * 2,
            &mut self.sum_scratch,
            &mut self.coeffs_out,
        )?;
        Ok(())
    }

    /// After the last column round: one folded `[K; 54]` digit row per
    /// witness plus the single remaining eq_beta_m entry.
    pub fn finalized_col_state_device(
        &self,
        device: &Device,
        kernels: &SumcheckKernels,
    ) -> Result<DeviceNcFinalState, CcsDeviceError> {
        if self.cur_len != 1 {
            return Err(CcsDeviceError::Shape("NC column point not fully folded"));
        }
        let digits_front = if self.digits_front_a {
            &self.digits_a
        } else {
            &self.digits_b
        };
        let eq_front = if self.eq_front_a { &self.eq_a } else { &self.eq_b };
        let (mode, width) = match &self.layout {
            NcDigitLayout::Strided { width, .. } => (NC_MODE_STRIDED, *width),
            NcDigitLayout::Dense { .. } => (NC_MODE_DENSE, 0),
        };
        // Pack the ~2KB of live state on device; downloading the ping-pong
        // buffers whole would move >100MB to read D lanes per witness.
        let stream = device.stream();
        let mut packed = uninit_u64_device_buffer(stream, (self.num_wits * D + 1) * 2)?;
        launch_nc_pack_final_state(
            &kernels.nc,
            stream,
            digits_front,
            eq_front,
            mode,
            width,
            self.wit_stride,
            self.num_wits,
            &mut packed,
        )?;
        Ok(DeviceNcFinalState {
            words: packed,
            witnesses: self.num_wits,
        })
    }

    pub fn finalized_col_state(
        &self,
        device: &Device,
        kernels: &SumcheckKernels,
    ) -> Result<NcFinalizedColState, CcsDeviceError> {
        let packed = self.finalized_col_state_device(device, kernels)?;
        download_finalized_col_state(device, &packed)
    }
}

pub fn download_finalized_col_state(
    device: &Device,
    packed: &DeviceNcFinalState,
) -> Result<NcFinalizedColState, CcsDeviceError> {
    let words = packed.words.to_host_vec(device.stream())?;
    device.sync()?;

    let read_k = |base: usize| k_from_device_words(words[base], words[base + 1]);
    let digit_rows = (0..packed.witnesses)
        .map(|wit| {
            let mut row = [K::ZERO; D];
            for (rho, slot) in row.iter_mut().enumerate() {
                *slot = read_k((wit * D + rho) * 2);
            }
            row
        })
        .collect();
    Ok(NcFinalizedColState {
        digit_rows,
        eq_beta_m0: read_k(packed.witnesses * D * 2),
    })
}

/// The `NcSumcheckBackend` the CPU prove driver calls; same contract shape
/// as [`super::DeviceFeBackend`].
pub struct DeviceNcBackend<'a> {
    pub(super) device: &'a Device,
    pub(super) kernels: &'a SumcheckKernels,
    oracle: Option<DeviceNcOracle>,
    oracle_workspace: Option<NcOracleWorkspace>,
    pub(super) phase_workspace: Option<NcPhaseWorkspace>,
    /// Caller-supplied fold witness planes; the digit tables initialize
    /// from them on device instead of a host K-word upload.
    witness_planes: Option<(&'a DeviceBuffer<u64>, usize)>,
    /// Last packed final column state returned to the CPU oracle. The
    /// adapter may consume this immediately for the Pi_CCS -> Pi_RLC handoff.
    last_final_state: Option<DeviceNcFinalState>,
    /// Coefficient-log shape from the last compact phase summary. Proof
    /// assembly can export the resident log later without rerunning Pi_CCS.
    pub(super) last_phase_log_shape: Option<PendingNcPhase>,
}

impl<'a> DeviceNcBackend<'a> {
    /// Share this fold's witness planes (see
    /// `DeviceFeBackend::set_witness_planes`; same contract and order).
    pub fn set_witness_planes(&mut self, planes: &'a DeviceBuffer<u64>, count: usize) {
        self.witness_planes = Some((planes, count));
    }

    pub fn new(device: &'a Device, kernels: &'a SumcheckKernels) -> Self {
        Self {
            device,
            kernels,
            oracle: None,
            oracle_workspace: None,
            phase_workspace: None,
            witness_planes: None,
            last_final_state: None,
            last_phase_log_shape: None,
        }
    }

    pub(crate) fn set_oracle_workspace(&mut self, workspace: Option<NcOracleWorkspace>) {
        self.oracle_workspace = workspace;
    }

    pub(crate) fn take_oracle_workspace(&mut self) -> Option<NcOracleWorkspace> {
        let mut workspace = self
            .oracle_workspace
            .take()
            .unwrap_or_else(NcOracleWorkspace::new);
        if let Some(oracle) = self.oracle.take() {
            oracle.return_to_workspace(&mut workspace);
        }
        Some(workspace)
    }

    pub(crate) fn set_phase_workspace(&mut self, workspace: Option<NcPhaseWorkspace>) {
        self.phase_workspace = workspace;
    }

    pub(crate) fn take_phase_workspace(&mut self) -> Option<NcPhaseWorkspace> {
        self.phase_workspace.take()
    }

    /// Move out the resident terminal NC state from the last prove.
    ///
    /// This is a diagnostic/integration handoff: callers must consume it
    /// before starting another prove on this backend.
    pub fn take_last_final_state(&mut self) -> Option<DeviceNcFinalState> {
        self.last_final_state.take()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_phase_with_prolog_and_tail_from_device_transcript(
        &mut self,
        transcript: &mut DeviceTranscript,
        col_rounds: usize,
        tail_rounds: usize,
        tail_coeff_count: usize,
        initial_sum: K,
        beta_a: &[K],
        gamma: K,
        public_challenges: Option<&DevicePublicChallenges>,
    ) -> Result<PendingNcPhase, CcsDeviceError> {
        let mut workspace = self
            .phase_workspace
            .take()
            .unwrap_or_else(NcPhaseWorkspace::new);
        let result = self
            .oracle
            .as_mut()
            .ok_or(CcsDeviceError::Shape("NC backend used before start"))?
            .enqueue_phase_with_prolog_and_tail_from_device_transcript(
                self.device,
                self.kernels,
                transcript,
                &mut workspace,
                col_rounds,
                tail_rounds,
                tail_coeff_count,
                initial_sum,
                beta_a,
                gamma,
                public_challenges,
            );
        self.phase_workspace = Some(workspace);
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_phase_with_prolog_and_tail(
        &mut self,
        col_rounds: usize,
        tail_rounds: usize,
        tail_coeff_count: usize,
        initial_sum: K,
        beta_a: &[K],
        gamma: K,
        public_challenges: Option<&DevicePublicChallenges>,
    ) -> Result<PreparedNcPhase, CcsDeviceError> {
        let mut workspace = self
            .phase_workspace
            .take()
            .unwrap_or_else(NcPhaseWorkspace::new);
        let result = DeviceNcOracle::prepare_phase_with_prolog_and_tail(
            self.device,
            self.kernels,
            &mut workspace,
            col_rounds,
            tail_rounds,
            tail_coeff_count,
            initial_sum,
            beta_a,
            gamma,
            public_challenges,
        );
        self.phase_workspace = Some(workspace);
        result
    }

    pub(crate) fn enqueue_prepared_phase_with_prolog_and_tail_from_device_transcript(
        &mut self,
        transcript: &mut DeviceTranscript,
        prepared: &PreparedNcPhase,
    ) -> Result<PendingNcPhase, CcsDeviceError> {
        let mut workspace = self
            .phase_workspace
            .take()
            .ok_or(CcsDeviceError::Shape("NC phase workspace missing before enqueue"))?;
        let result = self
            .oracle
            .as_mut()
            .ok_or(CcsDeviceError::Shape("NC backend used before start"))?
            .enqueue_prepared_phase_with_prolog_and_tail(
                self.device,
                self.kernels,
                transcript,
                &mut workspace,
                prepared,
            );
        self.phase_workspace = Some(workspace);
        result
    }

    pub(crate) fn graph_key(&self, prepared: &PreparedNcPhase) -> Result<NcPhaseGraphKey, CcsDeviceError> {
        let oracle = self
            .oracle
            .as_ref()
            .ok_or(CcsDeviceError::Shape("NC backend used before start"))?;
        let workspace = self
            .phase_workspace
            .as_ref()
            .ok_or(CcsDeviceError::Shape("NC phase workspace missing before graph key"))?;
        let (layout_tag, layout_width, layout_len, layout_rows) = oracle.layout_key();
        let mut allocations = GraphAllocations::new();
        oracle.record_graph_allocations(&mut allocations);
        workspace.record_graph_allocations(&mut allocations);
        if let Some((planes, _)) = self.witness_planes {
            allocations.push(planes);
        }
        Ok(NcPhaseGraphKey {
            col_rounds: prepared.col_rounds,
            tail_rounds: prepared.tail_rounds,
            tail_coeff_count: prepared.tail_coeff_count,
            col_coeff_words_per_round: prepared.col_coeff_words_per_round,
            tail_coeff_words_per_round: prepared.tail_coeff_words_per_round,
            num_wits: oracle.num_wits,
            wit_stride: oracle.wit_stride,
            cur_len: oracle.cur_len,
            layout_tag,
            layout_width,
            layout_len,
            layout_rows,
            allocations,
        })
    }

    pub(crate) fn mark_prepared_phase_replayed(
        &mut self,
        prepared: &PreparedNcPhase,
    ) -> Result<PendingNcPhase, CcsDeviceError> {
        self.oracle
            .as_mut()
            .ok_or(CcsDeviceError::Shape("NC backend used before start"))?
            .mark_col_rounds_replayed(prepared.col_rounds)?;
        Ok(PendingNcPhase {
            col_rounds: prepared.col_rounds,
            tail_rounds: prepared.tail_rounds,
            tail_coeff_count: prepared.tail_coeff_count,
            col_coeff_words_per_round: prepared.col_coeff_words_per_round,
            tail_coeff_words_per_round: prepared.tail_coeff_words_per_round,
        })
    }

    pub(crate) fn finish_phase_trace(
        &mut self,
        transcript: &DeviceTranscript,
        pending: PendingNcPhase,
    ) -> Result<NcPhaseRoundTrace, CcsDeviceError> {
        let workspace = self
            .phase_workspace
            .take()
            .ok_or(CcsDeviceError::Shape("NC phase workspace missing after enqueue"))?;
        let result = pending.download_trace(
            self.device,
            self.kernels,
            transcript,
            self.oracle
                .as_ref()
                .ok_or(CcsDeviceError::Shape("NC backend used before start"))?,
            &workspace,
        );
        self.phase_workspace = Some(workspace);
        let (trace, final_state) = result?;
        self.last_final_state = Some(final_state);
        Ok(trace)
    }

    pub(crate) fn finish_phase_summary(
        &mut self,
        transcript: &DeviceTranscript,
        pending: PendingNcPhase,
        initial_sum: K,
    ) -> Result<NcPhaseSummary, CcsDeviceError> {
        let workspace = self
            .phase_workspace
            .take()
            .ok_or(CcsDeviceError::Shape("NC phase workspace missing after enqueue"))?;
        let result = pending.download_summary(
            self.device,
            self.kernels,
            transcript,
            self.oracle
                .as_ref()
                .ok_or(CcsDeviceError::Shape("NC backend used before start"))?,
            &workspace,
            initial_sum,
        );
        self.phase_workspace = Some(workspace);
        let (summary, final_state) = result?;
        self.last_final_state = Some(final_state);
        self.last_phase_log_shape = Some(pending);
        Ok(summary)
    }
}

impl NcSumcheckBackend for DeviceNcBackend<'_> {
    fn start(&mut self, snapshot: &NcColSnapshot<'_>) -> bool {
        self.last_final_state = None;
        self.last_phase_log_shape = None;
        let mut workspace = self
            .oracle_workspace
            .take()
            .unwrap_or_else(NcOracleWorkspace::new);
        if let Some(oracle) = self.oracle.take() {
            oracle.return_to_workspace(&mut workspace);
        }
        let build_result = DeviceNcOracle::from_snapshot_with_workspace(
            self.device,
            self.kernels,
            snapshot,
            self.witness_planes,
            &mut workspace,
        );
        match build_result {
            Ok(oracle) => {
                self.oracle = Some(oracle);
                self.oracle_workspace = Some(workspace);
                true
            }
            Err(_error) => {
                self.oracle_workspace = Some(workspace);
                #[cfg(feature = "perf-timers")]
                eprintln!("[neo-prover-cuda] NC backend declined: {_error:?}");
                false
            }
        }
    }

    fn round_coeffs(&mut self) -> Vec<K> {
        self.oracle
            .as_mut()
            .expect("NC backend used before start")
            .round_coeffs(self.device, self.kernels)
            .expect("device NC round eval failed mid-prove")
    }

    fn fold(&mut self, r: K) {
        self.oracle
            .as_mut()
            .expect("NC backend used before start")
            .fold(self.device, self.kernels, r)
            .expect("device NC fold failed mid-prove")
    }

    fn finalized_col_state(&mut self) -> NcFinalizedColState {
        let packed = self
            .oracle
            .as_ref()
            .expect("NC backend used before start")
            .finalized_col_state_device(self.device, self.kernels)
            .expect("device NC finalized state pack failed mid-prove");
        let state = download_finalized_col_state(self.device, &packed)
            .expect("device NC finalized state download failed mid-prove");
        self.last_final_state = Some(packed);
        state
    }

    fn col_round_trace_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Option<NcColRoundTrace> {
        let trace = self
            .oracle
            .as_mut()
            .expect("NC backend used before start")
            .col_round_trace_from_transcript(self.device, self.kernels, transcript_state, transcript_absorbed, rounds)
            .expect("device NC column trace failed mid-prove");
        self.last_final_state = Some(
            self.oracle
                .as_ref()
                .expect("NC backend used before start")
                .finalized_col_state_device(self.device, self.kernels)
                .expect("device NC finalized state pack failed mid-prove"),
        );
        Some(trace)
    }

    fn col_round_trace_with_prolog(&mut self, request: NcColTraceRequest) -> Option<NcColRoundTrace> {
        let trace = self
            .oracle
            .as_mut()
            .expect("NC backend used before start")
            .col_round_trace_with_prolog(self.device, self.kernels, request)
            .expect("device NC prolog+column trace failed mid-prove");
        self.last_final_state = Some(
            self.oracle
                .as_ref()
                .expect("NC backend used before start")
                .finalized_col_state_device(self.device, self.kernels)
                .expect("device NC finalized state pack failed mid-prove"),
        );
        Some(trace)
    }
}
