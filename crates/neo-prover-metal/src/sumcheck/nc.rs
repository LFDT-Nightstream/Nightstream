//! NC column-sumcheck backend, resident execution, and canonical host fallback.
//!
//! Signed masks remain available after NC so Pi_RLC can reuse them. The backend
//! recycles its resident plan only after that handoff is complete.

use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_reductions::optimized_engine::legacy_split_nc::oracle::{NcColSnapshot, NcDigitTableView};
use neo_reductions::optimized_engine::legacy_split_nc::{NcColRoundTrace, NcFinalizedColState, NcSumcheckBackend};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::encoding::{
    decode_trace, dense_digit_rows, flatten_table, flatten_tables, fold_host, k_to_words, words_to_k,
};
use super::mask_residency;
use crate::{
    KWords, MetalError, MetalNcDigitInput, MetalNcFinalState, MetalNcSumcheckInputs, MetalNcSumcheckPlan,
    MetalNcSumcheckTrace, MetalResidentWitness, MetalSession, MetalWitnessMasks,
};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct NcSumcheckProfile {
    pub(crate) nc_rounds: usize,
    pub(crate) nc_input_witnesses: usize,
    pub(crate) nc_active_witnesses: usize,
    pub(crate) nc_mask_native_on_metal: bool,
    pub(crate) folded_tables: usize,
    pub(crate) failure: Option<String>,
}

impl NcSumcheckProfile {
    fn record_failure(&mut self, phase: &'static str, error: impl std::fmt::Display) {
        if self.failure.is_none() {
            self.failure = Some(format!("{phase}: {error}"));
        }
    }
}

/// Adapter between the canonical NC backend trait and one Metal session.
pub(crate) struct MetalNcBackend<'a> {
    session: &'a MetalSession,
    source: NcSource,
    shared_masks: Option<MetalWitnessMasks>,
    oracle: Option<NcOracle>,
    profile: NcSumcheckProfile,
}

#[derive(Clone)]
pub(super) struct NcSignedMasks {
    pub(super) words: Vec<u64>,
    pub(super) blocks: usize,
    pub(super) active_rows: usize,
    pub(super) rows: usize,
    pub(super) witness_count: usize,
}

pub(super) enum NcSource {
    Values(Vec<Vec<K>>),
    SignedMasks(NcSignedMasks),
}

/// Logical NC state with either compact signed masks or materialized rows.
struct NcOracle {
    cur_len: usize,
    beta_m: Vec<K>,
    eq_beta: Vec<K>,
    digit_values: Vec<Vec<K>>,
    initial_masks: Option<NcSignedMasks>,
    witness_count: usize,
    width: usize,
    dense: bool,
    weights: Vec<[K; D]>,
    initial_len: usize,
    initial_width: usize,
    initial_dense: bool,
    resident: Option<MetalNcSumcheckPlan>,
    // Stored until the next round so folding and coefficient evaluation share
    // a command submission.
    pending_challenge: Option<K>,
    challenges: Vec<K>,
    host_fallback: bool,
}

impl<'a> MetalNcBackend<'a> {
    pub(crate) fn new(
        session: &'a MetalSession,
        witnesses: &[&Mat<F>],
        assignment_len: usize,
        fresh_count: usize,
        fresh_device_masks: Option<&MetalWitnessMasks>,
        resident_id: Option<u64>,
    ) -> Result<Self, MetalError> {
        if fresh_count > witnesses.len() {
            return Err(MetalError::Shape("fresh witness count exceeds the NC input count"));
        }
        let source = mask_residency::select_source(
            witnesses,
            assignment_len,
            fresh_count,
            fresh_device_masks,
            resident_id.is_some(),
        );
        let shared_masks = match &source {
            NcSource::SignedMasks(masks) => Some(mask_residency::prepare_shared_masks(
                session,
                masks,
                fresh_count,
                fresh_device_masks,
                resident_id,
            )?),
            NcSource::Values(_) => None,
        };
        Ok(Self {
            session,
            source,
            shared_masks,
            oracle: None,
            profile: NcSumcheckProfile::default(),
        })
    }

    pub(crate) fn shared_masks(&self) -> Option<&MetalWitnessMasks> {
        self.shared_masks.as_ref()
    }

    pub(crate) fn profile(&self) -> NcSumcheckProfile {
        self.profile.clone()
    }

    pub(crate) fn enqueue_rlc_witness_mix_from_resident_masks(
        &self,
        rhos: &[i8],
        fresh_count: usize,
        input_count: usize,
        cols: usize,
        resident_id: Option<u64>,
    ) -> Result<Option<MetalResidentWitness>, MetalError> {
        if self.profile.failure.is_some() {
            return Ok(None);
        }
        let Some(plan) = self
            .oracle
            .as_ref()
            .and_then(|oracle| oracle.resident.as_ref())
        else {
            return Ok(None);
        };
        match resident_id {
            Some(resident_id) => self
                .session
                .enqueue_rlc_witness_mix_from_signed_masks_with_resident_id(
                    rhos,
                    plan,
                    fresh_count,
                    input_count,
                    cols,
                    resident_id,
                ),
            None => self
                .session
                .enqueue_rlc_witness_mix_from_signed_masks(rhos, plan, input_count, cols),
        }
    }

    pub(crate) fn recycle(&mut self) {
        if self.profile.failure.is_some() {
            return;
        }
        if let Some(plan) = self
            .oracle
            .as_mut()
            .and_then(|oracle| oracle.resident.take())
        {
            self.session.recycle_nc_sumcheck(plan);
        }
    }
}

impl NcSumcheckBackend for MetalNcBackend<'_> {
    fn start(&mut self, snapshot: &NcColSnapshot<'_>) -> bool {
        self.oracle = NcOracle::from_snapshot(snapshot, &self.source, self.shared_masks.as_ref(), self.session);
        if let Some(oracle) = self.oracle.as_ref() {
            self.profile.nc_input_witnesses = oracle.witness_count;
            self.profile.nc_active_witnesses = oracle
                .resident
                .as_ref()
                .map_or(0, MetalNcSumcheckPlan::active_witness_count);
        }
        self.oracle.is_some()
    }

    fn round_coeffs(&mut self) -> Vec<K> {
        self.profile.nc_rounds += 1;
        let oracle = self
            .oracle
            .as_mut()
            .expect("Metal NC backend used before start");
        match oracle.round_coeffs_metal(self.session) {
            Ok(coefficients) => {
                self.profile.nc_mask_native_on_metal |= oracle.initial_masks.is_some();
                coefficients
            }
            Err(error) => {
                self.profile
                    .record_failure("compute NC round coefficients", error);
                oracle.activate_host_fallback();
                oracle.round_coeffs()
            }
        }
    }

    fn fold(&mut self, challenge: K) {
        let oracle = self
            .oracle
            .as_mut()
            .expect("Metal NC backend used before start");
        self.profile.folded_tables += oracle.witness_count + 1;
        if let Err(error) = oracle.fold(challenge) {
            self.profile.record_failure("fold NC tables", error);
            oracle.activate_host_fallback();
            oracle.fold_host(challenge);
        }
    }

    fn finalized_col_state(&mut self) -> NcFinalizedColState {
        let oracle = self
            .oracle
            .as_mut()
            .expect("Metal NC backend used before start");
        match oracle.finalized_metal(self.session) {
            Ok(state) => state,
            Err(error) => {
                self.profile.record_failure("finalize NC columns", error);
                oracle.activate_host_fallback();
                oracle.finalized()
            }
        }
    }

    fn col_round_trace_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Option<NcColRoundTrace> {
        let oracle = self
            .oracle
            .as_mut()
            .expect("Metal NC backend used before start");
        match oracle.trace_metal(self.session, transcript_state, transcript_absorbed, rounds) {
            Ok(trace) => {
                self.profile.nc_rounds += rounds;
                self.profile.nc_mask_native_on_metal |= oracle.initial_masks.is_some();
                self.profile.folded_tables += (oracle.witness_count + 1) * rounds;
                Some(trace)
            }
            Err(error) => {
                self.profile
                    .record_failure("compute NC transcript trace", error);
                oracle.activate_host_fallback();
                None
            }
        }
    }
}

impl NcSource {
    fn witness_count(&self) -> usize {
        match self {
            Self::Values(values) => values.len(),
            Self::SignedMasks(masks) => masks.witness_count,
        }
    }

    fn rows(&self) -> Option<usize> {
        match self {
            Self::Values(values) => values.first().map(Vec::len),
            Self::SignedMasks(masks) => Some(masks.rows),
        }
    }

    fn values(&self) -> Option<&[Vec<K>]> {
        match self {
            Self::Values(values) => Some(values),
            Self::SignedMasks(_) => None,
        }
    }
}

impl NcOracle {
    /// Selects exactly one initial representation: fully deferred signed masks
    /// or fully materialized dense rows. Mixed ownership is rejected.
    fn from_snapshot(
        snapshot: &NcColSnapshot<'_>,
        source: &NcSource,
        shared_masks: Option<&MetalWitnessMasks>,
        session: &MetalSession,
    ) -> Option<Self> {
        if snapshot.cur_len < 2
            || !snapshot.cur_len.is_power_of_two()
            || snapshot.weights.len() != snapshot.digit_tables.len()
        {
            return None;
        }
        let all_deferred = snapshot
            .digit_tables
            .iter()
            .all(|table| matches!(table, NcDigitTableView::Deferred { len } if *len <= snapshot.cur_len));
        if source.witness_count() != snapshot.digit_tables.len()
            || source.rows() != Some(snapshot.cur_len)
            || source
                .values()
                .is_some_and(|values| values.iter().any(|values| values.len() != snapshot.cur_len))
            || (snapshot.eq_beta_m_tbl.len() != snapshot.cur_len
                && !(all_deferred && snapshot.eq_beta_m_tbl.is_empty()))
        {
            return None;
        }
        let none_deferred = snapshot
            .digit_tables
            .iter()
            .all(|table| !matches!(table, NcDigitTableView::Deferred { .. }));
        let (digit_values, initial_masks, width, dense) = if all_deferred {
            match source {
                NcSource::Values(values) => (values.clone(), None, 1, false),
                NcSource::SignedMasks(masks) => (Vec::new(), Some(masks.clone()), 1, false),
            }
        } else if none_deferred {
            let values = snapshot
                .digit_tables
                .iter()
                .map(|table| {
                    dense_digit_rows(table, snapshot.cur_len)
                        .map(|rows| rows.into_iter().flat_map(|row| row.into_iter()).collect())
                })
                .collect::<Option<Vec<Vec<K>>>>()?;
            (values, None, D, true)
        } else {
            return None;
        };
        let mut oracle = Self {
            cur_len: snapshot.cur_len,
            beta_m: snapshot.beta_m.to_vec(),
            eq_beta: snapshot.eq_beta_m_tbl.to_vec(),
            digit_values,
            initial_masks,
            witness_count: snapshot.digit_tables.len(),
            width,
            dense,
            weights: snapshot.weights.to_vec(),
            initial_len: snapshot.cur_len,
            initial_width: width,
            initial_dense: dense,
            resident: None,
            pending_challenge: None,
            challenges: Vec::new(),
            host_fallback: false,
        };
        oracle.resident = Some(oracle.prepare_resident(session, shared_masks).ok()?);
        Some(oracle)
    }

    fn round_coeffs(&self) -> Vec<K> {
        let mut coeffs = vec![K::ZERO; 5];
        for pair in 0..(self.cur_len / 2) {
            let index = 2 * pair;
            let e0 = self.eq_beta[index];
            let e1 = self.eq_beta[index + 1] - e0;
            let mut inner = [K::ZERO; 4];
            for witness in 0..self.witness_count {
                for lane in 0..D {
                    let weight = self.weights[witness][lane];
                    if weight == K::ZERO {
                        continue;
                    }
                    let a = self.digit_value(witness, index, lane);
                    let b = self.digit_value(witness, index + 1, lane) - a;
                    let a2 = a * a;
                    let b2 = b * b;
                    inner[0] += weight * (a2 * a - a);
                    inner[1] += weight * ((a2 * b) * K::from(F::from_u64(3)) - b);
                    inner[2] += weight * ((a * b2) * K::from(F::from_u64(3)));
                    inner[3] += weight * (b2 * b);
                }
            }
            coeffs[0] += e0 * inner[0];
            coeffs[1] += e0 * inner[1] + e1 * inner[0];
            coeffs[2] += e0 * inner[2] + e1 * inner[1];
            coeffs[3] += e0 * inner[3] + e1 * inner[2];
            coeffs[4] += e1 * inner[3];
        }
        coeffs
    }

    /// Encodes the selected logical representation into a resident plan while
    /// sharing an existing mask buffer whenever its shape matches.
    fn prepare_resident(
        &self,
        session: &MetalSession,
        shared_masks: Option<&MetalWitnessMasks>,
    ) -> Result<MetalNcSumcheckPlan, crate::MetalError> {
        let eq_point = flatten_table(&self.beta_m);
        let digit_values = self
            .initial_masks
            .is_none()
            .then(|| flatten_tables(&self.digit_values));
        let weights = self
            .weights
            .iter()
            .flat_map(|row| {
                row.iter().flat_map(|&value| {
                    let words = k_to_words(value);
                    [words.c0, words.c1]
                })
            })
            .collect::<Vec<_>>();
        let digits = match (&self.initial_masks, &digit_values) {
            (Some(masks), None) => MetalNcDigitInput::SignedMasks {
                words: &masks.words,
                blocks: masks.blocks,
                active_rows: masks.active_rows,
            },
            (None, Some(values)) => MetalNcDigitInput::Table(values),
            _ => return Err(crate::MetalError::Shape("resident NC digit source is inconsistent")),
        };
        session.prepare_nc_sumcheck(MetalNcSumcheckInputs {
            eq_point: &eq_point,
            digits,
            resident_masks: self.initial_masks.as_ref().and(shared_masks),
            weights: &weights,
            witness_count: self.witness_count,
            rows: self.cur_len,
            width: self.width,
            dense: self.dense,
        })
    }

    fn round_coeffs_metal(&mut self, session: &MetalSession) -> Result<Vec<K>, crate::MetalError> {
        let values_per_witness = if self.dense {
            self.cur_len * D
        } else {
            self.cur_len * self.width
        };
        let shape = [
            self.cur_len as u64,
            self.witness_count as u64,
            self.width as u64,
            u64::from(self.dense),
            values_per_witness as u64,
        ];
        let challenge = self.pending_challenge.take().map(k_to_words);
        session
            .nc_sumcheck_round(
                self.resident
                    .as_mut()
                    .ok_or(crate::MetalError::Shape("resident NC plan is unavailable"))?,
                &shape,
                challenge,
            )
            .map(|values| values.into_iter().map(words_to_k).collect())
    }

    fn trace_metal(
        &mut self,
        session: &MetalSession,
        transcript_state: [F; 8],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Result<NcColRoundTrace, crate::MetalError> {
        let trace = session.nc_sumcheck_trace(
            self.resident
                .as_mut()
                .ok_or(crate::MetalError::Shape("resident NC plan is unavailable"))?,
            transcript_state.map(|value| value.as_canonical_u64()),
            transcript_absorbed,
            rounds,
        )?;
        let MetalNcSumcheckTrace {
            rounds: round_trace,
            final_state,
        } = trace;
        let finalized = self.decode_final_state(final_state)?;
        self.cur_len = 1;
        self.pending_challenge = None;
        let (coeffs, challenges, transcript_after) = decode_trace(round_trace);
        self.challenges = challenges.clone();
        Ok(NcColRoundTrace {
            coeffs,
            challenges,
            transcript_after: Some(transcript_after),
            finalized,
        })
    }

    fn fold(&mut self, challenge: K) -> Result<(), crate::MetalError> {
        if self.host_fallback {
            self.fold_host(challenge);
            return Ok(());
        }
        if self.pending_challenge.replace(challenge).is_some() {
            return Err(crate::MetalError::Shape("resident NC fold challenge was not consumed"));
        }
        self.challenges.push(challenge);
        self.cur_len = self.cur_len.div_ceil(2);
        let keep_mask_native = self.initial_masks.is_some() && self.width == 32 && self.cur_len > 1;
        self.dense = self.dense || (2 * self.width > D && !keep_mask_native);
        self.width = if self.dense { D } else { 2 * self.width };
        Ok(())
    }

    fn activate_host_fallback(&mut self) {
        if self.host_fallback {
            return;
        }
        // Restore the initial representation, then replay every challenge that
        // the canonical transcript has already accepted.
        self.materialize_initial_masks();
        if self.eq_beta.is_empty() {
            self.eq_beta = neo_ccs::utils::tensor_point_parallel(&self.beta_m);
        }
        self.cur_len = self.initial_len;
        self.width = self.initial_width;
        self.dense = self.initial_dense;
        for challenge in self.challenges.clone() {
            self.fold_host(challenge);
        }
        self.pending_challenge = None;
        self.host_fallback = true;
    }

    fn fold_host(&mut self, challenge: K) {
        fold_host(&mut self.eq_beta, challenge);
        let next_rows = self.cur_len.div_ceil(2);
        let next_dense = self.dense || 2 * self.width > D;
        let next_width = if next_dense { D } else { 2 * self.width };
        let mut folded = Vec::with_capacity(self.witness_count);
        for witness in 0..self.witness_count {
            let mut values = vec![K::ZERO; next_rows * next_width];
            for row in 0..next_rows {
                for lane in 0..D {
                    let lo = self.digit_value(witness, 2 * row, lane);
                    let hi = if 2 * row + 1 < self.cur_len {
                        self.digit_value(witness, 2 * row + 1, lane)
                    } else {
                        K::ZERO
                    };
                    let value = lo + challenge * (hi - lo);
                    if next_dense {
                        values[row * D + lane] = value;
                    } else {
                        let start = (row * next_width) % D;
                        let slot = (lane + D - start) % D;
                        if slot < next_width {
                            values[row * next_width + slot] = value;
                        }
                    }
                }
            }
            folded.push(values);
        }
        self.digit_values = folded;
        self.cur_len = next_rows;
        self.width = next_width;
        self.dense = next_dense;
    }

    fn finalized(&self) -> NcFinalizedColState {
        debug_assert_eq!(self.cur_len, 1);
        let digit_rows = (0..self.witness_count)
            .map(|witness| std::array::from_fn(|lane| self.digit_value(witness, 0, lane)))
            .collect();
        NcFinalizedColState {
            digit_rows,
            eq_beta_m0: self.eq_beta[0],
        }
    }

    fn finalized_metal(&mut self, session: &MetalSession) -> Result<NcFinalizedColState, crate::MetalError> {
        if self.host_fallback {
            return Ok(self.finalized());
        }
        let state = session.finalize_nc_sumcheck(
            self.resident
                .as_mut()
                .ok_or(crate::MetalError::Shape("resident NC plan is unavailable"))?,
            self.pending_challenge.take().map(k_to_words),
        )?;
        self.decode_final_state(state)
    }

    fn decode_final_state(&self, state: MetalNcFinalState) -> Result<NcFinalizedColState, crate::MetalError> {
        let values_per_witness = if state.dense { D } else { state.width };
        if state.digit_words.len() != self.witness_count * values_per_witness * 2 {
            return Err(crate::MetalError::Shape("resident NC final state has invalid length"));
        }
        let digit_rows = state
            .digit_words
            .chunks_exact(values_per_witness * 2)
            .map(|words| {
                std::array::from_fn(|lane| {
                    if state.dense || lane < state.width {
                        words_to_k(KWords::new(words[2 * lane], words[2 * lane + 1]))
                    } else {
                        K::ZERO
                    }
                })
            })
            .collect();
        Ok(NcFinalizedColState {
            digit_rows,
            eq_beta_m0: words_to_k(state.eq_beta),
        })
    }

    fn digit_value(&self, witness: usize, row: usize, lane: usize) -> K {
        if self.dense {
            return self.digit_values[witness][row * D + lane];
        }
        let start = (row * self.width) % D;
        let slot = (lane + D - start) % D;
        if slot < self.width {
            self.digit_values[witness][row * self.width + slot]
        } else {
            K::ZERO
        }
    }

    fn materialize_initial_masks(&mut self) {
        if !self.digit_values.is_empty() {
            return;
        }
        let masks = self
            .initial_masks
            .as_ref()
            .expect("mask-native NC source is unavailable for host fallback");
        debug_assert_eq!(masks.rows, self.initial_len);
        debug_assert_eq!(masks.witness_count, self.witness_count);
        let mut tables = Vec::with_capacity(self.witness_count);
        for witness in 0..self.witness_count {
            let mut values = vec![K::ZERO; masks.rows];
            for row in 0..masks.active_rows {
                let block = row / D;
                let bit = 1u64 << (row % D);
                let base = 2 * (witness * masks.blocks + block);
                values[row] = if masks.words[base] & bit != 0 {
                    K::ONE
                } else if masks.words[base + 1] & bit != 0 {
                    K::ZERO - K::ONE
                } else {
                    K::ZERO
                };
            }
            tables.push(values);
        }
        self.digit_values = tables;
    }
}
