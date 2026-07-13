//! Metal-owned sumcheck table state used by the protocol engine backends.

use neo_ccs::Mat;
use neo_math::{from_complex, D, F, K};
use neo_reductions::optimized_engine::oracle::{NcColSnapshot, NcDigitTableView, RowPhaseSnapshot, RowTableSnapshot};
use neo_reductions::optimized_engine::{
    FeRowRoundTrace, FeSumcheckBackend, NcColRoundTrace, NcFinalizedColState, NcSumcheckBackend,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{KWords, MetalSession};
use crate::{
    MetalFeSumcheckInputs, MetalFeSumcheckPlan, MetalNcFinalState, MetalNcSumcheckInputs, MetalNcSumcheckPlan,
    MetalNcSumcheckTrace, MetalSumcheckTrace,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SumcheckProfile {
    pub fe_rounds: usize,
    pub nc_rounds: usize,
    pub folded_tables: usize,
    pub metal_failed: bool,
}

pub(crate) struct MetalFeBackend<'a> {
    session: &'a MetalSession,
    oracle: Option<FeOracle>,
    profile: SumcheckProfile,
}

pub(crate) struct MetalNcBackend<'a> {
    session: &'a MetalSession,
    source_values: Vec<Vec<K>>,
    oracle: Option<NcOracle>,
    profile: SumcheckProfile,
}

struct FeOracle {
    cur_len: usize,
    active_len: usize,
    sumcheck_degree_bound: usize,
    row_phase_deg_max: usize,
    tables: Vec<Vec<K>>,
    eq_beta: usize,
    eq_inputs: Option<usize>,
    eval: Option<usize>,
    mcs_tables: Vec<Vec<usize>>,
    gamma_to_k: K,
    gamma_pow_mcs: Vec<K>,
    zero_mcs: Vec<bool>,
    f_at_zero: K,
    f_terms: Vec<(K, Vec<(usize, u32)>)>,
    resident: Option<MetalFeSumcheckPlan>,
    pending_challenge: Option<K>,
    challenges: Vec<K>,
    host_fallback: bool,
}

struct NcOracle {
    cur_len: usize,
    eq_beta: Vec<K>,
    digit_values: Vec<Vec<K>>,
    width: usize,
    dense: bool,
    weights: Vec<[K; D]>,
    initial_len: usize,
    initial_width: usize,
    initial_dense: bool,
    resident: Option<MetalNcSumcheckPlan>,
    pending_challenge: Option<K>,
    challenges: Vec<K>,
    host_fallback: bool,
}

impl<'a> MetalFeBackend<'a> {
    pub(crate) fn new(session: &'a MetalSession) -> Self {
        Self {
            session,
            oracle: None,
            profile: SumcheckProfile::default(),
        }
    }

    pub(crate) fn profile(&self) -> SumcheckProfile {
        self.profile
    }
}

impl<'a> MetalNcBackend<'a> {
    pub(crate) fn new(session: &'a MetalSession, witnesses: &[&Mat<F>], assignment_len: usize) -> Self {
        let padded_len = assignment_len.next_power_of_two().max(2);
        let source_values = witnesses
            .iter()
            .map(|witness| {
                let mut values = vec![K::ZERO; padded_len];
                for column in 0..assignment_len {
                    values[column] = K::from(witness[(column % D, column / D)]);
                }
                values
            })
            .collect();
        Self {
            session,
            source_values,
            oracle: None,
            profile: SumcheckProfile::default(),
        }
    }

    pub(crate) fn profile(&self) -> SumcheckProfile {
        self.profile
    }
}

impl FeSumcheckBackend for MetalFeBackend<'_> {
    fn start(&mut self, snapshot: &RowPhaseSnapshot<'_>) -> bool {
        self.oracle = FeOracle::from_snapshot(snapshot, self.session);
        self.oracle.is_some()
    }

    fn round_coeffs(&mut self) -> Vec<K> {
        self.profile.fe_rounds += 1;
        let oracle = self
            .oracle
            .as_mut()
            .expect("Metal FE backend used before start");
        match oracle.round_coeffs_metal(self.session) {
            Ok(coefficients) => coefficients,
            Err(_) => {
                self.profile.metal_failed = true;
                oracle.activate_host_fallback();
                oracle.round_coeffs()
            }
        }
    }

    fn fold(&mut self, challenge: K) {
        let oracle = self
            .oracle
            .as_mut()
            .expect("Metal FE backend used before start");
        self.profile.folded_tables += oracle.tables.len();
        if oracle.fold(challenge).is_err() {
            self.profile.metal_failed = true;
            oracle.activate_host_fallback();
            oracle.fold_host(challenge);
        }
    }

    fn row_round_trace_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Option<FeRowRoundTrace> {
        let oracle = self
            .oracle
            .as_mut()
            .expect("Metal FE backend used before start");
        match oracle.trace_metal(self.session, transcript_state, transcript_absorbed, rounds) {
            Ok(trace) => {
                self.profile.fe_rounds += rounds;
                self.profile.folded_tables += oracle.tables.len() * rounds;
                Some(trace)
            }
            Err(_) => {
                self.profile.metal_failed = true;
                oracle.activate_host_fallback();
                None
            }
        }
    }
}

impl NcSumcheckBackend for MetalNcBackend<'_> {
    fn start(&mut self, snapshot: &NcColSnapshot<'_>) -> bool {
        self.oracle = NcOracle::from_snapshot(snapshot, &self.source_values, self.session);
        self.oracle.is_some()
    }

    fn round_coeffs(&mut self) -> Vec<K> {
        self.profile.nc_rounds += 1;
        let oracle = self
            .oracle
            .as_mut()
            .expect("Metal NC backend used before start");
        match oracle.round_coeffs_metal(self.session) {
            Ok(coefficients) => coefficients,
            Err(_) => {
                self.profile.metal_failed = true;
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
        self.profile.folded_tables += oracle.digit_values.len() + 1;
        if oracle.fold(challenge).is_err() {
            self.profile.metal_failed = true;
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
            Err(_) => {
                self.profile.metal_failed = true;
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
                self.profile.folded_tables += (oracle.digit_values.len() + 1) * rounds;
                Some(trace)
            }
            Err(_) => {
                self.profile.metal_failed = true;
                oracle.activate_host_fallback();
                None
            }
        }
    }
}

impl FeOracle {
    fn from_snapshot(snapshot: &RowPhaseSnapshot<'_>, session: &MetalSession) -> Option<Self> {
        if snapshot.cur_len < 2
            || !snapshot.cur_len.is_power_of_two()
            || snapshot.row_phase_deg_max > snapshot.sumcheck_degree_bound
            || snapshot.deferred_eval_tbl
            || snapshot.deferred_mcs.iter().any(|&deferred| deferred)
            || snapshot.zero_mcs.len() != snapshot.f_var_tables_by_mcs.len()
        {
            return None;
        }

        let mut tables = Vec::new();
        let eq_beta = push_k_table(&mut tables, snapshot.eq_beta_r_tbl, snapshot.cur_len)?;
        let eq_inputs = match snapshot.eq_r_inputs_tbl {
            Some(table) => Some(push_k_table(&mut tables, table, snapshot.cur_len)?),
            None => None,
        };
        let eval = match snapshot.eval_tbl {
            Some(table) => Some(push_k_table(&mut tables, table, snapshot.cur_len)?),
            None => None,
        };
        if eq_inputs.is_some() != eval.is_some() {
            return None;
        }

        let mut mcs_tables = Vec::with_capacity(snapshot.f_var_tables_by_mcs.len());
        for (mcs_idx, per_mcs) in snapshot.f_var_tables_by_mcs.iter().enumerate() {
            if snapshot.zero_mcs[mcs_idx] {
                mcs_tables.push(Vec::new());
                continue;
            }
            if per_mcs.len() != snapshot.f_var_count {
                return None;
            }
            let mut indices = Vec::with_capacity(per_mcs.len());
            for table in per_mcs {
                indices.push(push_row_table(&mut tables, table, snapshot.cur_len)?);
            }
            mcs_tables.push(indices);
        }

        let mut oracle = Self {
            cur_len: snapshot.cur_len,
            active_len: snapshot.active_len,
            sumcheck_degree_bound: snapshot.sumcheck_degree_bound,
            row_phase_deg_max: snapshot.row_phase_deg_max,
            tables,
            eq_beta,
            eq_inputs,
            eval,
            mcs_tables,
            gamma_to_k: snapshot.gamma_to_k,
            gamma_pow_mcs: snapshot.gamma_pow_mcs.to_vec(),
            zero_mcs: snapshot.zero_mcs.to_vec(),
            f_at_zero: snapshot.f_at_zero,
            f_terms: snapshot.f_terms.clone(),
            resident: None,
            pending_challenge: None,
            challenges: Vec::new(),
            host_fallback: false,
        };
        oracle.resident = Some(oracle.prepare_resident(session).ok()?);
        Some(oracle)
    }

    fn round_coeffs(&self) -> Vec<K> {
        let degree = self.row_phase_deg_max;
        let mut coeffs = vec![K::ZERO; self.sumcheck_degree_bound + 1];
        let mut inner = vec![K::ZERO; degree + 1];
        let mut term_poly = vec![K::ZERO; degree + 1];
        for pair in 0..self.active_len.div_ceil(2) {
            let index = 2 * pair;
            let eq0 = self.tables[self.eq_beta][index];
            let eq1 = self.tables[self.eq_beta][index + 1] - eq0;
            inner.fill(K::ZERO);

            for (mcs_idx, table_indices) in self.mcs_tables.iter().enumerate() {
                let gamma = self.gamma_pow_mcs.get(mcs_idx).copied().unwrap_or(K::ONE);
                if self.zero_mcs[mcs_idx] {
                    inner[0] += self.f_at_zero * gamma;
                    continue;
                }
                for (term_coeff, variables) in &self.f_terms {
                    term_poly.fill(K::ZERO);
                    term_poly[0] = *term_coeff * gamma;
                    let mut current_degree = 0usize;
                    for &(var_pos, exponent) in variables {
                        let table = &self.tables[table_indices[var_pos]];
                        let a = table[index];
                        let b = table[index + 1] - a;
                        for _ in 0..exponent {
                            poly_mul_affine(&mut term_poly, a, b, current_degree);
                            current_degree += 1;
                        }
                    }
                    for i in 0..=current_degree.min(degree) {
                        inner[i] += term_poly[i];
                    }
                }
            }

            coeffs[0] += eq0 * inner[0];
            for i in 1..=degree {
                coeffs[i] += eq0 * inner[i] + eq1 * inner[i - 1];
            }
            if let (Some(eq_inputs), Some(eval)) = (self.eq_inputs, self.eval) {
                let r0 = self.tables[eq_inputs][index];
                let r1 = self.tables[eq_inputs][index + 1] - r0;
                let v0 = self.tables[eval][index];
                let v1 = self.tables[eval][index + 1] - v0;
                coeffs[0] += self.gamma_to_k * r0 * v0;
                coeffs[1] += self.gamma_to_k * (r0 * v1 + r1 * v0);
                coeffs[2] += self.gamma_to_k * r1 * v1;
            }
        }
        coeffs
    }

    fn prepare_resident(&self, session: &MetalSession) -> Result<MetalFeSumcheckPlan, crate::MetalError> {
        let tables = flatten_tables(&self.tables);
        let mut mcs_headers = Vec::with_capacity(self.mcs_tables.len() * 3);
        let mut mcs_table_indices = Vec::new();
        for (mcs_idx, indices) in self.mcs_tables.iter().enumerate() {
            mcs_headers.push(u64::from(self.zero_mcs[mcs_idx]));
            mcs_headers.push(mcs_table_indices.len() as u64);
            mcs_headers.push(indices.len() as u64);
            mcs_table_indices.extend(indices.iter().map(|&index| index as u64));
        }
        let gammas = (0..self.mcs_tables.len())
            .flat_map(|index| {
                let value = self.gamma_pow_mcs.get(index).copied().unwrap_or(K::ONE);
                let words = k_to_words(value);
                [words.c0, words.c1]
            })
            .collect::<Vec<_>>();
        let mut term_headers = Vec::with_capacity(self.f_terms.len() * 4);
        let mut term_variables = Vec::new();
        for (coefficient, variables) in &self.f_terms {
            let words = k_to_words(*coefficient);
            term_headers.extend_from_slice(&[
                words.c0,
                words.c1,
                (term_variables.len() / 2) as u64,
                variables.len() as u64,
            ]);
            for &(position, exponent) in variables {
                term_variables.extend_from_slice(&[position as u64, exponent as u64]);
            }
        }
        let f_at_zero = k_to_words(self.f_at_zero);
        let gamma_to_k = k_to_words(self.gamma_to_k);
        let shape = [
            self.cur_len as u64,
            self.active_len as u64,
            (self.sumcheck_degree_bound + 1) as u64,
            self.row_phase_deg_max as u64,
            self.eq_beta as u64,
            self.eq_inputs.map_or(0, |index| index as u64 + 1),
            self.eval.map_or(0, |index| index as u64 + 1),
            self.mcs_tables.len() as u64,
            self.f_terms.len() as u64,
            f_at_zero.c0,
            f_at_zero.c1,
            gamma_to_k.c0,
            gamma_to_k.c1,
        ];
        session.prepare_fe_sumcheck(MetalFeSumcheckInputs {
            tables: &tables,
            shape: &shape,
            mcs_headers: &mcs_headers,
            mcs_table_indices: &mcs_table_indices,
            gammas: &gammas,
            term_headers: &term_headers,
            term_variables: &term_variables,
            table_count: self.tables.len(),
            coefficient_count: self.sumcheck_degree_bound + 1,
        })
    }

    fn round_coeffs_metal(&mut self, session: &MetalSession) -> Result<Vec<K>, crate::MetalError> {
        let f_at_zero = k_to_words(self.f_at_zero);
        let gamma_to_k = k_to_words(self.gamma_to_k);
        let shape = [
            self.cur_len as u64,
            self.active_len as u64,
            (self.sumcheck_degree_bound + 1) as u64,
            self.row_phase_deg_max as u64,
            self.eq_beta as u64,
            self.eq_inputs.map_or(0, |index| index as u64 + 1),
            self.eval.map_or(0, |index| index as u64 + 1),
            self.mcs_tables.len() as u64,
            self.f_terms.len() as u64,
            f_at_zero.c0,
            f_at_zero.c1,
            gamma_to_k.c0,
            gamma_to_k.c1,
        ];
        let challenge = self.pending_challenge.take().map(k_to_words);
        session
            .fe_sumcheck_round(
                self.resident
                    .as_mut()
                    .ok_or(crate::MetalError::Shape("resident FE plan is unavailable"))?,
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
    ) -> Result<FeRowRoundTrace, crate::MetalError> {
        let f_at_zero = k_to_words(self.f_at_zero);
        let gamma_to_k = k_to_words(self.gamma_to_k);
        let shape = [
            self.cur_len as u64,
            self.active_len as u64,
            (self.sumcheck_degree_bound + 1) as u64,
            self.row_phase_deg_max as u64,
            self.eq_beta as u64,
            self.eq_inputs.map_or(0, |index| index as u64 + 1),
            self.eval.map_or(0, |index| index as u64 + 1),
            self.mcs_tables.len() as u64,
            self.f_terms.len() as u64,
            f_at_zero.c0,
            f_at_zero.c1,
            gamma_to_k.c0,
            gamma_to_k.c1,
        ];
        let trace = session.fe_sumcheck_trace(
            self.resident
                .as_mut()
                .ok_or(crate::MetalError::Shape("resident FE plan is unavailable"))?,
            &shape,
            transcript_state.map(|value| value.as_canonical_u64()),
            transcript_absorbed,
            rounds,
        )?;
        self.cur_len >>= rounds;
        for _ in 0..rounds {
            self.active_len = self.active_len.div_ceil(2).max(1);
        }
        self.pending_challenge = None;
        let (coeffs, challenges, transcript_after) = decode_trace(trace);
        self.challenges = challenges.clone();
        Ok(FeRowRoundTrace {
            coeffs,
            challenges,
            transcript_after: Some(transcript_after),
            ajtai_y_eval: None,
        })
    }

    fn fold(&mut self, challenge: K) -> Result<(), crate::MetalError> {
        if self.host_fallback {
            self.fold_host(challenge);
            return Ok(());
        }
        if self.pending_challenge.replace(challenge).is_some() {
            return Err(crate::MetalError::Shape("resident FE fold challenge was not consumed"));
        }
        self.challenges.push(challenge);
        self.finish_fold();
        Ok(())
    }

    fn activate_host_fallback(&mut self) {
        if self.host_fallback {
            return;
        }
        for &challenge in &self.challenges {
            for table in &mut self.tables {
                fold_host(table, challenge);
            }
        }
        self.pending_challenge = None;
        self.host_fallback = true;
    }

    fn fold_host(&mut self, challenge: K) {
        for table in &mut self.tables {
            fold_host(table, challenge);
        }
        self.finish_fold();
    }

    fn finish_fold(&mut self) {
        self.cur_len /= 2;
        self.active_len = self.active_len.div_ceil(2).max(1);
    }
}

impl NcOracle {
    fn from_snapshot(snapshot: &NcColSnapshot<'_>, source_values: &[Vec<K>], session: &MetalSession) -> Option<Self> {
        if snapshot.cur_len < 2
            || !snapshot.cur_len.is_power_of_two()
            || snapshot.weights.len() != snapshot.digit_tables.len()
        {
            return None;
        }
        if source_values.len() != snapshot.digit_tables.len()
            || source_values
                .iter()
                .any(|values| values.len() != snapshot.cur_len)
            || snapshot.eq_beta_m_tbl.len() != snapshot.cur_len
        {
            return None;
        }
        let all_deferred = snapshot
            .digit_tables
            .iter()
            .all(|table| matches!(table, NcDigitTableView::Deferred { len } if *len <= snapshot.cur_len));
        let none_deferred = snapshot
            .digit_tables
            .iter()
            .all(|table| !matches!(table, NcDigitTableView::Deferred { .. }));
        let (digit_values, width, dense) = if all_deferred {
            (source_values.to_vec(), 1, false)
        } else if none_deferred {
            let values = snapshot
                .digit_tables
                .iter()
                .map(|table| {
                    dense_digit_rows(table, snapshot.cur_len)
                        .map(|rows| rows.into_iter().flat_map(|row| row.into_iter()).collect())
                })
                .collect::<Option<Vec<Vec<K>>>>()?;
            (values, D, true)
        } else {
            return None;
        };
        let mut oracle = Self {
            cur_len: snapshot.cur_len,
            eq_beta: snapshot.eq_beta_m_tbl.to_vec(),
            digit_values,
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
        oracle.resident = Some(oracle.prepare_resident(session).ok()?);
        Some(oracle)
    }

    fn round_coeffs(&self) -> Vec<K> {
        let mut coeffs = vec![K::ZERO; 5];
        for pair in 0..(self.cur_len / 2) {
            let index = 2 * pair;
            let e0 = self.eq_beta[index];
            let e1 = self.eq_beta[index + 1] - e0;
            let mut inner = [K::ZERO; 4];
            for witness in 0..self.digit_values.len() {
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

    fn prepare_resident(&self, session: &MetalSession) -> Result<MetalNcSumcheckPlan, crate::MetalError> {
        let eq_table = flatten_table(&self.eq_beta);
        let digit_values = flatten_tables(&self.digit_values);
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
        session.prepare_nc_sumcheck(MetalNcSumcheckInputs {
            eq_table: &eq_table,
            digit_values: &digit_values,
            weights: &weights,
            witness_count: self.digit_values.len(),
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
            self.digit_values.len() as u64,
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
        self.dense = self.dense || 2 * self.width > D;
        self.width = if self.dense { D } else { 2 * self.width };
        Ok(())
    }

    fn activate_host_fallback(&mut self) {
        if self.host_fallback {
            return;
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
        let mut folded = Vec::with_capacity(self.digit_values.len());
        for witness in 0..self.digit_values.len() {
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
        let digit_rows = self
            .digit_values
            .iter()
            .enumerate()
            .map(|(witness, _)| std::array::from_fn(|lane| self.digit_value(witness, 0, lane)))
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
        if state.digit_words.len() != self.digit_values.len() * values_per_witness * 2 {
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
}

fn push_k_table(tables: &mut Vec<Vec<K>>, values: &[K], expected: usize) -> Option<usize> {
    if values.len() != expected {
        return None;
    }
    let index = tables.len();
    tables.push(values.to_vec());
    Some(index)
}

fn push_row_table(tables: &mut Vec<Vec<K>>, table: &RowTableSnapshot<'_>, expected: usize) -> Option<usize> {
    if table.real.len() != expected || table.imag.is_some_and(|imag| imag.len() != expected) {
        return None;
    }
    let values = match table.imag {
        Some(imag) => table
            .real
            .iter()
            .zip(imag)
            .map(|(&real, &imag)| from_complex(real, imag))
            .collect(),
        None => table.real.iter().copied().map(K::from).collect(),
    };
    let index = tables.len();
    tables.push(values);
    Some(index)
}

fn dense_digit_rows(table: &NcDigitTableView<'_>, len: usize) -> Option<Vec<[K; D]>> {
    match table {
        NcDigitTableView::Zero { len: table_len } if *table_len == len => Some(vec![[K::ZERO; D]; len]),
        NcDigitTableView::Lane0(values) if values.len() == len => Some(
            values
                .iter()
                .map(|&value| {
                    let mut row = [K::ZERO; D];
                    row[0] = value;
                    row
                })
                .collect(),
        ),
        NcDigitTableView::Strided { width, values } if values.len() == len * *width => Some(
            values
                .chunks_exact(*width)
                .enumerate()
                .map(|(index, chunk)| {
                    let mut row = [K::ZERO; D];
                    for (offset, &value) in chunk.iter().enumerate() {
                        row[(index * *width + offset) % D] = value;
                    }
                    row
                })
                .collect(),
        ),
        NcDigitTableView::Dense(rows) if rows.len() == len => Some(rows.to_vec()),
        _ => None,
    }
}

fn poly_mul_affine(poly: &mut [K], a: K, b: K, current_degree: usize) {
    let mut previous = K::ZERO;
    for coefficient in poly.iter_mut().take(current_degree + 2) {
        let old = *coefficient;
        *coefficient = a * old + b * previous;
        previous = old;
    }
}

fn fold_host(table: &mut Vec<K>, challenge: K) {
    let half = table.len() / 2;
    for index in 0..half {
        let left = table[2 * index];
        table[index] = left + challenge * (table[2 * index + 1] - left);
    }
    table.truncate(half);
}

fn k_to_words(value: K) -> KWords {
    KWords::new(value.real().as_canonical_u64(), value.imag().as_canonical_u64())
}

fn flatten_tables(tables: &[Vec<K>]) -> Vec<u64> {
    tables
        .iter()
        .flat_map(|table| {
            table.iter().flat_map(|&value| {
                let words = k_to_words(value);
                [words.c0, words.c1]
            })
        })
        .collect()
}

fn flatten_table(table: &[K]) -> Vec<u64> {
    table
        .iter()
        .flat_map(|&value| {
            let words = k_to_words(value);
            [words.c0, words.c1]
        })
        .collect()
}

fn words_to_k(value: KWords) -> K {
    from_complex(F::from_u64(value.c0), F::from_u64(value.c1))
}

fn decode_trace(trace: MetalSumcheckTrace) -> (Vec<Vec<K>>, Vec<K>, ([F; 8], usize)) {
    let coeffs = trace
        .coeffs
        .into_iter()
        .map(|round| round.into_iter().map(words_to_k).collect())
        .collect();
    let challenges = trace.challenges.into_iter().map(words_to_k).collect();
    let state = trace.transcript_state.map(F::from_u64);
    (coeffs, challenges, (state, trace.transcript_absorbed))
}
