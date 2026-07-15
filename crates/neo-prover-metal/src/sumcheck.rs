//! Metal-owned sumcheck table state used by the protocol engine backends.

mod encoding;
mod mask_residency;

use std::mem::size_of;
use std::time::{Duration, Instant};

use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_reductions::optimized_engine::oracle::{NcColSnapshot, NcDigitTableView, RowPhaseSnapshot};
use neo_reductions::optimized_engine::{
    FeEvalTable, FeMcsRowTables, FeRowRoundTrace, FeSumcheckBackend, NcColRoundTrace, NcFinalizedColState,
    NcSumcheckBackend,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{
    KWords, MetalAjtaiRingForms, MetalDecFormPlan, MetalDeferredEvalTable, MetalDeferredMcsRowTables, MetalError,
    MetalFeOraclePlan, MetalFeSumcheckInputs, MetalFeSumcheckPlan, MetalFeTableInput, MetalNcDigitInput,
    MetalNcFinalState, MetalNcSumcheckInputs, MetalNcSumcheckPlan, MetalNcSumcheckTrace, MetalResidentWitness,
    MetalSession, MetalWitnessMasks,
};
use encoding::{
    decode_ajtai_y_eval, decode_trace, dense_digit_rows, flatten_table, flatten_tables, fold_host, k_to_words,
    poly_mul_affine, push_k_table, push_row_table, signed_unit_mask_words, words_to_k,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SumcheckProfile {
    pub fe_rounds: usize,
    pub fe_mcs_tables: usize,
    pub fe_mcs_table_bytes: usize,
    pub fe_seeded_build: Duration,
    pub fe_seeded_patch_entries: usize,
    pub fe_seeded_patch_bytes: usize,
    pub fe_explicit_coefficients: usize,
    pub fe_explicit_row_list_histogram: [usize; 8],
    pub fe_max_explicit_row_entries: usize,
    pub fe_carried_eval_on_metal: bool,
    pub ajtai_y_eval: Duration,
    pub ajtai_seeded_build: Duration,
    pub ajtai_device_eval: Duration,
    pub ajtai_tensor_gpu: Duration,
    pub ajtai_form_gpu: Duration,
    pub ajtai_tail_gpu: Duration,
    pub ajtai_seeded_patch_entries: usize,
    pub ajtai_seeded_patch_bytes: usize,
    pub ajtai_form_blocks: usize,
    pub ajtai_form_bytes: usize,
    pub ajtai_explicit_coefficients: usize,
    pub ajtai_signed_unit_coefficients: usize,
    pub ajtai_explicit_form_list_histogram: [usize; 8],
    pub ajtai_max_explicit_form_list_entries: usize,
    pub ajtai_parallel_form_lists: usize,
    pub ajtai_parallel_form_entries: usize,
    pub ajtai_y_eval_on_metal: bool,
    pub nc_rounds: usize,
    pub nc_input_witnesses: usize,
    pub nc_active_witnesses: usize,
    pub nc_mask_native_on_metal: bool,
    pub folded_tables: usize,
    pub metal_failed: bool,
}

pub(crate) struct MetalFeBackend<'a> {
    session: &'a MetalSession,
    oracle: Option<FeOracle>,
    y_eval_plan: Option<&'a MetalDecFormPlan>,
    y_eval_only: bool,
    ajtai_forms: Option<MetalAjtaiRingForms>,
    oracle_plan: Option<&'a MetalFeOraclePlan>,
    witness_masks: Option<&'a MetalWitnessMasks>,
    resident_running_id: Option<u64>,
    deferred_eval: Option<MetalDeferredEvalTable>,
    deferred_mcs: Vec<MetalDeferredMcsRowTables>,
    profile: SumcheckProfile,
}

pub(crate) struct MetalNcBackend<'a> {
    session: &'a MetalSession,
    source: NcSource,
    shared_masks: Option<MetalWitnessMasks>,
    oracle: Option<NcOracle>,
    profile: SumcheckProfile,
}

#[derive(Clone)]
struct NcSignedMasks {
    words: Vec<u64>,
    blocks: usize,
    active_rows: usize,
    rows: usize,
    witness_count: usize,
}

enum NcSource {
    Values(Vec<Vec<K>>),
    SignedMasks(NcSignedMasks),
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
    deferred_eval: Option<MetalDeferredEvalTable>,
    deferred_mcs: Vec<MetalDeferredMcsRowTables>,
    table_sources: Vec<FeTableSource>,
    beta_r: Vec<K>,
    r_inputs: Option<Vec<K>>,
}

enum FeTableSource {
    Host,
    Beta,
    Inputs,
    DeferredMcs { deferred: usize, table: usize },
    DeferredEval,
}

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
    pending_challenge: Option<K>,
    challenges: Vec<K>,
    host_fallback: bool,
}

impl<'a> MetalFeBackend<'a> {
    pub(crate) fn new(session: &'a MetalSession) -> Self {
        Self {
            session,
            oracle: None,
            y_eval_plan: None,
            y_eval_only: false,
            ajtai_forms: None,
            oracle_plan: None,
            witness_masks: None,
            resident_running_id: None,
            deferred_eval: None,
            deferred_mcs: Vec::new(),
            profile: SumcheckProfile::default(),
        }
    }

    pub(crate) fn y_eval_only(mut self, plan: Option<&'a MetalDecFormPlan>) -> Self {
        self.y_eval_plan = plan;
        self.y_eval_only = true;
        self
    }

    pub(crate) fn oracle_plan(mut self, plan: &'a MetalFeOraclePlan, resident_running_id: Option<u64>) -> Self {
        self.oracle_plan = Some(plan);
        self.resident_running_id = resident_running_id;
        self.y_eval_only = false;
        self
    }

    pub(crate) fn witness_masks(mut self, masks: Option<&'a MetalWitnessMasks>) -> Self {
        self.witness_masks = masks;
        self
    }

    pub(crate) fn profile(&self) -> SumcheckProfile {
        self.profile
    }

    pub(crate) fn take_ajtai_forms(&mut self) -> Option<MetalAjtaiRingForms> {
        self.ajtai_forms.take()
    }

    fn finish_ajtai_y_eval(
        &mut self,
        started: Instant,
        result: Result<(Vec<u64>, MetalAjtaiRingForms, crate::MetalAjtaiYProfile), crate::MetalError>,
        witness_count: usize,
        matrix_count: usize,
    ) -> Option<Vec<Vec<[K; D]>>> {
        self.profile.ajtai_y_eval += started.elapsed();
        match result {
            Ok((words, forms, ajtai_profile)) => {
                let Some(y_eval) = decode_ajtai_y_eval(&words, witness_count, matrix_count) else {
                    self.profile.metal_failed = true;
                    return None;
                };
                self.profile.ajtai_y_eval_on_metal = true;
                self.profile.ajtai_seeded_build = ajtai_profile.seeded_build;
                self.profile.ajtai_device_eval = ajtai_profile.device_eval;
                self.profile.ajtai_tensor_gpu = ajtai_profile.tensor_gpu;
                self.profile.ajtai_form_gpu = ajtai_profile.form_gpu;
                self.profile.ajtai_tail_gpu = ajtai_profile.tail_gpu;
                self.profile.ajtai_seeded_patch_entries = ajtai_profile.seeded_patch_entries;
                self.profile.ajtai_seeded_patch_bytes = ajtai_profile.seeded_patch_bytes;
                self.profile.ajtai_form_blocks = ajtai_profile.form_blocks;
                self.profile.ajtai_form_bytes = ajtai_profile.form_bytes;
                self.profile.ajtai_explicit_coefficients = ajtai_profile.explicit_coefficients;
                self.profile.ajtai_signed_unit_coefficients = ajtai_profile.signed_unit_coefficients;
                self.profile.ajtai_explicit_form_list_histogram = ajtai_profile.explicit_form_list_histogram;
                self.profile.ajtai_max_explicit_form_list_entries = ajtai_profile.max_explicit_form_list_entries;
                self.profile.ajtai_parallel_form_lists = ajtai_profile.parallel_form_lists;
                self.profile.ajtai_parallel_form_entries = ajtai_profile.parallel_form_entries;
                self.ajtai_forms = Some(forms);
                Some(y_eval)
            }
            Err(_) => {
                self.profile.metal_failed = true;
                None
            }
        }
    }
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
            profile: SumcheckProfile::default(),
        })
    }

    pub(crate) fn shared_masks(&self) -> Option<&MetalWitnessMasks> {
        self.shared_masks.as_ref()
    }

    pub(crate) fn profile(&self) -> SumcheckProfile {
        self.profile
    }

    pub(crate) fn enqueue_rlc_witness_mix_from_resident_masks(
        &self,
        rhos: &[i8],
        fresh_count: usize,
        input_count: usize,
        cols: usize,
        resident_id: Option<u64>,
    ) -> Result<Option<MetalResidentWitness>, MetalError> {
        if self.profile.metal_failed {
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
        if self.profile.metal_failed {
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

impl FeSumcheckBackend for MetalFeBackend<'_> {
    fn defers_row_equality_tables(&self) -> bool {
        !self.y_eval_only
    }

    fn start(&mut self, snapshot: &RowPhaseSnapshot<'_>) -> bool {
        if self.y_eval_only {
            return false;
        }
        self.oracle = FeOracle::from_snapshot(
            snapshot,
            self.session,
            self.deferred_eval.take(),
            std::mem::take(&mut self.deferred_mcs),
        );
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

    fn ajtai_y_eval(
        &mut self,
        cache: &neo_reductions::superneo_eval::SuperneoEvalCache,
        chi_r: &[K],
        n_eff: usize,
        witnesses: &[&Mat<F>],
    ) -> Option<Vec<Vec<[K; D]>>> {
        let plan = self.y_eval_plan?;
        if !plan.matches(cache) || witnesses.is_empty() {
            return None;
        }
        let started = Instant::now();
        let blocks = witnesses[0].cols();
        let resident_masks = self
            .witness_masks
            .filter(|masks| masks.matches(witnesses.len(), blocks));
        let mask_words = match resident_masks {
            Some(_) => Vec::new(),
            None => match signed_unit_mask_words(witnesses, blocks) {
                Some(words) => words,
                None => {
                    self.profile.ajtai_y_eval += started.elapsed();
                    return None;
                }
            },
        };
        let result = self.session.eval_ajtai_y_from_signed_masks(
            plan,
            cache,
            chi_r,
            n_eff,
            &mask_words,
            resident_masks,
            witnesses.len(),
        );
        self.finish_ajtai_y_eval(started, result, witnesses.len(), cache.matrix_caches().len())
    }

    fn ajtai_y_eval_from_row_challenges(
        &mut self,
        cache: &neo_reductions::superneo_eval::SuperneoEvalCache,
        row_challenges: &[K],
        n_eff: usize,
        witnesses: &[&Mat<F>],
    ) -> Option<Vec<Vec<[K; D]>>> {
        let plan = self.y_eval_plan?;
        if !plan.matches(cache) || witnesses.is_empty() {
            return None;
        }
        let started = Instant::now();
        let blocks = witnesses[0].cols();
        let resident_masks = self
            .witness_masks
            .filter(|masks| masks.matches(witnesses.len(), blocks));
        let mask_words = match resident_masks {
            Some(_) => Vec::new(),
            None => match signed_unit_mask_words(witnesses, blocks) {
                Some(words) => words,
                None => {
                    self.profile.ajtai_y_eval += started.elapsed();
                    return None;
                }
            },
        };
        let result = self
            .session
            .eval_ajtai_y_from_signed_masks_and_row_challenges(
                plan,
                cache,
                row_challenges,
                n_eff,
                &mask_words,
                resident_masks,
                witnesses.len(),
            );
        self.finish_ajtai_y_eval(started, result, witnesses.len(), cache.matrix_caches().len())
    }

    fn mcs_row_tables(
        &mut self,
        cache: &neo_reductions::superneo_eval::SuperneoEvalCache,
        mcs_idx: usize,
        f_var_indices: &[usize],
        z_blocks: &neo_reductions::superneo_eval::SuperneoZBlocks,
        n_eff: usize,
        n_pad: usize,
    ) -> Option<FeMcsRowTables> {
        let plan = self.oracle_plan?;
        if !plan.matches(cache) || !z_blocks.imag_all_zero() {
            return None;
        }
        match self.session.build_mcs_row_tables(
            plan,
            cache,
            mcs_idx,
            f_var_indices,
            z_blocks,
            self.witness_masks,
            n_eff,
            n_pad,
        ) {
            Ok(tables) => {
                self.profile.fe_mcs_tables += f_var_indices.len();
                self.profile.fe_mcs_table_bytes += f_var_indices.len() * n_pad * size_of::<u64>();
                self.profile.fe_seeded_build += tables.seeded_build();
                self.profile.fe_seeded_patch_entries += tables.seeded_patch_entries();
                self.profile.fe_seeded_patch_bytes += tables.seeded_patch_bytes();
                self.profile.fe_explicit_coefficients = plan.explicit_coefficients();
                self.profile.fe_explicit_row_list_histogram = plan.explicit_row_list_histogram();
                self.profile.fe_max_explicit_row_entries = plan.max_explicit_row_entries();
                self.deferred_mcs.push(tables);
                Some(FeMcsRowTables::Deferred)
            }
            Err(_) => {
                self.profile.metal_failed = true;
                None
            }
        }
    }

    fn serves_carried_eval_table(&self) -> bool {
        self.oracle_plan
            .is_some_and(MetalFeOraclePlan::supports_resident_eval)
            && self
                .resident_running_id
                .and_then(|id| self.session.resident_running_shape(id))
                .is_some()
    }

    fn carried_eval_table(
        &mut self,
        cache: &neo_reductions::superneo_eval::SuperneoEvalCache,
        carried_coeffs: &[K],
        _k_mcs: usize,
        weights: &[K; D],
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Option<FeEvalTable> {
        if !self.serves_carried_eval_table() {
            return None;
        }
        let (Some(plan), Some(resident_id)) = (self.oracle_plan, self.resident_running_id) else {
            return None;
        };
        let result = plan
            .matches(cache)
            .then_some(())
            .ok_or(crate::MetalError::Shape("resident Pi_CCS oracle plan is stale"))
            .and_then(|()| {
                self.session.build_carried_eval_table(
                    plan,
                    resident_id,
                    carried_coeffs,
                    weights,
                    mat_coeffs,
                    n_eff,
                    n_pad,
                )
            });
        match result {
            Ok(table) => {
                self.deferred_eval = Some(table);
                self.profile.fe_carried_eval_on_metal = true;
            }
            Err(_) => self.profile.metal_failed = true,
        }
        Some(FeEvalTable::Deferred)
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
        self.profile.folded_tables += oracle.witness_count + 1;
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
                self.profile.nc_mask_native_on_metal |= oracle.initial_masks.is_some();
                self.profile.folded_tables += (oracle.witness_count + 1) * rounds;
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
    fn from_snapshot(
        snapshot: &RowPhaseSnapshot<'_>,
        session: &MetalSession,
        deferred_eval: Option<MetalDeferredEvalTable>,
        deferred_mcs: Vec<MetalDeferredMcsRowTables>,
    ) -> Option<Self> {
        let initial_shape_valid = snapshot.cur_len >= 2
            && snapshot.cur_len.is_power_of_two()
            && snapshot.beta_r.len() == snapshot.cur_len.ilog2() as usize
            && (snapshot.eq_beta_r_tbl.is_empty() || snapshot.eq_beta_r_tbl.len() == snapshot.cur_len)
            && snapshot.row_phase_deg_max <= snapshot.sumcheck_degree_bound
            && snapshot.zero_mcs.len() == snapshot.f_var_tables_by_mcs.len()
            && snapshot.deferred_mcs.len() == snapshot.f_var_tables_by_mcs.len();
        if !initial_shape_valid {
            eprintln!(
                "nightstream Metal FE snapshot rejected: initial shape; n_pad={} row_degree={} canonical_degree={} zero_mcs={} deferred_flags={} mcs={}",
                snapshot.cur_len,
                snapshot.row_phase_deg_max,
                snapshot.sumcheck_degree_bound,
                snapshot.zero_mcs.len(),
                snapshot.deferred_mcs.len(),
                snapshot.f_var_tables_by_mcs.len(),
            );
            return None;
        }

        let mut tables = Vec::new();
        let mut table_sources = Vec::new();
        let eq_beta = tables.len();
        tables.push(snapshot.eq_beta_r_tbl.to_vec());
        table_sources.push(FeTableSource::Beta);
        let eq_inputs = match snapshot.eq_r_inputs_tbl {
            Some(table) => {
                let index = if table.is_empty() {
                    tables.push(Vec::new());
                    tables.len() - 1
                } else {
                    push_k_table(&mut tables, table, snapshot.cur_len)?
                };
                table_sources.push(FeTableSource::Inputs);
                Some(index)
            }
            None => None,
        };
        let deferred_eval = match (snapshot.deferred_eval_tbl, deferred_eval) {
            (true, Some(table)) if table.matches(snapshot.cur_len) => Some(table),
            (true, Some(table)) => {
                eprintln!(
                    "nightstream Metal FE snapshot rejected: deferred Eval buffer mismatch for n_pad={}",
                    snapshot.cur_len,
                );
                drop(table);
                return None;
            }
            (true, None) => {
                eprintln!("nightstream Metal FE snapshot rejected: deferred Eval buffer is missing");
                return None;
            }
            (false, _) => None,
        };
        let eval = match (snapshot.eval_tbl, deferred_eval.as_ref()) {
            (Some(table), None) => {
                let index = push_k_table(&mut tables, table, snapshot.cur_len)?;
                table_sources.push(FeTableSource::Host);
                Some(index)
            }
            (None, Some(_)) => {
                let index = tables.len();
                tables.push(Vec::new());
                table_sources.push(FeTableSource::DeferredEval);
                Some(index)
            }
            (None, None) => None,
            (Some(_), Some(_)) => return None,
        };
        if eq_inputs.is_some() != eval.is_some() {
            return None;
        }

        let mut mcs_tables = Vec::with_capacity(snapshot.f_var_tables_by_mcs.len());
        let mut used_deferred = vec![false; deferred_mcs.len()];
        for (mcs_idx, per_mcs) in snapshot.f_var_tables_by_mcs.iter().enumerate() {
            if snapshot.zero_mcs[mcs_idx] {
                if snapshot.deferred_mcs[mcs_idx] {
                    return None;
                }
                mcs_tables.push(Vec::new());
                continue;
            }
            if snapshot.deferred_mcs[mcs_idx] {
                if !per_mcs.is_empty() {
                    eprintln!(
                        "nightstream Metal FE snapshot rejected: deferred MCS {mcs_idx} retained {} host tables",
                        per_mcs.len(),
                    );
                    return None;
                }
                let Some((deferred, _)) = deferred_mcs.iter().enumerate().find(|(index, tables)| {
                    !used_deferred[*index] && tables.matches(mcs_idx, snapshot.cur_len, snapshot.f_var_count)
                }) else {
                    eprintln!(
                        "nightstream Metal FE snapshot rejected: no deferred buffer matches mcs={mcs_idx} n_pad={} f_vars={} candidates={}",
                        snapshot.cur_len,
                        snapshot.f_var_count,
                        deferred_mcs.len(),
                    );
                    return None;
                };
                used_deferred[deferred] = true;
                let mut indices = Vec::with_capacity(snapshot.f_var_count);
                for table in 0..snapshot.f_var_count {
                    let index = tables.len();
                    tables.push(Vec::new());
                    table_sources.push(FeTableSource::DeferredMcs { deferred, table });
                    indices.push(index);
                }
                mcs_tables.push(indices);
                continue;
            }
            if per_mcs.len() != snapshot.f_var_count {
                return None;
            }
            let mut indices = Vec::with_capacity(per_mcs.len());
            for table in per_mcs {
                indices.push(push_row_table(&mut tables, table, snapshot.cur_len)?);
                table_sources.push(FeTableSource::Host);
            }
            mcs_tables.push(indices);
        }
        if used_deferred.iter().any(|&used| !used) || table_sources.len() != tables.len() {
            eprintln!(
                "nightstream Metal FE snapshot rejected: source accounting; used={used_deferred:?} sources={} tables={}",
                table_sources.len(),
                tables.len(),
            );
            return None;
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
            deferred_eval,
            deferred_mcs,
            table_sources,
            beta_r: snapshot.beta_r.to_vec(),
            r_inputs: snapshot.r_inputs.map(<[K]>::to_vec),
        };
        oracle.resident = match oracle.prepare_resident(session) {
            Ok(plan) => Some(plan),
            Err(error) => {
                eprintln!(
                    "nightstream Metal FE startup failed: {error}; mcs={} deferred_mcs={} f_vars={} tables={} n_pad={} canonical_coefficients={} row_degree={} allocated_bytes={}",
                    snapshot.f_var_tables_by_mcs.len(),
                    oracle.deferred_mcs.len(),
                    snapshot.f_var_count,
                    oracle.tables.len(),
                    snapshot.cur_len,
                    snapshot.sumcheck_degree_bound + 1,
                    snapshot.row_phase_deg_max,
                    session.activity().current_allocated_bytes,
                );
                return None;
            }
        };
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
        let tables = self
            .table_sources
            .iter()
            .enumerate()
            .map(|(index, source)| match source {
                FeTableSource::Host => MetalFeTableInput::Host(&self.tables[index]),
                FeTableSource::Beta => MetalFeTableInput::TensorPoint(&self.beta_r),
                FeTableSource::Inputs => MetalFeTableInput::TensorPoint(
                    self.r_inputs
                        .as_deref()
                        .expect("FE input tensor source requires r_inputs"),
                ),
                FeTableSource::DeferredMcs { deferred, table } => MetalFeTableInput::DeferredMcs {
                    tables: &self.deferred_mcs[*deferred],
                    table: *table,
                },
                FeTableSource::DeferredEval => MetalFeTableInput::DeferredEval(
                    self.deferred_eval
                        .as_ref()
                        .expect("FE deferred Eval source requires a table"),
                ),
            })
            .collect::<Vec<_>>();
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
        assert!(
            self.deferred_eval.is_none() && self.deferred_mcs.is_empty(),
            "device-owned FE tables failed during the row phase"
        );
        if self.tables[self.eq_beta].is_empty() {
            self.tables[self.eq_beta] = neo_ccs::utils::tensor_point_parallel(&self.beta_r);
        }
        if let Some(eq_inputs) = self
            .eq_inputs
            .filter(|&index| self.tables[index].is_empty())
        {
            let point = self
                .r_inputs
                .as_deref()
                .expect("deferred FE input equality point");
            self.tables[eq_inputs] = neo_ccs::utils::tensor_point_parallel(point);
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
        self.dense = self.dense || 2 * self.width > D;
        self.width = if self.dense { D } else { 2 * self.width };
        Ok(())
    }

    fn activate_host_fallback(&mut self) {
        if self.host_fallback {
            return;
        }
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
