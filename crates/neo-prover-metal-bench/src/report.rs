//! Stable JSON-facing benchmark schema and timing policy.

use std::time::Duration;

use neo_prover_metal::{MetalActivity, MetalDeviceInfo, MetalNifsProfile};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const REPORT_SCHEMA_VERSION: u32 = 10;

const MAX_FIELD_ELEMENTS: usize = 1 << 22;
const MAX_POSEIDON_HASHES: usize = 1 << 20;
const MAX_POSEIDON_FIELDS_PER_HASH: usize = 1 << 16;
const MAX_KX_ELEMENTS: usize = 1 << 22;
const MAX_KX_ROUNDS: usize = 1 << 12;
const MAX_AJTAI_ROWS: usize = 1 << 10;
const MAX_AJTAI_COLS: usize = 1 << 14;
const MAX_FE_TABLE_ELEMENTS: usize = 1 << 24;
const MAX_LIFECYCLE_REPETITIONS: usize = 10;
const MAX_LIFECYCLE_SOAK_SECONDS: usize = 60;
const MAX_COMPOSITE_WORDS: usize = 1 << 26;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkConfig {
    pub samples: usize,
    pub field_elements: usize,
    pub poseidon_hashes: usize,
    pub poseidon_fields_per_hash: usize,
    pub kx_elements: usize,
    pub kx_rounds: usize,
    pub ajtai_rows: usize,
    pub ajtai_cols: usize,
    pub fe_table_elements: usize,
    pub lifecycle_repetitions: usize,
    #[serde(default)]
    pub lifecycle_soak_seconds: usize,
    pub run_sha256_lifecycle: bool,
    pub run_nebula_lifecycle: bool,
}

impl BenchmarkConfig {
    pub fn m0_m1() -> Self {
        Self {
            samples: 5,
            field_elements: 1 << 18,
            poseidon_hashes: 1 << 15,
            poseidon_fields_per_hash: 8,
            kx_elements: 1 << 18,
            kx_rounds: 64,
            ajtai_rows: 18,
            ajtai_cols: 8_377,
            fe_table_elements: 1 << 18,
            lifecycle_repetitions: 3,
            lifecycle_soak_seconds: 0,
            run_sha256_lifecycle: true,
            run_nebula_lifecycle: true,
        }
    }

    pub fn smoke() -> Self {
        Self {
            samples: 2,
            field_elements: 1 << 10,
            poseidon_hashes: 1 << 7,
            poseidon_fields_per_hash: 8,
            kx_elements: 1 << 9,
            kx_rounds: 4,
            ajtai_rows: 2,
            ajtai_cols: 3,
            fe_table_elements: 1 << 9,
            lifecycle_repetitions: 1,
            lifecycle_soak_seconds: 0,
            run_sha256_lifecycle: false,
            run_nebula_lifecycle: false,
        }
    }

    pub fn m6() -> Self {
        let mut config = Self::m0_m1();
        config.lifecycle_repetitions = 5;
        config.lifecycle_soak_seconds = 60;
        config
    }

    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.samples == 0 || self.samples > 25 {
            return Err(BenchmarkError::Config("samples must be in 1..=25"));
        }
        if self.poseidon_fields_per_hash == 0 || self.poseidon_fields_per_hash > MAX_POSEIDON_FIELDS_PER_HASH {
            return Err(BenchmarkError::Config("Poseidon fields per hash must be in 1..=65536"));
        }
        if self.field_elements == 0
            || self.poseidon_hashes == 0
            || self.kx_elements == 0
            || self.kx_rounds == 0
            || self.ajtai_rows == 0
            || self.ajtai_cols == 0
            || self.fe_table_elements < 2
            || !self.fe_table_elements.is_power_of_two()
        {
            return Err(BenchmarkError::Config(
                "primitive sizes must be nonzero and FE table size must be a power of two",
            ));
        }
        if self.field_elements > MAX_FIELD_ELEMENTS
            || self.poseidon_hashes > MAX_POSEIDON_HASHES
            || self.kx_elements > MAX_KX_ELEMENTS
            || self.kx_rounds > MAX_KX_ROUNDS
            || self.ajtai_rows > MAX_AJTAI_ROWS
            || self.ajtai_cols > MAX_AJTAI_COLS
            || self.fe_table_elements > MAX_FE_TABLE_ELEMENTS
            || self.lifecycle_repetitions > MAX_LIFECYCLE_REPETITIONS
            || self.lifecycle_soak_seconds > MAX_LIFECYCLE_SOAK_SECONDS
        {
            return Err(BenchmarkError::Config(
                "benchmark size exceeds the supported safety limit",
            ));
        }
        let poseidon_words = self
            .poseidon_hashes
            .checked_mul(self.poseidon_fields_per_hash)
            .ok_or(BenchmarkError::Config("benchmark dimensions overflow"))?;
        let kx_work = self
            .kx_elements
            .checked_mul(self.kx_rounds)
            .ok_or(BenchmarkError::Config("benchmark dimensions overflow"))?;
        let ajtai_words = self
            .ajtai_rows
            .checked_mul(self.ajtai_cols)
            .and_then(|words| words.checked_mul(neo_math::D))
            .ok_or(BenchmarkError::Config("benchmark dimensions overflow"))?;
        if poseidon_words > MAX_COMPOSITE_WORDS || kx_work > MAX_COMPOSITE_WORDS || ajtai_words > MAX_COMPOSITE_WORDS {
            return Err(BenchmarkError::Config(
                "benchmark composite size exceeds the supported safety limit",
            ));
        }
        if (self.run_sha256_lifecycle || self.run_nebula_lifecycle) && self.lifecycle_repetitions == 0 {
            return Err(BenchmarkError::Config(
                "lifecycle repetitions must be nonzero when lifecycle benchmarks are enabled",
            ));
        }
        Ok(())
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self::m0_m1()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TimingSummary {
    pub samples: usize,
    /// Ordered measurements before percentile sorting.
    pub raw_ms: Vec<f64>,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p95_ms: f64,
    pub coefficient_of_variation: f64,
}

impl TimingSummary {
    pub fn from_durations(mut values: Vec<Duration>) -> Self {
        assert!(!values.is_empty(), "timing summary needs samples");
        let raw_ms = values
            .iter()
            .map(|duration| duration.as_secs_f64() * 1e3)
            .collect::<Vec<_>>();
        values.sort_unstable();
        let milliseconds = values
            .iter()
            .map(|duration| duration.as_secs_f64() * 1e3)
            .collect::<Vec<_>>();
        let mean = milliseconds.iter().sum::<f64>() / milliseconds.len() as f64;
        let variance = milliseconds
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / milliseconds.len() as f64;
        let p95_index = ((milliseconds.len() as f64 * 0.95).ceil() as usize)
            .saturating_sub(1)
            .min(milliseconds.len() - 1);
        Self {
            samples: milliseconds.len(),
            raw_ms,
            median_ms: milliseconds[milliseconds.len() / 2],
            min_ms: milliseconds[0],
            max_ms: milliseconds[milliseconds.len() - 1],
            p95_ms: milliseconds[p95_index],
            coefficient_of_variation: if mean == 0.0 { 0.0 } else { variance.sqrt() / mean },
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ActivityReport {
    pub command_buffers: u64,
    pub dispatches: u64,
    pub host_waits: u64,
    pub allocated_bytes: u64,
    pub uploaded_bytes: u64,
    pub downloaded_bytes: u64,
    pub current_allocated_bytes: u64,
}

impl From<MetalActivity> for ActivityReport {
    fn from(value: MetalActivity) -> Self {
        Self {
            command_buffers: value.command_buffers,
            dispatches: value.dispatches,
            host_waits: value.host_waits,
            allocated_bytes: value.allocated_bytes,
            uploaded_bytes: value.uploaded_bytes,
            downloaded_bytes: value.downloaded_bytes,
            current_allocated_bytes: value.current_allocated_bytes,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CandidateReport {
    pub name: String,
    pub setup_ms: f64,
    pub timing: TimingSummary,
    pub throughput_per_second: f64,
    pub speedup_over_cpu: f64,
    pub activity: Option<ActivityReport>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PrimitiveReport {
    pub name: String,
    pub work_items: usize,
    pub parity_ok: bool,
    pub crossover_required: bool,
    pub cpu: TimingSummary,
    pub candidates: Vec<CandidateReport>,
    pub selected_candidate: String,
    pub selected_speedup_over_cpu: f64,
    pub crossover_gate_passed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DeviceReport {
    pub os: String,
    pub arch: String,
    pub gpu_name: String,
    pub unified_memory: bool,
    pub recommended_working_set_bytes: u64,
}

impl From<MetalDeviceInfo> for DeviceReport {
    fn from(value: MetalDeviceInfo) -> Self {
        Self {
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            gpu_name: value.name,
            unified_memory: value.unified_memory,
            recommended_working_set_bytes: value.recommended_working_set_bytes,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub device: DeviceReport,
    pub config: BenchmarkConfig,
    pub timing_contract: Vec<String>,
    pub primitives: Vec<PrimitiveReport>,
    pub lifecycle: Vec<LifecycleReport>,
    pub m1_parity_passed: bool,
    pub m1_crossover_passed: bool,
    pub m2_lifecycle_passed: bool,
    pub m2_crossover_passed: bool,
    pub m3_residency_passed: bool,
    pub m3_crossover_passed: bool,
    pub m4_projection_passed: bool,
    pub m4_crossover_passed: bool,
    pub m5_adapter_passed: bool,
    pub lifecycle_crossover: Vec<LifecycleCrossoverReport>,
    pub sustained: Option<SustainedLifecycleReport>,
    pub m6_pipeline_passed: bool,
    pub m6_crossover_passed: bool,
    pub m6_sustained_passed: bool,
    pub m6_passed: bool,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LifecycleCrossoverReport {
    pub name: String,
    pub crossover_required: bool,
    pub median_speedup_over_cpu: f64,
    pub p95_speedup_over_cpu: f64,
    pub median_target: f64,
    pub p95_target: f64,
    pub proof_parity_ok: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SustainedLifecycleReport {
    pub name: String,
    pub seconds_per_backend: usize,
    pub cpu_elapsed_ms: f64,
    pub metal_elapsed_ms: f64,
    pub cpu_proofs: usize,
    pub metal_proofs: usize,
    pub cpu_proofs_per_second: f64,
    pub metal_proofs_per_second: f64,
    pub speedup_over_cpu: f64,
    pub target_speedup: f64,
    pub proof_parity_ok: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LifecycleReport {
    pub name: String,
    pub backend: String,
    pub verification_mode: String,
    pub synthesis_ms: f64,
    pub preprocessing_ms: f64,
    pub online: TimingSummary,
    pub pipeline: Option<LifecyclePipelineReport>,
    pub verify_ms: TimingSummary,
    pub nifs_profile: Option<NifsStageReport>,
    pub audit_debug_chars: usize,
    pub semantic_result_ok: bool,
    pub proof_parity_ok: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LifecyclePipelineReport {
    pub synthesis_work: TimingSummary,
    pub fold_work: TimingSummary,
    pub final_materialization: TimingSummary,
    pub overlap_saved: TimingSummary,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NifsStageReport {
    pub folds_per_sample: usize,
    pub total: TimingSummary,
    pub pi_ccs: TimingSummary,
    pub ajtai_y_eval: TimingSummary,
    pub pi_rlc: TimingSummary,
    pub pi_dec: TimingSummary,
    pub dec_form_build: TimingSummary,
    pub dec_projection: TimingSummary,
    pub dec_host_materialization: TimingSummary,
    pub fe_on_metal: bool,
    pub ajtai_y_eval_on_metal: bool,
    pub nc_on_metal: bool,
    pub nc_mask_native_on_metal: bool,
    pub rlc_witness_on_metal: bool,
    pub rlc_witness_resident_only: bool,
    pub rlc_rho_small_coefficients: bool,
    pub dec_split_on_metal: bool,
    pub dec_recomposition_on_metal: bool,
    pub dec_forms_on_metal: bool,
    pub dec_y_on_metal: bool,
    pub dec_commit_on_metal: bool,
    pub resident_input_folds: usize,
    pub resident_output_folds: usize,
    pub deferred_proof_folds: usize,
    pub deferred_running_folds: usize,
    pub recursive_compile_reverify_required: bool,
    pub activity_per_sample: ActivityReport,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NifsProfileSample {
    total: Duration,
    pi_ccs: Duration,
    ajtai_y_eval: Duration,
    pi_rlc: Duration,
    pi_dec: Duration,
    dec_form_build: Duration,
    dec_projection: Duration,
    dec_host_materialization: Duration,
    folds: usize,
    fe_on_metal: bool,
    ajtai_y_eval_on_metal: bool,
    nc_on_metal: bool,
    nc_mask_native_on_metal: bool,
    rlc_witness_on_metal: bool,
    rlc_witness_resident_only: bool,
    rlc_rho_small_coefficients: bool,
    dec_split_on_metal: bool,
    dec_recomposition_on_metal: bool,
    dec_forms_on_metal: bool,
    dec_y_on_metal: bool,
    dec_commit_on_metal: bool,
    resident_input_folds: usize,
    resident_output_folds: usize,
    deferred_proof_folds: usize,
    deferred_running_folds: usize,
    recursive_compile_reverify_required: bool,
    activity: MetalActivity,
}

impl NifsProfileSample {
    pub(crate) fn from_profiles(profiles: Vec<MetalNifsProfile>) -> Self {
        let mut sample = Self {
            fe_on_metal: true,
            ajtai_y_eval_on_metal: true,
            nc_on_metal: true,
            nc_mask_native_on_metal: true,
            rlc_witness_on_metal: true,
            rlc_witness_resident_only: true,
            rlc_rho_small_coefficients: true,
            dec_split_on_metal: true,
            dec_recomposition_on_metal: true,
            dec_forms_on_metal: true,
            dec_y_on_metal: true,
            dec_commit_on_metal: true,
            ..Self::default()
        };
        for profile in profiles {
            sample.total += profile.total;
            sample.pi_ccs += profile.pi_ccs;
            sample.ajtai_y_eval += profile.ajtai_y_eval;
            sample.pi_rlc += profile.pi_rlc;
            sample.pi_dec += profile.pi_dec;
            sample.dec_form_build += profile.dec_form_build;
            sample.dec_projection += profile.dec_projection;
            sample.dec_host_materialization += profile.dec_host_materialization;
            sample.folds += 1;
            sample.fe_on_metal &= profile.fe_on_metal;
            sample.ajtai_y_eval_on_metal &= profile.ajtai_y_eval_on_metal;
            sample.nc_on_metal &= profile.nc_on_metal;
            sample.nc_mask_native_on_metal &= profile.nc_mask_native_on_metal;
            sample.rlc_witness_on_metal &= profile.rlc_witness_on_metal;
            sample.rlc_witness_resident_only &= profile.rlc_witness_resident_only;
            sample.rlc_rho_small_coefficients &= profile.rlc_rho_small_coefficients;
            sample.dec_split_on_metal &= profile.dec_split_on_metal;
            sample.dec_recomposition_on_metal &= profile.dec_recomposition_on_metal;
            sample.dec_forms_on_metal &= profile.dec_forms_on_metal;
            sample.dec_y_on_metal &= profile.dec_y_on_metal;
            sample.dec_commit_on_metal &= profile.dec_commit_on_metal;
            sample.resident_input_folds += usize::from(profile.resident_running_input);
            sample.resident_output_folds += usize::from(profile.resident_running_output);
            sample.deferred_proof_folds += usize::from(profile.proof_deferred);
            sample.deferred_running_folds += usize::from(profile.running_deferred);
            sample.recursive_compile_reverify_required |= profile.recursive_compile_reverify_required;
            add_activity(&mut sample.activity, profile.activity);
        }
        sample
    }
}

pub(crate) fn summarize_nifs_profiles(samples: Vec<NifsProfileSample>) -> NifsStageReport {
    assert!(!samples.is_empty(), "NIFS profile summary needs samples");
    let shape = samples[0];
    assert!(shape.folds > 0, "NIFS profile sample needs at least one fold");
    assert!(samples.iter().all(|sample| {
        sample.folds == shape.folds
            && sample.fe_on_metal == shape.fe_on_metal
            && sample.ajtai_y_eval_on_metal == shape.ajtai_y_eval_on_metal
            && sample.nc_on_metal == shape.nc_on_metal
            && sample.nc_mask_native_on_metal == shape.nc_mask_native_on_metal
            && sample.rlc_witness_on_metal == shape.rlc_witness_on_metal
            && sample.rlc_witness_resident_only == shape.rlc_witness_resident_only
            && sample.rlc_rho_small_coefficients == shape.rlc_rho_small_coefficients
            && sample.dec_split_on_metal == shape.dec_split_on_metal
            && sample.dec_recomposition_on_metal == shape.dec_recomposition_on_metal
            && sample.dec_forms_on_metal == shape.dec_forms_on_metal
            && sample.dec_y_on_metal == shape.dec_y_on_metal
            && sample.dec_commit_on_metal == shape.dec_commit_on_metal
            && sample.resident_input_folds == shape.resident_input_folds
            && sample.resident_output_folds == shape.resident_output_folds
            && sample.deferred_proof_folds == shape.deferred_proof_folds
            && sample.deferred_running_folds == shape.deferred_running_folds
            && sample.recursive_compile_reverify_required == shape.recursive_compile_reverify_required
    }));
    NifsStageReport {
        folds_per_sample: shape.folds,
        total: TimingSummary::from_durations(samples.iter().map(|sample| sample.total).collect()),
        pi_ccs: TimingSummary::from_durations(samples.iter().map(|sample| sample.pi_ccs).collect()),
        ajtai_y_eval: TimingSummary::from_durations(samples.iter().map(|sample| sample.ajtai_y_eval).collect()),
        pi_rlc: TimingSummary::from_durations(samples.iter().map(|sample| sample.pi_rlc).collect()),
        pi_dec: TimingSummary::from_durations(samples.iter().map(|sample| sample.pi_dec).collect()),
        dec_form_build: TimingSummary::from_durations(samples.iter().map(|sample| sample.dec_form_build).collect()),
        dec_projection: TimingSummary::from_durations(samples.iter().map(|sample| sample.dec_projection).collect()),
        dec_host_materialization: TimingSummary::from_durations(
            samples
                .iter()
                .map(|sample| sample.dec_host_materialization)
                .collect(),
        ),
        fe_on_metal: shape.fe_on_metal,
        ajtai_y_eval_on_metal: shape.ajtai_y_eval_on_metal,
        nc_on_metal: shape.nc_on_metal,
        nc_mask_native_on_metal: shape.nc_mask_native_on_metal,
        rlc_witness_on_metal: shape.rlc_witness_on_metal,
        rlc_witness_resident_only: shape.rlc_witness_resident_only,
        rlc_rho_small_coefficients: shape.rlc_rho_small_coefficients,
        dec_split_on_metal: shape.dec_split_on_metal,
        dec_recomposition_on_metal: shape.dec_recomposition_on_metal,
        dec_forms_on_metal: shape.dec_forms_on_metal,
        dec_y_on_metal: shape.dec_y_on_metal,
        dec_commit_on_metal: shape.dec_commit_on_metal,
        resident_input_folds: shape.resident_input_folds,
        resident_output_folds: shape.resident_output_folds,
        deferred_proof_folds: shape.deferred_proof_folds,
        deferred_running_folds: shape.deferred_running_folds,
        recursive_compile_reverify_required: shape.recursive_compile_reverify_required,
        activity_per_sample: shape.activity.into(),
    }
}

fn add_activity(total: &mut MetalActivity, next: MetalActivity) {
    total.command_buffers += next.command_buffers;
    total.dispatches += next.dispatches;
    total.host_waits += next.host_waits;
    total.allocated_bytes += next.allocated_bytes;
    total.uploaded_bytes += next.uploaded_bytes;
    total.downloaded_bytes += next.downloaded_bytes;
    total.current_allocated_bytes = next.current_allocated_bytes;
}

#[derive(Debug, Error)]
pub enum BenchmarkError {
    #[error("invalid benchmark configuration: {0}")]
    Config(&'static str),
    #[error("Metal benchmark failed: {0}")]
    Metal(#[from] neo_prover_metal::MetalError),
    #[error("benchmark parity failure in {0}")]
    Parity(&'static str),
    #[error("benchmark lifecycle failure: {0}")]
    Lifecycle(String),
    #[error("benchmark JSON failure: {0}")]
    Json(#[from] serde_json::Error),
}
