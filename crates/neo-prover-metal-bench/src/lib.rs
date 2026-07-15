//! Same-device CPU and Metal benchmark harness for the iOS prover backend.
//!
//! This crate owns benchmark workloads, timing policy, JSON reporting, and
//! the narrow iOS C ABI. It owns no proof or transcript semantics.

mod ffi;
mod lifecycle;
mod nebula;
mod parity;
mod primitives;
mod report;
mod sha256;

pub use ffi::{neo_metal_benchmark_free_bytes, neo_metal_benchmark_run_json};
pub use lifecycle::run_lifecycle_benchmarks;
pub use primitives::run_primitive_benchmarks;
pub use report::{
    ActivityReport, BenchmarkConfig, BenchmarkError, BenchmarkReport, CandidateReport, DeviceReport,
    LifecycleCrossoverReport, LifecyclePipelineReport, LifecycleReport, NifsStageReport, PrimitiveReport,
    SustainedLifecycleReport, TimingSummary, REPORT_SCHEMA_VERSION,
};

use neo_prover_metal::MetalSession;

pub fn run_benchmark(config: BenchmarkConfig) -> Result<BenchmarkReport, BenchmarkError> {
    config.validate()?;
    let session = MetalSession::new()?;
    let device = session.device_info()?.into();
    let primitives = run_primitive_benchmarks(&session, &config)?;
    let lifecycle = run_lifecycle_benchmarks(&config)?;
    let m1_parity_passed = primitives.iter().all(|primitive| primitive.parity_ok);
    let m1_crossover_passed = primitives
        .iter()
        .filter(|primitive| primitive.crossover_required)
        .all(|primitive| primitive.crossover_gate_passed);
    let metal_lifecycle = lifecycle
        .iter()
        .filter(|report| report.backend == "MetalNifsProver")
        .collect::<Vec<_>>();
    let lifecycle_crossover = metal_lifecycle
        .iter()
        .filter_map(|metal| {
            let cpu = lifecycle
                .iter()
                .find(|cpu| cpu.backend == "CPU" && cpu.name == metal.name)?;
            let median_target = 1.52;
            let p95_target = 1.50;
            let median_speedup = cpu.online.median_ms / metal.online.median_ms;
            let p95_speedup = cpu.online.p95_ms / metal.online.p95_ms;
            let crossover_required = metal.name == "sha256_serial_4_chunk";
            Some(LifecycleCrossoverReport {
                name: metal.name.clone(),
                crossover_required,
                median_speedup_over_cpu: median_speedup,
                p95_speedup_over_cpu: p95_speedup,
                median_target,
                p95_target,
                proof_parity_ok: metal.proof_parity_ok,
                passed: metal.proof_parity_ok && median_speedup >= median_target && p95_speedup >= p95_target,
            })
        })
        .collect::<Vec<_>>();
    let m2_lifecycle_passed = !metal_lifecycle.is_empty()
        && metal_lifecycle
            .iter()
            .all(|report| report.semantic_result_ok);
    let m2_crossover_passed = !metal_lifecycle.is_empty()
        && metal_lifecycle.iter().all(|metal| {
            lifecycle
                .iter()
                .find(|cpu| cpu.backend == "CPU" && cpu.name == metal.name)
                .is_some_and(|cpu| metal.online.median_ms < cpu.online.median_ms)
        });
    let m3_residency_passed = !metal_lifecycle.is_empty()
        && metal_lifecycle.iter().all(|report| {
            report.nifs_profile.as_ref().is_some_and(|profile| {
                let resident_fold_count = profile.folds_per_sample.saturating_sub(1);
                profile.folds_per_sample > 0
                    && profile.resident_input_folds == resident_fold_count
                    && profile.resident_output_folds == resident_fold_count
                    && profile.activity_per_sample.host_waits == profile.activity_per_sample.command_buffers
            })
        });
    let m3_crossover_passed = m2_crossover_passed;
    let m4_projection_passed = m3_residency_passed
        && metal_lifecycle.iter().all(|report| {
            report.nifs_profile.as_ref().is_some_and(|profile| {
                profile.rlc_witness_resident_only
                    && profile.dec_recomposition_on_metal
                    && profile.dec_forms_on_metal
                    && profile.dec_y_on_metal
                    && profile.dec_commit_on_metal
            })
        });
    let m4_crossover_passed = m4_projection_passed && m2_crossover_passed;
    let m5_adapter_passed = m4_projection_passed
        && metal_lifecycle.iter().all(|report| {
            report.nifs_profile.as_ref().is_some_and(|profile| {
                let recursive_folds = profile.folds_per_sample.saturating_sub(1);
                profile.deferred_proof_folds == profile.folds_per_sample
                    && profile.deferred_running_folds == recursive_folds
                    && !profile.recursive_compile_reverify_required
            })
        });
    let m6_pipeline_passed = metal_lifecycle
        .iter()
        .find(|report| report.name == "sha256_serial_4_chunk")
        .is_some_and(|report| {
            report.proof_parity_ok
                && report.nifs_profile.as_ref().is_some_and(|profile| {
                    profile.ajtai_y_eval_on_metal
                        && profile.nc_on_metal
                        && profile.nc_mask_native_on_metal
                        && profile.rlc_rho_small_coefficients
                })
                && report.pipeline.as_ref().is_some_and(|pipeline| {
                    pipeline.synthesis_work.median_ms > 0.0
                        && pipeline.fold_work.median_ms > 0.0
                        && pipeline.final_materialization.median_ms > 0.0
                        && pipeline.overlap_saved.max_ms > 0.0
                })
        });
    let m6_crossover_passed = m6_pipeline_passed
        && lifecycle_crossover
            .iter()
            .find(|report| report.crossover_required)
            .is_some_and(|_| {
                lifecycle_crossover
                    .iter()
                    .filter(|report| report.crossover_required)
                    .all(|report| report.passed)
            });
    #[cfg(target_vendor = "apple")]
    let sustained = (config.lifecycle_soak_seconds > 0)
        .then(|| lifecycle::run_sha256_sustained(config.lifecycle_soak_seconds))
        .transpose()?;
    #[cfg(not(target_vendor = "apple"))]
    let sustained: Option<SustainedLifecycleReport> = None;
    let m6_sustained_passed = sustained.as_ref().is_some_and(|report| report.passed);
    let m6_passed = m6_pipeline_passed && m6_crossover_passed && m6_sustained_passed;
    let mut notes = Vec::new();
    if cfg!(target_abi = "sim") {
        notes.push("iOS simulator result: correctness only, never a performance oracle".to_owned());
    } else if !cfg!(target_os = "ios") {
        notes.push("macOS development result: physical-iPhone crossover remains unmeasured".to_owned());
    }
    if !m1_crossover_passed {
        notes.push("M1 crossover threshold is 2x CPU for bulk primitives; parity remains authoritative".to_owned());
    }
    if !metal_lifecycle.is_empty() && !m3_residency_passed {
        notes.push("M3 residency is incomplete in at least one lifecycle".to_owned());
    } else if !metal_lifecycle.is_empty() && !m3_crossover_passed {
        notes.push("M3 residency passes, but the selected hybrid lifecycle is not yet faster than CPU".to_owned());
    }
    if !metal_lifecycle.is_empty() && !m4_projection_passed {
        notes.push(
            "M4 Pi_DEC split validation, child openings, or child commitments are not fully routed through Metal"
                .to_owned(),
        );
    } else if !metal_lifecycle.is_empty() && !m4_crossover_passed {
        notes.push("M4 Pi_DEC projection passes, but the complete lifecycle is not yet faster than CPU".to_owned());
    }
    if !metal_lifecycle.is_empty() && !m5_adapter_passed {
        notes.push("M5 deferred proof/running ownership or recursive replay removal is incomplete".to_owned());
    }
    if !metal_lifecycle.is_empty() && !m6_pipeline_passed {
        notes.push(
            "M6 timed SHA synthesis/fold overlap or canonical CPU/Metal proof-authority parity is incomplete"
                .to_owned(),
        );
    } else if !metal_lifecycle.is_empty() && !m6_crossover_passed {
        notes
            .push("M6 pipeline passes, but median or p95 complete-lifecycle crossover remains below target".to_owned());
    }
    if config.lifecycle_soak_seconds == 0 {
        notes.push(
            "M6 sustained gate was not run; use the M6 benchmark profile for the 60-second measurement".to_owned(),
        );
    } else if !m6_sustained_passed {
        notes.push("M6 sustained SHA throughput remains below the 1.15x CPU target".to_owned());
    }
    Ok(BenchmarkReport {
        schema_version: REPORT_SCHEMA_VERSION,
        device,
        config,
        timing_contract: vec![
            "CPU and Metal run on the same device with identical inputs".to_owned(),
            "one warm-up per backend precedes repeated measured samples".to_owned(),
            "SHA lifecycle samples are CPU/Metal pairs whose first backend alternates by sample index".to_owned(),
            "timing summaries retain ordered raw millisecond samples as well as aggregate statistics".to_owned(),
            "transfer-inclusive Metal candidates time upload, command submission, completion wait, and result download"
                .to_owned(),
            "resident Metal candidates report static allocation and ingress as setup_ms, then time dispatch through final result download"
                .to_owned(),
            "lifecycle setup is reported separately from append plus terminal materialization".to_owned(),
            "verification is timed separately and must accept every generated audit".to_owned(),
            "M3 NIFS stage timings and activity are aggregated per complete lifecycle sample".to_owned(),
            "M4 reports Pi_DEC form construction, Metal child projection, and canonical host materialization separately"
                .to_owned(),
            "M5 reports deferred proof and running carriers and whether recursive compile replay remains required"
                .to_owned(),
            "M6 times chunk synthesis inside the lifecycle, overlaps the next chunk with the current fold, and includes terminal materialization"
                .to_owned(),
            "M6 requires Ajtai Y_eval and mask-native NC column rounds to execute on Metal for every fold"
                .to_owned(),
            "M6 crossover requires 1.52x median and 1.50x p95 SHA lifecycle speedup with canonical CPU/Metal proof-authority parity"
                .to_owned(),
            "M6 sustained mode runs each backend for 60 seconds and requires 1.15x Metal throughput".to_owned(),
        ],
        primitives,
        lifecycle,
        m1_parity_passed,
        m1_crossover_passed,
        m2_lifecycle_passed,
        m2_crossover_passed,
        m3_residency_passed,
        m3_crossover_passed,
        m4_projection_passed,
        m4_crossover_passed,
        m5_adapter_passed,
        lifecycle_crossover,
        sustained,
        m6_pipeline_passed,
        m6_crossover_passed,
        m6_sustained_passed,
        m6_passed,
        notes,
    })
}

pub fn run_benchmark_json(config: BenchmarkConfig) -> Result<String, BenchmarkError> {
    Ok(serde_json::to_string_pretty(&run_benchmark(config)?)?)
}
