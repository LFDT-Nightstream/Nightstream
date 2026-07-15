#[cfg(target_vendor = "apple")]
use neo_prover_metal_bench::run_benchmark;
use neo_prover_metal_bench::{run_lifecycle_benchmarks, BenchmarkConfig, TimingSummary};

use std::time::Duration;

#[test]
fn timing_summary_retains_measurement_order() {
    let summary = TimingSummary::from_durations(vec![
        Duration::from_millis(3),
        Duration::from_millis(1),
        Duration::from_millis(2),
    ]);
    assert_eq!(summary.raw_ms, vec![3.0, 1.0, 2.0]);
    assert_eq!(summary.median_ms, 2.0);
    assert_eq!(summary.min_ms, 1.0);
    assert_eq!(summary.max_ms, 3.0);
}

#[test]
fn benchmark_config_rejects_unsafe_dimensions() {
    let mut config = BenchmarkConfig::smoke();
    config.field_elements = usize::MAX;
    assert!(config.validate().is_err());

    let mut config = BenchmarkConfig::smoke();
    config.poseidon_hashes = usize::MAX;
    config.poseidon_fields_per_hash = usize::MAX;
    assert!(config.validate().is_err());

    let mut config = BenchmarkConfig::smoke();
    config.poseidon_hashes = 1 << 20;
    config.poseidon_fields_per_hash = 1 << 16;
    assert!(config.validate().is_err());

    let mut config = BenchmarkConfig::smoke();
    config.lifecycle_repetitions = 0;
    config.run_nebula_lifecycle = true;
    assert!(config.validate().is_err());

    let mut config = BenchmarkConfig::smoke();
    config.lifecycle_soak_seconds = 61;
    assert!(config.validate().is_err());
}

#[test]
fn nebula_lifecycle_baseline_proves_and_verifies() {
    let mut config = BenchmarkConfig::smoke();
    config.run_nebula_lifecycle = true;
    let reports = run_lifecycle_benchmarks(&config).expect("Nebula lifecycle benchmark");
    let expected = if cfg!(target_vendor = "apple") { 2 } else { 1 };
    assert_eq!(reports.len(), expected);
    assert!(reports
        .iter()
        .all(|report| report.name == "nebula_memory_lane_2_step"));
    assert!(reports.iter().all(|report| report.semantic_result_ok));
    assert!(reports.iter().all(|report| report.proof_parity_ok));
    assert!(reports.iter().all(|report| report.pipeline.is_none()));
    assert!(reports.iter().all(|report| report.online.samples == 1));
    assert!(reports.iter().all(|report| report.verify_ms.samples == 1));
    #[cfg(target_vendor = "apple")]
    {
        let profile = reports
            .iter()
            .find(|report| report.backend == "MetalNifsProver")
            .and_then(|report| report.nifs_profile.as_ref())
            .expect("Metal Nebula NIFS profile");
        assert!(profile.folds_per_sample > 0);
        assert_eq!(profile.resident_input_folds, profile.folds_per_sample - 1);
        assert_eq!(profile.resident_output_folds, profile.folds_per_sample);
        assert!(profile.fe_on_metal);
        assert!(profile.ajtai_y_eval_on_metal);
        assert!(profile.nc_on_metal);
        assert!(profile.nc_mask_native_on_metal);
        assert!(profile.rlc_witness_on_metal);
        assert!(profile.rlc_witness_resident_only);
        assert!(profile.rlc_rho_small_coefficients);
        assert!(profile.dec_split_on_metal);
        assert!(profile.dec_recomposition_on_metal);
        assert!(profile.dec_forms_on_metal);
        assert!(profile.dec_y_on_metal);
        assert!(profile.dec_commit_on_metal);
        assert_eq!(profile.deferred_proof_folds, profile.folds_per_sample);
        assert_eq!(profile.deferred_running_folds, profile.folds_per_sample);
        assert!(!profile.recursive_compile_reverify_required);
        assert!(profile.activity_per_sample.host_waits <= profile.activity_per_sample.command_buffers);
    }
}

#[test]
#[ignore = "M0 production-core SHA lifecycle performance snapshot"]
fn sha256_lifecycle_baseline_proves_and_verifies() {
    let mut config = BenchmarkConfig::smoke();
    config.run_sha256_lifecycle = true;
    let reports = run_lifecycle_benchmarks(&config).expect("SHA lifecycle benchmark");
    let expected = if cfg!(target_vendor = "apple") { 2 } else { 1 };
    assert_eq!(reports.len(), expected);
    assert!(reports
        .iter()
        .all(|report| report.name == "sha256_serial_4_chunk"));
    assert!(reports.iter().all(|report| report.semantic_result_ok));
    assert!(reports.iter().all(|report| report.proof_parity_ok));
    assert!(reports.iter().all(|report| report.pipeline.is_some()));
    #[cfg(target_vendor = "apple")]
    {
        let cpu = reports
            .iter()
            .find(|report| report.backend == "CPU")
            .expect("CPU SHA lifecycle");
        let metal_report = reports
            .iter()
            .find(|report| report.backend == "MetalNifsProver")
            .expect("Metal SHA lifecycle");
        eprintln!(
            "M6 SHA lifecycle: CPU median={:.3}ms p95={:.3}ms; Metal median={:.3}ms p95={:.3}ms; speedup median={:.3}x p95={:.3}x",
            cpu.online.median_ms,
            cpu.online.p95_ms,
            metal_report.online.median_ms,
            metal_report.online.p95_ms,
            cpu.online.median_ms / metal_report.online.median_ms,
            cpu.online.p95_ms / metal_report.online.p95_ms,
        );
        let profile = reports
            .iter()
            .find(|report| report.backend == "MetalNifsProver")
            .and_then(|report| report.nifs_profile.as_ref())
            .expect("Metal SHA NIFS profile");
        eprintln!(
            "Metal NIFS medians: total={:.3}ms Pi_CCS={:.3}ms Y_eval={:.3}ms Pi_RLC={:.3}ms Pi_DEC={:.3}ms DEC forms={:.3}ms projection={:.3}ms host={:.3}ms",
            profile.total.median_ms,
            profile.pi_ccs.median_ms,
            profile.ajtai_y_eval.median_ms,
            profile.pi_rlc.median_ms,
            profile.pi_dec.median_ms,
            profile.dec_form_build.median_ms,
            profile.dec_projection.median_ms,
            profile.dec_host_materialization.median_ms,
        );
        let activity = &profile.activity_per_sample;
        eprintln!(
            "Metal activity/sample: command_buffers={} dispatches={} host_waits={} allocated={:.2}MiB uploaded={:.2}MiB downloaded={:.2}MiB current={:.2}MiB",
            activity.command_buffers,
            activity.dispatches,
            activity.host_waits,
            activity.allocated_bytes as f64 / (1024.0 * 1024.0),
            activity.uploaded_bytes as f64 / (1024.0 * 1024.0),
            activity.downloaded_bytes as f64 / (1024.0 * 1024.0),
            activity.current_allocated_bytes as f64 / (1024.0 * 1024.0),
        );
        assert_eq!(profile.resident_input_folds, profile.folds_per_sample - 1);
        assert_eq!(profile.resident_output_folds, profile.folds_per_sample);
        assert!(profile.rlc_witness_resident_only);
        assert!(profile.rlc_rho_small_coefficients);
        assert!(profile.ajtai_y_eval_on_metal);
        assert!(profile.nc_on_metal);
        assert!(profile.nc_mask_native_on_metal);
        assert!(profile.dec_forms_on_metal);
        assert!(profile.dec_y_on_metal);
        assert!(profile.dec_commit_on_metal);
        assert_eq!(profile.deferred_proof_folds, profile.folds_per_sample);
        assert_eq!(profile.deferred_running_folds, profile.folds_per_sample);
        assert!(!profile.recursive_compile_reverify_required);
    }
}

#[test]
#[cfg(target_vendor = "apple")]
#[ignore = "M6 runs five timed lifecycles plus 60 seconds per backend"]
fn m6_pipeline_crossover_and_sustained_gates_pass() {
    let report = run_benchmark(BenchmarkConfig::m6()).expect("M6 benchmark");
    assert!(
        report.m6_pipeline_passed,
        "timed synthesis/fold overlap or parity failed"
    );
    assert!(report.m6_crossover_passed, "median or p95 crossover target failed");
    assert!(report.m6_sustained_passed, "60-second sustained target failed");
    assert!(report.m6_passed);
}
