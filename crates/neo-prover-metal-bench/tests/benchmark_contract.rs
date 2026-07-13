#[cfg(target_vendor = "apple")]
use neo_prover_metal_bench::run_benchmark;
use neo_prover_metal_bench::{run_lifecycle_benchmarks, BenchmarkConfig};

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
        assert_eq!(profile.resident_output_folds, profile.folds_per_sample - 1);
        assert!(!profile.fe_on_metal);
        assert!(!profile.nc_on_metal);
        assert!(profile.rlc_witness_on_metal);
        assert!(profile.rlc_witness_resident_only);
        assert!(profile.rlc_rho_small_coefficients);
        assert!(profile.dec_split_on_metal);
        assert!(profile.dec_recomposition_on_metal);
        assert!(profile.dec_forms_on_metal);
        assert!(profile.dec_y_on_metal);
        assert!(profile.dec_commit_on_metal);
        assert_eq!(profile.deferred_proof_folds, profile.folds_per_sample);
        assert_eq!(profile.deferred_running_folds, profile.folds_per_sample - 1);
        assert!(!profile.recursive_compile_reverify_required);
        assert_eq!(
            profile.activity_per_sample.host_waits,
            profile.activity_per_sample.command_buffers
        );
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
        assert_eq!(profile.resident_input_folds, profile.folds_per_sample - 1);
        assert_eq!(profile.resident_output_folds, profile.folds_per_sample - 1);
        assert!(profile.rlc_witness_resident_only);
        assert!(profile.rlc_rho_small_coefficients);
        assert!(profile.dec_forms_on_metal);
        assert!(profile.dec_y_on_metal);
        assert!(profile.dec_commit_on_metal);
        assert_eq!(profile.deferred_proof_folds, profile.folds_per_sample);
        assert_eq!(profile.deferred_running_folds, profile.folds_per_sample - 1);
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
