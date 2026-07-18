#![cfg(target_vendor = "apple")]

use neo_prover_metal::MetalSession;
use neo_prover_metal_bench::{run_primitive_benchmarks, BenchmarkConfig};

#[test]
fn primitive_report_is_complete_and_parity_checked() {
    let session = MetalSession::new().expect("Metal session");
    let reports = run_primitive_benchmarks(&session, &BenchmarkConfig::smoke()).expect("primitive report");
    assert_eq!(reports.len(), 5);
    assert!(reports.iter().all(|report| report.parity_ok));
    assert_eq!(
        reports
            .iter()
            .filter(|report| report.crossover_required)
            .count(),
        4
    );
    assert!(
        !reports
            .iter()
            .find(|report| report.name == "goldilocks_ops")
            .expect("Goldilocks diagnostic")
            .crossover_required
    );
    assert!(reports.iter().all(|report| !report.candidates.is_empty()));
    assert!(reports.iter().all(|report| report
        .candidates
        .iter()
        .all(|candidate| candidate.activity.is_some())));
}
