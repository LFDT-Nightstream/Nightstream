use std::ptr;

use neo_prover_metal_bench::{neo_metal_benchmark_free_bytes, neo_metal_benchmark_run_json};

#[test]
fn ffi_rejects_malformed_configuration_without_returning_output() {
    let malformed = br#"{"samples": "not-a-number"}"#;
    let mut output = ptr::null_mut();
    let mut output_len = 0;
    let mut error = ptr::null_mut();
    let mut error_len = 0;
    let status = unsafe {
        neo_metal_benchmark_run_json(
            malformed.as_ptr(),
            malformed.len(),
            &mut output,
            &mut output_len,
            &mut error,
            &mut error_len,
        )
    };
    assert_eq!(status, 1);
    assert!(output.is_null());
    assert_eq!(output_len, 0);
    assert!(!error.is_null());
    let error_json = unsafe { std::slice::from_raw_parts(error, error_len) };
    let value: serde_json::Value = serde_json::from_slice(error_json).expect("error JSON");
    assert!(value["error"]
        .as_str()
        .is_some_and(|message| message.contains("invalid benchmark configuration")));
    unsafe { neo_metal_benchmark_free_bytes(error, error_len) };
}

#[test]
fn ffi_rejects_invalid_output_pointers() {
    let mut output_len = 0;
    let mut error = ptr::null_mut();
    let mut error_len = 0;
    let status = unsafe {
        neo_metal_benchmark_run_json(
            ptr::null(),
            0,
            ptr::null_mut(),
            &mut output_len,
            &mut error,
            &mut error_len,
        )
    };
    assert_eq!(status, -1);
}

#[test]
fn ffi_returns_a_parity_checked_smoke_report() {
    let configuration =
        serde_json::to_vec(&neo_prover_metal_bench::BenchmarkConfig::smoke()).expect("serialize smoke configuration");
    let mut output = ptr::null_mut();
    let mut output_len = 0;
    let mut error = ptr::null_mut();
    let mut error_len = 0;
    let status = unsafe {
        neo_metal_benchmark_run_json(
            configuration.as_ptr(),
            configuration.len(),
            &mut output,
            &mut output_len,
            &mut error,
            &mut error_len,
        )
    };
    assert_eq!(status, 0);
    assert!(error.is_null());
    assert_eq!(error_len, 0);
    assert!(!output.is_null());
    let report = unsafe { std::slice::from_raw_parts(output, output_len) };
    let value: serde_json::Value = serde_json::from_slice(report).expect("benchmark JSON");
    assert_eq!(value["schema_version"], 10);
    assert_eq!(value["m1_parity_passed"], true);
    assert_eq!(value["m2_lifecycle_passed"], false);
    assert_eq!(value["m3_residency_passed"], false);
    assert_eq!(value["m4_projection_passed"], false);
    assert_eq!(value["m4_crossover_passed"], false);
    assert_eq!(value["m5_adapter_passed"], false);
    assert_eq!(value["m6_pipeline_passed"], false);
    assert_eq!(value["m6_crossover_passed"], false);
    assert_eq!(value["m6_sustained_passed"], false);
    assert_eq!(value["m6_passed"], false);
    assert_eq!(value["primitives"].as_array().map(Vec::len), Some(5));
    unsafe { neo_metal_benchmark_free_bytes(output, output_len) };
}
