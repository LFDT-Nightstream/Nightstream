//! Example integration showing how to use constraint diagnostics
//! 
//! This example demonstrates:
//! 1. Capturing a diagnostic on constraint failure
//! 2. Exporting it to different formats
//! 3. Loading and replaying it

#[cfg(feature = "prove-diagnostics")]
use neo_fold::diagnostic::{
    capture_diagnostic, export_diagnostic, load_diagnostic,
    ContextInfo, FoldingContext, DiagnosticFormat,
};
use neo_ccs::{Mat, r1cs_to_ccs};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
#[cfg(feature = "prove-diagnostics")]
fn test_diagnostic_capture_and_replay() {
    // Create a simple failing R1CS constraint
    // Constraint: z[0] * z[1] = z[2]
    // With z = [2, 3, 7], we get 2*3 = 6 ≠ 7, residual = -1
    
    let a = Mat::from_row_major(1, 3, vec![
        F::ONE, F::ZERO, F::ZERO,
    ]);
    let b = Mat::from_row_major(1, 3, vec![
        F::ZERO, F::ONE, F::ZERO,
    ]);
    let c = Mat::from_row_major(1, 3, vec![
        F::ZERO, F::ZERO, F::ONE,
    ]);
    
    let ccs = r1cs_to_ccs(a, b, c);
    
    let witness = vec![
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(7),
    ];
    
    // Capture diagnostic (auto-computes residual)
    let diagnostic = capture_diagnostic(
        &ccs,
        0,  // row_index
        &witness,
        ContextInfo {
            step_idx: 0,
            instruction: "Test".to_string(),
            test: "test_diagnostic_example".to_string(),
            failure_reason: Some("Deliberate test failure: 2*3 ≠ 7".to_string()),
        },
        FoldingContext {
            phase: "Test".to_string(),
            b: 2,
            k: 12,
            norm_bound: 4096,
            expansion_factor: 2,
            challenge_set: "test@v1".to_string(),
        },
        None,  // transcript_challenges
    );
    
    // Verify diagnostic structure
    assert_eq!(diagnostic.structure.row_index, 0);
    assert_eq!(diagnostic.structure.t, 3);  // R1CS has 3 matrices
    assert_eq!(diagnostic.witness.size, 3);
    
    // Verify gradient blame was computed
    assert!(!diagnostic.witness.gradient_blame.is_empty());
    
    // Export to examples directory
    let examples_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("neo-diag/examples");
    std::fs::create_dir_all(&examples_dir).unwrap();
    
    std::env::set_var("NEO_DIAGNOSTICS", examples_dir.to_str().unwrap());
    std::env::set_var("NEO_DIAGNOSTIC_FORMAT", "json");
    
    let path = export_diagnostic(&diagnostic, DiagnosticFormat::Json).unwrap();
    println!("✅ Diagnostic exported to: {}", path.display());
    
    // Rename to a friendly example name
    let example_path = examples_dir.join("simple_r1cs_failure.json");
    std::fs::rename(&path, &example_path).ok();
    println!("✅ Example saved as: {}", example_path.display());
    
    // Load it back
    let example_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("neo-diag/examples/simple_r1cs_failure.json");
    let loaded = load_diagnostic(&example_path).unwrap();
    
    // Verify it matches
    assert_eq!(loaded.structure.hash, diagnostic.structure.hash);
    assert_eq!(loaded.eval.delta, diagnostic.eval.delta);
    
    println!("✅ Diagnostic loaded successfully");
    println!("   Expected: {}", loaded.eval.expected);
    println!("   Actual:   {}", loaded.eval.actual);
    println!("   Delta:    {} (signed: {})", loaded.eval.delta, loaded.eval.delta_signed);
    
    // Note: We keep the example file for documentation purposes
    // It's tracked in git as an example for neo-diag
}
