//! Tests for automatic parameter tuning based on circuit characteristics

use neo_params::NeoParams;

#[test]
fn compute_circuit_params_basic() {
    // Circuit with 100 constraints
    let (ell, d_sc) = NeoParams::compute_circuit_params(100, 3);
    assert_eq!(ell, 7); // log2(128)
    assert_eq!(d_sc, 3);
    
    // Circuit with 1024 constraints
    let (ell, d_sc) = NeoParams::compute_circuit_params(1024, 2);
    assert_eq!(ell, 10); // log2(1024)
    assert_eq!(d_sc, 2);
}

#[test]
fn goldilocks_for_circuit_works() {
    // Auto-tune for a circuit with ~100 constraints
    let (ell, d_sc) = NeoParams::compute_circuit_params(100, 3);
    let params = NeoParams::goldilocks_for_circuit(ell, d_sc, 2);
    
    // Should support the circuit
    let result = params.extension_check(ell, d_sc);
    assert!(result.is_ok());
    
    let summary = result.unwrap();
    assert!(summary.slack_bits >= 0, "Should have positive slack");
    
    println!("✅ Auto-tuned for 100 constraints: lambda={}, slack={}", 
             params.lambda, summary.slack_bits);
}

#[test]
fn automatic_workflow_example() {
    // Simulate what Session would do: detect circuit, auto-tune params
    let circuit_size = 500; // constraints
    let max_degree = 3;
    
    // Step 1: Compute circuit params
    let (ell, d_sc) = NeoParams::compute_circuit_params(circuit_size, max_degree);
    
    // Step 2: Auto-tune parameters
    let params = NeoParams::goldilocks_for_circuit(ell, d_sc, 2);
    
    // Step 3: Verify they work
    let check = params.extension_check(ell, d_sc);
    assert!(check.is_ok());
    
    println!("Circuit: {} constraints → ell={}, d_sc={}", 
             circuit_size, ell, d_sc);
    println!("Auto-tuned: lambda={} bits", params.lambda);
}

#[test]
fn small_circuits_still_work() {
    // The default should still work for small circuits
    let params = NeoParams::goldilocks_small_circuits();
    
    let result = params.extension_check(4, 3);
    assert!(result.is_ok());
}

