//! Integration test for Starstream TX IVC proof generation
//!
//! This test reads the exported test data from `test_starstream_tx_export.json`
//! and attempts to generate an IVC proof using the Neo proving system.
//!
//! According to the export metadata, this test SHOULD FAIL at Step 2 due to a
//! constraint violation (YieldResume expects current_program=300 but it's actually 400).

use std::fs;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use neo::{F, NeoParams};
use neo::ivc::{Accumulator, StepBindingSpec, IvcChainStepInput, prove_ivc_chain, verify_ivc_chain};
use neo::{NivcProgram, NivcStepSpec, NivcState, verify_nivc_chain};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs, CcsStructure};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Top-level structure matching the JSON export format
#[derive(Debug, Deserialize, Serialize)]
struct TestExport {
    metadata: Metadata,
    ivc_params: IvcParams,
    steps: Vec<StepData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_snark: Option<FinalSnark>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Metadata {
    test_name: String,
    field: String,
    modulus: String,
    num_steps: usize,
    should_fail: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    failure_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct IvcParams {
    y0: Vec<String>,
    step_spec: StepSpec,
}

#[derive(Debug, Deserialize, Serialize)]
struct StepSpec {
    y_len: usize,
    const1_index: usize,
    y_step_indices: Vec<usize>,
    y_prev_indices: Vec<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct StepData {
    step_idx: usize,
    instruction: String,
    witness: WitnessData,
    r1cs: R1csData,
}

#[derive(Debug, Deserialize, Serialize)]
struct WitnessData {
    instance: Vec<String>,
    witness: Vec<String>,
    z_full: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct R1csData {
    num_constraints: usize,
    num_variables: usize,
    num_public_inputs: usize,
    a_sparse: Vec<(usize, usize, String)>,
    b_sparse: Vec<(usize, usize, String)>,
    c_sparse: Vec<(usize, usize, String)>,
}

#[derive(Debug, Deserialize, Serialize)]
struct FinalSnark {
    witness: Vec<String>,
    public_input: Vec<String>,
    num_constraints: usize,
}

/// Parse a field element string (either decimal or hex)
fn parse_field_element(s: &str) -> F {
    if let Some(hex) = s.strip_prefix("0x") {
        F::from_u64(u64::from_str_radix(hex, 16).expect("valid hex"))
    } else {
        F::from_u64(s.parse::<u64>().expect("valid decimal"))
    }
}

/// Convert sparse matrix format to dense Mat
fn sparse_to_dense_mat(
    sparse: &[(usize, usize, String)],
    rows: usize,
    cols: usize,
) -> Mat<F> {
    let mut data = vec![F::ZERO; rows * cols];
    
    for (row, col, val_str) in sparse {
        let val = parse_field_element(val_str);
        data[row * cols + col] = val;
    }
    
    Mat::from_row_major(rows, cols, data)
}

/// Build CCS from R1CS data in the export
fn build_step_ccs(r1cs: &R1csData) -> CcsStructure<F> {
    let rows = r1cs.num_constraints;
    let cols = r1cs.num_variables;
    
    let a = sparse_to_dense_mat(&r1cs.a_sparse, rows, cols);
    let b = sparse_to_dense_mat(&r1cs.b_sparse, rows, cols);
    let c = sparse_to_dense_mat(&r1cs.c_sparse, rows, cols);
    
    r1cs_to_ccs(a, b, c)
}

/// Extract witness vector from step data
fn extract_witness(witness_data: &WitnessData) -> Vec<F> {
    witness_data.z_full
        .iter()
        .map(|s| parse_field_element(s))
        .collect()
}

/// Manually verify R1CS constraint satisfaction for debugging
fn verify_r1cs_constraints(step: &StepData) -> Result<(), String> {
    let z = extract_witness(&step.witness);
    let r1cs = &step.r1cs;
    
    // Compute A*z, B*z, C*z
    let mut az = vec![F::ZERO; r1cs.num_constraints];
    let mut bz = vec![F::ZERO; r1cs.num_constraints];
    let mut cz = vec![F::ZERO; r1cs.num_constraints];
    
    for (row, col, val_str) in &r1cs.a_sparse {
        let val = parse_field_element(val_str);
        az[*row] += val * z[*col];
    }
    
    for (row, col, val_str) in &r1cs.b_sparse {
        let val = parse_field_element(val_str);
        bz[*row] += val * z[*col];
    }
    
    for (row, col, val_str) in &r1cs.c_sparse {
        let val = parse_field_element(val_str);
        cz[*row] += val * z[*col];
    }
    
    // Check: A*z ∘ B*z = C*z
    let mut violations = Vec::new();
    for i in 0..r1cs.num_constraints {
        let lhs = az[i] * bz[i];
        let rhs = cz[i];
        if lhs != rhs {
            violations.push(format!(
                "Constraint {} violated: ({}) * ({}) = {} ≠ {} (instruction: {})",
                i,
                az[i].as_canonical_u64(),
                bz[i].as_canonical_u64(),
                lhs.as_canonical_u64(),
                rhs.as_canonical_u64(),
                step.instruction
            ));
        }
    }
    
    if !violations.is_empty() {
        return Err(format!(
            "Step {} has {} constraint violations:\n{}",
            step.step_idx,
            violations.len(),
            violations.join("\n")
        ));
    }
    
    Ok(())
}

#[test]
fn test_starstream_tx_ivc_proof() {
    println!("🧪 Testing Starstream TX IVC Proof Generation");
    println!("{}", "=".repeat(60));
    
    // Load the JSON export
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("tests/test_starstream_tx_export.json");
    
    let json_content = fs::read_to_string(&json_path)
        .expect("Failed to read test_starstream_tx_export.json");
    
    let export: TestExport = serde_json::from_str(&json_content)
        .expect("Failed to parse JSON export");
    
    println!("📋 Test Metadata:");
    println!("   Name: {}", export.metadata.test_name);
    println!("   Field: {} (modulus: {})", export.metadata.field, export.metadata.modulus);
    println!("   Steps: {}", export.metadata.num_steps);
    println!("   Should Fail: {}", export.metadata.should_fail);
    if let Some(reason) = &export.metadata.failure_reason {
        println!("   Failure Reason: {}", reason);
    }
    println!();
    
    // Parse initial state y0
    let y0: Vec<F> = export.ivc_params.y0
        .iter()
        .map(|s| parse_field_element(s))
        .collect();
    
    println!("🎯 Initial State y0: {:?}", 
        y0.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    println!();
    
    // Setup Neo parameters
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    // Build binding spec from step_spec
    let binding_spec = StepBindingSpec {
        y_step_offsets: export.ivc_params.step_spec.y_step_indices.clone(),
        step_program_input_witness_indices: vec![],  // No explicit public inputs binding
        y_prev_witness_indices: export.ivc_params.step_spec.y_prev_indices.clone(),
        const1_witness_index: export.ivc_params.step_spec.const1_index,
    };
    
    println!("📊 Binding Spec:");
    println!("   y_step_offsets: {:?}", binding_spec.y_step_offsets);
    println!("   y_prev_indices: {:?}", binding_spec.y_prev_witness_indices);
    println!("   const1_index: {}", binding_spec.const1_witness_index);
    println!();
    
    // Build step CCS - use the first step's R1CS structure
    // (All steps should have the same structure, just different witnesses)
    let step_ccs = build_step_ccs(&export.steps[0].r1cs);
    
    println!("🔧 Step CCS Structure:");
    println!("   Constraints (n): {}", step_ccs.n);
    println!("   Variables (m): {}", step_ccs.m);
    println!("   Matrices (t): {}", step_ccs.t());
    println!();
    
    // Initial accumulator
    let initial_accumulator = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: y0.clone(),
        step: 0,
    };
    
    // Prepare step inputs
    let step_inputs: Vec<IvcChainStepInput> = export.steps
        .iter()
        .map(|step| {
            let witness = extract_witness(&step.witness);
            IvcChainStepInput {
                witness,
                public_input: None,  // No explicit public inputs in this test
            }
        })
        .collect();
    
    println!("🚀 Attempting IVC Proof Generation...");
    println!("   This will run {} IVC steps", export.steps.len());
    if export.metadata.should_fail {
        println!("   ⚠️  Test metadata indicates this should fail: {}", 
            export.metadata.failure_reason.as_deref().unwrap_or("unknown reason"));
    }
    println!();
    
    // Generate proof (NO pre-flight checks - let the prover catch any issues!)
    let proof_result = prove_ivc_chain(
        &params,
        &step_ccs,
        &step_inputs,
        initial_accumulator.clone(),
        &binding_spec,
    );
    
    match proof_result {
        Ok(chain_proof) => {
            println!("✅ IVC Proof Generation SUCCEEDED");
            println!("   Chain Length: {}", chain_proof.chain_length);
            println!("   Final Accumulator Step: {}", chain_proof.final_accumulator.step);
            println!("   Final State: {:?}", 
                chain_proof.final_accumulator.y_compact
                    .iter()
                    .map(|f| f.as_canonical_u64())
                    .collect::<Vec<_>>());
            println!();
            
            // If should_fail is true but proof succeeded, this is a soundness bug!
            if export.metadata.should_fail {
                println!("🚨 SOUNDNESS BUG DETECTED! 🚨");
                println!("   Test metadata indicates this should fail, but proof generation succeeded.");
                println!("   This means the prover accepted invalid constraints!");
                if let Some(reason) = &export.metadata.failure_reason {
                    println!("   Expected failure: {}", reason);
                }
                panic!("Soundness violation: proof succeeded when it should have failed");
            }
            
            // Verify the proof
            println!("🔍 Verifying IVC Chain Proof...");
            let verify_result = verify_ivc_chain(
                &step_ccs,
                &chain_proof,
                &initial_accumulator,
                &binding_spec,
                &params,
            );
            
            match verify_result {
                Ok(true) => {
                    println!("✅ IVC Verification SUCCEEDED");
                    println!("   The proof is valid and verifies correctly.");
                }
                Ok(false) => {
                    println!("❌ IVC Verification FAILED");
                    println!("   The proof was generated but verification returned false.");
                    panic!("Verification failed for generated proof");
                }
                Err(e) => {
                    println!("❌ IVC Verification ERROR: {}", e);
                    panic!("Verification error: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ IVC Proof Generation FAILED");
            println!("   Error: {}", e);
            println!();
            
            // If should_fail is true, this is the expected outcome
            if export.metadata.should_fail {
                println!("✅ Expected failure occurred (should_fail=true)");
                println!("   The prover correctly rejected the invalid constraints.");
                if let Some(reason) = &export.metadata.failure_reason {
                    println!("   Expected reason: {}", reason);
                }
                println!("   Actual error: {}", e);
                
                // This is actually a success - the system is sound!
                println!();
                println!("🎉 Test PASSED: Prover correctly rejected invalid proof");
                return;
            } else {
                // If should_fail is false but proof failed, that's unexpected
                println!("⚠️  Unexpected failure (should_fail=false)");
                panic!("Proof generation failed unexpectedly: {}", e);
            }
        }
    }
    
    println!();
    println!("{}", "=".repeat(60));
    println!("🎉 Test Completed Successfully");
}

#[test]
fn test_json_parsing_only() {
    println!("🧪 Testing JSON Parsing");
    
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("tests/test_starstream_tx_export.json");
    
    let json_content = fs::read_to_string(&json_path)
        .expect("Failed to read test_starstream_tx_export.json");
    
    let export: TestExport = serde_json::from_str(&json_content)
        .expect("Failed to parse JSON export");
    
    println!("✅ Successfully parsed JSON");
    println!("   Test: {}", export.metadata.test_name);
    println!("   Steps: {}", export.steps.len());
    println!("   y0 length: {}", export.ivc_params.y0.len());
    
    for (i, step) in export.steps.iter().enumerate() {
        println!("   Step {}: {} constraints, {} variables",
            i, step.r1cs.num_constraints, step.r1cs.num_variables);
    }
}

#[test]
fn test_r1cs_constraint_verification() {
    println!("🧪 Testing R1CS Constraint Verification");
    println!("{}", "=".repeat(60));
    
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("tests/test_starstream_tx_export.json");
    
    let json_content = fs::read_to_string(&json_path)
        .expect("Failed to read test_starstream_tx_export.json");
    
    let export: TestExport = serde_json::from_str(&json_content)
        .expect("Failed to parse JSON export");
    
    println!("📋 Test: {}", export.metadata.test_name);
    println!("   Should Fail: {}", export.metadata.should_fail);
    if let Some(reason) = &export.metadata.failure_reason {
        println!("   Expected Failure: {}", reason);
    }
    println!();
    
    println!("🔍 Verifying R1CS constraints for each step:");
    println!();
    
    let mut found_violation = false;
    
    for step in &export.steps {
        println!("Step {}: {}", step.step_idx, step.instruction);
        println!("  {} constraints × {} variables", 
            step.r1cs.num_constraints, step.r1cs.num_variables);
        
        match verify_r1cs_constraints(step) {
            Ok(()) => {
                println!("  ✅ All constraints satisfied");
            }
            Err(e) => {
                println!("  ❌ CONSTRAINT VIOLATION DETECTED");
                println!("  {}", e.replace('\n', "\n  "));
                found_violation = true;
            }
        }
        println!();
    }
    
    // Assert test expectation matches reality
    if export.metadata.should_fail {
        assert!(found_violation, 
            "Test marked as should_fail=true but no constraint violations found");
        println!("✅ Test PASSED: Constraint violation detected as expected");
    } else {
        assert!(!found_violation, 
            "Test marked as should_fail=false but constraint violations found");
        println!("✅ Test PASSED: All constraints satisfied as expected");
    }
    
    println!();
    println!("{}", "=".repeat(60));
}

#[test]
fn test_starstream_tx_nivc_proof() {
    println!("🧪 Testing Starstream TX NIVC Proof Generation");
    println!("{}", "=".repeat(60));
    
    // Load the JSON export
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("tests/test_starstream_tx_export.json");
    
    let json_content = fs::read_to_string(&json_path)
        .expect("Failed to read test_starstream_tx_export.json");
    
    let export: TestExport = serde_json::from_str(&json_content)
        .expect("Failed to parse JSON export");
    
    println!("📋 Test Metadata:");
    println!("   Name: {}", export.metadata.test_name);
    println!("   Field: {} (modulus: {})", export.metadata.field, export.metadata.modulus);
    println!("   Steps: {}", export.metadata.num_steps);
    println!("   Should Fail: {}", export.metadata.should_fail);
    if let Some(reason) = &export.metadata.failure_reason {
        println!("   Failure Reason: {}", reason);
    }
    println!();
    
    // Parse initial state y0
    let y0: Vec<F> = export.ivc_params.y0
        .iter()
        .map(|s| parse_field_element(s))
        .collect();
    
    println!("🎯 Initial State y0: {:?}", 
        y0.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    println!();
    
    // Setup Neo parameters
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    // Build binding spec from step_spec
    let binding_spec = StepBindingSpec {
        y_step_offsets: export.ivc_params.step_spec.y_step_indices.clone(),
        step_program_input_witness_indices: vec![],  // No explicit public inputs binding
        y_prev_witness_indices: export.ivc_params.step_spec.y_prev_indices.clone(),
        const1_witness_index: export.ivc_params.step_spec.const1_index,
    };
    
    println!("📊 Binding Spec:");
    println!("   y_step_offsets: {:?}", binding_spec.y_step_offsets);
    println!("   y_prev_indices: {:?}", binding_spec.y_prev_witness_indices);
    println!("   const1_index: {}", binding_spec.const1_witness_index);
    println!();
    
    // Build step CCS - use the first step's R1CS structure
    let step_ccs = build_step_ccs(&export.steps[0].r1cs);
    
    println!("🔧 Step CCS Structure:");
    println!("   Constraints (n): {}", step_ccs.n);
    println!("   Variables (m): {}", step_ccs.m);
    println!("   Matrices (t): {}", step_ccs.t());
    println!();
    
    // Create NIVC program with a single step type
    // (In a more sophisticated version, we could have different step types for Nop, Resume, YieldResume)
    let program = NivcProgram::new(vec![
        NivcStepSpec {
            ccs: step_ccs.clone(),
            binding: binding_spec.clone(),
        },
    ]);
    
    println!("🔧 NIVC Program:");
    println!("   Number of step types (lanes): {}", program.len());
    println!();
    
    // Create NIVC state
    let mut nivc_state = match NivcState::new(params.clone(), program.clone(), y0.clone()) {
        Ok(state) => state,
        Err(e) => {
            panic!("Failed to create NIVC state: {}", e);
        }
    };
    
    println!("🚀 Attempting NIVC Proof Generation...");
    println!("   This will run {} NIVC steps", export.steps.len());
    if export.metadata.should_fail {
        println!("   ⚠️  Test metadata indicates this should fail: {}", 
            export.metadata.failure_reason.as_deref().unwrap_or("unknown reason"));
    }
    println!();
    
    // Execute steps (NO pre-flight checks - let the prover catch any issues!)
    let mut proof_generation_succeeded = true;
    let mut error_message = String::new();
    
    for (step_idx, step_data) in export.steps.iter().enumerate() {
        println!("   Executing Step {}: {} ...", step_idx, step_data.instruction);
        
        let witness = extract_witness(&step_data.witness);
        let step_io: Vec<F> = vec![];  // No explicit public I/O for this test
        
        // All steps use lane 0 (single step type)
        match nivc_state.step(0, &step_io, &witness) {
            Ok(_step_proof) => {
                println!("      ✅ Step {} succeeded", step_idx);
            }
            Err(e) => {
                println!("      ❌ Step {} failed: {}", step_idx, e);
                proof_generation_succeeded = false;
                error_message = e.to_string();
                break;
            }
        }
    }
    
    println!();
    
    if !proof_generation_succeeded {
        println!("❌ NIVC Proof Generation FAILED");
        println!("   Error: {}", error_message);
        println!();
        
        // If should_fail is true, this is the expected outcome
        if export.metadata.should_fail {
            println!("✅ Expected failure occurred (should_fail=true)");
            println!("   The NIVC prover correctly rejected the invalid constraints.");
            if let Some(reason) = &export.metadata.failure_reason {
                println!("   Expected reason: {}", reason);
            }
            println!("   Actual error: {}", error_message);
            
            // This is actually a success - the system is sound!
            println!();
            println!("🎉 Test PASSED: NIVC prover correctly rejected invalid proof");
            return;
        } else {
            // If should_fail is false but proof failed, that's unexpected
            println!("⚠️  Unexpected failure (should_fail=false)");
            panic!("NIVC proof generation failed unexpectedly: {}", error_message);
        }
    }
    
    // If we get here, all steps succeeded
    println!("✅ NIVC Proof Generation SUCCEEDED");
    println!("   All {} steps completed successfully", export.steps.len());
    println!();
    
    // If should_fail is true but proof succeeded, this is a soundness bug!
    if export.metadata.should_fail {
        println!("🚨 SOUNDNESS BUG DETECTED! 🚨");
        println!("   Test metadata indicates this should fail, but NIVC proof generation succeeded.");
        println!("   This means the NIVC prover accepted invalid constraints!");
        if let Some(reason) = &export.metadata.failure_reason {
            println!("   Expected failure: {}", reason);
        }
        println!();
        println!("   COMPARISON:");
        println!("   - IVC test: ✅ Correctly rejects with 'Extractor/binding mismatch'");
        println!("   - NIVC test: ❌ Incorrectly accepts and generates a proof");
        println!();
        panic!("Soundness violation: NIVC proof succeeded when it should have failed");
    }
    
    // Finalize and verify the NIVC chain
    println!("🔍 Finalizing NIVC Chain Proof...");
    let chain_proof = nivc_state.into_proof();
    println!("   Chain Length: {}", chain_proof.steps.len());
    println!("   Final Accumulator Step: {}", chain_proof.final_acc.step);
    println!("   Final Global State: {:?}", 
        chain_proof.final_acc.global_y
            .iter()
            .map(|f| f.as_canonical_u64())
            .collect::<Vec<_>>());
    println!();
    
    println!("🔍 Verifying NIVC Chain Proof...");
    let verify_result = verify_nivc_chain(&program, &params, &chain_proof, &y0);
    
    match verify_result {
        Ok(true) => {
            println!("✅ NIVC Verification SUCCEEDED");
            println!("   The proof is valid and verifies correctly.");
        }
        Ok(false) => {
            println!("❌ NIVC Verification FAILED");
            println!("   The proof was generated but verification returned false.");
            panic!("NIVC verification failed for generated proof");
        }
        Err(e) => {
            println!("❌ NIVC Verification ERROR: {}", e);
            panic!("NIVC verification error: {}", e);
        }
    }
    
    println!();
    println!("{}", "=".repeat(60));
    println!("🎉 Test Completed Successfully");
}

