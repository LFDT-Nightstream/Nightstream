//! Integration test for Starstream TX IVC proof generation (valid test case)

use std::fs;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use neo::{F, NeoParams};
use neo::{Accumulator, StepBindingSpec, IvcChainStepInput, prove_ivc_chain, verify_ivc_chain};
use neo::{NivcProgram, NivcStepSpec, NivcState, verify_nivc_chain};
use neo::{FoldingSession, NeoStep, StepArtifacts, StepDescriptor, verify_chain_with_descriptor};
use neo::AppInputBinding;
use neo_ccs::{Mat, r1cs::r1cs_to_ccs, CcsStructure};
use p3_field::PrimeCharacteristicRing;

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
    step_spec: JsonStepSpec,
}

#[derive(Debug, Deserialize, Serialize)]
struct JsonStepSpec {
    y_len: usize,
    const1_index: usize,
    y_step_indices: Vec<usize>,
    y_prev_indices: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    app_input_indices: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct StepData {
    step_idx: usize,
    instruction: String,
    witness: WitnessData,
    r1cs: R1csData,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct WitnessData {
    instance: Vec<String>,
    witness: Vec<String>,
    z_full: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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

fn parse_field_element(s: &str) -> F {
    if let Some(hex) = s.strip_prefix("0x") {
        F::from_u64(u64::from_str_radix(hex, 16).expect("valid hex"))
    } else {
        F::from_u64(s.parse::<u64>().expect("valid decimal"))
    }
}

fn sparse_to_dense_mat(sparse: &[(usize, usize, String)], rows: usize, cols: usize) -> Mat<F> {
    let mut data = vec![F::ZERO; rows * cols];
    for (row, col, val_str) in sparse {
        data[row * cols + col] = parse_field_element(val_str);
    }
    Mat::from_row_major(rows, cols, data)
}

fn build_step_ccs(r1cs: &R1csData) -> CcsStructure<F> {
    let a = sparse_to_dense_mat(&r1cs.a_sparse, r1cs.num_constraints, r1cs.num_variables);
    let b = sparse_to_dense_mat(&r1cs.b_sparse, r1cs.num_constraints, r1cs.num_variables);
    let c = sparse_to_dense_mat(&r1cs.c_sparse, r1cs.num_constraints, r1cs.num_variables);
    r1cs_to_ccs(a, b, c)
}

fn extract_witness(witness_data: &WitnessData) -> Vec<F> {
    witness_data.z_full.iter().map(|s| parse_field_element(s)).collect()
}

fn load_test_export() -> TestExport {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("tests/test_starstream_tx_export_valid.json");
    let json_content = fs::read_to_string(&json_path).expect("Failed to read JSON");
    serde_json::from_str(&json_content).expect("Failed to parse JSON")
}

fn handle_test_result(result: Result<bool, impl std::fmt::Display>, should_fail: bool, test_name: &str) {
    match result {
        Ok(true) if should_fail => {
            panic!("SOUNDNESS BUG: {} verification succeeded but should have failed", test_name);
        }
        Ok(false) if !should_fail => {
            panic!("{} verification failed unexpectedly", test_name);
        }
        Err(e) if !should_fail => {
            panic!("{} verification error: {}", test_name, e);
        }
        _ => {} // Expected outcome
    }
}

fn verify_r1cs_constraints(step: &StepData) -> Result<(), String> {
    let z = extract_witness(&step.witness);
    let r1cs = &step.r1cs;
    
    let mut az = vec![F::ZERO; r1cs.num_constraints];
    let mut bz = vec![F::ZERO; r1cs.num_constraints];
    let mut cz = vec![F::ZERO; r1cs.num_constraints];
    
    for (row, col, val_str) in &r1cs.a_sparse {
        az[*row] += parse_field_element(val_str) * z[*col];
    }
    for (row, col, val_str) in &r1cs.b_sparse {
        bz[*row] += parse_field_element(val_str) * z[*col];
    }
    for (row, col, val_str) in &r1cs.c_sparse {
        cz[*row] += parse_field_element(val_str) * z[*col];
    }
    
    for i in 0..r1cs.num_constraints {
        if az[i] * bz[i] != cz[i] {
            return Err(format!("Step {} constraint {} violated", step.step_idx, i));
        }
    }
    
    Ok(())
}

#[test]
fn test_json_parsing_only() {
    let export = load_test_export();
    assert!(export.steps.len() > 0);
    assert!(export.ivc_params.y0.len() > 0);
}

#[test]
fn test_r1cs_constraint_verification() {
    let export = load_test_export();
    let found_violation = export.steps.iter().any(|step| verify_r1cs_constraints(step).is_err());
    
    if export.metadata.should_fail {
        assert!(found_violation, "Expected constraint violations but found none");
    } else {
        assert!(!found_violation, "Found unexpected constraint violations");
    }
}

#[test]
fn test_starstream_tx_ivc_proof_valid() {
    let export = load_test_export();
    let y0: Vec<F> = export.ivc_params.y0.iter().map(|s| parse_field_element(s)).collect();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    let binding_spec = StepBindingSpec {
        y_step_offsets: export.ivc_params.step_spec.y_step_indices.clone(),
        step_program_input_witness_indices: vec![],
        const1_witness_index: export.ivc_params.step_spec.const1_index,
    };
    
    let step_ccs = build_step_ccs(&export.steps[0].r1cs);
    let initial_accumulator = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: y0.clone(),
        step: 0,
    };
    
    let step_inputs: Vec<IvcChainStepInput> = export.steps
        .iter()
        .enumerate()
        .map(|(i, step)| IvcChainStepInput {
            witness: extract_witness(&step.witness),
            public_input: None,
            step: i as u64,
        })
        .collect();
    
    let proof_result = prove_ivc_chain(&params, &step_ccs, &step_inputs, initial_accumulator.clone(), &binding_spec);
    
    match proof_result {
        Ok(chain_proof) => {
            let verify_result = verify_ivc_chain(&step_ccs, &chain_proof, &initial_accumulator, &binding_spec, &params);
            handle_test_result(verify_result, export.metadata.should_fail, "IVC");
        }
        Err(_e) if export.metadata.should_fail => {} // Expected failure
        Err(e) => panic!("IVC proof generation failed unexpectedly: {}", e),
    }
}

#[test]
fn test_starstream_tx_nivc_proof_valid() {
    let export = load_test_export();
    let y0: Vec<F> = export.ivc_params.y0.iter().map(|s| parse_field_element(s)).collect();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    let binding_spec = StepBindingSpec {
        y_step_offsets: export.ivc_params.step_spec.y_step_indices.clone(),
        step_program_input_witness_indices: vec![],
        const1_witness_index: export.ivc_params.step_spec.const1_index,
    };
    
    let step_ccs = build_step_ccs(&export.steps[0].r1cs);
    let program = NivcProgram::new(vec![NivcStepSpec {
        ccs: step_ccs.clone(),
        binding: binding_spec.clone(),
    }]);
    
    let mut nivc_state = NivcState::new(params.clone(), program.clone(), y0.clone())
        .expect("Failed to create NIVC state");
    
    let mut proof_succeeded = true;
    for step_data in &export.steps {
        let witness = extract_witness(&step_data.witness);
        if nivc_state.step(0, &vec![], &witness).is_err() {
            proof_succeeded = false;
            break;
        }
    }
    
    if !proof_succeeded {
        assert!(export.metadata.should_fail, "NIVC proof generation failed unexpectedly");
        return;
    }
    
    let chain_proof = nivc_state.into_proof();
    let verify_result = verify_nivc_chain(&program, &params, &chain_proof, &y0);
    handle_test_result(verify_result, export.metadata.should_fail, "NIVC");
}

struct TestStepCircuit {
    steps: Vec<StepData>,
    step_spec: neo::StepSpec,
    step_ccs: CcsStructure<F>,
}

impl NeoStep for TestStepCircuit {
    type ExternalInputs = ();
    
    fn state_len(&self) -> usize {
        self.step_spec.y_len
    }
    
    fn step_spec(&self) -> neo::StepSpec {
        self.step_spec.clone()
    }
    
    fn synthesize_step(&mut self, step_idx: usize, _z_prev: &[F], _inputs: &Self::ExternalInputs) -> StepArtifacts {
        StepArtifacts {
            ccs: self.step_ccs.clone(),
            witness: extract_witness(&self.steps[step_idx].witness),
            public_app_inputs: vec![],
            spec: self.step_spec.clone(),
        }
    }
}

#[test]
fn test_starstream_tx_session_api_valid() {
    let export = load_test_export();
    let y0: Vec<F> = export.ivc_params.y0.iter().map(|s| parse_field_element(s)).collect();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    let step_spec = neo::StepSpec {
        y_len: export.ivc_params.step_spec.y_len,
        const1_index: export.ivc_params.step_spec.const1_index,
        y_step_indices: export.ivc_params.step_spec.y_step_indices.clone(),
        app_input_indices: export.ivc_params.step_spec.app_input_indices.clone(),
    };
    
    let step_ccs = build_step_ccs(&export.steps[0].r1cs);
    let mut circuit = TestStepCircuit {
        steps: export.steps.clone(),
        step_spec: step_spec.clone(),
        step_ccs: step_ccs.clone(),
    };
    
    let mut session = FoldingSession::new(Some(&params), Some(y0.clone()), 0, AppInputBinding::TranscriptOnly);
    
    let mut proof_succeeded = true;
    for _ in 0..export.steps.len() {
        if session.prove_step(&mut circuit, &()).is_err() {
            proof_succeeded = false;
            break;
        }
    }
    
    if !proof_succeeded {
        assert!(export.metadata.should_fail, "Session proof generation failed unexpectedly");
        return;
    }
    
    let (chain_proof, step_ios) = session.finalize();
    let descriptor = StepDescriptor { ccs: step_ccs, spec: step_spec };
    let verify_result = verify_chain_with_descriptor(&descriptor, &chain_proof, &y0, &params, &step_ios, AppInputBinding::TranscriptOnly);
    handle_test_result(verify_result, export.metadata.should_fail, "Session");
}

