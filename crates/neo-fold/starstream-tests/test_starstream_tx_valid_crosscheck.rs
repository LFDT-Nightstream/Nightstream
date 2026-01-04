#![allow(non_snake_case)]
#![cfg(feature = "paper-exact")]

//! Integration test for Starstream TX using crosscheck folding mode

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use neo_fold::session::{FoldingSession, NeoStep, StepArtifacts, StepSpec};
use neo_fold::pi_ccs::FoldingMode;
use neo_ccs::{Mat, r1cs_to_ccs, CcsStructure};
use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule};
use rand_chacha::rand_core::SeedableRng;
use neo_params::NeoParams;
use neo_math::{F, D};
use p3_field::PrimeCharacteristicRing;
use neo_reductions::engines::CrosscheckCfg;

#[derive(Debug, Deserialize, Serialize)]
struct TestExport {
    metadata: Metadata,
    ivc_params: IvcParams,
    steps: Vec<StepData>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Metadata {
    test_name: String,
    field: String,
    modulus: String,
    num_steps: usize,
    should_fail: bool,
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
    let n = r1cs.num_constraints;
    let m = r1cs.num_variables;
    
    // If n > m, we need to pad with slack variables to make m' = n
    let m_padded = n.max(m);
    
    let a = sparse_to_dense_mat(&r1cs.a_sparse, n, m_padded);
    let b = sparse_to_dense_mat(&r1cs.b_sparse, n, m_padded);
    let c = sparse_to_dense_mat(&r1cs.c_sparse, n, m_padded);
    let s0 = r1cs_to_ccs(a, b, c);
    
    // ensure_identity_first_owned will now work since n == m_padded
    s0.ensure_identity_first_owned()
        .expect("ensure_identity_first_owned should succeed")
}

fn extract_witness(witness_data: &WitnessData) -> Vec<F> {
    witness_data.z_full.iter().map(|s| parse_field_element(s)).collect()
}

/// Pad witness to match CCS dimensions (adds slack variables if n > m_original)
fn pad_witness_to_m(mut z: Vec<F>, m_target: usize) -> Vec<F> {
    // Pad with zeros to reach m_target
    z.resize(m_target, F::ZERO);
    z
}

fn load_test_export() -> TestExport {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("starstream-tests/test_starstream_tx_export_valid.json");
    let json_content = fs::read_to_string(&json_path).expect("Failed to read JSON");
    serde_json::from_str(&json_content).expect("Failed to parse JSON")
}

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

#[derive(Clone)]
struct NoInputs;

struct StarstreamStepCircuit {
    steps: Vec<StepData>,
    step_spec: StepSpec,
    step_ccs: Arc<CcsStructure<F>>,
}

impl NeoStep for StarstreamStepCircuit {
    type ExternalInputs = NoInputs;
    
    fn state_len(&self) -> usize {
        self.step_spec.y_len
    }
    
    fn step_spec(&self) -> StepSpec {
        self.step_spec.clone()
    }
    
    fn synthesize_step(
        &mut self,
        step_idx: usize,
        _y_prev: &[F],
        _inputs: &Self::ExternalInputs,
    ) -> StepArtifacts {
        let z = extract_witness(&self.steps[step_idx].witness);
        let z_padded = pad_witness_to_m(z, self.step_ccs.m);
        StepArtifacts {
            ccs: self.step_ccs.clone(),
            witness: z_padded,
            public_app_inputs: vec![],
            spec: self.step_spec.clone(),
        }
    }
}

#[test]
fn test_starstream_tx_valid_crosscheck() {
    let export = load_test_export();
    let _y0: Vec<F> = export.ivc_params.y0.iter().map(|s| parse_field_element(s)).collect();
    
    // Determine n from the first step's R1CS (n = number of constraints, not variables)
    let n = export.steps[0].r1cs.num_constraints;
    let m = export.steps[0].r1cs.num_variables;
    
    // If n > m, we pad to m_padded = n for identity-first CCS
    let m_padded = n.max(m);
    
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n)
        .expect("goldilocks_auto_r1cs_ccs should find valid params");
    
    // Ajtai commitment is over Z ∈ F^{D×m}, so set it up for m_padded (after slack variable padding)
    setup_ajtai_for_dims(m_padded);
    let l = AjtaiSModule::from_global_for_dims(D, m_padded).expect("AjtaiSModule init");
    
    let step_spec = StepSpec {
        y_len: export.ivc_params.step_spec.y_len,
        const1_index: export.ivc_params.step_spec.const1_index,
        y_step_indices: export.ivc_params.step_spec.y_step_indices.clone(),
        // Include y_prev_indices in app_input_indices to complete the public input specification
        app_input_indices: Some(export.ivc_params.step_spec.y_prev_indices.clone()),
        m_in: export.steps[0].r1cs.num_public_inputs,
    };
    
    let step_ccs = Arc::new(build_step_ccs(&export.steps[0].r1cs));
    let mut circuit = StarstreamStepCircuit {
        steps: export.steps.clone(),
        step_spec: step_spec.clone(),
        step_ccs: step_ccs.clone(),
    };
    
    // Configure crosscheck with selective checks
    let crosscheck_cfg = CrosscheckCfg {
        fail_fast: true,
        initial_sum: true,
        per_round: true,
        terminal: true,
        outputs: true,
    };
    
    let mut session = FoldingSession::new(
        FoldingMode::OptimizedWithCrosscheck(crosscheck_cfg),
        params,
        l.clone()
    );
    
    // Execute all steps
    for _ in 0..export.steps.len() {
        session.add_step(&mut circuit, &NoInputs)
            .expect("add_step should succeed with crosscheck");
    }
    
    let run = session
        .fold_and_prove(step_ccs.as_ref())
        .expect("fold_and_prove should produce a FoldRun");
    
    assert_eq!(run.steps.len(), export.steps.len(), "should have correct number of steps");
    
    let mcss_public = session.mcss_public();
    let ok = session
        .verify(step_ccs.as_ref(), &mcss_public, &run)
        .expect("verify should run");
    assert!(ok, "crosscheck verification should pass");
}
