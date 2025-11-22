#![allow(non_snake_case)]
#![cfg(feature = "paper-exact")]

//! Integration test for Starstream TX using paper-exact folding mode with invalid witness
//! This test expects to fail because the witnesses and/or input don't properly work with the CCS circuit

use std::fs;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use neo_fold::session::{FoldingSession, NeoStep, StepArtifacts, StepSpec};
use neo_fold::pi_ccs::FoldingMode;
use neo_ccs::{Mat, r1cs_to_ccs, CcsStructure};
use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule};
use rand_chacha::rand_core::SeedableRng;
use neo_params::NeoParams;
use neo_math::{F, D};
use p3_field::PrimeCharacteristicRing;

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
    
    // ensure_identity_first will now work since n == m_padded
    s0.ensure_identity_first()
        .expect("ensure_identity_first should succeed")
}

fn extract_witness(witness_data: &WitnessData) -> Vec<F> {
    witness_data.z_full.iter().map(|s| parse_field_element(s)).collect()
}

/// Pad witness to match CCS dimensions (adds slack variables if n > m_original)
fn pad_witness_to_m(mut z: Vec<F>, m_target: usize) -> Vec<F> {
    z.resize(m_target, F::ZERO);
    z
}

fn load_test_export() -> TestExport {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("starstream-tests/test_starstream_tx_export_invalid.json");
    let json_content = fs::read_to_string(&json_path).expect("Failed to read JSON");
    serde_json::from_str(&json_content).expect("Failed to parse JSON")
}

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

/// Check if two CCS structures are exactly equal (dimensions and all matrix entries).
/// Simpler and collision-free compared to fingerprinting.
fn ccs_equal(a: &CcsStructure<F>, b: &CcsStructure<F>) -> bool {
    if (a.n, a.m, a.t()) != (b.n, b.m, b.t()) {
        return false;
    }
    a.matrices.iter().zip(&b.matrices).all(|(x, y)| x.as_slice() == y.as_slice())
}

#[derive(Clone)]
struct NoInputs;

struct StarstreamStepCircuit {
    steps: Vec<StepData>,
    step_spec: StepSpec,
    baseline_ccs: CcsStructure<F>,
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
        // Build CCS from THIS step's R1CS (already padded to square in build_step_ccs)
        let ccs_this = build_step_ccs(&self.steps[step_idx].r1cs);
        
        // Defensive check: all steps must share the EXACT SAME CCS content
        assert!(
            ccs_equal(&ccs_this, &self.baseline_ccs),
            "CCS content changed at step {} (constraint matrices differ from baseline)",
            step_idx
        );
        
        // Extract witness and pad to match CCS dimensions (m might be larger due to slack variables)
        let z_raw = extract_witness(&self.steps[step_idx].witness);
        let witness_padded = pad_witness_to_m(z_raw, ccs_this.m);
        
        StepArtifacts {
            ccs: ccs_this,
            witness: witness_padded,
            public_app_inputs: vec![],
            spec: self.step_spec.clone(),
        }
    }
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_starstream_tx_paper_exact_invalid() {
    let export = load_test_export();
    
    assert!(export.metadata.should_fail, "This test expects to use invalid data");
    
    let _y0: Vec<F> = export.ivc_params.y0.iter().map(|s| parse_field_element(s)).collect();
    
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
    
    // Add early validation that all steps have the same dimensions
    let n0 = export.steps[0].r1cs.num_constraints;
    let m0 = export.steps[0].r1cs.num_variables;
    assert!(
        export.steps.iter().all(|s|
            s.r1cs.num_constraints == n0 && s.r1cs.num_variables == m0
        ),
        "All steps must share the same (n, m) for this test"
    );
    
    let baseline_ccs = build_step_ccs(&export.steps[0].r1cs);
    let mut circuit = StarstreamStepCircuit {
        steps: export.steps.clone(),
        step_spec: step_spec.clone(),
        baseline_ccs: baseline_ccs.clone(),
    };
    
    let mut session = FoldingSession::new(FoldingMode::PaperExact, params, l.clone());
    
    for _ in 0..export.steps.len() {
        session.add_step(&mut circuit, &NoInputs)
            .expect("add_step should succeed (failure comes at fold_and_prove)");
    }
    
    let result = session.fold_and_prove(&baseline_ccs);
    
    assert!(
        result.is_err(),
        "finalize must fail on invalid witnesses (CCS/NC constraints don't hold)"
    );
}

