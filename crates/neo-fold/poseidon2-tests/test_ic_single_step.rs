#![allow(non_snake_case)]

//! Integration test for poseidon2 incremental commitment
//!
//! The poseidon2 implementation for this is in Starstream, in the
//! enzo/ivc-interleaving-proto branch currently.
//!
//! Maybe in the future it could be imported as a dependency to avoid using json
//! imports, but for now just use exported circuits.
//!
//! What the circuits compute is something of this form:
//!
//! let ic = [0, 0, 0, 0]
//! for i in 0..batch_size:
//!     ic = poseidon2(ic.concat([i, i, i, i]))
//!
//!
use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::{r1cs_to_ccs, CcsStructure, Mat};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{FoldingSession, NeoStep, StepArtifacts, StepSpec};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Serialize, Deserialize, Clone)]
struct SparseMatrix {
    rows: usize,
    cols: usize,
    entries: Vec<(usize, usize, u64)>,
}

#[derive(Serialize, Deserialize, Clone)]
struct TestExport {
    num_constraints: usize,
    num_variables: usize,
    matrix_a: SparseMatrix,
    matrix_b: SparseMatrix,
    matrix_c: SparseMatrix,
    // one witness per step
    witness: Vec<Vec<u64>>,
}

fn sparse_to_dense_mat(sparse: &SparseMatrix, rows: usize, cols: usize) -> Mat<F> {
    let mut data = vec![F::ZERO; rows * cols];
    for &(row, col, val) in &sparse.entries {
        data[row * cols + col] = F::from_u64(val);
    }
    Mat::from_row_major(rows, cols, data)
}

fn build_step_ccs(r1cs: &TestExport) -> CcsStructure<F> {
    let n = r1cs.num_constraints;
    let m = r1cs.num_variables;

    let n = n.max(m);

    let a = sparse_to_dense_mat(&r1cs.matrix_a, n, n);
    let b = sparse_to_dense_mat(&r1cs.matrix_b, n, n);
    let c = sparse_to_dense_mat(&r1cs.matrix_c, n, n);
    let s0 = r1cs_to_ccs(a, b, c);

    // ensure_identity_first will now work since n == m_padded
    s0.ensure_identity_first()
        .expect("ensure_identity_first should succeed")
}

/// Pad witness to match CCS dimensions (adds slack variables if n > m_original)
fn pad_witness_to_m(mut z: Vec<F>, m_target: usize) -> Vec<F> {
    // Pad with zeros to reach m_target
    z.resize(m_target, F::ZERO);
    z
}

fn load_test_export(batch_size: usize) -> TestExport {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join(format!(
        "poseidon2-tests/poseidon2_ic_circuit_batch_{batch_size}.json"
    ));
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

struct StepCircuit {
    steps: Vec<Vec<F>>,
    step_spec: StepSpec,
    step_ccs: CcsStructure<F>,
}

impl NeoStep for StepCircuit {
    type ExternalInputs = NoInputs;

    fn state_len(&self) -> usize {
        0
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
        let z = self.steps[step_idx].clone();
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
fn test_poseidon2_ic_batch_size_1() {
    test_poseidon2_ic_batch_size(1);
}

#[test]
fn test_poseidon2_ic_batch_size_10() {
    test_poseidon2_ic_batch_size(10);
}

#[test]
fn test_poseidon2_ic_batch_size_20() {
    test_poseidon2_ic_batch_size(20);
}

#[test]
fn test_poseidon2_ic_batch_size_30() {
    test_poseidon2_ic_batch_size(30);
}

#[test]
fn test_poseidon2_ic_batch_size_40() {
    test_poseidon2_ic_batch_size(40);
}

fn test_poseidon2_ic_batch_size(batch_size: usize) {
    let export = load_test_export(batch_size);
    let _y0: Vec<F> = vec![];

    let n = export.num_constraints;
    let m = export.num_variables;

    println!("num constraints: {n}");
    println!("num variables: {m}");

    let n = n.max(m);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n)
        .expect("goldilocks_auto_r1cs_ccs should find valid params");

    // TODO: needed for now for the decomposition to be bijective in pi_ccs
    //
    // note that this may require changing other parameters too for soundness,
    // but I think in theory it should work with b = 2
    params.b = 3;

    setup_ajtai_for_dims(n);
    let l = AjtaiSModule::from_global_for_dims(D, n).expect("AjtaiSModule init");

    let step_spec = StepSpec {
        y_len: 0,
        const1_index: 1,
        y_step_indices: vec![],
        app_input_indices: Some(vec![]),
        m_in: 1,
    };

    let step_ccs = build_step_ccs(&export);
    let mut circuit = StepCircuit {
        steps: export
            .witness
            .iter()
            .map(|step_witness| step_witness.iter().map(|f| F::from_u64(*f)).collect())
            .collect(),
        step_spec: step_spec.clone(),
        step_ccs: step_ccs.clone(),
    };

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l.clone());

    for _ in 0..export.witness.len() {
        let start = Instant::now();

        session
            .add_step(&mut circuit, &NoInputs)
            .expect("add_step should succeed with optimized");

        println!("Add step duration: {:?}", start.elapsed());
    }

    let start = Instant::now();
    let run = session
        .fold_and_prove(&step_ccs)
        .expect("fold_and_prove should produce a FoldRun");
    let finalize_duration = start.elapsed();

    println!("Proof generation time (finalize): {:?}", finalize_duration);

    assert_eq!(
        run.steps.len(),
        export.witness.len(),
        "should have correct number of steps"
    );

    let mcss_public = session.mcss_public();
    let ok = session
        .verify(&step_ccs, &mcss_public, &run)
        .expect("verify should run");
    assert!(ok, "optimized verification should pass");
}
