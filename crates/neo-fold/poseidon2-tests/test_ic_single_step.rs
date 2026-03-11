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
use neo_fold::{DeviceApi, MojoBackendConfig, ProverComputeBackend};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

use libloading::Library;

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

    // ensure_identity_first_owned will now work since n == m_padded
    s0.ensure_identity_first_owned()
        .expect("ensure_identity_first_owned should succeed")
}

/// Pad witness to match CCS dimensions (adds slack variables if n > m_original)
fn pad_witness_to_m(mut z: Vec<F>, m_target: usize) -> Vec<F> {
    // Pad with zeros to reach m_target
    z.resize(m_target, F::ZERO);
    z
}

fn load_test_export(batch_size: usize) -> TestExport {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join(format!("poseidon2-tests/poseidon2_ic_circuit_batch_{batch_size}.json"));
    let json_content = fs::read_to_string(&json_path).expect("Failed to read JSON");
    serde_json::from_str(&json_content).expect("Failed to parse JSON")
}

fn setup_ajtai_for_dims(m: usize) {
    let m_commit = m.div_ceil(D);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 4, m_commit).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

fn poseidon_prove_verify_params(n: usize) -> NeoParams {
    let base = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("goldilocks_auto_r1cs_ccs should find valid params");
    NeoParams::new(
        base.q,
        base.eta,
        base.d,
        base.kappa,
        base.m,
        4,
        16,
        base.T,
        base.s,
        base.lambda,
    )
    .expect("base-4 poseidon params")
}

fn mock_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("neo-gpu")
        .join("tests")
        .join("support")
        .join("mock-mojo-gpu")
        .join("Cargo.toml")
}

fn mock_library_name() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "libmock_mojo_gpu.dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "libmock_mojo_gpu.so"
    }
    #[cfg(target_os = "windows")]
    {
        "mock_mojo_gpu.dll"
    }
}

fn build_mock_library() -> &'static Path {
    static LIB_PATH: OnceLock<PathBuf> = OnceLock::new();
    LIB_PATH.get_or_init(|| {
        let manifest = mock_manifest_path();
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
        let status = Command::new(cargo)
            .arg("build")
            .arg("--release")
            .arg("--manifest-path")
            .arg(&manifest)
            .status()
            .expect("spawn cargo build for mock mojo gpu");
        assert!(status.success(), "mock mojo gpu build failed");

        manifest
            .parent()
            .expect("mock manifest parent")
            .join("target")
            .join("release")
            .join(mock_library_name())
            .canonicalize()
            .expect("canonical mock mojo gpu library path")
    })
}

fn real_mojo_library_name() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "libnightstream_mojo_gpu.dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "libnightstream_mojo_gpu.so"
    }
    #[cfg(target_os = "windows")]
    {
        "nightstream_mojo_gpu.dll"
    }
}

fn pixi_bin() -> OsString {
    if let Some(home) = std::env::var_os("HOME") {
        let candidate = PathBuf::from(home).join(".pixi").join("bin").join("pixi");
        if candidate.is_file() {
            return candidate.into_os_string();
        }
    }
    OsString::from("pixi")
}

fn build_real_mojo_library() -> &'static Path {
    static LIB_PATH: OnceLock<PathBuf> = OnceLock::new();
    LIB_PATH.get_or_init(|| {
        let project_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("gpu")
            .join("mojo");
        let output_dir = project_dir.join("build");
        let output = output_dir.join(real_mojo_library_name());
        std::fs::create_dir_all(&output_dir).expect("create mojo build directory");

        let status = Command::new(pixi_bin())
            .arg("run")
            .arg("mojo")
            .arg("build")
            .arg("--emit")
            .arg("shared-lib")
            .arg("src/lib.mojo")
            .arg("-o")
            .arg(&output)
            .current_dir(&project_dir)
            .status()
            .expect("spawn mojo build");
        assert!(status.success(), "real mojo gpu build failed");

        output
            .canonicalize()
            .expect("canonical real mojo gpu library path")
    })
}

fn required_accelerator_api() -> DeviceApi {
    #[cfg(target_os = "macos")]
    {
        DeviceApi::Metal
    }
    #[cfg(not(target_os = "macos"))]
    {
        DeviceApi::Cuda
    }
}

#[derive(Clone)]
struct NoInputs;

struct StepCircuit {
    steps: Vec<Vec<F>>,
    step_spec: StepSpec,
    step_ccs: Arc<CcsStructure<F>>,
}

impl NeoStep for StepCircuit {
    type ExternalInputs = NoInputs;

    fn state_len(&self) -> usize {
        0
    }

    fn step_spec(&self) -> StepSpec {
        self.step_spec.clone()
    }

    fn synthesize_step(&mut self, step_idx: usize, _y_prev: &[F], _inputs: &Self::ExternalInputs) -> StepArtifacts {
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

fn build_poseidon_session(
    batch_size: usize,
    compute_backend: Option<ProverComputeBackend>,
) -> (FoldingSession<AjtaiSModule>, Arc<CcsStructure<F>>) {
    let export = load_test_export(batch_size);

    let n = export.num_constraints.max(export.num_variables);
    let params = poseidon_prove_verify_params(n);

    setup_ajtai_for_dims(n);
    let l = AjtaiSModule::from_global_for_dims(D, n.div_ceil(D)).expect("AjtaiSModule init");

    let step_spec = StepSpec {
        y_len: 0,
        const1_index: 0,
        y_step_indices: vec![],
        app_input_indices: Some(vec![]),
        m_in: 1,
    };

    let step_ccs = Arc::new(build_step_ccs(&export));
    let mut circuit = StepCircuit {
        steps: export
            .witness
            .iter()
            .map(|step_witness| step_witness.iter().map(|f| F::from_u64(*f)).collect())
            .collect(),
        step_spec: step_spec.clone(),
        step_ccs: step_ccs.clone(),
    };

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l);
    if let Some(backend) = compute_backend {
        session = session.with_compute_backend(backend);
    }

    for step_idx in 0..export.witness.len() {
        session
            .add_step(&mut circuit, &NoInputs)
            .unwrap_or_else(|err| panic!("add_step {step_idx} should succeed under base-4 params: {err}"));
    }

    (session, step_ccs)
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

#[test]
fn test_poseidon2_ic_batch_size_1_mock_mojo_prove_verify() {
    type ResetFn = unsafe extern "C" fn();
    type CounterFn = unsafe extern "C" fn() -> usize;

    let mut cpu_session;
    let mut mojo_session;
    let ccs;

    (cpu_session, ccs) = build_poseidon_session(1, None);

    let cpu_run = cpu_session
        .fold_and_prove(ccs.as_ref())
        .expect("cpu fold_and_prove should succeed");
    assert!(
        cpu_session
            .verify_collected(ccs.as_ref(), &cpu_run)
            .expect("cpu verify_collected should run"),
        "cpu verify_collected should pass"
    );

    let mock_library = build_mock_library();
    let lib = unsafe { Library::new(mock_library) }.expect("load mock mojo gpu library");
    let reset = unsafe {
        *lib.get::<ResetFn>(b"nightstream_gpu_test_reset_counters\0")
            .expect("load counter reset symbol")
    };
    let session_open_calls = unsafe {
        *lib.get::<CounterFn>(b"nightstream_gpu_test_session_open_calls\0")
            .expect("load session open counter symbol")
    };
    let poseidon2_batch_calls = unsafe {
        *lib.get::<CounterFn>(b"nightstream_gpu_test_poseidon2_batch_calls\0")
            .expect("load poseidon2 batch counter symbol")
    };
    unsafe { reset() };

    let backend =
        ProverComputeBackend::Mojo(MojoBackendConfig::new(required_accelerator_api()).with_library_path(mock_library));
    (mojo_session, _) = build_poseidon_session(1, Some(backend));

    let mojo_run = mojo_session
        .fold_and_prove(ccs.as_ref())
        .expect("mock mojo fold_and_prove should succeed");
    assert!(
        mojo_session
            .verify_collected(ccs.as_ref(), &mojo_run)
            .expect("mock mojo verify_collected should run"),
        "mock mojo verify_collected should pass"
    );

    assert_eq!(
        serde_json::to_vec(&cpu_run).expect("serialize cpu run"),
        serde_json::to_vec(&mojo_run).expect("serialize mojo run"),
    );
    assert_eq!(unsafe { session_open_calls() }, 2);
    assert!(
        unsafe { poseidon2_batch_calls() } > 0,
        "mock mojo backend should use batched Poseidon2 during prove/verify"
    );
}

#[test]
#[ignore = "requires local Mojo shared library with a working accelerator session"]
fn test_poseidon2_ic_batch_size_40_real_mojo_gpu_prove_verify() {
    let backend = ProverComputeBackend::Mojo(
        MojoBackendConfig::new(required_accelerator_api()).with_library_path(build_real_mojo_library()),
    );
    let (mut session, ccs) = build_poseidon_session(40, Some(backend));

    let run = session
        .fold_and_prove(ccs.as_ref())
        .expect("real mojo gpu fold_and_prove should succeed");
    assert!(
        session
            .verify_collected(ccs.as_ref(), &run)
            .expect("real mojo gpu verify_collected should run"),
        "real mojo gpu verify_collected should pass"
    );
}

fn test_poseidon2_ic_batch_size(batch_size: usize) {
    let export = load_test_export(batch_size);
    let _y0: Vec<F> = vec![];

    let n = export.num_constraints;
    let m = export.num_variables;

    println!("num constraints: {n}");
    println!("num variables: {m}");

    let n = n.max(m);

    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("goldilocks_auto_r1cs_ccs should find valid params");

    setup_ajtai_for_dims(n);
    let l = AjtaiSModule::from_global_for_dims(D, n.div_ceil(D)).expect("AjtaiSModule init");

    let step_spec = StepSpec {
        y_len: 0,
        const1_index: 0,
        y_step_indices: vec![],
        app_input_indices: Some(vec![]),
        m_in: 1,
    };

    let step_ccs = Arc::new(build_step_ccs(&export));
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
    let start = Instant::now();

    let mut saw_expected_rejection = false;
    for _ in 0..export.witness.len() {
        let step_start = Instant::now();
        match session.add_step(&mut circuit, &NoInputs) {
            Ok(()) => {
                println!("Add step duration: {:?}", step_start.elapsed());
            }
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("not representable"),
                    "unexpected add_step failure under b=2: {msg}"
                );
                saw_expected_rejection = true;
                break;
            }
        }
    }
    println!("Poseidon2 b=2 range-check pass time: {:?}", start.elapsed());
    assert!(
        saw_expected_rejection,
        "poseidon2 witness unexpectedly passed b=2 representability guard; revisit this test"
    );
}
