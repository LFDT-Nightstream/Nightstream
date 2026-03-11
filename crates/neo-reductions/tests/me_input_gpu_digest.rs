use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use libloading::Library;
use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_gpu::{MojoBackendConfig, ProverComputeBackend};
use neo_math::{D, F, K};
use neo_reductions::engines::utils::{bind_me_inputs, bind_me_inputs_with_backend};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

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

fn sample_f(seed: u64) -> F {
    F::from_u64(seed.wrapping_mul(17).wrapping_add(3))
}

fn sample_k(seed: u64) -> K {
    sample_f(seed).into()
}

fn sample_k_vec(len: usize, seed: u64) -> Vec<K> {
    (0..len).map(|i| sample_k(seed + i as u64)).collect()
}

fn sample_mat(rows: usize, cols: usize, seed: u64) -> Mat<F> {
    let data = (0..rows * cols)
        .map(|i| sample_f(seed + i as u64))
        .collect::<Vec<_>>();
    Mat::from_row_major(rows, cols, data)
}

fn sample_commitment(seed: u64) -> Commitment {
    let mut c = Commitment::zeros(D, 1);
    for (i, slot) in c.data.iter_mut().enumerate() {
        *slot = sample_f(seed + i as u64);
    }
    c
}

fn sample_me_claim(idx: usize) -> CeClaim<Commitment, F, K> {
    let seed = (idx as u64) * 1000;
    let x_cols = 2 + (idx % 2);
    let y_row_len = 2 + (idx % 3);
    CeClaim {
        c: sample_commitment(seed + 1),
        X: sample_mat(D, x_cols, seed + 100),
        r: sample_k_vec(3, seed + 200),
        s_col: sample_k_vec(4, seed + 300),
        y_ring: vec![
            sample_k_vec(y_row_len, seed + 400),
            sample_k_vec(y_row_len + 1, seed + 500),
        ],
        ct: sample_k_vec(3, seed + 600),
        aux_openings: sample_k_vec(2, seed + 700),
        y_zcol: sample_k_vec(5, seed + 800),
        m_in: x_cols,
        fold_digest: std::array::from_fn(|j| seed.wrapping_add(j as u64) as u8),
        c_step_coords: (0..4).map(|j| sample_f(seed + 900 + j)).collect(),
        u_offset: idx,
        u_len: idx + 3,
    }
}

#[test]
fn bind_me_inputs_mojo_backend_matches_cpu_and_uses_batch_poseidon() {
    type ResetFn = unsafe extern "C" fn();
    type CounterFn = unsafe extern "C" fn() -> usize;

    let me_inputs = (0..6).map(sample_me_claim).collect::<Vec<_>>();
    let mut cpu_tr = Poseidon2Transcript::new(b"neo.reductions/me_input_gpu_digest");
    let mut mojo_tr = Poseidon2Transcript::new(b"neo.reductions/me_input_gpu_digest");

    bind_me_inputs(&mut cpu_tr, &me_inputs).expect("cpu bind_me_inputs");

    let mock_library = build_mock_library();
    let lib = unsafe { Library::new(mock_library) }.expect("load mock mojo gpu library");
    let reset = unsafe {
        *lib.get::<ResetFn>(b"nightstream_gpu_test_reset_counters\0")
            .expect("load counter reset symbol")
    };
    let batch_calls = unsafe {
        *lib.get::<CounterFn>(b"nightstream_gpu_test_poseidon2_batch_calls\0")
            .expect("load batch counter symbol")
    };
    unsafe { reset() };

    let backend = ProverComputeBackend::Mojo(MojoBackendConfig::auto().with_library_path(mock_library));
    bind_me_inputs_with_backend(&mut mojo_tr, &me_inputs, &backend).expect("mojo bind_me_inputs");

    assert_eq!(cpu_tr.digest32(), mojo_tr.digest32());
    assert!(
        unsafe { batch_calls() } > 0,
        "expected mock mojo backend to use the batched Poseidon2 symbol"
    );
}

#[test]
fn bind_me_inputs_auto_backend_falls_back_to_cpu_when_library_is_missing() {
    let me_inputs = (0..6).map(sample_me_claim).collect::<Vec<_>>();
    let mut cpu_tr = Poseidon2Transcript::new(b"neo.reductions/me_input_auto_digest");
    let mut auto_tr = Poseidon2Transcript::new(b"neo.reductions/me_input_auto_digest");

    bind_me_inputs(&mut cpu_tr, &me_inputs).expect("cpu bind_me_inputs");

    let backend = ProverComputeBackend::Mojo(
        MojoBackendConfig::auto().with_library_path("/tmp/nightstream-mojo-gpu-missing.dylib"),
    );
    bind_me_inputs_with_backend(&mut auto_tr, &me_inputs, &backend).expect("auto bind_me_inputs");

    assert_eq!(cpu_tr.digest32(), auto_tr.digest32());
}
