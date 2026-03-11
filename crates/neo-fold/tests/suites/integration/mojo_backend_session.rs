#![allow(non_snake_case)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use libloading::Library;
use neo_ajtai::AjtaiSModule;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{CcsBuilder, FoldingSession};
use neo_fold::{MojoBackendConfig, ProverComputeBackend};
use neo_math::F;
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

#[test]
fn test_session_mojo_backend_matches_cpu_single_step() {
    type ResetFn = unsafe extern "C" fn();
    type CounterFn = unsafe extern "C" fn() -> usize;
    let _counter_guard = super::lock_mock_backend_counters();

    let mut cs = CcsBuilder::<F>::new(1, 0).expect("CcsBuilder::new");
    cs.r1cs_terms([(1, F::ONE)], [(2, F::ONE)], [(3, F::ONE)]);
    let ccs = cs.build_rect(5, 0).expect("build_rect");

    let public_input = vec![F::ONE];
    let witness = vec![F::from_u64(2), F::from_u64(3), F::from_u64(6), F::ZERO];
    let seed = [19u8; 32];

    let mut cpu_session =
        FoldingSession::<AjtaiSModule>::new_ajtai_seeded(FoldingMode::Optimized, &ccs, seed).expect("new cpu session");
    cpu_session
        .add_step_io(&ccs, &public_input, &witness)
        .expect("cpu add_step_io");
    let cpu_run = cpu_session
        .fold_and_prove(&ccs)
        .expect("cpu fold_and_prove");
    assert!(cpu_session
        .verify_collected(&ccs, &cpu_run)
        .expect("cpu verify"));

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
    unsafe { reset() };

    let backend = ProverComputeBackend::Mojo(MojoBackendConfig::auto().with_library_path(mock_library));
    let mut mojo_session = FoldingSession::<AjtaiSModule>::new_ajtai_seeded(FoldingMode::Optimized, &ccs, seed)
        .expect("new mojo session")
        .with_compute_backend(backend);
    mojo_session
        .add_step_io(&ccs, &public_input, &witness)
        .expect("mojo add_step_io");
    let mojo_run = mojo_session
        .fold_and_prove(&ccs)
        .expect("mojo fold_and_prove");
    assert!(mojo_session
        .verify_collected(&ccs, &mojo_run)
        .expect("mojo verify"));

    assert_eq!(
        serde_json::to_vec(&cpu_run).expect("serialize cpu run"),
        serde_json::to_vec(&mojo_run).expect("serialize mojo run"),
    );
    assert_eq!(unsafe { session_open_calls() }, 2);
}

#[test]
fn test_session_mojo_backend_verify_opens_backend_once() {
    type ResetFn = unsafe extern "C" fn();
    type CounterFn = unsafe extern "C" fn() -> usize;
    let _counter_guard = super::lock_mock_backend_counters();

    let mut cs = CcsBuilder::<F>::new(1, 0).expect("CcsBuilder::new");
    cs.r1cs_terms([(1, F::ONE)], [(2, F::ONE)], [(3, F::ONE)]);
    let ccs = cs.build_rect(5, 0).expect("build_rect");

    let public_input = vec![F::ONE];
    let witness = vec![F::from_u64(2), F::from_u64(3), F::from_u64(6), F::ZERO];
    let seed = [23u8; 32];

    let mut cpu_session =
        FoldingSession::<AjtaiSModule>::new_ajtai_seeded(FoldingMode::Optimized, &ccs, seed).expect("new cpu session");
    cpu_session
        .add_step_io(&ccs, &public_input, &witness)
        .expect("cpu add_step_io");
    let cpu_run = cpu_session
        .fold_and_prove(&ccs)
        .expect("cpu fold_and_prove");

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

    let backend = ProverComputeBackend::Mojo(MojoBackendConfig::auto().with_library_path(mock_library));
    let mut mojo_session = FoldingSession::<AjtaiSModule>::new_ajtai_seeded(FoldingMode::Optimized, &ccs, seed)
        .expect("new mojo session")
        .with_compute_backend(backend);
    mojo_session
        .add_step_io(&ccs, &public_input, &witness)
        .expect("mojo add_step_io");

    unsafe { reset() };
    assert!(mojo_session
        .verify_collected(&ccs, &cpu_run)
        .expect("mojo verify"));
    assert_eq!(unsafe { session_open_calls() }, 1);
}
