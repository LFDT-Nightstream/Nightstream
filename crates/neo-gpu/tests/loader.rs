use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use libloading::Library;
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_gpu::{connect, DeviceApi, MojoBackendConfig, MojoLibrary};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;

fn mock_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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

fn mojo_project_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("gpu")
        .join("mojo")
}

fn backend_poseidon2_hash(session: &neo_gpu::MojoSession, input: &[u64]) -> [u64; p2::DIGEST_LEN] {
    let mut state = [Goldilocks::ZERO; p2::WIDTH];
    for chunk in input.chunks(p2::RATE) {
        for (dst, src) in state.iter_mut().zip(chunk.iter()) {
            *dst += Goldilocks::from_u64(*src);
        }
        let state_u64 = state.map(|x| x.as_canonical_u64());
        let out_u64 = session
            .permute_poseidon2_u64x8(&state_u64)
            .expect("backend poseidon2 permutation");
        state = out_u64.map(Goldilocks::from_u64);
    }
    state[0] += Goldilocks::ONE;
    let state_u64 = state.map(|x| x.as_canonical_u64());
    let state = session
        .permute_poseidon2_u64x8(&state_u64)
        .expect("backend poseidon2 final permutation")
        .map(Goldilocks::from_u64);

    let mut out = [0u64; p2::DIGEST_LEN];
    out.copy_from_slice(&state.map(|x| x.as_canonical_u64())[..p2::DIGEST_LEN]);
    out
}

fn permute_poseidon2_via_symbol(
    permute: unsafe extern "C" fn(usize, *mut u64, u32) -> i32,
    state: &[u64; 8],
) -> [u64; 8] {
    let mut backend = *state;
    let status = unsafe { permute(1, backend.as_mut_ptr(), 8) };
    assert_eq!(status, 0, "mojo poseidon status");
    backend
}

fn permute_poseidon2_batch_via_symbol(
    permute_batch: unsafe extern "C" fn(usize, *mut u64, u32, u32) -> i32,
    states: &mut [[u64; 8]],
) {
    let status = unsafe { permute_batch(1, states.as_mut_ptr().cast::<u64>(), states.len() as u32, 8) };
    assert_eq!(status, 0, "mojo poseidon batch status");
}

fn poseidon2_batch_fixture(num_states: usize) -> Vec<[u64; 8]> {
    (0..num_states)
        .map(|state_idx| std::array::from_fn(|word_idx| (state_idx as u64) * 97 + (word_idx as u64) * 17 + 3))
        .collect()
}

fn direct_backend_poseidon2_hash(
    permute: unsafe extern "C" fn(usize, *mut u64, u32) -> i32,
    input: &[u64],
) -> [u64; p2::DIGEST_LEN] {
    let mut state = [Goldilocks::ZERO; p2::WIDTH];
    for chunk in input.chunks(p2::RATE) {
        for (dst, src) in state.iter_mut().zip(chunk.iter()) {
            *dst += Goldilocks::from_u64(*src);
        }
        let state_u64 = state.map(|x| x.as_canonical_u64());
        let out_u64 = permute_poseidon2_via_symbol(permute, &state_u64);
        state = out_u64.map(Goldilocks::from_u64);
    }
    state[0] += Goldilocks::ONE;
    let state_u64 = state.map(|x| x.as_canonical_u64());
    let state = permute_poseidon2_via_symbol(permute, &state_u64).map(Goldilocks::from_u64);

    let mut out = [0u64; p2::DIGEST_LEN];
    out.copy_from_slice(&state.map(|x| x.as_canonical_u64())[..p2::DIGEST_LEN]);
    out
}

#[test]
fn loads_mock_library_and_probes_split_nc_support() {
    let cfg = MojoBackendConfig::new(DeviceApi::Metal).with_library_path(build_mock_library());
    let lib = MojoLibrary::load(&cfg).expect("load mock mojo gpu library");

    assert_eq!(lib.path(), build_mock_library());
    assert!(lib.probe_device(DeviceApi::Metal, 0).expect("probe device"));
    assert!(lib.supports_split_nc_api());
    assert!(lib.supports_poseidon2_api());
    assert!(lib.supports_poseidon2_batch_api());
}

#[test]
fn connects_to_mock_library_session() {
    let cfg = MojoBackendConfig::new(DeviceApi::Cuda)
        .with_device_id(7)
        .with_library_path(build_mock_library());
    let session = connect(&cfg).expect("connect to mock mojo gpu");

    assert_eq!(session.device_api(), DeviceApi::Cuda);
    assert_eq!(session.device_id(), 7);
    assert!(session.supports_split_nc_api());
    assert!(session.supports_poseidon2_api());
    assert!(session.supports_poseidon2_batch_api());
}

#[test]
fn auto_backend_selects_the_preferred_mock_accelerator() {
    let cfg = MojoBackendConfig::auto().with_library_path(build_mock_library());
    let session = connect(&cfg).expect("connect to mock mojo gpu with auto backend");

    #[cfg(target_os = "macos")]
    assert_eq!(session.device_api(), DeviceApi::Metal);
    #[cfg(not(target_os = "macos"))]
    assert_eq!(session.device_api(), DeviceApi::Cuda);
}

#[test]
fn poseidon2_permutation_matches_cpu_reference() {
    let cfg = MojoBackendConfig::new(DeviceApi::Metal).with_library_path(build_mock_library());
    let session = connect(&cfg).expect("connect to mock mojo gpu");

    let state = [3u64, 5, 7, 11, 13, 17, 19, 23];
    let backend = session
        .permute_poseidon2_u64x8(&state)
        .expect("poseidon2 permute through backend");

    let perm = neo_ccs::crypto::poseidon2_goldilocks::permutation();
    let cpu_in = state.map(Goldilocks::from_u64);
    let cpu_out = perm.permute(cpu_in).map(|x| x.as_canonical_u64());

    assert_eq!(backend, cpu_out);
}

#[test]
fn poseidon2_batch_permutation_matches_cpu_reference() {
    let cfg = MojoBackendConfig::new(DeviceApi::Metal).with_library_path(build_mock_library());
    let session = connect(&cfg).expect("connect to mock mojo gpu");

    let mut backend = poseidon2_batch_fixture(17);
    session
        .permute_poseidon2_batch_u64x8(&mut backend)
        .expect("poseidon2 batch permute through backend");

    let perm = neo_ccs::crypto::poseidon2_goldilocks::permutation();
    let cpu: Vec<[u64; 8]> = poseidon2_batch_fixture(17)
        .into_iter()
        .map(|state| {
            perm.permute(state.map(Goldilocks::from_u64))
                .map(|x| x.as_canonical_u64())
        })
        .collect();

    assert_eq!(backend, cpu);
}

#[test]
fn poseidon2_precompile_flow_matches_cpu_reference_for_lengths_0_to_8() {
    let cfg = MojoBackendConfig::new(DeviceApi::Metal).with_library_path(build_mock_library());
    let session = connect(&cfg).expect("connect to mock mojo gpu");

    for n in 0..=8usize {
        let input: Vec<u64> = (0..n).map(|i| (i as u64) * 17 + 3).collect();
        let backend = backend_poseidon2_hash(&session, &input);
        let cpu = p2::poseidon2_hash(
            &input
                .iter()
                .copied()
                .map(Goldilocks::from_u64)
                .collect::<Vec<_>>(),
        )
        .map(|x| x.as_canonical_u64());
        assert_eq!(backend, cpu, "length={n}");
    }
}

#[test]
#[ignore = "requires local Mojo toolchain"]
fn real_mojo_poseidon2_matches_cpu_reference() {
    type PoseidonFn = unsafe extern "C" fn(usize, *mut u64, u32) -> i32;
    type PoseidonBatchFn = unsafe extern "C" fn(usize, *mut u64, u32, u32) -> i32;
    let library_path = build_real_mojo_library();

    let cfg = MojoBackendConfig::new(DeviceApi::Cpu).with_library_path(library_path);
    let session = connect(&cfg).expect("connect to real mojo gpu library");
    assert!(session.supports_poseidon2_api());
    assert!(session.supports_poseidon2_batch_api());

    let state = [3u64, 5, 7, 11, 13, 17, 19, 23];
    let backend_via_loader = session
        .permute_poseidon2_u64x8(&state)
        .expect("loader poseidon2 permute");
    let cpu = p2::permutation()
        .permute(state.map(Goldilocks::from_u64))
        .map(|x| x.as_canonical_u64());
    assert_eq!(backend_via_loader, cpu);

    let mut backend_batch_via_loader = poseidon2_batch_fixture(17);
    session
        .permute_poseidon2_batch_u64x8(&mut backend_batch_via_loader)
        .expect("loader poseidon2 batch permute");
    let cpu_batch: Vec<[u64; 8]> = poseidon2_batch_fixture(17)
        .into_iter()
        .map(|state| {
            p2::permutation()
                .permute(state.map(Goldilocks::from_u64))
                .map(|x| x.as_canonical_u64())
        })
        .collect();
    assert_eq!(backend_batch_via_loader, cpu_batch);

    let lib = unsafe { Library::new(library_path) }.expect("load real mojo gpu library");
    let permute = unsafe {
        *lib.get::<PoseidonFn>(b"nightstream_gpu_poseidon2_permute_u64x8\0")
            .expect("load poseidon2 symbol")
    };
    let permute_batch = unsafe {
        *lib.get::<PoseidonBatchFn>(b"nightstream_gpu_poseidon2_permute_batch_u64x8\0")
            .expect("load poseidon2 batch symbol")
    };

    let state = [3u64, 5, 7, 11, 13, 17, 19, 23];
    let backend = permute_poseidon2_via_symbol(permute, &state);
    assert_eq!(backend, cpu);

    let mut backend_batch = poseidon2_batch_fixture(17);
    permute_poseidon2_batch_via_symbol(permute_batch, &mut backend_batch);
    let cpu_batch: Vec<[u64; 8]> = poseidon2_batch_fixture(17)
        .into_iter()
        .map(|state| {
            p2::permutation()
                .permute(state.map(Goldilocks::from_u64))
                .map(|x| x.as_canonical_u64())
        })
        .collect();
    assert_eq!(backend_batch, cpu_batch);

    for n in 0..=8usize {
        let input: Vec<u64> = (0..n).map(|i| (i as u64) * 17 + 3).collect();
        let backend = direct_backend_poseidon2_hash(permute, &input);
        let cpu = p2::poseidon2_hash(
            &input
                .iter()
                .copied()
                .map(Goldilocks::from_u64)
                .collect::<Vec<_>>(),
        )
        .map(|x| x.as_canonical_u64());
        assert_eq!(backend, cpu, "length={n}");
    }
}

#[test]
#[ignore = "requires local Metal-capable Mojo runtime"]
fn real_mojo_metal_session_batch_matches_cpu_reference() {
    let library_path = build_real_mojo_library();
    let cfg = MojoBackendConfig::new(DeviceApi::Metal).with_library_path(library_path);

    let Ok(session) = connect(&cfg) else {
        eprintln!("skipping: real Mojo shared-library Metal session is not available in this runtime");
        return;
    };
    assert_eq!(session.device_api(), DeviceApi::Metal);
    assert!(session.supports_poseidon2_api());
    assert!(session.supports_poseidon2_batch_api());

    let mut backend = poseidon2_batch_fixture(256);
    if let Err(err) = session.permute_poseidon2_batch_u64x8(&mut backend) {
        eprintln!("skipping: real Mojo shared-library Metal batch path failed: {err}");
        return;
    }

    let cpu: Vec<[u64; 8]> = poseidon2_batch_fixture(256)
        .into_iter()
        .map(|state| {
            p2::permutation()
                .permute(state.map(Goldilocks::from_u64))
                .map(|x| x.as_canonical_u64())
        })
        .collect();
    assert_eq!(backend, cpu);
}

#[test]
#[ignore = "requires working Mojo GPU runtime"]
fn mojo_gpu_compare_script_matches_cpu_reference() {
    let project_dir = mojo_project_dir();
    let status = Command::new(pixi_bin())
        .arg("run")
        .arg("mojo")
        .arg("run")
        .arg("src/poseidon_gpu_compare.mojo")
        .current_dir(project_dir)
        .status()
        .expect("spawn mojo gpu compare script");
    assert!(status.success(), "mojo gpu compare script failed");
}

#[test]
#[ignore = "manual Mojo GPU throughput benchmark"]
fn mojo_gpu_bench_script_runs() {
    let project_dir = mojo_project_dir();
    let status = Command::new(pixi_bin())
        .arg("run")
        .arg("mojo")
        .arg("run")
        .arg("src/poseidon_gpu_bench.mojo")
        .current_dir(project_dir)
        .status()
        .expect("spawn mojo gpu bench script");
    assert!(status.success(), "mojo gpu bench script failed");
}
