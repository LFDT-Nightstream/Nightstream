use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

use libloading::Library;
use neo_ajtai::{setup as ajtai_setup, AjtaiSModule, Commitment};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, Mat, SparsePoly, Term};
use neo_gpu::{DeviceApi, MojoBackendConfig, ProverComputeBackend};
use neo_math::{from_complex, D, F, K};
use neo_params::NeoParams;
use neo_reductions::accelerator::{BackendContext, SplitNcNcOracle, SplitNcOptimizedOracle};
use neo_reductions::api::{prove_with_backend, verify_with_backend, FoldingMode};
use neo_reductions::engines::optimized_engine::oracle::{NcOracle, OptimizedOracle, SparseCache};
use neo_reductions::engines::optimized_engine::{Challenges, PiCcsProof};
use neo_reductions::engines::utils::build_dims_and_policy;
use neo_reductions::sumcheck::RoundOracle;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn build_mock_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn lock_mock_backend() -> std::sync::MutexGuard<'static, ()> {
    build_mock_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
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

fn k(re: u64, im: u64) -> K {
    from_complex(F::from_u64(re), F::from_u64(im))
}

fn dense_mat<Ff: PrimeCharacteristicRing + Copy>(rows: usize, cols: usize, seed: u64) -> Mat<Ff> {
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let value = if (r + 2 * c) % 5 == 0 {
                Ff::from_u64(seed + (r as u64) * 17 + (c as u64) * 23 + 1)
            } else {
                Ff::ZERO
            };
            data.push(value);
        }
    }
    Mat::from_row_major(rows, cols, data)
}

fn identity_left(n: usize, m: usize) -> Mat<F> {
    let mut mat = Mat::zero(n, m, F::ZERO);
    for i in 0..n.min(m) {
        mat.set(i, i, F::ONE);
    }
    mat
}

fn zero_poly(t: usize) -> SparsePoly<F> {
    SparsePoly::new(t, Vec::new())
}

fn z_witness(seed: u64, m: usize) -> Mat<F> {
    assert!(m.is_multiple_of(D), "Split-NC test fixture requires SuperNeo-compatible width");
    let cols = m / D;
    let mut data = Vec::with_capacity(D * cols);
    for rho in 0..D {
        for blk in 0..cols {
            let c = blk * D + rho;
            data.push(F::from_u64(seed + (rho as u64) * 19 + (c as u64) * 29));
        }
    }
    Mat::from_row_major(D, cols, data)
}

struct OracleFixture {
    params: NeoParams,
    s: CcsStructure<F>,
    mcs_witnesses: Vec<CcsWitness<F>>,
    me_witnesses: Vec<Mat<F>>,
    ch: Challenges,
    r_inputs: Vec<K>,
    ell_d: usize,
    ell_n: usize,
    ell_m: usize,
    d_sc: usize,
    sparse: Arc<SparseCache<F>>,
}

fn build_oracle_fixture() -> OracleFixture {
    let n = D;
    let m = D;
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let matrices = vec![
        Mat::<F>::identity(n),
        dense_mat::<F>(n, m, 10),
        dense_mat::<F>(n, m, 20),
        dense_mat::<F>(n, m, 30),
    ];
    let f = SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 0, 1],
            },
        ],
    );
    let s = CcsStructure::new(matrices, f).expect("ccs");
    let dims = build_dims_and_policy(&params, &s).expect("dims");

    let mcs_witnesses = vec![
        CcsWitness {
            w: vec![],
            Z: z_witness(100, m),
        },
        CcsWitness {
            w: vec![],
            Z: z_witness(200, m),
        },
    ];
    let me_witnesses = vec![z_witness(300, m)];
    let ch = Challenges {
        alpha: (0..dims.ell_d)
            .map(|i| k(1_000 + i as u64, 2_000 + i as u64))
            .collect(),
        beta_a: (0..dims.ell_d)
            .map(|i| k(3_000 + i as u64, 4_000 + i as u64))
            .collect(),
        beta_r: (0..dims.ell_n)
            .map(|i| k(5_000 + i as u64, 6_000 + i as u64))
            .collect(),
        beta_m: (0..dims.ell_m)
            .map(|i| k(7_000 + i as u64, 8_000 + i as u64))
            .collect(),
        gamma: k(9_999, 11_111),
    };
    let r_inputs = (0..dims.ell_n)
        .map(|i| k(12_000 + i as u64, 13_000 + i as u64))
        .collect::<Vec<_>>();
    let sparse = Arc::new(SparseCache::build(&s));

    OracleFixture {
        params,
        s,
        mcs_witnesses,
        me_witnesses,
        ch,
        r_inputs,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
        sparse,
    }
}

fn build_prove_fixture(
    label: &'static [u8],
) -> (
    NeoParams,
    CcsStructure<F>,
    AjtaiSModule,
    Vec<CcsClaim<Commitment, F>>,
    Vec<CcsWitness<F>>,
    Poseidon2Transcript,
) {
    let n = 4;
    let m = D;
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let s = CcsStructure::new(vec![identity_left(n, m)], zero_poly(1)).expect("ccs");

    let witnesses = vec![
        CcsWitness {
            w: vec![F::ZERO; s.m],
            Z: Mat::from_row_major(D, m / D, vec![F::ZERO; D * (m / D)]),
        },
        CcsWitness {
            w: vec![F::ZERO; s.m],
            Z: Mat::from_row_major(D, m / D, vec![F::ZERO; D * (m / D)]),
        },
    ];

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m / D).expect("ajtai setup");
    let l = AjtaiSModule::new(Arc::new(pp));
    let claims = witnesses
        .iter()
        .map(|wit| CcsClaim {
            c: l.commit(&wit.Z),
            x: vec![],
            m_in: 0,
        })
        .collect::<Vec<_>>();

    (params, s, l, claims, witnesses, Poseidon2Transcript::new(label))
}

fn mock_backend() -> ProverComputeBackend {
    ProverComputeBackend::Mojo(MojoBackendConfig::new(DeviceApi::Cuda).with_library_path(build_mock_library()))
}

fn reset_mock_counters(lib: &Library) {
    type ResetFn = unsafe extern "C" fn();
    let reset = unsafe {
        *lib.get::<ResetFn>(b"nightstream_gpu_test_reset_counters\0")
            .expect("load counter reset symbol")
    };
    unsafe { reset() };
}

fn counter(lib: &Library, symbol: &[u8]) -> usize {
    type CounterFn = unsafe extern "C" fn() -> usize;
    let counter = unsafe { *lib.get::<CounterFn>(symbol).expect("load counter symbol") };
    unsafe { counter() }
}

fn assert_same_proof(lhs: &PiCcsProof, rhs: &PiCcsProof) {
    assert_eq!(lhs.variant, rhs.variant);
    assert_eq!(lhs.sumcheck_rounds, rhs.sumcheck_rounds);
    assert_eq!(lhs.sc_initial_sum, rhs.sc_initial_sum);
    assert_eq!(lhs.sumcheck_challenges, rhs.sumcheck_challenges);
    assert_eq!(lhs.sumcheck_rounds_nc, rhs.sumcheck_rounds_nc);
    assert_eq!(lhs.sc_initial_sum_nc, rhs.sc_initial_sum_nc);
    assert_eq!(lhs.sumcheck_challenges_nc, rhs.sumcheck_challenges_nc);
    assert_eq!(lhs.challenges_public.alpha, rhs.challenges_public.alpha);
    assert_eq!(lhs.challenges_public.beta_a, rhs.challenges_public.beta_a);
    assert_eq!(lhs.challenges_public.beta_r, rhs.challenges_public.beta_r);
    assert_eq!(lhs.challenges_public.beta_m, rhs.challenges_public.beta_m);
    assert_eq!(lhs.challenges_public.gamma, rhs.challenges_public.gamma);
    assert_eq!(lhs.sumcheck_final, rhs.sumcheck_final);
    assert_eq!(lhs.sumcheck_final_nc, rhs.sumcheck_final_nc);
    assert_eq!(lhs.header_digest, rhs.header_digest);
    assert_eq!(lhs._extra, rhs._extra);
}

#[test]
fn split_nc_fe_row_mojo_backend_matches_cpu_across_rounds() {
    let _guard = lock_mock_backend();
    let fixture = build_oracle_fixture();
    let backend = mock_backend();
    let backend_ctx = BackendContext::new(&backend).expect("backend context");

    let mock_library = build_mock_library();
    let lib = unsafe { Library::new(mock_library) }.expect("load mock mojo gpu library");
    reset_mock_counters(&lib);

    let mut cpu = OptimizedOracle::new_with_sparse(
        &fixture.s,
        &fixture.params,
        &fixture.mcs_witnesses,
        &fixture.me_witnesses,
        fixture.ch.clone(),
        fixture.ell_d,
        fixture.ell_n,
        fixture.d_sc,
        Some(&fixture.r_inputs),
        fixture.sparse.clone(),
    );
    let mut mojo = SplitNcOptimizedOracle::new_with_sparse(
        &fixture.s,
        &fixture.params,
        &fixture.mcs_witnesses,
        &fixture.me_witnesses,
        fixture.ch.clone(),
        fixture.ell_d,
        fixture.ell_n,
        fixture.d_sc,
        Some(&fixture.r_inputs),
        fixture.sparse.clone(),
        &backend_ctx,
    )
    .expect("split-nc mojo oracle");

    let total_rounds = cpu.num_rounds();
    for round in 0..total_rounds {
        let xs = vec![
            K::ZERO,
            K::ONE,
            k(50 + round as u64, 60 + round as u64),
            k(70 + round as u64, 80 + round as u64),
        ];
        assert_eq!(cpu.evals_at(&xs), mojo.evals_at(&xs), "round {round}");
        let r = k(90 + round as u64, 100 + round as u64);
        cpu.fold(r);
        mojo.fold(r);
    }

    assert!(
        counter(&lib, b"nightstream_gpu_test_fe_evals_at_calls\0") > 0,
        "expected FE row-phase to exercise the mock Split-NC evaluator"
    );
}

#[test]
fn split_nc_nc_col_mojo_backend_matches_cpu_across_rounds() {
    let _guard = lock_mock_backend();
    let fixture = build_oracle_fixture();
    let backend = mock_backend();
    let backend_ctx = BackendContext::new(&backend).expect("backend context");

    let mock_library = build_mock_library();
    let lib = unsafe { Library::new(mock_library) }.expect("load mock mojo gpu library");
    reset_mock_counters(&lib);

    let mut cpu = NcOracle::new(
        &fixture.s,
        &fixture.params,
        &fixture.mcs_witnesses,
        &fixture.me_witnesses,
        fixture.ch.clone(),
        fixture.ell_d,
        fixture.ell_m,
        fixture.d_sc,
    );
    let mut mojo = SplitNcNcOracle::new(
        &fixture.s,
        &fixture.params,
        &fixture.mcs_witnesses,
        &fixture.me_witnesses,
        fixture.ch.clone(),
        fixture.ell_d,
        fixture.ell_m,
        fixture.d_sc,
        &backend_ctx,
    )
    .expect("split-nc mojo nc oracle");

    let total_rounds = cpu.num_rounds();
    for round in 0..total_rounds {
        let xs = vec![
            K::ZERO,
            K::ONE,
            k(150 + round as u64, 160 + round as u64),
            k(170 + round as u64, 180 + round as u64),
        ];
        assert_eq!(cpu.evals_at(&xs), mojo.evals_at(&xs), "round {round}");
        let r = k(190 + round as u64, 200 + round as u64);
        cpu.fold(r);
        mojo.fold(r);
    }

    assert!(
        counter(&lib, b"nightstream_gpu_test_nc_evals_at_calls\0") > 0,
        "expected NC column-phase to exercise the mock Split-NC evaluator"
    );
}

#[test]
fn split_nc_optimized_prove_mojo_backend_matches_cpu_and_verifies() {
    let _guard = lock_mock_backend();
    let (params, s, l, claims, witnesses, mut cpu_tr) =
        build_prove_fixture(b"split_nc_gpu_parity/prove_cpu");
    let mut mojo_tr = Poseidon2Transcript::new(b"split_nc_gpu_parity/prove_cpu");
    let backend = mock_backend();

    let mock_library = build_mock_library();
    let lib = unsafe { Library::new(mock_library) }.expect("load mock mojo gpu library");
    reset_mock_counters(&lib);

    let (cpu_out, cpu_proof) = prove_with_backend(
        FoldingMode::Optimized,
        &mut cpu_tr,
        &params,
        &s,
        &claims,
        &witnesses,
        &[],
        &[],
        &l,
        &ProverComputeBackend::Cpu,
    )
    .expect("cpu prove");

    let (mojo_out, mojo_proof) = prove_with_backend(
        FoldingMode::Optimized,
        &mut mojo_tr,
        &params,
        &s,
        &claims,
        &witnesses,
        &[],
        &[],
        &l,
        &backend,
    )
    .expect("mojo prove");

    assert_eq!(cpu_out, mojo_out);
    assert_same_proof(&cpu_proof, &mojo_proof);
    assert!(
        counter(&lib, b"nightstream_gpu_test_fe_evals_at_calls\0") > 0,
        "expected FE evaluator use during optimized proof"
    );
    assert!(
        counter(&lib, b"nightstream_gpu_test_nc_evals_at_calls\0") > 0,
        "expected NC evaluator use during optimized proof"
    );

    let mut cpu_verify_tr = Poseidon2Transcript::new(b"split_nc_gpu_parity/prove_cpu");
    let ok_cpu = verify_with_backend(
        FoldingMode::Optimized,
        &mut cpu_verify_tr,
        &params,
        &s,
        &claims,
        &[],
        &cpu_out,
        &cpu_proof,
        &ProverComputeBackend::Cpu,
    )
    .expect("cpu verify");
    assert!(ok_cpu);

    let mut mojo_verify_tr = Poseidon2Transcript::new(b"split_nc_gpu_parity/prove_cpu");
    let ok_mojo = verify_with_backend(
        FoldingMode::Optimized,
        &mut mojo_verify_tr,
        &params,
        &s,
        &claims,
        &[],
        &mojo_out,
        &mojo_proof,
        &backend,
    )
    .expect("mojo verify");
    assert!(ok_mojo);
}

#[test]
fn split_nc_optimized_prove_falls_back_to_cpu_when_backend_is_missing() {
    let (params, s, l, claims, witnesses, mut cpu_tr) =
        build_prove_fixture(b"split_nc_gpu_parity/fallback_cpu");
    let mut fallback_tr = Poseidon2Transcript::new(b"split_nc_gpu_parity/fallback_cpu");

    let (cpu_out, cpu_proof) = prove_with_backend(
        FoldingMode::Optimized,
        &mut cpu_tr,
        &params,
        &s,
        &claims,
        &witnesses,
        &[],
        &[],
        &l,
        &ProverComputeBackend::Cpu,
    )
    .expect("cpu prove");

    let fallback_backend = ProverComputeBackend::Mojo(
        MojoBackendConfig::auto().with_library_path("/tmp/nightstream-mojo-gpu-does-not-exist.so"),
    );
    let (fallback_out, fallback_proof) = prove_with_backend(
        FoldingMode::Optimized,
        &mut fallback_tr,
        &params,
        &s,
        &claims,
        &witnesses,
        &[],
        &[],
        &l,
        &fallback_backend,
    )
    .expect("fallback prove");

    assert_eq!(cpu_out, fallback_out);
    assert_same_proof(&cpu_proof, &fallback_proof);
}

#[test]
#[ignore = "requires local Mojo toolchain"]
fn real_mojo_split_nc_optimized_prove_matches_cpu_and_verifies() {
    let (params, s, l, claims, witnesses, mut cpu_tr) =
        build_prove_fixture(b"split_nc_gpu_parity/real_mojo");
    let mut mojo_tr = Poseidon2Transcript::new(b"split_nc_gpu_parity/real_mojo");

    let (cpu_out, cpu_proof) = prove_with_backend(
        FoldingMode::Optimized,
        &mut cpu_tr,
        &params,
        &s,
        &claims,
        &witnesses,
        &[],
        &[],
        &l,
        &ProverComputeBackend::Cpu,
    )
    .expect("cpu prove");

    let mojo_backend =
        ProverComputeBackend::Mojo(MojoBackendConfig::new(DeviceApi::Cpu).with_library_path(build_real_mojo_library()));
    let (mojo_out, mojo_proof) = prove_with_backend(
        FoldingMode::Optimized,
        &mut mojo_tr,
        &params,
        &s,
        &claims,
        &witnesses,
        &[],
        &[],
        &l,
        &mojo_backend,
    )
    .expect("real mojo prove");

    assert_eq!(cpu_out, mojo_out);
    assert_same_proof(&cpu_proof, &mojo_proof);

    let mut verify_tr = Poseidon2Transcript::new(b"split_nc_gpu_parity/real_mojo");
    let ok = verify_with_backend(
        FoldingMode::Optimized,
        &mut verify_tr,
        &params,
        &s,
        &claims,
        &[],
        &mojo_out,
        &mojo_proof,
        &mojo_backend,
    )
    .expect("real mojo verify");
    assert!(ok);
}
