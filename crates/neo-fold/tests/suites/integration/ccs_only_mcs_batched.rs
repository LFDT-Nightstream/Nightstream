#![allow(non_snake_case)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::OnceLock;

use libloading::Library;
use neo_ajtai::{s_lincomb, s_mul, setup as ajtai_setup, AjtaiSModule, Commitment as Cmt};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, Mat, SparsePoly};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{
    fold_shard_prove, fold_shard_prove_ccs_only_batched, fold_shard_verify, fold_shard_verify_ccs_only_batched,
    CommitMixers,
};
use neo_fold::{MojoBackendConfig, ProverComputeBackend};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F, K};
use neo_memory::ajtai::{commit_cols_for_ccs_m, encode_vector_for_ccs_m};
use neo_memory::witness::{MemInstance, MemWitness, StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn default_mixers() -> Mixers {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        assert!(!cs.is_empty(), "mix_rhos_commits: empty commitments");
        let rq_els: Vec<RqEl> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }

    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        assert!(!cs.is_empty(), "combine_b_pows: empty commitments");
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for c in cs.iter().skip(1) {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, c);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }

    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn setup_ajtai_committer(params: &NeoParams, m: usize) -> AjtaiSModule {
    let m_commit = commit_cols_for_ccs_m(m);
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m_commit).expect("Ajtai setup should succeed");
    AjtaiSModule::new(Arc::new(pp))
}

fn identity_ccs(n: usize) -> CcsStructure<F> {
    CcsStructure::new(vec![Mat::identity(n)], SparsePoly::new(1, vec![])).expect("valid CCS")
}

fn build_step(params: &NeoParams, l: &AjtaiSModule, m: usize, m_in: usize, seed: u64) -> StepWitnessBundle<Cmt, F, K> {
    // SuperNeo packed NC checks require bounded coefficients; keep synthetic witnesses in {-1,0,1}.
    let z: Vec<F> = (0..m)
        .map(|i| match (seed.wrapping_add(i as u64)) % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        })
        .collect();
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    let Z = encode_vector_for_ccs_m(params, m, &z).expect("encode witness for CCS width");
    let c = l.commit(&Z);
    StepWitnessBundle::from((CcsClaim { c, x, m_in }, CcsWitness { w, Z }))
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

fn high_batch_params(n: usize) -> NeoParams {
    let base = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    NeoParams::new(
        base.q,
        base.eta,
        base.d,
        base.kappa,
        base.m,
        base.b,
        16,
        base.T,
        base.s,
        base.lambda,
    )
    .expect("high batch params")
}

#[test]
fn ccs_only_mcs_batched_k2_prove_verify() {
    let n = 8usize;
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    let steps: Vec<StepWitnessBundle<Cmt, F, K>> = (0..5usize)
        .map(|i| build_step(&params, &l, ccs.m, 2, 100 + (i as u64) * 100))
        .collect();
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

    // Reduction-level K>1 smoke: Π_CCS prove/verify must accept two MCS slots.
    {
        let mcs_list: Vec<CcsClaim<Cmt, F>> = steps[..2].iter().map(|s| s.mcs.0.clone()).collect();
        let mcs_wits: Vec<CcsWitness<F>> = steps[..2].iter().map(|s| s.mcs.1.clone()).collect();
        let mut tr0 = Poseidon2Transcript::new(b"neo.fold/session");
        let (out, pi) = neo_fold::pi_ccs::prove(
            FoldingMode::Optimized,
            &mut tr0,
            &params,
            &ccs,
            &mcs_list,
            &mcs_wits,
            &[],
            &[],
            &l,
        )
        .expect("pi_ccs prove k_mcs=2");
        let mut tr1 = Poseidon2Transcript::new(b"neo.fold/session");
        let ok = neo_fold::pi_ccs::verify(
            FoldingMode::Optimized,
            &mut tr1,
            &params,
            &ccs,
            &mcs_list,
            &[],
            &out,
            &pi,
        )
        .expect("pi_ccs verify result");
        assert!(ok, "pi_ccs verify should pass for k_mcs=2");
    }

    let mut tr_p = Poseidon2Transcript::new(b"neo.fold/session");
    let proof = fold_shard_prove_ccs_only_batched(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &ccs,
        &steps,
        &[],
        &[],
        &l,
        mixers,
        2,
        &ProverComputeBackend::Cpu,
    )
    .expect("prove");
    assert_eq!(proof.steps.len(), 3, "5 steps batched by 2 should yield 3 fold steps");

    let mut tr_v = Poseidon2Transcript::new(b"neo.fold/session");
    let outputs = fold_shard_verify_ccs_only_batched(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        &steps_public,
        &[],
        &proof,
        mixers,
        2,
    )
    .expect("verify");

    assert!(
        outputs.obligations.val.is_empty(),
        "ccs-only batched path must not emit val obligations"
    );
    assert_eq!(outputs.obligations.main.len(), params.k_rho as usize);
}

#[test]
fn ccs_only_default_shard_api_auto_batches() {
    let n = 8usize;
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    let steps: Vec<StepWitnessBundle<Cmt, F, K>> = (0..5usize)
        .map(|i| build_step(&params, &l, ccs.m, 2, 1000 + (i as u64) * 100))
        .collect();
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"neo.fold/session");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &ccs,
        &steps,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("prove");

    assert!(
        proof.steps.len() < steps.len(),
        "default shard API should auto-batch ccs-only steps when safe"
    );

    let mut tr_v = Poseidon2Transcript::new(b"neo.fold/session");
    let outputs = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        &steps_public,
        &[],
        &proof,
        mixers,
    )
    .expect("verify");

    assert!(
        outputs.obligations.val.is_empty(),
        "ccs-only path must not emit val obligations"
    );
    assert_eq!(outputs.obligations.main.len(), params.k_rho as usize);
}

#[test]
fn ccs_only_mcs_batched_mojo_backend_uses_batch_poseidon_for_rlc_binding() {
    type ResetFn = unsafe extern "C" fn();
    type CounterFn = unsafe extern "C" fn() -> usize;
    let _counter_guard = super::lock_mock_backend_counters();

    let n = 8usize;
    let ccs = identity_ccs(n);
    let params = high_batch_params(n);
    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();
    let batch_size = 40usize;

    let steps: Vec<StepWitnessBundle<Cmt, F, K>> = (0..batch_size)
        .map(|i| build_step(&params, &l, ccs.m, 2, 10_000 + (i as u64) * 97))
        .collect();
    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

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

    let mut tr_cpu = Poseidon2Transcript::new(b"neo.fold/ccs_only_gpu_rlc");
    let cpu_proof = fold_shard_prove_ccs_only_batched(
        FoldingMode::Optimized,
        &mut tr_cpu,
        &params,
        &ccs,
        &steps,
        &[],
        &[],
        &l,
        mixers,
        batch_size,
        &ProverComputeBackend::Cpu,
    )
    .expect("cpu prove");

    unsafe { reset() };
    let backend = ProverComputeBackend::Mojo(MojoBackendConfig::auto().with_library_path(mock_library));
    let mut tr_mojo = Poseidon2Transcript::new(b"neo.fold/ccs_only_gpu_rlc");
    let mojo_proof = fold_shard_prove_ccs_only_batched(
        FoldingMode::Optimized,
        &mut tr_mojo,
        &params,
        &ccs,
        &steps,
        &[],
        &[],
        &l,
        mixers,
        batch_size,
        &backend,
    )
    .expect("mojo prove");

    assert_eq!(
        serde_json::to_vec(&cpu_proof).expect("serialize cpu proof"),
        serde_json::to_vec(&mojo_proof).expect("serialize mojo proof"),
    );
    assert!(
        unsafe { batch_calls() } > 0,
        "expected mock mojo backend to use batched Poseidon2 for RLC input binding"
    );

    let mut tr_v = Poseidon2Transcript::new(b"neo.fold/ccs_only_gpu_rlc");
    let outputs = fold_shard_verify_ccs_only_batched(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        &steps_public,
        &[],
        &mojo_proof,
        mixers,
        batch_size,
    )
    .expect("verify");

    assert!(outputs.obligations.val.is_empty());
    assert_eq!(outputs.obligations.main.len(), params.k_rho as usize);
}

#[test]
fn ccs_only_mcs_batched_rejects_sidecars() {
    let n = 8usize;
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    let mut step = build_step(&params, &l, ccs.m, 2, 1234);
    step.mem_instances.push((
        MemInstance::<Cmt, F> {
            mem_id: 0,
            comms: Vec::new(),
            k: 2,
            d: 1,
            n_side: 2,
            steps: 1,
            lanes: 1,
            ell: 1,
            init: MemInit::Zero,
            init_digest: None,
            guest_addr_remap: None,
        },
        MemWitness { mats: Vec::new() },
    ));

    let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
    let err = fold_shard_prove_ccs_only_batched(
        FoldingMode::Optimized,
        &mut tr,
        &params,
        &ccs,
        &[step],
        &[],
        &[],
        &l,
        mixers,
        2,
        &ProverComputeBackend::Cpu,
    )
    .expect_err("sidecar steps should be rejected");

    assert!(
        err.to_string()
            .contains("ccs-only batching does not support mem/lut sidecars"),
        "unexpected error: {err}"
    );
}
