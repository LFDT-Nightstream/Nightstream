#![allow(non_snake_case)]

#[path = "../../common/fixtures.rs"]
mod fixtures;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use fixtures::build_twist_shout_2step_fixture;
use libloading::Library;
use neo_ccs::relations::{CcsClaim, CcsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{
    fold_shard_prove, fold_shard_prove_with_backend, fold_shard_verify, fold_shard_verify_with_backend,
    ShardSegmentKind,
};
use neo_fold::{MojoBackendConfig, ProverComputeBackend};
use neo_math::{F, K};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn build_ccs_only_step(fx: &fixtures::ShardFixture, salt: u64) -> StepWitnessBundle<neo_ajtai::Commitment, F, K> {
    let m = fx.ccs.m;
    let m_in = fx.steps_witness[0].mcs.0.m_in;
    // Keep CCS-only synthetic steps in a bounded coefficient range for SuperNeo NC validity.
    let z: Vec<F> = (0..m)
        .map(|i| match (salt.wrapping_add(i as u64)) % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        })
        .collect();
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    let Z = neo_memory::ajtai::encode_vector_for_ccs_m(&fx.params, z.len(), &z).expect("encode witness for CCS width");
    let c = fx.l.commit(&Z);
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

#[test]
fn mixed_ccs_only_and_route_a_segments_prove_verify() {
    let fx = build_twist_shout_2step_fixture(123);

    let mut steps_witness: Vec<StepWitnessBundle<neo_ajtai::Commitment, F, K>> =
        vec![build_ccs_only_step(&fx, 100), build_ccs_only_step(&fx, 200)];
    steps_witness.extend(fx.steps_witness.iter().cloned());
    steps_witness.push(build_ccs_only_step(&fx, 300));

    let steps_instance: Vec<StepInstanceBundle<neo_ajtai::Commitment, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"mixed-ccs-route-a/segment-test");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    assert!(
        proof
            .steps
            .iter()
            .any(|step| !step.mem.proofs.is_empty() || !step.batched_time.claimed_sums.is_empty()),
        "expected at least one Route-A step proof in mixed segmentation",
    );
    assert!(
        proof
            .steps
            .iter()
            .any(|step| step.mem.proofs.is_empty() && step.batched_time.claimed_sums.is_empty()),
        "expected at least one ccs-only batched step proof in mixed segmentation",
    );
    let seg = proof.segment_meta.as_ref().expect("segment_meta");
    assert_eq!(seg.len(), 3, "expected [CCS-only, Route-A, CCS-only] segments");
    assert_eq!(seg[0].kind, ShardSegmentKind::CcsOnly);
    assert_eq!(seg[0].public_steps, 2);
    assert!(seg[0].proof_steps >= 1);
    assert_eq!(seg[1].kind, ShardSegmentKind::RouteA);
    assert_eq!(seg[1].public_steps, 2);
    assert_eq!(seg[1].proof_steps, 1);
    assert_eq!(seg[2].kind, ShardSegmentKind::CcsOnly);
    assert_eq!(seg[2].public_steps, 1);
    assert_eq!(seg[2].proof_steps, 1);

    let mut tr_v = Poseidon2Transcript::new(b"mixed-ccs-route-a/segment-test");
    let outputs = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &fx.params,
        &fx.ccs,
        &steps_instance,
        &fx.acc_init,
        &proof,
        fx.mixers,
    )
    .expect("verify");

    assert_eq!(outputs.obligations.main.len(), fx.params.k_rho as usize);
}

#[test]
fn mixed_ccs_only_and_route_a_segments_mojo_backend_matches_cpu() {
    type ResetFn = unsafe extern "C" fn();
    type CounterFn = unsafe extern "C" fn() -> usize;
    let _counter_guard = super::lock_mock_backend_counters();

    let fx = build_twist_shout_2step_fixture(9001);
    let mut steps_witness: Vec<StepWitnessBundle<neo_ajtai::Commitment, F, K>> = (0..20)
        .map(|i| build_ccs_only_step(&fx, 1_000 + i as u64))
        .collect();
    steps_witness.extend(fx.steps_witness.iter().cloned());
    steps_witness.extend((0..20).map(|i| build_ccs_only_step(&fx, 2_000 + i as u64)));
    let steps_instance: Vec<StepInstanceBundle<neo_ajtai::Commitment, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_cpu = Poseidon2Transcript::new(b"mixed-ccs-route-a/mojo-backend");
    let cpu_proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_cpu,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("cpu prove");

    let mock_library = build_mock_library();
    let lib = unsafe { Library::new(mock_library) }.expect("load mock mojo gpu library");
    let reset = unsafe {
        *lib.get::<ResetFn>(b"nightstream_gpu_test_reset_counters\0")
            .expect("load counter reset symbol")
    };
    let poseidon2_batch_calls = unsafe {
        *lib.get::<CounterFn>(b"nightstream_gpu_test_poseidon2_batch_calls\0")
            .expect("load poseidon2 batch counter symbol")
    };
    let session_open_calls = unsafe {
        *lib.get::<CounterFn>(b"nightstream_gpu_test_session_open_calls\0")
            .expect("load session open counter symbol")
    };
    unsafe { reset() };

    let backend = ProverComputeBackend::Mojo(MojoBackendConfig::auto().with_library_path(mock_library));
    let mut tr_mojo = Poseidon2Transcript::new(b"mixed-ccs-route-a/mojo-backend");
    let mojo_proof = fold_shard_prove_with_backend(
        FoldingMode::Optimized,
        &mut tr_mojo,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
        &backend,
    )
    .expect("mojo prove");

    assert_eq!(
        serde_json::to_vec(&cpu_proof).expect("serialize cpu proof"),
        serde_json::to_vec(&mojo_proof).expect("serialize mojo proof"),
    );

    let mut tr_verify = Poseidon2Transcript::new(b"mixed-ccs-route-a/mojo-backend");
    let mojo_outputs = fold_shard_verify_with_backend(
        FoldingMode::Optimized,
        &mut tr_verify,
        &fx.params,
        &fx.ccs,
        &steps_instance,
        &fx.acc_init,
        &mojo_proof,
        fx.mixers,
        &backend,
    )
    .expect("mojo verify");

    assert_eq!(mojo_outputs.obligations.main.len(), fx.params.k_rho as usize);
    assert!(
        unsafe { poseidon2_batch_calls() } > 0,
        "mock mojo backend should use batched Poseidon2 in mixed shard flow"
    );
    assert_eq!(unsafe { session_open_calls() }, 2);
}

#[test]
fn mixed_verify_rejects_missing_segment_meta() {
    let fx = build_twist_shout_2step_fixture(456);

    let mut steps_witness: Vec<StepWitnessBundle<neo_ajtai::Commitment, F, K>> =
        vec![build_ccs_only_step(&fx, 11), build_ccs_only_step(&fx, 22)];
    steps_witness.extend(fx.steps_witness.iter().cloned());
    let steps_instance: Vec<StepInstanceBundle<neo_ajtai::Commitment, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"mixed-ccs-route-a/segment-meta-required");
    let mut proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("prove");
    proof.segment_meta = None;

    let mut tr_v = Poseidon2Transcript::new(b"mixed-ccs-route-a/segment-meta-required");
    let err = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &fx.params,
        &fx.ccs,
        &steps_instance,
        &fx.acc_init,
        &proof,
        fx.mixers,
    )
    .expect_err("missing segment_meta must fail");

    assert!(
        err.to_string().contains("requires segment_meta"),
        "unexpected error: {err}"
    );
}

#[test]
fn mixed_verify_rejects_tampered_route_a_kind_in_segment_meta() {
    let fx = build_twist_shout_2step_fixture(789);

    let mut steps_witness: Vec<StepWitnessBundle<neo_ajtai::Commitment, F, K>> =
        vec![build_ccs_only_step(&fx, 31), build_ccs_only_step(&fx, 32)];
    steps_witness.extend(fx.steps_witness.iter().cloned());
    let steps_instance: Vec<StepInstanceBundle<neo_ajtai::Commitment, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"mixed-ccs-route-a/tamper-route-kind");
    let mut proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    let seg = proof.segment_meta.as_mut().expect("segment_meta");
    let route_idx = seg
        .iter()
        .position(|e| e.kind == ShardSegmentKind::RouteA)
        .expect("route-a segment metadata entry");
    seg[route_idx].kind = ShardSegmentKind::CcsOnly;

    let mut tr_v = Poseidon2Transcript::new(b"mixed-ccs-route-a/tamper-route-kind");
    let err = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &fx.params,
        &fx.ccs,
        &steps_instance,
        &fx.acc_init,
        &proof,
        fx.mixers,
    )
    .expect_err("tampered route-a kind must fail");

    assert!(err.to_string().contains("marked CcsOnly"), "unexpected error: {err}");
}

#[test]
fn mixed_verify_rejects_truncated_segment_meta() {
    let fx = build_twist_shout_2step_fixture(790);

    let mut steps_witness: Vec<StepWitnessBundle<neo_ajtai::Commitment, F, K>> =
        vec![build_ccs_only_step(&fx, 41), build_ccs_only_step(&fx, 42)];
    steps_witness.extend(fx.steps_witness.iter().cloned());
    let steps_instance: Vec<StepInstanceBundle<neo_ajtai::Commitment, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"mixed-ccs-route-a/truncated-segment-meta");
    let mut proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    let seg = proof.segment_meta.as_mut().expect("segment_meta");
    seg.pop();

    let mut tr_v = Poseidon2Transcript::new(b"mixed-ccs-route-a/truncated-segment-meta");
    let err = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &fx.params,
        &fx.ccs,
        &steps_instance,
        &fx.acc_init,
        &proof,
        fx.mixers,
    )
    .expect_err("truncated segment_meta must fail");

    assert!(err.to_string().contains("consumed"), "unexpected error: {err}");
}

#[test]
fn mixed_verify_rejects_tampered_route_a_proof_steps() {
    let fx = build_twist_shout_2step_fixture(791);

    let mut steps_witness: Vec<StepWitnessBundle<neo_ajtai::Commitment, F, K>> =
        vec![build_ccs_only_step(&fx, 51), build_ccs_only_step(&fx, 52)];
    steps_witness.extend(fx.steps_witness.iter().cloned());
    let steps_instance: Vec<StepInstanceBundle<neo_ajtai::Commitment, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"mixed-ccs-route-a/tamper-route-proof-steps");
    let mut proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    let seg = proof.segment_meta.as_mut().expect("segment_meta");
    let route_idx = seg
        .iter()
        .position(|e| e.kind == ShardSegmentKind::RouteA)
        .expect("route-a segment metadata entry");
    seg[route_idx].proof_steps += 1;

    let mut tr_v = Poseidon2Transcript::new(b"mixed-ccs-route-a/tamper-route-proof-steps");
    let err = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &fx.params,
        &fx.ccs,
        &steps_instance,
        &fx.acc_init,
        &proof,
        fx.mixers,
    )
    .expect_err("tampered route-a proof_steps must fail");

    assert!(
        err.to_string()
            .contains("proof too short for segment_meta entry"),
        "unexpected error: {err}"
    );
}
