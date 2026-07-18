//! Genuine Nebula memory-lane CPU lifecycle baseline for M0.

use std::time::{Duration, Instant};

use neo_ajtai::{has_global_pp_for_dims, set_global_pp_seeded, Commitment};
use neo_ccs::{CcsStructure, LaneCommitments, Mat, SparsePoly};
use neo_fold_clean::config;
use neo_fold_clean::frontends::nebula::layout::StepPublicInput;
use neo_fold_clean::lifecycle::{
    self, extend, extend_nebula_open, extend_nebula_open_with_nifs_adapter, extend_with_nifs_adapter,
    finish_uncompressed_with_audit, finish_uncompressed_with_audit_and_nifs_adapter, preprocess, Preprocessing,
    UncompressedAudit,
};
use neo_fold_clean::paper::construction2::{NebulaConfig, StackShape};
use neo_fold_clean::paper::digest;
use neo_fold_clean::paper::relations::{CcsInstance, LaneRanges, LaneScheme};
use neo_math::{D, F, K};
use neo_prover_metal::MetalNifsProver;
use p3_field::PrimeCharacteristicRing;

use crate::parity::audit_authority_eq;
use crate::report::{summarize_nifs_profiles, BenchmarkError, LifecycleReport, NifsProfileSample, TimingSummary};

const STEPS_PER_SEGMENT: u64 = 2;
const M_IN: usize = 1_401;
const LANE_COLS: LaneRanges = LaneRanges {
    ops: 26..27,
    is: 27..28,
    fs: 28..29,
};
const M: usize = 29 * D;

struct NebulaFixture {
    prep: Preprocessing,
    advs: Vec<LaneCommitments<Commitment>>,
    d_pre: [[F; 4]; 3],
    setup: Duration,
}

pub fn run_nebula_cpu(repetitions: usize) -> Result<LifecycleReport, BenchmarkError> {
    let fixture = build_fixture()?;
    let mut online = Vec::with_capacity(repetitions);
    let mut verify = Vec::with_capacity(repetitions);
    let mut audit_bytes = 0;
    let mut semantic_result_ok = true;
    let warmup = honest_chain(&fixture)?;
    neo_fold_clean::verify_uncompressed_audit(&fixture.prep, &warmup)
        .map_err(|error| BenchmarkError::Lifecycle(format!("verify Nebula warm-up: {error}")))?;
    for _ in 0..repetitions {
        let started = Instant::now();
        let audit = honest_chain(&fixture)?;
        online.push(started.elapsed());
        let lane = audit.proof.state.nebula.as_ref();
        semantic_result_ok &= lane.is_some_and(|lane| lane.is_closed() && lane.seg_idx == 1 && lane.ts == 2);
        audit_bytes = format!("{audit:?}").len();

        let started = Instant::now();
        neo_fold_clean::verify_uncompressed_audit(&fixture.prep, &audit)
            .map_err(|error| BenchmarkError::Lifecycle(format!("verify Nebula audit: {error}")))?;
        verify.push(started.elapsed());
    }
    Ok(LifecycleReport {
        name: "nebula_memory_lane_2_step".to_owned(),
        backend: "CPU".to_owned(),
        verification_mode: "full_history_audit_replay".to_owned(),
        synthesis_ms: 0.0,
        preprocessing_ms: fixture.setup.as_secs_f64() * 1e3,
        online: TimingSummary::from_durations(online),
        pipeline: None,
        verify_ms: TimingSummary::from_durations(verify),
        nifs_profile: None,
        audit_debug_chars: audit_bytes,
        semantic_result_ok,
        proof_parity_ok: true,
    })
}

pub fn run_nebula_metal(repetitions: usize) -> Result<LifecycleReport, BenchmarkError> {
    let fixture = build_fixture()?;
    let mut metal = MetalNifsProver::new()?;
    let mut online = Vec::with_capacity(repetitions);
    let mut verify = Vec::with_capacity(repetitions);
    let mut audit_bytes = 0;
    let mut semantic_result_ok = true;
    let mut proof_parity_ok = true;
    let mut profile_samples = Vec::with_capacity(repetitions);
    let cpu_reference = honest_chain(&fixture)?;
    let (warmup, _) = honest_chain_metal(&fixture, &mut metal)?;
    neo_fold_clean::verify_uncompressed_audit(&fixture.prep, &warmup)
        .map_err(|error| BenchmarkError::Lifecycle(format!("verify Metal Nebula warm-up: {error}")))?;
    for _ in 0..repetitions {
        let started = Instant::now();
        let (audit, profiles) = honest_chain_metal(&fixture, &mut metal)?;
        online.push(started.elapsed());
        profile_samples.push(NifsProfileSample::from_profiles(profiles));
        let lane = audit.proof.state.nebula.as_ref();
        semantic_result_ok &= lane.is_some_and(|lane| lane.is_closed() && lane.seg_idx == 1 && lane.ts == 2);
        proof_parity_ok &= audit_authority_eq(&audit, &cpu_reference);
        audit_bytes = format!("{audit:?}").len();
        let started = Instant::now();
        neo_fold_clean::verify_uncompressed_audit(&fixture.prep, &audit)
            .map_err(|error| BenchmarkError::Lifecycle(format!("verify Metal Nebula audit: {error}")))?;
        verify.push(started.elapsed());
    }
    Ok(LifecycleReport {
        name: "nebula_memory_lane_2_step".to_owned(),
        backend: "MetalNifsProver".to_owned(),
        verification_mode: "full_history_audit_replay".to_owned(),
        synthesis_ms: 0.0,
        preprocessing_ms: fixture.setup.as_secs_f64() * 1e3,
        online: TimingSummary::from_durations(online),
        pipeline: None,
        verify_ms: TimingSummary::from_durations(verify),
        nifs_profile: Some(summarize_nifs_profiles(profile_samples)),
        audit_debug_chars: audit_bytes,
        semantic_result_ok,
        proof_parity_ok,
    })
}

fn build_fixture() -> Result<NebulaFixture, BenchmarkError> {
    let started = Instant::now();
    let structure = CcsStructure::new(vec![Mat::identity(M)], SparsePoly::new(1, vec![]))
        .map_err(|error| BenchmarkError::Lifecycle(format!("build Nebula structure: {error}")))?;
    let params = config::r1cs_params(structure.n, structure.m)
        .map_err(|error| BenchmarkError::Lifecycle(format!("derive Nebula params: {error}")))?;
    install_ajtai(&params, &structure)?;
    let base = preprocess(params, structure, Some(M_IN))
        .map_err(|error| BenchmarkError::Lifecycle(format!("preprocess Nebula: {error}")))?;
    let scheme = LaneScheme::from_seeds(base.params.kappa() as usize, LANE_COLS, [0xA7; 32], [0x7A; 32])
        .map_err(|error| BenchmarkError::Lifecycle(format!("build Nebula lane scheme: {error}")))?;
    let (advs, d_pre) = precommit(&base, &scheme)?;
    let config = NebulaConfig {
        scheme,
        steps_per_segment: STEPS_PER_SEGMENT,
        seg_max: 1,
        stacks: StackShape::NONE,
        plan_digest: [F::from_u64(11); 4],
        d_init: d_pre[1],
    };
    Ok(NebulaFixture {
        prep: base.with_nebula(config),
        advs,
        d_pre,
        setup: started.elapsed(),
    })
}

fn honest_chain(fixture: &NebulaFixture) -> Result<UncompressedAudit, BenchmarkError> {
    let audit = lifecycle::prove(&fixture.prep, Vec::<Vec<CcsInstance>>::new())
        .map_err(|error| BenchmarkError::Lifecycle(format!("start Nebula chain: {error}")))?;
    let gamma = derive_gamma(&fixture.prep, &audit, fixture.d_pre)?;
    let audit = extend_nebula_open(
        &fixture.prep,
        audit,
        vec![step_instance(&fixture.prep, gamma, 0, &fixture.advs[0])?],
        fixture.d_pre,
    )
    .map_err(|error| BenchmarkError::Lifecycle(format!("open Nebula segment: {error}")))?;
    let audit = extend(
        &fixture.prep,
        audit,
        vec![step_instance(&fixture.prep, gamma, 1, &fixture.advs[1])?],
    )
    .map_err(|error| BenchmarkError::Lifecycle(format!("extend Nebula segment: {error}")))?;
    finish_uncompressed_with_audit(&fixture.prep, audit)
        .map_err(|error| BenchmarkError::Lifecycle(format!("finish Nebula chain: {error}")))
}

fn honest_chain_metal(
    fixture: &NebulaFixture,
    metal: &mut MetalNifsProver,
) -> Result<(UncompressedAudit, Vec<neo_prover_metal::MetalNifsProfile>), BenchmarkError> {
    let _ = metal.take_last_profile();
    let mut profiles = Vec::new();
    let audit = lifecycle::prove_with_nifs_adapter(&fixture.prep, metal, Vec::<Vec<CcsInstance>>::new())
        .map_err(|error| BenchmarkError::Lifecycle(format!("start Metal Nebula chain: {error}")))?;
    let gamma = derive_gamma(&fixture.prep, &audit, fixture.d_pre)?;
    let audit = extend_nebula_open_with_nifs_adapter(
        &fixture.prep,
        metal,
        audit,
        vec![step_instance(&fixture.prep, gamma, 0, &fixture.advs[0])?],
        fixture.d_pre,
    )
    .map_err(|error| BenchmarkError::Lifecycle(format!("open Metal Nebula segment: {error}")))?;
    if let Some(profile) = metal.take_last_profile() {
        profiles.push(profile);
    }
    let audit = extend_with_nifs_adapter(
        &fixture.prep,
        metal,
        audit,
        vec![step_instance(&fixture.prep, gamma, 1, &fixture.advs[1])?],
    )
    .map_err(|error| BenchmarkError::Lifecycle(format!("extend Metal Nebula segment: {error}")))?;
    if let Some(profile) = metal.take_last_profile() {
        profiles.push(profile);
    }
    let audit = finish_uncompressed_with_audit_and_nifs_adapter(&fixture.prep, metal, audit)
        .map_err(|error| BenchmarkError::Lifecycle(format!("finish Metal Nebula chain: {error}")))?;
    if let Some(profile) = metal.take_last_profile() {
        profiles.push(profile);
    }
    Ok((audit, profiles))
}

fn precommit(
    prep: &Preprocessing,
    scheme: &LaneScheme,
) -> Result<(Vec<LaneCommitments<Commitment>>, [[F; 4]; 3]), BenchmarkError> {
    let mut advs = Vec::new();
    for step in 0..STEPS_PER_SEGMENT {
        let mut assignment = vec![F::ZERO; M];
        assignment[26 * D..].copy_from_slice(&lane_bits(step));
        let instance =
            CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &assignment, M_IN)
                .map_err(|error| BenchmarkError::Lifecycle(format!("precommit Nebula lane: {error}")))?;
        advs.push(
            scheme
                .commit(&instance.witness.Z)
                .map_err(|error| BenchmarkError::Lifecycle(format!("commit Nebula lane: {error}")))?,
        );
    }
    let memory_header = digest::nebula_chain_mem_header();
    let mut chains = [digest::nebula_chain_ops_header(), memory_header, memory_header];
    let tags: [&[u8]; 3] = [
        digest::NEBULA_CHAIN_OPS_TAG,
        digest::NEBULA_CHAIN_MEM_TAG,
        digest::NEBULA_CHAIN_MEM_TAG,
    ];
    for adv in &advs {
        let leaves = digest::nebula_lane_leaf_digests(adv);
        for lane in 0..3 {
            chains[lane] = digest::nebula_chain_link(&chains[lane], tags[lane], &leaves[lane]);
        }
    }
    Ok((advs, chains))
}

fn step_instance(
    prep: &Preprocessing,
    gamma: [K; 2],
    step: u64,
    adv: &LaneCommitments<Commitment>,
) -> Result<CcsInstance, BenchmarkError> {
    let input = StepPublicInput {
        seg_idx: 0,
        idx: step,
        ts_in: step,
        ts_out: step + 1,
        gamma,
        h_in: [K::ONE; 4],
        h_out: [K::ONE; 4],
        sp_in: [0; 2],
        sp_out: [0; 2],
    };
    let bits = input
        .encode(StackShape::NONE)
        .map_err(|error| BenchmarkError::Lifecycle(format!("encode Nebula input: {error}")))?;
    let mut assignment = vec![F::ZERO; M];
    assignment[0] = F::ONE;
    assignment[1..1 + bits.len()].copy_from_slice(&bits);
    assignment[26 * D..].copy_from_slice(&lane_bits(step));
    let mut instance =
        CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &assignment, M_IN)
            .map_err(|error| BenchmarkError::Lifecycle(format!("build Nebula instance: {error}")))?;
    instance.claim.adv = Some(adv.clone());
    Ok(instance)
}

fn derive_gamma(prep: &Preprocessing, audit: &UncompressedAudit, d_pre: [[F; 4]; 3]) -> Result<[K; 2], BenchmarkError> {
    let state = &audit.proof.state;
    let mut lane = state
        .nebula
        .clone()
        .ok_or_else(|| BenchmarkError::Lifecycle("Nebula base state omitted its lane".to_owned()))?;
    lane.open_segment(
        prep.nebula()
            .ok_or_else(|| BenchmarkError::Lifecycle("Nebula preprocessing omitted config".to_owned()))?,
        prep.vk.digest(),
        state.z_i,
        state.acc_digest,
        d_pre,
    )
    .map_err(|error| BenchmarkError::Lifecycle(format!("derive Nebula gamma: {error}")))?;
    lane.gamma
        .ok_or_else(|| BenchmarkError::Lifecycle("Nebula segment did not produce gamma".to_owned()))
}

fn lane_bits(step: u64) -> Vec<F> {
    (0..(3 * D) as u64)
        .map(|index| {
            F::from_u64(
                (step
                    .wrapping_mul(0x9E37)
                    .wrapping_add(index)
                    .rotate_left((index % 13) as u32))
                    & 1,
            )
        })
        .collect()
}

fn install_ajtai(params: &neo_fold_clean::Params, structure: &neo_fold_clean::Structure) -> Result<(), BenchmarkError> {
    let cols = structure.m.div_ceil(D);
    if has_global_pp_for_dims(D, cols) {
        return Ok(());
    }
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0x4e45_4f46_4f4c_4431_u64.to_le_bytes());
    match set_global_pp_seeded(D, params.kappa() as usize, cols, seed) {
        Ok(()) => Ok(()),
        Err(_) if has_global_pp_for_dims(D, cols) => Ok(()),
        Err(error) => Err(BenchmarkError::Lifecycle(format!(
            "install Nebula Ajtai parameters: {error}"
        ))),
    }
}
