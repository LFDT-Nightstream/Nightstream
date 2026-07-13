//! M0 CPU lifecycle baselines using production-core protocol parameters.

use std::hint::black_box;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use neo_fold_clean::frontends::bellpepper::BellpepperCcs;
use neo_fold_clean::frontends::r1cs_f_prime::{self, R1csChainBuilder, R1csFPrimePreprocessing};
use neo_fold_clean::paper::digest::digest_fields_as_digest32;
use neo_fold_clean::paper::params::Params;
use neo_prover_metal::MetalNifsProver;

use crate::report::{
    summarize_nifs_profiles, BenchmarkConfig, BenchmarkError, LifecyclePipelineReport, LifecycleReport,
    NifsProfileSample, SustainedLifecycleReport, TimingSummary,
};
use crate::sha256::{
    initial_sha_state, packed_state_derived_structure, serial_chunk, serial_state_lanes56_semantic_digest,
    sha_state_trace, SHA256_SERIAL_AJTAI_SEED,
};

const TRANSITIONS_PER_CHUNK: usize = 2;
const CHUNK_COUNT: usize = 4;

struct Sha256Fixture {
    states: Vec<Vec<u8>>,
    prep: R1csFPrimePreprocessing,
    setup_synthesis: Duration,
    preprocessing: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
struct PipelineSample {
    synthesis_work: Duration,
    fold_work: Duration,
    final_materialization: Duration,
    overlap_saved: Duration,
}

impl PipelineSample {
    fn summarize(samples: &[Self]) -> LifecyclePipelineReport {
        LifecyclePipelineReport {
            synthesis_work: TimingSummary::from_durations(samples.iter().map(|sample| sample.synthesis_work).collect()),
            fold_work: TimingSummary::from_durations(samples.iter().map(|sample| sample.fold_work).collect()),
            final_materialization: TimingSummary::from_durations(
                samples
                    .iter()
                    .map(|sample| sample.final_materialization)
                    .collect(),
            ),
            overlap_saved: TimingSummary::from_durations(samples.iter().map(|sample| sample.overlap_saved).collect()),
        }
    }
}

pub fn run_lifecycle_benchmarks(config: &BenchmarkConfig) -> Result<Vec<LifecycleReport>, BenchmarkError> {
    let mut reports = Vec::new();
    if config.run_sha256_lifecycle {
        reports.push(run_sha256_cpu(config.lifecycle_repetitions)?);
        #[cfg(target_vendor = "apple")]
        reports.push(run_sha256_metal(config.lifecycle_repetitions)?);
    }
    if config.run_nebula_lifecycle {
        reports.push(super::nebula::run_nebula_cpu(config.lifecycle_repetitions)?);
        #[cfg(target_vendor = "apple")]
        reports.push(super::nebula::run_nebula_metal(config.lifecycle_repetitions)?);
    }
    Ok(reports)
}

#[cfg(target_vendor = "apple")]
pub(crate) fn run_sha256_sustained(seconds: usize) -> Result<SustainedLifecycleReport, BenchmarkError> {
    if seconds == 0 {
        return Err(BenchmarkError::Config("sustained lifecycle duration must be nonzero"));
    }
    let fixture = build_sha256_fixture()?;
    let target = Duration::from_secs(seconds as u64);

    let (reference, _, _) = prove_sha(&fixture)?;
    neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, &reference)
        .map_err(|error| BenchmarkError::Lifecycle(format!("verify sustained CPU reference: {error}")))?;
    let reference_debug = format!("{reference:?}");

    let cpu_started = Instant::now();
    let mut cpu_proofs = 0usize;
    let mut last_cpu = None;
    while cpu_started.elapsed() < target {
        let (audit, _, _) = prove_sha(&fixture)?;
        last_cpu = Some(audit);
        cpu_proofs += 1;
    }
    let cpu_elapsed = cpu_started.elapsed();
    let last_cpu =
        last_cpu.ok_or_else(|| BenchmarkError::Lifecycle("sustained CPU run produced no proof".to_owned()))?;
    neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, &last_cpu)
        .map_err(|error| BenchmarkError::Lifecycle(format!("verify sustained CPU proof: {error}")))?;

    let mut metal = MetalNifsProver::new()?;
    let (metal_warmup, _, _, _) = prove_sha_metal(&fixture, &mut metal)?;
    neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, &metal_warmup)
        .map_err(|error| BenchmarkError::Lifecycle(format!("verify sustained Metal warm-up: {error}")))?;
    let metal_started = Instant::now();
    let mut metal_proofs = 0usize;
    let mut last_metal = None;
    while metal_started.elapsed() < target {
        let (audit, _, _, _) = prove_sha_metal(&fixture, &mut metal)?;
        last_metal = Some(audit);
        metal_proofs += 1;
    }
    let metal_elapsed = metal_started.elapsed();
    let last_metal =
        last_metal.ok_or_else(|| BenchmarkError::Lifecycle("sustained Metal run produced no proof".to_owned()))?;
    neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, &last_metal)
        .map_err(|error| BenchmarkError::Lifecycle(format!("verify sustained Metal proof: {error}")))?;

    let proof_parity_ok = format!("{last_cpu:?}") == reference_debug && format!("{last_metal:?}") == reference_debug;
    let cpu_rate = cpu_proofs as f64 / cpu_elapsed.as_secs_f64();
    let metal_rate = metal_proofs as f64 / metal_elapsed.as_secs_f64();
    let speedup = metal_rate / cpu_rate;
    let target_speedup = 1.15;
    Ok(SustainedLifecycleReport {
        name: "sha256_serial_4_chunk".to_owned(),
        seconds_per_backend: seconds,
        cpu_elapsed_ms: cpu_elapsed.as_secs_f64() * 1e3,
        metal_elapsed_ms: metal_elapsed.as_secs_f64() * 1e3,
        cpu_proofs,
        metal_proofs,
        cpu_proofs_per_second: cpu_rate,
        metal_proofs_per_second: metal_rate,
        speedup_over_cpu: speedup,
        target_speedup,
        proof_parity_ok,
        passed: proof_parity_ok && speedup >= target_speedup,
    })
}

fn run_sha256_metal(repetitions: usize) -> Result<LifecycleReport, BenchmarkError> {
    let fixture = build_sha256_fixture()?;
    let expected_digest = serial_state_lanes56_semantic_digest(
        fixture
            .states
            .last()
            .ok_or_else(|| BenchmarkError::Lifecycle("SHA state trace is empty".to_owned()))?,
    );
    let mut metal = MetalNifsProver::new()?;
    let mut online = Vec::with_capacity(repetitions);
    let mut verify = Vec::with_capacity(repetitions);
    let mut audit_bytes = 0;
    let mut semantic_result_ok = true;
    let mut proof_parity_ok = true;
    let mut profile_samples = Vec::with_capacity(repetitions);
    let mut pipeline_samples = Vec::with_capacity(repetitions);
    let (cpu_reference, _, _) = prove_sha(&fixture)?;
    let cpu_reference = format!("{cpu_reference:?}");
    let (warmup, _, _, _) = prove_sha_metal(&fixture, &mut metal)?;
    neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, &warmup)
        .map_err(|error| BenchmarkError::Lifecycle(format!("verify Metal SHA warm-up: {error}")))?;
    for _ in 0..repetitions {
        let started = Instant::now();
        let (audit, final_digest, profiles, pipeline) = prove_sha_metal(&fixture, &mut metal)?;
        online.push(started.elapsed());
        profile_samples.push(NifsProfileSample::from_profiles(profiles));
        pipeline_samples.push(pipeline);
        semantic_result_ok &= final_digest == expected_digest;
        proof_parity_ok &= format!("{audit:?}") == cpu_reference;
        audit_bytes = format!("{audit:?}").len();
        let started = Instant::now();
        neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, black_box(&audit))
            .map_err(|error| BenchmarkError::Lifecycle(format!("verify Metal SHA audit: {error}")))?;
        verify.push(started.elapsed());
    }
    Ok(LifecycleReport {
        name: "sha256_serial_4_chunk".to_owned(),
        backend: "MetalNifsProver".to_owned(),
        verification_mode: "full_history_audit_replay".to_owned(),
        synthesis_ms: fixture.setup_synthesis.as_secs_f64() * 1e3,
        preprocessing_ms: fixture.preprocessing.as_secs_f64() * 1e3,
        online: TimingSummary::from_durations(online),
        pipeline: Some(PipelineSample::summarize(&pipeline_samples)),
        verify_ms: TimingSummary::from_durations(verify),
        nifs_profile: Some(summarize_nifs_profiles(profile_samples)),
        audit_debug_chars: audit_bytes,
        semantic_result_ok,
        proof_parity_ok,
    })
}

fn run_sha256_cpu(repetitions: usize) -> Result<LifecycleReport, BenchmarkError> {
    let fixture = build_sha256_fixture()?;
    let expected_digest = serial_state_lanes56_semantic_digest(
        fixture
            .states
            .last()
            .ok_or_else(|| BenchmarkError::Lifecycle("SHA state trace is empty".to_owned()))?,
    );
    let mut online = Vec::with_capacity(repetitions);
    let mut verify = Vec::with_capacity(repetitions);
    let mut audit_bytes = 0;
    let mut semantic_result_ok = true;
    let mut pipeline_samples = Vec::with_capacity(repetitions);
    let (warmup, _, _) = prove_sha(&fixture)?;
    neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, &warmup)
        .map_err(|error| BenchmarkError::Lifecycle(format!("verify SHA warm-up: {error}")))?;
    for _ in 0..repetitions {
        let started = Instant::now();
        let (audit, final_digest, pipeline) = prove_sha(&fixture)?;
        online.push(started.elapsed());
        pipeline_samples.push(pipeline);
        semantic_result_ok &= final_digest == expected_digest;
        audit_bytes = format!("{audit:?}").len();

        let started = Instant::now();
        neo_fold_clean::verify_uncompressed_audit(&fixture.prep.prep, black_box(&audit))
            .map_err(|error| BenchmarkError::Lifecycle(format!("verify SHA audit: {error}")))?;
        verify.push(started.elapsed());
    }
    Ok(LifecycleReport {
        name: "sha256_serial_4_chunk".to_owned(),
        backend: "CPU".to_owned(),
        verification_mode: "full_history_audit_replay".to_owned(),
        synthesis_ms: fixture.setup_synthesis.as_secs_f64() * 1e3,
        preprocessing_ms: fixture.preprocessing.as_secs_f64() * 1e3,
        online: TimingSummary::from_durations(online),
        pipeline: Some(PipelineSample::summarize(&pipeline_samples)),
        verify_ms: TimingSummary::from_durations(verify),
        nifs_profile: None,
        audit_debug_chars: audit_bytes,
        semantic_result_ok,
        proof_parity_ok: true,
    })
}

fn build_sha256_fixture() -> Result<Sha256Fixture, BenchmarkError> {
    let synthesis_started = Instant::now();
    let total_transitions = TRANSITIONS_PER_CHUNK * CHUNK_COUNT;
    let states = sha_state_trace(&initial_sha_state(), total_transitions);
    let shape_chunk = serial_chunk(states[0].clone(), TRANSITIONS_PER_CHUNK);
    let setup_synthesis = synthesis_started.elapsed();
    let preprocessing_started = Instant::now();
    let (derived, _) = packed_state_derived_structure(&shape_chunk.sparse_r1cs, &Params::production(), &states[0]);
    let structure = derived.structure();
    let params = Params::for_ccs_shape(structure.ccs.n, structure.ccs.t(), structure.ccs.max_degree())
        .map_err(|error| BenchmarkError::Lifecycle(format!("derive SHA params: {error}")))?;
    if !params.has_production_core() || params.k_rho() < 14 {
        return Err(BenchmarkError::Lifecycle(
            "SHA benchmark did not retain production-core parameters".to_owned(),
        ));
    }
    let prepared = r1cs_f_prime::prepare_derived_structure(derived)
        .map_err(|error| BenchmarkError::Lifecycle(format!("prepare SHA relation: {error}")))?;
    let prep = r1cs_f_prime::preprocess_seeded_prepared_with_params(prepared, params, SHA256_SERIAL_AJTAI_SEED)
        .map_err(|error| BenchmarkError::Lifecycle(format!("preprocess SHA relation: {error}")))?;
    Ok(Sha256Fixture {
        states,
        prep,
        setup_synthesis,
        preprocessing: preprocessing_started.elapsed(),
    })
}

fn prove_sha(
    fixture: &Sha256Fixture,
) -> Result<(neo_fold_clean::UncompressedAudit, [u8; 32], PipelineSample), BenchmarkError> {
    let pipeline_started = Instant::now();
    let mut pipeline = PipelineSample::default();
    let mut chain = R1csChainBuilder::new(&fixture.prep)
        .map_err(|error| BenchmarkError::Lifecycle(format!("create SHA chain: {error}")))?;
    let mut final_digest = serial_state_lanes56_semantic_digest(&fixture.states[0]);
    let mut pending = Some(spawn_chunk(fixture.states[0].clone()));
    for index in 0..CHUNK_COUNT {
        let (chunk, synthesis) = join_chunk(pending.take().expect("SHA chunk synthesis is scheduled"))?;
        pipeline.synthesis_work += synthesis;
        if index + 1 < CHUNK_COUNT {
            pending = Some(spawn_chunk(fixture.states[(index + 1) * TRANSITIONS_PER_CHUNK].clone()));
        }
        let fold_started = Instant::now();
        let compiled = chain
            .append_assignment(chunk.assignment)
            .map_err(|error| BenchmarkError::Lifecycle(format!("append SHA chunk: {error}")))?;
        pipeline.fold_work += fold_started.elapsed();
        final_digest = digest_fields_as_digest32(compiled.semantic_state_digest_out);
    }
    let final_started = Instant::now();
    let audit = chain
        .finish_with_audit()
        .map_err(|error| BenchmarkError::Lifecycle(format!("finish SHA audit: {error}")))?;
    pipeline.final_materialization = final_started.elapsed();
    pipeline.overlap_saved = (pipeline.synthesis_work + pipeline.fold_work + pipeline.final_materialization)
        .saturating_sub(pipeline_started.elapsed());
    Ok((audit, final_digest, pipeline))
}

fn prove_sha_metal(
    fixture: &Sha256Fixture,
    metal: &mut MetalNifsProver,
) -> Result<
    (
        neo_fold_clean::UncompressedAudit,
        [u8; 32],
        Vec<neo_prover_metal::MetalNifsProfile>,
        PipelineSample,
    ),
    BenchmarkError,
> {
    let pipeline_started = Instant::now();
    let mut pipeline = PipelineSample::default();
    let _ = metal.take_last_profile();
    let mut profiles = Vec::new();
    let mut chain = R1csChainBuilder::new(&fixture.prep)
        .map_err(|error| BenchmarkError::Lifecycle(format!("create Metal SHA chain: {error}")))?;
    let mut final_digest = serial_state_lanes56_semantic_digest(&fixture.states[0]);
    let mut pending = Some(spawn_chunk(fixture.states[0].clone()));
    for index in 0..CHUNK_COUNT {
        let (chunk, synthesis) = join_chunk(pending.take().expect("SHA chunk synthesis is scheduled"))?;
        pipeline.synthesis_work += synthesis;
        if index + 1 < CHUNK_COUNT {
            pending = Some(spawn_chunk(fixture.states[(index + 1) * TRANSITIONS_PER_CHUNK].clone()));
        }
        let fold_started = Instant::now();
        let compiled = chain
            .append_assignment_with_nifs_adapter(chunk.assignment, metal)
            .map_err(|error| BenchmarkError::Lifecycle(format!("append Metal SHA chunk: {error}")))?;
        pipeline.fold_work += fold_started.elapsed();
        if let Some(profile) = metal.take_last_profile() {
            profiles.push(profile);
        }
        final_digest = digest_fields_as_digest32(compiled.semantic_state_digest_out);
    }
    let final_started = Instant::now();
    let audit = chain
        .finish_with_audit_and_nifs_adapter(metal)
        .map_err(|error| BenchmarkError::Lifecycle(format!("finish Metal SHA audit: {error}")))?;
    pipeline.final_materialization = final_started.elapsed();
    if let Some(profile) = metal.take_last_profile() {
        profiles.push(profile);
    }
    pipeline.overlap_saved = (pipeline.synthesis_work + pipeline.fold_work + pipeline.final_materialization)
        .saturating_sub(pipeline_started.elapsed());
    Ok((audit, final_digest, profiles, pipeline))
}

fn spawn_chunk(state: Vec<u8>) -> JoinHandle<(BellpepperCcs, Duration)> {
    std::thread::spawn(move || {
        let started = Instant::now();
        let chunk = serial_chunk(state, TRANSITIONS_PER_CHUNK);
        (chunk, started.elapsed())
    })
}

fn join_chunk(handle: JoinHandle<(BellpepperCcs, Duration)>) -> Result<(BellpepperCcs, Duration), BenchmarkError> {
    handle
        .join()
        .map_err(|_| BenchmarkError::Lifecycle("SHA chunk synthesis worker panicked".to_owned()))
}
