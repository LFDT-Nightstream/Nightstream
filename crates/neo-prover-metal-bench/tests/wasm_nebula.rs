#![cfg(target_vendor = "apple")]

//! Authentic reduced WASM + Nebula CPU/Metal proof benchmark.

use std::time::{Duration, Instant};

use neo_fold_clean::paper::nifs::{
    Error as NifsError, NifsFPrimeStepContext, NifsFreshInstancesRequest, NifsFreshSignedUnitInstancesRequest,
    NifsProverAdapter, NifsProverOutput, NifsProverRequest,
};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::CcsInstance;
use neo_prover_metal::{MetalNifsProfile, MetalNifsProver};
use neo_wasm::{
    WasmApplicationModule, WasmNebulaPreprocessing, WasmNebulaProof, WasmProver, WasmStepState, WasmVmStep,
};

const PROFILE_MANIFEST: &[u8] = include_bytes!("../../neo-wasm/tests/fixtures/wasm_benchmark_42x6.module.json");

const PROFILE_WAT: &str = r#"
(module
  (memory 1 1)
  (data (i32.const 0) "\2a\00\00\00")
  (func (export "main") (result i32)
    i32.const 0
    i32.load
    i32.const 6
    i32.mul))
"#;

struct Fixture {
    prep: WasmNebulaPreprocessing,
    trace: Vec<WasmVmStep>,
    final_state: WasmStepState,
}

struct ProfileCapture<'a> {
    metal: &'a mut MetalNifsProver,
    profiles: Vec<MetalNifsProfile>,
}

impl NifsProverAdapter for ProfileCapture<'_> {
    fn begin_f_prime_step(&mut self, context: NifsFPrimeStepContext) {
        self.metal.begin_f_prime_step(context);
    }

    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<NifsProverOutput, NifsError> {
        let output = self.metal.prove(request)?;
        if let Some(profile) = self.metal.take_last_profile() {
            self.profiles.push(profile);
        }
        Ok(output)
    }

    fn build_fresh_instances(
        &mut self,
        request: NifsFreshInstancesRequest<'_>,
    ) -> Result<Option<Vec<CcsInstance>>, NifsError> {
        self.metal.build_fresh_instances(request)
    }

    fn build_fresh_signed_unit_instances(
        &mut self,
        request: NifsFreshSignedUnitInstancesRequest<'_>,
    ) -> Result<Option<Vec<CcsInstance>>, NifsError> {
        self.metal.build_fresh_signed_unit_instances(request)
    }

    fn requires_recursive_compile_reverify(&self) -> bool {
        self.metal.requires_recursive_compile_reverify()
    }
}

#[test]
#[ignore = "authentic WASM + Nebula CPU/Metal proof benchmark"]
fn wasm_nebula_cpu_metal_benchmark_proves_and_verifies() {
    let wall_started = Instant::now();
    let fixture = build_fixture();
    let structure = fixture.prep.inner().relation().structure();
    print_fe_matrix_profile(structure);
    eprintln!(
        "WASM_NEBULA_PHASE fixture_ms={:.3} trace_rows={} relation_rows={} relation_columns={}",
        milliseconds(wall_started.elapsed()),
        fixture.trace.len(),
        structure.n,
        structure.m,
    );
    for (matrix_index, matrix) in structure.matrices.iter().enumerate() {
        for (block_index, seeded) in matrix.seeded_phi81_blocks().iter().enumerate() {
            eprintln!(
            "WASM_NEBULA_SEEDED matrix={} block={} rows={}..{} words={} word_width={} kappa={} message_cols={} chunk_size={} chunks_per_output={} transformed={}",
            matrix_index,
            block_index,
            seeded.row_start(),
            seeded.row_end(),
            seeded.word_starts().len(),
            seeded.word_width(),
            seeded.kappa(),
            seeded.message_cols(),
            seeded.chunk_size(),
            seeded.chunk_seeds_by_row().first().map_or(0, Vec::len),
            seeded.has_superneo_transformed_columns(),
        );
        }
    }
    assert!(
        structure
            .matrices
            .iter()
            .any(|matrix| !matrix.seeded_phi81_blocks().is_empty()),
        "benchmark must exercise the compact seeded-matrix path",
    );
    let mut metal = MetalNifsProver::new().expect("Metal prover");
    eprintln!(
        "WASM_NEBULA_PHASE metal_init_ms={:.3}",
        milliseconds(wall_started.elapsed())
    );
    let metal_static_elapsed = prepare_metal_static(&fixture, &mut metal);
    eprintln!(
        "WASM_NEBULA_PHASE metal_static_done_ms={:.3} metal_static_ms={:.3}",
        milliseconds(wall_started.elapsed()),
        milliseconds(metal_static_elapsed),
    );
    let (cpu, cpu_elapsed) = prove_cpu(&fixture);
    eprintln!(
        "WASM_NEBULA_PHASE cpu_done_ms={:.3} cpu_prove_ms={:.3}",
        milliseconds(wall_started.elapsed()),
        milliseconds(cpu_elapsed),
    );
    let (metal_proof, metal_elapsed) = prove_metal(&fixture, &mut metal);
    eprintln!(
        "WASM_NEBULA_PHASE metal_done_ms={:.3} metal_prove_ms={:.3}",
        milliseconds(wall_started.elapsed()),
        milliseconds(metal_elapsed),
    );
    eprintln!("WASM_NEBULA_LAST_METAL_PROFILE={:#?}", metal.last_profile());
    assert_same_terminal_authority(&cpu, &metal_proof);

    let cpu_ms = milliseconds(cpu_elapsed);
    let metal_ms = milliseconds(metal_elapsed);
    let report = serde_json::json!({
        "benchmark": "wasm_nebula_memory_arithmetic_reduced",
        "proof_pipeline": "preprocessed neo_wasm online prove; verification outside timing",
        "statistical_speed_gate": false,
        "samples": 1,
        "warmups_per_backend": 0,
        "order": "cpu_then_metal",
        "trace_rows": fixture.trace.len(),
        "relation": {
            "rows": structure.n,
            "columns": structure.m,
            "matrices": structure.t(),
            "degree": structure.max_degree(),
        },
        "cpu_ms": [cpu_ms],
        "metal_static_setup_ms": milliseconds(metal_static_elapsed),
        "metal_ms": [metal_ms],
        "speedup": cpu_ms / metal_ms,
        "canonical_verifier": "accepted_every_sample",
        "terminal_authority_parity": true,
    });
    println!(
        "WASM_NEBULA_METAL_BENCH={}",
        serde_json::to_string(&report).expect("serialize benchmark report")
    );
}

#[test]
#[ignore = "authentic WASM + Nebula FE matrix profile"]
fn wasm_nebula_fe_matrix_profile() {
    let fixture = build_fixture();
    print_fe_matrix_profile(fixture.prep.inner().relation().structure());
}

#[test]
#[ignore = "authentic WASM + Nebula Metal startup diagnostic"]
fn wasm_nebula_metal_startup_diagnostic() {
    let fixture = build_fixture();
    let mut metal = MetalNifsProver::new().expect("Metal prover");
    let static_elapsed = prepare_metal_static(&fixture, &mut metal);
    eprintln!("WASM_NEBULA_DIAGNOSTIC_STATIC_MS={:.3}", milliseconds(static_elapsed));
    let activity_before = metal.session().activity();
    let (proof, online_elapsed, profiles) = prove_metal_unverified_with_profiles(&fixture, &mut metal);
    let activity_after = metal.session().activity();
    let online_uploaded_bytes = activity_after
        .uploaded_bytes
        .saturating_sub(activity_before.uploaded_bytes);
    let online_downloaded_bytes = activity_after
        .downloaded_bytes
        .saturating_sub(activity_before.downloaded_bytes);
    eprintln!(
        "WASM_NEBULA_DIAGNOSTIC_ONLINE_TRANSFER uploaded_bytes={online_uploaded_bytes} downloaded_bytes={online_downloaded_bytes}"
    );
    eprintln!("WASM_NEBULA_DIAGNOSTIC_ONLINE_MS={:.3}", milliseconds(online_elapsed));
    neo_wasm::verify(&fixture.prep, &proof, fixture.final_state).expect("verify profiled Metal proof");
    for (fold, profile) in profiles.iter().enumerate() {
        eprintln!("WASM_NEBULA_DIAGNOSTIC_PROFILE fold={fold} {profile:#?}");
    }
    let profile = profiles.last().expect("last Metal fold profile");
    assert!(profile.pi_ccs.fe.carried_eval_on_metal);
    assert!(profile.pi_ccs.fe.seeded_patch_bytes < 1024 * 1024);
    assert_eq!(profile.pi_ccs.ajtai.seeded_patch_bytes, 0);
    assert!(profile.fresh.masks_reused);
    assert_eq!(profile.fresh.commit_count, 1);
    assert_eq!(profile.fresh.lane_commit_count, 1);
    assert!(profile.fresh.lanes_from_resident_masks);
    assert!(profile.pi_ccs.witness_masks_shared);
    assert!(profile.pi_rlc.witness_masks_reused);
    assert_eq!(profile.pi_ccs.nc.input_witnesses, 15);
    assert!(profile.pi_ccs.nc.active_witnesses < profile.pi_ccs.nc.input_witnesses);
    assert!(
        profiles
            .iter()
            .all(|profile| profile.activity.uploaded_bytes < 1024 * 1024),
        "online folds must reuse static Metal state instead of rewriting it from the CPU",
    );
    assert!(
        profiles
            .iter()
            .all(|profile| profile.activity.downloaded_bytes < 1024 * 1024),
        "online folds must keep private witness state resident instead of materializing it on the CPU",
    );
    assert!(
        online_uploaded_bytes < 16 * 1024 * 1024,
        "the complete online proof must upload only its compact fresh inputs, got {online_uploaded_bytes} bytes",
    );
    assert!(
        online_downloaded_bytes < 2 * 1024 * 1024,
        "the complete online proof must download only public proof surfaces, got {online_downloaded_bytes} bytes",
    );
}

fn print_fe_matrix_profile(structure: &neo_fold_clean::paper::relations::Structure) {
    let mut used = vec![false; structure.matrices.len()];
    for term in structure.f.terms() {
        for (index, &exponent) in term.exps.iter().enumerate() {
            used[index] |= exponent != 0;
        }
    }
    let selected = used
        .iter()
        .enumerate()
        .filter_map(|(index, &is_used)| is_used.then_some(index))
        .collect::<Vec<_>>();
    let seeded_selected = selected
        .iter()
        .copied()
        .filter(|&index| !structure.matrices[index].seeded_phi81_blocks().is_empty())
        .collect::<Vec<_>>();
    eprintln!(
        "WASM_NEBULA_FE_MATRIX_PROFILE selected={selected:?} seeded_selected={seeded_selected:?} seeded_blocks_by_matrix={:?}",
        structure
            .matrices
            .iter()
            .map(|matrix| matrix.seeded_phi81_blocks().len())
            .collect::<Vec<_>>(),
    );
}

#[test]
#[ignore = "authentic two-chain WASM + Nebula Metal throughput benchmark"]
fn wasm_nebula_two_chain_metal_throughput_proves_and_verifies() {
    const CHAINS: usize = 2;

    let fixture = build_fixture();
    let structure = fixture.prep.inner().relation().structure();
    let (cpu, cpu_elapsed) = prove_cpu(&fixture);
    let parallel_started = Instant::now();
    let runs = std::thread::scope(|scope| {
        let handles = (0..CHAINS)
            .map(|_| {
                scope.spawn(|| {
                    let mut metal = MetalNifsProver::new().expect("Metal prover");
                    let (proof, elapsed) = prove_metal_unverified(&fixture, &mut metal);
                    (proof, elapsed, metal.last_profile())
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("parallel Metal chain"))
            .collect::<Vec<_>>()
    });
    let parallel_wall = parallel_started.elapsed();

    for (proof, _, _) in &runs {
        neo_wasm::verify(&fixture.prep, proof, fixture.final_state).expect("verify parallel Metal proof");
        assert_same_terminal_authority(&cpu, proof);
    }

    let metal_thread_ms = runs
        .iter()
        .map(|(_, elapsed, _)| milliseconds(*elapsed))
        .collect::<Vec<_>>();
    let metal_thread_sum_ms = metal_thread_ms.iter().sum::<f64>();
    let parallel_wall_ms = milliseconds(parallel_wall);
    let cpu_aggregate_ms = milliseconds(cpu_elapsed) * CHAINS as f64;
    let report = serde_json::json!({
        "benchmark": "wasm_nebula_two_chain_throughput",
        "proof_pipeline": "two independent low-level Metal adapter proof calls",
        "chains": CHAINS,
        "cpu_single_ms": milliseconds(cpu_elapsed),
        "cpu_aggregate_ms": cpu_aggregate_ms,
        "metal_thread_ms": metal_thread_ms,
        "metal_thread_sum_ms": metal_thread_sum_ms,
        "metal_parallel_wall_ms": parallel_wall_ms,
        "aggregate_speedup": cpu_aggregate_ms / parallel_wall_ms,
        "metal_overlap": metal_thread_sum_ms / parallel_wall_ms,
        "canonical_verifier": "accepted_every_chain",
        "terminal_authority_parity": true,
        "relation": {
            "rows": structure.n,
            "columns": structure.m,
            "matrices": structure.t(),
        },
        "profiles": runs.iter().map(|(_, _, profile)| format!("{profile:?}")).collect::<Vec<_>>(),
    });
    println!(
        "WASM_NEBULA_METAL_MULTICHAIN={}",
        serde_json::to_string(&report).expect("serialize multichain benchmark report")
    );
}

fn build_fixture() -> Fixture {
    let module = WasmApplicationModule::from_json_slice(PROFILE_MANIFEST).expect("Lean-owned benchmark module");
    let independent_wasm = wat::parse_str(PROFILE_WAT).expect("valid profile WAT");
    assert_eq!(module.bytes(), independent_wasm);
    let artifacts = module.artifacts();
    let run = neo_wasm::collect_wasmtime_steps(module.bytes(), module.entrypoint(), &[]).expect("wasmtime trace");
    assert_eq!(run.results.as_slice(), &["252".to_owned()]);
    let trace = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("normalized trace");
    let final_state = trace.last().expect("nonempty WASM trace").state_after;
    let mut entry_pcs = artifacts
        .tables
        .function_entries
        .iter()
        .map(|&(_, entry_pc)| entry_pc)
        .collect::<Vec<_>>();
    entry_pcs.sort_unstable();
    entry_pcs.dedup();
    let [entry_pc] = entry_pcs.as_slice() else {
        panic!("benchmark expects one WASM function entry, got {entry_pcs:?}");
    };
    let prep = neo_wasm::nebula::preprocess_seeded_reduced_memory_test_only(
        test_params(),
        neo_wasm::WasmNebulaProfile::test_profile(),
        artifacts,
        &run.initial_locals,
        *entry_pc,
        0x57a5_b001,
    )
    .expect("WASM + Nebula preprocessing");
    Fixture {
        prep,
        trace,
        final_state,
    }
}

fn prove_cpu(fixture: &Fixture) -> (WasmNebulaProof, Duration) {
    let mut cpu = WasmProver::cpu();
    let started = Instant::now();
    let proof = cpu
        .prove(&fixture.prep, &fixture.trace)
        .expect("CPU WASM + Nebula proof");
    let elapsed = started.elapsed();
    neo_wasm::verify(&fixture.prep, &proof, fixture.final_state).expect("verify CPU WASM + Nebula proof");
    (proof, elapsed)
}

fn prove_metal(fixture: &Fixture, metal: &mut MetalNifsProver) -> (WasmNebulaProof, Duration) {
    let (proof, elapsed) = prove_metal_unverified(fixture, metal);
    neo_wasm::verify(&fixture.prep, &proof, fixture.final_state).expect("verify Metal WASM + Nebula proof");
    (proof, elapsed)
}

fn prove_metal_unverified(fixture: &Fixture, metal: &mut MetalNifsProver) -> (WasmNebulaProof, Duration) {
    prepare_metal_static(fixture, metal);
    let started = Instant::now();
    let proof = neo_wasm::nebula::prove_with_nifs_adapter(&fixture.prep, metal, &fixture.trace)
        .expect("Metal WASM + Nebula proof");
    let elapsed = started.elapsed();
    (proof, elapsed)
}

fn prove_metal_unverified_with_profiles(
    fixture: &Fixture,
    metal: &mut MetalNifsProver,
) -> (WasmNebulaProof, Duration, Vec<MetalNifsProfile>) {
    prepare_metal_static(fixture, metal);
    let started = Instant::now();
    let mut capture = ProfileCapture {
        metal,
        profiles: Vec::new(),
    };
    let proof = neo_wasm::nebula::prove_with_nifs_adapter(&fixture.prep, &mut capture, &fixture.trace)
        .expect("profiled Metal WASM + Nebula proof");
    let elapsed = started.elapsed();
    (proof, elapsed, capture.profiles)
}

fn prepare_metal_static(fixture: &Fixture, metal: &mut MetalNifsProver) -> Duration {
    let prep = &fixture.prep.inner().prep;
    let started = Instant::now();
    metal
        .prepare_static(
            &prep.log,
            prep.structure(),
            prep.optimized_cache(),
            prep.nebula().map(|config| &config.scheme),
        )
        .expect("prepare static Metal WASM + Nebula state");
    started.elapsed()
}

fn assert_same_terminal_authority(cpu: &WasmNebulaProof, metal: &WasmNebulaProof) {
    let cpu = cpu.inner();
    let metal = metal.inner();
    assert_eq!(cpu.state.chunk_count, metal.state.chunk_count);
    assert_eq!(cpu.state.step_count, metal.state.step_count);
    assert_eq!(cpu.state.z_0, metal.state.z_0);
    assert_eq!(cpu.state.z_i, metal.state.z_i);
    assert_eq!(cpu.state.pc, metal.state.pc);
    assert_eq!(
        cpu.state.initial_semantic_state_digest,
        metal.state.initial_semantic_state_digest
    );
    assert_eq!(cpu.state.semantic_state_digest, metal.state.semantic_state_digest);
    assert_eq!(cpu.state.acc_digest, metal.state.acc_digest);
    assert_eq!(cpu.state.public_trace, metal.state.public_trace);
    assert_eq!(cpu.state.nebula, metal.state.nebula);

    let cpu_running = cpu
        .state
        .proof
        .materialized_running()
        .expect("materialize CPU running authority")
        .expect("CPU active proof state");
    let metal_running = metal
        .state
        .proof
        .materialized_running()
        .expect("materialize Metal running authority")
        .expect("Metal active proof state");
    assert_eq!(cpu_running.claims, metal_running.claims);
    assert_eq!(cpu_running.witnesses, metal_running.witnesses);
    assert_eq!(cpu_running.parent_authority, metal_running.parent_authority);
    assert_instances_equivalent(
        &cpu.state.proof.latest().expect("CPU latest").instances,
        &metal.state.proof.latest().expect("Metal latest").instances,
    );

    let cpu_final = cpu.final_fold.as_ref().expect("CPU terminal fold");
    let metal_final = metal.final_fold.as_ref().expect("Metal terminal fold");
    assert_eq!(cpu_final.x_out, metal_final.x_out);
    assert_eq!(cpu_final.nifs.pi_ccs.outputs, metal_final.nifs.pi_ccs.outputs);
    assert_eq!(
        cpu_final.nifs.pi_ccs.outputs_digest,
        metal_final.nifs.pi_ccs.outputs_digest
    );
    assert_eq!(
        serde_json::to_vec(&cpu_final.nifs.pi_ccs.sumcheck).expect("CPU sumcheck JSON"),
        serde_json::to_vec(&metal_final.nifs.pi_ccs.sumcheck).expect("Metal sumcheck JSON"),
    );
    assert_eq!(cpu_final.nifs.pi_rlc.combined, metal_final.nifs.pi_rlc.combined);
    assert_eq!(cpu_final.nifs.pi_dec.children, metal_final.nifs.pi_dec.children);
    assert_eq!(
        cpu_final.terminal_inputs.pre_final_running.claims,
        metal_final.terminal_inputs.pre_final_running.claims
    );
    assert_eq!(
        cpu_final.terminal_inputs.pre_final_running.parent_authority,
        metal_final
            .terminal_inputs
            .pre_final_running
            .parent_authority
    );
    assert_instances_equivalent(
        &cpu_final.terminal_inputs.latest.instances,
        &metal_final.terminal_inputs.latest.instances,
    );
    assert_eq!(
        cpu_final.terminal_inputs.pre_nebula,
        metal_final.terminal_inputs.pre_nebula
    );
}

fn assert_instances_equivalent(cpu: &[CcsInstance], metal: &[CcsInstance]) {
    assert_eq!(cpu.len(), metal.len());
    for (cpu, metal) in cpu.iter().zip(metal) {
        assert_eq!(cpu.claim.c, metal.claim.c);
        assert_eq!(cpu.claim.x, metal.claim.x);
        assert_eq!(cpu.claim.m_in, metal.claim.m_in);
        assert_eq!(cpu.claim.adv, metal.claim.adv);
        assert_eq!(cpu.witness.w, metal.witness.w);
        assert_eq!(cpu.witness.Z, metal.witness.Z);
    }
}

fn test_params() -> Params {
    let raw = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        1 << 14,
        neo_params::goldilocks_paper_b2::B_BASE,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        20,
    )
    .expect("benchmark SuperNeo parameters");
    Params::test_only_from_neo_params(raw)
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
