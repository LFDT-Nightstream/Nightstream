//! Phase 5 end gate: the real SHA-256 serial-pair workload through the full
//! R1CS-F' lifecycle, CPU chain vs CUDA-adapter chain, byte-identical audits.

use neo_fold_clean::frontends::bellpepper::BellpepperCcs;
use neo_fold_clean::frontends::r1cs_f_prime::{self, R1csChainBuilder, R1csFPrimePreprocessing};
use neo_fold_clean::paper::construction2::{FoldProof, ProofState};
use neo_fold_clean::paper::digest::digest_fields_as_digest32;
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::{CeClaim, WitnessMat};
use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use crate::sha256_workload::{
    initial_sha_state, packed_state_derived_structure, serial_chunk, serial_state_lanes56_semantic_digest,
    sha_state_trace, SHA256_SERIAL_AJTAI_SEED,
};

use super::*;

/// SHA transitions per serial chunk (the reference serial-pair shape) and
/// chunk count. Folds are lazy — append k folds append k-1's latest, finish
/// folds the last — so 4 chunks give 5 folds, 3 of them at the steady
/// 15-witness shape where the device advantage shows.
const TRANSITIONS_PER_CHUNK: usize = 2;
const CHUNK_COUNT: usize = 4;
const MULTICHAIN_COUNT: usize = 2;
const MULTICHAIN8_COUNT: usize = 8;
const MULTICHAIN16_COUNT: usize = 16;

struct Sha256E2eFixture {
    states: Vec<Vec<u8>>,
    chunks: Vec<BellpepperCcs>,
    prep: R1csFPrimePreprocessing,
    synth_ms: f64,
    setup_ms: f64,
}

struct ChainRun {
    audit_repr: String,
    claims_only_audit_repr: String,
    terminal_witnesses: Vec<WitnessMat>,
    final_semantic_digest: [u8; 32],
    acc_digest_after_append: Vec<[u8; 32]>,
    recursive_folds_after_append: Vec<Option<NifsProof>>,
    append_ms: Vec<f64>,
    finish_ms: f64,
}

struct ParallelCudaRun {
    run: ChainRun,
    init_ms: f64,
    prepare_ms: f64,
}

struct OnlineTimer<'a> {
    done: &'a mpsc::Sender<()>,
    post_barrier: &'a Barrier,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MultichainMode {
    FullAudit,
    TerminalClaimsOnly,
}

impl MultichainMode {
    fn configure(self, cuda: &mut CudaNifsProver) {
        if self == Self::TerminalClaimsOnly {
            cuda.enable_terminal_claims_only_fast();
        }
    }
}

impl ChainRun {
    fn online_ms(&self) -> f64 {
        self.append_ms.iter().sum::<f64>() + self.finish_ms
    }
}

fn build_sha256_e2e_fixture() -> Sha256E2eFixture {
    let total_transitions = TRANSITIONS_PER_CHUNK * CHUNK_COUNT;
    let states = sha_state_trace(&initial_sha_state(), total_transitions);
    let (chunks, synth_ms) = timed(|| {
        (0..CHUNK_COUNT)
            .map(|i| serial_chunk(states[i * TRANSITIONS_PER_CHUNK].clone(), TRANSITIONS_PER_CHUNK))
            .collect::<Vec<_>>()
    });

    let (prep, setup_ms) = timed(|| {
        let (derived, _iterations) =
            packed_state_derived_structure(&chunks[0].sparse_r1cs, &Params::production(), &states[0]);
        let structure = derived.structure();
        let params = Params::for_ccs_shape(structure.ccs.n, structure.ccs.t(), structure.ccs.max_degree())
            .expect("e2e production params");
        let prepared = r1cs_f_prime::prepare_derived_structure(derived).expect("e2e prepare");
        r1cs_f_prime::preprocess_seeded_prepared_with_params(prepared, params, SHA256_SERIAL_AJTAI_SEED)
            .expect("e2e preprocess")
    });

    Sha256E2eFixture {
        states,
        chunks,
        prep,
        synth_ms,
        setup_ms,
    }
}

/// Drive one full chain: append every chunk, then finish with audit.
/// `adapter` selects the CUDA path; `None` is the canonical CPU chain.
fn run_chain(
    prep: &R1csFPrimePreprocessing,
    chunks: &[BellpepperCcs],
    initial_state: &[u8],
    adapter: Option<&mut CudaNifsProver>,
) -> ChainRun {
    run_chain_inner(prep, chunks, initial_state, adapter, None, true)
}

fn run_chain_without_append_snapshots(
    prep: &R1csFPrimePreprocessing,
    chunks: &[BellpepperCcs],
    initial_state: &[u8],
    adapter: Option<&mut CudaNifsProver>,
) -> ChainRun {
    run_chain_inner(prep, chunks, initial_state, adapter, None, false)
}

fn run_chain_inner(
    prep: &R1csFPrimePreprocessing,
    chunks: &[BellpepperCcs],
    initial_state: &[u8],
    mut adapter: Option<&mut CudaNifsProver>,
    online_timer: Option<OnlineTimer<'_>>,
    collect_append_snapshots: bool,
) -> ChainRun {
    let mut chain = R1csChainBuilder::new(prep).expect("e2e chain builder");
    let mut append_ms = Vec::with_capacity(chunks.len());
    let mut acc_digest_after_append = Vec::with_capacity(chunks.len());
    let mut recursive_folds_after_append = Vec::with_capacity(chunks.len());
    let mut final_semantic_digest = serial_state_lanes56_semantic_digest(initial_state);
    for chunk in chunks {
        let assignment = chunk.assignment.clone();
        let (compiled, ms) = match adapter.as_mut() {
            Some(cuda) => timed(|| {
                chain
                    .append_assignment_with_nifs_adapter(assignment, *cuda)
                    .expect("e2e CUDA append")
            }),
            None => timed(|| chain.append_assignment(assignment).expect("e2e CPU append")),
        };
        append_ms.push(ms);
        if collect_append_snapshots {
            let audit = chain.audit().expect("audit after append");
            acc_digest_after_append.push(audit.proof.state.acc_digest);
            recursive_folds_after_append.push(
                audit
                    .steps
                    .last()
                    .map(|step| match &step.fold {
                        FoldProof::NoFold => None,
                        FoldProof::Recursive(proof) => {
                            Some(proof.materialize().expect("recursive fold materialization"))
                        }
                    })
                    .unwrap_or(None),
            );
        }
        final_semantic_digest = digest_fields_as_digest32(compiled.semantic_state_digest_out);
    }
    let (audit, finish_ms) = match adapter {
        Some(cuda) => timed(|| {
            chain
                .finish_with_audit_and_nifs_adapter(cuda)
                .expect("e2e CUDA finish")
        }),
        None => timed(|| chain.finish_with_audit().expect("e2e CPU finish")),
    };
    if let Some(timer) = online_timer {
        timer
            .done
            .send(())
            .expect("signal multichain online completion");
        timer.post_barrier.wait();
    }
    let terminal_witnesses = terminal_witnesses_from_audit(&audit);
    let claims_only_audit_repr = terminal_claims_only_audit_repr(&audit);
    ChainRun {
        audit_repr: format!("{audit:?}"),
        claims_only_audit_repr,
        terminal_witnesses,
        final_semantic_digest,
        acc_digest_after_append,
        recursive_folds_after_append,
        append_ms,
        finish_ms,
    }
}

fn run_sha256_lifecycle(label: &str, configure_cuda: impl FnOnce(&mut CudaNifsProver)) -> (f64, f64, usize) {
    let total_transitions = TRANSITIONS_PER_CHUNK * CHUNK_COUNT;
    let fixture = build_sha256_e2e_fixture();

    let cpu = run_chain(&fixture.prep, &fixture.chunks, &fixture.states[0], None);
    let (mut cuda, adapter_ms) = timed(|| CudaNifsProver::new().expect("open CUDA NIFS prover"));
    configure_cuda(&mut cuda);
    let (_, cuda_prepare_ms) = timed(|| {
        cuda.prepare_static(&fixture.prep.prep.log, fixture.prep.prep.optimized_cache())
            .expect("prepare CUDA static state")
    });
    let gpu = run_chain(&fixture.prep, &fixture.chunks, &fixture.states[0], Some(&mut cuda));

    assert_recursive_folds_identical(&cpu.recursive_folds_after_append, &gpu.recursive_folds_after_append);
    assert_eq!(
        gpu.acc_digest_after_append, cpu.acc_digest_after_append,
        "per-append accumulator digest mismatch"
    );
    assert_audit_identical(&cpu.audit_repr, &gpu.audit_repr);
    assert_eq!(
        gpu.terminal_witnesses, cpu.terminal_witnesses,
        "terminal running witnesses mismatch"
    );
    let expected_digest = serial_state_lanes56_semantic_digest(fixture.states.last().expect("final state"));
    assert_eq!(cpu.final_semantic_digest, expected_digest, "CPU final semantic digest");
    assert_eq!(gpu.final_semantic_digest, expected_digest, "GPU final semantic digest");

    eprintln!(
        "  synth {synth_ms:.0}ms  setup {setup_ms:.0}ms  cuda-init {adapter_ms:.0}ms  cuda-prepare {cuda_prepare_ms:.0}ms"
        ,
        synth_ms = fixture.synth_ms,
        setup_ms = fixture.setup_ms
    );
    for (idx, (cpu_ms, gpu_ms)) in cpu.append_ms.iter().zip(&gpu.append_ms).enumerate() {
        eprintln!("  append {idx}: cpu {cpu_ms:.2}ms  gpu {gpu_ms:.2}ms");
    }
    eprintln!("  finish  : cpu {:.2}ms  gpu {:.2}ms", cpu.finish_ms, gpu.finish_ms);
    println!(
        "[parity {label}] OK: sha256 serial x{total_transitions} lifecycle identical (audit {} bytes); online prove cpu={:.2}ms gpu={:.2}ms ({:.2}x)",
        cpu.audit_repr.len(),
        cpu.online_ms(),
        gpu.online_ms(),
        cpu.online_ms() / gpu.online_ms(),
    );
    (cpu.online_ms(), gpu.online_ms(), cpu.audit_repr.len())
}

fn run_sha256_lifecycle_terminal_claims_only(
    label: &str,
    configure_cuda: impl FnOnce(&mut CudaNifsProver),
) -> (f64, f64, usize) {
    let total_transitions = TRANSITIONS_PER_CHUNK * CHUNK_COUNT;
    let fixture = build_sha256_e2e_fixture();

    let cpu = run_chain(&fixture.prep, &fixture.chunks, &fixture.states[0], None);
    let (mut cuda, adapter_ms) = timed(|| CudaNifsProver::new().expect("open CUDA NIFS prover"));
    cuda.enable_terminal_claims_only_fast();
    configure_cuda(&mut cuda);
    let (_, cuda_prepare_ms) = timed(|| {
        cuda.prepare_static(&fixture.prep.prep.log, fixture.prep.prep.optimized_cache())
            .expect("prepare CUDA static state")
    });
    let gpu = run_chain(&fixture.prep, &fixture.chunks, &fixture.states[0], Some(&mut cuda));

    assert_recursive_folds_identical(&cpu.recursive_folds_after_append, &gpu.recursive_folds_after_append);
    assert_eq!(
        gpu.acc_digest_after_append, cpu.acc_digest_after_append,
        "per-append accumulator digest mismatch"
    );
    assert_audit_identical(&cpu.claims_only_audit_repr, &gpu.claims_only_audit_repr);
    let expected_digest = serial_state_lanes56_semantic_digest(fixture.states.last().expect("final state"));
    assert_eq!(cpu.final_semantic_digest, expected_digest, "CPU final semantic digest");
    assert_eq!(gpu.final_semantic_digest, expected_digest, "GPU final semantic digest");

    eprintln!(
        "  synth {synth_ms:.0}ms  setup {setup_ms:.0}ms  cuda-init {adapter_ms:.0}ms  cuda-prepare {cuda_prepare_ms:.0}ms"
        ,
        synth_ms = fixture.synth_ms,
        setup_ms = fixture.setup_ms
    );
    for (idx, (cpu_ms, gpu_ms)) in cpu.append_ms.iter().zip(&gpu.append_ms).enumerate() {
        eprintln!("  append {idx}: cpu {cpu_ms:.2}ms  gpu {gpu_ms:.2}ms");
    }
    eprintln!("  finish  : cpu {:.2}ms  gpu {:.2}ms", cpu.finish_ms, gpu.finish_ms);
    println!(
        "[parity {label}] OK: sha256 serial x{total_transitions} lifecycle identical with terminal claims-only fast path (claims-only audit {} bytes); online prove cpu={:.2}ms gpu={:.2}ms ({:.2}x)",
        cpu.claims_only_audit_repr.len(),
        cpu.online_ms(),
        gpu.online_ms(),
        cpu.online_ms() / gpu.online_ms(),
    );
    (cpu.online_ms(), gpu.online_ms(), cpu.claims_only_audit_repr.len())
}

fn prepare_cuda_provers_on_context(
    fixture: &Sha256E2eFixture,
    count: usize,
    ctx: &Arc<cuda_core::CudaContext>,
    mode: MultichainMode,
) -> (Vec<CudaNifsProver>, f64, f64) {
    let (mut provers, init_ms) = timed(|| {
        (0..count)
            .map(|_| {
                let mut cuda = CudaNifsProver::new_on_context(Arc::clone(ctx)).expect("open CUDA NIFS prover stream");
                mode.configure(&mut cuda);
                cuda
            })
            .collect::<Vec<_>>()
    });
    let (_, prepare_ms) = timed(|| {
        for prover in &mut provers {
            prover
                .prepare_static(&fixture.prep.prep.log, fixture.prep.prep.optimized_cache())
                .expect("prepare CUDA static state");
        }
    });
    (provers, init_ms, prepare_ms)
}

fn assert_cuda_chain_matches_cpu(cpu: &ChainRun, gpu: &ChainRun, final_state: &[u8], mode: MultichainMode) {
    if !cpu.recursive_folds_after_append.is_empty() || !gpu.recursive_folds_after_append.is_empty() {
        assert_recursive_folds_identical(&cpu.recursive_folds_after_append, &gpu.recursive_folds_after_append);
        assert_eq!(
            gpu.acc_digest_after_append, cpu.acc_digest_after_append,
            "per-append accumulator digest mismatch"
        );
    }
    match mode {
        MultichainMode::FullAudit => {
            assert_audit_identical(&cpu.audit_repr, &gpu.audit_repr);
            assert_eq!(
                gpu.terminal_witnesses, cpu.terminal_witnesses,
                "terminal running witnesses mismatch"
            );
        }
        MultichainMode::TerminalClaimsOnly => {
            assert_audit_identical(&cpu.claims_only_audit_repr, &gpu.claims_only_audit_repr);
        }
    }
    let expected_digest = serial_state_lanes56_semantic_digest(final_state);
    assert_eq!(cpu.final_semantic_digest, expected_digest, "CPU final semantic digest");
    assert_eq!(gpu.final_semantic_digest, expected_digest, "GPU final semantic digest");
}

fn run_cuda_chains_sequential(fixture: &Sha256E2eFixture, provers: &mut [CudaNifsProver]) -> (Vec<ChainRun>, f64) {
    timed(|| {
        provers
            .iter_mut()
            .map(|cuda| {
                run_chain_without_append_snapshots(&fixture.prep, &fixture.chunks, &fixture.states[0], Some(cuda))
            })
            .collect()
    })
}

fn run_cuda_chains_parallel(
    fixture: &Sha256E2eFixture,
    count: usize,
    ctx: Arc<cuda_core::CudaContext>,
    mode: MultichainMode,
) -> (Vec<ParallelCudaRun>, f64) {
    let barrier = Arc::new(Barrier::new(count + 1));
    let post_barrier = Arc::new(Barrier::new(count + 1));
    let (online_tx, online_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        let handles = (0..count)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let post_barrier = Arc::clone(&post_barrier);
                let ctx = Arc::clone(&ctx);
                let online_tx = online_tx.clone();
                scope.spawn(move || {
                    let (mut cuda, init_ms) =
                        timed(|| CudaNifsProver::new_on_context(ctx).expect("open CUDA NIFS prover stream"));
                    mode.configure(&mut cuda);
                    let (_, prepare_ms) = timed(|| {
                        cuda.prepare_static(&fixture.prep.prep.log, fixture.prep.prep.optimized_cache())
                            .expect("prepare CUDA static state")
                    });
                    barrier.wait();
                    let run = run_chain_inner(
                        &fixture.prep,
                        &fixture.chunks,
                        &fixture.states[0],
                        Some(&mut cuda),
                        Some(OnlineTimer {
                            done: &online_tx,
                            post_barrier: &post_barrier,
                        }),
                        false,
                    );
                    ParallelCudaRun {
                        run,
                        init_ms,
                        prepare_ms,
                    }
                })
            })
            .collect::<Vec<_>>();
        drop(online_tx);
        barrier.wait();
        let online_start = Instant::now();
        for _ in 0..count {
            online_rx
                .recv()
                .expect("CUDA chain worker exited before online completion");
        }
        let online_wall_ms = online_start.elapsed().as_secs_f64() * 1000.0;
        post_barrier.wait();
        let runs = handles
            .into_iter()
            .map(|handle| handle.join().expect("CUDA chain worker panicked"))
            .collect::<Vec<_>>();
        (runs, online_wall_ms)
    })
}

fn terminal_witnesses_from_audit(audit: &neo_fold_clean::lifecycle::UncompressedAudit) -> Vec<WitnessMat> {
    match &audit.proof.state.proof {
        ProofState::Active { running, .. } => {
            running
                .materialize()
                .expect("terminal running materialization")
                .witnesses
        }
        ProofState::Initial => Vec::new(),
    }
}

fn terminal_claims_only_audit_repr(audit: &neo_fold_clean::lifecycle::UncompressedAudit) -> String {
    let mut audit = audit.clone();
    if let ProofState::Active { running, .. } = &mut audit.proof.state.proof {
        let running = running
            .as_materialized_mut()
            .expect("terminal running materialization");
        running.witnesses = (0..running.claims.len())
            .map(|_| WitnessMat::zero(0, 0, F::ZERO))
            .collect();
    }
    format!("{audit:?}")
}

fn assert_recursive_folds_identical(cpu: &[Option<NifsProof>], gpu: &[Option<NifsProof>]) {
    assert_eq!(gpu.len(), cpu.len(), "recursive fold snapshot count mismatch");
    for (append, (cpu, gpu)) in cpu.iter().zip(gpu.iter()).enumerate() {
        match (cpu, gpu) {
            (None, None) => {}
            (Some(cpu), Some(gpu)) => assert_nifs_proof_identical(append, cpu, gpu),
            (None, Some(_)) => panic!("append {append}: GPU emitted a recursive NIFS proof where CPU emitted NoFold"),
            (Some(_), None) => panic!("append {append}: CPU emitted a recursive NIFS proof where GPU emitted NoFold"),
        }
    }
}

fn assert_nifs_proof_identical(append: usize, cpu: &NifsProof, gpu: &NifsProof) {
    assert_pi_ccs_sumcheck_identical(append, &cpu.pi_ccs.sumcheck, &gpu.pi_ccs.sumcheck);
    if cpu.pi_ccs.outputs_digest != gpu.pi_ccs.outputs_digest {
        panic!("append {append}: Pi_CCS output digest mismatch");
    }
    assert_claims_identical(append, "Pi_CCS output", &cpu.pi_ccs.outputs, &gpu.pi_ccs.outputs);
    assert_claim_identical(append, "Pi_RLC combined", 0, &cpu.pi_rlc.combined, &gpu.pi_rlc.combined);
    assert_claims_identical(append, "Pi_DEC child", &cpu.pi_dec.children, &gpu.pi_dec.children);
}

fn assert_pi_ccs_sumcheck_identical(
    append: usize,
    cpu: &neo_reductions::api::PiCcsProof,
    gpu: &neo_reductions::api::PiCcsProof,
) {
    if cpu.variant != gpu.variant {
        panic!("append {append}: Pi_CCS sumcheck variant mismatch");
    }
    if cpu.sc_initial_sum != gpu.sc_initial_sum {
        panic!(
            "append {append}: Pi_CCS FE initial sum mismatch: cpu={:?} gpu={:?} alpha_equal={} gamma_equal={}",
            cpu.sc_initial_sum,
            gpu.sc_initial_sum,
            cpu.challenges_public.alpha == gpu.challenges_public.alpha,
            cpu.challenges_public.gamma == gpu.challenges_public.gamma,
        );
    }
    if cpu.sc_initial_sum_nc != gpu.sc_initial_sum_nc {
        panic!("append {append}: Pi_CCS NC initial sum mismatch");
    }
    if cpu.challenges_public.alpha != gpu.challenges_public.alpha {
        panic!("append {append}: Pi_CCS public alpha mismatch");
    }
    if cpu.challenges_public.beta_a != gpu.challenges_public.beta_a {
        panic!("append {append}: Pi_CCS public beta_a mismatch");
    }
    if cpu.challenges_public.beta_r != gpu.challenges_public.beta_r {
        panic!("append {append}: Pi_CCS public beta_r mismatch");
    }
    if cpu.challenges_public.beta_m != gpu.challenges_public.beta_m {
        panic!("append {append}: Pi_CCS public beta_m mismatch");
    }
    if cpu.challenges_public.gamma != gpu.challenges_public.gamma {
        panic!("append {append}: Pi_CCS public gamma mismatch");
    }
    assert_rounds_identical(
        append,
        "Pi_CCS FE round coefficients",
        &cpu.sumcheck_rounds,
        &gpu.sumcheck_rounds,
    );
    if cpu.sumcheck_challenges != gpu.sumcheck_challenges {
        panic!("append {append}: Pi_CCS FE challenges mismatch");
    }
    if cpu.sumcheck_final != gpu.sumcheck_final {
        panic!("append {append}: Pi_CCS FE final mismatch");
    }
    assert_rounds_identical(
        append,
        "Pi_CCS NC round coefficients",
        &cpu.sumcheck_rounds_nc,
        &gpu.sumcheck_rounds_nc,
    );
    if cpu.sumcheck_challenges_nc != gpu.sumcheck_challenges_nc {
        panic!("append {append}: Pi_CCS NC challenges mismatch");
    }
    if cpu.sumcheck_final_nc != gpu.sumcheck_final_nc {
        panic!("append {append}: Pi_CCS NC final mismatch");
    }
    if cpu.header_digest != gpu.header_digest {
        panic!("append {append}: Pi_CCS header_digest mismatch");
    }
    if cpu._extra != gpu._extra {
        panic!("append {append}: Pi_CCS extra payload mismatch");
    }
}

fn assert_rounds_identical(append: usize, label: &str, cpu: &[Vec<K>], gpu: &[Vec<K>]) {
    assert_eq!(gpu.len(), cpu.len(), "append {append}: {label} round count mismatch");
    for (round, (cpu_round, gpu_round)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(
            gpu_round.len(),
            cpu_round.len(),
            "append {append}: {label} round {round} coeff count mismatch"
        );
        for (degree, (cpu_coeff, gpu_coeff)) in cpu_round.iter().zip(gpu_round.iter()).enumerate() {
            if cpu_coeff != gpu_coeff {
                panic!("append {append}: {label} mismatch at round {round}, degree {degree}");
            }
        }
    }
}

fn assert_claims_identical(append: usize, label: &str, cpu: &[CeClaim], gpu: &[CeClaim]) {
    assert_eq!(gpu.len(), cpu.len(), "append {append}: {label} claim count mismatch");
    for (idx, (cpu, gpu)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_claim_identical(append, label, idx, cpu, gpu);
    }
}

fn assert_claim_identical(append: usize, label: &str, idx: usize, cpu: &CeClaim, gpu: &CeClaim) {
    if cpu == gpu {
        return;
    }
    if cpu.c != gpu.c {
        panic!("append {append}: {label} claim {idx} commitment mismatch");
    }
    if cpu.X != gpu.X {
        panic!("append {append}: {label} claim {idx} X mismatch");
    }
    if cpu.r != gpu.r {
        panic!("append {append}: {label} claim {idx} r mismatch");
    }
    if cpu.s_col != gpu.s_col {
        panic!("append {append}: {label} claim {idx} s_col mismatch");
    }
    if cpu.y_ring != gpu.y_ring {
        panic!("append {append}: {label} claim {idx} y_ring mismatch");
    }
    if cpu.ct != gpu.ct {
        panic!("append {append}: {label} claim {idx} ct mismatch");
    }
    if cpu.aux_openings != gpu.aux_openings {
        panic!("append {append}: {label} claim {idx} aux_openings mismatch");
    }
    if cpu.y_zcol != gpu.y_zcol {
        panic!("append {append}: {label} claim {idx} y_zcol mismatch");
    }
    if cpu.m_in != gpu.m_in {
        panic!("append {append}: {label} claim {idx} m_in mismatch");
    }
    if cpu.fold_digest != gpu.fold_digest {
        panic!("append {append}: {label} claim {idx} fold_digest mismatch");
    }
    if cpu.c_step_coords != gpu.c_step_coords {
        panic!("append {append}: {label} claim {idx} c_step_coords mismatch");
    }
    if cpu.u_offset != gpu.u_offset || cpu.u_len != gpu.u_len {
        panic!("append {append}: {label} claim {idx} u-range mismatch");
    }
    panic!("append {append}: {label} claim {idx} mismatch in an unclassified field");
}

/// Assert two audit `Debug` representations are identical; on mismatch print
/// a window around the first divergence instead of dumping both proofs.
fn assert_audit_identical(cpu: &str, gpu: &str) {
    if cpu == gpu {
        return;
    }
    let split = cpu
        .bytes()
        .zip(gpu.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(cpu.len().min(gpu.len()));
    let start = split.saturating_sub(60);
    let cpu_window = &cpu[start..(split + 60).min(cpu.len())];
    let gpu_window = &gpu[start..(split + 60).min(gpu.len())];
    panic!(
        "e2e audit mismatch at byte {split} (cpu len {}, gpu len {}):\n  cpu: …{cpu_window}…\n  gpu: …{gpu_window}…",
        cpu.len(),
        gpu.len()
    );
}

/// Phase 5 end gate: full lifecycle on the real SHA-256 serial workload.
/// The CPU chain and the CUDA-adapter chain run from the same preprocessing;
/// the audits (state, per-step proofs, terminal fold) must be identical and
/// the final semantic digest must match the sha2-computed ground truth.
pub fn e2e_bench() {
    let _ = run_sha256_lifecycle("e2e_bench", |_| {});
}

/// Full lifecycle fast-path benchmark that keeps private terminal witnesses
/// on device and compares only the public/proof/audit material. The regular
/// `e2e_bench` remains the full terminal-witness parity contract.
pub fn e2e_gpu_fast_bench() {
    let _ = run_sha256_lifecycle_terminal_claims_only("e2e_gpu_fast_bench", |_| {});
}

/// Throughput gate for the real dependency graph: independent SuperNeo chains
/// should be scheduled concurrently instead of serialized by the old CPU call
/// order. This does not change one chain's Fiat-Shamir spine; it measures
/// whether separate chains can fill GPU idle space while preserving full
/// byte-identical audit parity against the CPU chain.
pub fn e2e_multichain_bench() {
    run_e2e_multichain_bench("e2e_multichain_bench", MULTICHAIN_COUNT, MultichainMode::FullAudit);
}

/// Eight-chain throughput gate for the same dependency-graph scheduling
/// contract. It intentionally widens only across independent chains; every
/// chain's internal Fiat-Shamir order remains unchanged.
pub fn e2e_multichain8_bench() {
    run_e2e_multichain_bench("e2e_multichain8_bench", MULTICHAIN8_COUNT, MultichainMode::FullAudit);
}

/// Eight-chain throughput gate for the terminal-private fast-path contract.
/// Full terminal witness parity remains covered by `e2e_bench` and
/// `e2e_multichain8_bench`; this gate measures the dependency-graph schedule
/// when terminal private material stays device-resident.
pub fn e2e_multichain8_fast_bench() {
    run_e2e_multichain_bench(
        "e2e_multichain8_fast_bench",
        MULTICHAIN8_COUNT,
        MultichainMode::TerminalClaimsOnly,
    );
}

/// Sixteen-chain throughput gate for saturating the GPU with independent
/// SuperNeo chains. This is the dependency-graph version of "use more CUDA
/// cores": widen across chains that share no Fiat-Shamir dependency.
pub fn e2e_multichain16_fast_bench() {
    run_e2e_multichain_bench(
        "e2e_multichain16_fast_bench",
        MULTICHAIN16_COUNT,
        MultichainMode::TerminalClaimsOnly,
    );
}

fn run_e2e_multichain_bench(label: &str, chain_count: usize, mode: MultichainMode) {
    let fixture = build_sha256_e2e_fixture();
    let cpu = run_chain_without_append_snapshots(&fixture.prep, &fixture.chunks, &fixture.states[0], None);
    let final_state = fixture.states.last().expect("final state");
    let (sequential_ctx, sequential_context_ms) =
        timed(|| Device::open_context().expect("open shared CUDA context for sequential chains"));
    let (mut sequential_provers, sequential_init_ms, sequential_prepare_ms) =
        prepare_cuda_provers_on_context(&fixture, chain_count, &sequential_ctx, mode);
    let (sequential_runs, sequential_wall_ms) = run_cuda_chains_sequential(&fixture, &mut sequential_provers);
    for gpu in &sequential_runs {
        assert_cuda_chain_matches_cpu(&cpu, gpu, final_state, mode);
    }
    let sequential_thread_sum_ms = sequential_runs.iter().map(ChainRun::online_ms).sum::<f64>();
    drop(sequential_runs);
    drop(sequential_provers);
    drop(sequential_ctx);

    let (parallel_ctx, parallel_context_ms) =
        timed(|| Device::open_context().expect("open shared CUDA context for parallel chains"));
    let (parallel_runs, parallel_wall_ms) = run_cuda_chains_parallel(&fixture, chain_count, parallel_ctx, mode);
    for gpu in &parallel_runs {
        assert_cuda_chain_matches_cpu(&cpu, &gpu.run, final_state, mode);
    }

    let cpu_aggregate_ms = cpu.online_ms() * chain_count as f64;
    let parallel_thread_sum_ms = parallel_runs
        .iter()
        .map(|gpu| gpu.run.online_ms())
        .sum::<f64>();
    let parallel_init_ms = parallel_runs.iter().map(|gpu| gpu.init_ms).sum::<f64>();
    let parallel_prepare_ms = parallel_runs.iter().map(|gpu| gpu.prepare_ms).sum::<f64>();
    eprintln!(
        "  synth {:.0}ms  setup {:.0}ms  chains {}",
        fixture.synth_ms, fixture.setup_ms, chain_count
    );
    eprintln!(
        "  sequential context {:.0}ms  stream-init {:.0}ms  cuda-prepare {:.0}ms  outer-wall {:.2}ms  thread-sum {:.2}ms",
        sequential_context_ms, sequential_init_ms, sequential_prepare_ms, sequential_wall_ms, sequential_thread_sum_ms
    );
    eprintln!(
        "  parallel   context {:.0}ms  stream-init {:.0}ms  cuda-prepare {:.0}ms  outer-wall {:.2}ms  thread-sum {:.2}ms",
        parallel_context_ms, parallel_init_ms, parallel_prepare_ms, parallel_wall_ms, parallel_thread_sum_ms
    );
    println!(
        "[parity {label}] OK: {chain_count} independent sha256 chains byte-identical ({mode:?}); cpu aggregate={cpu_aggregate_ms:.2}ms sequential cuda={sequential_wall_ms:.2}ms ({:.2}x) parallel cuda={parallel_wall_ms:.2}ms ({:.2}x) overlap={:.2}x",
        cpu_aggregate_ms / sequential_wall_ms,
        cpu_aggregate_ms / parallel_wall_ms,
        sequential_wall_ms / parallel_wall_ms,
    );
}

/// Fast-path benchmark with the whole-FE device transcript path enabled.
///
/// This keeps the same terminal claims-only contract as `e2e_gpu_fast_bench`
/// so Pi_CCS phase scheduling can be compared without terminal witness export
/// noise.
pub fn e2e_whole_fe_fast_bench() {
    let _ = run_sha256_lifecycle_terminal_claims_only("e2e_whole_fe_fast_bench", |cuda| {
        cuda.enable_whole_fe_trace_fast();
    });
}

/// Full lifecycle with FE rows + Ajtai tail sourced from the device
/// transcript, but without CUDA graph capture/replay. This isolates the
/// structural CPU→GPU phase migration from the graph replay blocker.
pub fn e2e_whole_fe_bench() {
    let _ = run_sha256_lifecycle("e2e_whole_fe_bench", |cuda| {
        cuda.enable_whole_fe_trace_fast();
    });
}

/// Full lifecycle graph repro gate. It is intentionally separate from
/// `e2e_bench` so the production/default timing contract does not silently
/// change while the whole-FE graph path is still being stabilized.
pub fn e2e_graph_bench() {
    let _ = run_sha256_lifecycle("e2e_graph_bench", |cuda| {
        cuda.enable_whole_fe_graph_fast();
    });
}

/// Full lifecycle graph isolation gate. Only the first Π_CCS call uses the
/// whole-FE graph path; later folds stay on the default row-trace path.
pub fn e2e_graph_once_bench() {
    let _ = run_sha256_lifecycle("e2e_graph_once_bench", |cuda| {
        cuda.enable_whole_fe_graph_budget_fast(1);
    });
}

/// Full lifecycle graph isolation gate. The first two Π_CCS calls use the
/// whole-FE graph path; later folds stay on the default row-trace path.
pub fn e2e_graph_two_bench() {
    let _ = run_sha256_lifecycle("e2e_graph_two_bench", |cuda| {
        cuda.enable_whole_fe_graph_budget_fast(2);
    });
}

/// Full lifecycle graph isolation gate. The first three Π_CCS calls use the
/// whole-FE graph path; later folds stay on the default row-trace path.
pub fn e2e_graph_three_bench() {
    let _ = run_sha256_lifecycle("e2e_graph_three_bench", |cuda| {
        cuda.enable_whole_fe_graph_budget_fast(3);
    });
}

/// Full lifecycle graph ownership isolation gate. The first three Π_CCS calls
/// use fresh whole-FE graph captures instead of replaying a cached graph.
pub fn e2e_graph_three_recapture_bench() {
    let _ = run_sha256_lifecycle("e2e_graph_three_recapture_bench", |cuda| {
        cuda.enable_whole_fe_graph_recapture_budget_fast(3);
    });
}
