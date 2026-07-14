//! Full NIFS.P session gates (Phase 5).

use super::*;

const NIFS_TRANSCRIPT_LABEL: &[u8] = b"neo-prover-cuda/parity/nifs";

fn assert_rounds_identical(fold: usize, label: &str, gpu: &[Vec<K>], cpu: &[Vec<K>]) {
    assert_eq!(
        gpu.len(),
        cpu.len(),
        "pi_ccs {label} round count mismatch at fold {fold}: gpu={} cpu={}",
        gpu.len(),
        cpu.len()
    );
    for (round, (gpu_round, cpu_round)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_eq!(
            gpu_round.len(),
            cpu_round.len(),
            "pi_ccs {label} coeff count mismatch at fold {fold}, round {round}: gpu={} cpu={}",
            gpu_round.len(),
            cpu_round.len()
        );
        for (coeff, (gpu_coeff, cpu_coeff)) in gpu_round.iter().zip(cpu_round.iter()).enumerate() {
            assert_eq!(
                gpu_coeff, cpu_coeff,
                "pi_ccs {label} coeff mismatch at fold {fold}, round {round}, coeff {coeff}"
            );
        }
    }
}

fn assert_pi_ccs_sumcheck_identical(
    fold: usize,
    gpu: &neo_reductions::api::PiCcsProof,
    cpu: &neo_reductions::api::PiCcsProof,
) {
    assert_eq!(
        gpu.variant, cpu.variant,
        "pi_ccs sumcheck variant mismatch at fold {fold}"
    );
    assert_eq!(
        gpu.sc_initial_sum, cpu.sc_initial_sum,
        "pi_ccs FE initial sum mismatch at fold {fold}"
    );
    assert_eq!(
        gpu.sc_initial_sum_nc, cpu.sc_initial_sum_nc,
        "pi_ccs NC initial sum mismatch at fold {fold}"
    );
    assert_eq!(
        gpu.challenges_public.alpha, cpu.challenges_public.alpha,
        "pi_ccs public alpha mismatch at fold {fold}"
    );
    assert_eq!(
        gpu.challenges_public.beta_a, cpu.challenges_public.beta_a,
        "pi_ccs public beta_a mismatch at fold {fold}"
    );
    assert_eq!(
        gpu.challenges_public.beta_r, cpu.challenges_public.beta_r,
        "pi_ccs public beta_r mismatch at fold {fold}"
    );
    assert_eq!(
        gpu.challenges_public.beta_m, cpu.challenges_public.beta_m,
        "pi_ccs public beta_m mismatch at fold {fold}"
    );
    assert_eq!(
        gpu.challenges_public.gamma, cpu.challenges_public.gamma,
        "pi_ccs public gamma mismatch at fold {fold}"
    );
    assert_rounds_identical(fold, "FE rounds", &gpu.sumcheck_rounds, &cpu.sumcheck_rounds);
    assert_eq!(
        gpu.sumcheck_challenges, cpu.sumcheck_challenges,
        "pi_ccs FE challenges mismatch at fold {fold}"
    );
    assert_eq!(
        gpu.sumcheck_final, cpu.sumcheck_final,
        "pi_ccs FE final mismatch at fold {fold}"
    );
    assert_rounds_identical(fold, "NC rounds", &gpu.sumcheck_rounds_nc, &cpu.sumcheck_rounds_nc);
    assert_eq!(
        gpu.sumcheck_challenges_nc, cpu.sumcheck_challenges_nc,
        "pi_ccs NC challenges mismatch at fold {fold}"
    );
    assert_eq!(
        gpu.sumcheck_final_nc, cpu.sumcheck_final_nc,
        "pi_ccs NC final mismatch at fold {fold}"
    );
    assert_eq!(
        gpu.header_digest, cpu.header_digest,
        "pi_ccs header digest mismatch at fold {fold}"
    );
    assert_eq!(gpu._extra, cpu._extra, "pi_ccs extra payload mismatch at fold {fold}");

    let gpu_sumcheck = serde_json::to_string(gpu).expect("serialize gpu sumcheck");
    let cpu_sumcheck = serde_json::to_string(cpu).expect("serialize cpu sumcheck");
    assert_eq!(
        gpu_sumcheck, cpu_sumcheck,
        "pi_ccs sumcheck bytes mismatch at fold {fold}"
    );
}

/// Assert a GPU fold's running instance and proof are field-identical to
/// the CPU fold's, including full Π_CCS sumcheck bytes (via serde).
fn assert_nifs_fold_identical(fold: usize, gpu: &(RunningInstance, NifsProof), cpu: &(RunningInstance, NifsProof)) {
    let (gpu_running, gpu_proof) = gpu;
    let (cpu_running, cpu_proof) = cpu;
    assert_eq!(
        gpu_running.claims, cpu_running.claims,
        "running claims mismatch at fold {fold}"
    );
    assert_eq!(
        gpu_running.witnesses, cpu_running.witnesses,
        "running witnesses mismatch at fold {fold}"
    );
    assert_eq!(
        gpu_running.parent_authority, cpu_running.parent_authority,
        "parent authority mismatch at fold {fold}"
    );
    assert_eq!(
        gpu_proof.pi_ccs.outputs, cpu_proof.pi_ccs.outputs,
        "pi_ccs outputs mismatch at fold {fold}"
    );
    assert_eq!(
        gpu_proof.pi_ccs.outputs_digest, cpu_proof.pi_ccs.outputs_digest,
        "pi_ccs output digest mismatch at fold {fold}"
    );
    assert_pi_ccs_sumcheck_identical(fold, &gpu_proof.pi_ccs.sumcheck, &cpu_proof.pi_ccs.sumcheck);
    assert_eq!(
        gpu_proof.pi_rlc.combined, cpu_proof.pi_rlc.combined,
        "pi_rlc combined mismatch at fold {fold}"
    );
    assert_eq!(
        gpu_proof.pi_dec.children, cpu_proof.pi_dec.children,
        "pi_dec children mismatch at fold {fold}"
    );
}

fn assert_nifs_fold_public_identical(
    fold: usize,
    gpu: &(RunningInstance, NifsProof),
    cpu: &(RunningInstance, NifsProof),
) {
    let (gpu_running, gpu_proof) = gpu;
    let (cpu_running, cpu_proof) = cpu;
    assert_eq!(
        gpu_running.claims, cpu_running.claims,
        "running claims mismatch at cached fold {fold}"
    );
    assert_eq!(
        gpu_running.parent_authority, cpu_running.parent_authority,
        "parent authority mismatch at cached fold {fold}"
    );
    assert_eq!(
        gpu_proof.pi_ccs.outputs, cpu_proof.pi_ccs.outputs,
        "pi_ccs outputs mismatch at cached fold {fold}"
    );
    assert_eq!(
        gpu_proof.pi_ccs.outputs_digest, cpu_proof.pi_ccs.outputs_digest,
        "pi_ccs output digest mismatch at cached fold {fold}"
    );
    assert_pi_ccs_sumcheck_identical(fold, &gpu_proof.pi_ccs.sumcheck, &cpu_proof.pi_ccs.sumcheck);
    assert_eq!(
        gpu_proof.pi_rlc.combined, cpu_proof.pi_rlc.combined,
        "pi_rlc combined mismatch at cached fold {fold}"
    );
    assert_eq!(
        gpu_proof.pi_dec.children, cpu_proof.pi_dec.children,
        "pi_dec child claims mismatch at cached fold {fold}"
    );
}

/// One NIFS.P fold pair: the CPU chain and the CUDA adapter chain advance in
/// lockstep from identical transcripts, fresh batches, and running state.
struct NifsChainPair {
    cpu_tr: Transcript,
    gpu_tr: Transcript,
    cpu_running: RunningInstance,
    gpu_running: RunningInstance,
    cuda: CudaNifsProver,
}

impl NifsChainPair {
    fn new() -> Self {
        Self {
            cpu_tr: Transcript::with_label(NIFS_TRANSCRIPT_LABEL),
            gpu_tr: Transcript::with_label(NIFS_TRANSCRIPT_LABEL),
            cpu_running: RunningInstance::default(),
            gpu_running: RunningInstance::default(),
            cuda: CudaNifsProver::new().expect("open CUDA NIFS prover"),
        }
    }

    fn new_whole_phase() -> Self {
        let mut pair = Self::new();
        pair.cuda.enable_whole_fe_trace_for_parity();
        pair
    }

    /// Fold one fresh batch on both chains; returns (cpu_ms, gpu_ms).
    fn fold(&mut self, fixture: &Fixture, fold: usize, batch: &[CcsInstance]) -> (f64, f64) {
        self.fold_with_cache(fixture, fold, batch, false)
    }

    fn fold_with_cache(
        &mut self,
        fixture: &Fixture,
        fold: usize,
        batch: &[CcsInstance],
        cache_gpu_output: bool,
    ) -> (f64, f64) {
        let (cpu, cpu_ms) = timed(|| {
            nifs_cpu_prove(
                &mut self.cpu_tr,
                &fixture.prep.params,
                fixture.structure(),
                fixture.prep.optimized_cache(),
                &fixture.prep.log,
                ajtai_rlc_mixer,
                ajtai_dec_mixer,
                batch.to_vec(),
                &self.cpu_running,
            )
            .expect("CPU NIFS prove")
        });
        let (gpu, gpu_ms) = timed(|| {
            self.cuda
                .prove(NifsProverRequest {
                    tr: &mut self.gpu_tr,
                    pp: &fixture.prep.params,
                    s: fixture.structure(),
                    cache: fixture.prep.optimized_cache(),
                    log: &fixture.prep.log,
                    mix_rhos_commits: ajtai_rlc_mixer,
                    combine_b_pows: ajtai_dec_mixer,
                    fresh: batch.to_vec(),
                    running_carrier: None,
                    running: &self.gpu_running,
                    cache_output_for_next_step: cache_gpu_output,
                })
                .expect("GPU NIFS prove")
                .into_materialized_parts()
                .expect("GPU NIFS proof materialization")
        });
        if cache_gpu_output {
            assert_nifs_fold_public_identical(fold, &gpu, &cpu);
        } else {
            assert_nifs_fold_identical(fold, &gpu, &cpu);
        }
        assert_eq!(
            self.gpu_tr.snapshot(),
            self.cpu_tr.snapshot(),
            "transcript mismatch after fold {fold}"
        );
        self.cpu_running = cpu.0;
        self.gpu_running = gpu.0;
        (cpu_ms, gpu_ms)
    }
}

fn nifs_fresh_batches(fixture: &Fixture, folds: usize, k_fresh: usize, rng: &mut StdRng) -> Vec<Vec<CcsInstance>> {
    (0..folds)
        .map(|_| {
            (0..k_fresh)
                .map(|_| fixture.satisfying_binary_instance(rng))
                .collect()
        })
        .collect()
}

/// Phase 5 parity gate: full NIFS.P folds through the CUDA adapter must be
/// byte-identical to the CPU chain — proofs, running state, and transcript —
/// across multiple folds (empty running, then steady-state k children).
pub fn nifs() {
    const K_FRESH: usize = 2;
    const FOLDS: usize = 3;
    let fixture = Fixture::r1cs_identity(FIXTURE_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6e69_6673_5f70_3531);
    let batches = nifs_fresh_batches(&fixture, FOLDS, K_FRESH, &mut rng);

    let mut chains = NifsChainPair::new();
    let mut cpu_total = 0.0;
    let mut gpu_total = 0.0;
    for (fold, batch) in batches.iter().enumerate() {
        let (cpu_ms, gpu_ms) = chains.fold(&fixture, fold, batch);
        cpu_total += cpu_ms;
        gpu_total += gpu_ms;
    }
    println!(
        "[parity nifs] OK: {FOLDS} NIFS.P folds identical (m={FIXTURE_N}, K={K_FRESH}, proofs+running+transcript); total cpu={cpu_total:.2}ms gpu={gpu_total:.2}ms"
    );
}

pub fn nifs_whole_phase() {
    const K_FRESH: usize = 1;
    const FOLDS: usize = 4;
    let fixture = Fixture::r1cs_identity(FIXTURE_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6e69_6673_5f77_686f);
    let batches = nifs_fresh_batches(&fixture, FOLDS, K_FRESH, &mut rng);

    let mut chains = NifsChainPair::new_whole_phase();
    for (fold, batch) in batches.iter().enumerate() {
        let cache_gpu_output = fold + 1 < batches.len();
        let _ = chains.fold_with_cache(&fixture, fold, batch, cache_gpu_output);
    }
    println!("[parity nifs_whole_phase] OK: whole-Π_CCS phase backend matches NIFS.P folds");
}

/// Phase 5 perf gate at real scale: steady-state NIFS.P fold (fold 0 warms
/// PP/kernels/bar uploads and fills the running instance to k children).
pub fn nifs_bench() {
    const K_FRESH: usize = 2;
    const FOLDS: usize = 3;
    let fixture = Fixture::r1cs_identity(BENCH_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6e69_6673_5f62_6e63);
    let batches = nifs_fresh_batches(&fixture, FOLDS, K_FRESH, &mut rng);

    let mut chains = NifsChainPair::new();
    let mut cpu_steady = 0.0;
    let mut gpu_steady = 0.0;
    for (fold, batch) in batches.iter().enumerate() {
        let (cpu_ms, gpu_ms) = chains.fold(&fixture, fold, batch);
        if fold > 0 {
            cpu_steady += cpu_ms;
            gpu_steady += gpu_ms;
        }
    }
    let steady = (FOLDS - 1) as f64;
    println!(
        "[parity nifs_bench] OK: m={BENCH_N} {FOLDS} folds identical; steady-state per-fold cpu={:.2}ms gpu={:.2}ms",
        cpu_steady / steady,
        gpu_steady / steady,
    );
}
