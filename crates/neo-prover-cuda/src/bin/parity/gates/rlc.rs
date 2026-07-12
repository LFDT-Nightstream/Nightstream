//! Π_RLC gates (Phase 4).

use super::ccs::CcsProveCase;
use super::*;

/// One Π_RLC workload: honest Π_CCS output claims plus their witnesses,
/// optionally replicated to the real chain's K+k input count. Both provers
/// start from the same labeled transcript so the ρ challenges match.
struct RlcCase {
    claims: Vec<neo_fold_clean::CeClaim>,
    witnesses: Vec<neo_ccs::Mat<F>>,
}

const RLC_TRANSCRIPT_LABEL: &[u8] = b"neo-prover-cuda/parity/pi_rlc";

impl RlcCase {
    /// Run the CPU Π_CCS prover once to get honest CE claims, then cycle
    /// them up to `count` inputs (duplicates are legal Π_RLC inputs).
    fn from_ccs_outputs(fixture: &Fixture, k_fresh: usize, count: usize, rng: &mut StdRng) -> Self {
        let case = CcsProveCase::build(fixture, k_fresh, rng);
        let (outputs, _, _) = case.prove_cpu(fixture);
        let claims: Vec<neo_fold_clean::CeClaim> = outputs.iter().cycle().take(count).cloned().collect();
        let witnesses: Vec<neo_ccs::Mat<F>> = case
            .mcs_witnesses
            .iter()
            .map(|w| w.Z.clone())
            .cycle()
            .take(count)
            .collect();
        Self { claims, witnesses }
    }

    fn prove_cpu(&self, fixture: &Fixture) -> (pi_rlc::Output, pi_rlc::Proof, Transcript) {
        let mut tr = Transcript::with_label(RLC_TRANSCRIPT_LABEL);
        let (out, proof) = pi_rlc::prove(
            &mut tr,
            &fixture.prep.params,
            fixture.structure(),
            ajtai_rlc_mixer,
            &self.claims,
            &self.witnesses,
        )
        .expect("CPU pi_rlc prove");
        (out, proof, tr)
    }

    /// Device Π_RLC: ρ coefficients sampled by the device transcript,
    /// claim assembly on host, witness mix from the device rho buffer.
    fn prove_gpu(
        &self,
        fixture: &Fixture,
        device: &Device,
        kernels: &SumcheckKernels,
    ) -> (neo_fold_clean::CeClaim, neo_ccs::Mat<F>, Transcript) {
        let mut tr = Transcript::with_label(RLC_TRANSCRIPT_LABEL);
        let sampling_start =
            pi_rlc::begin_rho_sampling(&mut tr, &fixture.prep.params, &self.claims).expect("begin rho sampling");
        let (mut device_rhos, sampling_end) =
            device_rlc::sample_rhos_device(device, kernels, &fixture.prep.params, sampling_start, self.claims.len())
                .expect("device rho sampling");
        tr.restore_snapshot(sampling_end);
        let inputs_c: Vec<neo_ajtai::Commitment> = self.claims.iter().map(|claim| claim.c.clone()).collect();
        let commitment =
            device_rlc::mix_commitments_device_with_rho_coeffs(device, kernels.ring(), device_rhos.coeffs(), &inputs_c)
                .expect("device commitment mix");
        let mut combined = device_rlc::claim_shell_from_device_rhos(device, &self.claims, &mut device_rhos, commitment)
            .expect("device claim shell");
        let witness_refs: Vec<&neo_ccs::Mat<F>> = self.witnesses.iter().collect();
        let cols = self.witnesses[0].cols();
        let planes = upload_witness_planes(device, &witness_refs).expect("upload witness planes");
        let z_dev = device_rlc::mix_planes_device_with_rho_coeffs(
            device,
            kernels.ring(),
            device_rhos.coeffs(),
            &planes,
            self.witnesses.len(),
            cols,
        )
        .expect("device rlc mix");
        let x_dev = device_rlc::project_x_from_mixed_witness(
            device,
            kernels.rlc(),
            &z_dev,
            cols,
            fixture.structure().m,
            combined.m_in,
        )
        .expect("device X projection");
        combined.X = x_dev;
        let y_ring_dev = device_rlc::combine_y_ring(
            device,
            kernels.rlc(),
            device_rhos.coeffs(),
            &self.claims,
            fixture.structure().t(),
        )
        .expect("device y_ring combine");
        combined.y_ring = y_ring_dev;
        combined.ct = neo_reductions::common::ct_from_y_ring_for_ccs_m(
            &combined.y_ring,
            fixture.prep.params.inner(),
            fixture.structure().m,
        );
        let y_zcol_dev = device_rlc::combine_y_zcol(device, kernels.rlc(), device_rhos.coeffs(), &self.claims)
            .expect("device y_zcol combine");
        combined.y_zcol = y_zcol_dev;
        let rhos = device_rhos
            .mats(device, &fixture.prep.params)
            .expect("materialize rho matrices for projection schedule");
        let rhos = neo_reductions::common::rot_rhos_from_mats(
            fixture.prep.params.inner(),
            &rhos,
            "CUDA parity projection schedule",
        )
        .expect("validate rho matrices for projection schedule");
        pi_rlc::bind_backend_projection_schedule(&mut tr, &rhos, &self.claims, &combined)
            .expect("bind backend projection schedule");
        let z_mix = device_rlc::download_witness(device, &z_dev, cols).expect("download rlc mix");
        (combined, z_mix, tr)
    }
}

/// Π_RLC parity gate: the device witness mix plus host claim combination
/// must reproduce the paper prover bit-exactly, including the transcript.
pub fn rlc() {
    const K_FRESH: usize = 2;
    let fixture = Fixture::r1cs_identity(FIXTURE_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x726c_635f_7061_7231);
    let case = RlcCase::from_ccs_outputs(&fixture, K_FRESH, K_FRESH, &mut rng);

    let ((cpu_out, cpu_proof, cpu_tr), cpu_ms) = timed(|| case.prove_cpu(&fixture));
    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let ((gpu_claim, gpu_z, gpu_tr), gpu_ms) = timed(|| case.prove_gpu(&fixture, &device, &kernels));

    assert_eq!(gpu_claim, cpu_out.claim, "combined claim mismatch");
    assert_eq!(gpu_claim, cpu_proof.combined, "proof.combined mismatch");
    assert_eq!(gpu_z, cpu_out.witness, "Z_mix mismatch");
    assert_eq!(gpu_tr.snapshot(), cpu_tr.snapshot(), "post-prove transcript mismatch");
    println!(
        "[parity rlc] OK: combined claim + Z_mix identical (m={FIXTURE_N}, {K_FRESH} inputs); cpu={cpu_ms:.2}ms gpu={gpu_ms:.2}ms"
    );
}

/// Phase 4 perf gate at real scale and the real chain's input count
/// (K + k = 16): steady-state Π_RLC prove, warmed, parity per round.
pub fn rlc_bench() {
    const K_FRESH: usize = 2;
    const INPUTS: usize = 16;
    const ROUNDS: usize = 3;
    let fixture = Fixture::r1cs_identity(BENCH_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x726c_635f_6265_6e63);
    let case = RlcCase::from_ccs_outputs(&fixture, K_FRESH, INPUTS, &mut rng);

    let _warm = case.prove_cpu(&fixture);
    let (cpu, cpu_total_ms) = timed(|| {
        (0..ROUNDS)
            .map(|_| case.prove_cpu(&fixture))
            .collect::<Vec<_>>()
    });

    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let _warm = case.prove_gpu(&fixture, &device, &kernels);
    let (gpu, gpu_total_ms) = timed(|| {
        (0..ROUNDS)
            .map(|_| case.prove_gpu(&fixture, &device, &kernels))
            .collect::<Vec<_>>()
    });

    for (round, ((gpu_claim, gpu_z, gpu_tr), (cpu_out, _, cpu_tr))) in gpu.iter().zip(&cpu).enumerate() {
        assert_eq!(gpu_claim, &cpu_out.claim, "combined claim mismatch at round {round}");
        assert_eq!(gpu_z, &cpu_out.witness, "Z_mix mismatch at round {round}");
        assert_eq!(
            gpu_tr.snapshot(),
            cpu_tr.snapshot(),
            "transcript mismatch at round {round}"
        );
    }
    println!(
        "[parity rlc_bench] OK: m={BENCH_N} {INPUTS} inputs identical; per-prove cpu={:.2}ms gpu={:.2}ms",
        cpu_total_ms / ROUNDS as f64,
        gpu_total_ms / ROUNDS as f64,
    );
}
