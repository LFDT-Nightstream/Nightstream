//! Π_DEC gates (Phase 2).

use super::*;

/// Π_DEC gate: `DeviceDec::prove` must equal `paper::pi_dec::prove` on an
/// internally consistent parent, including the NC channel.
pub fn dec() {
    let fixture = Fixture::identity_ccs(FIXTURE_N, FIXTURE_T, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6465_635f_7061_7231);
    let witness = fixture.dec_parent_witness(&mut rng);
    let parent = fixture.consistent_parent_claim(&witness, &mut rng);
    let k = fixture.prep.params.k_rho();

    let ((cpu_children, cpu_proof), cpu_ms) = timed(|| cpu_dec_prove(&fixture, &parent, &witness));
    let mut gpu = DecGpu::open(&fixture);
    let ((gpu_children, gpu_proof), gpu_ms) = timed(|| gpu.prove(&fixture, &parent, &witness));

    assert_eq!(gpu_children.claims, cpu_children.claims, "child claims mismatch");
    assert_eq!(
        gpu_children.witnesses, cpu_children.witnesses,
        "child witnesses mismatch"
    );
    assert_eq!(gpu_proof.children, cpu_proof.children, "proof children mismatch");

    let resident = gpu.prove_resident(&fixture, &parent, &witness);
    let resident_commitments = gpu
        .ajtai
        .download_commitments(&gpu.device, &resident.child_commitment_words, k as usize)
        .expect("download resident DEC child commitments");
    assert_eq!(
        resident_commitments,
        cpu_children
            .claims
            .iter()
            .map(|claim| claim.c.clone())
            .collect::<Vec<_>>(),
        "resident DEC child commitment mismatch"
    );
    let surfaces = resident
        .child_surfaces
        .expect("resident DEC child surfaces");
    let decoded = surfaces
        .download_surfaces(&gpu.device)
        .expect("download resident DEC child surfaces");
    for (claim, child_surfaces) in cpu_children.claims.iter().zip(&decoded) {
        assert_eq!(claim.y_ring, child_surfaces[..fixture.structure().t()]);
        assert_eq!(claim.y_zcol, child_surfaces[fixture.structure().t()]);
    }
    let alpha = (0..D.next_power_of_two().trailing_zeros())
        .map(|_| rand_k(&mut rng))
        .collect::<Vec<_>>();
    let gamma = rand_k(&mut rng);
    let challenges = neo_reductions::optimized_engine::Challenges {
        alpha: alpha.clone(),
        beta_a: Vec::new(),
        beta_r: Vec::new(),
        beta_m: Vec::new(),
        gamma,
    };
    let expected = neo_reductions::optimized_engine::claimed_initial_sum_from_inputs_with_k_mcs(
        fixture.structure(),
        &challenges,
        1,
        &cpu_children.claims,
    );
    let actual = surfaces
        .claimed_initial_sum(&gpu.device, &gpu.kernels, &alpha, gamma, 1)
        .expect("resident DEC claimed initial sum");
    assert_eq!(actual, expected, "resident DEC claimed initial sum mismatch");

    let mut sparse_witness = neo_ccs::Mat::zero(witness.rows(), witness.cols(), F::ZERO);
    sparse_witness[(0, 0)] = F::ONE;
    let sparse_parent = fixture.consistent_parent_claim(&sparse_witness, &mut rng);
    let (sparse_cpu, _) = cpu_dec_prove(&fixture, &sparse_parent, &sparse_witness);
    let sparse_resident = gpu.prove_resident(&fixture, &sparse_parent, &sparse_witness);
    let sparse_surfaces = sparse_resident
        .child_surfaces
        .expect("sparse resident DEC child surfaces");
    let sparse_decoded = sparse_surfaces
        .download_surfaces(&gpu.device)
        .expect("download sparse resident DEC child surfaces");
    for (claim, child_surfaces) in sparse_cpu.claims.iter().zip(&sparse_decoded) {
        assert_eq!(claim.y_ring, child_surfaces[..fixture.structure().t()]);
        assert_eq!(claim.y_zcol, child_surfaces[fixture.structure().t()]);
    }
    let sparse_expected = neo_reductions::optimized_engine::claimed_initial_sum_from_inputs_with_k_mcs(
        fixture.structure(),
        &challenges,
        1,
        &sparse_cpu.claims,
    );
    let sparse_actual = sparse_surfaces
        .claimed_initial_sum(&gpu.device, &gpu.kernels, &alpha, gamma, 1)
        .expect("sparse resident DEC claimed initial sum");
    assert_eq!(
        sparse_actual, sparse_expected,
        "sparse resident DEC claimed initial sum mismatch"
    );
    println!(
        "[parity dec] OK: k={k} children (m={FIXTURE_N}, t={FIXTURE_T}) identical incl. NC channel; cpu={cpu_ms:.2}ms gpu={gpu_ms:.2}ms"
    );
}

/// Phase 2 perf gate at real sha256 scale (m = 452k, t = 3): steady-state
/// `DeviceDec::prove` vs the CPU prover on the same fixture, warmed, with
/// parity asserted on every round.
pub fn dec_bench() {
    const ROUNDS: usize = 2;
    let fixture = Fixture::identity_ccs(BENCH_N, FIXTURE_T, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6465_6362_656e_6368);
    let inputs: Vec<_> = (0..ROUNDS)
        .map(|_| {
            let witness = fixture.dec_parent_witness(&mut rng);
            let parent = fixture.consistent_parent_claim(&witness, &mut rng);
            (parent, witness)
        })
        .collect();

    let _warm = cpu_dec_prove(&fixture, &inputs[0].0, &inputs[0].1);
    let (cpu, cpu_total_ms) = timed(|| {
        inputs
            .iter()
            .map(|(parent, witness)| cpu_dec_prove(&fixture, parent, witness))
            .collect::<Vec<_>>()
    });

    let mut gpu = DecGpu::open(&fixture);
    let _warm = gpu.prove(&fixture, &inputs[0].0, &inputs[0].1);
    let (got, gpu_total_ms) = timed(|| {
        inputs
            .iter()
            .map(|(parent, witness)| gpu.prove(&fixture, parent, witness))
            .collect::<Vec<_>>()
    });

    for (round, ((gpu_children, _), (cpu_children, _))) in got.iter().zip(&cpu).enumerate() {
        assert_eq!(
            gpu_children.claims, cpu_children.claims,
            "claims mismatch at round {round}"
        );
    }
    println!(
        "[parity dec_bench] OK: m={BENCH_N} t={FIXTURE_T} identical; per-prove cpu={:.2}ms gpu={:.2}ms",
        cpu_total_ms / ROUNDS as f64,
        gpu_total_ms / ROUNDS as f64,
    );
}

type DecOutput = (
    neo_fold_clean::paper::pi_dec::Children,
    neo_fold_clean::paper::pi_dec::Proof,
);

fn cpu_dec_prove(fixture: &Fixture, parent: &neo_fold_clean::CeClaim, witness: &neo_ccs::Mat<F>) -> DecOutput {
    neo_fold_clean::paper::pi_dec::prove(
        &fixture.prep.params,
        fixture.structure(),
        fixture.prep.optimized_cache(),
        &fixture.prep.log,
        ajtai_dec_mixer,
        parent,
        witness,
    )
    .expect("CPU pi_dec prove")
}

/// One opened device with the fixture's PP uploaded and DEC kernels loaded.
struct DecGpu {
    device: Device,
    kernels: SumcheckKernels,
    ajtai: DeviceAjtai,
    dec: DeviceDec,
    bar_matrices: Option<DeviceBarMatrices>,
}

impl DecGpu {
    fn open(fixture: &Fixture) -> Self {
        let device = Device::open().expect("open CUDA device");
        let pp = fixture.prep.log.materialize_pp().expect("materialize PP");
        let ajtai = DeviceAjtai::upload(&device, &pp).expect("upload PP");
        let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
        let dec = DeviceDec::new(&device).expect("load DEC kernels");
        Self {
            device,
            kernels,
            ajtai,
            dec,
            bar_matrices: None,
        }
    }

    fn prove(&mut self, fixture: &Fixture, parent: &neo_fold_clean::CeClaim, witness: &neo_ccs::Mat<F>) -> DecOutput {
        let output = self
            .dec
            .prove(
                &self.device,
                &self.kernels,
                &mut self.ajtai,
                &mut self.bar_matrices,
                &fixture.prep.params,
                fixture.structure(),
                fixture.prep.optimized_cache(),
                ajtai_dec_mixer,
                parent,
                DecParentWitness::Host(witness),
                None,
                None,
                DecOutputMode::Full,
                DecRecompositionMode::Full,
            )
            .expect("GPU pi_dec prove");
        (output.children, output.proof)
    }

    fn prove_resident(
        &mut self,
        fixture: &Fixture,
        parent: &neo_fold_clean::CeClaim,
        witness: &neo_ccs::Mat<F>,
    ) -> neo_prover_cuda::reduce::dec::DecFoldOutput {
        self.dec
            .prove(
                &self.device,
                &self.kernels,
                &mut self.ajtai,
                &mut self.bar_matrices,
                &fixture.prep.params,
                fixture.structure(),
                fixture.prep.optimized_cache(),
                ajtai_dec_mixer,
                parent,
                DecParentWitness::Host(witness),
                None,
                None,
                DecOutputMode::ResidentOnly,
                DecRecompositionMode::DeferYAndXAndCommitment,
            )
            .expect("resident GPU pi_dec prove")
    }
}
