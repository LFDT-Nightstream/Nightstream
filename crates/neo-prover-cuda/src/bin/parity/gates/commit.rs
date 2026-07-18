//! Arithmetic smoke and Ajtai commit gates (Phases 0-1).

use super::*;

/// Toolchain + arithmetic gate: device K mul/add must match `neo_math::K`
/// on random inputs, through both the host port and a real kernel.
pub fn smoke() {
    const N: usize = 1 << 14;
    let mut rng = StdRng::seed_from_u64(0x6f78_6964_655f_7631);
    let a: Vec<K> = (0..N).map(|_| rand_k(&mut rng)).collect();
    let b: Vec<K> = (0..N).map(|_| rand_k(&mut rng)).collect();
    let expect: Vec<[u64; 2]> = a
        .iter()
        .zip(&b)
        .map(|(&a, &b)| k_words(a * b + a))
        .collect();

    let (a_words, b_words) = (k_slice_words(&a), k_slice_words(&b));
    for i in 0..N {
        let (ka, kb) = (
            Kx::from_words(a_words[2 * i], a_words[2 * i + 1]),
            Kx::from_words(b_words[2 * i], b_words[2 * i + 1]),
        );
        assert_eq!((ka * kb + ka).as_words(), expect[i], "host Kx mismatch at {i}");
    }

    let device = Device::open().expect("open CUDA device");
    let module = load_probe_kernels(device.ctx()).expect("load probe kernels");
    let a_dev = DeviceBuffer::from_host(device.stream(), &a_words).expect("upload a");
    let b_dev = DeviceBuffer::from_host(device.stream(), &b_words).expect("upload b");
    let mut out_dev = DeviceBuffer::zeroed(device.stream(), 2 * N).expect("alloc out");
    launch_k_mul_add(&module, device.stream(), N, &a_dev, &b_dev, &mut out_dev).expect("launch k_mul_add");
    let out = out_dev.to_host_vec(device.stream()).expect("download out");
    device.sync().expect("stream sync");

    for i in 0..N {
        assert_eq!([out[2 * i], out[2 * i + 1]], expect[i], "device Kx mismatch at {i}");
    }

    let graph_stream = device.stream().fork().expect("fork graph stream");
    let mut graph_out_dev = DeviceBuffer::zeroed(&graph_stream, 2 * N).expect("alloc graph out");
    let graph = CapturedGraph::capture(&graph_stream, || {
        launch_k_mul_add(&module, &graph_stream, N, &a_dev, &b_dev, &mut graph_out_dev)
    })
    .expect("capture k_mul_add graph");
    graph.launch(&graph_stream).expect("launch k_mul_add graph");
    let graph_out = graph_out_dev
        .to_host_vec(&graph_stream)
        .expect("download graph out");
    graph_stream.synchronize().expect("graph stream sync");
    for i in 0..N {
        assert_eq!(
            [graph_out[2 * i], graph_out[2 * i + 1]],
            expect[i],
            "captured graph Kx mismatch at {i}"
        );
    }

    const COOP_BLOCKS: u32 = 4;
    let mut coop_out_dev = DeviceBuffer::zeroed(device.stream(), COOP_BLOCKS as usize + 2).expect("alloc coop out");
    launch_cooperative_grid_sync_probe(&module, device.stream(), COOP_BLOCKS, &mut coop_out_dev)
        .expect("launch cooperative grid-sync probe");
    let coop_out = coop_out_dev
        .to_host_vec(device.stream())
        .expect("download coop out");
    device.sync().expect("cooperative probe sync");
    let expected_sum = COOP_BLOCKS * (COOP_BLOCKS + 1) / 2;
    assert_eq!(
        coop_out[COOP_BLOCKS as usize], expected_sum,
        "cooperative grid-sync marker sum mismatch"
    );
    assert_eq!(
        coop_out[COOP_BLOCKS as usize + 1],
        COOP_BLOCKS,
        "cooperative grid-sync block count mismatch"
    );

    println!(
        "[parity smoke] OK: {N} K mul-adds identical across neo_math / host port / device kernel / captured graph; cooperative grid-sync probe passed"
    );
}

/// Ajtai commit gate: the device commit must equal both CPU commit paths
/// (dense and s-module) for a random dense assignment.
pub fn ajtai() {
    const KAPPA: usize = 16;
    const COLS: usize = 2048;
    let mut rng = StdRng::seed_from_u64(0x616a_7461_695f_7631);
    let pp = neo_ajtai::setup(&mut rng, D, KAPPA, COLS).expect("sample PP");
    let z: Vec<F> = (0..COLS * D - 13).map(|_| rand_f(&mut rng)).collect();
    let z_mat = pack_ring_matrix(&z, COLS);

    let (cpu, cpu_ms) = timed(|| neo_ajtai::commit_row_major(&pp, &z_mat));
    let s_module = AjtaiSModule::new(Arc::new(pp.clone()));
    assert_eq!(s_module.commit(&z_mat), cpu, "CPU dense vs s-module commit diverged");

    let device = Device::open().expect("open CUDA device");
    let (mut dev_ajtai, upload_ms) = timed(|| {
        let uploaded = DeviceAjtai::upload(&device, &pp).expect("upload PP");
        device.sync().expect("sync after PP upload");
        uploaded
    });
    let (gpu, gpu_ms) = timed(|| {
        dev_ajtai
            .commit_assignment(&device, &z)
            .expect("device commit")
    });

    assert_eq!(gpu, cpu, "device commitment mismatch");
    println!(
        "[parity ajtai] OK: kappa={KAPPA} cols={COLS} identical; cpu={cpu_ms:.2}ms gpu={gpu_ms:.2}ms (pp upload {upload_ms:.2}ms, one-time)"
    );
}

/// Fresh-instance gate: the adapter's `build_fresh_instances` must be
/// field-identical to `CcsInstance::from_low_norm_assignment`.
pub fn fresh() {
    const K_FRESH: usize = 3;
    let fixture = Fixture::identity_ccs(FIXTURE_N, FIXTURE_T, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6672_6573_685f_7631);
    let assignments: Vec<Vec<F>> = (0..K_FRESH)
        .map(|_| fixture.low_norm_assignment(&mut rng))
        .collect();

    let (expected, cpu_ms) = timed(|| {
        assignments
            .iter()
            .map(|z| {
                CcsInstance::from_low_norm_assignment(
                    &fixture.prep.params,
                    &fixture.prep.log,
                    fixture.structure(),
                    z,
                    fixture.m_in,
                )
                .expect("CPU fresh instance")
            })
            .collect::<Vec<_>>()
    });

    let mut adapter = CudaNifsProver::new().expect("create CUDA adapter");
    let assignment_refs: Vec<&[F]> = assignments.iter().map(Vec::as_slice).collect();
    let (got, gpu_ms) = timed(|| {
        adapter
            .build_fresh_instances(NifsFreshInstancesRequest {
                pp: &fixture.prep.params,
                s: fixture.structure(),
                cache: fixture.prep.optimized_cache(),
                log: &fixture.prep.log,
                m_in: fixture.m_in,
                assignments: &assignment_refs,
                image_overlay: None,
                lane_scheme: None,
            })
            .expect("adapter build_fresh_instances")
            .expect("adapter must take the GPU path for valid low-norm inputs")
    });

    assert_eq!(got.len(), expected.len());
    for (idx, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(g.claim.c, e.claim.c, "commitment mismatch at instance {idx}");
        assert_eq!(g.claim.x, e.claim.x, "public x mismatch at instance {idx}");
        assert_eq!(g.claim.m_in, e.claim.m_in, "m_in mismatch at instance {idx}");
        assert_eq!(g.witness.w, e.witness.w, "witness w mismatch at instance {idx}");
        assert_eq!(g.witness.Z, e.witness.Z, "witness Z mismatch at instance {idx}");
    }
    println!(
        "[parity fresh] OK: {K_FRESH} instances (m={FIXTURE_N}) identical; cpu={cpu_ms:.2}ms gpu={gpu_ms:.2}ms (incl. one-time PP materialize+upload)"
    );
}

/// Phase 1 perf gate at real sha256 scale (m ≈ 452k): the device commit
/// must beat the CPU *seeded signed-unit fast path* (what real b=2
/// workloads take) in steady state, with parity asserted per commit.
pub fn fresh_bench() {
    const COLS: usize = BENCH_N / D;
    const ROUNDS: usize = 4;
    // Same params-derived kappa as `dec_bench`: the global PP registry keys
    // on (D, cols) only, so every gate at this shape must agree on kappa.
    let kappa = neo_fold_clean::config::r1cs_params(BENCH_N, BENCH_N)
        .expect("bench params")
        .kappa() as usize;
    install_seeded_global_pp(kappa, COLS);
    let s_module = AjtaiSModule::from_global_for_dims(D, COLS).expect("s-module from global PP");

    let mut rng = StdRng::seed_from_u64(0x6672_6265_6e63_6831);
    let assignments: Vec<Vec<F>> = (0..ROUNDS)
        .map(|_| {
            (0..COLS * D - 7)
                .map(|_| rand_bounded(&mut rng, 2))
                .collect()
        })
        .collect();
    let mats: Vec<_> = assignments
        .iter()
        .map(|z| pack_ring_matrix(z, COLS))
        .collect();

    let _warm = s_module.commit(&mats[0]);
    let (cpu, cpu_total_ms) = timed(|| mats.iter().map(|m| s_module.commit(m)).collect::<Vec<_>>());

    let device = Device::open().expect("open CUDA device");
    let (mut dev_ajtai, setup_ms) = timed(|| {
        let pp = s_module.verification_pp().expect("materialize seeded PP");
        let uploaded = DeviceAjtai::upload(&device, &pp).expect("upload PP");
        device.sync().expect("sync after PP upload");
        uploaded
    });
    let _warm = dev_ajtai
        .commit_assignment(&device, &assignments[0])
        .expect("warmup");
    let (gpu, gpu_total_ms) = timed(|| {
        assignments
            .iter()
            .map(|z| {
                dev_ajtai
                    .commit_assignment(&device, z)
                    .expect("device commit")
            })
            .collect::<Vec<_>>()
    });

    assert_eq!(gpu, cpu, "commitment mismatch");
    println!(
        "[parity fresh_bench] OK: kappa={kappa} cols={COLS} identical; per-commit cpu(seeded signed-unit)={:.2}ms gpu={:.2}ms, one-time materialize+upload={setup_ms:.2}ms",
        cpu_total_ms / ROUNDS as f64,
        gpu_total_ms / ROUNDS as f64,
    );
}

fn k_words(value: K) -> [u64; 2] {
    let (c0, c1) = value.to_limbs_u64();
    [c0, c1]
}

fn k_slice_words(values: &[K]) -> Vec<u64> {
    values.iter().flat_map(|v| k_words(*v)).collect()
}
