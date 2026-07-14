//! Π_CCS sumcheck gates (Phase 3).

use super::*;
use neo_prover_cuda::field::k_from_device_words;
use neo_prover_cuda::kernels::sumcheck_common::{
    launch_table_fold, launch_table_fold_from_challenge, load_sumcheck_common,
};

/// Π_CCS FE-channel gate: the device row-phase oracle must produce the same
/// round polynomials as the CPU `OptimizedOracle` across every FE round,
/// with both sides folded at the same random challenges. Runs twice: fresh
/// MCS only, then with carried ME inputs so the Eval channel
/// (`eq_r_inputs` / `eval_tbl`) is exercised too.
pub fn ccs_fe() {
    const K_FRESH: usize = 2;
    const K_CARRIED: usize = 2;
    let fixture = Fixture::r1cs_identity(FIXTURE_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6363_735f_6665_5f31);

    let rounds = fe_rounds_parity(&fixture, K_FRESH, 0, &mut rng);
    let rounds_me = fe_rounds_parity(&fixture, K_FRESH, K_CARRIED, &mut rng);
    table_fold_challenge_source_parity(&mut rng);
    let trace_rounds = fe_device_transcript_trace_parity(&fixture, K_FRESH, &mut rng);
    let cooperative_trace_rounds = fe_cooperative_transcript_trace_parity(&fixture, K_FRESH, &mut rng);
    let cooperative_all_trace_rounds = fe_cooperative_all_transcript_trace_parity(&fixture, K_FRESH, &mut rng);
    let tail_rounds = fe_ajtai_tail_parity(&fixture, K_FRESH, K_CARRIED, &mut rng);
    println!(
        "[parity ccs_fe] OK: {rounds} FE rounds identical (m={FIXTURE_N}, K={K_FRESH}), {rounds_me} rounds with {K_CARRIED} carried ME inputs, scalar/device challenge folds match, {trace_rounds} device-transcript FE rounds replay, {cooperative_trace_rounds} cooperative FE row rounds replay, {cooperative_all_trace_rounds} all-round cooperative FE rounds replay, and {tail_rounds} Ajtai-tail rounds from device-row-challenge Y_eval"
    );
}

fn eq_points(lhs: &[K], rhs: &[K]) -> K {
    assert_eq!(lhs.len(), rhs.len(), "eq point length mismatch");
    lhs.iter()
        .zip(rhs)
        .fold(K::ONE, |acc, (&a, &b)| acc * ((K::ONE - a) * (K::ONE - b) + a * b))
}

fn table_fold_challenge_source_parity(rng: &mut StdRng) {
    const TABLES: usize = 7;
    const CUR_LEN: usize = 64;
    const STRIDE: usize = CUR_LEN;
    const CHALLENGE_OFFSET: usize = 4;

    let mut table_words = vec![0u64; TABLES * STRIDE * 2];
    for table in 0..TABLES {
        for row in 0..STRIDE {
            let (c0, c1) = rand_k(rng).to_limbs_u64();
            let base = (table * STRIDE + row) * 2;
            table_words[base] = c0;
            table_words[base + 1] = c1;
        }
    }

    let r = rand_k(rng);
    let (r_c0, r_c1) = r.to_limbs_u64();
    let mut challenge_words = vec![0u64; CHALLENGE_OFFSET + 2];
    challenge_words[CHALLENGE_OFFSET] = r_c0;
    challenge_words[CHALLENGE_OFFSET + 1] = r_c1;

    let device = Device::open().expect("open CUDA device");
    let common = load_sumcheck_common(device.ctx()).expect("load sumcheck common kernels");
    let src = DeviceBuffer::from_host(device.stream(), &table_words).expect("upload fold table");
    let challenge = DeviceBuffer::from_host(device.stream(), &challenge_words).expect("upload fold challenge");
    let mut scalar_out = DeviceBuffer::zeroed(device.stream(), table_words.len()).expect("alloc scalar fold out");
    let mut device_out = DeviceBuffer::zeroed(device.stream(), table_words.len()).expect("alloc device fold out");

    launch_table_fold(
        &common,
        device.stream(),
        &src,
        TABLES,
        STRIDE,
        CUR_LEN,
        r_c0,
        r_c1,
        &mut scalar_out,
    )
    .expect("scalar challenge table fold");
    launch_table_fold_from_challenge(
        &common,
        device.stream(),
        &src,
        TABLES,
        STRIDE,
        CUR_LEN,
        &challenge,
        CHALLENGE_OFFSET,
        &mut device_out,
    )
    .expect("device challenge table fold");

    let scalar_words = scalar_out
        .to_host_vec(device.stream())
        .expect("download scalar fold output");
    let device_words = device_out
        .to_host_vec(device.stream())
        .expect("download device fold output");
    device.sync().expect("sync fold challenge source parity");

    let expected = expected_folded_table_words(&table_words, TABLES, STRIDE, CUR_LEN, r);
    assert_eq!(
        scalar_words, expected,
        "scalar table fold mismatch against CPU expectation"
    );
    assert_eq!(
        device_words, expected,
        "device challenge table fold mismatch against CPU expectation"
    );
}

fn expected_folded_table_words(table_words: &[u64], tables: usize, stride: usize, cur_len: usize, r: K) -> Vec<u64> {
    let mut out = vec![0u64; table_words.len()];
    for table in 0..tables {
        for row in 0..(cur_len / 2) {
            let lo = read_k_word(table_words, table, stride, 2 * row);
            let hi = read_k_word(table_words, table, stride, 2 * row + 1);
            let folded = lo + (hi - lo) * r;
            let (c0, c1) = folded.to_limbs_u64();
            let base = (table * stride + row) * 2;
            out[base] = c0;
            out[base + 1] = c1;
        }
    }
    out
}

fn read_k_word(table_words: &[u64], table: usize, stride: usize, row: usize) -> K {
    let base = (table * stride + row) * 2;
    k_from_device_words(table_words[base], table_words[base + 1])
}

fn fe_device_transcript_trace_parity(fixture: &Fixture, k_fresh: usize, rng: &mut StdRng) -> usize {
    fe_transcript_trace_parity(fixture, k_fresh, rng, FeTraceMode::Device)
}

fn fe_cooperative_transcript_trace_parity(fixture: &Fixture, k_fresh: usize, rng: &mut StdRng) -> usize {
    fe_transcript_trace_parity(fixture, k_fresh, rng, FeTraceMode::CooperativeRound)
}

fn fe_cooperative_all_transcript_trace_parity(fixture: &Fixture, k_fresh: usize, rng: &mut StdRng) -> usize {
    fe_transcript_trace_parity(fixture, k_fresh, rng, FeTraceMode::CooperativeAll)
}

#[derive(Clone, Copy, Debug)]
enum FeTraceMode {
    Device,
    CooperativeRound,
    CooperativeAll,
}

fn fe_transcript_trace_parity(fixture: &Fixture, k_fresh: usize, rng: &mut StdRng, mode: FeTraceMode) -> usize {
    let witnesses: Vec<neo_fold_clean::CcsWitness> = (0..k_fresh).map(|_| fixture.fresh_witness(rng)).collect();

    let params = fixture.prep.params.inner();
    let structure = fixture.structure();
    let cache = fixture.prep.optimized_cache();
    let dims = neo_reductions::engines::utils::build_dims_and_policy(params, structure).expect("dims");
    let challenges = neo_reductions::optimized_engine::Challenges {
        alpha: (0..dims.ell_d).map(|_| rand_k(rng)).collect(),
        beta_a: (0..dims.ell_d).map(|_| rand_k(rng)).collect(),
        beta_r: (0..dims.ell_n).map(|_| rand_k(rng)).collect(),
        beta_m: (0..dims.ell_m).map(|_| rand_k(rng)).collect(),
        gamma: rand_k(rng),
    };

    let mut cpu = neo_reductions::optimized_engine::CcsOracle::new_with_sparse_and_superneo_cache(
        structure,
        params,
        &witnesses,
        &[],
        challenges,
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        None,
        cache.sparse_arc(),
        cache.superneo_arc(),
    );

    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let mut gpu = DeviceFeBackend::new(&device, &kernels);
    assert!(gpu.start(&cpu.row_phase_snapshot()), "device FE backend start");
    let mut tr = neo_transcript::Poseidon2Transcript::new_raw_fields(&[rand_f(rng)]);
    let trace = match mode {
        FeTraceMode::Device => gpu
            .row_round_trace_from_transcript(tr.state(), tr.absorbed(), dims.ell_n)
            .expect("device FE trace from transcript"),
        FeTraceMode::CooperativeRound => {
            gpu.row_round_trace_from_transcript_cooperative(tr.state(), tr.absorbed(), dims.ell_n)
        }
        FeTraceMode::CooperativeAll => {
            gpu.row_round_trace_from_transcript_cooperative_all(tr.state(), tr.absorbed(), dims.ell_n)
        }
    };

    assert_eq!(trace.coeffs.len(), dims.ell_n, "FE trace coeff length mismatch");
    assert_eq!(trace.challenges.len(), dims.ell_n, "FE trace challenge length mismatch");

    let xs: Vec<K> = (0..=cpu.degree_bound())
        .map(|point| neo_math::from_complex(F::from_u64(point as u64), F::ZERO))
        .collect();
    for round in 0..dims.ell_n {
        let ys = cpu.evals_at(&xs);
        let coeffs = neo_reductions::sumcheck::interpolate_from_evals(&xs, &ys);
        assert_eq!(
            trace.coeffs[round], coeffs,
            "FE device trace coeff mismatch at round {round} (mode={mode:?})"
        );

        let coeff_fields = neo_reductions::sumcheck::round_coeff_fields(&coeffs);
        tr.append_fields_raw(&coeff_fields);
        let c = tr.challenge_fields_raw(2);
        let r = neo_math::from_complex(c[0], c[1]);
        assert_eq!(
            trace.challenges[round], r,
            "FE device trace challenge mismatch at round {round} (mode={mode:?})"
        );
        cpu.fold(r);
    }

    dims.ell_n
}

fn fe_ajtai_tail_parity(fixture: &Fixture, k_fresh: usize, k_carried: usize, rng: &mut StdRng) -> usize {
    let witnesses: Vec<neo_fold_clean::CcsWitness> = (0..k_fresh).map(|_| fixture.fresh_witness(rng)).collect();
    let me_witnesses: Vec<neo_ccs::Mat<F>> = (0..k_carried)
        .map(|_| fixture.fresh_witness(rng).Z)
        .collect();

    let params = fixture.prep.params.inner();
    let structure = fixture.structure();
    let cache = fixture.prep.optimized_cache();
    let dims = neo_reductions::engines::utils::build_dims_and_policy(params, structure).expect("dims");
    let challenges = neo_reductions::optimized_engine::Challenges {
        alpha: (0..dims.ell_d).map(|_| rand_k(rng)).collect(),
        beta_a: (0..dims.ell_d).map(|_| rand_k(rng)).collect(),
        beta_r: (0..dims.ell_n).map(|_| rand_k(rng)).collect(),
        beta_m: (0..dims.ell_m).map(|_| rand_k(rng)).collect(),
        gamma: rand_k(rng),
    };
    let r_inputs: Option<Vec<K>> = (k_carried > 0).then(|| (0..dims.ell_n).map(|_| rand_k(rng)).collect());
    let mut cpu = neo_reductions::optimized_engine::CcsOracle::new_with_sparse_and_superneo_cache(
        structure,
        params,
        &witnesses,
        &me_witnesses,
        challenges.clone(),
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        r_inputs.as_deref(),
        cache.sparse_arc(),
        cache.superneo_arc(),
    );

    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let all_mats: Vec<&neo_ccs::Mat<F>> = witnesses
        .iter()
        .map(|w| &w.Z)
        .chain(me_witnesses.iter())
        .collect();
    let planes = upload_witness_planes(&device, &all_mats).expect("upload Ajtai-tail witness planes");
    let mut gpu = DeviceFeBackend::new(&device, &kernels);
    gpu.set_witness_planes(&planes, all_mats.len());
    assert!(
        gpu.start(&cpu.row_phase_snapshot()),
        "device FE backend rejected snapshot"
    );

    let mut row_chals = Vec::with_capacity(dims.ell_n);
    for _ in 0..dims.ell_n {
        let r = rand_k(rng);
        cpu.fold(r);
        gpu.fold(r);
        row_chals.push(r);
    }

    let (eval_cache, chi_r, n_eff, witness_refs) = cpu.ajtai_backend_context().expect("Ajtai backend context");
    let y_eval_from_host_chi = gpu
        .device_ajtai_y_eval_surface(eval_cache, &chi_r, n_eff, &witness_refs)
        .expect("device Ajtai Y_eval surface")
        .expect("device Ajtai Y_eval applicable");
    let row_chal_words = row_chals
        .iter()
        .flat_map(|value| {
            let (c0, c1) = value.to_limbs_u64();
            [c0, c1]
        })
        .collect::<Vec<_>>();
    let row_chals_dev = cuda_core::DeviceBuffer::from_host(device.stream(), &row_chal_words)
        .expect("upload row challenges for device-chi parity");
    let y_eval = gpu
        .device_ajtai_y_eval_surface_from_device_challenges(
            eval_cache,
            &row_chals_dev,
            row_chals.len(),
            n_eff,
            &witness_refs,
        )
        .expect("device Ajtai Y_eval from device row challenges")
        .expect("device Ajtai Y_eval from device row challenges applicable");
    let y_eval_from_host_chi_host = gpu
        .download_ajtai_y_eval(&y_eval_from_host_chi)
        .expect("download host-chi Y_eval for parity");
    let y_eval_host = gpu
        .download_ajtai_y_eval(&y_eval)
        .expect("download device Ajtai Y_eval for CPU parity");
    assert_eq!(
        y_eval_host, y_eval_from_host_chi_host,
        "Ajtai Y_eval from device row challenges differs from host chi path"
    );
    let packed_surfaces = DevicePiCcsKSurfaces::pack(&device, &kernels, Some(&y_eval), None, D.next_power_of_two())
        .expect("pack Pi_CCS y_ring surfaces")
        .download_surfaces(&device)
        .expect("download Pi_CCS y_ring surfaces");
    assert_eq!(
        packed_surfaces.len(),
        y_eval_host.len(),
        "Pi_CCS packed y_ring claim count mismatch"
    );
    for (claim, (packed, expected_by_matrix)) in packed_surfaces.iter().zip(y_eval_host.iter()).enumerate() {
        assert_eq!(
            packed.len(),
            expected_by_matrix.len(),
            "Pi_CCS packed y_ring surface count mismatch for claim {claim}"
        );
        for (surface, (packed_row, expected_digits)) in packed.iter().zip(expected_by_matrix.iter()).enumerate() {
            let mut expected = vec![K::ZERO; D.next_power_of_two()];
            expected[..D].copy_from_slice(expected_digits);
            assert_eq!(
                *packed_row, expected,
                "Pi_CCS packed y_ring surface mismatch at claim {claim}, surface {surface}"
            );
        }
    }
    cpu.inject_ajtai_y_eval(y_eval_host);

    let eq_beta_r = eq_points(&row_chals, &challenges.beta_r);
    let eq_r_inputs = r_inputs
        .as_ref()
        .map_or(K::ZERO, |r_inputs| eq_points(&row_chals, r_inputs));
    let xs: Vec<K> = (0..=cpu.degree_bound())
        .map(|point| neo_math::from_complex(F::from_u64(point as u64), F::ZERO))
        .collect();
    let mut prefix = Vec::with_capacity(dims.ell_d);
    for round in 0..dims.ell_d {
        let ys = cpu.evals_at(&xs);
        let expected = neo_reductions::sumcheck::interpolate_from_evals(&xs, &ys);
        let actual = gpu
            .ajtai_tail_round_coeffs(
                &y_eval,
                DeviceFeTailRound {
                    alpha: &challenges.alpha,
                    beta_a: &challenges.beta_a,
                    prefix: &prefix,
                    gamma: challenges.gamma,
                    eq_beta_r,
                    eq_r_inputs,
                    k_mcs: k_fresh,
                    has_inputs: k_carried > 0,
                },
            )
            .expect("device Ajtai-tail coeffs");
        assert_eq!(actual, expected, "Ajtai-tail coefficient mismatch at round {round}");
        let r = rand_k(rng);
        cpu.fold(r);
        prefix.push(r);
    }

    dims.ell_d
}

/// Drive the CPU and device FE oracles side by side over every row round;
/// returns the round count. `k_carried > 0` adds ME witnesses and a shared
/// `r_inputs` point, enabling the Eval channel.
fn fe_rounds_parity(fixture: &Fixture, k_fresh: usize, k_carried: usize, rng: &mut StdRng) -> usize {
    let witnesses: Vec<neo_fold_clean::CcsWitness> = (0..k_fresh).map(|_| fixture.fresh_witness(rng)).collect();
    let me_witnesses: Vec<neo_ccs::Mat<F>> = (0..k_carried)
        .map(|_| fixture.fresh_witness(rng).Z)
        .collect();

    let params = fixture.prep.params.inner();
    let structure = fixture.structure();
    let cache = fixture.prep.optimized_cache();
    let dims = neo_reductions::engines::utils::build_dims_and_policy(params, structure).expect("dims");
    let challenges = neo_reductions::optimized_engine::Challenges {
        alpha: (0..dims.ell_d).map(|_| rand_k(rng)).collect(),
        beta_a: (0..dims.ell_d).map(|_| rand_k(rng)).collect(),
        beta_r: (0..dims.ell_n).map(|_| rand_k(rng)).collect(),
        beta_m: (0..dims.ell_m).map(|_| rand_k(rng)).collect(),
        gamma: rand_k(rng),
    };
    let r_inputs: Option<Vec<K>> = (k_carried > 0).then(|| (0..dims.ell_n).map(|_| rand_k(rng)).collect());

    let mut cpu = neo_reductions::optimized_engine::CcsOracle::new_with_sparse_and_superneo_cache(
        structure,
        params,
        &witnesses,
        &me_witnesses,
        challenges,
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        r_inputs.as_deref(),
        cache.sparse_arc(),
        cache.superneo_arc(),
    );

    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let mut gpu =
        DeviceFeOracle::from_snapshot(&device, &kernels, &cpu.row_phase_snapshot()).expect("device FE oracle");

    let xs: Vec<K> = (0..=cpu.degree_bound())
        .map(|point| neo_math::from_complex(F::from_u64(point as u64), F::ZERO))
        .collect();
    for round in 0..dims.ell_n {
        let cpu_ys = cpu.evals_at(&xs);
        let coeffs = gpu
            .round_coeffs(&device, &kernels)
            .expect("device round coeffs");
        let gpu_ys: Vec<K> = xs
            .iter()
            .map(|&x| neo_reductions::sumcheck::poly_eval_k(&coeffs, x))
            .collect();
        assert_eq!(
            gpu_ys, cpu_ys,
            "FE round {round} polynomial mismatch (k_carried={k_carried})"
        );
        let r = rand_k(rng);
        cpu.fold(r);
        gpu.fold(&device, &kernels, r).expect("device fold");
    }
    dims.ell_n
}

/// Π_CCS NC-channel gate: the device column-phase oracle must produce the
/// same 5-coefficient round polynomials as the CPU `NcOracle` across every
/// column round, and the same finalized digit rows afterwards.
pub fn ccs_nc() {
    const K_FRESH: usize = 2;
    let fixture = Fixture::r1cs_identity(FIXTURE_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6363_735f_6e63_5f31);
    let trace_rounds = nc_device_transcript_trace_parity(&fixture, K_FRESH, &mut rng);
    let witnesses: Vec<neo_fold_clean::CcsWitness> = (0..K_FRESH)
        .map(|_| fixture.fresh_witness(&mut rng))
        .collect();

    let params = fixture.prep.params.inner();
    let structure = fixture.structure();
    let dims = neo_reductions::engines::utils::build_dims_and_policy(params, structure).expect("dims");
    let challenges = neo_reductions::optimized_engine::Challenges {
        alpha: (0..dims.ell_d).map(|_| rand_k(&mut rng)).collect(),
        beta_a: (0..dims.ell_d).map(|_| rand_k(&mut rng)).collect(),
        beta_r: (0..dims.ell_n).map(|_| rand_k(&mut rng)).collect(),
        beta_m: (0..dims.ell_m).map(|_| rand_k(&mut rng)).collect(),
        gamma: rand_k(&mut rng),
    };

    let mut cpu = neo_reductions::optimized_engine::oracle::NcOracle::new(
        structure,
        params,
        &witnesses,
        &[],
        challenges,
        dims.ell_d,
        dims.ell_m,
        dims.d_sc,
    );

    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let mut gpu =
        DeviceNcOracle::from_snapshot(&device, &kernels, &cpu.col_phase_snapshot(), None).expect("device NC oracle");

    for round in 0..dims.ell_m {
        let cpu_coeffs = cpu
            .optimized_col_phase_round_coeffs()
            .expect("CPU col-phase coefficients");
        let gpu_coeffs = gpu
            .round_coeffs(&device, &kernels)
            .expect("device NC round coeffs");
        assert_eq!(gpu_coeffs, cpu_coeffs, "NC column round {round} coefficients mismatch");
        let r = rand_k(&mut rng);
        cpu.fold(r);
        gpu.fold(&device, &kernels, r).expect("device NC fold");
    }

    let final_state = gpu
        .finalized_col_state_device(&device, &kernels)
        .expect("device finalized NC state buffer");
    let finalized = gpu
        .finalized_col_state(&device, &kernels)
        .expect("device finalized state");
    assert_eq!(
        finalized.digit_rows,
        cpu.finalized_y_zcol_digits(),
        "finalized digit rows mismatch"
    );
    let packed_y_zcol = DevicePiCcsKSurfaces::pack(&device, &kernels, None, Some(&final_state), D.next_power_of_two())
        .expect("pack Pi_CCS y_zcol surface")
        .download_surfaces(&device)
        .expect("download Pi_CCS y_zcol surface");
    let expected_digits = cpu.finalized_y_zcol_digits();
    for (claim, (packed, expected_digits)) in packed_y_zcol.iter().zip(expected_digits.iter()).enumerate() {
        assert_eq!(packed.len(), 1, "Pi_CCS packed y_zcol surface count mismatch");
        let mut expected = vec![K::ZERO; D.next_power_of_two()];
        expected[..D].copy_from_slice(expected_digits);
        assert_eq!(
            packed[0], expected,
            "Pi_CCS packed y_zcol surface mismatch at claim {claim}"
        );
    }
    assert_eq!(
        finalized.eq_beta_m0,
        cpu.col_phase_snapshot().eq_beta_m_tbl[0],
        "finalized eq_beta_m mismatch"
    );
    println!(
        "[parity ccs_nc] OK: {} NC column rounds + finalized digit rows identical (m={FIXTURE_N}, K={K_FRESH}), {trace_rounds} device-transcript NC rounds replay",
        dims.ell_m,
    );
}

fn nc_device_transcript_trace_parity(fixture: &Fixture, k_fresh: usize, rng: &mut StdRng) -> usize {
    let witnesses: Vec<neo_fold_clean::CcsWitness> = (0..k_fresh).map(|_| fixture.fresh_witness(rng)).collect();

    let params = fixture.prep.params.inner();
    let structure = fixture.structure();
    let dims = neo_reductions::engines::utils::build_dims_and_policy(params, structure).expect("dims");
    let challenges = neo_reductions::optimized_engine::Challenges {
        alpha: (0..dims.ell_d).map(|_| rand_k(rng)).collect(),
        beta_a: (0..dims.ell_d).map(|_| rand_k(rng)).collect(),
        beta_r: (0..dims.ell_n).map(|_| rand_k(rng)).collect(),
        beta_m: (0..dims.ell_m).map(|_| rand_k(rng)).collect(),
        gamma: rand_k(rng),
    };
    let initial_sum = rand_k(rng);
    let mut tr = neo_transcript::Poseidon2Transcript::new_raw_fields(&[rand_f(rng)]);

    let mut cpu = neo_reductions::optimized_engine::oracle::NcOracle::new(
        structure,
        params,
        &witnesses,
        &[],
        challenges,
        dims.ell_d,
        dims.ell_m,
        dims.d_sc,
    );

    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let mut gpu =
        DeviceNcOracle::from_snapshot(&device, &kernels, &cpu.col_phase_snapshot(), None).expect("device NC oracle");
    let trace = gpu
        .col_round_trace_with_prolog(
            &device,
            &kernels,
            neo_reductions::optimized_engine::NcColTraceRequest {
                transcript_state: tr.state(),
                transcript_absorbed: tr.absorbed(),
                rounds: dims.ell_m,
                initial_sum,
            },
        )
        .expect("device NC transcript trace");

    tr.append_fields_raw(&[F::from_u64(
        neo_reductions::engines::utils::PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG,
    )]);
    tr.append_fields_raw(&[F::from_u64(
        neo_reductions::engines::utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
    )]);
    tr.append_fields_raw(&initial_sum.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(
        neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
    )]);

    assert_eq!(trace.coeffs.len(), dims.ell_m, "NC trace coeff length mismatch");
    assert_eq!(trace.challenges.len(), dims.ell_m, "NC trace challenge length mismatch");
    for round in 0..dims.ell_m {
        let coeffs = cpu
            .optimized_col_phase_round_coeffs()
            .expect("CPU NC transcript coeffs");
        assert_eq!(trace.coeffs[round], coeffs, "NC trace coeff mismatch at round {round}");
        let coeff_fields = neo_reductions::sumcheck::round_coeff_fields(&coeffs);
        tr.append_fields_raw(&coeff_fields);
        let c = tr.challenge_fields_raw(2);
        let r = neo_math::from_complex(c[0], c[1]);
        assert_eq!(
            trace.challenges[round], r,
            "NC trace challenge mismatch at round {round}"
        );
        cpu.fold(r);
    }

    assert_eq!(
        trace.finalized.digit_rows,
        cpu.finalized_y_zcol_digits(),
        "NC trace finalized digit rows mismatch"
    );
    assert_eq!(
        trace.finalized.eq_beta_m0,
        cpu.col_phase_snapshot().eq_beta_m_tbl[0],
        "NC trace finalized eq_beta_m mismatch"
    );
    let (state, absorbed) = trace
        .transcript_after
        .expect("device NC trace should return transcript snapshot");
    assert_eq!(state, tr.state(), "NC trace transcript state mismatch");
    assert_eq!(absorbed, tr.absorbed(), "NC trace transcript position mismatch");
    dims.ell_m
}

/// One Π_CCS prove workload: satisfying instances plus the transcript-bound
/// randomness, shared by `ccs_prove` and `ccs_bench`.
pub(super) struct CcsProveCase {
    mcs_list: Vec<neo_fold_clean::paper::relations::CcsClaim>,
    pub(super) mcs_witnesses: Vec<neo_fold_clean::CcsWitness>,
    me_inputs: Vec<neo_fold_clean::CeClaim>,
    me_witnesses: Vec<neo_ccs::Mat<F>>,
    digest: [F; 4],
    handle: [F; 4],
    transcript_init: [F; 1],
}

type CcsProveOutput = (
    Vec<neo_fold_clean::CeClaim>,
    neo_reductions::optimized_engine::PiCcsProof,
    neo_transcript::Poseidon2Transcript,
);

type CcsTerminalOutput = (
    neo_reductions::optimized_engine::PiCcsReplayTerminalState,
    neo_transcript::Poseidon2Transcript,
);

impl CcsProveCase {
    pub(super) fn build(fixture: &Fixture, k_fresh: usize, rng: &mut StdRng) -> Self {
        let instances: Vec<neo_fold_clean::CcsInstance> = (0..k_fresh)
            .map(|_| fixture.satisfying_binary_instance(rng))
            .collect();
        Self {
            mcs_list: instances.iter().map(|inst| inst.claim.clone()).collect(),
            mcs_witnesses: instances.into_iter().map(|inst| inst.witness).collect(),
            me_inputs: vec![],
            me_witnesses: vec![],
            digest: std::array::from_fn(|_| rand_f(rng)),
            handle: std::array::from_fn(|_| rand_f(rng)),
            transcript_init: [rand_f(rng)],
        }
    }

    pub(super) fn prove_cpu(&self, fixture: &Fixture) -> CcsProveOutput {
        let mut tr = neo_transcript::Poseidon2Transcript::new_raw_fields(&self.transcript_init);
        let (outputs, proof, _) =
            neo_reductions::optimized_engine::optimized_prove_with_cache_and_instance_digest_and_me_input_handle_and_perf(
                &mut tr,
                fixture.prep.params.inner(),
                fixture.structure(),
                &self.mcs_list,
                &self.mcs_witnesses,
                &self.me_inputs,
                &self.me_witnesses,
                self.digest,
                self.handle,
                &fixture.prep.log,
                fixture.prep.optimized_cache(),
            )
            .expect("CPU pi_ccs prove");
        (outputs, proof, tr)
    }

    fn terminal_cpu(&self, fixture: &Fixture) -> CcsTerminalOutput {
        let mut tr = neo_transcript::Poseidon2Transcript::new_raw_fields(&self.transcript_init);
        let terminal =
            neo_reductions::optimized_engine::optimized_replay_terminal_state_with_cache_instance_digest_and_me_input_handle_and_perf(
                &mut tr,
                fixture.prep.params.inner(),
                fixture.structure(),
                &self.mcs_list,
                &self.mcs_witnesses,
                &self.me_inputs,
                &self.me_witnesses,
                self.digest,
                self.handle,
                &fixture.prep.log,
                fixture.prep.optimized_cache(),
            )
            .expect("CPU pi_ccs terminal replay");
        (terminal, tr)
    }

    /// This case's witness planes, for the per-fold shared upload
    /// (`pi_rlc::upload_witness_planes` order contract).
    fn witness_planes(&self, device: &Device) -> DeviceBuffer<u64> {
        let mats: Vec<&neo_ccs::Mat<F>> = self
            .mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(self.me_witnesses.iter())
            .collect();
        neo_prover_cuda::ingest::upload_witness_planes(device, &mats).expect("upload case witness planes")
    }

    /// Backends are passed in (not built per prove) because a session holds
    /// them across folds — static uploads like the bar matrices must amortize.
    /// `planes` is this case's shared witness-planes buffer; the NC digit
    /// tables and Ajtai `Y_eval` source it like the real session does.
    fn prove_gpu<'a>(
        &self,
        fixture: &Fixture,
        fe_backend: &mut DeviceFeBackend<'a>,
        nc_backend: &mut DeviceNcBackend<'a>,
        planes: &'a DeviceBuffer<u64>,
    ) -> CcsProveOutput {
        let witness_count = self.mcs_witnesses.len() + self.me_witnesses.len();
        fe_backend.set_witness_planes(planes, witness_count);
        nc_backend.set_witness_planes(planes, witness_count);
        let mut tr = neo_transcript::Poseidon2Transcript::new_raw_fields(&self.transcript_init);
        let (outputs, proof, _) = neo_reductions::optimized_engine::optimized_prove_with_device_backends(
            &mut tr,
            fixture.prep.params.inner(),
            fixture.structure(),
            &self.mcs_list,
            &self.mcs_witnesses,
            &self.me_inputs,
            &self.me_witnesses,
            self.digest,
            self.handle,
            &fixture.prep.log,
            fixture.prep.optimized_cache(),
            Some(fe_backend),
            Some(nc_backend),
        )
        .expect("GPU-backed pi_ccs prove");
        (outputs, proof, tr)
    }

    fn prove_gpu_phase<'a>(
        &self,
        fixture: &Fixture,
        phase_backend: &mut DevicePiCcsPhaseBackend<'a>,
        planes: &'a DeviceBuffer<u64>,
    ) -> CcsProveOutput {
        phase_backend.set_witness_planes(planes, self.mcs_witnesses.len() + self.me_witnesses.len());
        phase_backend.enable_whole_fe_trace_for_parity();
        let mut tr = neo_transcript::Poseidon2Transcript::new_raw_fields(&self.transcript_init);
        let (outputs, proof, _) =
            neo_reductions::optimized_engine::optimized_prove_with_phase_backend_and_transcript_mode(
                &mut tr,
                fixture.prep.params.inner(),
                fixture.structure(),
                &self.mcs_list,
                &self.mcs_witnesses,
                &self.me_inputs,
                &self.me_witnesses,
                self.digest,
                self.handle,
                &fixture.prep.log,
                fixture.prep.optimized_cache(),
                Some(phase_backend),
                None,
                None,
                BackendTranscriptMode::Replay,
            )
            .expect("GPU whole-phase pi_ccs prove");
        (outputs, proof, tr)
    }

    fn terminal_gpu_phase_summary<'a>(
        &self,
        fixture: &Fixture,
        phase_backend: &mut DevicePiCcsPhaseBackend<'a>,
        planes: &'a DeviceBuffer<u64>,
    ) -> CcsTerminalOutput {
        phase_backend.set_witness_planes(planes, self.mcs_witnesses.len() + self.me_witnesses.len());
        phase_backend.enable_whole_fe_trace_for_parity();
        let mut tr = neo_transcript::Poseidon2Transcript::new_raw_fields(&self.transcript_init);
        let terminal =
            neo_reductions::optimized_engine::optimized_replay_terminal_state_with_phase_backend_and_transcript_mode(
                &mut tr,
                fixture.prep.params.inner(),
                fixture.structure(),
                &self.mcs_list,
                &self.mcs_witnesses,
                &self.me_inputs,
                &self.me_witnesses,
                self.digest,
                self.handle,
                &fixture.prep.log,
                fixture.prep.optimized_cache(),
                Some(phase_backend),
                BackendTranscriptMode::DeviceSnapshot,
            )
            .expect("GPU whole-phase pi_ccs terminal summary");
        (terminal, tr)
    }
}

/// Full Π_CCS prove gate: `optimized_prove_with_fe_backend` (device FE
/// rounds, everything else canonical CPU) must produce a bit-identical
/// `PiCcsProof`, ME outputs, and post-prove transcript state.
pub fn ccs_prove() {
    const K_FRESH: usize = 2;
    let fixture = Fixture::r1cs_identity(FIXTURE_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6363_735f_7076_5f31);
    let case = CcsProveCase::build(&fixture, K_FRESH, &mut rng);

    let ((cpu_outputs, cpu_proof, cpu_tr), cpu_ms) = timed(|| case.prove_cpu(&fixture));
    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let planes = case.witness_planes(&device);
    let mut fe_backend = DeviceFeBackend::new(&device, &kernels);
    fe_backend.enable_whole_fe_trace_for_parity();
    let mut nc_backend = DeviceNcBackend::new(&device, &kernels);
    let ((gpu_outputs, gpu_proof, gpu_tr), gpu_ms) =
        timed(|| case.prove_gpu(&fixture, &mut fe_backend, &mut nc_backend, &planes));

    assert_eq!(gpu_outputs, cpu_outputs, "ME outputs mismatch");
    assert_eq!(
        gpu_proof.sumcheck_rounds, cpu_proof.sumcheck_rounds,
        "FE rounds mismatch"
    );
    assert_eq!(
        gpu_proof.sumcheck_challenges, cpu_proof.sumcheck_challenges,
        "FE challenges mismatch"
    );
    assert_eq!(
        gpu_proof.sumcheck_rounds_nc, cpu_proof.sumcheck_rounds_nc,
        "NC rounds mismatch"
    );
    assert_eq!(
        gpu_proof.sumcheck_challenges_nc, cpu_proof.sumcheck_challenges_nc,
        "NC challenges mismatch"
    );
    assert_eq!(gpu_proof.sumcheck_final, cpu_proof.sumcheck_final, "FE final mismatch");
    assert_eq!(
        gpu_proof.sumcheck_final_nc, cpu_proof.sumcheck_final_nc,
        "NC final mismatch"
    );
    assert_eq!(gpu_proof.header_digest, cpu_proof.header_digest, "header mismatch");
    assert_eq!(gpu_tr.state(), cpu_tr.state(), "post-prove transcript state mismatch");
    assert_eq!(
        gpu_tr.absorbed(),
        cpu_tr.absorbed(),
        "post-prove transcript position mismatch"
    );
    println!(
        "[parity ccs_prove] OK: full Π_CCS proof identical (m={FIXTURE_N}, K={K_FRESH}, device FE+NC rounds); cpu={cpu_ms:.2}ms gpu={gpu_ms:.2}ms"
    );
}

/// Device Pi_CCS output-digest gate: the prover-side digest used before
/// Π_RLC samples rho can be built from resident K-surfaces without first
/// materializing those surfaces on the host.
pub fn ccs_output_digest() {
    const K_FRESH: usize = 2;
    let fixture = Fixture::r1cs_identity(FIXTURE_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6363_735f_6f75_7464);
    let case = CcsProveCase::build(&fixture, K_FRESH, &mut rng);

    let (cpu_outputs, _, _) = case.prove_cpu(&fixture);
    let cpu_digest = neo_fold_clean::paper::digest::pi_ccs_outputs_digest(&cpu_outputs);

    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let planes = case.witness_planes(&device);
    let mut fe_backend = DeviceFeBackend::new(&device, &kernels);
    let mut nc_backend = DeviceNcBackend::new(&device, &kernels);
    let (gpu_outputs, _, _) = case.prove_gpu(&fixture, &mut fe_backend, &mut nc_backend, &planes);
    assert_eq!(gpu_outputs, cpu_outputs, "ME outputs mismatch before digest check");

    let y_eval = fe_backend
        .take_last_y_eval_surface()
        .expect("FE backend retained resident Y_eval surface");
    let nc_final = nc_backend
        .take_last_final_state()
        .expect("NC backend retained resident finalized column state");
    let surfaces = DevicePiCcsKSurfaces::pack(&device, &kernels, Some(&y_eval), Some(&nc_final), D.next_power_of_two())
        .expect("pack resident Pi_CCS K surfaces");
    let shells = gpu_outputs
        .iter()
        .map(PiCcsOutputDigestShell::from_claim)
        .collect::<Vec<_>>();
    let gpu_digest = DevicePiCcsOutputsDigest::compute_from_shells(&device, &kernels, &shells, &surfaces)
        .expect("device Pi_CCS output digest")
        .download(&device)
        .expect("download device Pi_CCS output digest");
    assert_eq!(gpu_digest, cpu_digest, "Pi_CCS output digest mismatch");
    println!(
        "[parity ccs_output_digest] OK: device Pi_CCS output digest matches CPU from resident K-surfaces (m={FIXTURE_N}, K={K_FRESH})"
    );
}

/// Whole-FE graph replay gate: the first prove captures the FE graph and
/// the second same-shape prove replays it through the same backend/session
/// state. This catches stale captured pointers before the e2e session does.
pub fn ccs_graph_replay() {
    const K_FRESH: usize = 2;
    let fixture = Fixture::r1cs_identity(FIXTURE_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6363_735f_6772_6170);
    let case = CcsProveCase::build(&fixture, K_FRESH, &mut rng);

    let cpu = case.prove_cpu(&fixture);
    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let planes = case.witness_planes(&device);
    let mut fe_backend = DeviceFeBackend::new(&device, &kernels);
    fe_backend.enable_whole_fe_trace_for_parity();
    let mut nc_backend = DeviceNcBackend::new(&device, &kernels);

    let _capture = case.prove_gpu(&fixture, &mut fe_backend, &mut nc_backend, &planes);
    let replay = case.prove_gpu(&fixture, &mut fe_backend, &mut nc_backend, &planes);
    assert_graph_replay_proof(replay, cpu);
    println!("[parity ccs_graph_replay] OK: whole-FE graph replay is byte-identical");
}

/// Whole-phase terminal-summary gate: the backend may advance terminal Π_CCS
/// state without returning proof logs, but the canonical terminal state must
/// still be byte-identical to the CPU replay path.
pub fn ccs_phase_summary() {
    const K_FRESH: usize = 2;
    let fixture = Fixture::r1cs_identity(FIXTURE_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6363_735f_7375_6d6d);
    let case = CcsProveCase::build(&fixture, K_FRESH, &mut rng);

    let cpu = case.terminal_cpu(&fixture);
    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let planes = case.witness_planes(&device);
    let mut phase_backend = DevicePiCcsPhaseBackend::new(&device, &kernels);
    let gpu = case.terminal_gpu_phase_summary(&fixture, &mut phase_backend, &planes);

    assert_terminal_state_eq(gpu, cpu);
    println!("[parity ccs_phase_summary] OK: whole-Π_CCS terminal summary is byte-identical");
}

/// Real-scale graph replay gate: same mechanism as `ccs_graph_replay`, but
/// at the 15-witness shape that the SHA e2e chain reaches after warmup.
pub fn ccs_graph_replay_bench() {
    const K_FRESH: usize = 15;
    let fixture = Fixture::r1cs_identity(BENCH_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6363_735f_6772_6c67);
    let warm_case = CcsProveCase::build(&fixture, 1, &mut rng);
    let case = CcsProveCase::build(&fixture, K_FRESH, &mut rng);

    let cpu = case.prove_cpu(&fixture);
    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let warm_planes = warm_case.witness_planes(&device);
    let planes = case.witness_planes(&device);
    let mut fe_backend = DeviceFeBackend::new(&device, &kernels);
    fe_backend.enable_whole_fe_trace_for_parity();
    let mut nc_backend = DeviceNcBackend::new(&device, &kernels);

    let _small_shape_capture = warm_case.prove_gpu(&fixture, &mut fe_backend, &mut nc_backend, &warm_planes);
    let _capture = case.prove_gpu(&fixture, &mut fe_backend, &mut nc_backend, &planes);
    let replay = case.prove_gpu(&fixture, &mut fe_backend, &mut nc_backend, &planes);
    assert_graph_replay_proof(replay, cpu);
    println!("[parity ccs_graph_replay_bench] OK: real-scale whole-FE graph replay is byte-identical");
}

/// Real-scale whole-Π_CCS phase-backend gate: this exercises the structural
/// FE→NC device transcript seam directly, without the lifecycle wrapper.
pub fn ccs_phase_bench() {
    const K_FRESH: usize = 15;
    let fixture = Fixture::r1cs_identity(BENCH_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6363_735f_7068_6173);
    let case = CcsProveCase::build(&fixture, K_FRESH, &mut rng);

    let cpu = case.prove_cpu(&fixture);
    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let planes = case.witness_planes(&device);
    let mut phase_backend = DevicePiCcsPhaseBackend::new(&device, &kernels);
    let gpu = case.prove_gpu_phase(&fixture, &mut phase_backend, &planes);

    assert_graph_replay_proof(gpu, cpu);
    println!("[parity ccs_phase_bench] OK: real-scale whole-Π_CCS phase backend is byte-identical (K={K_FRESH})");
}

fn assert_graph_replay_proof(gpu: CcsProveOutput, cpu: CcsProveOutput) {
    let (gpu_outputs, gpu_proof, gpu_tr) = gpu;
    let (cpu_outputs, cpu_proof, cpu_tr) = cpu;
    assert_eq!(gpu_outputs, cpu_outputs, "ME outputs mismatch after graph replay");
    assert_eq!(
        gpu_proof.sumcheck_rounds, cpu_proof.sumcheck_rounds,
        "FE rounds mismatch after graph replay"
    );
    assert_eq!(
        gpu_proof.sumcheck_challenges, cpu_proof.sumcheck_challenges,
        "FE challenges mismatch after graph replay"
    );
    assert_eq!(
        gpu_proof.sumcheck_final, cpu_proof.sumcheck_final,
        "FE final mismatch after graph replay"
    );
    assert_eq!(
        gpu_proof.sumcheck_rounds_nc, cpu_proof.sumcheck_rounds_nc,
        "NC rounds mismatch after graph replay"
    );
    assert_eq!(
        gpu_proof.sumcheck_challenges_nc, cpu_proof.sumcheck_challenges_nc,
        "NC challenges mismatch after graph replay"
    );
    assert_eq!(
        gpu_proof.sumcheck_final_nc, cpu_proof.sumcheck_final_nc,
        "NC final mismatch after graph replay"
    );
    assert_eq!(
        gpu_proof.header_digest, cpu_proof.header_digest,
        "header mismatch after graph replay"
    );
    assert_eq!(
        gpu_tr.state(),
        cpu_tr.state(),
        "post-prove transcript state mismatch after graph replay"
    );
    assert_eq!(
        gpu_tr.absorbed(),
        cpu_tr.absorbed(),
        "post-prove transcript position mismatch after graph replay"
    );
}

fn assert_terminal_state_eq(gpu: CcsTerminalOutput, cpu: CcsTerminalOutput) {
    let (gpu_terminal, gpu_tr) = gpu;
    let (cpu_terminal, cpu_tr) = cpu;
    assert_eq!(gpu_terminal.me_outputs, cpu_terminal.me_outputs, "ME outputs mismatch");
    assert_eq!(
        gpu_terminal.challenges_public, cpu_terminal.challenges_public,
        "public challenges mismatch"
    );
    assert_eq!(
        gpu_terminal.row_chals, cpu_terminal.row_chals,
        "row challenges mismatch"
    );
    assert_eq!(
        gpu_terminal.alpha_prime, cpu_terminal.alpha_prime,
        "alpha_prime mismatch"
    );
    assert_eq!(gpu_terminal.s_col, cpu_terminal.s_col, "s_col mismatch");
    assert_eq!(
        gpu_terminal.alpha_prime_nc, cpu_terminal.alpha_prime_nc,
        "alpha_prime_nc mismatch"
    );
    assert_eq!(
        gpu_terminal.sumcheck_final, cpu_terminal.sumcheck_final,
        "FE final sum mismatch"
    );
    assert_eq!(
        gpu_terminal.sumcheck_final_nc, cpu_terminal.sumcheck_final_nc,
        "NC final sum mismatch"
    );
    assert_eq!(
        gpu_terminal.fold_digest, cpu_terminal.fold_digest,
        "fold digest mismatch"
    );
    assert_eq!(
        gpu_tr.state(),
        cpu_tr.state(),
        "post-terminal transcript state mismatch"
    );
    assert_eq!(
        gpu_tr.absorbed(),
        cpu_tr.absorbed(),
        "post-terminal transcript position mismatch"
    );
}

/// Phase 3 perf gate at real sha256 scale: steady-state full Π_CCS prove
/// with both device backends vs the CPU prover, warmed, parity per round.
pub fn ccs_bench() {
    const K_FRESH: usize = 2;
    const ROUNDS: usize = 2;
    let fixture = Fixture::r1cs_identity(BENCH_N, FIXTURE_M_IN);
    let mut rng = StdRng::seed_from_u64(0x6363_735f_6265_6e63);
    let cases: Vec<CcsProveCase> = (0..ROUNDS)
        .map(|_| CcsProveCase::build(&fixture, K_FRESH, &mut rng))
        .collect();

    let _warm = cases[0].prove_cpu(&fixture);
    let (cpu, cpu_total_ms) = timed(|| {
        cases
            .iter()
            .map(|case| case.prove_cpu(&fixture))
            .collect::<Vec<_>>()
    });

    let device = Device::open().expect("open CUDA device");
    let kernels = SumcheckKernels::load(&device).expect("load sumcheck kernels");
    let case_planes: Vec<DeviceBuffer<u64>> = cases
        .iter()
        .map(|case| case.witness_planes(&device))
        .collect();
    let mut fe_backend = DeviceFeBackend::new(&device, &kernels);
    let mut nc_backend = DeviceNcBackend::new(&device, &kernels);
    let _warm = cases[0].prove_gpu(&fixture, &mut fe_backend, &mut nc_backend, &case_planes[0]);
    let (gpu, gpu_total_ms) = timed(|| {
        cases
            .iter()
            .zip(&case_planes)
            .map(|(case, planes)| case.prove_gpu(&fixture, &mut fe_backend, &mut nc_backend, planes))
            .collect::<Vec<_>>()
    });

    for (round, ((gpu_outputs, gpu_proof, _), (cpu_outputs, cpu_proof, _))) in gpu.iter().zip(&cpu).enumerate() {
        assert_eq!(gpu_outputs, cpu_outputs, "ME outputs mismatch at round {round}");
        assert_eq!(
            gpu_proof.header_digest, cpu_proof.header_digest,
            "header mismatch at round {round}"
        );
        assert_eq!(
            gpu_proof.sumcheck_final_nc, cpu_proof.sumcheck_final_nc,
            "NC final mismatch at round {round}"
        );
    }
    println!(
        "[parity ccs_bench] OK: m={BENCH_N} K={K_FRESH} identical; per-prove cpu={:.2}ms gpu={:.2}ms",
        cpu_total_ms / ROUNDS as f64,
        gpu_total_ms / ROUNDS as f64,
    );
}
