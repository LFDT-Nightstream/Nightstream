use std::sync::Arc;
use std::time::Instant;

use neo_ajtai::{s_mul_add, scale_commitment_add_inplace, setup as ajtai_setup, AjtaiSModule, Commitment};
use neo_ccs::{poly::SparsePoly, poly::Term, CcsClaim, CcsStructure, CcsWitness, Mat};
use neo_fold_prototype::core::finalize::package_proof;
use neo_fold_prototype::core::proof::{partition_public_steps, FoldSchedule, StepInput};
use neo_fold_prototype::core::prover::CommitmentMixers;
use neo_fold_prototype::core::session::{prove_and_package, prove_run, verify_packaged, verify_run};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn fibonacci_trace_ccs(trace_len: usize) -> CcsStructure<F> {
    assert!(trace_len >= 3, "need at least three values for Fibonacci");
    assert!(trace_len <= D, "test trace must fit in one packed SuperNeo witness");

    let transitions = trace_len - 2;
    let mut m = Mat::zero(transitions, D, F::ZERO);
    for row in 0..transitions {
        m[(row, row)] = F::ONE;
        m[(row, row + 1)] = F::ONE;
        m[(row, row + 2)] = -F::ONE;
    }

    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    CcsStructure::new(vec![m], f).expect("valid Fibonacci CCS")
}

fn fibonacci_trace_from_seeds(a0: u64, a1: u64, trace_len: usize) -> Vec<u64> {
    assert!(trace_len >= 2, "need at least two seeds");

    let mut trace = Vec::with_capacity(trace_len);
    trace.push(a0);
    trace.push(a1);
    while trace.len() < trace_len {
        let next = trace[trace.len() - 2] + trace[trace.len() - 1];
        trace.push(next);
    }
    trace
}

fn fibonacci_chunked_traces(a0: u64, a1: u64, transitions_per_chunk: usize, total_transitions: usize) -> Vec<Vec<u64>> {
    assert!(transitions_per_chunk > 0, "chunk size must be positive");
    assert!(
        total_transitions.is_multiple_of(transitions_per_chunk),
        "total transitions must be divisible by chunk size"
    );

    let trace_len = transitions_per_chunk + 2;
    let mut traces = Vec::with_capacity(total_transitions / transitions_per_chunk);
    let mut seed0 = a0;
    let mut seed1 = a1;

    for _ in 0..(total_transitions / transitions_per_chunk) {
        let trace = fibonacci_trace_from_seeds(seed0, seed1, trace_len);
        seed0 = trace[transitions_per_chunk];
        seed1 = trace[transitions_per_chunk + 1];
        traces.push(trace);
    }

    traces
}

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn ajtai_mixers() -> CommitmentMixers<fn(&[Mat<F>], &[Commitment]) -> Commitment, fn(&[Commitment], u32) -> Commitment>
{
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Commitment]) -> Commitment {
        let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
        for (rho, c) in rhos.iter().zip(cs.iter()) {
            let rq = rot_matrix_to_rq(rho);
            s_mul_add(&mut acc, &rq, c);
        }
        acc
    }

    fn combine_b_pows(cs: &[Commitment], b: u32) -> Commitment {
        let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
        let base = F::from_u64(b as u64);
        let mut pow = F::ONE;
        for c in cs {
            scale_commitment_add_inplace(&mut acc, pow, c);
            pow *= base;
        }
        acc
    }

    CommitmentMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn make_ajtai_module(params: &NeoParams, witness_cols: usize) -> AjtaiSModule {
    let mut rng = ChaCha8Rng::seed_from_u64(1337);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, witness_cols).expect("Ajtai setup");
    AjtaiSModule::new(Arc::new(pp))
}

fn base2_params_with_k_rho(n_rows: usize, k_rho: u32) -> NeoParams {
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n_rows).expect("params");
    assert_eq!(params.b, 2, "test helper assumes base-2 decomposition");
    params.k_rho = k_rho;
    params.B = 1_u64.checked_shl(k_rho).expect("B fits in u64");
    params
}

fn fibonacci_step(log: &AjtaiSModule, label: &str, values: &[u64]) -> StepInput {
    assert!(values.len() <= D);

    let m_in = values.len();
    let mut z = vec![F::ZERO; D];
    for (idx, value) in values.iter().copied().enumerate() {
        z[idx] = F::from_u64(value);
    }

    let mut z_mat = Mat::zero(D, 1, F::ZERO);
    for (idx, value) in z.iter().copied().enumerate() {
        z_mat[(idx % D, idx / D)] = value;
    }

    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    StepInput {
        label: label.to_string(),
        mcs: CcsClaim {
            c: neo_ccs::traits::SModuleHomomorphism::commit(log, &z_mat),
            x,
            m_in,
        },
        witness: CcsWitness { w, Z: z_mat },
    }
}

#[test]
fn fibonacci_traces_fold_through_the_real_superneo_spine() {
    let traces = vec![vec![1_u64, 1, 2, 3, 5, 8, 13], vec![2_u64, 3, 5, 8, 13, 21, 34]];
    let params = NeoParams::goldilocks_auto_r1cs_ccs(traces[0].len() - 2).expect("params");
    let ccs = fibonacci_trace_ccs(traces[0].len());
    let log = make_ajtai_module(&params, 1);

    let steps = traces
        .iter()
        .enumerate()
        .map(|(idx, values)| fibonacci_step(&log, &format!("fib_trace_{idx}"), values))
        .collect::<Vec<_>>();

    let proof = prove_run(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps.clone(),
        &log,
        ajtai_mixers(),
    )
    .expect("prove run");

    assert_eq!(proof.chunks.len(), 2);
    assert_eq!(proof.chunks[0].ccs_outputs.len(), 1);
    assert_eq!(proof.chunks[0].dec.children.len(), params.k_rho as usize);
    assert_eq!(proof.chunks[1].ccs_outputs.len(), (params.k_rho as usize) + 1);
    assert_eq!(proof.final_main_claims.len(), params.k_rho as usize);

    let public_steps = steps
        .into_iter()
        .map(|step| step.public())
        .collect::<Vec<_>>();
    let verified = verify_run(
        FoldingMode::Optimized,
        &params,
        &ccs,
        &public_steps,
        &proof,
        ajtai_mixers(),
    )
    .expect("verify");
    assert_eq!(verified, proof.final_main_claims);
}

#[test]
fn fibonacci_traces_package_into_one_packaged_proof() {
    let traces = vec![vec![1_u64, 1, 2, 3, 5, 8, 13], vec![2_u64, 3, 5, 8, 13, 21, 34]];
    let params = NeoParams::goldilocks_auto_r1cs_ccs(traces[0].len() - 2).expect("params");
    let ccs = fibonacci_trace_ccs(traces[0].len());
    let log = make_ajtai_module(&params, 1);

    let steps = traces
        .iter()
        .enumerate()
        .map(|(idx, values)| fibonacci_step(&log, &format!("fib_trace_{idx}"), values))
        .collect::<Vec<_>>();

    let packaged = prove_and_package(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps,
        &log,
        ajtai_mixers(),
    )
    .expect("prove packaged");

    assert_eq!(
        packaged.proof.session.chunks[0].dec.children.len(),
        params.k_rho as usize
    );
    assert_eq!(packaged.statement.final_main_claims.len(), params.k_rho as usize);

    let verified =
        verify_packaged(FoldingMode::Optimized, &params, &ccs, &packaged, ajtai_mixers()).expect("verify packaged");
    assert_eq!(verified, packaged.statement.final_main_claims);
}

#[test]
fn fibonacci_traces_fold_through_ten_steps() {
    let trace_len = 7usize;
    let traces = (1_u64..=10)
        .map(|seed| fibonacci_trace_from_seeds(seed, seed + 1, trace_len))
        .collect::<Vec<_>>();
    let params = NeoParams::goldilocks_auto_r1cs_ccs(trace_len - 2).expect("params");
    let ccs = fibonacci_trace_ccs(trace_len);
    let log = make_ajtai_module(&params, 1);

    let steps = traces
        .iter()
        .enumerate()
        .map(|(idx, values)| fibonacci_step(&log, &format!("fib_trace_{idx}"), values))
        .collect::<Vec<_>>();

    let proof = prove_run(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps.clone(),
        &log,
        ajtai_mixers(),
    )
    .expect("prove run");

    assert_eq!(proof.chunks.len(), 10);
    assert_eq!(proof.chunks[0].ccs_outputs.len(), 1);
    for proved_step in proof.chunks.iter().skip(1) {
        assert_eq!(proved_step.ccs_outputs.len(), (params.k_rho as usize) + 1);
        assert_eq!(proved_step.dec.children.len(), params.k_rho as usize);
    }
    assert_eq!(proof.final_main_claims.len(), params.k_rho as usize);

    let public_steps = steps
        .into_iter()
        .map(|step| step.public())
        .collect::<Vec<_>>();
    let verified = verify_run(
        FoldingMode::Optimized,
        &params,
        &ccs,
        &public_steps,
        &proof,
        ajtai_mixers(),
    )
    .expect("verify");
    assert_eq!(verified, proof.final_main_claims);
}

#[test]
fn fibonacci_traces_fold_five_ten_transition_chunks_with_higher_k_rho() {
    let transitions_per_chunk = 10usize;
    let trace_len = transitions_per_chunk + 2;
    let traces = (1_u64..=5)
        .map(|seed| fibonacci_trace_from_seeds(seed, seed + 1, trace_len))
        .collect::<Vec<_>>();
    let params = base2_params_with_k_rho(transitions_per_chunk, 13);
    let ccs = fibonacci_trace_ccs(trace_len);
    let log = make_ajtai_module(&params, 1);

    let steps = traces
        .iter()
        .enumerate()
        .map(|(idx, values)| fibonacci_step(&log, &format!("fib_chunk_{idx}"), values))
        .collect::<Vec<_>>();

    let proof = prove_run(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps.clone(),
        &log,
        ajtai_mixers(),
    )
    .expect("five ten-transition chunks should fit once k_rho is raised");

    assert_eq!(proof.chunks.len(), 5);
    assert_eq!(proof.final_main_claims.len(), params.k_rho as usize);

    let public_steps = steps
        .into_iter()
        .map(|step| step.public())
        .collect::<Vec<_>>();
    let verified = verify_run(
        FoldingMode::Optimized,
        &params,
        &ccs,
        &public_steps,
        &proof,
        ajtai_mixers(),
    )
    .expect("verify");
    assert_eq!(verified, proof.final_main_claims);
}

#[test]
fn continuous_fifty_transition_fibonacci_exceeds_fixed_k_rho_budget() {
    let transitions_per_chunk = 10usize;
    let total_transitions = 50usize;
    let trace_len = transitions_per_chunk + 2;
    let traces = fibonacci_chunked_traces(1, 1, transitions_per_chunk, total_transitions);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(transitions_per_chunk).expect("params");
    let ccs = fibonacci_trace_ccs(trace_len);
    let log = make_ajtai_module(&params, 1);

    assert_eq!(traces.len(), 5);
    for pair in traces.windows(2) {
        assert_eq!(pair[0][transitions_per_chunk], pair[1][0]);
        assert_eq!(pair[0][transitions_per_chunk + 1], pair[1][1]);
    }

    let steps = traces
        .iter()
        .enumerate()
        .map(|(idx, values)| fibonacci_step(&log, &format!("fib_chunk_{idx}"), values))
        .collect::<Vec<_>>();

    let err = prove_run(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps,
        &log,
        ajtai_mixers(),
    )
    .expect_err("continuous 50-transition sequence should exceed fixed k_rho budget");
    assert!(format!("{err}").contains("DEC split"));
}

#[test]
fn fibonacci_fold_metrics_five_ten_transition_chunks_with_higher_k_rho() {
    let transitions_per_chunk = 10usize;
    let trace_len = transitions_per_chunk + 2;
    let traces = (1_u64..=5)
        .map(|seed| fibonacci_trace_from_seeds(seed, seed + 1, trace_len))
        .collect::<Vec<_>>();
    let params = base2_params_with_k_rho(transitions_per_chunk, 13);
    let ccs = fibonacci_trace_ccs(trace_len);
    let log = make_ajtai_module(&params, 1);

    let steps = traces
        .iter()
        .enumerate()
        .map(|(idx, values)| fibonacci_step(&log, &format!("fib_metrics_chunk_{idx}"), values))
        .collect::<Vec<_>>();
    let public_steps = steps.iter().map(|step| step.public()).collect::<Vec<_>>();

    let prove_start = Instant::now();
    let proof = prove_run(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps.clone(),
        &log,
        ajtai_mixers(),
    )
    .expect("prove run");
    let prove_ms = prove_start.elapsed().as_secs_f64() * 1000.0;

    let verify_start = Instant::now();
    let verified = verify_run(
        FoldingMode::Optimized,
        &params,
        &ccs,
        &public_steps,
        &proof,
        ajtai_mixers(),
    )
    .expect("verify");
    let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(verified, proof.final_main_claims);

    let package_start = Instant::now();
    let public_chunks =
        partition_public_steps(FoldSchedule::RowsPerChunk(1), public_steps.clone()).expect("public chunks");
    let packaged = package_proof(public_chunks, proof.clone()).expect("package run");
    let package_ms = package_start.elapsed().as_secs_f64() * 1000.0;

    let packaged_verify_start = Instant::now();
    let packaged_verified =
        verify_packaged(FoldingMode::Optimized, &params, &ccs, &packaged, ajtai_mixers()).expect("verify packaged");
    let packaged_verify_ms = packaged_verify_start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(packaged_verified, packaged.statement.final_main_claims);

    let run_bytes = bincode::serialize(&proof)
        .expect("serialize run proof")
        .len();
    let statement_bytes = bincode::serialize(&packaged.statement)
        .expect("serialize public statement")
        .len();
    let packaged_bytes = bincode::serialize(&packaged)
        .expect("serialize packaged proof")
        .len();

    println!(
        "fibonacci_metrics chunks={} transitions_per_chunk={} total_transitions={} k_rho={} run_prove_ms={:.3} run_verify_ms={:.3} package_ms={:.3} packaged_verify_ms={:.3} run_bytes={} statement_bytes={} packaged_bytes={} carried_claims={}",
        traces.len(),
        transitions_per_chunk,
        traces.len() * transitions_per_chunk,
        params.k_rho,
        prove_ms,
        verify_ms,
        package_ms,
        packaged_verify_ms,
        run_bytes,
        statement_bytes,
        packaged_bytes,
        proof.final_main_claims.len(),
    );
}
