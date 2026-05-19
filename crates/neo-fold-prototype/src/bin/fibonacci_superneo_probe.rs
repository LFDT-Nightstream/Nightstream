use std::env;
use std::error::Error;
use std::time::Instant;

use neo_ajtai::{s_mul_add, scale_commitment_add_inplace, set_global_pp_seeded, AjtaiSModule, Commitment};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsMatrix, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_fold_prototype::core::ivc::{
    build_superneo_ivc_relations_with_initial_carry_accumulator_handle_perf, build_superneo_ivc_relations_with_perf,
    SuperNeoIvcBuild,
};
use neo_fold_prototype::core::proof::{
    Carry, ChunkProvePerf, ChunkVerifyPerf, FoldSchedule, PackagedProof, RunProvePerf, RunVerifyPerf, StepInput,
};
use neo_fold_prototype::core::prover::CommitmentMixers;
use neo_fold_prototype::core::session::{prove_and_package_with_final_carry_perf, verify_packaged_with_perf};
use neo_fold_prototype::direct_ccs::{start_direct_ccs_proof_state, DirectCcsProgram, DirectCcsRecursiveIvcSnarkPerf};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F, K};
use neo_params::{goldilocks_paper_b2, NeoParams};
use neo_reductions::api::FoldingMode;
use neo_reductions::common::{ct_from_y_ring_for_ccs_m, decode_superneo_coeffs_from_witness_mat, RotRing};
use neo_reductions::engines::utils::build_dims_and_policy;
use neo_reductions::superneo_eval::{build_superneo_eval_cache, eval_all_mats_ring_cached};
use p3_field::PrimeCharacteristicRing;

#[path = "support/fibonacci_superneo_probe_support.rs"]
mod fibonacci_superneo_probe_support;

use fibonacci_superneo_probe_support::{print_final_summary, print_spartan, print_superneo_ivc_carrier};

type AppResult<T> = Result<T, Box<dyn Error>>;

const FIB_STEP_TRACE_LEN: usize = 3;

#[derive(Clone, Copy, Debug)]
struct Config {
    iterations: usize,
    rows_per_chunk: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            iterations: 8,
            rows_per_chunk: 1,
        }
    }
}

impl Config {
    fn from_args() -> AppResult<Option<Self>> {
        let mut config = Self::default();
        for arg in env::args().skip(1) {
            if arg == "--help" || arg == "-h" {
                return Ok(None);
            }
            if let Some(raw) = arg.strip_prefix("--iterations=") {
                config.iterations = parse_nonzero_usize("--iterations", raw)?;
                continue;
            }
            if let Some(raw) = arg.strip_prefix("--rows-per-chunk=") {
                config.rows_per_chunk = parse_nonzero_usize("--rows-per-chunk", raw)?;
                continue;
            }
            return Err(invalid_input(format!("unknown argument: {arg}")));
        }
        Ok(Some(config))
    }

    fn chunk_count(self) -> usize {
        self.iterations.div_ceil(self.rows_per_chunk)
    }

    fn fibonacci_values(self) -> usize {
        self.iterations + 2
    }
}

fn invalid_input(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn parse_nonzero_usize(name: &'static str, raw: &str) -> AppResult<usize> {
    let value = raw
        .parse::<usize>()
        .map_err(|err| invalid_input(format!("{name} must parse as usize: {err}")))?;
    if value == 0 {
        return Err(invalid_input(format!("{name} must be nonzero")));
    }
    Ok(value)
}

fn print_usage() {
    println!("fibonacci_superneo_probe");
    println!();
    println!("Direct Fibonacci CCS diagnostic for the generic SuperNeo spine.");
    println!("No VM frontend and no RV32IM relation are used.");
    println!();
    println!("Options:");
    println!("  --iterations=N       number of Fibonacci recurrence steps to generate [default: 8]");
    println!("  --rows-per-chunk=N   generated Fibonacci iterations per SuperNeo chunk [default: 1]");
}

fn print_paper_stage_map() {
    println!("== SuperNeo paper stage map ==");
    println!("section 5: field rows are embedded as ring coefficients; Mz checks become evaluated ring claims");
    println!("section 7.3 Pi_CCS: K fresh CCS rows + k carried CE claims -> K+k CE claims");
    println!("section 7.4 Pi_RLC: K+k CE claims -> one random-linear-combination parent CE claim");
    println!("section 7.5 Pi_DEC: one large-norm parent CE claim -> k_rho small-norm CE children");
    println!("next chunk: those Pi_DEC children are the incoming carried CE claims");
    println!("Spartan2: generic direct CCS/R1CS path; no RV32IM VM is used");
    println!();
}

fn fibonacci_trace_ccs(trace_len: usize) -> CcsStructure<F> {
    let transitions = trace_len - 2;
    let mut m = Mat::zero(transitions, D, F::ZERO);
    for row in 0..transitions {
        m[(row, row)] = F::ONE;
        m[(row, row + 1)] = F::ONE;
        m[(row, row + 2)] = -F::ONE;
    }

    let f = neo_ccs::poly::SparsePoly::new(
        1,
        vec![neo_ccs::poly::Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    CcsStructure::new(vec![m], f).expect("valid Fibonacci CCS")
}

fn fibonacci_trace_from_seeds(a0: u64, a1: u64, trace_len: usize) -> Vec<u64> {
    let mut trace = Vec::with_capacity(trace_len);
    trace.push(a0);
    trace.push(a1);
    while trace.len() < trace_len {
        let next = trace[trace.len() - 2]
            .checked_add(trace[trace.len() - 1])
            .expect("Fibonacci trace value overflowed u64");
        trace.push(next);
    }
    trace
}

fn fibonacci_step(log: &AjtaiSModule, label: &str, values: &[u64]) -> StepInput {
    let mut z = vec![F::ZERO; D];
    for (idx, value) in values.iter().copied().enumerate() {
        z[idx] = F::from_u64(value);
    }

    let mut z_mat = Mat::zero(D, 1, F::ZERO);
    for (idx, value) in z.iter().copied().enumerate() {
        z_mat[(idx % D, idx / D)] = value;
    }

    let m_in = values.len();
    StepInput {
        label: label.to_string(),
        mcs: CcsClaim {
            c: log.commit(&z_mat),
            x: z[..m_in].to_vec(),
            m_in,
        },
        witness: CcsWitness {
            w: z[m_in..].to_vec(),
            Z: z_mat,
        },
    }
}

fn make_ajtai_module(params: &NeoParams, witness_cols: usize) -> AppResult<AjtaiSModule> {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0x5355_5045_524e_454f_u64.to_le_bytes());
    set_global_pp_seeded(D, params.kappa as usize, witness_cols, seed)?;
    Ok(AjtaiSModule::from_global_for_dims(D, witness_cols)?)
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

fn ccs_matrix_nnz(s: &CcsStructure<F>) -> usize {
    s.matrices
        .iter()
        .map(|matrix| match matrix.as_csc() {
            Some(csc) => csc.vals.len(),
            None => matrix.rows(),
        })
        .sum()
}

fn padded_constraints(rows: usize) -> usize {
    if rows == 0 {
        0
    } else {
        rows.next_power_of_two()
    }
}

#[derive(Clone, Copy, Debug)]
struct SparseMatrixDiag {
    nnz: usize,
    max_row_nnz: usize,
    max_col_nnz: usize,
    empty_rows: usize,
    empty_cols: usize,
}

fn sparse_matrix_diag(matrix: &CcsMatrix<F>) -> SparseMatrixDiag {
    let rows = matrix.rows();
    let cols = matrix.cols();
    match matrix.as_csc() {
        Some(csc) => {
            let mut row_counts = vec![0usize; rows];
            let mut max_col_nnz = 0usize;
            let mut empty_cols = 0usize;
            for col in 0..cols {
                let start = csc.col_ptr[col];
                let end = csc.col_ptr[col + 1];
                let col_nnz = end - start;
                max_col_nnz = max_col_nnz.max(col_nnz);
                if col_nnz == 0 {
                    empty_cols += 1;
                }
                for &row in &csc.row_idx[start..end] {
                    if row < rows {
                        row_counts[row] += 1;
                    }
                }
            }
            SparseMatrixDiag {
                nnz: csc.vals.len(),
                max_row_nnz: row_counts.iter().copied().max().unwrap_or(0),
                max_col_nnz,
                empty_rows: row_counts.iter().filter(|&&count| count == 0).count(),
                empty_cols,
            }
        }
        None => {
            let nnz = rows.min(cols);
            SparseMatrixDiag {
                nnz,
                max_row_nnz: usize::from(nnz > 0),
                max_col_nnz: usize::from(nnz > 0),
                empty_rows: rows.saturating_sub(nnz),
                empty_cols: cols.saturating_sub(nnz),
            }
        }
    }
}

fn print_ccs_sparse_matrix_diagnostic(s: &CcsStructure<F>) {
    println!("== CCS sparse matrix diagnostic ==");
    for (idx, matrix) in s.matrices.iter().enumerate() {
        let diag = sparse_matrix_diag(matrix);
        let avg_row = if matrix.rows() == 0 {
            0.0
        } else {
            diag.nnz as f64 / matrix.rows() as f64
        };
        let avg_col = if matrix.cols() == 0 {
            0.0
        } else {
            diag.nnz as f64 / matrix.cols() as f64
        };
        println!(
            "M[{idx}]: rows={}, cols={}, nnz={}, max_row_nnz={}, avg_row_nnz={:.2}, max_col_nnz={}, avg_col_nnz={:.2}, empty_rows={}, empty_cols={}",
            matrix.rows(),
            matrix.cols(),
            diag.nnz,
            diag.max_row_nnz,
            avg_row,
            diag.max_col_nnz,
            avg_col,
            diag.empty_rows,
            diag.empty_cols
        );
    }
    println!(
        "SuperNeo eval cache: {}",
        if build_superneo_eval_cache(s).is_some() {
            "builds"
        } else {
            "unavailable"
        }
    );
    println!();
}

fn norm_budget_lhs(params: &NeoParams, fresh_k: usize, incoming_k: usize) -> u128 {
    ((fresh_k + incoming_k) as u128) * (params.T as u128) * ((params.b as u128).saturating_sub(1))
}

fn max_fresh_k_for_incoming(params: &NeoParams, incoming_k: usize) -> u128 {
    let denom = (params.T as u128) * ((params.b as u128).saturating_sub(1));
    if denom == 0 || params.B == 0 {
        return 0;
    }
    ((params.B as u128 - 1) / denom).saturating_sub(incoming_k as u128)
}

fn print_norm_budget(label: &str, params: &NeoParams, fresh_k: usize, incoming_k: usize) {
    let lhs = norm_budget_lhs(params, fresh_k, incoming_k);
    let rhs = params.B as u128;
    let slack = rhs as i128 - lhs as i128;
    let status = if lhs < rhs { "ok" } else { "OUT_OF_BUDGET" };
    println!(
        "{label}: fresh_K={}, incoming_k={}, lhs=(K+k)*T*(b-1)={}, B={}, slack={}, status={}",
        fresh_k, incoming_k, lhs, rhs, slack, status
    );
}

fn print_parameter_security_audit(
    params: &NeoParams,
    s: &CcsStructure<F>,
    semantic_steps: usize,
    rows_per_chunk: usize,
) {
    let dims = build_dims_and_policy(params, s).expect("valid dims");
    let ring = RotRing::goldilocks();
    let challenge_entropy_bits = D as f64 * (ring.alphabet.len() as f64).log2();
    let expected = NeoParams::goldilocks_paper_b2();
    let paper_b2_core_match = params.has_goldilocks_paper_b2_core()
        && ring.phi_coeffs == goldilocks_paper_b2::PHI_COEFFS.as_slice()
        && ring.alphabet == goldilocks_paper_b2::CHALLENGE_ALPHABET.as_slice()
        && ring.binv_floor == Some(goldilocks_paper_b2::B_INV_FLOOR);
    let entropy_status = if challenge_entropy_bits >= params.lambda as f64 {
        "ok"
    } else {
        "WARN"
    };
    let steady_max_fresh = max_fresh_k_for_incoming(params, params.k_rho as usize);
    let conservative_terms = ((dims.ell + dims.ell_nc) * s.max_degree().max(1) as usize).max(1);
    let conservative_sumcheck_bits = 64.0 * params.s as f64 - (conservative_terms as f64).log2();

    println!("== parameter/security audit ==");
    println!(
        "field=Goldilocks q_bits={} extension_s={} extension_field_bits~{} ring_degree_d={} cyclotomic=X^{} + X^{} + 1",
        u64::BITS - params.q.leading_zeros(),
        params.s,
        (u64::BITS - params.q.leading_zeros()) * params.s,
        D,
        goldilocks_paper_b2::D,
        goldilocks_paper_b2::PHI_MID_DEGREE
    );
    println!(
        "challenge_coeff_set={:?}, challenge_entropy_bits={:.2}, T_worst_case={}, b={}, k_dec={}, B={}, kappa={}, lambda={}",
        ring.alphabet,
        challenge_entropy_bits,
        params.T,
        params.b,
        params.k_rho,
        params.B,
        params.kappa,
        params.lambda
    );
    println!(
        "paper_appendix_b2_core_params_match={} (expected d={}, eta={}, kappa={}, k_dec={}, B={}, T={}, challenge_coeff_set={:?}, s={}, canonical_lambda={})",
        if paper_b2_core_match { "yes" } else { "no" },
        expected.d,
        expected.eta,
        expected.kappa,
        expected.k_rho,
        expected.B,
        expected.T,
        goldilocks_paper_b2::CHALLENGE_ALPHABET,
        expected.s,
        expected.lambda
    );
    println!(
        "effective_lambda={}{}",
        params.lambda,
        if params.lambda == expected.lambda {
            " (canonical paper profile)"
        } else {
            " (auto-lowered by extension policy for this CCS shape)"
        }
    );
    println!(
        "challenge_entropy_vs_lambda: {:.2} bits vs lambda={}, status={entropy_status}",
        challenge_entropy_bits, params.lambda
    );
    println!(
        "Pi_CCS soundness estimate: FE_rounds={}, NC_rounds={}, max_degree={}, conservative_error_bits~{:.1}",
        dims.ell,
        dims.ell_nc,
        s.max_degree(),
        conservative_sumcheck_bits
    );
    print_norm_budget("norm budget cold chunk", params, rows_per_chunk.min(semantic_steps), 0);
    print_norm_budget(
        "norm budget steady chunk",
        params,
        rows_per_chunk.min(semantic_steps),
        params.k_rho as usize,
    );
    println!("max_fresh_K_at_steady_state={steady_max_fresh}");
    if semantic_steps <= rows_per_chunk {
        println!(
            "note: legacy cold-start packaged proof has one chunk; direct F' carrier still starts from canonical CE(b)^k"
        );
    }
    if rows_per_chunk as u128 > steady_max_fresh {
        println!("warning: rows_per_chunk exceeds the steady-state norm budget for current parameters");
    }
    println!();
}

fn print_shape(config: Config, params: &NeoParams, s: &CcsStructure<F>) -> AppResult<()> {
    let dims = build_dims_and_policy(params, s)?;
    println!("== direct Fibonacci CCS ==");
    println!(
        "continuous trace: {} Fibonacci iterations, {} values",
        config.iterations,
        config.fibonacci_values()
    );
    println!(
        "generated fold steps: {}; constraints per fold step: 1",
        config.iterations
    );
    println!("equation: z[i] + z[i+1] - z[i+2] = 0, one CCS row per transition");
    println!(
        "ccs: n={} rows, m={} columns, t={} matrix, degree={}, matrix_nnz={}",
        s.n,
        s.m,
        s.t(),
        s.max_degree(),
        ccs_matrix_nnz(s)
    );
    println!(
        "pi_ccs dims: ell_d={}, ell_n={}, ell_m={}, ell={}, ell_nc={}, ell_max={}, d_sc={}",
        dims.ell_d, dims.ell_n, dims.ell_m, dims.ell, dims.ell_nc, dims.ell_max, dims.d_sc
    );
    println!(
        "params: b={}, k_rho={}, B={}, kappa={}, T={}, extension_s={}",
        params.b, params.k_rho, params.B, params.kappa, params.T, params.s
    );
    println!(
        "fold schedule: RowsPerChunk({}); expected chunks={}",
        config.rows_per_chunk,
        config.chunk_count()
    );
    println!();
    Ok(())
}

fn print_steps(config: Config, traces: &[Vec<u64>]) {
    println!("== generated fold steps ==");
    for (idx, trace) in traces.iter().enumerate() {
        let chunk = idx / config.rows_per_chunk;
        let row_in_chunk = idx % config.rows_per_chunk;
        println!(
            "step[{idx}]: chunk={chunk}, row_in_chunk={row_in_chunk}, label=fib_iter_{idx}, global_iteration={idx}, public_values={}, equation {} + {} - {} = 0",
            trace.len(),
            trace[0],
            trace[1],
            trace[trace.len() - 1]
        );
    }
    println!();
}

fn max_round_width(rounds: &[Vec<K>]) -> usize {
    rounds.iter().map(Vec::len).max().unwrap_or(0)
}

fn ce_shape_summary(claim: &CeClaim<Commitment, F, K>) -> String {
    let y_ring_cols = claim.y_ring.first().map(Vec::len).unwrap_or(0);
    format!(
        "commitment={}x{} elems={}, X={}x{}, r={}, s_col={}, y_ring={}x{}, ct={}, aux={}, y_zcol={}, m_in={}",
        claim.c.d,
        claim.c.kappa,
        claim.c.data.len(),
        claim.X.rows(),
        claim.X.cols(),
        claim.r.len(),
        claim.s_col.len(),
        claim.y_ring.len(),
        y_ring_cols,
        claim.ct.len(),
        claim.aux_openings.len(),
        claim.y_zcol.len(),
        claim.m_in
    )
}

fn print_fold_evolution(params: &NeoParams, build: &SuperNeoIvcBuild, perf: &RunProvePerf) {
    println!("== protocol evolution by chunk ==");
    for (idx, relation) in build.relations.iter().enumerate() {
        let fresh = relation.chunk.steps.len();
        let incoming_carry = relation.state_in.carry.claims.len();
        let pi_ccs_inputs = fresh + incoming_carry;
        let next_carry = relation.state_out.carry.claims.len();
        let replay = &relation.replay_witness.ccs_replay_proof;
        println!(
            "chunk[{idx}] start={} fresh_CCS_K={} incoming_CE_k={} Pi_CCS_inputs_K_plus_k={}",
            relation.chunk.start_index, fresh, incoming_carry, pi_ccs_inputs
        );
        print_norm_budget("  norm budget", params, fresh, incoming_carry);
        println!(
            "  Pi_CCS out: CE_claims={}, FE_rounds={} max_FE_round_width={}, FE_challenges={}, NC_rounds={} max_NC_round_width={}, NC_challenges={}, header_digest_bytes={}",
            relation.replay_witness.ccs_outputs.len(),
            replay.sumcheck_rounds.len(),
            max_round_width(&replay.sumcheck_rounds),
            replay.sumcheck_rounds.len(),
            replay.sumcheck_rounds_nc.len(),
            max_round_width(&replay.sumcheck_rounds_nc),
            replay.sumcheck_rounds_nc.len(),
            replay.header_digest.len()
        );
        if let Some(first) = relation.replay_witness.ccs_outputs.first() {
            println!("  first Pi_CCS CE shape: {}", ce_shape_summary(first));
        }
        if let Some(first_child) = relation.state_out.carry.claims.first() {
            println!(
                "  Pi_DEC out: children={} next_incoming_CE_k={} first_child={}",
                next_carry,
                next_carry,
                ce_shape_summary(first_child)
            );
        } else {
            println!("  Pi_DEC out: children=0 next_incoming_CE_k=0");
        }
        if let Some(chunk_perf) = perf.chunks.get(idx) {
            println!(
                "  timings ms: Pi_CCS={:.3}, Pi_RLC={:.3}, Pi_DEC_split={:.3}, Pi_DEC_commit={:.3}, Pi_DEC_check={:.3}, total={:.3}",
                chunk_perf.ccs_ms,
                chunk_perf.rlc_ms,
                chunk_perf.dec_split_ms,
                chunk_perf.dec_commit_ms,
                chunk_perf.dec_ms,
                chunk_perf.total_ms
            );
        }
    }
    println!();
}

fn print_superneo_embedding_diagnostic(
    params: &NeoParams,
    s: &CcsStructure<F>,
    build: &SuperNeoIvcBuild,
    first_step: &StepInput,
) -> AppResult<()> {
    let relation = build
        .relations
        .first()
        .ok_or_else(|| invalid_input("SuperNeo embedding diagnostic requires at least one direct relation"))?;
    let output = relation
        .replay_witness
        .ccs_outputs
        .first()
        .ok_or_else(|| invalid_input("SuperNeo embedding diagnostic requires at least one Pi_CCS output"))?;
    let cache = build_superneo_eval_cache(s)
        .ok_or_else(|| invalid_input("SuperNeo embedding diagnostic could not build SuperNeo eval cache"))?;
    let z = decode_superneo_coeffs_from_witness_mat(&first_step.witness.Z, s.m)?;
    let chi_r = neo_ccs::utils::tensor_point::<K>(&output.r);
    let expected_rows = eval_all_mats_ring_cached(&cache, &z, &chi_r, s.n);
    if expected_rows.len() != s.t() || output.y_ring.len() != s.t() {
        return Err(invalid_input(format!(
            "SuperNeo embedding diagnostic matrix count mismatch: expected {}, recomputed {}, proof {}",
            s.t(),
            expected_rows.len(),
            output.y_ring.len()
        )));
    }

    let mut nonconstant_nonzero = 0usize;
    for (matrix_idx, expected) in expected_rows.iter().enumerate() {
        let actual = &output.y_ring[matrix_idx];
        if actual.len() < D {
            return Err(invalid_input(format!(
                "SuperNeo embedding diagnostic expected y_ring[{matrix_idx}] to have at least {D} coefficients, got {}",
                actual.len()
            )));
        }
        for coeff_idx in 0..D {
            if actual[coeff_idx] != expected[coeff_idx] {
                return Err(invalid_input(format!(
                    "SuperNeo embedding diagnostic mismatch at matrix {matrix_idx}, coeff {coeff_idx}"
                )));
            }
        }
        if actual.iter().skip(D).any(|&coeff| coeff != K::ZERO) {
            return Err(invalid_input(format!(
                "SuperNeo embedding diagnostic expected padded y_ring[{matrix_idx}] tail after coeff {D} to be zero"
            )));
        }
        nonconstant_nonzero += actual
            .iter()
            .skip(1)
            .filter(|&&coeff| coeff != K::ZERO)
            .count();
    }

    let ct = ct_from_y_ring_for_ccs_m(&output.y_ring, params, s.m);
    if ct != output.ct {
        return Err(invalid_input(
            "SuperNeo embedding diagnostic: ct(y_ring) does not match CE scalar openings",
        ));
    }

    println!("== SuperNeo embedding diagnostic ==");
    println!("checked: first fresh Pi_CCS output recomputed from witness with eval_all_mats_ring_cached");
    println!(
        "embedding surface: y_ring_rows={}, ring_coeffs={}, padded_coeffs_per_row={}, nonconstant_nonzero_coeffs={}",
        output.y_ring.len(),
        D,
        output.y_ring.first().map(Vec::len).unwrap_or(0),
        nonconstant_nonzero
    );
    println!(
        "constant terms: ct(y_ring) matches CE scalar openings for {} matrices",
        output.ct.len()
    );
    println!();
    Ok(())
}

fn print_chunk_prove_perf(perf: &RunProvePerf) {
    println!("== prove stage breakdown ==");
    println!(
        "run totals: chunks={}, fresh_steps={}, incoming_main_claims={}, pi_ccs_outputs={}, dec_children={}, total_ms={:.3}",
        perf.chunk_count(),
        perf.fresh_steps(),
        perf.incoming_main_claims(),
        perf.ccs_outputs(),
        perf.dec_children(),
        perf.total_ms
    );
    println!(
        "stage totals ms: prepare={:.3}, pi_ccs={:.3}, dims={:.3}, pi_rlc_prepare={:.3}, pi_rlc={:.3}, pi_dec_split={:.3}, pi_dec_commit={:.3}, pi_dec={:.3}",
        perf.prepare_inputs_ms(),
        perf.ccs_ms(),
        perf.dims_ms(),
        perf.rlc_prepare_ms(),
        perf.rlc_ms(),
        perf.dec_split_ms(),
        perf.dec_commit_ms(),
        perf.dec_ms()
    );
    println!(
        "pi_ccs internals ms: bind={:.3}, sample_challenges={:.3}, fe_sumcheck={:.3}, nc_sumcheck={:.3}, output_materialize={:.3}",
        perf.ccs_bind_ms(),
        perf.ccs_sample_challenges_ms(),
        perf.ccs_fe_sumcheck_ms(),
        perf.ccs_nc_sumcheck_ms(),
        perf.ccs_output_materialize_ms()
    );
    println!();
    for (idx, chunk) in perf.chunks.iter().enumerate() {
        print_one_prove_chunk(idx, chunk);
    }
    println!();
}

fn print_one_prove_chunk(idx: usize, chunk: &ChunkProvePerf) {
    println!(
        "chunk[{idx}] start={} fresh_steps={} incoming_main={} pi_ccs_outputs={} dec_children={} total_ms={:.3}",
        chunk.start_index,
        chunk.fresh_steps,
        chunk.incoming_main_claims,
        chunk.ccs_outputs,
        chunk.dec_children,
        chunk.total_ms
    );
    println!("  1 prepare inputs: {:.3} ms", chunk.prepare_inputs_ms);
    println!(
        "  2 Pi_CCS: {:.3} ms = bind {:.3} + sample {:.3} + FE sumcheck {:.3} + NC sumcheck {:.3} + output {:.3}",
        chunk.ccs_ms,
        chunk.ccs_bind_ms,
        chunk.ccs_sample_challenges_ms,
        chunk.ccs_fe_sumcheck_ms,
        chunk.ccs_nc_sumcheck_ms,
        chunk.ccs_output_materialize_ms
    );
    println!(
        "  3 Pi_RLC: prepare {:.3} ms, fold {:.3} ms",
        chunk.rlc_prepare_ms, chunk.rlc_ms
    );
    println!(
        "  4 Pi_DEC: split {:.3} ms, commit {:.3} ms, prove/check {:.3} ms",
        chunk.dec_split_ms, chunk.dec_commit_ms, chunk.dec_ms
    );
}

fn timing_status(total_ms: f64, unattributed_ms: f64) -> &'static str {
    if total_ms <= 0.0 {
        "ok"
    } else if unattributed_ms.abs() <= 1.0 || unattributed_ms.abs() / total_ms <= 0.05 {
        "ok"
    } else {
        "WARN"
    }
}

fn print_timing_accounting(label: &str, total_ms: f64, named_ms: f64) {
    let unattributed_ms = total_ms - named_ms;
    let pct = if total_ms <= 0.0 {
        0.0
    } else {
        100.0 * unattributed_ms / total_ms
    };
    println!(
        "{label}: total={:.3} ms, named_sum={:.3} ms, unattributed={:.3} ms ({:.1}%), status={}",
        total_ms,
        named_ms,
        unattributed_ms,
        pct,
        timing_status(total_ms, unattributed_ms)
    );
}

fn print_prove_timing_accounting(perf: &RunProvePerf) {
    println!("== prove timing accounting ==");
    let named = perf.prepare_inputs_ms()
        + perf.ccs_ms()
        + perf.dims_ms()
        + perf.rlc_prepare_ms()
        + perf.rlc_ms()
        + perf.dec_split_ms()
        + perf.dec_commit_ms()
        + perf.dec_ms();
    let ccs_named = perf.ccs_bind_ms()
        + perf.ccs_sample_challenges_ms()
        + perf.ccs_fe_sumcheck_ms()
        + perf.ccs_nc_sumcheck_ms()
        + perf.ccs_output_materialize_ms();
    print_timing_accounting("run", perf.total_ms, named);
    print_timing_accounting("Pi_CCS internals", perf.ccs_ms(), ccs_named);
    println!();
}

fn print_verify_perf(perf: &RunVerifyPerf) {
    println!("== legacy cold-start packaged verify diagnostic ==");
    println!(
        "run totals: chunks={}, fresh_steps={}, incoming_main_claims={}, pi_ccs_outputs={}, dec_children={}, total_ms={:.3}",
        perf.chunk_count(),
        perf.fresh_steps(),
        perf.incoming_main_claims(),
        perf.ccs_outputs(),
        perf.dec_children(),
        perf.total_ms
    );
    println!(
        "stage totals ms: prepare={:.3}, pi_ccs={:.3}, digest_checks={:.3}, dims={:.3}, pi_rlc={:.3}, pi_dec={:.3}",
        perf.prepare_inputs_ms(),
        perf.ccs_ms(),
        perf.digest_checks_ms(),
        perf.dims_ms(),
        perf.rlc_ms(),
        perf.dec_ms()
    );
    println!(
        "pi_ccs verify internals ms: bind={:.3}, FE sumcheck={:.3}, NC sumcheck={:.3}, output_checks={:.3}, terminal={:.3}",
        perf.ccs_bind_ms(),
        perf.ccs_fe_sumcheck_ms(),
        perf.ccs_nc_sumcheck_ms(),
        perf.ccs_output_checks_ms(),
        perf.ccs_terminal_ms()
    );
    let named = perf.prepare_inputs_ms()
        + perf.ccs_ms()
        + perf.digest_checks_ms()
        + perf.dims_ms()
        + perf.rlc_ms()
        + perf.dec_ms();
    let ccs_named = perf.ccs_bind_ms()
        + perf.ccs_fe_sumcheck_ms()
        + perf.ccs_nc_sumcheck_ms()
        + perf.ccs_output_checks_ms()
        + perf.ccs_terminal_ms();
    print_timing_accounting("verify run", perf.total_ms, named);
    print_timing_accounting("verify Pi_CCS internals", perf.ccs_ms(), ccs_named);
    println!();
    for (idx, chunk) in perf.chunks.iter().enumerate() {
        print_one_verify_chunk(idx, chunk);
    }
    println!();
}

fn print_one_verify_chunk(idx: usize, chunk: &ChunkVerifyPerf) {
    println!(
        "verify chunk[{idx}] start={} fresh_steps={} incoming_main={} pi_ccs_outputs={} dec_children={} total_ms={:.3}",
        chunk.start_index,
        chunk.fresh_steps,
        chunk.incoming_main_claims,
        chunk.ccs_outputs,
        chunk.dec_children,
        chunk.total_ms
    );
    println!(
        "  Pi_CCS: bind {:.3}, FE {:.3}, NC {:.3}, output checks {:.3}, terminal {:.3}",
        chunk.ccs_bind_ms,
        chunk.ccs_fe_sumcheck_ms,
        chunk.ccs_nc_sumcheck_ms,
        chunk.ccs_output_checks_ms,
        chunk.ccs_terminal_ms
    );
    println!(
        "  Pi_RLC: challenge {:.3}, rho_mats {:.3}, x {:.3}, y {:.3}, aux {:.3}, commitment {:.3}, total {:.3}",
        chunk.rlc_challenge_ms,
        chunk.rlc_rho_mats_ms,
        chunk.rlc_x_ms,
        chunk.rlc_y_ms,
        chunk.rlc_aux_ms,
        chunk.rlc_commitment_ms,
        chunk.rlc_ms
    );
    println!("  Pi_DEC: {:.3} ms", chunk.dec_ms);
}

fn print_optimization_ranking(prove_perf: &RunProvePerf) {
    let ccs_named = prove_perf.ccs_bind_ms()
        + prove_perf.ccs_sample_challenges_ms()
        + prove_perf.ccs_fe_sumcheck_ms()
        + prove_perf.ccs_nc_sumcheck_ms()
        + prove_perf.ccs_output_materialize_ms();
    let ccs_unattributed = (prove_perf.ccs_ms() - ccs_named).max(0.0);
    let mut items = vec![
        ("Pi_DEC prove/check", prove_perf.dec_ms()),
        ("Pi_CCS FE sumcheck", prove_perf.ccs_fe_sumcheck_ms()),
        ("Pi_CCS unattributed", ccs_unattributed),
        ("Pi_DEC child commits", prove_perf.dec_commit_ms()),
    ];
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("== optimization ranking ==");
    for (rank, (label, ms)) in items.into_iter().take(5).enumerate() {
        println!("  {}. {}: {:.3} ms", rank + 1, label, ms);
    }
    println!();
}

fn print_folding_timing_table(perf: &RunProvePerf, ccs_rows_per_claim: usize) {
    let chunks = perf.chunks.iter().take(4).collect::<Vec<_>>();
    println!("== folding timing table ==");
    if perf.chunks.len() > chunks.len() {
        println!("showing first {} of {} foldings", chunks.len(), perf.chunks.len());
    }
    print_fold_table_row(
        "metric",
        chunks
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("fold[{idx}]")),
    );
    print_fold_table_row(
        "fresh CCS claims",
        chunks.iter().map(|chunk| chunk.fresh_steps.to_string()),
    );
    print_fold_table_row(
        "incoming CE claims",
        chunks
            .iter()
            .map(|chunk| chunk.incoming_main_claims.to_string()),
    );
    print_fold_table_row(
        "generate z/input ms",
        chunks
            .iter()
            .map(|chunk| format!("{:.3}", chunk.prepare_inputs_ms)),
    );
    print_fold_table_row("Pi_CCS ms", chunks.iter().map(|chunk| format!("{:.3}", chunk.ccs_ms)));
    print_fold_table_row(
        "  FE sumcheck ms (CCS rows)",
        chunks.iter().map(|chunk| {
            format!(
                "{:.3} ({})",
                chunk.ccs_fe_sumcheck_ms,
                chunk.fresh_steps * ccs_rows_per_claim
            )
        }),
    );
    print_fold_table_row(
        "  NC sumcheck ms (openings)",
        chunks
            .iter()
            .map(|chunk| format!("{:.3} ({})", chunk.ccs_nc_sumcheck_ms, chunk.ccs_outputs)),
    );
    print_fold_table_row(
        "Pi_RLC fold ms",
        chunks.iter().map(|chunk| format!("{:.3}", chunk.rlc_ms)),
    );
    print_fold_table_row(
        "Pi_DEC split ms",
        chunks
            .iter()
            .map(|chunk| format!("{:.3}", chunk.dec_split_ms)),
    );
    print_fold_table_row(
        "Pi_DEC commit ms",
        chunks
            .iter()
            .map(|chunk| format!("{:.3}", chunk.dec_commit_ms)),
    );
    print_fold_table_row(
        "Pi_DEC check ms",
        chunks.iter().map(|chunk| format!("{:.3}", chunk.dec_ms)),
    );
    print_fold_table_row(
        "chunk wall total ms",
        chunks.iter().map(|chunk| format!("{:.3}", chunk.total_ms)),
    );
    println!("note: Pi_CCS FE count is fresh CCS claims * CCS rows; Pi_CCS NC count is K+k claim openings");
    println!("note: Pi_RLC is the linear folding reduction; Pi_DEC decomposes the folded parent for the next chunk");
    println!();
}

fn print_fold_table_row<I>(label: &str, values: I)
where
    I: IntoIterator<Item = String>,
{
    print!("{label:<34}");
    for value in values {
        print!(" | {value:>12}");
    }
    println!();
}

fn print_constraint_breakdown(
    s: &CcsStructure<F>,
    prove_perf: &RunProvePerf,
    spartan_perf: Option<&DirectCcsRecursiveIvcSnarkPerf>,
) {
    let fresh = prove_perf.fresh_steps();
    let total_ccs_rows = s.n.saturating_mul(fresh);
    println!("== constraint breakdown ==");
    println!("circuit constraints:");
    println!("  Fibonacci CCS rows per claim: {}", s.n);
    println!("  folded claims: {fresh}");
    println!("  total semantic CCS rows folded: {total_ccs_rows}");
    println!(
        "  CCS columns={}, matrices={}, degree={}, nnz={}",
        s.m,
        s.t(),
        s.max_degree(),
        ccs_matrix_nnz(s)
    );
    println!("Spartan2 constraints:");
    if let Some(spartan_perf) = spartan_perf {
        let terminal = &spartan_perf.terminal;
        println!(
            "  backend R1CS constraints passed: {} terminal + {} folded F' chain + {} folded/default F' accumulator authority = {}",
            terminal.r1cs.sizes[0],
            spartan_perf.f_prime_chain_constraints,
            spartan_perf.f_prime_final_ce_constraints,
            terminal.r1cs.sizes[0] + spartan_perf.f_prime_chain_constraints + spartan_perf.f_prime_final_ce_constraints
        );
        println!(
            "  padded backend constraints: {} terminal + ~{} folded F' chain + ~{} folded/default F' accumulator authority",
            terminal.r1cs.sizes[4],
            padded_constraints(spartan_perf.f_prime_chain_constraints),
            padded_constraints(spartan_perf.f_prime_final_ce_constraints)
        );
        println!(
            "  backend public inputs={}, challenges={}, nnz_total={}",
            terminal.r1cs.sizes[8], terminal.r1cs.sizes[9], terminal.r1cs.nnz
        );
        println!(
            "  terminal direct CCS F' chunk constraints: {:?}",
            terminal.chunks.constraints_by_chunk
        );
        println!(
            "  terminal Construction-2 folded F' accumulator constraints: {}",
            terminal.constraints.construction2_fold
        );
        println!(
            "  final post-DEC CE relation constraints: {}",
            terminal.final_ce.relation_constraints
        );
        println!(
            "  folded/default F' accumulator authority constraints: {}",
            spartan_perf.f_prime_final_ce_constraints
        );
        println!(
            "  folded F' chain constraints: {}",
            spartan_perf.f_prime_chain_constraints
        );
    } else {
        println!("  not measured: Spartan terminal compression did not run");
    }
    println!();
}

fn run() -> AppResult<()> {
    let Some(config) = Config::from_args()? else {
        print_usage();
        return Ok(());
    };

    let params = NeoParams::goldilocks_auto_r1cs_ccs(1)?;
    let ccs = fibonacci_trace_ccs(FIB_STEP_TRACE_LEN);
    let log = make_ajtai_module(&params, 1)?;
    let full_trace = fibonacci_trace_from_seeds(1, 2, config.fibonacci_values());
    let traces = (0..config.iterations)
        .map(|idx| full_trace[idx..idx + FIB_STEP_TRACE_LEN].to_vec())
        .collect::<Vec<_>>();
    let steps = traces
        .iter()
        .enumerate()
        .map(|(idx, values)| fibonacci_step(&log, &format!("fib_iter_{idx}"), values))
        .collect::<Vec<_>>();
    let first_step_for_embedding_check = steps[0].clone();

    print_shape(config, &params, &ccs)?;
    print_ccs_sparse_matrix_diagnostic(&ccs);
    print_parameter_security_audit(&params, &ccs, config.iterations, config.rows_per_chunk);
    print_paper_stage_map();
    print_steps(config, &traces);

    let schedule = FoldSchedule::RowsPerChunk(config.rows_per_chunk);
    let public_steps = steps.iter().map(StepInput::public).collect::<Vec<_>>();
    let steps_for_direct_carrier = steps.clone();
    let steps_for_spartan = steps.clone();
    let (packaged, _legacy_prove_perf, final_carry) = prove_and_package_with_final_carry_perf(
        FoldingMode::Optimized,
        schedule,
        &params,
        &ccs,
        steps.clone(),
        &log,
        ajtai_mixers(),
    )?;
    let legacy_superneo_ivc =
        build_superneo_ivc_relations_with_perf(schedule, &params, &ccs, steps.clone(), &log, ajtai_mixers())?;
    let direct_program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, FIB_STEP_TRACE_LEN)?;
    let direct_initial_carry = direct_program.canonical_zero_carry()?;
    let direct_initial_carry_len = direct_initial_carry.claims.len();
    let direct_superneo_ivc = build_superneo_ivc_relations_with_initial_carry_accumulator_handle_perf(
        schedule,
        &params,
        &ccs,
        steps_for_direct_carrier,
        direct_initial_carry,
        &log,
        ajtai_mixers(),
    )?;
    let direct_prove_perf = direct_superneo_ivc.prove_perf();
    print_chunk_prove_perf(&direct_prove_perf);
    print_prove_timing_accounting(&direct_prove_perf);
    print_fold_evolution(&params, &direct_superneo_ivc, &direct_prove_perf);
    print_superneo_embedding_diagnostic(&params, &ccs, &direct_superneo_ivc, &first_step_for_embedding_check)?;

    let (verified_claims, verify_perf) =
        verify_packaged_with_perf(FoldingMode::Optimized, &params, &ccs, &packaged, ajtai_mixers())?;
    if verified_claims != packaged.statement.final_main_claims {
        return Err(invalid_input("verified final claims did not match packaged statement"));
    }
    if public_steps.len() != packaged.statement.public_step_count() {
        return Err(invalid_input("public step count changed during packaging"));
    }
    print_verify_perf(&verify_perf);
    if legacy_superneo_ivc.final_state.carry.claims != packaged.statement.final_main_claims {
        return Err(invalid_input(
            "legacy cold-start SuperNeo IVC carrier final claims did not match packaged native fold claims",
        ));
    }
    print_superneo_ivc_carrier(&direct_superneo_ivc, direct_initial_carry_len)?;
    let spartan_perf = print_spartan(
        &params,
        &ccs,
        &packaged,
        &final_carry,
        &steps_for_spartan,
        &direct_superneo_ivc,
        &log,
    )?;
    print_optimization_ranking(&direct_prove_perf);
    print_folding_timing_table(&direct_prove_perf, ccs.n);
    print_constraint_breakdown(&ccs, &direct_prove_perf, spartan_perf.as_ref());
    print_final_summary(&direct_prove_perf, spartan_perf.as_ref());

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
