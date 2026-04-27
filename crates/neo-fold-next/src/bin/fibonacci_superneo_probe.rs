// Diagnostic probe binary: the repo's 1,500-line implementation-file limit does not apply here.

use std::env;
use std::error::Error;

use neo_ajtai::{s_mul_add, scale_commitment_add_inplace, set_global_pp_seeded, AjtaiSModule, Commitment};
use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsMatrix, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_fold_next::decider::spartan2::{
    build_spartan2_self_bound_decider_relation, prove_spartan2_decider_with_perf, setup_spartan2_decider,
    verify_spartan2_decider, Spartan2DeciderShape, Spartan2DeciderTarget,
};
use neo_fold_next::finalize::FixedShapeChunkSummary;
use neo_fold_next::proof::{
    ChunkProvePerf, ChunkVerifyPerf, FoldSchedule, PackagedProof, PublicChunk, PublicStep, RunProvePerf, RunVerifyPerf,
    StepInput,
};
use neo_fold_next::prover::CommitmentMixers;
use neo_fold_next::run::{prove_and_package_with_perf, verify_packaged_with_perf};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F, K};
use neo_params::{goldilocks_paper_b2, NeoParams};
use neo_reductions::api::FoldingMode;
use neo_reductions::common::{ct_from_y_ring_for_ccs_m, decode_superneo_coeffs_from_witness_mat, RotRing};
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices};
use neo_reductions::superneo_eval::{build_superneo_eval_cache, eval_all_mats_ring_cached};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

type AppResult<T> = Result<T, Box<dyn Error>>;

const FIXED_SHAPE_DIGEST_FIELD_LEN: usize = 4;
const FIB_STEP_TRACE_LEN: usize = 3;
const PACKED_BYTES_PER_LIMB: usize = 7;

#[derive(Clone, Copy, Debug)]
struct Config {
    iterations: usize,
    rows_per_chunk: usize,
}

#[derive(Clone, Copy, Debug)]
struct SpartanRunSummary {
    prove_ms: f64,
    verify_ms: f64,
    final_proof_bytes: usize,
    snark_bytes: usize,
    backend_r1cs_constraints: usize,
    padded_backend_r1cs_constraints: usize,
    backend_public_inputs: usize,
    backend_challenges: usize,
    backend_nnz_total: usize,
    spartan_pcs_ms: f64,
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
    println!("No VM frontend and no RV64IM relation are used.");
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
    println!("Spartan2: proves the fixed-shape public/backend binding shell over the packaged run");
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

fn extend_packed_bytes_as_fields(dst: &mut Vec<F>, bytes: &[u8]) {
    dst.push(F::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(PACKED_BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        dst.push(F::from_u64(u64::from_le_bytes(limb)));
    }
}

fn packed_bytes_field_len(bytes_len: usize) -> usize {
    1 + bytes_len.div_ceil(PACKED_BYTES_PER_LIMB)
}

fn fixed_shape_summary_fields_for_spartan() -> usize {
    FixedShapeChunkSummary::packed_field_len() + FIXED_SHAPE_DIGEST_FIELD_LEN
}

fn spartan_transition_binding_fields() -> usize {
    2 * packed_bytes_field_len(32)
}

fn digest32_as_fields(digest: [u8; 32]) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
    [
        F::from_u64(u64::from_le_bytes(digest[0..8].try_into().expect("digest limb 0"))),
        F::from_u64(u64::from_le_bytes(digest[8..16].try_into().expect("digest limb 1"))),
        F::from_u64(u64::from_le_bytes(digest[16..24].try_into().expect("digest limb 2"))),
        F::from_u64(u64::from_le_bytes(digest[24..32].try_into().expect("digest limb 3"))),
    ]
}

fn digest_fields_as_digest32(fields: [F; FIXED_SHAPE_DIGEST_FIELD_LEN]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (index, field) in fields.into_iter().enumerate() {
        out[index * 8..(index + 1) * 8].copy_from_slice(&field.as_canonical_u64().to_le_bytes());
    }
    out
}

fn poseidon_digest_fields(input: &[F]) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
    poseidon2_hash(input)
}

fn fixed_shape_recursive_seed(domain: &[u8]) -> [u8; 32] {
    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields(&mut preimage, domain);
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

fn ccs_claim_digest_fields_into(claim: &CcsClaim<Commitment, F>, scratch: &mut Vec<F>) -> [F; 4] {
    scratch.clear();
    scratch.reserve(256);
    extend_packed_bytes_as_fields(scratch, b"neo.fold.next/finalize/ccs_claim_digest/v1");
    scratch.push(F::from_u64(claim.c.d as u64));
    scratch.push(F::from_u64(claim.c.kappa as u64));
    scratch.push(F::from_u64(claim.c.data.len() as u64));
    scratch.extend_from_slice(&claim.c.data);
    scratch.push(F::from_u64(claim.x.len() as u64));
    scratch.extend_from_slice(&claim.x);
    scratch.push(F::from_u64(claim.m_in as u64));
    poseidon_digest_fields(scratch)
}

fn public_step_digest_fields_into(step: &PublicStep, claim_scratch: &mut Vec<F>, step_scratch: &mut Vec<F>) -> [F; 4] {
    step_scratch.clear();
    step_scratch.reserve(96);
    extend_packed_bytes_as_fields(step_scratch, b"neo.fold.next/finalize/public_step_digest/v1");
    extend_packed_bytes_as_fields(step_scratch, step.label.as_bytes());
    step_scratch.extend_from_slice(&ccs_claim_digest_fields_into(&step.mcs, claim_scratch));
    poseidon_digest_fields(step_scratch)
}

fn public_chunk_digest_fields(chunk: &PublicChunk) -> [F; 4] {
    let mut claim_scratch = Vec::<F>::with_capacity(256);
    let mut step_scratch = Vec::<F>::with_capacity(96);
    let mut chunk_scratch = Vec::<F>::new();
    chunk_scratch.reserve(32 + (chunk.steps.len() * 4));
    extend_packed_bytes_as_fields(&mut chunk_scratch, b"neo.fold.next/finalize/public_chunk_digest/v1");
    chunk_scratch.push(F::from_u64(chunk.start_index as u64));
    chunk_scratch.push(F::from_u64(chunk.steps.len() as u64));
    for step in &chunk.steps {
        chunk_scratch.extend_from_slice(&public_step_digest_fields_into(
            step,
            &mut claim_scratch,
            &mut step_scratch,
        ));
    }
    poseidon_digest_fields(&chunk_scratch)
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
        println!("warning: this run has one SuperNeo chunk and does not exercise carried CE claims");
    }
    if rows_per_chunk as u128 > steady_max_fresh {
        println!("warning: rows_per_chunk exceeds the steady-state norm budget for current parameters");
    }
    println!();
}

fn diagnostic_relation_digest(params: &NeoParams, s: &CcsStructure<F>) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/fibonacci_superneo_probe/relation");
    tr.append_message(b"neo.fold.next/fibonacci_superneo_probe/relation/version", b"v1");
    tr.append_u64s(
        b"neo.fold.next/fibonacci_superneo_probe/relation/params",
        &[
            params.b as u64,
            params.k_rho as u64,
            params.B,
            params.kappa as u64,
            params.T as u64,
            params.s as u64,
            params.lambda as u64,
        ],
    );
    tr.append_u64s(
        b"neo.fold.next/fibonacci_superneo_probe/relation/ccs_shape",
        &[s.n as u64, s.m as u64, s.t() as u64, s.max_degree() as u64],
    );
    let matrix_digest = digest_ccs_matrices(s);
    tr.append_fields(
        b"neo.fold.next/fibonacci_superneo_probe/relation/ccs_matrix_digest",
        &matrix_digest,
    );
    tr.digest32()
}

fn diagnostic_chunk_transition_digest(
    chunk_index: usize,
    public_chunk: &PublicChunk,
    proof_chunk: &neo_fold_next::proof::ChunkProof,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/fibonacci_superneo_probe/chunk_transition");
    tr.append_message(
        b"neo.fold.next/fibonacci_superneo_probe/chunk_transition/version",
        b"v1",
    );
    tr.append_u64s(
        b"neo.fold.next/fibonacci_superneo_probe/chunk_transition/meta",
        &[
            chunk_index as u64,
            public_chunk.start_index as u64,
            public_chunk.steps.len() as u64,
            proof_chunk.ccs_outputs.len() as u64,
            proof_chunk.dec.children.len() as u64,
        ],
    );
    tr.append_message(
        b"neo.fold.next/fibonacci_superneo_probe/chunk_transition/pi_ccs_header_digest",
        &proof_chunk.ccs_proof.header_digest,
    );
    tr.append_message(
        b"neo.fold.next/fibonacci_superneo_probe/chunk_transition/chunk_relation_digest",
        &proof_chunk.relation_digest,
    );
    tr.digest32()
}

fn fixed_shape_summaries_and_transition_digests(
    packaged: &PackagedProof,
) -> AppResult<(Vec<FixedShapeChunkSummary>, Vec<[u8; 32]>)> {
    if packaged.statement.chunks.len() != packaged.proof.session.chunks.len() {
        return Err(invalid_input(
            "packaged public chunks and proof chunks have different lengths",
        ));
    }

    let mut summaries = Vec::with_capacity(packaged.statement.chunks.len());
    let mut transition_digests = Vec::with_capacity(packaged.statement.chunks.len());
    for (chunk_index, (public_chunk, proof_chunk)) in packaged
        .statement
        .chunks
        .iter()
        .zip(packaged.proof.session.chunks.iter())
        .enumerate()
    {
        let public_chunk_digest = digest_fields_as_digest32(public_chunk_digest_fields(public_chunk));
        summaries.push(FixedShapeChunkSummary::from_public_chunk(
            public_chunk,
            public_chunk_digest,
            proof_chunk.relation_digest,
        ));
        transition_digests.push(diagnostic_chunk_transition_digest(
            chunk_index,
            public_chunk,
            proof_chunk,
        ));
    }
    Ok((summaries, transition_digests))
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

fn print_fold_evolution(params: &NeoParams, packaged: &PackagedProof, perf: &RunProvePerf) {
    println!("== protocol evolution by chunk ==");
    let mut incoming_carry = 0usize;
    for (idx, proof_chunk) in packaged.proof.session.chunks.iter().enumerate() {
        let fresh = proof_chunk.chunk.steps.len();
        let pi_ccs_inputs = fresh + incoming_carry;
        let next_carry = proof_chunk.dec.children.len();
        println!(
            "chunk[{idx}] start={} fresh_CCS_K={} incoming_CE_k={} Pi_CCS_inputs_K_plus_k={}",
            proof_chunk.chunk.start_index, fresh, incoming_carry, pi_ccs_inputs
        );
        print_norm_budget("  norm budget", params, fresh, incoming_carry);
        println!(
            "  Pi_CCS out: CE_claims={}, FE_rounds={} max_FE_round_width={}, FE_challenges={}, NC_rounds={} max_NC_round_width={}, NC_challenges={}, header_digest_bytes={}",
            proof_chunk.ccs_outputs.len(),
            proof_chunk.ccs_proof.sumcheck_rounds.len(),
            max_round_width(&proof_chunk.ccs_proof.sumcheck_rounds),
            proof_chunk.ccs_proof.sumcheck_challenges.len(),
            proof_chunk.ccs_proof.sumcheck_rounds_nc.len(),
            max_round_width(&proof_chunk.ccs_proof.sumcheck_rounds_nc),
            proof_chunk.ccs_proof.sumcheck_challenges_nc.len(),
            proof_chunk.ccs_proof.header_digest.len()
        );
        if let Some(first) = proof_chunk.ccs_outputs.first() {
            println!("  first Pi_CCS CE shape: {}", ce_shape_summary(first));
        }
        println!("  Pi_RLC parent CE: {}", ce_shape_summary(&proof_chunk.rlc.parent));
        if let Some(first_child) = proof_chunk.dec.children.first() {
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
        incoming_carry = next_carry;
    }
    println!();
}

fn print_superneo_embedding_diagnostic(
    params: &NeoParams,
    s: &CcsStructure<F>,
    packaged: &PackagedProof,
    first_step: &StepInput,
) -> AppResult<()> {
    let proof_chunk = packaged
        .proof
        .session
        .chunks
        .first()
        .ok_or_else(|| invalid_input("SuperNeo embedding diagnostic requires at least one proof chunk"))?;
    let output = proof_chunk
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
    println!("== verify stage breakdown ==");
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

fn print_spartan_surface_evolution(target: &Spartan2DeciderTarget, shape: &Spartan2DeciderShape) {
    let digest_fields = packed_bytes_field_len(32);
    let chunk_summary_fields = fixed_shape_summary_fields_for_spartan();
    let transition_fields = spartan_transition_binding_fields();
    let statement_base_fields = 3 * digest_fields + 2 * FIXED_SHAPE_DIGEST_FIELD_LEN + 3;
    let statement_chunk_fields = target.statement.chunk_summaries.len() * chunk_summary_fields;
    let witness_base_fields = 2 + target.witness.base_component_digests.len() * digest_fields;
    let witness_transition_fields = target.witness.chunk_transition_bindings.len() * transition_fields;
    println!("== Spartan2 surface expansion ==");
    println!("digest packing: 32 bytes -> {digest_fields} Goldilocks fields using {PACKED_BYTES_PER_LIMB}-byte limbs");
    println!(
        "statement fields: base={} + chunks={}*{}={} => {}",
        statement_base_fields,
        target.statement.chunk_summaries.len(),
        chunk_summary_fields,
        statement_chunk_fields,
        shape.statement_public_io_len()
    );
    println!(
        "witness fields: counters/base={} + chunk_bindings={}*{}={} => {}",
        witness_base_fields,
        target.witness.chunk_transition_bindings.len(),
        transition_fields,
        witness_transition_fields,
        shape.witness_public_io_len()
    );
    println!(
        "backend public fields: statement={} + semantic_digest={} + binding_digest={} => {}",
        shape.statement_public_io_len(),
        FIXED_SHAPE_DIGEST_FIELD_LEN,
        FIXED_SHAPE_DIGEST_FIELD_LEN,
        shape.backend_public_io_len()
    );
    for (idx, summary) in target.statement.chunk_summaries.iter().enumerate() {
        println!(
            "  chunk_summary[{idx}]: start={}, public_steps={}, public_chunk_digest={:02x?}, relation_digest={:02x?}",
            summary.start_index,
            summary.public_step_count,
            &summary.public_chunk_digest[..4],
            &summary.chunk_relation_digest[..4]
        );
    }
    println!();
}

fn print_spartan(params: &NeoParams, s: &CcsStructure<F>, packaged: &PackagedProof) -> AppResult<SpartanRunSummary> {
    let (summaries, transition_digests) = fixed_shape_summaries_and_transition_digests(packaged)?;
    let relation_digest = diagnostic_relation_digest(params, s);
    let initial_handle_digest = digest32_as_fields(fixed_shape_recursive_seed(
        b"neo.fold.next/fibonacci_superneo_probe/initial_handle/v1",
    ));
    let relation = build_spartan2_self_bound_decider_relation(
        packaged.statement.digest,
        relation_digest,
        initial_handle_digest,
        packaged.statement.fold_schedule,
        packaged.statement.public_step_count() as u64,
        summaries,
        vec![packaged.proof.proof_digest],
        transition_digests,
    )?;
    let target = relation.target();
    let shape = target.shape();

    println!("== Spartan2 diagnostic decider surface ==");
    println!("target: generic fixed-shape backend-binding shell");
    println!(
        "note: this binds the packaged direct-CCS run into the public Spartan2 decider surface; it is not a VM recursion circuit"
    );
    println!(
        "fixed-shape chunks={}, semantic_steps={}, base_components={}, chunk_transitions={}",
        target.statement.chunk_summaries.len(),
        target.statement.semantic_step_count,
        target.witness.base_component_digests.len(),
        target.witness.chunk_transition_bindings.len()
    );
    println!(
        "io lengths: public_target={}, backend_public={}, backend_witness={}",
        shape.public_io_len(),
        shape.backend_public_io_len(),
        shape.backend_witness_field_len()
    );
    println!("relation_digest: {:02x?}", relation.digest);
    println!("native packaged proof digest: {:02x?}", packaged.proof.proof_digest);
    println!(
        "self-bound Spartan final proof digest: {:02x?}",
        relation.final_proof_digest
    );
    print_spartan_surface_evolution(&target, &shape);

    let (pk, vk) = setup_spartan2_decider(&shape)?;
    let sizes = pk.backend_shape_sizes();
    let stats = pk.backend_shape_debug_stats();
    println!(
        "backend R1CS sizes [cons, shared, precommitted, rest, padded_cons, padded_shared, padded_precommitted, padded_rest, public, challenges]: {:?}",
        sizes
    );
    println!(
        "backend R1CS nnz: A={}, B={}, C={}, total={}, max_row_total={}",
        stats.a_nnz, stats.b_nnz, stats.c_nnz, stats.total_nnz, stats.max_row_nnz_total
    );
    let semantic_steps = target.statement.semantic_step_count.max(1) as f64;
    let chunks = target.statement.chunk_summaries.len().max(1) as f64;
    println!(
        "backend R1CS scale: constraints_per_semantic_step={:.1}, constraints_per_chunk={:.1}",
        sizes[0] as f64 / semantic_steps,
        sizes[0] as f64 / chunks
    );
    let padding_rows = sizes[4].saturating_sub(sizes[0]);
    let padding_pct = if sizes[0] == 0 {
        0.0
    } else {
        100.0 * padding_rows as f64 / sizes[0] as f64
    };
    println!(
        "backend R1CS padding: actual_constraints={}, padded_constraints={}, padding_rows={}, padding_over_actual_pct={:.1}",
        sizes[0], sizes[4], padding_rows, padding_pct
    );
    println!("shape_digest: {:02x?}", pk.shape_digest());

    println!("== Spartan2 prove/verify ==");
    let (proof, perf) = prove_spartan2_decider_with_perf(&pk, &target)?;
    println!(
        "prove ms: relation_surface={:.3}, prep={:.3}, snark={:.3}, encode={:.3}, total={:.3}",
        perf.relation_surface_ms,
        perf.shell.prep_ms,
        perf.shell.snark_perf.total_ms,
        perf.shell.encode_ms,
        perf.total_ms
    );
    println!(
        "prove internals ms: tau={:.3}, matvec={:.3}, multilinears={:.3}, outer_sumcheck={:.3}, inner_claims={:.3}, eval_rx={:.3}, eval_table={:.3}, poly_abc={:.3}, poly_z={:.3}, inner_sumcheck={:.3}, pcs={:.3}",
        perf.shell.snark_perf.prepare_poly_tau_ms,
        perf.shell.snark_perf.matrix_vector_multiply_ms,
        perf.shell.snark_perf.prepare_multilinear_polys_ms,
        perf.shell.snark_perf.outer_sumcheck_ms,
        perf.shell.snark_perf.prepare_inner_claims_ms,
        perf.shell.snark_perf.compute_eval_rx_ms,
        perf.shell.snark_perf.compute_eval_table_sparse_ms,
        perf.shell.snark_perf.prepare_poly_abc_ms,
        perf.shell.snark_perf.prepare_poly_z_ms,
        perf.shell.snark_perf.inner_sumcheck_ms,
        perf.shell.snark_perf.pcs_prove_ms
    );
    let final_proof_bytes = bincode::serialize(&proof)?.len();
    let snark_bytes = proof.snark_bytes_len();
    println!(
        "proof bytes: final_serialized={}, snark_data={}, proof_digest={:02x?}",
        final_proof_bytes,
        snark_bytes,
        proof.digest()
    );
    println!(
        "proof bytes breakdown: snark_data={}, wrapper_overhead={}",
        snark_bytes,
        final_proof_bytes.saturating_sub(snark_bytes)
    );
    let verify_started = std::time::Instant::now();
    verify_spartan2_decider(&vk, &target, &proof)?;
    let verify_ms = verify_started.elapsed().as_secs_f64() * 1_000.0;
    println!("verify: ok");
    println!();
    Ok(SpartanRunSummary {
        prove_ms: perf.total_ms,
        verify_ms,
        final_proof_bytes,
        snark_bytes,
        backend_r1cs_constraints: sizes[0],
        padded_backend_r1cs_constraints: sizes[4],
        backend_public_inputs: sizes[8],
        backend_challenges: sizes[9],
        backend_nnz_total: stats.total_nnz,
        spartan_pcs_ms: perf.shell.snark_perf.pcs_prove_ms,
    })
}

fn print_final_summary(prove_perf: &RunProvePerf, spartan: SpartanRunSummary) {
    let chunk_folds = prove_perf.chunk_count();
    let fresh_steps = prove_perf.fresh_steps();
    let ms_per_chunk_fold = prove_perf.total_ms / chunk_folds.max(1) as f64;
    let ms_per_fresh_step = prove_perf.total_ms / fresh_steps.max(1) as f64;

    println!("== final summary ==");
    println!("proving (before spartan): {:.3} ms", prove_perf.total_ms);
    println!(
        "  number of folds: {} SuperNeo chunk fold(s) over {} fresh CCS step(s)",
        chunk_folds, fresh_steps
    );
    println!(
        "  time per fold: {:.3} ms/chunk fold ({:.3} ms/fresh CCS step)",
        ms_per_chunk_fold, ms_per_fresh_step
    );
    println!("proving (spartan): {:.3} ms", spartan.prove_ms);
    println!("proving (total): {:.3} ms", prove_perf.total_ms + spartan.prove_ms);
    println!("verifying (final proof): {:.3} ms", spartan.verify_ms);
    println!(
        "constraints passed to Spartan2: {} backend R1CS constraints (padded to {})",
        spartan.backend_r1cs_constraints, spartan.padded_backend_r1cs_constraints
    );
    println!(
        "size final proof: {} bytes (snark_data={}, wrapper_overhead={})",
        spartan.final_proof_bytes,
        spartan.snark_bytes,
        spartan
            .final_proof_bytes
            .saturating_sub(spartan.snark_bytes)
    );
    println!();
}

fn print_optimization_ranking(prove_perf: &RunProvePerf, spartan: SpartanRunSummary) {
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
        ("Spartan PCS", spartan.spartan_pcs_ms),
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

fn print_constraint_breakdown(s: &CcsStructure<F>, prove_perf: &RunProvePerf, spartan: SpartanRunSummary) {
    let fresh = prove_perf.fresh_steps();
    let total_ccs_rows = s.n.saturating_mul(fresh);
    let padding_rows = spartan
        .padded_backend_r1cs_constraints
        .saturating_sub(spartan.backend_r1cs_constraints);
    let padding_pct = if spartan.backend_r1cs_constraints == 0 {
        0.0
    } else {
        100.0 * padding_rows as f64 / spartan.backend_r1cs_constraints as f64
    };
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
    println!(
        "  backend R1CS constraints passed: {}",
        spartan.backend_r1cs_constraints
    );
    println!(
        "  padded backend constraints: {}",
        spartan.padded_backend_r1cs_constraints
    );
    println!("  padding rows: {padding_rows} ({padding_pct:.1}% over actual)");
    println!(
        "  backend public inputs={}, challenges={}, nnz_total={}",
        spartan.backend_public_inputs, spartan.backend_challenges, spartan.backend_nnz_total
    );
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
    let (packaged, prove_perf) = prove_and_package_with_perf(
        FoldingMode::Optimized,
        schedule,
        &params,
        &ccs,
        steps,
        &log,
        ajtai_mixers(),
    )?;
    print_chunk_prove_perf(&prove_perf);
    print_prove_timing_accounting(&prove_perf);
    print_fold_evolution(&params, &packaged, &prove_perf);
    print_superneo_embedding_diagnostic(&params, &ccs, &packaged, &first_step_for_embedding_check)?;

    let (verified_claims, verify_perf) =
        verify_packaged_with_perf(FoldingMode::Optimized, &params, &ccs, &packaged, ajtai_mixers())?;
    if verified_claims != packaged.statement.final_main_claims {
        return Err(invalid_input("verified final claims did not match packaged statement"));
    }
    if public_steps.len() != packaged.statement.public_step_count() {
        return Err(invalid_input("public step count changed during packaging"));
    }
    print_verify_perf(&verify_perf);
    let spartan_summary = print_spartan(&params, &ccs, &packaged)?;
    print_optimization_ranking(&prove_perf, spartan_summary);
    print_folding_timing_table(&prove_perf, ccs.n);
    print_constraint_breakdown(&ccs, &prove_perf, spartan_summary);
    print_final_summary(&prove_perf, spartan_summary);

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
