use std::env;
use std::error::Error;

use bellpepper::gadgets::boolean::{AllocatedBit, Boolean};
use bellpepper_core::{Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use ff::{Field, PrimeField};
use neo_ajtai::{s_mul_add, scale_commitment_add_inplace, set_global_pp_seeded, AjtaiSModule, Commitment};
use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsMatrix, CcsStructure, CcsWitness, CeClaim, CscMat, Mat};
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
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::common::{ct_from_y_ring_for_ccs_m, decode_superneo_coeffs_from_witness_mat, RotRing};
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices};
use neo_reductions::superneo_eval::{build_superneo_eval_cache, eval_all_mats_ring_cached};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};

type AppResult<T> = Result<T, Box<dyn Error>>;

const FIXED_SHAPE_DIGEST_FIELD_LEN: usize = 4;
const PACKED_BYTES_PER_LIMB: usize = 7;
const AUX_FLAG: u32 = 1 << 31;

#[derive(Clone, Copy, Debug)]
struct Config {
    preimage_bytes: usize,
    steps: usize,
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
    spartan_pcs_ms: f64,
}

#[derive(Clone, Copy, Debug)]
struct Sha256CircuitShape {
    constraints: usize,
    inputs: usize,
    aux: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            preimage_bytes: 3,
            steps: 1,
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
            if let Some(raw) = arg.strip_prefix("--preimage-bytes=") {
                config.preimage_bytes = parse_nonzero_usize("--preimage-bytes", raw)?;
                continue;
            }
            if let Some(raw) = arg.strip_prefix("--steps=") {
                config.steps = parse_nonzero_usize("--steps", raw)?;
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
        self.steps.div_ceil(self.rows_per_chunk)
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
    println!("sha256_superneo_probe");
    println!("Direct SHA256 Bellpepper->CCS diagnostic for the generic SuperNeo spine.");
    println!("No VM frontend and no RV64IM relation are used.");
    println!();
    println!("Options:");
    println!("  --preimage-bytes=N   SHA256 preimage length in bytes [default: 3]");
    println!("  --steps=N            number of repeated SHA256 CCS steps to fold [default: 1]");
    println!("  --rows-per-chunk=N   generated SHA256 steps per SuperNeo chunk [default: 1]");
}

fn print_paper_stage_map() {
    println!("== SuperNeo paper stage map ==");
    println!("Bellpepper SHA256 -> sparse R1CS -> CCS with f(Az,Bz,Cz)=Az*Bz-Cz");
    println!("section 5: field rows are embedded as ring coefficients; Mz checks become evaluated ring claims");
    println!("section 7.3 Pi_CCS: K fresh CCS rows + k carried CE claims -> K+k CE claims");
    println!("section 7.4 Pi_RLC: K+k CE claims -> one random-linear-combination parent CE claim");
    println!("section 7.5 Pi_DEC: one large-norm parent CE claim -> k_rho small-norm CE children");
    println!("Spartan2: proves the fixed-shape public/backend binding shell over the packaged run");
    println!();
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn sha256_preimage_for_len(preimage_len_bytes: usize) -> Vec<u8> {
    let mut preimage = vec![0u8; preimage_len_bytes];
    let pattern = b"abc";
    for (idx, byte) in preimage.iter_mut().enumerate() {
        *byte = pattern[idx % pattern.len()];
    }
    preimage
}

fn sha256_digest(preimage: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(preimage);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn expected_sha256_public_inputs_from_digest(digest: &[u8; 32]) -> Vec<F> {
    let digest_bits = bellpepper::gadgets::multipack::bytes_to_bits(digest);
    let mut out = Vec::with_capacity(1 + digest_bits.len());
    out.push(F::ONE);
    out.extend(
        digest_bits
            .into_iter()
            .map(|bit| if bit { F::ONE } else { F::ZERO }),
    );
    out
}

fn check_sha256_public_inputs_match_digest(digest: &[u8; 32], public_inputs: &[F]) -> AppResult<()> {
    let expected = expected_sha256_public_inputs_from_digest(digest);
    debug_assert_eq!(
        public_inputs, expected,
        "Bellpepper SHA256 public output bits must match real SHA256 digest"
    );
    if public_inputs != expected {
        return Err(invalid_input(
            "Bellpepper SHA256 public output bits do not match real SHA256 digest",
        ));
    }
    Ok(())
}

#[derive(PrimeField)]
#[PrimeFieldModulus = "18446744069414584321"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
struct FpGoldilocks([u64; 2]);

fn fp_to_u64(x: &FpGoldilocks) -> u64 {
    let bytes = x.to_repr();
    u64::from_le_bytes(bytes.0[0..8].try_into().expect("repr is at least 8 bytes"))
}

struct Sha256Circuit {
    preimage: Vec<u8>,
}

impl Circuit<FpGoldilocks> for Sha256Circuit {
    fn synthesize<CS: ConstraintSystem<FpGoldilocks>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let bit_values = bellpepper::gadgets::multipack::bytes_to_bits(&self.preimage)
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
        let preimage_bits = bit_values
            .into_iter()
            .enumerate()
            .map(|(idx, bit)| AllocatedBit::alloc(cs.namespace(|| format!("preimage_bit_{idx}")), bit))
            .map(|bit| bit.map(Boolean::from))
            .collect::<Result<Vec<_>, _>>()?;

        let hash_bits = bellpepper::gadgets::sha256::sha256(cs.namespace(|| "sha256"), &preimage_bits)?;
        for (bit_idx, bit) in hash_bits.iter().enumerate() {
            let value = bit
                .get_value()
                .ok_or(SynthesisError::AssignmentMissing)
                .map(|bit| if bit { FpGoldilocks::ONE } else { FpGoldilocks::ZERO })?;
            let input = cs.alloc_input(|| format!("hash_out_bit_{bit_idx}"), || Ok(value))?;
            cs.enforce(
                || format!("hash_out_bit_match_{bit_idx}"),
                |_| bit.lc(CS::one(), FpGoldilocks::ONE),
                |lc| lc + CS::one(),
                |lc| lc + input,
            );
        }
        Ok(())
    }
}

struct TripletConstraintSystem {
    inputs: Vec<F>,
    aux: Vec<F>,
    num_constraints: u32,
    a_trips: Vec<(u32, u32, F)>,
    b_trips: Vec<(u32, u32, F)>,
    c_trips: Vec<(u32, u32, F)>,
}

impl TripletConstraintSystem {
    fn new() -> Self {
        Self {
            inputs: vec![F::ONE],
            aux: Vec::new(),
            num_constraints: 0,
            a_trips: Vec::new(),
            b_trips: Vec::new(),
            c_trips: Vec::new(),
        }
    }

    fn push_lc_trips(row: u32, lc: &LinearCombination<FpGoldilocks>, trips: &mut Vec<(u32, u32, F)>) {
        for (var, coeff) in lc.iter() {
            let value = fp_to_u64(coeff);
            if value == 0 {
                continue;
            }
            let col = match var.0 {
                Index::Input(idx) => u32::try_from(idx).expect("input index fits u32"),
                Index::Aux(idx) => AUX_FLAG | u32::try_from(idx).expect("aux index fits u32"),
            };
            trips.push((row, col, F::from_u64(value)));
        }
    }

    fn resolve_triplets(trips: Vec<(u32, u32, F)>, num_inputs: usize) -> Vec<(usize, usize, F)> {
        trips
            .into_iter()
            .map(|(row, col, value)| {
                let row = row as usize;
                if (col & AUX_FLAG) == 0 {
                    (row, col as usize, value)
                } else {
                    let aux_idx = (col & !AUX_FLAG) as usize;
                    (row, num_inputs + aux_idx, value)
                }
            })
            .collect()
    }
}

impl ConstraintSystem<FpGoldilocks> for TripletConstraintSystem {
    type Root = Self;

    fn new() -> Self {
        Self::new()
    }

    fn alloc<FN, A, AR>(&mut self, _annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<FpGoldilocks, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let idx = self.aux.len();
        self.aux.push(F::from_u64(fp_to_u64(&f()?)));
        Ok(Variable::new_unchecked(Index::Aux(idx)))
    }

    fn alloc_input<FN, A, AR>(&mut self, _annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<FpGoldilocks, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let idx = self.inputs.len();
        self.inputs.push(F::from_u64(fp_to_u64(&f()?)));
        Ok(Variable::new_unchecked(Index::Input(idx)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<FpGoldilocks>) -> LinearCombination<FpGoldilocks>,
        LB: FnOnce(LinearCombination<FpGoldilocks>) -> LinearCombination<FpGoldilocks>,
        LC: FnOnce(LinearCombination<FpGoldilocks>) -> LinearCombination<FpGoldilocks>,
    {
        let row = self.num_constraints;
        self.num_constraints += 1;
        Self::push_lc_trips(row, &a(LinearCombination::zero()), &mut self.a_trips);
        Self::push_lc_trips(row, &b(LinearCombination::zero()), &mut self.b_trips);
        Self::push_lc_trips(row, &c(LinearCombination::zero()), &mut self.c_trips);
    }

    fn push_namespace<NR, N>(&mut self, _name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self) {}

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

fn bellpepper_sha256_ccs(preimage: &[u8]) -> AppResult<(CcsStructure<F>, Vec<F>, Sha256CircuitShape)> {
    let mut cs = TripletConstraintSystem::new();
    Sha256Circuit {
        preimage: preimage.to_vec(),
    }
    .synthesize(&mut cs)
    .map_err(|err| invalid_input(format!("SHA256 Bellpepper synthesis failed: {err:?}")))?;

    let TripletConstraintSystem {
        inputs,
        aux,
        num_constraints,
        a_trips,
        b_trips,
        c_trips,
    } = cs;
    let num_constraints = num_constraints as usize;
    let num_inputs = inputs.len();
    let num_aux = aux.len();
    let num_variables = num_inputs + num_aux;
    let mut witness = inputs;
    witness.extend(aux);
    let a = CcsMatrix::Csc(CscMat::from_triplets(
        TripletConstraintSystem::resolve_triplets(a_trips, num_inputs),
        num_constraints,
        num_variables,
    ));
    let b = CcsMatrix::Csc(CscMat::from_triplets(
        TripletConstraintSystem::resolve_triplets(b_trips, num_inputs),
        num_constraints,
        num_variables,
    ));
    let c = CcsMatrix::Csc(CscMat::from_triplets(
        TripletConstraintSystem::resolve_triplets(c_trips, num_inputs),
        num_constraints,
        num_variables,
    ));
    let ccs = neo_ccs::sparse_r1cs_to_ccs(a, b, c)?;
    Ok((
        ccs,
        witness,
        Sha256CircuitShape {
            constraints: num_constraints,
            inputs: num_inputs,
            aux: num_aux,
        },
    ))
}

fn pack_witness_mat(z: &[F], m: usize) -> Mat<F> {
    let mut z_mat = Mat::zero(D, m.div_ceil(D), F::ZERO);
    for idx in 0..m {
        z_mat[(idx % D, idx / D)] = z[idx];
    }
    z_mat
}

fn sha256_step(log: &AjtaiSModule, label: &str, ccs: &CcsStructure<F>, witness: &[F], m_in: usize) -> StepInput {
    let z_mat = pack_witness_mat(witness, ccs.m);
    StepInput {
        label: label.to_string(),
        mcs: CcsClaim {
            c: log.commit(&z_mat),
            x: witness[..m_in].to_vec(),
            m_in,
        },
        witness: CcsWitness {
            w: witness[m_in..].to_vec(),
            Z: z_mat,
        },
    }
}

fn make_ajtai_module(params: &NeoParams, witness_cols: usize) -> AppResult<AjtaiSModule> {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0x5348_4132_3536_5f50_u64.to_le_bytes());
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
    extend_packed_bytes_as_fields(step_scratch, b"neo.fold.next/finalize/public_step_digest/v1");
    extend_packed_bytes_as_fields(step_scratch, step.label.as_bytes());
    step_scratch.extend_from_slice(&ccs_claim_digest_fields_into(&step.mcs, claim_scratch));
    poseidon_digest_fields(step_scratch)
}

fn public_chunk_digest_fields(chunk: &PublicChunk) -> [F; 4] {
    let mut claim_scratch = Vec::<F>::with_capacity(256);
    let mut step_scratch = Vec::<F>::with_capacity(96);
    let mut chunk_scratch = Vec::<F>::new();
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
    let repo_profile_match = params.d as usize == D
        && params.eta == 81
        && params.kappa == 16
        && params.k_rho == 12
        && params.B == 4096
        && params.T == 216
        && params.s == 2;
    let steady_max_fresh = max_fresh_k_for_incoming(params, params.k_rho as usize);
    let conservative_terms = ((dims.ell + dims.ell_nc) * s.max_degree().max(1) as usize).max(1);
    let conservative_sumcheck_bits = 64.0 * params.s as f64 - (conservative_terms as f64).log2();

    println!("== parameter/security audit ==");
    println!(
        "field=Goldilocks q_bits=64 extension_s={} extension_field_bits~{} ring_degree_d={} cyclotomic=X^54 + X^27 + 1",
        params.s,
        64 * params.s,
        D
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
        "repo_goldilocks_profile_match={} (expected d=54, eta=81, kappa=16, k_dec=12, B=4096, T=216, s=2)",
        if repo_profile_match { "yes" } else { "no" }
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
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/sha256_superneo_probe/relation");
    tr.append_message(b"neo.fold.next/sha256_superneo_probe/relation/version", b"v1");
    tr.append_u64s(
        b"neo.fold.next/sha256_superneo_probe/relation/params",
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
        b"neo.fold.next/sha256_superneo_probe/relation/ccs_shape",
        &[s.n as u64, s.m as u64, s.t() as u64, s.max_degree() as u64],
    );
    tr.append_fields(
        b"neo.fold.next/sha256_superneo_probe/relation/ccs_matrix_digest",
        &digest_ccs_matrices(s),
    );
    tr.digest32()
}

fn diagnostic_chunk_transition_digest(
    chunk_index: usize,
    public_chunk: &PublicChunk,
    proof_chunk: &neo_fold_next::proof::ChunkProof,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/sha256_superneo_probe/chunk_transition");
    tr.append_message(b"neo.fold.next/sha256_superneo_probe/chunk_transition/version", b"v1");
    tr.append_u64s(
        b"neo.fold.next/sha256_superneo_probe/chunk_transition/meta",
        &[
            chunk_index as u64,
            public_chunk.start_index as u64,
            public_chunk.steps.len() as u64,
            proof_chunk.ccs_outputs.len() as u64,
            proof_chunk.dec.children.len() as u64,
        ],
    );
    tr.append_message(
        b"neo.fold.next/sha256_superneo_probe/chunk_transition/pi_ccs_header_digest",
        &proof_chunk.ccs_proof.header_digest,
    );
    tr.append_message(
        b"neo.fold.next/sha256_superneo_probe/chunk_transition/chunk_relation_digest",
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
        summaries.push(FixedShapeChunkSummary::from_public_chunk(
            public_chunk,
            digest_fields_as_digest32(public_chunk_digest_fields(public_chunk)),
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

fn print_shape(config: Config, params: &NeoParams, s: &CcsStructure<F>, shape: Sha256CircuitShape, digest: [u8; 32]) {
    let dims = build_dims_and_policy(params, s).expect("valid dims");
    println!("== direct SHA256 CCS ==");
    println!(
        "preimage: {} bytes; repeated fold steps={}; digest={}",
        config.preimage_bytes,
        config.steps,
        hex_lower(&digest)
    );
    println!(
        "bellpepper R1CS: constraints={}, inputs={}, aux={}, variables={}",
        shape.constraints,
        shape.inputs,
        shape.aux,
        shape.inputs + shape.aux
    );
    println!(
        "ccs: n={} rows, m={} columns, t={} matrices, degree={}, matrix_nnz={}",
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
}

fn print_steps(config: Config, step: &StepInput) {
    println!("== generated SHA256 fold steps ==");
    println!(
        "public input fields per step: {}; first field={}, digest_bit_inputs={}",
        step.mcs.x.len(),
        step.mcs.x[0].as_canonical_u64(),
        step.mcs.x.len().saturating_sub(1)
    );
    for idx in 0..config.steps {
        let chunk = idx / config.rows_per_chunk;
        let row_in_chunk = idx % config.rows_per_chunk;
        println!("step[{idx}]: chunk={chunk}, row_in_chunk={row_in_chunk}, label=sha256_step_{idx}");
    }
    println!();
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

fn print_verify_perf(perf: &RunVerifyPerf) {
    println!("== native packaged verify diagnostic ==");
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
    println!();
}

fn print_spartan(params: &NeoParams, s: &CcsStructure<F>, packaged: &PackagedProof) -> AppResult<SpartanRunSummary> {
    let (summaries, transition_digests) = fixed_shape_summaries_and_transition_digests(packaged)?;
    let relation_digest = diagnostic_relation_digest(params, s);
    let initial_handle_digest = digest32_as_fields(fixed_shape_recursive_seed(
        b"neo.fold.next/sha256_superneo_probe/initial_handle/v1",
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
        "prove internals ms: outer_sumcheck={:.3}, inner_sumcheck={:.3}, pcs={:.3}",
        perf.shell.snark_perf.outer_sumcheck_ms,
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
    print_optimization_ranking(prove_perf, spartan);
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
    println!("optimization ranking (measured):");
    for (rank, (label, ms)) in items.into_iter().take(5).enumerate() {
        println!("  {}. {}: {:.3} ms", rank + 1, label, ms);
    }
}

fn run() -> AppResult<()> {
    let Some(config) = Config::from_args()? else {
        print_usage();
        return Ok(());
    };

    let preimage = sha256_preimage_for_len(config.preimage_bytes);
    let digest = sha256_digest(&preimage);
    let build_started = std::time::Instant::now();
    let (ccs, witness, sha_shape) = bellpepper_sha256_ccs(&preimage)?;
    let build_ms = build_started.elapsed().as_secs_f64() * 1_000.0;
    check_sha256_public_inputs_match_digest(&digest, &witness[..sha_shape.inputs])?;
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n)?;
    let log = make_ajtai_module(&params, ccs.m.div_ceil(D))?;
    let base_step = sha256_step(&log, "sha256_step_0", &ccs, &witness, sha_shape.inputs);
    let steps = (0..config.steps)
        .map(|idx| {
            let mut step = base_step.clone();
            step.label = format!("sha256_step_{idx}");
            step
        })
        .collect::<Vec<_>>();
    let first_step_for_embedding_check = steps[0].clone();

    print_shape(config, &params, &ccs, sha_shape, digest);
    println!(
        "bellpepper->CCS build (not counted in final summary): {:.3} ms",
        build_ms
    );
    println!();
    print_ccs_sparse_matrix_diagnostic(&ccs);
    print_parameter_security_audit(&params, &ccs, config.steps, config.rows_per_chunk);
    print_paper_stage_map();
    print_steps(config, &base_step);

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
    print_final_summary(&prove_perf, spartan_summary);
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
