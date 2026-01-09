//! WHIR (Plonky3-based) backend glue for `ClosureProofV1::OpaqueBytes`.
//!
//! This module contains multiple incremental backends:
//! - a small deterministic **placeholder** proof (plumbing-only), and
//! - dev-milestone backends that prove *partial* obligation-closure predicates (e.g. Ajtai opening
//!   correctness, and optionally X-projection).
//!
//! The `FULL_CLOSURE` backend in this file *does* prove the full closure predicate
//! (Ajtai opening + bounds + ME consistency), but it is currently a **dev milestone**:
//! it materializes large evaluation tables in memory and encodes explicit obligations in the
//! payload, so it is not production-sized for large `(d, m)` or many obligations.

#![forbid(unsafe_code)]

mod seed;
mod sumcheck_math;

use crate::{contract, opaque, ClosureProofError, ClosureStatementV1};
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use seed::{derive_seed_v1, fixed_seed};

use neo_ajtai::Commitment as NeoCmt;
use neo_math::{F as NeoF, D as NeoD, KExtensions};
use p3_field::{PrimeCharacteristicRing as _, PrimeField64 as _};

use p3_challenger::DuplexChallenger;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

use crate::bounded::BoundedVec;
use crate::codec::{deserialize_payload, serialize_payload};

use whir_p3::{
    dft::EvalsDft,
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{errors::SecurityAssumption, FoldingFactor, ProtocolParameters},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::{reader::CommitmentReader, writer::CommitmentWriter},
        constraints::statement::Statement,
        parameters::WhirConfig,
        prover::Prover,
        verifier::Verifier,
    },
};

const WHIR_P3_NUM_VARIABLES: usize = 8;
const WHIR_P3_NUM_POINTS: usize = 2;
const WHIR_P3_DIGEST_ELEMS: usize = 8;
// Production security profile (matches the ~128-bit target in the Neo paper).
#[cfg(not(feature = "whir-p3-dev"))]
const WHIR_SECURITY_LEVEL_BITS: usize = 128;
// Dev/test profile (faster; NOT production-grade).
#[cfg(feature = "whir-p3-dev")]
const WHIR_SECURITY_LEVEL_BITS: usize = 32;
const MAX_WHIR_PROOF_DATA_U64: usize = 1 << 20; // 1M u64 limbs (~8 MiB)
const MAX_SUMCHECK_ROUNDS: usize = 64;
const MAX_SUMCHECK_EVALS_U64_PER_ROUND: usize = 1 << 14; // 16k u64 limbs

/// Hard safety limit for the current WHIR backends.
///
/// These backends materialize `Z` evaluations and public weight tables in memory. At production
/// sizes (e.g. `m=2^24`), this is not viable and must be replaced by a streaming backend.
const MAX_DEV_Z_EVALS_PADDED: usize = 1 << 22; // 4M

type F = Goldilocks;
type EF = F;
type Perm = Poseidon2Goldilocks<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct WhirP3PlaceholderPayloadV1 {
    /// WHIR prover transcript, as canonical u64 limbs of base-field elements.
    proof_data_u64: BoundedVec<u64, MAX_WHIR_PROOF_DATA_U64>,
}

fn digest_to_small_field_elems(d: &[u8; 32]) -> Vec<F> {
    // Map digest bytes to field elements using 32-bit chunks (always < modulus).
    let mut out = Vec::with_capacity(8);
    for chunk in d.chunks_exact(4) {
        let mut limb = [0u8; 4];
        limb.copy_from_slice(chunk);
        out.push(F::from_u64(u64::from_le_bytes([
            limb[0], limb[1], limb[2], limb[3], 0, 0, 0, 0,
        ])));
    }
    out
}

fn domain_separator_for_stmt(params: &WhirConfig<EF, F, MyHash, MyCompress, MyChallenger>, stmt: &ClosureStatementV1) -> DomainSeparator<EF, F> {
    let mut pattern = Vec::new();
    // Bind the backend + version.
    for &b in b"neo/closure-proof/whir-p3/opaque/v1" {
        pattern.push(F::from_u64(b as u64));
    }
    // Bind the statement digests (as small field elements).
    pattern.extend(digest_to_small_field_elems(&stmt.context_digest));
    pattern.extend(digest_to_small_field_elems(&stmt.pp_id_digest));
    pattern.extend(digest_to_small_field_elems(&stmt.obligations_digest));

    let mut ds = DomainSeparator::new(pattern);
    ds.commit_statement::<_, _, _, WHIR_P3_DIGEST_ELEMS>(params);
    ds.add_whir_proof::<_, _, _, WHIR_P3_DIGEST_ELEMS>(params);
    ds
}

fn make_params(num_variables: usize) -> WhirConfig<EF, F, MyHash, MyCompress, MyChallenger> {
    let mut rng = ChaCha8Rng::from_seed(fixed_seed(b"merkle_perm"));
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);

    let whir_params = ProtocolParameters {
        initial_statement: true,
        security_level: WHIR_SECURITY_LEVEL_BITS,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(2),
        merkle_hash,
        merkle_compress,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        univariate_skip: false,
    };

    WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params)
}

fn make_statement(stmt: &ClosureStatementV1, polynomial: &EvaluationsList<F>) -> Statement<EF> {
    let mut rng = ChaCha8Rng::from_seed(derive_seed_v1(b"points", stmt, None));
    let mut statement = Statement::<EF>::initialize(WHIR_P3_NUM_VARIABLES);

    for _ in 0..WHIR_P3_NUM_POINTS {
        let coords = (0..WHIR_P3_NUM_VARIABLES)
            .map(|_| EF::from_u64(rng.next_u64()))
            .collect::<Vec<_>>();
        let point = MultilinearPoint::new(coords);
        statement.add_unevaluated_constraint(point, polynomial);
    }

    statement
}

fn encode_proof_data(proof_data: &[F]) -> Vec<u64> {
    proof_data.iter().map(|x| x.as_canonical_u64()).collect()
}

fn decode_proof_data_u64(u64s: &[u64]) -> Vec<F> {
    u64s.iter().copied().map(F::from_u64).collect()
}

fn extract_commitment_root_u64_from_proof_data(
    params: &WhirConfig<EF, F, MyHash, MyCompress, MyChallenger>,
    domainsep: &DomainSeparator<EF, F>,
    challenger: MyChallenger,
    proof_data: &[F],
) -> Result<Vec<u64>, ClosureProofError> {
    let commitment_reader = CommitmentReader::new(params);
    let mut verifier_state = domainsep.to_verifier_state(proof_data.to_vec(), challenger);
    let parsed_commitment = commitment_reader
        .parse_commitment::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state)
        .map_err(|e| ClosureProofError::WhirP3(format!("parse_commitment failed: {e:?}")))?;
    Ok(parsed_commitment
        .root
        .as_ref()
        .iter()
        .map(|x| x.as_canonical_u64())
        .collect())
}

fn neo_f_to_whir(x: NeoF) -> F {
    x
}

fn whir_f_to_u64(x: F) -> u64 {
    x.as_canonical_u64()
}

fn u64_to_whir_f(x: u64) -> F {
    F::from_u64(x)
}

pub fn prove_whir_p3_placeholder_bytes_v1(stmt: &ClosureStatementV1) -> Vec<u8> {
    // Deterministically derive the polynomial from the statement digest so prover+verifier agree.
    let mut rng = ChaCha8Rng::from_seed(derive_seed_v1(b"poly", stmt, None));
    let num_evals = 1usize << WHIR_P3_NUM_VARIABLES;
    let polynomial = EvaluationsList::new(
        (0..num_evals)
            .map(|_| F::from_u64(rng.next_u64()))
            .collect::<Vec<_>>(),
    );

    let params = make_params(WHIR_P3_NUM_VARIABLES);
    let statement = make_statement(stmt, &polynomial);

    let domainsep = domain_separator_for_stmt(&params, stmt);

    // Challenger is separate from the Merkle permutation in WHIR's examples.
    let mut rng = ChaCha8Rng::from_seed(fixed_seed(b"challenger_perm"));
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    let committer = CommitmentWriter::new(&params);
    let dft_committer = EvalsDft::<F>::default();
    let witness = committer
        .commit::<WHIR_P3_DIGEST_ELEMS>(&dft_committer, &mut prover_state, polynomial)
        .expect("WHIR commit must succeed in placeholder mode");

    let prover = Prover(&params);
    let dft_prover = EvalsDft::<F>::default();
    prover
        .prove::<WHIR_P3_DIGEST_ELEMS>(&dft_prover, &mut prover_state, statement, witness)
        .expect("WHIR prove must succeed in placeholder mode");

    let payload = WhirP3PlaceholderPayloadV1 {
        proof_data_u64: encode_proof_data(prover_state.proof_data()).into(),
    };
    let payload_bytes = serialize_payload(&payload).expect("bincode serialize must succeed");
    opaque::encode_envelope(opaque::BACKEND_ID_WHIR_P3_PLACEHOLDER_V1, &payload_bytes)
}

pub fn verify_whir_p3_placeholder_payload_v1(
    stmt: &ClosureStatementV1,
    payload_bytes: &[u8],
) -> Result<(), ClosureProofError> {
    let payload: WhirP3PlaceholderPayloadV1 = deserialize_payload(payload_bytes)?;

    // Reconstruct deterministic polynomial + statement.
    let mut rng = ChaCha8Rng::from_seed(derive_seed_v1(b"poly", stmt, None));
    let num_evals = 1usize << WHIR_P3_NUM_VARIABLES;
    let polynomial = EvaluationsList::new(
        (0..num_evals)
            .map(|_| F::from_u64(rng.next_u64()))
            .collect::<Vec<_>>(),
    );

    let params = make_params(WHIR_P3_NUM_VARIABLES);
    let statement = make_statement(stmt, &polynomial);
    let domainsep = domain_separator_for_stmt(&params, stmt);

    let mut rng = ChaCha8Rng::from_seed(fixed_seed(b"challenger_perm"));
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

    let commitment_reader = CommitmentReader::new(&params);
    let verifier = Verifier::new(&params);

    let proof_data = decode_proof_data_u64(&payload.proof_data_u64);
    let mut verifier_state = domainsep.to_verifier_state(proof_data, challenger);

    let parsed_commitment = commitment_reader
        .parse_commitment::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state)
        .map_err(|e| ClosureProofError::WhirP3(format!("parse_commitment failed: {e:?}")))?;

    verifier
        .verify::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state, &parsed_commitment, &statement)
        .map_err(|e| ClosureProofError::WhirP3(format!("verify failed: {e:?}")))?;

    Ok(())
}

// -------------------------------------------------------------------------------------------------
// Ajtai opening backends (dev milestones)
// -------------------------------------------------------------------------------------------------
//
// This backend is an incremental step toward a real obligation-closure proof. It proves (via
// a batched random linear check) that Ajtai commitments open correctly to *some* witness matrices Z
// (one per obligation).
//
// Limitations:
// - It does NOT yet prove boundedness (range) of Z.
// - It does NOT yet prove ME y/r/y_scalars consistency for obligations.
//
// Those require a significantly more expressive proof of computation (e.g., GKR/STARK-style).

use crate::encoded::EncodedObligations;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct WhirP3AjtaiOpeningPayloadV1 {
    obligations: EncodedObligations,
    sumcheck: SumcheckProofV1,
    /// WHIR transcript/proof data as canonical u64 limbs of WHIR-field elements.
    whir_proof_data_u64: BoundedVec<u64, MAX_WHIR_PROOF_DATA_U64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SumcheckProofV1 {
    round_polys_u64: BoundedVec<[u64; 3], MAX_SUMCHECK_ROUNDS>,
    z_r_u64: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SumcheckProofV2 {
    /// For each round, the prover sends g(0), g(1), ..., g(deg) as canonical u64 limbs.
    ///
    /// The degree depends on the backend:
    /// - opening-only / opening+X uses `deg=2` (quadratic),
    /// - full closure mixes in a range-check term and uses `deg=2*b`.
    round_evals_u64: BoundedVec<
        BoundedVec<u64, MAX_SUMCHECK_EVALS_U64_PER_ROUND>,
        MAX_SUMCHECK_ROUNDS,
    >,
    z_r_u64: u64,
}

fn next_pow2_checked(n: usize) -> Result<usize, ClosureProofError> {
    n.checked_next_power_of_two().ok_or_else(|| {
        ClosureProofError::WhirP3("next_pow2 overflow".into())
    })
}

fn sumcheck_challenge(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    round_idx: usize,
    g_eval_u64: &[u64; 3],
) -> F {
    let mut h = blake3::Hasher::new();
    h.update(b"neo/closure-proof/whir-p3/ajtai-opening-only/sumcheck/chal/v1");
    h.update(&stmt.context_digest);
    h.update(&stmt.pp_id_digest);
    h.update(&stmt.obligations_digest);
    for &u in commitment_root_u64 {
        h.update(&u.to_le_bytes());
    }
    h.update(&(round_idx as u64).to_le_bytes());
    for &u in g_eval_u64 {
        h.update(&u.to_le_bytes());
    }
    // Take 64 bits (little-endian) and map into Goldilocks.
    let digest = h.finalize();
    let mut b = [0u8; 8];
    b.copy_from_slice(&digest.as_bytes()[0..8]);
    u64_to_whir_f(u64::from_le_bytes(b))
}

fn compress_in_place(evals: &mut Vec<F>, r: F) {
    debug_assert!(evals.len().is_power_of_two());
    debug_assert!(evals.len() >= 2);
    let mid = evals.len() / 2;
    for i in 0..mid {
        let p0 = evals[2 * i];
        let p1 = evals[2 * i + 1];
        evals[i] = r * (p1 - p0) + p0;
    }
    evals.truncate(mid);
}

fn derive_u_and_lambdas(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    kappa: usize,
    obligation_count: usize,
) -> (Vec<[NeoF; NeoD]>, Vec<NeoF>) {
    // u: κ vectors in F^d
    // λ: per-obligation weights in F
    let seed = derive_seed_v1(b"ajtai_opening_only/u_and_lambdas", stmt, Some(commitment_root_u64));
    let mut rng = ChaCha8Rng::from_seed(seed);

    let mut u_vecs = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        let mut v = [NeoF::ZERO; NeoD];
        for i in 0..NeoD {
            v[i] = NeoF::from_u64(rng.next_u64());
        }
        u_vecs.push(v);
    }

    let mut lambdas = Vec::with_capacity(obligation_count);
    for _ in 0..obligation_count {
        lambdas.push(NeoF::from_u64(rng.next_u64()));
    }

    (u_vecs, lambdas)
}

fn dot_u_commitment(u_vecs: &[[NeoF; NeoD]], c: &NeoCmt) -> Result<NeoF, ClosureProofError> {
    let d = c.d;
    let kappa = c.kappa;
    if d != NeoD {
        return Err(ClosureProofError::WhirP3(format!(
            "commitment d mismatch: got {d}, expected {NeoD}",
        )));
    }
    if kappa != u_vecs.len() {
        return Err(ClosureProofError::WhirP3(format!(
            "commitment κ mismatch: got {kappa}, expected {}",
            u_vecs.len(),
        )));
    }
    if c.data.len() != d * kappa {
        return Err(ClosureProofError::WhirP3(
            "commitment data length mismatch".into(),
        ));
    }

    let mut acc = NeoF::ZERO;
    for i in 0..kappa {
        for r in 0..d {
            acc += u_vecs[i][r] * c.data[i * d + r];
        }
    }
    Ok(acc)
}

fn sumcheck_challenge_full(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    round_idx: usize,
    g_evals_u64: &[u64],
) -> F {
    let mut h = blake3::Hasher::new();
    h.update(b"neo/closure-proof/whir-p3/full-closure/sumcheck/chal/v1");
    h.update(&stmt.context_digest);
    h.update(&stmt.pp_id_digest);
    h.update(&stmt.obligations_digest);
    for &u in commitment_root_u64 {
        h.update(&u.to_le_bytes());
    }
    h.update(&(round_idx as u64).to_le_bytes());
    for &u in g_evals_u64 {
        h.update(&u.to_le_bytes());
    }
    let digest = h.finalize();
    let mut b = [0u8; 8];
    b.copy_from_slice(&digest.as_bytes()[0..8]);
    u64_to_whir_f(u64::from_le_bytes(b))
}

fn range_vanishing_poly(z: F, base_b: u32) -> F {
    // Vanishing polynomial for the canonical Ajtai digit range:
    //   z ∈ {-(b-1), ..., 0, ..., (b-1)}  <=>  Π_{k=-(b-1)}^{b-1} (z - k) = 0.
    let b = base_b as i64;
    let mut acc = F::ONE;
    for k in (-(b - 1))..=(b - 1) {
        let fk = if k >= 0 {
            F::from_u64(k as u64)
        } else {
            F::ZERO - F::from_u64((-k) as u64)
        };
        acc *= z - fk;
    }
    acc
}

fn eq_poly_value(point: &[F], r0: &[F]) -> Result<F, ClosureProofError> {
    if point.len() != r0.len() {
        return Err(ClosureProofError::WhirP3(
            "eq_poly_value: point length mismatch".into(),
        ));
    }
    let mut acc = F::ONE;
    for (&x, &r) in point.iter().zip(r0.iter()) {
        // eq(x, r) = r*x + (1-r)*(1-x)
        acc *= r * x + (F::ONE - r) * (F::ONE - x);
    }
    Ok(acc)
}

fn prove_sumcheck_inner_product(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    mut z_evals: Vec<F>,
    mut w_evals: Vec<F>,
    claimed_sum: F,
) -> SumcheckProofV1 {
    assert_eq!(z_evals.len(), w_evals.len());
    assert!(z_evals.len().is_power_of_two());

    let n_vars = z_evals.len().ilog2() as usize;
    let mut claim = claimed_sum;
    let mut round_polys_u64 = Vec::with_capacity(n_vars);

    for round in 0..n_vars {
        // Compute g(X) evaluations at 0,1,2 for this round.
        //
        // Each variable fold pairs consecutive evaluations (lex order folds last variable first).
        let mut c0 = F::ZERO;
        let mut c2 = F::ZERO;
        for (z_pair, w_pair) in z_evals.chunks_exact(2).zip(w_evals.chunks_exact(2)) {
            let z0 = z_pair[0];
            let z1 = z_pair[1];
            let w0 = w_pair[0];
            let w1 = w_pair[1];
            c0 += z0 * w0;
            c2 += (z1 - z0) * (w1 - w0);
        }
        let c1 = claim - c0.double() - c2;
        let g0 = c0;
        let g1 = c0 + c1 + c2;
        let g2 = g1 + c1 + c2 + c2.double();

        let g_u64 = [whir_f_to_u64(g0), whir_f_to_u64(g1), whir_f_to_u64(g2)];
        round_polys_u64.push(g_u64);

        // Fiat–Shamir challenge for this round.
        let r = sumcheck_challenge(stmt, commitment_root_u64, round, &g_u64);

        // Update claim.
        claim = sumcheck_math::eval_quad([g0, g1, g2], r);

        // Fold z and w in place.
        compress_in_place(&mut z_evals, r);
        compress_in_place(&mut w_evals, r);
    }

    debug_assert_eq!(z_evals.len(), 1);
    debug_assert_eq!(w_evals.len(), 1);
    let z_r = z_evals[0];

    SumcheckProofV1 {
        round_polys_u64: round_polys_u64.into(),
        z_r_u64: whir_f_to_u64(z_r),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum AjtaiOpeningBackendKindV1 {
    OpeningOnly,
    OpeningPlusX,
}

impl AjtaiOpeningBackendKindV1 {
    fn backend_id(self) -> u32 {
        match self {
            Self::OpeningOnly => opaque::BACKEND_ID_WHIR_P3_AJTAI_OPENING_ONLY_V1,
            Self::OpeningPlusX => opaque::BACKEND_ID_WHIR_P3_AJTAI_OPENING_PLUS_X_V1,
        }
    }

    fn include_x(self) -> bool {
        match self {
            Self::OpeningOnly => false,
            Self::OpeningPlusX => true,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Full obligation closure backend (Ajtai opening + bounds + ME consistency)
// -------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct WhirP3FullClosurePayloadV1 {
    obligations: EncodedObligations,
    sumcheck: SumcheckProofV2,
    /// WHIR transcript/proof data as canonical u64 limbs of WHIR-field elements.
    whir_proof_data_u64: BoundedVec<u64, MAX_WHIR_PROOF_DATA_U64>,
}

fn mix_k_to_f(k: neo_math::K, delta: neo_math::F) -> neo_math::F {
    let coeffs = k.as_coeffs();
    coeffs[0] + delta * coeffs[1]
}

fn compute_rb_mix(
    r: &[neo_math::K],
    delta: neo_math::F,
) -> Vec<neo_math::F> {
    let rb = neo_ccs::utils::tensor_point::<neo_math::K>(r);
    rb.into_iter().map(|x| mix_k_to_f(x, delta)).collect()
}

fn chi_for_row_index(r: &[neo_math::K], idx: usize) -> neo_math::K {
    let mut acc = neo_math::K::ONE;
    for (bit, &ri) in r.iter().enumerate() {
        let is_one = ((idx >> bit) & 1) == 1;
        acc *= if is_one { ri } else { neo_math::K::ONE - ri };
    }
    acc
}

fn mix_in_me_consistency_core_and_bus(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    _params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    d: usize,
    m: usize,
    w_evals_whir: &mut [F],
    lambdas: &[neo_math::F],
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<(neo_math::F, neo_math::F), ClosureProofError> {
    // Randomize a single scalar ME-consistency check:
    //   Σ_i λ_i * Σ_j μ_j * Σ_ρ ν_ρ * y_i,j[ρ]  ==  Σ_i λ_i * Σ_ρ ν_ρ * Σ_c Z_i[ρ,c] * (Σ_j μ_j v_i,j[c])
    //
    // We work over the base field by mixing K components with a random δ.
    let seed = derive_seed_v1(b"full_closure/rng", stmt, Some(commitment_root_u64));
    let mut rng = ChaCha8Rng::from_seed(seed);

    // Mixer scalar γ_me; ensure nonzero to avoid accidentally disabling the check.
    let mut gamma_me = neo_math::F::from_u64(rng.next_u64());
    if gamma_me == neo_math::F::ZERO {
        gamma_me = neo_math::F::ONE;
    }
    let gamma_me_whir = neo_f_to_whir(gamma_me);

    // Mix K → F as c0 + δ*c1; δ must be nonzero with overwhelming probability.
    let mut delta_k = neo_math::F::from_u64(rng.next_u64());
    if delta_k == neo_math::F::ZERO {
        delta_k = neo_math::F::ONE;
    }

    // Row weights ν_ρ for ρ in 0..d.
    let mut nu = vec![neo_math::F::ZERO; d];
    for rho in 0..d {
        nu[rho] = neo_math::F::from_u64(rng.next_u64());
    }

    let core_t = ccs.t();
    let mut bus_cols_expected: Option<usize> = None;
    for me in obligations.main.iter().chain(obligations.val.iter()) {
        if me.y.len() != me.y_scalars.len() {
            return Err(ClosureProofError::WhirP3("ME y/y_scalars length mismatch".into()));
        }
        if me.y.len() < core_t {
            return Err(ClosureProofError::WhirP3("ME y.len() < core_t".into()));
        }
        let bus_cols = me.y.len() - core_t;
        match bus_cols_expected {
            None => bus_cols_expected = Some(bus_cols),
            Some(prev) if prev != bus_cols => {
                return Err(ClosureProofError::WhirP3(
                    "ME bus_cols mismatch across obligations".into(),
                ));
            }
            _ => {}
        }
        if bus_cols > 0 {
            let bus = bus.ok_or_else(|| {
                ClosureProofError::WhirP3(
                    "ME has bus openings but no BusLayout provided".into(),
                )
            })?;
            if bus.bus_cols != bus_cols || bus.m != m {
                return Err(ClosureProofError::WhirP3("BusLayout mismatch".into()));
            }
            if me.m_in != bus.m_in {
                return Err(ClosureProofError::WhirP3("ME m_in != bus.m_in".into()));
            }
        }
    }
    let bus_cols = bus_cols_expected.unwrap_or(0);

    // Matrix weights μ_j (core) and μ_bus[col_id] (bus openings).
    let mut mu_core = vec![neo_math::F::ZERO; core_t];
    for j in 0..core_t {
        mu_core[j] = neo_math::F::from_u64(rng.next_u64());
    }
    let mut mu_bus = vec![neo_math::F::ZERO; bus_cols];
    for col_id in 0..bus_cols {
        mu_bus[col_id] = neo_math::F::from_u64(rng.next_u64());
    }

    let mut claimed_me = neo_math::F::ZERO;
    let mut base_idx = 0usize;

    for (i, me) in obligations
        .main
        .iter()
        .chain(obligations.val.iter())
        .enumerate()
    {
        let lambda_i = lambdas
            .get(i)
            .copied()
            .ok_or_else(|| ClosureProofError::WhirP3("lambda count mismatch".into()))?;

        // Precompute r^b mix for this obligation.
        let rb_mix = compute_rb_mix(&me.r, delta_k);
        let n_eff = core::cmp::min(ccs.n, rb_mix.len());

        // Column weights s[c] = Σ_j μ_j * v_j[c] (mixed into base field).
        let mut col_weights = vec![neo_math::F::ZERO; m];

        for (j, mat) in ccs.matrices.iter().enumerate() {
            let mu_j = mu_core[j];
            if mu_j == neo_math::F::ZERO {
                continue;
            }
            match mat {
                neo_ccs::CcsMatrix::Identity { n } => {
                    let cap = core::cmp::min(n_eff, *n);
                    for idx in 0..cap {
                        col_weights[idx] += mu_j * rb_mix[idx];
                    }
                }
                neo_ccs::CcsMatrix::Csc(csc) => {
                    for c in 0..csc.ncols {
                        let s0 = csc.col_ptr[c];
                        let e0 = csc.col_ptr[c + 1];
                        for k in s0..e0 {
                            let row = csc.row_idx[k];
                            if row >= n_eff {
                                continue;
                            }
                            let wr = rb_mix[row];
                            if wr == neo_math::F::ZERO {
                                continue;
                            }
                            col_weights[c] += mu_j * wr * csc.vals[k];
                        }
                    }
                }
            }
        }

        if bus_cols > 0 {
            let bus = bus.expect("checked above");
            for col_id in 0..bus_cols {
                let mu = mu_bus[col_id];
                if mu == neo_math::F::ZERO {
                    continue;
                }
                for j in 0..bus.chunk_size {
                    let row = bus.time_index(j);
                    let w_time = chi_for_row_index(&me.r, row);
                    let w_time_mix = mix_k_to_f(w_time, delta_k);
                    let z_idx = bus.bus_cell(col_id, j);
                    if z_idx >= m {
                        return Err(ClosureProofError::WhirP3("bus_cell out of range".into()));
                    }
                    col_weights[z_idx] += mu * w_time_mix;
                }
            }
        }

        // Claimed sum from public y values.
        for j in 0..core_t {
            let mu_j = mu_core[j];
            if mu_j == neo_math::F::ZERO {
                continue;
            }
            let yj = me.y.get(j).ok_or_else(|| {
                ClosureProofError::WhirP3("ME y missing core entry".into())
            })?;
            if yj.len() < d {
                return Err(ClosureProofError::WhirP3("ME y row too short".into()));
            }
            let mut dot = neo_math::F::ZERO;
            for rho in 0..d {
                dot += nu[rho] * mix_k_to_f(yj[rho], delta_k);
            }
            claimed_me += lambda_i * mu_j * dot;
        }

        for col_id in 0..bus_cols {
            let mu = mu_bus[col_id];
            if mu == neo_math::F::ZERO {
                continue;
            }
            let yj = me
                .y
                .get(core_t + col_id)
                .ok_or_else(|| ClosureProofError::WhirP3("ME y missing bus entry".into()))?;
            if yj.len() < d {
                return Err(ClosureProofError::WhirP3("ME y bus row too short".into()));
            }
            let mut dot = neo_math::F::ZERO;
            for rho in 0..d {
                dot += nu[rho] * mix_k_to_f(yj[rho], delta_k);
            }
            claimed_me += lambda_i * mu * dot;
        }

        // Add Z weights for this obligation.
        for rho in 0..d {
            let row_scale = lambda_i * nu[rho];
            if row_scale == neo_math::F::ZERO {
                continue;
            }
            let row_base = base_idx
                .checked_add(rho * m)
                .ok_or_else(|| ClosureProofError::WhirP3("weight index overflow".into()))?;
            for c in 0..m {
                let idx = row_base + c;
                let w = row_scale * col_weights[c];
                if w != neo_math::F::ZERO {
                    w_evals_whir[idx] += gamma_me_whir * neo_f_to_whir(w);
                }
            }
        }

        base_idx = base_idx
            .checked_add(d * m)
            .ok_or_else(|| ClosureProofError::WhirP3("base_idx overflow".into()))?;
    }

    Ok((gamma_me, claimed_me))
}

fn build_eq_evals(r0: &[F]) -> Vec<F> {
    let mut evals = vec![F::ONE];
    for &ri in r0 {
        let one_minus = F::ONE - ri;
        let mut next = Vec::with_capacity(evals.len() * 2);
        for &v in &evals {
            next.push(v * one_minus);
            next.push(v * ri);
        }
        evals = next;
    }
    evals
}

fn prove_sumcheck_full_closure(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    mut z_evals: Vec<F>,
    mut w_evals: Vec<F>,
    r0: &[F],
    delta_range: F,
    base_b: u32,
    claimed_sum: F,
) -> SumcheckProofV2 {
    assert_eq!(z_evals.len(), w_evals.len());
    assert!(z_evals.len().is_power_of_two());
    assert_eq!(z_evals.len(), 1usize << r0.len());

    let deg = 2usize * (base_b as usize);
    let n_vars = z_evals.len().ilog2() as usize;
    let mut claim = claimed_sum;
    let mut round_evals_u64 = Vec::with_capacity(n_vars);

    let mut eq_evals = build_eq_evals(r0);
    assert_eq!(eq_evals.len(), z_evals.len());

    for round in 0..n_vars {
        let mut g_evals = vec![F::ZERO; deg + 1];

        for ((z_pair, w_pair), eq_pair) in z_evals
            .chunks_exact(2)
            .zip(w_evals.chunks_exact(2))
            .zip(eq_evals.chunks_exact(2))
        {
            let z0 = z_pair[0];
            let z1 = z_pair[1];
            let w0 = w_pair[0];
            let w1 = w_pair[1];
            let e0 = eq_pair[0];
            let e1 = eq_pair[1];

            let dz = z1 - z0;
            let dw = w1 - w0;
            let de = e1 - e0;

            for t in 0..=deg {
                let tt = F::from_u64(t as u64);
                let zt = tt * dz + z0;
                let wt = tt * dw + w0;
                let et = tt * de + e0;
                let rng = range_vanishing_poly(zt, base_b);
                g_evals[t] += zt * wt + delta_range * et * rng;
            }
        }

        let g_u64: Vec<u64> = g_evals.iter().map(|x| whir_f_to_u64(*x)).collect();
        round_evals_u64.push(g_u64.clone());

        // Sumcheck consistency: g(0)+g(1) must match the running claim.
        // For the honest prover this holds by construction; keep the same update rule as the verifier.
        debug_assert_eq!(g_evals[0] + g_evals[1], claim);

        let r = sumcheck_challenge_full(stmt, commitment_root_u64, round, &g_u64);
        claim = sumcheck_math::eval_lagrange_0_to_deg(&g_evals, r);

        compress_in_place(&mut z_evals, r);
        compress_in_place(&mut w_evals, r);
        compress_in_place(&mut eq_evals, r);
    }

    debug_assert_eq!(z_evals.len(), 1);
    let z_r = z_evals[0];

    SumcheckProofV2 {
        round_evals_u64: round_evals_u64
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>()
            .into(),
        z_r_u64: whir_f_to_u64(z_r),
    }
}

pub fn prove_whir_p3_full_closure_bytes_v1(
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    main_wits: &[neo_ccs::Mat<neo_math::F>],
    val_wits: &[neo_ccs::Mat<neo_math::F>],
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<Vec<u8>, ClosureProofError> {
    // Bind obligations to the statement digest.
    let expected_digest = contract::expected_obligations_digest(params, obligations, stmt.pp_id_digest);
    if expected_digest != stmt.obligations_digest {
        return Err(ClosureProofError::WhirP3(
            "obligations_digest mismatch (not bound to Phase-1 obligations)".into(),
        ));
    }
    if obligations.main.len() != main_wits.len() || obligations.val.len() != val_wits.len() {
        return Err(ClosureProofError::WhirP3("witness count mismatch".into()));
    }

    let d = params.d as usize;
    if d != NeoD {
        return Err(ClosureProofError::WhirP3("unexpected d (must match neo_math::D)".into()));
    }
    let m = ccs.m;

    // Enforce that the loaded seeded PP matches the statement's pp_id_digest.
    let (kappa, seed) = contract::require_global_pp_matches_statement(stmt.pp_id_digest, params, d, m)
        .map_err(ClosureProofError::WhirP3)?;

    // Commit to the concatenated witness Z (main then val), padded to a power of two.
    let obligation_count = obligations.main.len() + obligations.val.len();
    let z_len = obligation_count
        .checked_mul(d)
        .and_then(|x| x.checked_mul(m))
        .ok_or_else(|| ClosureProofError::WhirP3("z_len overflow".into()))?;
    let z_len_padded = next_pow2_checked(z_len.max(1))?;
    let num_vars = z_len_padded.ilog2() as usize;
    if z_len_padded > MAX_DEV_Z_EVALS_PADDED {
        return Err(ClosureProofError::WhirP3(format!(
            "Z too large for whir-p3 dev backend: z_len_padded={z_len_padded} exceeds MAX_DEV_Z_EVALS_PADDED={MAX_DEV_Z_EVALS_PADDED} (obligations={obligation_count}, d={d}, m={m})"
        )));
    }

    let mut z_evals_whir = vec![F::ZERO; z_len_padded];
    let mut fill_idx = 0usize;
    for Z in main_wits.iter().chain(val_wits.iter()) {
        if Z.rows() != d || Z.cols() != m {
            return Err(ClosureProofError::WhirP3("Z shape mismatch".into()));
        }
        for row in 0..d {
            for col in 0..m {
                z_evals_whir[fill_idx] = neo_f_to_whir(Z[(row, col)]);
                fill_idx += 1;
            }
        }
    }

    let z_poly = EvaluationsList::new(z_evals_whir.clone());

    // WHIR parameters and committer (as in placeholder).
    let params_whir = make_params(num_vars);
    let domainsep = domain_separator_for_stmt(&params_whir, stmt);

    let mut rng = ChaCha8Rng::from_seed(fixed_seed(b"challenger_perm"));
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    // Commitment phase (does not depend on statement points).
    let committer = CommitmentWriter::new(&params_whir);
    let dft_committer = EvalsDft::<F>::default();
    let witness = committer
        .commit::<WHIR_P3_DIGEST_ELEMS>(&dft_committer, &mut prover_state, z_poly.clone())
        .map_err(|e| ClosureProofError::WhirP3(format!("WHIR commit failed: {e:?}")))?;

    // Extract the commitment root limbs by parsing the WHIR commitment prefix.
    // This avoids relying on `proof_data` internal layout beyond WHIR's `parse_commitment` API.
    let commitment_root_u64 = extract_commitment_root_u64_from_proof_data(
        &params_whir,
        &domainsep,
        challenger,
        prover_state.proof_data(),
    )?;

    // Derive batching randomness (u vectors and per-obligation weights).
    let (u_vecs, lambdas) = derive_u_and_lambdas(stmt, &commitment_root_u64, kappa, obligation_count);

    // Build the Ajtai opening weight vector for the chosen u (one per Z entry).
    let w_u = neo_ajtai::compute_opening_weights_for_u_seeded(seed, m, &u_vecs);

    // Compute the claimed sum t = Σ_i λ_i · <u, c_i>.
    let mut claimed_sum_neo = NeoF::ZERO;
    for (idx, me) in obligations.main.iter().chain(obligations.val.iter()).enumerate() {
        let t_i = dot_u_commitment(&u_vecs, &me.c)?;
        claimed_sum_neo += lambdas[idx] * t_i;
    }

    // Build w_total = concat_i (λ_i * w_u), padded.
    let mut w_evals_whir = vec![F::ZERO; z_len_padded];
    let mut w_idx = 0usize;
    for lambda in lambdas.iter() {
        let lambda_whir = neo_f_to_whir(*lambda);
        for &w in &w_u {
            w_evals_whir[w_idx] = lambda_whir * neo_f_to_whir(w);
            w_idx += 1;
        }
    }
    debug_assert_eq!(w_idx, z_len);

    // X-projection: fold a random linear check into the same weight vector.
    let (gamma_x, claimed_x) =
        mix_in_x_projection(stmt, &commitment_root_u64, obligations, d, m, &mut w_evals_whir)?;
    claimed_sum_neo += gamma_x * claimed_x;

    // ME consistency: fold a random linear check into the same weight vector.
    let (gamma_me, claimed_me) = mix_in_me_consistency_core_and_bus(
        stmt,
        &commitment_root_u64,
        params,
        ccs,
        obligations,
        d,
        m,
        &mut w_evals_whir,
        &lambdas,
        bus,
    )?;
    claimed_sum_neo += gamma_me * claimed_me;

    let claimed_sum = neo_f_to_whir(claimed_sum_neo);

    // Range check: derive a random "Eq" point r0 and a mixing scalar δ_range.
    let mut rng = ChaCha8Rng::from_seed(derive_seed_v1(
        b"full_closure/range_rng",
        stmt,
        Some(commitment_root_u64.as_slice()),
    ));
    let mut delta_range_neo = NeoF::from_u64(rng.next_u64());
    if delta_range_neo == NeoF::ZERO {
        delta_range_neo = NeoF::ONE;
    }
    let delta_range = neo_f_to_whir(delta_range_neo);

    let mut r0 = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        r0.push(u64_to_whir_f(rng.next_u64()));
    }

    // Prove the combined sumcheck for:
    //   Σ_x [ Z(x)*W(x) + δ_range*Eq(x,r0)*Range(Z(x)) ] == claimed_sum.
    let sumcheck = prove_sumcheck_full_closure(
        stmt,
        &commitment_root_u64,
        z_evals_whir,
        w_evals_whir,
        &r0,
        delta_range,
        params.b,
        claimed_sum,
    );

    // Recover the sumcheck challenge point from the proof so we can build the WHIR statement.
    let deg = 2usize * (params.b as usize);
    if sumcheck.round_evals_u64.len() != num_vars {
        return Err(ClosureProofError::WhirP3("sumcheck rounds mismatch".into()));
    }
    let mut rands = Vec::with_capacity(num_vars);
    let mut claim = claimed_sum;
    for (round, g_u64) in sumcheck.round_evals_u64.iter().enumerate() {
        if g_u64.len() != deg + 1 {
            return Err(ClosureProofError::WhirP3("sumcheck degree mismatch".into()));
        }
        let g0 = u64_to_whir_f(g_u64[0]);
        let g1 = u64_to_whir_f(g_u64[1]);
        if g0 + g1 != claim {
            return Err(ClosureProofError::WhirP3("sumcheck consistency failed".into()));
        }
        let r = sumcheck_challenge_full(stmt, &commitment_root_u64, round, g_u64);
        let evals: Vec<F> = g_u64.iter().copied().map(u64_to_whir_f).collect();
        claim = sumcheck_math::eval_lagrange_0_to_deg(&evals, r);
        rands.push(r);
    }
    let mut coords = rands;
    coords.reverse();

    let z_r = u64_to_whir_f(sumcheck.z_r_u64);
    let mut statement = Statement::<EF>::initialize(num_vars);
    statement.add_evaluated_constraint(MultilinearPoint::new(coords), z_r);

    let dft_prover = EvalsDft::<F>::default();
    let prover = Prover(&params_whir);
    prover
        .prove::<WHIR_P3_DIGEST_ELEMS>(&dft_prover, &mut prover_state, statement, witness)
        .map_err(|e| ClosureProofError::WhirP3(format!("WHIR prove failed: {e:?}")))?;

    let payload = WhirP3FullClosurePayloadV1 {
        obligations: EncodedObligations::encode(obligations),
        sumcheck,
        whir_proof_data_u64: encode_proof_data(prover_state.proof_data()).into(),
    };
    let payload_bytes = serialize_payload(&payload)?;
    Ok(opaque::encode_envelope(
        opaque::BACKEND_ID_WHIR_P3_FULL_CLOSURE_V1,
        &payload_bytes,
    ))
}

pub fn verify_whir_p3_full_closure_payload_v1(
    stmt: &ClosureStatementV1,
    payload_bytes: &[u8],
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<(), ClosureProofError> {
    let payload: WhirP3FullClosurePayloadV1 = deserialize_payload(payload_bytes)?;

    let obligations = payload
        .obligations
        .decode()
        .ok_or(ClosureProofError::InvalidOpaqueProofEncoding)?;
    let expected_digest = contract::expected_obligations_digest(params, &obligations, stmt.pp_id_digest);
    if expected_digest != stmt.obligations_digest {
        return Err(ClosureProofError::WhirP3("obligations_digest mismatch".into()));
    }

    let d = params.d as usize;
    if d != NeoD {
        return Err(ClosureProofError::WhirP3("unexpected d (must match neo_math::D)".into()));
    }
    let m = ccs.m;
    let obligation_count = obligations.main.len() + obligations.val.len();
    let z_len = obligation_count
        .checked_mul(d)
        .and_then(|x| x.checked_mul(m))
        .ok_or_else(|| ClosureProofError::WhirP3("z_len overflow".into()))?;
    let z_len_padded = next_pow2_checked(z_len.max(1))?;
    let num_vars = z_len_padded.ilog2() as usize;
    if z_len_padded > MAX_DEV_Z_EVALS_PADDED {
        return Err(ClosureProofError::WhirP3(format!(
            "Z too large for whir-p3 dev backend: z_len_padded={z_len_padded} exceeds MAX_DEV_Z_EVALS_PADDED={MAX_DEV_Z_EVALS_PADDED} (obligations={obligation_count}, d={d}, m={m})"
        )));
    }

    // Enforce that the loaded seeded PP matches the statement's pp_id_digest.
    let (kappa, seed) = contract::require_global_pp_matches_statement(stmt.pp_id_digest, params, d, m)
        .map_err(ClosureProofError::WhirP3)?;

    // Extract commitment root limbs from the WHIR proof data (first digest elems).
    let proof_data = decode_proof_data_u64(&payload.whir_proof_data_u64);
    let params_whir = make_params(num_vars);
    let domainsep = domain_separator_for_stmt(&params_whir, stmt);
    let mut rng = ChaCha8Rng::from_seed(fixed_seed(b"challenger_perm"));
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
    let commitment_root_u64 = extract_commitment_root_u64_from_proof_data(
        &params_whir,
        &domainsep,
        challenger.clone(),
        &proof_data,
    )?;

    // Derive batching randomness.
    let (u_vecs, lambdas) = derive_u_and_lambdas(stmt, &commitment_root_u64, kappa, obligation_count);

    // Claimed opening sum from commitments.
    let mut claimed_sum_neo = NeoF::ZERO;
    for (idx, me) in obligations.main.iter().chain(obligations.val.iter()).enumerate() {
        let t_i = dot_u_commitment(&u_vecs, &me.c)?;
        claimed_sum_neo += lambdas[idx] * t_i;
    }

    // Build weight vector for Ajtai opening.
    let w_u = neo_ajtai::compute_opening_weights_for_u_seeded(seed, m, &u_vecs);
    let mut w_evals_whir = vec![F::ZERO; z_len_padded];
    let mut w_idx = 0usize;
    for lambda in &lambdas {
        let lambda_whir = neo_f_to_whir(*lambda);
        for &w in &w_u {
            w_evals_whir[w_idx] = lambda_whir * neo_f_to_whir(w);
            w_idx += 1;
        }
    }

    // X-projection check.
    let (gamma_x, claimed_x) =
        mix_in_x_projection(stmt, &commitment_root_u64, &obligations, d, m, &mut w_evals_whir)?;
    claimed_sum_neo += gamma_x * claimed_x;

    // ME consistency check.
    let (gamma_me, claimed_me) = mix_in_me_consistency_core_and_bus(
        stmt,
        &commitment_root_u64,
        params,
        ccs,
        &obligations,
        d,
        m,
        &mut w_evals_whir,
        &lambdas,
        bus,
    )?;
    claimed_sum_neo += gamma_me * claimed_me;

    let claimed_sum = neo_f_to_whir(claimed_sum_neo);

    // Re-derive r0 and δ_range.
    let mut rng = ChaCha8Rng::from_seed(derive_seed_v1(
        b"full_closure/range_rng",
        stmt,
        Some(commitment_root_u64.as_slice()),
    ));
    let mut delta_range_neo = NeoF::from_u64(rng.next_u64());
    if delta_range_neo == NeoF::ZERO {
        delta_range_neo = NeoF::ONE;
    }
    let delta_range = neo_f_to_whir(delta_range_neo);

    let mut r0 = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        r0.push(u64_to_whir_f(rng.next_u64()));
    }

    // Verify sumcheck.
    let deg = 2usize * (params.b as usize);
    if payload.sumcheck.round_evals_u64.len() != num_vars {
        return Err(ClosureProofError::WhirP3("sumcheck rounds mismatch".into()));
    }
    let mut claim = claimed_sum;
    let mut rands = Vec::with_capacity(num_vars);
    for (round, g_u64) in payload.sumcheck.round_evals_u64.iter().enumerate() {
        if g_u64.len() != deg + 1 {
            return Err(ClosureProofError::WhirP3("sumcheck degree mismatch".into()));
        }
        let g0 = u64_to_whir_f(g_u64[0]);
        let g1 = u64_to_whir_f(g_u64[1]);
        if g0 + g1 != claim {
            return Err(ClosureProofError::WhirP3("sumcheck consistency failed".into()));
        }
        let r = sumcheck_challenge_full(stmt, &commitment_root_u64, round, g_u64);
        let evals: Vec<F> = g_u64.iter().copied().map(u64_to_whir_f).collect();
        claim = sumcheck_math::eval_lagrange_0_to_deg(&evals, r);
        rands.push(r);
    }

    let z_r = u64_to_whir_f(payload.sumcheck.z_r_u64);

    // Compute w(r) by folding the public weight table.
    let mut w_fold = w_evals_whir;
    for &r in &rands {
        compress_in_place(&mut w_fold, r);
    }
    if w_fold.len() != 1 {
        return Err(ClosureProofError::WhirP3("w folding failed".into()));
    }
    let w_r = w_fold[0];

    let mut coords = rands;
    coords.reverse();
    let eq_r = eq_poly_value(&coords, &r0)?;
    let rng_r = range_vanishing_poly(z_r, params.b);

    if claim != z_r * w_r + delta_range * eq_r * rng_r {
        return Err(ClosureProofError::WhirP3("sumcheck final check failed".into()));
    }

    // Verify WHIR evaluation proof: Z(r) == z_r.
    let mut statement = Statement::<EF>::initialize(num_vars);
    statement.add_evaluated_constraint(MultilinearPoint::new(coords), z_r);

    let commitment_reader = CommitmentReader::new(&params_whir);
    let verifier = Verifier::new(&params_whir);

    let mut verifier_state = domainsep.to_verifier_state(proof_data, challenger);
    let parsed_commitment = commitment_reader
        .parse_commitment::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state)
        .map_err(|e| ClosureProofError::WhirP3(format!("parse_commitment failed: {e:?}")))?;

    verifier
        .verify::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state, &parsed_commitment, &statement)
        .map_err(|e| ClosureProofError::WhirP3(format!("verify failed: {e:?}")))?;

    // Extra explicit checks that are part of the closure contract but not enforced by the sumcheck.
    //
    // - y padding beyond d must be zero,
    // - y_scalars must be consistent with y (canonical base-b recomposition).
    let core_t = ccs.t();
    let bK = neo_math::K::from(neo_math::F::from_u64(params.b as u64));
    for me in obligations.main.iter().chain(obligations.val.iter()) {
        if me.y.len() != me.y_scalars.len() {
            return Err(ClosureProofError::WhirP3("ME y/y_scalars length mismatch".into()));
        }
        if me.y.len() < core_t {
            return Err(ClosureProofError::WhirP3("ME y.len() < core_t".into()));
        }
        for (j, yj) in me.y.iter().enumerate() {
            if yj.len() < d {
                return Err(ClosureProofError::WhirP3("ME y row too short".into()));
            }
            for rho in d..yj.len() {
                if yj[rho] != neo_math::K::ZERO {
                    return Err(ClosureProofError::WhirP3(format!(
                        "ME y padding nonzero at j={j}, rho={rho}"
                    )));
                }
            }

            let mut sc = neo_math::K::ZERO;
            let mut pow = neo_math::K::ONE;
            for rho in 0..d {
                sc += pow * yj[rho];
                pow *= bK;
            }
            if me.y_scalars[j] != sc {
                return Err(ClosureProofError::WhirP3(format!(
                    "ME y_scalars mismatch at j={j}"
                )));
            }
        }
    }

    Ok(())
}

fn mix_in_x_projection(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    d: usize,
    m: usize,
    w_evals_whir: &mut [F],
) -> Result<(NeoF, NeoF), ClosureProofError> {
    let seed = derive_seed_v1(b"ajtai_opening_plus_x/rng", stmt, Some(commitment_root_u64));
    let mut rng = ChaCha8Rng::from_seed(seed);

    // Mixer scalar γ; if γ=0 (prob ~2^-64), bump to 1 so the check isn't accidentally disabled.
    let mut gamma = NeoF::from_u64(rng.next_u64());
    if gamma == NeoF::ZERO {
        gamma = NeoF::ONE;
    }
    let gamma_whir = neo_f_to_whir(gamma);

    let mut claimed_x = NeoF::ZERO;
    let mut base_idx = 0usize;

    for me in obligations.main.iter().chain(obligations.val.iter()) {
        let m_in = me.m_in;
        if m_in > m {
            return Err(ClosureProofError::WhirP3("m_in exceeds commitment width".into()));
        }
        if me.X.rows() != d || me.X.cols() != m_in {
            return Err(ClosureProofError::WhirP3("X shape mismatch".into()));
        }
        for row in 0..d {
            for col in 0..m_in {
                let beta = NeoF::from_u64(rng.next_u64());
                claimed_x += beta * me.X[(row, col)];

                let idx = base_idx + row * m + col;
                w_evals_whir[idx] += gamma_whir * neo_f_to_whir(beta);
            }
        }
        base_idx += d * m;
    }

    Ok((gamma, claimed_x))
}

fn prove_whir_p3_ajtai_opening_bytes_v1_impl(
    kind: AjtaiOpeningBackendKindV1,
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    main_wits: &[neo_ccs::Mat<neo_math::F>],
    val_wits: &[neo_ccs::Mat<neo_math::F>],
) -> Result<Vec<u8>, ClosureProofError> {
    // Bind obligations to the statement digest.
    let expected_digest = contract::expected_obligations_digest(params, obligations, stmt.pp_id_digest);
    if expected_digest != stmt.obligations_digest {
        return Err(ClosureProofError::WhirP3(
            "obligations_digest mismatch (not bound to Phase-1 obligations)".into(),
        ));
    }
    if obligations.main.len() != main_wits.len() || obligations.val.len() != val_wits.len() {
        return Err(ClosureProofError::WhirP3("witness count mismatch".into()));
    }

    let d = params.d as usize;
    if d != NeoD {
        return Err(ClosureProofError::WhirP3("unexpected d (must match neo_math::D)".into()));
    }
    let m = ccs.m;

    // Commit to the concatenated witness Z (main then val), padded to a power of two.
    let obligation_count = obligations.main.len() + obligations.val.len();
    let z_len = obligation_count
        .checked_mul(d)
        .and_then(|x| x.checked_mul(m))
        .ok_or_else(|| ClosureProofError::WhirP3("z_len overflow".into()))?;
    let z_len_padded = next_pow2_checked(z_len.max(1))?;
    let num_vars = z_len_padded.ilog2() as usize;
    if z_len_padded > MAX_DEV_Z_EVALS_PADDED {
        return Err(ClosureProofError::WhirP3(format!(
            "Z too large for whir-p3 dev backend: z_len_padded={z_len_padded} exceeds MAX_DEV_Z_EVALS_PADDED={MAX_DEV_Z_EVALS_PADDED} (obligations={obligation_count}, d={d}, m={m})"
        )));
    }

    let mut z_evals_whir = vec![F::ZERO; z_len_padded];
    let mut fill_idx = 0usize;
    for Z in main_wits.iter().chain(val_wits.iter()) {
        if Z.rows() != d || Z.cols() != m {
            return Err(ClosureProofError::WhirP3("Z shape mismatch".into()));
        }
        for row in 0..d {
            for col in 0..m {
                z_evals_whir[fill_idx] = neo_f_to_whir(Z[(row, col)]);
                fill_idx += 1;
            }
        }
    }

    let z_poly = EvaluationsList::new(z_evals_whir.clone());

    // WHIR parameters and committer (as in placeholder).
    let params_whir = make_params(num_vars);
    let domainsep = domain_separator_for_stmt(&params_whir, stmt);

    let mut rng = ChaCha8Rng::from_seed(fixed_seed(b"challenger_perm"));
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    // Commitment phase (does not depend on statement points).
    let committer = CommitmentWriter::new(&params_whir);
    let dft_committer = EvalsDft::<F>::default();
    let witness = committer
        .commit::<WHIR_P3_DIGEST_ELEMS>(&dft_committer, &mut prover_state, z_poly.clone())
        .map_err(|e| ClosureProofError::WhirP3(format!("WHIR commit failed: {e:?}")))?;

    // Extract the commitment root limbs by parsing the WHIR commitment prefix.
    // This avoids relying on `proof_data` internal layout beyond WHIR's `parse_commitment` API.
    let commitment_root_u64 = extract_commitment_root_u64_from_proof_data(
        &params_whir,
        &domainsep,
        challenger,
        prover_state.proof_data(),
    )?;

    // Derive batching randomness.
    let (kappa, seed) = contract::require_global_pp_matches_statement(stmt.pp_id_digest, params, d, m)
        .map_err(ClosureProofError::WhirP3)?;
    let (u_vecs, lambdas) = derive_u_and_lambdas(stmt, &commitment_root_u64, kappa, obligation_count);

    // Build the Ajtai opening weight vector for the chosen u (one per Z entry).
    let w_u = neo_ajtai::compute_opening_weights_for_u_seeded(seed, m, &u_vecs);

    // Compute the claimed sum t = Σ_i λ_i · <u, c_i>.
    let mut claimed_sum_neo = NeoF::ZERO;
    for (idx, me) in obligations.main.iter().chain(obligations.val.iter()).enumerate() {
        let t_i = dot_u_commitment(&u_vecs, &me.c)?;
        claimed_sum_neo += lambdas[idx] * t_i;
    }

    // Build w_total = concat_i (λ_i * w_u), padded.
    let mut w_evals_whir = vec![F::ZERO; z_len_padded];
    let mut w_idx = 0usize;
    for lambda in lambdas.iter() {
        let lambda_whir = neo_f_to_whir(*lambda);
        for &w in &w_u {
            w_evals_whir[w_idx] = lambda_whir * neo_f_to_whir(w);
            w_idx += 1;
        }
    }
    debug_assert_eq!(w_idx, z_len);

    // Optional X-projection: fold a random linear check into the same sumcheck.
    if kind.include_x() {
        let (gamma, claimed_x) =
            mix_in_x_projection(stmt, &commitment_root_u64, obligations, d, m, &mut w_evals_whir)?;
        claimed_sum_neo += gamma * claimed_x;
    }
    let claimed_sum = neo_f_to_whir(claimed_sum_neo);

    // Prove the inner product via sumcheck, binding challenges to the commitment root.
    let sumcheck = prove_sumcheck_inner_product(
        stmt,
        &commitment_root_u64,
        z_evals_whir,
        w_evals_whir,
        claimed_sum,
    );

    // Build a WHIR statement for a single evaluation: z(r) == z_r.
    let z_r = u64_to_whir_f(sumcheck.z_r_u64);
    let mut statement = Statement::<EF>::initialize(num_vars);
    // Recompute r vector from sumcheck transcript.
    let rands = sumcheck
        .round_polys_u64
        .iter()
        .enumerate()
        .map(|(round, g)| sumcheck_challenge(stmt, &commitment_root_u64, round, g))
        .collect::<Vec<_>>();
    let mut coords = rands;
    coords.reverse();
    statement.add_evaluated_constraint(MultilinearPoint::new(coords), z_r);

    // Prove evaluation constraint via WHIR.
    let prover = Prover(&params_whir);
    let dft_prover = EvalsDft::<F>::default();
    prover
        .prove::<WHIR_P3_DIGEST_ELEMS>(&dft_prover, &mut prover_state, statement, witness)
        .map_err(|e| ClosureProofError::WhirP3(format!("WHIR prove failed: {e:?}")))?;

    let payload = WhirP3AjtaiOpeningPayloadV1 {
        obligations: EncodedObligations::encode(obligations),
        sumcheck,
        whir_proof_data_u64: encode_proof_data(prover_state.proof_data()).into(),
    };
    let payload_bytes = serialize_payload(&payload)?;
    Ok(opaque::encode_envelope(
        kind.backend_id(),
        &payload_bytes,
    ))
}

pub fn prove_whir_p3_ajtai_opening_only_bytes_v1(
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    main_wits: &[neo_ccs::Mat<neo_math::F>],
    val_wits: &[neo_ccs::Mat<neo_math::F>],
) -> Result<Vec<u8>, ClosureProofError> {
    prove_whir_p3_ajtai_opening_bytes_v1_impl(
        AjtaiOpeningBackendKindV1::OpeningOnly,
        stmt,
        params,
        ccs,
        obligations,
        main_wits,
        val_wits,
    )
}

pub fn prove_whir_p3_ajtai_opening_plus_x_bytes_v1(
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    main_wits: &[neo_ccs::Mat<neo_math::F>],
    val_wits: &[neo_ccs::Mat<neo_math::F>],
) -> Result<Vec<u8>, ClosureProofError> {
    prove_whir_p3_ajtai_opening_bytes_v1_impl(
        AjtaiOpeningBackendKindV1::OpeningPlusX,
        stmt,
        params,
        ccs,
        obligations,
        main_wits,
        val_wits,
    )
}

fn verify_whir_p3_ajtai_opening_payload_v1_impl(
    kind: AjtaiOpeningBackendKindV1,
    stmt: &ClosureStatementV1,
    payload_bytes: &[u8],
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
) -> Result<(), ClosureProofError> {
    let payload: WhirP3AjtaiOpeningPayloadV1 = deserialize_payload(payload_bytes)?;

    let obligations = payload
        .obligations
        .decode()
        .ok_or(ClosureProofError::InvalidOpaqueProofEncoding)?;
    let expected_digest = contract::expected_obligations_digest(params, &obligations, stmt.pp_id_digest);
    if expected_digest != stmt.obligations_digest {
        return Err(ClosureProofError::WhirP3("obligations_digest mismatch".into()));
    }

    let d = params.d as usize;
    if d != NeoD {
        return Err(ClosureProofError::WhirP3("unexpected d (must match neo_math::D)".into()));
    }
    let m = ccs.m;
    let obligation_count = obligations.main.len() + obligations.val.len();
    let z_len = obligation_count
        .checked_mul(d)
        .and_then(|x| x.checked_mul(m))
        .ok_or_else(|| ClosureProofError::WhirP3("z_len overflow".into()))?;
    let z_len_padded = next_pow2_checked(z_len.max(1))?;
    let num_vars = z_len_padded.ilog2() as usize;
    if z_len_padded > MAX_DEV_Z_EVALS_PADDED {
        return Err(ClosureProofError::WhirP3(format!(
            "Z too large for whir-p3 dev backend: z_len_padded={z_len_padded} exceeds MAX_DEV_Z_EVALS_PADDED={MAX_DEV_Z_EVALS_PADDED} (obligations={obligation_count}, d={d}, m={m})"
        )));
    }

    let proof_data = decode_proof_data_u64(&payload.whir_proof_data_u64);
    let params_whir = make_params(num_vars);
    let domainsep = domain_separator_for_stmt(&params_whir, stmt);
    let mut rng = ChaCha8Rng::from_seed(fixed_seed(b"challenger_perm"));
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
    let commitment_root_u64 = extract_commitment_root_u64_from_proof_data(
        &params_whir,
        &domainsep,
        challenger.clone(),
        &proof_data,
    )?;

    // Derive batching randomness.
    let (kappa, seed) = contract::require_global_pp_matches_statement(stmt.pp_id_digest, params, d, m)
        .map_err(ClosureProofError::WhirP3)?;
    let (u_vecs, lambdas) = derive_u_and_lambdas(stmt, &commitment_root_u64, kappa, obligation_count);

    // Claimed opening sum from commitments.
    let mut claimed_sum_neo = NeoF::ZERO;
    for (idx, me) in obligations.main.iter().chain(obligations.val.iter()).enumerate() {
        let t_i = dot_u_commitment(&u_vecs, &me.c)?;
        claimed_sum_neo += lambdas[idx] * t_i;
    }

    // Build w_total evaluation list.
    let w_u = neo_ajtai::compute_opening_weights_for_u_seeded(seed, m, &u_vecs);
    let mut w_evals_whir = vec![F::ZERO; z_len_padded];
    let mut w_idx = 0usize;
    for lambda in &lambdas {
        let lambda_whir = neo_f_to_whir(*lambda);
        for &w in &w_u {
            w_evals_whir[w_idx] = lambda_whir * neo_f_to_whir(w);
            w_idx += 1;
        }
    }

    if kind.include_x() {
        let (gamma, claimed_x) =
            mix_in_x_projection(stmt, &commitment_root_u64, &obligations, d, m, &mut w_evals_whir)?;
        claimed_sum_neo += gamma * claimed_x;
    }
    let claimed_sum = neo_f_to_whir(claimed_sum_neo);

    let z_r = u64_to_whir_f(payload.sumcheck.z_r_u64);
    if payload.sumcheck.round_polys_u64.len() != num_vars {
        return Err(ClosureProofError::WhirP3("sumcheck rounds mismatch".into()));
    }

    // Verify sumcheck and fold public weights to obtain w(r).
    let mut claim = claimed_sum;
    let mut w_fold = w_evals_whir;
    let mut rands = Vec::with_capacity(num_vars);
    for (round, g_u64) in payload.sumcheck.round_polys_u64.iter().enumerate() {
        let g0 = u64_to_whir_f(g_u64[0]);
        let g1 = u64_to_whir_f(g_u64[1]);
        let g2 = u64_to_whir_f(g_u64[2]);

        if g0 + g1 != claim {
            return Err(ClosureProofError::WhirP3("sumcheck consistency failed".into()));
        }

        let r = sumcheck_challenge(stmt, &commitment_root_u64, round, g_u64);
        rands.push(r);
        claim = sumcheck_math::eval_quad([g0, g1, g2], r);
        compress_in_place(&mut w_fold, r);
    }
    if w_fold.len() != 1 {
        return Err(ClosureProofError::WhirP3("w folding failed".into()));
    }
    let w_r = w_fold[0];
    if claim != z_r * w_r {
        return Err(ClosureProofError::WhirP3("sumcheck final check failed".into()));
    }
    let mut coords = rands;
    coords.reverse();

    // Verify WHIR evaluation proof: Z(r) == z_r.
    let mut statement = Statement::<EF>::initialize(num_vars);
    statement.add_evaluated_constraint(MultilinearPoint::new(coords), z_r);

    let commitment_reader = CommitmentReader::new(&params_whir);
    let verifier = Verifier::new(&params_whir);

    let mut verifier_state = domainsep.to_verifier_state(proof_data, challenger);
    let parsed_commitment = commitment_reader
        .parse_commitment::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state)
        .map_err(|e| ClosureProofError::WhirP3(format!("parse_commitment failed: {e:?}")))?;

    verifier
        .verify::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state, &parsed_commitment, &statement)
        .map_err(|e| ClosureProofError::WhirP3(format!("verify failed: {e:?}")))?;

    Ok(())
}

pub fn verify_whir_p3_ajtai_opening_only_payload_v1(
    stmt: &ClosureStatementV1,
    payload_bytes: &[u8],
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
) -> Result<(), ClosureProofError> {
    verify_whir_p3_ajtai_opening_payload_v1_impl(
        AjtaiOpeningBackendKindV1::OpeningOnly,
        stmt,
        payload_bytes,
        params,
        ccs,
    )
}

pub fn verify_whir_p3_ajtai_opening_plus_x_payload_v1(
    stmt: &ClosureStatementV1,
    payload_bytes: &[u8],
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
) -> Result<(), ClosureProofError> {
    verify_whir_p3_ajtai_opening_payload_v1_impl(
        AjtaiOpeningBackendKindV1::OpeningPlusX,
        stmt,
        payload_bytes,
        params,
        ccs,
    )
}
