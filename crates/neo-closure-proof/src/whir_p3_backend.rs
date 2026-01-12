//! WHIR (Plonky3-based) backend glue for `ClosureProofV1::OpaqueBytes`.
//!
//! This module currently contains a single dev-milestone backend: **WHIR full closure**.
//!
//! It proves:
//! - Ajtai commitment openings (batched),
//! - boundedness of the witness matrices `Z`,
//! - ME consistency (and bus openings when a `BusLayout` is provided).
//!
//! It is not production-sized yet: it still materializes large `2^n` evaluation tables, and
//! currently serializes explicit obligations in the payload (dev profile; backend id `5`).

#![forbid(unsafe_code)]

mod seed;
mod sumcheck_math;
mod weights_claims;

pub(crate) use sumcheck_math::eval_lagrange_0_to_deg;
pub(crate) use weights_claims::{compute_full_closure_public_weights_and_claims, FullClosurePublicWeightsAndClaims};

use crate::contract;
use crate::{opaque, ClosureProofError, ClosureStatementV1};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
pub(crate) use seed::{derive_seed_v1, fixed_seed};

use neo_ajtai::Commitment as NeoCmt;
use neo_math::{D as NeoD, F as NeoF};
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

use whir_p3::storage::{Buffer, MmapBuffer, DEFAULT_MMAP_THRESHOLD_BYTES};

pub(crate) const WHIR_P3_DIGEST_ELEMS: usize = 8;
// Security profile (matches the ~128-bit target in the Neo paper).
const WHIR_SECURITY_LEVEL_BITS: usize = 128;
pub(crate) const MAX_WHIR_PROOF_DATA_U64: usize = 1 << 20; // 1M u64 limbs (~8 MiB)
pub(crate) const MAX_SUMCHECK_ROUNDS: usize = 64;
pub(crate) const MAX_SUMCHECK_EVALS_U64_PER_ROUND: usize = 1 << 14; // 16k u64 limbs

/// Hard safety limit for the current WHIR backends.
///
/// These backends are intended to support very large `Z` (e.g. `m=2^24`). They still do linear
/// passes over large evaluation tables, but (when supported by lower layers) storage can be backed
/// by `mmap` rather than RAM.
const MAX_DEV_Z_EVALS_PADDED: usize = 1 << 31; // 2B (very large; primarily to avoid accidental overflow)

pub(crate) type F = Goldilocks;
pub(crate) type EF = F;
pub(crate) type Perm = Poseidon2Goldilocks<16>;
pub(crate) type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
pub(crate) type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
pub(crate) type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

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

pub(crate) fn domain_separator_for_stmt(
    params: &WhirConfig<EF, F, MyHash, MyCompress, MyChallenger>,
    stmt: &ClosureStatementV1,
) -> DomainSeparator<EF, F> {
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

    // Commitments first so the sumcheck Fiat–Shamir point can be bound to both commitment roots.
    ds.commit_statement::<_, _, _, WHIR_P3_DIGEST_ELEMS>(params); // Z commitment
    ds.commit_statement::<_, _, _, WHIR_P3_DIGEST_ELEMS>(params); // W commitment

    // Proofs/openings (at the sumcheck-derived point) come after both commitments.
    ds.add_whir_proof::<_, _, _, WHIR_P3_DIGEST_ELEMS>(params); // Z(r)
    ds.add_whir_proof::<_, _, _, WHIR_P3_DIGEST_ELEMS>(params); // W(r)

    ds
}

pub(crate) fn make_params(num_variables: usize) -> WhirConfig<EF, F, MyHash, MyCompress, MyChallenger> {
    let mut rng = ChaCha8Rng::from_seed(fixed_seed(b"merkle_perm"));
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);

    // For large `Z` (e.g. `m=2^24`), the first-round folding factor has an outsized impact on
    // Merkle tree height and DFT locality. Use a higher folding factor when possible.
    //
    // This is still a dev backend; production should pin/tune these parameters explicitly.
    let folding_factor = num_variables.clamp(2, 10);
    let whir_params = ProtocolParameters {
        initial_statement: true,
        security_level: WHIR_SECURITY_LEVEL_BITS,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(folding_factor),
        merkle_hash,
        merkle_compress,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        univariate_skip: false,
    };

    let mut cfg = WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params);
    // NOTE: WHIR's parameter synthesis may derive PoW bit counts that exceed what a 64-bit prime
    // field can support (and `p3-challenger` panics when asked to grind for >=64 bits). Since this
    // crate's WHIR backends are currently dev milestones, we disable PoW grinding entirely.
    cfg.max_pow_bits = 0;
    cfg.starting_folding_pow_bits = 0;
    for r in cfg.round_parameters.iter_mut() {
        r.pow_bits = 0;
        r.folding_pow_bits = 0;
    }
    cfg.final_pow_bits = 0;
    cfg.final_folding_pow_bits = 0;
    cfg
}

pub(crate) fn encode_proof_data(proof_data: &[F]) -> Vec<u64> {
    proof_data.iter().map(|x| x.as_canonical_u64()).collect()
}

pub(crate) fn whir_f_from_canonical_u64(x: u64) -> Result<F, ClosureProofError> {
    let f = F::from_u64(x);
    if f.as_canonical_u64() != x {
        return Err(ClosureProofError::InvalidOpaqueProofEncoding);
    }
    Ok(f)
}

pub(crate) fn decode_proof_data_u64_checked(u64s: &[u64]) -> Result<Vec<F>, ClosureProofError> {
    let mut out = Vec::with_capacity(u64s.len());
    for &x in u64s {
        out.push(whir_f_from_canonical_u64(x)?);
    }
    Ok(out)
}

pub(crate) fn extract_commitment_root_u64_from_proof_data(
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

pub(crate) fn extract_two_commitment_roots_u64_from_proof_data(
    params: &WhirConfig<EF, F, MyHash, MyCompress, MyChallenger>,
    domainsep: &DomainSeparator<EF, F>,
    challenger: MyChallenger,
    proof_data: &[F],
) -> Result<(Vec<u64>, Vec<u64>), ClosureProofError> {
    let commitment_reader = CommitmentReader::new(params);
    let mut verifier_state = domainsep.to_verifier_state(proof_data.to_vec(), challenger);

    let parsed_z = commitment_reader
        .parse_commitment::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state)
        .map_err(|e| ClosureProofError::WhirP3(format!("parse_commitment(Z) failed: {e:?}")))?;
    let root_z = parsed_z
        .root
        .as_ref()
        .iter()
        .map(|x| x.as_canonical_u64())
        .collect();

    let parsed_w = commitment_reader
        .parse_commitment::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state)
        .map_err(|e| ClosureProofError::WhirP3(format!("parse_commitment(W) failed: {e:?}")))?;
    let root_w = parsed_w
        .root
        .as_ref()
        .iter()
        .map(|x| x.as_canonical_u64())
        .collect();

    Ok((root_z, root_w))
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

// -------------------------------------------------------------------------------------------------
// Helpers shared by the full-closure backend
// -------------------------------------------------------------------------------------------------

use crate::encoded::EncodedObligations;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct SumcheckProofV2 {
    /// Initial sumcheck claim (canonical u64 limb).
    ///
    /// Round 0 must satisfy `g(0) + g(1) == claimed_sum`. This anchors the sumcheck transcript.
    pub(crate) claimed_sum_u64: u64,
    /// For each round, the prover sends g(0), g(1), ..., g(deg) as canonical u64 limbs.
    ///
    /// For the full-closure backend, `deg = 2*b` (mixing in a range-check term).
    pub(crate) round_evals_u64: BoundedVec<BoundedVec<u64, MAX_SUMCHECK_EVALS_U64_PER_ROUND>, MAX_SUMCHECK_ROUNDS>,
    pub(crate) z_r_u64: u64,
    pub(crate) w_r_u64: u64,
}

pub(crate) fn next_pow2_checked(n: usize) -> Result<usize, ClosureProofError> {
    n.checked_next_power_of_two()
        .ok_or_else(|| ClosureProofError::WhirP3("next_pow2 overflow".into()))
}

pub(crate) fn sumcheck_challenge_full(
    stmt: &ClosureStatementV1,
    commitment_root_z_u64: &[u64],
    commitment_root_w_u64: &[u64],
    round_idx: usize,
    g_evals_u64: &[u64],
) -> F {
    let mut h = blake3::Hasher::new();
    h.update(b"neo/closure-proof/whir-p3/full-closure/sumcheck/chal/v1");
    h.update(&stmt.context_digest);
    h.update(&stmt.pp_id_digest);
    h.update(&stmt.obligations_digest);
    for &u in commitment_root_z_u64 {
        h.update(&u.to_le_bytes());
    }
    h.update(b"root_w");
    for &u in commitment_root_w_u64 {
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

pub(crate) fn range_vanishing_poly(z: F, base_b: u32) -> F {
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

pub(crate) fn eq_poly_value(point: &[F], r0: &[F]) -> Result<F, ClosureProofError> {
    if point.len() != r0.len() {
        return Err(ClosureProofError::WhirP3("eq_poly_value: point length mismatch".into()));
    }
    let mut acc = F::ONE;
    for (&x, &r) in point.iter().zip(r0.iter()) {
        // eq(x, r) = r*x + (1-r)*(1-x)
        acc *= r * x + (F::ONE - r) * (F::ONE - x);
    }
    Ok(acc)
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

fn build_eq_evals(r0: &[F]) -> EvaluationsList<F> {
    let mut out = EvaluationsList::<F>::new_zeroed(1usize << r0.len());
    let out_slice = out.as_mut_slice();
    out_slice[0] = F::ONE;

    let mut cur = 1usize;
    for &ri in r0.iter().rev() {
        let one_minus = F::ONE - ri;
        for j in (0..cur).rev() {
            let v = out_slice[j];
            out_slice[j] = v * one_minus;
            out_slice[j + cur] = v * ri;
        }
        cur <<= 1;
    }

    out
}

pub(crate) fn prove_sumcheck_full_closure(
    stmt: &ClosureStatementV1,
    commitment_root_z_u64: &[u64],
    commitment_root_w_u64: &[u64],
    z_evals: &[F],
    mut w_evals: EvaluationsList<F>,
    r0: &[F],
    delta_range: F,
    base_b: u32,
    claimed_sum: F,
) -> SumcheckProofV2 {
    assert_eq!(z_evals.len(), w_evals.num_evals());
    assert!(z_evals.len().is_power_of_two());
    assert_eq!(z_evals.len(), 1usize << r0.len());

    let deg = 2usize * (base_b as usize);
    let n_vars = z_evals.len().ilog2() as usize;
    let mut claim = claimed_sum;
    let mut round_evals_u64 = Vec::with_capacity(n_vars);

    let mut eq_evals = build_eq_evals(r0);
    assert_eq!(eq_evals.num_evals(), z_evals.len());

    if n_vars == 0 {
        let z_r = *z_evals.first().unwrap_or(&F::ZERO);
        let w_r = *w_evals.as_slice().first().unwrap_or(&F::ZERO);
        return SumcheckProofV2 {
            claimed_sum_u64: whir_f_to_u64(claimed_sum),
            round_evals_u64: BoundedVec::from_vec_panicking(Vec::new()),
            z_r_u64: whir_f_to_u64(z_r),
            w_r_u64: whir_f_to_u64(w_r),
        };
    }

    let mut z_work = EvaluationsList::<F>::new_zeroed(z_evals.len() / 2);

    for round in 0..n_vars {
        let z_slice: &[F] = if round == 0 { z_evals } else { z_work.as_slice() };
        let w_slice = w_evals.as_slice();
        let eq_slice = eq_evals.as_slice();

        let mut g_evals = vec![F::ZERO; deg + 1];

        for ((z_pair, w_pair), eq_pair) in z_slice
            .chunks_exact(2)
            .zip(w_slice.chunks_exact(2))
            .zip(eq_slice.chunks_exact(2))
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

        let r = sumcheck_challenge_full(stmt, commitment_root_z_u64, commitment_root_w_u64, round, &g_u64);
        claim = sumcheck_math::eval_lagrange_0_to_deg(&g_evals, r);

        w_evals.compress(r);
        eq_evals.compress(r);

        if round == 0 {
            let dst = z_work.as_mut_slice();
            for (i, z_pair) in z_evals.chunks_exact(2).enumerate() {
                dst[i] = r * (z_pair[1] - z_pair[0]) + z_pair[0];
            }
        } else {
            z_work.compress(r);
        }
    }

    debug_assert_eq!(z_work.num_evals(), 1);
    let z_r = z_work.as_slice()[0];
    debug_assert_eq!(w_evals.num_evals(), 1);
    let w_r = w_evals.as_slice()[0];

    SumcheckProofV2 {
        claimed_sum_u64: whir_f_to_u64(claimed_sum),
        round_evals_u64: BoundedVec::from_vec_panicking(
            round_evals_u64
                .into_iter()
                .map(BoundedVec::from_vec_panicking)
                .collect::<Vec<_>>(),
        ),
        z_r_u64: whir_f_to_u64(z_r),
        w_r_u64: whir_f_to_u64(w_r),
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
        return Err(ClosureProofError::WhirP3(
            "unexpected d (must match neo_math::D)".into(),
        ));
    }
    let m = ccs.m;

    // Enforce that the loaded seeded PP matches the statement's pp_id_digest.
    let (kappa, pp_seed) = contract::require_global_pp_matches_statement(stmt.pp_id_digest, params, d, m)
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

    let mut z_poly = EvaluationsList::<F>::new_zeroed(z_len_padded);
    {
        let z_out = z_poly.as_mut_slice();
        let mut fill_idx = 0usize;
        for Z in main_wits.iter().chain(val_wits.iter()) {
            if Z.rows() != d || Z.cols() != m {
                return Err(ClosureProofError::WhirP3("Z shape mismatch".into()));
            }
            for row in 0..d {
                for col in 0..m {
                    z_out[fill_idx] = neo_f_to_whir(Z[(row, col)]);
                    fill_idx += 1;
                }
            }
        }
    }

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
        .commit::<WHIR_P3_DIGEST_ELEMS>(&dft_committer, &mut prover_state, z_poly)
        .map_err(|e| ClosureProofError::WhirP3(format!("WHIR commit failed: {e:?}")))?;

    // Extract the commitment root limbs by parsing the WHIR commitment prefix.
    // This avoids relying on `proof_data` internal layout beyond WHIR's `parse_commitment` API.
    let commitment_root_z_u64 = extract_commitment_root_u64_from_proof_data(
        &params_whir,
        &domainsep,
        challenger.clone(),
        prover_state.proof_data(),
    )?;

    // Commit to the deterministic weight table W before deriving the sumcheck challenge point, so
    // the Fiat–Shamir point is bound to both Z and W commitments.
    let weights_claims_commit = weights_claims::compute_full_closure_public_weights_and_claims(
        stmt,
        params,
        ccs,
        obligations,
        d,
        m,
        kappa,
        pp_seed,
        &commitment_root_z_u64,
        z_len_padded,
        num_vars,
        bus,
    )?;
    let weights_claims::FullClosurePublicWeightsAndClaims { w_evals, .. } = weights_claims_commit;

    let witness_w = committer
        .commit::<WHIR_P3_DIGEST_ELEMS>(&dft_committer, &mut prover_state, w_evals)
        .map_err(|e| ClosureProofError::WhirP3(format!("WHIR commit W failed: {e:?}")))?;

    let (commitment_root_z_u64_check, commitment_root_w_u64) = extract_two_commitment_roots_u64_from_proof_data(
        &params_whir,
        &domainsep,
        challenger.clone(),
        prover_state.proof_data(),
    )?;
    if commitment_root_z_u64_check != commitment_root_z_u64 {
        return Err(ClosureProofError::WhirP3("commitment root Z drift after committing W".into()));
    }

    // Recompute weights/claims for the sumcheck (the sumcheck prover consumes `w_evals` by folding
    // it in-place, so we keep it separate from the committed W table).
    let weights_claims_sumcheck = weights_claims::compute_full_closure_public_weights_and_claims(
        stmt,
        params,
        ccs,
        obligations,
        d,
        m,
        kappa,
        pp_seed,
        &commitment_root_z_u64,
        z_len_padded,
        num_vars,
        bus,
    )?;
    let weights_claims::FullClosurePublicWeightsAndClaims {
        claimed_sum,
        delta_range,
        r0,
        w_evals,
    } = weights_claims_sumcheck;

    // Prove the combined sumcheck for:
    //   Σ_x [ Z(x)*W(x) + δ_range*Eq(x,r0)*Range(Z(x)) ] == claimed_sum.
    let sumcheck = prove_sumcheck_full_closure(
        stmt,
        &commitment_root_z_u64,
        &commitment_root_w_u64,
        witness.polynomial.as_slice(),
        w_evals,
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
        let r = sumcheck_challenge_full(stmt, &commitment_root_z_u64, &commitment_root_w_u64, round, g_u64);
        let evals: Vec<F> = g_u64.iter().copied().map(u64_to_whir_f).collect();
        claim = sumcheck_math::eval_lagrange_0_to_deg(&evals, r);
        rands.push(r);
    }
    let mut coords = rands.clone();
    coords.reverse();

    let z_r = u64_to_whir_f(sumcheck.z_r_u64);
    let w_r = u64_to_whir_f(sumcheck.w_r_u64);
    let point = MultilinearPoint::new(coords);

    let mut statement_z = Statement::<EF>::initialize(num_vars);
    statement_z.add_evaluated_constraint(point.clone(), z_r);

    let dft_prover = EvalsDft::<F>::default();
    let prover = Prover(&params_whir);
    prover
        .prove::<WHIR_P3_DIGEST_ELEMS>(&dft_prover, &mut prover_state, statement_z, witness)
        .map_err(|e| ClosureProofError::WhirP3(format!("WHIR prove failed: {e:?}")))?;

    let mut statement_w = Statement::<EF>::initialize(num_vars);
    statement_w.add_evaluated_constraint(point, w_r);

    prover
        .prove::<WHIR_P3_DIGEST_ELEMS>(&dft_prover, &mut prover_state, statement_w, witness_w)
        .map_err(|e| ClosureProofError::WhirP3(format!("WHIR prove W failed: {e:?}")))?;

    let payload = WhirP3FullClosurePayloadV1 {
        obligations: EncodedObligations::encode(obligations),
        sumcheck,
        whir_proof_data_u64: BoundedVec::try_from_vec(encode_proof_data(prover_state.proof_data()))
            .map_err(|_| ClosureProofError::WhirP3("WHIR proof_data_u64 too large".into()))?,
    };
    let payload_bytes = serialize_payload(&payload)?;
    opaque::encode_envelope(opaque::BackendIdV1::WhirP3FullClosureV1.as_u32(), &payload_bytes)
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
        return Err(ClosureProofError::WhirP3(
            "unexpected d (must match neo_math::D)".into(),
        ));
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
    let (kappa, pp_seed) = contract::require_global_pp_matches_statement(stmt.pp_id_digest, params, d, m)
        .map_err(ClosureProofError::WhirP3)?;

    let proof_data = decode_proof_data_u64_checked(&payload.whir_proof_data_u64)?;
    let params_whir = make_params(num_vars);
    let domainsep = domain_separator_for_stmt(&params_whir, stmt);
    let mut rng = ChaCha8Rng::from_seed(fixed_seed(b"challenger_perm"));
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
    let commitment_reader = CommitmentReader::new(&params_whir);
    let verifier = Verifier::new(&params_whir);
    let mut verifier_state = domainsep.to_verifier_state(proof_data, challenger);

    // Parse Z and W commitment prefixes and extract roots (used for deterministic challenges).
    let parsed_commitment_z = commitment_reader
        .parse_commitment::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state)
        .map_err(|e| ClosureProofError::WhirP3(format!("parse_commitment(Z) failed: {e:?}")))?;
    let commitment_root_z_u64: Vec<u64> = parsed_commitment_z
        .root
        .as_ref()
        .iter()
        .map(|x| x.as_canonical_u64())
        .collect();

    let parsed_commitment_w = commitment_reader
        .parse_commitment::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state)
        .map_err(|e| ClosureProofError::WhirP3(format!("parse_commitment(W) failed: {e:?}")))?;
    let commitment_root_w_u64: Vec<u64> = parsed_commitment_w
        .root
        .as_ref()
        .iter()
        .map(|x| x.as_canonical_u64())
        .collect();

    let weights_claims = weights_claims::compute_full_closure_public_claims_and_rng(
        stmt,
        params,
        ccs,
        &obligations,
        d,
        m,
        kappa,
        &commitment_root_z_u64,
        num_vars,
        bus,
    )?;

    // Verify sumcheck.
    let deg = 2usize * (params.b as usize);
    if payload.sumcheck.round_evals_u64.len() != num_vars {
        return Err(ClosureProofError::WhirP3("sumcheck rounds mismatch".into()));
    }
    let claimed_sum = whir_f_from_canonical_u64(payload.sumcheck.claimed_sum_u64)?;
    if claimed_sum != weights_claims.claimed_sum {
        return Err(ClosureProofError::WhirP3("sumcheck claimed_sum mismatch".into()));
    }
    let mut claim = claimed_sum;
    let mut rands = Vec::with_capacity(num_vars);
    for (round, g_u64) in payload.sumcheck.round_evals_u64.iter().enumerate() {
        if g_u64.len() != deg + 1 {
            return Err(ClosureProofError::WhirP3("sumcheck degree mismatch".into()));
        }
        let g0 = whir_f_from_canonical_u64(g_u64[0])?;
        let g1 = whir_f_from_canonical_u64(g_u64[1])?;
        if g0 + g1 != claim {
            return Err(ClosureProofError::WhirP3("sumcheck consistency failed".into()));
        }
        let r = sumcheck_challenge_full(stmt, &commitment_root_z_u64, &commitment_root_w_u64, round, g_u64);
        let evals: Vec<F> = g_u64
            .iter()
            .copied()
            .map(whir_f_from_canonical_u64)
            .collect::<Result<_, _>>()?;
        claim = sumcheck_math::eval_lagrange_0_to_deg(&evals, r);
        rands.push(r);
    }

    let z_r = whir_f_from_canonical_u64(payload.sumcheck.z_r_u64)?;
    let w_r = whir_f_from_canonical_u64(payload.sumcheck.w_r_u64)?;

    let mut coords = rands.clone();
    coords.reverse();
    let eq_r = eq_poly_value(&coords, &weights_claims.r0)?;
    let rng_r = range_vanishing_poly(z_r, params.b);

    if claim != z_r * w_r + weights_claims.delta_range * eq_r * rng_r {
        return Err(ClosureProofError::WhirP3("sumcheck final check failed".into()));
    }

    // Verify WHIR evaluation proofs: Z(r) == z_r and W(r) == w_r.
    let point = MultilinearPoint::new(coords);

    let mut stmt_z = Statement::<EF>::initialize(num_vars);
    stmt_z.add_evaluated_constraint(point.clone(), z_r);
    verifier
        .verify::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state, &parsed_commitment_z, &stmt_z)
        .map_err(|e| ClosureProofError::WhirP3(format!("verify Z failed: {e:?}")))?;

    let mut stmt_w = Statement::<EF>::initialize(num_vars);
    stmt_w.add_evaluated_constraint(point, w_r);
    verifier
        .verify::<WHIR_P3_DIGEST_ELEMS>(&mut verifier_state, &parsed_commitment_w, &stmt_w)
        .map_err(|e| ClosureProofError::WhirP3(format!("verify W failed: {e:?}")))?;

    // Bind the committed W table to the obligations-derived weight computation.
    //
    // This is intentionally a dev-backend check: obligations are public in backend id `5`, so we
    // can recompute the expected opened evaluation `W(r)` and compare it against the commitment
    // opening, without materializing the full `2^n` weights table in memory.
    //
    // This closes a critical soundness gap where a malicious prover could previously commit to an
    // arbitrary W table and satisfy the sumcheck algebraically.
    let mut coords = rands;
    coords.reverse();
    let w_r_expected = weights_claims::compute_full_closure_public_w_r_expected_at_point(
        stmt,
        params,
        ccs,
        &obligations,
        d,
        m,
        kappa,
        pp_seed,
        &commitment_root_z_u64,
        z_len_padded,
        num_vars,
        &coords,
        bus,
    )?;
    if w_r_expected != w_r {
        return Err(ClosureProofError::WhirP3("W(r) mismatch vs obligations-derived weights".into()));
    }

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
                return Err(ClosureProofError::WhirP3(format!("ME y_scalars mismatch at j={j}")));
            }
        }
    }

    Ok(())
}
