//! WHIR (Plonky3-based) backend glue for an obligations-private Phase-2 closure proof.
//!
//! This is the **production target** backend: obligations must be kept private (no payload
//! obligations), and the verifier must be able to check using only:
//! - `ClosureStatementV1`,
//! - pinned context (`NeoParams`, CCS, optional `BusLayout`),
//! - proof bytes.
//!
//! NOTE: This backend is not yet production-audit-ready until it proves the missing
//! obligations→(weights, claimed_sum) binding described in
//! `docs/spartan-compression-phase2-obligations-private.md`.

#![forbid(unsafe_code)]

use crate::bounded::BoundedVec;
use crate::codec::{deserialize_payload, serialize_payload};
use crate::contract;
use crate::opaque;
use crate::whir_p3_backend::{
    decode_proof_data_u64_checked, eval_lagrange_0_to_deg, next_pow2_checked, range_vanishing_poly,
    sumcheck_challenge_full, whir_f_from_canonical_u64, MyChallenger, Perm, SumcheckProofV2, F, EF,
    MAX_WHIR_PROOF_DATA_U64, WHIR_P3_DIGEST_ELEMS,
};
use crate::{ClosureProofError, ClosureStatementV1};
use neo_ajtai::Commitment as NeoCmt;
use neo_math::D as NeoD;
use p3_field::{PrimeCharacteristicRing as _, PrimeField64 as _};
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use whir_p3::{
    dft::EvalsDft,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::{reader::CommitmentReader, writer::CommitmentWriter},
        constraints::statement::Statement,
        prover::Prover,
        verifier::Verifier,
    },
};

// Keep this aligned with `digest_binding::MAX_DIGEST_BINDING_PROOF_BYTES`.
const MAX_DIGEST_BINDING_PROOF_BYTES: usize = 16 * 1024 * 1024; // 16 MiB

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct WhirP3PrivateFullClosurePayloadV1 {
    /// Proof that private obligations hash to the public `stmt.obligations_digest`.
    ///
    /// Milestone 3 deliverable in `docs/spartan-compression-phase2-obligations-private.md`.
    digest_binding_proof: BoundedVec<u8, MAX_DIGEST_BINDING_PROOF_BYTES>,
    /// Sumcheck proof for:
    ///   Σ_x [ Z(x)*W(x) + δ_range*Eq(x,r0)*Range(Z(x)) ] == claimed_sum.
    ///
    /// This is the same format as the dev backend (id `5`).
    sumcheck: SumcheckProofV2,
    /// WHIR transcript/proof data as canonical u64 limbs of WHIR-field elements.
    whir_proof_data_u64: BoundedVec<u64, MAX_WHIR_PROOF_DATA_U64>,
}

pub fn prove_whir_p3_private_full_closure_bytes_v1(
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
                    z_out[fill_idx] = Z[(row, col)];
                    fill_idx += 1;
                }
            }
        }
    }

    // WHIR parameters + transcript domain separator.
    let params_whir = crate::whir_p3_backend::make_params(num_vars);
    let domainsep = crate::whir_p3_backend::domain_separator_for_stmt(&params_whir, stmt);

    let mut rng = ChaCha8Rng::from_seed(crate::whir_p3_backend::fixed_seed(b"challenger_perm"));
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    // Commitment phase (does not depend on statement points).
    let committer = CommitmentWriter::new(&params_whir);
    let dft_committer = EvalsDft::<F>::default();
    let witness = committer
        .commit::<WHIR_P3_DIGEST_ELEMS>(&dft_committer, &mut prover_state, z_poly)
        .map_err(|e| ClosureProofError::WhirP3(format!("WHIR commit failed: {e:?}")))?;

    // Extract the commitment root limbs by parsing the WHIR commitment prefix.
    let commitment_root_z_u64 = crate::whir_p3_backend::extract_commitment_root_u64_from_proof_data(
        &params_whir,
        &domainsep,
        challenger.clone(),
        prover_state.proof_data(),
    )?;

    // Commit to the deterministic weight table W before deriving the sumcheck Fiat–Shamir point.
    let weights_claims_commit = crate::whir_p3_backend::compute_full_closure_public_weights_and_claims(
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
    let crate::whir_p3_backend::FullClosurePublicWeightsAndClaims { w_evals, .. } = weights_claims_commit;

    let witness_w = committer
        .commit::<WHIR_P3_DIGEST_ELEMS>(&dft_committer, &mut prover_state, w_evals)
        .map_err(|e| ClosureProofError::WhirP3(format!("WHIR commit W failed: {e:?}")))?;

    let (commitment_root_z_u64_check, commitment_root_w_u64) =
        crate::whir_p3_backend::extract_two_commitment_roots_u64_from_proof_data(
            &params_whir,
            &domainsep,
            challenger.clone(),
            prover_state.proof_data(),
        )?;
    if commitment_root_z_u64_check != commitment_root_z_u64 {
        return Err(ClosureProofError::WhirP3("commitment root Z drift after committing W".into()));
    }

    // Recompute weights/claims for sumcheck (the sumcheck prover consumes `w_evals` by folding it in-place).
    let weights_claims_sumcheck = crate::whir_p3_backend::compute_full_closure_public_weights_and_claims(
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
    let crate::whir_p3_backend::FullClosurePublicWeightsAndClaims {
        claimed_sum,
        delta_range,
        r0,
        w_evals,
    } = weights_claims_sumcheck;

    let sumcheck = crate::whir_p3_backend::prove_sumcheck_full_closure(
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

    // Milestone 3/4: digest binding proof, extended to also bind the sumcheck claimed_sum to the
    // same private obligations witness.
    let digest_binding_proof = crate::prove_obligations_digest_binding_proof_v1(
        stmt,
        params,
        ccs,
        obligations,
        &commitment_root_z_u64,
        sumcheck.claimed_sum_u64,
    )?;

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
        let g0 = F::from_u64(g_u64[0]);
        let g1 = F::from_u64(g_u64[1]);
        if g0 + g1 != claim {
            return Err(ClosureProofError::WhirP3("sumcheck consistency failed".into()));
        }
        let r = sumcheck_challenge_full(stmt, &commitment_root_z_u64, &commitment_root_w_u64, round, g_u64);
        let evals: Vec<F> = g_u64.iter().copied().map(F::from_u64).collect();
        claim = eval_lagrange_0_to_deg(&evals, r);
        rands.push(r);
    }
    let mut coords = rands;
    coords.reverse();

    let z_r = F::from_u64(sumcheck.z_r_u64);
    let w_r = F::from_u64(sumcheck.w_r_u64);
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

    let payload = WhirP3PrivateFullClosurePayloadV1 {
        digest_binding_proof: BoundedVec::try_from_vec(digest_binding_proof)
            .map_err(|_| ClosureProofError::WhirP3("digest_binding_proof too large".into()))?,
        sumcheck,
        whir_proof_data_u64: BoundedVec::try_from_vec(crate::whir_p3_backend::encode_proof_data(prover_state.proof_data()))
            .map_err(|_| ClosureProofError::WhirP3("WHIR proof_data_u64 too large".into()))?,
    };
    let payload_bytes = serialize_payload(&payload)?;
    opaque::encode_envelope(opaque::BackendIdV1::WhirP3PrivateFullClosureV1.as_u32(), &payload_bytes)
}

pub fn verify_whir_p3_private_full_closure_payload_v1(
    stmt: &ClosureStatementV1,
    payload_bytes: &[u8],
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<(), ClosureProofError> {
    let payload: WhirP3PrivateFullClosurePayloadV1 = deserialize_payload(payload_bytes)?;

    // Decode the digest-binding shape first (without verifying the Spartan2 proof) so we can
    // size the WHIR instance and parse commitment roots.
    let shape_unverified = crate::digest_binding::decode_obligations_digest_binding_shape_v1(&payload.digest_binding_proof)?;

    let d = params.d as usize;
    if d != NeoD {
        return Err(ClosureProofError::WhirP3(
            "unexpected d (must match neo_math::D)".into(),
        ));
    }
    let m = ccs.m;

    // Enforce that the loaded seeded PP matches the statement's pp_id_digest.
    contract::require_global_pp_matches_statement(stmt.pp_id_digest, params, d, m).map_err(ClosureProofError::WhirP3)?;

    let obligation_count = (shape_unverified.main_len as usize)
        .checked_add(shape_unverified.val_len as usize)
        .ok_or_else(|| ClosureProofError::WhirP3("obligation_count overflow".into()))?;
    let z_len = obligation_count
        .checked_mul(d)
        .and_then(|x| x.checked_mul(m))
        .ok_or_else(|| ClosureProofError::WhirP3("z_len overflow".into()))?;
    let z_len_padded = next_pow2_checked(z_len.max(1))?;
    let num_vars = z_len_padded.ilog2() as usize;

    // Decode WHIR proof data and verify commitments/openings for Z(r) and W(r).
    let proof_data = decode_proof_data_u64_checked(&payload.whir_proof_data_u64)?;
    let params_whir = crate::whir_p3_backend::make_params(num_vars);
    let domainsep = crate::whir_p3_backend::domain_separator_for_stmt(&params_whir, stmt);

    let mut rng = ChaCha8Rng::from_seed(crate::whir_p3_backend::fixed_seed(b"challenger_perm"));
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

    // Milestone 3/4: verify the digest-binding Spartan2 proof *after* parsing the Z commitment
    // root, since the claimed_sum binding coefficients are derived from `(stmt, root_z)`.
    let (shape, claimed_sum_u64) = crate::digest_binding::verify_obligations_digest_binding_proof_v1_with_shape_and_claimed_sum(
        stmt,
        params,
        ccs,
        &commitment_root_z_u64,
        &payload.digest_binding_proof,
    )?;
    if shape != shape_unverified {
        return Err(ClosureProofError::WhirP3("digest-binding shape mismatch vs early decode".into()));
    }

    // Context sanity: ensure bus layout matches the obligation shape (prevents silent drift).
    let core_t = ccs.t();
    let y_len = shape.y_len as usize;
    if y_len < core_t {
        return Err(ClosureProofError::WhirP3("digest-binding shape y_len < ccs.t()".into()));
    }
    let bus_cols = y_len - core_t;
    match bus {
        None => {
            if bus_cols != 0 {
                return Err(ClosureProofError::WhirP3(
                    "digest-binding shape implies bus openings but no BusLayout provided".into(),
                ));
            }
        }
        Some(bus) => {
            if bus.bus_cols != bus_cols {
                return Err(ClosureProofError::WhirP3("BusLayout bus_cols mismatch vs digest-binding shape".into()));
            }
            if bus.m != m {
                return Err(ClosureProofError::WhirP3("BusLayout m mismatch vs CCS".into()));
            }
            if bus.m_in != shape.m_in as usize {
                return Err(ClosureProofError::WhirP3("BusLayout m_in mismatch vs digest-binding shape".into()));
            }
        }
    }

    // Range-check RNG (deterministic, statement-bound).
    let mut rng = ChaCha8Rng::from_seed(crate::whir_p3_backend::derive_seed_v1(
        b"full_closure/range_rng",
        stmt,
        Some(&commitment_root_z_u64),
    ));
    let mut delta_range = F::from_u64(rng.next_u64());
    if delta_range == F::ZERO {
        delta_range = F::ONE;
    }

    let mut r0 = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        r0.push(F::from_u64(rng.next_u64()));
    }

    // Verify sumcheck.
    let deg = 2usize * (params.b as usize);
    if payload.sumcheck.round_evals_u64.len() != num_vars {
        return Err(ClosureProofError::WhirP3("sumcheck rounds mismatch".into()));
    }

    // Milestone 4 (partial): bind the sumcheck claim to the same private obligations witness via
    // the digest-binding proof. The remaining work is to similarly bind the committed/opened `W`.
    if claimed_sum_u64 != payload.sumcheck.claimed_sum_u64 {
        return Err(ClosureProofError::WhirP3("digest-binding claimed_sum mismatch vs sumcheck".into()));
    }
    let mut claim = whir_f_from_canonical_u64(payload.sumcheck.claimed_sum_u64)?;
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
        claim = eval_lagrange_0_to_deg(&evals, r);
        rands.push(r);
    }

    let z_r = whir_f_from_canonical_u64(payload.sumcheck.z_r_u64)?;
    let w_r = whir_f_from_canonical_u64(payload.sumcheck.w_r_u64)?;

    let mut coords = rands;
    coords.reverse();
    let eq_r = crate::whir_p3_backend::eq_poly_value(&coords, &r0)?;
    let rng_r = range_vanishing_poly(z_r, params.b);

    let expected = z_r * w_r + delta_range * eq_r * rng_r;
    if claim != expected {
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

    let _ = bus; // currently unused in obligations-private verifier
    Ok(())
}
