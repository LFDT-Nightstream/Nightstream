//! Optimized-engine verifier implementation for Π_CCS.
//!
//! The verifier keeps formula-equivalent RHS assembly while avoiding dependencies on
//! `paper_exact_engine` module paths.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::optimized_engine::{OptimizedStructureCache, PiCcsProof, PiCcsProofVariant, PiCcsVerifyPerf};
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::KExtensions;
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use super::replay_binding::{NcReplayMode, ReplayBinding};
use crate::engines::utils;

/// Optimized verifier implementation.
pub fn optimized_verify(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    let cache = OptimizedStructureCache::build(s)?;
    optimized_verify_with_cache(tr, params, s, mcs_list, me_inputs, me_outputs, proof, &cache)
}

pub fn optimized_verify_with_cache(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
) -> Result<bool, PiCcsError> {
    let (ok, _perf) =
        optimized_verify_with_cache_and_perf(tr, params, s, mcs_list, me_inputs, me_outputs, proof, cache)?;
    Ok(ok)
}

pub fn optimized_verify_with_cache_and_perf(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
) -> Result<(bool, PiCcsVerifyPerf), PiCcsError> {
    optimized_verify_with_cache_and_public_instance_digest_impl(
        tr,
        params,
        s,
        mcs_list,
        me_inputs,
        me_outputs,
        proof,
        cache,
        ReplayBinding::claims(),
    )
}

pub fn optimized_verify_with_cache_and_instance_digest_and_perf(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
    public_instance_digest: [F; 4],
) -> Result<(bool, PiCcsVerifyPerf), PiCcsError> {
    optimized_verify_with_cache_and_public_instance_digest_impl(
        tr,
        params,
        s,
        mcs_list,
        me_inputs,
        me_outputs,
        proof,
        cache,
        ReplayBinding::instance_digest(public_instance_digest),
    )
}

pub fn optimized_verify_with_cache_and_instance_digest_and_me_input_handle_and_perf(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
    public_instance_digest: [F; 4],
    me_input_accumulator_handle: [F; 4],
) -> Result<(bool, PiCcsVerifyPerf), PiCcsError> {
    optimized_verify_with_cache_and_public_instance_digest_impl(
        tr,
        params,
        s,
        mcs_list,
        me_inputs,
        me_outputs,
        proof,
        cache,
        ReplayBinding::legacy_handle(public_instance_digest, me_input_accumulator_handle),
    )
}

pub(super) fn optimized_verify_with_cache_and_public_instance_digest_impl(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
    binding: ReplayBinding,
) -> Result<(bool, PiCcsVerifyPerf), PiCcsError> {
    let total_started = std::time::Instant::now();
    if mcs_list.is_empty() {
        return Err(PiCcsError::InvalidInput("optimized_verify: empty mcs_list".into()));
    }

    let bind_started = std::time::Instant::now();
    let dims = utils::build_dims_and_policy(params, s)?;
    crate::api::validate_mcs_claims("optimized_verify", s, mcs_list)?;
    crate::api::validate_ce_claims_shape("optimized_verify: me_inputs", s, dims.ell_m, me_inputs)?;
    crate::api::validate_ce_claims_shape("optimized_verify: me_outputs", s, dims.ell_m, me_outputs)?;
    crate::api::validate_pi_ccs_outputs("optimized_verify: me_outputs", s, me_outputs)?;
    let _ = utils::shared_me_input_r(me_inputs, dims.ell_n)?;
    let _ = utils::shared_me_input_r(me_outputs, dims.ell_n)?;
    utils::validate_mcs_output_x_recomposition(params, s.m, mcs_list, me_outputs)?;
    let block_pending = match &binding.nc_mode {
        NcReplayMode::LegacyFlat => None,
        NcReplayMode::BlockLaneDelayed(pending) => Some(pending.clone()),
    };
    let block_mode = block_pending.is_some();
    if block_mode && params.b != 2 {
        return Err(PiCcsError::InvalidInput(
            "block-lane delayed Π_CCS requires the strict base-two norm relation".into(),
        ));
    }
    let transcript_variant = if block_mode {
        utils::PiCcsTranscriptVariant::BlockLaneNcDelayedV1
    } else {
        utils::PiCcsTranscriptVariant::SplitNcV1
    };
    let bind_header_instances_started = std::time::Instant::now();
    let bind_header_perf = if let Some(public_instance_digest) = binding.public_instance_digest {
        utils::bind_header_and_instance_digest_with_digest_for_variant(
            tr,
            params,
            s,
            dims,
            cache.mat_digest(),
            &public_instance_digest,
            transcript_variant,
        )?
    } else {
        utils::bind_header_and_instances_with_digest_for_variant(
            tr,
            params,
            s,
            mcs_list,
            dims,
            cache.mat_digest(),
            transcript_variant,
        )?
    };
    let bind_header_instances_ms = bind_header_instances_started.elapsed().as_secs_f64() * 1_000.0;
    let bind_me_inputs_started = std::time::Instant::now();
    if let Some(handle) = binding.me_input_accumulator_handle {
        utils::bind_me_inputs_accumulator_handle(tr, me_inputs.len(), &handle)?;
    } else {
        utils::bind_me_inputs(tr, me_inputs)?;
    }
    let bind_me_inputs_ms = bind_me_inputs_started.elapsed().as_secs_f64() * 1_000.0;
    let bind_sample_challenges_started = std::time::Instant::now();
    let mut ch = utils::sample_challenges(tr, dims.ell_d, dims.ell)?;
    let block_challenges = if block_mode {
        Some(super::block_lane_replay::sample_challenges(tr, dims, &mut ch)?)
    } else {
        ch.beta_m = utils::sample_beta_m(tr, dims.ell_m)?;
        None
    };
    let bind_sample_challenges_ms = bind_sample_challenges_started.elapsed().as_secs_f64() * 1_000.0;
    let bind_ms = bind_started.elapsed().as_secs_f64() * 1_000.0;

    // Compute the public claimed sum T from ME inputs and α
    // (this is the only legitimate initial sum for sumcheck).
    let claimed_initial = super::claimed_initial_sum_from_inputs_with_k_mcs(s, &ch, mcs_list.len(), me_inputs);

    // Optional tightness check: if prover sent a sum, verify it matches T.
    // This helps debug forged proofs.
    if let Some(x) = proof.sc_initial_sum {
        if x != claimed_initial {
            return Err(PiCcsError::SumcheckError(
                "initial sum mismatch: proof claims different value than public T".into(),
            ));
        }
    }

    let expected_variant = if block_mode {
        PiCcsProofVariant::BlockLaneNcDelayedV1
    } else {
        PiCcsProofVariant::SplitNcV1
    };
    if proof.variant != expected_variant {
        return Err(PiCcsError::ProtocolError(
            "Π_CCS proof variant does not match verifier entry point".into(),
        ));
    }

    let want_rounds_fe = dims
        .ell_n
        .checked_add(dims.ell_d)
        .ok_or_else(|| PiCcsError::ProtocolError("ell_n + ell_d overflow".into()))?;
    let want_rounds_nc = if block_mode {
        super::oracle::BLOCK_LANE_NC_BLOCK_VARIABLES + super::oracle::BLOCK_LANE_NC_LANE_VARIABLES
    } else {
        dims.ell_nc
    };

    if proof.sumcheck_rounds.len() != want_rounds_fe {
        return Err(PiCcsError::InvalidInput(format!(
            "split Π_CCS: sumcheck_rounds.len()={}, expected {}",
            proof.sumcheck_rounds.len(),
            want_rounds_fe
        )));
    }
    if proof.sumcheck_rounds_nc.len() != want_rounds_nc {
        return Err(PiCcsError::InvalidInput(format!(
            "split Π_CCS: sumcheck_rounds_nc.len()={}, expected {}",
            proof.sumcheck_rounds_nc.len(),
            want_rounds_nc
        )));
    }

    // -----------------------------
    // FE sumcheck
    // -----------------------------
    let fe_sumcheck_started = std::time::Instant::now();
    tr.append_fields_raw(&[F::from_u64(utils::PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)]);
    tr.append_fields_raw(&[F::from_u64(utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    tr.append_fields_raw(&claimed_initial.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(crate::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);
    let (r_all, running_sum, ok) =
        crate::sumcheck::verify_sumcheck_rounds_poseidon_v3(tr, dims.d_sc, claimed_initial, &proof.sumcheck_rounds);
    if !ok {
        return Err(PiCcsError::SumcheckError("rounds invalid".into()));
    }
    if r_all.len() != want_rounds_fe {
        return Err(PiCcsError::ProtocolError(format!(
            "split Π_CCS: expected {} FE challenges, got {}",
            want_rounds_fe,
            r_all.len()
        )));
    }
    let (r_prime, alpha_prime) = r_all.split_at(dims.ell_n);
    let fe_sumcheck_ms = fe_sumcheck_started.elapsed().as_secs_f64() * 1_000.0;

    // -----------------------------
    // NC-only sumcheck
    // -----------------------------
    let nc_sumcheck_started = std::time::Instant::now();
    tr.append_fields_raw(&[F::from_u64(utils::PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)]);
    let claimed_nc = block_challenges.as_ref().map_or(K::ZERO, |challenges| {
        super::block_lane_terminal::claimed_initial(challenges, block_pending.as_ref().and_then(Option::as_ref))
    });
    if proof
        .sc_initial_sum_nc
        .is_some_and(|claimed| claimed != claimed_nc)
    {
        return Err(PiCcsError::SumcheckError(
            "NC initial sum does not match the verifier-computed delayed claim".into(),
        ));
    }
    tr.append_fields_raw(&[F::from_u64(utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    tr.append_fields_raw(&claimed_nc.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(crate::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);
    let nc_degree = if block_mode {
        super::oracle::BLOCK_LANE_NC_ROUND_COEFFICIENTS - 1
    } else {
        dims.d_sc
    };
    let (r_all_nc, running_sum_nc, ok_nc) =
        crate::sumcheck::verify_sumcheck_rounds_poseidon_v3(tr, nc_degree, claimed_nc, &proof.sumcheck_rounds_nc);
    if !ok_nc {
        return Err(PiCcsError::SumcheckError("NC rounds invalid".into()));
    }
    if r_all_nc.len() != want_rounds_nc {
        return Err(PiCcsError::ProtocolError(format!(
            "split Π_CCS: expected {} NC challenges, got {}",
            want_rounds_nc,
            r_all_nc.len()
        )));
    }
    let nc_point_variables = if block_mode {
        super::oracle::BLOCK_LANE_NC_BLOCK_VARIABLES
    } else {
        dims.ell_m
    };
    let (s_col_prime, alpha_prime_nc) = r_all_nc.split_at(nc_point_variables);
    let nc_sumcheck_ms = nc_sumcheck_started.elapsed().as_secs_f64() * 1_000.0;

    let output_checks_started = std::time::Instant::now();
    let r_inputs = utils::shared_me_input_r(me_inputs, dims.ell_n)?;

    // Strictly enforce NC channel presence and transcript-derived points.
    utils::validate_me_outputs_against_inputs(s, params, mcs_list, me_inputs, me_outputs, r_prime, s_col_prime)?;
    if block_mode
        && me_outputs.iter().any(|output| {
            output.y_zcol[neo_math::D..]
                .iter()
                .any(|value| *value != K::ZERO)
        })
    {
        return Err(PiCcsError::ProtocolError(
            "block-lane Π_CCS output contains nonzero lane padding".into(),
        ));
    }
    let output_checks_ms = output_checks_started.elapsed().as_secs_f64() * 1_000.0;

    // RHS assembly (FE-only; NC is verified separately)
    let terminal_started = std::time::Instant::now();
    let rhs = super::rhs_terminal_identity_fe_with_k_mcs(
        s,
        params,
        &ch,
        r_prime,
        alpha_prime,
        me_outputs,
        mcs_list.len(),
        r_inputs,
    );

    let rhs_nc = if let Some(challenges) = block_challenges.as_ref() {
        super::block_lane_terminal::rhs(
            challenges,
            s_col_prime,
            alpha_prime_nc,
            me_outputs,
            mcs_list.len(),
            block_pending.as_ref().and_then(Option::as_ref),
        )?
    } else {
        super::rhs_terminal_identity_nc(params, &ch, s_col_prime, alpha_prime_nc, me_outputs)
    };

    let ok_fe = running_sum == rhs;
    let ok_nc = running_sum_nc == rhs_nc;

    #[cfg(feature = "debug-logs")]
    if !(ok_fe && ok_nc) {
        eprintln!("\n[verify] split Π_CCS mismatch:");
        eprintln!("[verify]   FE: running_sum={:?}", running_sum);
        eprintln!("[verify]   FE: rhs        ={:?}", rhs);
        eprintln!("[verify]   NC: running_sum={:?}", running_sum_nc);
        eprintln!("[verify]   NC: rhs        ={:?}", rhs_nc);
        eprintln!("[verify]   ok_fe={}, ok_nc={}", ok_fe, ok_nc);
        eprintln!(
            "[verify]   sizes: k_mcs={}, k_me_in={}, k_out={}",
            mcs_list.len(),
            me_inputs.len(),
            me_outputs.len()
        );
    }

    let perf = PiCcsVerifyPerf {
        bind_ms,
        bind_header_instances_ms,
        bind_header_prefix_ms: bind_header_perf.prefix_ms,
        bind_header_poly_ms: bind_header_perf.poly_ms,
        bind_header_public_instances_ms: bind_header_perf.public_instances_ms,
        bind_me_inputs_ms,
        bind_sample_challenges_ms,
        fe_sumcheck_ms,
        nc_sumcheck_ms,
        output_checks_ms,
        terminal_ms: terminal_started.elapsed().as_secs_f64() * 1_000.0,
        total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
    };

    Ok((ok_fe && ok_nc, perf))
}
