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
        tr, params, s, mcs_list, me_inputs, me_outputs, proof, cache, None,
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
        Some(public_instance_digest),
    )
}

fn optimized_verify_with_cache_and_public_instance_digest_impl(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
    public_instance_digest: Option<[F; 4]>,
) -> Result<(bool, PiCcsVerifyPerf), PiCcsError> {
    let total_started = std::time::Instant::now();
    if mcs_list.is_empty() {
        return Err(PiCcsError::InvalidInput("optimized_verify: empty mcs_list".into()));
    }

    let bind_started = std::time::Instant::now();
    let dims = utils::build_dims_and_policy(params, s)?;
    let bind_header_instances_started = std::time::Instant::now();
    let bind_header_perf = if let Some(public_instance_digest) = public_instance_digest {
        utils::bind_header_and_instance_digest_with_digest(
            tr,
            params,
            s,
            dims,
            cache.mat_digest(),
            &public_instance_digest,
        )?
    } else {
        utils::bind_header_and_instances_with_digest(tr, params, s, mcs_list, dims, cache.mat_digest())?
    };
    let bind_header_instances_ms = bind_header_instances_started.elapsed().as_secs_f64() * 1_000.0;
    let bind_me_inputs_started = std::time::Instant::now();
    utils::bind_me_inputs(tr, me_inputs)?;
    let bind_me_inputs_ms = bind_me_inputs_started.elapsed().as_secs_f64() * 1_000.0;
    let bind_sample_challenges_started = std::time::Instant::now();
    let mut ch = utils::sample_challenges(tr, dims.ell_d, dims.ell)?;
    ch.beta_m = utils::sample_beta_m(tr, dims.ell_m)?;
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

    if proof.variant != PiCcsProofVariant::SplitNcV1 {
        return Err(PiCcsError::ProtocolError("unsupported Π_CCS proof variant".into()));
    }

    let want_rounds_fe = dims
        .ell_n
        .checked_add(dims.ell_d)
        .ok_or_else(|| PiCcsError::ProtocolError("ell_n + ell_d overflow".into()))?;
    let want_rounds_nc = dims.ell_nc;

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
    let claimed_nc = K::ZERO;
    tr.append_fields_raw(&[F::from_u64(utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    tr.append_fields_raw(&claimed_nc.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(crate::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);
    let (r_all_nc, running_sum_nc, ok_nc) =
        crate::sumcheck::verify_sumcheck_rounds_poseidon_v3(tr, dims.d_sc, claimed_nc, &proof.sumcheck_rounds_nc);
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
    let (s_col_prime, alpha_prime_nc) = r_all_nc.split_at(dims.ell_m);
    let nc_sumcheck_ms = nc_sumcheck_started.elapsed().as_secs_f64() * 1_000.0;

    let output_checks_started = std::time::Instant::now();
    let r_inputs = utils::shared_me_input_r(me_inputs, dims.ell_n)?;

    // Strictly enforce NC channel presence and transcript-derived points.
    utils::validate_me_outputs_against_inputs(s, params, mcs_list, me_inputs, me_outputs, r_prime, s_col_prime)?;
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

    let rhs_nc = super::rhs_terminal_identity_nc(params, &ch, s_col_prime, alpha_prime_nc, me_outputs);

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
