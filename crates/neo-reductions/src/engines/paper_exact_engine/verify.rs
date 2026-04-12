//! Paper-exact verify implementation for PiCcsEngine.
//!
//! This module contains the verify logic for the paper-exact engine,
//! which validates the sumcheck proof using paper-exact RHS assembly.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::optimized_engine::{PiCcsProof, PiCcsProofVariant};
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::KExtensions;
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

/// Paper-exact verify implementation.
///
/// This function verifies the sumcheck proof using the paper-exact
/// RHS terminal identity evaluation.
pub fn paper_exact_verify(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    if mcs_list.is_empty() {
        return Err(PiCcsError::InvalidInput("paper_exact_verify: empty mcs_list".into()));
    }

    let dims = crate::engines::utils::build_dims_and_policy(params, s)?;
    crate::engines::utils::bind_header_and_instances(tr, params, s, mcs_list, dims)?;
    crate::engines::utils::bind_me_inputs(tr, me_inputs)?;
    let mut ch = crate::engines::utils::sample_challenges(tr, dims.ell_d, dims.ell)?;
    ch.beta_m = crate::engines::utils::sample_beta_m(tr, dims.ell_m)?;

    // Compute the public claimed sum T from ME inputs and α
    // (this is the only legitimate initial sum for sumcheck).
    let claimed_initial =
        crate::paper_exact_engine::claimed_initial_sum_from_inputs_with_k_mcs(s, &ch, mcs_list.len(), me_inputs);

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

    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)]);
    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
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

    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)]);
    let claimed_nc = K::ZERO;
    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
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

    let r_inputs = crate::engines::utils::shared_me_input_r(me_inputs, dims.ell_n)?;

    let d_pad = 1usize
        .checked_shl(dims.ell_d as u32)
        .ok_or_else(|| PiCcsError::ProtocolError("d_pad shift overflow".into()))?;
    let want_outputs = mcs_list
        .len()
        .checked_add(me_inputs.len())
        .ok_or_else(|| PiCcsError::ProtocolError("mcs_list.len() + me_inputs.len() overflow".into()))?;
    if me_outputs.len() != want_outputs {
        return Err(PiCcsError::InvalidInput(format!(
            "split Π_CCS: me_outputs.len()={}, expected {} (= |mcs_list| + |me_inputs|)",
            me_outputs.len(),
            want_outputs
        )));
    }
    for (idx, out) in me_outputs.iter().enumerate() {
        if out.r.as_slice() != r_prime {
            return Err(PiCcsError::ProtocolError(format!(
                "split Π_CCS: me_outputs[{idx}].r does not match FE r'"
            )));
        }
        if out.s_col.as_slice() != s_col_prime {
            return Err(PiCcsError::ProtocolError(format!(
                "split Π_CCS: me_outputs[{idx}].s_col does not match NC s'"
            )));
        }
        if out.y_zcol.len() != d_pad {
            return Err(PiCcsError::ProtocolError(format!(
                "split Π_CCS: me_outputs[{idx}].y_zcol.len()={}, expected {}",
                out.y_zcol.len(),
                d_pad
            )));
        }

        if idx < mcs_list.len() {
            let inst = &mcs_list[idx];
            if out.c != inst.c {
                return Err(PiCcsError::ProtocolError(format!(
                    "split Π_CCS: me_outputs[{idx}].c does not match mcs_list[{idx}].c"
                )));
            }
            if out.m_in != inst.m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "split Π_CCS: me_outputs[{idx}].m_in={}, expected {}",
                    out.m_in, inst.m_in
                )));
            }
            if inst.x.len() != inst.m_in {
                return Err(PiCcsError::InvalidInput(format!(
                    "split Π_CCS: mcs_list[{idx}].x.len()={}, expected m_in={}",
                    inst.x.len(),
                    inst.m_in
                )));
            }
            if out.X.rows() != D || out.X.cols() != inst.m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "split Π_CCS: me_outputs[{idx}].X shape mismatch (got {}×{}, expected {}×{})",
                    out.X.rows(),
                    out.X.cols(),
                    D,
                    inst.m_in
                )));
            }
        } else {
            let me_idx = idx - mcs_list.len();
            let inp = &me_inputs[me_idx];
            if out.c != inp.c {
                return Err(PiCcsError::ProtocolError(format!(
                    "split Π_CCS: me_outputs[{idx}].c does not match me_inputs[{me_idx}].c"
                )));
            }
            if out.m_in != inp.m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "split Π_CCS: me_outputs[{idx}].m_in={}, expected {}",
                    out.m_in, inp.m_in
                )));
            }
            if out.X != inp.X {
                return Err(PiCcsError::ProtocolError(format!(
                    "split Π_CCS: me_outputs[{idx}].X does not match me_inputs[{me_idx}].X"
                )));
            }
        }
    }

    crate::engines::utils::validate_ct_constant_term(s, params, me_outputs)?;
    // MCS-derived outputs must expose X consistent with public x.
    crate::engines::utils::validate_mcs_output_x_recomposition(params, s.m, mcs_list, me_outputs)?;

    let rhs = crate::paper_exact_engine::rhs_terminal_identity_fe_paper_exact_with_k_mcs(
        s,
        params,
        &ch,
        r_prime,
        alpha_prime,
        me_outputs,
        mcs_list.len(),
        r_inputs,
    );
    let rhs_nc = crate::paper_exact_engine::rhs_terminal_identity_nc_paper_exact(
        params,
        &ch,
        s_col_prime,
        alpha_prime_nc,
        me_outputs,
    );
    Ok(running_sum == rhs && running_sum_nc == rhs_nc)
}
