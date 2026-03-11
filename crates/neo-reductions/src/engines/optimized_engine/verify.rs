//! Optimized-engine verifier implementation for Π_CCS.
//!
//! The verifier keeps formula-equivalent RHS assembly while avoiding dependencies on
//! `paper_exact_engine` module paths.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::optimized_engine::{PiCcsProof, PiCcsProofVariant};
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_gpu::ProverComputeBackend;
use neo_math::KExtensions;
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
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
    optimized_verify_with_backend(
        tr,
        params,
        s,
        mcs_list,
        me_inputs,
        me_outputs,
        proof,
        &ProverComputeBackend::Cpu,
    )
}

/// Optimized verifier implementation with an explicit compute backend.
pub fn optimized_verify_with_backend(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    compute_backend: &ProverComputeBackend,
) -> Result<bool, PiCcsError> {
    let backend_ctx = crate::accelerator::BackendContext::new(compute_backend)?;
    optimized_verify_with_context(tr, params, s, mcs_list, me_inputs, me_outputs, proof, &backend_ctx)
}

/// Optimized verifier implementation with a reusable backend context.
pub fn optimized_verify_with_context(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    backend_ctx: &crate::accelerator::BackendContext,
) -> Result<bool, PiCcsError> {
    if mcs_list.is_empty() {
        return Err(PiCcsError::InvalidInput("optimized_verify: empty mcs_list".into()));
    }

    let dims = utils::build_dims_and_policy(params, s)?;
    utils::bind_header_and_instances(tr, params, s, mcs_list, dims)?;
    utils::bind_me_inputs_with_context(tr, me_inputs, backend_ctx)?;
    let mut ch = utils::sample_challenges(tr, dims.ell_d, dims.ell)?;
    ch.beta_m = utils::sample_beta_m(tr, dims.ell_m)?;

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
    tr.append_message(b"sumcheck/fe", b"");
    tr.append_fields(b"sumcheck/initial_sum", &claimed_initial.as_coeffs());
    let (r_all, running_sum, ok) =
        crate::sumcheck::verify_sumcheck_rounds(tr, dims.d_sc, claimed_initial, &proof.sumcheck_rounds);
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

    // -----------------------------
    // NC-only sumcheck
    // -----------------------------
    tr.append_message(b"sumcheck/nc", b"");
    let claimed_nc = K::ZERO;
    tr.append_fields(b"sumcheck/initial_sum", &claimed_nc.as_coeffs());
    let (r_all_nc, running_sum_nc, ok_nc) =
        crate::sumcheck::verify_sumcheck_rounds(tr, dims.d_sc, claimed_nc, &proof.sumcheck_rounds_nc);
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

    let r_inputs = utils::shared_me_input_r(me_inputs, dims.ell_n)?;

    // Strictly enforce NC channel presence and transcript-derived points.
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

        // Outputs must be aligned with their corresponding input instances.
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

    utils::validate_ct_constant_term(s, params, me_outputs)?;
    // MCS-derived outputs must expose X consistent with public x.
    utils::validate_mcs_output_x_recomposition(params, s.m, mcs_list, me_outputs)?;

    // RHS assembly (FE-only; NC is verified separately)
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

    Ok(ok_fe && ok_nc)
}
