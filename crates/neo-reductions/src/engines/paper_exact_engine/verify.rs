//! Paper-exact verify implementation for PiCcsEngine.
//!
//! This module contains the verify logic for the paper-exact engine,
//! which validates the sumcheck proof using paper-exact RHS assembly.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::optimized_engine::PiCcsProof;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, McsInstance, MeInstance};
use neo_math::KExtensions;
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;

/// Paper-exact verify implementation.
///
/// This function verifies the sumcheck proof using the paper-exact
/// RHS terminal identity evaluation.
pub fn paper_exact_verify(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_outputs: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    let dims = crate::engines::utils::build_dims_and_policy(params, s)?;
    crate::engines::utils::bind_header_and_instances(tr, params, s, mcs_list, dims.ell, dims.d_sc, 0)?;
    crate::engines::utils::bind_me_inputs(tr, me_inputs)?;
    let ch = crate::engines::utils::sample_challenges(tr, dims.ell_d, dims.ell)?;

    // Compute the public claimed sum T from ME inputs and α
    // (this is the only legitimate initial sum for sumcheck).
    let claimed_initial = crate::paper_exact_engine::claimed_initial_sum_from_inputs(s, &ch, me_inputs);

    // Optional tightness check: if prover sent a sum, verify it matches T.
    // This helps debug forged proofs.
    if let Some(x) = proof.sc_initial_sum {
        if x != claimed_initial {
            return Err(PiCcsError::SumcheckError(
                "initial sum mismatch: proof claims different value than public T".into(),
            ));
        }
    }

    // Bind T and run the sumcheck verification
    tr.append_fields(b"sumcheck/initial_sum", &claimed_initial.as_coeffs());
    let (r_all, running_sum, ok) =
        crate::sumcheck::verify_sumcheck_rounds(tr, dims.d_sc, claimed_initial, &proof.sumcheck_rounds);
    if !ok {
        return Err(PiCcsError::SumcheckError("rounds invalid".into()));
    }

    let (r_prime, alpha_prime) = r_all.split_at(dims.ell_n);

    // Validate ME input r (if provided)
    for (idx, me) in me_inputs.iter().enumerate() {
        if me.r.len() != dims.ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "ME input r length mismatch at accumulator #{}: expected ell_n = {}, got {}",
                idx,
                dims.ell_n,
                me.r.len()
            )));
        }
    }

    // Paper-exact RHS assembly
    let rhs = crate::paper_exact_engine::rhs_terminal_identity_paper_exact(
        s,
        params,
        &ch,
        r_prime,
        alpha_prime,
        me_outputs,
        me_inputs.first().map(|mi| mi.r.as_slice()),
    );
    Ok(running_sum == rhs)
}
