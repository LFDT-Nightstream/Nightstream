//! Selected PiCCS validation and shared native proving. Copied from neo-fold-clean.
use super::{
    has_zero_evaluation_padding, kernels as engine, superneo_has_canonical_x_shape, transcript::Transcript, CcsClaim,
    CcsWitness, CeClaim, Params, RunningInstance, Structure, EVALUATION_WIDTH,
};
use neo_math::D;
pub use neo_reductions::api::PiCcsProof as SumcheckProof;
use neo_reductions::{optimized_engine::optimized_prove_with_matrix_rows, superneo_eval::MatrixRows};
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("PiCCS shape: {0}")]
    Shape(&'static str),
    #[error(transparent)]
    Engine(#[from] engine::Error),
}
#[derive(Clone, Debug, PartialEq)]
pub struct Proof {
    pub sumcheck: SumcheckProof,
    pub outputs: Vec<CeClaim>,
}
fn reject_auxiliary(fresh: &CcsClaim, running: &[CeClaim]) -> Result<(), Error> {
    if fresh.adv.is_some() || running.iter().any(|c| c.adv.is_some()) {
        return Err(Error::Shape("plain claims cannot carry auxiliary commitments"));
    }
    Ok(())
}
pub(crate) fn prove_from_parts_with_rows(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    rows: &dyn MatrixRows,
    workspace_bytes: usize,
    fresh_claim: &CcsClaim,
    fresh_witness: &CcsWitness,
    running: &RunningInstance,
) -> Result<Proof, Error> {
    reject_auxiliary(fresh_claim, &running.claims)?;
    validate_input_shape(pp, s, fresh_claim, fresh_witness, running)?;
    let (outputs, sumcheck, _, _) = optimized_prove_with_matrix_rows(
        tr.inner_mut(),
        pp.inner(),
        s,
        std::slice::from_ref(fresh_claim),
        std::slice::from_ref(fresh_witness),
        &running.claims,
        &running.witnesses,
        rows,
        workspace_bytes,
        None,
    )
    .map_err(engine::Error::from)?;
    validate_v1_1_claims(s, &outputs)?;
    Ok(Proof { sumcheck, outputs })
}

pub fn verify(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    fresh_claim: &CcsClaim,
    running: &RunningInstance,
    proof: &Proof,
) -> Result<Vec<CeClaim>, Error> {
    reject_auxiliary(fresh_claim, &running.claims)?;
    if proof.outputs.iter().any(|c| c.adv.is_some()) {
        return Err(Error::Shape("plain outputs cannot carry auxiliary commitments"));
    }
    validate_verifier_shape(pp, s, running, &proof.outputs)?;
    let ok = engine::verify_pi_ccs(
        tr.inner_mut(),
        pp,
        s,
        fresh_claim,
        running,
        &proof.outputs,
        &proof.sumcheck,
    )?;
    if !ok {
        return Err(Error::Shape("engine returned false on verify"));
    }
    Ok(proof.outputs.clone())
}

fn validate_canonical_x_shape(claims: &[CeClaim], label: &'static str) -> Result<(), Error> {
    for claim in claims {
        if !superneo_has_canonical_x_shape(&claim.X, claim.m_in) {
            return Err(Error::Shape(label));
        }
    }
    Ok(())
}

fn validate_input_shape(
    pp: &Params,
    s: &Structure,
    fresh_claim: &CcsClaim,
    fresh_witness: &CcsWitness,
    running: &RunningInstance,
) -> Result<(), Error> {
    if !running.prover_shape_is_valid() {
        return Err(Error::Shape("running: |claims| \u{2260} |witnesses|"));
    }
    if !running.is_empty() && running.claims.len() as u32 != pp.k_rho() {
        return Err(Error::Shape("running length does not match params.k_rho()"));
    }
    if fresh_claim.m_in > s.m {
        return Err(Error::Shape("fresh m_in exceeds structure.m"));
    }
    if fresh_claim.m_in % D != 0 {
        return Err(Error::Shape("fresh m_in must contain whole degree-D ring elements"));
    }
    if fresh_claim.x.len() != fresh_claim.m_in {
        return Err(Error::Shape("fresh x length does not match m_in"));
    }
    if fresh_witness.private_len(fresh_claim.m_in, s.m).is_none() {
        return Err(Error::Shape("fresh m_in + witness length must equal structure.m"));
    }
    validate_canonical_x_shape(
        &running.claims,
        "running X must use the canonical coefficient embedding",
    )?;
    validate_v1_1_claims(s, &running.claims)?;
    Ok(())
}

fn validate_verifier_shape(
    pp: &Params,
    s: &Structure,
    running: &RunningInstance,
    fold_outputs: &[CeClaim],
) -> Result<(), Error> {
    let running_claims = &running.claims;
    if !running_claims.is_empty() && running_claims.len() as u32 != pp.k_rho() {
        return Err(Error::Shape("running length does not match params.k_rho()"));
    }
    if fold_outputs.len() != 1 + running_claims.len() {
        return Err(Error::Shape("|fold_outputs| \u{2260} 1 + k"));
    }
    validate_canonical_x_shape(running_claims, "running X must use the canonical coefficient embedding")?;
    validate_canonical_x_shape(
        fold_outputs,
        "fold output X must use the canonical coefficient embedding",
    )?;
    validate_v1_1_claims(s, running_claims)?;
    validate_v1_1_claims(s, fold_outputs)?;
    Ok(())
}

fn validate_v1_1_claims(s: &Structure, claims: &[CeClaim]) -> Result<(), Error> {
    for claim in claims {
        validate_v1_1_claim(s, claim)?;
    }
    Ok(())
}

fn validate_v1_1_claim(s: &Structure, claim: &CeClaim) -> Result<(), Error> {
    let assignment_width = neo_reductions::common::superneo_carrier_width(s.m);
    let ell_n = s
        .domain_rows()
        .max(assignment_width)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;

    if claim.r.len() != ell_n {
        return Err(Error::Shape("CE r length must match the joint row point"));
    }
    if claim.eval_k.len() != EVALUATION_WIDTH {
        return Err(Error::Shape("CE Eval_K must use the padded ring degree"));
    }
    if !has_zero_evaluation_padding(&claim.eval_k) {
        return Err(Error::Shape("CE Eval_K padding lanes must be zero"));
    }
    if claim.eval_a.len() != s.t() {
        return Err(Error::Shape("CE Eval_A count must equal the CCS matrix count"));
    }
    for row in &claim.eval_a {
        if row.len() != EVALUATION_WIDTH {
            return Err(Error::Shape("CE Eval_A rows must use the padded ring degree"));
        }
        if !has_zero_evaluation_padding(row) {
            return Err(Error::Shape("CE Eval_A padding lanes must be zero"));
        }
    }
    Ok(())
}
