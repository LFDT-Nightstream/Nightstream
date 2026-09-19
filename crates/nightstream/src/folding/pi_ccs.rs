//! Selected PiCCS validation and shared native proving. Copied from neo-fold-clean.
use super::{
    kernels as engine, superneo_has_canonical_x_shape, transcript::Transcript, CcsClaim, CcsWitness, CeClaim, Params,
    RunningInstance, Structure,
};
use neo_math::{D, K};
pub use neo_reductions::api::PiCcsProof as SumcheckProof;
use neo_reductions::{optimized_engine::optimized_prove_with_row_cache, superneo_eval::SuperneoEvalCache};
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("PiCCS shape: {0}")]
    Shape(&'static str),
    #[error("PiCCS auxiliary forwarding mismatch")]
    AdvForwarding,
    #[error(transparent)]
    Engine(#[from] engine::Error),
}
#[derive(Clone, Debug, PartialEq)]
pub struct Proof {
    pub sumcheck: SumcheckProof,
    pub outputs: Vec<CeClaim>,
}
fn reject_auxiliary(fresh: &[CcsClaim], running: &[CeClaim]) -> Result<(), Error> {
    if fresh.iter().any(|c| c.adv.is_some()) || running.iter().any(|c| c.adv.is_some()) {
        return Err(Error::Shape("plain claims cannot carry auxiliary commitments"));
    }
    Ok(())
}
pub(crate) fn prove_from_parts_with_rows(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    cache: &SuperneoEvalCache,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
) -> Result<Proof, Error> {
    reject_auxiliary(fresh_claims, &running.claims)?;
    validate_input_shape(pp, s, fresh_claims, fresh_witnesses, running)?;
    let (mut outputs, sumcheck, _, _) = optimized_prove_with_row_cache(
        tr.inner_mut(),
        pp.inner(),
        s,
        fresh_claims,
        fresh_witnesses,
        &running.claims,
        &running.witnesses,
        cache,
    )
    .map_err(engine::Error::from)?;
    forward_adv(fresh_claims, &running.claims, &mut outputs)?;
    validate_v1_1_claims(s, &outputs)?;
    Ok(Proof { sumcheck, outputs })
}

fn forward_adv(fresh: &[CcsClaim], running: &[CeClaim], outputs: &mut [CeClaim]) -> Result<(), Error> {
    if outputs.len() != fresh.len() + running.len() {
        return Err(Error::Shape("|outputs| \u{2260} K + k in adv forwarding"));
    }
    let inputs = fresh
        .iter()
        .map(|c| &c.adv)
        .chain(running.iter().map(|c| &c.adv));
    for (output, adv) in outputs.iter_mut().zip(inputs) {
        output.adv = adv.clone();
    }
    Ok(())
}

fn validate_adv_forwarding(fresh: &[CcsClaim], running: &[CeClaim], outputs: &[CeClaim]) -> Result<(), Error> {
    if outputs.len() != fresh.len() + running.len() {
        return Err(Error::Shape("|outputs| \u{2260} K + k in adv forwarding"));
    }
    let inputs = fresh
        .iter()
        .map(|c| &c.adv)
        .chain(running.iter().map(|c| &c.adv));
    for (output, adv) in outputs.iter().zip(inputs) {
        if output.adv != *adv {
            return Err(Error::AdvForwarding);
        }
    }
    Ok(())
}

pub fn verify(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    proof: &Proof,
) -> Result<Vec<CeClaim>, Error> {
    reject_auxiliary(fresh_claims, &running.claims)?;
    if proof.outputs.iter().any(|c| c.adv.is_some()) {
        return Err(Error::Shape("plain outputs cannot carry auxiliary commitments"));
    }
    validate_verifier_shape(pp, s, fresh_claims, running, &proof.outputs)?;
    validate_adv_forwarding(fresh_claims, &running.claims, &proof.outputs)?;
    let ok = engine::verify_pi_ccs(
        tr.inner_mut(),
        pp,
        s,
        fresh_claims,
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
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
) -> Result<(), Error> {
    if fresh_claims.is_empty() {
        return Err(Error::Shape("K (fresh) must be \u{2265} 1"));
    }
    validate_fresh_count_within_rlc_guard(pp, fresh_claims.len())?;
    if fresh_claims.len() != fresh_witnesses.len() {
        return Err(Error::Shape("|fresh_claims| \u{2260} |fresh_witnesses|"));
    }
    if !running.prover_shape_is_valid() {
        return Err(Error::Shape("running: |claims| \u{2260} |witnesses|"));
    }
    if !running.is_empty() && running.claims.len() as u32 != pp.k_rho() {
        return Err(Error::Shape("running length does not match params.k_rho()"));
    }
    for (idx, claim) in fresh_claims.iter().enumerate() {
        if claim.m_in > s.m {
            return Err(Error::Shape("fresh m_in exceeds structure.m"));
        }
        if claim.m_in % D != 0 {
            return Err(Error::Shape("fresh m_in must contain whole degree-D ring elements"));
        }
        if claim.x.len() != claim.m_in {
            return Err(Error::Shape("fresh x length does not match m_in"));
        }
        if fresh_witnesses[idx].private_len(claim.m_in, s.m).is_none() {
            return Err(Error::Shape("fresh m_in + witness length must equal structure.m"));
        }
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
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    fold_outputs: &[CeClaim],
) -> Result<(), Error> {
    let running_claims = &running.claims;
    if fresh_claims.is_empty() {
        return Err(Error::Shape("K (fresh) must be \u{2265} 1"));
    }
    validate_fresh_count_within_rlc_guard(pp, fresh_claims.len())?;
    if !running_claims.is_empty() && running_claims.len() as u32 != pp.k_rho() {
        return Err(Error::Shape("running length does not match params.k_rho()"));
    }
    let expected_outputs = fresh_claims.len() + running_claims.len();
    if fold_outputs.len() != expected_outputs {
        return Err(Error::Shape("|fold_outputs| \u{2260} K + k"));
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

fn validate_fresh_count_within_rlc_guard(pp: &Params, fresh_len: usize) -> Result<(), Error> {
    if fresh_len > pp.max_fresh_count() {
        return Err(Error::Shape("K (fresh) exceeds params.max_fresh_count()"));
    }
    Ok(())
}

fn validate_v1_1_claims(s: &Structure, claims: &[CeClaim]) -> Result<(), Error> {
    for claim in claims {
        validate_v1_1_claim(s, claim)?;
    }
    Ok(())
}

fn validate_v1_1_claim(s: &Structure, claim: &CeClaim) -> Result<(), Error> {
    let d_pad = D.next_power_of_two();
    let assignment_width = neo_reductions::common::superneo_carrier_width(s.m);
    let ell_n =
        s.n.max(assignment_width)
            .next_power_of_two()
            .max(2)
            .trailing_zeros() as usize;

    if claim.r.len() != ell_n {
        return Err(Error::Shape("CE r length must match the joint row point"));
    }
    if claim.eval_k.len() != d_pad {
        return Err(Error::Shape("CE Eval_K must use the padded ring degree"));
    }
    if claim
        .eval_k
        .iter()
        .skip(D)
        .any(|&lane| lane != K::default())
    {
        return Err(Error::Shape("CE Eval_K padding lanes must be zero"));
    }
    if claim.eval_a.len() != s.t() {
        return Err(Error::Shape("CE Eval_A count must equal the CCS matrix count"));
    }
    for row in &claim.eval_a {
        if row.len() != d_pad {
            return Err(Error::Shape("CE Eval_A rows must use the padded ring degree"));
        }
        if row.iter().skip(D).any(|&lane| lane != K::default()) {
            return Err(Error::Shape("CE Eval_A padding lanes must be zero"));
        }
    }
    Ok(())
}
