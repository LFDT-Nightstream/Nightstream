//! Selected PiRLC. Public verification recomputes every parent coordinate.
//! Legacy projection advice used a cloned transcript and is not part of this path.
use super::{
    has_zero_evaluation_padding, kernels as engine, superneo_has_canonical_x_shape, transcript::Transcript, CeClaim,
    Params, RlcMixer, Structure, EVALUATION_WIDTH,
};
use neo_ccs::Mat;
use neo_math::F;
use p3_field::PrimeField64;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("PiRLC requires nonempty inputs")]
    Shape,
    #[error("PiRLC witness count: {claims} claims, {witnesses} witnesses")]
    WitnessMismatch { claims: usize, witnesses: usize },
    #[error("PiRLC parent does not match")]
    VerifyRejected,
    #[error("PiRLC frame differs")]
    FoldDigest,
    #[error("PiRLC noncanonical frame in {owner} lane {lane}")]
    FoldDigestCanonicality { owner: &'static str, lane: usize },
    #[error("PiRLC public shape in {0}")]
    NoncanonicalXShape(&'static str),
    #[error("PiRLC point shape in {0}")]
    RShape(&'static str),
    #[error("PiRLC input points differ")]
    RConsistency,
    #[error("PiRLC evaluation shape in {0}")]
    EvaluationShape(&'static str),
    #[error("PiRLC evaluation padding in {0}")]
    EvaluationPadding(&'static str),
    #[error("plain claims cannot carry auxiliary commitments")]
    Auxiliary,
    #[error(transparent)]
    Engine(#[from] engine::Error),
}
#[derive(Clone, Debug, PartialEq)]
pub struct Proof {
    pub combined: CeClaim,
}
pub(crate) struct Output {
    pub claim: CeClaim,
    pub witness: Mat<F>,
}
pub(crate) fn prove_refs(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mix: RlcMixer,
    claims: &[CeClaim],
    witnesses: &[&Mat<F>],
) -> Result<(Output, Proof), Error> {
    validate_input_shape(claims, witnesses)?;
    validate_inputs_before_rho(s, claims)?;
    let rhos = engine::sample_rho_n(tr.inner_mut(), pp, claims.len())?;
    let (combined, witness) = engine::prove_pi_rlc_refs(pp, s, &rhos, claims, witnesses, mix)?;
    validate_combined_claim(s, claims, &combined)?;
    Ok((
        Output {
            claim: combined.clone(),
            witness,
        },
        Proof { combined },
    ))
}
/// PiRLC with an accelerator-owned parent witness. `mix_witnesses` returns
/// the accelerator's form of `Σ ρ_i·Z_i`; the public claim is canonical.
#[cfg(feature = "metal")]
pub(crate) fn prove_refs_resident<R>(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mix: RlcMixer,
    claims: &[CeClaim],
    witnesses: &[&Mat<F>],
    mix_witnesses: impl Fn(&[Mat<F>], &[&Mat<F>]) -> R,
) -> Result<(CeClaim, R, Proof), Error> {
    validate_input_shape(claims, witnesses)?;
    validate_inputs_before_rho(s, claims)?;
    let rhos = engine::sample_rho_n(tr.inner_mut(), pp, claims.len())?;
    let (combined, resident) = engine::prove_pi_rlc_refs_resident(pp, s, &rhos, claims, witnesses, mix, mix_witnesses)?;
    validate_combined_claim(s, claims, &combined)?;
    Ok((combined.clone(), resident, Proof { combined }))
}
pub(crate) fn verify(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mix: RlcMixer,
    claims: &[CeClaim],
    proof: &Proof,
) -> Result<CeClaim, Error> {
    validate_inputs_before_rho(s, claims)?;
    let rhos = engine::sample_rho_n(tr.inner_mut(), pp, claims.len())?;
    validate_combined_claim(s, claims, &proof.combined)?;
    if !engine::verify_pi_rlc(pp, s, &rhos, claims, &proof.combined, mix)? {
        return Err(Error::VerifyRejected);
    }
    Ok(proof.combined.clone())
}
fn validate_inputs_before_rho(s: &Structure, inputs: &[CeClaim]) -> Result<(), Error> {
    if inputs.is_empty() {
        return Err(Error::Shape);
    }
    for input in inputs {
        if input.adv.is_some() {
            return Err(Error::Auxiliary);
        }
        validate_fold_digest_canonical("input", input)?;
        validate_canonical_x_shape_one("input", input)?;
        validate_r_shape_one("input", s, input)?;
        validate_evaluation_shape_one("input", s, input)?;
        validate_evaluation_padding_zero_one("input", input)?;
    }
    Ok(())
}
fn validate_combined_claim(s: &Structure, inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    if combined.adv.is_some() {
        return Err(Error::Auxiliary);
    }
    validate_fold_digest_canonical("combined", combined)?;
    validate_canonical_x_shape(inputs, combined)?;
    validate_r_shape(s, inputs, combined)?;
    validate_r_consistency(inputs, combined)?;
    validate_evaluation_shape(s, inputs, combined)?;
    validate_evaluation_padding_zero(inputs, combined)?;
    validate_fold_digest_consistency(inputs, combined)
}
fn validate_input_shape(claims: &[CeClaim], witnesses: &[&Mat<F>]) -> Result<(), Error> {
    if claims.is_empty() {
        return Err(Error::Shape);
    }
    if claims.len() != witnesses.len() {
        return Err(Error::WitnessMismatch {
            claims: claims.len(),
            witnesses: witnesses.len(),
        });
    }
    Ok(())
}

fn validate_canonical_x_shape(inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        validate_canonical_x_shape_one("input", input)?;
    }
    validate_canonical_x_shape_one("combined", combined)?;
    Ok(())
}

fn validate_canonical_x_shape_one(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    if !superneo_has_canonical_x_shape(&claim.X, claim.m_in) {
        return Err(Error::NoncanonicalXShape(owner));
    }
    Ok(())
}

fn validate_r_shape(s: &crate::folding::Structure, inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        validate_r_shape_one("input", s, input)?;
    }
    validate_r_shape_one("combined", s, combined)?;
    Ok(())
}

fn validate_r_shape_one(owner: &'static str, s: &crate::folding::Structure, claim: &CeClaim) -> Result<(), Error> {
    let assignment_width = neo_reductions::common::superneo_carrier_width(s.m);
    let expected = s
        .domain_rows()
        .max(assignment_width)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    if claim.r.len() != expected {
        return Err(Error::RShape(owner));
    }
    Ok(())
}

fn validate_r_consistency(inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        if input.r != combined.r {
            return Err(Error::RConsistency);
        }
    }
    Ok(())
}

fn validate_evaluation_shape(
    s: &crate::folding::Structure,
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<(), Error> {
    for input in inputs {
        validate_evaluation_shape_one("input", s, input)?;
    }
    validate_evaluation_shape_one("combined", s, combined)?;
    Ok(())
}

fn validate_evaluation_shape_one(
    owner: &'static str,
    s: &crate::folding::Structure,
    claim: &CeClaim,
) -> Result<(), Error> {
    if claim.eval_k.len() != EVALUATION_WIDTH
        || claim.eval_a.len() != s.t()
        || claim.eval_a.iter().any(|row| row.len() != EVALUATION_WIDTH)
    {
        return Err(Error::EvaluationShape(owner));
    }
    Ok(())
}

fn validate_evaluation_padding_zero(inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        validate_evaluation_padding_zero_one("input", input)?;
    }
    validate_evaluation_padding_zero_one("combined", combined)?;
    Ok(())
}

fn validate_evaluation_padding_zero_one(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    if !has_zero_evaluation_padding(&claim.eval_k)
        || !claim
            .eval_a
            .iter()
            .all(|row| has_zero_evaluation_padding(row))
    {
        return Err(Error::EvaluationPadding(owner));
    }
    Ok(())
}

fn validate_fold_digest_consistency(inputs: &[CeClaim], combined: &CeClaim) -> Result<(), Error> {
    for input in inputs {
        if input.fold_digest != combined.fold_digest {
            return Err(Error::FoldDigest);
        }
    }
    Ok(())
}

fn validate_fold_digest_canonical(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    for (lane, chunk) in claim.fold_digest.chunks_exact(8).enumerate() {
        let value = u64::from_le_bytes(chunk.try_into().expect("fold_digest lanes are 8 bytes"));
        if value >= F::ORDER_U64 {
            return Err(Error::FoldDigestCanonicality { owner, lane });
        }
    }
    Ok(())
}
