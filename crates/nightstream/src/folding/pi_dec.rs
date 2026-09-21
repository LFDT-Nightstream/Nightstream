//! Selected radix-two decomposition under the existing production key.
use super::{
    ajtai_dec_mixer, kernels as engine, superneo_has_canonical_x_shape, superneo_public_x_cols, CeClaim, DecMixer,
    Params, Structure,
};
use neo_ajtai::nightstream_fprime_setup::{
    commit_production_signed_unit_prefix_matrices, PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_VERIFIER_ROWS,
};
use neo_ccs::Mat;
use neo_math::{balanced::within_nc_bound, D, F, K};
use neo_reductions::superneo_eval::SuperneoEvalCache;
use p3_field::PrimeField64;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("PiDEC child count {got}, expected {expected}")]
    ChildCount { expected: usize, got: usize },
    #[error("PiDEC public recomposition failed")]
    VerifyRejected,
    #[error("PiDEC public shape in {0}")]
    NoncanonicalXShape(&'static str),
    #[error("PiDEC child public input exceeds unit norm")]
    ChildXLowNorm,
    #[error("PiDEC frame differs")]
    FoldDigest,
    #[error("PiDEC noncanonical frame in {owner} lane {lane}")]
    FoldDigestCanonicality { owner: &'static str, lane: usize },
    #[error("PiDEC point shape in {0}")]
    RShape(&'static str),
    #[error("PiDEC evaluation shape in {0}")]
    EvaluationShape(&'static str),
    #[error("PiDEC evaluation padding in {0}")]
    EvaluationPadding(&'static str),
    #[error("plain claims cannot carry auxiliary commitments")]
    Auxiliary,
    #[error(transparent)]
    Engine(#[from] engine::Error),
    #[error(transparent)]
    ProductionCommitment(#[from] neo_ajtai::AjtaiError),
}
pub(crate) struct Children {
    pub claims: Vec<CeClaim>,
    pub witnesses: Vec<Mat<F>>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Proof {
    pub children: Vec<CeClaim>,
}
pub(crate) fn prove_with_production_key(
    pp: &Params,
    s: &Structure,
    cache: &SuperneoEvalCache,
    parent: &CeClaim,
    parent_witness: Mat<F>,
) -> Result<(Children, Proof), Error> {
    let production = Params::production();
    let blocks = s.m.div_ceil(D);
    if pp.b() != production.b()
        || pp.k_rho() != production.k_rho()
        || pp.big_b() != production.big_b()
        || u64::from(pp.inner().kappa) != PRODUCTION_VERIFIER_ROWS
        || blocks == 0
        || blocks > PRODUCTION_MESSAGE_COLUMNS as usize
        || cache.relation_shape() != Some((s.n, blocks * D, s.t()))
    {
        return Err(engine::Error::from(neo_reductions::PiCcsError::InvalidInput(
            "selected PiDEC production key or row-cache shape mismatch".into(),
        ))
        .into());
    }
    if parent.adv.is_some() {
        return Err(Error::Auxiliary);
    }
    neo_reductions::common::validate_superneo_witness_mat(&parent_witness, s.m).map_err(engine::Error::from)?;
    let (digits, flags) =
        neo_reductions::common::split_b_matrix_k_with_nonzero_flags(&parent_witness, pp.k_rho() as usize, pp.b())
            .map_err(engine::Error::from)?;
    // The split owns every coefficient needed by commitments and openings.
    drop(parent_witness);
    let commitments = commit_production_signed_unit_prefix_matrices(&digits).map_err(|error| error.into_error())?;
    let (children, ok_y, ok_x, ok_c) =
        neo_reductions::api::dec_children_with_commit_superneo_cached_from_trusted_split_digits(
            neo_reductions::api::FoldingMode::Optimized,
            s,
            pp.inner(),
            parent,
            &digits,
            &flags,
            D.next_power_of_two().trailing_zeros() as usize,
            &commitments,
            ajtai_dec_mixer,
            Some(cache),
            None,
            None,
        );
    if children.is_empty() {
        return Err(engine::Error::PiDecFailed.into());
    }
    if !(ok_y && ok_x && ok_c) {
        return Err(engine::Error::PiDecPublicCheckFailed { ok_y, ok_x, ok_c }.into());
    }
    let proof = Proof { children };
    let claims = verify(pp, s, ajtai_dec_mixer, parent, &proof)?;
    Ok((
        Children {
            claims,
            witnesses: digits,
        },
        proof,
    ))
}

pub fn verify(
    pp: &Params,
    s: &Structure,
    combine: DecMixer,
    parent: &CeClaim,
    proof: &Proof,
) -> Result<Vec<CeClaim>, Error> {
    validate_verifier_inputs(pp, s, combine, parent, proof)?;
    let ok = engine::verify_pi_dec(pp, s, parent, &proof.children, |cs, b| combine(cs, b));
    if !ok {
        return Err(Error::VerifyRejected);
    }
    Ok(proof.children.clone())
}

fn validate_verifier_inputs(
    pp: &Params,
    s: &Structure,
    combine: DecMixer,
    parent: &CeClaim,
    proof: &Proof,
) -> Result<(), Error> {
    validate_child_count(pp, proof.children.len())?;
    validate_fold_digest_canonical("parent", parent)?;
    for child in &proof.children {
        validate_fold_digest_canonical("child", child)?;
    }
    validate_r_shape(s, parent, &proof.children)?;
    validate_evaluation_shape(s, parent, &proof.children)?;
    validate_canonical_x_shape(parent, &proof.children)?;
    validate_child_x_low_norm(pp, &proof.children)?;
    validate_evaluation_padding_zero(parent, &proof.children)?;
    validate_fold_digest_consistency(parent, &proof.children)?;
    let _ = combine;
    if parent.adv.is_some() || proof.children.iter().any(|c| c.adv.is_some()) {
        return Err(Error::Auxiliary);
    }
    Ok(())
}

fn validate_r_shape(s: &Structure, parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    validate_r_shape_one("parent", s, parent)?;
    for child in children {
        validate_r_shape_one("child", s, child)?;
    }
    Ok(())
}

fn validate_r_shape_one(owner: &'static str, s: &Structure, claim: &CeClaim) -> Result<(), Error> {
    let expected =
        s.n.max(neo_reductions::common::superneo_carrier_width(s.m))
            .next_power_of_two()
            .max(2)
            .trailing_zeros() as usize;
    if claim.r.len() != expected {
        return Err(Error::RShape(owner));
    }
    Ok(())
}

fn validate_child_count(pp: &Params, got: usize) -> Result<(), Error> {
    let expected = pp.k_rho() as usize;
    if got != expected {
        return Err(Error::ChildCount { expected, got });
    }
    Ok(())
}

fn validate_canonical_x_shape(parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    if !superneo_has_canonical_x_shape(&parent.X, parent.m_in) {
        return Err(Error::NoncanonicalXShape("parent"));
    }
    for child in children {
        if !superneo_has_canonical_x_shape(&child.X, child.m_in) {
            return Err(Error::NoncanonicalXShape("child"));
        }
    }
    Ok(())
}

fn validate_child_x_low_norm(pp: &Params, children: &[CeClaim]) -> Result<(), Error> {
    let b = pp.b();
    for child in children {
        let active_cols = superneo_public_x_cols(child.m_in);
        if active_cols > child.X.cols() {
            return Err(Error::ChildXLowNorm);
        }
        for r in 0..child.X.rows() {
            for c in 0..active_cols {
                if !within_nc_bound(child.X[(r, c)], b) {
                    return Err(Error::ChildXLowNorm);
                }
            }
        }
    }
    Ok(())
}

fn validate_fold_digest_consistency(parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    for child in children {
        if child.fold_digest != parent.fold_digest {
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

fn validate_evaluation_shape(s: &Structure, parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    validate_evaluation_shape_one("parent", s, parent)?;
    for child in children {
        validate_evaluation_shape_one("child", s, child)?;
    }
    Ok(())
}

fn validate_evaluation_shape_one(owner: &'static str, s: &Structure, claim: &CeClaim) -> Result<(), Error> {
    let width = D.next_power_of_two();
    if claim.eval_k.len() != width || claim.eval_a.len() != s.t() || claim.eval_a.iter().any(|row| row.len() != width) {
        return Err(Error::EvaluationShape(owner));
    }
    Ok(())
}

fn validate_evaluation_padding_zero(parent: &CeClaim, children: &[CeClaim]) -> Result<(), Error> {
    validate_evaluation_padding_zero_one("parent", parent)?;
    for child in children {
        validate_evaluation_padding_zero_one("child", child)?;
    }
    Ok(())
}

fn validate_evaluation_padding_zero_one(owner: &'static str, claim: &CeClaim) -> Result<(), Error> {
    if claim
        .eval_k
        .iter()
        .skip(D)
        .any(|&lane| lane != K::default())
    {
        return Err(Error::EvaluationPadding(owner));
    }
    for row in &claim.eval_a {
        for &lane in row.iter().skip(D) {
            if lane != K::default() {
                return Err(Error::EvaluationPadding(owner));
            }
        }
    }
    Ok(())
}
