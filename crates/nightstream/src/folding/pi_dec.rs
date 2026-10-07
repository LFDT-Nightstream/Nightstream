//! Selected radix-two decomposition under the existing production key.
use super::{
    ajtai_dec_mixer, has_canonical_evaluations, kernels as engine, superneo_has_canonical_x_shape,
    superneo_public_x_cols, CeClaim, DecMixer, Params, Structure,
};
use neo_ajtai::nightstream_fprime_setup::{
    commit_production_signed_unit_prefix_matrices, MAX_MESSAGE_COLUMNS, PRODUCTION_VERIFIER_ROWS,
};
use neo_ccs::Mat;
use neo_math::{balanced::within_nc_bound, D, F};
use neo_reductions::superneo_eval::{eval_real_v1_1_openings_from_rows, MatrixRows, MatrixShape, SuperneoZBlocks};
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
    #[error("PiDEC noncanonical evaluations in {0}")]
    Evaluation(&'static str),
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
    rows: &dyn MatrixRows,
    workspace_bytes: usize,
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
        || blocks > MAX_MESSAGE_COLUMNS as usize
        || rows.shape()
            != (MatrixShape {
                rows: s.n,
                columns: blocks * D,
                matrices: s.t(),
            })
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
    let blocks = digits
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, s.m))
        .collect::<Result<Vec<_>, _>>()
        .map_err(engine::Error::from)?;
    let openings =
        eval_real_v1_1_openings_from_rows(rows, &parent.r, &blocks, workspace_bytes).map_err(engine::Error::from)?;
    drop(blocks);
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
            None,
            None,
            Some(&openings),
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
    validate_verifier_inputs(pp, s, parent, proof)?;
    let ok = engine::verify_pi_dec(pp, s, parent, &proof.children, |cs, b| combine(cs, b));
    if !ok {
        return Err(Error::VerifyRejected);
    }
    Ok(proof.children.clone())
}

fn validate_verifier_inputs(pp: &Params, s: &Structure, parent: &CeClaim, proof: &Proof) -> Result<(), Error> {
    validate_children(pp, s, &proof.children)?;
    validate_claim("parent", s, parent)?;
    validate_fold_digest_canonical("parent", parent)?;
    for child in &proof.children {
        validate_fold_digest_canonical("child", child)?;
    }
    validate_fold_digest_consistency(parent, &proof.children)
}

/// The child-family checks that need no parent: the selected count, each
/// claim's point, evaluation and public shapes, and unit-norm public inputs.
/// Frames are not checked here.
pub(crate) fn validate_children(pp: &Params, s: &Structure, children: &[CeClaim]) -> Result<(), Error> {
    validate_child_count(pp, children.len())?;
    for child in children {
        validate_claim("child", s, child)?;
    }
    validate_child_x_low_norm(pp, children)
}

fn validate_claim(owner: &'static str, s: &Structure, claim: &CeClaim) -> Result<(), Error> {
    let point = s
        .domain_rows()
        .max(neo_reductions::common::superneo_carrier_width(s.m))
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    if claim.r.len() != point {
        return Err(Error::RShape(owner));
    }
    if !has_canonical_evaluations(claim, s.t()) {
        return Err(Error::Evaluation(owner));
    }
    if !superneo_has_canonical_x_shape(&claim.X, claim.m_in) {
        return Err(Error::NoncanonicalXShape(owner));
    }
    if claim.adv.is_some() {
        return Err(Error::Auxiliary);
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
