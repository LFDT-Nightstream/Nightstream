//! Direct NIFS reduction seam backed only by `neo-reductions` PaperExact code.
//!
//! Owns argument marshaling for PaperExact PiCCS, PiRLC, and PiDEC. It does
//! not use optimized caches or accelerator callbacks.

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::F;
use neo_reductions::api as reductions;
use neo_reductions::common::{split_b_matrix_k_with_nonzero_flags, RotRho};
use thiserror::Error;

use crate::paper::construction2::RunningInstance;
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, CcsWitness, CeClaim, Structure};

fn ell_d() -> usize {
    neo_math::D.next_power_of_two().trailing_zeros() as usize
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("engine.paper_exact: {0}")]
    Reductions(#[from] neo_reductions::error::PiCcsError),
    #[error("engine.paper_exact: PiDEC returned an empty child set")]
    PiDecFailed,
    #[error("engine.paper_exact: PiDEC public checks failed (y={ok_y}, X={ok_x}, c={ok_c})")]
    PiDecPublicCheckFailed { ok_y: bool, ok_x: bool, ok_c: bool },
}

#[allow(clippy::too_many_arguments)]
pub fn prove_pi_ccs_parts<L>(
    transcript: &mut neo_transcript::Poseidon2Transcript,
    params: &Params,
    structure: &Structure,
    fresh_claims: &[CcsClaim],
    fresh_witnesses: &[CcsWitness],
    running: &RunningInstance,
    commitment: &L,
) -> Result<(Vec<CeClaim>, reductions::PiCcsProof), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Commitment>,
{
    let public_instance_digest = crate::paper::reductions::paper_exact_protocol::pi_ccs_instance_digest(
        fresh_claims,
        running.claims.len(),
        running.parent_authority.as_ref(),
    );
    let running_accumulator_handle = crate::paper::reductions::paper_exact_protocol::accumulator_handle(
        &running.claims,
        running.parent_authority.as_ref(),
    );
    neo_reductions::engines::paper_exact_engine::paper_exact_prove_with_instance_digest_and_me_input_handle(
        transcript,
        params.inner(),
        structure,
        fresh_claims,
        fresh_witnesses,
        &running.claims,
        &running.witnesses,
        public_instance_digest,
        running_accumulator_handle,
        commitment,
    )
    .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
pub fn verify_pi_ccs(
    transcript: &mut neo_transcript::Poseidon2Transcript,
    params: &Params,
    structure: &Structure,
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    outputs: &[CeClaim],
    proof: &reductions::PiCcsProof,
) -> Result<bool, Error> {
    let public_instance_digest = crate::paper::reductions::paper_exact_protocol::pi_ccs_instance_digest(
        fresh_claims,
        running.claims.len(),
        running.parent_authority.as_ref(),
    );
    let running_accumulator_handle = crate::paper::reductions::paper_exact_protocol::accumulator_handle(
        &running.claims,
        running.parent_authority.as_ref(),
    );
    neo_reductions::engines::paper_exact_engine::paper_exact_verify_with_instance_digest_and_me_input_handle(
        transcript,
        params.inner(),
        structure,
        fresh_claims,
        &running.claims,
        outputs,
        proof,
        public_instance_digest,
        running_accumulator_handle,
    )
    .map_err(Into::into)
}

pub fn prove_pi_rlc_refs<MR>(
    params: &Params,
    structure: &Structure,
    rhos: &[RotRho],
    claims: &[CeClaim],
    witnesses: &[&Mat<F>],
    mix_commitments: MR,
) -> Result<(CeClaim, Mat<F>), Error>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment,
{
    let rho_matrices = rhos
        .iter()
        .map(|rho| rho.as_mat().clone())
        .collect::<Vec<_>>();
    let witness_values = witnesses
        .iter()
        .map(|witness| (*witness).clone())
        .collect::<Vec<_>>();
    Ok(
        neo_reductions::engines::paper_exact_engine::rlc_reduction_paper_exact_with_commit_mix(
            structure,
            params.inner(),
            &rho_matrices,
            claims,
            &witness_values,
            ell_d(),
            mix_commitments,
        ),
    )
}

/// Sample PiRLC challenges without using the optimized sampler.
pub fn sample_rho_n(
    transcript: &mut neo_transcript::Poseidon2Transcript,
    params: &Params,
    count: usize,
) -> Result<Vec<RotRho>, Error> {
    let ring = params.ring();
    if count == 0 || ring.phi_coeffs.len() != neo_math::D || ring.alphabet.len() < 2 {
        return Err(neo_reductions::error::PiCcsError::InvalidInput(
            "PaperExact PiRLC sampler shape is invalid".into(),
        )
        .into());
    }
    if let Some(limit) = ring.binv_floor {
        let minimum = *ring.alphabet.iter().min().expect("nonempty alphabet") as i64;
        let maximum = *ring.alphabet.iter().max().expect("nonempty alphabet") as i64;
        if (maximum - minimum).unsigned_abs() >= limit {
            return Err(neo_reductions::error::PiCcsError::InvalidInput(
                "PaperExact PiRLC strong-set bound failed".into(),
            )
            .into());
        }
    }
    // One sampler owner: the shared typed sampler carries the corrected
    // whole-digest schedule (PI_RLC_SAMPLER_DIGEST_ROUNDS) that the Lean
    // conformance suite certifies; a local reimplementation diverged from
    // the optimized engine at the prover-transcript boundary.
    neo_reductions::common::sample_rot_rhos_n_typed(transcript, params.inner(), &params.ring(), count)
        .map_err(Into::into)
}

/// Recompute the public PiRLC parent with the direct schoolbook reference.
/// The paper layer checks the auxiliary `adv` commitment tuple separately.
pub fn verify_pi_rlc<MR>(
    params: &Params,
    structure: &Structure,
    rhos: &[RotRho],
    claims: &[CeClaim],
    expected: &CeClaim,
    mix_commitments: MR,
) -> bool
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment,
{
    let rho_matrices = rhos
        .iter()
        .map(|rho| rho.as_mat().clone())
        .collect::<Vec<_>>();
    let direct = neo_reductions::engines::paper_exact_engine::rlc_claim_paper_exact_with_commit_mix(
        structure,
        params.inner(),
        &rho_matrices,
        claims,
        ell_d(),
        mix_commitments,
    );
    direct.c == expected.c
        && direct.X == expected.X
        && direct.r == expected.r
        && direct.y_ring == expected.y_ring
        && direct.ct == expected.ct
        && direct.m_in == expected.m_in
        && direct.fold_digest == expected.fold_digest
}

#[allow(clippy::too_many_arguments)]
pub fn prove_pi_dec<L, MB>(
    params: &Params,
    structure: &Structure,
    commitment: &L,
    parent: &CeClaim,
    parent_witness: &Mat<F>,
    combine_commitments: MB,
) -> Result<(Vec<CeClaim>, Vec<Mat<F>>), Error>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
    MB: Fn(&[Commitment], u32) -> Commitment,
{
    let child_count = params.k_rho() as usize;
    let (witnesses, digit_nonzero) = split_b_matrix_k_with_nonzero_flags(parent_witness, child_count, params.b())?;
    let nonzero_witnesses = witnesses
        .iter()
        .zip(&digit_nonzero)
        .filter_map(|(witness, &nonzero)| nonzero.then_some(witness))
        .collect::<Vec<_>>();
    let mut nonzero_commitments = commitment.commit_many(&nonzero_witnesses).into_iter();
    let child_commitments = digit_nonzero
        .iter()
        .map(|&nonzero| {
            if nonzero {
                nonzero_commitments
                    .next()
                    .expect("PaperExact PiDEC commitment count matches nonzero witnesses")
            } else {
                Commitment::zeros(parent.c.d, parent.c.kappa)
            }
        })
        .collect::<Vec<_>>();
    debug_assert!(nonzero_commitments.next().is_none());

    let (children, ok_y, ok_x, ok_c) =
        neo_reductions::engines::paper_exact_engine::dec_reduction_paper_exact_with_commit_check(
            structure,
            params.inner(),
            parent,
            &witnesses,
            ell_d(),
            &child_commitments,
            combine_commitments,
        );
    if children.is_empty() {
        return Err(Error::PiDecFailed);
    }
    if !(ok_y && ok_x && ok_c) {
        return Err(Error::PiDecPublicCheckFailed { ok_y, ok_x, ok_c });
    }
    Ok((children, witnesses))
}

/// Check PiDEC public recomposition with direct PaperExact loops.
pub fn verify_pi_dec<MB>(params: &Params, parent: &CeClaim, children: &[CeClaim], combine_commitments: MB) -> bool
where
    MB: Fn(&[Commitment], u32) -> Commitment,
{
    neo_reductions::engines::paper_exact_engine::verify_dec_public_paper_exact(
        params.inner(),
        parent,
        children,
        combine_commitments,
    )
}
