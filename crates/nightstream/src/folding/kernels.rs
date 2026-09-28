//! Arguments for the shared optimized arithmetic kernels.
use super::{CcsClaim, CeClaim, Params, RunningInstance, Structure};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::F;
use neo_reductions::{
    api as nr,
    api::FoldingMode,
    common::{sample_rot_rhos_n_typed, RotRho},
    optimized_engine::pi_ccs_verify,
};
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("PiDEC returned no children")]
    PiDecFailed,
    #[error(transparent)]
    Reductions(#[from] neo_reductions::PiCcsError),
    #[error("PiDEC public checks failed: y={ok_y}, x={ok_x}, c={ok_c}")]
    PiDecPublicCheckFailed { ok_y: bool, ok_x: bool, ok_c: bool },
}
fn ell_d() -> usize {
    neo_math::D.next_power_of_two().trailing_zeros() as usize
}

pub fn verify_pi_ccs(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    s: &Structure,
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    fold_outputs: &[CeClaim],
    proof: &nr::PiCcsProof,
) -> Result<bool, Error> {
    let ok = pi_ccs_verify(tr, pp.inner(), s, fresh_claims, &running.claims, fold_outputs, proof)?;
    Ok(ok)
}

pub fn sample_rho_n(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    count: usize,
) -> Result<Vec<RotRho>, Error> {
    sample_rot_rhos_n_typed(tr, pp.inner(), &pp.ring(), count).map_err(Into::into)
}

pub fn prove_pi_rlc_refs<MR>(
    pp: &Params,
    s: &Structure,
    rhos: &[RotRho],
    me_inputs: &[CeClaim],
    witnesses: &[&Mat<F>],
    mix_rhos_commits: MR,
) -> Result<(CeClaim, Mat<F>), Error>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment,
{
    nr::rlc_with_commit_refs(
        FoldingMode::Optimized,
        s,
        pp.inner(),
        rhos,
        me_inputs,
        witnesses,
        ell_d(),
        mix_rhos_commits,
    )
    .map_err(Into::into)
}

pub fn verify_pi_rlc<MR>(
    pp: &Params,
    s: &Structure,
    rhos: &[RotRho],
    me_inputs: &[CeClaim],
    expected: &CeClaim,
    mix_rhos_commits: MR,
) -> Result<bool, Error>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment,
{
    let (ok, perf) = nr::rlc_public_matches_verified_inputs_with_perf(
        s,
        pp.inner(),
        rhos,
        me_inputs,
        expected,
        mix_rhos_commits,
        ell_d(),
    )?;
    let _ = perf;
    Ok(ok)
}

pub fn verify_pi_dec<MB>(pp: &Params, s: &Structure, parent: &CeClaim, children: &[CeClaim], combine_b_pows: MB) -> bool
where
    MB: Fn(&[Commitment], u32) -> Commitment,
{
    nr::verify_dec_public(s, pp.inner(), parent, children, combine_b_pows, ell_d())
}
