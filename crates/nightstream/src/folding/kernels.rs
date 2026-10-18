//! Arguments for the shared optimized arithmetic kernels.
use super::{CcsClaim, CeClaim, Params, RlcMixer, RunningInstance, Structure};
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
    fresh_claim: &CcsClaim,
    running: &RunningInstance,
    fold_outputs: &[CeClaim],
    proof: &nr::PiCcsProof,
) -> Result<bool, Error> {
    let ok = pi_ccs_verify(
        tr,
        pp.inner(),
        s,
        std::slice::from_ref(fresh_claim),
        &running.claims,
        fold_outputs,
        proof,
    )?;
    Ok(ok)
}

pub fn sample_rho_n(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    count: usize,
) -> Result<Vec<RotRho>, Error> {
    sample_rot_rhos_n_typed(tr, pp.inner(), &pp.ring(), count).map_err(Into::into)
}

pub fn prove_pi_rlc_refs(
    pp: &Params,
    s: &Structure,
    rhos: &[RotRho],
    me_inputs: &[CeClaim],
    witnesses: &[&Mat<F>],
    mix_rhos_commits: RlcMixer,
) -> Result<(CeClaim, Mat<F>), Error> {
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

#[cfg(feature = "metal")]
pub fn prove_pi_rlc_refs_resident<MW, R>(
    pp: &Params,
    s: &Structure,
    rhos: &[RotRho],
    me_inputs: &[CeClaim],
    witnesses: &[&Mat<F>],
    mix_rhos_commits: RlcMixer,
    mix_witnesses: MW,
) -> Result<(CeClaim, R), Error>
where
    MW: Fn(&[Mat<F>], &[&Mat<F>]) -> R,
{
    nr::rlc_with_commit_refs_and_resident_witness(
        FoldingMode::Optimized,
        s,
        pp.inner(),
        rhos,
        me_inputs,
        witnesses,
        ell_d(),
        mix_rhos_commits,
        mix_witnesses,
    )
    .map_err(Into::into)
}

pub fn verify_pi_rlc(
    pp: &Params,
    s: &Structure,
    rhos: &[RotRho],
    me_inputs: &[CeClaim],
    expected: &CeClaim,
    mix_rhos_commits: RlcMixer,
) -> Result<bool, Error> {
    nr::rlc_public_matches(s, pp.inner(), rhos, me_inputs, expected, mix_rhos_commits, ell_d()).map_err(Into::into)
}

pub fn verify_pi_dec<MB>(pp: &Params, s: &Structure, parent: &CeClaim, children: &[CeClaim], combine_b_pows: MB) -> bool
where
    MB: Fn(&[Commitment], u32) -> Commitment,
{
    nr::verify_dec_public(s, pp.inner(), parent, children, combine_b_pows, ell_d())
}
