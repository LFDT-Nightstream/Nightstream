//! Arguments for the shared optimized arithmetic kernels.
use super::{CeClaim, Params, RlcMixer, Structure};
use crate::engine::Backend;
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::F;
use neo_reductions::{
    api as nr,
    api::FoldingMode,
    common::{sample_rot_rhos_n_typed, RotRho},
};
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("PiDEC returned no children")]
    PiDecFailed,
    #[error(transparent)]
    Reductions(#[from] neo_reductions::PiCcsError),
    #[error("PiDEC public checks failed: y={ok_y}, x={ok_x}, c={ok_c}")]
    PiDecPublicCheckFailed { ok_y: bool, ok_x: bool, ok_c: bool },
    #[error(transparent)]
    Engine(#[from] crate::engine::EngineError),
}
impl From<crate::engine::DeviceError> for Error {
    fn from(error: crate::engine::DeviceError) -> Self {
        match error {
            crate::engine::DeviceError::Engine(error) => Self::Engine(error),
            crate::engine::DeviceError::Rejected(error) => Self::Reductions(error),
        }
    }
}
impl From<crate::engine::CommitError> for Error {
    fn from(error: crate::engine::CommitError) -> Self {
        match error {
            crate::engine::CommitError::Engine(error) => Self::Engine(error),
            crate::engine::CommitError::Input(reason) => {
                Self::Reductions(neo_reductions::PiCcsError::InvalidInput(reason))
            }
        }
    }
}
impl From<nr::DecRejected> for Error {
    fn from(nr::DecRejected { ok_y, ok_x, ok_c }: nr::DecRejected) -> Self {
        Self::PiDecPublicCheckFailed { ok_y, ok_x, ok_c }
    }
}
fn ell_d() -> usize {
    neo_math::D.next_power_of_two().trailing_zeros() as usize
}

pub(crate) fn sample_rho_n(
    tr: &mut neo_transcript::Poseidon2Transcript,
    pp: &Params,
    count: usize,
) -> Result<Vec<RotRho>, Error> {
    sample_rot_rhos_n_typed(tr, pp.inner(), &pp.ring(), count).map_err(Into::into)
}

/// PiRLC's parent claim and witness. A device engine mixes the witnesses;
/// the host mixes the commitments and the claims.
pub(crate) fn prove_pi_rlc_refs(
    backend: &Backend,
    pp: &Params,
    s: &Structure,
    rhos: &[RotRho],
    me_inputs: &[CeClaim],
    witnesses: &[&Mat<F>],
    mix_rhos_commits: RlcMixer,
) -> Result<(CeClaim, Mat<F>), Error> {
    match backend {
        Backend::Optimized => nr::rlc_with_commit_refs(
            FoldingMode::Optimized,
            s,
            pp.inner(),
            rhos,
            me_inputs,
            witnesses,
            ell_d(),
            mix_rhos_commits,
        )
        .map_err(Into::into),
        #[cfg(feature = "metal")]
        Backend::Metal(device) => {
            let (claim, witness) = nr::rlc_with_commit_refs_and_resident_witness(
                FoldingMode::Optimized,
                s,
                pp.inner(),
                rhos,
                me_inputs,
                witnesses,
                ell_d(),
                mix_rhos_commits,
                |rhos, witnesses| device.mix_rlc_witnesses(rhos, witnesses),
            )?;
            Ok((claim, witness?))
        }
    }
}

pub(crate) fn verify_pi_dec<MB>(
    pp: &Params,
    s: &Structure,
    parent: &CeClaim,
    children: &[CeClaim],
    combine_b_pows: MB,
) -> Result<(), neo_reductions::PiCcsError>
where
    MB: Fn(&[Commitment], u32) -> Commitment,
{
    nr::verify_dec_public(s, pp.inner(), parent, children, combine_b_pows, ell_d())
}
