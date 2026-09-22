//! Native selected C/R/D composition copied from neo-fold-clean's Stage 1 path.
//! Witnesses remain prover data. Public claims acquire authority only through verification.

mod claims;
mod compose;
pub(crate) mod kernels;
mod params;
pub(crate) mod pi_ccs;
pub(crate) mod pi_dec;
pub(crate) mod pi_rlc;
mod proof;
pub(crate) mod transcript;

pub(crate) use claims::{ajtai_dec_mixer, ajtai_rlc_mixer, superneo_has_canonical_x_shape};
pub use claims::{CcsInstance, RunningInstance};
pub(crate) use compose::{prove_owned_with_rows, validate_running_parent_authority, verify};
use neo_ajtai::Commitment;
pub(crate) use neo_ccs::superneo_public_x_cols;
use neo_math::{F, K};
pub(crate) use params::Params;
pub use proof::NifsProof;
pub type CcsClaim = neo_ccs::CcsClaim<Commitment, F>;
pub type CeClaim = neo_ccs::CeClaim<Commitment, F, K>;
pub type CcsWitness = neo_ccs::CcsWitness<F>;
pub(crate) type Structure = neo_ccs::CcsStructure<F>;
pub(crate) type RlcMixer = fn(&[neo_ccs::Mat<F>], &[Commitment]) -> Commitment;
pub(crate) type DecMixer = fn(&[Commitment], u32) -> Commitment;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    PiCcs(#[from] pi_ccs::Error),
    #[error(transparent)]
    PiRlc(#[from] pi_rlc::Error),
    #[error(transparent)]
    PiDec(#[from] pi_dec::Error),
}
