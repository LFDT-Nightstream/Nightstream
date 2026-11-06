//! Native selected C/R/D composition.
//! Witnesses remain prover data. Public claims acquire authority only through verification.

mod claims;
pub(crate) mod kernels;
pub(crate) mod option_b;
mod params;
pub(crate) mod pi_ccs;
pub(crate) mod pi_dec;
pub(crate) mod pi_rlc;
pub(crate) mod transcript;

pub(crate) use claims::{
    ajtai_dec_mixer, ajtai_rlc_mixer, has_canonical_evaluations, superneo_has_canonical_x_shape, EVALUATION_WIDTH,
};
pub(crate) use claims::{CcsInstance, RunningInstance};
use neo_ajtai::Commitment;
pub(crate) use neo_ccs::superneo_public_x_cols;
use neo_math::{F, K};
pub(crate) use params::Params;
pub(crate) type CcsClaim = neo_ccs::CcsClaim<Commitment, F>;
pub(crate) type CeClaim = neo_ccs::CeClaim<Commitment, F, K>;
pub(crate) type CcsWitness = neo_ccs::CcsWitness<F>;
pub(crate) type Structure = neo_ccs::CcsStructure<F>;
pub(crate) use neo_reductions::api::RlcMixer;
pub(crate) type DecMixer = fn(&[Commitment], u32) -> Commitment;

/// What every engine call reads about one circuit: the folding parameters,
/// the structure, its matrix rows and the rows' working-storage allowance.
#[derive(Clone, Copy)]
pub(crate) struct FoldContext<'a> {
    pub(crate) params: &'a Params,
    pub(crate) structure: &'a Structure,
    pub(crate) rows: &'a dyn neo_reductions::superneo_eval::MatrixRows,
    pub(crate) workspace_bytes: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Ccs(#[from] pi_ccs::Error),
    #[error(transparent)]
    Rlc(#[from] pi_rlc::Error),
    #[error(transparent)]
    Dec(#[from] pi_dec::Error),
}
