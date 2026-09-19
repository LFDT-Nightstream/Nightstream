//! Selected native proof construction from the prepared relation.
use super::PreparedLifecycle;
use crate::folding::{self as nifs, transcript::Transcript, CcsInstance, Params, RunningInstance};
use neo_math::D;
use nightstream_fprime::{
    PackageError, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_SOURCE_COUNT, PI_DEC_V1_1_CHILD_COUNT,
};
#[derive(Debug, thiserror::Error)]
pub enum ProveError {
    #[error("selected proof input: {0}")]
    Input(&'static str),
    #[error(transparent)]
    Package(#[from] PackageError),
    #[error(transparent)]
    Parameters(#[from] neo_params::ParamsError),
    #[error(transparent)]
    Nifs(#[from] nifs::Error),
}
impl PreparedLifecycle {
    pub fn prove(
        &self,
        fresh: Vec<CcsInstance>,
        running: RunningInstance,
    ) -> Result<(RunningInstance, nifs::NifsProof), ProveError> {
        self.validate_prover_sources(&fresh, &running)?;
        let params = Params::for_ccs_shape(
            self.structure.n,
            self.structure.m,
            self.structure.t(),
            self.structure.max_degree(),
        )?;
        let cache = self.build_superneo_cache()?;
        let mut transcript = Transcript::session();
        Ok(nifs::prove_owned_with_rows(
            &mut transcript,
            &params,
            &self.structure,
            cache,
            fresh,
            running,
        )?)
    }

    fn validate_prover_sources(&self, fresh: &[CcsInstance], running: &RunningInstance) -> Result<(), ProveError> {
        if fresh.len() != PI_CCS_V1_1_SOURCE_COUNT - PI_DEC_V1_1_CHILD_COUNT
            || running.claims.len() != PI_DEC_V1_1_CHILD_COUNT
            || !running.prover_shape_is_valid()
        {
            return Err(ProveError::Input("source counts do not match the selected profile"));
        }
        let blocks = self.structure.m.div_ceil(D);
        let public = PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS;
        for source in fresh {
            if source.claim.m_in != public
                || source.claim.x.len() != public
                || source.witness.Z.rows() != D
                || source.witness.Z.cols() != blocks
                || source.claim.adv.is_some()
                || (0..public).any(|index| source.witness.Z[(index % D, index / D)] != source.claim.x[index])
            {
                return Err(ProveError::Input("fresh carrier or public prefix does not match"));
            }
        }
        for (claim, witness) in running.claims.iter().zip(&running.witnesses) {
            if claim.m_in != public
                || claim.X.rows() != D
                || claim.X.cols() != public / D
                || witness.rows() != D
                || witness.cols() != blocks
                || claim.adv.is_some()
                || (0..public).any(|index| witness[(index % D, index / D)] != claim.X[(index % D, index / D)])
            {
                return Err(ProveError::Input("running carrier or public prefix does not match"));
            }
        }
        Ok(())
    }
}
