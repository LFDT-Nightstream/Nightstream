//! Selected native proof construction from the prepared relation.
use super::PreparedLifecycle;
use crate::engine::{paper_exact, Backend};
use crate::folding::{self as nifs, transcript::Transcript, CcsInstance, Params, RunningInstance};
use neo_math::D;
use nightstream_fprime::{
    PackageError, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_SOURCE_COUNT, PI_DEC_V1_1_CHILD_COUNT,
};
#[derive(Debug, thiserror::Error)]
pub enum ProveError {
    #[error(transparent)]
    Engine(#[from] crate::EngineError),
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
        let mut transcript = Transcript::session();
        let rows = self.matrix_rows();
        let workspace_bytes = self.matrix_workspace_bytes()?;
        Ok(match &self.backend {
            #[cfg(feature = "metal")]
            Backend::Metal(device) => {
                let mut device = device.lock().map_err(|_| crate::EngineError::Failure {
                    engine: crate::Engine::Metal,
                    reason: "device session lock was poisoned".into(),
                })?;
                crate::engine::metal::prove(
                    &mut device,
                    &mut transcript,
                    &params,
                    &self.structure,
                    &rows,
                    workspace_bytes,
                    fresh,
                    running,
                )?
            }
            Backend::Optimized => nifs::prove_owned_with_rows(
                &mut transcript,
                &params,
                &self.structure,
                &rows,
                workspace_bytes,
                fresh,
                running,
            )?,
            Backend::PaperExact => paper_exact::prove(
                &mut transcript,
                &params,
                &self.structure,
                &paper_exact::PackageRows(&self.package),
                fresh,
                running,
            )?,
            Backend::Crosscheck => crate::engine::crosscheck::prove(
                &mut transcript,
                &params,
                &self.structure,
                &rows,
                workspace_bytes,
                &paper_exact::PackageRows(&self.package),
                fresh,
                running,
            )?,
        })
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
