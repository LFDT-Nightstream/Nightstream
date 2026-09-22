//! Verifier-owned preprocessing for one Lean F′ plus Nebula manifest.
//!
//! Owns: exact relation reconstruction, shape-derived parameters, seeded
//! Ajtai registration, terminal-induction capability, and instance creation.
//!
//! Does not own: manifest generation, native Step witnesses, Nebula witnesses,
//! recursive proving, terminal proving, or application semantics.

use neo_ajtai::set_global_pp_seeded;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::lifecycle::{self, Preprocessing};
use crate::paper::relations::{CcsInstance, RelationError};

use super::lean_manifest::ColumnId;
use super::lean_nebula_combined_manifest::{LeanNebulaCombinedEmissionError, LeanNebulaCombinedManifest};

#[derive(Debug, Error)]
pub enum LeanNebulaCombinedError {
    #[error(transparent)]
    Emission(#[from] LeanNebulaCombinedEmissionError),
    #[error("Lean F-prime plus Nebula assignment does not satisfy the manifest relation")]
    Unsatisfied,
    #[error(transparent)]
    Params(#[from] neo_params::ParamsError),
    #[error(transparent)]
    Ajtai(#[from] neo_ajtai::AjtaiError),
    #[error(transparent)]
    Relation(#[from] RelationError),
    #[error(transparent)]
    Lifecycle(#[from] lifecycle::Error),
}

/// One verifier-owned combined relation and its lifecycle preprocessing.
pub struct LeanNebulaCombinedPreprocessing {
    manifest: LeanNebulaCombinedManifest,
    preprocessing: Preprocessing,
}

impl LeanNebulaCombinedPreprocessing {
    /// Reconstruct and preprocess the exact relation selected by `manifest`.
    pub fn new(manifest: LeanNebulaCombinedManifest) -> Result<Self, LeanNebulaCombinedError> {
        let mut public = vec![F::ZERO; manifest.public_carrier_width()];
        public[0] = F::ONE;
        let private = vec![F::ZERO; manifest.nebula_private_width()];
        let relation = manifest.emit(&public, |_| Some(F::ZERO), &private)?;
        let structure = relation.structure().clone();
        let params = crate::config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())?;
        let message_cols = structure.m.div_ceil(D);
        set_global_pp_seeded(D, params.kappa() as usize, message_cols, manifest.ajtai_setup_seed())?;
        let preprocessing =
            lifecycle::preprocess(params, structure, Some(manifest.public_carrier_width()))?.with_terminal_induction();
        Ok(Self {
            manifest,
            preprocessing,
        })
    }

    pub fn manifest(&self) -> &LeanNebulaCombinedManifest {
        &self.manifest
    }

    pub fn preprocessing(&self) -> &Preprocessing {
        &self.preprocessing
    }

    /// Build one foldable instance from one exact combined assignment.
    pub fn build_instance(
        &self,
        public_values: &[F],
        native_values: impl FnMut(&ColumnId) -> Option<F>,
        nebula_private: &[F],
    ) -> Result<CcsInstance, LeanNebulaCombinedError> {
        let emission = self
            .manifest
            .emit(public_values, native_values, nebula_private)?;
        if !emission.is_satisfied() {
            return Err(LeanNebulaCombinedError::Unsatisfied);
        }
        Ok(CcsInstance::from_low_norm_assignment(
            &self.preprocessing.params,
            &self.preprocessing.log,
            self.preprocessing.structure(),
            emission.assignment(),
            self.manifest.public_carrier_width(),
        )?)
    }
}
