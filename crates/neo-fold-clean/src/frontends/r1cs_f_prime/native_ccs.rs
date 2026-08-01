//! Verifier-owned preprocessing for one Lean native-CCS F′ manifest.
//!
//! Owns: relation reconstruction, shape-derived parameters, exact seeded
//! Ajtai registration, terminal-induction capability, and satisfying-instance
//! construction.
//!
//! Does not own: manifest generation, application witness generation,
//! recursive proving, terminal Spartan proving, or operator-selected setup.

use neo_ajtai::set_global_pp_seeded;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::lifecycle::{self, Preprocessing};
use crate::paper::relations::{CcsInstance, RelationError};

use super::lean_manifest::ColumnId;
use super::lean_native_ccs_manifest::{LeanNativeCcsEmissionError, LeanNativeCcsManifest};

#[derive(Debug, Error)]
pub enum LeanNativeCcsError {
    #[error(transparent)]
    Emission(#[from] LeanNativeCcsEmissionError),
    #[error("Lean native CCS assignment does not satisfy the manifest relation")]
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

/// One verifier-owned native F′ relation and its lifecycle preprocessing.
///
/// This object retains the validated manifest that selected the relation. It
/// cannot be assembled from an unrelated [`Preprocessing`] and manifest.
pub struct LeanNativeCcsPreprocessing {
    manifest: LeanNativeCcsManifest,
    preprocessing: Preprocessing,
}

impl LeanNativeCcsPreprocessing {
    /// Reconstruct and preprocess the exact relation selected by `manifest`.
    ///
    /// The manifest is the complete F′ relation, so the outer lifecycle uses
    /// its stateless mode. The application state is already constrained and
    /// hashed inside the manifest rows. A second caller-selected semantic
    /// state would create a competing authority.
    pub fn new(manifest: LeanNativeCcsManifest) -> Result<Self, LeanNativeCcsError> {
        let relation = manifest.emit_phi81_step(|_| Some(F::ZERO))?;
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

    pub fn manifest(&self) -> &LeanNativeCcsManifest {
        &self.manifest
    }

    pub fn preprocessing(&self) -> &Preprocessing {
        &self.preprocessing
    }

    /// Build one foldable instance from an exact manifest-column assignment.
    ///
    /// The relation check runs before the Ajtai commitment. Missing columns,
    /// selector drift, a failed CCS row, and a non-low-norm assignment all
    /// reject.
    pub fn build_instance(
        &self,
        values: impl FnMut(&ColumnId) -> Option<F>,
    ) -> Result<CcsInstance, LeanNativeCcsError> {
        let emission = self.manifest.emit_phi81_step(values)?;
        if !emission.is_satisfied() {
            return Err(LeanNativeCcsError::Unsatisfied);
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
