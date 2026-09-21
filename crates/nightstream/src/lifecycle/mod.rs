//! Selected recursive state and terminal openings for one prepared circuit.
//! Application witness and output values come from the Rust application owner.
//! The generated rows and fixed key determine whether those values are accepted.

use crate::engine::{Engine, Prover};
use crate::folding::{CcsInstance, RunningInstance, Structure};
use neo_ajtai::nightstream_fprime_setup::{
    authority_words, PRODUCTION_CARRIER_WIDTH, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS,
};
use neo_math::D;
use neo_reductions::superneo_eval::SuperneoEvalCache;
use nightstream_fprime::{
    LoadedPerApplicationPackage, PackageError, PiCcsV1_1PackageInputs, PiDecV1_1PackageInputs, Stage1VerifierBinding,
    WitnessAssignment,
};
use std::sync::{Arc, OnceLock};
mod base;
mod complete;
mod evaluation;
mod extend;
mod inputs;
mod prove;
mod step_inputs;
mod verify;
pub use complete::{CompleteStepError, Stage1Envelope};
pub use extend::ExtendError;
pub use inputs::{
    encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage,
    PiCcsV1_1PackageBridgeError, PiCcsV1_1ProofInputs,
};
pub use prove::ProveError;
pub use step_inputs::{Stage1State, Stage1StepInputs, StepInputError};
pub use verify::VerifyError;

pub struct PreparedLifecycle {
    package: LoadedPerApplicationPackage,
    structure: Structure,
    binding: Stage1VerifierBinding,
    cache: OnceLock<Arc<SuperneoEvalCache>>,
    prover: Prover,
}
impl PreparedLifecycle {
    pub(crate) fn from_package(
        package: LoadedPerApplicationPackage,
        binding: Stage1VerifierBinding,
        prover: Prover,
    ) -> Result<Self, PackageError> {
        if package.production_verifier_binding()? != binding {
            return Err(PackageError::Invalid(
                "lifecycle binding differs from the prepared relation",
            ));
        }
        let structure = package.ccs_structure_header()?;
        validate_key_prefix(structure.m, binding.verifier_context().commitment_key_words())?;
        if matches!(prover.engine(), Engine::PaperExact | Engine::Crosscheck) {
            package.validate_all_matrix_rows()?;
        }
        Ok(Self {
            package,
            structure,
            binding,
            cache: OnceLock::new(),
            prover,
        })
    }
    pub(crate) fn engine(&self) -> Engine {
        self.prover.engine()
    }
    #[cfg(test)]
    pub(crate) fn structure(&self) -> &Structure {
        &self.structure
    }
    pub fn package_identity(&self) -> [u64; 4] {
        self.binding.package_identity()
    }
    pub(crate) fn execute_step_witness(
        &self,
        c: &PiCcsV1_1PackageInputs,
        d: &PiDecV1_1PackageInputs,
        application_witness: &[u64],
    ) -> Result<WitnessAssignment, PackageError> {
        self.package
            .execute_stage1_v1_1_witness(c, d, application_witness)
    }
}
/// The package binds its exact prefix dimensions; the selected seed and rows are fixed.
fn validate_key_prefix(logical_width: usize, commitment_key_words: &[u64]) -> Result<(), PackageError> {
    if logical_width == 0 || logical_width > PRODUCTION_CARRIER_WIDTH {
        return Err(PackageError::Invalid("logical width exceeds the selected key prefix"));
    }
    let columns = logical_width.div_ceil(D) as u64;
    let expected = authority_words(PRODUCTION_VERIFIER_ROWS, columns, &PRODUCTION_SEED);
    if commitment_key_words != expected {
        return Err(PackageError::Invalid(
            "commitment authority differs from the exact production key prefix",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct LatestInstance {
    instances: Vec<CcsInstance>,
}
impl LatestInstance {
    fn from_instances(instances: Vec<CcsInstance>) -> Self {
        Self { instances }
    }
}
#[derive(Clone, Debug)]
enum ProofState {
    Initial,
    Active {
        running: RunningInstance,
        latest: LatestInstance,
    },
}
impl ProofState {
    fn initial() -> Self {
        Self::Initial
    }
    fn active(running: RunningInstance, latest: LatestInstance) -> Self {
        Self::Active { running, latest }
    }
    fn is_initial(&self) -> bool {
        matches!(self, Self::Initial)
    }
    fn running(&self) -> Option<&RunningInstance> {
        match self {
            Self::Initial => None,
            Self::Active { running, .. } => Some(running),
        }
    }
    fn latest(&self) -> Option<&LatestInstance> {
        match self {
            Self::Initial => None,
            Self::Active { latest, .. } => Some(latest),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/lifecycle_native/mod.rs"]
mod tests;
