//! Selected prover-envelope completion. Child witnesses are checked against
//! the verified claims and retained by move. The emitted witness program
//! supplies the fresh assignment committed under the fixed production key.
//! Full CE evaluations and terminal acceptance remain verifier obligations.

use neo_ajtai::nightstream_fprime_setup::PRODUCTION_MESSAGE_COLUMNS;
use neo_ccs::Mat;
use neo_math::{D, F};
use neo_reductions::common::project_x_from_witness_mat;
use nightstream_fprime::{PackageError, PI_DEC_V1_1_CHILD_COUNT};
use p3_field::PrimeCharacteristicRing;

use super::{LatestInstance, ProofState};
use super::{PreparedLifecycle, Stage1State, Stage1StepInputs};
use crate::folding::{CcsClaim, CcsInstance, CcsWitness, RunningInstance};

/// Prover data for the selected state. An active envelope carries exactly
/// one fresh instance; construction does not establish terminal acceptance.
#[derive(Debug)]
pub struct Stage1Envelope {
    state: Stage1State,
    proof: ProofState,
}

impl Stage1Envelope {
    /// Load untrusted active proof data. Only the selected terminal verifier
    /// can establish acceptance of its statement and openings.
    #[cfg(test)]
    pub(crate) fn from_parts(state: Stage1State, running: RunningInstance, fresh: CcsInstance) -> Self {
        Self {
            state,
            proof: ProofState::active(running, LatestInstance::from_instances(vec![fresh])),
        }
    }

    /// The exact bottom case has zero iterations and no running or fresh proof.
    pub(crate) fn initial(z0: [F; 4]) -> Self {
        Self {
            state: Stage1State::new(0, z0, z0),
            proof: ProofState::initial(),
        }
    }

    pub fn state(&self) -> &Stage1State {
        &self.state
    }

    pub(crate) fn running(&self) -> Option<&RunningInstance> {
        self.proof.running()
    }

    pub(crate) fn fresh(&self) -> Option<&CcsInstance> {
        self.proof
            .latest()
            .and_then(|latest| latest.instances.first())
    }

    pub(crate) fn is_initial(&self) -> bool {
        self.proof.is_initial()
    }

    pub(super) fn into_parts(self) -> (Stage1State, ProofState) {
        (self.state, self.proof)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CompleteStepError {
    #[error("selected envelope input: {0}")]
    Input(&'static str),
    #[error("selected child {index}: {reason}")]
    ChildWitness { index: usize, reason: &'static str },
    #[error(transparent)]
    Package(#[from] PackageError),
    #[error(transparent)]
    Commitment(#[from] crate::engine::EngineError),
}

impl PreparedLifecycle {
    /// Complete one prover envelope from the checked NIFS caller packet and
    /// its supplied child witnesses. Running completion tails are retained;
    /// the newly generated fresh carrier has a zero completion tail.
    pub fn complete_step(
        &self,
        inputs: Stage1StepInputs,
        child_witnesses: Vec<Mat<F>>,
    ) -> Result<Stage1Envelope, CompleteStepError> {
        if child_witnesses.len() != PI_DEC_V1_1_CHILD_COUNT
            || inputs.next_running().claims.len() != PI_DEC_V1_1_CHILD_COUNT
        {
            return Err(CompleteStepError::Input(
                "child witness count differs from the selected profile",
            ));
        }
        for (index, (claim, witness)) in inputs
            .next_running()
            .claims
            .iter()
            .zip(&child_witnesses)
            .enumerate()
        {
            let projected = project_x_from_witness_mat(witness, self.structure.m, claim.m_in).map_err(|_| {
                CompleteStepError::ChildWitness {
                    index,
                    reason: "witness shape or public dimensions differ from the selected carrier",
                }
            })?;
            if projected != claim.X {
                return Err(CompleteStepError::ChildWitness {
                    index,
                    reason: "witness public projection differs from the verified child",
                });
            }
        }
        // Validate every coefficient, including the running tails, before
        // sharing exact indexed key coefficients across the child witnesses.
        #[cfg(test)]
        let started = std::time::Instant::now();
        let commitments = self.prover.commit(&child_witnesses)?;
        #[cfg(test)]
        eprintln!("complete child commitments elapsed={:?}", started.elapsed());
        for (index, (claim, commitment)) in inputs
            .next_running()
            .claims
            .iter()
            .zip(commitments)
            .enumerate()
        {
            if commitment != claim.c {
                return Err(CompleteStepError::ChildWitness {
                    index,
                    reason: "fixed-key commitment differs from the verified child",
                });
            }
        }

        #[cfg(test)]
        let started = std::time::Instant::now();
        let logical = self.package.execute_stage1_v1_1_ccs_assignment(
            inputs.pi_ccs(),
            inputs.pi_dec(),
            inputs.application_witness(),
        )?;
        #[cfg(test)]
        eprintln!("complete CCS assignment elapsed={:?}", started.elapsed());
        #[cfg(test)]
        let started = std::time::Instant::now();
        let blocks = self.structure.m.div_ceil(D);
        if logical.len() != self.structure.m || blocks == 0 || blocks > PRODUCTION_MESSAGE_COLUMNS as usize {
            return Err(CompleteStepError::Input(
                "fresh logical assignment differs from the fixed-key carrier",
            ));
        }
        let public_width = self.package.logical_public_input_count();
        if inputs.next_public_input().len() != public_width {
            return Err(CompleteStepError::Input("fresh public input has the wrong width"));
        }
        for (column, &expected) in inputs.next_public_input().iter().enumerate() {
            if logical.value(column)? != expected {
                return Err(CompleteStepError::Input(
                    "fresh logical public output differs from the caller packet",
                ));
            }
        }

        let mut positive = vec![0u64; blocks];
        let mut negative = vec![0u64; blocks];
        for (column, &value) in logical.balanced_values().iter().enumerate() {
            let mask = 1u64 << (column % D);
            match value {
                0 => {}
                1 => positive[column / D] |= mask,
                -1 => negative[column / D] |= mask,
                _ => {
                    return Err(CompleteStepError::Input(
                        "fresh logical assignment exceeds the strict unit norm",
                    ))
                }
            }
        }
        // Bits outside the logical assignment stay zero in the complete carrier.
        let packed = Mat::<F>::compact_signed_unit_from_column_masks(D, blocks, &positive, &negative)
            .map_err(CompleteStepError::Input)?;
        drop((logical, positive, negative));
        #[cfg(test)]
        eprintln!("complete logical packing elapsed={:?}", started.elapsed());
        #[cfg(test)]
        let started = std::time::Instant::now();
        let commitment = self.prover.commit(std::slice::from_ref(&packed))?.remove(0);
        #[cfg(test)]
        eprintln!("complete fresh commitment elapsed={:?}", started.elapsed());
        let fresh = CcsInstance {
            claim: CcsClaim {
                c: commitment,
                x: inputs
                    .next_public_input()
                    .iter()
                    .copied()
                    .map(F::from_u64)
                    .collect(),
                m_in: public_width,
                adv: None,
            },
            witness: CcsWitness {
                w: Vec::new(),
                Z: packed,
            },
        };
        let state = inputs.next_state();
        let mut running = inputs.into_running();
        running.witnesses = child_witnesses;
        Ok(Stage1Envelope {
            state,
            proof: ProofState::active(running, LatestInstance::from_instances(vec![fresh])),
        })
    }
}
