//! Wire-adjacent state for the superseded block/lane SplitNC protocol.
//!
//! These types support accelerator replay and deferred proof assembly. They
//! are not part of the canonical `PaperRectangularV1` protocol.

use crate::error::PiCcsError;
use neo_math::K;

use super::{Challenges, PiCcsProof, PiCcsProofVariant, PiCcsProvePerf, PiDecProverPrecompute};

#[derive(Debug, Clone)]
pub struct PiCcsTerminalOutputShell {
    pub count: usize,
    pub m_in: usize,
    pub row_chals: Vec<K>,
    pub s_col: Vec<K>,
    pub has_y_zcol: bool,
    pub fold_digest: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct PiCcsReplayTerminalState {
    pub variant: PiCcsProofVariant,
    pub me_outputs: Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>>,
    pub output_shell: PiCcsTerminalOutputShell,
    pub sc_initial_sum: K,
    pub sc_initial_sum_nc: K,
    pub challenges_public: Challenges,
    pub row_chals: Vec<K>,
    pub alpha_prime: Vec<K>,
    pub s_col: Vec<K>,
    pub alpha_prime_nc: Vec<K>,
    pub sumcheck_final: K,
    pub sumcheck_final_nc: K,
    pub fold_digest: [u8; 32],
    pub perf: PiCcsProvePerf,
    #[doc(hidden)]
    pub pi_dec_precompute: Option<PiDecProverPrecompute>,
}

#[derive(Debug, Clone)]
pub struct PiCcsReplayOutputs {
    pub me_outputs: Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>>,
    pub fold_digest: [u8; 32],
    pub perf: PiCcsProvePerf,
}

#[derive(Debug, Clone)]
pub struct PiCcsReplayWitnessOutputs {
    pub me_outputs: Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>>,
    pub replay_proof: PiCcsReplayProofWitness,
    pub perf: PiCcsProvePerf,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PiCcsReplayProofWitness {
    pub sumcheck_rounds: Vec<Vec<K>>,
    pub sumcheck_rounds_nc: Vec<Vec<K>>,
    pub header_digest: [u8; 32],
}

impl PiCcsReplayProofWitness {
    pub fn from_proof(proof: &PiCcsProof) -> Result<Self, PiCcsError> {
        if proof.variant != PiCcsProofVariant::SplitNcV1 {
            return Err(PiCcsError::ProtocolError(
                "unsupported legacy SplitNC replay proof variant".into(),
            ));
        }
        let header_digest: [u8; 32] = proof
            .header_digest
            .as_slice()
            .try_into()
            .map_err(|_| PiCcsError::ProtocolError("PiCCS header digest must be 32 bytes".into()))?;
        Ok(Self {
            sumcheck_rounds: proof.sumcheck_rounds.clone(),
            sumcheck_rounds_nc: proof.sumcheck_rounds_nc.clone(),
            header_digest,
        })
    }

    pub fn to_pi_ccs_proof(&self) -> PiCcsProof {
        let mut proof = PiCcsProof::new(self.sumcheck_rounds.clone(), None);
        proof.variant = PiCcsProofVariant::SplitNcV1;
        proof.sumcheck_rounds_nc = self.sumcheck_rounds_nc.clone();
        proof.header_digest = self.header_digest.to_vec();
        proof
    }
}
