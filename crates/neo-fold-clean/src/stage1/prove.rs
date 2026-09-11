//! Selected PiCCS proving from actual source claims and witnesses.

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_reductions::{optimized_engine::optimized_prove_with_row_cache, PiCcsError};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::{PI_CCS_V1_1_SOURCE_COUNT, PI_DEC_V1_1_CHILD_COUNT};
use p3_field::PrimeCharacteristicRing;

use super::Poseidon2HashChainV1Package;
use crate::paper::{params::Params, reductions::pi_ccs::Proof};

impl Poseidon2HashChainV1Package {
    /// Prove the selected PiCCS phase. Header, parameters, transcript and
    /// matrix arithmetic come from this package, not caller-supplied caches.
    pub fn prove_pi_ccs(
        &self,
        fresh_claims: &[CcsClaim<Commitment, F>],
        fresh_witnesses: &[CcsWitness<F>],
        running_claims: &[CeClaim<Commitment, F, K>],
        running_witnesses: &[Mat<F>],
    ) -> Result<Proof, PiCcsError> {
        if fresh_claims.len() != PI_CCS_V1_1_SOURCE_COUNT - PI_DEC_V1_1_CHILD_COUNT
            || running_claims.len() != PI_DEC_V1_1_CHILD_COUNT
            || fresh_claims.len() != fresh_witnesses.len()
            || running_claims.len() != running_witnesses.len()
        {
            return Err(PiCcsError::InvalidInput("selected PiCCS source counts".into()));
        }
        let params = Params::for_ccs_shape(
            self.structure.n,
            self.structure.m,
            self.structure.t(),
            self.structure.max_degree(),
        )
        .map_err(|error| PiCcsError::InvalidInput(error.to_string()))?;
        let cache = self
            .build_superneo_cache()
            .map_err(|error| PiCcsError::InvalidInput(error.to_string()))?;
        let mut transcript = Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 8], 0);
        let (outputs, sumcheck, _, _) = optimized_prove_with_row_cache(
            &mut transcript,
            params.inner(),
            &self.structure,
            fresh_claims,
            fresh_witnesses,
            running_claims,
            running_witnesses,
            &cache,
        )?;
        Ok(Proof { sumcheck, outputs })
    }
}
