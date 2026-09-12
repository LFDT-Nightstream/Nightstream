//! Selected proving from actual source claims and witnesses.

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_reductions::{optimized_engine::optimized_prove_with_row_cache, PiCcsError};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::{
    PackageError, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_SOURCE_COUNT, PI_DEC_V1_1_CHILD_COUNT,
};
use p3_field::PrimeCharacteristicRing;

use super::Poseidon2HashChainV1Package;
use crate::engine::transcript::Transcript;
use crate::paper::{
    construction2::RunningInstance,
    nifs,
    params::Params,
    pi_ccs, pi_rlc,
    reductions::pi_ccs::Proof,
    relations::{ajtai_dec_mixer, ajtai_rlc_mixer, CcsInstance},
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

impl Poseidon2HashChainV1Package {
    /// Prove C, R, then D through the normal NIFS owner. This package fixes
    /// the header, key and parameters and builds one row cache for both C/D.
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
            &cache,
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

    /// Fixture boundary for the actual C/R prefix. This returns a parent,
    /// not a complete NIFS proof. The full prover uses the same phase helper.
    /// The verifier replay also checks the original running-parent authority
    /// and the exact producer transcript position before any output is saved.
    #[doc(hidden)]
    pub fn prove_parent(
        &self,
        fresh: Vec<CcsInstance>,
        running: RunningInstance,
    ) -> Result<(pi_ccs::Proof, pi_rlc::Output), ProveError> {
        self.validate_prover_sources(&fresh, &running)?;
        let params = Params::for_ccs_shape(
            self.structure.n,
            self.structure.m,
            self.structure.t(),
            self.structure.max_degree(),
        )?;
        let fresh_claims = fresh
            .iter()
            .map(|source| source.claim.clone())
            .collect::<Vec<_>>();
        let prior = running.claims_only();
        let cache = self.build_superneo_cache()?;
        let mut transcript = Transcript::session();
        let (proof, parent) =
            nifs::prove_parent_with_rows(&mut transcript, &params, &self.structure, &cache, fresh, running)?;
        drop(cache);
        let mut replay = Transcript::session();
        nifs::validate_running_parent_authority(&params, &self.structure, ajtai_dec_mixer, &prior)?;
        let outputs = pi_ccs::verify(&mut replay, &params, &self.structure, &fresh_claims, &prior, &proof)
            .map_err(nifs::Error::from)?;
        pi_rlc::verify(
            &mut replay,
            &params,
            &self.structure,
            ajtai_rlc_mixer,
            &outputs,
            &pi_rlc::Proof {
                combined: parent.claim.clone(),
            },
        )
        .map_err(nifs::Error::from)?;
        if replay.snapshot() != transcript.snapshot() {
            return Err(ProveError::Input(
                "C/R producer and verifier transcript positions differ",
            ));
        }
        Ok((proof, parent))
    }

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
