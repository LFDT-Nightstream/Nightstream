//! Append flow for the recursive direct-CCS carrier.

use super::super::*;
use super::DirectCcsRecursiveIvcState;

impl DirectCcsRecursiveIvcState {
    pub fn append_step<MR, MB>(
        &self,
        step: DirectCcsStep,
        log: &AjtaiSModule,
        mixers: CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let f_prime_chain = self.preflight_prior_f_prime_instance_if_any(mixers)?;
        let construction2_fold = f_prime_chain
            .state()
            .map(DirectCcsIvcState::latest_construction2_fold_context)
            .transpose()?;
        let construction2_accumulator_digest =
            self.construction2_accumulator_digest_after_prior_step(&f_prime_chain, step.clone(), log, mixers)?;
        let direct = self
            .direct
            .append_step_with_construction2_accumulator_digest(step, log, mixers, construction2_accumulator_digest)?
            .with_latest_construction2_fold_context(construction2_fold)?;
        Ok(Self { direct, f_prime_chain })
    }

    pub fn append_relation<MR, MB>(
        &self,
        relation: &SuperNeoIvcStepRelation,
        log: &AjtaiSModule,
        mixers: CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let f_prime_chain = self.preflight_prior_f_prime_instance_if_any(mixers)?;
        let construction2_fold = f_prime_chain
            .state()
            .map(DirectCcsIvcState::latest_construction2_fold_context)
            .transpose()?;
        let construction2_accumulator_digest =
            self.construction2_accumulator_digest_after_prior_relation(&f_prime_chain, relation, log, mixers)?;
        let direct = self
            .direct
            .append_relation_with_construction2_accumulator_digest(
                relation,
                log,
                mixers,
                construction2_accumulator_digest,
            )?
            .with_latest_construction2_fold_context(construction2_fold)?;
        Ok(Self { direct, f_prime_chain })
    }

    fn preflight_prior_f_prime_instance_if_any<MR, MB>(
        &self,
        mixers: CommitmentMixers<MR, MB>,
    ) -> Result<DirectCcsFPrimeChain, DirectCcsFPrimeSnarkError>
    where
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        if self.direct.final_state().chunk_count == 0 {
            return Ok(self.f_prime_chain.clone());
        }
        self.f_prime_chain
            .preflight_compact_low_norm_source_from_direct_state(&self.direct, mixers)
    }

    fn construction2_accumulator_digest_after_prior_step<MR, MB>(
        &self,
        f_prime_chain: &DirectCcsFPrimeChain,
        step: DirectCcsStep,
        log: &AjtaiSModule,
        mixers: CommitmentMixers<MR, MB>,
    ) -> Result<[u8; 32], DirectCcsFPrimeSnarkError>
    where
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        if let Some(state) = f_prime_chain.state() {
            return Ok(direct_accumulator_digest_from_claims(
                state.params(),
                &state.final_state().carry.claims,
            ));
        }
        let _ = (step, log, mixers);
        Ok(self.direct.construction2_accumulator_digest)
    }

    fn construction2_accumulator_digest_after_prior_relation<MR, MB>(
        &self,
        f_prime_chain: &DirectCcsFPrimeChain,
        relation: &SuperNeoIvcStepRelation,
        log: &AjtaiSModule,
        mixers: CommitmentMixers<MR, MB>,
    ) -> Result<[u8; 32], DirectCcsFPrimeSnarkError>
    where
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        if let Some(state) = f_prime_chain.state() {
            return Ok(direct_accumulator_digest_from_claims(
                state.params(),
                &state.final_state().carry.claims,
            ));
        }
        let _ = (relation, log, mixers);
        Ok(self.direct.construction2_accumulator_digest)
    }
}
