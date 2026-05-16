//! Recursive append flow for the direct-CCS carrier.
//!
//! Each append performs the Construction-2/F' carrier work around one current
//! SuperNeo step:
//! 1. fold the previous direct step into the private F' authority chain,
//! 2. append the current direct step to the public SuperNeo carrier,
//! 3. attach the prior F' fold context to the current terminal circuit.

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
        let prior_f_prime = self.advance_prior_f_prime_authority(mixers)?;
        let direct = self.append_current_direct_step(step, log, mixers, &prior_f_prime)?;
        Ok(self.with_next_direct_state(direct, prior_f_prime))
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
        let prior_f_prime = self.advance_prior_f_prime_authority(mixers)?;
        let direct = self.append_current_verified_relation(relation, log, mixers, &prior_f_prime)?;
        Ok(self.with_next_direct_state(direct, prior_f_prime))
    }

    fn advance_prior_f_prime_authority<MR, MB>(
        &self,
        mixers: CommitmentMixers<MR, MB>,
    ) -> Result<PriorFPrimeAuthority, DirectCcsFPrimeSnarkError>
    where
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let chain = if self.direct.final_state().chunk_count == 0 {
            self.f_prime_chain.clone()
        } else {
            self.f_prime_chain
                .append_latest_step_authority_from_direct_state(&self.direct, mixers)?
        };
        PriorFPrimeAuthority::from_chain(&self.direct, chain)
    }

    fn append_current_direct_step<MR, MB>(
        &self,
        step: DirectCcsStep,
        log: &AjtaiSModule,
        mixers: CommitmentMixers<MR, MB>,
        prior_f_prime: &PriorFPrimeAuthority,
    ) -> Result<DirectCcsIvcState, DirectCcsFPrimeSnarkError>
    where
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        self.direct
            .append_step_with_f_prime_accumulator(step, log, mixers, prior_f_prime.accumulator_digest)?
            .with_latest_construction2_fold_context(prior_f_prime.fold_context.clone())
    }

    fn append_current_verified_relation<MR, MB>(
        &self,
        relation: &SuperNeoIvcStepRelation,
        log: &AjtaiSModule,
        mixers: CommitmentMixers<MR, MB>,
        prior_f_prime: &PriorFPrimeAuthority,
    ) -> Result<DirectCcsIvcState, DirectCcsFPrimeSnarkError>
    where
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        self.direct
            .append_relation_with_f_prime_accumulator(relation, log, mixers, prior_f_prime.accumulator_digest)?
            .with_latest_construction2_fold_context(prior_f_prime.fold_context.clone())
    }

    fn with_next_direct_state(&self, direct: DirectCcsIvcState, prior_f_prime: PriorFPrimeAuthority) -> Self {
        Self {
            direct,
            f_prime_chain: prior_f_prime.chain,
        }
    }
}

struct PriorFPrimeAuthority {
    chain: DirectCcsFPrimeChain,
    fold_context: Option<DirectCcsConstruction2FoldContext>,
    accumulator_digest: [u8; 32],
}

impl PriorFPrimeAuthority {
    fn from_chain(direct: &DirectCcsIvcState, chain: DirectCcsFPrimeChain) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let fold_context = chain
            .state()
            .map(DirectCcsIvcState::latest_construction2_fold_context)
            .transpose()?;
        let accumulator_digest = match chain.state() {
            Some(state) => direct_accumulator_digest_from_claims(state.params(), &state.final_state().carry.claims),
            None => direct.construction2_accumulator_digest,
        };
        Ok(Self {
            chain,
            fold_context,
            accumulator_digest,
        })
    }
}
