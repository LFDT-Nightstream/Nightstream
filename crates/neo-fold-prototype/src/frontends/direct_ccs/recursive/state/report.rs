//! Summary/reporting flow for the recursive direct-CCS carrier.

use super::super::*;
use super::DirectCcsRecursiveIvcState;

impl DirectCcsRecursiveIvcState {
    pub fn summary(&self) -> DirectCcsRecursiveIvcSummary {
        self.summary_inner(false)
    }

    pub fn summary_with_verifier_body_measurement(&self) -> DirectCcsRecursiveIvcSummary {
        self.summary_inner(true)
    }

    fn summary_inner(&self, measure_verifier_body: bool) -> DirectCcsRecursiveIvcSummary {
        let semantic_chunks = self.direct.final_state().chunk_count;
        let f_prime_summary = self.f_prime_chain.summary();
        let carried_f_prime_ce_claims = self
            .f_prime_chain
            .state()
            .map_or(0, |state| state.final_state().carry.claims.len());
        let folded_f_prime_r2_steps = f_prime_summary.map_or(0, |summary| summary.folded_r2_steps);
        let expected_folded_f_prime_r2_steps = semantic_chunks.saturating_sub(1);
        let f_prime_chain_has_authority = f_prime_summary.is_some_and(|summary| summary.has_proof_authority);
        let f_prime_encoder_required = expected_folded_f_prime_r2_steps > 0 && !f_prime_chain_has_authority;
        let f_prime_encoder_status = if measure_verifier_body {
            DirectCcsFPrimeEncoderStatus::from_direct_state_with_verifier_body_measurement(
                &self.direct,
                f_prime_encoder_required,
                true,
            )
        } else {
            DirectCcsFPrimeEncoderStatus::from_direct_state(&self.direct, f_prime_encoder_required)
        };
        let f_prime_encoder_blocker = f_prime_encoder_status.blocker;
        let standalone_proof_authority_ready = semantic_chunks > 0
            && folded_f_prime_r2_steps.checked_add(1) == Some(semantic_chunks)
            && (expected_folded_f_prime_r2_steps == 0 || f_prime_chain_has_authority);
        DirectCcsRecursiveIvcSummary {
            semantic: DirectCcsRecursiveSemanticSummary {
                chunks: semantic_chunks,
                steps: self.direct.final_state().step_count,
                terminal_chunks_synthesized: u64::from(semantic_chunks > 0),
                carried_ce_claims: self.direct.final_state().carry.claims.len(),
            },
            f_prime: DirectCcsRecursiveFPrimeSummary::from_encoder_status(
                folded_f_prime_r2_steps,
                carried_f_prime_ce_claims,
                f_prime_encoder_required,
                f_prime_encoder_status,
            ),
            proof: DirectCcsRecursiveProofSummary {
                standalone_authority_ready: standalone_proof_authority_ready,
                encoder_blocker: f_prime_encoder_blocker,
            },
        }
    }
}
