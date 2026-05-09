//! Recursive compression flow for the direct-CCS carrier.

use super::super::super::f_prime::chain::DIRECT_CCS_F_PRIME_LOW_NORM_ENCODER_BLOCKER;
use super::super::*;
use super::DirectCcsRecursiveIvcState;

impl DirectCcsRecursiveIvcState {
    pub fn compress_recursive_snark(
        &self,
    ) -> Result<
        (
            DirectCcsRecursiveIvcSnark,
            DirectCcsRecursiveIvcSnarkVerifierKey,
            DirectCcsRecursiveIvcSnarkPerf,
        ),
        DirectCcsFPrimeSnarkError,
    > {
        let mut emit = |_message: &str| {};
        self.compress_recursive_snark_with_trace(&mut emit)
    }

    pub fn compress_recursive_snark_with_trace(
        &self,
        emit: &mut dyn FnMut(&str),
    ) -> Result<
        (
            DirectCcsRecursiveIvcSnark,
            DirectCcsRecursiveIvcSnarkVerifierKey,
            DirectCcsRecursiveIvcSnarkPerf,
        ),
        DirectCcsFPrimeSnarkError,
    > {
        if self.direct.final_state().chunk_count == 0 {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct recursive IVC compression requires at least one appended F' step".into(),
            ));
        }
        let expected_folded_f_prime_steps = self.direct.final_state().chunk_count.saturating_sub(1);
        let folded_f_prime_steps = self
            .f_prime_chain
            .summary()
            .map_or(0, |summary| summary.folded_r2_steps);
        let summary = self.summary_with_verifier_body_measurement();
        if folded_f_prime_steps != expected_folded_f_prime_steps || !summary.proof.standalone_authority_ready {
            let blocker = summary
                .proof
                .encoder_blocker
                .unwrap_or(DIRECT_CCS_F_PRIME_LOW_NORM_ENCODER_BLOCKER);
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct recursive IVC refuses to fold terminal committed/source-image machinery; \
                 a low-norm enc(F') relation for the verifier-shaped direct F' body must be implemented \
                 before multi-step direct CCS recursive compression can be proof-complete; "
                    .to_owned()
                    + blocker,
            ));
        }
        let f_prime_chain_state = self.f_prime_accumulator_state_for_compression()?;
        let (terminal, terminal_vk, terminal_perf) = self.direct.compress_snark_with_trace(emit)?;
        let f_prime_final_claims = canonical_direct_ce_claims(&f_prime_chain_state.final_state().carry.claims);
        let f_prime_final_ce_claims = f_prime_final_claims.len() as u64;
        let f_prime_final_digest =
            direct_accumulator_digest_from_claims(f_prime_chain_state.params(), &f_prime_final_claims);
        let (
            f_prime_chain_snark,
            f_prime_chain_vk,
            f_prime_chain_perf,
            f_prime_chain_verify_ms,
            f_prime_chain_constraints,
            f_prime_chain_proof_bytes,
        ) = if folded_f_prime_steps == 0 {
            (None, None, None, 0.0, 0, 0)
        } else {
            if f_prime_chain_state.final_state().chunk_count != folded_f_prime_steps
                || f_prime_chain_state.final_state().step_count != folded_f_prime_steps
            {
                return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
            }
            emit("direct_ccs_recursive.phase=f_prime_chain_compress.start");
            let (chain_snark, chain_vk, chain_perf) = f_prime_chain_state.compress_snark_with_trace(emit)?;
            emit("direct_ccs_recursive.phase=f_prime_chain_compress.done");
            if chain_snark.public_image().accumulator_out_digest != f_prime_final_digest
                || chain_snark.public_image().chunk_count_out != folded_f_prime_steps
                || chain_snark.public_image().step_count_out != folded_f_prime_steps
            {
                return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
            }
            let chain_verify_started = Instant::now();
            chain_snark.verify(&chain_vk, chain_snark.public_image())?;
            let chain_verify_ms = chain_verify_started.elapsed().as_secs_f64() * 1_000.0;
            let chain_constraints = chain_perf.terminal_f_prime_constraints;
            let chain_proof_bytes = chain_perf.final_proof_bytes;
            (
                Some(chain_snark),
                Some(chain_vk),
                Some(chain_perf),
                chain_verify_ms,
                chain_constraints,
                chain_proof_bytes,
            )
        };
        let (f_prime_chain_setup_ms, f_prime_chain_prove_ms) = f_prime_chain_perf
            .as_ref()
            .map_or((0.0, 0.0), |perf| (perf.setup_ms, perf.prove_ms));
        let public_image = DirectCcsRecursiveIvcPublicImage::from_terminal_and_f_prime_accumulator(
            terminal.public_image().clone(),
            f_prime_final_digest,
            f_prime_chain_state.params().b,
            folded_f_prime_steps,
            f_prime_final_ce_claims,
        )?;

        let expected_f_prime_default_accumulator_digest = if folded_f_prime_steps == 0 {
            f_prime_final_digest
        } else {
            self.default_f_prime_accumulator_digest_for_direct_state(&self.direct)?
        };
        let (
            f_prime_final_ce_proof,
            f_prime_final_ce_vk,
            f_prime_final_ce_setup_ms,
            f_prime_final_ce_prove_ms,
            f_prime_final_ce_verify_ms,
            f_prime_final_ce_constraints,
            f_prime_final_ce_digest_constraints,
            f_prime_final_ce_digest_match_constraints,
            f_prime_final_ce_relation_constraints,
            f_prime_final_ce_public_inputs,
            f_prime_final_ce_proof_bytes,
        ) = if folded_f_prime_steps == 0 {
            emit("direct_ccs_recursive.phase=f_prime_final_ce_proof.default_accumulator_skipped");
            (None, None, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0)
        } else {
            let witnesses = direct_ce_bundle_witnesses(&f_prime_chain_state.final_state().carry.witnesses)?;
            emit("direct_ccs_recursive.phase=f_prime_final_ce_measure.start");
            let measured = measure_direct_ce_bundle_relation(
                f_prime_chain_state.params(),
                f_prime_chain_state.structure(),
                &f_prime_final_claims,
                &witnesses,
            )?;
            emit("direct_ccs_recursive.phase=f_prime_final_ce_measure.done");
            emit("direct_ccs_recursive.phase=f_prime_final_ce_setup.start");
            let setup_started = Instant::now();
            let (pk, vk) = setup_direct_ce_bundle_relation(
                f_prime_chain_state.params(),
                f_prime_chain_state.structure(),
                &f_prime_final_claims,
                &witnesses,
            )?;
            let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;
            emit("direct_ccs_recursive.phase=f_prime_final_ce_setup.done");
            emit("direct_ccs_recursive.phase=f_prime_final_ce_prove.start");
            let prove_started = Instant::now();
            let proof = prove_direct_ce_bundle_relation(
                &pk,
                f_prime_chain_state.params(),
                f_prime_chain_state.structure(),
                &f_prime_final_claims,
                &witnesses,
            )?;
            let prove_ms = prove_started.elapsed().as_secs_f64() * 1_000.0;
            let proof_bytes = proof.snark_bytes_len();
            emit("direct_ccs_recursive.phase=f_prime_final_ce_prove.done");
            emit("direct_ccs_recursive.phase=f_prime_final_ce_verify.start");
            let verify_started = Instant::now();
            verify_direct_ce_bundle_relation(&vk, &f_prime_final_claims, &proof)?;
            let verify_ms = verify_started.elapsed().as_secs_f64() * 1_000.0;
            emit("direct_ccs_recursive.phase=f_prime_final_ce_verify.done");
            (
                Some(proof),
                Some(vk),
                setup_ms,
                prove_ms,
                verify_ms,
                measured.total_constraints,
                measured.digest_constraints,
                measured.digest_match_constraints,
                measured.ce_relation_constraints,
                measured.public_input_count,
                proof_bytes,
            )
        };
        let perf = DirectCcsRecursiveIvcSnarkPerf {
            terminal: terminal_perf.clone(),
            f_prime_chain: f_prime_chain_perf.clone(),
            f_prime_chain_setup_ms,
            f_prime_chain_prove_ms,
            f_prime_chain_verify_ms,
            f_prime_chain_constraints,
            f_prime_chain_proof_bytes,
            f_prime_final_ce_setup_ms,
            f_prime_final_ce_prove_ms,
            f_prime_final_ce_verify_ms,
            f_prime_final_ce_constraints,
            f_prime_final_ce_digest_constraints,
            f_prime_final_ce_digest_match_constraints,
            f_prime_final_ce_relation_constraints,
            f_prime_final_ce_public_inputs,
            f_prime_final_ce_claims: f_prime_final_ce_claims as usize,
            total_prove_ms: terminal_perf.total_prove_ms
                + f_prime_chain_perf
                    .as_ref()
                    .map_or(0.0, |perf| perf.total_prove_ms)
                + f_prime_final_ce_prove_ms,
            total_verify_ms: terminal_perf.total_verify_ms + f_prime_chain_verify_ms + f_prime_final_ce_verify_ms,
            terminal_proof_bytes: terminal_perf.final_proof_bytes,
            f_prime_final_ce_proof_bytes,
            total_proof_bytes: terminal_perf.final_proof_bytes
                + f_prime_chain_proof_bytes
                + f_prime_final_ce_proof_bytes,
        };
        Ok((
            DirectCcsRecursiveIvcSnark {
                terminal,
                f_prime_chain: f_prime_chain_snark,
                f_prime_final_claims: if folded_f_prime_steps == 0 {
                    Vec::new()
                } else {
                    f_prime_final_claims
                },
                f_prime_final_ce_proof,
                public_image,
            },
            DirectCcsRecursiveIvcSnarkVerifierKey {
                terminal: terminal_vk,
                f_prime_chain: f_prime_chain_vk,
                f_prime_final_ce: f_prime_final_ce_vk,
                expected_f_prime_default_accumulator_digest,
                expected_f_prime_accumulator_base: f_prime_chain_state.params().b,
                expected_f_prime_final_ce_claims: f_prime_final_ce_claims,
            },
            perf,
        ))
    }

    fn default_f_prime_accumulator_digest_for_direct_state(
        &self,
        direct: &DirectCcsIvcState,
    ) -> Result<[u8; 32], DirectCcsFPrimeSnarkError> {
        Ok(direct.construction2_accumulator_digest)
    }

    fn f_prime_accumulator_state_for_compression(&self) -> Result<DirectCcsIvcState, DirectCcsFPrimeSnarkError> {
        if let Some(state) = self.f_prime_chain.state() {
            return Ok(state.clone());
        }
        let public_input_len = self.direct.public_input_len.ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input(
                "direct recursive IVC default F' accumulator requires a fixed direct public input length".into(),
            )
        })?;
        let program = DirectCcsProgram::new_with_public_input_len(
            self.direct.params(),
            self.direct.structure(),
            public_input_len,
        )?;
        DirectCcsIvcState::new_with_canonical_zero_carry(program)
    }
}
