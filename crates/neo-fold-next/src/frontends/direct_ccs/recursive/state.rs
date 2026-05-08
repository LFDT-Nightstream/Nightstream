//! Owns the direct-CCS recursive carrier state machine.
//!
//! The public flow is append, summarize, then compress. Helpers below keep the
//! prior F' chain preflight and default accumulator handling out of the type umbrella.

use super::*;

impl DirectCcsRecursiveIvcState {
    pub fn new_with_canonical_zero_carry(program: DirectCcsProgram) -> Result<Self, DirectCcsFPrimeSnarkError> {
        Ok(Self {
            direct: DirectCcsIvcState::new_with_canonical_zero_carry(program)?,
            f_prime_chain: DirectCcsFPrimeChain::new(),
        })
    }

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

    pub fn direct_state(&self) -> &DirectCcsIvcState {
        &self.direct
    }

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
        let low_norm_source_shape = f_prime_encoder_status.low_norm_source_r1cs_shape;
        let verifier_body_shape = f_prime_encoder_status.verifier_body_shape.as_ref();
        let f_prime_encoder_available = f_prime_encoder_status.low_norm_relation_available;
        let standalone_proof_authority_ready = semantic_chunks > 0
            && folded_f_prime_r2_steps.checked_add(1) == Some(semantic_chunks)
            && (expected_folded_f_prime_r2_steps == 0 || f_prime_chain_has_authority);
        DirectCcsRecursiveIvcSummary {
            semantic_chunks,
            semantic_steps: self.direct.final_state().step_count,
            terminal_chunks_synthesized: u64::from(semantic_chunks > 0),
            carried_semantic_ce_claims: self.direct.final_state().carry.claims.len(),
            folded_f_prime_r2_steps,
            carried_f_prime_ce_claims,
            native_f_prime_evaluator_available: f_prime_encoder_status.native_evaluator_available,
            f_prime_encoder_required,
            f_prime_encoder_available,
            compact_f_prime_image_digest: f_prime_encoder_status.compact_image_digest,
            low_norm_f_prime_source_available: f_prime_encoder_status.low_norm_source_available,
            low_norm_f_prime_source_len: f_prime_encoder_status.low_norm_source_len,
            low_norm_f_prime_source_digest: f_prime_encoder_status.low_norm_source_digest,
            low_norm_f_prime_source_r1cs_constraints: low_norm_source_shape.map_or(0, |shape| shape.constraint_count),
            low_norm_f_prime_source_r1cs_variables: low_norm_source_shape.map_or(0, |shape| shape.variable_count),
            low_norm_f_prime_source_r1cs_nnz: low_norm_source_shape.map_or(0, |shape| shape.nonzero_entries),
            low_norm_f_prime_source_public_inputs: low_norm_source_shape.map_or(0, |shape| shape.public_input_len),
            low_norm_f_prime_source_private_bits: low_norm_source_shape.map_or(0, |shape| shape.source_len),
            low_norm_f_prime_source_counter_carry_bits: low_norm_source_shape
                .map_or(0, |shape| shape.counter_carry_bits),
            low_norm_f_prime_source_digest_count: f_prime_encoder_status.low_norm_source_digest_count,
            low_norm_f_prime_source_u64_count: f_prime_encoder_status.low_norm_source_u64_count,
            low_norm_f_prime_source_encoded_public_input_count: f_prime_encoder_status
                .low_norm_source_encoded_public_input_count,
            low_norm_f_prime_source_field_lane_count: f_prime_encoder_status.low_norm_source_field_lane_count,
            low_norm_f_prime_source_construction2_commitment_fields: f_prime_encoder_status
                .low_norm_source_construction2_commitment_fields,
            low_norm_f_prime_nifs_payload_shape: f_prime_encoder_status.nifs_payload_shape,
            f_prime_verifier_body_measured: verifier_body_shape.is_some(),
            f_prime_verifier_body_measure_skipped: f_prime_encoder_status.verifier_body_measure_skipped,
            f_prime_verifier_body_public_inputs: verifier_body_shape.map_or(0, |shape| shape.public_inputs),
            f_prime_verifier_body_constraints: verifier_body_shape.map_or(0, |shape| shape.constraints),
            f_prime_verifier_body_nifs_constraints: verifier_body_shape.map_or(0, |shape| shape.nifs_constraints()),
            f_prime_verifier_body_nifs_chunk_meta_constraints: verifier_body_shape
                .map_or(0, |shape| shape.nifs_chunk_meta_constraints),
            f_prime_verifier_body_nifs_pi_ccs_constraints: verifier_body_shape
                .map_or(0, |shape| shape.nifs_pi_ccs_constraints),
            f_prime_verifier_body_nifs_pi_rlc_constraints: verifier_body_shape
                .map_or(0, |shape| shape.nifs_pi_rlc_constraints),
            f_prime_verifier_body_nifs_pi_dec_constraints: verifier_body_shape
                .map_or(0, |shape| shape.nifs_pi_dec_constraints),
            f_prime_verifier_body_construction2_fold_constraints: verifier_body_shape
                .map_or(0, |shape| shape.construction2_fold_constraints),
            f_prime_verifier_body_public_link_constraints: verifier_body_shape
                .map_or(0, |shape| shape.public_link_constraints),
            f_prime_verifier_body_chunk_done_constraints: verifier_body_shape
                .map_or(0, |shape| shape.chunk_done_constraints),
            f_prime_verifier_body_final_ce_relation_constraints: verifier_body_shape
                .map_or(0, |shape| shape.final_ce_relation_constraints),
            f_prime_exact_encoder_row_cap: DIRECT_CCS_F_PRIME_EXACT_ENCODER_MAX_R1CS_CONSTRAINTS,
            low_norm_f_prime_source_shell_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.shell_constraints()),
            low_norm_f_prime_source_bit_constraints: low_norm_source_shape.map_or(0, |shape| shape.bit_constraints),
            low_norm_f_prime_source_x_out_link_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.x_out_link_constraints),
            low_norm_f_prime_source_construction2_boundary_link_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.construction2_boundary_link_constraints),
            low_norm_f_prime_source_construction2_instance_digest_link_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.construction2_instance_digest_link_constraints),
            low_norm_f_prime_source_construction2_commitment_shape_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.construction2_commitment_shape_constraints),
            low_norm_f_prime_source_structural_counter_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.structural_counter_constraints),
            low_norm_f_prime_source_structural_fixed_arity_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.structural_fixed_arity_constraints),
            low_norm_f_prime_source_structural_counter_carry_bit_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.structural_counter_carry_bit_constraints),
            low_norm_f_prime_source_canonical_field_lane_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.canonical_field_lane_constraints),
            low_norm_f_prime_source_canonical_field_lane_aux_bits: low_norm_source_shape
                .map_or(0, |shape| shape.canonical_field_lane_aux_bits),
            low_norm_f_prime_source_poseidon_digest_recomputation_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.poseidon_digest_recomputation_constraints),
            low_norm_f_prime_source_nifs_v_verifier_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.nifs_v_verifier_constraints),
            low_norm_f_prime_source_authority_constraints: low_norm_source_shape
                .map_or(0, |shape| shape.authority_constraints()),
            f_prime_encoder_blocker: f_prime_encoder_status.blocker,
            standalone_proof_authority_ready,
        }
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

    fn default_f_prime_accumulator_digest_for_direct_state(
        &self,
        direct: &DirectCcsIvcState,
    ) -> Result<[u8; 32], DirectCcsFPrimeSnarkError> {
        Ok(direct.construction2_accumulator_digest)
    }

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
        if folded_f_prime_steps != expected_folded_f_prime_steps || !summary.standalone_proof_authority_ready {
            let blocker = summary
                .f_prime_encoder_blocker
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
