//! Owns the compact native F' image for direct CCS.
//!
//! This is the paper-shaped Construction-2 boundary that a future low-norm
//! `enc(F')` relation must prove. It intentionally carries only compact
//! counters, handles, and digests; terminal source packing and final CE checks
//! remain terminal-compression responsibilities.

use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};

use super::ivc::{DirectCcsFPrimeSnarkError, DirectCcsIvcState};
use super::public_image::{
    direct_boundary_update_digest, direct_public_trace_update_digest, direct_state_x_out, DirectCcsIvcPublicImage,
    DIRECT_CCS_TRIVIAL_PC,
};
use crate::construction2::{Construction2EncodedPublicInput, Construction2PublicBoundary};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsCompactFPrimeImage {
    pub mat_digest: [F; 4],
    pub vk_fs_digest: [u8; 32],
    pub pc: u64,
    pub chunk_count_in: u64,
    pub step_count_in: u64,
    pub chunk_count_out: u64,
    pub step_count_out: u64,
    pub initial_boundary_digest: [u8; 32],
    pub current_boundary_in_digest: [u8; 32],
    pub current_boundary_out_digest: [u8; 32],
    pub public_trace_in_digest: [u8; 32],
    pub public_trace_out_digest: [u8; 32],
    pub semantic_accumulator_in_digest: [u8; 32],
    pub semantic_accumulator_out_digest: [u8; 32],
    pub f_prime_accumulator_in_digest: [u8; 32],
    pub f_prime_accumulator_out_digest: [u8; 32],
    pub x_in: Construction2EncodedPublicInput,
    pub x_out: Construction2EncodedPublicInput,
    pub construction2_u_in_digest: [u8; 32],
    pub construction2_u_out_digest: [u8; 32],
    pub latest_chunk_digest: [F; 4],
    pub latest_fold_digest: [u8; 32],
    pub latest_chunk_relation_digest: [u8; 32],
    pub fresh_claims: u64,
    pub incoming_ce_claims: u64,
    pub output_ce_claims: u64,
    pub final_ce_claims: u64,
    pub nifs_chunk_index: u64,
    pub nifs_fresh_claims: u64,
    pub nifs_incoming_ce_claims: u64,
    pub nifs_pi_ccs_outputs: u64,
    pub nifs_final_ce_claims: u64,
    pub nifs_fe_sumcheck_rounds: u64,
    pub nifs_fe_sumcheck_messages: u64,
    pub nifs_nc_sumcheck_rounds: u64,
    pub nifs_nc_sumcheck_messages: u64,
    pub nifs_transcript_absorbed_in: u64,
    pub nifs_transcript_absorbed_out: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsNativeFPrimeAdvice {
    compact_image: DirectCcsCompactFPrimeImage,
    construction2_u_in: Construction2PublicBoundary,
    construction2_u_out: Construction2PublicBoundary,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsFPrimeNifsPayloadShape {
    pub chunk_index: u64,
    pub fresh_claims: usize,
    pub incoming_ce_claims: usize,
    pub pi_ccs_outputs: usize,
    pub final_ce_claims: usize,
    pub fe_sumcheck_rounds: usize,
    pub fe_sumcheck_messages: usize,
    pub nc_sumcheck_rounds: usize,
    pub nc_sumcheck_messages: usize,
    pub transcript_absorbed_in: usize,
    pub transcript_absorbed_out: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsFPrimeLowNormSourceImage {
    values: Vec<F>,
    mat_digest_bit_offset: usize,
    vk_fs_digest_bit_offset: usize,
    pc_bit_offset: usize,
    chunk_count_in_bit_offset: usize,
    step_count_in_bit_offset: usize,
    chunk_count_out_bit_offset: usize,
    step_count_out_bit_offset: usize,
    initial_boundary_digest_bit_offset: usize,
    current_boundary_in_digest_bit_offset: usize,
    current_boundary_out_digest_bit_offset: usize,
    public_trace_in_digest_bit_offset: usize,
    public_trace_out_digest_bit_offset: usize,
    semantic_accumulator_in_digest_bit_offset: usize,
    semantic_accumulator_out_digest_bit_offset: usize,
    f_prime_accumulator_in_digest_bit_offset: usize,
    f_prime_accumulator_out_digest_bit_offset: usize,
    compact_x_in_bit_offset: usize,
    compact_x_out_bit_offset: usize,
    compact_construction2_u_in_digest_bit_offset: usize,
    compact_construction2_u_out_digest_bit_offset: usize,
    latest_chunk_digest_bit_offset: usize,
    latest_fold_digest_bit_offset: usize,
    latest_chunk_relation_digest_bit_offset: usize,
    fresh_claims_bit_offset: usize,
    incoming_ce_claims_bit_offset: usize,
    output_ce_claims_bit_offset: usize,
    final_ce_claims_bit_offset: usize,
    nifs_chunk_index_bit_offset: usize,
    nifs_fresh_claims_bit_offset: usize,
    nifs_incoming_ce_claims_bit_offset: usize,
    nifs_pi_ccs_outputs_bit_offset: usize,
    nifs_final_ce_claims_bit_offset: usize,
    nifs_fe_sumcheck_rounds_bit_offset: usize,
    nifs_fe_sumcheck_messages_bit_offset: usize,
    nifs_nc_sumcheck_rounds_bit_offset: usize,
    nifs_nc_sumcheck_messages_bit_offset: usize,
    nifs_transcript_absorbed_in_bit_offset: usize,
    nifs_transcript_absorbed_out_bit_offset: usize,
    construction2_u_in_fresh_digest_bit_offset: usize,
    construction2_u_in_commitment_digest_bit_offset: usize,
    construction2_u_in_commitment_d_bit_offset: usize,
    construction2_u_in_commitment_kappa_bit_offset: usize,
    construction2_u_in_x_i_bit_offset: usize,
    construction2_u_out_fresh_digest_bit_offset: usize,
    construction2_u_out_commitment_digest_bit_offset: usize,
    construction2_u_out_commitment_d_bit_offset: usize,
    construction2_u_out_commitment_kappa_bit_offset: usize,
    construction2_u_out_x_i_bit_offset: usize,
    field_lane_bit_offsets: Vec<usize>,
    digest_count: usize,
    u64_count: usize,
    encoded_public_input_count: usize,
    field_lane_count: usize,
    construction2_commitment_fields: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsNativeFPrimeStepImage {
    compact_image: DirectCcsCompactFPrimeImage,
    construction2_u_out: Construction2PublicBoundary,
    terminal_public_image: DirectCcsIvcPublicImage,
}

impl DirectCcsCompactFPrimeImage {
    pub fn from_latest_state(state: &DirectCcsIvcState) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let last = state.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input("direct compact F' image requires an appended step".into())
        })?;
        if last.relation.chunk.steps.is_empty() {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct compact F' image requires at least one fresh CCS claim".into(),
            ));
        }
        if last.construction2_u_i.x_i() != &last.x_i || state.construction2_u_i.x_i() != &last.x_out {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct compact F' image Construction-2 u_i/x_i linkage is inconsistent".into(),
            ));
        }
        let nifs_payload = DirectCcsFPrimeNifsPayloadShape::from_latest_state(state)?;
        let image = Self {
            mat_digest: state.mat_digest,
            vk_fs_digest: state.vk_fs_digest,
            pc: DIRECT_CCS_TRIVIAL_PC,
            chunk_count_in: last.relation.state_in.chunk_count,
            step_count_in: last.relation.state_in.step_count,
            chunk_count_out: state.state.chunk_count,
            step_count_out: state.state.step_count,
            initial_boundary_digest: state.initial_boundary_digest,
            current_boundary_in_digest: last.current_boundary_in_digest,
            current_boundary_out_digest: last.current_boundary_out_digest,
            public_trace_in_digest: last.public_trace_in_digest,
            public_trace_out_digest: last.public_trace_out_digest,
            semantic_accumulator_in_digest: last.accumulator_in_digest,
            semantic_accumulator_out_digest: last.accumulator_out_digest,
            f_prime_accumulator_in_digest: last.construction2_accumulator_in_digest,
            f_prime_accumulator_out_digest: last.construction2_accumulator_out_digest,
            x_in: last.x_i.clone(),
            x_out: last.x_out.clone(),
            construction2_u_in_digest: Construction2PublicBoundary::from_fresh_instance(&last.construction2_u_i)
                .fresh_instance_digest,
            construction2_u_out_digest: Construction2PublicBoundary::from_fresh_instance(&state.construction2_u_i)
                .fresh_instance_digest,
            latest_chunk_digest: last.surface.replay.handoff.public_chunk_instance_digest,
            latest_fold_digest: last.relation.fold_digest,
            latest_chunk_relation_digest: last.relation.chunk_relation_digest,
            fresh_claims: last.relation.chunk.steps.len() as u64,
            incoming_ce_claims: last.relation.state_in.carry.claims.len() as u64,
            output_ce_claims: last.relation.replay_witness.ccs_outputs.len() as u64,
            final_ce_claims: state.state.carry.claims.len() as u64,
            nifs_chunk_index: nifs_payload.chunk_index,
            nifs_fresh_claims: nifs_payload.fresh_claims as u64,
            nifs_incoming_ce_claims: nifs_payload.incoming_ce_claims as u64,
            nifs_pi_ccs_outputs: nifs_payload.pi_ccs_outputs as u64,
            nifs_final_ce_claims: nifs_payload.final_ce_claims as u64,
            nifs_fe_sumcheck_rounds: nifs_payload.fe_sumcheck_rounds as u64,
            nifs_fe_sumcheck_messages: nifs_payload.fe_sumcheck_messages as u64,
            nifs_nc_sumcheck_rounds: nifs_payload.nc_sumcheck_rounds as u64,
            nifs_nc_sumcheck_messages: nifs_payload.nc_sumcheck_messages as u64,
            nifs_transcript_absorbed_in: nifs_payload.transcript_absorbed_in as u64,
            nifs_transcript_absorbed_out: nifs_payload.transcript_absorbed_out as u64,
        };
        image.validate()?;
        if image.incoming_ce_claims != state.params().k_rho as u64
            || image.final_ce_claims != state.params().k_rho as u64
        {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        Ok(image)
    }

    pub fn validate(&self) -> Result<(), DirectCcsFPrimeSnarkError> {
        if self.pc != DIRECT_CCS_TRIVIAL_PC
            || self.chunk_count_in.checked_add(1) != Some(self.chunk_count_out)
            || self.step_count_in >= self.step_count_out
            || self.fresh_claims == 0
            || self.incoming_ce_claims == 0
            || self.output_ce_claims != self.incoming_ce_claims + self.fresh_claims
            || self.final_ce_claims != self.incoming_ce_claims
            || self.nifs_chunk_index != self.chunk_count_in
            || self.nifs_fresh_claims != self.fresh_claims
            || self.nifs_incoming_ce_claims != self.incoming_ce_claims
            || self.nifs_pi_ccs_outputs != self.output_ce_claims
            || self.nifs_final_ce_claims != self.final_ce_claims
            || self.nifs_fe_sumcheck_rounds == 0
            || self.nifs_fe_sumcheck_messages == 0
            || self.nifs_nc_sumcheck_rounds == 0
            || self.nifs_nc_sumcheck_messages == 0
            || self.nifs_transcript_absorbed_out < self.nifs_transcript_absorbed_in
        {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        if self.current_boundary_out_digest
            != direct_boundary_update_digest(self.current_boundary_in_digest, self.latest_chunk_digest)
            || self.public_trace_out_digest
                != direct_public_trace_update_digest(self.public_trace_in_digest, self.latest_chunk_digest)
        {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        let expected_x_in = direct_state_x_out(
            self.vk_fs_digest,
            &self.mat_digest,
            self.chunk_count_in,
            self.step_count_in,
            self.initial_boundary_digest,
            self.current_boundary_in_digest,
            self.pc,
            self.semantic_accumulator_in_digest,
            self.f_prime_accumulator_in_digest,
            self.public_trace_in_digest,
        );
        let expected_x_out = direct_state_x_out(
            self.vk_fs_digest,
            &self.mat_digest,
            self.chunk_count_out,
            self.step_count_out,
            self.initial_boundary_digest,
            self.current_boundary_out_digest,
            self.pc,
            self.semantic_accumulator_out_digest,
            self.f_prime_accumulator_out_digest,
            self.public_trace_out_digest,
        );
        if self.x_in != expected_x_in || self.x_out != expected_x_out {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        Ok(())
    }

    pub fn expected_digest(&self) -> Result<[u8; 32], DirectCcsFPrimeSnarkError> {
        self.validate()?;
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/compact_f_prime_image");
        tr.append_message(b"neo.fold.next/direct_ccs/compact_f_prime_image/version", b"v1");
        tr.append_fields(
            b"neo.fold.next/direct_ccs/compact_f_prime_image/mat_digest",
            &self.mat_digest,
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/compact_f_prime_image/vk_fs",
            &self.vk_fs_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/direct_ccs/compact_f_prime_image/meta",
            &[
                self.pc,
                self.chunk_count_in,
                self.step_count_in,
                self.chunk_count_out,
                self.step_count_out,
                self.fresh_claims,
                self.incoming_ce_claims,
                self.output_ce_claims,
                self.final_ce_claims,
                self.nifs_chunk_index,
                self.nifs_fresh_claims,
                self.nifs_incoming_ce_claims,
                self.nifs_pi_ccs_outputs,
                self.nifs_final_ce_claims,
                self.nifs_fe_sumcheck_rounds,
                self.nifs_fe_sumcheck_messages,
                self.nifs_nc_sumcheck_rounds,
                self.nifs_nc_sumcheck_messages,
                self.nifs_transcript_absorbed_in,
                self.nifs_transcript_absorbed_out,
            ],
        );
        for (label, digest) in [
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/initial_boundary"[..],
                self.initial_boundary_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/current_boundary_in"[..],
                self.current_boundary_in_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/current_boundary_out"[..],
                self.current_boundary_out_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/public_trace_in"[..],
                self.public_trace_in_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/public_trace_out"[..],
                self.public_trace_out_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/semantic_accumulator_in"[..],
                self.semantic_accumulator_in_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/semantic_accumulator_out"[..],
                self.semantic_accumulator_out_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/f_prime_accumulator_in"[..],
                self.f_prime_accumulator_in_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/f_prime_accumulator_out"[..],
                self.f_prime_accumulator_out_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/x_in"[..],
                self.x_in.bytes(),
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/x_out"[..],
                self.x_out.bytes(),
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/u_in"[..],
                self.construction2_u_in_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/u_out"[..],
                self.construction2_u_out_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/latest_fold"[..],
                self.latest_fold_digest,
            ),
            (
                &b"neo.fold.next/direct_ccs/compact_f_prime_image/latest_chunk_relation"[..],
                self.latest_chunk_relation_digest,
            ),
        ] {
            tr.append_message(label, &digest);
        }
        tr.append_fields(
            b"neo.fold.next/direct_ccs/compact_f_prime_image/latest_chunk",
            &self.latest_chunk_digest,
        );
        Ok(tr.digest32())
    }

    pub fn terminal_public_image(
        &self,
        construction2_u_i: crate::construction2::Construction2PublicBoundary,
    ) -> DirectCcsIvcPublicImage {
        DirectCcsIvcPublicImage {
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            current_boundary_digest: self.current_boundary_out_digest,
            pc: self.pc,
            chunk_count_out: self.chunk_count_out,
            step_count_out: self.step_count_out,
            x_out: self.x_out.clone(),
            accumulator_out_digest: self.semantic_accumulator_out_digest,
            public_trace_out_digest: self.public_trace_out_digest,
            construction2_accumulator_digest: self.f_prime_accumulator_out_digest,
            construction2_u_i,
        }
    }
}

impl DirectCcsNativeFPrimeAdvice {
    pub fn from_latest_state(state: &DirectCcsIvcState) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let compact_image = DirectCcsCompactFPrimeImage::from_latest_state(state)?;
        let last = state.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input("direct native F' advice requires an appended step".into())
        })?;
        let advice = Self {
            compact_image,
            construction2_u_in: Construction2PublicBoundary::from_fresh_instance(&last.construction2_u_i),
            construction2_u_out: Construction2PublicBoundary::from_fresh_instance(&state.construction2_u_i),
        };
        advice.validate()?;
        Ok(advice)
    }

    pub fn compact_image(&self) -> &DirectCcsCompactFPrimeImage {
        &self.compact_image
    }

    pub fn construction2_u_in(&self) -> &Construction2PublicBoundary {
        &self.construction2_u_in
    }

    pub fn construction2_u_out(&self) -> &Construction2PublicBoundary {
        &self.construction2_u_out
    }

    pub fn low_norm_source_image(&self) -> Result<DirectCcsFPrimeLowNormSourceImage, DirectCcsFPrimeSnarkError> {
        DirectCcsFPrimeLowNormSourceImage::from_native_advice(self)
    }

    pub fn evaluate(&self) -> Result<DirectCcsNativeFPrimeStepImage, DirectCcsFPrimeSnarkError> {
        self.validate()?;
        Ok(DirectCcsNativeFPrimeStepImage {
            compact_image: self.compact_image.clone(),
            terminal_public_image: self
                .compact_image
                .terminal_public_image(self.construction2_u_out.clone()),
            construction2_u_out: self.construction2_u_out.clone(),
        })
    }

    fn validate(&self) -> Result<(), DirectCcsFPrimeSnarkError> {
        self.compact_image.validate()?;
        validate_construction2_boundary_digest(
            &self.construction2_u_in,
            &self.compact_image.x_in,
            self.compact_image.construction2_u_in_digest,
            "input",
        )?;
        validate_construction2_boundary_digest(
            &self.construction2_u_out,
            &self.compact_image.x_out,
            self.compact_image.construction2_u_out_digest,
            "output",
        )
    }
}

impl DirectCcsFPrimeNifsPayloadShape {
    pub fn from_latest_state(state: &DirectCcsIvcState) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let last = state.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input("direct F' NIFS payload shape requires an appended step".into())
        })?;
        Ok(Self {
            chunk_index: last.relation.chunk_index,
            fresh_claims: last.relation.chunk.steps.len(),
            incoming_ce_claims: last.relation.state_in.carry.claims.len(),
            pi_ccs_outputs: last.relation.replay_witness.ccs_outputs.len(),
            final_ce_claims: last.relation.state_out.carry.claims.len(),
            fe_sumcheck_rounds: last
                .relation
                .replay_witness
                .ccs_replay_proof
                .sumcheck_rounds
                .len(),
            fe_sumcheck_messages: last
                .relation
                .replay_witness
                .ccs_replay_proof
                .sumcheck_rounds
                .iter()
                .map(Vec::len)
                .sum(),
            nc_sumcheck_rounds: last
                .relation
                .replay_witness
                .ccs_replay_proof
                .sumcheck_rounds_nc
                .len(),
            nc_sumcheck_messages: last
                .relation
                .replay_witness
                .ccs_replay_proof
                .sumcheck_rounds_nc
                .iter()
                .map(Vec::len)
                .sum(),
            transcript_absorbed_in: last.relation.state_in.transcript.absorbed,
            transcript_absorbed_out: last.relation.state_out.transcript.absorbed,
        })
    }
}

impl DirectCcsFPrimeLowNormSourceImage {
    pub fn from_native_advice(advice: &DirectCcsNativeFPrimeAdvice) -> Result<Self, DirectCcsFPrimeSnarkError> {
        advice.validate()?;
        let mut builder = DirectCcsFPrimeLowNormSourceBuilder::default();
        let image = advice.compact_image();
        builder.mat_digest_bit_offset = builder.values.len();
        for value in image.mat_digest {
            builder.push_field_u64(value);
        }
        builder.vk_fs_digest_bit_offset = builder.push_digest(image.vk_fs_digest);
        builder.pc_bit_offset = builder.push_u64(image.pc);
        builder.chunk_count_in_bit_offset = builder.push_u64(image.chunk_count_in);
        builder.step_count_in_bit_offset = builder.push_u64(image.step_count_in);
        builder.chunk_count_out_bit_offset = builder.push_u64(image.chunk_count_out);
        builder.step_count_out_bit_offset = builder.push_u64(image.step_count_out);
        builder.initial_boundary_digest_bit_offset = builder.push_digest(image.initial_boundary_digest);
        builder.current_boundary_in_digest_bit_offset = builder.push_digest(image.current_boundary_in_digest);
        builder.current_boundary_out_digest_bit_offset = builder.push_digest(image.current_boundary_out_digest);
        builder.public_trace_in_digest_bit_offset = builder.push_digest(image.public_trace_in_digest);
        builder.public_trace_out_digest_bit_offset = builder.push_digest(image.public_trace_out_digest);
        builder.semantic_accumulator_in_digest_bit_offset = builder.push_digest(image.semantic_accumulator_in_digest);
        builder.semantic_accumulator_out_digest_bit_offset = builder.push_digest(image.semantic_accumulator_out_digest);
        builder.f_prime_accumulator_in_digest_bit_offset = builder.push_digest(image.f_prime_accumulator_in_digest);
        builder.f_prime_accumulator_out_digest_bit_offset = builder.push_digest(image.f_prime_accumulator_out_digest);
        builder.compact_x_in_bit_offset = builder.push_encoded_public_input(&image.x_in);
        builder.compact_x_out_bit_offset = builder.push_encoded_public_input(&image.x_out);
        builder.compact_construction2_u_in_digest_bit_offset = builder.push_digest(image.construction2_u_in_digest);
        builder.latest_chunk_digest_bit_offset = builder.values.len();
        for value in image.latest_chunk_digest {
            builder.push_field_u64(value);
        }
        builder.latest_fold_digest_bit_offset = builder.push_digest(image.latest_fold_digest);
        builder.latest_chunk_relation_digest_bit_offset = builder.push_digest(image.latest_chunk_relation_digest);
        builder.fresh_claims_bit_offset = builder.push_u64(image.fresh_claims);
        builder.incoming_ce_claims_bit_offset = builder.push_u64(image.incoming_ce_claims);
        builder.output_ce_claims_bit_offset = builder.push_u64(image.output_ce_claims);
        builder.final_ce_claims_bit_offset = builder.push_u64(image.final_ce_claims);
        builder.nifs_chunk_index_bit_offset = builder.push_u64(image.nifs_chunk_index);
        builder.nifs_fresh_claims_bit_offset = builder.push_u64(image.nifs_fresh_claims);
        builder.nifs_incoming_ce_claims_bit_offset = builder.push_u64(image.nifs_incoming_ce_claims);
        builder.nifs_pi_ccs_outputs_bit_offset = builder.push_u64(image.nifs_pi_ccs_outputs);
        builder.nifs_final_ce_claims_bit_offset = builder.push_u64(image.nifs_final_ce_claims);
        builder.nifs_fe_sumcheck_rounds_bit_offset = builder.push_u64(image.nifs_fe_sumcheck_rounds);
        builder.nifs_fe_sumcheck_messages_bit_offset = builder.push_u64(image.nifs_fe_sumcheck_messages);
        builder.nifs_nc_sumcheck_rounds_bit_offset = builder.push_u64(image.nifs_nc_sumcheck_rounds);
        builder.nifs_nc_sumcheck_messages_bit_offset = builder.push_u64(image.nifs_nc_sumcheck_messages);
        builder.nifs_transcript_absorbed_in_bit_offset = builder.push_u64(image.nifs_transcript_absorbed_in);
        builder.nifs_transcript_absorbed_out_bit_offset = builder.push_u64(image.nifs_transcript_absorbed_out);
        let in_offsets = builder.push_construction2_boundary(advice.construction2_u_in());
        builder.construction2_u_in_fresh_digest_bit_offset = in_offsets.fresh_digest;
        builder.construction2_u_in_commitment_digest_bit_offset = in_offsets.commitment_digest;
        builder.construction2_u_in_commitment_d_bit_offset = in_offsets.commitment_d;
        builder.construction2_u_in_commitment_kappa_bit_offset = in_offsets.commitment_kappa;
        builder.construction2_u_in_x_i_bit_offset = in_offsets.x_i;
        Ok(builder.finish())
    }

    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn compact_x_in_bit_offset(&self) -> usize {
        self.compact_x_in_bit_offset
    }

    pub fn mat_digest_bit_offset(&self) -> usize {
        self.mat_digest_bit_offset
    }

    pub fn vk_fs_digest_bit_offset(&self) -> usize {
        self.vk_fs_digest_bit_offset
    }

    pub fn pc_bit_offset(&self) -> usize {
        self.pc_bit_offset
    }

    pub fn chunk_count_in_bit_offset(&self) -> usize {
        self.chunk_count_in_bit_offset
    }

    pub fn step_count_in_bit_offset(&self) -> usize {
        self.step_count_in_bit_offset
    }

    pub fn chunk_count_out_bit_offset(&self) -> usize {
        self.chunk_count_out_bit_offset
    }

    pub fn step_count_out_bit_offset(&self) -> usize {
        self.step_count_out_bit_offset
    }

    pub fn initial_boundary_digest_bit_offset(&self) -> usize {
        self.initial_boundary_digest_bit_offset
    }

    pub fn current_boundary_in_digest_bit_offset(&self) -> usize {
        self.current_boundary_in_digest_bit_offset
    }

    pub fn current_boundary_out_digest_bit_offset(&self) -> usize {
        self.current_boundary_out_digest_bit_offset
    }

    pub fn public_trace_in_digest_bit_offset(&self) -> usize {
        self.public_trace_in_digest_bit_offset
    }

    pub fn public_trace_out_digest_bit_offset(&self) -> usize {
        self.public_trace_out_digest_bit_offset
    }

    pub fn semantic_accumulator_in_digest_bit_offset(&self) -> usize {
        self.semantic_accumulator_in_digest_bit_offset
    }

    pub fn semantic_accumulator_out_digest_bit_offset(&self) -> usize {
        self.semantic_accumulator_out_digest_bit_offset
    }

    pub fn f_prime_accumulator_in_digest_bit_offset(&self) -> usize {
        self.f_prime_accumulator_in_digest_bit_offset
    }

    pub fn f_prime_accumulator_out_digest_bit_offset(&self) -> usize {
        self.f_prime_accumulator_out_digest_bit_offset
    }

    pub fn compact_x_out_bit_offset(&self) -> usize {
        self.compact_x_out_bit_offset
    }

    pub fn compact_construction2_u_in_digest_bit_offset(&self) -> usize {
        self.compact_construction2_u_in_digest_bit_offset
    }

    pub fn compact_construction2_u_out_digest_bit_offset(&self) -> usize {
        self.compact_construction2_u_out_digest_bit_offset
    }

    pub fn latest_chunk_digest_bit_offset(&self) -> usize {
        self.latest_chunk_digest_bit_offset
    }

    pub fn latest_fold_digest_bit_offset(&self) -> usize {
        self.latest_fold_digest_bit_offset
    }

    pub fn latest_chunk_relation_digest_bit_offset(&self) -> usize {
        self.latest_chunk_relation_digest_bit_offset
    }

    pub fn fresh_claims_bit_offset(&self) -> usize {
        self.fresh_claims_bit_offset
    }

    pub fn incoming_ce_claims_bit_offset(&self) -> usize {
        self.incoming_ce_claims_bit_offset
    }

    pub fn output_ce_claims_bit_offset(&self) -> usize {
        self.output_ce_claims_bit_offset
    }

    pub fn final_ce_claims_bit_offset(&self) -> usize {
        self.final_ce_claims_bit_offset
    }

    pub fn nifs_chunk_index_bit_offset(&self) -> usize {
        self.nifs_chunk_index_bit_offset
    }

    pub fn nifs_fresh_claims_bit_offset(&self) -> usize {
        self.nifs_fresh_claims_bit_offset
    }

    pub fn nifs_incoming_ce_claims_bit_offset(&self) -> usize {
        self.nifs_incoming_ce_claims_bit_offset
    }

    pub fn nifs_pi_ccs_outputs_bit_offset(&self) -> usize {
        self.nifs_pi_ccs_outputs_bit_offset
    }

    pub fn nifs_final_ce_claims_bit_offset(&self) -> usize {
        self.nifs_final_ce_claims_bit_offset
    }

    pub fn nifs_fe_sumcheck_rounds_bit_offset(&self) -> usize {
        self.nifs_fe_sumcheck_rounds_bit_offset
    }

    pub fn nifs_fe_sumcheck_messages_bit_offset(&self) -> usize {
        self.nifs_fe_sumcheck_messages_bit_offset
    }

    pub fn nifs_nc_sumcheck_rounds_bit_offset(&self) -> usize {
        self.nifs_nc_sumcheck_rounds_bit_offset
    }

    pub fn nifs_nc_sumcheck_messages_bit_offset(&self) -> usize {
        self.nifs_nc_sumcheck_messages_bit_offset
    }

    pub fn nifs_transcript_absorbed_in_bit_offset(&self) -> usize {
        self.nifs_transcript_absorbed_in_bit_offset
    }

    pub fn nifs_transcript_absorbed_out_bit_offset(&self) -> usize {
        self.nifs_transcript_absorbed_out_bit_offset
    }

    pub fn construction2_u_in_fresh_digest_bit_offset(&self) -> usize {
        self.construction2_u_in_fresh_digest_bit_offset
    }

    pub fn construction2_u_in_commitment_digest_bit_offset(&self) -> usize {
        self.construction2_u_in_commitment_digest_bit_offset
    }

    pub fn construction2_u_in_commitment_d_bit_offset(&self) -> usize {
        self.construction2_u_in_commitment_d_bit_offset
    }

    pub fn construction2_u_in_commitment_kappa_bit_offset(&self) -> usize {
        self.construction2_u_in_commitment_kappa_bit_offset
    }

    pub fn construction2_u_in_x_i_bit_offset(&self) -> usize {
        self.construction2_u_in_x_i_bit_offset
    }

    pub fn construction2_u_out_fresh_digest_bit_offset(&self) -> usize {
        self.construction2_u_out_fresh_digest_bit_offset
    }

    pub fn construction2_u_out_commitment_digest_bit_offset(&self) -> usize {
        self.construction2_u_out_commitment_digest_bit_offset
    }

    pub fn construction2_u_out_commitment_d_bit_offset(&self) -> usize {
        self.construction2_u_out_commitment_d_bit_offset
    }

    pub fn construction2_u_out_commitment_kappa_bit_offset(&self) -> usize {
        self.construction2_u_out_commitment_kappa_bit_offset
    }

    pub fn construction2_u_out_x_i_bit_offset(&self) -> usize {
        self.construction2_u_out_x_i_bit_offset
    }

    pub fn digest_count(&self) -> usize {
        self.digest_count
    }

    pub fn u64_count(&self) -> usize {
        self.u64_count
    }

    pub fn encoded_public_input_count(&self) -> usize {
        self.encoded_public_input_count
    }

    pub fn construction2_commitment_fields(&self) -> usize {
        self.construction2_commitment_fields
    }

    pub fn field_lane_bit_offsets(&self) -> &[usize] {
        &self.field_lane_bit_offsets
    }

    pub fn field_lane_count(&self) -> usize {
        self.field_lane_count
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/f_prime_low_norm_source");
        tr.append_message(b"neo.fold.next/direct_ccs/f_prime_low_norm_source/version", b"v1");
        tr.append_u64s(
            b"neo.fold.next/direct_ccs/f_prime_low_norm_source/counts",
            &[
                self.values.len() as u64,
                self.mat_digest_bit_offset as u64,
                self.vk_fs_digest_bit_offset as u64,
                self.pc_bit_offset as u64,
                self.chunk_count_in_bit_offset as u64,
                self.step_count_in_bit_offset as u64,
                self.chunk_count_out_bit_offset as u64,
                self.step_count_out_bit_offset as u64,
                self.initial_boundary_digest_bit_offset as u64,
                self.current_boundary_in_digest_bit_offset as u64,
                self.current_boundary_out_digest_bit_offset as u64,
                self.public_trace_in_digest_bit_offset as u64,
                self.public_trace_out_digest_bit_offset as u64,
                self.semantic_accumulator_in_digest_bit_offset as u64,
                self.semantic_accumulator_out_digest_bit_offset as u64,
                self.f_prime_accumulator_in_digest_bit_offset as u64,
                self.f_prime_accumulator_out_digest_bit_offset as u64,
                self.compact_x_in_bit_offset as u64,
                self.compact_x_out_bit_offset as u64,
                self.compact_construction2_u_in_digest_bit_offset as u64,
                self.compact_construction2_u_out_digest_bit_offset as u64,
                self.latest_chunk_digest_bit_offset as u64,
                self.latest_fold_digest_bit_offset as u64,
                self.latest_chunk_relation_digest_bit_offset as u64,
                self.fresh_claims_bit_offset as u64,
                self.incoming_ce_claims_bit_offset as u64,
                self.output_ce_claims_bit_offset as u64,
                self.final_ce_claims_bit_offset as u64,
                self.nifs_chunk_index_bit_offset as u64,
                self.nifs_fresh_claims_bit_offset as u64,
                self.nifs_incoming_ce_claims_bit_offset as u64,
                self.nifs_pi_ccs_outputs_bit_offset as u64,
                self.nifs_final_ce_claims_bit_offset as u64,
                self.nifs_fe_sumcheck_rounds_bit_offset as u64,
                self.nifs_fe_sumcheck_messages_bit_offset as u64,
                self.nifs_nc_sumcheck_rounds_bit_offset as u64,
                self.nifs_nc_sumcheck_messages_bit_offset as u64,
                self.nifs_transcript_absorbed_in_bit_offset as u64,
                self.nifs_transcript_absorbed_out_bit_offset as u64,
                self.construction2_u_in_fresh_digest_bit_offset as u64,
                self.construction2_u_in_commitment_digest_bit_offset as u64,
                self.construction2_u_in_commitment_d_bit_offset as u64,
                self.construction2_u_in_commitment_kappa_bit_offset as u64,
                self.construction2_u_in_x_i_bit_offset as u64,
                self.construction2_u_out_fresh_digest_bit_offset as u64,
                self.construction2_u_out_commitment_digest_bit_offset as u64,
                self.construction2_u_out_commitment_d_bit_offset as u64,
                self.construction2_u_out_commitment_kappa_bit_offset as u64,
                self.construction2_u_out_x_i_bit_offset as u64,
                self.digest_count as u64,
                self.u64_count as u64,
                self.encoded_public_input_count as u64,
                self.field_lane_count as u64,
                self.construction2_commitment_fields as u64,
            ],
        );
        tr.append_u64s(
            b"neo.fold.next/direct_ccs/f_prime_low_norm_source/field_lane_offsets",
            &self
                .field_lane_bit_offsets
                .iter()
                .copied()
                .map(|offset| offset as u64)
                .collect::<Vec<_>>(),
        );
        tr.append_fields(b"neo.fold.next/direct_ccs/f_prime_low_norm_source/values", &self.values);
        tr.digest32()
    }
}

#[derive(Default)]
struct DirectCcsFPrimeLowNormSourceBuilder {
    values: Vec<F>,
    mat_digest_bit_offset: usize,
    vk_fs_digest_bit_offset: usize,
    pc_bit_offset: usize,
    chunk_count_in_bit_offset: usize,
    step_count_in_bit_offset: usize,
    chunk_count_out_bit_offset: usize,
    step_count_out_bit_offset: usize,
    initial_boundary_digest_bit_offset: usize,
    current_boundary_in_digest_bit_offset: usize,
    current_boundary_out_digest_bit_offset: usize,
    public_trace_in_digest_bit_offset: usize,
    public_trace_out_digest_bit_offset: usize,
    semantic_accumulator_in_digest_bit_offset: usize,
    semantic_accumulator_out_digest_bit_offset: usize,
    f_prime_accumulator_in_digest_bit_offset: usize,
    f_prime_accumulator_out_digest_bit_offset: usize,
    compact_x_in_bit_offset: usize,
    compact_x_out_bit_offset: usize,
    compact_construction2_u_in_digest_bit_offset: usize,
    compact_construction2_u_out_digest_bit_offset: usize,
    latest_chunk_digest_bit_offset: usize,
    latest_fold_digest_bit_offset: usize,
    latest_chunk_relation_digest_bit_offset: usize,
    fresh_claims_bit_offset: usize,
    incoming_ce_claims_bit_offset: usize,
    output_ce_claims_bit_offset: usize,
    final_ce_claims_bit_offset: usize,
    nifs_chunk_index_bit_offset: usize,
    nifs_fresh_claims_bit_offset: usize,
    nifs_incoming_ce_claims_bit_offset: usize,
    nifs_pi_ccs_outputs_bit_offset: usize,
    nifs_final_ce_claims_bit_offset: usize,
    nifs_fe_sumcheck_rounds_bit_offset: usize,
    nifs_fe_sumcheck_messages_bit_offset: usize,
    nifs_nc_sumcheck_rounds_bit_offset: usize,
    nifs_nc_sumcheck_messages_bit_offset: usize,
    nifs_transcript_absorbed_in_bit_offset: usize,
    nifs_transcript_absorbed_out_bit_offset: usize,
    construction2_u_in_fresh_digest_bit_offset: usize,
    construction2_u_in_commitment_digest_bit_offset: usize,
    construction2_u_in_commitment_d_bit_offset: usize,
    construction2_u_in_commitment_kappa_bit_offset: usize,
    construction2_u_in_x_i_bit_offset: usize,
    construction2_u_out_fresh_digest_bit_offset: usize,
    construction2_u_out_commitment_digest_bit_offset: usize,
    construction2_u_out_commitment_d_bit_offset: usize,
    construction2_u_out_commitment_kappa_bit_offset: usize,
    construction2_u_out_x_i_bit_offset: usize,
    digest_count: usize,
    u64_count: usize,
    encoded_public_input_count: usize,
    field_lane_bit_offsets: Vec<usize>,
    field_lane_count: usize,
    construction2_commitment_fields: usize,
}

impl DirectCcsFPrimeLowNormSourceBuilder {
    fn push_digest(&mut self, digest: [u8; 32]) -> usize {
        let start = self.values.len();
        self.digest_count += 1;
        self.push_field_lane_offsets(start, 4);
        for byte in digest {
            for bit_index in 0..8 {
                self.push_bit((byte >> bit_index) & 1);
            }
        }
        start
    }

    fn push_encoded_public_input(&mut self, input: &Construction2EncodedPublicInput) -> usize {
        let start = self.values.len();
        self.encoded_public_input_count += 1;
        self.push_field_lane_offsets(start, 4);
        for bit in input.bit_image() {
            self.push_bit(bit);
        }
        start
    }

    fn push_construction2_boundary(&mut self, boundary: &Construction2PublicBoundary) -> Construction2BoundaryOffsets {
        let fresh_offset = self.push_digest(boundary.fresh_instance_digest);
        let commitment_digest_offset = self.push_digest(boundary.commitment_digest);
        let commitment_d_offset = self.values.len();
        self.push_u64(boundary.commitment_d);
        let commitment_kappa_offset = self.values.len();
        self.push_u64(boundary.commitment_kappa);
        let x_i_offset = self.push_encoded_public_input(&boundary.x_i);
        Construction2BoundaryOffsets {
            fresh_digest: fresh_offset,
            commitment_digest: commitment_digest_offset,
            commitment_d: commitment_d_offset,
            commitment_kappa: commitment_kappa_offset,
            x_i: x_i_offset,
        }
    }

    fn push_field_u64(&mut self, value: F) {
        self.field_lane_bit_offsets.push(self.values.len());
        self.field_lane_count += 1;
        self.push_u64(value.as_canonical_u64());
    }

    fn push_u64(&mut self, value: u64) -> usize {
        let start = self.values.len();
        self.u64_count += 1;
        for bit_index in 0..64 {
            self.push_bit(((value >> bit_index) & 1) as u8);
        }
        start
    }

    fn push_bit(&mut self, bit: u8) {
        debug_assert!(bit <= 1);
        self.values.push(F::from_u64(bit as u64));
    }

    fn push_field_lane_offsets(&mut self, start: usize, count: usize) {
        self.field_lane_count += count;
        self.field_lane_bit_offsets
            .extend((0..count).map(|idx| start + idx * 64));
    }

    fn finish(self) -> DirectCcsFPrimeLowNormSourceImage {
        DirectCcsFPrimeLowNormSourceImage {
            values: self.values,
            mat_digest_bit_offset: self.mat_digest_bit_offset,
            vk_fs_digest_bit_offset: self.vk_fs_digest_bit_offset,
            pc_bit_offset: self.pc_bit_offset,
            chunk_count_in_bit_offset: self.chunk_count_in_bit_offset,
            step_count_in_bit_offset: self.step_count_in_bit_offset,
            chunk_count_out_bit_offset: self.chunk_count_out_bit_offset,
            step_count_out_bit_offset: self.step_count_out_bit_offset,
            initial_boundary_digest_bit_offset: self.initial_boundary_digest_bit_offset,
            current_boundary_in_digest_bit_offset: self.current_boundary_in_digest_bit_offset,
            current_boundary_out_digest_bit_offset: self.current_boundary_out_digest_bit_offset,
            public_trace_in_digest_bit_offset: self.public_trace_in_digest_bit_offset,
            public_trace_out_digest_bit_offset: self.public_trace_out_digest_bit_offset,
            semantic_accumulator_in_digest_bit_offset: self.semantic_accumulator_in_digest_bit_offset,
            semantic_accumulator_out_digest_bit_offset: self.semantic_accumulator_out_digest_bit_offset,
            f_prime_accumulator_in_digest_bit_offset: self.f_prime_accumulator_in_digest_bit_offset,
            f_prime_accumulator_out_digest_bit_offset: self.f_prime_accumulator_out_digest_bit_offset,
            compact_x_in_bit_offset: self.compact_x_in_bit_offset,
            compact_x_out_bit_offset: self.compact_x_out_bit_offset,
            compact_construction2_u_in_digest_bit_offset: self.compact_construction2_u_in_digest_bit_offset,
            compact_construction2_u_out_digest_bit_offset: self.compact_construction2_u_out_digest_bit_offset,
            latest_chunk_digest_bit_offset: self.latest_chunk_digest_bit_offset,
            latest_fold_digest_bit_offset: self.latest_fold_digest_bit_offset,
            latest_chunk_relation_digest_bit_offset: self.latest_chunk_relation_digest_bit_offset,
            fresh_claims_bit_offset: self.fresh_claims_bit_offset,
            incoming_ce_claims_bit_offset: self.incoming_ce_claims_bit_offset,
            output_ce_claims_bit_offset: self.output_ce_claims_bit_offset,
            final_ce_claims_bit_offset: self.final_ce_claims_bit_offset,
            nifs_chunk_index_bit_offset: self.nifs_chunk_index_bit_offset,
            nifs_fresh_claims_bit_offset: self.nifs_fresh_claims_bit_offset,
            nifs_incoming_ce_claims_bit_offset: self.nifs_incoming_ce_claims_bit_offset,
            nifs_pi_ccs_outputs_bit_offset: self.nifs_pi_ccs_outputs_bit_offset,
            nifs_final_ce_claims_bit_offset: self.nifs_final_ce_claims_bit_offset,
            nifs_fe_sumcheck_rounds_bit_offset: self.nifs_fe_sumcheck_rounds_bit_offset,
            nifs_fe_sumcheck_messages_bit_offset: self.nifs_fe_sumcheck_messages_bit_offset,
            nifs_nc_sumcheck_rounds_bit_offset: self.nifs_nc_sumcheck_rounds_bit_offset,
            nifs_nc_sumcheck_messages_bit_offset: self.nifs_nc_sumcheck_messages_bit_offset,
            nifs_transcript_absorbed_in_bit_offset: self.nifs_transcript_absorbed_in_bit_offset,
            nifs_transcript_absorbed_out_bit_offset: self.nifs_transcript_absorbed_out_bit_offset,
            construction2_u_in_fresh_digest_bit_offset: self.construction2_u_in_fresh_digest_bit_offset,
            construction2_u_in_commitment_digest_bit_offset: self.construction2_u_in_commitment_digest_bit_offset,
            construction2_u_in_commitment_d_bit_offset: self.construction2_u_in_commitment_d_bit_offset,
            construction2_u_in_commitment_kappa_bit_offset: self.construction2_u_in_commitment_kappa_bit_offset,
            construction2_u_in_x_i_bit_offset: self.construction2_u_in_x_i_bit_offset,
            construction2_u_out_fresh_digest_bit_offset: self.construction2_u_out_fresh_digest_bit_offset,
            construction2_u_out_commitment_digest_bit_offset: self.construction2_u_out_commitment_digest_bit_offset,
            construction2_u_out_commitment_d_bit_offset: self.construction2_u_out_commitment_d_bit_offset,
            construction2_u_out_commitment_kappa_bit_offset: self.construction2_u_out_commitment_kappa_bit_offset,
            construction2_u_out_x_i_bit_offset: self.construction2_u_out_x_i_bit_offset,
            field_lane_bit_offsets: self.field_lane_bit_offsets,
            digest_count: self.digest_count,
            u64_count: self.u64_count,
            encoded_public_input_count: self.encoded_public_input_count,
            field_lane_count: self.field_lane_count,
            construction2_commitment_fields: self.construction2_commitment_fields,
        }
    }
}

struct Construction2BoundaryOffsets {
    fresh_digest: usize,
    commitment_digest: usize,
    commitment_d: usize,
    commitment_kappa: usize,
    x_i: usize,
}

impl DirectCcsNativeFPrimeStepImage {
    pub fn compact_image(&self) -> &DirectCcsCompactFPrimeImage {
        &self.compact_image
    }

    pub fn construction2_u_out(&self) -> &Construction2PublicBoundary {
        &self.construction2_u_out
    }

    pub fn terminal_public_image(&self) -> &DirectCcsIvcPublicImage {
        &self.terminal_public_image
    }
}

fn validate_construction2_boundary_digest(
    boundary: &Construction2PublicBoundary,
    expected_x_i: &Construction2EncodedPublicInput,
    expected_instance_digest: [u8; 32],
    role: &str,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    if !boundary.has_canonical_commitment_shape()
        || boundary.commitment_digest != boundary.expected_commitment_digest()
        || boundary.fresh_instance_digest != boundary.expected_fresh_instance_digest()
        || &boundary.x_i != expected_x_i
    {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct native F' advice {role} Construction-2 boundary is inconsistent"
        )));
    }
    if boundary.fresh_instance_digest != expected_instance_digest {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct native F' advice {role} Construction-2 instance digest is inconsistent"
        )));
    }
    Ok(())
}
