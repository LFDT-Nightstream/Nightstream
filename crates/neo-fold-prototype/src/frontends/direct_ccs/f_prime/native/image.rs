//! Compact F' image for the latest direct-CCS transition.
//!
//! The image binds counters, accumulator handles, boundary digests, and NIFS
//! payload sizes, but deliberately omits terminal final-CE material.

use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use super::super::super::public_image::{
    direct_boundary_update_digest, direct_public_trace_update_digest, direct_state_x_out, DirectCcsIvcPublicImage,
    DIRECT_CCS_TRIVIAL_PC,
};
use super::super::super::state::{DirectCcsFPrimeSnarkError, DirectCcsIvcState};
use super::nifs::DirectCcsFPrimeNifsPayloadShape;
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
