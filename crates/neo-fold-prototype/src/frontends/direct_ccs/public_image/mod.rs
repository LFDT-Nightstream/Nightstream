//! Verifier-facing public image for generic direct CCS/R1CS IVC.
//!
//! This image is the non-VM Construction-2 boundary: it binds the fixed direct
//! CCS relation, the folded state counters, the private terminal accumulator
//! digest, and the committed F' boundary. It does not expose or hash final CE
//! projections.

mod digest;

use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::construction2::{Construction2EncodedPublicInput, Construction2PublicBoundary};
use crate::spartan_backend::SpartanF;
use digest::{
    digest32_as_spartan_fields, digest32_has_canonical_field_limb_bytes, field_to_spartan, u64_halves_as_spartan_fields,
};
pub(crate) use digest::{
    direct_boundary_update_digest, direct_initial_boundary_digest, direct_public_trace_seed_digest,
    direct_public_trace_update_digest, direct_state_x_out, direct_vk_fs_digest,
};

pub const DIRECT_CCS_TRIVIAL_PC: u64 = 1;
const DIRECT_CCS_TERMINAL_PUBLIC_VALUES_LEN: usize = 292;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsStatement {
    pub mat_digest: [F; 4],
    pub vk_fs_digest: [u8; 32],
    pub initial_boundary_digest: [u8; 32],
    pub current_boundary_digest: [u8; 32],
    pub pc: u64,
    pub chunk_count_out: u64,
    pub step_count_out: u64,
    pub x_out: Construction2EncodedPublicInput,
    pub accumulator_out_digest: [u8; 32],
    pub public_trace_out_digest: [u8; 32],
    pub construction2_accumulator_digest: [u8; 32],
    pub construction2_u_i: Construction2PublicBoundary,
}

impl DirectCcsStatement {
    pub(crate) fn terminal_public_values(&self) -> Vec<SpartanF> {
        let mut values = Vec::with_capacity(DIRECT_CCS_TERMINAL_PUBLIC_VALUES_LEN);
        values.extend(self.mat_digest.iter().copied().map(field_to_spartan));
        values.extend(u64_halves_as_spartan_fields(self.chunk_count_out));
        values.extend(u64_halves_as_spartan_fields(self.step_count_out));
        values.extend(digest32_as_spartan_fields(self.vk_fs_digest));
        values.extend(digest32_as_spartan_fields(self.initial_boundary_digest));
        values.extend(digest32_as_spartan_fields(self.current_boundary_digest));
        values.extend(digest32_as_spartan_fields(self.x_out.bytes()));
        values.extend(self.x_out.field_image().into_iter().map(field_to_spartan));
        values.extend(digest32_as_spartan_fields(self.accumulator_out_digest));
        values.extend(digest32_as_spartan_fields(self.public_trace_out_digest));
        values.extend(digest32_as_spartan_fields(self.construction2_accumulator_digest));
        values
    }

    pub fn validate_final_construction2_public_boundary(&self) -> Result<(), String> {
        if self.chunk_count_out == 0 || self.step_count_out == 0 {
            return Err("direct CCS public image must close at least one folded step".into());
        }
        if self.pc != DIRECT_CCS_TRIVIAL_PC {
            return Err("direct CCS public image pc does not match the fixed direct relation".into());
        }
        if self.construction2_u_i.x_i != self.x_out {
            return Err("direct CCS public image Construction-2 u_i.x_i does not match x_out".into());
        }
        if self.x_out
            != direct_state_x_out(
                self.vk_fs_digest,
                &self.mat_digest,
                self.chunk_count_out,
                self.step_count_out,
                self.initial_boundary_digest,
                self.current_boundary_digest,
                self.pc,
                self.accumulator_out_digest,
                self.construction2_accumulator_digest,
                self.public_trace_out_digest,
            )
        {
            return Err("direct CCS public image x_out does not bind output counter/digest state".into());
        }
        for (label, digest) in [
            ("x_out", self.x_out.bytes()),
            ("vk_fs_digest", self.vk_fs_digest),
            ("initial_boundary_digest", self.initial_boundary_digest),
            ("current_boundary_digest", self.current_boundary_digest),
            ("accumulator_out_digest", self.accumulator_out_digest),
            ("public_trace_out_digest", self.public_trace_out_digest),
            (
                "construction2_accumulator_digest",
                self.construction2_accumulator_digest,
            ),
            (
                "construction2_u_i.fresh_instance_digest",
                self.construction2_u_i.fresh_instance_digest,
            ),
            (
                "construction2_u_i.commitment_digest",
                self.construction2_u_i.commitment_digest,
            ),
        ] {
            if !digest32_has_canonical_field_limb_bytes(digest) {
                return Err(format!(
                    "direct CCS public image {label} is not a canonical four-limb field encoding"
                ));
            }
        }
        if !self.construction2_u_i.has_canonical_commitment_shape() {
            return Err("direct CCS public image Construction-2 commitment shape is not canonical".into());
        }
        if self.construction2_u_i.commitment_digest != self.construction2_u_i.expected_commitment_digest() {
            return Err(
                "direct CCS public image Construction-2 commitment digest does not bind commitment data".into(),
            );
        }
        if self.construction2_u_i.fresh_instance_digest != self.construction2_u_i.expected_fresh_instance_digest() {
            return Err(
                "direct CCS public image Construction-2 fresh-instance digest does not bind commitment and x_i".into(),
            );
        }
        Ok(())
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/statement");
        tr.append_message(b"neo.fold.next/direct_ccs/statement/version", b"v1");
        tr.append_fields(b"neo.fold.next/direct_ccs/statement/mat_digest", &self.mat_digest);
        tr.append_message(b"neo.fold.next/direct_ccs/statement/vk_fs", &self.vk_fs_digest);
        tr.append_u64s(
            b"neo.fold.next/direct_ccs/statement/meta",
            &[self.pc, self.chunk_count_out, self.step_count_out],
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/statement/initial_boundary",
            &self.initial_boundary_digest,
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/statement/current_boundary",
            &self.current_boundary_digest,
        );
        tr.append_message(b"neo.fold.next/direct_ccs/statement/x_out", &self.x_out.bytes());
        tr.append_message(
            b"neo.fold.next/direct_ccs/statement/accumulator_out",
            &self.accumulator_out_digest,
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/statement/public_trace_out",
            &self.public_trace_out_digest,
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/statement/construction2_accumulator",
            &self.construction2_accumulator_digest,
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/statement/construction2_u_i",
            &self.construction2_u_i.expected_digest(),
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsIvcPublicImage {
    pub mat_digest: [F; 4],
    pub vk_fs_digest: [u8; 32],
    pub initial_boundary_digest: [u8; 32],
    pub current_boundary_digest: [u8; 32],
    pub pc: u64,
    pub chunk_count_out: u64,
    pub step_count_out: u64,
    pub x_out: Construction2EncodedPublicInput,
    pub accumulator_out_digest: [u8; 32],
    pub public_trace_out_digest: [u8; 32],
    pub construction2_accumulator_digest: [u8; 32],
    pub construction2_u_i: Construction2PublicBoundary,
}

impl DirectCcsIvcPublicImage {
    pub fn statement(&self) -> DirectCcsStatement {
        DirectCcsStatement {
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            current_boundary_digest: self.current_boundary_digest,
            pc: self.pc,
            chunk_count_out: self.chunk_count_out,
            step_count_out: self.step_count_out,
            x_out: self.x_out.clone(),
            accumulator_out_digest: self.accumulator_out_digest,
            public_trace_out_digest: self.public_trace_out_digest,
            construction2_accumulator_digest: self.construction2_accumulator_digest,
            construction2_u_i: self.construction2_u_i.clone(),
        }
    }

    pub fn validate_final_construction2_public_boundary(&self) -> Result<(), String> {
        self.statement()
            .validate_final_construction2_public_boundary()
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        self.statement().expected_digest()
    }
}
