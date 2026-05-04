//! Owns the verifier-facing public image for generic direct CCS/R1CS IVC.
//!
//! This image is the non-VM Construction-2 boundary: it binds the fixed direct
//! CCS relation, the latest folded state counters, the private terminal
//! accumulator digest, and the committed `F'` boundary. It does not expose or
//! hash final CE projections.

use neo_math::F;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};

use crate::construction2::{Construction2EncodedPublicInput, Construction2PublicBoundary};
use crate::finalize::{digest32_as_fields, digest_fields_as_digest32};
use crate::spartan_backend::SpartanF;

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

pub(crate) fn direct_state_x_out(
    vk_fs_digest: [u8; 32],
    mat_digest: &[F; 4],
    chunk_count: u64,
    step_count: u64,
    initial_boundary_digest: [u8; 32],
    current_boundary_digest: [u8; 32],
    pc: u64,
    semantic_accumulator_digest: [u8; 32],
    construction2_accumulator_digest: [u8; 32],
    public_trace_digest: [u8; 32],
) -> Construction2EncodedPublicInput {
    Construction2EncodedPublicInput::from_digest_bytes(direct_state_image_digest(
        vk_fs_digest,
        mat_digest,
        chunk_count,
        step_count,
        initial_boundary_digest,
        current_boundary_digest,
        pc,
        semantic_accumulator_digest,
        construction2_accumulator_digest,
        public_trace_digest,
    ))
}

fn direct_state_image_digest(
    vk_fs_digest: [u8; 32],
    mat_digest: &[F; 4],
    chunk_count: u64,
    step_count: u64,
    initial_boundary_digest: [u8; 32],
    current_boundary_digest: [u8; 32],
    pc: u64,
    semantic_accumulator_digest: [u8; 32],
    construction2_accumulator_digest: [u8; 32],
    public_trace_digest: [u8; 32],
) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/f_prime_x_out/v2");
    preimage.extend(digest32_as_fields(vk_fs_digest));
    preimage.extend(mat_digest.iter().copied());
    preimage.extend(u64_halves_as_native_fields(chunk_count));
    preimage.extend(u64_halves_as_native_fields(step_count));
    preimage.extend(digest32_as_fields(initial_boundary_digest));
    preimage.extend(digest32_as_fields(current_boundary_digest));
    preimage.extend(u64_halves_as_native_fields(pc));
    preimage.extend(digest32_as_fields(semantic_accumulator_digest));
    preimage.extend(digest32_as_fields(construction2_accumulator_digest));
    preimage.extend(digest32_as_fields(public_trace_digest));
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(crate) fn direct_vk_fs_digest(
    params: &NeoParams,
    mat_digest: &[F; 4],
    public_input_len: Option<usize>,
) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/vk_fs/v1");
    preimage.extend(mat_digest.iter().copied());
    preimage.extend([
        F::from_u64(params.q),
        F::from_u64(params.eta as u64),
        F::from_u64(params.d as u64),
        F::from_u64(params.kappa as u64),
        F::from_u64(params.m),
        F::from_u64(params.b as u64),
        F::from_u64(params.k_rho as u64),
        F::from_u64(params.B),
        F::from_u64(params.T as u64),
        F::from_u64(params.s as u64),
        F::from_u64(params.lambda as u64),
        F::from_u64(public_input_len.map_or(u64::MAX, |len| len as u64)),
    ]);
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(crate) fn direct_initial_boundary_digest(mat_digest: &[F; 4], public_input_len: Option<usize>) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/initial_boundary/v1");
    preimage.extend(mat_digest.iter().copied());
    preimage.push(F::from_u64(public_input_len.map_or(u64::MAX, |len| len as u64)));
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(crate) fn direct_boundary_update_digest(boundary_digest: [u8; 32], latest_chunk_digest: [F; 4]) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/current_boundary_update/v1");
    preimage.extend(digest32_as_fields(boundary_digest));
    preimage.extend(latest_chunk_digest);
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(crate) fn direct_public_trace_seed_digest(mat_digest: &[F; 4]) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/public_trace_seed/v1");
    preimage.extend(mat_digest.iter().copied());
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(crate) fn direct_public_trace_update_digest(
    public_trace_digest: [u8; 32],
    latest_chunk_digest: [F; 4],
) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/public_trace_update/v1");
    preimage.extend(digest32_as_fields(public_trace_digest));
    preimage.extend(latest_chunk_digest);
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

fn direct_domain_fields(domain: &[u8]) -> Vec<F> {
    crate::superneo_circuit::claim::packed_bytes_field_values(domain)
        .into_iter()
        .map(|value| F::from_u64(value.to_canonical_u64()))
        .collect()
}

fn digest32_as_spartan_fields(digest: [u8; 32]) -> [SpartanF; 4] {
    digest32_as_fields(digest).map(field_to_spartan)
}

fn u64_halves_as_native_fields(value: u64) -> [F; 2] {
    [F::from_u64(value & 0xffff_ffff), F::from_u64(value >> 32)]
}

fn u64_halves_as_spartan_fields(value: u64) -> [SpartanF; 2] {
    [
        SpartanF::from_canonical_u64(value & 0xffff_ffff),
        SpartanF::from_canonical_u64(value >> 32),
    ]
}

fn field_to_spartan(value: F) -> SpartanF {
    SpartanF::from_canonical_u64(value.as_canonical_u64())
}

fn digest32_has_canonical_field_limb_bytes(digest: [u8; 32]) -> bool {
    digest.chunks_exact(8).all(|chunk| {
        let limb = u64::from_le_bytes(chunk.try_into().expect("digest32 has 8-byte limbs"));
        F::from_u64(limb).as_canonical_u64() == limb
    })
}
