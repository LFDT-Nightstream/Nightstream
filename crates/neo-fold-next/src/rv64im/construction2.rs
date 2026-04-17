//! Owns the explicit HyperNova Construction-2 target surface for native RV64IM F'.
//!
//! This module owns:
//! - the canonical state image used for `x = enc_inst(H(...))`
//! - the canonical native witness-domain shape for the future `enc(F')` CCS instance
//! - the deterministic native witness image for the currently wired Construction-2 inputs
//!
//! It does not own:
//! - native `F'` step evaluation
//! - any claim that the current legacy replay lane already defines `u = (c, x)`

use neo_ajtai::{set_global_pp_seeded, AjtaiSModule, Commitment};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{
    rlc_public_matches_verified_inputs_with_perf, sample_rot_rhos_n_typed, verify_dec_public, RotRing,
};
use neo_reductions::engines::utils::build_dims_and_policy;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};

use crate::finalize::{digest32_as_fields, FixedShapeChunkSummary};
use crate::proof::Carry;
use crate::proof::{ChunkInput, StepInput};
use crate::rv64im::chunk_fold_step::{adapt_rv64im_chunk_to_fresh_ccs, Rv64imAccumulatorHandle, Rv64imChunkFoldCarry};
use crate::rv64im::chunk_relation::{
    rv64im_step_handle, trace_rv64im_chunk_relation_with_replay_rounds, Rv64imChunkRelationTrace,
};
use crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcRelation;
use crate::rv64im::construction2_default::Rv64imMainRecursionConstruction2DefaultPair;
use crate::rv64im::f_prime::{
    evaluate_rv64im_main_recursion_f_prime_advice, Rv64imEncodedPublicInput, Rv64imMainRecursionAccumulatorSlot,
    Rv64imMainRecursionAccumulatorSurface, Rv64imMainRecursionFPrimeAdvice, Rv64imVerifierKeyFs, RV64IM_ENC_INST_BITS,
    RV64IM_ENC_INST_RING_DEGREE, RV64IM_ENC_INST_RING_SLOTS, RV64IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE,
};
use crate::rv64im::final_relation::{
    rv64im_chunk_fold_carried_transcript_snapshot, Rv64imChunkFoldState, Rv64imChunkFoldTranscriptSnapshot,
};
use crate::rv64im::kernel::{
    rv64im_ajtai_mixers, rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache,
    rv64im_public_chunk_digest, Rv64imChunkBridgeHandoff,
};
use crate::rv64im::main_relation_spartan::{
    build_rv64im_main_recursion_f_prime_claim_cover, Rv64imChunkStepIvcShape, Rv64imMainRecursionFPrimeClaimCover,
};
use crate::rv64im::recursion_shape::build_rv64im_recursion_shape;
use crate::rv64im::SimpleKernelError;
use crate::witness_layout::{commit_cols_for_full_width, encode_vector_for_full_width};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionConstruction2Commitment(Commitment);

impl Rv64imMainRecursionConstruction2Commitment {
    pub fn commitment(&self) -> &Commitment {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionConstruction2FreshInstance {
    c_i: Rv64imMainRecursionConstruction2Commitment,
    x_i: Rv64imEncodedPublicInput,
}

impl Rv64imMainRecursionConstruction2FreshInstance {
    pub fn commitment(&self) -> &Rv64imMainRecursionConstruction2Commitment {
        &self.c_i
    }

    pub fn x_i(&self) -> &Rv64imEncodedPublicInput {
        &self.x_i
    }

    pub(crate) fn x_i_mut(&mut self) -> &mut Rv64imEncodedPublicInput {
        &mut self.x_i
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let encoded = bincode::serialize(self).expect("rv64im construction2 fresh instance encodes");
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_construction2_fresh_instance");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_construction2_fresh_instance/version",
            b"v1",
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_construction2_fresh_instance/encoded",
            &encoded,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionConstruction2FPrimeWitnessImage {
    logical_values: Vec<F>,
}

impl Rv64imMainRecursionConstruction2FPrimeWitnessImage {
    pub fn logical_values(&self) -> &[F] {
        &self.logical_values
    }

    pub fn logical_field_count(&self) -> u64 {
        self.logical_values.len() as u64
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_construction2_f_prime_witness");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_construction2_f_prime_witness/version",
            b"v1",
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_construction2_f_prime_witness/len",
            &[self.logical_values.len() as u64],
        );
        tr.append_fields_iter(
            b"neo.fold.next/rv64im/main_recursion_construction2_f_prime_witness/logical_values",
            self.logical_values.len(),
            self.logical_values.iter().copied(),
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage {
    pub(crate) binary_values: Vec<F>,
}

impl Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage {
    pub fn binary_values(&self) -> &[F] {
        &self.binary_values
    }

    pub fn low_norm_field_count(&self) -> u64 {
        self.binary_values.len() as u64
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr =
            Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_construction2_f_prime_low_norm_witness");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_construction2_f_prime_low_norm_witness/version",
            b"v1",
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_construction2_f_prime_low_norm_witness/len",
            &[self.binary_values.len() as u64],
        );
        tr.append_fields_iter(
            b"neo.fold.next/rv64im/main_recursion_construction2_f_prime_low_norm_witness/binary_values",
            self.binary_values.len(),
            self.binary_values.iter().copied(),
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainRecursionConstruction2NifsBridge<'a> {
    input_fresh_instance: &'a Rv64imMainRecursionConstruction2FreshInstance,
    low_norm_witness_image: Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage,
    state_in: &'a Rv64imChunkFoldState,
    chunk_index: u64,
    pi_fold: Rv64imMainRecursionConstruction2PiFoldProof,
    chunk_replay_input: Rv64imMainRecursionConstruction2ReplayInput,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainRecursionConstruction2VerifiedStep {
    pub state: Rv64imChunkFoldState,
    pub canonical_step_statement_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Rv64imMainRecursionConstruction2VerifiedStepStatement {
    pub(crate) chunk_index: u64,
    pub(crate) step_lo: u64,
    pub(crate) step_hi: u64,
    pub(crate) state_in: [u8; 32],
    pub(crate) state_out: [u8; 32],
    pub(crate) public_chunk_digest: [u8; 32],
    pub(crate) chunk_relation_digest: [u8; 32],
}

impl Rv64imMainRecursionConstruction2VerifiedStepStatement {
    pub(crate) fn fixed_shape_chunk_summary(&self) -> Result<FixedShapeChunkSummary, SimpleKernelError> {
        let public_step_count = self.step_hi.checked_sub(self.step_lo).ok_or_else(|| {
            SimpleKernelError::Bridge(
                "RV64IM Construction-2 verified-step summary underflowed the public step span".into(),
            )
        })?;
        Ok(FixedShapeChunkSummary {
            start_index: self.step_lo,
            public_step_count,
            public_chunk_digest: self.public_chunk_digest,
            chunk_relation_digest: self.chunk_relation_digest,
        })
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr =
            Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_construction2_verified_step_statement");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_construction2_verified_step_statement/version",
            b"v1",
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_construction2_verified_step_statement/meta",
            &[self.chunk_index, self.step_lo, self.step_hi],
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_construction2_verified_step_statement/state_in",
            &self.state_in,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_construction2_verified_step_statement/state_out",
            &self.state_out,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_construction2_verified_step_statement/public_chunk_digest",
            &self.public_chunk_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_construction2_verified_step_statement/chunk_relation_digest",
            &self.chunk_relation_digest,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rv64imMainRecursionConstruction2PiCcsOutputPayload {
    pub(crate) y_ring: Vec<Vec<K>>,
    pub(crate) y_zcol: Vec<K>,
}

impl Rv64imMainRecursionConstruction2PiCcsOutputPayload {
    fn from_ce_claim(
        claim: &neo_ccs::CeClaim<Commitment, F, K>,
        expected_fold_digest: [u8; 32],
    ) -> Result<Self, SimpleKernelError> {
        if claim.fold_digest != expected_fold_digest {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Construction-2 Pi_CCS output fold digest drifted from the replay transport header digest"
                    .into(),
            ));
        }
        if !claim.aux_openings.is_empty() {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Construction-2 Pi_CCS output payload cannot carry aux openings before they are owned explicitly"
                    .into(),
            ));
        }
        if !claim.c_step_coords.is_empty() || claim.u_offset != 0 || claim.u_len != 0 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Construction-2 Pi_CCS output payload cannot carry Pattern-A shell fields before they are owned explicitly"
                    .into(),
            ));
        }
        if claim.ct.len() != claim.y_ring.len() {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Construction-2 Pi_CCS output scalar view length does not match y_ring".into(),
            ));
        }
        for (row_idx, (row, ct)) in claim.y_ring.iter().zip(claim.ct.iter()).enumerate() {
            let constant_term = row.first().copied().ok_or_else(|| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM Construction-2 Pi_CCS output y_ring[{row_idx}] is empty"
                ))
            })?;
            if constant_term != *ct {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV64IM Construction-2 Pi_CCS output ct[{row_idx}] does not match y_ring[{row_idx}][0]"
                )));
            }
        }
        Ok(Self {
            y_ring: claim.y_ring.clone(),
            y_zcol: claim.y_zcol.clone(),
        })
    }

    fn validate_matches_ce_claim(
        &self,
        claim: &neo_ccs::CeClaim<Commitment, F, K>,
        expected_fold_digest: [u8; 32],
    ) -> Result<(), SimpleKernelError> {
        let rebuilt = Self::from_ce_claim(claim, expected_fold_digest)?;
        if rebuilt != *self {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Construction-2 Pi_CCS output payload does not match the verified replay output".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rv64imMainRecursionConstruction2PiDecChildPayload {
    pub(crate) c: Commitment,
    pub(crate) y_ring: Vec<Vec<K>>,
}

fn validate_rv64im_main_recursion_construction2_ce_claim_surface(
    claim: &neo_ccs::CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<(), SimpleKernelError> {
    if claim.ct.len() != claim.y_ring.len() {
        return Err(SimpleKernelError::Proof(format!(
            "{label} scalar view length does not match y_ring"
        )));
    }
    for (row_idx, (row, ct)) in claim.y_ring.iter().zip(claim.ct.iter()).enumerate() {
        let constant_term = row
            .first()
            .copied()
            .ok_or_else(|| SimpleKernelError::Proof(format!("{label} y_ring[{row_idx}] is empty")))?;
        if constant_term != *ct {
            return Err(SimpleKernelError::Proof(format!(
                "{label} ct[{row_idx}] does not match y_ring[{row_idx}][0]"
            )));
        }
    }
    Ok(())
}

impl Rv64imMainRecursionConstruction2PiDecChildPayload {
    fn from_ce_claim(claim: &neo_ccs::CeClaim<Commitment, F, K>) -> Result<Self, SimpleKernelError> {
        if claim.ct.len() != claim.y_ring.len() {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Construction-2 Pi_DEC child scalar view length does not match y_ring".into(),
            ));
        }
        for (row_idx, (row, ct)) in claim.y_ring.iter().zip(claim.ct.iter()).enumerate() {
            let constant_term = row.first().copied().ok_or_else(|| {
                SimpleKernelError::Bridge(format!("RV64IM Construction-2 Pi_DEC child y_ring[{row_idx}] is empty"))
            })?;
            if constant_term != *ct {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV64IM Construction-2 Pi_DEC child ct[{row_idx}] does not match y_ring[{row_idx}][0]"
                )));
            }
        }
        Ok(Self {
            c: claim.c.clone(),
            y_ring: claim.y_ring.clone(),
        })
    }

    fn validate_matches_ce_claim(&self, claim: &neo_ccs::CeClaim<Commitment, F, K>) -> Result<(), SimpleKernelError> {
        let rebuilt = Self::from_ce_claim(claim)?;
        if rebuilt != *self {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Construction-2 Pi_DEC child payload does not match the verified replay output".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainRecursionConstruction2PiCcsReplayPayload {
    pub(crate) sumcheck_rounds: Vec<Vec<K>>,
    pub(crate) sumcheck_rounds_nc: Vec<Vec<K>>,
}

impl Rv64imMainRecursionConstruction2PiCcsReplayPayload {
    fn from_chunk_step_relation(relation: &Rv64imChunkStepIvcRelation) -> Self {
        let replay_transport = &relation.witness.replay_witness.ccs_replay_proof;
        Self {
            sumcheck_rounds: replay_transport.sumcheck_rounds.clone(),
            sumcheck_rounds_nc: replay_transport.sumcheck_rounds_nc.clone(),
        }
    }

    fn tamper_first_sumcheck_coeff(&mut self) -> Result<(), SimpleKernelError> {
        let coeff = self
            .sumcheck_rounds
            .first_mut()
            .and_then(|round| round.first_mut())
            .or_else(|| {
                self.sumcheck_rounds_nc
                    .first_mut()
                    .and_then(|round| round.first_mut())
            })
            .ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV64IM Construction-2 Pi_CCS replay payload must carry at least one sumcheck coefficient".into(),
                )
            })?;
        *coeff += K::ONE;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainRecursionConstruction2PiFoldProof {
    pub(crate) ccs_output_payloads: Vec<Rv64imMainRecursionConstruction2PiCcsOutputPayload>,
    pub(crate) ccs_replay_payload: Rv64imMainRecursionConstruction2PiCcsReplayPayload,
    pub(crate) dec_child_payloads: Vec<Rv64imMainRecursionConstruction2PiDecChildPayload>,
}

impl Rv64imMainRecursionConstruction2PiFoldProof {
    pub(crate) fn tamper_ccs_replay_first_round_coeff(&mut self) -> Result<(), SimpleKernelError> {
        self.ccs_replay_payload.tamper_first_sumcheck_coeff()
    }

    pub(crate) fn tamper_dec_child_commitment_first_word(&mut self, index: usize) -> Result<(), SimpleKernelError> {
        let child = self
            .dec_child_payloads
            .get_mut(index)
            .ok_or_else(|| SimpleKernelError::Bridge(format!("invalid Construction-2 DEC child index {index}")))?;
        let word = child.c.data.first_mut().ok_or_else(|| {
            SimpleKernelError::Bridge(format!("Construction-2 DEC child {index} commitment has zero words"))
        })?;
        *word += F::ONE;
        Ok(())
    }
}

pub(crate) fn build_rv64im_main_recursion_construction2_pi_fold_from_trace(
    trace: &Rv64imChunkRelationTrace,
) -> Result<Rv64imMainRecursionConstruction2PiFoldProof, SimpleKernelError> {
    let ccs_output_payloads = trace
        .ccs_outputs
        .iter()
        .map(|claim| {
            Rv64imMainRecursionConstruction2PiCcsOutputPayload::from_ce_claim(claim, trace.terminal_state.fold_digest)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Rv64imMainRecursionConstruction2PiFoldProof {
        ccs_output_payloads,
        ccs_replay_payload: Rv64imMainRecursionConstruction2PiCcsReplayPayload {
            sumcheck_rounds: trace.ccs_replay_proof.sumcheck_rounds.clone(),
            sumcheck_rounds_nc: trace.ccs_replay_proof.sumcheck_rounds_nc.clone(),
        },
        dec_child_payloads: trace
            .children
            .iter()
            .map(Rv64imMainRecursionConstruction2PiDecChildPayload::from_ce_claim)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

pub(crate) fn build_rv64im_main_recursion_construction2_pi_fold_from_relation(
    relation: &Rv64imChunkStepIvcRelation,
) -> Result<Rv64imMainRecursionConstruction2PiFoldProof, SimpleKernelError> {
    let (trace, replay_payload) = trace_and_validate_rv64im_main_recursion_construction2_relation(relation)?;
    let mut pi_fold = build_rv64im_main_recursion_construction2_pi_fold_from_trace(&trace)?;
    pi_fold.ccs_replay_payload = replay_payload;
    Ok(pi_fold)
}

#[derive(Clone, Debug)]
struct Rv64imMainRecursionConstruction2ReplayInput {
    chunk_input: ChunkInput,
    bridge_handoff: Rv64imChunkBridgeHandoff,
}

impl Rv64imMainRecursionConstruction2ReplayInput {
    fn from_verified_kernel_handoff(handoff: &crate::rv64im::kernel::Rv64imVerifiedKernelChunkHandoff) -> Self {
        let mut bridge_handoff = handoff.bridge_handoff.clone();
        for binding in &mut bridge_handoff.step_bindings {
            binding.digest = binding.expected_digest();
        }
        bridge_handoff.digest = bridge_handoff.expected_digest();
        Self {
            chunk_input: handoff.chunk_input.clone(),
            bridge_handoff,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionConstruction2FPrimeCcsShape {
    pub verifier_key_fs_digest: [u8; 32],
    pub recursion_shape_digest: [u8; 32],
    pub x_i_bit_len: u64,
    pub x_i_ring_slot_count: u64,
    pub x_i_ring_degree: u64,
    pub phi_side_commitment_word_lens: Vec<u64>,
    pub step_cover_shape: Rv64imChunkStepIvcShape,
    pub claim_cover: Rv64imMainRecursionFPrimeClaimCover,
}

impl Rv64imMainRecursionConstruction2FPrimeCcsShape {
    pub fn expected_digest(&self) -> [u8; 32] {
        let encoded = bincode::serialize(self).expect("rv64im construction2 native F' shape encodes");
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_construction2_f_prime_ccs_shape");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_construction2_f_prime_ccs_shape/version",
            b"v1",
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_construction2_f_prime_ccs_shape/encoded",
            &encoded,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionConstruction2StateImage {
    vk_fs: Rv64imVerifierKeyFs,
    step_index: u64,
    z_0: [u8; 32],
    z_i: [u8; 32],
    pc_i: u64,
    accumulator_instance_digest: [u8; 32],
}

impl Rv64imMainRecursionConstruction2StateImage {
    pub fn from_parts(
        vk_fs: Rv64imVerifierKeyFs,
        step_index: u64,
        z_0: [u8; 32],
        z_i: [u8; 32],
        pc_i: u64,
        accumulator_instance_digest: [u8; 32],
    ) -> Self {
        Self {
            vk_fs,
            step_index,
            z_0,
            z_i,
            pc_i,
            accumulator_instance_digest,
        }
    }

    pub fn encoded_public_input(&self) -> Rv64imEncodedPublicInput {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_f_prime_x_out");
        tr.append_message(b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/version", b"v4");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/vk_fs",
            &self.vk_fs.expected_digest(),
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/meta",
            &[self.step_index, self.pc_i],
        );
        tr.append_fields_iter(
            b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/z_0",
            4,
            digest32_as_fields(self.z_0),
        );
        tr.append_fields_iter(
            b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/z_i",
            4,
            digest32_as_fields(self.z_i),
        );
        tr.append_fields_iter(
            b"neo.fold.next/rv64im/main_recursion_f_prime_x_out/accumulator_instance_digest",
            4,
            digest32_as_fields(self.accumulator_instance_digest),
        );
        Rv64imEncodedPublicInput::from_digest_bytes(tr.digest32())
    }
}

fn merge_phi_side_commitment_word_cover(cover: &mut Vec<u64>, commitments: &[Vec<u64>]) {
    for (idx, words) in commitments.iter().enumerate() {
        let len = words.len() as u64;
        if let Some(existing) = cover.get_mut(idx) {
            *existing = (*existing).max(len);
        } else {
            cover.push(len);
        }
    }
}

fn append_u64_field(out: &mut Vec<F>, value: u64) {
    out.push(F::from_u64(value));
}

fn append_field_bits_le(out: &mut Vec<F>, value: F) {
    let word = value.as_canonical_u64();
    for bit_index in 0..64 {
        out.push(F::from_u64((word >> bit_index) & 1));
    }
}

fn append_digest_fields(out: &mut Vec<F>, digest: [u8; 32]) {
    out.extend(digest32_as_fields(digest));
}

fn append_commitment_fields(out: &mut Vec<F>, commitment: &Commitment) {
    append_u64_field(out, commitment.d as u64);
    append_u64_field(out, commitment.kappa as u64);
    append_u64_field(out, commitment.data.len() as u64);
    out.extend(commitment.data.iter().copied());
}

fn append_f_slice(out: &mut Vec<F>, values: &[F]) {
    append_u64_field(out, values.len() as u64);
    out.extend(values.iter().copied());
}

fn append_f_matrix(out: &mut Vec<F>, matrix: &neo_ccs::Mat<F>) {
    append_u64_field(out, matrix.rows() as u64);
    append_u64_field(out, matrix.cols() as u64);
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            out.push(matrix[(row, col)]);
        }
    }
}

fn append_k_value(out: &mut Vec<F>, value: &K) {
    out.extend(value.as_coeffs());
}

fn append_k_slice(out: &mut Vec<F>, values: &[K]) {
    append_u64_field(out, values.len() as u64);
    for value in values {
        append_k_value(out, value);
    }
}

fn append_k_rows(out: &mut Vec<F>, rows: &[Vec<K>]) {
    append_u64_field(out, rows.len() as u64);
    for row in rows {
        append_k_slice(out, row);
    }
}

fn append_ccs_claim_fields(out: &mut Vec<F>, claim: &neo_ccs::CcsClaim<Commitment, F>) {
    append_commitment_fields(out, &claim.c);
    append_u64_field(out, claim.m_in as u64);
    append_f_slice(out, &claim.x);
}

fn append_ccs_witness_fields(out: &mut Vec<F>, witness: &neo_ccs::CcsWitness<F>) {
    append_f_slice(out, &witness.w);
    append_f_matrix(out, &witness.Z);
}

fn append_ce_claim_fields(out: &mut Vec<F>, claim: &neo_ccs::CeClaim<Commitment, F, K>) {
    append_commitment_fields(out, &claim.c);
    append_u64_field(out, claim.m_in as u64);
    append_f_matrix(out, &claim.X);
    append_k_slice(out, &claim.r);
    append_k_slice(out, &claim.s_col);
    append_k_rows(out, &claim.y_ring);
    append_k_slice(out, &claim.ct);
    append_k_slice(out, &claim.aux_openings);
    append_k_slice(out, &claim.y_zcol);
}

fn append_pi_ccs_output_payload_fields(out: &mut Vec<F>, payload: &Rv64imMainRecursionConstruction2PiCcsOutputPayload) {
    append_k_rows(out, &payload.y_ring);
    append_k_slice(out, &payload.y_zcol);
}

fn append_pi_dec_child_payload_fields(out: &mut Vec<F>, payload: &Rv64imMainRecursionConstruction2PiDecChildPayload) {
    append_commitment_fields(out, &payload.c);
    append_k_rows(out, &payload.y_ring);
}

fn append_step_input_fields(out: &mut Vec<F>, step: &StepInput) {
    append_ccs_claim_fields(out, &step.mcs);
    append_ccs_witness_fields(out, &step.witness);
}

fn append_chunk_input_fields(out: &mut Vec<F>, chunk_input: &ChunkInput) {
    append_u64_field(out, chunk_input.start_index as u64);
    append_u64_field(out, chunk_input.steps.len() as u64);
    for step in &chunk_input.steps {
        append_step_input_fields(out, step);
    }
}

fn append_construction2_fresh_instance_fields(
    out: &mut Vec<F>,
    fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
) {
    append_commitment_fields(out, fresh_instance.commitment().commitment());
    out.extend(fresh_instance.x_i().field_image());
}

fn append_phi_side_fields(out: &mut Vec<F>, advice: &Rv64imMainRecursionFPrimeAdvice) {
    append_u64_field(out, advice.phi_side().commitment_count());
    for words in advice.phi_side().commitment_words() {
        append_u64_field(out, words.len() as u64);
        for &word in words {
            append_u64_field(out, word);
        }
    }
}

fn append_pi_fold_fields(out: &mut Vec<F>, pi_fold: &Rv64imMainRecursionConstruction2PiFoldProof) {
    append_u64_field(out, pi_fold.ccs_output_payloads.len() as u64);
    for payload in &pi_fold.ccs_output_payloads {
        append_pi_ccs_output_payload_fields(out, payload);
    }
    append_u64_field(out, pi_fold.ccs_replay_payload.sumcheck_rounds.len() as u64);
    for round in &pi_fold.ccs_replay_payload.sumcheck_rounds {
        append_k_slice(out, round);
    }
    append_u64_field(out, pi_fold.ccs_replay_payload.sumcheck_rounds_nc.len() as u64);
    for round in &pi_fold.ccs_replay_payload.sumcheck_rounds_nc {
        append_k_slice(out, round);
    }
    append_u64_field(out, pi_fold.dec_child_payloads.len() as u64);
    for payload in &pi_fold.dec_child_payloads {
        append_pi_dec_child_payload_fields(out, payload);
    }
}

fn validate_rv64im_main_recursion_construction2_advice(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<(), SimpleKernelError> {
    if !RV64IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE && !advice.side_witness().is_zero() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 shape builder cannot admit side-witness cargo before phi_side is wired"
                .into(),
        ));
    }
    if advice.verified_kernel_handoff().chunk_input.steps.len() != 1 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 shape builder requires one public step per recursive relation".into(),
        ));
    }
    let fresh = adapt_rv64im_chunk_to_fresh_ccs(advice.verified_kernel_handoff());
    if fresh.fresh_claims.len() != 1 || fresh.fresh_witnesses.len() != 1 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 shape builder requires exactly one fresh CCS instance per recursive relation"
                .into(),
        ));
    }
    Ok(())
}

fn validate_rv64im_main_recursion_construction2_input_fresh_instance(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
) -> Result<(), SimpleKernelError> {
    if advice.chunk_count_in() == 0 {
        let expected_default = build_rv64im_main_recursion_construction2_default_fresh_instance(
            advice.verifier_key_fs(),
            crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape(
                &build_rv64im_main_recursion_construction2_f_prime_ccs_shape(core::slice::from_ref(advice))?,
            )?,
        )?;
        if current_input_fresh_instance != &expected_default {
            return Err(SimpleKernelError::Bridge(
                "RV64IM native Construction-2 base-case input fresh instance is not the canonical default witness-backed u_perp".into(),
            ));
        }
    } else if current_input_fresh_instance.x_i() != advice.x_i() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 witness image input fresh instance x_i does not match the carried native F' public input"
                .into(),
        ));
    }
    Ok(())
}

fn build_rv64im_main_recursion_construction2_step_shape(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<Rv64imChunkStepIvcShape, SimpleKernelError> {
    let current_input_fresh_instance = advice.construction2_input_fresh_instance().ok_or_else(|| {
        SimpleKernelError::Bridge(
            "RV64IM native Construction-2 shape builder requires the threaded Construction-2 input u_i".into(),
        )
    })?;
    let bridge = build_rv64im_main_recursion_construction2_nifs_bridge(advice, current_input_fresh_instance)?;
    let verified_step = verify_rv64im_main_recursion_construction2_nifs_step(&bridge)?;
    let fresh = adapt_rv64im_chunk_to_fresh_ccs(advice.verified_kernel_handoff());
    Ok(Rv64imChunkStepIvcShape {
        // Recursive-step cover shape intentionally treats terminality as a selector, not a circuit family split.
        terminal_step: false,
        state_in_claim_count: advice.running_state().carry.main.claims.len() as u64,
        state_out_claim_count: verified_step.state.carry.main.claims.len() as u64,
        fresh_claim_count: fresh.fresh_claims.len() as u64,
        fresh_witness_count: fresh.fresh_witnesses.len() as u64,
        ccs_output_count: bridge.pi_fold.ccs_output_payloads.len() as u64,
        child_count: bridge.pi_fold.dec_child_payloads.len() as u64,
        transcript_in_absorbed: advice.running_state().transcript.absorbed as u64,
        transcript_out_absorbed: verified_step.state.transcript.absorbed as u64,
        fe_round_lengths: bridge
            .pi_fold
            .ccs_replay_payload
            .sumcheck_rounds
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
        nc_round_lengths: bridge
            .pi_fold
            .ccs_replay_payload
            .sumcheck_rounds_nc
            .iter()
            .map(|round| round.len() as u64)
            .collect(),
    })
}

fn rv64im_main_recursion_construction2_commitment_seed(full_width: usize) -> Result<[u8; 32], SimpleKernelError> {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_construction2_commitment_seed");
    tr.append_message(
        b"neo.fold.next/rv64im/main_recursion_construction2_commitment_seed/version",
        b"v1",
    );
    tr.append_message(
        b"neo.fold.next/rv64im/main_recursion_construction2_commitment_seed/shape_digest",
        &build_rv64im_recursion_shape()?.canonical_digest(),
    );
    tr.append_u64s(
        b"neo.fold.next/rv64im/main_recursion_construction2_commitment_seed/full_width",
        &[full_width as u64],
    );
    Ok(tr.digest32())
}

fn build_rv64im_main_recursion_construction2_commitment_context(
    full_width: usize,
) -> Result<(NeoParams, AjtaiSModule), SimpleKernelError> {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Construction-2 commitment params failed for full width {full_width}: {err}"
        ))
    })?;
    let m = commit_cols_for_full_width(full_width);
    let seed = rv64im_main_recursion_construction2_commitment_seed(full_width)?;
    set_global_pp_seeded(D, params.kappa as usize, m, seed).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Construction-2 commitment seed setup failed for (d,m)=({D},{m}): {err}"
        ))
    })?;
    let log = AjtaiSModule::from_global_for_dims(D, m).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Construction-2 commitment module failed for (d,m)=({D},{m}): {err}"
        ))
    })?;
    Ok((params, log))
}

pub fn build_rv64im_main_recursion_construction2_f_prime_ccs_shape(
    advices: &[Rv64imMainRecursionFPrimeAdvice],
) -> Result<Rv64imMainRecursionConstruction2FPrimeCcsShape, SimpleKernelError> {
    let first = advices.first().ok_or_else(|| {
        SimpleKernelError::Build("RV64IM native Construction-2 shape builder requires at least one F' advice".into())
    })?;
    let recursion_shape_digest = build_rv64im_recursion_shape()?.canonical_digest();
    let verifier_key_fs_digest = first.verifier_key_fs().expected_digest();
    if first.verifier_key_fs().main_lane_shape_digest != recursion_shape_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 shape builder vk_fs does not match the canonical recursion shape digest"
                .into(),
        ));
    }

    let mut step_cover_shape = Rv64imChunkStepIvcShape::recursive_step_cover_seed();
    let mut phi_side_commitment_word_lens = Vec::new();
    for advice in advices {
        validate_rv64im_main_recursion_construction2_advice(advice)?;
        if advice.verifier_key_fs().expected_digest() != verifier_key_fs_digest {
            return Err(SimpleKernelError::Bridge(
                "RV64IM native Construction-2 shape builder requires one canonical vk_fs across the F' advice chain"
                    .into(),
            ));
        }
        if advice.verifier_key_fs().main_lane_shape_digest != recursion_shape_digest {
            return Err(SimpleKernelError::Bridge(
                "RV64IM native Construction-2 shape builder found an F' advice whose vk_fs shape digest drifted".into(),
            ));
        }
        step_cover_shape =
            step_cover_shape.recursive_step_cover_merge(&build_rv64im_main_recursion_construction2_step_shape(advice)?);
        merge_phi_side_commitment_word_cover(&mut phi_side_commitment_word_lens, advice.phi_side().commitment_words());
    }

    Ok(Rv64imMainRecursionConstruction2FPrimeCcsShape {
        verifier_key_fs_digest,
        recursion_shape_digest,
        x_i_bit_len: RV64IM_ENC_INST_BITS as u64,
        x_i_ring_slot_count: RV64IM_ENC_INST_RING_SLOTS as u64,
        x_i_ring_degree: RV64IM_ENC_INST_RING_DEGREE as u64,
        phi_side_commitment_word_lens,
        step_cover_shape,
        claim_cover: build_rv64im_main_recursion_f_prime_claim_cover(advices)
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?,
    })
}

pub fn build_rv64im_main_recursion_construction2_input_state_image(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Rv64imMainRecursionConstruction2StateImage {
    Rv64imMainRecursionConstruction2StateImage::from_parts(
        advice.verifier_key_fs().clone(),
        advice.chunk_count_in(),
        *advice.z_0(),
        *advice.z_i(),
        advice.pc_i(),
        advice.folded_accumulator_in_digest(),
    )
}

pub fn build_rv64im_main_recursion_construction2_output_state_image(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<Rv64imMainRecursionConstruction2StateImage, SimpleKernelError> {
    let step_image = evaluate_rv64im_main_recursion_f_prime_advice(advice)?;
    Ok(Rv64imMainRecursionConstruction2StateImage::from_parts(
        advice.verifier_key_fs().clone(),
        step_image.chunk_count(),
        *advice.z_0(),
        *step_image.z_next(),
        step_image.pc_next(),
        step_image.folded_accumulator_digest(),
    ))
}

pub fn build_rv64im_main_recursion_construction2_x_i(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<Rv64imEncodedPublicInput, SimpleKernelError> {
    Ok(build_rv64im_main_recursion_construction2_output_state_image(advice)?.encoded_public_input())
}

pub fn build_rv64im_main_recursion_construction2_f_prime_witness_image(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
) -> Result<Rv64imMainRecursionConstruction2FPrimeWitnessImage, SimpleKernelError> {
    validate_rv64im_main_recursion_construction2_advice(advice)?;
    validate_rv64im_main_recursion_construction2_input_fresh_instance(advice, current_input_fresh_instance)?;
    let pi_fold = advice.construction2_pi_fold();

    let mut logical_values = Vec::new();
    append_u64_field(&mut logical_values, advice.chunk_count_in());
    append_digest_fields(&mut logical_values, *advice.z_0());
    append_digest_fields(&mut logical_values, *advice.z_i());
    append_u64_field(&mut logical_values, advice.pc_i());
    append_phi_side_fields(&mut logical_values, advice);
    append_u64_field(
        &mut logical_values,
        advice.running_state().carry.main.claims.len() as u64,
    );
    for claim in &advice.running_state().carry.main.claims {
        append_ce_claim_fields(&mut logical_values, claim);
    }
    append_construction2_fresh_instance_fields(&mut logical_values, current_input_fresh_instance);
    append_chunk_input_fields(&mut logical_values, &advice.verified_kernel_handoff().chunk_input);
    append_pi_fold_fields(&mut logical_values, pi_fold);

    Ok(Rv64imMainRecursionConstruction2FPrimeWitnessImage { logical_values })
}

pub fn build_rv64im_main_recursion_construction2_f_prime_low_norm_witness_image(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
) -> Result<Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage, SimpleKernelError> {
    let logical_image =
        build_rv64im_main_recursion_construction2_f_prime_witness_image(advice, current_input_fresh_instance)?;
    let mut binary_values = Vec::with_capacity(logical_image.logical_values().len() * 64);
    for &value in logical_image.logical_values() {
        append_field_bits_le(&mut binary_values, value);
    }
    Ok(Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage { binary_values })
}

pub fn build_rv64im_main_recursion_construction2_default_low_norm_witness_image(
    vk_fs: &Rv64imVerifierKeyFs,
    full_width: usize,
) -> Result<Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage, SimpleKernelError> {
    Ok(
        build_rv64im_main_recursion_construction2_default_pair(vk_fs, full_width)?
            .w_perp()
            .clone(),
    )
}

pub fn build_rv64im_main_recursion_construction2_default_pair(
    vk_fs: &Rv64imVerifierKeyFs,
    full_width: usize,
) -> Result<Rv64imMainRecursionConstruction2DefaultPair, SimpleKernelError> {
    crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_default_pair_for_full_width(
        vk_fs, full_width,
    )
}

pub(crate) fn build_rv64im_main_recursion_construction2_nifs_bridge<'a>(
    advice: &'a Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &'a Rv64imMainRecursionConstruction2FreshInstance,
) -> Result<Rv64imMainRecursionConstruction2NifsBridge<'a>, SimpleKernelError> {
    validate_rv64im_main_recursion_construction2_advice(advice)?;
    validate_rv64im_main_recursion_construction2_input_fresh_instance(advice, current_input_fresh_instance)?;
    let low_norm_witness_image =
        build_rv64im_main_recursion_construction2_f_prime_low_norm_witness_image(advice, current_input_fresh_instance)?;
    if low_norm_witness_image.binary_values().is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 NIFS bridge cannot carry an empty enc(F') low-norm witness image".into(),
        ));
    }
    Ok(Rv64imMainRecursionConstruction2NifsBridge {
        input_fresh_instance: current_input_fresh_instance,
        low_norm_witness_image,
        state_in: advice.running_state(),
        chunk_index: advice.chunk_index(),
        pi_fold: advice.construction2_pi_fold().clone(),
        chunk_replay_input: Rv64imMainRecursionConstruction2ReplayInput::from_verified_kernel_handoff(
            advice.verified_kernel_handoff(),
        ),
    })
}

pub(crate) fn verify_rv64im_main_recursion_construction2_nifs_step(
    bridge: &Rv64imMainRecursionConstruction2NifsBridge<'_>,
) -> Result<Rv64imMainRecursionConstruction2VerifiedStep, SimpleKernelError> {
    if !bridge.input_fresh_instance.x_i().is_binary_low_norm() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 NIFS bridge carries a non-binary x_i image".into(),
        ));
    }
    if bridge.low_norm_witness_image.binary_values().is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 NIFS bridge cannot verify with an empty enc(F') low-norm witness image"
                .into(),
        ));
    }
    let (trace, mut transcript) = verify_rv64im_main_recursion_construction2_pi_ccs(bridge)?;
    verify_rv64im_main_recursion_construction2_pi_rlc(&trace, &mut transcript)?;
    verify_rv64im_main_recursion_construction2_pi_dec(bridge, &trace)?;
    let state = derive_rv64im_main_recursion_construction2_next_state_from_trace(
        bridge.state_in,
        &bridge.chunk_replay_input,
        &trace,
        &transcript,
    )?;
    Ok(Rv64imMainRecursionConstruction2VerifiedStep {
        canonical_step_statement_digest: build_rv64im_main_recursion_construction2_verified_step_statement(
            bridge, &state, &trace,
        )
        .expected_digest(),
        state,
    })
}

fn verify_rv64im_main_recursion_construction2_pi_ccs(
    bridge: &Rv64imMainRecursionConstruction2NifsBridge<'_>,
) -> Result<(Rv64imChunkRelationTrace, Poseidon2Transcript), SimpleKernelError> {
    validate_rv64im_main_recursion_construction2_chunk_replay_input(bridge.state_in, &bridge.chunk_replay_input)?;
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(
        bridge.state_in.transcript.state,
        bridge.state_in.transcript.absorbed,
    );
    let trace = trace_rv64im_chunk_relation_with_replay_rounds(
        bridge.chunk_index as usize,
        &bridge.chunk_replay_input.chunk_input,
        &bridge.chunk_replay_input.bridge_handoff,
        &bridge.state_in.carry.main,
        &bridge.pi_fold.ccs_replay_payload.sumcheck_rounds,
        &bridge.pi_fold.ccs_replay_payload.sumcheck_rounds_nc,
        &mut transcript,
        params,
        structure,
        log,
        optimized_cache,
    )
    .map_err(|err| SimpleKernelError::Proof(format!("RV64IM Construction-2 Pi_CCS replay failed: {err}")))?;
    if trace.ccs_outputs.len() != bridge.pi_fold.ccs_output_payloads.len() {
        return Err(SimpleKernelError::Proof(format!(
            "RV64IM Construction-2 Pi_CCS output count mismatch: derived {}, carried {}",
            trace.ccs_outputs.len(),
            bridge.pi_fold.ccs_output_payloads.len()
        )));
    }
    for (idx, (claim, payload)) in trace
        .ccs_outputs
        .iter()
        .zip(bridge.pi_fold.ccs_output_payloads.iter())
        .enumerate()
    {
        payload
            .validate_matches_ce_claim(claim, trace.terminal_state.fold_digest)
            .map_err(|err| {
                SimpleKernelError::Proof(format!(
                    "RV64IM Construction-2 Pi_CCS output payload mismatch at output {idx}: {err}"
                ))
            })?;
    }
    Ok((trace, transcript))
}

fn sample_rv64im_main_recursion_construction2_pi_rlc_rhos(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    claim_count: usize,
) -> Result<Vec<neo_reductions::api::RotRho>, SimpleKernelError> {
    let ring = RotRing::goldilocks();
    sample_rot_rhos_n_typed(transcript, params, &ring, claim_count).map_err(|err| {
        SimpleKernelError::Proof(format!("RV64IM Construction-2 Pi_RLC challenge sampling failed: {err}"))
    })
}

pub(crate) fn audit_rv64im_main_recursion_construction2_pi_rlc_rho_mats(
    bridge: &Rv64imMainRecursionConstruction2NifsBridge<'_>,
) -> Result<Vec<Mat<F>>, SimpleKernelError> {
    let (trace, mut transcript) = verify_rv64im_main_recursion_construction2_pi_ccs(bridge)?;
    let (params, _, _) = rv64im_cached_root_main_lane_context()?;
    Ok(
        sample_rv64im_main_recursion_construction2_pi_rlc_rhos(&mut transcript, params, trace.ccs_outputs.len())?
            .into_iter()
            .map(|rho| rho.into_mat())
            .collect(),
    )
}

fn verify_rv64im_main_recursion_construction2_pi_rlc(
    trace: &Rv64imChunkRelationTrace,
    transcript: &mut Poseidon2Transcript,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_main_recursion_construction2_ce_claim_surface(
        &trace.parent,
        "RV64IM Construction-2 Pi_RLC parent claim",
    )?;
    let (params, _, structure) = rv64im_cached_root_main_lane_context()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Proof(format!("RV64IM Construction-2 Pi_RLC dims build failed: {err}")))?;
    let rhos = sample_rv64im_main_recursion_construction2_pi_rlc_rhos(transcript, params, trace.ccs_outputs.len())?;
    let mixers = rv64im_ajtai_mixers();
    let (parent_matches, _) = rlc_public_matches_verified_inputs_with_perf(
        structure,
        params,
        &rhos,
        &trace.ccs_outputs,
        &trace.parent,
        mixers.mix_rhos_commits,
        dims.ell_d,
    )
    .map_err(|err| SimpleKernelError::Proof(format!("RV64IM Construction-2 Pi_RLC public recompute failed: {err}")))?;
    if !parent_matches {
        return Err(SimpleKernelError::Proof(
            "RV64IM Construction-2 Pi_RLC parent claim does not match the independently recomputed RLC fold".into(),
        ));
    }
    Ok(())
}

fn verify_rv64im_main_recursion_construction2_pi_dec(
    bridge: &Rv64imMainRecursionConstruction2NifsBridge<'_>,
    trace: &Rv64imChunkRelationTrace,
) -> Result<(), SimpleKernelError> {
    if trace.children.len() != bridge.pi_fold.dec_child_payloads.len() {
        return Err(SimpleKernelError::Proof(format!(
            "RV64IM Construction-2 DEC child count mismatch: derived {}, carried {}",
            trace.children.len(),
            bridge.pi_fold.dec_child_payloads.len()
        )));
    }
    for (idx, (claim, payload)) in trace
        .children
        .iter()
        .zip(bridge.pi_fold.dec_child_payloads.iter())
        .enumerate()
    {
        validate_rv64im_main_recursion_construction2_ce_claim_surface(
            claim,
            &format!("RV64IM Construction-2 Pi_DEC child {idx}"),
        )?;
        payload.validate_matches_ce_claim(claim).map_err(|err| {
            SimpleKernelError::Proof(format!(
                "RV64IM Construction-2 Pi_DEC child payload mismatch at child {idx}: {err}"
            ))
        })?;
    }
    let (params, _, structure) = rv64im_cached_root_main_lane_context()?;
    let dims = build_dims_and_policy(params, structure)
        .map_err(|err| SimpleKernelError::Proof(format!("RV64IM Construction-2 Pi_DEC dims build failed: {err}")))?;
    let mixers = rv64im_ajtai_mixers();
    if !verify_dec_public(
        structure,
        params,
        &trace.parent,
        &trace.children,
        mixers.combine_b_pows,
        dims.ell_d,
    ) {
        return Err(SimpleKernelError::Proof(
            "RV64IM Construction-2 Pi_DEC child decomposition does not match the independently recomputed public fold identities"
                .into(),
        ));
    }
    Ok(())
}

fn build_rv64im_main_recursion_construction2_verified_step_statement(
    bridge: &Rv64imMainRecursionConstruction2NifsBridge<'_>,
    next_state: &Rv64imChunkFoldState,
    trace: &crate::rv64im::chunk_relation::Rv64imChunkRelationTrace,
) -> Rv64imMainRecursionConstruction2VerifiedStepStatement {
    build_rv64im_main_recursion_construction2_verified_step_statement_from_parts(
        bridge.chunk_replay_input.bridge_handoff.chunk_index,
        &bridge.chunk_replay_input.chunk_input,
        bridge.state_in,
        next_state,
        trace,
    )
}

fn build_rv64im_main_recursion_construction2_verified_step_statement_from_parts(
    chunk_index: u64,
    chunk_input: &ChunkInput,
    state_in: &Rv64imChunkFoldState,
    next_state: &Rv64imChunkFoldState,
    trace: &crate::rv64im::chunk_relation::Rv64imChunkRelationTrace,
) -> Rv64imMainRecursionConstruction2VerifiedStepStatement {
    let public_chunk = chunk_input.public();
    let step_lo = public_chunk.start_index as u64;
    let step_hi = step_lo + public_chunk.steps.len() as u64;
    Rv64imMainRecursionConstruction2VerifiedStepStatement {
        chunk_index,
        step_lo,
        step_hi,
        state_in: state_in.carry.terminal_handle.0,
        state_out: next_state.carry.terminal_handle.0,
        public_chunk_digest: rv64im_public_chunk_digest(&public_chunk),
        chunk_relation_digest: trace.chunk_relation_digest,
    }
}

fn trace_and_validate_rv64im_main_recursion_construction2_relation(
    relation: &Rv64imChunkStepIvcRelation,
) -> Result<
    (
        Rv64imChunkRelationTrace,
        Rv64imMainRecursionConstruction2PiCcsReplayPayload,
    ),
    SimpleKernelError,
> {
    let replay_input =
        Rv64imMainRecursionConstruction2ReplayInput::from_verified_kernel_handoff(&relation.witness.handoff);
    validate_rv64im_main_recursion_construction2_chunk_replay_input(&relation.witness.state_in, &replay_input)?;
    let (params, log, structure) = rv64im_cached_root_main_lane_context()?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(
        relation.witness.state_in.transcript.state,
        relation.witness.state_in.transcript.absorbed,
    );
    let replay_payload = Rv64imMainRecursionConstruction2PiCcsReplayPayload::from_chunk_step_relation(relation);
    let trace = trace_rv64im_chunk_relation_with_replay_rounds(
        relation.witness.handoff.bridge_handoff.chunk_index as usize,
        &replay_input.chunk_input,
        &replay_input.bridge_handoff,
        &relation.witness.state_in.carry.main,
        &replay_payload.sumcheck_rounds,
        &replay_payload.sumcheck_rounds_nc,
        &mut transcript,
        params,
        structure,
        log,
        optimized_cache,
    )?;
    let expected_next_state = derive_rv64im_main_recursion_construction2_next_state_from_trace(
        &relation.witness.state_in,
        &replay_input,
        &trace,
        &transcript,
    )?;
    if expected_next_state.carry.main.claims != relation.witness.state_out.carry.main.claims
        || expected_next_state.carry.main.witnesses != relation.witness.state_out.carry.main.witnesses
        || expected_next_state.carry.terminal_handle != relation.witness.state_out.carry.terminal_handle
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 relation replay carry_out does not match the verified chunk relation trace".into(),
        ));
    }
    if expected_next_state.transcript != relation.witness.state_out.transcript {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 relation replay transcript_out does not match the verified chunk relation trace"
                .into(),
        ));
    }
    Ok((trace, replay_payload))
}

pub(crate) fn build_rv64im_main_recursion_construction2_verified_step_statement_from_relation(
    relation: &Rv64imChunkStepIvcRelation,
) -> Result<Rv64imMainRecursionConstruction2VerifiedStepStatement, SimpleKernelError> {
    let (trace, _) = trace_and_validate_rv64im_main_recursion_construction2_relation(relation)?;
    Ok(
        build_rv64im_main_recursion_construction2_verified_step_statement_from_parts(
            relation.witness.handoff.bridge_handoff.chunk_index,
            &relation.witness.handoff.chunk_input,
            &relation.witness.state_in,
            &relation.witness.state_out,
            &trace,
        ),
    )
}

pub(crate) fn build_rv64im_main_recursion_construction2_canonical_step_statement_digest_from_relation(
    relation: &Rv64imChunkStepIvcRelation,
) -> Result<[u8; 32], SimpleKernelError> {
    Ok(build_rv64im_main_recursion_construction2_verified_step_statement_from_relation(relation)?.expected_digest())
}

fn validate_rv64im_main_recursion_construction2_chunk_replay_input(
    state_in: &Rv64imChunkFoldState,
    replay_input: &Rv64imMainRecursionConstruction2ReplayInput,
) -> Result<(), SimpleKernelError> {
    if replay_input.bridge_handoff.chunk_start_index != replay_input.chunk_input.start_index as u64 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 bridge replay input chunk metadata drifted from the carried chunk input".into(),
        ));
    }
    if replay_input.bridge_handoff.public_step_count != replay_input.chunk_input.steps.len() as u64 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 bridge replay input step count drifted from the carried chunk input".into(),
        ));
    }
    if state_in.transcript.absorbed > neo_params::poseidon2_goldilocks::RATE {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 bridge transcript snapshot absorbed count exceeds the Poseidon2 rate".into(),
        ));
    }
    Ok(())
}

fn apply_rv64im_main_recursion_construction2_verified_slot_to_carry(
    carry_in: &Carry,
    pc_i: u64,
    verified_claim: neo_ccs::CeClaim<Commitment, F, K>,
    verified_witness: Mat<F>,
) -> Result<Carry, SimpleKernelError> {
    let mut accumulator =
        Rv64imMainRecursionAccumulatorSurface::try_from_carry(carry_in, "Construction-2 native next-state input")?;
    accumulator.update_pc_slot(
        pc_i,
        Rv64imMainRecursionAccumulatorSlot::from_parts(verified_claim, verified_witness),
    )?;
    Ok(accumulator.into_carry())
}

fn derive_rv64im_main_recursion_construction2_next_state_from_trace(
    state_in: &Rv64imChunkFoldState,
    replay_input: &Rv64imMainRecursionConstruction2ReplayInput,
    trace: &crate::rv64im::chunk_relation::Rv64imChunkRelationTrace,
    transcript: &Poseidon2Transcript,
) -> Result<Rv64imChunkFoldState, SimpleKernelError> {
    if trace.children.len() != 1 || trace.z_split.len() != 1 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 native next-state derivation requires exactly one verified CE slot in the current single-PC specialization"
                .into(),
        ));
    }
    let next_carry = Rv64imChunkFoldCarry {
        main: apply_rv64im_main_recursion_construction2_verified_slot_to_carry(
            &state_in.carry.main,
            crate::rv64im::f_prime::RV64IM_MAIN_RECURSION_TRIVIAL_PC,
            trace.children[0].clone(),
            trace.z_split[0].clone(),
        )?,
        terminal_handle: Rv64imAccumulatorHandle(rv64im_step_handle(
            state_in.carry.terminal_handle.0,
            replay_input.bridge_handoff.chunk_index as usize,
            replay_input.chunk_input.start_index,
            replay_input.chunk_input.steps.len(),
            trace.chunk_relation_digest,
        )),
    };
    let transcript_out = rv64im_chunk_fold_carried_transcript_snapshot(&Rv64imChunkFoldTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    });
    Ok(Rv64imChunkFoldState {
        carry: next_carry,
        transcript: transcript_out,
    })
}

fn build_rv64im_main_recursion_construction2_full_z_image_with_x_i(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
    x_i: Rv64imEncodedPublicInput,
) -> Result<(Rv64imEncodedPublicInput, Vec<F>), SimpleKernelError> {
    let low_norm_witness =
        build_rv64im_main_recursion_construction2_f_prime_low_norm_witness_image(advice, current_input_fresh_instance)?;
    let mut full_vector = Vec::with_capacity(RV64IM_ENC_INST_BITS + low_norm_witness.binary_values().len());
    full_vector.extend(x_i.field_image());
    full_vector.extend_from_slice(low_norm_witness.binary_values());
    Ok((x_i, full_vector))
}

pub(crate) fn build_rv64im_main_recursion_construction2_fresh_instance_from_full_vector(
    chunk_count_out: u64,
    x_i: Rv64imEncodedPublicInput,
    full_vector: &[F],
) -> Result<Rv64imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    let (params, log) = build_rv64im_main_recursion_construction2_commitment_context(full_vector.len())?;
    let packed = encode_vector_for_full_width(&params, full_vector.len(), full_vector).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM native Construction-2 fresh instance encoding failed for chunk {chunk_count_out}: {err}"
        ))
    })?;
    Ok(Rv64imMainRecursionConstruction2FreshInstance {
        c_i: Rv64imMainRecursionConstruction2Commitment(log.commit(&packed)),
        x_i,
    })
}

pub fn build_rv64im_main_recursion_construction2_default_fresh_instance(
    vk_fs: &Rv64imVerifierKeyFs,
    full_width: usize,
) -> Result<Rv64imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    Ok(
        build_rv64im_main_recursion_construction2_default_pair(vk_fs, full_width)?
            .u_perp()
            .clone(),
    )
}

pub(crate) fn build_rv64im_main_recursion_construction2_fresh_instance_with_input_and_x_i(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
    x_i: Rv64imEncodedPublicInput,
) -> Result<Rv64imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    validate_rv64im_main_recursion_construction2_advice(advice)?;
    let (x_i, full_vector) =
        build_rv64im_main_recursion_construction2_full_z_image_with_x_i(advice, current_input_fresh_instance, x_i)?;
    build_rv64im_main_recursion_construction2_fresh_instance_from_full_vector(
        advice.chunk_count_in() + 1,
        x_i,
        &full_vector,
    )
}

pub fn build_rv64im_main_recursion_construction2_fresh_instance_with_input(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
) -> Result<Rv64imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    build_rv64im_main_recursion_construction2_fresh_instance_with_input_and_x_i(
        advice,
        current_input_fresh_instance,
        build_rv64im_main_recursion_construction2_x_i(advice)?,
    )
}

pub fn build_rv64im_main_recursion_construction2_fresh_instance(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<Rv64imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    let shape = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(core::slice::from_ref(advice))?;
    if advice.chunk_count_in() > 0 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 fresh instance builder for an inductive F' step still requires the prior-step output u_i = (c_i, x_i) to be threaded explicitly; use the explicit input-threaded builder"
                .into(),
        ));
    }
    build_rv64im_main_recursion_construction2_default_fresh_instance(
        advice.verifier_key_fs(),
        crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape(
            &build_rv64im_main_recursion_construction2_f_prime_ccs_shape(core::slice::from_ref(advice))?,
        )?,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM native Construction-2 base-case fresh instance build failed after wiring the binary low-norm enc(F') image (shape digest {:?}): {err}",
            shape.expected_digest(),
        ))
    })
}
