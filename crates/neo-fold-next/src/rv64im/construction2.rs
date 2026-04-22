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

use neo_ajtai::{
    audit_commit_row_major_seeded_binary_cols_with_chunk_seeds, commit_row_major_seeded_binary_cols_with_chunk_seeds,
    get_global_pp_seeded_params_for_dims, has_global_pp_for_dims, seeded_pp_chunk_seeds, set_global_pp_seeded,
    AjtaiSModule, Commitment, SeededBinaryColsCommitAudit,
};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{rlc_public, sample_rot_rhos_n_typed, verify_dec_public, RotRing};
use neo_reductions::optimized_engine::optimized_verify_with_cache_and_instance_digest_and_perf;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::chunk_relation::ChunkReplayWitness;
use crate::finalize::{digest32_as_fields, public_chunk_digest, FixedShapeChunkSummary};
use crate::proof::Carry;
use crate::proof::{ChunkInput, StepInput};
use crate::rv64im::chunk_fold_step::{adapt_rv64im_chunk_to_fresh_ccs, Rv64imAccumulatorHandle, Rv64imChunkFoldCarry};
use crate::rv64im::chunk_relation::{
    rv64im_chunk_relation_digest_from_fold_digest, rv64im_step_handle, trace_rv64im_chunk_relation_with_replay_rounds,
    Rv64imChunkRelationTrace,
};
use crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcRelation;
use crate::rv64im::construction2_default::Rv64imMainRecursionConstruction2DefaultPair;
use crate::rv64im::f_prime::{
    evaluate_rv64im_main_recursion_f_prime_advice, Rv64imEncodedPublicInput,
    Rv64imMainRecursionConstruction2NifsVerifyPerf, Rv64imMainRecursionFPrimeAdvice, Rv64imVerifierKeyFs,
    RV64IM_ENC_INST_BITS, RV64IM_ENC_INST_RING_DEGREE, RV64IM_ENC_INST_RING_SLOTS,
    RV64IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE,
};
use crate::rv64im::final_relation::{
    rv64im_chunk_fold_carried_transcript_snapshot, Rv64imChunkFoldState, Rv64imChunkFoldTranscriptSnapshot,
};
use crate::rv64im::kernel::{
    rv64im_ajtai_mixers, rv64im_cached_root_main_lane_optimized_cache, rv64im_public_chunk_digest,
    rv64im_root_main_lane_context_for_claim_count, Rv64imChunkBridgeHandoff,
};
use crate::rv64im::main_relation_spartan::{
    build_rv64im_main_recursion_f_prime_claim_cover, Rv64imChunkStepIvcShape, Rv64imMainRecursionFPrimeClaimCover,
};
use crate::rv64im::SimpleKernelError;
use crate::witness_layout::commit_cols_for_full_width;

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn emit_debug_timing(trace_prefix: Option<&str>, label: &str, elapsed_ms: f64) {
    if let Some(prefix) = trace_prefix {
        eprintln!("{prefix}.{label}={elapsed_ms:.2}ms");
        let _ = io::stderr().flush();
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Rv64imMainRecursionConstruction2FreshInstanceBuildPerf {
    pub canonical_full_width_ms: f64,
    pub commitment_context_ms: f64,
    pub pack_image_ms: f64,
    pub commit_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionConstruction2Commitment(Commitment);

impl Rv64imMainRecursionConstruction2Commitment {
    pub fn commitment(&self) -> &Commitment {
        &self.0
    }

    pub(crate) fn from_commitment(commitment: Commitment) -> Self {
        Self(commitment)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionConstruction2FreshInstance {
    c_i: Rv64imMainRecursionConstruction2Commitment,
    x_i: Rv64imEncodedPublicInput,
}

impl Rv64imMainRecursionConstruction2FreshInstance {
    pub(crate) fn from_parts(c_i: Rv64imMainRecursionConstruction2Commitment, x_i: Rv64imEncodedPublicInput) -> Self {
        Self { c_i, x_i }
    }

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
    state_in: &'a Rv64imChunkFoldState,
    expected_state_out: &'a Rv64imChunkFoldState,
    chunk_index: u64,
    pi_fold: &'a Rv64imMainRecursionConstruction2PiFoldProof,
    replay_witness: &'a ChunkReplayWitness,
    chunk_replay_input: Rv64imMainRecursionConstruction2ReplayInput,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imMainRecursionConstruction2VerifiedStep {
    pub state: Rv64imChunkFoldState,
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
    pub(crate) ccs_replay_payload: Rv64imMainRecursionConstruction2PiCcsReplayPayload,
}

impl Rv64imMainRecursionConstruction2PiFoldProof {
    pub(crate) fn tamper_ccs_replay_first_round_coeff(&mut self) -> Result<(), SimpleKernelError> {
        self.ccs_replay_payload.tamper_first_sumcheck_coeff()
    }
}

pub(crate) fn build_rv64im_main_recursion_construction2_pi_fold_from_trace(
    trace: &Rv64imChunkRelationTrace,
) -> Result<Rv64imMainRecursionConstruction2PiFoldProof, SimpleKernelError> {
    Ok(Rv64imMainRecursionConstruction2PiFoldProof {
        ccs_replay_payload: Rv64imMainRecursionConstruction2PiCcsReplayPayload {
            sumcheck_rounds: trace.ccs_replay_proof.sumcheck_rounds.clone(),
            sumcheck_rounds_nc: trace.ccs_replay_proof.sumcheck_rounds_nc.clone(),
        },
    })
}

pub(crate) fn build_rv64im_main_recursion_construction2_pi_fold_from_replay_witness(
    replay_witness: &ChunkReplayWitness,
) -> Rv64imMainRecursionConstruction2PiFoldProof {
    Rv64imMainRecursionConstruction2PiFoldProof {
        ccs_replay_payload: Rv64imMainRecursionConstruction2PiCcsReplayPayload {
            sumcheck_rounds: replay_witness.ccs_replay_proof.sumcheck_rounds.clone(),
            sumcheck_rounds_nc: replay_witness.ccs_replay_proof.sumcheck_rounds_nc.clone(),
        },
    }
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

trait Rv64imConstruction2BitSink {
    fn push_bit(&mut self, value: F);

    fn skip_zero_bits(&mut self, count: usize) {
        for _ in 0..count {
            self.push_bit(F::ZERO);
        }
    }

    fn push_u64_bits_le(&mut self, word: u64) {
        if word == 0 {
            self.skip_zero_bits(64);
            return;
        }
        for bit_index in 0..64 {
            self.push_bit(F::from_u64((word >> bit_index) & 1));
        }
    }
}

impl Rv64imConstruction2BitSink for Vec<F> {
    fn push_bit(&mut self, value: F) {
        self.push(value);
    }

    fn skip_zero_bits(&mut self, count: usize) {
        self.resize(self.len() + count, F::ZERO);
    }

    fn push_u64_bits_le(&mut self, word: u64) {
        if word == 0 {
            self.skip_zero_bits(64);
            return;
        }
        for bit_index in 0..64 {
            self.push(F::from_u64((word >> bit_index) & 1));
        }
    }
}

struct PackedBinaryMatBitSink {
    full_width: usize,
    written: usize,
    column_bits: Vec<u64>,
}

impl PackedBinaryMatBitSink {
    fn new(full_width: usize) -> Result<Self, SimpleKernelError> {
        if full_width == 0 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM native Construction-2 packed bit sink requires a non-zero full width".into(),
            ));
        }
        Ok(Self {
            full_width,
            written: 0,
            column_bits: vec![0u64; commit_cols_for_full_width(full_width)],
        })
    }

    fn finish(self, label: &str) -> Result<Vec<u64>, SimpleKernelError> {
        if self.written != self.full_width {
            return Err(SimpleKernelError::Bridge(format!(
                "{label}: packed bit image wrote {} bits but expected {}",
                self.written, self.full_width
            )));
        }
        Ok(self.column_bits)
    }
}

impl Rv64imConstruction2BitSink for PackedBinaryMatBitSink {
    fn push_bit(&mut self, value: F) {
        assert!(
            self.written < self.full_width,
            "packed binary Construction-2 image overflowed its canonical full width"
        );
        let block = self.written / D;
        let rho = self.written % D;
        assert!(
            value == F::ZERO || value == F::ONE,
            "packed binary Construction-2 image received a non-binary field slot"
        );
        if value == F::ONE {
            self.column_bits[block] |= 1u64 << rho;
        }
        self.written += 1;
    }

    fn skip_zero_bits(&mut self, count: usize) {
        let next = self
            .written
            .checked_add(count)
            .expect("packed binary Construction-2 image bit index overflowed");
        assert!(
            next <= self.full_width,
            "packed binary Construction-2 image overflowed its canonical full width"
        );
        self.written = next;
    }

    fn push_u64_bits_le(&mut self, word: u64) {
        if word == 0 {
            self.skip_zero_bits(64);
            return;
        }
        let next = self
            .written
            .checked_add(64)
            .expect("packed binary Construction-2 image bit index overflowed");
        assert!(
            next <= self.full_width,
            "packed binary Construction-2 image overflowed its canonical full width"
        );

        let block = self.written / D;
        let offset = self.written % D;
        let first_len = (D - offset).min(64);
        let first_mask = if first_len == 64 {
            u64::MAX
        } else {
            (1u64 << first_len) - 1
        };
        self.column_bits[block] |= (word & first_mask) << offset;

        let mut consumed = first_len;
        if consumed < 64 {
            let second_len = (64 - consumed).min(D);
            let second_mask = if second_len == 64 {
                u64::MAX
            } else {
                (1u64 << second_len) - 1
            };
            self.column_bits[block + 1] |= (word >> consumed) & second_mask;
            consumed += second_len;
            if consumed < 64 {
                let third_len = 64 - consumed;
                let third_mask = if third_len == 64 {
                    u64::MAX
                } else {
                    (1u64 << third_len) - 1
                };
                self.column_bits[block + 2] |= (word >> consumed) & third_mask;
            }
        }

        self.written = next;
    }
}

fn append_u64_field_bits(out: &mut impl Rv64imConstruction2BitSink, value: u64) {
    append_field_bits_le(out, F::from_u64(value));
}

fn append_field_bits_le(out: &mut impl Rv64imConstruction2BitSink, value: F) {
    out.push_u64_bits_le(value.as_canonical_u64());
}

fn append_binary_field_slots(out: &mut impl Rv64imConstruction2BitSink, values: &[F], label: &str) {
    for (bit_index, &value) in values.iter().enumerate() {
        assert!(
            value == F::ZERO || value == F::ONE,
            "{label}: field slot {bit_index} is not binary"
        );
        out.push_bit(value);
    }
}

fn append_digest_fields(out: &mut Vec<F>, digest: [u8; 32]) {
    out.extend(digest32_as_fields(digest));
}

fn append_digest_field_bits(out: &mut impl Rv64imConstruction2BitSink, digest: [u8; 32]) {
    for value in digest32_as_fields(digest) {
        append_field_bits_le(out, value);
    }
}

fn append_commitment_fields(out: &mut Vec<F>, commitment: &Commitment) {
    out.extend(commitment.data.iter().copied());
}

fn append_commitment_field_bits(out: &mut impl Rv64imConstruction2BitSink, commitment: &Commitment) {
    for &value in &commitment.data {
        append_field_bits_le(out, value);
    }
}

fn append_f_slice(out: &mut Vec<F>, values: &[F]) {
    out.extend(values.iter().copied());
}

fn append_f_slice_bits(out: &mut impl Rv64imConstruction2BitSink, values: &[F]) {
    for &value in values {
        append_field_bits_le(out, value);
    }
}

fn append_f_matrix(out: &mut Vec<F>, matrix: &neo_ccs::Mat<F>) {
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            out.push(matrix[(row, col)]);
        }
    }
}

fn append_f_matrix_bits(out: &mut impl Rv64imConstruction2BitSink, matrix: &neo_ccs::Mat<F>) {
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            append_field_bits_le(out, matrix[(row, col)]);
        }
    }
}

fn append_k_value(out: &mut Vec<F>, value: &K) {
    out.extend(value.as_coeffs());
}

fn append_k_value_bits(out: &mut impl Rv64imConstruction2BitSink, value: &K) {
    for coeff in value.as_coeffs() {
        append_field_bits_le(out, coeff);
    }
}

fn append_k_slice(out: &mut Vec<F>, values: &[K]) {
    for value in values {
        append_k_value(out, value);
    }
}

fn append_k_slice_bits(out: &mut impl Rv64imConstruction2BitSink, values: &[K]) {
    for value in values {
        append_k_value_bits(out, value);
    }
}

fn append_k_rows(out: &mut Vec<F>, rows: &[Vec<K>]) {
    for row in rows {
        append_k_slice(out, row);
    }
}

fn append_k_rows_bits(out: &mut impl Rv64imConstruction2BitSink, rows: &[Vec<K>]) {
    for row in rows {
        append_k_slice_bits(out, row);
    }
}

fn append_ccs_claim_fields(out: &mut Vec<F>, claim: &neo_ccs::CcsClaim<Commitment, F>) {
    append_commitment_fields(out, &claim.c);
    append_f_slice(out, &claim.x);
}

fn append_ccs_claim_field_bits(out: &mut impl Rv64imConstruction2BitSink, claim: &neo_ccs::CcsClaim<Commitment, F>) {
    append_commitment_field_bits(out, &claim.c);
    append_f_slice_bits(out, &claim.x);
}

fn append_ccs_witness_fields(out: &mut Vec<F>, witness: &neo_ccs::CcsWitness<F>) {
    append_f_slice(out, &witness.w);
    append_f_matrix(out, &witness.Z);
}

fn append_ccs_witness_field_bits(out: &mut impl Rv64imConstruction2BitSink, witness: &neo_ccs::CcsWitness<F>) {
    append_f_slice_bits(out, &witness.w);
    append_f_matrix_bits(out, &witness.Z);
}

fn validate_ce_claim_ct_alias(claim: &neo_ccs::CeClaim<Commitment, F, K>) -> Result<(), SimpleKernelError> {
    if claim.ct.len() > claim.y_ring.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 carried CE claim ct vector exceeds y_ring rows".into(),
        ));
    }
    for (idx, expected) in claim.ct.iter().enumerate() {
        let actual = claim
            .y_ring
            .get(idx)
            .and_then(|row| row.first())
            .copied()
            .ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV64IM native Construction-2 carried CE claim ct alias row is missing".into(),
                )
            })?;
        if actual != *expected {
            return Err(SimpleKernelError::Bridge(
                "RV64IM native Construction-2 carried CE claim ct is not the y_ring constant term".into(),
            ));
        }
    }
    Ok(())
}

fn validate_shared_state_in_claims(claims: &[neo_ccs::CeClaim<Commitment, F, K>]) -> Result<(), SimpleKernelError> {
    let Some((first, rest)) = claims.split_first() else {
        return Ok(());
    };
    if first.X.rows() == 0 || first.X.cols() != first.m_in {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 carried CE claim must expose packed X with cols == m_in".into(),
        ));
    }
    validate_ce_claim_ct_alias(first)?;
    for claim in rest {
        if claim.r != first.r {
            return Err(SimpleKernelError::Bridge(
                "RV64IM native Construction-2 carried CE claims must share one evaluation point r".into(),
            ));
        }
        if claim.X.rows() == 0 || claim.X.cols() != claim.m_in {
            return Err(SimpleKernelError::Bridge(
                "RV64IM native Construction-2 carried CE claim must expose packed X with cols == m_in".into(),
            ));
        }
        validate_ce_claim_ct_alias(claim)?;
    }
    Ok(())
}

fn append_compact_x_fields(out: &mut Vec<F>, claim: &neo_ccs::CeClaim<Commitment, F, K>) {
    for col in 0..claim.m_in {
        out.push(claim.X[(col % claim.X.rows(), col)]);
    }
}

fn append_compact_x_field_bits(out: &mut impl Rv64imConstruction2BitSink, claim: &neo_ccs::CeClaim<Commitment, F, K>) {
    for col in 0..claim.m_in {
        append_field_bits_le(out, claim.X[(col % claim.X.rows(), col)]);
    }
}

fn append_state_in_claim_fields(
    out: &mut Vec<F>,
    claims: &[neo_ccs::CeClaim<Commitment, F, K>],
) -> Result<(), SimpleKernelError> {
    validate_shared_state_in_claims(claims)?;
    if let Some(first) = claims.first() {
        append_k_slice(out, &first.r);
    }
    for claim in claims {
        append_commitment_fields(out, &claim.c);
        append_compact_x_fields(out, claim);
        append_k_rows(out, &claim.y_ring);
    }
    Ok(())
}

fn append_state_in_claim_field_bits(
    out: &mut impl Rv64imConstruction2BitSink,
    claims: &[neo_ccs::CeClaim<Commitment, F, K>],
) -> Result<(), SimpleKernelError> {
    validate_shared_state_in_claims(claims)?;
    if let Some(first) = claims.first() {
        append_k_slice_bits(out, &first.r);
    }
    for claim in claims {
        append_commitment_field_bits(out, &claim.c);
        append_compact_x_field_bits(out, claim);
        append_k_rows_bits(out, &claim.y_ring);
    }
    Ok(())
}

fn append_step_input_fields(out: &mut Vec<F>, step: &StepInput) {
    append_ccs_claim_fields(out, &step.mcs);
    append_ccs_witness_fields(out, &step.witness);
}

fn append_step_input_field_bits(out: &mut impl Rv64imConstruction2BitSink, step: &StepInput) {
    append_ccs_claim_field_bits(out, &step.mcs);
    append_ccs_witness_field_bits(out, &step.witness);
}

fn append_chunk_input_fields(out: &mut Vec<F>, chunk_input: &ChunkInput) {
    append_u64_field(out, chunk_input.start_index as u64);
    append_u64_field(out, chunk_input.steps.len() as u64);
    for step in &chunk_input.steps {
        append_step_input_fields(out, step);
    }
}

fn append_chunk_input_field_bits(out: &mut impl Rv64imConstruction2BitSink, chunk_input: &ChunkInput) {
    append_u64_field_bits(out, chunk_input.start_index as u64);
    append_u64_field_bits(out, chunk_input.steps.len() as u64);
    for step in &chunk_input.steps {
        append_step_input_field_bits(out, step);
    }
}

fn append_construction2_fresh_instance_fields(
    out: &mut Vec<F>,
    fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
) {
    append_commitment_fields(out, fresh_instance.commitment().commitment());
    out.extend(fresh_instance.x_i().field_image());
}

fn append_construction2_fresh_instance_field_bits(
    out: &mut impl Rv64imConstruction2BitSink,
    fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
) {
    append_commitment_field_bits(out, fresh_instance.commitment().commitment());
    for value in fresh_instance.x_i().field_image() {
        append_field_bits_le(out, value);
    }
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

fn append_phi_side_field_bits(out: &mut impl Rv64imConstruction2BitSink, advice: &Rv64imMainRecursionFPrimeAdvice) {
    append_u64_field_bits(out, advice.phi_side().commitment_count());
    for words in advice.phi_side().commitment_words() {
        append_u64_field_bits(out, words.len() as u64);
        for &word in words {
            append_u64_field_bits(out, word);
        }
    }
}

fn append_pi_fold_fields(out: &mut Vec<F>, pi_fold: &Rv64imMainRecursionConstruction2PiFoldProof) {
    append_u64_field(out, pi_fold.ccs_replay_payload.sumcheck_rounds.len() as u64);
    for round in &pi_fold.ccs_replay_payload.sumcheck_rounds {
        append_k_slice(out, round);
    }
    append_u64_field(out, pi_fold.ccs_replay_payload.sumcheck_rounds_nc.len() as u64);
    for round in &pi_fold.ccs_replay_payload.sumcheck_rounds_nc {
        append_k_slice(out, round);
    }
}

fn append_pi_fold_field_bits(
    out: &mut impl Rv64imConstruction2BitSink,
    pi_fold: &Rv64imMainRecursionConstruction2PiFoldProof,
) {
    append_u64_field_bits(out, pi_fold.ccs_replay_payload.sumcheck_rounds.len() as u64);
    for round in &pi_fold.ccs_replay_payload.sumcheck_rounds {
        append_k_slice_bits(out, round);
    }
    append_u64_field_bits(out, pi_fold.ccs_replay_payload.sumcheck_rounds_nc.len() as u64);
    for round in &pi_fold.ccs_replay_payload.sumcheck_rounds_nc {
        append_k_slice_bits(out, round);
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
    let step_cap = advice.verifier_key_fs().step_cap()?;
    let active_step_count = advice.verified_kernel_handoff().chunk_input.steps.len();
    if active_step_count == 0 || active_step_count > step_cap {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM native Construction-2 shape builder requires 1..={step_cap} public steps per recursive relation; got {active_step_count}"
        )));
    }
    if !advice.bridge_handoff_halted_out() && active_step_count != step_cap {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM native Construction-2 non-terminal recursive relation must carry exactly step_cap={step_cap} public steps; got {active_step_count}"
        )));
    }
    let fresh = adapt_rv64im_chunk_to_fresh_ccs(advice.verified_kernel_handoff());
    if fresh.fresh_claims.len() != active_step_count || fresh.fresh_witnesses.len() != active_step_count {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM native Construction-2 shape builder requires one fresh CCS instance per public step; got {} claims, {} witnesses, {active_step_count} public steps",
            fresh.fresh_claims.len(),
            fresh.fresh_witnesses.len(),
        )));
    }
    Ok(())
}

fn validate_rv64im_main_recursion_construction2_input_fresh_instance(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
) -> Result<(), SimpleKernelError> {
    if advice.chunk_count_in() == 0 {
        let canonical_full_width =
            crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_canonical_full_width(
                advice.verifier_key_fs(),
                advice.phi_side(),
            )?;
        let expected_default = build_rv64im_main_recursion_construction2_default_fresh_instance(
            advice.verifier_key_fs(),
            canonical_full_width,
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
        ccs_output_count: (advice.running_state().carry.main.claims.len()
            + advice.verified_kernel_handoff().chunk_input.steps.len()) as u64,
        child_count: verified_step.state.carry.main.claims.len() as u64,
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

fn rv64im_main_recursion_construction2_commitment_seed(
    full_width: usize,
    step_cap: usize,
) -> Result<[u8; 32], SimpleKernelError> {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_construction2_commitment_seed");
    tr.append_message(
        b"neo.fold.next/rv64im/main_recursion_construction2_commitment_seed/version",
        b"v1",
    );
    tr.append_message(
        b"neo.fold.next/rv64im/main_recursion_construction2_commitment_seed/shape_digest",
        &crate::rv64im::recursion_shape::build_rv64im_recursion_shape_for_step_cap(step_cap)?.canonical_digest(),
    );
    tr.append_u64s(
        b"neo.fold.next/rv64im/main_recursion_construction2_commitment_seed/full_width",
        &[full_width as u64],
    );
    Ok(tr.digest32())
}

struct Rv64imMainRecursionConstruction2CommitmentContext {
    log: AjtaiSModule,
    kappa: usize,
    m: usize,
    chunk_size: usize,
    chunk_seeds_by_row: Vec<Vec<[u8; 32]>>,
}

static RV64IM_MAIN_RECURSION_CONSTRUCTION2_COMMITMENT_CONTEXTS: OnceLock<
    Mutex<HashMap<(usize, usize), Arc<Rv64imMainRecursionConstruction2CommitmentContext>>>,
> = OnceLock::new();

impl Rv64imMainRecursionConstruction2CommitmentContext {
    fn new(full_width: usize, step_cap: usize) -> Result<Self, SimpleKernelError> {
        let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM Construction-2 commitment params failed for full width {full_width}: {err}"
            ))
        })?;
        let m = commit_cols_for_full_width(full_width);
        let seed = rv64im_main_recursion_construction2_commitment_seed(full_width, step_cap)?;
        let want_kappa = params.kappa as usize;
        if has_global_pp_for_dims(D, m) {
            if let Ok((kappa, registered_seed)) = get_global_pp_seeded_params_for_dims(D, m) {
                if kappa != want_kappa || registered_seed != seed {
                    return Err(SimpleKernelError::Bridge(format!(
                        "RV64IM Construction-2 commitment PP mismatch for (d,m)=({D},{m})"
                    )));
                }
            }
        } else {
            set_global_pp_seeded(D, want_kappa, m, seed).map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM Construction-2 commitment seed setup failed for (d,m)=({D},{m}): {err}"
                ))
            })?;
        }
        let log = AjtaiSModule::from_global_for_dims(D, m).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM Construction-2 commitment module failed for (d,m)=({D},{m}): {err}"
            ))
        })?;
        let (chunk_size, chunk_seeds_by_row) = seeded_pp_chunk_seeds(seed, want_kappa, m);
        Ok(Self {
            log,
            kappa: want_kappa,
            m,
            chunk_size,
            chunk_seeds_by_row,
        })
    }

    fn commit(&self, packed: &Mat<F>) -> Commitment {
        self.log.commit(packed)
    }

    fn commit_binary_columns(&self, column_bits: &[u64]) -> Commitment {
        commit_row_major_seeded_binary_cols_with_chunk_seeds(
            D,
            self.kappa,
            self.m,
            column_bits,
            self.chunk_size,
            &self.chunk_seeds_by_row,
        )
    }
}

fn rv64im_main_recursion_construction2_commitment_context(
    full_width: usize,
    step_cap: usize,
) -> Result<Arc<Rv64imMainRecursionConstruction2CommitmentContext>, SimpleKernelError> {
    let contexts = RV64IM_MAIN_RECURSION_CONSTRUCTION2_COMMITMENT_CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut contexts = contexts
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM Construction-2 commitment context cache poisoned".into()))?;
    if let Some(context) = contexts.get(&(full_width, step_cap)) {
        return Ok(Arc::clone(context));
    }
    let context = Arc::new(Rv64imMainRecursionConstruction2CommitmentContext::new(
        full_width, step_cap,
    )?);
    contexts.insert((full_width, step_cap), Arc::clone(&context));
    Ok(context)
}

fn build_rv64im_main_recursion_construction2_commitment_log(
    full_width: usize,
    step_cap: usize,
) -> Result<Arc<Rv64imMainRecursionConstruction2CommitmentContext>, SimpleKernelError> {
    rv64im_main_recursion_construction2_commitment_context(full_width, step_cap)
}

pub fn build_rv64im_main_recursion_construction2_f_prime_ccs_shape(
    advices: &[Rv64imMainRecursionFPrimeAdvice],
) -> Result<Rv64imMainRecursionConstruction2FPrimeCcsShape, SimpleKernelError> {
    let first = advices.first().ok_or_else(|| {
        SimpleKernelError::Build("RV64IM native Construction-2 shape builder requires at least one F' advice".into())
    })?;
    let recursion_shape_digest =
        crate::rv64im::recursion_shape::build_rv64im_recursion_shape_for_step_cap(first.verifier_key_fs().step_cap()?)?
            .canonical_digest();
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
    append_state_in_claim_fields(&mut logical_values, &advice.running_state().carry.main.claims)?;
    append_construction2_fresh_instance_fields(&mut logical_values, current_input_fresh_instance);
    append_chunk_input_fields(&mut logical_values, &advice.verified_kernel_handoff().chunk_input);
    append_pi_fold_fields(&mut logical_values, pi_fold);

    Ok(Rv64imMainRecursionConstruction2FPrimeWitnessImage { logical_values })
}

pub fn build_rv64im_main_recursion_construction2_f_prime_low_norm_witness_image(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
) -> Result<Rv64imMainRecursionConstruction2FPrimeLowNormWitnessImage, SimpleKernelError> {
    let pi_fold = advice.construction2_pi_fold();
    let canonical_full_width =
        crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_canonical_full_width(
            advice.verifier_key_fs(),
            advice.phi_side(),
        )?;
    let mut binary_values = Vec::with_capacity(canonical_full_width.saturating_sub(RV64IM_ENC_INST_BITS));
    append_u64_field_bits(&mut binary_values, advice.chunk_count_in());
    append_digest_field_bits(&mut binary_values, *advice.z_0());
    append_digest_field_bits(&mut binary_values, *advice.z_i());
    append_u64_field_bits(&mut binary_values, advice.pc_i());
    append_phi_side_field_bits(&mut binary_values, advice);
    append_u64_field_bits(
        &mut binary_values,
        advice.running_state().carry.main.claims.len() as u64,
    );
    append_state_in_claim_field_bits(&mut binary_values, &advice.running_state().carry.main.claims)?;
    append_construction2_fresh_instance_field_bits(&mut binary_values, current_input_fresh_instance);
    append_chunk_input_field_bits(&mut binary_values, &advice.verified_kernel_handoff().chunk_input);
    append_pi_fold_field_bits(&mut binary_values, pi_fold);
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
    build_rv64im_main_recursion_construction2_nifs_bridge_with_trace(advice, current_input_fresh_instance, None)
}

pub(crate) fn build_rv64im_main_recursion_construction2_nifs_bridge_with_trace<'a>(
    advice: &'a Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &'a Rv64imMainRecursionConstruction2FreshInstance,
    trace_prefix: Option<&str>,
) -> Result<Rv64imMainRecursionConstruction2NifsBridge<'a>, SimpleKernelError> {
    let total_started = Instant::now();
    let started = Instant::now();
    validate_rv64im_main_recursion_construction2_advice(advice)?;
    emit_debug_timing(trace_prefix, "validate_advice", elapsed_ms(started));
    let started = Instant::now();
    validate_rv64im_main_recursion_construction2_input_fresh_instance(advice, current_input_fresh_instance)?;
    emit_debug_timing(trace_prefix, "validate_input_fresh_instance", elapsed_ms(started));
    let started = Instant::now();
    let bridge = Rv64imMainRecursionConstruction2NifsBridge {
        input_fresh_instance: current_input_fresh_instance,
        state_in: advice.running_state(),
        expected_state_out: advice.fresh_state_out(),
        chunk_index: advice.chunk_index(),
        pi_fold: advice.construction2_pi_fold(),
        replay_witness: advice.main_circuit_replay_witness(),
        chunk_replay_input: Rv64imMainRecursionConstruction2ReplayInput::from_verified_kernel_handoff(
            advice.verified_kernel_handoff(),
        ),
    };
    emit_debug_timing(trace_prefix, "materialize_bridge", elapsed_ms(started));
    emit_debug_timing(trace_prefix, "total", elapsed_ms(total_started));
    Ok(bridge)
}

pub(crate) fn verify_rv64im_main_recursion_construction2_nifs_step(
    bridge: &Rv64imMainRecursionConstruction2NifsBridge<'_>,
) -> Result<Rv64imMainRecursionConstruction2VerifiedStep, SimpleKernelError> {
    Ok(verify_rv64im_main_recursion_construction2_nifs_step_with_perf_and_trace(bridge, None)?.0)
}

pub(crate) fn verify_rv64im_main_recursion_construction2_nifs_step_with_perf_and_trace(
    bridge: &Rv64imMainRecursionConstruction2NifsBridge<'_>,
    trace_prefix: Option<&str>,
) -> Result<
    (
        Rv64imMainRecursionConstruction2VerifiedStep,
        Rv64imMainRecursionConstruction2NifsVerifyPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();
    let mut perf = Rv64imMainRecursionConstruction2NifsVerifyPerf::default();
    let started = Instant::now();
    if !bridge.input_fresh_instance.x_i().is_binary_low_norm() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM native Construction-2 NIFS bridge carries a non-binary x_i image".into(),
        ));
    }
    emit_debug_timing(trace_prefix, "prechecks", elapsed_ms(started));
    let started = Instant::now();
    let (transcript, chunk_relation_digest) = verify_rv64im_main_recursion_construction2_verified_relation(bridge)?;
    perf.chunk_relation_verify_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "chunk_relation_verify", perf.chunk_relation_verify_ms);
    let started = Instant::now();
    let state = derive_rv64im_main_recursion_construction2_next_state_from_expected_state_out(
        bridge,
        chunk_relation_digest,
        &transcript,
    )?;
    perf.derive_next_state_ms = elapsed_ms(started);
    emit_debug_timing(trace_prefix, "derive_next_state", perf.derive_next_state_ms);
    perf.total_ms = elapsed_ms(total_started);
    emit_debug_timing(trace_prefix, "total", perf.total_ms);
    Ok((Rv64imMainRecursionConstruction2VerifiedStep { state }, perf))
}

fn verify_rv64im_main_recursion_construction2_pi_ccs(
    bridge: &Rv64imMainRecursionConstruction2NifsBridge<'_>,
) -> Result<(Rv64imChunkRelationTrace, Poseidon2Transcript), SimpleKernelError> {
    validate_rv64im_main_recursion_construction2_chunk_replay_input(bridge.state_in, &bridge.chunk_replay_input)?;
    let (params, log, structure) =
        rv64im_root_main_lane_context_for_claim_count(bridge.state_in.carry.main.claims.len())?;
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
        &params,
        structure,
        log,
        optimized_cache,
    )
    .map_err(|err| SimpleKernelError::Proof(format!("RV64IM Construction-2 Pi_CCS replay failed: {err}")))?;
    let pi_ccs_transcript = Poseidon2Transcript::from_state_and_absorbed(
        trace.pi_ccs_post_transcript.state,
        trace.pi_ccs_post_transcript.absorbed,
    );
    for (idx, claim) in trace.ccs_outputs.iter().enumerate() {
        validate_rv64im_main_recursion_construction2_ce_claim_surface(
            claim,
            &format!("RV64IM Construction-2 Pi_CCS output {idx}"),
        )?;
    }
    Ok((trace, pi_ccs_transcript))
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
    let (params, _, _) = rv64im_root_main_lane_context_for_claim_count(bridge.state_in.carry.main.claims.len())?;
    Ok(
        sample_rv64im_main_recursion_construction2_pi_rlc_rhos(&mut transcript, &params, trace.ccs_outputs.len())?
            .into_iter()
            .map(|rho| rho.into_mat())
            .collect(),
    )
}

pub(crate) fn build_rv64im_main_recursion_construction2_verified_step_statement_from_parts(
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

pub(crate) fn build_rv64im_main_recursion_construction2_verified_step_statement_from_summary(
    chunk_index: u64,
    chunk_summary: &FixedShapeChunkSummary,
    state_in: &Rv64imChunkFoldState,
    next_state: &Rv64imChunkFoldState,
) -> Rv64imMainRecursionConstruction2VerifiedStepStatement {
    let step_lo = chunk_summary.start_index;
    let step_hi = step_lo + chunk_summary.public_step_count;
    Rv64imMainRecursionConstruction2VerifiedStepStatement {
        chunk_index,
        step_lo,
        step_hi,
        state_in: state_in.carry.terminal_handle.0,
        state_out: next_state.carry.terminal_handle.0,
        public_chunk_digest: chunk_summary.public_chunk_digest,
        chunk_relation_digest: chunk_summary.chunk_relation_digest,
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
    let (params, log, structure) =
        rv64im_root_main_lane_context_for_claim_count(relation.witness.state_in.carry.main.claims.len())?;
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
        &params,
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

fn verify_rv64im_main_recursion_construction2_verified_relation(
    bridge: &Rv64imMainRecursionConstruction2NifsBridge<'_>,
) -> Result<(Poseidon2Transcript, [u8; 32]), SimpleKernelError> {
    validate_rv64im_main_recursion_construction2_chunk_replay_input(bridge.state_in, &bridge.chunk_replay_input)?;
    let (params, _, structure) =
        rv64im_root_main_lane_context_for_claim_count(bridge.state_in.carry.main.claims.len())?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(
        bridge.state_in.transcript.state,
        bridge.state_in.transcript.absorbed,
    );
    transcript.append_fields_raw(&[
        F::from_u64(14),
        F::from_u64(bridge.chunk_replay_input.chunk_input.start_index as u64),
        F::from_u64(bridge.chunk_replay_input.chunk_input.steps.len() as u64),
    ]);
    let fresh_claims = bridge
        .chunk_replay_input
        .chunk_input
        .steps
        .iter()
        .map(|step| step.mcs.clone())
        .collect::<Vec<_>>();
    let ccs_proof = bridge.replay_witness.ccs_replay_proof.to_pi_ccs_proof();
    let (ok, _) = optimized_verify_with_cache_and_instance_digest_and_perf(
        &mut transcript,
        &params,
        structure,
        &fresh_claims,
        &bridge.state_in.carry.main.claims,
        &bridge.replay_witness.ccs_outputs,
        &ccs_proof,
        &optimized_cache,
        public_chunk_digest(&bridge.chunk_replay_input.chunk_input.public()),
    )
    .map_err(|err| SimpleKernelError::Proof(format!("RV64IM Construction-2 Pi_CCS verify failed: {err}")))?;
    if !ok {
        return Err(SimpleKernelError::Proof(
            "RV64IM Construction-2 Pi_CCS replay witness does not verify against the carried chunk relation".into(),
        ));
    }
    let verified_fold_digest = transcript.digest32();
    if verified_fold_digest != bridge.replay_witness.ccs_replay_proof.header_digest {
        return Err(SimpleKernelError::Proof(
            "RV64IM Construction-2 Pi_CCS replay header digest does not match the verifier transcript".into(),
        ));
    }
    for (idx, claim) in bridge.replay_witness.ccs_outputs.iter().enumerate() {
        validate_rv64im_main_recursion_construction2_ce_claim_surface(
            claim,
            &format!("RV64IM Construction-2 Pi_CCS output {idx}"),
        )?;
    }
    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, structure)
        .map_err(|err| SimpleKernelError::Proof(format!("RV64IM Construction-2 public verifier dims failed: {err}")))?;
    let rhos = sample_rv64im_main_recursion_construction2_pi_rlc_rhos(
        &mut transcript,
        &params,
        bridge.replay_witness.ccs_outputs.len(),
    )?;
    let parent = rlc_public(
        structure,
        &params,
        &rhos,
        &bridge.replay_witness.ccs_outputs,
        rv64im_ajtai_mixers().mix_rhos_commits,
        dims.ell_d,
    )
    .map_err(|err| SimpleKernelError::Proof(format!("RV64IM Construction-2 Pi_RLC public verify failed: {err}")))?;
    if !verify_dec_public(
        structure,
        &params,
        &parent,
        &bridge.expected_state_out.carry.main.claims,
        rv64im_ajtai_mixers().combine_b_pows,
        dims.ell_d,
    ) {
        return Err(SimpleKernelError::Proof(
            "RV64IM Construction-2 Pi_DEC public verification failed against the carried next-state claims".into(),
        ));
    }
    for (idx, claim) in bridge
        .expected_state_out
        .carry
        .main
        .claims
        .iter()
        .enumerate()
    {
        validate_rv64im_main_recursion_construction2_ce_claim_surface(
            claim,
            &format!("RV64IM Construction-2 verified child {idx}"),
        )?;
    }
    let chunk_relation_digest = rv64im_chunk_relation_digest_from_fold_digest(
        rv64im_public_chunk_digest(&bridge.chunk_replay_input.chunk_input.public()),
        verified_fold_digest,
        bridge.chunk_replay_input.bridge_handoff.digest,
    );
    Ok((transcript, chunk_relation_digest))
}

fn derive_rv64im_main_recursion_construction2_next_state_from_verified_relation(
    state_in: &Rv64imChunkFoldState,
    replay_input: &Rv64imMainRecursionConstruction2ReplayInput,
    verified_relation: crate::chunk_relation::ChunkRelationResult,
    chunk_relation_digest: [u8; 32],
    transcript: &Poseidon2Transcript,
) -> Result<Rv64imChunkFoldState, SimpleKernelError> {
    let crate::chunk_relation::ChunkRelationResult { next_main, .. } = verified_relation;
    if next_main.witnesses.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 native next-state derivation requires a non-empty Π_DEC digit witness".into(),
        ));
    }
    if next_main.claims.len() != next_main.witnesses.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM Construction-2 native next-state derivation: Π_DEC children ({}) and digit witnesses ({}) disagree",
            next_main.claims.len(),
            next_main.witnesses.len(),
        )));
    }
    let next_carry = Rv64imChunkFoldCarry::from_main(
        next_main,
        Rv64imAccumulatorHandle(rv64im_step_handle(
            state_in.carry.terminal_handle.0,
            replay_input.bridge_handoff.chunk_index as usize,
            replay_input.chunk_input.start_index,
            replay_input.chunk_input.steps.len(),
            chunk_relation_digest,
        )),
    );
    let transcript_out = rv64im_chunk_fold_carried_transcript_snapshot(&Rv64imChunkFoldTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    });
    Ok(Rv64imChunkFoldState {
        carry: next_carry,
        transcript: transcript_out,
    })
}

fn derive_rv64im_main_recursion_construction2_next_state_from_expected_state_out(
    bridge: &Rv64imMainRecursionConstruction2NifsBridge<'_>,
    chunk_relation_digest: [u8; 32],
    transcript: &Poseidon2Transcript,
) -> Result<Rv64imChunkFoldState, SimpleKernelError> {
    let expected_terminal_handle = Rv64imAccumulatorHandle(rv64im_step_handle(
        bridge.state_in.carry.terminal_handle.0,
        bridge.chunk_replay_input.bridge_handoff.chunk_index as usize,
        bridge.chunk_replay_input.chunk_input.start_index,
        bridge.chunk_replay_input.chunk_input.steps.len(),
        chunk_relation_digest,
    ));
    if bridge.expected_state_out.carry.main.claims.len() != bridge.expected_state_out.carry.main.witnesses.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 carried next-state CE claim and witness counts diverged".into(),
        ));
    }
    if bridge.expected_state_out.carry.terminal_handle != expected_terminal_handle {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 carried next-state terminal handle does not match the public verified relation"
                .into(),
        ));
    }
    let transcript_out = rv64im_chunk_fold_carried_transcript_snapshot(&Rv64imChunkFoldTranscriptSnapshot {
        state: transcript.state(),
        absorbed: transcript.absorbed(),
    });
    if bridge.expected_state_out.transcript != transcript_out {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 carried next-state transcript does not match the public verified relation".into(),
        ));
    }
    Ok(bridge.expected_state_out.clone())
}

fn derive_rv64im_main_recursion_construction2_next_state_from_trace(
    state_in: &Rv64imChunkFoldState,
    replay_input: &Rv64imMainRecursionConstruction2ReplayInput,
    trace: &crate::rv64im::chunk_relation::Rv64imChunkRelationTrace,
    transcript: &Poseidon2Transcript,
) -> Result<Rv64imChunkFoldState, SimpleKernelError> {
    let verified_relation = crate::chunk_relation::ChunkRelationResult {
        next_main: Carry {
            claims: trace.children.clone(),
            witnesses: trace.z_split.clone(),
        },
        artifacts: crate::chunk_relation::ChunkRelationArtifacts {
            relation_digest: trace.chunk_relation_digest,
        },
    };
    derive_rv64im_main_recursion_construction2_next_state_from_verified_relation(
        state_in,
        replay_input,
        verified_relation,
        trace.chunk_relation_digest,
        transcript,
    )
}

fn encode_binary_vector_for_full_width(full_width: usize, witness: &[F], label: &str) -> Result<Mat<F>, String> {
    if witness.len() != full_width {
        return Err(format!(
            "{label}: witness length {} != full_width {}",
            witness.len(),
            full_width
        ));
    }
    if full_width == 0 {
        return Err(format!("{label}: full_width must be > 0"));
    }
    let cols = commit_cols_for_full_width(full_width);
    let mut out = Mat::zero(D, cols, F::ZERO);
    for (column, &value) in witness.iter().enumerate() {
        if value != F::ZERO && value != F::ONE {
            return Err(format!("{label}: witness coefficient at index {column} is not binary"));
        }
        let block = column / D;
        let rho = column % D;
        out[(rho, block)] = value;
    }
    Ok(out)
}

pub(crate) fn build_rv64im_main_recursion_construction2_fresh_instance_from_full_vector(
    chunk_count_out: u64,
    step_cap: usize,
    x_i: Rv64imEncodedPublicInput,
    full_vector: &[F],
) -> Result<Rv64imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    let context = build_rv64im_main_recursion_construction2_commitment_log(full_vector.len(), step_cap)?;
    let packed = encode_binary_vector_for_full_width(
        full_vector.len(),
        full_vector,
        "RV64IM native Construction-2 fresh instance",
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM native Construction-2 fresh instance encoding failed for chunk {chunk_count_out}: {err}"
        ))
    })?;
    Ok(Rv64imMainRecursionConstruction2FreshInstance {
        c_i: Rv64imMainRecursionConstruction2Commitment(context.commit(&packed)),
        x_i,
    })
}

pub(crate) fn debug_trace_build_rv64im_main_recursion_construction2_fresh_instance_from_full_vector(
    chunk_count_out: u64,
    step_cap: usize,
    x_i: Rv64imEncodedPublicInput,
    full_vector: &[F],
    trace_prefix: &str,
) -> Result<Rv64imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    let emit = |label: &str, elapsed_ms: f64| {
        eprintln!("{trace_prefix}.{label}={elapsed_ms:.2}ms");
        let _ = io::stderr().flush();
    };
    let started = Instant::now();
    let context = build_rv64im_main_recursion_construction2_commitment_log(full_vector.len(), step_cap)?;
    emit("commitment_context", started.elapsed().as_secs_f64() * 1_000.0);
    let started = Instant::now();
    let packed = encode_binary_vector_for_full_width(
        full_vector.len(),
        full_vector,
        "RV64IM native Construction-2 fresh instance",
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM native Construction-2 fresh instance encoding failed for chunk {chunk_count_out}: {err}"
        ))
    })?;
    emit("encode_vector", started.elapsed().as_secs_f64() * 1_000.0);
    let started = Instant::now();
    let commitment = context.commit(&packed);
    emit("commit", started.elapsed().as_secs_f64() * 1_000.0);
    Ok(Rv64imMainRecursionConstruction2FreshInstance {
        c_i: Rv64imMainRecursionConstruction2Commitment(commitment),
        x_i,
    })
}

fn build_rv64im_main_recursion_construction2_fresh_instance_packed_image(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
    x_i: &Rv64imEncodedPublicInput,
) -> Result<Vec<u64>, SimpleKernelError> {
    let pi_fold = advice.construction2_pi_fold();
    let canonical_full_width =
        crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_canonical_full_width(
            advice.verifier_key_fs(),
            advice.phi_side(),
        )?;
    let mut sink = PackedBinaryMatBitSink::new(canonical_full_width)?;
    append_binary_field_slots(&mut sink, &x_i.field_image(), "RV64IM native Construction-2 x_i");
    append_u64_field_bits(&mut sink, advice.chunk_count_in());
    append_digest_field_bits(&mut sink, *advice.z_0());
    append_digest_field_bits(&mut sink, *advice.z_i());
    append_u64_field_bits(&mut sink, advice.pc_i());
    append_phi_side_field_bits(&mut sink, advice);
    append_u64_field_bits(&mut sink, advice.running_state().carry.main.claims.len() as u64);
    append_state_in_claim_field_bits(&mut sink, &advice.running_state().carry.main.claims)?;
    append_construction2_fresh_instance_field_bits(&mut sink, current_input_fresh_instance);
    append_chunk_input_field_bits(&mut sink, &advice.verified_kernel_handoff().chunk_input);
    append_pi_fold_field_bits(&mut sink, pi_fold);
    sink.finish("RV64IM native Construction-2 fresh instance")
}

fn debug_trace_build_rv64im_main_recursion_construction2_fresh_instance_from_input_direct(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
    x_i: Rv64imEncodedPublicInput,
    trace_prefix: &str,
) -> Result<Rv64imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    let emit = |label: &str, elapsed_ms: f64| {
        eprintln!("{trace_prefix}.{label}={elapsed_ms:.2}ms");
        let _ = io::stderr().flush();
    };
    let started = Instant::now();
    let canonical_full_width =
        crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_canonical_full_width(
            advice.verifier_key_fs(),
            advice.phi_side(),
        )?;
    emit("canonical_full_width", started.elapsed().as_secs_f64() * 1_000.0);
    let started = Instant::now();
    let context = build_rv64im_main_recursion_construction2_commitment_log(
        canonical_full_width,
        advice.verifier_key_fs().step_cap()?,
    )?;
    emit("commitment_context", started.elapsed().as_secs_f64() * 1_000.0);
    let started = Instant::now();
    let packed = build_rv64im_main_recursion_construction2_fresh_instance_packed_image(
        advice,
        current_input_fresh_instance,
        &x_i,
    )?;
    emit("pack_image", started.elapsed().as_secs_f64() * 1_000.0);
    let started = Instant::now();
    let commitment =
        Rv64imMainRecursionConstruction2Commitment::from_commitment(context.commit_binary_columns(&packed));
    emit("commit", started.elapsed().as_secs_f64() * 1_000.0);
    Ok(Rv64imMainRecursionConstruction2FreshInstance { c_i: commitment, x_i })
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
    Ok(
        build_rv64im_main_recursion_construction2_fresh_instance_with_input_and_x_i_with_perf(
            advice,
            current_input_fresh_instance,
            x_i,
        )?
        .0,
    )
}

pub(crate) fn build_rv64im_main_recursion_construction2_fresh_instance_with_input_and_x_i_with_perf(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
    x_i: Rv64imEncodedPublicInput,
) -> Result<
    (
        Rv64imMainRecursionConstruction2FreshInstance,
        Rv64imMainRecursionConstruction2FreshInstanceBuildPerf,
    ),
    SimpleKernelError,
> {
    validate_rv64im_main_recursion_construction2_advice(advice)?;
    let total_started = Instant::now();
    let mut perf = Rv64imMainRecursionConstruction2FreshInstanceBuildPerf::default();
    let started = Instant::now();
    let canonical_full_width =
        crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_canonical_full_width(
            advice.verifier_key_fs(),
            advice.phi_side(),
        )?;
    perf.canonical_full_width_ms = elapsed_ms(started);
    let started = Instant::now();
    let context = build_rv64im_main_recursion_construction2_commitment_log(
        canonical_full_width,
        advice.verifier_key_fs().step_cap()?,
    )?;
    perf.commitment_context_ms = elapsed_ms(started);
    let started = Instant::now();
    let packed = build_rv64im_main_recursion_construction2_fresh_instance_packed_image(
        advice,
        current_input_fresh_instance,
        &x_i,
    )?;
    perf.pack_image_ms = elapsed_ms(started);
    let started = Instant::now();
    let commitment =
        Rv64imMainRecursionConstruction2Commitment::from_commitment(context.commit_binary_columns(&packed));
    perf.commit_ms = elapsed_ms(started);
    perf.total_ms = elapsed_ms(total_started);
    Ok((
        Rv64imMainRecursionConstruction2FreshInstance { c_i: commitment, x_i },
        perf,
    ))
}

pub(crate) fn audit_rv64im_main_recursion_construction2_binary_commit(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
    x_i: &Rv64imEncodedPublicInput,
) -> Result<SeededBinaryColsCommitAudit, SimpleKernelError> {
    validate_rv64im_main_recursion_construction2_advice(advice)?;
    let canonical_full_width =
        crate::rv64im::construction2_default::build_rv64im_main_recursion_construction2_canonical_full_width(
            advice.verifier_key_fs(),
            advice.phi_side(),
        )?;
    let context = build_rv64im_main_recursion_construction2_commitment_log(
        canonical_full_width,
        advice.verifier_key_fs().step_cap()?,
    )?;
    let packed = build_rv64im_main_recursion_construction2_fresh_instance_packed_image(
        advice,
        current_input_fresh_instance,
        x_i,
    )?;
    let (audited_commitment, audit) = audit_commit_row_major_seeded_binary_cols_with_chunk_seeds(
        D,
        context.kappa,
        context.m,
        &packed,
        context.chunk_size,
        &context.chunk_seeds_by_row,
    );
    let live_commitment = context.commit_binary_columns(&packed);
    if audited_commitment != live_commitment {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Construction-2 binary-column audit diverged from the live commit_binary_columns path".into(),
        ));
    }
    Ok(audit)
}

pub(crate) fn debug_trace_build_rv64im_main_recursion_construction2_fresh_instance_with_input_and_x_i(
    advice: &Rv64imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &Rv64imMainRecursionConstruction2FreshInstance,
    x_i: Rv64imEncodedPublicInput,
    trace_prefix: &str,
) -> Result<Rv64imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    validate_rv64im_main_recursion_construction2_advice(advice)?;
    debug_trace_build_rv64im_main_recursion_construction2_fresh_instance_from_input_direct(
        advice,
        current_input_fresh_instance,
        x_i,
        &format!("{trace_prefix}.from_input_direct"),
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
