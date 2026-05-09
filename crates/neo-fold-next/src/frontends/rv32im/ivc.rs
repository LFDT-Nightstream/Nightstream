//! Owns the native RV32IM IVC prover carrier.
//!
//! The carrier is for append/resume/compress witness construction only. It is
//! not a proof verifier and must not replay historical chunks as acceptance
//! evidence.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::chunk_folding::ChunkReplayWitness;
use crate::proof::FoldSchedule;
use crate::rv32im::chunk::step_ivc::{
    rv32im_chunk_step_ivc_initial_state_for_step_cap, validate_rv32im_chunk_step_ivc_surface,
    Rv32imChunkStepIvcRelation, Rv32imChunkStepIvcStatement, Rv32imChunkStepIvcWitness,
};
use crate::rv32im::construction2::{
    build_rv32im_main_recursion_construction2_pi_fold_from_replay_witness,
    build_rv32im_main_recursion_construction2_verified_step_statement_digest_from_step_statement,
    build_rv32im_main_recursion_construction2_verified_step_statement_from_summary,
    Rv32imMainRecursionConstruction2FreshInstance, Rv32imMainRecursionConstruction2PublicBoundary,
};
use crate::rv32im::encoded_public_input::{
    digest32_has_canonical_field_limb_bytes, encoded_public_input_has_canonical_field_limb_bytes,
};
use crate::rv32im::f_prime::{
    build_rv32im_main_recursion_verifier_key_fs_for_step_cap, evaluate_rv32im_main_recursion_f_prime_advice_with_perf,
    rv32im_main_recursion_x_out, Rv32imEncodedPublicInput, Rv32imMainRecursionFPrimeAdvice, Rv32imMainRecursionPhiSide,
    Rv32imVerifierKeyFs, RV32IM_MAIN_RECURSION_TRIVIAL_PC,
};
use crate::rv32im::final_relation::{rv32im_chunk_fold_carry_recursive_accumulator_digest, Rv32imChunkFoldState};
use crate::rv32im::kernel::Rv32imVerifiedKernelChunkHandoff;
use crate::rv32im::SimpleKernelError;

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

pub fn derive_rv32im_ivc_step_cap(
    fold_schedule: FoldSchedule,
    semantic_step_count: usize,
) -> Result<usize, SimpleKernelError> {
    match fold_schedule {
        FoldSchedule::RowsPerChunk(rows) => {
            if rows == 0 {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM native IVC step_cap cannot be derived from RowsPerChunk(0)".into(),
                ));
            }
            Ok(rows)
        }
        FoldSchedule::WholeTrace => {
            if semantic_step_count == 0 {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM native IVC WholeTrace step_cap requires at least one semantic step".into(),
                ));
            }
            Ok(semantic_step_count)
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32imIvcAppendPerf {
    pub validate_state_surface_ms: f64,
    pub validate_relation_surface_ms: f64,
    pub validate_next_relation_surface_ms: f64,
    pub verified_step_statement_ms: f64,
    pub fixed_shape_chunk_summary_ms: f64,
    pub main_circuit_trace_ms: f64,
    pub construction2_pi_fold_ms: f64,
    pub advice_build_ms: f64,
    pub evaluate_f_prime_ms: f64,
    pub evaluate_f_prime_build_nifs_bridge_ms: f64,
    pub evaluate_f_prime_verify_nifs_step_ms: f64,
    pub evaluate_f_prime_verify_chunk_relation_ms: f64,
    pub evaluate_f_prime_verify_derive_next_state_ms: f64,
    pub evaluate_f_prime_build_u_next_ms: f64,
    pub evaluate_f_prime_build_u_next_pack_image_ms: f64,
    pub evaluate_f_prime_build_u_next_commit_ms: f64,
    pub derive_committed_u_next_ms: f64,
    pub finalize_state_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv32imIvcPublicImage {
    pub vk_fs_digest: [u8; 32],
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_0: [u8; 32],
    pub z_i: [u8; 32],
    pub pc: u64,
    pub x_i: Rv32imEncodedPublicInput,
    pub construction2_u_i: Rv32imMainRecursionConstruction2PublicBoundary,
    pub folded_accumulator_digest: [u8; 32],
    pub terminal_bridge_handoff_digest: [u8; 32],
    pub terminal_verified_step_statement_digest: [u8; 32],
    pub terminal_statement: Option<Rv32imChunkStepIvcStatement>,
}

impl Rv32imIvcPublicImage {
    /// Checks the public metadata that the compressed Construction-2 verifier
    /// consumes directly. Witness-bearing CE and F' relations are checked by
    /// the paired Spartan proofs, not by this structural guard.
    pub fn validate_final_construction2_public_boundary(&self) -> Result<(), SimpleKernelError> {
        if self.chunk_count == 0 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image must close at least one recursive step".into(),
            ));
        }
        if self.step_count == 0 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image must close at least one semantic step".into(),
            ));
        }
        if self.pc != RV32IM_MAIN_RECURSION_TRIVIAL_PC {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image pc does not match the trivial recursion lane".into(),
            ));
        }
        if self.construction2_u_i.x_i != self.x_i {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image Construction-2 u_i.x_i does not match x_i".into(),
            ));
        }
        if !encoded_public_input_has_canonical_field_limb_bytes(&self.x_i) {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image x_i is not a canonical four-limb field encoding".into(),
            ));
        }
        for (label, digest) in [
            ("vk_fs_digest", self.vk_fs_digest),
            ("z_0", self.z_0),
            ("z_i", self.z_i),
            (
                "construction2_u_i.fresh_instance_digest",
                self.construction2_u_i.fresh_instance_digest,
            ),
            (
                "construction2_u_i.commitment_digest",
                self.construction2_u_i.commitment_digest,
            ),
            ("folded_accumulator_digest", self.folded_accumulator_digest),
            ("terminal_bridge_handoff_digest", self.terminal_bridge_handoff_digest),
            (
                "terminal_verified_step_statement_digest",
                self.terminal_verified_step_statement_digest,
            ),
        ] {
            if !digest32_has_canonical_field_limb_bytes(digest) {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV32IM IVC compressed public image {label} is not a canonical four-limb field encoding"
                )));
            }
        }
        if !self.construction2_u_i.has_canonical_commitment_shape() {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image Construction-2 u_i commitment shape is not canonical".into(),
            ));
        }
        if self.construction2_u_i.commitment_digest != self.construction2_u_i.expected_commitment_digest() {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image Construction-2 u_i commitment digest does not bind commitment data"
                    .into(),
            ));
        }
        if self.construction2_u_i.fresh_instance_digest != self.construction2_u_i.expected_fresh_instance_digest() {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image Construction-2 u_i digest does not bind commitment and x_i".into(),
            ));
        }
        let terminal_statement = self.terminal_statement.as_ref().ok_or_else(|| {
            SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image must carry the terminal chunk statement metadata".into(),
            )
        })?;
        for (label, digest) in [
            (
                "terminal_statement.step_public.program_digest",
                terminal_statement.step_public.program_digest,
            ),
            (
                "terminal_statement.step_public.state_in",
                terminal_statement.step_public.state_in,
            ),
            (
                "terminal_statement.step_public.state_out",
                terminal_statement.step_public.state_out,
            ),
            (
                "terminal_statement.chunk_summary.public_chunk_digest",
                terminal_statement.chunk_summary.public_chunk_digest,
            ),
            (
                "terminal_statement.chunk_summary.chunk_relation_digest",
                terminal_statement.chunk_summary.chunk_relation_digest,
            ),
        ] {
            if !digest32_has_canonical_field_limb_bytes(digest) {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV32IM IVC compressed public image {label} is not a canonical four-limb field encoding"
                )));
            }
        }
        if terminal_statement.step_public.chunk_index.checked_add(1) != Some(self.chunk_count) {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image terminal chunk index does not close chunk_count".into(),
            ));
        }
        if terminal_statement.step_public.step_hi != self.step_count {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image terminal step_hi does not close step_count".into(),
            ));
        }
        if terminal_statement.step_public.step_lo != terminal_statement.chunk_summary.start_index {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image terminal step_lo does not match chunk summary start".into(),
            ));
        }
        if terminal_statement.chunk_summary.public_step_count == 0 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image terminal chunk must carry at least one public step".into(),
            ));
        }
        let Some(summary_step_hi) = terminal_statement
            .chunk_summary
            .start_index
            .checked_add(terminal_statement.chunk_summary.public_step_count)
        else {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image terminal chunk summary overflows step_count".into(),
            ));
        };
        if summary_step_hi != terminal_statement.step_public.step_hi {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image terminal chunk summary does not close step_hi".into(),
            ));
        }
        if terminal_statement.step_public.state_out != self.z_i {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image terminal state_out does not match z_i".into(),
            ));
        }
        if !terminal_statement.step_public.halted_out {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image terminal chunk must be halted".into(),
            ));
        }
        let expected_terminal_statement_digest =
            build_rv32im_main_recursion_construction2_verified_step_statement_digest_from_step_statement(
                terminal_statement,
            )?;
        if self.terminal_verified_step_statement_digest != expected_terminal_statement_digest {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC compressed public image terminal verified-step digest does not bind terminal metadata"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/ivc_public_image");
        tr.append_message(b"neo.fold.next/rv32im/ivc_public_image/version", b"v4");
        tr.append_message(
            b"neo.fold.next/rv32im/ivc_public_image/vk_fs_digest",
            &self.vk_fs_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv32im/ivc_public_image/meta",
            &[self.chunk_count, self.step_count, self.pc],
        );
        tr.append_message(b"neo.fold.next/rv32im/ivc_public_image/z_0", &self.z_0);
        tr.append_message(b"neo.fold.next/rv32im/ivc_public_image/z_i", &self.z_i);
        tr.append_message(b"neo.fold.next/rv32im/ivc_public_image/x_i", &self.x_i.bytes());
        tr.append_message(
            b"neo.fold.next/rv32im/ivc_public_image/construction2_u_i",
            &self.construction2_u_i.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/rv32im/ivc_public_image/folded_accumulator_digest",
            &self.folded_accumulator_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv32im/ivc_public_image/terminal_bridge_handoff_digest",
            &self.terminal_bridge_handoff_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv32im/ivc_public_image/terminal_verified_step_statement_digest",
            &self.terminal_verified_step_statement_digest,
        );
        match self.terminal_statement.as_ref() {
            Some(statement) => {
                tr.append_u64s(b"neo.fold.next/rv32im/ivc_public_image/has_terminal_statement", &[1]);
                tr.append_message(
                    b"neo.fold.next/rv32im/ivc_public_image/terminal_statement",
                    &statement.expected_digest(),
                );
            }
            None => {
                tr.append_u64s(b"neo.fold.next/rv32im/ivc_public_image/has_terminal_statement", &[0]);
            }
        }
        tr.digest32()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32imIvcState {
    step_cap: u64,
    vk_fs_digest: [u8; 32],
    z_0: [u8; 32],
    z_i: [u8; 32],
    pc: u64,
    chunk_count: u64,
    step_count: u64,
    phi_side: Rv32imMainRecursionPhiSide,
    folded_accumulator_digest: [u8; 32],
    x_i: Rv32imEncodedPublicInput,
    construction2_u_i: Rv32imMainRecursionConstruction2FreshInstance,
    running_state: Rv32imChunkFoldState,
    last_step: Option<Rv32imIvcStepRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Rv32imIvcStepRecord {
    statement: Rv32imChunkStepIvcStatement,
    handoff: Rv32imVerifiedKernelChunkHandoff,
    phi_side: Rv32imMainRecursionPhiSide,
    x_i: Rv32imEncodedPublicInput,
    construction2_u_i: Rv32imMainRecursionConstruction2FreshInstance,
    state_in: Rv32imChunkFoldState,
    state_out: Rv32imChunkFoldState,
    replay_witness: ChunkReplayWitness,
    terminal_step: bool,
}

impl Rv32imIvcState {
    pub fn init_with_step_cap(step_cap: usize) -> Result<Self, SimpleKernelError> {
        Self::init_with_step_cap_and_phi_side(step_cap, Rv32imMainRecursionPhiSide::zero())
    }

    pub fn init_with_step_cap_and_phi_side(
        step_cap: usize,
        phi_side: Rv32imMainRecursionPhiSide,
    ) -> Result<Self, SimpleKernelError> {
        let step_cap = u64::try_from(step_cap)
            .map_err(|_| SimpleKernelError::Bridge("RV32IM native IVC step_cap overflowed u64".into()))?;
        if step_cap == 0 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM native IVC step_cap must be at least one public step".into(),
            ));
        }
        let vk_fs = build_rv32im_main_recursion_verifier_key_fs_for_step_cap(step_cap as usize)?;
        let running_state = rv32im_chunk_step_ivc_initial_state_for_step_cap(step_cap as usize);
        let z_0 = running_state.carry.terminal_handle.0;
        let folded_accumulator_digest = rv32im_chunk_fold_carry_recursive_accumulator_digest(&running_state.carry);
        let x_i = rv32im_main_recursion_x_out(
            &vk_fs,
            0,
            z_0,
            z_0,
            RV32IM_MAIN_RECURSION_TRIVIAL_PC,
            folded_accumulator_digest,
        );
        let full_width =
            crate::rv32im::build_rv32im_main_recursion_construction2_canonical_full_width(&vk_fs, &phi_side)?;
        let default_pair = crate::rv32im::build_rv32im_main_recursion_construction2_default_pair(&vk_fs, full_width)?;
        Ok(Self {
            step_cap,
            vk_fs_digest: vk_fs.expected_digest(),
            z_0,
            z_i: z_0,
            pc: RV32IM_MAIN_RECURSION_TRIVIAL_PC,
            chunk_count: 0,
            step_count: 0,
            phi_side,
            folded_accumulator_digest,
            x_i,
            construction2_u_i: default_pair.u_perp().clone(),
            running_state,
            last_step: None,
        })
    }

    pub fn append(&self, relation: &Rv32imChunkStepIvcRelation) -> Result<Self, SimpleKernelError> {
        Ok(self.append_with_perf(relation)?.0)
    }

    pub fn append_with_perf(
        &self,
        relation: &Rv32imChunkStepIvcRelation,
    ) -> Result<(Self, Rv32imIvcAppendPerf), SimpleKernelError> {
        let total_started = Instant::now();
        let mut perf = Rv32imIvcAppendPerf::default();

        let started = Instant::now();
        let (vk_fs, folded_accumulator_in_digest) = self.append_surface()?;
        perf.validate_state_surface_ms = elapsed_ms(started);

        let started = Instant::now();
        validate_rv32im_chunk_step_ivc_surface(&relation.statement, &relation.witness)?;
        perf.validate_relation_surface_ms = elapsed_ms(started);

        let started = Instant::now();
        self.validate_next_relation_surface(relation)?;
        perf.validate_next_relation_surface_ms = elapsed_ms(started);

        let started = Instant::now();
        let main_circuit_chunk_summary = relation.statement.chunk_summary.clone();
        perf.fixed_shape_chunk_summary_ms = elapsed_ms(started);

        let started = Instant::now();
        let native_verified_step_statement =
            build_rv32im_main_recursion_construction2_verified_step_statement_from_summary(
                relation.witness.handoff.bridge_handoff.chunk_index,
                relation.witness.terminal_step,
                &main_circuit_chunk_summary,
                &relation.witness.state_in,
                &relation.witness.state_out,
            );
        perf.verified_step_statement_ms = elapsed_ms(started);

        let started = Instant::now();
        let construction2_pi_fold =
            build_rv32im_main_recursion_construction2_pi_fold_from_replay_witness(&relation.witness.replay_witness);
        perf.construction2_pi_fold_ms = elapsed_ms(started);

        let started = Instant::now();
        let advice = Rv32imMainRecursionFPrimeAdvice::from_parts_with_folded_accumulator_in_digest(
            vk_fs,
            self.chunk_count,
            self.z_0,
            self.z_i,
            self.pc,
            crate::rv32im::Rv32imMainRecursionSideLaneWitness::zero(),
            self.phi_side.clone(),
            self.running_state.clone(),
            folded_accumulator_in_digest,
            self.x_i.clone(),
            Some(self.construction2_u_i.clone()),
            native_verified_step_statement,
            relation.witness.terminal_step,
            relation.witness.handoff.clone(),
            relation.witness.state_out.clone(),
            relation.witness.replay_witness.clone(),
            construction2_pi_fold,
        )?;
        perf.advice_build_ms = elapsed_ms(started);

        let started = Instant::now();
        let (step_image, eval_perf) = evaluate_rv32im_main_recursion_f_prime_advice_with_perf(&advice)?;
        let step_image = step_image.into_parts();
        perf.evaluate_f_prime_ms = elapsed_ms(started);
        perf.evaluate_f_prime_build_nifs_bridge_ms = eval_perf.build_nifs_bridge_ms;
        perf.evaluate_f_prime_verify_nifs_step_ms = eval_perf.verify_nifs_step_ms;
        perf.evaluate_f_prime_verify_chunk_relation_ms = eval_perf.verify_nifs_chunk_relation_ms;
        perf.evaluate_f_prime_verify_derive_next_state_ms = eval_perf.verify_nifs_derive_next_state_ms;
        perf.evaluate_f_prime_build_u_next_ms = eval_perf.build_construction2_u_next_ms;
        perf.evaluate_f_prime_build_u_next_pack_image_ms = eval_perf.build_construction2_u_next_pack_image_ms;
        perf.evaluate_f_prime_build_u_next_commit_ms = eval_perf.build_construction2_u_next_commit_ms;

        let started = Instant::now();
        let next_step_count = self
            .step_count
            .checked_add(relation.statement.chunk_summary.public_step_count)
            .ok_or_else(|| SimpleKernelError::Bridge("RV32IM IVC step_count overflowed during append".into()))?;
        if next_step_count != relation.statement.step_public.step_hi {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC append step_count does not match the appended relation step_hi".into(),
            ));
        }
        let step_record = Rv32imIvcStepRecord {
            statement: relation.statement.clone(),
            handoff: relation.witness.handoff.clone(),
            phi_side: self.phi_side.clone(),
            x_i: self.x_i.clone(),
            construction2_u_i: self.construction2_u_i.clone(),
            state_in: relation.witness.state_in.clone(),
            state_out: step_image.next_state.clone(),
            replay_witness: relation.witness.replay_witness.clone(),
            terminal_step: relation.witness.terminal_step,
        };
        let next = Self {
            step_cap: self.step_cap,
            vk_fs_digest: self.vk_fs_digest,
            z_0: self.z_0,
            z_i: step_image.z_next,
            pc: step_image.pc_next,
            chunk_count: step_image.chunk_count,
            step_count: next_step_count,
            phi_side: step_image.phi_side,
            folded_accumulator_digest: step_image.folded_accumulator_digest,
            x_i: step_image.x_out,
            construction2_u_i: step_image.construction2_u_next,
            running_state: step_image.next_state,
            last_step: Some(step_record),
        };
        perf.finalize_state_ms = elapsed_ms(started);
        perf.total_ms = elapsed_ms(total_started);
        Ok((next, perf))
    }

    pub fn public_image(&self) -> Rv32imIvcPublicImage {
        Rv32imIvcPublicImage {
            vk_fs_digest: self.vk_fs_digest,
            chunk_count: self.chunk_count,
            step_count: self.step_count,
            z_0: self.z_0,
            z_i: self.z_i,
            pc: self.pc,
            x_i: self.x_i.clone(),
            construction2_u_i: Rv32imMainRecursionConstruction2PublicBoundary::from_fresh_instance(
                &self.construction2_u_i,
            ),
            folded_accumulator_digest: self.folded_accumulator_digest(),
            terminal_bridge_handoff_digest: self.terminal_bridge_handoff_digest(),
            terminal_verified_step_statement_digest: self.terminal_verified_step_statement_digest(),
            terminal_statement: self.last_step.as_ref().map(|step| step.statement.clone()),
        }
    }

    pub fn chunk_count(&self) -> u64 {
        self.chunk_count
    }

    pub fn step_cap(&self) -> u64 {
        self.step_cap
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn z_0(&self) -> [u8; 32] {
        self.z_0
    }

    pub fn z_i(&self) -> [u8; 32] {
        self.z_i
    }

    pub fn x_i(&self) -> &Rv32imEncodedPublicInput {
        &self.x_i
    }

    pub fn construction2_u_i(&self) -> &Rv32imMainRecursionConstruction2FreshInstance {
        &self.construction2_u_i
    }

    pub fn running_state(&self) -> &Rv32imChunkFoldState {
        &self.running_state
    }

    pub fn latest_terminal_statement(&self) -> Option<&Rv32imChunkStepIvcStatement> {
        self.last_step.as_ref().map(|step| &step.statement)
    }

    pub(crate) fn latest_relation_and_advice(
        &self,
    ) -> Result<(Rv32imChunkStepIvcRelation, Rv32imMainRecursionFPrimeAdvice), SimpleKernelError> {
        let vk_fs = self.canonical_vk_fs()?;
        let last_step = self.last_step.as_ref().ok_or_else(|| {
            SimpleKernelError::Bridge("RV32IM IVC compression requires at least one appended recursive step".into())
        })?;
        Ok((last_step.relation()?, last_step.advice(vk_fs, self.z_0)?))
    }

    pub(crate) fn validate_current_surface_for_compression(&self) -> Result<(), SimpleKernelError> {
        self.validate_surface()
    }

    pub(crate) fn terminal_bridge_handoff_digest(&self) -> [u8; 32] {
        self.last_step
            .as_ref()
            .map(|step| step.handoff.bridge_handoff.digest)
            .unwrap_or([0u8; 32])
    }

    pub(crate) fn terminal_verified_step_statement_digest(&self) -> [u8; 32] {
        self.last_step
            .as_ref()
            .and_then(|step| {
                build_rv32im_main_recursion_construction2_verified_step_statement_digest_from_step_statement(
                    &step.statement,
                )
                .ok()
            })
            .unwrap_or([0u8; 32])
    }

    fn canonical_vk_fs(&self) -> Result<crate::rv32im::Rv32imVerifierKeyFs, SimpleKernelError> {
        let vk_fs = build_rv32im_main_recursion_verifier_key_fs_for_step_cap(self.step_cap_usize()?)?;
        if vk_fs.expected_digest() != self.vk_fs_digest {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC state verifier-key digest does not match the canonical current F' structure".into(),
            ));
        }
        Ok(vk_fs)
    }

    fn folded_accumulator_digest(&self) -> [u8; 32] {
        self.folded_accumulator_digest
    }

    fn recomputed_folded_accumulator_digest(&self) -> [u8; 32] {
        rv32im_chunk_fold_carry_recursive_accumulator_digest(&self.running_state.carry)
    }

    fn append_surface(&self) -> Result<(Rv32imVerifierKeyFs, [u8; 32]), SimpleKernelError> {
        let vk_fs = self.canonical_vk_fs()?;
        let folded_accumulator_digest = self.folded_accumulator_digest();
        if self.running_state.transcript.absorbed > neo_params::poseidon2_goldilocks::RATE {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC transcript snapshot absorbed count exceeds the Poseidon2 rate".into(),
            ));
        }
        if self.z_0
            != rv32im_chunk_step_ivc_initial_state_for_step_cap(self.step_cap_usize()?)
                .carry
                .terminal_handle
                .0
        {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC z_0 does not match the canonical seed terminal handle".into(),
            ));
        }
        if self.z_i != self.running_state.carry.terminal_handle.0 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC z_i does not match the carried terminal handle".into(),
            ));
        }
        if self.pc != RV32IM_MAIN_RECURSION_TRIVIAL_PC {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC pc does not match the trivial recursion control lane".into(),
            ));
        }
        if self.step_cap_usize()? == 0 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC state carries a zero native step_cap".into(),
            ));
        }
        if self.running_state.carry.main.claims.len() != self.running_state.carry.main.witnesses.len() {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC carried CE claim and witness counts diverged".into(),
            ));
        }
        self.running_state
            .carry
            .validate_projection_digests("running_state")?;
        let expected_x_i = rv32im_main_recursion_x_out(
            &vk_fs,
            self.chunk_count,
            self.z_0,
            self.z_i,
            self.pc,
            folded_accumulator_digest,
        );
        if self.x_i != expected_x_i {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC x_i does not match the carried native public image".into(),
            ));
        }
        if self.chunk_count == 0 {
            if self.step_count != 0 {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM IVC base state cannot carry a non-zero step count".into(),
                ));
            }
            if self.last_step.is_some() {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM IVC base state cannot carry a terminal step record".into(),
                ));
            }
            let full_width =
                crate::rv32im::build_rv32im_main_recursion_construction2_canonical_full_width(&vk_fs, &self.phi_side)?;
            let default_pair =
                crate::rv32im::build_rv32im_main_recursion_construction2_default_pair(&vk_fs, full_width)?;
            if self.construction2_u_i != *default_pair.u_perp() {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM IVC base state does not carry the canonical Construction-2 default pair".into(),
                ));
            }
        } else if self.construction2_u_i.x_i() != &self.x_i {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC carried Construction-2 fresh instance does not bind to the carried x_i".into(),
            ));
        }
        Ok((vk_fs, folded_accumulator_digest))
    }

    fn validated_surface(&self) -> Result<(Rv32imVerifierKeyFs, [u8; 32]), SimpleKernelError> {
        let (vk_fs, folded_accumulator_digest) = self.append_surface()?;
        let recomputed_folded_accumulator_digest = self.recomputed_folded_accumulator_digest();
        if folded_accumulator_digest != recomputed_folded_accumulator_digest {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC folded accumulator digest does not match the carried running state".into(),
            ));
        }
        if self.chunk_count == 0 {
        } else {
            let last_step = self.last_step.as_ref().ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV32IM IVC non-base prover state must carry the latest appended relation".into(),
                )
            })?;
            if !rv32im_chunk_fold_states_match(&last_step.state_out, &self.running_state) {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM IVC latest step output does not match the carried running state".into(),
                ));
            }
            self.validate_relation_step_cap(
                last_step.statement.chunk_summary.public_step_count,
                last_step.terminal_step,
            )?;
            if last_step.statement.step_public.chunk_index + 1 != self.chunk_count {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM IVC chunk_count does not match the latest terminal statement chunk index".into(),
                ));
            }
            if last_step.statement.step_public.step_hi != self.step_count {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM IVC step_count does not match the latest terminal statement step_hi".into(),
                ));
            }
            if last_step.statement.step_public.state_out != self.z_i {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM IVC latest terminal statement state_out does not match z_i".into(),
                ));
            }
        }
        Ok((vk_fs, folded_accumulator_digest))
    }

    fn validate_surface(&self) -> Result<(), SimpleKernelError> {
        let _ = self.validated_surface()?;
        Ok(())
    }

    fn validate_next_relation_surface(&self, relation: &Rv32imChunkStepIvcRelation) -> Result<(), SimpleKernelError> {
        if relation.witness.state_in.carry.main.claims != self.running_state.carry.main.claims
            || relation.witness.state_in.carry.main.witnesses != self.running_state.carry.main.witnesses
            || relation.witness.state_in.carry.terminal_handle != self.running_state.carry.terminal_handle
            || relation.witness.state_in.transcript != self.running_state.transcript
        {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC append requires the next relation to start from the carried running state".into(),
            ));
        }
        if relation.statement.step_public.chunk_index != self.chunk_count {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC append chunk index does not match the carried chunk count".into(),
            ));
        }
        if relation.statement.step_public.step_lo != self.step_count {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC append step_lo does not match the carried semantic step count".into(),
            ));
        }
        if relation.statement.step_public.state_in != self.z_i {
            return Err(SimpleKernelError::Bridge(
                "RV32IM IVC append step_public.state_in does not match the carried z_i".into(),
            ));
        }
        if let Some(last_step) = self.last_step.as_ref() {
            if relation.statement.step_public.program_digest != last_step.statement.step_public.program_digest {
                return Err(SimpleKernelError::Bridge(
                    "RV32IM IVC append program digest drifted across carried folds".into(),
                ));
            }
        }
        self.validate_relation_step_cap(
            relation.statement.chunk_summary.public_step_count,
            relation.witness.terminal_step,
        )?;
        Ok(())
    }

    fn step_cap_usize(&self) -> Result<usize, SimpleKernelError> {
        usize::try_from(self.step_cap).map_err(|_| {
            SimpleKernelError::Bridge(
                "RV32IM IVC state step_cap does not fit into the local native recursion width".into(),
            )
        })
    }

    fn validate_relation_step_cap(&self, public_step_count: u64, terminal_step: bool) -> Result<(), SimpleKernelError> {
        let active_step_count = usize::try_from(public_step_count).map_err(|_| {
            SimpleKernelError::Bridge(
                "RV32IM IVC relation public_step_count does not fit into the local native step-cap model".into(),
            )
        })?;
        let step_cap = self.step_cap_usize()?;
        if active_step_count == 0 || active_step_count > step_cap {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM IVC relation carries {active_step_count} public steps outside the frozen native step_cap={step_cap}"
            )));
        }
        if !terminal_step && active_step_count != step_cap {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM IVC non-terminal relation must carry exactly step_cap={step_cap} public steps; got {active_step_count}"
            )));
        }
        Ok(())
    }
}

impl Rv32imIvcStepRecord {
    fn relation(&self) -> Result<Rv32imChunkStepIvcRelation, SimpleKernelError> {
        let witness = Rv32imChunkStepIvcWitness {
            handoff: self.handoff.clone(),
            state_in: self.state_in.clone(),
            state_out: self.state_out.clone(),
            replay_witness: self.replay_witness.clone(),
            terminal_step: self.terminal_step,
        };
        validate_rv32im_chunk_step_ivc_surface(&self.statement, &witness)?;
        Ok(Rv32imChunkStepIvcRelation {
            statement: self.statement.clone(),
            witness,
        })
    }

    fn advice(
        &self,
        vk_fs: Rv32imVerifierKeyFs,
        z_0: [u8; 32],
    ) -> Result<Rv32imMainRecursionFPrimeAdvice, SimpleKernelError> {
        let native_verified_step_statement =
            build_rv32im_main_recursion_construction2_verified_step_statement_from_summary(
                self.handoff.bridge_handoff.chunk_index,
                self.terminal_step,
                &self.statement.chunk_summary,
                &self.state_in,
                &self.state_out,
            );
        let construction2_pi_fold =
            build_rv32im_main_recursion_construction2_pi_fold_from_replay_witness(&self.replay_witness);
        Rv32imMainRecursionFPrimeAdvice::from_parts(
            vk_fs,
            self.statement.step_public.chunk_index,
            z_0,
            self.statement.step_public.state_in,
            RV32IM_MAIN_RECURSION_TRIVIAL_PC,
            crate::rv32im::Rv32imMainRecursionSideLaneWitness::zero(),
            self.phi_side.clone(),
            self.state_in.clone(),
            self.x_i.clone(),
            Some(self.construction2_u_i.clone()),
            native_verified_step_statement,
            self.terminal_step,
            self.handoff.clone(),
            self.state_out.clone(),
            self.replay_witness.clone(),
            construction2_pi_fold,
        )
    }
}

fn rv32im_chunk_fold_states_match(lhs: &Rv32imChunkFoldState, rhs: &Rv32imChunkFoldState) -> bool {
    lhs.carry.main.claims == rhs.carry.main.claims
        && lhs.carry.main.witnesses == rhs.carry.main.witnesses
        && lhs.carry.main_projection_digests == rhs.carry.main_projection_digests
        && lhs.carry.terminal_handle == rhs.carry.terminal_handle
        && lhs.transcript == rhs.transcript
}

pub(crate) fn build_rv32im_ivc_prover_state_from_relations(
    relations: &[Rv32imChunkStepIvcRelation],
    step_cap: usize,
) -> Result<Rv32imIvcState, SimpleKernelError> {
    let mut state = Rv32imIvcState::init_with_step_cap(step_cap)?;
    for relation in relations {
        state = state.append(relation)?;
    }
    Ok(state)
}
