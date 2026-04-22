//! Owns the native RV64IM IVC carrier that can be serialized, resumed, and
//! appended without Spartan.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::chunk_relation::ChunkReplayWitness;
use crate::proof::FoldSchedule;
use crate::rv64im::chunk_fold_step::verify_rv64im_chunk_fold_verifier_step;
use crate::rv64im::chunk_step_ivc::{
    rv64im_chunk_step_ivc_initial_state_for_step_cap, validate_rv64im_chunk_step_ivc_surface,
    Rv64imChunkStepIvcRelation, Rv64imChunkStepIvcStatement, Rv64imChunkStepIvcWitness,
};
use crate::rv64im::construction2::{
    build_rv64im_main_recursion_construction2_pi_fold_from_replay_witness,
    build_rv64im_main_recursion_construction2_verified_step_statement_from_summary,
    Rv64imMainRecursionConstruction2FreshInstance,
};
use crate::rv64im::f_prime::{
    build_rv64im_main_recursion_verifier_key_fs_for_step_cap, evaluate_rv64im_main_recursion_f_prime_advice,
    rv64im_main_recursion_x_out, Rv64imEncodedPublicInput, Rv64imMainRecursionFPrimeAdvice, Rv64imMainRecursionPhiSide,
    Rv64imVerifierKeyFs, RV64IM_MAIN_RECURSION_TRIVIAL_PC,
};
use crate::rv64im::final_relation::{rv64im_chunk_fold_carry_recursive_accumulator_digest, Rv64imChunkFoldState};
use crate::rv64im::kernel::{
    rv64im_cached_root_main_lane_optimized_cache, rv64im_root_main_lane_context_for_claim_count,
    Rv64imVerifiedKernelChunkHandoff,
};
use crate::rv64im::SimpleKernelError;

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

pub fn derive_rv64im_ivc_step_cap(
    fold_schedule: FoldSchedule,
    semantic_step_count: usize,
) -> Result<usize, SimpleKernelError> {
    match fold_schedule {
        FoldSchedule::RowsPerChunk(rows) => {
            if rows == 0 {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM native IVC step_cap cannot be derived from RowsPerChunk(0)".into(),
                ));
            }
            Ok(rows)
        }
        FoldSchedule::WholeTrace => {
            if semantic_step_count == 0 {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM native IVC WholeTrace step_cap requires at least one semantic step".into(),
                ));
            }
            Ok(semantic_step_count)
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv64imIvcAppendPerf {
    pub validate_state_surface_ms: f64,
    pub validate_relation_surface_ms: f64,
    pub validate_next_relation_surface_ms: f64,
    pub verified_step_statement_ms: f64,
    pub fixed_shape_chunk_summary_ms: f64,
    pub main_circuit_trace_ms: f64,
    pub construction2_pi_fold_ms: f64,
    pub advice_build_ms: f64,
    pub evaluate_f_prime_ms: f64,
    pub finalize_state_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv64imIvcVerifyPerf {
    pub validate_state_surface_ms: f64,
    pub build_terminal_relation_ms: f64,
    pub verified_step_statement_ms: f64,
    pub context_lookup_ms: f64,
    pub replay_step_ms: f64,
    pub compare_running_state_ms: f64,
    pub transcript_snapshot_ms: f64,
    pub compare_step_public_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imIvcPublicImage {
    pub vk_fs_digest: [u8; 32],
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_0: [u8; 32],
    pub z_i: [u8; 32],
    pub pc: u64,
    pub x_i: Rv64imEncodedPublicInput,
    pub folded_accumulator_digest: [u8; 32],
    pub terminal_statement: Option<Rv64imChunkStepIvcStatement>,
}

impl Rv64imIvcPublicImage {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/ivc_public_image");
        tr.append_message(b"neo.fold.next/rv64im/ivc_public_image/version", b"v1");
        tr.append_message(
            b"neo.fold.next/rv64im/ivc_public_image/vk_fs_digest",
            &self.vk_fs_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/ivc_public_image/meta",
            &[self.chunk_count, self.step_count, self.pc],
        );
        tr.append_message(b"neo.fold.next/rv64im/ivc_public_image/z_0", &self.z_0);
        tr.append_message(b"neo.fold.next/rv64im/ivc_public_image/z_i", &self.z_i);
        tr.append_message(b"neo.fold.next/rv64im/ivc_public_image/x_i", &self.x_i.bytes());
        tr.append_message(
            b"neo.fold.next/rv64im/ivc_public_image/folded_accumulator_digest",
            &self.folded_accumulator_digest,
        );
        match self.terminal_statement.as_ref() {
            Some(statement) => {
                tr.append_u64s(b"neo.fold.next/rv64im/ivc_public_image/has_terminal_statement", &[1]);
                tr.append_message(
                    b"neo.fold.next/rv64im/ivc_public_image/terminal_statement",
                    &statement.expected_digest(),
                );
            }
            None => {
                tr.append_u64s(b"neo.fold.next/rv64im/ivc_public_image/has_terminal_statement", &[0]);
            }
        }
        tr.digest32()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imIvcState {
    step_cap: u64,
    vk_fs_digest: [u8; 32],
    z_0: [u8; 32],
    z_i: [u8; 32],
    pc: u64,
    chunk_count: u64,
    step_count: u64,
    phi_side: Rv64imMainRecursionPhiSide,
    folded_accumulator_digest: [u8; 32],
    x_i: Rv64imEncodedPublicInput,
    construction2_u_i: Rv64imMainRecursionConstruction2FreshInstance,
    running_state: Rv64imChunkFoldState,
    last_step: Option<Rv64imIvcStepRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Rv64imIvcStepRecord {
    statement: Rv64imChunkStepIvcStatement,
    handoff: Rv64imVerifiedKernelChunkHandoff,
    state_in: Rv64imChunkFoldState,
    replay_witness: ChunkReplayWitness,
    terminal_step: bool,
}

impl Rv64imIvcState {
    pub fn init_with_step_cap(step_cap: usize) -> Result<Self, SimpleKernelError> {
        Self::init_with_step_cap_and_phi_side(step_cap, Rv64imMainRecursionPhiSide::zero())
    }

    pub fn init_with_step_cap_and_phi_side(
        step_cap: usize,
        phi_side: Rv64imMainRecursionPhiSide,
    ) -> Result<Self, SimpleKernelError> {
        let step_cap = u64::try_from(step_cap)
            .map_err(|_| SimpleKernelError::Bridge("RV64IM native IVC step_cap overflowed u64".into()))?;
        if step_cap == 0 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM native IVC step_cap must be at least one public step".into(),
            ));
        }
        let vk_fs = build_rv64im_main_recursion_verifier_key_fs_for_step_cap(step_cap as usize)?;
        let running_state = rv64im_chunk_step_ivc_initial_state_for_step_cap(step_cap as usize);
        let z_0 = running_state.carry.terminal_handle.0;
        let folded_accumulator_digest = rv64im_chunk_fold_carry_recursive_accumulator_digest(&running_state.carry);
        let x_i = rv64im_main_recursion_x_out(
            &vk_fs,
            0,
            z_0,
            z_0,
            RV64IM_MAIN_RECURSION_TRIVIAL_PC,
            folded_accumulator_digest,
        );
        let full_width =
            crate::rv64im::build_rv64im_main_recursion_construction2_canonical_full_width(&vk_fs, &phi_side)?;
        let default_pair = crate::rv64im::build_rv64im_main_recursion_construction2_default_pair(&vk_fs, full_width)?;
        Ok(Self {
            step_cap,
            vk_fs_digest: vk_fs.expected_digest(),
            z_0,
            z_i: z_0,
            pc: RV64IM_MAIN_RECURSION_TRIVIAL_PC,
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

    pub fn append(&self, relation: &Rv64imChunkStepIvcRelation) -> Result<Self, SimpleKernelError> {
        Ok(self.append_with_perf(relation)?.0)
    }

    pub fn append_with_perf(
        &self,
        relation: &Rv64imChunkStepIvcRelation,
    ) -> Result<(Self, Rv64imIvcAppendPerf), SimpleKernelError> {
        let total_started = Instant::now();
        let mut perf = Rv64imIvcAppendPerf::default();

        let started = Instant::now();
        let (vk_fs, folded_accumulator_in_digest) = self.append_surface()?;
        perf.validate_state_surface_ms = elapsed_ms(started);

        let started = Instant::now();
        validate_rv64im_chunk_step_ivc_surface(&relation.statement, &relation.witness)?;
        perf.validate_relation_surface_ms = elapsed_ms(started);

        let started = Instant::now();
        self.validate_next_relation_surface(relation)?;
        perf.validate_next_relation_surface_ms = elapsed_ms(started);

        let started = Instant::now();
        let main_circuit_chunk_summary = relation.statement.chunk_summary.clone();
        perf.fixed_shape_chunk_summary_ms = elapsed_ms(started);

        let started = Instant::now();
        let native_verified_step_statement =
            build_rv64im_main_recursion_construction2_verified_step_statement_from_summary(
                relation.witness.handoff.bridge_handoff.chunk_index,
                &main_circuit_chunk_summary,
                &relation.witness.state_in,
                &relation.witness.state_out,
            );
        perf.verified_step_statement_ms = elapsed_ms(started);

        let started = Instant::now();
        let construction2_pi_fold =
            build_rv64im_main_recursion_construction2_pi_fold_from_replay_witness(&relation.witness.replay_witness);
        perf.construction2_pi_fold_ms = elapsed_ms(started);

        let started = Instant::now();
        let advice = Rv64imMainRecursionFPrimeAdvice::from_parts_with_folded_accumulator_in_digest(
            vk_fs,
            self.chunk_count,
            self.z_0,
            self.z_i,
            self.pc,
            crate::rv64im::Rv64imMainRecursionSideLaneWitness::zero(),
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
        let step_image = evaluate_rv64im_main_recursion_f_prime_advice(&advice)?.into_parts();
        perf.evaluate_f_prime_ms = elapsed_ms(started);

        let started = Instant::now();
        let next_step_count = self
            .step_count
            .checked_add(relation.statement.chunk_summary.public_step_count)
            .ok_or_else(|| SimpleKernelError::Bridge("RV64IM IVC step_count overflowed during append".into()))?;
        if next_step_count != relation.statement.step_public.step_hi {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC append step_count does not match the appended relation step_hi".into(),
            ));
        }
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
            last_step: Some(Rv64imIvcStepRecord {
                statement: relation.statement.clone(),
                handoff: relation.witness.handoff.clone(),
                state_in: relation.witness.state_in.clone(),
                replay_witness: relation.witness.replay_witness.clone(),
                terminal_step: relation.witness.terminal_step,
            }),
        };
        perf.finalize_state_ms = elapsed_ms(started);
        perf.total_ms = elapsed_ms(total_started);
        Ok((next, perf))
    }

    pub fn verify(&self) -> Result<(), SimpleKernelError> {
        self.verify_with_perf().map(|_| ())
    }

    pub fn verify_with_perf(&self) -> Result<Rv64imIvcVerifyPerf, SimpleKernelError> {
        let total_started = Instant::now();
        let mut perf = Rv64imIvcVerifyPerf::default();

        let started = Instant::now();
        self.validate_surface()?;
        perf.validate_state_surface_ms = elapsed_ms(started);

        if let Some(last_step) = self.last_step.as_ref() {
            let started = Instant::now();
            let witness = Rv64imChunkStepIvcWitness {
                handoff: last_step.handoff.clone(),
                state_in: last_step.state_in.clone(),
                state_out: self.running_state.clone(),
                replay_witness: last_step.replay_witness.clone(),
                terminal_step: last_step.terminal_step,
            };
            validate_rv64im_chunk_step_ivc_surface(&last_step.statement, &witness)?;
            perf.build_terminal_relation_ms = elapsed_ms(started);

            let started = Instant::now();
            let (params, log, structure) =
                rv64im_root_main_lane_context_for_claim_count(last_step.state_in.carry.main.claims.len())?;
            let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()?;
            perf.context_lookup_ms = elapsed_ms(started);

            let started = Instant::now();
            let mut transcript = Poseidon2Transcript::from_state_and_absorbed(
                last_step.state_in.transcript.state,
                last_step.state_in.transcript.absorbed,
            );
            let step = verify_rv64im_chunk_fold_verifier_step(
                last_step.statement.step_public.program_digest,
                last_step.statement.step_public.chunk_index as usize,
                last_step.terminal_step,
                &last_step.handoff,
                &last_step.state_in.carry,
                &last_step.replay_witness,
                &mut transcript,
                &params,
                structure,
                log,
                &optimized_cache,
            )?;
            perf.replay_step_ms = elapsed_ms(started);

            let started = Instant::now();
            if step.public_chunk_digest != last_step.statement.chunk_summary.public_chunk_digest {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC verify replayed terminal relation does not reproduce the carried public chunk digest"
                        .into(),
                ));
            }
            if step.chunk_relation_digest != last_step.statement.chunk_summary.chunk_relation_digest {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC verify replayed terminal relation does not reproduce the carried chunk relation digest"
                        .into(),
                ));
            }
            perf.verified_step_statement_ms = elapsed_ms(started);

            let started = Instant::now();
            if step.next_carry.main.claims != self.running_state.carry.main.claims
                || step.next_carry.main.witnesses != self.running_state.carry.main.witnesses
                || step.next_carry.terminal_handle != self.running_state.carry.terminal_handle
            {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC verify replayed terminal relation does not produce the carried running state".into(),
                ));
            }
            perf.compare_running_state_ms = elapsed_ms(started);

            let started = Instant::now();
            let transcript_out = crate::rv64im::final_relation::rv64im_chunk_fold_carried_transcript_snapshot(
                &crate::rv64im::final_relation::Rv64imChunkFoldTranscriptSnapshot {
                    state: transcript.state(),
                    absorbed: transcript.absorbed(),
                },
            );
            if transcript_out != self.running_state.transcript {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC verify replayed terminal relation does not produce the carried transcript snapshot"
                        .into(),
                ));
            }
            perf.transcript_snapshot_ms = elapsed_ms(started);

            let started = Instant::now();
            if step.step_public != last_step.statement.step_public {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC verify replayed terminal relation does not reproduce the carried step public".into(),
                ));
            }
            perf.compare_step_public_ms = elapsed_ms(started);
        }

        perf.total_ms = elapsed_ms(total_started);
        Ok(perf)
    }

    pub fn public_image(&self) -> Rv64imIvcPublicImage {
        Rv64imIvcPublicImage {
            vk_fs_digest: self.vk_fs_digest,
            chunk_count: self.chunk_count,
            step_count: self.step_count,
            z_0: self.z_0,
            z_i: self.z_i,
            pc: self.pc,
            x_i: self.x_i.clone(),
            folded_accumulator_digest: self.folded_accumulator_digest(),
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

    pub fn x_i(&self) -> &Rv64imEncodedPublicInput {
        &self.x_i
    }

    pub fn construction2_u_i(&self) -> &Rv64imMainRecursionConstruction2FreshInstance {
        &self.construction2_u_i
    }

    pub fn running_state(&self) -> &Rv64imChunkFoldState {
        &self.running_state
    }

    pub fn latest_terminal_statement(&self) -> Option<&Rv64imChunkStepIvcStatement> {
        self.last_step.as_ref().map(|step| &step.statement)
    }

    pub(crate) fn build_terminal_relation(&self) -> Result<Rv64imChunkStepIvcRelation, SimpleKernelError> {
        let last_step = self.last_step.as_ref().ok_or_else(|| {
            SimpleKernelError::Bridge("RV64IM IVC compression requires at least one appended fold".into())
        })?;
        let witness = Rv64imChunkStepIvcWitness {
            handoff: last_step.handoff.clone(),
            state_in: last_step.state_in.clone(),
            state_out: self.running_state.clone(),
            replay_witness: last_step.replay_witness.clone(),
            terminal_step: last_step.terminal_step,
        };
        validate_rv64im_chunk_step_ivc_surface(&last_step.statement, &witness)?;
        Ok(Rv64imChunkStepIvcRelation {
            statement: last_step.statement.clone(),
            witness,
        })
    }

    fn canonical_vk_fs(&self) -> Result<crate::rv64im::Rv64imVerifierKeyFs, SimpleKernelError> {
        let vk_fs = build_rv64im_main_recursion_verifier_key_fs_for_step_cap(self.step_cap_usize()?)?;
        if vk_fs.expected_digest() != self.vk_fs_digest {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC state verifier-key digest does not match the canonical current F' structure".into(),
            ));
        }
        Ok(vk_fs)
    }

    fn folded_accumulator_digest(&self) -> [u8; 32] {
        self.folded_accumulator_digest
    }

    fn recomputed_folded_accumulator_digest(&self) -> [u8; 32] {
        rv64im_chunk_fold_carry_recursive_accumulator_digest(&self.running_state.carry)
    }

    fn append_surface(&self) -> Result<(Rv64imVerifierKeyFs, [u8; 32]), SimpleKernelError> {
        let vk_fs = self.canonical_vk_fs()?;
        let folded_accumulator_digest = self.folded_accumulator_digest();
        if self.running_state.transcript.absorbed > neo_params::poseidon2_goldilocks::RATE {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC transcript snapshot absorbed count exceeds the Poseidon2 rate".into(),
            ));
        }
        if self.z_0
            != rv64im_chunk_step_ivc_initial_state_for_step_cap(self.step_cap_usize()?)
                .carry
                .terminal_handle
                .0
        {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC z_0 does not match the canonical seed terminal handle".into(),
            ));
        }
        if self.z_i != self.running_state.carry.terminal_handle.0 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC z_i does not match the carried terminal handle".into(),
            ));
        }
        if self.pc != RV64IM_MAIN_RECURSION_TRIVIAL_PC {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC pc does not match the trivial recursion control lane".into(),
            ));
        }
        if self.step_cap_usize()? == 0 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC state carries a zero native step_cap".into(),
            ));
        }
        if self.running_state.carry.main.claims.len() != self.running_state.carry.main.witnesses.len() {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC carried CE claim and witness counts diverged".into(),
            ));
        }
        let expected_x_i = rv64im_main_recursion_x_out(
            &vk_fs,
            self.chunk_count,
            self.z_0,
            self.z_i,
            self.pc,
            folded_accumulator_digest,
        );
        if self.x_i != expected_x_i {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC x_i does not match the carried native public image".into(),
            ));
        }
        if self.chunk_count == 0 {
            if self.step_count != 0 {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC base state cannot carry a non-zero step count".into(),
                ));
            }
            if self.last_step.is_some() {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC base state cannot carry a terminal step record".into(),
                ));
            }
            let full_width =
                crate::rv64im::build_rv64im_main_recursion_construction2_canonical_full_width(&vk_fs, &self.phi_side)?;
            let default_pair =
                crate::rv64im::build_rv64im_main_recursion_construction2_default_pair(&vk_fs, full_width)?;
            if self.construction2_u_i != *default_pair.u_perp() {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC base state does not carry the canonical Construction-2 default pair".into(),
                ));
            }
        } else if self.construction2_u_i.x_i() != &self.x_i {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC carried Construction-2 fresh instance does not bind to the carried x_i".into(),
            ));
        }
        Ok((vk_fs, folded_accumulator_digest))
    }

    fn validated_surface(&self) -> Result<(Rv64imVerifierKeyFs, [u8; 32]), SimpleKernelError> {
        let (vk_fs, folded_accumulator_digest) = self.append_surface()?;
        let recomputed_folded_accumulator_digest = self.recomputed_folded_accumulator_digest();
        if folded_accumulator_digest != recomputed_folded_accumulator_digest {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC folded accumulator digest does not match the carried running state".into(),
            ));
        }
        if self.chunk_count == 0 {
        } else {
            let last_step = self.last_step.as_ref().ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV64IM IVC non-base state must carry the latest relation needed for native verification".into(),
                )
            })?;
            self.validate_relation_step_cap(
                last_step.statement.chunk_summary.public_step_count,
                last_step.terminal_step,
            )?;
            if last_step.statement.step_public.chunk_index + 1 != self.chunk_count {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC chunk_count does not match the latest terminal statement chunk index".into(),
                ));
            }
            if last_step.statement.step_public.step_hi != self.step_count {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC step_count does not match the latest terminal statement step_hi".into(),
                ));
            }
            if last_step.statement.step_public.state_out != self.z_i {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC latest terminal statement state_out does not match z_i".into(),
                ));
            }
        }
        Ok((vk_fs, folded_accumulator_digest))
    }

    fn validate_surface(&self) -> Result<(), SimpleKernelError> {
        let _ = self.validated_surface()?;
        Ok(())
    }

    fn validate_next_relation_surface(&self, relation: &Rv64imChunkStepIvcRelation) -> Result<(), SimpleKernelError> {
        if relation.witness.state_in.carry.main.claims != self.running_state.carry.main.claims
            || relation.witness.state_in.carry.main.witnesses != self.running_state.carry.main.witnesses
            || relation.witness.state_in.carry.terminal_handle != self.running_state.carry.terminal_handle
            || relation.witness.state_in.transcript != self.running_state.transcript
        {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC append requires the next relation to start from the carried running state".into(),
            ));
        }
        if relation.statement.step_public.chunk_index != self.chunk_count {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC append chunk index does not match the carried chunk count".into(),
            ));
        }
        if relation.statement.step_public.step_lo != self.step_count {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC append step_lo does not match the carried semantic step count".into(),
            ));
        }
        if relation.statement.step_public.state_in != self.z_i {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC append step_public.state_in does not match the carried z_i".into(),
            ));
        }
        if let Some(last_step) = self.last_step.as_ref() {
            if relation.statement.step_public.program_digest != last_step.statement.step_public.program_digest {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM IVC append program digest drifted across carried folds".into(),
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
                "RV64IM IVC state step_cap does not fit into the local native recursion width".into(),
            )
        })
    }

    fn validate_relation_step_cap(&self, public_step_count: u64, terminal_step: bool) -> Result<(), SimpleKernelError> {
        let active_step_count = usize::try_from(public_step_count).map_err(|_| {
            SimpleKernelError::Bridge(
                "RV64IM IVC relation public_step_count does not fit into the local native step-cap model".into(),
            )
        })?;
        let step_cap = self.step_cap_usize()?;
        if active_step_count == 0 || active_step_count > step_cap {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM IVC relation carries {active_step_count} public steps outside the frozen native step_cap={step_cap}"
            )));
        }
        if !terminal_step && active_step_count != step_cap {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM IVC non-terminal relation must carry exactly step_cap={step_cap} public steps; got {active_step_count}"
            )));
        }
        Ok(())
    }
}

pub(crate) fn build_rv64im_ivc_state_from_relations(
    relations: &[Rv64imChunkStepIvcRelation],
    step_cap: usize,
) -> Result<Rv64imIvcState, SimpleKernelError> {
    let mut state = Rv64imIvcState::init_with_step_cap(step_cap)?;
    for relation in relations {
        state = state.append(relation)?;
    }
    Ok(state)
}
