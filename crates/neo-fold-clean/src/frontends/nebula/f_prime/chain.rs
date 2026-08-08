//! Prover lifecycle for the authoritative three-arm Nebula F' relation.
//!
//! Each append first folds the previous claim, then synthesizes the current
//! field-native verifier execution and encodes it into the fixed low-norm
//! relation. Nebula application data is consumed with the same one-step delay;
//! finalization consumes the trailing claim.

use neo_ajtai::Commitment;
use neo_ccs::LaneCommitments;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use super::{
    enforce_nebula_application_f_prime_base_step, enforce_nebula_application_f_prime_recursive_step,
    enforce_nebula_f_prime_base_step, enforce_nebula_f_prime_recursive_step, NebulaFPrimeBranch, NebulaFPrimeError,
    NebulaFPrimeRelation, NebulaFPrimeRelationError,
};
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::frontends::direct_ccs::ajtai;
use crate::frontends::nebula::application::{ApplicationSegmentTrace, NebulaApplication};
use crate::frontends::nebula::circuit::StepData;
use crate::frontends::nebula::fingerprint::Gammas;
use crate::frontends::nebula::layout::LayoutError;
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::nebula::trace::SegmentTrace;
use crate::frontends::r1cs_f_prime::lowering::{
    normalized_field_assignment, normalized_source_column, LowNormR1csError,
};
use crate::lifecycle::{self, Preprocessing, Uncompressed, UncompressedAudit};
use crate::paper::construction2::{
    LaneCommitmentMode, NebulaError, NebulaLane, SemanticStateAdvance, SemanticStateMode, State,
};
use crate::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest_for_uniform_shape,
    initial_boundary_digest, public_trace_seed_digest, state_x_out_digest_with_mode, AccumulatorHandle,
    StateXOutDigestMode,
};
use crate::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL;
use crate::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use crate::paper::f_prime::r1cs::{
    FPrimeBaseInputs, FPrimePublicInputLayout, FPrimeRecursiveInputs, FPrimeStateIn, FPrimeStepConfig,
    F_PRIME_ENC_INST_BITS,
};
use crate::paper::f_prime::source_image::{BitRange, FPrimeSourceImage};
use crate::paper::nifs::circuit::NifsVCircuitMessages;
use crate::paper::nifs::{
    NifsFreshInstancesRequest, NifsFreshSignedUnitInstancesRequest, NifsProof, NifsProverAdapter,
};
use crate::paper::params::Params;
use crate::paper::relations::{CcsInstance, LaneSchemeError, RelationError};

#[derive(Debug, Error)]
pub enum NebulaFPrimeChainError {
    #[error(transparent)]
    Relation(#[from] NebulaFPrimeRelationError),
    #[error(transparent)]
    Composition(#[from] NebulaFPrimeError),
    #[error(transparent)]
    Lifecycle(#[from] lifecycle::Error),
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error(transparent)]
    Lanes(#[from] LaneSchemeError),
    #[error(transparent)]
    Instance(#[from] RelationError),
    #[error(transparent)]
    Nebula(#[from] NebulaError),
    #[error("folded Nebula F': trace was produced under different plan constants")]
    PlanMismatch,
    #[error(
        "folded Nebula F': trace position (segment {trace_seg}, step {trace_step}, ts {trace_ts}) \
         does not match delayed lane (segment {lane_seg}, step {lane_step}, ts {lane_ts})"
    )]
    ChainPositionMismatch {
        trace_seg: u64,
        trace_step: u64,
        trace_ts: u64,
        lane_seg: u64,
        lane_step: u64,
        lane_ts: u64,
    },
    #[error("folded Nebula F': current segment has no active gamma")]
    SegmentNotOpen,
    #[error("folded Nebula F': expected an active lifecycle state")]
    ExpectedActiveState,
    #[error("folded Nebula F': expected a recursive fold proof")]
    ExpectedRecursiveFold,
    #[error("folded Nebula F': field arm {branch:?} synthesized {rows}x{columns} with {public_columns} public columns; expected {expected_rows}x{expected_columns} with {expected_public_columns} public columns")]
    ArmShapeMismatch {
        branch: NebulaFPrimeBranch,
        rows: usize,
        columns: usize,
        public_columns: usize,
        expected_rows: usize,
        expected_columns: usize,
        expected_public_columns: usize,
    },
    #[error(
        "folded Nebula F': live {branch:?} field column {source_col:?} (`{source_family}` offset {source_family_col:?}) normalized to {normalized_col} carries {value}, exceeding inferred width {width}"
    )]
    LowNormWidthViolation {
        branch: NebulaFPrimeBranch,
        normalized_col: usize,
        source_col: Option<usize>,
        source_family: &'static str,
        source_family_col: Option<usize>,
        width: usize,
        value: u64,
    },
    #[error("folded Nebula F': remapped lane commitment does not match the segment precommit pass")]
    LaneCommitmentMismatch,
    #[error("folded Nebula F': chain builder has no appended steps")]
    EmptyChain,
    #[error("folded Nebula F': preprocessing includes an application; append an ApplicationSegmentTrace")]
    ApplicationTraceRequired,
    #[error("folded Nebula F': preprocessing has no application relation")]
    UnexpectedApplicationTrace,
    #[error("folded Nebula F': application semantic input does not match the carried state")]
    SemanticInputMismatch,
    #[cfg(feature = "perf-timers")]
    #[error("folded Nebula F' profiler requested {requested} steps from a {available}-step segment")]
    InvalidProfileStepCount { requested: usize, available: usize },
}

/// Verifier-owned fixed-point relation, plan, and lifecycle preprocessing.
/// No public constructor accepts an already-built `Preprocessing`, preventing
/// callers from attaching terminal-induction authority to an arbitrary CCS.
pub struct NebulaFPrimePreprocessing {
    pub prep: Preprocessing,
    relation: NebulaFPrimeRelation,
    plan: NebulaPlan,
}

impl NebulaFPrimePreprocessing {
    /// Compile and preprocess the authoritative fixed point using the global
    /// verifier-owned Ajtai setup.
    pub fn new(params: Params, plan: NebulaPlan) -> Result<Self, NebulaFPrimeChainError> {
        let relation = NebulaFPrimeRelation::compile_fixed_point(&params, &plan)?;
        Self::from_relation(params, plan, relation, None)
    }

    pub fn new_with_application(
        params: Params,
        plan: NebulaPlan,
        application: NebulaApplication,
    ) -> Result<Self, NebulaFPrimeChainError> {
        let relation = NebulaFPrimeRelation::compile_application_fixed_point(&params, &plan, application)?;
        Self::from_relation(params, plan, relation, None)
    }

    /// Deterministic test/demo constructor. Production callers use [`Self::new`].
    #[doc(hidden)]
    pub fn new_seeded(params: Params, plan: NebulaPlan, seed: u64) -> Result<Self, NebulaFPrimeChainError> {
        let relation = NebulaFPrimeRelation::compile_fixed_point(&params, &plan)?;
        let log = ajtai::setup_seeded(&params, relation.structure(), seed);
        Self::from_relation(params, plan, relation, Some(log))
    }

    #[doc(hidden)]
    pub fn new_seeded_with_application(
        params: Params,
        plan: NebulaPlan,
        application: NebulaApplication,
        seed: u64,
    ) -> Result<Self, NebulaFPrimeChainError> {
        let relation = NebulaFPrimeRelation::compile_application_fixed_point(&params, &plan, application)?;
        let log = ajtai::setup_seeded(&params, relation.structure(), seed);
        Self::from_relation(params, plan, relation, Some(log))
    }

    fn from_relation(
        params: Params,
        plan: NebulaPlan,
        mut relation: NebulaFPrimeRelation,
        test_log: Option<neo_ajtai::AjtaiSModule>,
    ) -> Result<Self, NebulaFPrimeChainError> {
        let public_input_len = Some(relation.public_input_len());
        let mut prep = match test_log {
            Some(log) => {
                lifecycle::preprocess_shared_with_test_log(params, relation.structure_arc(), log, public_input_len)
            }
            None => lifecycle::preprocess_shared(params, relation.structure_arc(), public_input_len),
        }?
        .with_nebula(relation.nebula_config().clone())
        .with_terminal_induction();
        if let Some(application) = relation.application() {
            let mode = crate::frontends::r1cs_f_prime::semantic_state_mode_for_plan(application.recursive_plan());
            let initial =
                crate::frontends::r1cs_f_prime::initial_semantic_state_digest_for_plan(application.recursive_plan());
            prep = prep
                .with_semantic_state_mode(mode)
                .with_initial_semantic_state_digest(initial)?;
        }
        relation.bind_preprocessing(&prep)?;
        Ok(Self { prep, relation, plan })
    }

    pub fn relation(&self) -> &NebulaFPrimeRelation {
        &self.relation
    }

    pub fn plan(&self) -> &NebulaPlan {
        &self.plan
    }
}

/// Serial K=1 prover for the authoritative fixed relation.
pub struct NebulaFPrimeChainBuilder<'a> {
    prep: &'a NebulaFPrimePreprocessing,
    audit: Option<UncompressedAudit>,
}

impl<'a> NebulaFPrimeChainBuilder<'a> {
    pub fn new(prep: &'a NebulaFPrimePreprocessing) -> Self {
        Self { prep, audit: None }
    }

    /// Append one complete memory segment. Its last claim remains delayed
    /// until the next append or terminal finalization.
    pub fn append_segment(&mut self, trace: &SegmentTrace) -> Result<(), NebulaFPrimeChainError> {
        self.append_segment_inner(trace, None)
    }

    /// Append one complete memory segment while routing recursive NIFS
    /// proving through `adapter`.
    pub fn append_segment_with_nifs_adapter(
        &mut self,
        trace: &SegmentTrace,
        adapter: &mut dyn NifsProverAdapter,
    ) -> Result<(), NebulaFPrimeChainError> {
        self.append_segment_inner(trace, Some(adapter))
    }

    fn append_segment_inner(
        &mut self,
        trace: &SegmentTrace,
        mut adapter: Option<&mut dyn NifsProverAdapter>,
    ) -> Result<(), NebulaFPrimeChainError> {
        if self.prep.relation.application().is_some() {
            return Err(NebulaFPrimeChainError::ApplicationTraceRequired);
        }
        let plan = self.prep.plan();
        if trace.params() != plan.params() {
            return Err(NebulaFPrimeChainError::PlanMismatch);
        }
        let (expected_advs, d_pre) = precommit_segment(plan, trace)?;
        let params = plan.params();
        let mut ts_in = trace.ts_in;
        let mut h_in = [K::ONE; 4];
        let mut sp_in = [0; 2];

        for step in 0..params.steps_per_segment() {
            let prepared = if let Some(adapter) = adapter.as_mut() {
                self.prepare_step(None, Some(&mut **adapter))?
            } else {
                self.prepare_step(None, None)?
            };
            let post = prepared.post();
            let lane = post
                .nebula
                .as_ref()
                .ok_or(NebulaFPrimeChainError::ExpectedActiveState)?;
            if lane.seg_idx != trace.seg_idx || lane.idx != step as u64 || lane.ts != ts_in {
                return Err(NebulaFPrimeChainError::ChainPositionMismatch {
                    trace_seg: trace.seg_idx,
                    trace_step: step as u64,
                    trace_ts: ts_in,
                    lane_seg: lane.seg_idx,
                    lane_step: lane.idx,
                    lane_ts: lane.ts,
                });
            }
            if lane.h != h_in || lane.sp != sp_in {
                return Err(NebulaFPrimeChainError::ChainPositionMismatch {
                    trace_seg: trace.seg_idx,
                    trace_step: step as u64,
                    trace_ts: ts_in,
                    lane_seg: lane.seg_idx,
                    lane_step: lane.idx,
                    lane_ts: lane.ts,
                });
            }

            let current_d_pre = (step == 0).then_some(d_pre);
            let gamma = gamma_for_current_claim(self.prep, post, current_d_pre)?;
            let data = StepData {
                seg_idx: trace.seg_idx,
                idx: step as u64,
                ts_in,
                h_in,
                sp_in,
                ops: trace.step_ops(step),
                is_cells: &trace.is_cells[step * params.b_scan..(step + 1) * params.b_scan],
                fs_cells: &trace.fs_cells[step * params.b_scan..(step + 1) * params.b_scan],
            };
            let (s_mem_assignment, step_x) = plan.circuit().witness(
                &Gammas {
                    gamma1: gamma[0],
                    gamma2: gamma[1],
                },
                &data,
            )?;
            ts_in = step_x.ts_out;
            h_in = step_x.h_out;
            sp_in = step_x.sp_out;

            let instance = if let Some(adapter) = adapter.as_mut() {
                self.synthesize_instance(&prepared, &s_mem_assignment, None, current_d_pre, Some(&mut **adapter))?
            } else {
                self.synthesize_instance(&prepared, &s_mem_assignment, None, current_d_pre, None)?
            };
            if instance.claim.adv.as_ref() != Some(&expected_advs[step]) {
                return Err(NebulaFPrimeChainError::LaneCommitmentMismatch);
            }
            if let Some(adapter) = adapter.as_mut() {
                self.deposit(prepared, instance, Some(&mut **adapter))?;
            } else {
                self.deposit(prepared, instance, None)?;
            }
        }
        Ok(())
    }

    /// Append one complete application segment. Each application assignment
    /// occupies one scan step and its declarative ports occupy fixed `S_mem`
    /// slots, including canonical holes.
    pub fn append_application_segment(
        &mut self,
        trace: &ApplicationSegmentTrace,
    ) -> Result<(), NebulaFPrimeChainError> {
        self.append_application_steps(trace, self.prep.plan().params().steps_per_segment(), None)
    }

    /// Append one complete application segment while routing recursive NIFS
    /// proving through `adapter`.
    pub fn append_application_segment_with_nifs_adapter(
        &mut self,
        trace: &ApplicationSegmentTrace,
        adapter: &mut dyn NifsProverAdapter,
    ) -> Result<(), NebulaFPrimeChainError> {
        self.append_application_steps(trace, self.prep.plan().params().steps_per_segment(), Some(adapter))
    }

    /// Append a nonterminal prefix of a production segment for per-fold
    /// profiling. The resulting audit is intentionally not terminal-verifiable.
    #[cfg(feature = "perf-timers")]
    #[doc(hidden)]
    pub fn append_application_prefix_for_profile(
        &mut self,
        trace: &ApplicationSegmentTrace,
        step_count: usize,
    ) -> Result<(), NebulaFPrimeChainError> {
        let available = self.prep.plan().params().steps_per_segment();
        if step_count == 0 || step_count > available {
            return Err(NebulaFPrimeChainError::InvalidProfileStepCount {
                requested: step_count,
                available,
            });
        }
        self.append_application_steps(trace, step_count, None)
    }

    fn append_application_steps(
        &mut self,
        trace: &ApplicationSegmentTrace,
        step_count: usize,
        mut adapter: Option<&mut dyn NifsProverAdapter>,
    ) -> Result<(), NebulaFPrimeChainError> {
        #[cfg(feature = "perf-timers")]
        let segment_started = std::time::Instant::now();
        let application = self
            .prep
            .relation
            .application()
            .ok_or(NebulaFPrimeChainError::UnexpectedApplicationTrace)?;
        let plan = self.prep.plan();
        let memory = trace.memory();
        if memory.params() != plan.params() {
            return Err(NebulaFPrimeChainError::PlanMismatch);
        }
        #[cfg(feature = "perf-timers")]
        let precommit_started = std::time::Instant::now();
        let (expected_advs, d_pre) = precommit_application_segment(plan, trace)?;
        #[cfg(feature = "perf-timers")]
        let precommit_elapsed = precommit_started.elapsed();
        let params = plan.params();
        let mut ts_in = memory.ts_in;
        let mut h_in = [K::ONE; 4];
        let mut sp_in = [0; 2];
        let mut assignments = trace.assignment_cursor();

        for step in 0..step_count {
            #[cfg(feature = "perf-timers")]
            let step_started = std::time::Instant::now();
            #[cfg(feature = "perf-timers")]
            let application_started = std::time::Instant::now();
            let assignment = assignments
                .next()
                .expect("application trace has one assignment per memory step");
            let semantic =
                crate::frontends::r1cs_f_prime::ivc::shape::semantic_values(application.recursive_plan(), assignment)
                    .map_err(NebulaFPrimeRelationError::from)?;
            #[cfg(feature = "perf-timers")]
            let application_elapsed = application_started.elapsed();
            #[cfg(feature = "perf-timers")]
            let prepare_started = std::time::Instant::now();
            let prepared = if let Some(adapter) = adapter.as_mut() {
                self.prepare_step(Some(semantic), Some(&mut **adapter))?
            } else {
                self.prepare_step(Some(semantic), None)?
            };
            #[cfg(feature = "perf-timers")]
            let prepare_elapsed = prepare_started.elapsed();
            #[cfg(feature = "perf-timers")]
            let branch = prepared.branch();
            let post = prepared.post();
            let lane = post
                .nebula
                .as_ref()
                .ok_or(NebulaFPrimeChainError::ExpectedActiveState)?;
            if lane.seg_idx != memory.seg_idx || lane.idx != step as u64 || lane.ts != ts_in {
                return Err(NebulaFPrimeChainError::ChainPositionMismatch {
                    trace_seg: memory.seg_idx,
                    trace_step: step as u64,
                    trace_ts: ts_in,
                    lane_seg: lane.seg_idx,
                    lane_step: lane.idx,
                    lane_ts: lane.ts,
                });
            }
            if lane.h != h_in || lane.sp != sp_in {
                return Err(NebulaFPrimeChainError::ChainPositionMismatch {
                    trace_seg: memory.seg_idx,
                    trace_step: step as u64,
                    trace_ts: ts_in,
                    lane_seg: lane.seg_idx,
                    lane_step: lane.idx,
                    lane_ts: lane.ts,
                });
            }

            #[cfg(feature = "perf-timers")]
            let memory_started = std::time::Instant::now();
            let current_d_pre = (step == 0).then_some(d_pre);
            let gamma = gamma_for_current_claim(self.prep, post, current_d_pre)?;
            let data = StepData {
                seg_idx: memory.seg_idx,
                idx: step as u64,
                ts_in,
                h_in,
                sp_in,
                ops: &[],
                is_cells: &memory.is_cells[step * params.b_scan..(step + 1) * params.b_scan],
                fs_cells: &memory.fs_cells[step * params.b_scan..(step + 1) * params.b_scan],
            };
            let (s_mem_assignment, step_x) = plan.circuit().witness_slots(
                &Gammas {
                    gamma1: gamma[0],
                    gamma2: gamma[1],
                },
                &data,
                trace.slots(step),
            )?;
            ts_in = step_x.ts_out;
            h_in = step_x.h_out;
            sp_in = step_x.sp_out;
            #[cfg(feature = "perf-timers")]
            let memory_elapsed = memory_started.elapsed();

            #[cfg(feature = "perf-timers")]
            let synthesis_started = std::time::Instant::now();
            let instance = if let Some(adapter) = adapter.as_mut() {
                self.synthesize_instance(
                    &prepared,
                    &s_mem_assignment,
                    Some(assignment),
                    current_d_pre,
                    Some(&mut **adapter),
                )?
            } else {
                self.synthesize_instance(&prepared, &s_mem_assignment, Some(assignment), current_d_pre, None)?
            };
            #[cfg(feature = "perf-timers")]
            let synthesis_elapsed = synthesis_started.elapsed();
            if instance.claim.adv.as_ref() != Some(&expected_advs[step]) {
                return Err(NebulaFPrimeChainError::LaneCommitmentMismatch);
            }
            #[cfg(feature = "perf-timers")]
            let deposit_started = std::time::Instant::now();
            if let Some(adapter) = adapter.as_mut() {
                self.deposit(prepared, instance, Some(&mut **adapter))?;
            } else {
                self.deposit(prepared, instance, None)?;
            }
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "[wasm-nebula-step] segment={} step={} branch={branch:?} app={:.3}s prior-fold={:.3}s memory={:.3}s fprime={:.3}s deposit={:.3}s total={:.3}s",
                memory.seg_idx,
                step,
                application_elapsed.as_secs_f64(),
                prepare_elapsed.as_secs_f64(),
                memory_elapsed.as_secs_f64(),
                synthesis_elapsed.as_secs_f64(),
                deposit_started.elapsed().as_secs_f64(),
                step_started.elapsed().as_secs_f64(),
            );
        }
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[wasm-nebula-segment] segment={} steps={} precommit={:.3}s total={:.3}s",
            memory.seg_idx,
            step_count,
            precommit_elapsed.as_secs_f64(),
            segment_started.elapsed().as_secs_f64(),
        );
        Ok(())
    }

    pub fn audit(&self) -> Option<&UncompressedAudit> {
        self.audit.as_ref()
    }

    pub fn into_audit(self) -> Result<UncompressedAudit, NebulaFPrimeChainError> {
        self.audit.ok_or(NebulaFPrimeChainError::EmptyChain)
    }

    pub fn finish(self) -> Result<Uncompressed, NebulaFPrimeChainError> {
        #[cfg(feature = "perf-timers")]
        let started = std::time::Instant::now();
        let prep = &self.prep.prep;
        let proof = lifecycle::finish_uncompressed(prep, self.into_audit()?)?;
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[wasm-nebula-finish] terminal_fold={:.3}s chunks={} steps={} final_fold={}",
            started.elapsed().as_secs_f64(),
            proof.state.chunk_count,
            proof.state.step_count,
            proof.final_fold.is_some(),
        );
        Ok(proof)
    }

    pub fn finish_with_audit(self) -> Result<UncompressedAudit, NebulaFPrimeChainError> {
        let prep = &self.prep.prep;
        Ok(lifecycle::finish_uncompressed_with_audit(prep, self.into_audit()?)?)
    }

    /// Finalize while routing the terminal latest-to-running fold through
    /// `adapter` and dropping the audit trail.
    pub fn finish_with_nifs_adapter(
        self,
        adapter: &mut dyn NifsProverAdapter,
    ) -> Result<Uncompressed, NebulaFPrimeChainError> {
        let prep = &self.prep.prep;
        Ok(lifecycle::finish_uncompressed_with_audit_and_nifs_adapter(prep, adapter, self.into_audit()?)?.proof)
    }

    /// Finalize while routing the terminal latest-to-running fold through
    /// `adapter` and retaining the audit trail.
    pub fn finish_with_audit_and_nifs_adapter(
        self,
        adapter: &mut dyn NifsProverAdapter,
    ) -> Result<UncompressedAudit, NebulaFPrimeChainError> {
        let prep = &self.prep.prep;
        Ok(lifecycle::finish_uncompressed_with_audit_and_nifs_adapter(
            prep,
            adapter,
            self.into_audit()?,
        )?)
    }

    fn prepare_step(
        &mut self,
        semantic: Option<crate::frontends::r1cs_f_prime::ivc::shape::SemanticValues>,
        adapter: Option<&mut dyn NifsProverAdapter>,
    ) -> Result<PreparedStep, NebulaFPrimeChainError> {
        let Some(audit) = self.audit.take() else {
            let pre = StateCoordinates::base(&self.prep.prep);
            if semantic.is_some_and(|values| {
                values
                    .input
                    .is_some_and(|input| input != pre.semantic_state_digest)
            }) {
                return Err(NebulaFPrimeChainError::SemanticInputMismatch);
            }
            let post = pre.base_advance(&self.prep.prep, semantic.and_then(|values| values.output))?;
            return Ok(PreparedStep::Base { pre, post });
        };

        let pre = StateCoordinates::from_state(&audit.proof.state);
        if semantic.is_some_and(|values| {
            values
                .input
                .is_some_and(|input| input != pre.semantic_state_digest)
        }) {
            self.audit = Some(audit);
            return Err(NebulaFPrimeChainError::SemanticInputMismatch);
        }
        let branch = if pre.step_count <= 1 {
            NebulaFPrimeBranch::BootstrapRecursive
        } else {
            NebulaFPrimeBranch::Recursive
        };
        let semantic_output = semantic.and_then(|values| values.output);
        let semantic_advance = match semantic_output {
            Some(output) => SemanticStateAdvance::Stateful(digest_fields_as_digest32(output)),
            None => SemanticStateAdvance::Stateless,
        };
        let fold = match adapter {
            Some(adapter) => crate::lifecycle::prove::prepare_recursive_step_with_nifs_adapter(
                &self.prep.prep,
                adapter,
                audit,
                semantic_advance,
            )?,
            None => crate::lifecycle::prove::prepare_recursive_step(&self.prep.prep, audit, semantic_advance)?,
        };
        let nifs = fold.nifs_proof()?;
        #[cfg(feature = "perf-timers")]
        {
            let combined = &nifs.pi_rlc.combined;
            let child = nifs.pi_dec.children.first();
            eprintln!(
                "[folded-f-prime] fresh={} running={} outputs={} children={} sumcheck={}x{} combined=(adv={}, c={}, X={}x{}, r={}, y_rows={}) child={:?}",
                fold.fresh().len(),
                fold.running().len(),
                nifs.pi_ccs.outputs.len(),
                nifs.pi_dec.children.len(),
                nifs.pi_ccs.sumcheck.sumcheck_rounds.len(),
                nifs.pi_ccs.sumcheck.sumcheck_rounds.first().map_or(0, Vec::len),
                combined.adv.is_some(),
                combined.c.data.len(),
                combined.X.rows(),
                combined.X.cols(),
                combined.r.len(),
                combined.y_ring.len(),
                child.map(|claim| (
                    claim.adv.is_some(),
                    claim.c.data.len(),
                    claim.X.rows(),
                    claim.X.cols(),
                    claim.r.len(),
                    claim.y_ring.len(),
                )),
            );
        }
        let pre = StateCoordinates::from_protocol(fold.pre());
        let post = StateCoordinates::from_protocol(fold.post());
        Ok(PreparedStep::Recursive {
            branch,
            pre,
            post,
            nifs,
            fold,
        })
    }

    fn synthesize_instance(
        &self,
        prepared: &PreparedStep,
        s_mem_assignment: &[F],
        application_assignment: Option<&[F]>,
        current_d_pre: Option<[[F; 4]; 3]>,
        adapter: Option<&mut dyn NifsProverAdapter>,
    ) -> Result<CcsInstance, NebulaFPrimeChainError> {
        #[cfg(feature = "perf-timers")]
        let total_started = std::time::Instant::now();
        let nifs = self.prep.prep.nifs_v_circuit_config()?;
        let nebula = self.prep.plan.config();
        let cfg = FPrimeStepConfig {
            nifs,
            b: self.prep.prep.params.b(),
            transcript_label: F_PRIME_STEP_TRANSCRIPT_LABEL,
            public_input_layout: FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(nebula.stacks)),
            nebula: Some(&nebula),
            state_x_out_digest_mode: self
                .prep
                .relation
                .application()
                .map_or(StateXOutDigestMode::Stateless, |application| {
                    crate::frontends::r1cs_f_prime::ivc::shape::digest_mode(application.recursive_plan())
                }),
        };
        let application = match (self.prep.relation.application(), application_assignment) {
            (Some(application), Some(assignment)) => Some((application, assignment)),
            (Some(_), None) => return Err(NebulaFPrimeChainError::ApplicationTraceRequired),
            (None, Some(_)) => return Err(NebulaFPrimeChainError::UnexpectedApplicationTrace),
            (None, None) => None,
        };

        let mut source = FPrimeSourceImage::new();
        let pre = prepared.pre();
        let post = prepared.post();
        let chunk_count_in_word = source.push_u64_le(pre.chunk_count);
        let step_count_in_word = source.push_u64_le(pre.step_count);
        let pc_word = source.push_u64_le(pre.pc);
        // Preprocessing already owns the authoritative rows. Proving only
        // evaluates that fixed circuit to obtain its witness.
        let mut builder = R1csBuilder::new_witness_only();
        #[cfg(feature = "perf-timers")]
        let setup_elapsed = total_started.elapsed();
        #[cfg(feature = "perf-timers")]
        let enforce_started = std::time::Instant::now();

        let (branch, output) = match prepared {
            PreparedStep::Base { .. } => {
                let public_x_out_bits = source.push_enc_inst(post.x_out_fields(&self.prep.prep));
                let inputs = FPrimeBaseInputs {
                    state: pre.as_f_prime_state(&self.prep.prep),
                    chunk_digest: post.z_i,
                    semantic_state_digest_out: post.semantic_state_digest,
                    rows_in_chunk: 1,
                    source_image: &source,
                    chunk_count_in_word,
                    step_count_in_word,
                    pc_word,
                    public_x_out_bits,
                };
                let output = match application {
                    Some((application, assignment)) => enforce_nebula_application_f_prime_base_step(
                        &mut builder,
                        self.prep.plan.circuit(),
                        s_mem_assignment,
                        application,
                        assignment,
                        current_d_pre,
                        &cfg,
                        &inputs,
                    )?,
                    None => enforce_nebula_f_prime_base_step(
                        &mut builder,
                        self.prep.plan.circuit(),
                        s_mem_assignment,
                        current_d_pre,
                        &cfg,
                        &inputs,
                    )?,
                };
                (NebulaFPrimeBranch::Base, output)
            }
            PreparedStep::Recursive { branch, nifs, fold, .. } => {
                let prior_public = source.push_f_prime_public_input(pre.x_out_fields(&self.prep.prep));
                let prior_x_out_bits = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
                let public_x_out_bits = source.push_enc_inst(post.x_out_fields(&self.prep.prep));
                let messages = NifsVCircuitMessages {
                    fresh: fold.fresh(),
                    running: fold.running(),
                    running_parent_authority: fold.running_parent_authority(),
                    pi_ccs: &nifs.pi_ccs,
                    combined: &nifs.pi_rlc.combined,
                    children: &nifs.pi_dec.children,
                };
                let inputs = FPrimeRecursiveInputs {
                    state: pre.as_f_prime_state(&self.prep.prep),
                    chunk_digest: post.z_i,
                    semantic_state_digest_out: post.semantic_state_digest,
                    acc_digest_out: post.acc_digest,
                    nifs_msg: messages,
                    rows_in_chunk: 1,
                    source_image: &source,
                    chunk_count_in_word,
                    step_count_in_word,
                    pc_word,
                    prior_x_out_bits,
                    public_x_out_bits,
                };
                let output = match application {
                    Some((application, assignment)) => enforce_nebula_application_f_prime_recursive_step(
                        &mut builder,
                        &self.prep.prep.params,
                        self.prep.plan.circuit(),
                        s_mem_assignment,
                        application,
                        assignment,
                        current_d_pre,
                        &cfg,
                        &inputs,
                    )?,
                    None => enforce_nebula_f_prime_recursive_step(
                        &mut builder,
                        &self.prep.prep.params,
                        self.prep.plan.circuit(),
                        s_mem_assignment,
                        current_d_pre,
                        &cfg,
                        &inputs,
                    )?,
                };
                (*branch, output)
            }
        };
        #[cfg(feature = "perf-timers")]
        let enforce_elapsed = enforce_started.elapsed();

        let public_outputs = output.public_outputs();
        let expected = self.prep.relation.arm_shape(branch);
        let actual = (builder.rows(), builder.witness().len(), 1 + public_outputs.len());
        if actual != (expected.rows, expected.columns, expected.public_columns) {
            return Err(NebulaFPrimeChainError::ArmShapeMismatch {
                branch,
                rows: actual.0,
                columns: actual.1,
                public_columns: actual.2,
                expected_rows: expected.rows,
                expected_columns: expected.columns,
                expected_public_columns: expected.public_columns,
            });
        }
        #[cfg(feature = "perf-timers")]
        let normalize_started = std::time::Instant::now();
        let field_assignment =
            normalized_field_assignment(&builder, &public_outputs).map_err(NebulaFPrimeRelationError::from)?;
        let builder_columns = actual.1;
        let column_family_ranges = builder.column_family_ranges().to_vec();
        drop(builder);
        crate::heap::release_unused_pages();
        #[cfg(feature = "perf-timers")]
        let normalize_elapsed = normalize_started.elapsed();
        #[cfg(feature = "perf-timers")]
        let instance_started = std::time::Instant::now();
        let instance: Result<CcsInstance, NebulaFPrimeChainError> = if let Some(adapter) = adapter {
            #[cfg(feature = "perf-timers")]
            let encode_started = std::time::Instant::now();
            let assignment = self
                .prep
                .relation
                .encode_signed_unit_for_deferred_nifs(branch, &field_assignment)?;
            #[cfg(feature = "perf-timers")]
            let encode_elapsed = encode_started.elapsed();
            let assignments = [assignment];
            #[cfg(feature = "perf-timers")]
            let adapter_started = std::time::Instant::now();
            let mut accelerated = adapter
                .build_fresh_signed_unit_instances(NifsFreshSignedUnitInstancesRequest {
                    pp: &self.prep.prep.params,
                    s: self.prep.prep.structure(),
                    cache: self.prep.prep.optimized_cache(),
                    log: &self.prep.prep.log,
                    m_in: self.prep.relation.public_input_len(),
                    assignments: &assignments,
                    lane_scheme: Some(&self.prep.relation.nebula_config().scheme),
                })
                .map_err(crate::paper::construction2::Error::from)
                .map_err(lifecycle::Error::from)?;
            if accelerated.is_none() {
                let dense = assignments[0].to_dense();
                let dense_assignments = [dense.as_slice()];
                accelerated = adapter
                    .build_fresh_instances(NifsFreshInstancesRequest {
                        pp: &self.prep.prep.params,
                        s: self.prep.prep.structure(),
                        cache: self.prep.prep.optimized_cache(),
                        log: &self.prep.prep.log,
                        m_in: self.prep.relation.public_input_len(),
                        assignments: &dense_assignments,
                        lane_scheme: Some(&self.prep.relation.nebula_config().scheme),
                    })
                    .map_err(crate::paper::construction2::Error::from)
                    .map_err(lifecycle::Error::from)?;
            }
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "[fprime-metal-instance] branch={branch:?} encode={:.3}s adapter={:.3}s",
                encode_elapsed.as_secs_f64(),
                adapter_started.elapsed().as_secs_f64(),
            );
            match accelerated {
                Some(mut instances) if instances.len() == 1 => {
                    let mut instance = instances.pop().expect("one accelerated fresh instance");
                    if instance.claim.adv.is_none() {
                        self.prep.relation.attach_lane_commitment(&mut instance)?;
                    }
                    Ok(instance)
                }
                Some(_) => Err(
                    lifecycle::Error::Construction2(crate::paper::construction2::Error::Nifs(
                        crate::paper::nifs::Error::BackendUnavailable {
                            backend: "adapter",
                            reason: "adapter returned the wrong number of Nebula fresh instances",
                        },
                    ))
                    .into(),
                ),
                None => self
                    .prep
                    .relation
                    .build_instance(&self.prep.prep, branch, &field_assignment)
                    .map_err(NebulaFPrimeChainError::from),
            }
        } else {
            self.prep
                .relation
                .build_instance(&self.prep.prep, branch, &field_assignment)
                .map_err(NebulaFPrimeChainError::from)
        };
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[fprime-synthesize] branch={branch:?} setup={:.3}s witness={:.3}s normalize={:.3}s instance={:.3}s total={:.3}s rows={} field_cols={} normalized_cols={}",
            setup_elapsed.as_secs_f64(),
            enforce_elapsed.as_secs_f64(),
            normalize_elapsed.as_secs_f64(),
            instance_started.elapsed().as_secs_f64(),
            total_started.elapsed().as_secs_f64(),
            actual.0,
            actual.1,
            field_assignment.len(),
        );
        match instance {
            Err(NebulaFPrimeChainError::Relation(NebulaFPrimeRelationError::LowNorm(
                LowNormR1csError::InferredWidthViolation { col, width, value },
            ))) => {
                let source_col = normalized_source_column(builder_columns, &public_outputs, col);
                let source_range = source_col.and_then(|source_col| {
                    column_family_ranges
                        .iter()
                        .filter(|range| range.column_start <= source_col && source_col < range.column_end)
                        .min_by_key(|range| range.column_end - range.column_start)
                });
                Err(NebulaFPrimeChainError::LowNormWidthViolation {
                    branch,
                    normalized_col: col,
                    source_col,
                    source_family: source_range.map_or("unclassified", |range| range.name),
                    source_family_col: source_col
                        .zip(source_range)
                        .map(|(source_col, range)| source_col - range.column_start),
                    width,
                    value,
                })
            }
            Err(error) => Err(error),
            Ok(instance) => Ok(instance),
        }
    }

    fn deposit(
        &mut self,
        prepared: PreparedStep,
        instance: CcsInstance,
        adapter: Option<&mut dyn NifsProverAdapter>,
    ) -> Result<(), NebulaFPrimeChainError> {
        self.audit = Some(match prepared {
            PreparedStep::Base { post, .. } => match (self.prep.prep.semantic_state_mode(), adapter) {
                (SemanticStateMode::Stateful, Some(adapter)) => {
                    crate::lifecycle::prove::prove_one_with_semantic_state_and_nifs_adapter(
                        &self.prep.prep,
                        adapter,
                        vec![instance],
                        self.prep.prep.initial_semantic_state_digest(),
                        digest_fields_as_digest32(post.semantic_state_digest),
                    )?
                }
                (SemanticStateMode::Stateful, None) => crate::lifecycle::prove::prove_one_with_semantic_state(
                    &self.prep.prep,
                    vec![instance],
                    self.prep.prep.initial_semantic_state_digest(),
                    digest_fields_as_digest32(post.semantic_state_digest),
                )?,
                (_, Some(adapter)) => lifecycle::prove_with_nifs_adapter(&self.prep.prep, adapter, [vec![instance]])?,
                (_, None) => lifecycle::prove(&self.prep.prep, [vec![instance]])?,
            },
            PreparedStep::Recursive { fold, .. } => fold.complete(instance)?,
        });
        Ok(())
    }
}

enum PreparedStep {
    Base {
        pre: StateCoordinates,
        post: StateCoordinates,
    },
    Recursive {
        branch: NebulaFPrimeBranch,
        pre: StateCoordinates,
        post: StateCoordinates,
        nifs: NifsProof,
        fold: crate::lifecycle::prove::PreparedRecursiveStep,
    },
}

impl PreparedStep {
    #[cfg(feature = "perf-timers")]
    fn branch(&self) -> NebulaFPrimeBranch {
        match self {
            Self::Base { .. } => NebulaFPrimeBranch::Base,
            Self::Recursive { branch, .. } => *branch,
        }
    }

    fn pre(&self) -> &StateCoordinates {
        match self {
            Self::Base { pre, .. } | Self::Recursive { pre, .. } => pre,
        }
    }

    fn post(&self) -> &StateCoordinates {
        match self {
            Self::Base { post, .. } | Self::Recursive { post, .. } => post,
        }
    }
}

#[derive(Clone)]
struct StateCoordinates {
    chunk_count: u64,
    step_count: u64,
    z_0: [F; 4],
    z_i: [F; 4],
    pc: u64,
    semantic_state_digest: [F; 4],
    acc_digest: [F; 4],
    public_trace: [F; 4],
    nebula: Option<NebulaLane>,
}

impl StateCoordinates {
    fn base(prep: &Preprocessing) -> Self {
        let z_0 = digest32_as_fields(initial_boundary_digest(prep.structure_digest(), prep.public_input_len));
        let empty = AccumulatorHandle::empty().digest_fields();
        Self {
            chunk_count: 0,
            step_count: 0,
            z_0,
            z_i: z_0,
            pc: crate::paper::construction2::TRIVIAL_PC,
            semantic_state_digest: digest32_as_fields(prep.initial_semantic_state_digest()),
            acc_digest: empty,
            public_trace: digest32_as_fields(public_trace_seed_digest(prep.structure_digest())),
            nebula: prep.nebula().map(NebulaLane::base),
        }
    }

    fn from_state(state: &State) -> Self {
        Self {
            chunk_count: state.chunk_count,
            step_count: state.step_count,
            z_0: digest32_as_fields(state.z_0),
            z_i: digest32_as_fields(state.z_i),
            pc: state.pc,
            semantic_state_digest: digest32_as_fields(state.semantic_state_digest),
            acc_digest: digest32_as_fields(state.acc_digest),
            public_trace: digest32_as_fields(state.public_trace),
            nebula: state.nebula.clone(),
        }
    }

    fn from_protocol(state: &crate::paper::construction2::StateCoordinates) -> Self {
        Self {
            chunk_count: state.chunk_count,
            step_count: state.step_count,
            z_0: digest32_as_fields(state.z_0),
            z_i: digest32_as_fields(state.z_i),
            pc: state.pc,
            semantic_state_digest: digest32_as_fields(state.semantic_state_digest),
            acc_digest: digest32_as_fields(state.acc_digest),
            public_trace: digest32_as_fields(state.public_trace),
            nebula: state.nebula.clone(),
        }
    }

    fn base_advance(
        &self,
        prep: &Preprocessing,
        semantic_output: Option<[F; 4]>,
    ) -> Result<Self, NebulaFPrimeChainError> {
        let z_i = f_prime_chunk_public_digest_for_uniform_shape(
            self.step_count,
            1,
            D,
            prep.params.kappa() as usize,
            prep.public_input_len
                .expect("folded F' fixes public input length"),
        );
        let m_in = prep
            .public_input_len
            .expect("folded F' fixes public input length");
        let running = crate::paper::construction2::RunningInstance::canonical_zero(
            &prep.params,
            prep.structure(),
            m_in,
            LaneCommitmentMode::Nebula,
        )
        .map_err(crate::paper::construction2::Error::from)
        .map_err(lifecycle::Error::from)?;
        let acc_digest = digest32_as_fields(
            running
                .accumulator_digest(prep.structure())
                .map_err(crate::paper::construction2::Error::from)
                .map_err(lifecycle::Error::from)?,
        );
        Ok(Self {
            chunk_count: 1,
            step_count: 1,
            z_i,
            semantic_state_digest: semantic_output.unwrap_or(acc_digest),
            acc_digest,
            public_trace: z_i,
            ..self.clone()
        })
    }

    fn as_f_prime_state(&self, prep: &Preprocessing) -> FPrimeStateIn {
        FPrimeStateIn {
            vk_fs_digest: digest32_as_fields(prep.vk.digest()),
            pi_ccs_header_bundle: prep.pi_ccs_header_bundle(),
            chunk_count_in: self.chunk_count,
            step_count_in: self.step_count,
            z_0: self.z_0,
            z_i_in: self.z_i,
            pc: self.pc,
            semantic_state_digest_in: self.semantic_state_digest,
            acc_digest_in: self.acc_digest,
            public_trace_in: self.public_trace,
            nebula: self.nebula.clone(),
        }
    }

    fn x_out_fields(&self, prep: &Preprocessing) -> [F; 4] {
        digest32_as_fields(state_x_out_digest_with_mode(
            match prep.semantic_state_mode() {
                SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
                SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
            },
            prep.vk.digest(),
            prep.pi_ccs_header_bundle(),
            prep.structure_digest(),
            self.chunk_count,
            self.step_count,
            digest_fields_as_digest32(self.z_0),
            digest_fields_as_digest32(self.z_i),
            self.pc,
            digest_fields_as_digest32(self.semantic_state_digest),
            digest_fields_as_digest32(self.acc_digest),
            digest_fields_as_digest32(self.public_trace),
            self.nebula.as_ref().map(NebulaLane::digest),
        ))
    }
}

fn gamma_for_current_claim(
    prep: &NebulaFPrimePreprocessing,
    post: &StateCoordinates,
    d_pre: Option<[[F; 4]; 3]>,
) -> Result<[K; 2], NebulaFPrimeChainError> {
    let mut lane = post
        .nebula
        .clone()
        .ok_or(NebulaFPrimeChainError::ExpectedActiveState)?;
    if let Some(d_pre) = d_pre {
        lane.open_segment(
            prep.prep
                .nebula()
                .ok_or(NebulaFPrimeChainError::ExpectedActiveState)?,
            prep.prep.vk.digest(),
            digest_fields_as_digest32(post.z_i),
            digest_fields_as_digest32(post.acc_digest),
            d_pre,
        )?;
    }
    lane.gamma.ok_or(NebulaFPrimeChainError::SegmentNotOpen)
}

fn precommit_segment(
    plan: &NebulaPlan,
    trace: &SegmentTrace,
) -> Result<(Vec<LaneCommitments<Commitment>>, [[F; 4]; 3]), NebulaFPrimeChainError> {
    let params = plan.params();
    let mut advs = Vec::with_capacity(params.steps_per_segment());
    for step in 0..params.steps_per_segment() {
        let ops = params.encode_ops_lane(trace.step_ops(step))?;
        let is = params.encode_scan_lane(&trace.is_cells[step * params.b_scan..(step + 1) * params.b_scan])?;
        let fs = params.encode_scan_lane(&trace.fs_cells[step * params.b_scan..(step + 1) * params.b_scan])?;
        advs.push(plan.scheme().commit_bits(&ops, &is, &fs)?);
    }
    let d_pre = crate::paper::digest::nebula_lane_chains(advs.iter());
    Ok((advs, d_pre))
}

fn precommit_application_segment(
    plan: &NebulaPlan,
    trace: &ApplicationSegmentTrace,
) -> Result<(Vec<LaneCommitments<Commitment>>, [[F; 4]; 3]), NebulaFPrimeChainError> {
    let params = plan.params();
    let memory = trace.memory();
    let mut advs = Vec::with_capacity(params.steps_per_segment());
    for step in 0..params.steps_per_segment() {
        let ops = params.encode_op_slots(trace.slots(step))?;
        let is = params.encode_scan_lane(&memory.is_cells[step * params.b_scan..(step + 1) * params.b_scan])?;
        let fs = params.encode_scan_lane(&memory.fs_cells[step * params.b_scan..(step + 1) * params.b_scan])?;
        advs.push(plan.scheme().commit_bits(&ops, &is, &fs)?);
    }
    let d_pre = crate::paper::digest::nebula_lane_chains(advs.iter());
    Ok((advs, d_pre))
}
