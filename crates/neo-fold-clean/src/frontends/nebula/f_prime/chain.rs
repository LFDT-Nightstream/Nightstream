//! Prover lifecycle for the authoritative three-arm Nebula F' relation.
//!
//! Each append first folds the previous claim, then synthesizes the current
//! field-native verifier execution and encodes it into the fixed low-norm
//! relation. Nebula application data is consumed with the same one-step delay;
//! finalization consumes the trailing claim.

use neo_ajtai::Commitment;
use neo_ccs::{LaneCommitments, Mat};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use super::{
    enforce_nebula_f_prime_base_step, enforce_nebula_f_prime_recursive_step, NebulaFPrimeBranch, NebulaFPrimeError,
    NebulaFPrimeRelation, NebulaFPrimeRelationError,
};
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::frontends::direct_ccs::ajtai;
use crate::frontends::nebula::circuit::StepData;
use crate::frontends::nebula::fingerprint::Gammas;
use crate::frontends::nebula::layout::LayoutError;
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::nebula::trace::SegmentTrace;
use crate::frontends::r1cs_f_prime::lowering::normalized_field_assignment;
use crate::lifecycle::{self, Preprocessing, Uncompressed, UncompressedAudit};
use crate::paper::construction2::{FoldProof, NebulaError, NebulaLane, ProofState, State};
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
use crate::paper::nifs::NifsProof;
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, CcsInstance, CcsWitness, CeClaim, LaneSchemeError, RelationError};

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
    #[error("folded Nebula F': live {branch:?} field synthesis is unsatisfied at row {row:?}")]
    FieldSynthesisUnsatisfied {
        branch: NebulaFPrimeBranch,
        row: Option<usize>,
    },
    #[error("folded Nebula F': remapped lane commitment does not match the segment precommit pass")]
    LaneCommitmentMismatch,
    #[error("folded Nebula F': chain builder has no appended steps")]
    EmptyChain,
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
        Self::from_relation(params, plan, relation)
    }

    /// Deterministic test/demo constructor. Production callers use [`Self::new`].
    #[doc(hidden)]
    pub fn new_seeded(params: Params, plan: NebulaPlan, seed: u64) -> Result<Self, NebulaFPrimeChainError> {
        let relation = NebulaFPrimeRelation::compile_fixed_point(&params, &plan)?;
        let _ = ajtai::setup_seeded(&params, relation.structure(), seed);
        Self::from_relation(params, plan, relation)
    }

    fn from_relation(
        params: Params,
        plan: NebulaPlan,
        mut relation: NebulaFPrimeRelation,
    ) -> Result<Self, NebulaFPrimeChainError> {
        let prep = lifecycle::preprocess(params, relation.structure().clone(), Some(relation.public_input_len()))?
            .with_nebula(relation.nebula_config().clone())
            .with_terminal_induction();
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
            let prepared = self.prepare_step()?;
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

            let instance = self.synthesize_instance(&prepared, &s_mem_assignment, current_d_pre)?;
            if instance.claim.adv.as_ref() != Some(&expected_advs[step]) {
                return Err(NebulaFPrimeChainError::LaneCommitmentMismatch);
            }
            self.deposit(prepared, instance)?;
        }
        Ok(())
    }

    pub fn audit(&self) -> Option<&UncompressedAudit> {
        self.audit.as_ref()
    }

    pub fn into_audit(self) -> Result<UncompressedAudit, NebulaFPrimeChainError> {
        self.audit.ok_or(NebulaFPrimeChainError::EmptyChain)
    }

    pub fn finish(self) -> Result<Uncompressed, NebulaFPrimeChainError> {
        let prep = &self.prep.prep;
        Ok(lifecycle::finish_uncompressed(prep, self.into_audit()?)?)
    }

    pub fn finish_with_audit(self) -> Result<UncompressedAudit, NebulaFPrimeChainError> {
        let prep = &self.prep.prep;
        Ok(lifecycle::finish_uncompressed_with_audit(prep, self.into_audit()?)?)
    }

    fn prepare_step(&mut self) -> Result<PreparedStep, NebulaFPrimeChainError> {
        let Some(audit) = self.audit.take() else {
            let pre = StateCoordinates::base(&self.prep.prep);
            let post = pre.base_advance(&self.prep.prep);
            return Ok(PreparedStep::Base { pre, post });
        };

        let pre = StateCoordinates::from_state(&audit.proof.state);
        let (running, running_parent_authority, fresh, placeholder) = match &audit.proof.state.proof {
            ProofState::Active { running, latest } => {
                let prior = latest
                    .instances
                    .first()
                    .ok_or(NebulaFPrimeChainError::ExpectedActiveState)?;
                let placeholder = CcsInstance {
                    claim: prior.claim.clone(),
                    witness: CcsWitness {
                        w: Vec::new(),
                        Z: Mat::zero(0, 0, F::ZERO),
                    },
                };
                (
                    running.claims.clone(),
                    running.parent_authority.clone(),
                    latest.claims(),
                    placeholder,
                )
            }
            ProofState::Initial => return Err(NebulaFPrimeChainError::ExpectedActiveState),
        };
        let branch = if running.is_empty() {
            NebulaFPrimeBranch::BootstrapRecursive
        } else {
            NebulaFPrimeBranch::Recursive
        };
        let pending = lifecycle::extend(&self.prep.prep, audit, vec![placeholder])?;
        let nifs = match &pending
            .steps
            .last()
            .ok_or(NebulaFPrimeChainError::ExpectedRecursiveFold)?
            .fold
        {
            FoldProof::Recursive(proof) => proof.clone(),
            FoldProof::NoFold => return Err(NebulaFPrimeChainError::ExpectedRecursiveFold),
        };
        #[cfg(feature = "perf-timers")]
        {
            let combined = &nifs.pi_rlc.combined;
            let child = nifs.pi_dec.children.first();
            eprintln!(
                "[folded-f-prime] fresh={} running={} outputs={} children={} fe_rounds={}x{} nc_rounds={}x{} combined=(adv={}, c={}, X={}x{}, r={}, s={}, y_rows={}, yz={}) child={:?}",
                fresh.len(),
                running.len(),
                nifs.pi_ccs.outputs.len(),
                nifs.pi_dec.children.len(),
                nifs.pi_ccs.sumcheck.sumcheck_rounds.len(),
                nifs.pi_ccs.sumcheck.sumcheck_rounds.first().map_or(0, Vec::len),
                nifs.pi_ccs.sumcheck.sumcheck_rounds_nc.len(),
                nifs.pi_ccs.sumcheck.sumcheck_rounds_nc.first().map_or(0, Vec::len),
                combined.adv.is_some(),
                combined.c.data.len(),
                combined.X.rows(),
                combined.X.cols(),
                combined.r.len(),
                combined.s_col.len(),
                combined.y_ring.len(),
                combined.y_zcol.len(),
                child.map(|claim| (
                    claim.adv.is_some(),
                    claim.c.data.len(),
                    claim.X.rows(),
                    claim.X.cols(),
                    claim.r.len(),
                    claim.s_col.len(),
                    claim.y_ring.len(),
                    claim.y_zcol.len(),
                )),
            );
        }
        let post = StateCoordinates::from_state(&pending.proof.state);
        Ok(PreparedStep::Recursive {
            branch,
            pre,
            post,
            fresh,
            running,
            running_parent_authority,
            nifs,
            pending,
        })
    }

    fn synthesize_instance(
        &self,
        prepared: &PreparedStep,
        s_mem_assignment: &[F],
        current_d_pre: Option<[[F; 4]; 3]>,
    ) -> Result<CcsInstance, NebulaFPrimeChainError> {
        let nifs = self.prep.prep.nifs_v_circuit_config()?;
        let nebula = self.prep.plan.config();
        let cfg = FPrimeStepConfig {
            nifs,
            b: self.prep.prep.params.b(),
            transcript_label: F_PRIME_STEP_TRANSCRIPT_LABEL,
            public_input_layout: FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(nebula.stacks)),
            nebula: Some(&nebula),
            state_x_out_digest_mode: StateXOutDigestMode::Stateless,
        };

        let mut source = FPrimeSourceImage::new();
        let pre = prepared.pre();
        let post = prepared.post();
        let chunk_count_in_word = source.push_u64_le(pre.chunk_count);
        let step_count_in_word = source.push_u64_le(pre.step_count);
        let pc_word = source.push_u64_le(pre.pc);
        let mut builder = R1csBuilder::new();

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
                let output = enforce_nebula_f_prime_base_step(
                    &mut builder,
                    self.prep.plan.circuit(),
                    s_mem_assignment,
                    current_d_pre,
                    &cfg,
                    &inputs,
                )?;
                (NebulaFPrimeBranch::Base, output)
            }
            PreparedStep::Recursive {
                branch,
                fresh,
                running,
                running_parent_authority,
                nifs,
                ..
            } => {
                let prior_public = source.push_f_prime_public_input(pre.x_out_fields(&self.prep.prep));
                let prior_x_out_bits = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
                let public_x_out_bits = source.push_enc_inst(post.x_out_fields(&self.prep.prep));
                let messages = NifsVCircuitMessages {
                    fresh,
                    running,
                    running_parent_authority: running_parent_authority.as_ref(),
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
                let output = enforce_nebula_f_prime_recursive_step(
                    &mut builder,
                    &self.prep.prep.params,
                    self.prep.plan.circuit(),
                    s_mem_assignment,
                    current_d_pre,
                    &cfg,
                    &inputs,
                )?;
                (*branch, output)
            }
        };

        if !builder.is_satisfied() {
            return Err(NebulaFPrimeChainError::FieldSynthesisUnsatisfied {
                branch,
                row: builder.first_unsatisfied_row(),
            });
        }
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
        let field_assignment =
            normalized_field_assignment(&builder, &public_outputs).map_err(NebulaFPrimeRelationError::from)?;
        self.prep
            .relation
            .build_instance(&self.prep.prep, branch, &field_assignment)
            .map_err(Into::into)
    }

    fn deposit(&mut self, prepared: PreparedStep, instance: CcsInstance) -> Result<(), NebulaFPrimeChainError> {
        self.audit = Some(match prepared {
            PreparedStep::Base { .. } => lifecycle::prove(&self.prep.prep, [vec![instance]])?,
            PreparedStep::Recursive { mut pending, .. } => {
                let claim = instance.claim.clone();
                match &mut pending.proof.state.proof {
                    ProofState::Active { latest, .. } => {
                        *latest = crate::paper::construction2::LatestInstance::from_instances(vec![instance]);
                    }
                    ProofState::Initial => return Err(NebulaFPrimeChainError::ExpectedActiveState),
                }
                *pending
                    .public_batches
                    .last_mut()
                    .ok_or(NebulaFPrimeChainError::ExpectedActiveState)? = vec![claim];
                pending
            }
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
        fresh: Vec<CcsClaim>,
        running: Vec<CeClaim>,
        running_parent_authority: Option<CeClaim>,
        nifs: NifsProof,
        pending: UncompressedAudit,
    },
}

impl PreparedStep {
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
            semantic_state_digest: empty,
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

    fn base_advance(&self, prep: &Preprocessing) -> Self {
        let z_i = f_prime_chunk_public_digest_for_uniform_shape(
            self.step_count,
            1,
            D,
            prep.params.kappa() as usize,
            prep.public_input_len
                .expect("folded F' fixes public input length"),
        );
        Self {
            chunk_count: 1,
            step_count: 1,
            z_i,
            public_trace: z_i,
            ..self.clone()
        }
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
            StateXOutDigestMode::Stateless,
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
