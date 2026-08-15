//! Prover lifecycle for the generic implementation R1CS IVC relation.

use neo_math::{D, F};

use super::relation::{R1csIvcBranch, R1csIvcRelation};
use super::shape::{digest_mode, enforce_base_application, enforce_recursive_application, semantic_values};
use super::R1csIvcError;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::frontends::direct_ccs::ajtai;
use crate::frontends::f_prime::recursive_plan::RecursiveStepImagePlan;
use crate::frontends::r1cs_f_prime::lowering::normalized_field_assignment;
use crate::frontends::r1cs_f_prime::R1csShape;
use crate::lifecycle::{self, Preprocessing, Uncompressed, UncompressedAudit};
use crate::paper::construction2::{LaneCommitmentMode, SemanticStateAdvance, SemanticStateMode, State};
use crate::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest_for_uniform_shape,
    initial_boundary_digest, public_trace_seed_digest, state_x_out_digest_with_mode, AccumulatorHandle,
};
use crate::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL;
use crate::paper::f_prime::r1cs::{
    FPrimeBaseInputs, FPrimePublicInputLayout, FPrimeRecursiveInputs, FPrimeStateIn, FPrimeStepConfig,
    F_PRIME_ENC_INST_BITS,
};
use crate::paper::f_prime::source_image::{BitRange, FPrimeSourceImage};
use crate::paper::nifs::circuit::NifsVCircuitMessages;
use crate::paper::params::Params;
use crate::paper::relations::CcsInstance;

/// Verifier-owned application, fixed recursive relation, and lifecycle keys.
pub struct R1csIvcPreprocessing {
    pub prep: Preprocessing,
    relation: R1csIvcRelation,
    app: R1csShape,
    plan: RecursiveStepImagePlan,
}

impl R1csIvcPreprocessing {
    pub fn new(params: Params, app: impl Into<R1csShape>, plan: RecursiveStepImagePlan) -> Result<Self, R1csIvcError> {
        let app = app.into();
        let relation = R1csIvcRelation::compile_fixed_point(&params, &app, &plan)?;
        Self::from_relation(params, app, plan, relation, None)
    }

    #[doc(hidden)]
    pub fn new_seeded(
        params: Params,
        app: impl Into<R1csShape>,
        plan: RecursiveStepImagePlan,
        seed: u64,
    ) -> Result<Self, R1csIvcError> {
        let app = app.into();
        let relation = R1csIvcRelation::compile_fixed_point(&params, &app, &plan)?;
        let log = ajtai::setup_seeded(&params, relation.structure(), seed);
        Self::from_relation(params, app, plan, relation, Some(log))
    }

    fn from_relation(
        params: Params,
        app: R1csShape,
        plan: RecursiveStepImagePlan,
        mut relation: R1csIvcRelation,
        test_log: Option<neo_ajtai::AjtaiSModule>,
    ) -> Result<Self, R1csIvcError> {
        let mode = super::super::semantic_state_mode_for_plan(&plan);
        let initial = super::super::initial_semantic_state_digest_for_plan(&plan);
        let public_input_len = Some(relation.public_input_len());
        let prep = match test_log {
            Some(log) => {
                lifecycle::preprocess_with_test_log(params, relation.structure().clone(), log, public_input_len)
            }
            None => lifecycle::preprocess(params, relation.structure().clone(), public_input_len),
        }?
        .with_terminal_induction()
        .with_semantic_state_mode(mode)
        .with_initial_semantic_state_digest(initial)?;
        relation.bind_preprocessing(&prep)?;
        Ok(Self {
            prep,
            relation,
            app,
            plan,
        })
    }

    pub fn relation(&self) -> &R1csIvcRelation {
        &self.relation
    }

    pub fn app(&self) -> &R1csShape {
        &self.app
    }

    pub fn plan(&self) -> &RecursiveStepImagePlan {
        &self.plan
    }
}

/// Serial `K=1` HyperNova lifecycle over one verifier-owned R1CS app.
pub struct R1csIvc<'a> {
    prep: &'a R1csIvcPreprocessing,
    audit: Option<UncompressedAudit>,
}

/// Exact normalized field assignment synthesized for one accepted IVC step.
///
/// This diagnostic value is not a proof. A consumer must replay it against
/// the exact exported source arm before it is used as a solver background.
#[derive(Debug)]
pub struct R1csIvcConstraintWitnessAudit {
    branch: R1csIvcBranch,
    source_assignment: Vec<F>,
}

impl R1csIvcConstraintWitnessAudit {
    pub fn branch(&self) -> R1csIvcBranch {
        self.branch
    }

    pub fn source_assignment(&self) -> &[F] {
        &self.source_assignment
    }
}

impl<'a> R1csIvc<'a> {
    pub fn new(prep: &'a R1csIvcPreprocessing) -> Self {
        Self { prep, audit: None }
    }

    pub fn extend(&mut self, assignment: Vec<F>) -> Result<(), R1csIvcError> {
        self.extend_with_constraint_witness_audit(assignment)?;
        Ok(())
    }

    /// Extend the lifecycle and return the exact normalized source assignment
    /// used for this accepted step.
    #[doc(hidden)]
    pub fn extend_with_constraint_witness_audit(
        &mut self,
        assignment: Vec<F>,
    ) -> Result<R1csIvcConstraintWitnessAudit, R1csIvcError> {
        self.prep.app.is_satisfied_by(&assignment)?;
        let semantic = semantic_values(&self.prep.plan, &assignment)?;
        let prepared = self.prepare_step(semantic.input, semantic.output)?;
        let SynthesizedSourceInstance {
            branch,
            source_assignment,
            instance,
        } = self.synthesize_instance(&prepared, &assignment)?;
        self.deposit(prepared, instance)?;
        Ok(R1csIvcConstraintWitnessAudit {
            branch,
            source_assignment,
        })
    }

    pub fn audit(&self) -> Option<&UncompressedAudit> {
        self.audit.as_ref()
    }

    pub fn into_audit(self) -> Result<UncompressedAudit, R1csIvcError> {
        self.audit.ok_or(R1csIvcError::EmptyChain)
    }

    /// Produce the compact HyperNova proof. For a plain authoritative F'
    /// relation, finalization keeps `(running, latest)` separate; the verifier
    /// checks both satisfactions instead of replaying history or adding a
    /// terminal fold.
    pub fn finish(self) -> Result<Uncompressed, R1csIvcError> {
        Ok(lifecycle::finish_uncompressed(&self.prep.prep, self.into_audit()?)?)
    }

    fn prepare_step(
        &mut self,
        semantic_input: Option<[F; 4]>,
        semantic_output: Option<[F; 4]>,
    ) -> Result<PreparedStep, R1csIvcError> {
        let Some(audit) = self.audit.take() else {
            let pre = StateCoordinates::base(&self.prep.prep);
            if semantic_input.is_some_and(|input| input != pre.semantic_state_digest) {
                return Err(R1csIvcError::SemanticInputMismatch);
            }
            let post = pre.base_advance(&self.prep.prep, semantic_output)?;
            return Ok(PreparedStep::Base { pre, post });
        };

        let pre = StateCoordinates::from_state(&audit.proof.state);
        if semantic_input.is_some_and(|input| input != pre.semantic_state_digest) {
            self.audit = Some(audit);
            return Err(R1csIvcError::SemanticInputMismatch);
        }
        let branch = if pre.step_count <= 1 {
            R1csIvcBranch::BootstrapRecursive
        } else {
            R1csIvcBranch::Recursive
        };
        let semantic_advance = if let Some(output) = semantic_output {
            SemanticStateAdvance::Stateful(digest_fields_as_digest32(output))
        } else {
            SemanticStateAdvance::Stateless
        };
        let fold = crate::lifecycle::prove::prepare_recursive_step(&self.prep.prep, audit, semantic_advance)?;
        let nifs = fold.nifs_proof()?;
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
        assignment: &[F],
    ) -> Result<SynthesizedSourceInstance, R1csIvcError> {
        #[cfg(feature = "perf-timers")]
        let synth_start = std::time::Instant::now();
        let cfg = FPrimeStepConfig {
            nifs: self.prep.prep.nifs_v_circuit_config()?,
            b: self.prep.prep.params.b(),
            transcript_label: F_PRIME_STEP_TRANSCRIPT_LABEL,
            public_input_layout: FPrimePublicInputLayout::plain(),
            nebula: None,
            state_x_out_digest_mode: digest_mode(&self.prep.plan),
        };
        let pre = prepared.pre();
        let post = prepared.post();
        let mut source = FPrimeSourceImage::new();
        let chunk_count_in_word = source.push_u64_le(pre.chunk_count);
        let step_count_in_word = source.push_u64_le(pre.step_count);
        let pc_word = source.push_u64_le(pre.pc);
        let mut builder = R1csBuilder::new_witness_only();

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
                let output =
                    enforce_base_application(&mut builder, &self.prep.app, assignment, &self.prep.plan, &cfg, &inputs)?;
                (R1csIvcBranch::Base, output)
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
                let output = enforce_recursive_application(
                    &mut builder,
                    &self.prep.prep.params,
                    &self.prep.app,
                    assignment,
                    &self.prep.plan,
                    &cfg,
                    &inputs,
                )?;
                (*branch, output)
            }
        };
        #[cfg(feature = "perf-timers")]
        let enforce_elapsed = synth_start.elapsed();

        let expected = self.prep.relation.arm_shape(branch);
        let public_outputs = output.x_out_bits.clone();
        let actual = (builder.rows(), builder.witness().len(), 1 + public_outputs.len());
        if actual != (expected.rows, expected.columns, expected.public_columns) {
            return Err(R1csIvcError::ArmShapeMismatch {
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
        let normalize_start = std::time::Instant::now();
        let field_assignment = normalized_field_assignment(&builder, &public_outputs)?;
        #[cfg(feature = "perf-timers")]
        let normalize_elapsed = normalize_start.elapsed();
        #[cfg(feature = "perf-timers")]
        let instance_start = std::time::Instant::now();
        let instance = self
            .prep
            .relation
            .build_instance(&self.prep.prep, branch, &field_assignment)?;
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-ivc-synth] witness {:>7.2}s normalize {:>7.2}s encode+instance {:>7.2}s total {:>7.2}s",
            enforce_elapsed.as_secs_f64(),
            normalize_elapsed.as_secs_f64(),
            instance_start.elapsed().as_secs_f64(),
            synth_start.elapsed().as_secs_f64(),
        );
        Ok(SynthesizedSourceInstance {
            branch,
            source_assignment: field_assignment,
            instance,
        })
    }

    fn deposit(&mut self, prepared: PreparedStep, instance: CcsInstance) -> Result<(), R1csIvcError> {
        self.audit = Some(match prepared {
            PreparedStep::Base { post, .. } => {
                if matches!(self.prep.prep.semantic_state_mode(), SemanticStateMode::Stateful) {
                    crate::lifecycle::prove::prove_one_with_semantic_state(
                        &self.prep.prep,
                        vec![instance],
                        self.prep.prep.initial_semantic_state_digest(),
                        digest_fields_as_digest32(post.semantic_state_digest),
                    )?
                } else {
                    lifecycle::prove(&self.prep.prep, [vec![instance]])?
                }
            }
            PreparedStep::Recursive { fold, .. } => fold.complete(instance)?,
        });
        Ok(())
    }
}

struct SynthesizedSourceInstance {
    branch: R1csIvcBranch,
    source_assignment: Vec<F>,
    instance: CcsInstance,
}

enum PreparedStep {
    Base {
        pre: StateCoordinates,
        post: StateCoordinates,
    },
    Recursive {
        branch: R1csIvcBranch,
        pre: StateCoordinates,
        post: StateCoordinates,
        nifs: crate::paper::nifs::NifsProof,
        fold: crate::lifecycle::prove::PreparedRecursiveStep,
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
        }
    }

    fn base_advance(&self, prep: &Preprocessing, semantic_output: Option<[F; 4]>) -> Result<Self, R1csIvcError> {
        let z_i = f_prime_chunk_public_digest_for_uniform_shape(
            self.step_count,
            1,
            D,
            prep.params.kappa() as usize,
            prep.public_input_len
                .expect("R1CS IVC fixes public input length"),
        );
        let m_in = prep
            .public_input_len
            .expect("R1CS IVC fixes public input length");
        let running = crate::paper::construction2::RunningInstance::canonical_zero(
            &prep.params,
            prep.structure(),
            m_in,
            LaneCommitmentMode::Plain,
        )
        .map_err(crate::paper::construction2::Error::from)
        .map_err(lifecycle::Error::from)?;
        let acc_digest = digest32_as_fields(
            running
                .accumulator_digest(prep.params.b(), prep.structure())
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
            nebula: None,
        }
    }

    fn x_out_fields(&self, prep: &Preprocessing) -> [F; 4] {
        digest32_as_fields(state_x_out_digest_with_mode(
            match prep.semantic_state_mode() {
                SemanticStateMode::Stateless => crate::paper::digest::StateXOutDigestMode::Stateless,
                SemanticStateMode::Stateful => crate::paper::digest::StateXOutDigestMode::Stateful,
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
            None,
        ))
    }
}
