//! Field-R1CS source arms for the phased Nebula F-prime lifecycle.
//!
//! Owns the complete base or recursive F-prime verifier, verifier-key advice
//! replay, the private delayed Nebula input, and the 640 logical public fields
//! shared with one selected phase. It does not own the phase relation, the
//! private delayed-input link to that phase, selective lowering, a fixed point,
//! or terminal acceptance.

use std::ops::Range;

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

use super::streaming_phase_envelope::{enforce_streaming_phase_semantic_digest, streaming_phase_semantic_digest};
use super::streaming_public::NebulaFPrimeStreamingPublicLayout;
use super::streaming_state_envelope::enforce_streaming_state_x_out;
use super::{relation_config, NebulaFPrimeError, NebulaFPrimeRelationError};
use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, normalized_field_column, SparseR1cs};
use crate::lifecycle::Preprocessing;
use crate::paper::construction2::{
    running::zero_lane_commitments, LaneCommitmentMode, LatestInstance, NebulaConfig, NebulaLane, ProofState,
    RunningInstance, State,
};
use crate::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest_for_uniform_shape,
    initial_boundary_digest, public_trace_seed_digest, state_x_out_digest_from_preimage, AccumulatorHandle,
    StateXOutDigestMode,
};
use crate::paper::f_prime::digest_circuit::{
    alloc_constant, enforce_initial_boundary_digest_circuit, enforce_public_trace_seed_digest_circuit,
    enforce_vk_fs_digest_circuit, enforce_vk_fs_policy_digest_circuit, StateXOutDigestInputs,
};
use crate::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL;
use crate::paper::f_prime::nebula_lane_circuit::{
    delayed_nebula_public_suffix_len, enforce_nebula_lane_constant_circuit, enforce_nebula_lane_digest_selected_circuit,
};
use crate::paper::f_prime::r1cs::{
    encode_x_out_public_bits, enforce_f_prime_base_step_circuit,
    enforce_f_prime_recursive_step_circuit_with_private_nebula_input, FPrimeBaseInputs, FPrimeRecursiveInputs,
    FPrimeStateIn, FPrimeStateWires, FPrimeStepConfig, FPrimeStepOutput, F_PRIME_ENC_INST_BITS,
};
use crate::paper::f_prime::source_image::{BitRange, FPrimeSourceImage};
use crate::paper::f_prime::stage as fprime_stage;
use crate::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs;
use crate::paper::reductions::pi_ccs_circuit::{PiCcsVerifierConfig, PiCcsVerifierRelation};
use crate::paper::relations::{CcsClaim, CcsInstance, CeClaim};

const LOGICAL_PUBLIC_OUTPUTS: usize = 640;
const X_OUT_PREIMAGE_FIELDS: usize = 32;
const PRIVATE_DELAYED_INPUT_FAMILY: &str = "fprime.recursive.nebula.private_delayed_input.raw_bits";
const BASE_BEFORE_PAYLOAD_FAMILY: &str = "fprime.streaming.base.phase.before.delayed_payload.raw_bits";
const BASE_BEFORE_LOCAL_FAMILY: &str = "fprime.streaming.base.phase.before.local_state_digest";
const BASE_AFTER_LOCAL_FAMILY: &str = "fprime.streaming.base.phase.after.local_state_digest";
const BASE_AFTER_PAYLOAD_FAMILY: &str = "fprime.streaming.base.phase.after.delayed_payload.raw_bits";
const RECURSIVE_BEFORE_LOCAL_FAMILY: &str = "fprime.streaming.recursive.phase.before.local_state_digest";
const RECURSIVE_AFTER_LOCAL_FAMILY: &str = "fprime.streaming.recursive.phase.after.local_state_digest";
const RECURSIVE_AFTER_PAYLOAD_FAMILY: &str = "fprime.streaming.recursive.phase.after.delayed_payload.raw_bits";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimeStreamingLifecycleArm {
    Base,
    Recursive,
}

impl NebulaFPrimeStreamingLifecycleArm {
    const fn index(self) -> usize {
        match self {
            Self::Base => 0,
            Self::Recursive => 1,
        }
    }
}

/// Exact field-R1CS source rows before selective low-norm lowering.
pub struct NebulaFPrimeStreamingLifecycleSourceArms {
    arms: [SparseR1cs; 2],
    base_assignment: Vec<F>,
    recursive_assignment: Option<Vec<F>>,
    recursive_delayed_input_fields: Range<usize>,
    phase_envelope_fields: [NebulaFPrimeStreamingPhaseEnvelopeFields; 2],
    x_out_preimage_columns: [NebulaFPrimeStreamingXOutPreimageColumns; 2],
    after_nebula_lane_columns: [NebulaFPrimeStreamingLaneSourceColumns; 2],
}

/// Exact normalized source columns consumed by the before-state and
/// after-state `x_out` Poseidon2 rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingXOutPreimageColumns {
    before: [usize; X_OUT_PREIMAGE_FIELDS],
    after: [usize; X_OUT_PREIMAGE_FIELDS],
}

impl NebulaFPrimeStreamingXOutPreimageColumns {
    pub fn before(&self) -> &[usize; X_OUT_PREIMAGE_FIELDS] {
        &self.before
    }

    pub fn after(&self) -> &[usize; X_OUT_PREIMAGE_FIELDS] {
        &self.after
    }
}

/// Exact normalized source columns for one post-step Nebula lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLaneSourceColumns {
    pub program_binding_digest: [usize; 4],
    pub open: usize,
    pub seg_idx: usize,
    pub idx: usize,
    pub ts: usize,
    pub gamma: [[usize; 2]; 2],
    pub h: [[usize; 2]; 4],
    pub sp: [usize; 2],
    pub d_pre: [[usize; 4]; 3],
    pub d_seen: [[usize; 4]; 3],
    pub d_mem: [usize; 4],
}

impl NebulaFPrimeStreamingLaneSourceColumns {
    pub fn all(&self) -> Vec<usize> {
        let mut columns = Vec::with_capacity(50);
        columns.extend(self.program_binding_digest);
        columns.extend([self.open, self.seg_idx, self.idx, self.ts]);
        columns.extend(self.gamma.into_iter().flatten());
        columns.extend(self.h.into_iter().flatten());
        columns.extend(self.sp);
        columns.extend(self.d_pre.into_iter().flatten());
        columns.extend(self.d_seen.into_iter().flatten());
        columns.extend(self.d_mem);
        debug_assert_eq!(columns.len(), 50);
        columns
    }
}

/// Exact normalized source fields that the selected phase must own and link.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingPhaseEnvelopeFields {
    before_local_state_digest: Range<usize>,
    before_delayed_payload: Range<usize>,
    after_local_state_digest: Range<usize>,
    after_delayed_payload: Range<usize>,
}

impl NebulaFPrimeStreamingPhaseEnvelopeFields {
    pub fn before_local_state_digest(&self) -> Range<usize> {
        self.before_local_state_digest.clone()
    }

    pub fn before_delayed_payload(&self) -> Range<usize> {
        self.before_delayed_payload.clone()
    }

    pub fn after_local_state_digest(&self) -> Range<usize> {
        self.after_local_state_digest.clone()
    }

    pub fn after_delayed_payload(&self) -> Range<usize> {
        self.after_delayed_payload.clone()
    }
}

impl NebulaFPrimeStreamingLifecycleSourceArms {
    pub fn arm(&self, arm: NebulaFPrimeStreamingLifecycleArm) -> &SparseR1cs {
        &self.arms[arm.index()]
    }

    pub fn into_arms(self) -> [SparseR1cs; 2] {
        self.arms
    }

    /// Exact normalized satisfying Rust assignment for the base source arm.
    pub fn base_assignment(&self) -> &[F] {
        &self.base_assignment
    }

    /// Exact normalized satisfying Rust assignment for the recursive arm.
    /// Shape-only synthesis returns `None`; proof-backed synthesis returns it.
    pub fn recursive_assignment(&self) -> Option<&[F]> {
        self.recursive_assignment.as_deref()
    }

    /// Exact normalized source-field slice for
    /// `[step_x_bits || open || bits(D_pre)]` in the recursive arm.
    pub fn recursive_delayed_input_fields(&self) -> Range<usize> {
        self.recursive_delayed_input_fields.clone()
    }

    pub fn phase_envelope_fields(
        &self,
        arm: NebulaFPrimeStreamingLifecycleArm,
    ) -> &NebulaFPrimeStreamingPhaseEnvelopeFields {
        &self.phase_envelope_fields[arm.index()]
    }

    pub fn x_out_preimage_columns(
        &self,
        arm: NebulaFPrimeStreamingLifecycleArm,
    ) -> &NebulaFPrimeStreamingXOutPreimageColumns {
        &self.x_out_preimage_columns[arm.index()]
    }

    pub fn after_nebula_lane_columns(
        &self,
        arm: NebulaFPrimeStreamingLifecycleArm,
    ) -> &NebulaFPrimeStreamingLaneSourceColumns {
        &self.after_nebula_lane_columns[arm.index()]
    }
}

struct FinalizedPublic {
    outputs: Vec<Var>,
    before_x_out_preimage: Vec<Var>,
    after_x_out_preimage: Vec<Var>,
    after_nebula_lane: crate::paper::f_prime::nebula_lane_circuit::NebulaLaneWires,
}

struct SynthesizedLifecycleArm {
    source: SparseR1cs,
    assignment: Vec<F>,
    x_out_preimage_columns: NebulaFPrimeStreamingXOutPreimageColumns,
    after_nebula_lane_columns: NebulaFPrimeStreamingLaneSourceColumns,
}

struct ShapeContext<'a> {
    params: &'a Params,
    folded: PiCcsVerifierRelation,
    config: NebulaConfig,
    structure_digest: [F; 4],
    matrix_digest: [F; 4],
    ajtai_pp_digest: [F; 4],
    vk_fs_digest: [F; 4],
    joint_variables: usize,
    joint_degree: usize,
    public: NebulaFPrimeStreamingPublicLayout,
}

/// Bind generic verifier preprocessing to the fixed streaming lifecycle
/// policy and the exact Nebula plan.
pub fn prepare_streaming_lifecycle_preprocessing(
    preprocessing: Preprocessing,
    plan: &NebulaPlan,
) -> Result<Preprocessing, NebulaFPrimeRelationError> {
    let public = NebulaFPrimeStreamingPublicLayout::production();
    if preprocessing.public_input_len != Some(public.columns()) {
        return Err(NebulaFPrimeRelationError::Geometry(format!(
            "streaming lifecycle preprocessing public width {:?} != {}",
            preprocessing.public_input_len,
            public.columns()
        )));
    }
    let payload_len = delayed_nebula_public_suffix_len(plan.config().stacks);
    let initial_semantic_state_digest = streaming_phase_semantic_digest([F::ZERO; 4], &vec![F::ZERO; payload_len]);
    preprocessing
        .with_initial_semantic_state_digest(digest_fields_as_digest32(initial_semantic_state_digest))
        .map(|preprocessing| {
            let mut config = plan.config();
            config.initial_semantic_state_digest = initial_semantic_state_digest;
            preprocessing
                .with_semantic_state_mode(crate::paper::construction2::SemanticStateMode::Stateful)
                .with_terminal_induction()
                .with_nebula(config)
        })
        .map_err(|error| {
            NebulaFPrimeRelationError::Geometry(format!("streaming lifecycle preprocessing policy: {error}"))
        })
}

#[derive(Clone, Copy)]
struct VerifierAdvice {
    structure_digest: [F; 4],
    ajtai_pp_digest: [F; 4],
    initial_semantic_state_digest: [F; 4],
}

/// Synthesize the two complete lifecycle source arms for one verifier-owned
/// folded relation. The base assignment is satisfying. The recursive rows are
/// exact, but a recursive assignment requires a real NIFS proof and is not
/// supplied by this shape constructor.
///
/// The returned rows are not a frozen profile until selective compilation
/// reaches its checked fixed point.
pub fn synthesize_streaming_lifecycle_source_arms(
    preprocessing: &Preprocessing,
    plan: &NebulaPlan,
) -> Result<NebulaFPrimeStreamingLifecycleSourceArms, NebulaFPrimeRelationError> {
    let context = streaming_shape_context(preprocessing, plan)?;
    let base = synthesize_base(&context)?;
    let recursive = synthesize_recursive_shape(&context)?;
    assemble_lifecycle_source_arms(
        base,
        recursive,
        false,
        delayed_nebula_public_suffix_len(context.config.stacks),
    )
}

/// Synthesize the same exact source rows with a real recursive NIFS proof and
/// retain both satisfying assignments.
pub fn synthesize_streaming_lifecycle_source_arms_with_recursive_assignment(
    preprocessing: &Preprocessing,
    plan: &NebulaPlan,
    fresh: &CcsInstance,
    private_delayed: &[F],
) -> Result<NebulaFPrimeStreamingLifecycleSourceArms, NebulaFPrimeRelationError> {
    let context = streaming_shape_context(preprocessing, plan)?;
    let expected_delayed = delayed_nebula_public_suffix_len(context.config.stacks);
    if private_delayed.len() != expected_delayed {
        return Err(NebulaFPrimeRelationError::Geometry(format!(
            "streaming recursive delayed input width {} != {expected_delayed}",
            private_delayed.len()
        )));
    }
    let base = synthesize_base(&context)?;
    let recursive = synthesize_recursive_with_instance(&context, preprocessing, plan, fresh, private_delayed)?;
    assemble_lifecycle_source_arms(base, recursive, true, expected_delayed)
}

/// Emit only the deterministic source rows needed during verifier-shape
/// fixed-point discovery. Placeholder advice values do not cross a trust
/// boundary and the returned assignments are discarded.
pub(super) fn synthesize_streaming_lifecycle_source_arm_shapes(
    params: &Params,
    folded: PiCcsVerifierRelation,
    plan: &NebulaPlan,
) -> Result<[SparseR1cs; 2], NebulaFPrimeRelationError> {
    let context = streaming_shape_context_for_folded(params, folded, plan)?;
    let base = synthesize_base(&context)?;
    let recursive = synthesize_recursive_shape(&context)?;
    Ok([base.source, recursive.source])
}

fn streaming_shape_context<'a>(
    preprocessing: &'a Preprocessing,
    plan: &NebulaPlan,
) -> Result<ShapeContext<'a>, NebulaFPrimeRelationError> {
    validate_streaming_preprocessing(preprocessing, plan)?;
    let params = &preprocessing.params;
    let folded = PiCcsVerifierRelation::from_structure(preprocessing.structure());
    streaming_shape_context_from_parts(
        params,
        folded,
        plan,
        *preprocessing.structure_digest(),
        preprocessing.pi_ccs_header_bundle(),
        preprocessing.ajtai_pp_digest(),
        digest32_as_fields(preprocessing.vk.digest()),
    )
}

fn streaming_shape_context_for_folded<'a>(
    params: &'a Params,
    folded: PiCcsVerifierRelation,
    plan: &NebulaPlan,
) -> Result<ShapeContext<'a>, NebulaFPrimeRelationError> {
    streaming_shape_context_from_parts(
        params,
        folded,
        plan,
        [F::ZERO; 4],
        [F::ZERO; 4],
        [F::ZERO; 4],
        [F::ZERO; 4],
    )
}

#[allow(clippy::too_many_arguments)]
fn streaming_shape_context_from_parts<'a>(
    params: &'a Params,
    folded: PiCcsVerifierRelation,
    plan: &NebulaPlan,
    structure_digest: [F; 4],
    matrix_digest: [F; 4],
    ajtai_pp_digest: [F; 4],
    vk_fs_digest: [F; 4],
) -> Result<ShapeContext<'a>, NebulaFPrimeRelationError> {
    if !params.has_production_core() {
        return Err(NebulaFPrimeRelationError::Geometry(
            "streaming lifecycle shape does not use the SuperNeo Appendix B.2 Goldilocks core".into(),
        ));
    }
    let dims = neo_reductions::engines::pi_ccs_joint::build_joint_dims_for_shape(
        params.inner(),
        folded.n(),
        folded.m(),
        folded.t(),
        folded.max_degree(),
        1,
        params.k_rho() as usize,
    )
    .map_err(|error| NebulaFPrimeRelationError::Geometry(format!("streaming verifier dimensions: {error}")))?;
    let mut config = relation_config(plan, None);
    let delayed_len = delayed_nebula_public_suffix_len(config.stacks);
    let zero_payload = vec![F::ZERO; delayed_len];
    config.initial_semantic_state_digest = streaming_phase_semantic_digest([F::ZERO; 4], &zero_payload);
    Ok(ShapeContext {
        params,
        folded,
        config,
        structure_digest,
        matrix_digest,
        ajtai_pp_digest,
        vk_fs_digest,
        joint_variables: dims.variables,
        joint_degree: dims.degree,
        public: NebulaFPrimeStreamingPublicLayout::production(),
    })
}

fn assemble_lifecycle_source_arms(
    base: SynthesizedLifecycleArm,
    mut recursive: SynthesizedLifecycleArm,
    retain_recursive_assignment: bool,
    delayed_len: usize,
) -> Result<NebulaFPrimeStreamingLifecycleSourceArms, NebulaFPrimeRelationError> {
    let recursive_delayed_input_fields =
        exact_column_family(&recursive.source, PRIVATE_DELAYED_INPUT_FAMILY, delayed_len)?;
    if recursive_delayed_input_fields.start < recursive.source.m_in {
        return Err(NebulaFPrimeRelationError::Geometry(
            "streaming delayed Nebula input overlaps the public prefix".into(),
        ));
    }
    if base
        .source
        .column_family_ranges()
        .iter()
        .any(|family| family.name == PRIVATE_DELAYED_INPUT_FAMILY)
    {
        return Err(NebulaFPrimeRelationError::Geometry(
            "base lifecycle arm unexpectedly owns delayed Nebula input".into(),
        ));
    }
    let phase_envelope_fields = [
        exact_phase_envelope_fields(&base.source, NebulaFPrimeStreamingLifecycleArm::Base, delayed_len)?,
        exact_phase_envelope_fields(
            &recursive.source,
            NebulaFPrimeStreamingLifecycleArm::Recursive,
            delayed_len,
        )?,
    ];
    if phase_envelope_fields[NebulaFPrimeStreamingLifecycleArm::Recursive.index()].before_delayed_payload
        != recursive_delayed_input_fields
    {
        return Err(NebulaFPrimeRelationError::Geometry(
            "recursive phase envelope does not reuse the exact delayed Nebula input".into(),
        ));
    }
    let recursive_assignment = retain_recursive_assignment.then(|| std::mem::take(&mut recursive.assignment));
    Ok(NebulaFPrimeStreamingLifecycleSourceArms {
        arms: [base.source, recursive.source],
        base_assignment: base.assignment,
        recursive_assignment,
        recursive_delayed_input_fields,
        phase_envelope_fields,
        x_out_preimage_columns: [base.x_out_preimage_columns, recursive.x_out_preimage_columns],
        after_nebula_lane_columns: [base.after_nebula_lane_columns, recursive.after_nebula_lane_columns],
    })
}

fn validate_streaming_preprocessing(
    preprocessing: &Preprocessing,
    plan: &NebulaPlan,
) -> Result<(), NebulaFPrimeRelationError> {
    if !preprocessing.params.has_production_core() {
        return Err(NebulaFPrimeRelationError::Geometry(
            "streaming lifecycle preprocessing does not use the SuperNeo Appendix B.2 Goldilocks core".into(),
        ));
    }
    let public = NebulaFPrimeStreamingPublicLayout::production();
    if preprocessing.public_input_len != Some(public.columns()) {
        return Err(NebulaFPrimeRelationError::Geometry(format!(
            "streaming lifecycle preprocessing public width {:?} != {}",
            preprocessing.public_input_len,
            public.columns()
        )));
    }
    if preprocessing.semantic_state_mode() != crate::paper::construction2::SemanticStateMode::Stateful {
        return Err(NebulaFPrimeRelationError::Geometry(
            "streaming lifecycle preprocessing is not stateful".into(),
        ));
    }
    if !preprocessing.enforces_terminal_induction() {
        return Err(NebulaFPrimeRelationError::Geometry(
            "streaming lifecycle preprocessing lacks recursive terminal induction".into(),
        ));
    }
    let expected = plan.config();
    let actual = preprocessing.nebula().ok_or_else(|| {
        NebulaFPrimeRelationError::Geometry("streaming lifecycle preprocessing has no Nebula plan".into())
    })?;
    if actual.steps_per_segment != expected.steps_per_segment
        || actual.seg_max != expected.seg_max
        || actual.stacks != expected.stacks
        || actual.plan_digest != expected.plan_digest
        || actual.d_init != expected.d_init
    {
        return Err(NebulaFPrimeRelationError::Geometry(
            "streaming lifecycle preprocessing differs from the exact Nebula plan".into(),
        ));
    }
    let expected_initial = digest_fields_as_digest32(actual.initial_semantic_state_digest);
    if preprocessing.initial_semantic_state_digest() != expected_initial {
        return Err(NebulaFPrimeRelationError::Geometry(
            "streaming lifecycle preprocessing has a different initial semantic state".into(),
        ));
    }
    Ok(())
}

fn exact_phase_envelope_fields(
    arm: &SparseR1cs,
    kind: NebulaFPrimeStreamingLifecycleArm,
    payload_len: usize,
) -> Result<NebulaFPrimeStreamingPhaseEnvelopeFields, NebulaFPrimeRelationError> {
    let (before_local, before_payload, after_local, after_payload) = match kind {
        NebulaFPrimeStreamingLifecycleArm::Base => (
            BASE_BEFORE_LOCAL_FAMILY,
            BASE_BEFORE_PAYLOAD_FAMILY,
            BASE_AFTER_LOCAL_FAMILY,
            BASE_AFTER_PAYLOAD_FAMILY,
        ),
        NebulaFPrimeStreamingLifecycleArm::Recursive => (
            RECURSIVE_BEFORE_LOCAL_FAMILY,
            PRIVATE_DELAYED_INPUT_FAMILY,
            RECURSIVE_AFTER_LOCAL_FAMILY,
            RECURSIVE_AFTER_PAYLOAD_FAMILY,
        ),
    };
    Ok(NebulaFPrimeStreamingPhaseEnvelopeFields {
        before_local_state_digest: exact_column_family(arm, before_local, 4)?,
        before_delayed_payload: exact_column_family(arm, before_payload, payload_len)?,
        after_local_state_digest: exact_column_family(arm, after_local, 4)?,
        after_delayed_payload: exact_column_family(arm, after_payload, payload_len)?,
    })
}

fn exact_column_family(
    arm: &SparseR1cs,
    name: &'static str,
    expected_len: usize,
) -> Result<Range<usize>, NebulaFPrimeRelationError> {
    let mut matches = arm
        .column_family_ranges()
        .iter()
        .filter(|family| family.name == name);
    let family = matches
        .next()
        .ok_or_else(|| NebulaFPrimeRelationError::Geometry(format!("streaming lifecycle is missing {name}")))?;
    if matches.next().is_some() {
        return Err(NebulaFPrimeRelationError::Geometry(format!(
            "streaming lifecycle contains duplicate {name} ranges"
        )));
    }
    let range = family.column_start..family.column_end;
    if range.len() != expected_len {
        return Err(NebulaFPrimeRelationError::Geometry(format!(
            "streaming lifecycle {name} width {} != {expected_len}",
            range.len()
        )));
    }
    Ok(range)
}

fn synthesize_base(context: &ShapeContext<'_>) -> Result<SynthesizedLifecycleArm, NebulaFPrimeRelationError> {
    let advice = shape_verifier_advice(context);
    let semantic_digest = context.config.initial_semantic_state_digest;
    let state = shape_state(
        context,
        advice,
        false,
        semantic_digest,
        AccumulatorHandle::empty().digest_fields(),
    );
    let mut source = FPrimeSourceImage::new();
    let chunk_count_in_word = source.push_u64_le(0);
    let step_count_in_word = source.push_u64_le(0);
    let pc_word = source.push_u64_le(1);
    let public_x_out_bits = source.push_enc_inst([F::ZERO; 4]);
    let inputs = FPrimeBaseInputs {
        state,
        chunk_digest: chunk_digest(context, 0),
        semantic_state_digest_out: semantic_digest,
        rows_in_chunk: 1,
        source_image: &source,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        public_x_out_bits,
    };
    let cfg = step_config(context);
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage(fprime_stage::BASE_ROOT);
    let output = enforce_f_prime_base_step_circuit(&mut builder, &cfg, &inputs).map_err(NebulaFPrimeError::from)?;
    let public = finalize_public(
        &mut builder,
        context,
        advice,
        NebulaFPrimeStreamingLifecycleArm::Base,
        &output,
    )?;
    let x_out_preimage_columns = exact_x_out_preimage_columns(builder.cols(), &public)?;
    let after_nebula_lane_columns =
        exact_lane_source_columns(builder.cols(), &public.outputs, &public.after_nebula_lane)?;
    let (source, mut assignment) = lower_field_r1cs(builder, &public.outputs)?.into_parts();
    populate_public_x_out_bits(
        &mut assignment,
        1,
        x_out_preimage_columns.after(),
        "base after-state x_out",
    )?;
    Ok(SynthesizedLifecycleArm {
        source,
        assignment,
        x_out_preimage_columns,
        after_nebula_lane_columns,
    })
}

fn synthesize_recursive_shape(
    context: &ShapeContext<'_>,
) -> Result<SynthesizedLifecycleArm, NebulaFPrimeRelationError> {
    let advice = shape_verifier_advice(context);
    let public_input_len = context.public.columns();
    let ce = zero_ce_claim(context, public_input_len);
    let running = vec![ce.clone(); context.params.k_rho() as usize];
    let running_parent = Some(ce.clone());
    let fresh = [zero_fresh_claim(context, public_input_len)];
    let outputs = vec![ce.clone(); fresh.len() + running.len()];
    let sumcheck = pi_ccs::SumcheckProof::new(vec![vec![K::ZERO; context.joint_degree + 1]; context.joint_variables]);
    let proof = pi_ccs::Proof {
        outputs_digest: crate::paper::digest::pi_ccs_outputs_digest(&outputs),
        outputs,
        sumcheck,
    };
    let combined = ce.clone();
    let children = vec![ce; context.params.k_rho() as usize];
    let running_digest =
        AccumulatorHandle::from_running_parts(context.params.b(), &running, running_parent.as_ref()).digest_fields();
    let output_digest =
        AccumulatorHandle::from_running_parts(context.params.b(), &children, Some(&combined)).digest_fields();
    let semantic_digest = context.config.initial_semantic_state_digest;
    let state = shape_state(context, advice, true, semantic_digest, running_digest);
    let private_delayed = vec![F::ZERO; delayed_nebula_public_suffix_len(context.config.stacks)];
    synthesize_recursive_from_messages(
        context,
        advice,
        state,
        &fresh,
        &running,
        running_parent.as_ref(),
        &proof,
        &combined,
        &children,
        output_digest,
        &private_delayed,
    )
}

fn synthesize_recursive_with_instance(
    context: &ShapeContext<'_>,
    preprocessing: &Preprocessing,
    plan: &NebulaPlan,
    fresh: &CcsInstance,
    private_delayed: &[F],
) -> Result<SynthesizedLifecycleArm, NebulaFPrimeRelationError> {
    let public_input_len = context.public.columns();
    if fresh.claim.m_in != public_input_len {
        return Err(NebulaFPrimeRelationError::Geometry(format!(
            "streaming recursive fresh public width {} != {public_input_len}",
            fresh.claim.m_in
        )));
    }
    if fresh.claim.adv.is_none() {
        return Err(NebulaFPrimeRelationError::Geometry(
            "streaming recursive fresh claim has no Nebula commitment sidecar".into(),
        ));
    }
    let running = RunningInstance::canonical_zero(
        context.params,
        preprocessing.structure(),
        public_input_len,
        LaneCommitmentMode::Nebula,
    )
    .map_err(|error| NebulaFPrimeRelationError::Geometry(format!("streaming recursive zero accumulator: {error}")))?;
    let running_digest =
        AccumulatorHandle::from_running_parts(context.params.b(), &running.claims, running.parent_authority.as_ref())
            .digest_fields();
    let advice = shape_verifier_advice(context);
    let semantic_digest = streaming_phase_semantic_digest([F::ZERO; 4], private_delayed);
    let state = shape_state(context, advice, true, semantic_digest, running_digest);
    let native_state = State {
        chunk_count: state.chunk_count_in,
        step_count: state.step_count_in,
        z_0: digest_fields_as_digest32(state.z_0),
        z_i: digest_fields_as_digest32(state.z_i_in),
        initial_semantic_state_digest: digest_fields_as_digest32(context.config.initial_semantic_state_digest),
        semantic_state_digest: digest_fields_as_digest32(state.semantic_state_digest_in),
        pc: state.pc,
        acc_digest: digest_fields_as_digest32(state.acc_digest_in),
        public_trace: digest_fields_as_digest32(state.public_trace_in),
        proof: ProofState::active(running.clone(), LatestInstance::from_instances(vec![fresh.clone()])),
        nebula: Some(NebulaLane::base(&context.config)),
    };
    let mut transcript = crate::paper::f_prime::native::f_prime_step_transcript(
        &preprocessing.vk,
        preprocessing.structure_digest(),
        &native_state,
        chunk_digest(context, 1),
    );
    let (output_running, proof) = crate::paper::nifs::prove(
        &mut transcript,
        context.params,
        preprocessing.structure(),
        preprocessing.optimized_cache(),
        &preprocessing.log,
        Some(plan.scheme()),
        preprocessing.mix_rhos_commits(),
        preprocessing.combine_b_pows(),
        vec![fresh.clone()],
        &running,
    )
    .map_err(|error| NebulaFPrimeRelationError::Geometry(format!("streaming recursive NIFS proof: {error}")))?;
    let output_digest = AccumulatorHandle::from_running_parts(
        context.params.b(),
        &output_running.claims,
        output_running.parent_authority.as_ref(),
    )
    .digest_fields();
    synthesize_recursive_from_messages(
        context,
        advice,
        state,
        std::slice::from_ref(&fresh.claim),
        &running.claims,
        running.parent_authority.as_ref(),
        &proof.pi_ccs,
        &proof.pi_rlc.combined,
        &output_running.claims,
        output_digest,
        private_delayed,
    )
}

#[allow(clippy::too_many_arguments)]
fn synthesize_recursive_from_messages(
    context: &ShapeContext<'_>,
    advice: VerifierAdvice,
    state: FPrimeStateIn,
    fresh: &[CcsClaim],
    running: &[CeClaim],
    running_parent: Option<&CeClaim>,
    proof: &pi_ccs::Proof,
    combined: &CeClaim,
    children: &[CeClaim],
    output_digest: [F; 4],
    private_delayed: &[F],
) -> Result<SynthesizedLifecycleArm, NebulaFPrimeRelationError> {
    let nifs_msg = NifsVCircuitMessages {
        fresh,
        running,
        running_parent_authority: running_parent,
        pi_ccs: proof,
        combined,
        children,
    };
    let semantic_digest = context.config.initial_semantic_state_digest;
    let mut source = FPrimeSourceImage::new();
    let chunk_count_in_word = source.push_u64_le(1);
    let step_count_in_word = source.push_u64_le(1);
    let pc_word = source.push_u64_le(1);
    let prior_public = source.push_f_prime_public_input([F::ZERO; 4]);
    let prior_x_out_bits = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
    let public_x_out_bits = source.push_enc_inst([F::ZERO; 4]);
    let inputs = FPrimeRecursiveInputs {
        state,
        chunk_digest: chunk_digest(context, 1),
        semantic_state_digest_out: semantic_digest,
        acc_digest_out: output_digest,
        nifs_msg,
        rows_in_chunk: 1,
        source_image: &source,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        prior_x_out_bits,
        public_x_out_bits,
    };
    let cfg = step_config(context);
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage(fprime_stage::RECURSIVE_ROOT);
    let output = enforce_f_prime_recursive_step_circuit_with_private_nebula_input(
        &mut builder,
        context.params,
        &cfg,
        &inputs,
        private_delayed,
    )
    .map_err(NebulaFPrimeError::from)?;
    let public = finalize_public(
        &mut builder,
        context,
        advice,
        NebulaFPrimeStreamingLifecycleArm::Recursive,
        &output,
    )?;
    let x_out_preimage_columns = exact_x_out_preimage_columns(builder.cols(), &public)?;
    let after_nebula_lane_columns =
        exact_lane_source_columns(builder.cols(), &public.outputs, &public.after_nebula_lane)?;
    let (source, mut assignment) = lower_field_r1cs(builder, &public.outputs)?.into_parts();
    populate_public_x_out_bits(
        &mut assignment,
        1,
        x_out_preimage_columns.after(),
        "recursive after-state x_out",
    )?;
    populate_public_x_out_bits(
        &mut assignment,
        1 + F_PRIME_ENC_INST_BITS,
        x_out_preimage_columns.before(),
        "recursive before-state x_out",
    )?;
    Ok(SynthesizedLifecycleArm {
        source,
        assignment,
        x_out_preimage_columns,
        after_nebula_lane_columns,
    })
}

fn populate_public_x_out_bits(
    assignment: &mut [F],
    public_start: usize,
    preimage_columns: &[usize; X_OUT_PREIMAGE_FIELDS],
    label: &'static str,
) -> Result<(), NebulaFPrimeRelationError> {
    let preimage = preimage_columns.map(|column| assignment[column]);
    let digest = digest32_as_fields(state_x_out_digest_from_preimage(&preimage));
    let bits = encode_x_out_public_bits(digest);
    let public_end = public_start + bits.len();
    let assignment_width = assignment.len();
    let target = assignment
        .get_mut(public_start..public_end)
        .ok_or_else(|| {
            NebulaFPrimeRelationError::Geometry(format!(
                "streaming lifecycle {label} public bit range {public_start}..{public_end} exceeds assignment width {}",
                assignment_width
            ))
        })?;
    target.copy_from_slice(&bits);
    Ok(())
}

fn step_config<'a>(context: &'a ShapeContext<'a>) -> FPrimeStepConfig<'a> {
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: PiCcsVerifierConfig {
                params: context.params,
                structure: context.folded.clone(),
                matrix_digest: context.matrix_digest,
            },
        },
        b: context.params.b(),
        transcript_label: F_PRIME_STEP_TRANSCRIPT_LABEL,
        public_input_layout: context.public.f_prime_layout(),
        nebula: Some(&context.config),
        state_x_out_digest_mode: StateXOutDigestMode::Stateful,
    }
}

fn shape_verifier_advice(context: &ShapeContext<'_>) -> VerifierAdvice {
    VerifierAdvice {
        structure_digest: context.structure_digest,
        ajtai_pp_digest: context.ajtai_pp_digest,
        initial_semantic_state_digest: context.config.initial_semantic_state_digest,
    }
}

fn shape_state(
    context: &ShapeContext<'_>,
    advice: VerifierAdvice,
    recursive: bool,
    semantic_digest: [F; 4],
    acc_digest: [F; 4],
) -> FPrimeStateIn {
    let public_input_len = Some(context.public.columns());
    let z_0 = initial_boundary_digest(&advice.structure_digest, public_input_len);
    let initial_trace = public_trace_seed_digest(&advice.structure_digest);
    let prior_boundary = if recursive {
        digest_fields_as_digest32(chunk_digest(context, 0))
    } else {
        z_0
    };
    FPrimeStateIn {
        vk_fs_digest: context.vk_fs_digest,
        pi_ccs_header_bundle: context.matrix_digest,
        chunk_count_in: u64::from(recursive),
        step_count_in: u64::from(recursive),
        z_0: digest32_as_fields(z_0),
        z_i_in: digest32_as_fields(prior_boundary),
        pc: 1,
        semantic_state_digest_in: semantic_digest,
        acc_digest_in: acc_digest,
        public_trace_in: if recursive {
            digest32_as_fields(prior_boundary)
        } else {
            digest32_as_fields(initial_trace)
        },
        nebula: Some(NebulaLane::base(&context.config)),
    }
}

fn chunk_digest(context: &ShapeContext<'_>, start_index: u64) -> [F; 4] {
    f_prime_chunk_public_digest_for_uniform_shape(
        start_index,
        1,
        D,
        context.params.kappa() as usize,
        context.public.columns(),
    )
}

fn finalize_public(
    builder: &mut R1csBuilder,
    context: &ShapeContext<'_>,
    advice: VerifierAdvice,
    arm: NebulaFPrimeStreamingLifecycleArm,
    output: &FPrimeStepOutput,
) -> Result<FinalizedPublic, NebulaFPrimeRelationError> {
    enforce_verifier_advice(builder, context, advice, arm, &output.state_in)?;
    enforce_phase_envelope(builder, context, arm, output)?;
    let (finalize_stage, context_stage, family) = match arm {
        NebulaFPrimeStreamingLifecycleArm::Base => (
            fprime_stage::BASE_FINALIZE,
            fprime_stage::BASE_CONTEXT_LINK,
            "fprime.streaming.base.public_envelope",
        ),
        NebulaFPrimeStreamingLifecycleArm::Recursive => (
            fprime_stage::RECURSIVE_FINALIZE,
            fprime_stage::RECURSIVE_CONTEXT_LINK,
            "fprime.streaming.recursive.public_envelope",
        ),
    };
    builder.begin_encoding_stage(finalize_stage);
    builder.begin_encoding_stage(context_stage);
    let row_start = builder.rows();
    if arm == NebulaFPrimeStreamingLifecycleArm::Recursive {
        builder.enforce_eq(
            &Lc::from_var(output.state_in.chunk_count),
            &Lc::from_var(output.state_in.step_count),
        );
    }
    let (before_bits, before_x_out_preimage) = match arm {
        NebulaFPrimeStreamingLifecycleArm::Base => state_x_out_public(builder, &output.state_in)?,
        NebulaFPrimeStreamingLifecycleArm::Recursive => {
            let prior = output.prior_link.as_ref().ok_or_else(|| {
                NebulaFPrimeRelationError::Geometry("recursive lifecycle arm has no prior link".into())
            })?;
            (prior.encoded_bits.clone(), prior.preimage.clone())
        }
    };
    let before_cursor = decompose_var_to_u64_bits(builder, output.state_in.step_count);
    let after_cursor = decompose_var_to_u64_bits(builder, output.state_out.step_count);
    builder.record_row_family(family, row_start);

    let mut public = Vec::with_capacity(LOGICAL_PUBLIC_OUTPUTS);
    public.extend_from_slice(&output.x_out_bits);
    public.extend(before_bits);
    public.extend(before_cursor);
    public.extend(after_cursor);
    if public.len() != LOGICAL_PUBLIC_OUTPUTS {
        return Err(NebulaFPrimeRelationError::Geometry(format!(
            "streaming lifecycle logical public width {} != {LOGICAL_PUBLIC_OUTPUTS}",
            public.len()
        )));
    }
    let after_nebula_lane = output.state_out.nebula.ok_or_else(|| {
        NebulaFPrimeRelationError::Geometry("streaming lifecycle state-out has no Nebula lane".into())
    })?;
    Ok(FinalizedPublic {
        outputs: public,
        before_x_out_preimage,
        after_x_out_preimage: output.x_out_preimage.clone(),
        after_nebula_lane,
    })
}

fn exact_x_out_preimage_columns(
    source_columns: usize,
    public: &FinalizedPublic,
) -> Result<NebulaFPrimeStreamingXOutPreimageColumns, NebulaFPrimeRelationError> {
    Ok(NebulaFPrimeStreamingXOutPreimageColumns {
        before: normalized_preimage_columns(
            source_columns,
            &public.outputs,
            &public.before_x_out_preimage,
            "before-state x_out",
        )?,
        after: normalized_preimage_columns(
            source_columns,
            &public.outputs,
            &public.after_x_out_preimage,
            "after-state x_out",
        )?,
    })
}

fn normalized_preimage_columns(
    source_columns: usize,
    public_outputs: &[Var],
    preimage: &[Var],
    label: &'static str,
) -> Result<[usize; X_OUT_PREIMAGE_FIELDS], NebulaFPrimeRelationError> {
    if preimage.len() != X_OUT_PREIMAGE_FIELDS {
        return Err(NebulaFPrimeRelationError::Geometry(format!(
            "streaming lifecycle {label} preimage width {} != {X_OUT_PREIMAGE_FIELDS}",
            preimage.len()
        )));
    }
    preimage
        .iter()
        .map(|wire| {
            normalized_field_column(source_columns, public_outputs, wire.col()).ok_or_else(|| {
                NebulaFPrimeRelationError::Geometry(format!(
                    "streaming lifecycle {label} source column {} is outside width {source_columns}",
                    wire.col()
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|columns: Vec<usize>| {
            NebulaFPrimeRelationError::Geometry(format!(
                "streaming lifecycle {label} normalized width {} != {X_OUT_PREIMAGE_FIELDS}",
                columns.len()
            ))
        })
}

fn exact_lane_source_columns(
    source_columns: usize,
    public_outputs: &[Var],
    lane: &crate::paper::f_prime::nebula_lane_circuit::NebulaLaneWires,
) -> Result<NebulaFPrimeStreamingLaneSourceColumns, NebulaFPrimeRelationError> {
    Ok(NebulaFPrimeStreamingLaneSourceColumns {
        program_binding_digest: normalized_var_array(
            source_columns,
            public_outputs,
            lane.program_binding_digest,
            "post-step Nebula program binding",
        )?,
        open: normalized_var_column(source_columns, public_outputs, lane.open, "post-step Nebula open")?,
        seg_idx: normalized_var_column(
            source_columns,
            public_outputs,
            lane.seg_idx,
            "post-step Nebula segment index",
        )?,
        idx: normalized_var_column(source_columns, public_outputs, lane.idx, "post-step Nebula step index")?,
        ts: normalized_var_column(source_columns, public_outputs, lane.ts, "post-step Nebula timestamp")?,
        gamma: normalized_k_array(source_columns, public_outputs, lane.gamma, "post-step Nebula gamma")?,
        h: normalized_k_array(source_columns, public_outputs, lane.h, "post-step Nebula products")?,
        sp: normalized_var_array(
            source_columns,
            public_outputs,
            lane.sp,
            "post-step Nebula stack pointers",
        )?,
        d_pre: normalized_digest_array(source_columns, public_outputs, lane.d_pre, "post-step Nebula D_pre")?,
        d_seen: normalized_digest_array(source_columns, public_outputs, lane.d_seen, "post-step Nebula D_seen")?,
        d_mem: normalized_var_array(source_columns, public_outputs, lane.d_mem, "post-step Nebula D_mem")?,
    })
}

fn normalized_var_column(
    source_columns: usize,
    public_outputs: &[Var],
    wire: Var,
    label: &'static str,
) -> Result<usize, NebulaFPrimeRelationError> {
    normalized_field_column(source_columns, public_outputs, wire.col()).ok_or_else(|| {
        NebulaFPrimeRelationError::Geometry(format!(
            "streaming lifecycle {label} source column {} is outside width {source_columns}",
            wire.col()
        ))
    })
}

fn normalized_var_array<const N: usize>(
    source_columns: usize,
    public_outputs: &[Var],
    wires: [Var; N],
    label: &'static str,
) -> Result<[usize; N], NebulaFPrimeRelationError> {
    wires
        .into_iter()
        .map(|wire| normalized_var_column(source_columns, public_outputs, wire, label))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|columns: Vec<usize>| {
            NebulaFPrimeRelationError::Geometry(format!(
                "streaming lifecycle {label} normalized width {} != {N}",
                columns.len()
            ))
        })
}

fn normalized_k_array<const N: usize>(
    source_columns: usize,
    public_outputs: &[Var],
    wires: [crate::engine::r1cs_circuit::field_ext::KVar; N],
    label: &'static str,
) -> Result<[[usize; 2]; N], NebulaFPrimeRelationError> {
    wires
        .into_iter()
        .map(|wire| normalized_var_array(source_columns, public_outputs, [wire.c0, wire.c1], label))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|columns: Vec<[usize; 2]>| {
            NebulaFPrimeRelationError::Geometry(format!(
                "streaming lifecycle {label} normalized K-width {} != {N}",
                columns.len()
            ))
        })
}

fn normalized_digest_array<const N: usize>(
    source_columns: usize,
    public_outputs: &[Var],
    wires: [[Var; 4]; N],
    label: &'static str,
) -> Result<[[usize; 4]; N], NebulaFPrimeRelationError> {
    wires
        .into_iter()
        .map(|wire| normalized_var_array(source_columns, public_outputs, wire, label))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|columns: Vec<[usize; 4]>| {
            NebulaFPrimeRelationError::Geometry(format!(
                "streaming lifecycle {label} normalized digest count {} != {N}",
                columns.len()
            ))
        })
}

fn enforce_phase_envelope(
    builder: &mut R1csBuilder,
    context: &ShapeContext<'_>,
    arm: NebulaFPrimeStreamingLifecycleArm,
    output: &FPrimeStepOutput,
) -> Result<(), NebulaFPrimeRelationError> {
    let (stage, row_family, before_local_family, after_local_family, after_payload_family) = match arm {
        NebulaFPrimeStreamingLifecycleArm::Base => (
            fprime_stage::BASE_SEMANTIC_LINKS,
            "fprime.streaming.base.phase.semantic_envelope",
            BASE_BEFORE_LOCAL_FAMILY,
            BASE_AFTER_LOCAL_FAMILY,
            BASE_AFTER_PAYLOAD_FAMILY,
        ),
        NebulaFPrimeStreamingLifecycleArm::Recursive => (
            fprime_stage::RECURSIVE_SEMANTIC_LINKS,
            "fprime.streaming.recursive.phase.semantic_envelope",
            RECURSIVE_BEFORE_LOCAL_FAMILY,
            RECURSIVE_AFTER_LOCAL_FAMILY,
            RECURSIVE_AFTER_PAYLOAD_FAMILY,
        ),
    };
    builder.begin_encoding_stage(stage);
    let row_start = builder.rows();
    let payload_len = delayed_nebula_public_suffix_len(context.config.stacks);

    let before_local_start = builder.cols();
    let before_local = alloc_digest(builder, [F::ZERO; 4]);
    builder.record_column_family(before_local_family, before_local_start);

    let before_payload = match arm {
        NebulaFPrimeStreamingLifecycleArm::Base => {
            let column_start = builder.cols();
            let payload = builder.alloc_vec(&vec![F::ZERO; payload_len]);
            builder.record_column_family(BASE_BEFORE_PAYLOAD_FAMILY, column_start);
            for &bit in &payload {
                builder.enforce_zero(&Lc::from_var(bit));
            }
            payload
        }
        NebulaFPrimeStreamingLifecycleArm::Recursive => {
            output.private_delayed_nebula_input.clone().ok_or_else(|| {
                NebulaFPrimeRelationError::Geometry(
                    "recursive lifecycle arm has no private delayed Nebula input".into(),
                )
            })?
        }
    };
    if before_payload.len() != payload_len {
        return Err(NebulaFPrimeRelationError::Geometry(format!(
            "phase-before delayed payload width {} != {payload_len}",
            before_payload.len()
        )));
    }

    let after_local_start = builder.cols();
    let after_local = alloc_digest(builder, [F::ZERO; 4]);
    builder.record_column_family(after_local_family, after_local_start);

    let after_payload_start = builder.cols();
    let after_payload = builder.alloc_vec(&vec![F::ZERO; payload_len]);
    builder.record_column_family(after_payload_family, after_payload_start);

    let before_semantic = enforce_streaming_phase_semantic_digest(builder, before_local, &before_payload, false);
    let after_semantic = enforce_streaming_phase_semantic_digest(builder, after_local, &after_payload, true);
    bind_digest(builder, &output.state_in.semantic_state_digest, &before_semantic);
    bind_digest(builder, &output.state_out.semantic_state_digest, &after_semantic);
    builder.record_row_family(row_family, row_start);
    Ok(())
}

fn enforce_verifier_advice(
    builder: &mut R1csBuilder,
    context: &ShapeContext<'_>,
    advice: VerifierAdvice,
    arm: NebulaFPrimeStreamingLifecycleArm,
    state: &FPrimeStateWires,
) -> Result<(), NebulaFPrimeRelationError> {
    let (stage, family) = match arm {
        NebulaFPrimeStreamingLifecycleArm::Base => {
            (fprime_stage::BASE_VERIFIER_KEY, "fprime.streaming.base.verifier_advice")
        }
        NebulaFPrimeStreamingLifecycleArm::Recursive => (
            fprime_stage::RECURSIVE_VERIFIER_KEY,
            "fprime.streaming.recursive.verifier_advice",
        ),
    };
    builder.begin_encoding_stage(stage);
    let row_start = builder.rows();
    let column_start = builder.witness().len();
    let structure_digest = alloc_bound_digest(builder, advice.structure_digest);
    let ajtai_pp_digest = alloc_bound_digest(builder, advice.ajtai_pp_digest);
    let initial_semantic_state_digest = alloc_bound_digest(builder, advice.initial_semantic_state_digest);
    let base_vk = enforce_vk_fs_digest_circuit(
        builder,
        context.params,
        structure_digest,
        state.pi_ccs_header_bundle,
        ajtai_pp_digest,
        Some(context.public.columns()),
        initial_semantic_state_digest,
    );
    let policy_vk = enforce_vk_fs_policy_digest_circuit(builder, base_vk, true, true, true);
    bind_digest(builder, &state.vk_fs_digest, &policy_vk);
    let initial_boundary =
        enforce_initial_boundary_digest_circuit(builder, structure_digest, Some(context.public.columns()));
    bind_digest(builder, &state.z_0, &initial_boundary);
    match arm {
        NebulaFPrimeStreamingLifecycleArm::Base => {
            bind_digest(builder, &state.semantic_state_digest, &initial_semantic_state_digest);
            let initial_trace = enforce_public_trace_seed_digest_circuit(builder, structure_digest);
            bind_digest(builder, &state.public_trace, &initial_trace);
            let lane = state
                .nebula
                .as_ref()
                .ok_or_else(|| NebulaFPrimeRelationError::Geometry("base lifecycle arm has no Nebula lane".into()))?;
            enforce_nebula_lane_constant_circuit(builder, lane, &NebulaLane::base(&context.config));
        }
        NebulaFPrimeStreamingLifecycleArm::Recursive => {
            bind_digest(builder, &state.public_trace, &state.z_i);
        }
    }
    builder.record_row_family(family, row_start);
    builder.record_column_family(family, column_start);
    Ok(())
}

fn state_x_out_public(
    builder: &mut R1csBuilder,
    state: &FPrimeStateWires,
) -> Result<(Vec<Var>, Vec<Var>), NebulaFPrimeRelationError> {
    let lane = state
        .nebula
        .as_ref()
        .ok_or_else(|| NebulaFPrimeRelationError::Geometry("streaming lifecycle state has no Nebula lane".into()))?;
    let lane_digest = enforce_nebula_lane_digest_selected_circuit(builder, lane);
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: state.vk_fs_digest,
        pi_ccs_header_bundle: state.pi_ccs_header_bundle,
        structure_digest: state.pi_ccs_header_bundle,
        chunk_count: state.chunk_count,
        step_count: state.step_count,
        initial_boundary: state.z_0,
        current_boundary: state.z_i,
        pc: state.pc,
        semantic_acc: state.semantic_state_digest,
        construction2_acc: state.acc_digest,
        public_trace: state.public_trace,
    };
    let wires = enforce_streaming_state_x_out(builder, &inputs, lane_digest);
    Ok((wires.public_bits.to_vec(), wires.preimage))
}

fn bind_digest(builder: &mut R1csBuilder, left: &[Var; 4], right: &[Var; 4]) {
    for lane in 0..4 {
        builder.enforce_eq(&Lc::from_var(left[lane]), &Lc::from_var(right[lane]));
    }
}

fn alloc_digest(builder: &mut R1csBuilder, values: [F; 4]) -> [Var; 4] {
    values.map(|value| builder.alloc(value))
}

fn alloc_bound_digest(builder: &mut R1csBuilder, values: [F; 4]) -> [Var; 4] {
    values.map(|value| alloc_constant(builder, value))
}

fn zero_fresh_claim(context: &ShapeContext<'_>, m_in: usize) -> CcsClaim {
    let mut x = vec![F::ZERO; m_in];
    x[0] = F::ONE;
    CcsClaim {
        c: Commitment::zeros(D, context.params.kappa() as usize),
        x,
        m_in,
        adv: Some(zero_lane_commitments(context.params)),
    }
}

fn zero_ce_claim(context: &ShapeContext<'_>, m_in: usize) -> CeClaim {
    let d_pad = D.next_power_of_two();
    CeClaim {
        c: Commitment::zeros(D, context.params.kappa() as usize),
        X: Mat::zero(D, crate::paper::relations::superneo_public_x_cols(m_in), F::ZERO),
        r: vec![K::ZERO; context.joint_variables],
        y_ring: vec![vec![K::ZERO; d_pad]; context.folded.t() + 1],
        ct: vec![K::ZERO; context.folded.t() + 1],
        m_in,
        fold_digest: [0u8; 32],
        adv: Some(zero_lane_commitments(context.params)),
    }
}

const _: () = assert!(LOGICAL_PUBLIC_OUTPUTS + 1 == 641);
