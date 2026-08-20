//! Composition of one current `S_mem` execution with authoritative F'.
//!
//! The current application suffix is produced here; the core F' relation
//! consumes the previous claim's suffix through NIFS.V. Keeping those two
//! directions in one wrapper makes Nebula's one-step delay explicit.

mod chain;
mod constraint_source_audit;
mod constraint_witness_audit;
mod encoder_artifact;
mod relation_artifact;
mod shape;
mod streaming_claim_replay;
mod streaming_lifecycle_fixed_point;
mod streaming_lifecycle_profile;
mod streaming_lifecycle_relation;
mod streaming_lifecycle_verifier_key_recipe;
mod streaming_phase_envelope;
mod streaming_pi_ccs_round_relation;
mod streaming_pi_ccs_start_relation;
mod streaming_pi_ccs_state;
mod streaming_pi_rlc_family_relation;
mod streaming_pi_rlc_family_replay;
mod streaming_prelude_relation;
mod streaming_prior_state_replay_relation;
mod streaming_program;
mod streaming_public;
mod streaming_relation;
mod streaming_state_envelope;
mod streaming_terminal_profile;

pub use chain::{
    NebulaFPrimeChainBuilder, NebulaFPrimeChainError, NebulaFPrimePreparedProfile, NebulaFPrimePreprocessing,
};
pub use constraint_source_audit::NebulaFPrimeConstraintSourceAudit;
pub use constraint_witness_audit::NebulaFPrimeConstraintWitnessAudit;
pub use encoder_artifact::{NebulaFPrimeEncoderArtifactReceipt, VerifiedNebulaFPrimeEncoderArtifact};
#[doc(hidden)]
pub use streaming_claim_replay::{
    build_production_claim_coordinate_overlay_low_norm_r1cs, build_production_claim_replay_base_low_norm_r1cs,
    claim_replay_shape_audit_for_chunk_fields, production_claim_coordinate_overlay_kind_count,
    production_claim_coordinate_overlay_kind_map, production_claim_coordinate_overlay_link_runs,
    production_claim_coordinate_overlay_links, production_claim_coordinate_overlay_shape_audit,
    production_claim_replay_base_shape_audit, production_claim_replay_shape_audit,
    production_claim_running_commitment_field_map, production_claim_running_public_field_map,
    production_claim_statement_fresh_field_map, NebulaFPrimeClaimCoordinateOverlayLinkRun,
    NebulaFPrimeClaimCoordinateOverlayShapeAudit, NebulaFPrimeClaimCoordinateOverlaySynthesis,
    NebulaFPrimeClaimReplayArmKind, NebulaFPrimeClaimReplayBaseShapeAudit, NebulaFPrimeClaimReplayError,
    NebulaFPrimeClaimReplayFieldArmAudit, NebulaFPrimeClaimReplayShapeAudit, NebulaFPrimeClaimReplaySynthesis,
};
#[doc(hidden)]
pub use streaming_lifecycle_fixed_point::{
    production_streaming_lifecycle_full_source_fixed_point_audit,
    NebulaFPrimeStreamingLifecycleFullSourceFixedPointAudit, NebulaFPrimeStreamingLifecycleFullSourceFixedPointRound,
    NebulaFPrimeStreamingLifecycleFullSourceFixedPointShape, NebulaFPrimeStreamingLifecycleFullSourceShape,
    STREAMING_LIFECYCLE_FULL_SOURCE_JOINT_DOMAIN_BITS,
};
#[doc(hidden)]
pub use streaming_lifecycle_profile::{
    production_streaming_lifecycle_profile, NebulaFPrimeStreamingLifecycleArmProfile,
    NebulaFPrimeStreamingLifecycleColumnLayout, NebulaFPrimeStreamingLifecycleDecoderTerm,
    NebulaFPrimeStreamingLifecycleFieldBinding, NebulaFPrimeStreamingLifecycleFinalRowRun,
    NebulaFPrimeStreamingLifecycleProfile, NebulaFPrimeStreamingLifecycleProfileError,
    NebulaFPrimeStreamingLifecycleRewriteBinding, NebulaFPrimeStreamingLifecycleSourceRunBinding,
    NebulaFPrimeStreamingLifecycleStageBinding, NebulaFPrimeStreamingLifecycleXOutBinding,
    STREAMING_LIFECYCLE_BASE_SOURCE_ARTIFACT_ID, STREAMING_LIFECYCLE_FINAL_ARTIFACT_ID, STREAMING_LIFECYCLE_PROFILE_ID,
    STREAMING_LIFECYCLE_RECURSIVE_SOURCE_ARTIFACT_ID,
};
#[doc(hidden)]
pub use streaming_lifecycle_relation::{
    prepare_streaming_lifecycle_preprocessing, synthesize_streaming_lifecycle_source_arms,
    synthesize_streaming_lifecycle_source_arms_with_recursive_assignment, NebulaFPrimeStreamingLaneSourceColumns,
    NebulaFPrimeStreamingLifecycleArm, NebulaFPrimeStreamingLifecycleSourceArms,
    NebulaFPrimeStreamingPhaseEnvelopeFields, NebulaFPrimeStreamingXOutHashAudit,
    NebulaFPrimeStreamingXOutPreimageColumns, NebulaFPrimeStreamingXOutPreimageValues,
    NebulaFPrimeStreamingXOutPublicWordAudit,
};
#[doc(hidden)]
pub use streaming_lifecycle_verifier_key_recipe::{
    extract_base_verifier_key_core_recipes, NebulaFPrimeStreamingVerifierKeyCoreRecipes,
    NebulaFPrimeStreamingVerifierKeyDigestBinding, NebulaFPrimeStreamingVerifierKeyHashBlock,
    NebulaFPrimeStreamingVerifierKeyHashRecipes,
};
#[doc(hidden)]
pub use streaming_phase_envelope::{
    enforce_streaming_lifecycle_semantic_link, enforce_streaming_lifecycle_source_semantic_link,
    enforce_streaming_phase_semantic_digest, streaming_phase_semantic_digest, StreamingLifecycleBeforePayloadRule,
    StreamingLifecycleSemanticLinkWires, STREAMING_CARRY_PHASE_ENVELOPE_FAMILY,
    STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS, STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY,
    STREAMING_LIFECYCLE_SEMANTIC_LINK_FAMILY, STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY, STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY,
};
#[doc(hidden)]
pub use streaming_pi_ccs_round_relation::{
    production_pi_ccs_round_source_arm, NebulaFPrimePiCcsRoundArithmeticBinding, NebulaFPrimePiCcsRoundColumnLayout,
    NebulaFPrimePiCcsRoundRelationError, NebulaFPrimePiCcsRoundShapeAudit, NebulaFPrimePiCcsRoundSourceStage,
    NebulaFPrimePiCcsRoundSynthesis, PI_CCS_ROUND_AFTER_LAST_PROGRAM_CURSOR, PI_CCS_ROUND_ARITHMETIC_BINDING,
    PI_CCS_ROUND_COMPACT_ARITHMETIC_ARTIFACT_ID, PI_CCS_ROUND_FINAL_COMMON_PUBLIC_COLUMNS,
    PI_CCS_ROUND_FIRST_PROGRAM_CURSOR, PI_CCS_ROUND_LIFECYCLE_SCOPE, PI_CCS_ROUND_PROFILE_ID,
    PI_CCS_ROUND_SOURCE_ARTIFACT_ID, PI_CCS_ROUND_SOURCE_COLUMNS, PI_CCS_ROUND_SOURCE_COLUMN_LAYOUT,
    PI_CCS_ROUND_SOURCE_POSEIDON2_PERMUTATIONS, PI_CCS_ROUND_SOURCE_PUBLIC_COLUMNS, PI_CCS_ROUND_SOURCE_ROWS,
    PI_CCS_ROUND_SOURCE_SHA256, PI_CCS_ROUND_SOURCE_STAGE_SCHEDULE, STREAMING_PI_CCS_ROUND_AFTER_STATE_FAMILY,
    STREAMING_PI_CCS_ROUND_ARITHMETIC_FAMILY, STREAMING_PI_CCS_ROUND_BEFORE_STATE_FAMILY,
    STREAMING_PI_CCS_ROUND_COEFFICIENT_FAMILY, STREAMING_PI_CCS_ROUND_LIFECYCLE_CARRY_FAMILY,
    STREAMING_PI_CCS_ROUND_STATE_DIGEST_FAMILY, STREAMING_PI_CCS_ROUND_STATE_TRANSITION_FAMILY,
    STREAMING_PI_CCS_ROUND_TRANSCRIPT_FAMILY,
};
#[doc(hidden)]
pub use streaming_pi_ccs_start_relation::{
    production_pi_ccs_start_source_arm, NebulaFPrimePiCcsStartColumnLayout, NebulaFPrimePiCcsStartRelationError,
    NebulaFPrimePiCcsStartShapeAudit, NebulaFPrimePiCcsStartSourceStage, NebulaFPrimePiCcsStartSynthesis,
    PI_CCS_START_AFTER_PROGRAM_CURSOR, PI_CCS_START_BEFORE_PROGRAM_CURSOR, PI_CCS_START_FINAL_BINDING_STATUS,
    PI_CCS_START_FINAL_COMMON_PUBLIC_COLUMNS, PI_CCS_START_LIFECYCLE_SCOPE, PI_CCS_START_PROFILE_ID,
    PI_CCS_START_SOURCE_ARTIFACT_ID, PI_CCS_START_SOURCE_COLUMNS, PI_CCS_START_SOURCE_COLUMN_LAYOUT,
    PI_CCS_START_SOURCE_HASH_SCHEMA, PI_CCS_START_SOURCE_POSEIDON2_PERMUTATIONS, PI_CCS_START_SOURCE_PUBLIC_COLUMNS,
    PI_CCS_START_SOURCE_ROWS, PI_CCS_START_SOURCE_SHA256, PI_CCS_START_SOURCE_STAGE_SCHEDULE,
    STREAMING_PI_CCS_START_CLAIM_STATE_FAMILY, STREAMING_PI_CCS_START_CONTEXT_FAMILY,
    STREAMING_PI_CCS_START_INITIAL_CLAIM_FAMILY, STREAMING_PI_CCS_START_LIFECYCLE_CARRY_FAMILY,
    STREAMING_PI_CCS_START_READY_FAMILY, STREAMING_PI_CCS_START_TRANSCRIPT_FAMILY,
    STREAMING_PI_CCS_START_VARIABLE_BINDING_FAMILY, STREAMING_PI_CCS_START_VARIABLE_FIELDS_FAMILY,
};
#[doc(hidden)]
pub use streaming_pi_rlc_family_relation::{
    build_production_pi_rlc_family_body_low_norm_r1cs, build_production_pi_rlc_family_overlay_low_norm_r1cs,
    production_pi_rlc_family_body_algebra_retained_audit, production_pi_rlc_family_body_carry_retained_audit,
    production_pi_rlc_family_body_compact_layout_and_decoder_runs_for_ranges,
    production_pi_rlc_family_body_compiler_audit, production_pi_rlc_family_body_decoder_runs,
    production_pi_rlc_family_body_low_norm_shape_audit, production_pi_rlc_family_body_opening_rows_audit,
    production_pi_rlc_family_body_projected_rows_with_source_provenance,
    production_pi_rlc_family_body_residual_retained_audit, production_pi_rlc_family_body_row_ledger,
    production_pi_rlc_family_body_source_arms, production_pi_rlc_family_normalized_link_audit,
    production_pi_rlc_family_overlay_kind_map, production_pi_rlc_family_overlay_link_runs,
    production_pi_rlc_family_overlay_links, production_pi_rlc_family_overlay_retained_audit,
    NebulaFPrimePiRlcBodyAlgebraRetainedAudit, NebulaFPrimePiRlcBodyCarryRetainedAudit,
    NebulaFPrimePiRlcBodyFixedEmittedRun, NebulaFPrimePiRlcBodyFixedFamily, NebulaFPrimePiRlcBodyOpeningRowsAudit,
    NebulaFPrimePiRlcBodyResidualRetainedAudit, NebulaFPrimePiRlcBodyRetainedRun, NebulaFPrimePiRlcBodyRewriteBatch,
    NebulaFPrimePiRlcBodyRewriteKind, NebulaFPrimePiRlcFamilyBodyLowNormShapeAudit,
    NebulaFPrimePiRlcFamilyBodyRowLedger, NebulaFPrimePiRlcFamilyBodyShapeAudit, NebulaFPrimePiRlcFamilyBodySynthesis,
    NebulaFPrimePiRlcFamilyNormalizedLinkAudit, NebulaFPrimePiRlcFamilyNormalizedLinkRunAudit,
    NebulaFPrimePiRlcFamilyOverlayLinkRun, NebulaFPrimePiRlcFamilyOverlayRetainedAudit,
    NebulaFPrimePiRlcFamilyOverlayShapeAudit, NebulaFPrimePiRlcFamilyOverlaySynthesis,
    NebulaFPrimePiRlcFamilyRelationError, PI_RLC_FAMILY_BODY_EVEN_COLUMNS, PI_RLC_FAMILY_BODY_EVEN_ROWS,
    PI_RLC_FAMILY_BODY_EVEN_SOURCE_ROWS, PI_RLC_FAMILY_BODY_ODD_COLUMNS, PI_RLC_FAMILY_BODY_ODD_ROWS,
    PI_RLC_FAMILY_BODY_ODD_SOURCE_ROWS, PI_RLC_FAMILY_BODY_PUBLIC_OUTPUTS, PI_RLC_FAMILY_BODY_SOURCE_ROWS,
    PI_RLC_FAMILY_COUNT, PI_RLC_FAMILY_LINK_FIELDS, PI_RLC_FAMILY_OVERLAY_COLUMNS, PI_RLC_FAMILY_OVERLAY_ROWS,
    PI_RLC_GLOBAL_INPUT_FIELDS, PI_RLC_MESSAGE_COLUMNS,
};
#[doc(hidden)]
pub use streaming_pi_rlc_family_replay::{
    NebulaFPrimePiRlcFamilyReplayArmKind, NebulaFPrimePiRlcFamilyReplayCallAudit,
    NebulaFPrimePiRlcFamilyReplayCallClass, NebulaFPrimePiRlcFamilyReplayScope,
    NebulaFPrimePiRlcFamilyReplayShapeAudit, NebulaFPrimePiRlcFamilyReplaySynthesis,
};
#[doc(hidden)]
pub use streaming_prelude_relation::{
    production_streaming_prelude_source_arm, NebulaFPrimeStreamingPreludeSynthesis,
    STREAMING_PRELUDE_INITIAL_REPLAY_STATE_FAMILY, STREAMING_PRELUDE_INITIAL_REPLAY_STATE_ROWS_FAMILY,
};
#[doc(hidden)]
pub use streaming_prior_state_replay_relation::{
    production_prior_state_replay_final_source_arm, production_prior_state_replay_full_source_arm,
    NebulaFPrimePriorStateReplayArmKind, NebulaFPrimePriorStateReplayColumnLayout,
    NebulaFPrimePriorStateReplayShapeAudit, NebulaFPrimePriorStateReplaySourceStage,
    NebulaFPrimePriorStateReplaySynthesis, PRIOR_STATE_REPLAY_AFTER_LAST_PROGRAM_CURSOR, PRIOR_STATE_REPLAY_CHUNKS,
    PRIOR_STATE_REPLAY_CHUNK_FIELDS, PRIOR_STATE_REPLAY_FINAL_COMMON_PUBLIC_COLUMNS, PRIOR_STATE_REPLAY_FINAL_FIELDS,
    PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMNS, PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMN_LAYOUT,
    PRIOR_STATE_REPLAY_FINAL_SOURCE_POSEIDON2_PERMUTATIONS, PRIOR_STATE_REPLAY_FINAL_SOURCE_ROWS,
    PRIOR_STATE_REPLAY_FINAL_SOURCE_SHA256, PRIOR_STATE_REPLAY_FINAL_SOURCE_STAGE_SCHEDULE,
    PRIOR_STATE_REPLAY_FINAL_TARGET_BINDING_STATUS, PRIOR_STATE_REPLAY_FIRST_PROGRAM_CURSOR,
    PRIOR_STATE_REPLAY_FRAME_FIELDS, PRIOR_STATE_REPLAY_FULL_CHUNKS, PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMNS,
    PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMN_LAYOUT, PRIOR_STATE_REPLAY_FULL_SOURCE_POSEIDON2_PERMUTATIONS,
    PRIOR_STATE_REPLAY_FULL_SOURCE_ROWS, PRIOR_STATE_REPLAY_FULL_SOURCE_SHA256,
    PRIOR_STATE_REPLAY_FULL_SOURCE_STAGE_SCHEDULE, PRIOR_STATE_REPLAY_LIFECYCLE_SCOPE, PRIOR_STATE_REPLAY_PROFILE_ID,
    PRIOR_STATE_REPLAY_SOURCE_ARTIFACT_ID, PRIOR_STATE_REPLAY_SOURCE_HASH_SCHEMA,
    PRIOR_STATE_REPLAY_SOURCE_PUBLIC_COLUMNS, STREAMING_PRIOR_STATE_REPLAY_AFTER_STATE_FAMILY,
    STREAMING_PRIOR_STATE_REPLAY_BEFORE_STATE_FAMILY, STREAMING_PRIOR_STATE_REPLAY_CHUNK_FAMILY,
    STREAMING_PRIOR_STATE_REPLAY_FINAL_TARGET_FAMILY, STREAMING_PRIOR_STATE_REPLAY_LIFECYCLE_CARRY_FAMILY,
    STREAMING_PRIOR_STATE_REPLAY_STATE_TRANSITION_FAMILY,
};
#[doc(hidden)]
pub use streaming_program::{
    NebulaFPrimeStreamingCircuitKind, NebulaFPrimeStreamingPhase, NebulaFPrimeStreamingProgramAudit,
    NebulaFPrimeStreamingRun, NebulaFPrimeStreamingWorkItem,
};
#[doc(hidden)]
pub use streaming_public::NebulaFPrimeStreamingPublicLayout;
#[doc(hidden)]
pub use streaming_relation::{
    build_production_combined_overlay_low_norm_r1cs, build_production_streaming_schedule_low_norm_r1cs,
    build_production_streaming_schedule_with_overlays_low_norm_r1cs, production_combined_overlay_kind_count,
    production_combined_overlay_kind_map, production_combined_overlay_links, production_phase_envelope_link_profile,
    NebulaFPrimeStreamingPhaseEnvelopeLinkProfile, NebulaFPrimeStreamingRelationError,
};
#[doc(hidden)]
pub use streaming_state_envelope::enforce_streaming_state_x_out_bits;
#[doc(hidden)]
pub use streaming_terminal_profile::{
    production_streaming_terminal_profile, NebulaFPrimeStreamingTerminalColumnLayout,
    NebulaFPrimeStreamingTerminalDecoderTerm, NebulaFPrimeStreamingTerminalFieldBinding,
    NebulaFPrimeStreamingTerminalFieldDomain, NebulaFPrimeStreamingTerminalFinalRowRun,
    NebulaFPrimeStreamingTerminalLaneBinding, NebulaFPrimeStreamingTerminalProfile,
    NebulaFPrimeStreamingTerminalProfileError, NebulaFPrimeStreamingTerminalSliceBinding,
    NebulaFPrimeStreamingTerminalSourceStageBinding, NebulaFPrimeStreamingTerminalXOutBinding,
    STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS, STREAMING_TERMINAL_FINAL_ARTIFACT_ID, STREAMING_TERMINAL_PROFILE_ID,
    STREAMING_TERMINAL_SOURCE_ARTIFACT_ID,
};

use std::sync::Arc;

use neo_math::D;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::frontends::nebula::application::{enforce_memory_ports, ApplicationError, NebulaApplication};
use crate::frontends::nebula::circuit::{SMemCircuit, SMemR1csError};
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_low_norm_width_for_norm_base_with_alignment,
    prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix, selective_polynomial,
    FieldR1csLoweringError, LowNormEncoderArtifactError, LowNormR1csError, MultiBranchLowNormR1cs,
    PreparedSelectiveLowNormR1cs, SelectiveLowNormShapeSummary, SelectiveLowNormWidthAudit, SparseR1cs,
};
use crate::lifecycle::Preprocessing;
use crate::paper::construction2::NebulaConfig;
use crate::paper::digest;
use crate::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use crate::paper::f_prime::r1cs::{
    enforce_f_prime_base_step_circuit, enforce_f_prime_recursive_step_circuit, Error as FPrimeError, FPrimeBaseInputs,
    FPrimeRecursiveInputs, FPrimeStepConfig, FPrimeStepOutput,
};
use crate::paper::f_prime::stage as fprime_stage;
use crate::paper::nifs::NifsFreshSignedUnitAssignment;
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs_circuit::PiCcsVerifierRelation;
use crate::paper::relations::{CcsInstance, LaneRanges, LaneSchemeError, RelationError, Structure};

#[derive(Debug, Error)]
pub enum NebulaFPrimeError {
    #[error(transparent)]
    Application(#[from] ApplicationError),
    #[error(transparent)]
    App(#[from] crate::frontends::direct_ccs::FrontendError),
    #[error(transparent)]
    R1csIvc(#[from] crate::frontends::r1cs_f_prime::ivc::R1csIvcError),
    #[error("composed Nebula F': FPrimeStepConfig has no Nebula configuration")]
    MissingNebulaConfig,
    #[error("composed Nebula F': S_mem public step width {actual} != configured width {expected}")]
    StepPublicWidth { actual: usize, expected: usize },
    #[error("composed Nebula F': configured suffix width {actual} != delayed Nebula width {expected}")]
    SuffixWidth { actual: usize, expected: usize },
    #[error(transparent)]
    SMem(#[from] SMemR1csError),
    #[error(transparent)]
    FPrime(#[from] FPrimeError),
}

#[derive(Debug, Error)]
pub enum NebulaFPrimeRelationError {
    #[error(transparent)]
    Application(#[from] ApplicationError),
    #[error(transparent)]
    R1csIvc(#[from] crate::frontends::r1cs_f_prime::ivc::R1csIvcError),
    #[error(transparent)]
    LowNorm(#[from] LowNormR1csError),
    #[error(transparent)]
    Lanes(#[from] LaneSchemeError),
    #[error(transparent)]
    Relation(#[from] RelationError),
    #[error(transparent)]
    FieldR1cs(#[from] FieldR1csLoweringError),
    #[error(transparent)]
    EncoderArtifact(#[from] LowNormEncoderArtifactError),
    #[error(transparent)]
    Composition(#[from] NebulaFPrimeError),
    #[error("fixed Nebula F': {0}")]
    Geometry(String),
    #[error("fixed Nebula F': encoded branch does not satisfy the authoritative relation at row {row}")]
    Unsatisfied { row: usize },
    #[error("fixed Nebula F': preprocessing was built for a different relation")]
    PreprocessingMismatch,
    #[error("fixed Nebula F': program does not match the prepared relation profile: {0}")]
    PreparedProfileMismatch(&'static str),
    #[error(
        "fixed Nebula F': relation-shape discovery entered a cycle after {rounds} rounds \
         (last verifier relation {input_rows}x{input_cols}, next {output_rows}x{output_cols})"
    )]
    NoFixedPoint {
        rounds: usize,
        input_rows: usize,
        input_cols: usize,
        output_rows: usize,
        output_cols: usize,
    },
}

/// Lifecycle branches of the single folded relation. Both recursive branches
/// use the same recursive circuit. Their different witness values describe
/// whether a delayed projection is present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimeBranch {
    Base,
    BootstrapRecursive,
    Recursive,
}

/// Field-native dimensions of one authoritative F' arm, before low-norm
/// bit lowering. `columns` includes the implicit constant-one column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeFieldArmShape {
    pub rows: usize,
    pub columns: usize,
    pub public_columns: usize,
    pub poseidon2_permutations: usize,
}

/// Shape-only audit of all three fixed-relation arms.
/// This deliberately stops before low-norm compilation, whose output can be
/// much larger than the field-native matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeFieldShapeAudit {
    pub verifier_rows: usize,
    pub verifier_columns: usize,
    pub base: NebulaFPrimeFieldArmShape,
    pub bootstrap_recursive: NebulaFPrimeFieldArmShape,
    pub recursive: NebulaFPrimeFieldArmShape,
}

impl NebulaFPrimeBranch {
    const fn logical_index(self) -> usize {
        match self {
            Self::Base => 0,
            Self::BootstrapRecursive => 1,
            Self::Recursive => 2,
        }
    }

    /// Physical selective-relation arm used by this lifecycle branch.
    #[doc(hidden)]
    pub const fn relation_arm_index(self) -> usize {
        match self {
            Self::Base => 0,
            Self::BootstrapRecursive | Self::Recursive => 1,
        }
    }
}

/// One foldable low-norm relation for all Nebula F' lifecycle branches.
/// Bootstrap and steady recursion share one physical recursive arm because
/// they use the same R1CS relation.
pub struct NebulaFPrimeRelation {
    relation: Arc<MultiBranchLowNormR1cs>,
    config: NebulaConfig,
    application: Option<NebulaApplication>,
    arm_shapes: [NebulaFPrimeFieldArmShape; 3],
    width_audit: Option<Arc<SelectiveLowNormWidthAudit>>,
    preprocessing_digest: Option<[F; 4]>,
}

struct NebulaFixedPointCandidate {
    prepared: PreparedSelectiveLowNormR1cs,
    rounds: usize,
    verifier_rows: usize,
    verifier_columns: usize,
}

impl NebulaFPrimeRelation {
    /// Compile the two distinct relation arms to a verifier-shape fixed point.
    ///
    /// Recursive-arm matrices are synthesized from shape-correct placeholder
    /// messages. Their witness values need not satisfy the rows: R1CS shape and
    /// coefficients must be deterministic functions of `(params, folded
    /// relation shape)`. The active encoder test supplies honest assignments
    /// to both compiled arms, including bootstrap and interior recursive
    /// witnesses, and
    /// therefore fails if live synthesis drifts from this fixed relation.
    pub fn compile_fixed_point(params: &Params, plan: &NebulaPlan) -> Result<Self, NebulaFPrimeRelationError> {
        Self::compile_fixed_point_inner(params, plan, None)
    }

    pub fn compile_application_fixed_point(
        params: &Params,
        plan: &NebulaPlan,
        application: NebulaApplication,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        application.validate_for(plan)?;
        Self::compile_fixed_point_inner(params, plan, Some(application))
    }

    fn compile_fixed_point_inner(
        params: &Params,
        plan: &NebulaPlan,
        application: Option<NebulaApplication>,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        let candidate = Self::discover_fixed_point(params, plan, application.as_ref())?;
        Self::compile_owned_selected(candidate.prepared, plan, application)
    }

    fn discover_fixed_point(
        params: &Params,
        plan: &NebulaPlan,
        application: Option<&NebulaApplication>,
    ) -> Result<NebulaFixedPointCandidate, NebulaFPrimeRelationError> {
        // Fixed-point discovery starts inside the output compiler's relation
        // family. The previous S_mem seed forced one full transition from a
        // 15-matrix degree-4 relation before the 13-matrix degree-8 selective
        // shape could begin converging. The exact terminal signature remains
        // the return condition, so this changes only discovery work.
        let verifier_row_domain = usize::try_from(params.m())
            .map_err(|_| NebulaFPrimeRelationError::Geometry("parameter row domain exceeds usize".into()))?;
        let verifier_assignment_domain = verifier_row_domain / D * D;
        let mut verifier_relation =
            PiCcsVerifierRelation::from_parts(verifier_row_domain, verifier_assignment_domain, selective_polynomial());
        let mut seen = Vec::new();
        loop {
            #[cfg(feature = "perf-timers")]
            let round = seen.len();
            #[cfg(feature = "perf-timers")]
            let round_started = std::time::Instant::now();
            let input_signature = verifier_relation_signature(&verifier_relation);
            seen.push(input_signature);
            #[cfg(feature = "perf-timers")]
            let synthesis_started = std::time::Instant::now();
            let arms = shape::synthesize_arm_shapes(params, &verifier_relation, plan, application)?;
            #[cfg(feature = "perf-timers")]
            let synthesis_elapsed = synthesis_started.elapsed();
            #[cfg(feature = "perf-timers")]
            let arm_shapes = [(arms.base.n, arms.base.m), (arms.recursive.n, arms.recursive.m)];
            #[cfg(feature = "perf-timers")]
            let lowering_started = std::time::Instant::now();
            let prepared = prepare_low_norm_relation(vec![arms.base, arms.recursive], plan, params.b())?;
            let next_shape = prepared.shape_summary();
            let output_signature = shape_summary_signature(&next_shape);
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "[fprime-fixed-point] round={round} input={}x{} t={} u={} arms=base:{}x{},recursive:{}x{} output={}x{} t={} u={} synth={:.3}s lower={:.3}s total={:.3}s",
                input_signature.0,
                input_signature.1,
                input_signature.2,
                input_signature.3,
                arm_shapes[0].0,
                arm_shapes[0].1,
                arm_shapes[1].0,
                arm_shapes[1].1,
                next_shape.rows,
                next_shape.columns,
                next_shape.polynomial.arity(),
                next_shape.polynomial.max_degree(),
                synthesis_elapsed.as_secs_f64(),
                lowering_started.elapsed().as_secs_f64(),
                round_started.elapsed().as_secs_f64(),
            );
            if input_signature == output_signature {
                return Ok(NebulaFixedPointCandidate {
                    prepared,
                    rounds: seen.len(),
                    verifier_rows: input_signature.0,
                    verifier_columns: input_signature.1,
                });
            }
            if seen.contains(&output_signature) {
                return Err(NebulaFPrimeRelationError::NoFixedPoint {
                    rounds: seen.len(),
                    input_rows: verifier_relation.n(),
                    input_cols: verifier_relation.m(),
                    output_rows: next_shape.rows,
                    output_cols: next_shape.columns,
                });
            }
            verifier_relation =
                PiCcsVerifierRelation::from_parts(next_shape.rows, next_shape.columns, next_shape.polynomial);
        }
    }

    /// Measure the three field-native arms without constructing their
    /// low-norm union. This is the safe entry point for fixed-relation cost audits.
    pub fn audit_field_shapes(
        params: &Params,
        verifier_structure: &Structure,
        plan: &NebulaPlan,
    ) -> Result<NebulaFPrimeFieldShapeAudit, NebulaFPrimeRelationError> {
        shape::audit_arm_shapes(params, verifier_structure, plan)
    }

    /// Attribute the exact low-norm assignment width without allocating the
    /// compiled CCS matrices.
    pub fn audit_low_norm_width(
        params: &Params,
        verifier_structure: &Structure,
        plan: &NebulaPlan,
    ) -> Result<SelectiveLowNormWidthAudit, NebulaFPrimeRelationError> {
        let verifier_relation = PiCcsVerifierRelation::from_structure(verifier_structure);
        let arms = shape::synthesize_arm_shapes(params, &verifier_relation, plan, None)?;
        let circuit = plan.circuit();
        let logical_public_fields = circuit.logical_public_input_len();
        let shared_private_fields = circuit.cols() - logical_public_fields;
        Ok(
            audit_multi_branch_selective_low_norm_width_for_norm_base_with_alignment(
                &[arms.base, arms.recursive],
                shared_private_fields,
                D,
                logical_public_fields % D,
                params.b(),
            )?,
        )
    }

    /// Compile already-synthesized base and recursive arms. Both arms must
    /// come from this module's composition functions, which allocate the
    /// same current `S_mem` assignment before branch-specific F' advice.
    pub fn compile(
        base: &SparseR1cs,
        recursive: &SparseR1cs,
        plan: &NebulaPlan,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        let arms = vec![base.clone(), recursive.clone()];
        Self::compile_owned(arms, plan, None)
    }

    fn compile_owned(
        arms: Vec<SparseR1cs>,
        plan: &NebulaPlan,
        application: Option<NebulaApplication>,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        let prepared = prepare_low_norm_relation(arms, plan, 2)?;
        Self::compile_owned_selected(prepared, plan, application)
    }

    fn compile_owned_selected(
        prepared: PreparedSelectiveLowNormR1cs,
        plan: &NebulaPlan,
        application: Option<NebulaApplication>,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        let circuit = plan.circuit();
        let shape = prepared.shape_summary();
        let base = prepared.arm(0);
        let recursive = prepared.arm(1);
        let base_shape = NebulaFPrimeFieldArmShape {
            rows: base.n,
            columns: base.m,
            public_columns: base.m_in,
            poseidon2_permutations: base.poseidon2_permutations(),
        };
        let recursive_shape = NebulaFPrimeFieldArmShape {
            rows: recursive.n,
            columns: recursive.m,
            public_columns: recursive.m_in,
            poseidon2_permutations: recursive.poseidon2_permutations(),
        };
        let source_public_input_lens = [base.m_in, recursive.m_in];
        let arm_shapes = [base_shape, recursive_shape, recursive_shape];
        let relation = prepared.finish()?;
        let compiler_audit = relation.selective_compiler_audit().ok_or_else(|| {
            NebulaFPrimeRelationError::Geometry("exact selective relation has no compiler audit".into())
        })?;
        let width_audit = compiler_audit.width().clone();
        if relation_signature(relation.structure()) != shape_summary_signature(&shape)
            || relation.public_input_len() != shape.public_input_len
            || compiler_audit.width().total_coordinates != shape.total_coordinates
        {
            return Err(NebulaFPrimeRelationError::Geometry(
                "lightweight shape differs from the emitted selective relation".into(),
            ));
        }
        let remapped_ranges = remap_lane_ranges(&relation, source_public_input_lens, circuit)?;
        let mut config = relation_config(plan, application.as_ref());
        config.scheme = config.scheme.remap_ranges(remapped_ranges)?;
        Ok(Self {
            relation: Arc::new(relation),
            config,
            application,
            arm_shapes,
            width_audit: Some(Arc::new(width_audit)),
            preprocessing_digest: None,
        })
    }

    pub fn structure(&self) -> &Structure {
        self.relation.structure()
    }

    pub(crate) fn structure_arc(&self) -> std::sync::Arc<Structure> {
        self.relation.structure_arc()
    }

    pub fn public_input_len(&self) -> usize {
        self.relation.public_input_len()
    }

    pub fn nebula_config(&self) -> &NebulaConfig {
        &self.config
    }

    pub fn application(&self) -> Option<&NebulaApplication> {
        self.application.as_ref()
    }

    #[doc(hidden)]
    pub fn field_arm_shapes(&self) -> [NebulaFPrimeFieldArmShape; 3] {
        self.arm_shapes
    }

    #[doc(hidden)]
    pub fn low_norm_width_audit(&self) -> Option<&SelectiveLowNormWidthAudit> {
        self.width_audit.as_deref()
    }

    /// Return the checked, read-only compiler snapshot for conformance audits.
    #[doc(hidden)]
    pub fn selective_snapshot(
        &self,
    ) -> Result<
        crate::frontends::r1cs_f_prime::lowering::SelectiveLowNormSnapshot<'_>,
        crate::frontends::r1cs_f_prime::lowering::SelectiveSnapshotError,
    > {
        self.relation.selective_snapshot()
    }

    fn bind_program_profile(
        &self,
        plan: &NebulaPlan,
        application: Option<NebulaApplication>,
    ) -> Result<Self, NebulaFPrimeRelationError> {
        match (self.application.as_ref(), application.as_ref()) {
            (Some(reference), Some(candidate)) if !reference.same_relation_profile_as(candidate) => {
                return Err(NebulaFPrimeRelationError::PreparedProfileMismatch(
                    "application relation, recursive plan, or memory routing differs",
                ));
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(NebulaFPrimeRelationError::PreparedProfileMismatch(
                    "application presence differs",
                ));
            }
            _ => {}
        }
        if let Some(application) = application.as_ref() {
            application.validate_for(plan)?;
        }

        let mut config = relation_config(plan, application.as_ref());
        if config.steps_per_segment != self.config.steps_per_segment
            || config.seg_max != self.config.seg_max
            || config.stacks != self.config.stacks
            || config.scheme.seeded_setup() != self.config.scheme.seeded_setup()
        {
            return Err(NebulaFPrimeRelationError::PreparedProfileMismatch(
                "Nebula geometry or commitment setup differs",
            ));
        }
        config.scheme = config
            .scheme
            .remap_ranges(self.config.scheme.lane_ranges().clone())?;
        Ok(Self {
            relation: Arc::clone(&self.relation),
            config,
            application,
            arm_shapes: self.arm_shapes,
            width_audit: self.width_audit.clone(),
            preprocessing_digest: None,
        })
    }

    #[doc(hidden)]
    pub fn shares_compiled_relation_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.relation, &other.relation)
    }

    fn arm_shape(&self, branch: NebulaFPrimeBranch) -> NebulaFPrimeFieldArmShape {
        self.arm_shapes[branch.logical_index()]
    }

    pub(super) fn bind_preprocessing(&mut self, prep: &Preprocessing) -> Result<(), NebulaFPrimeRelationError> {
        let structure = self.structure();
        let prep_structure = prep.structure();
        if (structure.n, structure.m, structure.t(), structure.max_degree())
            != (
                prep_structure.n,
                prep_structure.m,
                prep_structure.t(),
                prep_structure.max_degree(),
            )
            || prep.public_input_len != Some(self.public_input_len())
        {
            return Err(NebulaFPrimeRelationError::PreprocessingMismatch);
        }
        self.preprocessing_digest = Some(*prep.structure_digest());
        Ok(())
    }

    pub fn encode(
        &self,
        branch: NebulaFPrimeBranch,
        field_assignment: &[F],
    ) -> Result<Vec<F>, NebulaFPrimeRelationError> {
        #[cfg(feature = "perf-timers")]
        let encode_started = std::time::Instant::now();
        let assignment = self.encode_for_deferred_nifs(branch, field_assignment)?;
        #[cfg(feature = "perf-timers")]
        let encode_elapsed = encode_started.elapsed();
        #[cfg(feature = "perf-timers")]
        let validate_started = std::time::Instant::now();
        if let Some(row) = self.relation.first_unsatisfied_row(&assignment) {
            #[cfg(feature = "perf-timers")]
            if let Some(audit) = self.relation.selective_compiler_audit() {
                if let Some(run) = audit
                    .rows()
                    .emitted_runs()
                    .iter()
                    .find(|run| run.emitted_rows().contains(&row))
                {
                    let source_row = run.arm().and_then(|arm| {
                        audit.rows().arms()[arm]
                            .source_runs()
                            .iter()
                            .find_map(|source| {
                                let emitted = source.emitted_start()?;
                                let source_rows = source.source_rows();
                                (emitted <= row && row < emitted + source_rows.len())
                                    .then_some(source_rows.start + row - emitted)
                            })
                    });
                    let stage = run
                        .arm()
                        .zip(run.source_stage_occurrence())
                        .and_then(|(arm, occurrence)| {
                            audit.source_arm_physical_stages()[arm]
                                .get(occurrence)
                                .map(|stage| stage.path())
                        });
                    eprintln!(
                        "[fprime-unsatisfied] row={row} family={:?} arm={:?} source_row={source_row:?} stage={stage:?} rewrite={:?}",
                        run.family(),
                        run.arm(),
                        run.rewrite_id().map(|id| id.index()),
                    );
                }
            }
            return Err(NebulaFPrimeRelationError::Unsatisfied { row });
        }
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[fprime-encode] branch={branch:?} lower={:.3}s validate={:.3}s committed={}",
            encode_elapsed.as_secs_f64(),
            validate_started.elapsed().as_secs_f64(),
            assignment.len(),
        );
        Ok(assignment)
    }

    /// Encode a synthesized arm while deferring the full relation scan to
    /// the NIFS proof that immediately consumes this instance. Encoding still
    /// enforces field width, aliases, selectors, and derived-value geometry;
    /// only the redundant prover-side sparse matrix evaluation is omitted.
    pub(super) fn encode_for_deferred_nifs(
        &self,
        branch: NebulaFPrimeBranch,
        field_assignment: &[F],
    ) -> Result<Vec<F>, NebulaFPrimeRelationError> {
        self.relation
            .encode(branch.relation_arm_index(), field_assignment)
            .map_err(Into::into)
    }

    pub(super) fn encode_signed_unit_for_deferred_nifs(
        &self,
        branch: NebulaFPrimeBranch,
        field_assignment: &[F],
    ) -> Result<NifsFreshSignedUnitAssignment, NebulaFPrimeRelationError> {
        self.relation
            .encode_signed_unit(branch.relation_arm_index(), field_assignment)
            .map_err(Into::into)
    }

    pub(super) fn build_instance_from_encoded(
        &self,
        prep: &Preprocessing,
        assignment: &[F],
    ) -> Result<CcsInstance, NebulaFPrimeRelationError> {
        let mut instance = self.ccs_instance_from_encoded(prep, assignment)?;
        self.attach_lane_commitment(&mut instance)?;
        Ok(instance)
    }

    /// Encode, commit, and attach the product-commitment sidecar used by the
    /// delayed Nebula transition. The full witness commitment and the three
    /// lane commitments are disjoint maps over the same fixed assignment.
    pub fn build_instance(
        &self,
        prep: &Preprocessing,
        branch: NebulaFPrimeBranch,
        field_assignment: &[F],
    ) -> Result<CcsInstance, NebulaFPrimeRelationError> {
        #[cfg(feature = "perf-timers")]
        let total_started = std::time::Instant::now();
        #[cfg(feature = "perf-timers")]
        let encode_started = std::time::Instant::now();
        let assignment = self.encode(branch, field_assignment)?;
        #[cfg(feature = "perf-timers")]
        let encode_elapsed = encode_started.elapsed();
        #[cfg(feature = "perf-timers")]
        let instance_started = std::time::Instant::now();
        let mut instance = self.ccs_instance_from_encoded(prep, &assignment)?;
        #[cfg(feature = "perf-timers")]
        let instance_elapsed = instance_started.elapsed();
        #[cfg(feature = "perf-timers")]
        let adv_started = std::time::Instant::now();
        self.attach_lane_commitment(&mut instance)?;
        #[cfg(feature = "perf-timers")]
        let adv_elapsed = adv_started.elapsed();
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[fprime-instance] branch={branch:?} encode={:.3}s ccs+commit={:.3}s adv={:.3}s total={:.3}s field_cols={} committed={} packed={}x{}",
            encode_elapsed.as_secs_f64(),
            instance_elapsed.as_secs_f64(),
            adv_elapsed.as_secs_f64(),
            total_started.elapsed().as_secs_f64(),
            field_assignment.len(),
            assignment.len(),
            instance.witness.Z.rows(),
            instance.witness.Z.cols(),
        );
        Ok(instance)
    }

    fn ccs_instance_from_encoded(
        &self,
        prep: &Preprocessing,
        assignment: &[F],
    ) -> Result<CcsInstance, NebulaFPrimeRelationError> {
        let structure_matches = self.preprocessing_digest.map_or_else(
            || digest::structure_digest(self.structure()) == *prep.structure_digest(),
            |bound| bound == *prep.structure_digest(),
        );
        if !structure_matches || prep.public_input_len != Some(self.public_input_len()) {
            return Err(NebulaFPrimeRelationError::PreprocessingMismatch);
        }
        CcsInstance::from_low_norm_assignment(
            &prep.params,
            &prep.log,
            prep.structure(),
            assignment,
            self.public_input_len(),
        )
        .map_err(Into::into)
    }

    pub(super) fn attach_lane_commitment(&self, instance: &mut CcsInstance) -> Result<(), NebulaFPrimeRelationError> {
        let adv = self.config.scheme.commit(&instance.witness.Z)?;
        if adv.ops.kappa != instance.claim.c.kappa {
            return Err(NebulaFPrimeRelationError::Geometry(
                "lane and full-witness commitments use different kappa".into(),
            ));
        }
        instance.claim.adv = Some(adv);
        Ok(())
    }
}

fn relation_config(plan: &NebulaPlan, application: Option<&NebulaApplication>) -> NebulaConfig {
    let mut config = plan.config();
    if let Some(application) = application {
        let initial =
            crate::frontends::r1cs_f_prime::initial_semantic_state_digest_for_plan(application.recursive_plan());
        config.initial_semantic_state_digest = digest::digest32_as_fields(initial);
    }
    config
}

fn relation_signature(structure: &Structure) -> (usize, usize, usize, u32) {
    (structure.n, structure.m, structure.t(), structure.max_degree())
}

fn verifier_relation_signature(relation: &PiCcsVerifierRelation) -> (usize, usize, usize, u32) {
    (relation.n(), relation.m(), relation.t(), relation.max_degree())
}

fn shape_summary_signature(shape: &SelectiveLowNormShapeSummary) -> (usize, usize, usize, u32) {
    (
        shape.rows,
        shape.columns,
        shape.polynomial.arity(),
        shape.polynomial.max_degree(),
    )
}

fn prepare_low_norm_relation(
    arms: Vec<SparseR1cs>,
    plan: &NebulaPlan,
    norm_base: u32,
) -> Result<PreparedSelectiveLowNormR1cs, NebulaFPrimeRelationError> {
    let circuit = plan.circuit();
    let logical_public_fields = circuit.logical_public_input_len();
    let shared_private_bit_fields = circuit.cols() - logical_public_fields;
    #[cfg(feature = "perf-timers")]
    let started = std::time::Instant::now();
    let prepared = prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        arms,
        shared_private_bit_fields,
        shared_private_bit_fields,
        D,
        logical_public_fields % D,
        norm_base,
    )?;
    #[cfg(feature = "perf-timers")]
    let shape = prepared.shape_summary();
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[fprime-low-norm-shape] shared_private_fields={shared_private_bit_fields} rows={} columns={} coordinates={} total={:.3}s",
        shape.rows,
        shape.columns,
        shape.total_coordinates,
        started.elapsed().as_secs_f64(),
    );
    Ok(prepared)
}

fn remap_lane_ranges(
    relation: &MultiBranchLowNormR1cs,
    source_public_input_lens: [usize; 2],
    circuit: &SMemCircuit,
) -> Result<LaneRanges, NebulaFPrimeRelationError> {
    let source = circuit.lane_ranges();
    Ok(LaneRanges {
        ops: remap_lane_range(relation, source_public_input_lens, circuit, source.ops)?,
        is: remap_lane_range(relation, source_public_input_lens, circuit, source.is)?,
        fs: remap_lane_range(relation, source_public_input_lens, circuit, source.fs)?,
    })
}

fn remap_lane_range(
    relation: &MultiBranchLowNormR1cs,
    source_public_input_lens: [usize; 2],
    circuit: &SMemCircuit,
    source_ring_columns: core::ops::Range<usize>,
) -> Result<core::ops::Range<usize>, NebulaFPrimeRelationError> {
    let source_start = source_ring_columns.start * D;
    let source_end = source_ring_columns.end * D;
    if source_start < circuit.m_in() || source_end > circuit.cols() {
        return Err(NebulaFPrimeRelationError::Geometry(
            "S_mem lane lies outside its private assignment prefix".into(),
        ));
    }

    let mut fixed_start = None;
    let mut expected = 0usize;
    for source_col in source_start..source_end {
        // Normalization moves only the logical public fields to the public
        // prefix. The zero fields that complete that prefix to one ring
        // column stay at the start of private advice, before these lanes.
        let private_offset = source_col - circuit.logical_public_input_len();
        let slots: Vec<(usize, usize)> = source_public_input_lens
            .iter()
            .enumerate()
            .map(|(arm, &public_input_len)| {
                relation
                    .field_slot(arm, public_input_len + private_offset)
                    .ok_or_else(|| NebulaFPrimeRelationError::Geometry("missing S_mem lane slot".into()))
            })
            .collect::<Result<_, _>>()?;
        if slots.iter().any(|slot| *slot != slots[0]) || slots[0].1 != 1 {
            return Err(NebulaFPrimeRelationError::Geometry(
                "S_mem lane is not a shared one-bit slot".into(),
            ));
        }
        let base_slot = slots[0];
        match fixed_start {
            None => {
                fixed_start = Some(base_slot.0);
                expected = base_slot.0;
            }
            Some(_) if base_slot.0 != expected => {
                return Err(NebulaFPrimeRelationError::Geometry(
                    "S_mem lane slots are not contiguous in the fixed assignment".into(),
                ))
            }
            Some(_) => {}
        }
        expected += 1;
    }
    let fixed_start = fixed_start.ok_or_else(|| NebulaFPrimeRelationError::Geometry("empty S_mem lane".into()))?;
    if fixed_start % D != 0 || expected % D != 0 {
        return Err(NebulaFPrimeRelationError::Geometry(
            "S_mem lane is not aligned to whole ring columns after fixed-shape lowering".into(),
        ));
    }
    Ok(fixed_start / D..expected / D)
}

/// Wires of one composed application/F' execution.
pub struct NebulaFPrimeStepOutput {
    pub f_prime: FPrimeStepOutput,
    /// Exact `S_mem` assignment wires, including its constant-one column.
    pub s_mem: Vec<Var>,
    /// Application assignment wires. Empty for the memory-only relation.
    pub application: Vec<Var>,
    /// `[step_x_bits || open || bits(D_pre)]` produced for the next step.
    pub current_public_suffix: Vec<Var>,
}

impl NebulaFPrimeStepOutput {
    /// Public field columns passed to `lower_field_r1cs`. The lowering adds
    /// the one implicit constant column in front of this sequence.
    pub fn public_outputs(&self) -> Vec<Var> {
        let mut out = self.f_prime.x_out_bits.clone();
        out.extend_from_slice(&self.current_public_suffix);
        out
    }
}

pub fn enforce_nebula_f_prime_base_step(
    builder: &mut R1csBuilder,
    s_mem: &SMemCircuit,
    s_mem_assignment: &[F],
    current_d_pre: Option<[[F; 4]; 3]>,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeBaseInputs<'_>,
) -> Result<NebulaFPrimeStepOutput, NebulaFPrimeError> {
    builder.begin_encoding_stage(fprime_stage::BASE_ROOT);
    builder.begin_encoding_stage(fprime_stage::BASE_APPLICATION);
    let current = enforce_current_application(builder, s_mem, s_mem_assignment, None, current_d_pre, cfg)?;
    let f_prime_column_start = builder.witness().len();
    let f_prime = enforce_f_prime_base_step_circuit(builder, cfg, inputs)?;
    builder.record_column_family("nebula.f_prime", f_prime_column_start);
    Ok(NebulaFPrimeStepOutput {
        f_prime,
        s_mem: current.s_mem,
        application: current.application,
        current_public_suffix: current.public_suffix,
    })
}

pub fn enforce_nebula_f_prime_recursive_step(
    builder: &mut R1csBuilder,
    pp: &Params,
    s_mem: &SMemCircuit,
    s_mem_assignment: &[F],
    current_d_pre: Option<[[F; 4]; 3]>,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeRecursiveInputs<'_>,
) -> Result<NebulaFPrimeStepOutput, NebulaFPrimeError> {
    builder.begin_encoding_stage(fprime_stage::RECURSIVE_ROOT);
    builder.begin_encoding_stage(fprime_stage::RECURSIVE_APPLICATION);
    let current = enforce_current_application(builder, s_mem, s_mem_assignment, None, current_d_pre, cfg)?;
    let f_prime_column_start = builder.witness().len();
    let f_prime = enforce_f_prime_recursive_step_circuit(builder, pp, cfg, inputs)?;
    builder.record_column_family("nebula.f_prime", f_prime_column_start);
    Ok(NebulaFPrimeStepOutput {
        f_prime,
        s_mem: current.s_mem,
        application: current.application,
        current_public_suffix: current.public_suffix,
    })
}

pub fn enforce_nebula_application_f_prime_base_step(
    builder: &mut R1csBuilder,
    s_mem: &SMemCircuit,
    s_mem_assignment: &[F],
    application: &NebulaApplication,
    application_assignment: &[F],
    current_d_pre: Option<[[F; 4]; 3]>,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeBaseInputs<'_>,
) -> Result<NebulaFPrimeStepOutput, NebulaFPrimeError> {
    builder.begin_encoding_stage(fprime_stage::BASE_ROOT);
    builder.begin_encoding_stage(fprime_stage::BASE_APPLICATION);
    let current = enforce_current_application(
        builder,
        s_mem,
        s_mem_assignment,
        Some((application, application_assignment)),
        current_d_pre,
        cfg,
    )?;
    let f_prime_column_start = builder.witness().len();
    let f_prime = enforce_f_prime_base_step_circuit(builder, cfg, inputs)?;
    builder.record_column_family("nebula.f_prime", f_prime_column_start);
    if let Some(semantic) = current.semantic {
        builder.begin_encoding_stage(fprime_stage::BASE_SEMANTIC_LINKS);
        crate::frontends::r1cs_f_prime::ivc::shape::bind_semantic_state(
            builder,
            application.recursive_plan(),
            &f_prime,
            semantic,
            true,
        );
    }
    Ok(NebulaFPrimeStepOutput {
        f_prime,
        s_mem: current.s_mem,
        application: current.application,
        current_public_suffix: current.public_suffix,
    })
}

pub fn enforce_nebula_application_f_prime_recursive_step(
    builder: &mut R1csBuilder,
    pp: &Params,
    s_mem: &SMemCircuit,
    s_mem_assignment: &[F],
    application: &NebulaApplication,
    application_assignment: &[F],
    current_d_pre: Option<[[F; 4]; 3]>,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeRecursiveInputs<'_>,
) -> Result<NebulaFPrimeStepOutput, NebulaFPrimeError> {
    builder.begin_encoding_stage(fprime_stage::RECURSIVE_ROOT);
    builder.begin_encoding_stage(fprime_stage::RECURSIVE_APPLICATION);
    let current = enforce_current_application(
        builder,
        s_mem,
        s_mem_assignment,
        Some((application, application_assignment)),
        current_d_pre,
        cfg,
    )?;
    let f_prime_column_start = builder.witness().len();
    let f_prime = enforce_f_prime_recursive_step_circuit(builder, pp, cfg, inputs)?;
    builder.record_column_family("nebula.f_prime", f_prime_column_start);
    if let Some(semantic) = current.semantic {
        builder.begin_encoding_stage(fprime_stage::RECURSIVE_SEMANTIC_LINKS);
        crate::frontends::r1cs_f_prime::ivc::shape::bind_semantic_state(
            builder,
            application.recursive_plan(),
            &f_prime,
            semantic,
            false,
        );
    }
    Ok(NebulaFPrimeStepOutput {
        f_prime,
        s_mem: current.s_mem,
        application: current.application,
        current_public_suffix: current.public_suffix,
    })
}

struct CurrentApplication {
    s_mem: Vec<Var>,
    application: Vec<Var>,
    public_suffix: Vec<Var>,
    semantic: Option<crate::frontends::r1cs_f_prime::ivc::shape::SemanticWires>,
}

fn enforce_current_application(
    builder: &mut R1csBuilder,
    circuit: &SMemCircuit,
    assignment: &[F],
    application: Option<(&NebulaApplication, &[F])>,
    current_d_pre: Option<[[F; 4]; 3]>,
    cfg: &FPrimeStepConfig<'_>,
) -> Result<CurrentApplication, NebulaFPrimeError> {
    let nebula = cfg.nebula.ok_or(NebulaFPrimeError::MissingNebulaConfig)?;
    let expected_step_width = nebula.stacks.x_bits();
    let actual_step_width = circuit.logical_public_input_len() - 1;
    if actual_step_width != expected_step_width {
        return Err(NebulaFPrimeError::StepPublicWidth {
            actual: actual_step_width,
            expected: expected_step_width,
        });
    }
    let expected_suffix = delayed_nebula_public_suffix_len(nebula.stacks);
    if cfg.public_input_layout.suffix_len() != expected_suffix {
        return Err(NebulaFPrimeError::SuffixWidth {
            actual: cfg.public_input_layout.suffix_len(),
            expected: expected_suffix,
        });
    }

    let s_mem_start = builder.rows();
    let s_mem_column_start = builder.witness().len();
    let s_mem = if application.is_some() {
        let vars = circuit.allocate_r1cs_assignment(builder, assignment)?;
        builder.record_row_family("nebula.application.s_mem_assignment", s_mem_start);
        builder.record_column_family("nebula.application.s_mem_assignment", s_mem_column_start);
        vars
    } else {
        let vars = circuit.enforce_in_r1cs(builder, assignment)?;
        builder.record_row_family("nebula.application.s_mem", s_mem_start);
        builder.record_column_family("nebula.application.s_mem", s_mem_column_start);
        vars
    };
    let (application_vars, semantic) = if let Some((application, application_assignment)) = application {
        let relation_start = builder.rows();
        let relation_column_start = builder.witness().len();
        let vars = application.shape().enforce_in_f_prime(
            builder,
            application_assignment,
            crate::frontends::r1cs_f_prime::ivc::shape::pin_app_constant(application.recursive_plan()),
        )?;
        builder.record_row_family("nebula.application.relation", relation_start);
        builder.record_column_family("nebula.application.relation", relation_column_start);
        let s_mem_constraints_start = builder.rows();
        let s_mem_constraints_column_start = builder.witness().len();
        circuit.enforce_allocated_r1cs(builder, &s_mem)?;
        builder.record_row_family("nebula.application.s_mem_constraints", s_mem_constraints_start);
        builder.record_column_family("nebula.application.s_mem_constraints", s_mem_constraints_column_start);
        let memory_start = builder.rows();
        let memory_column_start = builder.witness().len();
        enforce_memory_ports(
            builder,
            circuit,
            &s_mem,
            application_assignment,
            &vars,
            application.memory(),
        )?;
        builder.record_row_family("nebula.application.memory_ports", memory_start);
        builder.record_column_family("nebula.application.memory_ports", memory_column_start);
        let semantic_start = builder.rows();
        let semantic_column_start = builder.witness().len();
        let semantic = crate::frontends::r1cs_f_prime::ivc::shape::enforce_semantic_digests(
            builder,
            application.recursive_plan(),
            application_assignment,
            &vars,
        )?;
        builder.record_row_family("nebula.application.semantic", semantic_start);
        builder.record_column_family("nebula.application.semantic", semantic_column_start);
        (vars, Some(semantic))
    } else {
        (Vec::new(), None)
    };
    let suffix_column_start = builder.witness().len();
    let mut suffix = s_mem[1..circuit.logical_public_input_len()].to_vec();
    let open = builder.alloc(if current_d_pre.is_some() { F::ONE } else { F::ZERO });
    enforce_bit(builder, open);
    suffix.push(open);

    let mut not_open = Lc::from_const(F::ONE);
    not_open.add_term(open, -F::ONE);
    for digest in current_d_pre.unwrap_or([[F::ZERO; 4]; 3]) {
        for lane in digest {
            let value = lane.as_canonical_u64();
            for bit in 0..64 {
                let wire = builder.alloc(F::from_u64((value >> bit) & 1));
                enforce_bit(builder, wire);
                builder.enforce(&not_open, &Lc::from_var(wire), &Lc::zero());
                suffix.push(wire);
            }
        }
    }
    builder.record_column_family("nebula.application.public_suffix", suffix_column_start);
    debug_assert_eq!(suffix.len(), expected_suffix);
    Ok(CurrentApplication {
        s_mem,
        application: application_vars,
        public_suffix: suffix,
        semantic,
    })
}
