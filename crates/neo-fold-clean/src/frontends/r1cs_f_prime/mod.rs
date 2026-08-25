//! Authoritative recursive R1CS F' relation and its low-norm lowering.
//!
//! The public lifecycle is [`ivc`]. The remaining modules own relation
//! construction, selective lowering, audit artifacts, and terminal Spartan
//! verification.

mod grouped_phase;
pub mod ivc;
pub mod lean_manifest;
pub mod lean_native_ccs_manifest;
pub mod lean_nebula_combined_manifest;
mod linked_overlay;
pub mod lowering;
pub mod native_ccs;
pub mod nebula_combined_ccs;
mod relation_artifact;
mod selective;
mod selective_audit;
mod selective_census;
mod selective_row_artifact;
mod selective_selection_audit;
mod selective_selector_coverage;
pub mod structure;
pub mod terminal_r1cs;
mod ternary_encoding;

pub use grouped_phase::{
    build_grouped_phase_low_norm_r1cs, build_scheduled_grouped_phase_low_norm_r1cs,
    build_scheduled_grouped_phase_low_norm_r1cs_with_field_links, GroupedPhaseError, GroupedPhaseLayout,
    GroupedPhaseLowNormR1cs, ScheduledCommonPhaseFieldLink, ScheduledCursorBits, ScheduledGroupedPhaseError,
    ScheduledGroupedPhaseLayout, ScheduledGroupedPhaseLowNormR1cs, ScheduledPhaseKindLinks,
};
pub use lean_native_ccs_manifest::LeanNativeCcsManifest;
pub use lean_nebula_combined_manifest::{LeanNebulaCombinedManifest, NebulaCombinedEmission};
pub use linked_overlay::{
    build_linked_overlay_low_norm_r1cs, build_scheduled_linked_overlay_low_norm_r1cs,
    build_scheduled_linked_overlay_low_norm_r1cs_with_phase_field_links, LinkedOverlayError, LinkedOverlayLayout,
    LinkedOverlayLowNormR1cs, OverlayBaseFieldPin, OverlayFieldLink, OverlayKindLinks, ScheduledLinkedOverlayLayout,
    ScheduledLinkedOverlayLowNormR1cs,
};
pub(crate) use lowering::normalized_field_column;
pub use lowering::{
    build_fixed_shape_low_norm_r1cs, build_fixed_shape_low_norm_r1cs_with_shared_private_prefix,
    build_multi_branch_low_norm_r1cs, build_multi_branch_low_norm_r1cs_with_alignment, lower_field_r1cs,
    lower_sparse_r1cs_to_low_norm, FieldR1csLoweringError, FixedR1csBranch, FixedShapeLowNormR1cs,
    LowNormEncoderArtifactError, LowNormEncoderArtifactLimits, LowNormEncoderArtifactReceipt, LowNormR1cs,
    LowNormR1csError, LoweredFieldR1cs, MultiBranchLowNormR1cs, VerifiedLowNormEncoderArtifact,
};
pub use native_ccs::{LeanNativeCcsError, LeanNativeCcsPreprocessing};
pub use nebula_combined_ccs::{LeanNebulaCombinedError, LeanNebulaCombinedPreprocessing};
pub use relation_artifact::{R1CS_F_PRIME_COMPILER_ID, R1CS_F_PRIME_CONTRACT_ID, R1CS_F_PRIME_PROFILE_ID};
pub(crate) use selective::{
    audit_multi_branch_selective_compact_layout_and_decoder_runs_with_shared_bit_prefix,
    audit_multi_branch_selective_compiler_with_shared_bit_prefix,
    audit_multi_branch_selective_decoder_runs_with_shared_bit_prefix,
    audit_multi_branch_selective_low_norm_shape_with_alignment,
    audit_multi_branch_selective_low_norm_width_for_norm_base_with_alignment,
    prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix,
    project_rows_with_complete_source_provenance_with_alignment, PreparedSelectiveLowNormR1cs, SelectiveLowNormShape,
    SelectiveLowNormShapeSummary,
};
pub use selective::{
    audit_multi_branch_selective_low_norm_width_with_alignment,
    audit_multi_branch_selective_low_norm_width_with_shared_bit_prefix,
    audit_multi_branch_selective_rows_with_alignment,
    audit_multi_branch_selective_rows_with_complete_source_provenance_with_alignment,
    build_multi_branch_selective_low_norm_r1cs_with_alignment,
    build_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix, SelectiveCompactLayoutAudit,
    SelectiveProjectedDecoderProvenance, SelectiveProjectedDecoderRunProvenance, SelectiveProjectedDerivedProductSum,
    SelectiveProjectedExplicitRunCensus, SelectiveProjectedGeometricRun, SelectiveProjectedPort,
    SelectiveProjectedPoseidon2OutputStep, SelectiveProjectedPoseidon2SboxStep, SelectiveProjectedProductFactor,
    SelectiveProjectedPublicCoordinate, SelectiveProjectedPublicCoordinateSource, SelectiveProjectedRetainedStep,
    SelectiveProjectedRewriteOutput, SelectiveProjectedRewriteStep, SelectiveProjectedRowArtifact,
    SelectiveProjectedRowsAudit, SelectiveProjectedSourceDecoder, SelectiveProjectedSourceDecoderRun,
    SelectiveProjectedSourceDecoderStridedRun, SelectiveProjectedSourceDecoderTemplate,
    SelectiveProjectedSourceDecoderTemplateInstances, SelectiveProjectedSourceDefinition,
    SelectiveProjectedSourceFamilyRange, SelectiveProjectedSourceImage, SelectiveProjectedSourceLinearCombination,
    SelectiveProjectedSourceProvenance, SelectiveProjectedSourceResolution, SelectiveProjectedSourceResolutionRun,
    SelectiveProjectedSourceSlot, SelectiveProjectedSourceTerm, SelectiveProjectedTerm,
};
#[doc(hidden)]
pub use selective::{is_canonical_selective_low_norm_polynomial, selective_polynomial};
pub use selective_audit::{
    SelectiveArmRowMappingAudit, SelectiveArmWidthAudit, SelectiveCanonicalOpeningAudit, SelectiveCompilerAudit,
    SelectiveEmittedRowFamily, SelectiveEmittedRowRunAudit, SelectiveFamilyWidthAudit, SelectiveLayoutAudit,
    SelectiveLinearDefinitionAudit, SelectiveLinearDefinitionTermAudit, SelectiveLowNormWidthAudit,
    SelectivePhysicalStageWidthAudit, SelectiveRewriteAudit, SelectiveRewriteId, SelectiveRewriteKind,
    SelectiveRowMappingAudit, SelectiveSourceRowDisposition, SelectiveSourceRowRunAudit, SelectiveTraceWidthAudit,
};
pub use selective_census::{
    SelectiveMatrixTag, SelectivePortCensus, SelectiveStructureCensus, SelectiveStructureCensusError,
};
pub use selective_row_artifact::{
    SelectiveMatrixRow, SelectiveRowArtifact, SelectiveRowArtifactError, SelectiveRowTerm,
    SELECTIVE_ROW_ARTIFACT_SCHEMA_VERSION,
};
pub use selective_selection_audit::SelectiveFirstAcceptedSelectionAudit;
pub use selective_selector_coverage::{
    SelectiveGatePort, SelectiveSelectorGateCoverage, SelectiveSelectorGateCoverageError, SelectiveSelectorGateRun,
    SelectiveSelectorOwnerGateRun, SelectiveSelectorPolynomialTerm, SELECTIVE_SELECTOR_GATE_COVERAGE_SCHEMA_VERSION,
};
pub use structure::{build_r1cs_f_prime_structure, R1csRowAnchors, R1csShape, SparseR1cs};
pub use terminal_r1cs::{
    compile_combined_terminal_r1cs, compile_combined_terminal_r1cs_statement, finish_combined_with_spartan,
    finish_with_spartan, verify_combined_spartan, verify_spartan, TerminalR1csError, TerminalSpartanProof,
    TerminalSpartanStatement,
};

use thiserror::Error;

use crate::frontends::f_prime::recursive_plan::{RecursiveStepImagePlan, StateXOutPlanOptions};
use crate::paper::construction2::SemanticStateMode;

pub(crate) fn semantic_state_mode_for_plan(plan: &RecursiveStepImagePlan) -> SemanticStateMode {
    match plan.state_x_out.as_ref() {
        Some(state) if has_semantic_binding(state) => SemanticStateMode::Stateful,
        _ => SemanticStateMode::Stateless,
    }
}

fn has_semantic_binding(state: &StateXOutPlanOptions) -> bool {
    !state.semantic_state_in_var_indices.is_empty()
        || !state.semantic_state_out_var_indices.is_empty()
        || !state.app_public_input_var_indices.is_empty()
        || !state.app_public_input_bit_var_indices.is_empty()
}

pub(crate) fn initial_semantic_state_digest_for_plan(plan: &RecursiveStepImagePlan) -> [u8; 32] {
    plan.state_x_out
        .as_ref()
        .and_then(|state| state.initial_semantic_state_digest_anchor)
        .unwrap_or_else(crate::paper::digest::empty_semantic_state_digest)
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("R1CS F' plan has {plan_limbs} limbs; the application needs {expected}")]
    PlanLimbsMismatch { plan_limbs: usize, expected: usize },
    #[error("R1CS F' plan has {got} application widths; the R1CS has {expected} variables")]
    PlanAppPrivateWidthCountMismatch { got: usize, expected: usize },
    #[error("R1CS F' application width at variable {index} is {width}; it must be in 1..=64")]
    PlanAppPrivateWidthInvalid { index: usize, width: usize },
    #[error("R1CS F' application width {width} at variable {index} is below the proven width {proven_width}")]
    PlanAppPrivateWidthTooNarrow {
        index: usize,
        width: usize,
        proven_width: usize,
    },
    #[error("R1CS F' packed public variable {index} is not Boolean-constrained")]
    PlanPackedPublicInputBooleanUnconstrained { index: usize },
    #[error("R1CS F' packed public variable {index} has width {width}; it must have width 1")]
    PlanPackedPublicInputWidthNotOne { index: usize, width: usize },
    #[error("R1CS F' plan has no state_x_out binding")]
    PlanMissingStateXOut,
    #[error("R1CS F' public binding does not cover variables 0..{m_in} (field={actual:?}, bits={actual_bit:?})")]
    PlanAppPublicInputMismatch {
        actual: Vec<usize>,
        actual_bit: Vec<usize>,
        m_in: usize,
    },
    #[error("R1CS F' semantic input has no semantic output")]
    PlanSemanticStatePartial,
    #[error("R1CS F' semantic input has no verifier-owned initial digest")]
    PlanSemanticStateMissingAnchor,
    #[error("R1CS F' plan has an initial semantic digest but no semantic binding")]
    PlanSemanticStateAnchorWithoutIndices,
    #[error("R1CS F' semantic variable {index} is outside the {m}-variable application")]
    PlanSemanticStateIndexOutOfRange { index: usize, m: usize },
    #[error("R1CS F' public variable {index} is not part of the explicit semantic transition")]
    PlanPublicInputNotSemanticBound { index: usize },
}

/// Validate the verifier-owned application and semantic-state shape.
pub(crate) fn validate_plan(plan: &RecursiveStepImagePlan, r1cs: &R1csShape) -> Result<(), Error> {
    let state = plan
        .state_x_out
        .as_ref()
        .ok_or(Error::PlanMissingStateXOut)?;
    let boolean_vars = r1cs.boolean_constrained_variables();
    let proven_widths = if plan.app_private_var_widths.is_empty() {
        Vec::new()
    } else {
        r1cs.conservative_app_private_var_widths()
    };
    if !plan.app_private_var_widths.is_empty() && plan.app_private_var_widths.len() != r1cs.m() {
        return Err(Error::PlanAppPrivateWidthCountMismatch {
            got: plan.app_private_var_widths.len(),
            expected: r1cs.m(),
        });
    }
    if let Some((index, &width)) = plan
        .app_private_var_widths
        .iter()
        .enumerate()
        .find(|(_, &width)| !(1..=64).contains(&width))
    {
        return Err(Error::PlanAppPrivateWidthInvalid { index, width });
    }
    if let Some((index, &width)) = plan
        .app_private_var_widths
        .iter()
        .enumerate()
        .find(|(index, width)| **width < proven_widths[*index])
    {
        return Err(Error::PlanAppPrivateWidthTooNarrow {
            index,
            width,
            proven_width: proven_widths[index],
        });
    }
    let app_bits = if plan.app_private_var_widths.is_empty() {
        r1cs.m() * 64
    } else {
        plan.app_private_var_widths.iter().sum()
    };
    let expected_limbs = app_bits + 1;
    if plan.limbs != expected_limbs {
        return Err(Error::PlanLimbsMismatch {
            plan_limbs: plan.limbs,
            expected: expected_limbs,
        });
    }

    let expected: Vec<_> = (0..r1cs.m_in()).collect();
    let field_binding =
        state.app_public_input_var_indices == expected && state.app_public_input_bit_var_indices.is_empty();
    let bit_binding =
        state.app_public_input_var_indices.is_empty() && state.app_public_input_bit_var_indices == expected;
    if !field_binding && !bit_binding {
        return Err(Error::PlanAppPublicInputMismatch {
            actual: state.app_public_input_var_indices.clone(),
            actual_bit: state.app_public_input_bit_var_indices.clone(),
            m_in: r1cs.m_in(),
        });
    }
    if bit_binding {
        for &index in &state.app_public_input_bit_var_indices {
            let width = plan
                .app_private_var_widths
                .get(index)
                .copied()
                .unwrap_or(64);
            if width != 1 {
                return Err(Error::PlanPackedPublicInputWidthNotOne { index, width });
            }
            if index != 0 && !boolean_vars[index] {
                return Err(Error::PlanPackedPublicInputBooleanUnconstrained { index });
            }
        }
    }

    let has_input = !state.semantic_state_in_var_indices.is_empty();
    let has_explicit_output = !state.semantic_state_out_var_indices.is_empty();
    let has_output = has_explicit_output || field_binding || bit_binding;
    if has_input && !has_output {
        return Err(Error::PlanSemanticStatePartial);
    }
    for &index in state
        .semantic_state_in_var_indices
        .iter()
        .chain(&state.semantic_state_out_var_indices)
    {
        if index >= r1cs.m() {
            return Err(Error::PlanSemanticStateIndexOutOfRange { index, m: r1cs.m() });
        }
    }
    if has_explicit_output {
        for index in state
            .app_public_input_var_indices
            .iter()
            .chain(&state.app_public_input_bit_var_indices)
            .copied()
        {
            if !state.semantic_state_in_var_indices.contains(&index)
                && !state.semantic_state_out_var_indices.contains(&index)
            {
                return Err(Error::PlanPublicInputNotSemanticBound { index });
            }
        }
    }
    let has_anchor = state.initial_semantic_state_digest_anchor.is_some();
    if has_input && !has_anchor {
        return Err(Error::PlanSemanticStateMissingAnchor);
    }
    if !has_input && !has_output && has_anchor {
        return Err(Error::PlanSemanticStateAnchorWithoutIndices);
    }
    Ok(())
}
