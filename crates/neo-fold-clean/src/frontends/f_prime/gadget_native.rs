//! Exact gadget-native low-norm lowering for field-valued R1CS.
//!
//! Owns: validation, lowering orchestration, assignment materialization, and
//! constraint emission for traced low-norm replacements.
//!
//! Does not own: source relation semantics, source trace emission, or the
//! inverse-plan API and decoding implementation (owned by `plan`).
//!
//! Emits constraints: yes, in the lowered CCS relation.
//!
//! Authority boundary: the source R1CS is the local implementation arithmetic
//! reference, not the protocol semantics. Every untraced row is lowered
//! generically, and each traced replacement must match its source rows exactly
//! to avoid introducing new implementation divergence. Permission to retain or
//! remove a protocol obligation additionally requires the independent Lean
//! paper-semantics and necessity/refinement chain.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | Poseidon2 | Preserve `x^7` while removing product temporaries | yes | this file | existing Poseidon2 refinement |
//! | Extension field | Preserve both Karatsuba output limbs | yes | this file | open concrete trace bridge |
//! | Balanced ternary | Validate all 124 source rows and classify shared coordinates | no | `balanced_ternary` | `SisLowering`; exact shared-slot bridge open |
//! | Ordinary private field | Exact 41-coordinate shifted centered materialization and linear decode; retain the local centered family | yes, via `coordinate_gates` | `ordinary_private_field` | `CenteredTernaryField.decode_encodeDigit`; production refinement open |
//! | Value slots | Place/decode Boolean, canonical-binary, ordinary-centered, centered aliases, and SIS-balanced values | canonical prefix rows | `slots` | linear-substitution bridge open |
//! | Inverse plan | Reconstruct every source column from the accepted low-norm assignment | no | `plan` / `model` | exact whole-plan refinement open |
//! | Product-sum batch | Preserve mixed SSA identities and reconstruction | yes | `product_sum` | `PiRlcAlgebra.Refinement.ProductSum` |
//! | Chunk acceptance | Replace four canonical inverse rows by a 14-edge aggregate tree | yes | `acceptance` | `PiRlcChallenge.Refinement.AggregateAcceptanceRows` |
//! | Packed mod-5 | Replace 20 source rows by eight nonresidue-packed equations | yes | `mod5` | `PiRlcChallenge.Refinement.PackedChunkRows` |
//! | Ring action | Preserve production Toom-3 convolution | yes | this file | exact-or-bad-root boundary is separate |
//! | First-accepted selection | Replace aggregates by guarded equations | yes | `selection` | `PiRlcChallenge.Refinement.SelectionRows` |
//! | Canonical u64 | Validate 69 source rows and report role/stage census | no | `canonical_u64` | exact refinement open |
//! | Stage profile | Attribute source/lowered costs to production paths | no | `profile` | diagnostic only |

use std::collections::BTreeMap;

use neo_ccs::CcsStructure;
use neo_math::{D, F};
use p3_field::{Field, PrimeCharacteristicRing};
use thiserror::Error;

use crate::engine::r1cs_circuit::builder::BALANCED_TERNARY_DIGITS;
use crate::engine::r1cs_circuit::{Lc, R1csEncodingTrace, R1csSnapshot, RingMulToom3TraceEntry, Sbox7TraceEntry, Var};

mod acceptance;
mod balanced_ternary;
mod boolean_dedup;
mod canonical_u64;
mod coordinate_gates;
mod fresh_assignment;
mod gates;
mod mod5;
mod model;
mod ordinary_private_field;
mod plan;
mod product_sum;
mod profile;
mod projection_identity;
mod selection;
mod selector_gated;
mod shared_slots;
mod slots;
mod source_allocation;
mod source_manifest;
mod source_schedule;
#[doc(hidden)]
pub use acceptance::{
    audit_r1cs_gadget_native_aggregate_acceptance_outer_image, AggregateAcceptanceAudit,
    AggregateAcceptanceBitOuterImage, AggregateAcceptanceBooleanRowOwner, AggregateAcceptanceChunkOuterImageAudit,
    AggregateAcceptanceDecodedImage, AggregateAcceptanceLinearDefinitionAudit, AggregateAcceptanceMatrixRowAudit,
    AggregateAcceptanceOuterImageAudit, AggregateAcceptancePhysicalRowAudit, AggregateAcceptanceSourceRowAudit,
};
pub use canonical_u64::{
    audit_r1cs_gadget_native_canonical_u64, CanonicalU64Audit, CanonicalU64AuditEntry, CanonicalU64Census,
    CanonicalU64Classification, CanonicalU64StageCensus,
};
pub use coordinate_gates::{
    GadgetNativeBooleanFamily, GadgetNativeBooleanPairingBreakdown, GadgetNativeCenteredFamily,
    GadgetNativeCoordinateGateSchedule, GadgetNativeCoordinateGroupAudit, GadgetNativeCoordinateGroupFamily,
    GadgetNativeCoordinateRowAudit, GadgetNativePairTailCount,
};
use gates::{gate, one_selector, TraceGateBuilder};
#[doc(hidden)]
pub use gates::{GadgetNativeCoordinateGateRoles, GADGET_NATIVE_RESIDUAL_PAIR_NONRESIDUE};
#[doc(hidden)]
pub use mod5::{PackedMod5DecoderAudit, PackedMod5ProductDecoderAudit};
use model::{LinearDefinition, ProductDefinition, RingSyntheticSlots, SourceColumn, TraceMarks};
#[doc(hidden)]
pub use ordinary_private_field::{
    encode_ordinary_private_field, ORDINARY_PRIVATE_DIGITS, ORDINARY_PRIVATE_RADIX_40, ORDINARY_PRIVATE_RADIX_41,
    ORDINARY_PRIVATE_SHIFT,
};
pub use plan::{GadgetNativePlan, GadgetNativePlanTestMutation};
pub use profile::{
    profile_r1cs_gadget_native_stages, GadgetNativeCanonicalBinaryFieldRowBreakdown, GadgetNativeEncodedRowBreakdown,
    GadgetNativeEstimate, GadgetNativeStageEstimate, GadgetNativeStageProfile, GadgetNativeStageProfileError,
};
#[doc(hidden)]
pub use projection_identity::{
    audit_projection_identity_compaction, ProjectionCoefficientZero, ProjectionEvaluationCompactionAudit,
    ProjectionEvaluationKind, ProjectionFinalCoefficient, ProjectionFinalFactorAudit,
    ProjectionFinalLimbCompactionAudit, ProjectionFinalOperand, ProjectionIdentityCompactionAudit,
    ProjectionIdentityCompactionSchema, ProjectionRetainedBindingAudit,
};
pub use selector_gated::{
    estimate_selector_gated_r1cs_gadget_native, SelectorGatedGadgetNativeCostLayout, SelectorGatedGadgetNativeEstimate,
};
pub use shared_slots::{BalancedTernarySharedSlotPlan, GadgetNativeConstraintRow};
use slots::{
    decode_slot, emit_goldilocks_canonicality, push_balanced_ternary_slot, push_boolean_slot, push_field_slot,
    push_ordinary_private_field_slot, slot_terms, ValueSlot, GOLDILOCKS_CANONICALITY_PAIR_ROWS,
};
pub use source_allocation::{
    audit_r1cs_gadget_native_ordinary_placement, gadget_native_source_loop_width, GadgetNativeOrdinaryPlacement,
    GadgetNativeOrdinaryPlacementManifest, GadgetNativeOrdinaryPlacementManifestTestMutation,
};
pub use source_manifest::{
    audit_r1cs_gadget_native_source_manifest, GadgetNativeSourceManifest, GadgetNativeSourceManifestTestMutation,
};
pub use source_schedule::GadgetNativeSourceRole;

const FIELD_BITS: usize = 64;
const HIGH_BITS_START: usize = 32;
const CANONICAL_PREFIX_AUX: usize = 31;
const CANONICAL_SLOT_WIDTH: usize = FIELD_BITS + CANONICAL_PREFIX_AUX;
const TOOM_SPLIT: usize = 18;
const TOOM_COEFFICIENTS: usize = 2 * TOOM_SPLIT - 1;
const TOOM_EVALUATIONS: usize = 5;
const MAX_PRODUCT_TERMS: usize = TOOM_SPLIT;

fn canonical_superneo_public_input_len(public_bits: usize) -> Result<usize, GadgetNativeError> {
    let logical_len = 1usize
        .checked_add(public_bits)
        .ok_or(GadgetNativeError::SourceAllocationOverflow { column: 0 })?;
    logical_len
        .div_ceil(D)
        .checked_mul(D)
        .ok_or(GadgetNativeError::SourceAllocationOverflow { column: 0 })
}

/// One materialized gadget-native CCS instance before Ajtai commitment.
#[derive(Clone, Debug)]
pub struct EncodedGadgetNativeR1cs {
    pub structure: CcsStructure<F>,
    pub assignment: Vec<F>,
    pub plan: GadgetNativePlan,
}

impl EncodedGadgetNativeR1cs {
    pub fn public_input(&self) -> &[F] {
        &self.assignment[..self.plan.public_input_len]
    }

    pub fn private_witness(&self) -> &[F] {
        &self.assignment[self.plan.public_input_len..]
    }

    pub fn first_unsatisfied_row(&self) -> Option<usize> {
        evaluate_ccs(&self.structure, &self.assignment)
            .into_iter()
            .position(|value| value != F::ZERO)
    }

    pub fn is_satisfied(&self) -> bool {
        self.first_unsatisfied_row().is_none()
    }

    pub fn decode_source(&self) -> Result<Vec<F>, GadgetNativeError> {
        if let Some(row) = self.first_unsatisfied_row() {
            return Err(GadgetNativeError::UnsatisfiedEncoding { row });
        }
        self.plan.decode_source(&self.assignment)
    }
}

#[derive(Debug, Error)]
pub enum GadgetNativeError {
    #[error("gadget-native encoding requires source column zero to equal one")]
    SourceConstantOne,
    #[error("public source column {column} is out of range for {cols} columns")]
    PublicColumnOutOfRange { column: usize, cols: usize },
    #[error("public source column {column} appears more than once")]
    DuplicatePublicColumn { column: usize },
    #[error("public source column {column} is not explicitly Boolean")]
    PublicColumnNotBoolean { column: usize },
    #[error("selector-gated branches expose different public bit counts ({base} versus {recursive})")]
    BranchPublicInputLength { base: usize, recursive: usize },
    #[error("encoding trace {gadget} row range {start}..{end} is malformed for a {rows}-row source")]
    TraceRowRange {
        gadget: &'static str,
        start: usize,
        end: usize,
        rows: usize,
    },
    #[error("encoding trace row {row} is claimed by more than one gadget")]
    OverlappingTraceRow { row: usize },
    #[error("encoding trace for {gadget} does not match source R1CS row {row}")]
    TraceRowMismatch { gadget: &'static str, row: usize },
    #[error("encoding trace {gadget} has malformed fixed arity")]
    TraceArity { gadget: &'static str },
    #[error("source column {column} is projected by more than one gadget")]
    DuplicateGadgetDefinition { column: usize },
    #[error("source column {column} has conflicting source-schedule ownership: {detail}")]
    SourceDecisionConflict { column: usize, detail: &'static str },
    #[error("projected source column {column} has no recognized exact decision")]
    UnclassifiedProjectedSourceColumn { column: usize },
    #[error("source column {column} is not eligible for any exact materialization decision")]
    UnclassifiedSourceColumn { column: usize },
    #[error("source decision width {got} does not match expected width {expected}")]
    SourceDecisionWidth { expected: usize, got: usize },
    #[error("materialized source column {column} disagrees with its exact decision")]
    SourceMaterializationMismatch { column: usize },
    #[error("source allocation for column {column} expected {expected_start}..{expected_end}, got {actual_start}..{actual_end}")]
    SourceAllocationMismatch {
        column: usize,
        expected_start: usize,
        expected_end: usize,
        actual_start: usize,
        actual_end: usize,
    },
    #[error("source allocation overflowed while visiting column {column}")]
    SourceAllocationOverflow { column: usize },
    #[error("source allocation phase ended at {actual}; expected {expected}")]
    SourceAllocationPhaseEnd { expected: usize, actual: usize },
    #[error("ordinary-placement manifest has invalid {detail}")]
    SourceAllocationManifest { detail: &'static str },
    #[error("balanced-ternary opening {opening} has invalid {detail}")]
    BalancedTernaryGeometry {
        opening: usize,
        detail: &'static str,
    },
    #[error("canonical-u64 decomposition {decomposition} has invalid {detail}")]
    CanonicalU64Geometry {
        decomposition: usize,
        detail: &'static str,
    },
    #[error("canonical-u64 stage trace has invalid {detail}")]
    CanonicalU64StageSchedule { detail: &'static str },
    #[error("packed mod-5 chunk {chunk} has invalid {detail}")]
    PackedMod5Geometry { chunk: usize, detail: &'static str },
    #[error("chunk acceptance block {chunk} has invalid {detail}")]
    AcceptanceGeometry { chunk: usize, detail: &'static str },
    #[error("chunk acceptance block {chunk} witness disagrees with canonical source column {column}")]
    AcceptanceWitness { chunk: usize, column: usize },
    #[error("packed mod-5 chunk {chunk} witness disagrees with projected source column {column}")]
    PackedMod5Witness { chunk: usize, column: usize },
    #[error("public source column {column} is internal to a balanced-ternary opening")]
    PublicBalancedTernaryColumn { column: usize },
    #[error("gadget-derived source column {column} is not topological")]
    NonTopologicalDefinition { column: usize },
    #[error("gadget-derived source column {column} escapes its recorded source rows")]
    GadgetTemporaryEscapes { column: usize },
    #[error("product-sum batch {batch} has invalid {detail}")]
    ProductSumGeometry { batch: usize, detail: &'static str },
    #[error("product-sum batch {batch} identity {identity} does not follow from its exact source rows")]
    ProductSumIdentityMismatch { batch: usize, identity: usize },
    #[error("product-sum batch {batch} identities do not uniquely bind all retained columns")]
    ProductSumRetainedRank { batch: usize },
    #[error("product-sum batch {batch} references projected column {column} without a linear representation")]
    ProductSumUnavailableDependency { batch: usize, column: usize },
    #[error(transparent)]
    ProjectionIdentityTrace(
        #[from] crate::engine::r1cs_circuit::projection_identity_trace::ProjectionIdentityTraceError,
    ),
    #[error("projection-identity compaction rejected the fixed production manifest: {detail}")]
    ProjectionIdentityManifest { detail: &'static str },
    #[error("public source column {column} is an internal gadget temporary")]
    PublicGadgetTemporary { column: usize },
    #[error("source column {column} has no linear low-norm representation in row {row}")]
    MissingDecodedColumn { column: usize, row: usize },
    #[error("Boolean row-dedup plan disagrees with the concrete singleton slot map at source row {row}")]
    BooleanDedupPlanMismatch { row: usize },
    #[error("Boolean row-dedup ownership overlaps another removed source row at row {row}")]
    BooleanDedupOwnershipOverlap { row: usize },
    #[error("ring trace output {output} has inconsistent coefficient for convolution {evaluation}:{coefficient}")]
    RingCoefficientMismatch {
        output: usize,
        evaluation: usize,
        coefficient: usize,
    },
    #[error("ring trace output expression references non-product source column {column}")]
    RingOutputUnknownColumn { column: usize },
    #[error("source Boolean column {column} has non-Boolean witness value")]
    BooleanWitness { column: usize },
    #[error("source centered column {column} is not in {{-1, 0, 1}}")]
    CenteredWitness { column: usize },
    #[error("source field column {column} disagrees with its balanced-ternary digits")]
    BalancedTernaryWitness { column: usize },
    #[error("encoded assignment length {got} does not match plan length {expected}")]
    EncodedLength { expected: usize, got: usize },
    #[error("encoded assignment's constant column is not one")]
    EncodedConstantOne,
    #[error("encoded bit for source column {column} is not zero or one")]
    NonBinaryDigit { column: usize },
    #[error("encoded centered digit for source column {column} is not in {{-1, 0, 1}}")]
    NonCenteredDigit { column: usize },
    #[error("ordinary-private field {column} has {got} coordinates; expected {expected}")]
    OrdinaryPrivateWidth {
        column: usize,
        expected: usize,
        got: usize,
    },
    #[error("64-bit encoding for source column {column} is noncanonical ({value} >= p)")]
    NonCanonicalField { column: usize, value: u64 },
    #[error("canonicality auxiliary {offset} for source column {column} is inconsistent")]
    CanonicalAuxMismatch { column: usize, offset: usize },
    #[error("gadget-native low-norm relation is unsatisfied at row {row}")]
    UnsatisfiedEncoding { row: usize },
    #[error("common coordinate gate schedule has invalid {detail}")]
    CoordinateGateSchedule { detail: &'static str },
}

/// Compute exact production dimensions without allocating the bit witness or
/// sparse CCS matrices.
pub fn estimate_r1cs_gadget_native(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    public_bit_columns: &[usize],
) -> Result<GadgetNativeEstimate, GadgetNativeError> {
    let schedule = source_schedule::ValidatedSourceSchedule::checked(source, trace, public_bit_columns)?;
    estimate_r1cs_gadget_native_from_schedule(source, trace, public_bit_columns, &schedule)
}

fn estimate_r1cs_gadget_native_from_schedule(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    public_bit_columns: &[usize],
    schedule: &source_schedule::ValidatedSourceSchedule,
) -> Result<GadgetNativeEstimate, GadgetNativeError> {
    let is_public = &schedule.is_public;
    let explicit_bits = &schedule.explicit_bits;
    let marks = &schedule.marks;
    let linearly_derived = schedule
        .decisions()
        .iter()
        .map(|decision| matches!(decision, source_schedule::SourceColumnDecision::GenericLinear(_)))
        .collect::<Vec<_>>();
    let linearly_derived_source_cols = schedule
        .decisions()
        .iter()
        .filter(|decision| decision.role() == GadgetNativeSourceRole::LinearlyDerived)
        .count();
    let redundant_boolean_rows =
        boolean_dedup::ExactBooleanRows::from_plan(source, is_public, explicit_bits, &linearly_derived, marks);
    let removed_source_rows = marks
        .balanced_ternary
        .reduction_removed_rows(&schedule.removed_definition_rows, redundant_boolean_rows.rows())?;

    let mut one_bit_source_cols = 0usize;
    let mut canonical_binary_field_source_cols = 0usize;
    let mut ordinary_private_field_source_cols = 0usize;
    let mut balanced_ternary_field_source_cols = 0usize;
    let mut balanced_ternary_alias_source_cols = 0usize;
    let balanced_ternary_binary_source_cols = schedule.balanced_binary_columns();
    for decision in schedule.decisions().iter().skip(1) {
        match decision {
            source_schedule::SourceColumnDecision::PublicBit
            | source_schedule::SourceColumnDecision::PrivateBoolean(_) => one_bit_source_cols += 1,
            source_schedule::SourceColumnDecision::BalancedOpening { .. } => {
                balanced_ternary_field_source_cols += 1;
            }
            source_schedule::SourceColumnDecision::BalancedDigitAlias { .. } => {
                balanced_ternary_alias_source_cols += 1;
            }
            source_schedule::SourceColumnDecision::CanonicalField(
                source_schedule::CanonicalFieldKind::OrdinaryPrivate,
            ) => ordinary_private_field_source_cols += 1,
            source_schedule::SourceColumnDecision::CanonicalField(
                source_schedule::CanonicalFieldKind::DirectCanonicalU64,
            ) => canonical_binary_field_source_cols += 1,
            source_schedule::SourceColumnDecision::ConstantOne
            | source_schedule::SourceColumnDecision::GenericLinear(_)
            | source_schedule::SourceColumnDecision::Projected(_) => {}
        }
    }
    let coordinate_pairing =
        coordinate_gates::PlannedCoordinatePairing::checked(source, trace, &schedule, &linearly_derived)?;
    let boolean_pairing = coordinate_pairing.boolean_total();
    let centered_pairing = coordinate_pairing.centered_total();
    let ordinary_private_centered_pairing =
        coordinate_pairing.centered_family_total(GadgetNativeCenteredFamily::OrdinaryPrivateField);
    let sis_centered_pairing = coordinate_pairing.centered_family_total(GadgetNativeCenteredFamily::SisOpening);
    let synthetic_ring_fields = trace
        .ring_muls_toom3()
        .len()
        .saturating_mul(TOOM_EVALUATIONS * TOOM_COEFFICIENTS);
    let synthetic_product_sum_fields = marks.product_sums.synthetic_fields();
    let acceptance_chunks = marks.acceptance.len();
    let acceptance_encoded_cols = acceptance_chunks.saturating_mul(acceptance::ENCODED_COORDINATES_PER_CHUNK);
    let acceptance_tree_output_cols = acceptance_chunks.saturating_mul(acceptance::TREE_OUTPUTS_PER_CHUNK);
    let acceptance_tree_bit_pair_rows = acceptance_chunks.saturating_mul(acceptance::TREE_BIT_PAIR_ROWS_PER_CHUNK);
    let acceptance_product_aggregate_rows =
        acceptance_chunks.saturating_mul(acceptance::PRODUCT_AGGREGATE_ROWS_PER_CHUNK);
    let acceptance_root_binding_rows = acceptance_chunks.saturating_mul(acceptance::ROOT_BINDING_ROWS_PER_CHUNK);
    let packed_mod5_chunks = marks.mod5.len();
    let packed_mod5_encoded_cols = packed_mod5_chunks.saturating_mul(mod5::ENCODED_COORDINATES_PER_CHUNK);
    let packed_mod5_synthetic_cols = packed_mod5_chunks.saturating_mul(mod5::RESIDUE_COORDINATES_PER_CHUNK);
    let packed_mod5_low_bit_pair_rows = packed_mod5_chunks.saturating_mul(mod5::LOW_BIT_PAIR_ROWS_PER_CHUNK);
    let packed_mod5_high_bit_pair_rows = packed_mod5_chunks.saturating_mul(mod5::HIGH_BIT_PAIR_ROWS_PER_CHUNK);
    let packed_mod5_residue_pair_rows = packed_mod5_chunks.saturating_mul(mod5::RESIDUE_PAIR_ROWS_PER_CHUNK);
    let ordinary_private_encoded_cols =
        ordinary_private_field_source_cols.saturating_mul(ordinary_private_field::ORDINARY_PRIVATE_DIGITS);
    let sis_centered_encoded_cols = balanced_ternary_field_source_cols.saturating_mul(BALANCED_TERNARY_DIGITS);
    let centered_encoded_cols = ordinary_private_encoded_cols.saturating_add(sis_centered_encoded_cols);
    let public_input_len = canonical_superneo_public_input_len(public_bit_columns.len())?;
    let public_padding = public_input_len - (1 + public_bit_columns.len());
    let encoded_cols = 1usize
        .saturating_add(one_bit_source_cols)
        .saturating_add(public_padding)
        .saturating_add(centered_encoded_cols)
        .saturating_add(acceptance_tree_output_cols)
        .saturating_add(packed_mod5_synthetic_cols)
        .saturating_add(
            canonical_binary_field_source_cols
                .saturating_add(synthetic_ring_fields)
                .saturating_add(synthetic_product_sum_fields)
                .saturating_mul(CANONICAL_SLOT_WIDTH),
        );
    let fallback_source_rows =
        redundant_boolean_rows.retained_fallback_count(&marks.covered_rows, &removed_source_rows);
    let custom_rows = trace
        .sbox7()
        .len()
        .saturating_add(
            trace
                .k_muls()
                .iter()
                .enumerate()
                .filter(|(index, _)| !marks.product_sums.is_nested_k_mul(*index))
                .count()
                .saturating_mul(2),
        )
        .saturating_add(marks.product_sums.encoded_rows())
        .saturating_add(
            trace
                .ring_muls_toom3()
                .len()
                .saturating_mul(TOOM_EVALUATIONS * TOOM_COEFFICIENTS + 54),
        )
        .saturating_add(
            trace
                .first_accepted_selections()
                .iter()
                .map(selection::encoded_rows)
                .sum::<usize>(),
        )
        .saturating_add(acceptance_tree_bit_pair_rows)
        .saturating_add(acceptance_product_aggregate_rows)
        .saturating_add(acceptance_root_binding_rows)
        .saturating_add(packed_mod5_low_bit_pair_rows)
        .saturating_add(packed_mod5_high_bit_pair_rows)
        .saturating_add(packed_mod5_residue_pair_rows);
    let coordinate_rows = boolean_pairing
        .total_rows()
        .saturating_add(centered_pairing.total_rows());
    let encoded_rows = coordinate_rows
        .saturating_add(
            canonical_binary_field_source_cols
                .saturating_add(synthetic_ring_fields)
                .saturating_add(synthetic_product_sum_fields)
                .saturating_mul(GOLDILOCKS_CANONICALITY_PAIR_ROWS),
        )
        .saturating_add(fallback_source_rows)
        .saturating_add(custom_rows);
    Ok(GadgetNativeEstimate {
        source_rows: source.rows(),
        source_cols: source.cols(),
        public_input_len,
        encoded_cols,
        encoded_rows,
        max_degree: 8,
        one_bit_source_cols,
        canonical_binary_field_source_cols,
        ordinary_private_field_source_cols,
        balanced_ternary_field_source_cols,
        balanced_ternary_alias_source_cols,
        balanced_ternary_binary_source_cols,
        centered_encoded_cols,
        centered_pairing,
        ordinary_private_encoded_cols,
        ordinary_private_centered_pairing,
        sis_centered_encoded_cols,
        sis_centered_pairing,
        synthetic_ring_fields,
        synthetic_product_sum_fields,
        acceptance_chunks,
        acceptance_encoded_cols,
        acceptance_tree_output_cols,
        acceptance_tree_bit_pair_rows,
        acceptance_product_aggregate_rows,
        acceptance_root_binding_rows,
        packed_mod5_chunks,
        packed_mod5_encoded_cols,
        packed_mod5_low_bit_pair_rows,
        packed_mod5_high_bit_pair_rows,
        packed_mod5_residue_pair_rows,
        boolean_pairing,
        linearly_derived_source_cols,
        gadget_derived_source_cols: marks
            .gadget_columns
            .iter()
            .filter(|&&derived| derived)
            .count()
            .saturating_sub(marks.mod5.linear_column_count()),
        redundant_boolean_source_rows: redundant_boolean_rows.count(),
        fallback_source_rows,
    })
}

/// Materialize the exact gadget-native lowering. Intended for differential
/// tests and bounded relations; use [`estimate_r1cs_gadget_native`] before
/// attempting a production-sized allocation.
pub fn encode_r1cs_gadget_native(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    public_bit_columns: &[usize],
) -> Result<EncodedGadgetNativeR1cs, GadgetNativeError> {
    let mut source_schedule =
        source_schedule::ValidatedSourceSchedule::checked(source, trace, public_bit_columns)?.into_materialization();
    let decisions = std::mem::take(&mut source_schedule.decisions);
    let linearly_derived = decisions
        .iter()
        .map(|decision| matches!(decision, source_schedule::SourceColumnDecision::GenericLinear(_)))
        .collect::<Vec<_>>();
    let is_public = &source_schedule.is_public;
    let explicit_bits = &source_schedule.explicit_bits;
    let marks = &source_schedule.marks;
    let removed_definition_rows = &source_schedule.removed_definition_rows;

    let mut assignment = vec![F::ONE];
    let mut source_columns = vec![None; source.cols()];
    source_columns[0] = Some(SourceColumn::One);
    let mut canonical_slots = Vec::new();
    let mut balanced_slots_by_field = vec![None; source.cols()];

    for &column in public_bit_columns {
        let slot = push_boolean_slot(&mut assignment, source.witness()[column], column)?;
        source_columns[column] = Some(SourceColumn::Encoded(slot));
    }
    let public_input_len = canonical_superneo_public_input_len(public_bit_columns.len())?;
    assignment.resize(public_input_len, F::ZERO);
    let mut source_allocation = source_allocation::SourceAllocationCursor::new(public_input_len);

    for (column, decision) in decisions.into_iter().enumerate() {
        let allocation_step = source_allocation.step(column, decision.role())?;
        let allocation_start = assignment.len();
        match decision {
            source_schedule::SourceColumnDecision::ConstantOne | source_schedule::SourceColumnDecision::PublicBit => {}
            source_schedule::SourceColumnDecision::PrivateBoolean(_) => {
                let slot = push_boolean_slot(&mut assignment, source.witness()[column], column)?;
                source_columns[column] = Some(SourceColumn::Encoded(slot));
            }
            source_schedule::SourceColumnDecision::BalancedOpening { opening } => {
                let slot =
                    push_balanced_ternary_slot(&mut assignment, source, &trace.balanced_ternary_openings()[opening])?;
                balanced_slots_by_field[column] = Some(slot);
                source_columns[column] = Some(SourceColumn::Encoded(slot));
            }
            source_schedule::SourceColumnDecision::BalancedDigitAlias { field, digit } => {
                let parent = balanced_slots_by_field[field]
                    .expect("validated balanced-ternary field precedes its digit aliases");
                source_columns[column] = Some(SourceColumn::Encoded(ValueSlot::centered_alias(parent, digit)));
            }
            source_schedule::SourceColumnDecision::CanonicalField(kind) => {
                let slot = match kind {
                    source_schedule::CanonicalFieldKind::OrdinaryPrivate => {
                        push_ordinary_private_field_slot(&mut assignment, source.witness()[column])
                    }
                    source_schedule::CanonicalFieldKind::DirectCanonicalU64 => {
                        let slot = push_field_slot(&mut assignment, source.witness()[column]);
                        canonical_slots.push(slot);
                        slot
                    }
                };
                source_columns[column] = Some(SourceColumn::Encoded(slot));
            }
            source_schedule::SourceColumnDecision::GenericLinear(definition) => {
                source_columns[column] = Some(SourceColumn::Linear(definition));
            }
            source_schedule::SourceColumnDecision::Projected(projected) => match projected {
                source_schedule::ProjectedColumnDecision::Product(definition) => {
                    source_columns[column] = Some(SourceColumn::Product(definition));
                }
                source_schedule::ProjectedColumnDecision::GadgetLinear(definition) => {
                    source_columns[column] = Some(SourceColumn::GadgetLinear(definition));
                }
                source_schedule::ProjectedColumnDecision::AcceptanceInverse
                | source_schedule::ProjectedColumnDecision::Mod5Linear(_)
                | source_schedule::ProjectedColumnDecision::Mod5Product => {}
            },
        }
        allocation_step.check_observed(allocation_start, assignment.len())?;
    }
    source_allocation.check_phase_end(assignment.len())?;

    let mut acceptance_slots =
        acceptance::allocate_and_install(source, trace, &marks.acceptance, &mut assignment, &mut source_columns)?;
    let mod5_slots = mod5::allocate_and_install(source, trace, &marks.mod5, &mut assignment, &mut source_columns)?;

    let mut ring_slots = Vec::with_capacity(trace.ring_muls_toom3().len());
    for ring in trace.ring_muls_toom3() {
        let mut coefficients = Vec::with_capacity(TOOM_EVALUATIONS * TOOM_COEFFICIENTS);
        for convolution in &ring.convolutions {
            for coefficient in 0..TOOM_COEFFICIENTS {
                let value = (0..TOOM_SPLIT)
                    .filter_map(|left| {
                        let right = coefficient.checked_sub(left)?;
                        (right < TOOM_SPLIT)
                            .then_some(source.witness()[convolution.products[left * TOOM_SPLIT + right].col()])
                    })
                    .fold(F::ZERO, |sum, product| sum + product);
                let slot = push_field_slot(&mut assignment, value);
                canonical_slots.push(slot);
                coefficients.push(slot);
            }
        }
        ring_slots.push(RingSyntheticSlots { coefficients });
    }
    let product_sum_slots =
        product_sum::allocate_carries(source, &marks.product_sums, &mut assignment, &mut canonical_slots);

    let source_columns = source_columns
        .into_iter()
        .map(|column| column.expect("every source column has one encoding definition"))
        .collect::<Vec<_>>();
    source_schedule.validate_materialized(&source_columns)?;
    let planned_redundant_boolean_rows =
        boolean_dedup::ExactBooleanRows::from_plan(source, is_public, explicit_bits, &linearly_derived, marks);
    let redundant_boolean_rows = boolean_dedup::ExactBooleanRows::checked_concrete(
        source,
        &source_columns,
        &marks.covered_rows,
        &planned_redundant_boolean_rows,
        removed_definition_rows,
    )?;
    let decoded_terms = build_source_terms(&source_columns)?;
    let acceptance_translated_source_rows = acceptance::translated_boolean_source_rows(source, trace, &source_columns)?;
    let mut acceptance_translated_index = 0usize;
    let mut acceptance_translated_boolean_rows = Vec::with_capacity(acceptance_translated_source_rows.len());
    let mut balanced_ternary_reduction = shared_slots::ReductionPlan::checked(
        trace,
        &source_columns,
        &marks.balanced_ternary,
        removed_definition_rows,
        redundant_boolean_rows.rows(),
        assignment.len(),
    )?;
    let coordinate_gates = coordinate_gates::build_schedule(coordinate_gates::CoordinateGateInputs {
        source,
        trace,
        source_columns: &source_columns,
        ring_slots: &ring_slots,
        product_sum_slots: &product_sum_slots,
        product_sums: &marks.product_sums,
        balanced: &marks.balanced_ternary,
        reduction: &balanced_ternary_reduction,
        acceptance: &acceptance_slots,
        mod5: &mod5_slots,
        public_padding: (1 + public_bit_columns.len())..public_input_len,
        encoded_columns: assignment.len(),
    })?;
    balanced_ternary_reduction.install_coordinate_rows(&coordinate_gates)?;
    let mut gates = TraceGateBuilder::new();
    coordinate_gates.emit(&mut gates)?;
    for &slot in &canonical_slots {
        emit_goldilocks_canonicality(&mut gates, slot);
    }

    for row in 0..source.rows() {
        let is_acceptance_translated = acceptance_translated_source_rows
            .get(acceptance_translated_index)
            .is_some_and(|&candidate| candidate == row);
        if marks.covered_rows[row]
            || removed_definition_rows[row]
            || redundant_boolean_rows.rows()[row]
            || balanced_ternary_reduction.omits_source_row(row)
        {
            if is_acceptance_translated {
                return Err(GadgetNativeError::AcceptanceGeometry {
                    chunk: 0,
                    detail: "translated Boolean source row was removed",
                });
            }
            continue;
        }
        balanced_ternary_reduction.before_emit(row, gates.rows)?;
        if is_acceptance_translated {
            acceptance_translated_boolean_rows.push((row, gates.rows));
            acceptance_translated_index += 1;
        }
        gates.product_sum(
            one_selector(),
            vec![(
                translate_source_row(source.a_row(row), &decoded_terms, row)?,
                translate_source_row(source.b_row(row), &decoded_terms, row)?,
            )],
            translate_source_row(source.c_row(row), &decoded_terms, row)?,
        );
        balanced_ternary_reduction.after_emit(row, gates.rows)?;
    }
    if acceptance_translated_index != acceptance_translated_source_rows.len() {
        return Err(GadgetNativeError::AcceptanceGeometry {
            chunk: 0,
            detail: "translated Boolean source-row census",
        });
    }
    for event in trace.sbox7() {
        gates.sbox7(
            one_selector(),
            translate_event_lc(&event.input, &decoded_terms, event.source_rows.start)?,
            source_terms(event.output.col(), &decoded_terms, event.source_rows.start)?,
        );
    }
    for (index, event) in trace.k_muls().iter().enumerate() {
        if !marks.product_sums.is_nested_k_mul(index) {
            product_sum::emit_k_mul(event, &decoded_terms, &mut gates)?;
        }
    }
    product_sum::emit(&marks.product_sums, &product_sum_slots, &decoded_terms, &mut gates)?;
    for (event, slots) in trace.ring_muls_toom3().iter().zip(ring_slots.iter()) {
        emit_ring_mul(event, slots, &decoded_terms, &mut gates)?;
    }
    for event in trace.first_accepted_selections() {
        selection::emit(event, &decoded_terms, &mut gates)?;
    }
    acceptance::emit(trace, &mut acceptance_slots, &decoded_terms, &mut gates)?;
    mod5::emit(trace, &mod5_slots, &decoded_terms, &mut gates)?;

    let balanced_ternary_openings = balanced_ternary_reduction.finish()?;
    let structure = gates.finish(assignment.len());
    let plan = GadgetNativePlan {
        source_columns,
        source_roles: source_schedule.roles,
        ring_slots,
        product_sum_slots,
        acceptance_slots,
        mod5_slots,
        balanced_ternary_openings,
        public_columns: public_bit_columns.to_vec(),
        public_input_len,
        encoded_cols: assignment.len(),
        coordinate_gates,
        acceptance_translated_boolean_rows,
    };
    plan.validate_materialization_contract()?;
    Ok(EncodedGadgetNativeR1cs {
        structure,
        assignment,
        plan,
    })
}

fn validate_source_one(source: &R1csSnapshot) -> Result<(), GadgetNativeError> {
    if source.witness().first().copied() != Some(F::ONE) {
        return Err(GadgetNativeError::SourceConstantOne);
    }
    Ok(())
}

fn reject_public_gadget_columns(gadget_columns: &[bool], is_public: &[bool]) -> Result<(), GadgetNativeError> {
    if let Some(column) = (1..gadget_columns.len()).find(|&column| gadget_columns[column] && is_public[column]) {
        return Err(GadgetNativeError::PublicGadgetTemporary { column });
    }
    Ok(())
}

fn validate_and_mark_trace(source: &R1csSnapshot, trace: &R1csEncodingTrace) -> Result<TraceMarks, GadgetNativeError> {
    let mut covered_rows = vec![false; source.rows()];
    let mut gadget_columns = vec![false; source.cols()];
    let projection_batches = projection_identity::exact_product_sum_batches(source, trace)?;
    let product_sums = product_sum::ValidatedProductSums::validate_and_claim(
        source,
        trace,
        projection_batches,
        &mut covered_rows,
        &mut gadget_columns,
    )?;
    for event in trace.sbox7() {
        validate_sbox(source, event)?;
        claim_rows(source, "Poseidon2 sbox7", &event.source_rows, &mut covered_rows)?;
        for variable in event.intermediates {
            claim_gadget_column(variable.col(), &mut gadget_columns)?;
        }
    }
    for (index, event) in trace.k_muls().iter().enumerate() {
        product_sum::validate_k_mul(source, event)?;
        if product_sums.is_nested_k_mul(index) {
            continue;
        }
        claim_rows(source, "K multiplication", &event.source_rows, &mut covered_rows)?;
        for variable in event.intermediates {
            claim_gadget_column(variable.col(), &mut gadget_columns)?;
        }
    }
    for event in trace.ring_muls_toom3() {
        validate_ring_mul(source, event)?;
        claim_rows(
            source,
            "Toom-3 ring multiplication",
            &event.source_rows,
            &mut covered_rows,
        )?;
        for convolution in &event.convolutions {
            for variable in &convolution.products {
                claim_gadget_column(variable.col(), &mut gadget_columns)?;
            }
        }
    }
    for event in trace.first_accepted_selections() {
        selection::validate(source, event)?;
        for row in event.one_hot_rows.clone() {
            if covered_rows[row] {
                return Err(GadgetNativeError::OverlappingTraceRow { row });
            }
        }
        claim_rows(
            source,
            "first-accepted selection products",
            &event.product_rows,
            &mut covered_rows,
        )?;
        claim_rows(
            source,
            "first-accepted selection bindings",
            &event.bind_rows,
            &mut covered_rows,
        )?;
        selection::claim_products(event, &mut gadget_columns)?;
    }
    let acceptance =
        acceptance::ValidatedAcceptance::validate_and_claim(source, trace, &mut covered_rows, &mut gadget_columns)?;
    let mod5 = mod5::ValidatedMod5::validate_and_claim(source, trace, &mut covered_rows, &mut gadget_columns)?;
    let canonical_u64 = canonical_u64::ValidatedCanonicalU64::validate(source, trace, &covered_rows)?;
    let balanced_ternary = balanced_ternary::ValidatedBalancedTernary::validate(source, trace, &covered_rows)?;
    // Dependency safety is global: only now are all projected columns known.
    product_sums.validate_emitted_dependencies(&gadget_columns)?;
    for row in 0..source.rows() {
        if covered_rows[row] {
            continue;
        }
        for &(column, _) in source
            .a_row(row)
            .iter()
            .chain(source.b_row(row))
            .chain(source.c_row(row))
        {
            if gadget_columns[column] && !mod5.permits_escape(column) {
                return Err(GadgetNativeError::GadgetTemporaryEscapes { column });
            }
        }
    }
    Ok(TraceMarks {
        covered_rows,
        gadget_columns,
        product_sums,
        balanced_ternary,
        canonical_u64,
        acceptance,
        mod5,
    })
}

fn claim_rows(
    source: &R1csSnapshot,
    gadget: &'static str,
    range: &std::ops::Range<usize>,
    covered: &mut [bool],
) -> Result<(), GadgetNativeError> {
    if range.start >= range.end || range.end > source.rows() {
        return Err(GadgetNativeError::TraceRowRange {
            gadget,
            start: range.start,
            end: range.end,
            rows: source.rows(),
        });
    }
    for row in range.clone() {
        if std::mem::replace(&mut covered[row], true) {
            return Err(GadgetNativeError::OverlappingTraceRow { row });
        }
    }
    Ok(())
}

fn claim_gadget_column(column: usize, claimed: &mut [bool]) -> Result<(), GadgetNativeError> {
    if column == 0 || column >= claimed.len() || std::mem::replace(&mut claimed[column], true) {
        return Err(GadgetNativeError::DuplicateGadgetDefinition { column });
    }
    Ok(())
}

fn validate_sbox(source: &R1csSnapshot, event: &Sbox7TraceEntry) -> Result<(), GadgetNativeError> {
    if event
        .source_rows
        .end
        .saturating_sub(event.source_rows.start)
        != 4
    {
        return Err(GadgetNativeError::TraceArity {
            gadget: "Poseidon2 sbox7",
        });
    }
    let [x2, x4, x6] = event.intermediates;
    let rows = [
        (event.input.clone(), event.input.clone(), Lc::from_var(x2)),
        (Lc::from_var(x2), Lc::from_var(x2), Lc::from_var(x4)),
        (Lc::from_var(x2), Lc::from_var(x4), Lc::from_var(x6)),
        (event.input.clone(), Lc::from_var(x6), Lc::from_var(event.output)),
    ];
    validate_expected_rows(source, "Poseidon2 sbox7", event.source_rows.start, &rows)
}

fn validate_ring_mul(source: &R1csSnapshot, event: &RingMulToom3TraceEntry) -> Result<(), GadgetNativeError> {
    let product_rows = TOOM_EVALUATIONS * TOOM_SPLIT * TOOM_SPLIT;
    if event.rho.len() != 54
        || event.c.len() != 54
        || event.convolutions.len() != TOOM_EVALUATIONS
        || event.reduced_output_lcs.len() != 54
        || event.output.len() != 54
        || event
            .source_rows
            .end
            .saturating_sub(event.source_rows.start)
            != product_rows + 54
        || event.convolutions.iter().any(|convolution| {
            convolution.lhs.len() != TOOM_SPLIT
                || convolution.rhs.len() != TOOM_SPLIT
                || convolution.products.len() != TOOM_SPLIT * TOOM_SPLIT
        })
    {
        return Err(GadgetNativeError::TraceArity {
            gadget: "Toom-3 ring multiplication",
        });
    }
    let mut row = event.source_rows.start;
    for convolution in &event.convolutions {
        for left in 0..TOOM_SPLIT {
            for right in 0..TOOM_SPLIT {
                validate_row(
                    source,
                    "Toom-3 ring multiplication",
                    row,
                    &convolution.lhs[left],
                    &convolution.rhs[right],
                    &Lc::from_var(convolution.products[left * TOOM_SPLIT + right]),
                )?;
                row += 1;
            }
        }
    }
    for output in 0..54 {
        let diff = Lc::from_var(event.output[output]).add_scaled(&event.reduced_output_lcs[output], -F::ONE);
        validate_row(
            source,
            "Toom-3 ring multiplication",
            row,
            &diff,
            &Lc::from_var(Var::ONE),
            &Lc::zero(),
        )?;
        row += 1;
    }
    Ok(())
}

fn validate_expected_rows(
    source: &R1csSnapshot,
    gadget: &'static str,
    first: usize,
    rows: &[(Lc, Lc, Lc)],
) -> Result<(), GadgetNativeError> {
    for (offset, (a, b, c)) in rows.iter().enumerate() {
        validate_row(source, gadget, first + offset, a, b, c)?;
    }
    Ok(())
}

fn validate_row(
    source: &R1csSnapshot,
    gadget: &'static str,
    row: usize,
    a: &Lc,
    b: &Lc,
    c: &Lc,
) -> Result<(), GadgetNativeError> {
    if row >= source.rows()
        || source.a_row(row) != normalize_lc(a)
        || source.b_row(row) != normalize_lc(b)
        || source.c_row(row) != normalize_lc(c)
    {
        return Err(GadgetNativeError::TraceRowMismatch { gadget, row });
    }
    Ok(())
}

fn normalize_lc(lc: &Lc) -> Vec<(usize, F)> {
    let mut terms = BTreeMap::<usize, F>::new();
    for &(column, coefficient) in &lc.terms {
        *terms.entry(column).or_insert(F::ZERO) += coefficient;
    }
    *terms.entry(0).or_insert(F::ZERO) += lc.constant;
    terms
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect()
}

fn build_product_definitions(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    product_sums: &product_sum::ValidatedProductSums,
) -> Result<(Vec<Option<ProductDefinition>>, Vec<Option<LinearDefinition>>), GadgetNativeError> {
    let mut definitions = vec![None; source.cols()];
    let mut linear = vec![None; source.cols()];
    for event in trace.sbox7() {
        let [x2, x4, x6] = event.intermediates;
        set_product_definition(&mut definitions, x2, event.input.clone(), event.input.clone())?;
        set_product_definition(&mut definitions, x4, Lc::from_var(x2), Lc::from_var(x2))?;
        set_product_definition(&mut definitions, x6, Lc::from_var(x2), Lc::from_var(x4))?;
    }
    product_sum::define_unbatched_k_muls(&mut definitions, trace, product_sums)?;
    for event in trace.ring_muls_toom3() {
        for convolution in &event.convolutions {
            for left in 0..TOOM_SPLIT {
                for right in 0..TOOM_SPLIT {
                    set_product_definition(
                        &mut definitions,
                        convolution.products[left * TOOM_SPLIT + right],
                        convolution.lhs[left].clone(),
                        convolution.rhs[right].clone(),
                    )?;
                }
            }
        }
    }
    for event in trace.first_accepted_selections() {
        selection::define_products(&mut definitions, event)?;
    }
    product_sums.install_definitions(&mut definitions, &mut linear)?;
    Ok((definitions, linear))
}

fn set_product_definition(
    definitions: &mut [Option<ProductDefinition>],
    variable: Var,
    left: Lc,
    right: Lc,
) -> Result<(), GadgetNativeError> {
    let column = variable.col();
    if column == 0 || column >= definitions.len() || definitions[column].is_some() {
        return Err(GadgetNativeError::DuplicateGadgetDefinition { column });
    }
    if left
        .terms
        .iter()
        .chain(right.terms.iter())
        .any(|&(input, _)| input >= column)
    {
        return Err(GadgetNativeError::NonTopologicalDefinition { column });
    }
    definitions[column] = Some(ProductDefinition { left, right });
    Ok(())
}

fn select_linear_definition_columns(
    source: &R1csSnapshot,
    is_public: &[bool],
    marks: &TraceMarks,
) -> (Vec<bool>, Vec<bool>) {
    let mut defined = marks.gadget_columns.clone();
    for (column, value) in defined.iter_mut().enumerate() {
        *value |= marks.balanced_ternary.is_structural(column);
    }
    let mut selected = vec![false; source.cols()];
    let mut selected_rows = vec![false; source.rows()];
    for row in 0..source.rows() {
        if marks.covered_rows[row] {
            continue;
        }
        let Some((column, _)) = linear_definition_candidate(source, row, is_public, &defined) else {
            continue;
        };
        defined[column] = true;
        selected[column] = true;
        selected_rows[row] = true;
    }
    (selected, selected_rows)
}

fn build_linear_definitions(
    source: &R1csSnapshot,
    is_public: &[bool],
    marks: &TraceMarks,
) -> (Vec<Option<LinearDefinition>>, Vec<bool>) {
    let mut definitions = vec![None; source.cols()];
    let mut removed_rows = vec![false; source.rows()];
    let mut defined = marks.gadget_columns.clone();
    for (column, value) in defined.iter_mut().enumerate() {
        *value |= marks.balanced_ternary.is_structural(column);
    }
    for row in 0..source.rows() {
        if marks.covered_rows[row] {
            continue;
        }
        let Some((column, coefficient)) = linear_definition_candidate(source, row, is_public, &defined) else {
            continue;
        };
        let (positive, negative) = linear_difference(source, row).expect("candidate row is linear");
        let inverse = coefficient.inverse();
        let mut terms = Vec::new();
        visit_difference_terms(positive, negative, |input, input_coefficient| {
            if input != column {
                let derived = -input_coefficient * inverse;
                if derived != F::ZERO {
                    terms.push((input, derived));
                }
            }
        });
        definitions[column] = Some(LinearDefinition {
            terms,
            source_row: Some(row),
        });
        removed_rows[row] = true;
        defined[column] = true;
    }
    (definitions, removed_rows)
}

fn linear_definition_candidate(
    source: &R1csSnapshot,
    row: usize,
    is_public: &[bool],
    defined: &[bool],
) -> Option<(usize, F)> {
    let (positive, negative) = linear_difference(source, row)?;
    let mut highest = None;
    visit_difference_terms(positive, negative, |column, coefficient| {
        if column != 0 {
            highest = Some((column, coefficient));
        }
    });
    let (column, coefficient) = highest?;
    if is_public[column] || defined[column] {
        return None;
    }
    Some((column, coefficient))
}

fn linear_difference(source: &R1csSnapshot, row: usize) -> Option<(&[(usize, F)], &[(usize, F)])> {
    if source.b_row(row) == [(0, F::ONE)] {
        Some((source.a_row(row), source.c_row(row)))
    } else if source.a_row(row) == [(0, F::ONE)] {
        Some((source.b_row(row), source.c_row(row)))
    } else {
        None
    }
}

fn visit_difference_terms(positive: &[(usize, F)], negative: &[(usize, F)], mut visit: impl FnMut(usize, F)) {
    let mut left = 0usize;
    let mut right = 0usize;
    while left < positive.len() || right < negative.len() {
        let column = match (positive.get(left), negative.get(right)) {
            (Some(&(a, _)), Some(&(b, _))) => a.min(b),
            (Some(&(a, _)), None) => a,
            (None, Some(&(b, _))) => b,
            (None, None) => unreachable!(),
        };
        let mut coefficient = F::ZERO;
        if positive
            .get(left)
            .is_some_and(|&(candidate, _)| candidate == column)
        {
            coefficient += positive[left].1;
            left += 1;
        }
        if negative
            .get(right)
            .is_some_and(|&(candidate, _)| candidate == column)
        {
            coefficient -= negative[right].1;
            right += 1;
        }
        if coefficient != F::ZERO {
            visit(column, coefficient);
        }
    }
}

fn build_source_terms(columns: &[SourceColumn]) -> Result<Vec<Option<Vec<(usize, F)>>>, GadgetNativeError> {
    let mut decoded = vec![None; columns.len()];
    decoded[0] = Some(vec![(0, F::ONE)]);
    for column in 1..columns.len() {
        decoded[column] = match &columns[column] {
            SourceColumn::One => unreachable!("only source column zero is ONE"),
            SourceColumn::Encoded(slot) => Some(slot_terms(*slot)),
            SourceColumn::EncodedLinear(terms) => Some(terms.clone()),
            SourceColumn::Product(_) => None,
            SourceColumn::CanonicalNonzeroInverse(_) => None,
            SourceColumn::GadgetLinear(_) => None,
            SourceColumn::Linear(definition) => {
                let mut combined = BTreeMap::<usize, F>::new();
                for &(input, scale) in &definition.terms {
                    let Some(input_terms) = &decoded[input] else {
                        return Err(GadgetNativeError::MissingDecodedColumn { column: input, row: 0 });
                    };
                    for &(encoded_column, coefficient) in input_terms {
                        *combined.entry(encoded_column).or_insert(F::ZERO) += scale * coefficient;
                    }
                }
                Some(
                    combined
                        .into_iter()
                        .filter(|(_, coefficient)| *coefficient != F::ZERO)
                        .collect(),
                )
            }
        };
    }
    Ok(decoded)
}

fn translate_source_row(
    row_terms: &[(usize, F)],
    decoded: &[Option<Vec<(usize, F)>>],
    row: usize,
) -> Result<Vec<(usize, F)>, GadgetNativeError> {
    let mut out = Vec::new();
    for &(source_column, scale) in row_terms {
        let terms = source_terms(source_column, decoded, row)?;
        out.extend(
            terms
                .into_iter()
                .map(|(column, coefficient)| (column, scale * coefficient)),
        );
    }
    Ok(out)
}

fn translate_event_lc(
    lc: &Lc,
    decoded: &[Option<Vec<(usize, F)>>],
    row: usize,
) -> Result<Vec<(usize, F)>, GadgetNativeError> {
    let mut source_row = lc.terms.clone();
    if lc.constant != F::ZERO {
        source_row.push((0, lc.constant));
    }
    translate_source_row(&source_row, decoded, row)
}

fn source_terms(
    column: usize,
    decoded: &[Option<Vec<(usize, F)>>],
    row: usize,
) -> Result<Vec<(usize, F)>, GadgetNativeError> {
    decoded[column]
        .clone()
        .ok_or(GadgetNativeError::MissingDecodedColumn { column, row })
}

fn emit_ring_mul(
    event: &RingMulToom3TraceEntry,
    slots: &RingSyntheticSlots,
    decoded: &[Option<Vec<(usize, F)>>],
    gates: &mut TraceGateBuilder,
) -> Result<(), GadgetNativeError> {
    let row = event.source_rows.start;
    for (evaluation, convolution) in event.convolutions.iter().enumerate() {
        for coefficient in 0..TOOM_COEFFICIENTS {
            let mut products = Vec::new();
            for left in 0..TOOM_SPLIT {
                let Some(right) = coefficient.checked_sub(left) else {
                    continue;
                };
                if right >= TOOM_SPLIT {
                    continue;
                }
                products.push((
                    translate_event_lc(&convolution.lhs[left], decoded, row)?,
                    translate_event_lc(&convolution.rhs[right], decoded, row)?,
                ));
            }
            gates.product_sum(
                one_selector(),
                products,
                slot_terms(slots.coefficient(evaluation, coefficient)),
            );
        }
    }

    let product_groups = ring_product_groups(event);
    for output in 0..54 {
        let rhs = collapse_ring_output(output, &event.reduced_output_lcs[output], &product_groups, slots)?;
        gates.linear(
            one_selector(),
            source_terms(event.output[output].col(), decoded, row)?,
            rhs,
        );
    }
    Ok(())
}

fn ring_product_groups(event: &RingMulToom3TraceEntry) -> BTreeMap<usize, (usize, usize)> {
    let mut groups = BTreeMap::new();
    for (evaluation, convolution) in event.convolutions.iter().enumerate() {
        for left in 0..TOOM_SPLIT {
            for right in 0..TOOM_SPLIT {
                groups.insert(
                    convolution.products[left * TOOM_SPLIT + right].col(),
                    (evaluation, left + right),
                );
            }
        }
    }
    groups
}

fn collapse_ring_output(
    output: usize,
    lc: &Lc,
    product_groups: &BTreeMap<usize, (usize, usize)>,
    slots: &RingSyntheticSlots,
) -> Result<Vec<(usize, F)>, GadgetNativeError> {
    let normalized = normalize_lc(lc);
    let by_column = normalized.into_iter().collect::<BTreeMap<_, _>>();
    let mut group_coefficients = BTreeMap::<(usize, usize), F>::new();
    for (&column, &coefficient) in &by_column {
        if column == 0 {
            continue;
        }
        let Some(&group) = product_groups.get(&column) else {
            return Err(GadgetNativeError::RingOutputUnknownColumn { column });
        };
        if let Some(previous) = group_coefficients.insert(group, coefficient) {
            if previous != coefficient {
                return Err(GadgetNativeError::RingCoefficientMismatch {
                    output,
                    evaluation: group.0,
                    coefficient: group.1,
                });
            }
        }
    }
    for (&column, &(evaluation, coefficient)) in product_groups {
        let actual = by_column.get(&column).copied().unwrap_or(F::ZERO);
        let expected = group_coefficients
            .get(&(evaluation, coefficient))
            .copied()
            .unwrap_or(F::ZERO);
        if actual != expected {
            return Err(GadgetNativeError::RingCoefficientMismatch {
                output,
                evaluation,
                coefficient,
            });
        }
    }
    let mut out = Vec::new();
    if let Some(&constant) = by_column.get(&0) {
        out.push((0, constant));
    }
    for ((evaluation, coefficient), scale) in group_coefficients {
        out.extend(scaled_terms(
            slot_terms(slots.coefficient(evaluation, coefficient)),
            scale,
        ));
    }
    Ok(out)
}

fn scaled_terms(terms: Vec<(usize, F)>, scale: F) -> Vec<(usize, F)> {
    terms
        .into_iter()
        .map(|(column, coefficient)| (column, coefficient * scale))
        .collect()
}

fn eval_lc_from_source(lc: &Lc, source: &[F]) -> F {
    lc.terms
        .iter()
        .fold(lc.constant, |value, &(column, coefficient)| {
            value + coefficient * source[column]
        })
}

fn evaluate_ccs(structure: &CcsStructure<F>, assignment: &[F]) -> Vec<F> {
    let mut matrix_z = vec![vec![F::ZERO; structure.n]; structure.matrices.len()];
    for (index, matrix) in structure.matrices.iter().enumerate() {
        matrix.add_mul_into(assignment, &mut matrix_z[index], structure.n);
    }
    (0..structure.n)
        .map(|row| {
            let point = matrix_z
                .iter()
                .map(|values| values[row])
                .collect::<Vec<_>>();
            structure.f.eval(&point)
        })
        .collect()
}
