//! Stage attribution for the exact gadget-native `enc(F')` estimate.
//!
//! Owns: exact attribution and aggregation of source and lowered costs.
//!
//! Does not own: stage placement, constraint emission, or semantic authority.
//!
//! Emits constraints: no.
//!
//! Authority boundary: only validated source ranges and gadget traces are
//! counted; every reported total must reconcile with the production estimate.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | Source stage ranges | Attribute columns and fallback rows | no | source emitters | no theorem claim |
//! | Gadget events | Attribute projected columns and replacement rows | no | traced lowering | matching refinement leaf |
//! | Common Boolean bitness | Prove every retained Boolean slot is in `{0, 1}` | no | parent common gate | no theorem claim |
//! | Ordinary-private centered-unit | Pair 41 coordinates per field within each physical stage; retain one stage-family odd tail | no | `ordinary_private_field` / `coordinate_gates` | `CenteredTernaryField.gateWord_iff_alphabetWord` |
//! | SIS centered-unit | Pair SIS opening digits independently from ordinary-private coordinates | no | `balanced_ternary` / `coordinate_gates` | `ResidualPairFamilies.centeredUnitScheduleHolds_iff` |
//! | Direct canonical-u64 source fields | 64 raw-bit roots, 31 prefix-bit roots, and 32 Goldilocks-bound relations in 16 per-slot residual-pair rows | no | canonical source slots | `ResidualPairFamilies.oneProductPairHolds_iff` |
//! | Canonical synthetic ring fields | The same per-slot split for Toom-3 materialization fields | no | ring-mul lowering | same model theorem |
//! | Canonical synthetic product-sum fields | The same per-slot split for compact product-sum fields | no | product-sum lowering | same model theorem |
//! | Aggregate acceptance leaves | Partition each four-row/two-column source block into tree bit pairs, one product aggregate, and one root binding | no | `acceptance` trace validator | `AggregateAcceptanceRows`; exact artifact bridge open |
//! | Packed Mod-5 leaves | Partition each validated 20-row/19-column source block into low-bit, high-bit, and residue obligations | no | `mod5` trace validator | `PackedChunkRows`; exact artifact bridge separate |
//! | Custom/fallback families | Attribute every remaining emitted row to its exact gate owner | no | traced lowering | matching refinement leaf |

use std::collections::BTreeMap;

use thiserror::Error;

use crate::engine::r1cs_circuit::alphabet_sampling::pi_rlc_challenge_stage;
use crate::engine::r1cs_circuit::builder::BALANCED_TERNARY_DIGITS;
use crate::engine::r1cs_circuit::{R1csEncodingTrace, R1csSnapshot};

use super::slots::{GOLDILOCKS_CANONICALITY_PAIR_ROWS, GOLDILOCKS_CANONICALITY_RELATIONS};
use super::{
    GadgetNativeBooleanPairingBreakdown, GadgetNativeError, GadgetNativePairTailCount, CANONICAL_SLOT_WIDTH,
    TOOM_COEFFICIENTS, TOOM_EVALUATIONS,
};

const K_MUL_ROWS: usize = 2;
const RING_MUL_ROWS: usize = TOOM_EVALUATIONS * TOOM_COEFFICIENTS + 54;

/// Exact size of the gadget-native lowering without materializing its sparse
/// matrices or bit assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GadgetNativeEstimate {
    pub source_rows: usize,
    pub source_cols: usize,
    pub public_input_len: usize,
    pub encoded_cols: usize,
    pub encoded_rows: usize,
    pub max_degree: u32,
    pub one_bit_source_cols: usize,
    /// Direct canonical-u64 source fields that retain 95-coordinate binary slots.
    pub canonical_binary_field_source_cols: usize,
    /// Ordinary private fields lowered to exact 41-coordinate centered words.
    pub ordinary_private_field_source_cols: usize,
    pub balanced_ternary_field_source_cols: usize,
    pub balanced_ternary_alias_source_cols: usize,
    pub balanced_ternary_binary_source_cols: usize,
    pub centered_encoded_cols: usize,
    /// Exact physical-stage pairing of centered-unit residuals.
    pub centered_pairing: GadgetNativePairTailCount,
    pub ordinary_private_encoded_cols: usize,
    pub ordinary_private_centered_pairing: GadgetNativePairTailCount,
    pub sis_centered_encoded_cols: usize,
    pub sis_centered_pairing: GadgetNativePairTailCount,
    pub synthetic_ring_fields: usize,
    pub synthetic_product_sum_fields: usize,
    pub acceptance_chunks: usize,
    pub acceptance_encoded_cols: usize,
    pub acceptance_tree_output_cols: usize,
    pub acceptance_tree_bit_pair_rows: usize,
    pub acceptance_product_aggregate_rows: usize,
    pub acceptance_root_binding_rows: usize,
    pub packed_mod5_chunks: usize,
    pub packed_mod5_encoded_cols: usize,
    pub packed_mod5_low_bit_pair_rows: usize,
    pub packed_mod5_high_bit_pair_rows: usize,
    pub packed_mod5_residue_pair_rows: usize,
    /// Exact stage-reset common-Boolean pair/tail schedule.
    pub boolean_pairing: GadgetNativeBooleanPairingBreakdown,
    pub linearly_derived_source_cols: usize,
    pub gadget_derived_source_cols: usize,
    /// Exact source `v * (v - 1) = 0` rows already enforced by the common
    /// bitness gate on the same singleton encoded slot.
    pub redundant_boolean_source_rows: usize,
    pub fallback_source_rows: usize,
}

/// Exact common-gate rows for one canonical Goldilocks field origin.
///
/// These are deliberately separate because a future lowering may prove that
/// one origin can omit an alphabet family while another still requires it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GadgetNativeCanonicalBinaryFieldRowBreakdown {
    /// Stage-local nonresidue pairs plus an ordinary odd tail, if present.
    pub raw_bits: GadgetNativePairTailCount,
    /// Stage-local nonresidue pairs plus an ordinary odd tail, if present.
    pub prefix_aux: GadgetNativePairTailCount,
    /// 31 prefix-product equations plus the final Goldilocks bound equation.
    pub canonicality_relations: usize,
    /// Physical nonresidue-seven pair rows, reset within every canonical slot.
    pub canonicality_pair_rows: usize,
}

impl GadgetNativeCanonicalBinaryFieldRowBreakdown {
    fn from_pairing(raw_bits: GadgetNativePairTailCount, prefix_aux: GadgetNativePairTailCount, fields: usize) -> Self {
        Self {
            raw_bits,
            prefix_aux,
            canonicality_relations: fields * GOLDILOCKS_CANONICALITY_RELATIONS,
            canonicality_pair_rows: fields * GOLDILOCKS_CANONICALITY_PAIR_ROWS,
        }
    }

    pub fn total(self) -> usize {
        self.raw_bits.total_rows() + self.prefix_aux.total_rows() + self.canonicality_pair_rows
    }
}

/// Why the low-norm compiler emits rows for one stage.
///
/// These components are disjoint and their sum is exactly the stage's
/// `encoded_rows`. Keeping the formula explicit prevents a large stage total
/// from hiding canonicalization or generic-lowering overhead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GadgetNativeEncodedRowBreakdown {
    /// Stage-local pair/tail rows for retained ordinary Boolean slots.
    pub common_boolean: GadgetNativePairTailCount,
    /// Organizational parent: exact sum of the two centered children below.
    pub common_centered_unit: GadgetNativePairTailCount,
    /// Stage-local ordinary-private centered residual pair/tail rows.
    pub ordinary_private_centered_unit: GadgetNativePairTailCount,
    /// Stage-local SIS-opening centered residual pair/tail rows.
    pub sis_centered_unit: GadgetNativePairTailCount,
    pub canonical_binary_source_fields: GadgetNativeCanonicalBinaryFieldRowBreakdown,
    pub synthetic_ring_fields: GadgetNativeCanonicalBinaryFieldRowBreakdown,
    pub synthetic_product_sum_fields: GadgetNativeCanonicalBinaryFieldRowBreakdown,
    pub fallback: usize,
    pub sbox: usize,
    pub k_mul: usize,
    pub product_sum: usize,
    pub ring_mul: usize,
    pub acceptance_tree_bit_pair: usize,
    pub acceptance_product_aggregate: usize,
    pub acceptance_root_binding: usize,
    pub packed_mod5_low_bit_pair: usize,
    pub packed_mod5_high_bit_pair: usize,
    pub packed_mod5_residue_pair: usize,
    pub selection_accept_aggregate: usize,
    pub selection_prefix_aggregate: usize,
    pub selection_symbol_aggregate: usize,
}

impl GadgetNativeEncodedRowBreakdown {
    pub fn total(self) -> usize {
        self.common_boolean.total_rows()
            + self.ordinary_private_centered_unit.total_rows()
            + self.sis_centered_unit.total_rows()
            + self.canonical_binary_source_fields.total()
            + self.synthetic_ring_fields.total()
            + self.synthetic_product_sum_fields.total()
            + self.fallback
            + self.sbox
            + self.k_mul
            + self.product_sum
            + self.ring_mul
            + self.acceptance_tree_bit_pair
            + self.acceptance_product_aggregate
            + self.acceptance_root_binding
            + self.packed_mod5_low_bit_pair
            + self.packed_mod5_high_bit_pair
            + self.packed_mod5_residue_pair
            + self.selection_accept_aggregate
            + self.selection_prefix_aggregate
            + self.selection_symbol_aggregate
    }
}

/// Exact contribution of one source-emission range or validated constraint
/// family. Organizational parents are represented by zero-cost entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GadgetNativeStageEstimate {
    pub label: &'static str,
    /// Number of disjoint emission ranges aggregated into this estimate.
    pub occurrences: usize,
    pub source_rows: usize,
    pub source_cols: usize,
    pub one_bit_source_cols: usize,
    pub canonical_binary_field_source_cols: usize,
    pub ordinary_private_field_source_cols: usize,
    pub balanced_ternary_field_source_cols: usize,
    pub balanced_ternary_alias_source_cols: usize,
    pub balanced_ternary_binary_source_cols: usize,
    pub linearly_derived_source_cols: usize,
    pub gadget_derived_source_cols: usize,
    pub synthetic_ring_fields: usize,
    pub synthetic_product_sum_fields: usize,
    pub acceptance_chunks: usize,
    pub acceptance_encoded_cols: usize,
    pub acceptance_tree_output_cols: usize,
    pub acceptance_tree_bit_pair_rows: usize,
    pub acceptance_product_aggregate_rows: usize,
    pub acceptance_root_binding_rows: usize,
    /// Counted on the low-bit-pair leaf so each packed chunk has one owner.
    pub packed_mod5_chunks: usize,
    pub packed_mod5_encoded_cols: usize,
    pub packed_mod5_low_bit_pair_rows: usize,
    pub packed_mod5_high_bit_pair_rows: usize,
    pub packed_mod5_residue_pair_rows: usize,
    /// Exact stage-reset common-Boolean pair/tail schedule.
    pub boolean_pairing: GadgetNativeBooleanPairingBreakdown,
    /// Low-norm columns contributed by this stage; excludes the global ONE.
    pub encoded_cols: usize,
    pub centered_encoded_cols: usize,
    /// Physical-stage centered residual pair/tail schedule.
    pub centered_pairing: GadgetNativePairTailCount,
    pub ordinary_private_encoded_cols: usize,
    pub ordinary_private_centered_pairing: GadgetNativePairTailCount,
    pub sis_centered_encoded_cols: usize,
    pub sis_centered_pairing: GadgetNativePairTailCount,
    pub encoded_rows: usize,
    /// Source Boolean rows omitted because the exact same singleton slot is
    /// already constrained by the common value-encoding bitness gate.
    pub redundant_boolean_source_rows: usize,
    pub fallback_source_rows: usize,
    pub poseidon_permutations: usize,
    pub poseidon_hash_permutations: usize,
    pub poseidon_hashes: usize,
    pub sboxes: usize,
    pub k_muls: usize,
    pub product_sum_batches: usize,
    pub product_sum_identities: usize,
    pub product_sum_rows: usize,
    pub ring_muls: usize,
    pub selection_accept_aggregate_rows: usize,
    pub selection_prefix_aggregate_rows: usize,
    pub selection_symbol_aggregate_rows: usize,
    /// `input fields -> (calls, permutations)` for one-shot Poseidon hashes.
    pub hash_histogram: BTreeMap<usize, (usize, usize)>,
}

impl GadgetNativeStageEstimate {
    fn empty(label: &'static str) -> Self {
        Self {
            label,
            occurrences: 0,
            source_rows: 0,
            source_cols: 0,
            one_bit_source_cols: 0,
            canonical_binary_field_source_cols: 0,
            ordinary_private_field_source_cols: 0,
            balanced_ternary_field_source_cols: 0,
            balanced_ternary_alias_source_cols: 0,
            balanced_ternary_binary_source_cols: 0,
            linearly_derived_source_cols: 0,
            gadget_derived_source_cols: 0,
            synthetic_ring_fields: 0,
            synthetic_product_sum_fields: 0,
            acceptance_chunks: 0,
            acceptance_encoded_cols: 0,
            acceptance_tree_output_cols: 0,
            acceptance_tree_bit_pair_rows: 0,
            acceptance_product_aggregate_rows: 0,
            acceptance_root_binding_rows: 0,
            packed_mod5_chunks: 0,
            packed_mod5_encoded_cols: 0,
            packed_mod5_low_bit_pair_rows: 0,
            packed_mod5_high_bit_pair_rows: 0,
            packed_mod5_residue_pair_rows: 0,
            boolean_pairing: GadgetNativeBooleanPairingBreakdown::default(),
            encoded_cols: 0,
            centered_encoded_cols: 0,
            centered_pairing: GadgetNativePairTailCount::default(),
            ordinary_private_encoded_cols: 0,
            ordinary_private_centered_pairing: GadgetNativePairTailCount::default(),
            sis_centered_encoded_cols: 0,
            sis_centered_pairing: GadgetNativePairTailCount::default(),
            encoded_rows: 0,
            redundant_boolean_source_rows: 0,
            fallback_source_rows: 0,
            poseidon_permutations: 0,
            poseidon_hash_permutations: 0,
            poseidon_hashes: 0,
            sboxes: 0,
            k_muls: 0,
            product_sum_batches: 0,
            product_sum_identities: 0,
            product_sum_rows: 0,
            ring_muls: 0,
            selection_accept_aggregate_rows: 0,
            selection_prefix_aggregate_rows: 0,
            selection_symbol_aggregate_rows: 0,
            hash_histogram: BTreeMap::new(),
        }
    }

    fn add(&mut self, other: &Self) {
        self.occurrences += other.occurrences;
        self.source_rows += other.source_rows;
        self.source_cols += other.source_cols;
        self.one_bit_source_cols += other.one_bit_source_cols;
        self.canonical_binary_field_source_cols += other.canonical_binary_field_source_cols;
        self.ordinary_private_field_source_cols += other.ordinary_private_field_source_cols;
        self.balanced_ternary_field_source_cols += other.balanced_ternary_field_source_cols;
        self.balanced_ternary_alias_source_cols += other.balanced_ternary_alias_source_cols;
        self.balanced_ternary_binary_source_cols += other.balanced_ternary_binary_source_cols;
        self.linearly_derived_source_cols += other.linearly_derived_source_cols;
        self.gadget_derived_source_cols += other.gadget_derived_source_cols;
        self.synthetic_ring_fields += other.synthetic_ring_fields;
        self.synthetic_product_sum_fields += other.synthetic_product_sum_fields;
        self.acceptance_chunks += other.acceptance_chunks;
        self.acceptance_encoded_cols += other.acceptance_encoded_cols;
        self.acceptance_tree_output_cols += other.acceptance_tree_output_cols;
        self.acceptance_tree_bit_pair_rows += other.acceptance_tree_bit_pair_rows;
        self.acceptance_product_aggregate_rows += other.acceptance_product_aggregate_rows;
        self.acceptance_root_binding_rows += other.acceptance_root_binding_rows;
        self.packed_mod5_chunks += other.packed_mod5_chunks;
        self.packed_mod5_encoded_cols += other.packed_mod5_encoded_cols;
        self.packed_mod5_low_bit_pair_rows += other.packed_mod5_low_bit_pair_rows;
        self.packed_mod5_high_bit_pair_rows += other.packed_mod5_high_bit_pair_rows;
        self.packed_mod5_residue_pair_rows += other.packed_mod5_residue_pair_rows;
        self.boolean_pairing.add(other.boolean_pairing);
        self.encoded_cols += other.encoded_cols;
        self.centered_encoded_cols += other.centered_encoded_cols;
        self.centered_pairing.add(other.centered_pairing);
        self.ordinary_private_encoded_cols += other.ordinary_private_encoded_cols;
        self.ordinary_private_centered_pairing
            .add(other.ordinary_private_centered_pairing);
        self.sis_centered_encoded_cols += other.sis_centered_encoded_cols;
        self.sis_centered_pairing.add(other.sis_centered_pairing);
        self.encoded_rows += other.encoded_rows;
        self.redundant_boolean_source_rows += other.redundant_boolean_source_rows;
        self.fallback_source_rows += other.fallback_source_rows;
        self.poseidon_permutations += other.poseidon_permutations;
        self.poseidon_hash_permutations += other.poseidon_hash_permutations;
        self.poseidon_hashes += other.poseidon_hashes;
        self.sboxes += other.sboxes;
        self.k_muls += other.k_muls;
        self.product_sum_batches += other.product_sum_batches;
        self.product_sum_identities += other.product_sum_identities;
        self.product_sum_rows += other.product_sum_rows;
        self.ring_muls += other.ring_muls;
        self.selection_accept_aggregate_rows += other.selection_accept_aggregate_rows;
        self.selection_prefix_aggregate_rows += other.selection_prefix_aggregate_rows;
        self.selection_symbol_aggregate_rows += other.selection_symbol_aggregate_rows;
        for (&input_len, &(calls, permutations)) in &other.hash_histogram {
            let entry = self.hash_histogram.entry(input_len).or_default();
            entry.0 += calls;
            entry.1 += permutations;
        }
    }

    pub fn encoded_row_breakdown(&self) -> GadgetNativeEncodedRowBreakdown {
        let packed_boolean_cols = self
            .packed_mod5_low_bit_pair_rows
            .checked_mul(2)
            .and_then(|low| low.checked_add(self.packed_mod5_high_bit_pair_rows))
            .expect("packed mod-5 Boolean-column count");
        let common_boolean_coordinates = self
            .one_bit_source_cols
            .checked_sub(self.balanced_ternary_binary_source_cols)
            .and_then(|cols| cols.checked_sub(packed_boolean_cols))
            .and_then(|cols| {
                cols.checked_sub(
                    self.acceptance_encoded_cols
                        .checked_sub(self.acceptance_tree_output_cols)?,
                )
            })
            .expect("balanced-ternary Boolean aliases must be a subset of one-bit source columns");
        let packed_synthetic_cols = self
            .packed_mod5_encoded_cols
            .checked_sub(packed_boolean_cols)
            .expect("packed mod-5 Boolean columns must be part of its encoded columns");
        assert_eq!(
            self.encoded_cols,
            self.one_bit_source_cols
                + self.centered_encoded_cols
                + self.acceptance_tree_output_cols
                + packed_synthetic_cols
                + (self.canonical_binary_field_source_cols
                    + self.synthetic_ring_fields
                    + self.synthetic_product_sum_fields)
                    * CANONICAL_SLOT_WIDTH,
            "stage encoded-column ownership must match its disjoint slot families"
        );
        assert_eq!(
            self.boolean_pairing.common.coordinates, common_boolean_coordinates,
            "common Boolean pairing census must cover retained ordinary coordinates"
        );
        assert_eq!(
            self.centered_pairing.coordinates, self.centered_encoded_cols,
            "centered residual pairing census must cover retained centered coordinates"
        );
        assert_eq!(
            self.ordinary_private_encoded_cols,
            self.ordinary_private_field_source_cols * super::ordinary_private_field::ORDINARY_PRIVATE_DIGITS,
            "ordinary-private coordinates must be exactly 41 per source field"
        );
        assert_eq!(
            self.ordinary_private_centered_pairing.coordinates, self.ordinary_private_encoded_cols,
            "ordinary-private centered rows must cover their exact coordinates"
        );
        assert_eq!(
            self.sis_centered_pairing.coordinates, self.sis_centered_encoded_cols,
            "SIS centered rows must cover their exact coordinates"
        );
        let mut centered_children = self.ordinary_private_centered_pairing;
        centered_children.add(self.sis_centered_pairing);
        assert_eq!(
            self.centered_pairing, centered_children,
            "centered organizational parent must equal its two disjoint children"
        );
        GadgetNativeEncodedRowBreakdown {
            common_boolean: self.boolean_pairing.common,
            common_centered_unit: self.centered_pairing,
            ordinary_private_centered_unit: self.ordinary_private_centered_pairing,
            sis_centered_unit: self.sis_centered_pairing,
            canonical_binary_source_fields: GadgetNativeCanonicalBinaryFieldRowBreakdown::from_pairing(
                self.boolean_pairing.source_raw64,
                self.boolean_pairing.source_prefix31,
                self.canonical_binary_field_source_cols,
            ),
            synthetic_ring_fields: GadgetNativeCanonicalBinaryFieldRowBreakdown::from_pairing(
                self.boolean_pairing.synthetic_ring_raw64,
                self.boolean_pairing.synthetic_ring_prefix31,
                self.synthetic_ring_fields,
            ),
            synthetic_product_sum_fields: GadgetNativeCanonicalBinaryFieldRowBreakdown::from_pairing(
                self.boolean_pairing.synthetic_product_sum_raw64,
                self.boolean_pairing.synthetic_product_sum_prefix31,
                self.synthetic_product_sum_fields,
            ),
            fallback: self.fallback_source_rows,
            sbox: self.sboxes,
            k_mul: self.k_muls * K_MUL_ROWS,
            product_sum: self.product_sum_rows,
            ring_mul: self.ring_muls * RING_MUL_ROWS,
            acceptance_tree_bit_pair: self.acceptance_tree_bit_pair_rows,
            acceptance_product_aggregate: self.acceptance_product_aggregate_rows,
            acceptance_root_binding: self.acceptance_root_binding_rows,
            packed_mod5_low_bit_pair: self.packed_mod5_low_bit_pair_rows,
            packed_mod5_high_bit_pair: self.packed_mod5_high_bit_pair_rows,
            packed_mod5_residue_pair: self.packed_mod5_residue_pair_rows,
            selection_accept_aggregate: self.selection_accept_aggregate_rows,
            selection_prefix_aggregate: self.selection_prefix_aggregate_rows,
            selection_symbol_aggregate: self.selection_symbol_aggregate_rows,
        }
    }
}

/// Stage breakdown plus the reconciled whole-branch estimate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GadgetNativeStageProfile {
    pub total: GadgetNativeEstimate,
    pub stages: Vec<GadgetNativeStageEstimate>,
}

impl GadgetNativeStageProfile {
    /// Coalesce repeated disjoint ranges with the same stable leaf label.
    pub fn aggregate_by_label(&self) -> Vec<GadgetNativeStageEstimate> {
        let mut totals = Vec::<GadgetNativeStageEstimate>::new();
        for stage in &self.stages {
            if let Some(total) = totals.iter_mut().find(|total| total.label == stage.label) {
                total.add(stage);
            } else {
                totals.push(stage.clone());
            }
        }
        totals
    }

    /// Sum one semantic owner and all of its dot-delimited descendants.
    pub fn aggregate_prefix(&self, prefix: &'static str) -> Option<GadgetNativeStageEstimate> {
        let mut total = GadgetNativeStageEstimate::empty(prefix);
        for stage in &self.stages {
            let is_descendant = stage.label == prefix
                || stage
                    .label
                    .strip_prefix(prefix)
                    .is_some_and(|suffix| suffix.starts_with('.'));
            if is_descendant {
                total.add(stage);
            }
        }
        (total.occurrences != 0).then_some(total)
    }
}

#[derive(Debug, Error)]
pub enum GadgetNativeStageProfileError {
    #[error(transparent)]
    Encoding(#[from] GadgetNativeError),
    #[error("encoding stage trace must start at row 0/column 1 and end at the source dimensions")]
    Boundary,
    #[error("encoding stage checkpoints are not monotonic")]
    Order,
    #[error("{gadget} event rows {start}..{end} cross a stage boundary")]
    CrossStage {
        gadget: &'static str,
        start: usize,
        end: usize,
    },
    #[error("packed mod-5 trace must exactly match one chunk.mod5 stage range")]
    PackedMod5Stage,
    #[error("aggregate acceptance trace must exactly match one chunk.accept stage range")]
    AcceptanceStage,
}

#[derive(Clone, Copy)]
struct StageRange {
    label: &'static str,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
}

/// Attribute the exact low-norm estimate to named R1CS emission stages.
pub fn profile_r1cs_gadget_native_stages(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    public_bit_columns: &[usize],
) -> Result<GadgetNativeStageProfile, GadgetNativeStageProfileError> {
    let schedule = super::source_schedule::ValidatedSourceSchedule::checked(source, trace, public_bit_columns)?;
    let is_public = &schedule.is_public;
    let explicit_bits = &schedule.explicit_bits;
    let marks = &schedule.marks;
    let linear_columns = schedule
        .decisions()
        .iter()
        .map(|decision| matches!(decision, super::source_schedule::SourceColumnDecision::GenericLinear(_)))
        .collect::<Vec<_>>();
    let linear_definition_rows = &schedule.removed_definition_rows;
    let redundant_boolean_rows =
        super::boolean_dedup::ExactBooleanRows::from_plan(source, is_public, explicit_bits, &linear_columns, marks);
    redundant_boolean_rows.require_disjoint(linear_definition_rows)?;
    let removed_rows = marks
        .balanced_ternary
        .reduction_removed_rows(linear_definition_rows, redundant_boolean_rows.rows())?;
    let ranges = stage_ranges(source, trace)?;
    let mut stages = ranges
        .iter()
        .map(|range| GadgetNativeStageEstimate {
            label: range.label,
            occurrences: 1,
            source_rows: range.row_end - range.row_start,
            source_cols: range.col_end - range.col_start,
            one_bit_source_cols: 0,
            canonical_binary_field_source_cols: 0,
            ordinary_private_field_source_cols: 0,
            balanced_ternary_field_source_cols: 0,
            balanced_ternary_alias_source_cols: 0,
            balanced_ternary_binary_source_cols: 0,
            linearly_derived_source_cols: 0,
            gadget_derived_source_cols: 0,
            synthetic_ring_fields: 0,
            synthetic_product_sum_fields: 0,
            acceptance_chunks: 0,
            acceptance_encoded_cols: 0,
            acceptance_tree_output_cols: 0,
            acceptance_tree_bit_pair_rows: 0,
            acceptance_product_aggregate_rows: 0,
            acceptance_root_binding_rows: 0,
            packed_mod5_chunks: 0,
            packed_mod5_encoded_cols: 0,
            packed_mod5_low_bit_pair_rows: 0,
            packed_mod5_high_bit_pair_rows: 0,
            packed_mod5_residue_pair_rows: 0,
            boolean_pairing: GadgetNativeBooleanPairingBreakdown::default(),
            encoded_cols: 0,
            centered_encoded_cols: 0,
            centered_pairing: GadgetNativePairTailCount::default(),
            ordinary_private_encoded_cols: 0,
            ordinary_private_centered_pairing: GadgetNativePairTailCount::default(),
            sis_centered_encoded_cols: 0,
            sis_centered_pairing: GadgetNativePairTailCount::default(),
            encoded_rows: 0,
            redundant_boolean_source_rows: 0,
            fallback_source_rows: 0,
            poseidon_permutations: 0,
            poseidon_hash_permutations: 0,
            poseidon_hashes: 0,
            sboxes: 0,
            k_muls: 0,
            product_sum_batches: 0,
            product_sum_identities: 0,
            product_sum_rows: 0,
            ring_muls: 0,
            selection_accept_aggregate_rows: 0,
            selection_prefix_aggregate_rows: 0,
            selection_symbol_aggregate_rows: 0,
            hash_histogram: BTreeMap::new(),
        })
        .collect::<Vec<_>>();

    for (range, stage) in ranges.iter().zip(&mut stages) {
        for column in range.col_start..range.col_end {
            let decision = &schedule.decisions()[column];
            if marks.gadget_columns[column] {
                stage.gadget_derived_source_cols += 1;
            } else if decision.role() == super::GadgetNativeSourceRole::LinearlyDerived {
                stage.linearly_derived_source_cols += 1;
            } else if marks.balanced_ternary.is_structural(column) {
                // The opening stage owns the shared word, even when its source
                // field was allocated by an earlier message-construction stage.
                continue;
            } else {
                match decision {
                    super::source_schedule::SourceColumnDecision::PublicBit
                    | super::source_schedule::SourceColumnDecision::PrivateBoolean(_) => {
                        stage.one_bit_source_cols += 1;
                    }
                    super::source_schedule::SourceColumnDecision::CanonicalField(
                        super::source_schedule::CanonicalFieldKind::OrdinaryPrivate,
                    ) => stage.ordinary_private_field_source_cols += 1,
                    super::source_schedule::SourceColumnDecision::CanonicalField(
                        super::source_schedule::CanonicalFieldKind::DirectCanonicalU64,
                    ) => stage.canonical_binary_field_source_cols += 1,
                    _ => {}
                }
            }
        }
        stage.redundant_boolean_source_rows = (range.row_start..range.row_end)
            .filter(|&row| redundant_boolean_rows.rows()[row])
            .count();
        stage.fallback_source_rows = (range.row_start..range.row_end)
            .filter(|&row| !marks.covered_rows[row] && !removed_rows[row] && !redundant_boolean_rows.rows()[row])
            .count();
    }

    reattribute_acceptance_stages(&ranges, &mut stages, trace)?;
    reattribute_packed_mod5_stages(&ranges, &mut stages, trace)?;

    for event in trace.sbox7() {
        let stage = event_stage(
            &ranges,
            event.source_rows.start,
            event.source_rows.end,
            "Poseidon2 sbox7",
        )?;
        stages[stage].sboxes += 1;
    }
    for opening in trace.balanced_ternary_openings() {
        let stage = event_stage(
            &ranges,
            opening.digit_rows.start,
            opening.transition_rows.end,
            "balanced-ternary opening",
        )?;
        stages[stage].balanced_ternary_field_source_cols += 1;
        stages[stage].balanced_ternary_alias_source_cols += BALANCED_TERNARY_DIGITS;
        let binary = opening.negative_cols.len() + opening.borrow_cols.len();
        stages[stage].one_bit_source_cols += binary;
        stages[stage].balanced_ternary_binary_source_cols += binary;
    }
    for (index, event) in trace.k_muls().iter().enumerate() {
        if marks.product_sums.is_nested_k_mul(index) {
            continue;
        }
        let stage = event_stage(
            &ranges,
            event.source_rows.start,
            event.source_rows.end,
            "K multiplication",
        )?;
        stages[stage].k_muls += 1;
    }
    for identity in marks.product_sums.costs() {
        let stage = event_stage(
            &ranges,
            identity.stage_row,
            identity.stage_row + 1,
            "product-sum identity",
        )?;
        stages[stage].product_sum_batches += usize::from(identity.starts_batch);
        stages[stage].product_sum_identities += 1;
        stages[stage].product_sum_rows += identity.encoded_rows;
        stages[stage].synthetic_product_sum_fields += identity.synthetic_fields;
    }
    for event in trace.ring_muls_toom3() {
        let stage = event_stage(
            &ranges,
            event.source_rows.start,
            event.source_rows.end,
            "Toom-3 ring multiplication",
        )?;
        stages[stage].ring_muls += 1;
        stages[stage].synthetic_ring_fields += TOOM_EVALUATIONS * TOOM_COEFFICIENTS;
    }
    for event in trace.first_accepted_selections() {
        let accept_stage = event_stage(
            &ranges,
            event.bind_rows.start,
            event.bind_rows.start + 1,
            "first-accepted selection accept binding",
        )?;
        let prefix_stage = event_stage(
            &ranges,
            event.bind_rows.start + 1,
            event.bind_rows.start + 2,
            "first-accepted selection prefix binding",
        )?;
        let symbol_stage = event_stage(
            &ranges,
            event.bind_rows.start + 2,
            event.bind_rows.end,
            "first-accepted selection symbol binding",
        )?;
        let rows = super::selection::aggregate_rows_per_family(event);
        stages[accept_stage].selection_accept_aggregate_rows += rows;
        stages[prefix_stage].selection_prefix_aggregate_rows += rows;
        stages[symbol_stage].selection_symbol_aggregate_rows += rows;
    }
    for event in trace.poseidon_permutations() {
        let stage = event_stage(
            &ranges,
            event.source_rows.start,
            event.source_rows.end,
            "Poseidon2 permutation",
        )?;
        stages[stage].poseidon_permutations += 1;
    }
    for event in trace.poseidon_hashes() {
        let stage = event_stage(
            &ranges,
            event.source_rows.start,
            event.source_rows.end,
            "Poseidon2 hash",
        )?;
        let permutations = event.permutation_range.len();
        stages[stage].poseidon_hashes += 1;
        stages[stage].poseidon_hash_permutations += permutations;
        let entry = stages[stage]
            .hash_histogram
            .entry(event.input_len)
            .or_default();
        entry.0 += 1;
        entry.1 += permutations;
    }

    for stage in &mut stages {
        let field_slots =
            stage.canonical_binary_field_source_cols + stage.synthetic_ring_fields + stage.synthetic_product_sum_fields;
        let packed_boolean_cols = stage.packed_mod5_low_bit_pair_rows * 2 + stage.packed_mod5_high_bit_pair_rows;
        let packed_synthetic_cols = stage
            .packed_mod5_encoded_cols
            .checked_sub(packed_boolean_cols)
            .expect("packed mod-5 Boolean-column ownership");
        stage.ordinary_private_encoded_cols =
            stage.ordinary_private_field_source_cols * super::ordinary_private_field::ORDINARY_PRIVATE_DIGITS;
        stage.ordinary_private_centered_pairing =
            GadgetNativePairTailCount::from_coordinates(stage.ordinary_private_encoded_cols);
        stage.sis_centered_encoded_cols = stage.balanced_ternary_field_source_cols * BALANCED_TERNARY_DIGITS;
        stage.sis_centered_pairing = GadgetNativePairTailCount::from_coordinates(stage.sis_centered_encoded_cols);
        stage.centered_encoded_cols = stage
            .ordinary_private_encoded_cols
            .saturating_add(stage.sis_centered_encoded_cols);
        stage.centered_pairing = stage.ordinary_private_centered_pairing;
        stage.centered_pairing.add(stage.sis_centered_pairing);
        stage.encoded_cols = stage.one_bit_source_cols
            + stage.centered_encoded_cols
            + stage.acceptance_tree_output_cols
            + packed_synthetic_cols
            + field_slots * CANONICAL_SLOT_WIDTH;
        let common_boolean_coordinates = stage
            .one_bit_source_cols
            .saturating_sub(stage.balanced_ternary_binary_source_cols)
            .saturating_sub(packed_boolean_cols)
            .saturating_sub(
                stage
                    .acceptance_encoded_cols
                    .saturating_sub(stage.acceptance_tree_output_cols),
            );
        stage.boolean_pairing = GadgetNativeBooleanPairingBreakdown::one_stage(
            common_boolean_coordinates,
            stage.canonical_binary_field_source_cols,
            stage.synthetic_ring_fields,
            stage.synthetic_product_sum_fields,
        );
        stage.encoded_rows = stage.boolean_pairing.total_rows()
            + stage.centered_pairing.total_rows()
            + field_slots * GOLDILOCKS_CANONICALITY_PAIR_ROWS
            + stage.fallback_source_rows
            + stage.sboxes
            + stage.k_muls * K_MUL_ROWS
            + stage.product_sum_rows
            + stage.ring_muls * RING_MUL_ROWS
            + stage.acceptance_tree_bit_pair_rows
            + stage.acceptance_product_aggregate_rows
            + stage.acceptance_root_binding_rows
            + stage.packed_mod5_low_bit_pair_rows
            + stage.packed_mod5_high_bit_pair_rows
            + stage.packed_mod5_residue_pair_rows
            + stage.selection_accept_aggregate_rows
            + stage.selection_prefix_aggregate_rows
            + stage.selection_symbol_aggregate_rows;
    }

    let sum = |f: fn(&GadgetNativeStageEstimate) -> usize| stages.iter().map(f).sum::<usize>();
    let encoded_cols = 1 + sum(|stage| stage.encoded_cols);
    let mut boolean_pairing = GadgetNativeBooleanPairingBreakdown::default();
    for stage in &stages {
        boolean_pairing.add(stage.boolean_pairing);
    }
    let total = GadgetNativeEstimate {
        source_rows: source.rows(),
        source_cols: source.cols(),
        public_input_len: 1 + public_bit_columns.len(),
        encoded_cols,
        encoded_rows: sum(|stage| stage.encoded_rows),
        max_degree: 8,
        one_bit_source_cols: sum(|stage| stage.one_bit_source_cols),
        canonical_binary_field_source_cols: sum(|stage| stage.canonical_binary_field_source_cols),
        ordinary_private_field_source_cols: sum(|stage| stage.ordinary_private_field_source_cols),
        balanced_ternary_field_source_cols: sum(|stage| stage.balanced_ternary_field_source_cols),
        balanced_ternary_alias_source_cols: sum(|stage| stage.balanced_ternary_alias_source_cols),
        balanced_ternary_binary_source_cols: sum(|stage| stage.balanced_ternary_binary_source_cols),
        centered_encoded_cols: sum(|stage| stage.centered_encoded_cols),
        centered_pairing: stages
            .iter()
            .fold(GadgetNativePairTailCount::default(), |mut total, stage| {
                total.add(stage.centered_pairing);
                total
            }),
        ordinary_private_encoded_cols: sum(|stage| stage.ordinary_private_encoded_cols),
        ordinary_private_centered_pairing: stages.iter().fold(
            GadgetNativePairTailCount::default(),
            |mut total, stage| {
                total.add(stage.ordinary_private_centered_pairing);
                total
            },
        ),
        sis_centered_encoded_cols: sum(|stage| stage.sis_centered_encoded_cols),
        sis_centered_pairing: stages
            .iter()
            .fold(GadgetNativePairTailCount::default(), |mut total, stage| {
                total.add(stage.sis_centered_pairing);
                total
            }),
        synthetic_ring_fields: sum(|stage| stage.synthetic_ring_fields),
        synthetic_product_sum_fields: sum(|stage| stage.synthetic_product_sum_fields),
        acceptance_chunks: sum(|stage| stage.acceptance_chunks),
        acceptance_encoded_cols: sum(|stage| stage.acceptance_encoded_cols),
        acceptance_tree_output_cols: sum(|stage| stage.acceptance_tree_output_cols),
        acceptance_tree_bit_pair_rows: sum(|stage| stage.acceptance_tree_bit_pair_rows),
        acceptance_product_aggregate_rows: sum(|stage| stage.acceptance_product_aggregate_rows),
        acceptance_root_binding_rows: sum(|stage| stage.acceptance_root_binding_rows),
        packed_mod5_chunks: sum(|stage| stage.packed_mod5_chunks),
        packed_mod5_encoded_cols: sum(|stage| stage.packed_mod5_encoded_cols),
        packed_mod5_low_bit_pair_rows: sum(|stage| stage.packed_mod5_low_bit_pair_rows),
        packed_mod5_high_bit_pair_rows: sum(|stage| stage.packed_mod5_high_bit_pair_rows),
        packed_mod5_residue_pair_rows: sum(|stage| stage.packed_mod5_residue_pair_rows),
        boolean_pairing,
        linearly_derived_source_cols: sum(|stage| stage.linearly_derived_source_cols),
        gadget_derived_source_cols: sum(|stage| stage.gadget_derived_source_cols),
        redundant_boolean_source_rows: sum(|stage| stage.redundant_boolean_source_rows),
        fallback_source_rows: sum(|stage| stage.fallback_source_rows),
    };
    Ok(GadgetNativeStageProfile { total, stages })
}

/// Replace each validated four-row acceptance range with its exact semantic
/// source owner and three disjoint nine-row lowering leaves.
fn reattribute_acceptance_stages(
    ranges: &[StageRange],
    stages: &mut Vec<GadgetNativeStageEstimate>,
    trace: &R1csEncodingTrace,
) -> Result<(), GadgetNativeStageProfileError> {
    let mut claimed_ranges = vec![false; ranges.len()];
    for event in trace.acceptance_chunks() {
        let index = event_stage(
            ranges,
            event.source_rows.start,
            event.source_rows.end,
            "aggregate acceptance chunk",
        )?;
        let range = ranges[index];
        if claimed_ranges[index]
            || range.label != pi_rlc_challenge_stage::CHUNK_ACCEPT
            || range.row_start != event.source_rows.start
            || range.row_end != event.source_rows.end
            || range.col_start != event.allocated_columns.start
            || range.col_end != event.allocated_columns.end
        {
            return Err(GadgetNativeStageProfileError::AcceptanceStage);
        }
        claimed_ranges[index] = true;
        stages[index] = stage_occurrence(pi_rlc_challenge_stage::CHUNK_ACCEPT);
        stages.push(stage_occurrence(pi_rlc_challenge_stage::CHUNK_ACCEPT_PACKED));

        let mut tree = stage_occurrence(pi_rlc_challenge_stage::ACCEPT_TREE_BIT_PAIRS);
        tree.acceptance_chunks = 1;
        tree.acceptance_encoded_cols = super::acceptance::TREE_OUTPUTS_PER_CHUNK;
        tree.acceptance_tree_output_cols = super::acceptance::TREE_OUTPUTS_PER_CHUNK;
        tree.acceptance_tree_bit_pair_rows = super::acceptance::TREE_BIT_PAIR_ROWS_PER_CHUNK;
        tree.encoded_cols = super::acceptance::TREE_OUTPUTS_PER_CHUNK;
        stages.push(tree);

        let mut aggregate = stage_occurrence(pi_rlc_challenge_stage::ACCEPT_PRODUCT_AGGREGATE);
        aggregate.acceptance_product_aggregate_rows = super::acceptance::PRODUCT_AGGREGATE_ROWS_PER_CHUNK;
        stages.push(aggregate);

        let mut root = stage_occurrence(pi_rlc_challenge_stage::ACCEPT_ROOT_BINDING);
        root.source_rows = 4;
        root.source_cols = 2;
        root.one_bit_source_cols = 1;
        root.gadget_derived_source_cols = 1;
        root.acceptance_encoded_cols = 1;
        root.acceptance_root_binding_rows = super::acceptance::ROOT_BINDING_ROWS_PER_CHUNK;
        root.encoded_cols = 1;
        stages.push(root);
    }
    if ranges
        .iter()
        .enumerate()
        .any(|(index, range)| range.label == pi_rlc_challenge_stage::CHUNK_ACCEPT && !claimed_ranges[index])
    {
        return Err(GadgetNativeStageProfileError::AcceptanceStage);
    }
    Ok(())
}

/// Replace each validated flat Mod-5 source range with the exact mathematical
/// partition used by the packed lowering. The three leaves partition both the
/// 20 source rows/19 source columns and the 8 rows/15 encoded coordinates.
fn reattribute_packed_mod5_stages(
    ranges: &[StageRange],
    stages: &mut Vec<GadgetNativeStageEstimate>,
    trace: &R1csEncodingTrace,
) -> Result<(), GadgetNativeStageProfileError> {
    let mut claimed_ranges = vec![false; ranges.len()];
    for event in trace.mod5_chunks() {
        let index = event_stage(
            ranges,
            event.source_rows.start,
            event.source_rows.end,
            "packed mod-5 chunk",
        )?;
        let range = ranges[index];
        if claimed_ranges[index]
            || range.label != pi_rlc_challenge_stage::CHUNK_MOD5
            || range.row_start != event.source_rows.start
            || range.row_end != event.source_rows.end
            || range.col_start != event.allocated_columns.start
            || range.col_end != event.allocated_columns.end
        {
            return Err(GadgetNativeStageProfileError::PackedMod5Stage);
        }
        claimed_ranges[index] = true;
        stages[index] = stage_occurrence(pi_rlc_challenge_stage::CHUNK_MOD5);
        stages.push(stage_occurrence(pi_rlc_challenge_stage::CHUNK_MOD5_PACKED));

        let mut low = stage_occurrence(pi_rlc_challenge_stage::LOW_BIT_PAIRS);
        low.source_rows = 12;
        low.source_cols = 12;
        low.one_bit_source_cols = 12;
        low.packed_mod5_chunks = 1;
        low.packed_mod5_encoded_cols = 12;
        low.packed_mod5_low_bit_pair_rows = 6;
        low.encoded_cols = 12;
        stages.push(low);

        let mut high = stage_occurrence(pi_rlc_challenge_stage::HIGH_BIT_PAIR);
        high.source_rows = 4;
        high.source_cols = 3;
        high.one_bit_source_cols = 1;
        high.linearly_derived_source_cols = 2;
        high.packed_mod5_encoded_cols = 1;
        high.packed_mod5_high_bit_pair_rows = 1;
        high.encoded_cols = 1;
        stages.push(high);

        let mut residue = stage_occurrence(pi_rlc_challenge_stage::RESIDUE_PAIR);
        residue.source_rows = 4;
        residue.source_cols = 4;
        residue.linearly_derived_source_cols = 1;
        residue.gadget_derived_source_cols = 3;
        residue.packed_mod5_encoded_cols = 2;
        residue.packed_mod5_residue_pair_rows = 1;
        residue.encoded_cols = 2;
        stages.push(residue);
    }
    if ranges
        .iter()
        .enumerate()
        .any(|(index, range)| range.label == pi_rlc_challenge_stage::CHUNK_MOD5 && !claimed_ranges[index])
    {
        return Err(GadgetNativeStageProfileError::PackedMod5Stage);
    }
    Ok(())
}

fn stage_occurrence(label: &'static str) -> GadgetNativeStageEstimate {
    let mut stage = GadgetNativeStageEstimate::empty(label);
    stage.occurrences = 1;
    stage
}

fn stage_ranges(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
) -> Result<Vec<StageRange>, GadgetNativeStageProfileError> {
    let checkpoints = trace.stages();
    if checkpoints.len() < 2
        || checkpoints[0].row != 0
        || checkpoints[0].col != 1
        || checkpoints
            .last()
            .is_none_or(|last| last.row != source.rows() || last.col != source.cols())
    {
        return Err(GadgetNativeStageProfileError::Boundary);
    }
    let mut ranges = Vec::with_capacity(checkpoints.len() - 1);
    for pair in checkpoints.windows(2) {
        let (start, end) = (&pair[0], &pair[1]);
        if start.row > end.row || start.col > end.col {
            return Err(GadgetNativeStageProfileError::Order);
        }
        ranges.push(StageRange {
            label: start.label,
            row_start: start.row,
            row_end: end.row,
            col_start: start.col,
            col_end: end.col,
        });
    }
    Ok(ranges)
}

fn event_stage(
    ranges: &[StageRange],
    start: usize,
    end: usize,
    gadget: &'static str,
) -> Result<usize, GadgetNativeStageProfileError> {
    let Some(index) = ranges
        .iter()
        .position(|range| start >= range.row_start && start < range.row_end)
    else {
        return Err(GadgetNativeStageProfileError::CrossStage { gadget, start, end });
    };
    if end > ranges[index].row_end {
        return Err(GadgetNativeStageProfileError::CrossStage { gadget, start, end });
    }
    Ok(index)
}
