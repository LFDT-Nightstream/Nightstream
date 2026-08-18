//! Selective low-norm CCS compiler for recorded verifier traces.
//!
//! Owns: selective slot planning, exact temporary substitution, retained-value
//! encodings, direct trace rows, and width-audit composition.
//!
//! Does not own: source trace recording, semantic proof of each trace family,
//! outer `F'` orchestration, or folding verification.
//!
//! Emits constraints: yes. Ordinary source rows remain the arithmetic
//! authority. A temporary is removed only when constrained retained operands
//! reconstruct it under the recorded trace contract.

use std::collections::BTreeSet;

use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use super::lowering::{DerivedProductSumEncoding, LowNormR1csError, MultiBranchLowNormR1cs};
use super::selective_audit::{
    physical_stage_width_audits, retained_trace_widths, row_family_width_audits, SelectiveArmWidthAudit,
    SelectiveCanonicalOpeningAudit, SelectiveCompilerAudit, SelectiveLayoutAudit, SelectiveLinearDefinitionAudit,
    SelectiveLinearDefinitionTermAudit, SelectiveLowNormWidthAudit, SelectiveRewriteKind, SelectiveRowMappingAudit,
};
use super::SparseR1cs;
use crate::engine::r1cs_circuit::builder::{
    BalancedTernaryDecomposition, CanonicalU64Decomposition, ProductFactorTrace,
};
use crate::engine::r1cs_circuit::Lc;

#[path = "selective_canonical.rs"]
mod canonical;
#[path = "selective_combined_audit.rs"]
mod combined_audit;
#[path = "selective_definitions.rs"]
mod definitions;
#[path = "selective_emit.rs"]
mod emit;
#[path = "selective_projected_decoder.rs"]
mod projected_decoder;
#[path = "selective_projected_rows.rs"]
mod projected_rows;
#[path = "selective_rows.rs"]
mod rows;
#[path = "selective_shape.rs"]
mod shape;
#[path = "selective_structure.rs"]
mod structure;
#[path = "selective_terms.rs"]
mod terms;
pub(crate) use combined_audit::audit_multi_branch_selective_compact_layout_and_decoder_runs_with_shared_bit_prefix;
pub use combined_audit::SelectiveCompactLayoutAudit;
use definitions::{find_linear_definitions, LinearDefinitions};
use emit::{lc_from_column, trace_error};
pub use projected_decoder::{
    SelectiveProjectedDecoderProvenance, SelectiveProjectedDecoderRunProvenance, SelectiveProjectedSourceDecoder,
    SelectiveProjectedSourceDecoderRun, SelectiveProjectedSourceDecoderStridedRun,
    SelectiveProjectedSourceDecoderTemplate, SelectiveProjectedSourceDecoderTemplateInstances,
    SelectiveProjectedSourceFamilyRange, SelectiveProjectedSourceResolution, SelectiveProjectedSourceResolutionRun,
};
pub(crate) use projected_rows::{
    project_rows_with_alignment, project_rows_with_complete_source_provenance_with_alignment,
};
pub use projected_rows::{
    SelectiveProjectedDerivedProductSum, SelectiveProjectedExplicitRunCensus, SelectiveProjectedGeometricRun,
    SelectiveProjectedPort, SelectiveProjectedPoseidon2SboxStep, SelectiveProjectedProductFactor,
    SelectiveProjectedPublicCoordinate, SelectiveProjectedPublicCoordinateSource, SelectiveProjectedRetainedStep,
    SelectiveProjectedRewriteOutput, SelectiveProjectedRewriteStep, SelectiveProjectedRowArtifact,
    SelectiveProjectedRowsAudit, SelectiveProjectedSourceDefinition, SelectiveProjectedSourceImage,
    SelectiveProjectedSourceLinearCombination, SelectiveProjectedSourceProvenance, SelectiveProjectedSourceSlot,
    SelectiveProjectedSourceTerm, SelectiveProjectedTerm,
};
use rows::{balanced_ternary_decompositions_by_digit_start, skipped_selective_rows, PreparedSelectiveRows};
pub(crate) use shape::{
    audit_multi_branch_selective_low_norm_shape_with_alignment, SelectiveLowNormShape, SelectiveLowNormShapeSummary,
};
#[doc(hidden)]
pub use shape::{is_canonical_selective_low_norm_polynomial, selective_polynomial};

pub(super) const EVAL_GROUP_SIZE: usize = 5;
const BALANCED_FIELD_WIDTH: usize = 41;
const SEPTENARY_FIELD_WIDTH: usize = 23;
const CANON_CHUNK_WIDTH: usize = 2;
const CANON_CHUNK_COUNT: usize = BALANCED_FIELD_WIDTH.div_ceil(CANON_CHUNK_WIDTH);
const BINARY_FIELD_WIDTH: usize = 64;
const BIT: usize = 0;
pub(super) const GENERAL_SELECTOR: usize = 1;
pub(super) const A: usize = 2;
pub(super) const B: usize = 3;
pub(super) const C: usize = 4;
const SBOX_INPUT: usize = 5;
const CENTERED_UNIT: usize = 6;
const EVAL_SELECTOR: usize = 7;
// Two-trit canonical rows use these otherwise ordinary evaluation-factor
// ports as one-hot selectors for the normalized base-9 bound in 0..=4.
// GENERAL_SELECTOR gates the family, so evaluation rows remain disjoint.
const CANON_CHUNK_CLASS_SELECTORS: [usize; 5] = [8, 9, 10, 11, 12];
const EVAL_PAIRS: [(usize, usize); EVAL_GROUP_SIZE] =
    [(BIT, A), (B, SBOX_INPUT), (CENTERED_UNIT, 8), (9, 10), (11, 12)];
pub(super) const SELECTIVE_ARITY: usize = 13;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SelectiveEncoding {
    norm_base: u32,
    general_field_width: usize,
}

impl SelectiveEncoding {
    fn for_norm_base(norm_base: u32) -> Result<Self, LowNormR1csError> {
        match norm_base {
            2 => Ok(Self {
                norm_base,
                general_field_width: BALANCED_FIELD_WIDTH,
            }),
            4 => Ok(Self {
                norm_base,
                general_field_width: SEPTENARY_FIELD_WIDTH,
            }),
            _ => Err(trace_error("selective lowering supports only norm base two or four")),
        }
    }

    pub(super) fn general_field_width(self) -> usize {
        self.general_field_width
    }

    pub(super) fn outer_norm_proves_centered_unit(self) -> bool {
        self.norm_base == 2
    }
}

struct SelectiveLayout {
    encoding: SelectiveEncoding,
    plans: Vec<SelectiveArmPlan>,
    slots: Vec<Vec<Option<(usize, usize)>>>,
    aliases: Vec<Vec<Option<(usize, usize)>>>,
    equal_aliases: Vec<Vec<Option<usize>>>,
    derived_product_sums: Vec<Vec<DerivedProductSumEncoding>>,
    selector_cols: Vec<usize>,
    public_padding_cols: Vec<usize>,
    private_padding_cols: Vec<usize>,
    public_input_len: usize,
    columns: usize,
    prepared_rows: PreparedSelectiveRows,
    compiler_audit: SelectiveCompilerAudit,
}

struct SelectiveLayoutCore {
    encoding: SelectiveEncoding,
    plans: Vec<SelectiveArmPlan>,
    slots: Vec<Vec<Option<(usize, usize)>>>,
    aliases: Vec<Vec<Option<(usize, usize)>>>,
    equal_aliases: Vec<Vec<Option<usize>>>,
    derived_product_sums: Vec<Vec<DerivedProductSumEncoding>>,
    selector_cols: Vec<usize>,
    public_padding_cols: Vec<usize>,
    private_padding_cols: Vec<usize>,
    public_input_len: usize,
    columns: usize,
    logical_public_input_len: usize,
    shared_private_start: usize,
    branch_start: usize,
    branch_coordinates: Vec<usize>,
    prepared_rows: PreparedSelectiveRows,
}

/// An owned selective layout that can be inspected for its exact shape and
/// then consumed once to emit the matching relation. Owning the source arms
/// prevents the final emitter from receiving different inputs.
pub(crate) struct PreparedSelectiveLowNormR1cs {
    arms: Vec<SparseR1cs>,
    shared_private_fields: usize,
    layout: SelectiveLayoutCore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SelectiveLayoutSummary {
    pub rows: usize,
    pub columns: usize,
    pub public_input_len: usize,
    pub total_coordinates: usize,
}

impl SelectiveLayoutCore {
    fn summary(&self) -> SelectiveLayoutSummary {
        SelectiveLayoutSummary {
            rows: self.prepared_rows.total_rows(),
            columns: self.columns.next_multiple_of(D),
            public_input_len: self.public_input_len,
            total_coordinates: self.columns,
        }
    }
}

impl PreparedSelectiveLowNormR1cs {
    pub(crate) fn shape_summary(&self) -> SelectiveLowNormShapeSummary {
        let layout = self.layout.summary();
        SelectiveLowNormShapeSummary {
            rows: layout.rows,
            columns: layout.columns,
            public_input_len: layout.public_input_len,
            polynomial: selective_polynomial(),
            total_coordinates: layout.total_coordinates,
        }
    }

    pub(crate) fn arm(&self, index: usize) -> &SparseR1cs {
        &self.arms[index]
    }

    /// Finish the exact compiler audit without emitting the final CCS
    /// matrices, and return the source arms from the same prepared plan.
    pub(crate) fn into_source_audit_parts(self) -> Result<(Vec<SparseR1cs>, SelectiveCompilerAudit), LowNormR1csError> {
        let Self {
            arms,
            shared_private_fields,
            layout,
        } = self;
        let layout = finish_selective_layout(&arms, shared_private_fields, layout)?;
        Ok((arms, layout.compiler_audit))
    }

    pub(crate) fn finish(self) -> Result<MultiBranchLowNormR1cs, LowNormR1csError> {
        let Self {
            arms,
            shared_private_fields,
            layout,
        } = self;
        let layout = finish_selective_layout(&arms, shared_private_fields, layout)?;
        build_selective_relation(&arms, shared_private_fields, layout)
    }
}

struct SelectiveArmPlan {
    widths: Vec<usize>,
    centered: Vec<bool>,
    source_boolean_rows: Vec<bool>,
    equality_roots: Vec<usize>,
    definitions: LinearDefinitions,
}

/// Compile one-hot field-R1CS arms to selective degree-eight CCS.
///
/// Two distinct zero regions are inserted and accounted independently:
/// the logical public prefix is first completed to `modulus`, then selectors
/// are allocated, and a second region places shared private advice at
/// `residue (mod modulus)`. This ordering keeps branch selectors out of the
/// active SuperNeo public ring carrier.
pub fn build_multi_branch_selective_low_norm_r1cs_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<MultiBranchLowNormR1cs, LowNormR1csError> {
    build_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        arms,
        shared_private_fields,
        shared_private_fields,
        modulus,
        residue,
    )
}

/// Project selected exact rows from the same emitter term stream used by the
/// full selective relation, without allocating arrays over every final column.
#[doc(hidden)]
pub fn audit_multi_branch_selective_rows_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
    selected_rows: &[usize],
) -> Result<SelectiveProjectedRowsAudit, LowNormR1csError> {
    project_rows_with_alignment(
        arms,
        shared_private_fields,
        shared_private_fields,
        modulus,
        residue,
        selected_rows,
    )
}

/// Project selected rows and include the exact source-column and rewrite
/// provenance used by assurance artifacts.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn audit_multi_branch_selective_rows_with_complete_source_provenance_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
    selected_rows: &[usize],
    source_arm: usize,
    source_columns: &[usize],
    retained_row_pairs: &[(usize, usize)],
) -> Result<SelectiveProjectedRowsAudit, LowNormR1csError> {
    project_rows_with_complete_source_provenance_with_alignment(
        arms,
        shared_private_fields,
        shared_private_fields,
        modulus,
        residue,
        selected_rows,
        source_arm,
        source_columns,
        retained_row_pairs,
    )
}

/// Compute complete run-compressed source decoders from one exact selective
/// layout without emitting its matrices. Each requested interval is checked
/// pointwise against the same slots, aliases, and elimination plan that the
/// production emitter consumes.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn audit_multi_branch_selective_decoder_runs_with_shared_bit_prefix(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    norm_base: u32,
    requests: &[(usize, std::ops::Range<usize>)],
) -> Result<Vec<SelectiveProjectedDecoderRunProvenance>, LowNormR1csError> {
    let layout = prepare_selective_layout_for_encoding(
        arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        SelectiveEncoding::for_norm_base(norm_base)?,
    )?;
    combined_audit::decoder_runs_from_layout(arms, &layout, requests)
}

/// Return the complete compiler ledger from the exact prepared layout without
/// adapting it through a finished-relation snapshot or emitting matrices.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn audit_multi_branch_selective_compiler_with_shared_bit_prefix(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    norm_base: u32,
) -> Result<SelectiveCompilerAudit, LowNormR1csError> {
    Ok(prepare_selective_layout_for_encoding(
        arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        SelectiveEncoding::for_norm_base(norm_base)?,
    )?
    .compiler_audit)
}

pub fn build_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<MultiBranchLowNormR1cs, LowNormR1csError> {
    let layout = prepare_selective_layout(arms, shared_private_fields, shared_private_bit_fields, modulus, residue)?;
    build_selective_relation(arms, shared_private_fields, layout)
}

fn build_selective_relation(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    layout: SelectiveLayout,
) -> Result<MultiBranchLowNormR1cs, LowNormR1csError> {
    let structure = structure::build_structure(
        arms,
        layout.encoding,
        &layout.plans,
        &layout.slots,
        &layout.aliases,
        &layout.equal_aliases,
        shared_private_fields,
        &layout.derived_product_sums,
        &layout.selector_cols,
        &layout.public_padding_cols,
        &layout.private_padding_cols,
        layout.columns,
        &layout.prepared_rows,
    )?;
    build_selective_relation_from_structure(arms, layout, structure)
}

fn build_selective_relation_from_structure(
    arms: &[SparseR1cs],
    layout: SelectiveLayout,
    structure: crate::paper::relations::Structure,
) -> Result<MultiBranchLowNormR1cs, LowNormR1csError> {
    Ok(MultiBranchLowNormR1cs::from_compiler_parts(
        structure,
        layout.public_input_len,
        layout.selector_cols,
        arms[0].m_in,
        layout.slots,
        layout.aliases,
        layout.equal_aliases,
        layout
            .plans
            .iter()
            .map(|plan| plan.centered.clone())
            .collect(),
        layout.derived_product_sums,
        Some(layout.compiler_audit),
    ))
}

pub(crate) fn prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
    arms: Vec<SparseR1cs>,
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    norm_base: u32,
) -> Result<PreparedSelectiveLowNormR1cs, LowNormR1csError> {
    let encoding = SelectiveEncoding::for_norm_base(norm_base)?;
    let layout = prepare_selective_layout_core(
        &arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        encoding,
    )?;
    Ok(PreparedSelectiveLowNormR1cs {
        arms,
        shared_private_fields,
        layout,
    })
}

/// Compute selective-lowering width without constructing output matrices.
/// Public-carrier and private-alignment padding remain separate audit fields.
pub fn audit_multi_branch_selective_low_norm_width_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLowNormWidthAudit, LowNormR1csError> {
    audit_multi_branch_selective_low_norm_width_with_shared_bit_prefix(
        arms,
        shared_private_fields,
        shared_private_fields,
        modulus,
        residue,
    )
}

pub(crate) fn audit_multi_branch_selective_low_norm_width_for_norm_base_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
    norm_base: u32,
) -> Result<SelectiveLowNormWidthAudit, LowNormR1csError> {
    let encoding = SelectiveEncoding::for_norm_base(norm_base)?;
    Ok(prepare_selective_layout_for_encoding(
        arms,
        shared_private_fields,
        shared_private_fields,
        modulus,
        residue,
        encoding,
    )?
    .compiler_audit
    .into_width())
}

pub fn audit_multi_branch_selective_low_norm_width_with_shared_bit_prefix(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLowNormWidthAudit, LowNormR1csError> {
    Ok(
        prepare_selective_layout(arms, shared_private_fields, shared_private_bit_fields, modulus, residue)?
            .compiler_audit
            .into_width(),
    )
}

fn prepare_selective_layout_core(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    encoding: SelectiveEncoding,
) -> Result<SelectiveLayoutCore, LowNormR1csError> {
    #[cfg(feature = "perf-timers")]
    let total_started = std::time::Instant::now();
    if arms.len() < 2 {
        return Err(LowNormR1csError::TooFewArms(arms.len()));
    }
    if modulus == 0 {
        return Err(LowNormR1csError::ZeroAlignmentModulus);
    }
    if shared_private_bit_fields > shared_private_fields {
        return Err(trace_error("shared bit prefix exceeds shared private prefix"));
    }
    for arm in arms {
        arm.validate_shape()?;
        if arm.m_in == 0 {
            return Err(LowNormR1csError::MissingPublicConstant);
        }
    }

    let public_field_count = arms[0].m_in;
    #[cfg(feature = "perf-timers")]
    let phase_started = std::time::Instant::now();
    let plans = arms
        .par_iter()
        .map(|arm| selective_arm_plan(arm, shared_private_fields, shared_private_bit_fields, encoding))
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(feature = "perf-timers")]
    let plans_elapsed = phase_started.elapsed();
    let widths = plans
        .iter()
        .map(|plan| plan.widths.as_slice())
        .collect::<Vec<_>>();
    validate_shared_shapes(arms, &widths, public_field_count, shared_private_fields)?;
    #[cfg(feature = "perf-timers")]
    let phase_started = std::time::Instant::now();
    let aliases = arms
        .par_iter()
        .zip(&plans)
        .map(|(arm, plan)| decomposition_aliases(arm, &plan.widths, shared_private_fields))
        .collect::<Vec<_>>();
    let equal_aliases = arms
        .par_iter()
        .zip(&plans)
        .zip(&aliases)
        .map(|((arm, plan), aliases)| equality_aliases(arm, plan, aliases, shared_private_fields))
        .collect::<Vec<_>>();
    #[cfg(feature = "perf-timers")]
    let aliases_elapsed = phase_started.elapsed();
    #[cfg(feature = "perf-timers")]
    let phase_started = std::time::Instant::now();
    let mut cursor = 1usize;
    let mut slots = arms.iter().map(|arm| vec![None; arm.m]).collect::<Vec<_>>();
    for col in 1..public_field_count {
        let width = widths[0][col];
        let slot = Some((cursor, width));
        for arm_slots in &mut slots {
            arm_slots[col] = slot;
        }
        cursor += width;
    }
    let logical_public_input_len = cursor;
    let public_padding_len = (modulus - cursor % modulus) % modulus;
    let public_padding_cols = (cursor..cursor + public_padding_len).collect::<Vec<_>>();
    cursor += public_padding_len;
    let public_input_len = cursor;
    let selector_cols = (0..arms.len())
        .map(|_| {
            let selector = cursor;
            cursor += 1;
            selector
        })
        .collect::<Vec<_>>();
    let residue = residue % modulus;
    let padding_len = (residue + modulus - cursor % modulus) % modulus;
    let private_padding_cols = (cursor..cursor + padding_len).collect::<Vec<_>>();
    cursor += padding_len;
    let shared_private_start = cursor;

    for offset in 0..shared_private_fields {
        let shared_cursor = cursor;
        let mut shared_slot = None;
        for (arm_index, arm) in arms.iter().enumerate() {
            let source = arm.m_in + offset;
            let mut arm_cursor = shared_cursor;
            assign_slot(
                &mut slots[arm_index],
                &widths[arm_index],
                &aliases[arm_index],
                &equal_aliases[arm_index],
                source,
                &mut arm_cursor,
            )?;
            if arm_index == 0 {
                cursor = arm_cursor;
                shared_slot = slots[arm_index][source];
            } else if arm_cursor != cursor || slots[arm_index][source] != shared_slot {
                return Err(trace_error(
                    "shared private decomposition aliases disagree across selective arms",
                ));
            }
        }
    }
    let branch_start = cursor;
    let mut arm_cursors = vec![branch_start; arms.len()];
    for (arm_index, arm) in arms.iter().enumerate() {
        let mut arm_cursor = branch_start;
        for col in arm.m_in + shared_private_fields..arm.m {
            assign_slot(
                &mut slots[arm_index],
                &widths[arm_index],
                &aliases[arm_index],
                &equal_aliases[arm_index],
                col,
                &mut arm_cursor,
            )?;
        }
        arm_cursors[arm_index] = arm_cursor;
    }
    let branch_coordinates = arm_cursors
        .iter()
        .map(|&arm_cursor| arm_cursor - branch_start)
        .collect::<Vec<_>>();
    let mut derived_product_sums = (0..arms.len()).map(|_| Vec::new()).collect::<Vec<Vec<_>>>();
    for (arm_index, arm) in arms.iter().enumerate() {
        for trace in arm.polynomial_evaluation_traces() {
            for limb in 0..2 {
                let product_indices = (1..trace.coefficient_cols.len()).collect::<Vec<_>>();
                let groups = product_indices.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
                let mut previous = None;
                for group in groups.iter().take(groups.len().saturating_sub(1)) {
                    let slot = (arm_cursors[arm_index], encoding.general_field_width());
                    arm_cursors[arm_index] += encoding.general_field_width();
                    let index = derived_product_sums[arm_index].len();
                    derived_product_sums[arm_index].push(DerivedProductSumEncoding {
                        slot,
                        factors: group
                            .iter()
                            .map(|&index| ProductFactorTrace {
                                left: lc_from_column(trace.coefficient_cols[index]),
                                right: lc_from_column(trace.power_cols[index][limb]),
                                coefficient: F::ONE,
                            })
                            .collect(),
                        previous,
                    });
                    previous = Some(index);
                }
            }
        }
        for batch in arm.product_sum_batch_traces() {
            for identity in &batch.identities {
                if identity.factors.len() <= EVAL_GROUP_SIZE {
                    continue;
                }
                let groups = identity.factors.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
                let mut previous = None;
                for group in groups.iter().take(groups.len() - 1) {
                    let slot = (arm_cursors[arm_index], encoding.general_field_width());
                    arm_cursors[arm_index] += encoding.general_field_width();
                    let index = derived_product_sums[arm_index].len();
                    derived_product_sums[arm_index].push(DerivedProductSumEncoding {
                        slot,
                        factors: group.to_vec(),
                        previous,
                    });
                    previous = Some(index);
                }
            }
        }
        cursor = cursor.max(arm_cursors[arm_index]);
    }
    #[cfg(feature = "perf-timers")]
    let layout_elapsed = phase_started.elapsed();

    #[cfg(feature = "perf-timers")]
    let phase_started = std::time::Instant::now();
    let prepared_rows = PreparedSelectiveRows::prepare(
        arms,
        &plans,
        &slots,
        &aliases,
        &equal_aliases,
        shared_private_fields,
        &derived_product_sums,
        selector_cols.len(),
        public_padding_cols.len(),
        private_padding_cols.len(),
        cursor,
        encoding,
    )?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[selective-layout-core] shared_private_fields={shared_private_fields} plans={:.3}s aliases={:.3}s layout={:.3}s rows={:.3}s total={:.3}s",
        plans_elapsed.as_secs_f64(),
        aliases_elapsed.as_secs_f64(),
        layout_elapsed.as_secs_f64(),
        phase_started.elapsed().as_secs_f64(),
        total_started.elapsed().as_secs_f64(),
    );
    Ok(SelectiveLayoutCore {
        encoding,
        plans,
        slots,
        aliases,
        equal_aliases,
        derived_product_sums,
        selector_cols,
        public_padding_cols,
        private_padding_cols,
        public_input_len,
        columns: cursor,
        logical_public_input_len,
        shared_private_start,
        branch_start,
        branch_coordinates,
        prepared_rows,
    })
}

fn prepare_selective_layout(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLayout, LowNormR1csError> {
    prepare_selective_layout_for_encoding(
        arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        SelectiveEncoding::for_norm_base(2)?,
    )
}

fn prepare_selective_layout_for_encoding(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    encoding: SelectiveEncoding,
) -> Result<SelectiveLayout, LowNormR1csError> {
    let core = prepare_selective_layout_core(
        arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        encoding,
    )?;
    finish_selective_layout(arms, shared_private_fields, core)
}

fn finish_selective_layout(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    core: SelectiveLayoutCore,
) -> Result<SelectiveLayout, LowNormR1csError> {
    #[cfg(feature = "perf-timers")]
    let total_started = std::time::Instant::now();
    let SelectiveLayoutCore {
        encoding,
        plans,
        slots,
        aliases,
        equal_aliases,
        derived_product_sums,
        selector_cols,
        public_padding_cols,
        private_padding_cols,
        public_input_len,
        columns,
        logical_public_input_len,
        shared_private_start,
        branch_start,
        branch_coordinates,
        prepared_rows,
    } = core;
    #[cfg(feature = "perf-timers")]
    let phase_started = std::time::Instant::now();
    let arms_audit: Vec<SelectiveArmWidthAudit> = arms
        .par_iter()
        .enumerate()
        .map(|(arm_index, arm)| {
            let branch_range = arm.m_in + shared_private_fields..arm.m;
            let branch_widths = &plans[arm_index].widths[branch_range.clone()];
            let eliminated_columns = branch_widths.iter().filter(|&&width| width == 0).count();
            let unit_columns = branch_widths.iter().filter(|&&width| width == 1).count();
            let balanced_columns = branch_widths
                .iter()
                .filter(|&&width| width == encoding.general_field_width())
                .count();
            let binary_columns = branch_widths
                .iter()
                .filter(|&&width| width == BINARY_FIELD_WIDTH)
                .count();
            let retained_coordinates_before_aliases = branch_widths.iter().sum();
            let decomposition_aliases = aliases[arm_index][branch_range.clone()]
                .iter()
                .filter(|alias| alias.is_some())
                .count();
            let equality_aliases = equal_aliases[arm_index][branch_range]
                .iter()
                .filter(|alias| alias.is_some())
                .count();
            let derived_product_sums = derived_product_sums[arm_index].len();
            let derived_coordinates = derived_product_sums * encoding.general_field_width();
            SelectiveArmWidthAudit {
                branch_source_columns: branch_widths.len(),
                eliminated_columns,
                unit_columns,
                balanced_columns,
                binary_columns,
                retained_coordinates_before_aliases,
                decomposition_aliases,
                equality_aliases,
                branch_coordinates: branch_coordinates[arm_index],
                derived_product_sums,
                derived_coordinates,
                total_branch_coordinates: branch_coordinates[arm_index] + derived_coordinates,
                traces: retained_trace_widths(arm, &plans[arm_index].widths),
                row_families: row_family_width_audits(
                    arm,
                    &plans[arm_index].widths,
                    arm.m_in + shared_private_fields,
                    encoding.general_field_width(),
                    BINARY_FIELD_WIDTH,
                ),
                physical_stages: physical_stage_width_audits(
                    arm,
                    &plans[arm_index].widths,
                    &aliases[arm_index],
                    &equal_aliases[arm_index],
                    &plans[arm_index].definitions.by_column,
                    &plans[arm_index].centered,
                    &plans[arm_index].source_boolean_rows,
                    encoding.general_field_width(),
                    BALANCED_FIELD_WIDTH,
                    encoding.outer_norm_proves_centered_unit(),
                    arm.m_in + shared_private_fields,
                ),
            }
        })
        .collect();
    for (arm_index, arm) in arms_audit.iter().enumerate() {
        if !arm.physical_stages.is_empty() {
            assert_eq!(
                arm.physical_stages
                    .iter()
                    .map(|stage| stage.source_column_count)
                    .sum::<usize>(),
                arm.branch_source_columns,
                "physical-stage decoder dispositions do not cover selective arm {arm_index}",
            );
            assert_eq!(
                arm.physical_stages
                    .iter()
                    .map(|stage| stage.linear_definition_columns)
                    .sum::<usize>(),
                plans[arm_index].definitions.entries.len(),
                "physical-stage definitions do not cover selective arm {arm_index}",
            );
            assert_eq!(
                arm.physical_stages
                    .iter()
                    .map(|stage| stage.allocated_coordinates)
                    .sum::<usize>(),
                arm.branch_coordinates,
                "physical-stage width does not cover selective arm {arm_index}",
            );
        }
    }
    #[cfg(feature = "perf-timers")]
    let width_audit_elapsed = phase_started.elapsed();
    let width_audit = SelectiveLowNormWidthAudit {
        constant_coordinate: 1,
        logical_public_coordinates: logical_public_input_len - 1,
        public_carrier_padding: public_padding_cols.len(),
        public_coordinates: public_input_len - 1,
        selector_coordinates: selector_cols.len(),
        alignment_padding: private_padding_cols.len(),
        shared_private_coordinates: branch_start - public_input_len - selector_cols.len() - private_padding_cols.len(),
        branch_start,
        arms: arms_audit,
        total_coordinates: columns,
    };
    let layout_audit = SelectiveLayoutAudit::from_prepared_layout(
        logical_public_input_len,
        public_input_len,
        public_padding_cols.clone(),
        selector_cols.clone(),
        private_padding_cols.clone(),
        shared_private_start..branch_start,
        branch_start..columns,
        columns..columns.next_multiple_of(D),
    );
    let row_audit = prepared_rows.audit();
    let first_accepted_selections =
        super::selective_selection_audit::audit_first_accepted_selections(arms, &row_audit)?;
    #[cfg(feature = "perf-timers")]
    let phase_started = std::time::Instant::now();
    let canonical_openings = audit_canonical_openings(arms, &slots, &row_audit, columns)?;
    let source_arm_linear_definitions = plans
        .iter()
        .map(|plan| {
            let mut definitions = plan
                .definitions
                .entries
                .iter()
                .map(|definition| {
                    SelectiveLinearDefinitionAudit::new(
                        definition.row,
                        definition.target,
                        definition.rhs.constant,
                        definition
                            .rhs
                            .terms
                            .iter()
                            .map(|&(column, coefficient)| SelectiveLinearDefinitionTermAudit::new(column, coefficient))
                            .collect(),
                    )
                })
                .collect::<Vec<_>>();
            definitions.sort_unstable_by_key(SelectiveLinearDefinitionAudit::target);
            definitions
        })
        .collect();
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[selective-layout-audit] shared_private_fields={shared_private_fields} width_audit={:.3}s openings={:.3}s total={:.3}s",
        width_audit_elapsed.as_secs_f64(),
        phase_started.elapsed().as_secs_f64(),
        total_started.elapsed().as_secs_f64(),
    );
    Ok(SelectiveLayout {
        encoding,
        plans,
        slots,
        aliases,
        equal_aliases,
        derived_product_sums,
        selector_cols,
        public_padding_cols,
        private_padding_cols,
        public_input_len,
        columns,
        prepared_rows,
        compiler_audit: SelectiveCompilerAudit::new(
            layout_audit,
            width_audit,
            row_audit,
            canonical_openings,
            arms.iter()
                .map(|arm| arm.physical_stage_ranges().to_vec())
                .collect(),
            source_arm_linear_definitions,
            first_accepted_selections,
        ),
    })
}

fn audit_canonical_openings(
    arms: &[SparseR1cs],
    slots: &[Vec<Option<(usize, usize)>>],
    rows: &SelectiveRowMappingAudit,
    columns: usize,
) -> Result<Vec<Vec<SelectiveCanonicalOpeningAudit>>, LowNormR1csError> {
    let mut result = Vec::with_capacity(arms.len());
    for (arm_index, arm) in arms.iter().enumerate() {
        let decompositions = balanced_ternary_decompositions_by_digit_start(arm.balanced_ternary_decompositions())?;
        let rewrites = rows
            .rewrites()
            .iter()
            .filter(|rewrite| {
                rewrite.arm() == arm_index && rewrite.kind() == SelectiveRewriteKind::ShiftedTernaryCanonical
            })
            .collect::<Vec<_>>();
        if rewrites.len() != arm.shifted_ternary_canonical_traces().len() {
            return Err(trace_error("shifted-ternary trace and emitted-rewrite counts disagree"));
        }

        let mut openings = Vec::with_capacity(rewrites.len());
        for (trace, rewrite) in arm.shifted_ternary_canonical_traces().iter().zip(rewrites) {
            let decomposition = decompositions
                .get(&trace.digit_columns_start)
                .copied()
                .ok_or_else(|| trace_error("shifted-ternary trace has no source-field decomposition"))?;
            if decomposition
                .digit_cols
                .iter()
                .copied()
                .ne(trace.digit_columns_start..trace.digit_columns_start + BALANCED_FIELD_WIDTH)
            {
                return Err(trace_error("shifted-ternary digit columns are not one exact word"));
            }

            let digit_coordinates = decomposition
                .digit_cols
                .iter()
                .map(|&column| match slots[arm_index][column] {
                    Some((coordinate, 1)) => Ok(coordinate),
                    _ => Err(trace_error("shifted-ternary digit does not own one final coordinate")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let Some((source_coordinate, source_width)) = slots[arm_index][decomposition.field_col] else {
                return Err(trace_error("shifted-ternary source field has no final low-norm slot"));
            };
            if source_width != BALANCED_FIELD_WIDTH
                || digit_coordinates
                    .iter()
                    .enumerate()
                    .any(|(digit, &coordinate)| coordinate != source_coordinate + digit)
            {
                return Err(trace_error(&format!(
                    "arm {arm_index} shifted-ternary field {} uses slot ({source_coordinate}, {source_width}) \
                     but its digit coordinates are not that exact alias",
                    decomposition.field_col
                )));
            }
            for column in trace.negative_columns_start..trace.negative_columns_start + BALANCED_FIELD_WIDTH {
                if slots[arm_index][column].is_some() {
                    return Err(trace_error(
                        "shifted-ternary negative indicator survived selective lowering",
                    ));
                }
            }

            let mut borrow_coordinates = Vec::with_capacity(CANON_CHUNK_COUNT - 1);
            for index in 0..BALANCED_FIELD_WIDTH - 1 {
                let slot = slots[arm_index][trace.borrow_columns_start + index];
                if index % CANON_CHUNK_WIDTH == 0 {
                    if slot.is_some() {
                        return Err(trace_error(
                            "internal shifted-ternary borrow survived selective lowering",
                        ));
                    }
                } else {
                    let Some((coordinate, 1)) = slot else {
                        return Err(trace_error(
                            "shifted-ternary chunk endpoint does not own one final coordinate",
                        ));
                    };
                    borrow_coordinates.push(coordinate);
                }
            }
            if digit_coordinates.len() != BALANCED_FIELD_WIDTH
                || borrow_coordinates.len() != CANON_CHUNK_COUNT - 1
                || rewrite.emitted_rows().len() != CANON_CHUNK_COUNT
            {
                return Err(trace_error("shifted-ternary compact geometry drifted"));
            }
            let mut opening_coordinates = BTreeSet::new();
            for &coordinate in digit_coordinates.iter().chain(&borrow_coordinates) {
                if coordinate >= columns || !opening_coordinates.insert(coordinate) {
                    return Err(trace_error(
                        "one shifted-ternary opening overlaps itself or escapes the final assignment",
                    ));
                }
            }
            openings.push(SelectiveCanonicalOpeningAudit::new(
                decomposition.field_col,
                digit_coordinates,
                borrow_coordinates,
                rewrite.emitted_rows(),
            ));
        }
        result.push(openings);
    }
    Ok(result)
}

fn selective_arm_plan(
    arm: &SparseR1cs,
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    encoding: SelectiveEncoding,
) -> Result<SelectiveArmPlan, LowNormR1csError> {
    let shared_end = arm
        .m_in
        .checked_add(shared_private_fields)
        .filter(|&end| end <= arm.m)
        .ok_or_else(|| trace_error("shared private prefix exceeds the source arm"))?;
    let mut widths = vec![encoding.general_field_width(); arm.m];
    let mut centered = vec![false; arm.m];
    widths[0] = 0;
    let mut eliminated = vec![false; arm.m];
    for trace in arm.poseidon2_traces() {
        if trace.row_start >= trace.row_end || trace.row_end > arm.n {
            return Err(trace_error("Poseidon2 row range is outside the source arm"));
        }
        for &col in &trace.allocated_columns {
            if col == 0 || col >= arm.m {
                return Err(trace_error("Poseidon2 temporary column is outside the source arm"));
            }
            eliminated[col] = true;
        }
        for sbox in &trace.sboxes {
            eliminated[sbox.output_col] = false;
        }
        for &col in &trace.output_cols {
            eliminated[col] = false;
        }
    }
    for trace in arm.polynomial_evaluation_traces() {
        if trace.row_start >= trace.row_end || trace.row_end > arm.n {
            return Err(trace_error("polynomial-evaluation row range is outside the source arm"));
        }
        if trace.coefficient_cols.is_empty()
            || trace.coefficient_cols.len() != trace.power_cols.len()
            || trace.coefficient_cols.len() > D
        {
            return Err(trace_error(
                "polynomial-evaluation trace has invalid coefficient geometry",
            ));
        }
        for &col in &trace.allocated_columns {
            if col == 0 || col >= arm.m {
                return Err(trace_error("polynomial-evaluation temporary is outside the source arm"));
            }
            eliminated[col] = true;
        }
        for &col in &trace.output_cols {
            eliminated[col] = false;
        }
    }
    for trace in arm.product_sum_batch_traces() {
        if trace.row_start >= trace.row_end || trace.row_end > arm.n || trace.identities.is_empty() {
            return Err(trace_error("product-sum batch trace has invalid geometry"));
        }
        for &column in &trace.allocated_columns {
            if column == 0 || column >= arm.m {
                return Err(trace_error("product-sum temporary is outside the source arm"));
            }
            eliminated[column] = true;
        }
        for &column in &trace.retained_columns {
            eliminated[column] = false;
        }
    }
    for trace in arm.centered_unit_traces() {
        if trace.row_start >= trace.row_end || trace.row_end > arm.n {
            return Err(trace_error("centered-unit trace has invalid row geometry"));
        }
        for &column in &trace.allocated_columns {
            if column == 0 || column >= arm.m {
                return Err(trace_error("centered-unit temporary is outside the source arm"));
            }
            eliminated[column] = true;
        }
        eliminated[trace.value_col] = false;
    }
    for trace in arm.shifted_ternary_canonical_traces() {
        let digit_end = trace.digit_columns_start + BALANCED_FIELD_WIDTH;
        let negative_end = trace.negative_columns_start + BALANCED_FIELD_WIDTH;
        let borrow_end = trace.borrow_columns_start + BALANCED_FIELD_WIDTH - 1;
        if digit_end > arm.m || negative_end > arm.m || borrow_end > arm.m {
            return Err(trace_error("shifted-ternary trace columns exceed the source arm"));
        }
        for column in trace.negative_columns_start..negative_end {
            eliminated[column] = true;
        }
        for column in trace.digit_columns_start..digit_end {
            eliminated[column] = false;
        }
        for (index, column) in (trace.borrow_columns_start..borrow_end).enumerate() {
            // One retained endpoint serves each two-trit transition. Borrows
            // internal to a pair are exact polynomial substitutions.
            eliminated[column] = index % 2 == 0;
        }
    }
    for col in 1..arm.m {
        if eliminated[col] {
            widths[col] = 0;
        }
    }
    if eliminated[1..arm.m_in].iter().any(|&value| value) {
        return Err(trace_error("selective trace attempted to eliminate a public output"));
    }
    let skipped = skipped_selective_rows(arm)?;
    let definitions = find_linear_definitions(arm, shared_end, &eliminated, &skipped)?;
    for (column, definition) in definitions.by_column.iter().enumerate() {
        if definition.is_some() {
            widths[column] = 0;
        }
    }
    for decomposition in arm.canonical_u64_decompositions() {
        if widths[decomposition.field_col] != 0 {
            widths[decomposition.field_col] = BINARY_FIELD_WIDTH;
        }
        for &bit_col in &decomposition.bit_cols {
            if widths[bit_col] != 0 {
                widths[bit_col] = 1;
            }
        }
    }
    for decomposition in arm.balanced_ternary_decompositions() {
        if widths[decomposition.field_col] != 0 {
            widths[decomposition.field_col] = BALANCED_FIELD_WIDTH;
        }
        for &digit_col in &decomposition.digit_cols {
            if widths[digit_col] != 0 {
                widths[digit_col] = 1;
                centered[digit_col] = true;
            }
        }
    }
    for &column in arm.centered_unit_columns() {
        if widths[column] != 0 {
            widths[column] = 1;
            centered[column] = true;
        }
    }
    for &column in arm.boolean_columns() {
        if widths[column] != 0 {
            widths[column] = 1;
            centered[column] = false;
        }
    }
    // The public prefix and the leading part of the shared private prefix are
    // verifier-owned bit surfaces. Remaining shared fields retain the widths
    // inferred from their identical per-arm relations.
    let shared_bit_end = arm.m_in + shared_private_bit_fields;
    widths[1..arm.m_in].fill(1);
    widths[arm.m_in..shared_bit_end].fill(1);
    centered[..shared_bit_end].fill(false);
    let equality_roots = propagate_low_norm_equalities(arm, &mut widths, &mut centered, shared_bit_end, &skipped)?;
    for decomposition in arm.balanced_ternary_decompositions() {
        if decomposition.field_col >= shared_bit_end && widths[decomposition.field_col] != 0 {
            // The Ajtai word already needs these 41 coordinates. Keep them as
            // the authoritative field slot even when an equality class also
            // contains a bit, so the digit word aliases the field exactly.
            widths[decomposition.field_col] = BALANCED_FIELD_WIDTH;
        }
    }
    let mut removed_rows = skipped;
    for definition in &definitions.entries {
        if let Some(row) = definition.row {
            removed_rows[row] = true;
        }
    }
    let mut source_boolean_rows = vec![false; arm.m];
    for &(column, row) in arm.boolean_constraint_rows() {
        if column < arm.m && row < arm.n && !removed_rows[row] {
            source_boolean_rows[column] = true;
        }
    }
    Ok(SelectiveArmPlan {
        widths,
        centered,
        source_boolean_rows,
        equality_roots,
        definitions,
    })
}

fn propagate_low_norm_equalities(
    arm: &SparseR1cs,
    widths: &mut [usize],
    centered: &mut [bool],
    shared_end: usize,
    skipped: &[bool],
) -> Result<Vec<usize>, LowNormR1csError> {
    let mut parents = (0..arm.m).collect::<Vec<_>>();
    for &(row, lhs, rhs) in arm.equality_pairs() {
        if !skipped[row] {
            union(&mut parents, lhs, rhs);
        }
    }
    for column in 0..arm.m {
        let root = find(&mut parents, column);
        parents[column] = root;
    }

    let mut boolean = vec![false; arm.m];
    boolean[..shared_end].fill(true);
    for &column in arm.boolean_columns() {
        boolean[column] = true;
    }
    for decomposition in arm.canonical_u64_decompositions() {
        for &column in &decomposition.bit_cols {
            boolean[column] = true;
        }
    }
    let mut class_boolean = vec![false; arm.m];
    let mut class_centered = vec![false; arm.m];
    for column in 1..arm.m {
        if widths[column] == 0 {
            continue;
        }
        let root = parents[column];
        class_boolean[root] |= boolean[column];
        class_centered[root] |= centered[column];
    }
    for column in 1..arm.m {
        if widths[column] == 0 {
            continue;
        }
        let root = parents[column];
        if class_boolean[root] {
            widths[column] = 1;
            centered[column] = false;
        } else if class_centered[root] {
            widths[column] = 1;
            centered[column] = true;
        }
    }
    Ok(parents)
}

fn equality_aliases(
    arm: &SparseR1cs,
    plan: &SelectiveArmPlan,
    canonical_aliases: &[Option<(usize, usize)>],
    shared_private_fields: usize,
) -> Vec<Option<usize>> {
    let branch_start = arm.m_in + shared_private_fields;
    let mut representative = vec![None; arm.m];
    for column in 1..arm.m {
        if plan.widths[column] == 0 || canonical_aliases[column].is_some() {
            continue;
        }
        let root = plan.equality_roots[column];
        representative[root].get_or_insert(column);
    }

    let mut aliases = vec![None; arm.m];
    for column in branch_start..arm.m {
        if plan.widths[column] == 0 || canonical_aliases[column].is_some() {
            continue;
        }
        let Some(source) = representative[plan.equality_roots[column]] else {
            continue;
        };
        if source < column
            && plan.widths[source] == plan.widths[column]
            && plan.centered[source] == plan.centered[column]
        {
            aliases[column] = Some(source);
        }
    }
    aliases
}

fn find(parents: &mut [usize], mut value: usize) -> usize {
    while parents[value] != value {
        parents[value] = parents[parents[value]];
        value = parents[value];
    }
    value
}

fn union(parents: &mut [usize], lhs: usize, rhs: usize) {
    let lhs = find(parents, lhs);
    let rhs = find(parents, rhs);
    if lhs != rhs {
        let (root, child) = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
        parents[child] = root;
    }
}

fn validate_shared_shapes(
    arms: &[SparseR1cs],
    widths: &[&[usize]],
    public_fields: usize,
    shared_private_fields: usize,
) -> Result<(), LowNormR1csError> {
    for (arm_index, arm) in arms.iter().enumerate().skip(1) {
        if arm.m_in != public_fields {
            return Err(LowNormR1csError::ArmPublicInputArity {
                arm: arm_index,
                actual: arm.m_in,
                expected: public_fields,
            });
        }
    }
    for col in 1..public_fields {
        for arm_index in 1..arms.len() {
            if widths[arm_index][col] != widths[0][col] {
                return Err(LowNormR1csError::ArmFieldWidth {
                    arm: arm_index,
                    col,
                    actual: widths[arm_index][col],
                    expected: widths[0][col],
                });
            }
        }
    }
    for (arm_index, arm) in arms.iter().enumerate() {
        let private_fields = arm.m - arm.m_in;
        if private_fields < shared_private_fields {
            return Err(LowNormR1csError::ArmSharedPrefixTooLong {
                arm: arm_index,
                actual: private_fields,
                required: shared_private_fields,
            });
        }
    }
    for offset in 0..shared_private_fields {
        let expected = widths[0][arms[0].m_in + offset];
        for arm_index in 1..arms.len() {
            let col = arms[arm_index].m_in + offset;
            if widths[arm_index][col] != expected {
                return Err(LowNormR1csError::ArmFieldWidth {
                    arm: arm_index,
                    col,
                    actual: widths[arm_index][col],
                    expected,
                });
            }
        }
    }
    Ok(())
}

fn decomposition_aliases(
    arm: &SparseR1cs,
    widths: &[usize],
    shared_private_fields: usize,
) -> Vec<Option<(usize, usize)>> {
    let mut aliases = vec![None; arm.m];
    let shared_end = arm.m_in + shared_private_fields;
    for CanonicalU64Decomposition { field_col, bit_cols } in arm.canonical_u64_decompositions() {
        if *field_col == 0 || widths[*field_col] != BINARY_FIELD_WIDTH {
            continue;
        }
        if (arm.m_in..shared_end).contains(field_col) {
            continue;
        }
        let usable = bit_cols.iter().all(|&bit_col| {
            bit_col > *field_col
                && bit_col < arm.m
                && bit_col >= arm.m_in
                && !(arm.m_in..shared_end).contains(&bit_col)
                && widths[bit_col] == 1
                && aliases[bit_col].is_none()
        });
        if usable {
            for (bit, &bit_col) in bit_cols.iter().enumerate() {
                aliases[bit_col] = Some((*field_col, bit));
            }
        }
    }
    for BalancedTernaryDecomposition { field_col, digit_cols } in arm.balanced_ternary_decompositions() {
        if *field_col == 0 || widths[*field_col] != BALANCED_FIELD_WIDTH {
            continue;
        }
        let usable = digit_cols.iter().all(|&digit_col| {
            digit_col > *field_col
                && digit_col < arm.m
                && digit_col >= arm.m_in
                && widths[digit_col] == 1
                && aliases[digit_col].is_none()
        });
        if usable {
            for (digit, &digit_col) in digit_cols.iter().enumerate() {
                aliases[digit_col] = Some((*field_col, digit));
            }
        }
    }
    aliases
}

fn assign_slot(
    slots: &mut [Option<(usize, usize)>],
    widths: &[usize],
    aliases: &[Option<(usize, usize)>],
    equal_aliases: &[Option<usize>],
    field_col: usize,
    cursor: &mut usize,
) -> Result<(), LowNormR1csError> {
    if widths[field_col] == 0 {
        return Ok(());
    }
    if let Some(source) = equal_aliases[field_col] {
        slots[field_col] = slots[source];
        return Ok(());
    }
    if let Some((source, bit)) = aliases[field_col] {
        let (start, width) =
            slots[source].ok_or_else(|| trace_error("decomposition source slot does not precede its child"))?;
        if bit >= width {
            return Err(trace_error("decomposition child exceeds its source slot"));
        }
        slots[field_col] = Some((start + bit, 1));
    } else {
        slots[field_col] = Some((*cursor, widths[field_col]));
        *cursor += widths[field_col];
    }
    Ok(())
}
