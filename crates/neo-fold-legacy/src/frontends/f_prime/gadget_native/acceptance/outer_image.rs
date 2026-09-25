//! Read-only outer-image evidence for aggregate sampler acceptance.
//!
//! Owns: exact production-decoder images for the sixteen chunk bits, removed
//! linear-definition provenance, Boolean-row ownership, source/active row
//! placement, and lossless sparse row images read from materialized CCS.
//!
//! Does not own: the nine-row emitter, sampler semantics, selector-gated
//! materialization, or permission to remove any protocol obligation.
//!
//! Emits constraints: no. It consumes the exact decoder and row schedules
//! produced by the parent materializer.
//!
//! Authority boundary: generated row data is evidence only. Removed source
//! definitions are identified separately from emitted physical owner rows;
//! counts are derived from the records and never authorize acceptance.
//!
//! | Evidence branch | Mathematical obligation | Production owner | Lean owner |
//! |---|---|---|---|
//! | decoded bits | singleton or exact transitive `build_source_terms` image | `gadget_native::build_source_terms` | outer-image refinement open |
//! | removed definitions | each sparse image has exact source-row derivation provenance | source linear schedule | linear-substitution bridge open |
//! | Boolean owners | singleton pair/tail or emitted translated source row | coordinate/fallback emitters | `ChunkBitOuterImage` placement open |
//! | active rows | every chunk names its exact nine physical CCS rows | `acceptance::emit` | aggregate leaf placement open |

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::engine::r1cs_circuit::{R1csEncodingTrace, R1csSnapshot, Var};

use super::{
    geometry, validate_canonical_witness, validate_geometry, validate_rows, PRODUCT_AGGREGATE_ROWS_PER_CHUNK,
    ROOT_BINDING_ROWS_PER_CHUNK, TREE_BIT_PAIR_ROWS_PER_CHUNK, TREE_OUTPUTS_PER_CHUNK,
};
use crate::frontends::f_prime::gadget_native::coordinate_gates::{
    GadgetNativeBooleanFamily, GadgetNativeCoordinateGroupFamily, GadgetNativeCoordinateRowAudit,
    PlannedCoordinatePairing,
};
use crate::frontends::f_prime::gadget_native::gates::{gate, TraceGateBuilder};
use crate::frontends::f_prime::gadget_native::shared_slots::matrix_row;
use crate::frontends::f_prime::gadget_native::slots::{
    slot_terms, ValueEncoding, ValueSlot, GOLDILOCKS_CANONICALITY_PAIR_ROWS,
};
use crate::frontends::f_prime::gadget_native::source_allocation::visit_planned_source_slots;
use crate::frontends::f_prime::gadget_native::source_schedule::{SourceColumnDecision, ValidatedSourceSchedule};
use crate::frontends::f_prime::gadget_native::{
    build_source_terms, linear_difference, selection, visit_difference_terms, EncodedGadgetNativeR1cs,
    GadgetNativeError, GadgetNativePlan, SourceColumn, TOOM_COEFFICIENTS, TOOM_EVALUATIONS,
};

const SOURCE_INPUTS_PER_CHUNK: usize = 16;

/// Exact source-column decoder kind used by one acceptance input bit.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AggregateAcceptanceDecodedImage {
    Singleton { encoded_column: usize },
    SparseLinear { terms: Vec<(usize, F)> },
}

/// Exact physical row that owns Booleanity for one substituted input bit.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregateAcceptanceBooleanRowOwner {
    CoordinatePairLeft {
        encoded_row: usize,
        family: GadgetNativeBooleanFamily,
        paired_column: usize,
    },
    CoordinatePairRight {
        encoded_row: usize,
        family: GadgetNativeBooleanFamily,
        paired_column: usize,
    },
    CoordinateTail {
        encoded_row: usize,
        family: GadgetNativeBooleanFamily,
    },
    TranslatedSource {
        source_row: usize,
        encoded_row: usize,
    },
}

impl AggregateAcceptanceBooleanRowOwner {
    pub fn encoded_row(self) -> usize {
        match self {
            Self::CoordinatePairLeft { encoded_row, .. }
            | Self::CoordinatePairRight { encoded_row, .. }
            | Self::CoordinateTail { encoded_row, .. }
            | Self::TranslatedSource { encoded_row, .. } => encoded_row,
        }
    }
}

/// One acceptance bit after the production source decoder is fully expanded.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateAcceptanceBitOuterImage {
    pub source_column: usize,
    pub source_boolean_row: usize,
    pub decoded: AggregateAcceptanceDecodedImage,
    pub linear_definition_columns: Vec<usize>,
    pub boolean_owner: AggregateAcceptanceBooleanRowOwner,
}

/// One removed generic source definition in exact dependency order. Its
/// `source_row` is provenance for transitive substitution; it is not an
/// emitted physical enforcement row.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateAcceptanceLinearDefinitionAudit {
    pub source_column: usize,
    pub source_row: usize,
    pub terms: Vec<(usize, F)>,
}

/// Lossless sparse source-R1CS row read from the supplied source snapshot.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateAcceptanceSourceRowAudit {
    pub row: usize,
    pub a: Vec<(usize, F)>,
    pub b: Vec<(usize, F)>,
    pub c: Vec<(usize, F)>,
}

/// One nonempty matrix image in a physical gadget-native CCS row.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateAcceptanceMatrixRowAudit {
    pub matrix: usize,
    pub terms: Vec<(usize, F)>,
}

/// Lossless sparse image of one actual physical gadget-native CCS row.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateAcceptancePhysicalRowAudit {
    pub row: usize,
    pub matrices: Vec<AggregateAcceptanceMatrixRowAudit>,
}

/// Exact placement of one source chunk and its nine-row active replacement.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateAcceptanceChunkOuterImageAudit {
    pub source_rows: Range<usize>,
    pub source_accept_column: usize,
    pub source_inverse_column: usize,
    pub bits: [AggregateAcceptanceBitOuterImage; SOURCE_INPUTS_PER_CHUNK],
    pub encoded_accept: usize,
    pub encoded_outputs: Range<usize>,
    pub active_rows: Range<usize>,
}

/// Recursive-branch-compatible acceptance placement extracted from an actual
/// materialized gadget-native relation. This is evidence, not protocol
/// authority, and makes no fixed-selector claim.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateAcceptanceOuterImageAudit {
    pub source_columns: usize,
    pub source_row_count: usize,
    pub encoded_columns: usize,
    pub encoded_rows: usize,
    pub matrix_arity: usize,
    pub linear_definitions: Vec<AggregateAcceptanceLinearDefinitionAudit>,
    pub chunks: Vec<AggregateAcceptanceChunkOuterImageAudit>,
    pub source_rows: Vec<AggregateAcceptanceSourceRowAudit>,
    pub physical_rows: Vec<AggregateAcceptancePhysicalRowAudit>,
}

impl EncodedGadgetNativeR1cs {
    /// Export the exact aggregate-acceptance outer image from this already
    /// materialized relation. All decoded bit expressions come from the same
    /// `build_source_terms` result consumed by production row emission.
    #[doc(hidden)]
    pub fn aggregate_acceptance_outer_image_audit(
        &self,
        source: &R1csSnapshot,
        trace: &R1csEncodingTrace,
    ) -> Result<AggregateAcceptanceOuterImageAudit, GadgetNativeError> {
        if source.cols() != self.plan.source_columns.len()
            || trace.acceptance_chunks().len() != self.plan.acceptance_slots.chunks.len()
        {
            return Err(geometry(0, "outer-image source/plan census"));
        }
        let decoded = build_source_terms(&self.plan.source_columns)?;
        let bit_rows = canonical_boolean_source_rows(source, trace)?;
        let boolean_families = boolean_family_by_row(&self.plan)?;
        let mut definitions = BTreeMap::new();
        let mut source_row_ids = BTreeSet::new();
        let mut physical_row_ids = BTreeSet::new();
        let mut chunks = Vec::with_capacity(trace.acceptance_chunks().len());
        let empty_gadget_columns = vec![false; source.cols()];

        for (chunk, event) in trace.acceptance_chunks().iter().enumerate() {
            validate_geometry(source, chunk, event, &empty_gadget_columns)?;
            validate_rows(source, event)?;
            validate_canonical_witness(source, chunk, event)?;
            let slots = self
                .plan
                .acceptance_slots
                .chunks
                .get(chunk)
                .ok_or_else(|| geometry(chunk, "outer-image chunk slot"))?;
            if slots.inverse_column != event.inverse.col()
                || slots.active_rows.len()
                    != TREE_BIT_PAIR_ROWS_PER_CHUNK + PRODUCT_AGGREGATE_ROWS_PER_CHUNK + ROOT_BINDING_ROWS_PER_CHUNK
                || slots.active_rows.end > self.structure.n
            {
                return Err(geometry(chunk, "outer-image active placement"));
            }
            source_row_ids.extend(event.source_rows.clone());
            physical_row_ids.extend(slots.active_rows.clone());

            let mut bits = Vec::with_capacity(SOURCE_INPUTS_PER_CHUNK);
            for variable in event.chunk_bits {
                let source_column = variable.col();
                let source_row = bit_rows
                    .get(&source_column)
                    .copied()
                    .ok_or_else(|| geometry(chunk, "unique source Boolean row for input bit"))?;
                source_row_ids.insert(source_row);
                let terms = normalize_terms(decoded[source_column].as_deref().ok_or(
                    GadgetNativeError::MissingDecodedColumn {
                        column: source_column,
                        row: event.source_rows.start,
                    },
                )?);
                let (decoded_image, linear_definition_columns, boolean_owner) = match &self.plan.source_columns
                    [source_column]
                {
                    SourceColumn::Encoded(slot)
                        if slot.width == 1
                            && slot.encoding == ValueEncoding::Boolean
                            && terms == [(slot.start, F::ONE)] =>
                    {
                        if translated_boolean_row(&self.plan, source_row).is_some() {
                            return Err(geometry(chunk, "singleton Boolean source row was not deduplicated"));
                        }
                        let owner = coordinate_boolean_owner(&self.plan, &boolean_families, slot.start, chunk)?;
                        validate_coordinate_boolean_owner(&self.structure.matrices, owner, slot.start, chunk)?;
                        (
                            AggregateAcceptanceDecodedImage::Singleton {
                                encoded_column: slot.start,
                            },
                            Vec::new(),
                            owner,
                        )
                    }
                    SourceColumn::Linear(_) => {
                        let mut closure = BTreeSet::new();
                        collect_linear_definitions(
                            source,
                            &self.plan,
                            source_column,
                            chunk,
                            &mut closure,
                            &mut definitions,
                        )?;
                        for definition_column in &closure {
                            let SourceColumn::Linear(definition) = &self.plan.source_columns[*definition_column] else {
                                return Err(geometry(chunk, "linear definition provenance kind"));
                            };
                            let definition_row = definition
                                .source_row
                                .ok_or_else(|| geometry(chunk, "removed linear definition source row"))?;
                            source_row_ids.insert(definition_row);
                        }
                        let encoded_row = translated_boolean_row(&self.plan, source_row)
                            .ok_or_else(|| geometry(chunk, "translated Boolean source-row placement"))?;
                        validate_translated_boolean_row(&self.structure.matrices, encoded_row, &terms, chunk)?;
                        (
                            AggregateAcceptanceDecodedImage::SparseLinear { terms },
                            closure.into_iter().collect(),
                            AggregateAcceptanceBooleanRowOwner::TranslatedSource {
                                source_row,
                                encoded_row,
                            },
                        )
                    }
                    _ => return Err(geometry(chunk, "acceptance bit decoder kind")),
                };
                physical_row_ids.insert(boolean_owner.encoded_row());
                bits.push(AggregateAcceptanceBitOuterImage {
                    source_column,
                    source_boolean_row: source_row,
                    decoded: decoded_image,
                    linear_definition_columns,
                    boolean_owner,
                });
            }

            chunks.push(AggregateAcceptanceChunkOuterImageAudit {
                source_rows: event.source_rows.clone(),
                source_accept_column: event.accept.col(),
                source_inverse_column: event.inverse.col(),
                bits: bits
                    .try_into()
                    .map_err(|_| geometry(chunk, "sixteen outer-image bits"))?,
                encoded_accept: slots.accept.start,
                encoded_outputs: slots.outputs[0].start..slots.outputs[0].start + TREE_OUTPUTS_PER_CHUNK,
                active_rows: slots.active_rows.clone(),
            });
        }

        let source_rows = source_row_ids
            .into_iter()
            .map(|row| AggregateAcceptanceSourceRowAudit {
                row,
                a: source.a_row(row).to_vec(),
                b: source.b_row(row).to_vec(),
                c: source.c_row(row).to_vec(),
            })
            .collect();
        let physical_rows = physical_row_ids
            .into_iter()
            .map(|row| physical_row_audit(&self.structure.matrices, self.structure.n, row))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AggregateAcceptanceOuterImageAudit {
            source_columns: source.cols(),
            source_row_count: source.rows(),
            encoded_columns: self.structure.m,
            encoded_rows: self.structure.n,
            matrix_arity: self.structure.matrices.len(),
            linear_definitions: definitions.into_values().collect(),
            chunks,
            source_rows,
            physical_rows,
        })
    }
}

/// Extract the same aggregate-acceptance outer image without materializing the
/// complete encoded witness or any CCS matrix. The source schedule, allocation
/// cursor, coordinate-pairing plan, fallback-row partition, and the existing
/// nine-row emitter are all replayed directly from the validated production
/// source and trace.
#[doc(hidden)]
pub fn audit_r1cs_gadget_native_aggregate_acceptance_outer_image(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    public_bit_columns: &[usize],
) -> Result<AggregateAcceptanceOuterImageAudit, GadgetNativeError> {
    let schedule = ValidatedSourceSchedule::checked(source, trace, public_bit_columns)?;
    let linearly_derived = schedule
        .decisions()
        .iter()
        .map(|decision| matches!(decision, SourceColumnDecision::GenericLinear(_)))
        .collect::<Vec<_>>();
    let pairing = PlannedCoordinatePairing::checked(source, trace, &schedule, &linearly_derived)?;
    let estimate = crate::frontends::f_prime::gadget_native::estimate_r1cs_gadget_native_from_schedule(
        source,
        trace,
        public_bit_columns,
        &schedule,
    )?;
    if pairing.total_rows() != estimate.boolean_pairing.total_rows() + estimate.centered_pairing.total_rows()
        || estimate.acceptance_chunks != trace.acceptance_chunks().len()
    {
        return Err(geometry(0, "selective outer-image estimate reconciliation"));
    }

    let requested_bits = trace
        .acceptance_chunks()
        .iter()
        .flat_map(|event| event.chunk_bits)
        .map(Var::col)
        .collect::<BTreeSet<_>>();
    let requested_accepts = trace
        .acceptance_chunks()
        .iter()
        .map(|event| event.accept.col())
        .collect::<BTreeSet<_>>();
    let dependencies = selective_dependency_closure(&schedule, &requested_bits)?;
    let singleton_bits = requested_bits
        .iter()
        .copied()
        .filter(|&column| !matches!(schedule.decisions()[column], SourceColumnDecision::GenericLinear(_)))
        .collect::<BTreeSet<_>>();
    let (source_phase_end, source_slots, common_owners) = planned_slots_and_common_owners(
        trace,
        public_bit_columns,
        &schedule,
        &pairing,
        &dependencies,
        &requested_accepts,
        &singleton_bits,
    )?;
    let decoded = decode_selective_columns(&schedule, &dependencies, &source_slots)?;
    let bit_rows = canonical_boolean_source_rows(source, trace)?;

    let translated_source_rows = requested_bits
        .iter()
        .copied()
        .filter(|&column| matches!(schedule.decisions()[column], SourceColumnDecision::GenericLinear(_)))
        .map(|column| {
            bit_rows
                .get(&column)
                .copied()
                .ok_or_else(|| geometry(0, "translated Boolean source-row owner"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let translated_rows = planned_translated_boolean_rows(
        source,
        &schedule,
        &linearly_derived,
        &pairing,
        &estimate,
        &translated_source_rows,
    )?;
    let acceptance_row_start = planned_acceptance_row_start(trace, &schedule, &pairing, &estimate)?;

    let mut acceptance_slots = planned_acceptance_slots(trace, source_phase_end, &source_slots)?;
    let mut gates = TraceGateBuilder::new();
    super::emit_with_decoded_lookup(trace, &mut acceptance_slots, &mut gates, &mut |column, row| {
        decoded
            .get(&column)
            .cloned()
            .ok_or(GadgetNativeError::MissingDecodedColumn { column, row })
    })?;
    let active_row_count = gates.rows;
    if active_row_count
        != estimate.acceptance_tree_bit_pair_rows
            + estimate.acceptance_product_aggregate_rows
            + estimate.acceptance_root_binding_rows
    {
        return Err(geometry(0, "selective active-row census"));
    }
    let active_sparse_rows = gates.into_sparse_rows();

    let mut definitions = BTreeMap::<usize, AggregateAcceptanceLinearDefinitionAudit>::new();
    let mut source_row_ids = BTreeSet::new();
    let mut physical_rows = BTreeMap::<usize, AggregateAcceptancePhysicalRowAudit>::new();
    for (relative_row, matrices) in active_sparse_rows.into_iter().enumerate() {
        insert_physical_row(
            &mut physical_rows,
            AggregateAcceptancePhysicalRowAudit {
                row: acceptance_row_start + relative_row,
                matrices: matrices
                    .into_iter()
                    .map(|(matrix, terms)| AggregateAcceptanceMatrixRowAudit { matrix, terms })
                    .collect(),
            },
        )?;
    }

    let mut chunks = Vec::with_capacity(trace.acceptance_chunks().len());
    for (chunk, event) in trace.acceptance_chunks().iter().enumerate() {
        source_row_ids.extend(event.source_rows.clone());
        let slots = acceptance_slots
            .chunks
            .get(chunk)
            .ok_or_else(|| geometry(chunk, "selective acceptance slot"))?;
        let mut bits = Vec::with_capacity(SOURCE_INPUTS_PER_CHUNK);
        for bit in event.chunk_bits {
            let source_column = bit.col();
            let source_row = bit_rows
                .get(&source_column)
                .copied()
                .ok_or_else(|| geometry(chunk, "selective Boolean source row"))?;
            source_row_ids.insert(source_row);
            let terms = decoded
                .get(&source_column)
                .cloned()
                .ok_or(GadgetNativeError::MissingDecodedColumn {
                    column: source_column,
                    row: event.source_rows.start,
                })?;
            let (decoded_image, linear_definition_columns, boolean_owner) = if matches!(
                schedule.decisions()[source_column],
                SourceColumnDecision::GenericLinear(_)
            ) {
                let mut closure = BTreeSet::new();
                collect_selective_linear_definitions(
                    source,
                    &schedule,
                    source_column,
                    chunk,
                    &mut closure,
                    &mut definitions,
                )?;
                for definition_column in &closure {
                    let SourceColumnDecision::GenericLinear(definition) = &schedule.decisions()[*definition_column]
                    else {
                        return Err(geometry(chunk, "selective linear-definition kind"));
                    };
                    source_row_ids.insert(
                        definition
                            .source_row
                            .ok_or_else(|| geometry(chunk, "selective linear-definition row"))?,
                    );
                }
                let encoded_row = translated_rows
                    .get(&source_row)
                    .copied()
                    .ok_or_else(|| geometry(chunk, "selective translated-row placement"))?;
                (
                    AggregateAcceptanceDecodedImage::SparseLinear { terms: terms.clone() },
                    closure.into_iter().collect(),
                    AggregateAcceptanceBooleanRowOwner::TranslatedSource {
                        source_row,
                        encoded_row,
                    },
                )
            } else {
                let slot = source_slots
                    .get(&source_column)
                    .copied()
                    .ok_or_else(|| geometry(chunk, "selective singleton source slot"))?;
                if slot.width != 1 || slot.encoding != ValueEncoding::Boolean || terms != [(slot.start, F::ONE)] {
                    return Err(geometry(chunk, "selective singleton decoder"));
                }
                (
                    AggregateAcceptanceDecodedImage::Singleton {
                        encoded_column: slot.start,
                    },
                    Vec::new(),
                    common_owners
                        .get(&source_column)
                        .copied()
                        .ok_or_else(|| geometry(chunk, "selective common Boolean owner"))?,
                )
            };
            insert_physical_row(&mut physical_rows, planned_boolean_owner_row(boolean_owner, &terms)?)?;
            bits.push(AggregateAcceptanceBitOuterImage {
                source_column,
                source_boolean_row: source_row,
                decoded: decoded_image,
                linear_definition_columns,
                boolean_owner,
            });
        }

        chunks.push(AggregateAcceptanceChunkOuterImageAudit {
            source_rows: event.source_rows.clone(),
            source_accept_column: event.accept.col(),
            source_inverse_column: event.inverse.col(),
            bits: bits
                .try_into()
                .map_err(|_| geometry(chunk, "sixteen selective outer-image bits"))?,
            encoded_accept: slots.accept.start,
            encoded_outputs: slots.outputs[0].start..slots.outputs[0].start + TREE_OUTPUTS_PER_CHUNK,
            active_rows: acceptance_row_start + slots.active_rows.start..acceptance_row_start + slots.active_rows.end,
        });
    }

    let source_rows = source_row_ids
        .into_iter()
        .map(|row| AggregateAcceptanceSourceRowAudit {
            row,
            a: source.a_row(row).to_vec(),
            b: source.b_row(row).to_vec(),
            c: source.c_row(row).to_vec(),
        })
        .collect();
    Ok(AggregateAcceptanceOuterImageAudit {
        source_columns: source.cols(),
        source_row_count: source.rows(),
        encoded_columns: estimate.encoded_cols,
        encoded_rows: estimate.encoded_rows,
        matrix_arity: gate::ARITY,
        linear_definitions: definitions.into_values().collect(),
        chunks,
        source_rows,
        physical_rows: physical_rows.into_values().collect(),
    })
}

fn selective_dependency_closure(
    schedule: &ValidatedSourceSchedule,
    roots: &BTreeSet<usize>,
) -> Result<BTreeSet<usize>, GadgetNativeError> {
    let mut closure = BTreeSet::new();
    let mut stack = roots.iter().copied().collect::<Vec<_>>();
    while let Some(column) = stack.pop() {
        if column >= schedule.decisions().len() || !closure.insert(column) {
            continue;
        }
        match &schedule.decisions()[column] {
            SourceColumnDecision::GenericLinear(definition) => {
                for &(input, _) in &definition.terms {
                    if input >= column {
                        return Err(GadgetNativeError::NonTopologicalDefinition { column });
                    }
                    stack.push(input);
                }
            }
            SourceColumnDecision::BalancedDigitAlias { field, .. } => {
                if *field >= column {
                    return Err(GadgetNativeError::NonTopologicalDefinition { column });
                }
                stack.push(*field);
            }
            SourceColumnDecision::Projected(_) => {
                return Err(geometry(0, "acceptance bit uses an unavailable projected decoder"));
            }
            _ => {}
        }
    }
    closure.insert(0);
    Ok(closure)
}

#[allow(clippy::too_many_arguments)]
fn planned_slots_and_common_owners(
    trace: &R1csEncodingTrace,
    public_bit_columns: &[usize],
    schedule: &ValidatedSourceSchedule,
    pairing: &PlannedCoordinatePairing,
    dependencies: &BTreeSet<usize>,
    requested_accepts: &BTreeSet<usize>,
    singleton_bits: &BTreeSet<usize>,
) -> Result<
    (
        usize,
        BTreeMap<usize, ValueSlot>,
        BTreeMap<usize, AggregateAcceptanceBooleanRowOwner>,
    ),
    GadgetNativeError,
> {
    let specialized = crate::frontends::f_prime::gadget_native::coordinate_gates::specialized_boolean_columns(
        trace,
        schedule.decisions().len(),
    );
    let mut slots = BTreeMap::new();
    let mut owners = BTreeMap::new();
    let mut pending = vec![None::<(usize, usize)>; pairing.stage_count()];
    let mut common_counts = vec![0usize; pairing.stage_count()];
    let source_phase_end = visit_planned_source_slots(schedule, public_bit_columns, |column, slot| {
        if dependencies.contains(&column) || requested_accepts.contains(&column) {
            slots.insert(column, slot);
        }
        if slot.encoding != ValueEncoding::Boolean
            || schedule.marks.balanced_ternary.is_binary(column)
            || specialized[column]
        {
            return Ok(());
        }
        let stage = pairing.source_stage(column)?;
        let ordinal = common_counts[stage];
        common_counts[stage] += 1;
        if let Some((left_source, left_encoded)) = pending[stage].take() {
            let row = pairing.stage_row_start(stage) + ordinal / 2;
            if singleton_bits.contains(&left_source) {
                owners.insert(
                    left_source,
                    AggregateAcceptanceBooleanRowOwner::CoordinatePairLeft {
                        encoded_row: row,
                        family: GadgetNativeBooleanFamily::Common,
                        paired_column: slot.start,
                    },
                );
            }
            if singleton_bits.contains(&column) {
                owners.insert(
                    column,
                    AggregateAcceptanceBooleanRowOwner::CoordinatePairRight {
                        encoded_row: row,
                        family: GadgetNativeBooleanFamily::Common,
                        paired_column: left_encoded,
                    },
                );
            }
        } else {
            pending[stage] = Some((column, slot.start));
        }
        Ok(())
    })?;
    for stage in 0..pairing.stage_count() {
        if common_counts[stage] != pairing.stage_boolean(stage).common.coordinates {
            return Err(geometry(0, "selective common-coordinate census"));
        }
        if let Some((source_column, encoded_column)) = pending[stage] {
            if singleton_bits.contains(&source_column) {
                owners.insert(
                    source_column,
                    AggregateAcceptanceBooleanRowOwner::CoordinateTail {
                        encoded_row: pairing.stage_row_start(stage) + common_counts[stage] / 2,
                        family: GadgetNativeBooleanFamily::Common,
                    },
                );
            }
            if encoded_column == 0 {
                return Err(geometry(0, "selective common-coordinate tail"));
            }
        }
    }
    if owners.len() != singleton_bits.len() {
        return Err(geometry(0, "selective singleton-owner census"));
    }
    Ok((source_phase_end, slots, owners))
}

fn decode_selective_columns(
    schedule: &ValidatedSourceSchedule,
    dependencies: &BTreeSet<usize>,
    slots: &BTreeMap<usize, ValueSlot>,
) -> Result<BTreeMap<usize, Vec<(usize, F)>>, GadgetNativeError> {
    let mut decoded = BTreeMap::<usize, Vec<(usize, F)>>::new();
    for &column in dependencies {
        let terms = match &schedule.decisions()[column] {
            SourceColumnDecision::ConstantOne => vec![(0, F::ONE)],
            SourceColumnDecision::PublicBit
            | SourceColumnDecision::PrivateBoolean(_)
            | SourceColumnDecision::BalancedOpening { .. }
            | SourceColumnDecision::BalancedDigitAlias { .. }
            | SourceColumnDecision::CanonicalField(_) => slot_terms(
                *slots
                    .get(&column)
                    .ok_or_else(|| geometry(0, "selective source slot"))?,
            ),
            SourceColumnDecision::GenericLinear(definition) => {
                let mut combined = Vec::new();
                for &(input, scale) in &definition.terms {
                    let input_terms = decoded
                        .get(&input)
                        .ok_or(GadgetNativeError::MissingDecodedColumn { column: input, row: 0 })?;
                    combined.extend(
                        input_terms
                            .iter()
                            .map(|&(encoded_column, coefficient)| (encoded_column, scale * coefficient)),
                    );
                }
                normalize_terms(&combined)
            }
            SourceColumnDecision::Projected(_) => {
                return Err(geometry(0, "selective projected decoder"));
            }
        };
        decoded.insert(column, terms);
    }
    Ok(decoded)
}

fn collect_selective_linear_definitions(
    source: &R1csSnapshot,
    schedule: &ValidatedSourceSchedule,
    column: usize,
    chunk: usize,
    closure: &mut BTreeSet<usize>,
    definitions: &mut BTreeMap<usize, AggregateAcceptanceLinearDefinitionAudit>,
) -> Result<(), GadgetNativeError> {
    if !closure.insert(column) {
        return Ok(());
    }
    let SourceColumnDecision::GenericLinear(definition) = &schedule.decisions()[column] else {
        return Err(geometry(chunk, "selective linear-definition decoder kind"));
    };
    let row = definition
        .source_row
        .ok_or_else(|| geometry(chunk, "selective removed linear-definition row"))?;
    let terms = normalize_terms(&definition.terms);
    validate_linear_definition(source, row, column, &terms, chunk)?;
    for &(input, _) in &terms {
        if matches!(schedule.decisions()[input], SourceColumnDecision::GenericLinear(_)) {
            collect_selective_linear_definitions(source, schedule, input, chunk, closure, definitions)?;
        }
    }
    let audit = AggregateAcceptanceLinearDefinitionAudit {
        source_column: column,
        source_row: row,
        terms,
    };
    if let Some(existing) = definitions.get(&column) {
        if existing != &audit {
            return Err(geometry(chunk, "inconsistent selective linear provenance"));
        }
    } else {
        definitions.insert(column, audit);
    }
    Ok(())
}

fn planned_translated_boolean_rows(
    source: &R1csSnapshot,
    schedule: &ValidatedSourceSchedule,
    linearly_derived: &[bool],
    pairing: &PlannedCoordinatePairing,
    estimate: &crate::frontends::f_prime::gadget_native::GadgetNativeEstimate,
    requested: &BTreeSet<usize>,
) -> Result<BTreeMap<usize, usize>, GadgetNativeError> {
    let redundant = crate::frontends::f_prime::gadget_native::boolean_dedup::ExactBooleanRows::from_plan(
        source,
        &schedule.is_public,
        &schedule.explicit_bits,
        linearly_derived,
        &schedule.marks,
    );
    let removed = schedule
        .marks
        .balanced_ternary
        .reduction_removed_rows(&schedule.removed_definition_rows, redundant.rows())?;
    let canonical_fields = estimate.canonical_binary_field_source_cols
        + estimate.synthetic_ring_fields
        + estimate.synthetic_product_sum_fields;
    let fallback_start = pairing.total_rows() + canonical_fields * GOLDILOCKS_CANONICALITY_PAIR_ROWS;
    let mut retained = 0usize;
    let mut placements = BTreeMap::new();
    for row in 0..source.rows() {
        if schedule.marks.covered_rows[row] || removed[row] || redundant.rows()[row] {
            continue;
        }
        if requested.contains(&row) {
            placements.insert(row, fallback_start + retained);
        }
        retained += 1;
    }
    if retained != estimate.fallback_source_rows || placements.len() != requested.len() {
        return Err(geometry(0, "selective translated-row census"));
    }
    Ok(placements)
}

fn planned_acceptance_row_start(
    trace: &R1csEncodingTrace,
    schedule: &ValidatedSourceSchedule,
    pairing: &PlannedCoordinatePairing,
    estimate: &crate::frontends::f_prime::gadget_native::GadgetNativeEstimate,
) -> Result<usize, GadgetNativeError> {
    let canonical_fields = estimate.canonical_binary_field_source_cols
        + estimate.synthetic_ring_fields
        + estimate.synthetic_product_sum_fields;
    let k_mul_rows = trace
        .k_muls()
        .iter()
        .enumerate()
        .filter(|(index, _)| !schedule.marks.product_sums.is_nested_k_mul(*index))
        .count()
        * 2;
    let selection_rows = trace
        .first_accepted_selections()
        .iter()
        .map(selection::encoded_rows)
        .sum::<usize>();
    let start = pairing.total_rows()
        + canonical_fields * GOLDILOCKS_CANONICALITY_PAIR_ROWS
        + estimate.fallback_source_rows
        + trace.sbox7().len()
        + k_mul_rows
        + schedule.marks.product_sums.encoded_rows()
        + trace.ring_muls_toom3().len() * (TOOM_EVALUATIONS * TOOM_COEFFICIENTS + 54)
        + selection_rows;
    let acceptance_rows = estimate.acceptance_tree_bit_pair_rows
        + estimate.acceptance_product_aggregate_rows
        + estimate.acceptance_root_binding_rows;
    let mod5_rows = estimate.packed_mod5_low_bit_pair_rows
        + estimate.packed_mod5_high_bit_pair_rows
        + estimate.packed_mod5_residue_pair_rows;
    if start + acceptance_rows + mod5_rows != estimate.encoded_rows {
        return Err(geometry(0, "selective physical-row partition"));
    }
    Ok(start)
}

fn planned_acceptance_slots(
    trace: &R1csEncodingTrace,
    source_phase_end: usize,
    source_slots: &BTreeMap<usize, ValueSlot>,
) -> Result<super::AcceptanceSlots, GadgetNativeError> {
    let mut next_output = source_phase_end;
    let mut chunks = Vec::with_capacity(trace.acceptance_chunks().len());
    for (chunk, event) in trace.acceptance_chunks().iter().enumerate() {
        let accept = source_slots
            .get(&event.accept.col())
            .copied()
            .ok_or_else(|| geometry(chunk, "planned accept slot"))?;
        if accept.width != 1 || accept.encoding != ValueEncoding::Boolean {
            return Err(geometry(chunk, "planned accept encoding"));
        }
        let outputs = std::array::from_fn(|offset| ValueSlot {
            start: next_output + offset,
            width: 1,
            encoding: ValueEncoding::Boolean,
        });
        next_output += TREE_OUTPUTS_PER_CHUNK;
        chunks.push(super::AcceptanceChunkSlots {
            accept,
            outputs,
            inverse_column: event.inverse.col(),
            active_rows: 0..0,
        });
    }
    Ok(super::AcceptanceSlots {
        chunks,
        omitted_coordinates: Vec::new(),
    })
}

fn planned_boolean_owner_row(
    owner: AggregateAcceptanceBooleanRowOwner,
    decoded_terms: &[(usize, F)],
) -> Result<AggregateAcceptancePhysicalRowAudit, GadgetNativeError> {
    let matrices = match owner {
        AggregateAcceptanceBooleanRowOwner::CoordinatePairLeft { paired_column, .. } => vec![
            AggregateAcceptanceMatrixRowAudit {
                matrix: gate::SELECTOR,
                terms: vec![(0, F::ONE)],
            },
            AggregateAcceptanceMatrixRowAudit {
                matrix: gate::QUADRATIC_BIT_LEFT,
                terms: decoded_terms.to_vec(),
            },
            AggregateAcceptanceMatrixRowAudit {
                matrix: gate::QUADRATIC_BIT_RIGHT,
                terms: vec![(paired_column, F::ONE)],
            },
        ],
        AggregateAcceptanceBooleanRowOwner::CoordinatePairRight { paired_column, .. } => vec![
            AggregateAcceptanceMatrixRowAudit {
                matrix: gate::SELECTOR,
                terms: vec![(0, F::ONE)],
            },
            AggregateAcceptanceMatrixRowAudit {
                matrix: gate::QUADRATIC_BIT_LEFT,
                terms: vec![(paired_column, F::ONE)],
            },
            AggregateAcceptanceMatrixRowAudit {
                matrix: gate::QUADRATIC_BIT_RIGHT,
                terms: decoded_terms.to_vec(),
            },
        ],
        AggregateAcceptanceBooleanRowOwner::CoordinateTail { .. } => vec![
            AggregateAcceptanceMatrixRowAudit {
                matrix: gate::SELECTOR,
                terms: vec![(0, F::ONE)],
            },
            AggregateAcceptanceMatrixRowAudit {
                matrix: gate::BITNESS,
                terms: decoded_terms.to_vec(),
            },
        ],
        AggregateAcceptanceBooleanRowOwner::TranslatedSource { .. } => {
            let mut minus_one = decoded_terms.to_vec();
            minus_one.push((0, -F::ONE));
            vec![
                AggregateAcceptanceMatrixRowAudit {
                    matrix: gate::SELECTOR,
                    terms: vec![(0, F::ONE)],
                },
                AggregateAcceptanceMatrixRowAudit {
                    matrix: gate::PRODUCT_LEFT,
                    terms: decoded_terms.to_vec(),
                },
                AggregateAcceptanceMatrixRowAudit {
                    matrix: gate::PRODUCT_RIGHT,
                    terms: normalize_terms(&minus_one),
                },
            ]
        }
    };
    Ok(AggregateAcceptancePhysicalRowAudit {
        row: owner.encoded_row(),
        matrices,
    })
}

fn insert_physical_row(
    rows: &mut BTreeMap<usize, AggregateAcceptancePhysicalRowAudit>,
    row: AggregateAcceptancePhysicalRowAudit,
) -> Result<(), GadgetNativeError> {
    if let Some(existing) = rows.get(&row.row) {
        if existing != &row {
            return Err(geometry(0, "inconsistent selective physical row"));
        }
    } else {
        rows.insert(row.row, row);
    }
    Ok(())
}

fn normalize_terms(terms: &[(usize, F)]) -> Vec<(usize, F)> {
    let mut normalized = BTreeMap::new();
    for &(column, coefficient) in terms {
        *normalized.entry(column).or_insert(F::ZERO) += coefficient;
    }
    normalized
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect()
}

fn canonical_boolean_source_rows(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
) -> Result<BTreeMap<usize, usize>, GadgetNativeError> {
    let requested = trace
        .acceptance_chunks()
        .iter()
        .flat_map(|event| event.chunk_bits)
        .map(Var::col)
        .collect::<BTreeSet<_>>();
    let mut rows = BTreeMap::new();
    for decomposition in trace.canonical_u64_decompositions() {
        for (offset, bit) in decomposition.bits.iter().enumerate() {
            let column = bit.col();
            if !requested.contains(&column) {
                continue;
            }
            let row = decomposition.source_rows.start + offset;
            if row >= source.rows()
                || crate::frontends::f_prime::gadget_native::boolean_dedup::exact_bit_row_column(source, row)
                    != Some(column)
                || rows.insert(column, row).is_some()
            {
                return Err(geometry(0, "canonical source Boolean row for input bit"));
            }
        }
    }
    if rows.len() != requested.len() {
        return Err(geometry(0, "canonical source Boolean row census"));
    }
    Ok(rows)
}

pub(in crate::frontends::f_prime::gadget_native) fn translated_boolean_source_rows(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    source_columns: &[SourceColumn],
) -> Result<Vec<usize>, GadgetNativeError> {
    let bit_rows = canonical_boolean_source_rows(source, trace)?;
    let mut rows = Vec::new();
    for (chunk, event) in trace.acceptance_chunks().iter().enumerate() {
        for bit in event.chunk_bits {
            if matches!(source_columns.get(bit.col()), Some(SourceColumn::Linear(_))) {
                rows.push(
                    bit_rows
                        .get(&bit.col())
                        .copied()
                        .ok_or_else(|| geometry(chunk, "canonical source Boolean row for translated input bit"))?,
                );
            }
        }
    }
    rows.sort_unstable();
    if rows.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(geometry(0, "duplicate translated Boolean source row"));
    }
    Ok(rows)
}

fn boolean_family_by_row(plan: &GadgetNativePlan) -> Result<Vec<Option<GadgetNativeBooleanFamily>>, GadgetNativeError> {
    let schedule = &plan.coordinate_gates;
    let mut families = vec![None; schedule.rows().len()];
    for group in schedule.groups() {
        let GadgetNativeCoordinateGroupFamily::Boolean(family) = group.family else {
            continue;
        };
        for row in group.encoded_rows.clone() {
            if row >= families.len() || families[row].replace(family).is_some() {
                return Err(geometry(0, "Boolean coordinate-family row partition"));
            }
        }
    }
    Ok(families)
}

fn coordinate_boolean_owner(
    plan: &GadgetNativePlan,
    families: &[Option<GadgetNativeBooleanFamily>],
    column: usize,
    chunk: usize,
) -> Result<AggregateAcceptanceBooleanRowOwner, GadgetNativeError> {
    let row = plan
        .coordinate_gates
        .row_for_column(column)
        .ok_or_else(|| geometry(chunk, "singleton Boolean coordinate row"))?;
    let family = families
        .get(row)
        .copied()
        .flatten()
        .ok_or_else(|| geometry(chunk, "singleton Boolean coordinate family"))?;
    match plan.coordinate_gates.rows().get(row).copied() {
        Some(GadgetNativeCoordinateRowAudit::BooleanPair { left, right, .. }) if left == column => {
            Ok(AggregateAcceptanceBooleanRowOwner::CoordinatePairLeft {
                encoded_row: row,
                family,
                paired_column: right,
            })
        }
        Some(GadgetNativeCoordinateRowAudit::BooleanPair { left, right, .. }) if right == column => {
            Ok(AggregateAcceptanceBooleanRowOwner::CoordinatePairRight {
                encoded_row: row,
                family,
                paired_column: left,
            })
        }
        Some(GadgetNativeCoordinateRowAudit::BooleanTail { coordinate, .. }) if coordinate == column => {
            Ok(AggregateAcceptanceBooleanRowOwner::CoordinateTail {
                encoded_row: row,
                family,
            })
        }
        _ => Err(geometry(chunk, "singleton Boolean coordinate owner")),
    }
}

fn translated_boolean_row(plan: &GadgetNativePlan, source_row: usize) -> Option<usize> {
    plan.acceptance_translated_boolean_rows
        .binary_search_by_key(&source_row, |&(source, _)| source)
        .ok()
        .map(|index| plan.acceptance_translated_boolean_rows[index].1)
}

fn collect_linear_definitions(
    source: &R1csSnapshot,
    plan: &GadgetNativePlan,
    column: usize,
    chunk: usize,
    closure: &mut BTreeSet<usize>,
    definitions: &mut BTreeMap<usize, AggregateAcceptanceLinearDefinitionAudit>,
) -> Result<(), GadgetNativeError> {
    if !closure.insert(column) {
        return Ok(());
    }
    let SourceColumn::Linear(definition) = plan
        .source_columns
        .get(column)
        .ok_or_else(|| geometry(chunk, "linear definition source column"))?
    else {
        return Err(geometry(chunk, "linear definition decoder kind"));
    };
    let row = definition
        .source_row
        .ok_or_else(|| geometry(chunk, "removed linear definition source row"))?;
    let terms = normalize_terms(&definition.terms);
    validate_linear_definition(source, row, column, &terms, chunk)?;
    for &(input, _) in &terms {
        if matches!(plan.source_columns.get(input), Some(SourceColumn::Linear(_))) {
            collect_linear_definitions(source, plan, input, chunk, closure, definitions)?;
        }
    }
    let audit = AggregateAcceptanceLinearDefinitionAudit {
        source_column: column,
        source_row: row,
        terms,
    };
    if let Some(existing) = definitions.get(&column) {
        if existing != &audit {
            return Err(geometry(chunk, "inconsistent linear definition provenance"));
        }
    } else {
        definitions.insert(column, audit);
    }
    Ok(())
}

fn validate_linear_definition(
    source: &R1csSnapshot,
    row: usize,
    column: usize,
    expected: &[(usize, F)],
    chunk: usize,
) -> Result<(), GadgetNativeError> {
    let (positive, negative) =
        linear_difference(source, row).ok_or_else(|| geometry(chunk, "linear provenance source row"))?;
    let mut output_coefficient = F::ZERO;
    visit_difference_terms(positive, negative, |candidate, coefficient| {
        if candidate == column {
            output_coefficient += coefficient;
        }
    });
    if output_coefficient == F::ZERO {
        return Err(geometry(chunk, "linear provenance output coefficient"));
    }
    let inverse = output_coefficient.inverse();
    let mut actual = Vec::new();
    visit_difference_terms(positive, negative, |candidate, coefficient| {
        if candidate != column {
            actual.push((candidate, -coefficient * inverse));
        }
    });
    if normalize_terms(&actual) != expected {
        return Err(geometry(chunk, "linear provenance terms"));
    }
    Ok(())
}

fn validate_coordinate_boolean_owner(
    matrices: &[neo_ccs::CcsMatrix<F>],
    owner: AggregateAcceptanceBooleanRowOwner,
    column: usize,
    chunk: usize,
) -> Result<(), GadgetNativeError> {
    let row = owner.encoded_row();
    let rows = physical_matrix_rows(matrices, row, chunk)?;
    if rows[gate::SELECTOR] != [(0, F::ONE)] {
        return Err(geometry(chunk, "Boolean owner selector"));
    }
    let exact = match owner {
        AggregateAcceptanceBooleanRowOwner::CoordinatePairLeft { paired_column, .. } => {
            rows[gate::QUADRATIC_BIT_LEFT] == [(column, F::ONE)]
                && rows[gate::QUADRATIC_BIT_RIGHT] == [(paired_column, F::ONE)]
        }
        AggregateAcceptanceBooleanRowOwner::CoordinatePairRight { paired_column, .. } => {
            rows[gate::QUADRATIC_BIT_LEFT] == [(paired_column, F::ONE)]
                && rows[gate::QUADRATIC_BIT_RIGHT] == [(column, F::ONE)]
        }
        AggregateAcceptanceBooleanRowOwner::CoordinateTail { .. } => rows[gate::BITNESS] == [(column, F::ONE)],
        AggregateAcceptanceBooleanRowOwner::TranslatedSource { .. } => false,
    };
    let allowed = match owner {
        AggregateAcceptanceBooleanRowOwner::CoordinatePairLeft { .. }
        | AggregateAcceptanceBooleanRowOwner::CoordinatePairRight { .. } => {
            [gate::SELECTOR, gate::QUADRATIC_BIT_LEFT, gate::QUADRATIC_BIT_RIGHT]
        }
        AggregateAcceptanceBooleanRowOwner::CoordinateTail { .. } => [gate::SELECTOR, gate::BITNESS, gate::BITNESS],
        AggregateAcceptanceBooleanRowOwner::TranslatedSource { .. } => unreachable!(),
    };
    if !exact
        || rows
            .iter()
            .enumerate()
            .any(|(matrix, terms)| !allowed.contains(&matrix) && !terms.is_empty())
    {
        return Err(geometry(chunk, "Boolean coordinate owner row"));
    }
    Ok(())
}

fn validate_translated_boolean_row(
    matrices: &[neo_ccs::CcsMatrix<F>],
    row: usize,
    terms: &[(usize, F)],
    chunk: usize,
) -> Result<(), GadgetNativeError> {
    let rows = physical_matrix_rows(matrices, row, chunk)?;
    let mut minus_one = terms.to_vec();
    minus_one.push((0, -F::ONE));
    let exact = rows[gate::SELECTOR] == [(0, F::ONE)]
        && rows[gate::PRODUCT_LEFT] == terms
        && rows[gate::PRODUCT_RIGHT] == normalize_terms(&minus_one)
        && rows[gate::PRODUCT_OUT].is_empty()
        && rows.iter().enumerate().all(|(matrix, row_terms)| {
            matches!(matrix, gate::SELECTOR | gate::PRODUCT_LEFT | gate::PRODUCT_RIGHT) || row_terms.is_empty()
        });
    if !exact {
        return Err(geometry(chunk, "translated Boolean owner row"));
    }
    Ok(())
}

fn physical_matrix_rows(
    matrices: &[neo_ccs::CcsMatrix<F>],
    row: usize,
    chunk: usize,
) -> Result<Vec<Vec<(usize, F)>>, GadgetNativeError> {
    matrices
        .iter()
        .map(|matrix| matrix_row(matrix, row).ok_or_else(|| geometry(chunk, "non-CSC physical matrix")))
        .collect()
}

fn physical_row_audit(
    matrices: &[neo_ccs::CcsMatrix<F>],
    encoded_rows: usize,
    row: usize,
) -> Result<AggregateAcceptancePhysicalRowAudit, GadgetNativeError> {
    if row >= encoded_rows {
        return Err(geometry(0, "physical row bound"));
    }
    let matrices = physical_matrix_rows(matrices, row, 0)?
        .into_iter()
        .enumerate()
        .filter_map(|(matrix, terms)| {
            (!terms.is_empty()).then_some(AggregateAcceptanceMatrixRowAudit { matrix, terms })
        })
        .collect();
    Ok(AggregateAcceptancePhysicalRowAudit { row, matrices })
}
