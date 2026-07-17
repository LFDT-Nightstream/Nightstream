//! Exact nine-row lowering for one sampler chunk acceptance block.
//!
//! Owns: four-row/two-column source validation, canonical inverse projection,
//! fourteen product-tree coordinates, and the exact nine-row replacement.
//!
//! Does not own: chunk-bit derivation, Mod-5 arithmetic, prefix counting,
//! first-accepted selection, or selector-gated inactive materialization.
//! Read-only production-placement evidence belongs to `outer_image`.
//!
//! Emits constraints: yes. Four source rows become seven packed output-bit
//! rows, one radix-three product aggregate, and one root/acceptance binding.
//!
//! Authority boundary: the sixteen checked source bits remain the local
//! implementation arithmetic reference. The inverse is reconstructed
//! canonically from their exact little-endian difference; tree outputs and
//! `accept` are bound by the emitted rows. Independent sampler semantics must
//! still justify this check family.
//!
//! | Stage path | Mathematical obligation | Coordinates | Rows | Lean theorem |
//! |---|---|---:|---:|---|
//! | `chunk.accept.packed.tree_bit_pairs` | Fourteen tree outputs are Boolean | 14 | 7 | `productTreeOutputBitRows_iff` |
//! | `chunk.accept.packed.product_aggregate` | All fourteen product edges hold | 0 | 1 | `productTreeAggregateRow_iff` |
//! | `chunk.accept.packed.root_binding` | Roots derive the retained accept bit | 1 | 1 | `aggregateAcceptanceRows_iff_sourceMeaning` |
//! | `chunk.accept.outer_image` | Delegate exact source/physical placement evidence | 16 inputs | 0 | `outer_image`; production placement bridge open |

use std::ops::Range;

use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::engine::r1cs_circuit::{AcceptanceTraceEntry, Lc, R1csEncodingTrace, R1csSnapshot, Var};

use super::gates::{one_selector, TraceGateBuilder};
use super::slots::{slot_terms, ValueEncoding, ValueSlot};
use super::{
    claim_gadget_column, claim_rows, scaled_terms, source_terms, validate_row, GadgetNativeError, GadgetNativePlan,
    SourceColumn,
};

mod outer_image;
pub(super) use outer_image::translated_boolean_source_rows;
pub use outer_image::{
    audit_r1cs_gadget_native_aggregate_acceptance_outer_image, AggregateAcceptanceBitOuterImage,
    AggregateAcceptanceBooleanRowOwner, AggregateAcceptanceChunkOuterImageAudit, AggregateAcceptanceDecodedImage,
    AggregateAcceptanceLinearDefinitionAudit, AggregateAcceptanceMatrixRowAudit, AggregateAcceptanceOuterImageAudit,
    AggregateAcceptancePhysicalRowAudit, AggregateAcceptanceSourceRowAudit,
};

const GADGET: &str = "sampler chunk acceptance";
const REJECTION_BUCKET: u64 = 65_535;
const SOURCE_ROWS_PER_CHUNK: usize = 4;
const SOURCE_COLUMNS_PER_CHUNK: usize = 2;
pub(super) const TREE_OUTPUTS_PER_CHUNK: usize = 14;
pub(super) const ENCODED_COORDINATES_PER_CHUNK: usize = TREE_OUTPUTS_PER_CHUNK + 1;
pub(super) const TREE_BIT_PAIR_ROWS_PER_CHUNK: usize = 7;
pub(super) const PRODUCT_AGGREGATE_ROWS_PER_CHUNK: usize = 1;
pub(super) const ROOT_BINDING_ROWS_PER_CHUNK: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProjectedRole {
    CanonicalNonzeroInverse,
}

/// Exact source ownership established before acceptance rows are removed.
pub(super) struct ValidatedAcceptance {
    projected_roles: Vec<Option<ProjectedRole>>,
    chunks: usize,
}

impl ValidatedAcceptance {
    pub(super) fn validate_and_claim(
        source: &R1csSnapshot,
        trace: &R1csEncodingTrace,
        covered_rows: &mut [bool],
        gadget_columns: &mut [bool],
    ) -> Result<Self, GadgetNativeError> {
        let mut projected_roles = vec![None; source.cols()];
        let mut strict_owner = vec![None; source.cols()];
        for (chunk, event) in trace.acceptance_chunks().iter().enumerate() {
            validate_geometry(source, chunk, event, gadget_columns)?;
            validate_rows(source, event)?;
            validate_canonical_witness(source, chunk, event)?;
            claim_rows(source, GADGET, &event.source_rows, covered_rows)?;
            let inverse = event.inverse.col();
            if projected_roles[inverse]
                .replace(ProjectedRole::CanonicalNonzeroInverse)
                .is_some()
            {
                return Err(geometry(chunk, "duplicate projected inverse"));
            }
            claim_gadget_column(inverse, gadget_columns)?;
            strict_owner[inverse] = Some(chunk);
        }

        for row in 0..source.rows() {
            for &(column, _) in source
                .a_row(row)
                .iter()
                .chain(source.b_row(row))
                .chain(source.c_row(row))
            {
                if let Some(chunk) = strict_owner[column] {
                    if !trace.acceptance_chunks()[chunk].source_rows.contains(&row) {
                        return Err(GadgetNativeError::GadgetTemporaryEscapes { column });
                    }
                }
            }
        }

        Ok(Self {
            projected_roles,
            chunks: trace.acceptance_chunks().len(),
        })
    }

    pub(super) fn len(&self) -> usize {
        self.chunks
    }

    pub(super) fn projected_role(&self, column: usize) -> Option<ProjectedRole> {
        self.projected_roles[column]
    }
}

fn validate_geometry(
    source: &R1csSnapshot,
    chunk: usize,
    event: &AcceptanceTraceEntry,
    gadget_columns: &[bool],
) -> Result<(), GadgetNativeError> {
    if event.source_rows.len() != SOURCE_ROWS_PER_CHUNK || event.source_rows.end > source.rows() {
        return Err(geometry(chunk, "four-row source interval"));
    }
    if event.allocated_columns.len() != SOURCE_COLUMNS_PER_CHUNK
        || event.allocated_columns.start == 0
        || event.allocated_columns.end > source.cols()
    {
        return Err(geometry(chunk, "two-column allocation interval"));
    }
    let first = event.allocated_columns.start;
    if event.accept.col() != first || event.inverse.col() != first + 1 {
        return Err(geometry(chunk, "production column role order"));
    }
    if gadget_columns[event.accept.col()] || gadget_columns[event.inverse.col()] {
        return Err(geometry(chunk, "source column owned by another gadget"));
    }
    if event
        .chunk_bits
        .iter()
        .any(|variable| variable.col() == 0 || variable.col() >= first)
    {
        return Err(geometry(chunk, "topological chunk-bit inputs"));
    }
    Ok(())
}

fn validate_rows(source: &R1csSnapshot, event: &AcceptanceTraceEntry) -> Result<(), GadgetNativeError> {
    let accept = Lc::from_var(event.accept);
    let accept_minus_one = accept.clone().add_scaled(&Lc::from_const(F::ONE), -F::ONE);
    validate_row(
        source,
        GADGET,
        event.source_rows.start,
        &accept,
        &accept_minus_one,
        &Lc::zero(),
    )?;

    let difference = acceptance_difference(event);
    let one_minus_accept = Lc::from_const(F::ONE).add_scaled(&accept, -F::ONE);
    validate_row(
        source,
        GADGET,
        event.source_rows.start + 1,
        &one_minus_accept,
        &difference,
        &Lc::zero(),
    )?;
    validate_row(
        source,
        GADGET,
        event.source_rows.start + 2,
        &difference,
        &Lc::from_var(event.inverse),
        &accept,
    )?;
    validate_row(
        source,
        GADGET,
        event.source_rows.start + 3,
        &one_minus_accept,
        &Lc::from_var(event.inverse),
        &Lc::zero(),
    )
}

fn validate_canonical_witness(
    source: &R1csSnapshot,
    chunk: usize,
    event: &AcceptanceTraceEntry,
) -> Result<(), GadgetNativeError> {
    let difference = super::eval_lc_from_source(&acceptance_difference(event), source.witness());
    let expected = canonical_inverse(difference);
    if source.witness()[event.inverse.col()] != expected {
        return Err(GadgetNativeError::AcceptanceWitness {
            chunk,
            column: event.inverse.col(),
        });
    }
    Ok(())
}

fn acceptance_difference(event: &AcceptanceTraceEntry) -> Lc {
    let mut difference = little_endian_lc(&event.chunk_bits);
    difference.add_constant(-F::from_u64(REJECTION_BUCKET));
    difference
}

fn little_endian_lc<const N: usize>(bits: &[Var; N]) -> Lc {
    let mut out = Lc::zero();
    let mut power = F::ONE;
    for &bit in bits {
        out.add_term(bit, power);
        power += power;
    }
    out
}

fn canonical_inverse(value: F) -> F {
    if value == F::ZERO {
        F::ZERO
    } else {
        value.inverse()
    }
}

fn geometry(chunk: usize, detail: &'static str) -> GadgetNativeError {
    GadgetNativeError::AcceptanceGeometry { chunk, detail }
}

#[derive(Clone, Debug)]
struct AcceptanceChunkSlots {
    accept: ValueSlot,
    outputs: [ValueSlot; TREE_OUTPUTS_PER_CHUNK],
    inverse_column: usize,
    active_rows: Range<usize>,
}

/// Assignment coordinates omitted from the common coordinate-gate schedule.
#[derive(Clone, Debug, Default)]
pub(super) struct AcceptanceSlots {
    chunks: Vec<AcceptanceChunkSlots>,
    omitted_coordinates: Vec<bool>,
}

impl AcceptanceSlots {
    pub(super) fn omits_coordinate(&self, column: usize) -> bool {
        self.omitted_coordinates
            .get(column)
            .copied()
            .unwrap_or(false)
    }
}

/// Read-only role schedule for an exact acceptance artifact exporter.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct AggregateAcceptanceAudit<'a> {
    pub encoded_accept: usize,
    pub encoded_outputs: Range<usize>,
    pub inverse_source_column: usize,
    pub inverse_difference: &'a Lc,
    pub radix_weights: [F; TREE_OUTPUTS_PER_CHUNK],
    pub active_rows: Range<usize>,
}

impl GadgetNativePlan {
    #[doc(hidden)]
    pub fn aggregate_acceptance_audit(&self, chunk: usize) -> Option<AggregateAcceptanceAudit<'_>> {
        let slots = self.acceptance_slots.chunks.get(chunk)?;
        let start = slots.outputs[0].start;
        if !slots
            .outputs
            .iter()
            .enumerate()
            .all(|(offset, slot)| slot.start == start + offset && slot.width == 1)
        {
            return None;
        }
        let SourceColumn::CanonicalNonzeroInverse(difference) = self.source_columns.get(slots.inverse_column)? else {
            return None;
        };
        Some(AggregateAcceptanceAudit {
            encoded_accept: slots.accept.start,
            encoded_outputs: start..start + TREE_OUTPUTS_PER_CHUNK,
            inverse_source_column: slots.inverse_column,
            inverse_difference: difference,
            radix_weights: radix_weights(),
            active_rows: slots.active_rows.clone(),
        })
    }
}

pub(super) fn allocate_and_install(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    validated: &ValidatedAcceptance,
    assignment: &mut Vec<F>,
    source_columns: &mut [Option<SourceColumn>],
) -> Result<AcceptanceSlots, GadgetNativeError> {
    if validated.len() != trace.acceptance_chunks().len() {
        return Err(geometry(0, "validated trace census"));
    }
    let mut chunks = Vec::with_capacity(validated.len());
    for (chunk, event) in trace.acceptance_chunks().iter().enumerate() {
        let accept = encoded_boolean(source_columns, event.accept.col(), chunk)?;
        let bits = event
            .chunk_bits
            .map(|variable| source.witness()[variable.col()]);
        let values = product_tree_values(&bits);
        let outputs = values.map(|value| push_boolean_coordinate(assignment, value));
        let difference = acceptance_difference(event);
        let inverse = canonical_inverse(super::eval_lc_from_source(&difference, source.witness()));
        if inverse != source.witness()[event.inverse.col()] {
            return Err(GadgetNativeError::AcceptanceWitness {
                chunk,
                column: event.inverse.col(),
            });
        }
        if source_columns[event.inverse.col()]
            .replace(SourceColumn::CanonicalNonzeroInverse(difference))
            .is_some()
        {
            return Err(geometry(chunk, "projected inverse definition overlap"));
        }
        chunks.push(AcceptanceChunkSlots {
            accept,
            outputs,
            inverse_column: event.inverse.col(),
            active_rows: 0..0,
        });
    }

    let mut omitted_coordinates = vec![false; assignment.len()];
    for chunk in &chunks {
        for slot in std::iter::once(&chunk.accept).chain(&chunk.outputs) {
            if slot.width != 1 || std::mem::replace(&mut omitted_coordinates[slot.start], true) {
                return Err(geometry(0, "overlapping aggregate coordinate"));
            }
        }
    }
    Ok(AcceptanceSlots {
        chunks,
        omitted_coordinates,
    })
}

fn encoded_boolean(
    source_columns: &[Option<SourceColumn>],
    column: usize,
    chunk: usize,
) -> Result<ValueSlot, GadgetNativeError> {
    match source_columns.get(column).and_then(Option::as_ref) {
        Some(SourceColumn::Encoded(slot)) if slot.width == 1 && matches!(slot.encoding, ValueEncoding::Boolean) => {
            Ok(*slot)
        }
        _ => Err(geometry(chunk, "retained accept encoding")),
    }
}

fn push_boolean_coordinate(assignment: &mut Vec<F>, value: F) -> ValueSlot {
    let start = assignment.len();
    assignment.push(value);
    ValueSlot {
        start,
        width: 1,
        encoding: ValueEncoding::Boolean,
    }
}

fn product_tree_values(bits: &[F; 16]) -> [F; TREE_OUTPUTS_PER_CHUNK] {
    let mut outputs = [F::ZERO; TREE_OUTPUTS_PER_CHUNK];
    for index in 0..TREE_OUTPUTS_PER_CHUNK {
        let (left, right) = edge_values(bits, &outputs, index);
        outputs[index] = left * right;
    }
    outputs
}

fn edge_values(bits: &[F; 16], outputs: &[F; TREE_OUTPUTS_PER_CHUNK], index: usize) -> (F, F) {
    match index {
        0 => (bits[0], bits[1]),
        1 => (bits[2], bits[3]),
        2 => (bits[4], bits[5]),
        3 => (bits[6], bits[7]),
        4 => (outputs[0], outputs[1]),
        5 => (outputs[2], outputs[3]),
        6 => (outputs[4], outputs[5]),
        7 => (bits[8], bits[9]),
        8 => (bits[10], bits[11]),
        9 => (bits[12], bits[13]),
        10 => (bits[14], bits[15]),
        11 => (outputs[7], outputs[8]),
        12 => (outputs[9], outputs[10]),
        13 => (outputs[11], outputs[12]),
        _ => unreachable!("fourteen-edge product tree"),
    }
}

pub(super) fn emit(
    trace: &R1csEncodingTrace,
    slots: &mut AcceptanceSlots,
    decoded: &[Option<Vec<(usize, F)>>],
    gates: &mut TraceGateBuilder,
) -> Result<(), GadgetNativeError> {
    emit_with_decoded_lookup(trace, slots, gates, &mut |column, row| {
        source_terms(column, decoded, row)
    })
}

pub(super) fn emit_with_decoded_lookup(
    trace: &R1csEncodingTrace,
    slots: &mut AcceptanceSlots,
    gates: &mut TraceGateBuilder,
    lookup: &mut impl FnMut(usize, usize) -> Result<Vec<(usize, F)>, GadgetNativeError>,
) -> Result<(), GadgetNativeError> {
    if trace.acceptance_chunks().len() != slots.chunks.len() {
        return Err(geometry(0, "emission trace census"));
    }
    for (chunk, (event, slots)) in trace
        .acceptance_chunks()
        .iter()
        .zip(&mut slots.chunks)
        .enumerate()
    {
        let start = gates.rows;
        for pair in 0..TREE_BIT_PAIR_ROWS_PER_CHUNK {
            gates.quadratic_bit_pair(
                slot_terms(slots.outputs[2 * pair]),
                slot_terms(slots.outputs[2 * pair + 1]),
            );
        }

        let weights = radix_weights();
        let mut products = Vec::with_capacity(TREE_OUTPUTS_PER_CHUNK);
        let mut out = Vec::with_capacity(TREE_OUTPUTS_PER_CHUNK);
        for index in 0..TREE_OUTPUTS_PER_CHUNK {
            let (left, right) = edge_terms(event, slots, lookup, index)?;
            products.push((scaled_terms(left, weights[index]), right));
            out.extend(scaled_terms(slot_terms(slots.outputs[index]), weights[index]));
        }
        gates.product_sum(one_selector(), products, out);

        gates.product_sum(
            one_selector(),
            vec![(slot_terms(slots.outputs[6]), slot_terms(slots.outputs[13]))],
            vec![(0, F::ONE), (slots.accept.start, -F::ONE)],
        );
        slots.active_rows = start..gates.rows;
        if slots.active_rows.len()
            != TREE_BIT_PAIR_ROWS_PER_CHUNK + PRODUCT_AGGREGATE_ROWS_PER_CHUNK + ROOT_BINDING_ROWS_PER_CHUNK
        {
            return Err(geometry(chunk, "nine-row active interval"));
        }
    }
    Ok(())
}

fn edge_terms(
    event: &AcceptanceTraceEntry,
    slots: &AcceptanceChunkSlots,
    lookup: &mut impl FnMut(usize, usize) -> Result<Vec<(usize, F)>, GadgetNativeError>,
    index: usize,
) -> Result<(Vec<(usize, F)>, Vec<(usize, F)>), GadgetNativeError> {
    let mut bit = |index: usize| lookup(event.chunk_bits[index].col(), event.source_rows.start);
    let output = |index: usize| slot_terms(slots.outputs[index]);
    match index {
        0 => Ok((bit(0)?, bit(1)?)),
        1 => Ok((bit(2)?, bit(3)?)),
        2 => Ok((bit(4)?, bit(5)?)),
        3 => Ok((bit(6)?, bit(7)?)),
        4 => Ok((output(0), output(1))),
        5 => Ok((output(2), output(3))),
        6 => Ok((output(4), output(5))),
        7 => Ok((bit(8)?, bit(9)?)),
        8 => Ok((bit(10)?, bit(11)?)),
        9 => Ok((bit(12)?, bit(13)?)),
        10 => Ok((bit(14)?, bit(15)?)),
        11 => Ok((output(7), output(8))),
        12 => Ok((output(9), output(10))),
        13 => Ok((output(11), output(12))),
        _ => unreachable!("fourteen-edge product tree"),
    }
}

fn radix_weights() -> [F; TREE_OUTPUTS_PER_CHUNK] {
    let mut weights = [F::ZERO; TREE_OUTPUTS_PER_CHUNK];
    let mut weight = F::ONE;
    for slot in &mut weights {
        *slot = weight;
        weight *= F::from_u64(3);
    }
    weights
}
