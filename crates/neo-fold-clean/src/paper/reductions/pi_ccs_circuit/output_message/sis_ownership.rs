//! Exact source-row ownership for the Π_CCS output SIS input binding.
//!
//! Owns: the correspondence from typed preimage source columns to canonical
//! balanced-ternary opening rows and auxiliary columns.
//!
//! Does not own: output truth, seeded Φ81
//! soundness, Poseidon2 soundness, gadget-native lowering, or row removal.
//!
//! Emits constraints: no.
//!
//! Authority boundary: this audit accepts only an already-emitted R1CS trace.
//! It classifies physical provenance; it does not make the trace authoritative.
//!
//! | Leaf | Mathematical obligation | Physical source ownership |
//! |---|---|---|
//! | `verifier_shape` | canonical opening of domain and dimension fields | exact opening rows for unique verifier-owned columns |
//! | `y_ring` | canonical opening of accepted FE output limbs | exact opening rows for unique `y_ring` columns |
//! | `shared_input_binding` | commitment allocation and seeded Φ81 map | input-binding rows/columns not in a canonical opening |
//! | `digest_compression` | independent short SIS binding | complete shared phase span |
//! | `envelope` | domain/shape envelope and Poseidon2 | complete shared phase span |

use std::collections::HashMap;
use std::ops::Range;

use crate::engine::r1cs_circuit::builder::BALANCED_TERNARY_DIGITS;
use crate::engine::r1cs_circuit::{BalancedTernaryOpeningTraceEntry, R1csEncodingTrace};
use crate::paper::reductions::accumulator_sis_circuit::{SisAccumulatorCircuitLayout, SisCircuitSpan};
use crate::paper::reductions::pi_ccs_output_message::R1csInputOwner;

use super::super::Error;
use super::PiCcsOutputsPreimage;

const DIGIT_ROWS_PER_OPENING: usize = 2 * BALANCED_TERNARY_DIGITS;
const TRANSITION_ROWS_PER_OPENING: usize = BALANCED_TERNARY_DIGITS;
const AUXILIARY_COLUMNS_PER_OPENING: usize = 3 * BALANCED_TERNARY_DIGITS - 1;

/// Exact canonical-opening contribution of one semantic input owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsOutputSisOwnerAudit {
    owner: R1csInputOwner,
    field_occurrences: usize,
    unique_source_columns: usize,
    new_openings: usize,
    reused_openings: usize,
    digit_rows: usize,
    reconstruction_rows: usize,
    transition_rows: usize,
    auxiliary_columns: usize,
}

impl PiCcsOutputSisOwnerAudit {
    pub fn owner(&self) -> R1csInputOwner {
        self.owner
    }

    pub fn field_occurrences(&self) -> usize {
        self.field_occurrences
    }

    pub fn unique_source_columns(&self) -> usize {
        self.unique_source_columns
    }

    pub fn new_openings(&self) -> usize {
        self.new_openings
    }

    pub fn reused_openings(&self) -> usize {
        self.reused_openings
    }

    pub fn digit_rows(&self) -> usize {
        self.digit_rows
    }

    pub fn reconstruction_rows(&self) -> usize {
        self.reconstruction_rows
    }

    pub fn transition_rows(&self) -> usize {
        self.transition_rows
    }

    pub fn source_rows(&self) -> usize {
        self.digit_rows + self.reconstruction_rows + self.transition_rows
    }

    pub fn auxiliary_columns(&self) -> usize {
        self.auxiliary_columns
    }
}

/// Complete source-R1CS ownership partition for one Π_CCS output SIS call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsOutputSisAudit {
    owners: [PiCcsOutputSisOwnerAudit; 2],
    input_binding_rows: Range<usize>,
    input_binding_columns: Range<usize>,
    shared_input_binding_rows: usize,
    shared_input_binding_columns: usize,
    digest_compression_rows: Range<usize>,
    digest_compression_columns: Range<usize>,
    envelope_rows: Range<usize>,
    envelope_columns: Range<usize>,
}

impl PiCcsOutputSisAudit {
    pub fn owners(&self) -> &[PiCcsOutputSisOwnerAudit; 2] {
        &self.owners
    }

    pub fn owner(&self, owner: R1csInputOwner) -> &PiCcsOutputSisOwnerAudit {
        &self.owners[owner_index(owner)]
    }

    pub fn input_binding_rows(&self) -> Range<usize> {
        self.input_binding_rows.clone()
    }

    pub fn input_binding_columns(&self) -> Range<usize> {
        self.input_binding_columns.clone()
    }

    pub fn shared_input_binding_rows(&self) -> usize {
        self.shared_input_binding_rows
    }

    pub fn shared_input_binding_columns(&self) -> usize {
        self.shared_input_binding_columns
    }

    pub fn digest_compression_rows(&self) -> Range<usize> {
        self.digest_compression_rows.clone()
    }

    pub fn digest_compression_columns(&self) -> Range<usize> {
        self.digest_compression_columns.clone()
    }

    pub fn envelope_rows(&self) -> Range<usize> {
        self.envelope_rows.clone()
    }

    pub fn envelope_columns(&self) -> Range<usize> {
        self.envelope_columns.clone()
    }
}

#[derive(Clone, Copy, Default)]
struct OwnerCounts {
    field_occurrences: usize,
    unique_source_columns: usize,
    new_openings: usize,
    reused_openings: usize,
    digit_rows: usize,
    reconstruction_rows: usize,
    transition_rows: usize,
    auxiliary_columns: usize,
}

/// Reconcile every typed Π_CCS output field with the unique canonical opening
/// of its R1CS source column. Global SIS phases remain explicitly shared.
pub fn audit_pi_ccs_output_sis(
    preimage: &PiCcsOutputsPreimage,
    layout: &SisAccumulatorCircuitLayout,
    trace: &R1csEncodingTrace,
) -> Result<PiCcsOutputSisAudit, Error> {
    validate_phase_partition(layout, trace)?;

    let mut counts = [OwnerCounts::default(); 2];
    let mut source_owners = HashMap::new();
    for field in preimage.fields() {
        let owner = field.r1cs_input_owner();
        let owner_count = &mut counts[owner_index(owner)];
        owner_count.field_occurrences += 1;
        match source_owners.insert(field.source_column(), owner) {
            None => owner_count.unique_source_columns += 1,
            Some(previous) if previous == owner => {}
            Some(previous) => {
                return Err(Error::Shape(format!(
                    "Pi_CCS output SIS source column {} is shared by owners {previous:?} and {owner:?}",
                    field.source_column()
                )));
            }
        }
    }

    let openings = trace.balanced_ternary_openings();
    let mut opening_by_source = HashMap::with_capacity(openings.len());
    for (index, opening) in openings.iter().enumerate() {
        if let Some(previous) = opening_by_source.insert(opening.field_col, index) {
            return Err(Error::Shape(format!(
                "balanced-ternary source column {} has duplicate openings {previous} and {index}",
                opening.field_col
            )));
        }
    }

    let input_span = layout.input_binding();
    let input_openings = input_span.balanced_ternary_openings();
    let mut opening_row_ranges = Vec::with_capacity(input_openings.len());
    let mut opening_column_ranges = Vec::with_capacity(input_openings.len());

    for (&source_column, &owner) in &source_owners {
        let index = opening_by_source
            .get(&source_column)
            .copied()
            .ok_or_else(|| {
                Error::Shape(format!(
                    "Pi_CCS output SIS source column {source_column} has no canonical opening trace"
                ))
            })?;
        let owner_count = &mut counts[owner_index(owner)];
        if index < input_openings.start {
            let opening = &openings[index];
            if opening.transition_rows.end > input_span.rows().start {
                return Err(Error::Shape(format!(
                    "reused opening {index} overlaps the Pi_CCS output input-binding phase"
                )));
            }
            owner_count.reused_openings += 1;
        } else if input_openings.contains(&index) {
            let shape = validate_opening(index, &openings[index], input_span)?;
            owner_count.new_openings += 1;
            owner_count.digit_rows += DIGIT_ROWS_PER_OPENING;
            owner_count.reconstruction_rows += 1;
            owner_count.transition_rows += TRANSITION_ROWS_PER_OPENING;
            owner_count.auxiliary_columns += AUXILIARY_COLUMNS_PER_OPENING;
            opening_row_ranges.push(shape.rows);
            opening_column_ranges.push(shape.columns);
        } else {
            return Err(Error::Shape(format!(
                "Pi_CCS output SIS source column {source_column} is opened only after its input-binding phase"
            )));
        }
    }

    for index in input_openings.clone() {
        let opening = &openings[index];
        if !source_owners.contains_key(&opening.field_col) {
            return Err(Error::Shape(format!(
                "Pi_CCS output input-binding opening {index} has unowned source column {}",
                opening.field_col
            )));
        }
    }

    validate_disjoint("opening row", &mut opening_row_ranges)?;
    validate_disjoint("opening auxiliary-column", &mut opening_column_ranges)?;

    let owned_rows: usize = counts.iter().map(|owner| owner_rows(*owner)).sum();
    let owned_columns: usize = counts.iter().map(|owner| owner.auxiliary_columns).sum();
    let input_rows = input_span.rows();
    let input_columns = input_span.columns();
    let shared_input_binding_rows = range_len(&input_rows)
        .checked_sub(owned_rows)
        .ok_or_else(|| Error::Shape("Pi_CCS output SIS opening rows exceed the input-binding phase".into()))?;
    let shared_input_binding_columns = range_len(&input_columns)
        .checked_sub(owned_columns)
        .ok_or_else(|| Error::Shape("Pi_CCS output SIS opening columns exceed the input-binding phase".into()))?;

    Ok(PiCcsOutputSisAudit {
        owners: [
            finish_owner(R1csInputOwner::VerifierShape, counts[0]),
            finish_owner(R1csInputOwner::YRingOutput, counts[1]),
        ],
        input_binding_rows: input_rows,
        input_binding_columns: input_columns,
        shared_input_binding_rows,
        shared_input_binding_columns,
        digest_compression_rows: layout.digest_compression().rows(),
        digest_compression_columns: layout.digest_compression().columns(),
        envelope_rows: layout.envelope().rows(),
        envelope_columns: layout.envelope().columns(),
    })
}

struct OpeningShape {
    rows: Range<usize>,
    columns: Range<usize>,
}

fn validate_opening(
    index: usize,
    opening: &BalancedTernaryOpeningTraceEntry,
    phase: &SisCircuitSpan,
) -> Result<OpeningShape, Error> {
    if range_len(&opening.digit_rows) != DIGIT_ROWS_PER_OPENING
        || range_len(&opening.transition_rows) != TRANSITION_ROWS_PER_OPENING
        || opening.digit_rows.end != opening.reconstruction_row
        || opening.reconstruction_row + 1 != opening.transition_rows.start
    {
        return Err(Error::Shape(format!(
            "balanced-ternary opening {index} does not have the exact 82 + 1 + 41 row layout"
        )));
    }
    let rows = opening.digit_rows.start..opening.transition_rows.end;
    if rows.start < phase.rows().start || rows.end > phase.rows().end {
        return Err(Error::Shape(format!(
            "balanced-ternary opening {index} rows {rows:?} escape phase {:?}",
            phase.rows()
        )));
    }

    if !consecutive(&opening.digit_cols)
        || !consecutive(&opening.negative_cols)
        || !consecutive(&opening.borrow_cols)
        || opening.digit_cols[BALANCED_TERNARY_DIGITS - 1] + 1 != opening.negative_cols[0]
        || opening.negative_cols[BALANCED_TERNARY_DIGITS - 1] + 1 != opening.borrow_cols[0]
    {
        return Err(Error::Shape(format!(
            "balanced-ternary opening {index} auxiliary columns are not the exact contiguous layout"
        )));
    }
    let columns = opening.digit_cols[0]..opening.borrow_cols[BALANCED_TERNARY_DIGITS - 2] + 1;
    if range_len(&columns) != AUXILIARY_COLUMNS_PER_OPENING
        || columns.start < phase.columns().start
        || columns.end > phase.columns().end
    {
        return Err(Error::Shape(format!(
            "balanced-ternary opening {index} columns {columns:?} escape phase {:?}",
            phase.columns()
        )));
    }
    Ok(OpeningShape { rows, columns })
}

fn validate_phase_partition(layout: &SisAccumulatorCircuitLayout, trace: &R1csEncodingTrace) -> Result<(), Error> {
    let input = layout.input_binding();
    let compression = layout.digest_compression();
    let envelope = layout.envelope();
    if input.rows().end != compression.rows().start
        || compression.rows().end != envelope.rows().start
        || input.columns().end != compression.columns().start
        || compression.columns().end != envelope.columns().start
        || input.balanced_ternary_openings().end != compression.balanced_ternary_openings().start
        || compression.balanced_ternary_openings().end != envelope.balanced_ternary_openings().start
        || envelope.balanced_ternary_openings().end > trace.balanced_ternary_openings().len()
    {
        return Err(Error::Shape(
            "Pi_CCS output SIS phase frontiers do not form one ordered partition".into(),
        ));
    }
    if !envelope.balanced_ternary_openings().is_empty() {
        return Err(Error::Shape(
            "Pi_CCS output SIS envelope unexpectedly owns balanced-ternary openings".into(),
        ));
    }
    for index in compression.balanced_ternary_openings() {
        validate_opening(index, &trace.balanced_ternary_openings()[index], compression)?;
    }
    Ok(())
}

fn validate_disjoint(label: &str, ranges: &mut [Range<usize>]) -> Result<(), Error> {
    ranges.sort_unstable_by_key(|range| range.start);
    for pair in ranges.windows(2) {
        if pair[0].end > pair[1].start {
            return Err(Error::Shape(format!(
                "Pi_CCS output SIS {label} ranges {:?} and {:?} overlap",
                pair[0], pair[1]
            )));
        }
    }
    Ok(())
}

fn consecutive<const N: usize>(columns: &[usize; N]) -> bool {
    columns.windows(2).all(|pair| pair[1] == pair[0] + 1)
}

fn range_len(range: &Range<usize>) -> usize {
    range.end - range.start
}

fn owner_index(owner: R1csInputOwner) -> usize {
    match owner {
        R1csInputOwner::VerifierShape => 0,
        R1csInputOwner::YRingOutput => 1,
    }
}

fn owner_rows(counts: OwnerCounts) -> usize {
    counts.digit_rows + counts.reconstruction_rows + counts.transition_rows
}

fn finish_owner(owner: R1csInputOwner, counts: OwnerCounts) -> PiCcsOutputSisOwnerAudit {
    PiCcsOutputSisOwnerAudit {
        owner,
        field_occurrences: counts.field_occurrences,
        unique_source_columns: counts.unique_source_columns,
        new_openings: counts.new_openings,
        reused_openings: counts.reused_openings,
        digit_rows: counts.digit_rows,
        reconstruction_rows: counts.reconstruction_rows,
        transition_rows: counts.transition_rows,
        auxiliary_columns: counts.auxiliary_columns,
    }
}
