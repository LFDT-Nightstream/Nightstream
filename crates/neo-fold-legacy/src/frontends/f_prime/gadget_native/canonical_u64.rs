//! Exact canonical-u64 trace validation and diagnostic classification.
//!
//! Owns: validation of the 69-row canonical Goldilocks decomposition,
//! source-column roles, stage ownership, and decomposition census reporting.
//!
//! Does not own: row removal, slot aliasing, source semantics, or field
//! authority. A valid report is diagnostic provenance only.
//!
//! Emits constraints: no.
//!
//! Authority boundary: all roles and rows are reconstructed from the source
//! R1CS. Trace metadata never authorizes a lowering by itself.
//!
//! | Row family | Rows | Exact equation | Role owner |
//! |---|---:|---|---|
//! | Bit alphabet | 64 | `b_i * (b_i - 1) = 0` | `bits[0..64]` |
//! | Recomposition | 1 | `(field - sum 2^i b_i) * 1 = 0` | `field`, `bits` |
//! | High flag alphabet | 1 | `h * (h - 1) = 0` | `high_is_max` |
//! | High flag forward | 1 | `h * (hi - 0xffffffff) = 0` | `high_is_max` |
//! | High flag reverse | 1 | `(hi - 0xffffffff) * inv = 1 - h` | `inverse` |
//! | Canonical bound | 1 | `h * lo = 0` | all low bits |

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::{CanonicalU64TraceEntry, Lc, R1csEncodingTrace, R1csSnapshot, Var};

use super::boolean_dedup::validate_public_columns;
use super::{
    reject_public_gadget_columns, select_linear_definition_columns, validate_and_mark_trace, validate_row,
    validate_source_one, GadgetNativeError,
};

const SOURCE_ROWS: usize = 69;
const BIT_ROWS: usize = 64;

/// Validate and classify every canonical-u64 gadget without changing the
/// production lowering plan or claiming any source row.
pub fn audit_r1cs_gadget_native_canonical_u64(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    public_bit_columns: &[usize],
) -> Result<CanonicalU64Audit, GadgetNativeError> {
    validate_source_one(source)?;
    let (is_public, _) = validate_public_columns(source, public_bit_columns)?;
    let marks = validate_and_mark_trace(source, trace)?;
    reject_public_gadget_columns(&marks.gadget_columns, &is_public)?;
    marks.balanced_ternary.reject_public_columns(&is_public)?;
    let (linearly_derived, _) = select_linear_definition_columns(source, &is_public, &marks);
    marks.canonical_u64.report(source, trace, &linearly_derived)
}

/// Exclusive diagnostic category for one validated decomposition.
///
/// `EqualityLinked` takes precedence when every bit has an exact external
/// equality. The independent `field_linearly_derived` flag remains visible,
/// so a decomposition that has both properties is never hidden by the census.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalU64Classification {
    Direct,
    EqualityLinked,
    Linear,
}

/// Counts for one complete trace or one named emission stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CanonicalU64Census {
    pub total: usize,
    pub direct: usize,
    pub equality_linked: usize,
    pub linear: usize,
    pub field_linearly_derived: usize,
}

/// Exact classification and ownership for one validated source gadget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalU64AuditEntry {
    pub decomposition: usize,
    pub stage: &'static str,
    pub source_rows: Range<usize>,
    pub field_column: usize,
    pub classification: CanonicalU64Classification,
    pub equality_linked_bits: usize,
    pub field_linearly_derived: bool,
}

/// Per-stage decomposition census, aggregated by stable stage path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalU64StageCensus {
    pub stage: &'static str,
    pub census: CanonicalU64Census,
}

/// Diagnostic report for all canonical-u64 gadgets in one source relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalU64Audit {
    pub entries: Vec<CanonicalU64AuditEntry>,
    pub census: CanonicalU64Census,
    pub stages: Vec<CanonicalU64StageCensus>,
}

#[derive(Clone, Debug)]
struct ValidatedEntry {
    stage: &'static str,
}

/// Validated trace roles retained by the parent lowering.
///
/// This type deliberately has no row-claiming or alias-planning method.
pub(super) struct ValidatedCanonicalU64 {
    entries: Vec<ValidatedEntry>,
}

impl ValidatedCanonicalU64 {
    pub(super) fn validate(
        source: &R1csSnapshot,
        trace: &R1csEncodingTrace,
        covered_rows: &[bool],
    ) -> Result<Self, GadgetNativeError> {
        if trace.canonical_u64_decompositions().is_empty() {
            return Ok(Self { entries: Vec::new() });
        }
        let stages = stage_ranges(source, trace)?;
        let mut owned_columns = HashSet::new();
        let mut previous_end = 0usize;
        let mut entries = Vec::with_capacity(trace.canonical_u64_decompositions().len());

        for (index, event) in trace.canonical_u64_decompositions().iter().enumerate() {
            validate_geometry(source, index, event, &mut owned_columns)?;
            if index > 0 && event.source_rows.start < previous_end {
                return Err(geometry(index, "overlapping or out-of-order row schedule"));
            }
            if event.source_rows.clone().any(|row| covered_rows[row]) {
                return Err(geometry(index, "row overlap with a replacing trace"));
            }
            validate_rows(source, event)?;
            let stage = event_stage(index, event, &stages)?;
            entries.push(ValidatedEntry { stage: stage.label });
            previous_end = event.source_rows.end;
        }
        Ok(Self { entries })
    }

    pub(super) fn report(
        &self,
        source: &R1csSnapshot,
        trace: &R1csEncodingTrace,
        linearly_derived: &[bool],
    ) -> Result<CanonicalU64Audit, GadgetNativeError> {
        if linearly_derived.len() != source.cols() {
            return Err(geometry(0, "linear-classification width"));
        }
        let equality_links = equality_linked_bits(source, trace);
        let mut entries = Vec::with_capacity(self.entries.len());
        let mut census = CanonicalU64Census::default();
        let mut stages = BTreeMap::<&'static str, CanonicalU64Census>::new();

        for (index, ((event, validated), linked)) in trace
            .canonical_u64_decompositions()
            .iter()
            .zip(&self.entries)
            .zip(equality_links)
            .enumerate()
        {
            let field_linearly_derived = linearly_derived[event.field.col()];
            let equality_linked_bits = linked.iter().filter(|&&is_linked| is_linked).count();
            let classification = if equality_linked_bits == BIT_ROWS {
                CanonicalU64Classification::EqualityLinked
            } else if field_linearly_derived {
                CanonicalU64Classification::Linear
            } else {
                CanonicalU64Classification::Direct
            };
            record(&mut census, classification, field_linearly_derived);
            record(
                stages.entry(validated.stage).or_default(),
                classification,
                field_linearly_derived,
            );
            entries.push(CanonicalU64AuditEntry {
                decomposition: index,
                stage: validated.stage,
                source_rows: event.source_rows.clone(),
                field_column: event.field.col(),
                classification,
                equality_linked_bits,
                field_linearly_derived,
            });
        }
        Ok(CanonicalU64Audit {
            entries,
            census,
            stages: stages
                .into_iter()
                .map(|(stage, census)| CanonicalU64StageCensus { stage, census })
                .collect(),
        })
    }
}

#[derive(Clone, Copy)]
struct StageRange {
    label: &'static str,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
}

fn stage_ranges(source: &R1csSnapshot, trace: &R1csEncodingTrace) -> Result<Vec<StageRange>, GadgetNativeError> {
    let checkpoints = trace.stages();
    if checkpoints.len() < 2
        || checkpoints[0].row != 0
        || checkpoints[0].col != 1
        || checkpoints
            .last()
            .is_none_or(|last| last.row != source.rows() || last.col != source.cols())
    {
        return Err(GadgetNativeError::CanonicalU64StageSchedule { detail: "boundary" });
    }
    let mut ranges = Vec::with_capacity(checkpoints.len() - 1);
    for pair in checkpoints.windows(2) {
        let (start, end) = (&pair[0], &pair[1]);
        if start.row > end.row || start.col > end.col {
            return Err(GadgetNativeError::CanonicalU64StageSchedule { detail: "order" });
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

fn event_stage<'a>(
    index: usize,
    event: &CanonicalU64TraceEntry,
    stages: &'a [StageRange],
) -> Result<&'a StageRange, GadgetNativeError> {
    let Some(stage) = stages
        .iter()
        .find(|stage| event.source_rows.start >= stage.row_start && event.source_rows.start < stage.row_end)
    else {
        return Err(geometry(index, "unowned stage rows"));
    };
    let first_fresh = event.bits[0].col();
    if event.source_rows.end > stage.row_end || first_fresh < stage.col_start || event.inverse.col() >= stage.col_end {
        return Err(geometry(index, "cross-stage row or column schedule"));
    }
    Ok(stage)
}

fn validate_geometry(
    source: &R1csSnapshot,
    index: usize,
    event: &CanonicalU64TraceEntry,
    owned_columns: &mut HashSet<usize>,
) -> Result<(), GadgetNativeError> {
    if event.source_rows.len() != SOURCE_ROWS || event.source_rows.end > source.rows() {
        return Err(geometry(index, "69-row schedule"));
    }
    let field = event.field.col();
    let first_bit = event.bits[0].col();
    if field == Var::ONE.col() || field >= source.cols() || first_bit <= field {
        return Err(geometry(index, "field column"));
    }
    for (bit, variable) in event.bits.iter().enumerate() {
        if variable.col() != first_bit + bit {
            return Err(geometry(index, "bit-column schedule"));
        }
    }
    if event.high_is_max.col() != first_bit + BIT_ROWS || event.inverse.col() != first_bit + BIT_ROWS + 1 {
        return Err(geometry(index, "auxiliary-column schedule"));
    }
    for column in std::iter::once(field)
        .chain(event.bits.iter().map(|bit| bit.col()))
        .chain([event.high_is_max.col(), event.inverse.col()])
    {
        if column == Var::ONE.col() || column >= source.cols() || !owned_columns.insert(column) {
            return Err(geometry(index, "duplicate or out-of-range role column"));
        }
    }
    Ok(())
}

fn validate_rows(source: &R1csSnapshot, event: &CanonicalU64TraceEntry) -> Result<(), GadgetNativeError> {
    for (offset, &bit) in event.bits.iter().enumerate() {
        let value = Lc::from_var(bit);
        let minus_one = value.clone().add_scaled(&Lc::from_const(F::ONE), -F::ONE);
        validate_row(
            source,
            "canonical-u64 bit alphabet",
            event.source_rows.start + offset,
            &value,
            &minus_one,
            &Lc::zero(),
        )?;
    }

    let mut reconstruction = Lc::from_var(event.field);
    let mut power = F::ONE;
    for &bit in &event.bits {
        reconstruction.add_term(bit, -power);
        power += power;
    }
    validate_row(
        source,
        "canonical-u64 recomposition",
        event.source_rows.start + BIT_ROWS,
        &reconstruction,
        &Lc::from_var(Var::ONE),
        &Lc::zero(),
    )?;

    let high_flag = Lc::from_var(event.high_is_max);
    let high_minus_one = high_flag
        .clone()
        .add_scaled(&Lc::from_const(F::ONE), -F::ONE);
    validate_row(
        source,
        "canonical-u64 high flag alphabet",
        event.source_rows.start + 65,
        &high_flag,
        &high_minus_one,
        &Lc::zero(),
    )?;

    let mut high = Lc::zero();
    let mut high_power = F::ONE;
    for &bit in &event.bits[32..] {
        high.add_term(bit, high_power);
        high_power += high_power;
    }
    let high_max = F::from_u64(0xffff_ffff);
    let high_diff = high.add_scaled(&Lc::from_const(high_max), -F::ONE);
    validate_row(
        source,
        "canonical-u64 high flag forward",
        event.source_rows.start + 66,
        &high_flag,
        &high_diff,
        &Lc::zero(),
    )?;
    let one_minus_flag = Lc::from_const(F::ONE).add_scaled(&high_flag, -F::ONE);
    validate_row(
        source,
        "canonical-u64 high flag reverse",
        event.source_rows.start + 67,
        &high_diff,
        &Lc::from_var(event.inverse),
        &one_minus_flag,
    )?;

    let mut low = Lc::zero();
    let mut low_power = F::ONE;
    for &bit in &event.bits[..32] {
        low.add_term(bit, low_power);
        low_power += low_power;
    }
    validate_row(
        source,
        "canonical-u64 canonical bound",
        event.source_rows.start + 68,
        &high_flag,
        &low,
        &Lc::zero(),
    )
}

fn equality_linked_bits(source: &R1csSnapshot, trace: &R1csEncodingTrace) -> Vec<[bool; BIT_ROWS]> {
    let mut bit_owner = HashMap::<usize, (usize, usize)>::new();
    let mut role_owner = HashMap::<usize, usize>::new();
    for (event_index, event) in trace.canonical_u64_decompositions().iter().enumerate() {
        role_owner.insert(event.field.col(), event_index);
        role_owner.insert(event.high_is_max.col(), event_index);
        role_owner.insert(event.inverse.col(), event_index);
        for (bit_index, bit) in event.bits.iter().enumerate() {
            bit_owner.insert(bit.col(), (event_index, bit_index));
            role_owner.insert(bit.col(), event_index);
        }
    }
    let mut linked = vec![[false; BIT_ROWS]; trace.canonical_u64_decompositions().len()];
    for row in 0..source.rows() {
        let Some((left, right)) = exact_equality_pair(source, row) else {
            continue;
        };
        for (bit_column, other) in [(left, right), (right, left)] {
            let Some(&(event, bit)) = bit_owner.get(&bit_column) else {
                continue;
            };
            if role_owner.get(&other).copied() != Some(event) {
                linked[event][bit] = true;
            }
        }
    }
    linked
}

fn exact_equality_pair(source: &R1csSnapshot, row: usize) -> Option<(usize, usize)> {
    if !source.c_row(row).is_empty() {
        return None;
    }
    if source.b_row(row) == [(Var::ONE.col(), F::ONE)] {
        equality_difference(source.a_row(row))
    } else if source.a_row(row) == [(Var::ONE.col(), F::ONE)] {
        equality_difference(source.b_row(row))
    } else {
        None
    }
}

fn equality_difference(row: &[(usize, F)]) -> Option<(usize, usize)> {
    let [(left_column, left), (right_column, right)] = row else {
        return None;
    };
    if *left_column == Var::ONE.col() || *right_column == Var::ONE.col() {
        return None;
    }
    if *left == F::ONE && *right == -F::ONE {
        Some((*left_column, *right_column))
    } else if *left == -F::ONE && *right == F::ONE {
        Some((*right_column, *left_column))
    } else {
        None
    }
}

fn record(census: &mut CanonicalU64Census, classification: CanonicalU64Classification, field_linear: bool) {
    census.total += 1;
    match classification {
        CanonicalU64Classification::Direct => census.direct += 1,
        CanonicalU64Classification::EqualityLinked => census.equality_linked += 1,
        CanonicalU64Classification::Linear => census.linear += 1,
    }
    census.field_linearly_derived += usize::from(field_linear);
}

fn geometry(decomposition: usize, detail: &'static str) -> GadgetNativeError {
    GadgetNativeError::CanonicalU64Geometry { decomposition, detail }
}
