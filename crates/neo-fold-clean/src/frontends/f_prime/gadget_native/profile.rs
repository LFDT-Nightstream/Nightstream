//! Stage attribution for the exact gadget-native `enc(F')` estimate.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::engine::r1cs_circuit::{R1csEncodingTrace, R1csSnapshot};

use super::{
    linear_definition_candidate, reject_public_gadget_columns, validate_and_mark_trace, validate_public_columns,
    validate_source_one, GadgetNativeError, GadgetNativeEstimate, CANONICAL_SLOT_WIDTH, TOOM_COEFFICIENTS,
    TOOM_EVALUATIONS,
};

const CANONICALITY_ROWS: usize = 32;
const K_MUL_ROWS: usize = 2;
const RING_MUL_ROWS: usize = TOOM_EVALUATIONS * TOOM_COEFFICIENTS + 54;

/// Exact contribution of one sequential circuit-emission stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GadgetNativeStageEstimate {
    pub label: &'static str,
    pub source_rows: usize,
    pub source_cols: usize,
    pub one_bit_source_cols: usize,
    pub canonical_field_source_cols: usize,
    pub linearly_derived_source_cols: usize,
    pub gadget_derived_source_cols: usize,
    pub synthetic_ring_fields: usize,
    /// Low-norm columns contributed by this stage; excludes the global ONE.
    pub encoded_cols: usize,
    pub encoded_rows: usize,
    pub fallback_source_rows: usize,
    pub poseidon_permutations: usize,
    pub poseidon_hash_permutations: usize,
    pub poseidon_hashes: usize,
    pub sboxes: usize,
    pub k_muls: usize,
    pub ring_muls: usize,
    /// `input fields -> (calls, permutations)` for one-shot Poseidon hashes.
    pub hash_histogram: BTreeMap<usize, (usize, usize)>,
}

/// Stage breakdown plus the reconciled whole-branch estimate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GadgetNativeStageProfile {
    pub total: GadgetNativeEstimate,
    pub stages: Vec<GadgetNativeStageEstimate>,
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
    validate_source_one(source)?;
    let (is_public, explicit_bits) = validate_public_columns(source, public_bit_columns)?;
    let marks = validate_and_mark_trace(source, trace)?;
    reject_public_gadget_columns(&marks.gadget_columns, &is_public)?;
    let (linear_columns, removed_rows) = select_linear_rows(source, &is_public, &marks);
    let ranges = stage_ranges(source, trace)?;
    let mut stages = ranges
        .iter()
        .map(|range| GadgetNativeStageEstimate {
            label: range.label,
            source_rows: range.row_end - range.row_start,
            source_cols: range.col_end - range.col_start,
            one_bit_source_cols: 0,
            canonical_field_source_cols: 0,
            linearly_derived_source_cols: 0,
            gadget_derived_source_cols: 0,
            synthetic_ring_fields: 0,
            encoded_cols: 0,
            encoded_rows: 0,
            fallback_source_rows: 0,
            poseidon_permutations: 0,
            poseidon_hash_permutations: 0,
            poseidon_hashes: 0,
            sboxes: 0,
            k_muls: 0,
            ring_muls: 0,
            hash_histogram: BTreeMap::new(),
        })
        .collect::<Vec<_>>();

    for (range, stage) in ranges.iter().zip(&mut stages) {
        for column in range.col_start..range.col_end {
            if marks.gadget_columns[column] {
                stage.gadget_derived_source_cols += 1;
            } else if linear_columns[column] {
                stage.linearly_derived_source_cols += 1;
            } else if is_public[column] || explicit_bits[column] {
                stage.one_bit_source_cols += 1;
            } else {
                stage.canonical_field_source_cols += 1;
            }
        }
        stage.fallback_source_rows = (range.row_start..range.row_end)
            .filter(|&row| !marks.covered_rows[row] && !removed_rows[row])
            .count();
    }

    for event in trace.sbox7() {
        let stage = event_stage(
            &ranges,
            event.source_rows.start,
            event.source_rows.end,
            "Poseidon2 sbox7",
        )?;
        stages[stage].sboxes += 1;
    }
    for event in trace.k_muls() {
        let stage = event_stage(
            &ranges,
            event.source_rows.start,
            event.source_rows.end,
            "K multiplication",
        )?;
        stages[stage].k_muls += 1;
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
        let field_slots = stage.canonical_field_source_cols + stage.synthetic_ring_fields;
        stage.encoded_cols = stage.one_bit_source_cols + field_slots * CANONICAL_SLOT_WIDTH;
        stage.encoded_rows = stage.encoded_cols
            + field_slots * CANONICALITY_ROWS
            + stage.fallback_source_rows
            + stage.sboxes
            + stage.k_muls * K_MUL_ROWS
            + stage.ring_muls * RING_MUL_ROWS;
    }

    let sum = |f: fn(&GadgetNativeStageEstimate) -> usize| stages.iter().map(f).sum::<usize>();
    let encoded_cols = 1 + sum(|stage| stage.encoded_cols);
    let total = GadgetNativeEstimate {
        source_rows: source.rows(),
        source_cols: source.cols(),
        public_input_len: 1 + public_bit_columns.len(),
        encoded_cols,
        encoded_rows: sum(|stage| stage.encoded_rows),
        max_degree: 8,
        one_bit_source_cols: sum(|stage| stage.one_bit_source_cols),
        canonical_field_source_cols: sum(|stage| stage.canonical_field_source_cols),
        synthetic_ring_fields: sum(|stage| stage.synthetic_ring_fields),
        linearly_derived_source_cols: sum(|stage| stage.linearly_derived_source_cols),
        gadget_derived_source_cols: sum(|stage| stage.gadget_derived_source_cols),
        fallback_source_rows: sum(|stage| stage.fallback_source_rows),
    };
    Ok(GadgetNativeStageProfile { total, stages })
}

fn select_linear_rows(source: &R1csSnapshot, is_public: &[bool], marks: &super::TraceMarks) -> (Vec<bool>, Vec<bool>) {
    let mut defined = marks.gadget_columns.clone();
    let mut columns = vec![false; source.cols()];
    let mut rows = vec![false; source.rows()];
    for row in 0..source.rows() {
        if marks.covered_rows[row] {
            continue;
        }
        if let Some((column, _)) = linear_definition_candidate(source, row, is_public, &defined) {
            defined[column] = true;
            columns[column] = true;
            rows[row] = true;
        }
    }
    (columns, rows)
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
