//! Exact normalized port audit for the retained PiRLC carry block.
//!
//! Owns exhaustive comparison of both source parity matrices with the
//! independent challenge-carry recipe and of all retained normalized rows
//! with the canonical radix images of that recipe. It does not own range
//! enforcement, assignment values, selector authority, or lifecycle state.

use std::ops::Range;

use neo_ccs::CcsMatrix;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use crate::frontends::r1cs_f_prime::{
    prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix, SelectiveSourceRowDisposition,
};

use super::retained_algebra::{append_radix_image, canonical_terms, Term};
use super::{
    production_pi_rlc_family_body_source_arms, NebulaFPrimePiRlcFamilyRelationError, LANE_COUNT,
    REPLAY_AUXILIARY_START, SOURCE_COUNT,
};

const SCHEMA_VERSION: u64 = 1;
const CHALLENGE_FIELDS: usize = SOURCE_COUNT * LANE_COUNT;

const LOCAL_CHALLENGE_START: usize = 1;
const BEFORE_CHALLENGE_START: usize = 163_826;
const AFTER_CHALLENGE_START: usize = 164_744;
const BEFORE_CURSOR_COLUMN: usize = 165_662;
const AFTER_CURSOR_COLUMN: usize = 165_663;
const LOCAL_COLUMNS: usize = 165_664;

const SOURCE_COLUMN_SHIFT: usize = 640;
const SOURCE_CHALLENGE_START: usize = LOCAL_CHALLENGE_START + SOURCE_COLUMN_SHIFT;
const SOURCE_BEFORE_CHALLENGE_START: usize = BEFORE_CHALLENGE_START + SOURCE_COLUMN_SHIFT;
const SOURCE_AFTER_CHALLENGE_START: usize = AFTER_CHALLENGE_START + SOURCE_COLUMN_SHIFT;
const SOURCE_BEFORE_CURSOR_COLUMN: usize = BEFORE_CURSOR_COLUMN + SOURCE_COLUMN_SHIFT;
const SOURCE_AFTER_CURSOR_COLUMN: usize = AFTER_CURSOR_COLUMN + SOURCE_COLUMN_SHIFT;

const SOURCE_ROW_START: usize = 163_609;
const DECODE_ROWS: usize = CHALLENGE_FIELDS;
const CHALLENGE_ROWS: usize = CHALLENGE_FIELDS;
const CARRY_ROWS: usize = DECODE_ROWS + CHALLENGE_ROWS + 1;

const FINAL_ROWS: usize = 491_046;
const FINAL_COLUMNS: usize = 8_858_862;
const SELECTOR_COLUMNS: [usize; 2] = [648, 649];
const EMITTED_STARTS: [usize; 2] = [69_607, 305_118];
const FINAL_CHALLENGE_START: usize = 702;
const DIRECT_SOURCE_START: usize = 164_140;
const FINAL_DIRECT_START: usize = 2_129_045;
const GENERAL_WIDTH: usize = 41;

const PORT_COUNT: usize = 13;
const GENERAL_SELECTOR_PORT: usize = 1;
const A_PORT: usize = 2;
const B_PORT: usize = 3;
const C_PORT: usize = 4;

type SourceRow = [Vec<Term>; 3];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcBodyCarryRetainedAudit {
    schema_version: u64,
    source_row_start: usize,
    source_rows: usize,
    local_columns: usize,
    source_column_shift: usize,
    final_rows: usize,
    final_columns: usize,
    selector_columns: [usize; 2],
    emitted_starts: [usize; 2],
    source_starts: [usize; 5],
    final_starts: [usize; 5],
    widths: [usize; 5],
    radices: [u64; 5],
    source_nnz: [usize; 3],
    final_port_nnz: [usize; PORT_COUNT],
}

impl NebulaFPrimePiRlcBodyCarryRetainedAudit {
    pub const fn schema_version(&self) -> u64 {
        self.schema_version
    }

    pub const fn source_row_start(&self) -> usize {
        self.source_row_start
    }

    pub const fn source_rows(&self) -> usize {
        self.source_rows
    }

    pub const fn local_columns(&self) -> usize {
        self.local_columns
    }

    pub const fn source_column_shift(&self) -> usize {
        self.source_column_shift
    }

    pub const fn final_rows(&self) -> usize {
        self.final_rows
    }

    pub const fn final_columns(&self) -> usize {
        self.final_columns
    }

    pub const fn selector_columns(&self) -> [usize; 2] {
        self.selector_columns
    }

    pub const fn emitted_starts(&self) -> [usize; 2] {
        self.emitted_starts
    }

    pub const fn source_starts(&self) -> [usize; 5] {
        self.source_starts
    }

    pub const fn final_starts(&self) -> [usize; 5] {
        self.final_starts
    }

    pub const fn widths(&self) -> [usize; 5] {
        self.widths
    }

    pub const fn radices(&self) -> [u64; 5] {
        self.radices
    }

    pub const fn source_nnz(&self) -> [usize; 3] {
        self.source_nnz
    }

    pub const fn final_port_nnz(&self) -> [usize; PORT_COUNT] {
        self.final_port_nnz
    }
}

fn carry_error(reason: impl Into<String>) -> NebulaFPrimePiRlcFamilyRelationError {
    NebulaFPrimePiRlcFamilyRelationError::CarryRetained(reason.into())
}

fn source_range() -> Range<usize> {
    SOURCE_ROW_START..SOURCE_ROW_START + CARRY_ROWS
}

fn expected_source_row(row: usize) -> SourceRow {
    if row < DECODE_ROWS {
        return [
            canonical_terms(vec![
                (0, F::from_u64(2)),
                (SOURCE_CHALLENGE_START + row, -F::ONE),
                (SOURCE_BEFORE_CHALLENGE_START + row, F::ONE),
            ]),
            vec![(0, F::ONE)],
            Vec::new(),
        ];
    }
    if row < DECODE_ROWS + CHALLENGE_ROWS {
        let challenge = row - DECODE_ROWS;
        return [
            canonical_terms(vec![
                (SOURCE_BEFORE_CHALLENGE_START + challenge, -F::ONE),
                (SOURCE_AFTER_CHALLENGE_START + challenge, F::ONE),
            ]),
            vec![(0, F::ONE)],
            Vec::new(),
        ];
    }
    [
        canonical_terms(vec![
            (0, -F::ONE),
            (SOURCE_BEFORE_CURSOR_COLUMN, -F::ONE),
            (SOURCE_AFTER_CURSOR_COLUMN, F::ONE),
        ]),
        vec![(0, F::ONE)],
        Vec::new(),
    ]
}

fn source_rows_in_range(matrix: &CcsMatrix<F>) -> Result<Vec<Vec<Term>>, NebulaFPrimePiRlcFamilyRelationError> {
    let selected = source_range();
    let intersects = |start: usize, stop: usize| start < selected.end && selected.start < stop;
    if matrix
        .seeded_phi81_blocks()
        .iter()
        .any(|block| intersects(block.row_start(), block.row_end()))
        || matrix
            .geometric_runs()
            .iter()
            .any(|run| selected.contains(&run.row()))
    {
        return Err(carry_error(
            "compact source matrix content intersects the retained carry rows",
        ));
    }
    let mut rows = vec![Vec::new(); CARRY_ROWS];
    match matrix {
        CcsMatrix::Identity { n } => {
            if *n < selected.end {
                return Err(carry_error("source identity matrix is shorter than the carry block"));
            }
            for row in selected {
                rows[row - SOURCE_ROW_START].push((row, F::ONE));
            }
        }
        CcsMatrix::Csc(_) | CcsMatrix::CscWithSeededPhi81 { .. } => {
            let csc = matrix
                .sparse_component()
                .ok_or_else(|| carry_error("source matrix has no sparse component"))?;
            if !csc.is_canonical() {
                return Err(carry_error("source matrix CSC is not canonical"));
            }
            for column in 0..csc.ncols {
                for entry in csc.column_range(column) {
                    let row = csc.row_index(entry);
                    if selected.contains(&row) {
                        rows[row - SOURCE_ROW_START].push((column, csc.vals[entry]));
                    }
                }
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            return Err(carry_error("source matrix content is unavailable"));
        }
    }
    Ok(rows)
}

fn selected_row_index(row: usize) -> Option<usize> {
    EMITTED_STARTS.iter().enumerate().find_map(|(arm, &start)| {
        (start..start + CARRY_ROWS)
            .contains(&row)
            .then_some(arm * CARRY_ROWS + row - start)
    })
}

fn selected_final_rows(matrix: &CcsMatrix<F>) -> Result<Vec<Vec<Term>>, NebulaFPrimePiRlcFamilyRelationError> {
    let intersects = |start: usize, stop: usize| {
        EMITTED_STARTS
            .iter()
            .any(|&selected| start < selected + CARRY_ROWS && selected < stop)
    };
    if matrix
        .seeded_phi81_blocks()
        .iter()
        .any(|block| intersects(block.row_start(), block.row_end()))
        || matrix
            .geometric_runs()
            .iter()
            .any(|run| selected_row_index(run.row()).is_some())
    {
        return Err(carry_error(
            "compact final matrix content intersects retained carry rows",
        ));
    }
    let mut rows = vec![Vec::new(); 2 * CARRY_ROWS];
    match matrix {
        CcsMatrix::Identity { n } => {
            if *n < FINAL_ROWS {
                return Err(carry_error("final identity matrix has the wrong row domain"));
            }
            for &start in &EMITTED_STARTS {
                for row in start..start + CARRY_ROWS {
                    rows[selected_row_index(row).expect("selected identity row")].push((row, F::ONE));
                }
            }
        }
        CcsMatrix::Csc(_) | CcsMatrix::CscWithSeededPhi81 { .. } => {
            let csc = matrix
                .sparse_component()
                .ok_or_else(|| carry_error("final matrix has no sparse component"))?;
            if !csc.is_canonical() {
                return Err(carry_error("final matrix CSC is not canonical"));
            }
            for column in 0..csc.ncols {
                for entry in csc.column_range(column) {
                    if let Some(index) = selected_row_index(csc.row_index(entry)) {
                        rows[index].push((column, csc.vals[entry]));
                    }
                }
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            return Err(carry_error("final matrix content is unavailable"));
        }
    }
    Ok(rows)
}

fn source_slot(column: usize) -> Option<(usize, usize)> {
    if (SOURCE_CHALLENGE_START..SOURCE_CHALLENGE_START + CHALLENGE_FIELDS).contains(&column) {
        Some((
            FINAL_CHALLENGE_START + (column - SOURCE_CHALLENGE_START) * GENERAL_WIDTH,
            GENERAL_WIDTH,
        ))
    } else if (SOURCE_BEFORE_CHALLENGE_START..=SOURCE_AFTER_CURSOR_COLUMN).contains(&column) {
        Some((
            FINAL_DIRECT_START + (column - DIRECT_SOURCE_START) * GENERAL_WIDTH,
            GENERAL_WIDTH,
        ))
    } else {
        None
    }
}

fn final_image(source: &[Term]) -> Result<Vec<Term>, NebulaFPrimePiRlcFamilyRelationError> {
    let mut terms = Vec::new();
    for &(column, coefficient) in source {
        if column == 0 {
            terms.push((0, coefficient));
        } else {
            let (start, width) = source_slot(column)
                .ok_or_else(|| carry_error("carry source term is outside the declared normalized slots"))?;
            append_radix_image(&mut terms, start, width, coefficient);
        }
    }
    Ok(canonical_terms(terms))
}

fn expected_final_port(arm: usize, row: usize, port: usize) -> Result<Vec<Term>, NebulaFPrimePiRlcFamilyRelationError> {
    match port {
        GENERAL_SELECTOR_PORT => Ok(vec![(SELECTOR_COLUMNS[arm], F::ONE)]),
        A_PORT => final_image(&expected_source_row(row)[0]),
        B_PORT => final_image(&expected_source_row(row)[1]),
        C_PORT => final_image(&expected_source_row(row)[2]),
        _ => Ok(Vec::new()),
    }
}

pub fn production_pi_rlc_family_body_carry_retained_audit(
) -> Result<NebulaFPrimePiRlcBodyCarryRetainedAudit, NebulaFPrimePiRlcFamilyRelationError> {
    let arms = production_pi_rlc_family_body_source_arms()?;
    if arms.len() != 2
        || arms
            .iter()
            .any(|arm| arm.n < SOURCE_ROW_START + CARRY_ROWS || arm.m <= SOURCE_AFTER_CURSOR_COLUMN)
    {
        return Err(carry_error(
            "source parity shape does not contain the complete carry block",
        ));
    }

    let mut source_nnz = [0usize; 3];
    for (arm_index, arm) in arms.iter().enumerate() {
        for (port, matrix) in [&arm.a, &arm.b, &arm.c].into_iter().enumerate() {
            let actual = source_rows_in_range(matrix)?;
            let mut count = 0;
            for (row, terms) in actual.iter().enumerate() {
                let expected = expected_source_row(row)[port].clone();
                if *terms != expected {
                    return Err(carry_error(format!(
                        "source parity matrix differs from the exact carry recipe: port={port}, arm={arm_index}, source_row={}",
                        SOURCE_ROW_START + row,
                    )));
                }
                count += terms.len();
            }
            if arm_index == 0 {
                source_nnz[port] = count;
            } else if source_nnz[port] != count {
                return Err(carry_error("source parity carry matrices have different sparsity"));
            }
        }
    }

    let relation = prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        arms,
        REPLAY_AUXILIARY_START - 1,
        0,
        D,
        0,
        crate::config::B_BASE,
    )?
    .finish()?;
    if relation.structure().n != FINAL_ROWS
        || relation.structure().m != FINAL_COLUMNS
        || relation.selector_cols() != SELECTOR_COLUMNS
        || relation.structure().matrices.len() != PORT_COUNT
    {
        return Err(carry_error("normalized relation has the wrong carry audit shape"));
    }
    let compiler = relation
        .selective_compiler_audit()
        .ok_or_else(|| carry_error("normalized relation omitted its compiler audit"))?;
    let expected_source = source_range();
    for (arm, (&emitted_start, mapping)) in EMITTED_STARTS
        .iter()
        .zip(compiler.rows().arms())
        .enumerate()
    {
        let retained = mapping
            .source_runs()
            .iter()
            .find(|run| run.source_rows() == expected_source);
        let Some(retained) = retained else {
            return Err(carry_error("compiler audit omitted the retained carry run"));
        };
        if retained.disposition() != SelectiveSourceRowDisposition::Retained
            || retained.emitted_start() != Some(emitted_start)
            || arm > 1
        {
            return Err(carry_error("compiler ledger differs from the retained carry interval"));
        }
    }

    let mut final_port_nnz = [0usize; PORT_COUNT];
    for (port, matrix) in relation.structure().matrices.iter().enumerate() {
        let actual = selected_final_rows(matrix)?;
        for arm in 0..2 {
            for row in 0..CARRY_ROWS {
                let terms = &actual[arm * CARRY_ROWS + row];
                let expected = expected_final_port(arm, row, port)?;
                if *terms != expected {
                    let first_difference = terms
                        .iter()
                        .zip(expected.iter())
                        .position(|(actual, expected)| actual != expected);
                    return Err(carry_error(format!(
                        "normalized port differs from the exact carry source image: port={port}, arm={arm}, source_row={}, actual_len={}, expected_len={}, first_difference={first_difference:?}, actual_first={:?}, expected_first={:?}",
                        SOURCE_ROW_START + row,
                        terms.len(),
                        expected.len(),
                        terms.get(first_difference.unwrap_or(0)),
                        expected.get(first_difference.unwrap_or(0)),
                    )));
                }
                final_port_nnz[port] += terms.len();
            }
        }
    }

    Ok(NebulaFPrimePiRlcBodyCarryRetainedAudit {
        schema_version: SCHEMA_VERSION,
        source_row_start: SOURCE_ROW_START,
        source_rows: CARRY_ROWS,
        local_columns: LOCAL_COLUMNS,
        source_column_shift: SOURCE_COLUMN_SHIFT,
        final_rows: FINAL_ROWS,
        final_columns: FINAL_COLUMNS,
        selector_columns: SELECTOR_COLUMNS,
        emitted_starts: EMITTED_STARTS,
        source_starts: [
            SOURCE_CHALLENGE_START,
            SOURCE_BEFORE_CHALLENGE_START,
            SOURCE_AFTER_CHALLENGE_START,
            SOURCE_BEFORE_CURSOR_COLUMN,
            SOURCE_AFTER_CURSOR_COLUMN,
        ],
        final_starts: [
            FINAL_CHALLENGE_START,
            FINAL_DIRECT_START + (SOURCE_BEFORE_CHALLENGE_START - DIRECT_SOURCE_START) * GENERAL_WIDTH,
            FINAL_DIRECT_START + (SOURCE_AFTER_CHALLENGE_START - DIRECT_SOURCE_START) * GENERAL_WIDTH,
            FINAL_DIRECT_START + (SOURCE_BEFORE_CURSOR_COLUMN - DIRECT_SOURCE_START) * GENERAL_WIDTH,
            FINAL_DIRECT_START + (SOURCE_AFTER_CURSOR_COLUMN - DIRECT_SOURCE_START) * GENERAL_WIDTH,
        ],
        widths: [GENERAL_WIDTH; 5],
        radices: [3; 5],
        source_nnz,
        final_port_nnz,
    })
}

const _: () = assert!(CARRY_ROWS == 1_837);
const _: () = assert!(SOURCE_ROW_START + CARRY_ROWS == 165_446);
const _: () = assert!(SOURCE_AFTER_CURSOR_COLUMN == 166_303);
