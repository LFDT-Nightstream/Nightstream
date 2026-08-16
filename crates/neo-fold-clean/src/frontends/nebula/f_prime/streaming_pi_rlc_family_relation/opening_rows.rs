//! Exact normalized row audit for the PiRLC input openings.
//!
//! Owns the active digit domain rows, the retained zero-word rows, and every
//! rewritten two-trit canonical row in both parity arms. It does not own the
//! outer norm premise, assignment values, or semantic canonicality.

use neo_ccs::CcsMatrix;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::frontends::r1cs_f_prime::{
    prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix, SelectiveEmittedRowFamily,
    SelectiveRewriteKind,
};

use super::{production_pi_rlc_family_body_source_arms, NebulaFPrimePiRlcFamilyRelationError, REPLAY_AUXILIARY_START};

const SCHEMA_VERSION: u64 = 1;
const ARM_COUNT: usize = 2;
const OPENING_COUNT: usize = 810;
const DIGIT_COUNT: usize = 41;
const BORROW_COUNT: usize = 20;
const CHUNK_COUNT: usize = 21;
const SOURCE_ZERO_ROW_START: usize = 43_794;
const SOURCE_ZERO_DIGIT_START: usize = 46_055;
const SOURCE_FIELD_START: usize = 1_451;
const SOURCE_DIGIT_START: usize = 46_096;
const SOURCE_DIGIT_STRIDE: usize = 122;
const SOURCE_CANONICAL_ROW_START: usize = 43_835;
const SOURCE_CANONICAL_ROW_STRIDE: usize = 124;
const CENTERED_ROW_START: usize = 2;
const CENTERED_ROW_COUNT: usize = OPENING_COUNT * DIGIT_COUNT / 2;
const ZERO_EMITTED_STARTS: [usize; ARM_COUNT] = [78_090, 202_066];
const CANONICAL_EMITTED_STARTS: [usize; ARM_COUNT] = [141_262, 265_410];
const SELECTOR_COLUMNS: [usize; ARM_COUNT] = [648, 649];
const FINAL_DIGIT_START: usize = 19_332;
const FINAL_DIGIT_STRIDE: usize = DIGIT_COUNT;
const FINAL_ZERO_START: usize = 1_059_804;
const FINAL_BORROW_START: usize = 1_059_845;
const FINAL_BORROW_STRIDE: usize = BORROW_COUNT;
const FINAL_ROWS: usize = 282_459;
const FINAL_COLUMNS: usize = 2_521_314;
const PORT_COUNT: usize = 13;
const BIT_PORT: usize = 0;
const GENERAL_SELECTOR_PORT: usize = 1;
const A_PORT: usize = 2;
const B_PORT: usize = 3;
const C_PORT: usize = 4;
const SBOX_INPUT_PORT: usize = 5;
const CENTERED_UNIT_PORT: usize = 6;
const EVALUATION_SELECTOR_PORT: usize = 7;
const CLASS_SELECTOR_START: usize = 8;

type Term = (usize, F);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcBodyOpeningRowsAudit {
    schema_version: u64,
    arm_count: usize,
    opening_count: usize,
    digit_count: usize,
    borrow_count: usize,
    chunk_count: usize,
    source_zero_row_start: usize,
    source_zero_digit_start: usize,
    source_field_start: usize,
    source_digit_start: usize,
    source_digit_stride: usize,
    source_canonical_row_start: usize,
    source_canonical_row_stride: usize,
    centered_row_start: usize,
    centered_row_count: usize,
    zero_emitted_starts: [usize; ARM_COUNT],
    canonical_emitted_starts: [usize; ARM_COUNT],
    selector_columns: [usize; ARM_COUNT],
    final_digit_start: usize,
    final_digit_stride: usize,
    final_zero_start: usize,
    final_borrow_start: usize,
    final_borrow_stride: usize,
    final_rows: usize,
    final_columns: usize,
    normalized_chunk_bounds: [usize; CHUNK_COUNT],
    complemented_chunks: [bool; CHUNK_COUNT],
    source_zero_nnz: [usize; 3],
    final_port_nnz: [usize; PORT_COUNT],
}

macro_rules! audit_getter {
    ($name:ident, $ty:ty) => {
        pub const fn $name(&self) -> $ty {
            self.$name
        }
    };
}

impl NebulaFPrimePiRlcBodyOpeningRowsAudit {
    audit_getter!(schema_version, u64);
    audit_getter!(arm_count, usize);
    audit_getter!(opening_count, usize);
    audit_getter!(digit_count, usize);
    audit_getter!(borrow_count, usize);
    audit_getter!(chunk_count, usize);
    audit_getter!(source_zero_row_start, usize);
    audit_getter!(source_zero_digit_start, usize);
    audit_getter!(source_field_start, usize);
    audit_getter!(source_digit_start, usize);
    audit_getter!(source_digit_stride, usize);
    audit_getter!(source_canonical_row_start, usize);
    audit_getter!(source_canonical_row_stride, usize);
    audit_getter!(centered_row_start, usize);
    audit_getter!(centered_row_count, usize);
    audit_getter!(zero_emitted_starts, [usize; ARM_COUNT]);
    audit_getter!(canonical_emitted_starts, [usize; ARM_COUNT]);
    audit_getter!(selector_columns, [usize; ARM_COUNT]);
    audit_getter!(final_digit_start, usize);
    audit_getter!(final_digit_stride, usize);
    audit_getter!(final_zero_start, usize);
    audit_getter!(final_borrow_start, usize);
    audit_getter!(final_borrow_stride, usize);
    audit_getter!(final_rows, usize);
    audit_getter!(final_columns, usize);
    audit_getter!(normalized_chunk_bounds, [usize; CHUNK_COUNT]);
    audit_getter!(complemented_chunks, [bool; CHUNK_COUNT]);
    audit_getter!(source_zero_nnz, [usize; 3]);
    audit_getter!(final_port_nnz, [usize; PORT_COUNT]);
}

fn opening_error(reason: impl Into<String>) -> NebulaFPrimePiRlcFamilyRelationError {
    NebulaFPrimePiRlcFamilyRelationError::OpeningRows(reason.into())
}

fn canonical_terms(mut terms: Vec<Term>) -> Vec<Term> {
    terms.sort_unstable_by_key(|term| term.0);
    let mut result: Vec<Term> = Vec::with_capacity(terms.len());
    for (column, coefficient) in terms {
        if let Some((last_column, last_coefficient)) = result.last_mut() {
            if *last_column == column {
                *last_coefficient += coefficient;
                if *last_coefficient == F::ZERO {
                    result.pop();
                }
                continue;
            }
        }
        if coefficient != F::ZERO {
            result.push((column, coefficient));
        }
    }
    result
}

fn rows_in_range(
    matrix: &CcsMatrix<F>,
    start: usize,
    length: usize,
) -> Result<Vec<Vec<Term>>, NebulaFPrimePiRlcFamilyRelationError> {
    let mut rows = vec![Vec::new(); length];
    match matrix {
        CcsMatrix::Identity { n } => {
            if start + length > *n {
                return Err(opening_error("identity matrix is shorter than the requested row range"));
            }
            for offset in 0..length {
                rows[offset].push((start + offset, F::ONE));
            }
        }
        CcsMatrix::Csc(_) | CcsMatrix::CscWithSeededPhi81 { .. } => {
            let csc = matrix
                .sparse_component()
                .ok_or_else(|| opening_error("matrix has no sparse component"))?;
            if !csc.is_canonical() {
                return Err(opening_error("matrix CSC is not canonical"));
            }
            for column in 0..csc.ncols {
                for entry in csc.column_range(column) {
                    let row = csc.row_index(entry);
                    if (start..start + length).contains(&row) {
                        rows[row - start].push((column, csc.vals[entry]));
                    }
                }
            }
        }
        CcsMatrix::VerifierArtifact { .. } => return Err(opening_error("matrix content is unavailable")),
    }
    Ok(rows)
}

fn selected_row_index(row: usize) -> Option<usize> {
    if (CENTERED_ROW_START..CENTERED_ROW_START + CENTERED_ROW_COUNT).contains(&row) {
        return Some(row - CENTERED_ROW_START);
    }
    let mut cursor = CENTERED_ROW_COUNT;
    for arm in 0..ARM_COUNT {
        if (ZERO_EMITTED_STARTS[arm]..ZERO_EMITTED_STARTS[arm] + DIGIT_COUNT).contains(&row) {
            return Some(cursor + row - ZERO_EMITTED_STARTS[arm]);
        }
        cursor += DIGIT_COUNT;
        let canonical_count = OPENING_COUNT * CHUNK_COUNT;
        if (CANONICAL_EMITTED_STARTS[arm]..CANONICAL_EMITTED_STARTS[arm] + canonical_count).contains(&row) {
            return Some(cursor + row - CANONICAL_EMITTED_STARTS[arm]);
        }
        cursor += canonical_count;
    }
    None
}

fn selected_rows(matrix: &CcsMatrix<F>) -> Result<Vec<Vec<Term>>, NebulaFPrimePiRlcFamilyRelationError> {
    let selected_count = CENTERED_ROW_COUNT + ARM_COUNT * (DIGIT_COUNT + OPENING_COUNT * CHUNK_COUNT);
    let mut rows = vec![Vec::new(); selected_count];
    match matrix {
        CcsMatrix::Identity { n } => {
            if *n < FINAL_ROWS {
                return Err(opening_error("final identity matrix has the wrong row domain"));
            }
            for row in 0..FINAL_ROWS {
                if let Some(index) = selected_row_index(row) {
                    rows[index].push((row, F::ONE));
                }
            }
        }
        CcsMatrix::Csc(_) | CcsMatrix::CscWithSeededPhi81 { .. } => {
            let csc = matrix
                .sparse_component()
                .ok_or_else(|| opening_error("final matrix has no sparse component"))?;
            if !csc.is_canonical() {
                return Err(opening_error("final matrix CSC is not canonical"));
            }
            for column in 0..csc.ncols {
                for entry in csc.column_range(column) {
                    if let Some(index) = selected_row_index(csc.row_index(entry)) {
                        rows[index].push((column, csc.vals[entry]));
                    }
                }
            }
        }
        CcsMatrix::VerifierArtifact { .. } => return Err(opening_error("final matrix content is unavailable")),
    }
    Ok(rows)
}

fn chunk_geometry() -> ([usize; CHUNK_COUNT], [bool; CHUNK_COUNT]) {
    let mut bound = F::ORDER_U64 - 1;
    let mut normalized = [0usize; CHUNK_COUNT];
    let mut complemented = [false; CHUNK_COUNT];
    for chunk in 0..CHUNK_COUNT {
        let digit = 2 * chunk;
        let first = bound % 3;
        bound /= 3;
        let second = if digit + 1 < DIGIT_COUNT {
            let value = bound % 3;
            bound /= 3;
            value
        } else {
            0
        };
        let chunk_bound = first + 3 * second;
        complemented[chunk] = chunk_bound > 4;
        normalized[chunk] = if complemented[chunk] {
            (8 - chunk_bound) as usize
        } else {
            chunk_bound as usize
        };
    }
    assert_eq!(bound, 0);
    (normalized, complemented)
}

fn expected_centered_port(row: usize, port: usize) -> Vec<Term> {
    let left = FINAL_DIGIT_START + 2 * row;
    let right = left + 1;
    match port {
        GENERAL_SELECTOR_PORT | EVALUATION_SELECTOR_PORT => vec![(0, F::ONE)],
        A_PORT => vec![(right, F::ONE)],
        CENTERED_UNIT_PORT => vec![(left, F::ONE)],
        _ => Vec::new(),
    }
}

fn expected_zero_port(arm: usize, digit: usize, port: usize) -> Vec<Term> {
    match port {
        GENERAL_SELECTOR_PORT => vec![(SELECTOR_COLUMNS[arm], F::ONE)],
        A_PORT => vec![(FINAL_ZERO_START + digit, F::ONE)],
        B_PORT => vec![(0, F::ONE)],
        _ => Vec::new(),
    }
}

fn expected_canonical_port(
    arm: usize,
    opening: usize,
    chunk: usize,
    port: usize,
    normalized: &[usize; CHUNK_COUNT],
    complemented: &[bool; CHUNK_COUNT],
) -> Vec<Term> {
    let scale = if complemented[chunk] { -F::ONE } else { F::ONE };
    let digit_start = FINAL_DIGIT_START + opening * FINAL_DIGIT_STRIDE;
    let borrow_start = FINAL_BORROW_START + opening * FINAL_BORROW_STRIDE;
    let mut terms = Vec::new();
    match port {
        BIT_PORT => {
            if chunk != 0 {
                terms.push((borrow_start + chunk - 1, scale));
            }
            if complemented[chunk] {
                terms.push((0, F::ONE));
            }
        }
        GENERAL_SELECTOR_PORT => terms.push((SELECTOR_COLUMNS[arm], F::ONE)),
        A_PORT => {
            let second = 2 * chunk + 1;
            if second < DIGIT_COUNT {
                terms.push((digit_start + second, scale));
            } else {
                terms.push((0, -scale));
            }
        }
        C_PORT | SBOX_INPUT_PORT => {
            if chunk + 1 != CHUNK_COUNT {
                terms.push((borrow_start + chunk, scale));
            }
            if complemented[chunk] {
                terms.push((0, F::ONE));
            }
        }
        CENTERED_UNIT_PORT => terms.push((digit_start + 2 * chunk, scale)),
        class if class == CLASS_SELECTOR_START + normalized[chunk] => {
            terms.push((SELECTOR_COLUMNS[arm], F::ONE));
        }
        _ => {}
    }
    canonical_terms(terms)
}

pub fn production_pi_rlc_family_body_opening_rows_audit(
) -> Result<NebulaFPrimePiRlcBodyOpeningRowsAudit, NebulaFPrimePiRlcFamilyRelationError> {
    let arms = production_pi_rlc_family_body_source_arms()?;
    if arms.len() != ARM_COUNT {
        return Err(opening_error("production body does not have two parity arms"));
    }

    let mut source_zero_nnz = [0usize; 3];
    for (arm_index, arm) in arms.iter().enumerate() {
        if arm.shifted_ternary_canonical_traces().len() != OPENING_COUNT
            || arm.balanced_ternary_decompositions().len() < OPENING_COUNT
        {
            return Err(opening_error("source arm has the wrong shifted-ternary opening count"));
        }
        for (opening, trace) in arm.shifted_ternary_canonical_traces().iter().enumerate() {
            let row_start = SOURCE_CANONICAL_ROW_START + opening * SOURCE_CANONICAL_ROW_STRIDE;
            if trace.field_column != SOURCE_FIELD_START + opening
                || trace.digit_columns_start != SOURCE_DIGIT_START + opening * SOURCE_DIGIT_STRIDE
                || trace.negative_columns_start != trace.digit_columns_start + DIGIT_COUNT
                || trace.borrow_columns_start != trace.digit_columns_start + 2 * DIGIT_COUNT
                || trace.digit_rows_start != row_start
                || trace.reconstruction_row != row_start + 2 * DIGIT_COUNT
                || trace.transition_rows_start != row_start + 2 * DIGIT_COUNT + 1
            {
                return Err(opening_error("source shifted-ternary trace geometry drifted"));
            }
        }

        for (port, matrix) in [&arm.a, &arm.b, &arm.c].into_iter().enumerate() {
            let rows = rows_in_range(matrix, SOURCE_ZERO_ROW_START, DIGIT_COUNT)?;
            let mut count = 0;
            for (digit, actual) in rows.iter().enumerate() {
                let expected = match port {
                    0 => vec![(SOURCE_ZERO_DIGIT_START + digit, F::ONE)],
                    1 => vec![(0, F::ONE)],
                    _ => Vec::new(),
                };
                if *actual != expected {
                    return Err(opening_error("source zero-word row differs from the exact pin recipe"));
                }
                count += actual.len();
            }
            if arm_index == 0 {
                source_zero_nnz[port] = count;
            } else if source_zero_nnz[port] != count {
                return Err(opening_error("source zero-word parity sparsity differs"));
            }
        }
    }

    let relation = prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        arms,
        REPLAY_AUXILIARY_START - 1,
        0,
        D,
        0,
        4,
    )?
    .finish()?;
    if relation.structure().n != FINAL_ROWS
        || relation.structure().m != FINAL_COLUMNS
        || relation.selector_cols() != SELECTOR_COLUMNS
        || relation.structure().matrices.len() != PORT_COUNT
    {
        return Err(opening_error("normalized relation has the wrong opening-row shape"));
    }
    let compiler = relation
        .selective_compiler_audit()
        .ok_or_else(|| opening_error("normalized relation omitted its compiler audit"))?;
    if compiler.canonical_openings().len() != ARM_COUNT {
        return Err(opening_error("compiler audit has the wrong parity count"));
    }
    for arm in 0..ARM_COUNT {
        let openings = &compiler.canonical_openings()[arm];
        if openings.len() != OPENING_COUNT {
            return Err(opening_error("compiler audit has the wrong opening count"));
        }
        for (opening, audit) in openings.iter().enumerate() {
            if audit.source_field() != SOURCE_FIELD_START + opening
                || audit.digit_coordinates()
                    != (FINAL_DIGIT_START + opening * FINAL_DIGIT_STRIDE
                        ..FINAL_DIGIT_START + (opening + 1) * FINAL_DIGIT_STRIDE)
                        .collect::<Vec<_>>()
                || audit.borrow_coordinates()
                    != (FINAL_BORROW_START + opening * FINAL_BORROW_STRIDE
                        ..FINAL_BORROW_START + (opening + 1) * FINAL_BORROW_STRIDE)
                        .collect::<Vec<_>>()
                || audit.emitted_rows()
                    != (CANONICAL_EMITTED_STARTS[arm] + opening * CHUNK_COUNT
                        ..CANONICAL_EMITTED_STARTS[arm] + (opening + 1) * CHUNK_COUNT)
            {
                return Err(opening_error("compiler opening slot geometry drifted"));
            }
        }
    }
    let canonical_rewrites = compiler
        .rows()
        .rewrites()
        .iter()
        .filter(|rewrite| rewrite.kind() == SelectiveRewriteKind::ShiftedTernaryCanonical)
        .count();
    let canonical_runs = compiler
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == SelectiveEmittedRowFamily::ShiftedTernaryCanonical)
        .count();
    if canonical_rewrites != ARM_COUNT * OPENING_COUNT || canonical_runs != ARM_COUNT * OPENING_COUNT {
        return Err(opening_error("compiler canonical row ownership census drifted"));
    }

    let (normalized_chunk_bounds, complemented_chunks) = chunk_geometry();
    let mut final_port_nnz = [0usize; PORT_COUNT];
    for (port, matrix) in relation.structure().matrices.iter().enumerate() {
        let actual = selected_rows(matrix)?;
        let mut cursor = 0;
        for row in 0..CENTERED_ROW_COUNT {
            let expected = expected_centered_port(row, port);
            if actual[cursor] != expected {
                return Err(opening_error(format!(
                    "active digit centered-domain image drifted: port={port}, row={row}"
                )));
            }
            final_port_nnz[port] += actual[cursor].len();
            cursor += 1;
        }
        for arm in 0..ARM_COUNT {
            for digit in 0..DIGIT_COUNT {
                let expected = expected_zero_port(arm, digit, port);
                if actual[cursor] != expected {
                    return Err(opening_error(format!(
                        "zero-word image drifted: port={port}, arm={arm}, digit={digit}"
                    )));
                }
                final_port_nnz[port] += actual[cursor].len();
                cursor += 1;
            }
            for opening in 0..OPENING_COUNT {
                for chunk in 0..CHUNK_COUNT {
                    let expected = expected_canonical_port(
                        arm,
                        opening,
                        chunk,
                        port,
                        &normalized_chunk_bounds,
                        &complemented_chunks,
                    );
                    if actual[cursor] != expected {
                        return Err(opening_error(format!(
                            "canonical row image drifted: port={port}, arm={arm}, opening={opening}, chunk={chunk}, actual={:?}, expected={expected:?}",
                            actual[cursor]
                        )));
                    }
                    final_port_nnz[port] += actual[cursor].len();
                    cursor += 1;
                }
            }
        }
        if cursor != actual.len() {
            return Err(opening_error("selected opening-row census drifted"));
        }
    }

    Ok(NebulaFPrimePiRlcBodyOpeningRowsAudit {
        schema_version: SCHEMA_VERSION,
        arm_count: ARM_COUNT,
        opening_count: OPENING_COUNT,
        digit_count: DIGIT_COUNT,
        borrow_count: BORROW_COUNT,
        chunk_count: CHUNK_COUNT,
        source_zero_row_start: SOURCE_ZERO_ROW_START,
        source_zero_digit_start: SOURCE_ZERO_DIGIT_START,
        source_field_start: SOURCE_FIELD_START,
        source_digit_start: SOURCE_DIGIT_START,
        source_digit_stride: SOURCE_DIGIT_STRIDE,
        source_canonical_row_start: SOURCE_CANONICAL_ROW_START,
        source_canonical_row_stride: SOURCE_CANONICAL_ROW_STRIDE,
        centered_row_start: CENTERED_ROW_START,
        centered_row_count: CENTERED_ROW_COUNT,
        zero_emitted_starts: ZERO_EMITTED_STARTS,
        canonical_emitted_starts: CANONICAL_EMITTED_STARTS,
        selector_columns: SELECTOR_COLUMNS,
        final_digit_start: FINAL_DIGIT_START,
        final_digit_stride: FINAL_DIGIT_STRIDE,
        final_zero_start: FINAL_ZERO_START,
        final_borrow_start: FINAL_BORROW_START,
        final_borrow_stride: FINAL_BORROW_STRIDE,
        final_rows: FINAL_ROWS,
        final_columns: FINAL_COLUMNS,
        normalized_chunk_bounds,
        complemented_chunks,
        source_zero_nnz,
        final_port_nnz,
    })
}

const _: () = assert!(CENTERED_ROW_COUNT == 16_605);
const _: () = assert!(OPENING_COUNT * CHUNK_COUNT == 17_010);
