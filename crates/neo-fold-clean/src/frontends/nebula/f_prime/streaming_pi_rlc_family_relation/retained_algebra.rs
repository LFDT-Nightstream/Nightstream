//! Exact normalized port audit for the retained PiRLC algebra block.
//!
//! Owns exhaustive comparison of both source parity matrices with the
//! independent algebra recipe and of all retained normalized port rows with
//! the canonical radix images of those source rows. It does not own the
//! semantic meaning of the recipe, assignment values, or lifecycle state.

use std::collections::BTreeMap;

use neo_ccs::CcsMatrix;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use crate::frontends::r1cs_f_prime::{
    prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix, SelectiveSourceRowDisposition,
};

use super::{production_pi_rlc_family_body_source_arms, NebulaFPrimePiRlcFamilyRelationError, REPLAY_AUXILIARY_START};

const SCHEMA_VERSION: u64 = 1;
const SOURCE_COUNT: usize = 15;
const LANE_COUNT: usize = 54;
const LOCAL_CHALLENGE_START: usize = 1;
const LOCAL_INPUT_START: usize = LOCAL_CHALLENGE_START + SOURCE_COUNT * LANE_COUNT;
const LOCAL_OUTPUT_START: usize = LOCAL_INPUT_START + SOURCE_COUNT * LANE_COUNT;
const LOCAL_PRODUCT_START: usize = LOCAL_OUTPUT_START + LANE_COUNT;
const PRODUCT_ROWS: usize = SOURCE_COUNT * LANE_COUNT * LANE_COUNT;
const ALGEBRA_ROWS: usize = PRODUCT_ROWS + LANE_COUNT;
const LOCAL_COLUMNS: usize = LOCAL_PRODUCT_START + PRODUCT_ROWS;

const SOURCE_COLUMN_SHIFT: usize = 640;
const SOURCE_CHALLENGE_START: usize = LOCAL_CHALLENGE_START + SOURCE_COLUMN_SHIFT;
const SOURCE_INPUT_START: usize = LOCAL_INPUT_START + SOURCE_COLUMN_SHIFT;
const SOURCE_OUTPUT_START: usize = LOCAL_OUTPUT_START + SOURCE_COLUMN_SHIFT;
const SOURCE_PRODUCT_START: usize = LOCAL_PRODUCT_START + SOURCE_COLUMN_SHIFT;

const FINAL_ROWS: usize = 282_459;
const FINAL_COLUMNS: usize = 2_521_314;
const SELECTOR_COLUMNS: [usize; 2] = [648, 649];
const EMITTED_STARTS: [usize; 2] = [34_296, 158_272];
const FINAL_CHALLENGE_START: usize = 702;
const FINAL_INPUT_START: usize = 19_332;
const FINAL_OUTPUT_START: usize = 52_542;
const FINAL_PRODUCT_START: usize = FINAL_OUTPUT_START + LANE_COUNT * 23;
const GENERAL_WIDTH: usize = 23;
const INPUT_WIDTH: usize = 41;

const PORT_COUNT: usize = 13;
const GENERAL_SELECTOR_PORT: usize = 1;
const A_PORT: usize = 2;
const C_PORT: usize = 4;

pub(super) type Term = (usize, F);
type SourceRow = [Vec<Term>; 3];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcBodyAlgebraRetainedAudit {
    schema_version: u64,
    source_rows: usize,
    local_columns: usize,
    source_column_shift: usize,
    final_rows: usize,
    final_columns: usize,
    selector_columns: [usize; 2],
    emitted_starts: [usize; 2],
    source_starts: [usize; 4],
    final_starts: [usize; 4],
    widths: [usize; 4],
    radices: [u64; 4],
    source_nnz: [usize; 3],
    final_port_nnz: [usize; PORT_COUNT],
}

impl NebulaFPrimePiRlcBodyAlgebraRetainedAudit {
    pub const fn schema_version(&self) -> u64 {
        self.schema_version
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

    pub const fn source_starts(&self) -> [usize; 4] {
        self.source_starts
    }

    pub const fn final_starts(&self) -> [usize; 4] {
        self.final_starts
    }

    pub const fn widths(&self) -> [usize; 4] {
        self.widths
    }

    pub const fn radices(&self) -> [u64; 4] {
        self.radices
    }

    pub const fn source_nnz(&self) -> [usize; 3] {
        self.source_nnz
    }

    pub const fn final_port_nnz(&self) -> [usize; PORT_COUNT] {
        self.final_port_nnz
    }
}

fn algebra_error(reason: impl Into<String>) -> NebulaFPrimePiRlcFamilyRelationError {
    NebulaFPrimePiRlcFamilyRelationError::AlgebraRetained(reason.into())
}

fn challenge_column(source: usize, lane: usize) -> usize {
    LOCAL_CHALLENGE_START + source * LANE_COUNT + lane
}

fn input_column(source: usize, lane: usize) -> usize {
    LOCAL_INPUT_START + source * LANE_COUNT + lane
}

fn output_column(lane: usize) -> usize {
    LOCAL_OUTPUT_START + lane
}

fn product_column(source: usize, left: usize, right: usize) -> usize {
    LOCAL_PRODUCT_START + (source * LANE_COUNT + left) * LANE_COUNT + right
}

fn reduced_monomial(degree: usize) -> Vec<(usize, F)> {
    if degree < LANE_COUNT {
        vec![(degree, F::ONE)]
    } else if degree < LANE_COUNT + LANE_COUNT / 2 {
        vec![(degree - LANE_COUNT, -F::ONE), (degree - LANE_COUNT / 2, -F::ONE)]
    } else {
        vec![(degree - 3 * LANE_COUNT / 2, F::ONE)]
    }
}

pub(super) fn canonical_terms(terms: Vec<Term>) -> Vec<Term> {
    let mut canonical = BTreeMap::<usize, F>::new();
    for (column, coefficient) in terms {
        *canonical.entry(column).or_insert(F::ZERO) += coefficient;
    }
    canonical.retain(|_, coefficient| *coefficient != F::ZERO);
    canonical.into_iter().collect()
}

fn expected_local_source_row(row: usize) -> SourceRow {
    if row < PRODUCT_ROWS {
        let source = row / (LANE_COUNT * LANE_COUNT);
        let within_source = row % (LANE_COUNT * LANE_COUNT);
        let left = within_source / LANE_COUNT;
        let right = within_source % LANE_COUNT;
        return [
            vec![(0, -F::from_u64(2)), (challenge_column(source, left), F::ONE)],
            vec![(input_column(source, right), F::ONE)],
            vec![(product_column(source, left, right), F::ONE)],
        ];
    }

    let output = row - PRODUCT_ROWS;
    let mut products = Vec::new();
    for source in 0..SOURCE_COUNT {
        for left in 0..LANE_COUNT {
            for right in 0..LANE_COUNT {
                for (lane, coefficient) in reduced_monomial(left + right) {
                    if lane == output {
                        products.push((product_column(source, left, right), coefficient));
                    }
                }
            }
        }
    }
    [
        vec![(0, F::ONE)],
        canonical_terms(products),
        vec![(output_column(output), F::ONE)],
    ]
}

fn shifted_source_row(row: usize) -> SourceRow {
    expected_local_source_row(row).map(|terms| {
        terms
            .into_iter()
            .map(|(column, coefficient)| {
                let column = if column == 0 { 0 } else { column + SOURCE_COLUMN_SHIFT };
                (column, coefficient)
            })
            .collect()
    })
}

fn prefix_rows(matrix: &CcsMatrix<F>) -> Result<Vec<Vec<Term>>, NebulaFPrimePiRlcFamilyRelationError> {
    if matrix
        .seeded_phi81_blocks()
        .iter()
        .any(|block| block.row_start() < ALGEBRA_ROWS)
        || matrix
            .geometric_runs()
            .iter()
            .any(|run| run.row() < ALGEBRA_ROWS)
    {
        return Err(algebra_error(
            "compact source matrix content intersects the algebra rows",
        ));
    }
    let mut rows = vec![Vec::new(); ALGEBRA_ROWS];
    match matrix {
        CcsMatrix::Identity { n } => {
            if *n < ALGEBRA_ROWS {
                return Err(algebra_error(
                    "source identity matrix is shorter than the algebra block",
                ));
            }
            for (row, terms) in rows.iter_mut().enumerate() {
                terms.push((row, F::ONE));
            }
        }
        CcsMatrix::Csc(_) | CcsMatrix::CscWithSeededPhi81 { .. } => {
            let csc = matrix
                .sparse_component()
                .ok_or_else(|| algebra_error("source matrix has no sparse component"))?;
            if !csc.is_canonical() {
                return Err(algebra_error("source matrix CSC is not canonical"));
            }
            for column in 0..csc.ncols {
                for entry in csc.column_range(column) {
                    let row = csc.row_index(entry);
                    if row < ALGEBRA_ROWS {
                        rows[row].push((column, csc.vals[entry]));
                    }
                }
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            return Err(algebra_error("source matrix content is unavailable"));
        }
    }
    Ok(rows)
}

fn selected_row_index(row: usize) -> Option<usize> {
    EMITTED_STARTS.iter().enumerate().find_map(|(arm, &start)| {
        (start..start + ALGEBRA_ROWS)
            .contains(&row)
            .then_some(arm * ALGEBRA_ROWS + row - start)
    })
}

fn selected_rows(matrix: &CcsMatrix<F>) -> Result<Vec<Vec<Term>>, NebulaFPrimePiRlcFamilyRelationError> {
    let intersects = |start: usize, stop: usize| {
        EMITTED_STARTS
            .iter()
            .any(|&selected| start < selected + ALGEBRA_ROWS && selected < stop)
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
        return Err(algebra_error(
            "compact final matrix content intersects retained algebra rows",
        ));
    }
    let mut rows = vec![Vec::new(); 2 * ALGEBRA_ROWS];
    match matrix {
        CcsMatrix::Identity { n } => {
            if *n < FINAL_ROWS {
                return Err(algebra_error("final identity matrix has the wrong row domain"));
            }
            for &start in &EMITTED_STARTS {
                for row in start..start + ALGEBRA_ROWS {
                    rows[selected_row_index(row).expect("selected identity row")].push((row, F::ONE));
                }
            }
        }
        CcsMatrix::Csc(_) | CcsMatrix::CscWithSeededPhi81 { .. } => {
            let csc = matrix
                .sparse_component()
                .ok_or_else(|| algebra_error("final matrix has no sparse component"))?;
            if !csc.is_canonical() {
                return Err(algebra_error("final matrix CSC is not canonical"));
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
            return Err(algebra_error("final matrix content is unavailable"));
        }
    }
    Ok(rows)
}

pub(super) fn append_radix_image(terms: &mut Vec<Term>, start: usize, width: usize, coefficient: F) {
    let radix = match width {
        INPUT_WIDTH => F::from_u64(3),
        GENERAL_WIDTH => F::from_u64(7),
        _ => F::from_u64(2),
    };
    let mut power = coefficient;
    for offset in 0..width {
        terms.push((start + offset, power));
        power *= radix;
    }
}

fn source_slot(column: usize) -> Option<(usize, usize)> {
    if (SOURCE_CHALLENGE_START..SOURCE_INPUT_START).contains(&column) {
        Some((
            FINAL_CHALLENGE_START + (column - SOURCE_CHALLENGE_START) * GENERAL_WIDTH,
            GENERAL_WIDTH,
        ))
    } else if (SOURCE_INPUT_START..SOURCE_OUTPUT_START).contains(&column) {
        Some((
            FINAL_INPUT_START + (column - SOURCE_INPUT_START) * INPUT_WIDTH,
            INPUT_WIDTH,
        ))
    } else if (SOURCE_OUTPUT_START..SOURCE_PRODUCT_START).contains(&column) {
        Some((
            FINAL_OUTPUT_START + (column - SOURCE_OUTPUT_START) * GENERAL_WIDTH,
            GENERAL_WIDTH,
        ))
    } else if (SOURCE_PRODUCT_START..SOURCE_PRODUCT_START + PRODUCT_ROWS).contains(&column) {
        Some((
            FINAL_PRODUCT_START + (column - SOURCE_PRODUCT_START) * GENERAL_WIDTH,
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
                .ok_or_else(|| algebra_error("algebra source term is outside the declared normalized slots"))?;
            append_radix_image(&mut terms, start, width, coefficient);
        }
    }
    Ok(canonical_terms(terms))
}

fn expected_final_port(arm: usize, row: usize, port: usize) -> Result<Vec<Term>, NebulaFPrimePiRlcFamilyRelationError> {
    match port {
        GENERAL_SELECTOR_PORT => Ok(vec![(SELECTOR_COLUMNS[arm], F::ONE)]),
        A_PORT..=C_PORT => final_image(&shifted_source_row(row)[port - A_PORT]),
        _ => Ok(Vec::new()),
    }
}

pub fn production_pi_rlc_family_body_algebra_retained_audit(
) -> Result<NebulaFPrimePiRlcBodyAlgebraRetainedAudit, NebulaFPrimePiRlcFamilyRelationError> {
    let arms = production_pi_rlc_family_body_source_arms()?;
    if arms.len() != 2
        || arms
            .iter()
            .any(|arm| arm.n < ALGEBRA_ROWS || arm.m < SOURCE_PRODUCT_START + PRODUCT_ROWS)
    {
        return Err(algebra_error(
            "source parity shape does not contain the complete algebra block",
        ));
    }

    let mut source_nnz = [0usize; 3];
    for (arm_index, arm) in arms.iter().enumerate() {
        for (port, matrix) in [&arm.a, &arm.b, &arm.c].into_iter().enumerate() {
            let actual = prefix_rows(matrix)?;
            let mut count = 0;
            for (row, terms) in actual.iter().enumerate() {
                let expected = shifted_source_row(row)[port].clone();
                if *terms != expected {
                    return Err(algebra_error(
                        "source parity matrix differs from the exact algebra recipe",
                    ));
                }
                count += terms.len();
            }
            if arm_index == 0 {
                source_nnz[port] = count;
            } else if source_nnz[port] != count {
                return Err(algebra_error("source parity algebra matrices have different sparsity"));
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
        return Err(algebra_error("normalized relation has the wrong algebra audit shape"));
    }
    let compiler = relation
        .selective_compiler_audit()
        .ok_or_else(|| algebra_error("normalized relation omitted its compiler audit"))?;
    for (arm, (&emitted_start, mapping)) in EMITTED_STARTS
        .iter()
        .zip(compiler.rows().arms())
        .enumerate()
    {
        let Some(first) = mapping.source_runs().first() else {
            return Err(algebra_error("compiler audit omitted the first retained algebra run"));
        };
        if first.source_rows() != (0..ALGEBRA_ROWS)
            || first.disposition() != SelectiveSourceRowDisposition::Retained
            || first.emitted_start() != Some(emitted_start)
            || mapping.retained_emitted_rows().start > emitted_start
            || arm > 1
        {
            return Err(algebra_error(
                "compiler ledger differs from the retained algebra interval",
            ));
        }
    }

    let mut final_port_nnz = [0usize; PORT_COUNT];
    for (port, matrix) in relation.structure().matrices.iter().enumerate() {
        let actual = selected_rows(matrix)?;
        for arm in 0..2 {
            for row in 0..ALGEBRA_ROWS {
                let terms = &actual[arm * ALGEBRA_ROWS + row];
                let expected = expected_final_port(arm, row, port)?;
                if *terms != expected {
                    let first_difference = terms
                        .iter()
                        .zip(expected.iter())
                        .position(|(actual, expected)| actual != expected);
                    return Err(algebra_error(format!(
                        "normalized port differs from the exact algebra source image: \
                         port={port}, arm={arm}, source_row={row}, \
                         actual_len={}, expected_len={}, first_difference={first_difference:?}, \
                         actual_first={:?}, expected_first={:?}",
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

    Ok(NebulaFPrimePiRlcBodyAlgebraRetainedAudit {
        schema_version: SCHEMA_VERSION,
        source_rows: ALGEBRA_ROWS,
        local_columns: LOCAL_COLUMNS,
        source_column_shift: SOURCE_COLUMN_SHIFT,
        final_rows: FINAL_ROWS,
        final_columns: FINAL_COLUMNS,
        selector_columns: SELECTOR_COLUMNS,
        emitted_starts: EMITTED_STARTS,
        source_starts: [
            SOURCE_CHALLENGE_START,
            SOURCE_INPUT_START,
            SOURCE_OUTPUT_START,
            SOURCE_PRODUCT_START,
        ],
        final_starts: [
            FINAL_CHALLENGE_START,
            FINAL_INPUT_START,
            FINAL_OUTPUT_START,
            FINAL_PRODUCT_START,
        ],
        widths: [GENERAL_WIDTH, INPUT_WIDTH, GENERAL_WIDTH, GENERAL_WIDTH],
        radices: [7, 3, 7, 7],
        source_nnz,
        final_port_nnz,
    })
}

const _: () = assert!(ALGEBRA_ROWS == 43_794);
const _: () = assert!(LOCAL_COLUMNS == 45_415);
const _: () = assert!(SOURCE_PRODUCT_START + PRODUCT_ROWS == 46_055);
const _: () = assert!(FINAL_PRODUCT_START == 53_784);
