//! Exact normalized port audit for the production PiRLC family overlays.
//!
//! Owns exhaustive comparison of all 110 source seeded maps with the
//! independent family-position recipe and of every retained normalized row
//! with its compact seeded block, selector, constant-one, and radix-three
//! output images. It does not own body-to-overlay links, assignment values,
//! selector authority, lifecycle state, or Module-SIS hardness.

use neo_ajtai::seeded_pp_chunk_seeds;
use neo_ccs::{CcsMatrix, SeededPhi81LinearBlock};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::frontends::r1cs_f_prime::{
    prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix, SelectiveSourceRowDisposition,
};
use crate::paper::reductions::accumulator_sis_circuit::PI_RLC_INPUT_COORDINATE_SIS_CONFIG;

use super::retained_algebra::{append_radix_image, canonical_terms, Term};
use super::{
    production_pi_rlc_family_overlay_sparse_arms, NebulaFPrimePiRlcFamilyRelationError, COMMITMENT_OUTPUT_FIELDS,
    DIGIT_COUNT, FAMILY_INPUT_FIELDS, PI_RLC_FAMILY_COUNT, PI_RLC_FAMILY_OVERLAY_ACTIVE_DIGIT_START,
    PI_RLC_FAMILY_OVERLAY_COLUMNS, PI_RLC_FAMILY_OVERLAY_OUTPUT_START, PI_RLC_FAMILY_OVERLAY_ROWS,
    PI_RLC_FAMILY_OVERLAY_ZERO_DIGIT_START, PI_RLC_GLOBAL_INPUT_FIELDS, PI_RLC_MESSAGE_COLUMNS,
};

const SCHEMA_VERSION: u64 = 1;
const FINAL_ROWS: usize = 12_001;
const FINAL_COLUMNS: usize = 42_228;
const SELECTOR_START: usize = 1;
const RETAINED_START: usize = 111;
const RETAINED_STRIDE: usize = PI_RLC_FAMILY_OVERLAY_ROWS;
pub(super) const FINAL_ZERO_DIGIT_START: usize = 111;
pub(super) const FINAL_ACTIVE_DIGIT_START: usize = 152;
pub(super) const FINAL_OUTPUT_START: usize = 37_790;
pub(super) const INPUT_WIDTH: usize = 1;
pub(super) const OUTPUT_WIDTH: usize = 41;
pub(super) const OUTPUT_RADIX: u64 = 3;
const PORT_COUNT: usize = 13;
const GENERAL_SELECTOR_PORT: usize = 1;
const A_PORT: usize = 2;
const B_PORT: usize = 3;
const C_PORT: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcFamilyOverlayRetainedAudit {
    schema_version: u64,
    family_count: usize,
    source_rows: usize,
    source_columns: usize,
    final_rows: usize,
    final_columns: usize,
    selector_start: usize,
    selector_count: usize,
    retained_start: usize,
    retained_stride: usize,
    source_starts: [usize; 3],
    final_starts: [usize; 3],
    widths: [usize; 2],
    radices: [u64; 2],
    chunk_size: usize,
    chunk_seeds_by_row: Vec<Vec<[u8; 32]>>,
    source_explicit_nnz: [usize; 3],
    final_block_counts: [usize; PORT_COUNT],
    final_explicit_port_nnz: [usize; PORT_COUNT],
}

impl NebulaFPrimePiRlcFamilyOverlayRetainedAudit {
    pub const fn schema_version(&self) -> u64 {
        self.schema_version
    }

    pub const fn family_count(&self) -> usize {
        self.family_count
    }

    pub const fn source_rows(&self) -> usize {
        self.source_rows
    }

    pub const fn source_columns(&self) -> usize {
        self.source_columns
    }

    pub const fn final_rows(&self) -> usize {
        self.final_rows
    }

    pub const fn final_columns(&self) -> usize {
        self.final_columns
    }

    pub const fn selector_start(&self) -> usize {
        self.selector_start
    }

    pub const fn selector_count(&self) -> usize {
        self.selector_count
    }

    pub const fn retained_start(&self) -> usize {
        self.retained_start
    }

    pub const fn retained_stride(&self) -> usize {
        self.retained_stride
    }

    pub const fn source_starts(&self) -> [usize; 3] {
        self.source_starts
    }

    pub const fn final_starts(&self) -> [usize; 3] {
        self.final_starts
    }

    pub const fn widths(&self) -> [usize; 2] {
        self.widths
    }

    pub const fn radices(&self) -> [u64; 2] {
        self.radices
    }

    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn chunk_seeds_by_row(&self) -> &[Vec<[u8; 32]>] {
        &self.chunk_seeds_by_row
    }

    pub const fn source_explicit_nnz(&self) -> [usize; 3] {
        self.source_explicit_nnz
    }

    pub const fn final_block_counts(&self) -> [usize; PORT_COUNT] {
        self.final_block_counts
    }

    pub const fn final_explicit_port_nnz(&self) -> [usize; PORT_COUNT] {
        self.final_explicit_port_nnz
    }
}

fn overlay_error(reason: impl Into<String>) -> NebulaFPrimePiRlcFamilyRelationError {
    NebulaFPrimePiRlcFamilyRelationError::OverlayRetained(reason.into())
}

fn source_word_starts(family: usize) -> Vec<usize> {
    let mut starts = vec![PI_RLC_FAMILY_OVERLAY_ZERO_DIGIT_START; PI_RLC_GLOBAL_INPUT_FIELDS];
    let family_start = family * FAMILY_INPUT_FIELDS;
    for offset in 0..FAMILY_INPUT_FIELDS {
        starts[family_start + offset] = PI_RLC_FAMILY_OVERLAY_ACTIVE_DIGIT_START + offset * DIGIT_COUNT;
    }
    starts
}

fn final_word_starts(family: usize) -> Vec<usize> {
    let mut starts = vec![FINAL_ZERO_DIGIT_START; PI_RLC_GLOBAL_INPUT_FIELDS];
    let family_start = family * FAMILY_INPUT_FIELDS;
    for offset in 0..FAMILY_INPUT_FIELDS {
        starts[family_start + offset] = FINAL_ACTIVE_DIGIT_START + offset * DIGIT_COUNT;
    }
    starts
}

fn expected_source_block(
    family: usize,
    chunk_size: usize,
    chunk_seeds_by_row: &[Vec<[u8; 32]>],
) -> Result<SeededPhi81LinearBlock, NebulaFPrimePiRlcFamilyRelationError> {
    SeededPhi81LinearBlock::new_with_word_width(
        0,
        source_word_starts(family),
        DIGIT_COUNT,
        PI_RLC_INPUT_COORDINATE_SIS_CONFIG.kappa,
        PI_RLC_MESSAGE_COLUMNS,
        chunk_size,
        chunk_seeds_by_row.to_vec(),
    )
    .map_err(|error| overlay_error(error.to_string()))
}

fn expected_final_block(
    family: usize,
    chunk_size: usize,
    chunk_seeds_by_row: &[Vec<[u8; 32]>],
) -> Result<SeededPhi81LinearBlock, NebulaFPrimePiRlcFamilyRelationError> {
    SeededPhi81LinearBlock::new_with_word_width(
        RETAINED_START + family * RETAINED_STRIDE,
        final_word_starts(family),
        DIGIT_COUNT,
        PI_RLC_INPUT_COORDINATE_SIS_CONFIG.kappa,
        PI_RLC_MESSAGE_COLUMNS,
        chunk_size,
        chunk_seeds_by_row.to_vec(),
    )
    .map_err(|error| overlay_error(error.to_string()))
}

fn explicit_rows(
    matrix: &CcsMatrix<F>,
    row_count: usize,
) -> Result<Vec<Vec<Term>>, NebulaFPrimePiRlcFamilyRelationError> {
    if !matrix.seeded_phi81_blocks().is_empty() || !matrix.geometric_runs().is_empty() {
        return Err(overlay_error(
            "source explicit-row scan intersects compact matrix content",
        ));
    }
    let mut rows = vec![Vec::new(); row_count];
    match matrix {
        CcsMatrix::Identity { n } => {
            if *n < row_count {
                return Err(overlay_error("source identity matrix is shorter than the overlay rows"));
            }
            for (row, terms) in rows.iter_mut().enumerate() {
                terms.push((row, F::ONE));
            }
        }
        CcsMatrix::Csc(csc) | CcsMatrix::CscWithSeededPhi81 { csc, .. } => {
            if !csc.is_canonical() {
                return Err(overlay_error("source overlay CSC is not canonical"));
            }
            for column in 0..csc.ncols {
                for entry in csc.column_range(column) {
                    let row = csc.row_index(entry);
                    if row < row_count {
                        rows[row].push((column, csc.vals[entry]));
                    }
                }
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            return Err(overlay_error("source overlay matrix content is unavailable"));
        }
    }
    Ok(rows)
}

fn selected_row_index(row: usize) -> Option<usize> {
    if row < RETAINED_START {
        return None;
    }
    let offset = row - RETAINED_START;
    (offset < PI_RLC_FAMILY_COUNT * PI_RLC_FAMILY_OVERLAY_ROWS).then_some(offset)
}

fn selected_final_explicit_rows(matrix: &CcsMatrix<F>) -> Result<Vec<Vec<Term>>, NebulaFPrimePiRlcFamilyRelationError> {
    let mut rows = vec![Vec::new(); PI_RLC_FAMILY_COUNT * PI_RLC_FAMILY_OVERLAY_ROWS];
    match matrix {
        CcsMatrix::Identity { n } => {
            if *n < FINAL_ROWS {
                return Err(overlay_error("final identity matrix has the wrong row domain"));
            }
            for row in RETAINED_START..RETAINED_START + rows.len() {
                rows[row - RETAINED_START].push((row, F::ONE));
            }
        }
        CcsMatrix::Csc(csc) | CcsMatrix::CscWithSeededPhi81 { csc, .. } => {
            if !csc.is_canonical() {
                return Err(overlay_error("final overlay CSC is not canonical"));
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
            return Err(overlay_error("final overlay matrix content is unavailable"));
        }
    }
    for run in matrix.geometric_runs() {
        if let Some(index) = selected_row_index(run.row()) {
            run.for_each_term(|_, column, coefficient| rows[index].push((column, coefficient)));
        }
    }
    for row in &mut rows {
        *row = canonical_terms(core::mem::take(row));
    }
    Ok(rows)
}

fn expected_final_port(family: usize, output: usize, port: usize) -> Vec<Term> {
    match port {
        GENERAL_SELECTOR_PORT => vec![(SELECTOR_START + family, F::ONE)],
        B_PORT => vec![(0, F::ONE)],
        C_PORT => {
            let mut terms = Vec::with_capacity(OUTPUT_WIDTH);
            append_radix_image(
                &mut terms,
                FINAL_OUTPUT_START + output * OUTPUT_WIDTH,
                OUTPUT_WIDTH,
                F::ONE,
            );
            canonical_terms(terms)
        }
        _ => Vec::new(),
    }
}

pub fn production_pi_rlc_family_overlay_retained_audit(
) -> Result<NebulaFPrimePiRlcFamilyOverlayRetainedAudit, NebulaFPrimePiRlcFamilyRelationError> {
    let arms = production_pi_rlc_family_overlay_sparse_arms()?;
    if arms.len() != PI_RLC_FAMILY_COUNT
        || arms
            .iter()
            .any(|arm| arm.n != PI_RLC_FAMILY_OVERLAY_ROWS || arm.m != PI_RLC_FAMILY_OVERLAY_COLUMNS)
    {
        return Err(overlay_error(
            "source overlay family shapes differ from the exact recipe",
        ));
    }

    let (chunk_size, chunk_seeds_by_row) = seeded_pp_chunk_seeds(
        PI_RLC_INPUT_COORDINATE_SIS_CONFIG.seed,
        PI_RLC_INPUT_COORDINATE_SIS_CONFIG.kappa,
        PI_RLC_MESSAGE_COLUMNS,
    );
    let mut source_explicit_nnz = [0usize; 3];
    for (family, arm) in arms.iter().enumerate() {
        if arm.a.seeded_phi81_blocks() != [expected_source_block(family, chunk_size, &chunk_seeds_by_row)?]
            || !arm.a.geometric_runs().is_empty()
        {
            return Err(overlay_error(format!(
                "source family {family} seeded block differs from the verifier-owned recipe"
            )));
        }
        let a_explicit = match &arm.a {
            CcsMatrix::CscWithSeededPhi81 { csc, .. } => csc.vals.len(),
            _ => return Err(overlay_error("source A port did not retain compact seeded metadata")),
        };
        if a_explicit != 0 {
            return Err(overlay_error("source seeded A port has unexpected explicit terms"));
        }
        for (port, matrix) in [&arm.b, &arm.c].into_iter().enumerate() {
            let actual = explicit_rows(matrix, PI_RLC_FAMILY_OVERLAY_ROWS)?;
            let matrix_port = port + 1;
            for (output, terms) in actual.iter().enumerate() {
                let expected = if matrix_port == 1 {
                    vec![(0, F::ONE)]
                } else {
                    vec![(PI_RLC_FAMILY_OVERLAY_OUTPUT_START + output, F::ONE)]
                };
                if *terms != expected {
                    return Err(overlay_error(format!(
                        "source family {family} port {matrix_port} row {output} differs from the exact recipe"
                    )));
                }
                source_explicit_nnz[matrix_port] += terms.len();
            }
        }
    }

    let relation = prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        arms,
        PI_RLC_FAMILY_OVERLAY_COLUMNS - 1,
        0,
        1,
        0,
        crate::config::B_BASE,
    )?
    .finish()?;
    if relation.structure().n != FINAL_ROWS
        || relation.structure().m != FINAL_COLUMNS
        || relation.selector_cols() != (SELECTOR_START..SELECTOR_START + PI_RLC_FAMILY_COUNT).collect::<Vec<_>>()
        || relation.structure().matrices.len() != PORT_COUNT
    {
        return Err(overlay_error("normalized overlay relation has the wrong exact shape"));
    }
    let snapshot = relation
        .selective_snapshot()
        .map_err(|error| overlay_error(error.to_string()))?;
    for (family, mapping) in snapshot.compiler_audit().rows().arms().iter().enumerate() {
        if mapping.source_runs().len() != 1
            || mapping.source_runs()[0].source_rows() != (0..PI_RLC_FAMILY_OVERLAY_ROWS)
            || mapping.source_runs()[0].disposition() != SelectiveSourceRowDisposition::Retained
            || mapping.source_runs()[0].emitted_start() != Some(RETAINED_START + family * RETAINED_STRIDE)
            || mapping.retained_emitted_rows()
                != (RETAINED_START + family * RETAINED_STRIDE..RETAINED_START + (family + 1) * RETAINED_STRIDE)
        {
            return Err(overlay_error(format!(
                "compiler row ledger differs for overlay family {family}"
            )));
        }
        let plan = snapshot
            .arm(family)
            .ok_or_else(|| overlay_error("normalized overlay snapshot omitted an arm"))?;
        for source in 1..PI_RLC_FAMILY_OVERLAY_OUTPUT_START {
            let slot = plan
                .slot(source)
                .ok_or_else(|| overlay_error("normalized overlay digit omitted its direct slot"))?;
            if slot.start() != source + 110 || slot.len() != INPUT_WIDTH || !plan.centered_columns()[source] {
                return Err(overlay_error(format!(
                    "normalized overlay digit slot differs: family={family}, source={source}"
                )));
            }
        }
        for source in PI_RLC_FAMILY_OVERLAY_OUTPUT_START..PI_RLC_FAMILY_OVERLAY_COLUMNS {
            let slot = plan
                .slot(source)
                .ok_or_else(|| overlay_error("normalized overlay output omitted its radix-three slot"))?;
            let expected = FINAL_OUTPUT_START + (source - PI_RLC_FAMILY_OVERLAY_OUTPUT_START) * OUTPUT_WIDTH;
            if slot.start() != expected || slot.len() != OUTPUT_WIDTH || plan.centered_columns()[source] {
                return Err(overlay_error(format!(
                    "normalized overlay output slot differs: family={family}, source={source}"
                )));
            }
        }
    }

    let final_blocks = relation.structure().matrices[A_PORT].seeded_phi81_blocks();
    if final_blocks.len() != PI_RLC_FAMILY_COUNT {
        return Err(overlay_error("normalized A port has the wrong seeded-block count"));
    }
    for (family, actual) in final_blocks.iter().enumerate() {
        let expected = expected_final_block(family, chunk_size, &chunk_seeds_by_row)?;
        if *actual != expected {
            return Err(overlay_error(format!(
                "normalized seeded block differs for overlay family {family}"
            )));
        }
    }
    for (port, matrix) in relation.structure().matrices.iter().enumerate() {
        if port != A_PORT && !matrix.seeded_phi81_blocks().is_empty() {
            return Err(overlay_error(format!(
                "normalized overlay port {port} has an unexpected seeded block"
            )));
        }
    }

    let mut final_explicit_port_nnz = [0usize; PORT_COUNT];
    for (port, matrix) in relation.structure().matrices.iter().enumerate() {
        let actual = selected_final_explicit_rows(matrix)?;
        for family in 0..PI_RLC_FAMILY_COUNT {
            for output in 0..PI_RLC_FAMILY_OVERLAY_ROWS {
                let terms = &actual[family * PI_RLC_FAMILY_OVERLAY_ROWS + output];
                let expected = expected_final_port(family, output, port);
                if *terms != expected {
                    return Err(overlay_error(format!(
                        "normalized overlay port differs: port={port}, family={family}, output={output}, actual_len={}, expected_len={}",
                        terms.len(),
                        expected.len(),
                    )));
                }
                final_explicit_port_nnz[port] += terms.len();
            }
        }
    }

    let mut final_block_counts = [0usize; PORT_COUNT];
    for (port, matrix) in relation.structure().matrices.iter().enumerate() {
        final_block_counts[port] = matrix.seeded_phi81_blocks().len();
    }
    Ok(NebulaFPrimePiRlcFamilyOverlayRetainedAudit {
        schema_version: SCHEMA_VERSION,
        family_count: PI_RLC_FAMILY_COUNT,
        source_rows: PI_RLC_FAMILY_OVERLAY_ROWS,
        source_columns: PI_RLC_FAMILY_OVERLAY_COLUMNS,
        final_rows: FINAL_ROWS,
        final_columns: FINAL_COLUMNS,
        selector_start: SELECTOR_START,
        selector_count: PI_RLC_FAMILY_COUNT,
        retained_start: RETAINED_START,
        retained_stride: RETAINED_STRIDE,
        source_starts: [
            PI_RLC_FAMILY_OVERLAY_ZERO_DIGIT_START,
            PI_RLC_FAMILY_OVERLAY_ACTIVE_DIGIT_START,
            PI_RLC_FAMILY_OVERLAY_OUTPUT_START,
        ],
        final_starts: [FINAL_ZERO_DIGIT_START, FINAL_ACTIVE_DIGIT_START, FINAL_OUTPUT_START],
        widths: [INPUT_WIDTH, OUTPUT_WIDTH],
        radices: [2, OUTPUT_RADIX],
        chunk_size,
        chunk_seeds_by_row,
        source_explicit_nnz,
        final_block_counts,
        final_explicit_port_nnz,
    })
}

const _: () = assert!(PI_RLC_FAMILY_OVERLAY_ROWS == 108);
const _: () = assert!(PI_RLC_FAMILY_OVERLAY_COLUMNS == 37_788);
const _: () = assert!(RETAINED_START + PI_RLC_FAMILY_COUNT * RETAINED_STRIDE == 11_991);
const _: () = assert!(FINAL_OUTPUT_START + COMMITMENT_OUTPUT_FIELDS * OUTPUT_WIDTH == 42_218);
