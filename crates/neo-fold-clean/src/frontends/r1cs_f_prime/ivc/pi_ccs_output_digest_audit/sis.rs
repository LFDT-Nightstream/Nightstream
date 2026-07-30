//! Canonical-opening and seeded-Φ81 ownership for the stabilized output digest.
//!
//! Owns: the exact field-word to source-column map; canonical 82+1+41 row
//! openings; both seeded block geometries; and output columns recovered from
//! the emitted R1CS equations.
//!
//! Does not own: source semantic truth, SIS binding security, seed derivation,
//! Poseidon2, transcript placement, costs, or row removal.
//!
//! Emits constraints: no.
//!
//! Authority boundary: typed field paths classify existing columns. They do
//! not make `y_ring` or `y_zcol` authoritative. Seeded outputs are recovered
//! from B/C rows rather than trusted from a side trace.
//!
//! | Leaf | Mathematical/physical obligation |
//! |---|---|
//! | `openings.reused` | a previously constrained source column supplies this word |
//! | `openings.primary` | one exact 82+1+41 canonical opening precedes the primary map |
//! | `primary` | ordered field words feed the rank-2 seeded map |
//! | `openings.compression` | every primary output receives one canonical opening |
//! | `compression` | ordered primary outputs feed the rank-1 seeded map |

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use neo_ccs::{CcsMatrix, SeededPhi81LinearBlock};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::BALANCED_TERNARY_DIGITS;
use crate::frontends::r1cs_f_prime::SparseR1cs;
use crate::paper::reductions::pi_ccs_output_message::{Profile, R1csInputOwner};

use super::invalid;
use crate::frontends::r1cs_f_prime::ivc::R1csIvcError;

const DIGIT_ROWS: usize = 2 * BALANCED_TERNARY_DIGITS;
const TRANSITION_ROWS: usize = BALANCED_TERNARY_DIGITS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalOpeningPlacement {
    Reused,
    Primary,
    Compression,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalOpeningAudit {
    pub source_column: usize,
    pub digit_columns: Range<usize>,
    pub negative_columns: Range<usize>,
    pub borrow_columns: Range<usize>,
    pub digit_rows: Range<usize>,
    pub reconstruction_row: usize,
    pub transition_rows: Range<usize>,
    pub placement: CanonicalOpeningPlacement,
    pub owner: Option<R1csInputOwner>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeededPhi81BlockAudit {
    block: SeededPhi81LinearBlock,
    input_columns: Vec<usize>,
    output_columns: Vec<usize>,
}

impl SeededPhi81BlockAudit {
    pub fn block(&self) -> &SeededPhi81LinearBlock {
        &self.block
    }

    /// Ordered source columns consumed by the block's canonical ternary
    /// words. For the primary block this is the complete typed PiCCS output
    /// preimage; for compression it is the primary block's output vector.
    pub fn input_columns(&self) -> &[usize] {
        &self.input_columns
    }

    pub fn output_columns(&self) -> &[usize] {
        &self.output_columns
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsOutputSisPhysicalAudit {
    openings: Vec<CanonicalOpeningAudit>,
    primary: SeededPhi81BlockAudit,
    compression: SeededPhi81BlockAudit,
}

impl PiCcsOutputSisPhysicalAudit {
    pub fn openings(&self) -> &[CanonicalOpeningAudit] {
        &self.openings
    }

    pub fn primary(&self) -> &SeededPhi81BlockAudit {
        &self.primary
    }

    pub fn compression(&self) -> &SeededPhi81BlockAudit {
        &self.compression
    }
}

pub(super) fn recover(
    arm: &SparseR1cs,
    sis_start: usize,
    claim_start: usize,
    profile: Profile,
    primary: &SeededPhi81LinearBlock,
    compression: &SeededPhi81LinearBlock,
) -> Result<PiCcsOutputSisPhysicalAudit, R1csIvcError> {
    let (primary_outputs, compression_outputs) = recover_outputs(arm, primary, compression)?;
    let mut opening_map = recover_openings(arm, sis_start, claim_start, primary, compression)?;
    validate_primary_words(profile, primary, &mut opening_map)?;
    validate_compression_words(compression, &primary_outputs, &opening_map)?;
    let primary_inputs = ordered_input_columns(primary, &opening_map)?;
    let compression_inputs = ordered_input_columns(compression, &opening_map)?;
    if compression_inputs != primary_outputs {
        return Err(invalid(
            "compression SIS input columns differ from the primary SIS outputs",
        ));
    }

    Ok(PiCcsOutputSisPhysicalAudit {
        openings: opening_map.into_values().collect(),
        primary: SeededPhi81BlockAudit {
            block: primary.clone(),
            input_columns: primary_inputs,
            output_columns: primary_outputs,
        },
        compression: SeededPhi81BlockAudit {
            block: compression.clone(),
            input_columns: compression_inputs,
            output_columns: compression_outputs,
        },
    })
}

fn ordered_input_columns(
    block: &SeededPhi81LinearBlock,
    openings: &BTreeMap<usize, CanonicalOpeningAudit>,
) -> Result<Vec<usize>, R1csIvcError> {
    block
        .word_starts()
        .iter()
        .map(|word_start| {
            openings
                .get(word_start)
                .map(|opening| opening.source_column)
                .ok_or_else(|| {
                    invalid(format!(
                        "seeded Phi81 word column {word_start} has no canonical source opening"
                    ))
                })
        })
        .collect()
}

fn recover_openings(
    arm: &SparseR1cs,
    sis_start: usize,
    claim_start: usize,
    primary: &SeededPhi81LinearBlock,
    compression: &SeededPhi81LinearBlock,
) -> Result<BTreeMap<usize, CanonicalOpeningAudit>, R1csIvcError> {
    let required = primary
        .word_starts()
        .iter()
        .chain(compression.word_starts())
        .copied()
        .collect::<BTreeSet<_>>();
    let mut sources = BTreeMap::new();
    for decomposition in arm.balanced_ternary_decompositions() {
        let start = decomposition.digit_cols[0];
        if decomposition.digit_cols != std::array::from_fn(|offset| start + offset) {
            return Err(invalid(format!(
                "balanced-ternary word at column {start} is not contiguous"
            )));
        }
        if sources.insert(start, decomposition.field_col).is_some() {
            return Err(invalid(format!(
                "balanced-ternary word column {start} has duplicate source decompositions"
            )));
        }
    }

    let mut openings = BTreeMap::new();
    for trace in arm.shifted_ternary_canonical_traces() {
        let start = trace.digit_columns_start;
        let source_column = *sources.get(&start).ok_or_else(|| {
            invalid(format!(
                "canonical opening at digit column {start} has no source decomposition"
            ))
        })?;
        if trace.negative_columns_start != start + BALANCED_TERNARY_DIGITS
            || trace.borrow_columns_start != start + 2 * BALANCED_TERNARY_DIGITS
            || trace.reconstruction_row != trace.digit_rows_start + DIGIT_ROWS
            || trace.transition_rows_start != trace.reconstruction_row + 1
        {
            return Err(invalid(format!(
                "canonical opening at digit column {start} does not have the exact 82+1+41 layout"
            )));
        }
        let transition_end = trace.transition_rows_start + TRANSITION_ROWS;
        let placement = if transition_end <= sis_start {
            CanonicalOpeningPlacement::Reused
        } else if sis_start <= trace.digit_rows_start && transition_end <= primary.row_start() {
            CanonicalOpeningPlacement::Primary
        } else if primary.row_end() <= trace.digit_rows_start && transition_end <= compression.row_start() {
            CanonicalOpeningPlacement::Compression
        } else if trace.digit_rows_start < claim_start {
            return Err(invalid(format!(
                "canonical opening at digit column {start} overlaps a PiCCS output SIS phase boundary"
            )));
        } else {
            continue;
        };
        if !required.contains(&start) {
            if placement == CanonicalOpeningPlacement::Reused {
                continue;
            }
            return Err(invalid(format!(
                "new canonical opening at digit column {start} is not consumed by its SIS phase"
            )));
        }
        let opening = CanonicalOpeningAudit {
            source_column,
            digit_columns: start..start + BALANCED_TERNARY_DIGITS,
            negative_columns: trace.negative_columns_start..trace.negative_columns_start + BALANCED_TERNARY_DIGITS,
            borrow_columns: trace.borrow_columns_start..trace.borrow_columns_start + BALANCED_TERNARY_DIGITS - 1,
            digit_rows: trace.digit_rows_start..trace.digit_rows_start + DIGIT_ROWS,
            reconstruction_row: trace.reconstruction_row,
            transition_rows: trace.transition_rows_start..transition_end,
            placement,
            owner: None,
        };
        if openings.insert(start, opening).is_some() {
            return Err(invalid(format!(
                "digit column {start} has duplicate canonical openings"
            )));
        }
    }
    Ok(openings)
}

fn validate_primary_words(
    profile: Profile,
    primary: &SeededPhi81LinearBlock,
    openings: &mut BTreeMap<usize, CanonicalOpeningAudit>,
) -> Result<(), R1csIvcError> {
    if primary.word_starts().len() != profile.field_count() {
        return Err(invalid("primary SIS word count differs from its recovered profile"));
    }
    let mut owner_by_source = BTreeMap::<usize, R1csInputOwner>::new();
    for (index, &word_start) in primary.word_starts().iter().enumerate() {
        let owner = profile
            .decode(index)
            .ok_or_else(|| invalid(format!("primary SIS field {index} has no typed path")))?
            .r1cs_input_owner();
        let opening = openings.get_mut(&word_start).ok_or_else(|| {
            invalid(format!(
                "primary SIS field {index} word column {word_start} has no canonical opening"
            ))
        })?;
        if opening.placement == CanonicalOpeningPlacement::Compression {
            return Err(invalid(format!(
                "primary SIS field {index} is opened only in the compression phase"
            )));
        }
        match owner_by_source.insert(opening.source_column, owner) {
            Some(previous) if previous != owner => {
                return Err(invalid(format!(
                    "source column {} is shared by typed owners {previous:?} and {owner:?}",
                    opening.source_column
                )));
            }
            _ => {}
        }
        match opening.owner {
            Some(previous) if previous != owner => {
                return Err(invalid(format!(
                    "canonical opening at digit column {word_start} is shared by typed owners {previous:?} and {owner:?}"
                )));
            }
            Some(_) => {}
            None => opening.owner = Some(owner),
        }
    }
    Ok(())
}

fn validate_compression_words(
    compression: &SeededPhi81LinearBlock,
    primary_outputs: &[usize],
    openings: &BTreeMap<usize, CanonicalOpeningAudit>,
) -> Result<(), R1csIvcError> {
    if compression.word_starts().len() != primary_outputs.len() {
        return Err(invalid(format!(
            "compression SIS has {} words for {} primary outputs",
            compression.word_starts().len(),
            primary_outputs.len()
        )));
    }
    for (index, (&word_start, &output_column)) in compression
        .word_starts()
        .iter()
        .zip(primary_outputs)
        .enumerate()
    {
        let opening = openings.get(&word_start).ok_or_else(|| {
            invalid(format!(
                "compression SIS word {index} column {word_start} has no canonical opening"
            ))
        })?;
        if opening.placement != CanonicalOpeningPlacement::Compression || opening.source_column != output_column {
            return Err(invalid(format!(
                "compression SIS word {index} does not canonically open primary output column {output_column}"
            )));
        }
    }
    Ok(())
}

fn recover_outputs(
    arm: &SparseR1cs,
    primary: &SeededPhi81LinearBlock,
    compression: &SeededPhi81LinearBlock,
) -> Result<(Vec<usize>, Vec<usize>), R1csIvcError> {
    reject_unowned_a_compact_overlap(&arm.a, primary, compression)?;
    let a = scan_sparse_rows(&arm.a, primary, compression, "A")?;
    if a.primary
        .iter()
        .chain(&a.compression)
        .any(|row| !row.is_empty())
    {
        return Err(invalid("seeded Phi81 rows contain unexpected ordinary A terms"));
    }
    reject_compact_overlap(&arm.b, primary, compression, "B")?;
    reject_compact_overlap(&arm.c, primary, compression, "C")?;
    let b = scan_sparse_rows(&arm.b, primary, compression, "B")?;
    let c = scan_sparse_rows(&arm.c, primary, compression, "C")?;
    for row in b.primary.iter().chain(&b.compression) {
        if row.as_slice() != [(0, F::ONE)] {
            return Err(invalid(
                "seeded Phi81 row does not multiply by the constant-one B column",
            ));
        }
    }
    Ok((
        exact_output_columns(&c.primary, "primary")?,
        exact_output_columns(&c.compression, "compression")?,
    ))
}

fn reject_unowned_a_compact_overlap(
    matrix: &CcsMatrix<F>,
    primary: &SeededPhi81LinearBlock,
    compression: &SeededPhi81LinearBlock,
) -> Result<(), R1csIvcError> {
    let overlaps = |start: usize, end: usize| {
        (start < primary.row_end() && primary.row_start() < end)
            || (start < compression.row_end() && compression.row_start() < end)
    };
    if matrix.seeded_phi81_blocks().iter().any(|block| {
        !std::ptr::eq(block, primary)
            && !std::ptr::eq(block, compression)
            && overlaps(block.row_start(), block.row_end())
    }) || matrix
        .geometric_runs()
        .iter()
        .any(|run| overlaps(run.row(), run.row() + 1))
    {
        return Err(invalid(
            "PiCCS output A matrix contains an unowned compact contribution in seeded rows",
        ));
    }
    Ok(())
}

struct BlockRowTerms {
    primary: Vec<Vec<(usize, F)>>,
    compression: Vec<Vec<(usize, F)>>,
}

fn scan_sparse_rows(
    matrix: &CcsMatrix<F>,
    primary: &SeededPhi81LinearBlock,
    compression: &SeededPhi81LinearBlock,
    label: &str,
) -> Result<BlockRowTerms, R1csIvcError> {
    let csc = matrix
        .sparse_component()
        .ok_or_else(|| invalid(format!("PiCCS output {label} matrix has no sparse component")))?;
    let mut rows = BlockRowTerms {
        primary: vec![Vec::new(); primary.row_end() - primary.row_start()],
        compression: vec![Vec::new(); compression.row_end() - compression.row_start()],
    };
    for column in 0..csc.ncols {
        for entry in csc.column_range(column) {
            let row = csc.row_index(entry);
            if (primary.row_start()..primary.row_end()).contains(&row) {
                rows.primary[row - primary.row_start()].push((column, csc.vals[entry]));
            } else if (compression.row_start()..compression.row_end()).contains(&row) {
                rows.compression[row - compression.row_start()].push((column, csc.vals[entry]));
            }
        }
    }
    Ok(rows)
}

fn reject_compact_overlap(
    matrix: &CcsMatrix<F>,
    primary: &SeededPhi81LinearBlock,
    compression: &SeededPhi81LinearBlock,
    label: &str,
) -> Result<(), R1csIvcError> {
    let overlaps = |start: usize, end: usize| {
        (start < primary.row_end() && primary.row_start() < end)
            || (start < compression.row_end() && compression.row_start() < end)
    };
    if matrix
        .seeded_phi81_blocks()
        .iter()
        .any(|block| overlaps(block.row_start(), block.row_end()))
        || matrix
            .geometric_runs()
            .iter()
            .any(|run| overlaps(run.row(), run.row() + 1))
    {
        return Err(invalid(format!(
            "PiCCS output {label} matrix contains a compact contribution in seeded output rows"
        )));
    }
    Ok(())
}

fn exact_output_columns(rows: &[Vec<(usize, F)>], label: &str) -> Result<Vec<usize>, R1csIvcError> {
    rows.iter()
        .enumerate()
        .map(|(index, row)| match row.as_slice() {
            [(column, coefficient)] if *coefficient == F::ONE => Ok(*column),
            _ => Err(invalid(format!(
                "{label} seeded Phi81 output row {index} is not one exact C output wire"
            ))),
        })
        .collect()
}
