//! Source-language and decoder extraction for aggregate acceptance.
//!
//! Owns: internal validation of the exact 4-row/2-column role schema, the
//! canonical inverse decoder, and source/encoded column geometry needed to
//! normalize the nine lowered rows.
//!
//! Does not own: generated artifact data, emitted matrix rows, the CCS
//! polynomial, or recursive physical placement.
//!
//! | Branch | Exact obligation | Failure mode |
//! |---|---|---|
//! | Source roles | one, 16 bits, accept, inverse have unique columns | overlap or order drift |
//! | Source rows | all 64 chunks normalize to the same four equations | row/coefficient drift |
//! | Inverse decoder | `inverse = 0` iff difference is zero, otherwise `difference^-1` | decoder or escape drift |
//! | Geometry | source and encoded columns retain their role order | range/order drift |

use std::collections::{BTreeMap, BTreeSet};

use neo_fold_clean::engine::r1cs_circuit::{AcceptanceTraceEntry, Lc, R1csEncodingTrace, R1csSnapshot, Var};
use neo_fold_clean::frontends::f_prime::gadget_native::EncodedGadgetNativeR1cs;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::{
    signed, CanonicalInverseDecoder, ChunkGeometry, RoleTerm, SourceRole, SourceRow, ACCEPTANCE_COORDINATES_PER_CHUNK,
    ACTIVE_ROWS_PER_CHUNK, CHUNKS, SOURCE_COLUMNS_PER_CHUNK, SOURCE_INPUTS_PER_CHUNK, SOURCE_ROWS_PER_CHUNK,
};

pub(super) struct SourceAudit {
    pub chunks: Vec<ChunkGeometry>,
}

fn source_role_map(event: &AcceptanceTraceEntry) -> BTreeMap<usize, SourceRole> {
    let mut roles = BTreeMap::from([(Var::ONE.col(), SourceRole::One)]);
    for (index, variable) in event.chunk_bits.iter().enumerate() {
        assert!(roles
            .insert(variable.col(), SourceRole::ChunkBit(index))
            .is_none());
    }
    assert!(roles
        .insert(event.accept.col(), SourceRole::Accept)
        .is_none());
    assert!(roles
        .insert(event.inverse.col(), SourceRole::Inverse)
        .is_none());
    assert_eq!(roles.len(), SOURCE_INPUTS_PER_CHUNK + SOURCE_COLUMNS_PER_CHUNK + 1);
    roles
}

fn role_terms(row: &[(usize, F)], roles: &BTreeMap<usize, SourceRole>) -> Vec<RoleTerm<SourceRole>> {
    row.iter()
        .map(|&(column, coefficient)| RoleTerm {
            role: *roles
                .get(&column)
                .unwrap_or_else(|| panic!("unowned acceptance source column {column}")),
            coefficient: signed(coefficient),
        })
        .collect()
}

fn source_schema(source: &R1csSnapshot, event: &AcceptanceTraceEntry) -> Vec<SourceRow> {
    assert_eq!(event.source_rows.len(), SOURCE_ROWS_PER_CHUNK);
    assert_eq!(event.allocated_columns.len(), SOURCE_COLUMNS_PER_CHUNK);
    assert_eq!(event.accept.col(), event.allocated_columns.start);
    assert_eq!(event.inverse.col(), event.allocated_columns.start + 1);
    let roles = source_role_map(event);
    event
        .source_rows
        .clone()
        .map(|row| SourceRow {
            a: role_terms(source.a_row(row), &roles),
            b: role_terms(source.b_row(row), &roles),
            c: role_terms(source.c_row(row), &roles),
        })
        .collect()
}

fn decoder_terms(lc: &Lc, roles: &BTreeMap<usize, SourceRole>) -> Vec<RoleTerm<SourceRole>> {
    let mut terms = lc
        .terms
        .iter()
        .map(|&(column, coefficient)| RoleTerm {
            role: *roles
                .get(&column)
                .unwrap_or_else(|| panic!("unowned canonical-inverse input {column}")),
            coefficient: signed(coefficient),
        })
        .collect::<Vec<_>>();
    if lc.constant != F::ZERO {
        terms.push(RoleTerm {
            role: SourceRole::One,
            coefficient: signed(lc.constant),
        });
    }
    terms
}

fn inverse_owned_offsets(rows: &[SourceRow]) -> Vec<usize> {
    rows.iter()
        .enumerate()
        .filter_map(|(row, equation)| {
            equation
                .a
                .iter()
                .chain(&equation.b)
                .chain(&equation.c)
                .any(|term| term.role == SourceRole::Inverse)
                .then_some(row)
        })
        .collect()
}

fn encoded_singleton(encoded: &EncodedGadgetNativeR1cs, column: usize) -> usize {
    let range = encoded
        .plan
        .encoded_range_for_source_column(column)
        .unwrap_or_else(|| panic!("source column {column} must have one retained coordinate"));
    assert_eq!(range.len(), 1);
    range.start
}

pub(super) fn extract(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    encoded: &EncodedGadgetNativeR1cs,
) -> SourceAudit {
    let events = trace.acceptance_chunks();
    assert_eq!(events.len(), CHUNKS);
    let source_rows = source_schema(source, &events[0]);
    let owned_offsets = inverse_owned_offsets(&source_rows);
    assert_eq!(owned_offsets, [2, 3], "canonical inverse source-row roles");

    let mut strict_inverse_owner = vec![None; source.cols()];
    let mut source_row_roles = BTreeSet::new();
    let mut source_input_roles = BTreeSet::new();
    let mut source_allocated_roles = BTreeSet::new();
    let mut encoded_input_roles = BTreeSet::new();
    let mut encoded_acceptance_roles = BTreeSet::new();
    let mut chunks = Vec::with_capacity(CHUNKS);
    let mut representative_decoder = None;
    for (chunk, event) in events.iter().enumerate() {
        assert_eq!(
            source_schema(source, event),
            source_rows,
            "source role schema drift at chunk {chunk}"
        );
        assert!(event
            .chunk_bits
            .iter()
            .all(|variable| variable.col() < event.allocated_columns.start));
        assert!(strict_inverse_owner[event.inverse.col()]
            .replace(chunk)
            .is_none());
        for row in event.source_rows.clone() {
            assert!(source_row_roles.insert(row), "duplicate source-row role {row}");
        }
        for variable in event.chunk_bits {
            assert!(
                source_input_roles.insert(variable.col()),
                "duplicate source-input role {}",
                variable.col()
            );
        }
        for column in event.allocated_columns.clone() {
            assert!(
                source_allocated_roles.insert(column),
                "duplicate source-allocated role {column}"
            );
        }

        let audit = encoded
            .plan
            .aggregate_acceptance_audit(chunk)
            .expect("role-specific aggregate-acceptance audit");
        assert_eq!(audit.inverse_source_column, event.inverse.col());
        assert_eq!(audit.encoded_outputs.len(), ACCEPTANCE_COORDINATES_PER_CHUNK - 1);
        assert_eq!(audit.radix_weights.len(), ACCEPTANCE_COORDINATES_PER_CHUNK - 1);
        assert_eq!(encoded_singleton(encoded, event.accept.col()), audit.encoded_accept);

        let roles = source_role_map(event);
        let decoder = CanonicalInverseDecoder {
            output: SourceRole::Inverse,
            difference: decoder_terms(audit.inverse_difference, &roles),
            owned_row_offsets: owned_offsets.clone(),
        };
        if let Some(representative) = &representative_decoder {
            assert_eq!(representative, &decoder, "inverse decoder drift at chunk {chunk}");
        } else {
            representative_decoder = Some(decoder);
        }

        let encoded_input_columns = event
            .chunk_bits
            .iter()
            .map(|variable| encoded_singleton(encoded, variable.col()))
            .collect::<Vec<_>>();
        assert_eq!(encoded_input_columns.len(), SOURCE_INPUTS_PER_CHUNK);
        assert_eq!(
            encoded_input_columns
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            SOURCE_INPUTS_PER_CHUNK
        );
        let mut encoded_acceptance_columns = vec![audit.encoded_accept];
        encoded_acceptance_columns.extend(audit.encoded_outputs.clone());
        assert_eq!(encoded_acceptance_columns.len(), ACCEPTANCE_COORDINATES_PER_CHUNK);
        assert_eq!(
            encoded_acceptance_columns
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            ACCEPTANCE_COORDINATES_PER_CHUNK
        );
        assert!(encoded_input_columns
            .iter()
            .all(|column| !encoded_acceptance_columns.contains(column)));
        for &column in &encoded_input_columns {
            assert!(
                encoded_input_roles.insert(column),
                "duplicate encoded-input role {column}"
            );
        }
        for &column in &encoded_acceptance_columns {
            assert!(
                encoded_acceptance_roles.insert(column),
                "duplicate encoded-acceptance role {column}"
            );
        }

        chunks.push(ChunkGeometry {
            source_row_start: event.source_rows.start,
            source_row_end: event.source_rows.end,
            source_column_start: event.allocated_columns.start,
            source_column_end: event.allocated_columns.end,
            source_input_columns: event
                .chunk_bits
                .iter()
                .map(|variable| variable.col())
                .collect(),
            source_accept_column: event.accept.col(),
            source_inverse_column: event.inverse.col(),
            encoded_input_columns,
            encoded_acceptance_columns,
            active_row_start: 0,
            active_row_end: ACTIVE_ROWS_PER_CHUNK,
        });
    }

    for row in 0..source.rows() {
        for &(column, _) in source
            .a_row(row)
            .iter()
            .chain(source.b_row(row))
            .chain(source.c_row(row))
        {
            if let Some(chunk) = strict_inverse_owner[column] {
                assert!(
                    events[chunk].source_rows.contains(&row),
                    "canonical inverse escaped chunk {chunk} at source row {row}"
                );
            }
        }
    }

    assert_eq!(source_row_roles.len(), CHUNKS * SOURCE_ROWS_PER_CHUNK);
    assert_eq!(source_input_roles.len(), CHUNKS * SOURCE_INPUTS_PER_CHUNK);
    assert_eq!(source_allocated_roles.len(), CHUNKS * SOURCE_COLUMNS_PER_CHUNK);
    assert!(source_input_roles.is_disjoint(&source_allocated_roles));
    assert_eq!(encoded_input_roles.len(), CHUNKS * SOURCE_INPUTS_PER_CHUNK);
    assert_eq!(
        encoded_acceptance_roles.len(),
        CHUNKS * ACCEPTANCE_COORDINATES_PER_CHUNK
    );
    assert!(encoded_input_roles.is_disjoint(&encoded_acceptance_roles));

    let _ = representative_decoder.expect("representative canonical inverse decoder");
    SourceAudit { chunks }
}
