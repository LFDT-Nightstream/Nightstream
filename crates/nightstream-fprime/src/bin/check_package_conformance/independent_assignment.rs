//! Independent evaluation of one sealed-package assignment.

use rayon::prelude::*;

use super::*;

const ROW_OWNER_NAMES: [&str; 12] = [
    "statement_binding",
    "statement_absorption",
    "challenge_derivation",
    "round_transcript",
    "initial_claim",
    "sumcheck_chain",
    "eval_k",
    "eval_a",
    "ccs_terminal",
    "norm_terminal",
    "final_identity",
    "output_binding",
];

const COLUMN_OWNER_NAMES: [&str; 14] = [
    "external",
    "statement_binding",
    "statement_absorption",
    "challenge_derivation",
    "round_transcript",
    "initial_claim",
    "sumcheck_chain",
    "eval_k",
    "eval_a",
    "ccs_terminal",
    "norm_terminal",
    "final_identity",
    "output_binding",
    "r1cs_intermediate",
];

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawOwnershipAudit {
    schema_version: u64,
    structural_identity: [String; 4],
    row_spans: Vec<RawOwnerSpan>,
    column_spans: Vec<RawOwnerSpan>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOwnerSpan {
    owner: String,
    start: u64,
    count: u64,
}

fn identity_word(value: &str) -> u64 {
    let word = value
        .parse::<u64>()
        .expect("ownership identity decimal word");
    assert_eq!(value, word.to_string(), "canonical ownership identity word");
    assert!(word < GOLDILOCKS_MODULUS, "canonical ownership identity field word");
    word
}

fn owner_spans(raw: Vec<RawOwnerSpan>, expected: &[&'static str], allow_zero: bool) -> Vec<OwnerSpan> {
    assert_eq!(raw.len(), expected.len(), "ownership span count");
    let spans = raw
        .into_iter()
        .zip(expected.iter().copied())
        .map(|(span, expected_owner)| {
            assert_eq!(span.owner, expected_owner, "ownership tag order");
            let start = word(span.start);
            let count = word(span.count);
            assert!(allow_zero || count != 0, "nonempty row-owner span");
            let end = start.checked_add(count).expect("ownership span end");
            OwnerSpan {
                name: expected_owner,
                start,
                end,
            }
        })
        .collect::<Vec<_>>();
    for adjacent in spans.windows(2) {
        assert_eq!(adjacent[0].end, adjacent[1].start, "ownership span adjacency");
    }
    spans
}

fn ownership_inventory(bytes: &[u8], expected_structural_identity: [u64; 4]) -> OwnershipInventory {
    let raw: RawOwnershipAudit = serde_json::from_slice(bytes).expect("independent PiCCS ownership audit decode");
    assert_eq!(raw.schema_version, 1, "PiCCS ownership audit schema");
    let structural_identity = raw.structural_identity.map(|value| identity_word(&value));
    assert_eq!(
        structural_identity, expected_structural_identity,
        "PiCCS ownership audit structural identity",
    );
    let row_spans = owner_spans(raw.row_spans, &ROW_OWNER_NAMES, false);
    let column_spans = owner_spans(raw.column_spans, &COLUMN_OWNER_NAMES, true);
    let zero_column_names = column_spans
        .iter()
        .filter(|span| span.start == span.end)
        .map(|span| span.name)
        .collect::<Vec<_>>();
    assert_eq!(
        zero_column_names,
        ["statement_binding", "sumcheck_chain"],
        "exact zero-column owner families",
    );
    assert_eq!(column_spans.first().map(|span| span.start), Some(0));
    OwnershipInventory {
        row_spans,
        column_spans,
    }
}

fn reference_layout(raw: &RawPackage) -> ReferenceLayout {
    let domain_size = 1usize << 28;
    ReferenceLayout {
        unpadded_rows: word(raw.3 .0),
        unpadded_constant: word(raw.3 .2),
        public_columns: word(raw.3 .3),
        domain_size,
        final_columns: domain_size + 1 + word(raw.3 .3),
    }
}

fn sealed_raw(bytes: &[u8]) -> (RawPackage, ReferenceLayout) {
    let sealed: Value = serde_json::from_slice(bytes).expect("independent sealed-package decode");
    let raw: RawPackage = serde_json::from_value(sealed[1].clone()).expect("independent raw-package decode");
    assert_eq!(raw.0, 8, "independent raw-package schema");
    let layout = reference_layout(&raw);
    (raw, layout)
}

fn expanded_raw(bytes: &[u8]) -> (RawPackage, ReferenceLayout) {
    let raw: RawPackage = serde_json::from_slice(bytes).expect("independent Lean final-package decode");
    assert_eq!(raw.0, 8, "independent Lean final-package schema");
    let layout = reference_layout(&raw);
    (raw, layout)
}

fn compare_raw_matrices(
    raw: &RawPackage,
    layout: &ReferenceLayout,
    matrices: &nightstream_fprime::PackageR1cs,
) -> [usize; 3] {
    for matrix in [matrices.a(), matrices.b(), matrices.c()] {
        assert_eq!(matrix.rows(), layout.domain_size);
        assert_eq!(matrix.columns(), layout.final_columns);
    }
    let schedule = events(raw);
    schedule.par_iter().for_each(|&event| {
        for ordinal in 0..event_row_count(event, raw) {
            let row_index = event.row_start() + ordinal;
            for (name, side, matrix) in [
                ("A", MatrixSide::A, matrices.a()),
                ("B", MatrixSide::B, matrices.b()),
                ("C", MatrixSide::C, matrices.c()),
            ] {
                compare_row(
                    matrix,
                    row_index,
                    &expected_row(event, &raw.5, ordinal, side, layout),
                    name,
                );
            }
        }
    });
    for matrix in [matrices.a(), matrices.b(), matrices.c()] {
        let end = matrix.nonzero_count();
        assert!(matrix.row_offsets()[layout.unpadded_rows..]
            .iter()
            .all(|offset| *offset == end));
    }
    [
        matrices.a().nonzero_count(),
        matrices.b().nonzero_count(),
        matrices.c().nonzero_count(),
    ]
}

/// Compare Rust's final padded matrix objects with every raw Lean row carried
/// inside the sealed package.
pub fn compare_sealed_matrices(bytes: &[u8], matrices: &nightstream_fprime::PackageR1cs) -> [usize; 3] {
    let (raw, layout) = sealed_raw(bytes);
    compare_raw_matrices(&raw, &layout, matrices)
}

/// Compare Rust's final padded matrix objects with the separately emitted
/// Lean final-package reference.
pub fn compare_lean_expanded_matrices(bytes: &[u8], matrices: &nightstream_fprime::PackageR1cs) -> [usize; 3] {
    let (raw, layout) = expanded_raw(bytes);
    compare_raw_matrices(&raw, &layout, matrices)
}

/// Require one real raw-row rejection for each named PiCCS row and column
/// owner. Raw expansion locates each owned source column, and the separate
/// raw-package evaluator makes every acceptance decision.
pub fn check_piccs_owner_mutations(
    bytes: &[u8],
    ownership_bytes: &[u8],
    expected_structural_identity: [u64; 4],
    expected_row_start: usize,
    expected_row_end: usize,
    private_values: &[u64],
    public_values: &[u64],
) -> PiCcsOwnerMutationReport {
    let (raw, layout) = sealed_raw(bytes);
    let inventory = ownership_inventory(ownership_bytes, expected_structural_identity);
    assert!(expected_row_start < expected_row_end, "PiCCS row interval");
    assert_eq!(
        inventory.row_spans.first().map(|span| span.start),
        Some(expected_row_start),
        "PiCCS owner rows start at the independently fixed prefix boundary",
    );
    assert_eq!(
        inventory.row_spans.last().map(|span| span.end),
        Some(expected_row_end),
        "PiCCS owner rows cover the independently checked prefix",
    );
    assert!(expected_row_end <= layout.unpadded_rows, "PiCCS row prefix bound");
    let row_mutations =
        owner_mutations::row_owner_mutation_checks(&raw, &inventory.row_spans, private_values, public_values);
    let column_mutations = owner_mutations::column_owner_mutation_checks(
        &raw,
        &inventory.row_spans,
        &inventory.column_spans,
        &layout,
        private_values,
        public_values,
    );
    let public_mutations = owner_mutations::public_segment_mutation_checks(
        &raw,
        &inventory.row_spans,
        &layout,
        private_values,
        public_values,
    );
    PiCcsOwnerMutationReport {
        row_families: inventory.row_spans.len(),
        row_mutations,
        column_families: inventory.column_spans.len(),
        zero_column_families: inventory
            .column_spans
            .iter()
            .filter(|span| span.start == span.end)
            .count(),
        column_mutations,
        public_segments: raw.3 .6.len(),
        public_mutations,
    }
}
