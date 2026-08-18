//! Same-wire lifecycle semantic-link regression and compact Lean artifact gate.

#[path = "../gadgets/lean_artifact_support.rs"]
#[allow(dead_code)]
mod lean_artifact_support;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Range;
use std::path::{Path, PathBuf};

use lean_artifact_support::{lean_nat_list, sha256_hex};
use neo_fold_clean::engine::r1cs_circuit::builder::{Poseidon2HashAudit, Poseidon2HashRoundAuditKind};
use neo_fold_clean::engine::r1cs_circuit::{enforce_poseidon2_permutation, R1csBuilder};
use neo_fold_clean::frontends::nebula::f_prime::{
    enforce_streaming_lifecycle_semantic_link, enforce_streaming_lifecycle_source_semantic_link,
    streaming_phase_semantic_digest, StreamingLifecycleBeforePayloadRule, StreamingLifecycleSemanticLinkWires,
    STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS, STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY,
    STREAMING_LIFECYCLE_SEMANTIC_LINK_FAMILY,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: usize = 1;
const PROFILE_ID: &str = "nightstream/goldilocks/streaming-lifecycle-semantic-link/v1";
const ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingLifecycleSemanticLink.lean";
const SOURCE_ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingLifecycleSourceSemanticLink.lean";
const DIGEST_FIELDS: usize = 4;
const DOMAIN_FIELDS: usize = 10;
const HASH_CONSTANT_FIELDS: usize = DOMAIN_FIELDS + 1;
const HASH_INPUT_FIELDS: usize = HASH_CONSTANT_FIELDS + DIGEST_FIELDS + STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS;
const ABSORB_ROUNDS: usize = HASH_INPUT_FIELDS / 4;
const HASH_ROUNDS: usize = ABSORB_ROUNDS + 1;
const POSEIDON2_ROWS: usize = 600;
const ABSORB_ROUND_ROWS: usize = 4 + POSEIDON2_ROWS;
const HASH_TRACE_ROWS: usize = 1 + ABSORB_ROUNDS * ABSORB_ROUND_ROWS + 1 + POSEIDON2_ROWS;
const HASH_TOTAL_ROWS: usize = HASH_CONSTANT_FIELDS + HASH_TRACE_ROWS;
const PAYLOAD_ROWS: usize = 2 * STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS;
const EQUALITY_ROWS: usize = 2 * DIGEST_FIELDS;
const TOTAL_ROWS: usize = PAYLOAD_ROWS + 2 * HASH_TOTAL_ROWS + EQUALITY_ROWS;
const BASE_SOURCE_ROWS: usize = TOTAL_ROWS;
const RECURSIVE_SOURCE_ROWS: usize = TOTAL_ROWS - STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS;
const BASE_SOURCE_ROWS_SHA256: &str = "b647c041e8e632e49c3863a8e27f2ee496dc9347d7bb705a2342b753cf1ad9ba";
const RECURSIVE_SOURCE_ROWS_SHA256: &str = "14bb0c9f0ae92dd3134de5b07a35ea8e7dbff36ab546c4172de53a736ba967fd";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceScope {
    Base,
    Recursive,
}

impl SourceScope {
    const fn profile_id(self) -> &'static str {
        match self {
            Self::Base => "nightstream/goldilocks/streaming-lifecycle-source-semantic-link/base/v1",
            Self::Recursive => "nightstream/goldilocks/streaming-lifecycle-source-semantic-link/recursive/v1",
        }
    }

    const fn source_identity(self) -> &'static str {
        match self {
            Self::Base => "rust:streaming-lifecycle-source-semantic-link/base/v1",
            Self::Recursive => "rust:streaming-lifecycle-source-semantic-link/recursive/v1",
        }
    }

    const fn lean_constructor(self) -> &'static str {
        match self {
            Self::Base => ".base",
            Self::Recursive => ".recursive",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct SparseRow {
    a: Vec<(usize, F)>,
    b: Vec<(usize, F)>,
    c: Vec<(usize, F)>,
}

struct BuiltLink {
    builder: R1csBuilder,
    before_semantic: [usize; DIGEST_FIELDS],
    after_semantic: [usize; DIGEST_FIELDS],
    before_local: [usize; DIGEST_FIELDS],
    after_local: [usize; DIGEST_FIELDS],
    before_payload: [usize; STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS],
    after_payload: [usize; STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS],
}

struct LinkArtifact {
    source_rows_sha256: String,
    row_count: usize,
    column_count: usize,
    before_semantic: [usize; DIGEST_FIELDS],
    after_semantic: [usize; DIGEST_FIELDS],
    before_local: [usize; DIGEST_FIELDS],
    after_local: [usize; DIGEST_FIELDS],
    before_payload_start: usize,
    after_payload_start: usize,
    before_hash_constant_start: usize,
    after_hash_constant_start: usize,
    before_hash_output: [usize; DIGEST_FIELDS],
    after_hash_output: [usize; DIGEST_FIELDS],
    equality_row_start: usize,
    constant_values: Vec<u64>,
}

struct SourceLinkArtifact {
    scope: SourceScope,
    source_rows_sha256: String,
    row_count: usize,
    column_count: usize,
    before_semantic: [usize; DIGEST_FIELDS],
    after_semantic: [usize; DIGEST_FIELDS],
    before_local: [usize; DIGEST_FIELDS],
    after_local: [usize; DIGEST_FIELDS],
    before_payload_start: usize,
    after_payload_start: usize,
    before_hash_constant_start: usize,
    after_hash_constant_start: usize,
    before_hash_output: [usize; DIGEST_FIELDS],
    after_hash_output: [usize; DIGEST_FIELDS],
    before_payload_row_start: usize,
    before_hash_constant_row_start: usize,
    after_payload_row_start: usize,
    after_hash_constant_row_start: usize,
    equality_row_start: usize,
    constant_values: Vec<u64>,
}

fn build_link() -> BuiltLink {
    let before_local_values = [F::from_u64(3); DIGEST_FIELDS];
    let after_local_values = [F::from_u64(5); DIGEST_FIELDS];
    let before_payload_values = std::array::from_fn(|index| F::from_bool(index % 2 == 0));
    let after_payload_values = std::array::from_fn(|index| F::from_bool(index % 3 == 0));
    let before_semantic_values = streaming_phase_semantic_digest(before_local_values, &before_payload_values);
    let after_semantic_values = streaming_phase_semantic_digest(after_local_values, &after_payload_values);

    let mut builder = R1csBuilder::new();
    let before_semantic_wires = before_semantic_values.map(|value| builder.alloc(value));
    let after_semantic_wires = after_semantic_values.map(|value| builder.alloc(value));
    let before_local_wires = before_local_values.map(|value| builder.alloc(value));
    let after_local_wires = after_local_values.map(|value| builder.alloc(value));
    let before_payload_wires = before_payload_values.map(|value| builder.alloc(value));
    let after_payload_wires = after_payload_values.map(|value| builder.alloc(value));
    enforce_streaming_lifecycle_semantic_link(
        &mut builder,
        StreamingLifecycleSemanticLinkWires {
            before_semantic_digest: before_semantic_wires,
            after_semantic_digest: after_semantic_wires,
            before_local_state_digest: before_local_wires,
            after_local_state_digest: after_local_wires,
            before_delayed_payload: &before_payload_wires,
            after_delayed_payload: &after_payload_wires,
        },
    );
    BuiltLink {
        builder,
        before_semantic: before_semantic_wires.map(|wire| wire.col()),
        after_semantic: after_semantic_wires.map(|wire| wire.col()),
        before_local: before_local_wires.map(|wire| wire.col()),
        after_local: after_local_wires.map(|wire| wire.col()),
        before_payload: before_payload_wires.map(|wire| wire.col()),
        after_payload: after_payload_wires.map(|wire| wire.col()),
    }
}

fn build_source_link(scope: SourceScope) -> BuiltLink {
    let before_local_values = [F::from_u64(3); DIGEST_FIELDS];
    let after_local_values = [F::from_u64(5); DIGEST_FIELDS];
    let before_payload_values = match scope {
        SourceScope::Base => [F::ZERO; STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS],
        SourceScope::Recursive => std::array::from_fn(|index| F::from_bool(index % 2 == 0)),
    };
    let after_payload_values = std::array::from_fn(|index| F::from_bool(index % 3 == 0));
    let before_semantic_values = streaming_phase_semantic_digest(before_local_values, &before_payload_values);
    let after_semantic_values = streaming_phase_semantic_digest(after_local_values, &after_payload_values);

    let mut builder = R1csBuilder::new();
    let before_semantic_wires = before_semantic_values.map(|value| builder.alloc(value));
    let after_semantic_wires = after_semantic_values.map(|value| builder.alloc(value));
    let before_local_wires = before_local_values.map(|value| builder.alloc(value));
    let before_payload_wires = before_payload_values.map(|value| builder.alloc(value));
    let after_local_wires = after_local_values.map(|value| builder.alloc(value));
    let after_payload_wires = after_payload_values.map(|value| builder.alloc(value));
    let before_payload_rule = match scope {
        SourceScope::Base => StreamingLifecycleBeforePayloadRule::EnforceZero,
        SourceScope::Recursive => StreamingLifecycleBeforePayloadRule::ReuseBinary,
    };
    enforce_streaming_lifecycle_source_semantic_link(
        &mut builder,
        StreamingLifecycleSemanticLinkWires {
            before_semantic_digest: before_semantic_wires,
            after_semantic_digest: after_semantic_wires,
            before_local_state_digest: before_local_wires,
            after_local_state_digest: after_local_wires,
            before_delayed_payload: &before_payload_wires,
            after_delayed_payload: &after_payload_wires,
        },
        before_payload_rule,
    );
    BuiltLink {
        builder,
        before_semantic: before_semantic_wires.map(|wire| wire.col()),
        after_semantic: after_semantic_wires.map(|wire| wire.col()),
        before_local: before_local_wires.map(|wire| wire.col()),
        after_local: after_local_wires.map(|wire| wire.col()),
        before_payload: before_payload_wires.map(|wire| wire.col()),
        after_payload: after_payload_wires.map(|wire| wire.col()),
    }
}

fn normalize_terms(terms: impl IntoIterator<Item = (usize, F)>) -> Vec<(usize, F)> {
    let mut totals = BTreeMap::<usize, F>::new();
    for (column, coefficient) in terms {
        *totals.entry(column).or_insert(F::ZERO) += coefficient;
    }
    totals
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect()
}

fn normalized_matrix_range(
    total_rows: usize,
    row_range: &Range<usize>,
    trips: &[(usize, usize, F)],
) -> Vec<Vec<(usize, F)>> {
    let mut raw = vec![Vec::new(); row_range.len()];
    for &(row, column, coefficient) in trips {
        assert!(row < total_rows);
        if row_range.contains(&row) {
            raw[row - row_range.start].push((column, coefficient));
        }
    }
    raw.into_iter().map(normalize_terms).collect()
}

fn normalized_rows(builder: &R1csBuilder) -> Vec<SparseRow> {
    let (a, b, c) = builder.sparse_triplets();
    let range = 0..builder.rows();
    let a = normalized_matrix_range(builder.rows(), &range, a);
    let b = normalized_matrix_range(builder.rows(), &range, b);
    let c = normalized_matrix_range(builder.rows(), &range, c);
    (0..builder.rows())
        .map(|row| SparseRow {
            a: a[row].clone(),
            b: b[row].clone(),
            c: c[row].clone(),
        })
        .collect()
}

fn linear_row(output: usize, terms: &[(usize, F)]) -> SparseRow {
    SparseRow {
        a: normalize_terms(
            std::iter::once((output, F::ONE)).chain(
                terms
                    .iter()
                    .map(|&(column, coefficient)| (column, -coefficient)),
            ),
        ),
        b: vec![(0, F::ONE)],
        c: Vec::new(),
    }
}

fn bit_row(column: usize) -> SparseRow {
    SparseRow {
        a: vec![(column, F::ONE)],
        b: normalize_terms([(column, F::ONE), (0, -F::ONE)]),
        c: Vec::new(),
    }
}

fn zero_row(column: usize) -> SparseRow {
    linear_row(column, &[])
}

fn rename_row(row: &SparseRow, column_map: &impl Fn(usize) -> usize) -> SparseRow {
    let rename = |terms: &[(usize, F)]| {
        normalize_terms(
            terms
                .iter()
                .map(|&(column, coefficient)| (column_map(column), coefficient)),
        )
    };
    SparseRow {
        a: rename(&row.a),
        b: rename(&row.b),
        c: rename(&row.c),
    }
}

fn poseidon2_template() -> Vec<SparseRow> {
    let mut builder = R1csBuilder::new();
    let inputs = std::array::from_fn(|lane| builder.alloc(F::from_usize(lane + 1)));
    let _ = enforce_poseidon2_permutation(&mut builder, &inputs);
    assert_eq!(builder.rows(), POSEIDON2_ROWS);
    normalized_rows(&builder)
}

fn assert_row(rows: &[SparseRow], row: usize, expected: SparseRow, label: &str) {
    assert_eq!(rows[row], expected, "{label} at row {row}");
}

fn assert_poseidon2_call(
    rows: &[SparseRow],
    call_rows: Range<usize>,
    inputs: [usize; 8],
    first_allocated: usize,
    template: &[SparseRow],
) {
    assert_eq!(call_rows.len(), POSEIDON2_ROWS);
    for (offset, source) in template.iter().enumerate() {
        let expected = rename_row(source, &|column| match column {
            0 => 0,
            1..=8 => inputs[column - 1],
            _ => first_allocated + column - 9,
        });
        assert_row(rows, call_rows.start + offset, expected, "Poseidon2 permutation");
    }
}

fn validate_hash(
    builder: &R1csBuilder,
    rows: &[SparseRow],
    hash: &Poseidon2HashAudit,
    constant_start: usize,
    local_columns: [usize; DIGEST_FIELDS],
    payload_columns: &[usize],
    template: &[SparseRow],
) -> Vec<u64> {
    let expected_inputs = (constant_start..constant_start + HASH_CONSTANT_FIELDS)
        .chain(local_columns)
        .chain(payload_columns.iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(hash.input_cols, expected_inputs);
    assert_eq!(hash.zero_col, constant_start + HASH_CONSTANT_FIELDS);
    assert_eq!(hash.zero_row, hash.row_start);
    assert_eq!(hash.rounds.len(), HASH_ROUNDS);
    assert_eq!(hash.row_end - hash.row_start, HASH_TRACE_ROWS);

    let constant_row_start = hash.row_start - HASH_CONSTANT_FIELDS;
    let constants = (0..HASH_CONSTANT_FIELDS)
        .map(|offset| {
            let column = constant_start + offset;
            let value = builder.witness()[column].as_canonical_u64();
            assert_row(
                rows,
                constant_row_start + offset,
                linear_row(column, &[(0, F::from_u64(value))]),
                "phase preimage constant",
            );
            value
        })
        .collect::<Vec<_>>();
    assert_row(rows, hash.zero_row, linear_row(hash.zero_col, &[]), "hash zero");

    let mut prior_outputs = [hash.zero_col; 8];
    for round_index in 0..ABSORB_ROUNDS {
        let round = &hash.rounds[round_index];
        let Poseidon2HashRoundAuditKind::Absorb { chunk_cols } = &round.kind else {
            panic!("data round must absorb")
        };
        let chunk = &hash.input_cols[4 * round_index..4 * round_index + 4];
        assert_eq!(chunk_cols, chunk);
        assert_eq!(round.state_before_cols, prior_outputs);
        let definition_start = hash.row_start + 1 + round_index * ABSORB_ROUND_ROWS;
        let column_start = hash.zero_col + 1 + round_index * ABSORB_ROUND_ROWS;
        let inputs = std::array::from_fn(|lane| {
            if lane < 4 {
                column_start + lane
            } else {
                prior_outputs[lane]
            }
        });
        assert_eq!(round.permutation_input_cols, inputs);
        assert_eq!(
            round.defining_rows,
            (definition_start..definition_start + 4).collect::<Vec<_>>()
        );
        for lane in 0..4 {
            assert_row(
                rows,
                definition_start + lane,
                linear_row(inputs[lane], &[(prior_outputs[lane], F::ONE), (chunk[lane], F::ONE)]),
                "hash absorb definition",
            );
        }
        let first_allocated = column_start + 4;
        assert_poseidon2_call(
            rows,
            definition_start + 4..definition_start + ABSORB_ROUND_ROWS,
            inputs,
            first_allocated,
            template,
        );
        let outputs = std::array::from_fn(|lane| first_allocated + 592 + lane);
        assert_eq!(round.permutation_output_cols, outputs);
        prior_outputs = outputs;
    }

    let pad = &hash.rounds[ABSORB_ROUNDS];
    assert_eq!(pad.kind, Poseidon2HashRoundAuditKind::Pad);
    assert_eq!(pad.state_before_cols, prior_outputs);
    let pad_row = hash.row_start + 1 + ABSORB_ROUNDS * ABSORB_ROUND_ROWS;
    let pad_column = hash.zero_col + 1 + ABSORB_ROUNDS * ABSORB_ROUND_ROWS;
    let inputs = std::array::from_fn(|lane| if lane == 0 { pad_column } else { prior_outputs[lane] });
    assert_eq!(pad.permutation_input_cols, inputs);
    assert_eq!(pad.defining_rows, vec![pad_row]);
    assert_row(
        rows,
        pad_row,
        linear_row(pad_column, &[(prior_outputs[0], F::ONE), (0, F::ONE)]),
        "hash padding definition",
    );
    let first_allocated = pad_column + 1;
    assert_poseidon2_call(
        rows,
        pad_row + 1..pad_row + 1 + POSEIDON2_ROWS,
        inputs,
        first_allocated,
        template,
    );
    let outputs = std::array::from_fn(|lane| first_allocated + 592 + lane);
    assert_eq!(pad.permutation_output_cols, outputs);
    assert_eq!(hash.output_cols, outputs[..DIGEST_FIELDS]);
    assert_eq!(hash.row_end, pad_row + 1 + POSEIDON2_ROWS);
    constants
}

fn source_rows_sha256(rows: &[SparseRow]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"nightstream-r1cs-row-range-v1\0");
    hasher.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        for (matrix, terms) in [(0_u8, &row.a), (1, &row.b), (2, &row.c)] {
            hasher.update([matrix]);
            hasher.update((terms.len() as u64).to_le_bytes());
            for &(column, coefficient) in terms {
                hasher.update((column as u64).to_le_bytes());
                hasher.update(coefficient.as_canonical_u64().to_le_bytes());
            }
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn build_source_artifact(scope: SourceScope) -> SourceLinkArtifact {
    let built = build_source_link(scope);
    let builder = &built.builder;
    assert!(builder.is_satisfied());
    let expected_rows = match scope {
        SourceScope::Base => BASE_SOURCE_ROWS,
        SourceScope::Recursive => RECURSIVE_SOURCE_ROWS,
    };
    assert_eq!(builder.rows(), expected_rows);
    assert_eq!(builder.cols(), 665_149);
    let rows = normalized_rows(builder);
    let audits = builder.poseidon2_hash_audits();
    let [before_hash, after_hash] = audits.as_slice() else {
        panic!("source semantic link must own two Poseidon2 hashes")
    };

    let before_row_start = match scope {
        SourceScope::Base => {
            for (row, &column) in built.before_payload.iter().enumerate() {
                assert_row(&rows, row, zero_row(column), "base before-payload zero");
            }
            STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS
        }
        SourceScope::Recursive => 0,
    };
    assert_eq!(before_hash.row_start - HASH_CONSTANT_FIELDS, before_row_start);

    let template = poseidon2_template();
    let before_constants = validate_hash(
        builder,
        &rows,
        before_hash,
        before_hash.input_cols[0],
        built.before_local,
        &built.before_payload,
        &template,
    );
    let after_bits_start = before_hash.row_end;
    for (offset, &column) in built.after_payload.iter().enumerate() {
        assert_row(&rows, after_bits_start + offset, bit_row(column), "after-payload bit");
    }
    assert_eq!(
        after_hash.row_start - HASH_CONSTANT_FIELDS,
        after_bits_start + STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS,
    );
    let after_constants = validate_hash(
        builder,
        &rows,
        after_hash,
        after_hash.input_cols[0],
        built.after_local,
        &built.after_payload,
        &template,
    );
    assert_eq!(before_constants, after_constants);

    let equality_row_start = after_hash.row_end;
    for lane in 0..DIGEST_FIELDS {
        assert_row(
            &rows,
            equality_row_start + lane,
            linear_row(built.before_semantic[lane], &[(before_hash.output_cols[lane], F::ONE)]),
            "before semantic equality",
        );
        assert_row(
            &rows,
            equality_row_start + DIGEST_FIELDS + lane,
            linear_row(built.after_semantic[lane], &[(after_hash.output_cols[lane], F::ONE)]),
            "after semantic equality",
        );
    }
    assert_eq!(equality_row_start + EQUALITY_ROWS, expected_rows);
    SourceLinkArtifact {
        scope,
        source_rows_sha256: source_rows_sha256(&rows),
        row_count: builder.rows(),
        column_count: builder.cols(),
        before_semantic: built.before_semantic,
        after_semantic: built.after_semantic,
        before_local: built.before_local,
        after_local: built.after_local,
        before_payload_start: built.before_payload[0],
        after_payload_start: built.after_payload[0],
        before_hash_constant_start: before_hash.input_cols[0],
        after_hash_constant_start: after_hash.input_cols[0],
        before_hash_output: before_hash.output_cols,
        after_hash_output: after_hash.output_cols,
        before_payload_row_start: 0,
        before_hash_constant_row_start: before_hash.row_start - HASH_CONSTANT_FIELDS,
        after_payload_row_start: after_bits_start,
        after_hash_constant_row_start: after_hash.row_start - HASH_CONSTANT_FIELDS,
        equality_row_start,
        constant_values: before_constants,
    }
}

fn build_artifact() -> LinkArtifact {
    let built = build_link();
    let builder = &built.builder;
    assert!(builder.is_satisfied());
    assert_eq!(builder.rows(), TOTAL_ROWS);
    let rows = normalized_rows(builder);
    let families = builder.row_family_ranges();
    assert_eq!(families.len(), 2);
    assert_eq!(families[0].name, STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY);
    assert_eq!(families[0].row_start..families[0].row_end, 0..PAYLOAD_ROWS);
    assert_eq!(families[1].name, STREAMING_LIFECYCLE_SEMANTIC_LINK_FAMILY);
    assert_eq!(families[1].row_start..families[1].row_end, PAYLOAD_ROWS..TOTAL_ROWS);
    for (row, &column) in built
        .before_payload
        .iter()
        .chain(&built.after_payload)
        .enumerate()
    {
        assert_row(&rows, row, bit_row(column), "payload bit");
    }

    let audits = builder.poseidon2_hash_audits();
    let [before_hash, after_hash] = audits.as_slice() else {
        panic!("semantic link must own two Poseidon2 hashes")
    };
    let before_constant_start = before_hash.input_cols[0];
    let after_constant_start = after_hash.input_cols[0];
    let template = poseidon2_template();
    let before_constants = validate_hash(
        builder,
        &rows,
        before_hash,
        before_constant_start,
        built.before_local,
        &built.before_payload,
        &template,
    );
    let after_constants = validate_hash(
        builder,
        &rows,
        after_hash,
        after_constant_start,
        built.after_local,
        &built.after_payload,
        &template,
    );
    assert_eq!(before_constants, after_constants);
    assert_eq!(before_hash.row_start - HASH_CONSTANT_FIELDS, PAYLOAD_ROWS);
    assert_eq!(
        after_hash.row_start - HASH_CONSTANT_FIELDS,
        PAYLOAD_ROWS + HASH_TOTAL_ROWS
    );
    let equality_row_start = after_hash.row_end;
    for lane in 0..DIGEST_FIELDS {
        assert_row(
            &rows,
            equality_row_start + 2 * lane,
            linear_row(built.before_semantic[lane], &[(before_hash.output_cols[lane], F::ONE)]),
            "before semantic equality",
        );
        assert_row(
            &rows,
            equality_row_start + 2 * lane + 1,
            linear_row(built.after_semantic[lane], &[(after_hash.output_cols[lane], F::ONE)]),
            "after semantic equality",
        );
    }
    assert_eq!(equality_row_start + EQUALITY_ROWS, TOTAL_ROWS);
    assert_eq!(
        built.before_payload,
        std::array::from_fn(|offset| built.before_payload[0] + offset)
    );
    assert_eq!(
        built.after_payload,
        std::array::from_fn(|offset| built.after_payload[0] + offset)
    );

    LinkArtifact {
        source_rows_sha256: source_rows_sha256(&rows),
        row_count: builder.rows(),
        column_count: builder.cols(),
        before_semantic: built.before_semantic,
        after_semantic: built.after_semantic,
        before_local: built.before_local,
        after_local: built.after_local,
        before_payload_start: built.before_payload[0],
        after_payload_start: built.after_payload[0],
        before_hash_constant_start: before_constant_start,
        after_hash_constant_start: after_constant_start,
        before_hash_output: before_hash.output_cols,
        after_hash_output: after_hash.output_cols,
        equality_row_start,
        constant_values: before_constants,
    }
}

fn render_artifact() -> String {
    assert_eq!(STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS, 2_169);
    assert_eq!(HASH_INPUT_FIELDS, 2_184);
    assert_eq!(HASH_TOTAL_ROWS, 330_397);
    assert_eq!(TOTAL_ROWS, 665_140);
    let artifact = build_artifact();
    let mut payload = String::new();
    writeln!(
        payload,
        "def phaseConstantValues : List Nat := {}",
        lean_nat_list(artifact.constant_values.iter().map(|&value| value as usize)),
    )
    .unwrap();
    writeln!(
        payload,
        "\ndef rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := {SCHEMA_VERSION}, profileId := \"{PROFILE_ID}\",\n    \
            sourceIdentity := \"rust:streaming-lifecycle-semantic-link/v1\",\n    \
            sourceRowsSha256 := \"{}\", rowCount := {}, columnCount := {},\n    \
            constantValues := phaseConstantValues,\n    \
            beforeSemanticColumns := {}, afterSemanticColumns := {},\n    \
            beforeLocalColumns := {}, afterLocalColumns := {},\n    \
            beforePayloadStartColumn := {}, afterPayloadStartColumn := {},\n    \
            beforeHashConstantStartColumn := {}, afterHashConstantStartColumn := {},\n    \
            beforeHashOutputColumns := {}, afterHashOutputColumns := {},\n    \
            equalityRowStart := {} }}",
        artifact.source_rows_sha256,
        artifact.row_count,
        artifact.column_count,
        lean_nat_list(artifact.before_semantic),
        lean_nat_list(artifact.after_semantic),
        lean_nat_list(artifact.before_local),
        lean_nat_list(artifact.after_local),
        artifact.before_payload_start,
        artifact.after_payload_start,
        artifact.before_hash_constant_start,
        artifact.after_hash_constant_start,
        lean_nat_list(artifact.before_hash_output),
        lean_nat_list(artifact.after_hash_output),
        artifact.equality_row_start,
    )
    .unwrap();
    let artifact_hash = sha256_hex(&payload);
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSemanticLinkSchema\n\n\
         /-! Generated compact geometry for the Rust lifecycle semantic-link family.\n\n\
         The Rust generator compares every represented source row with the compact recipe.\n\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSemanticLink\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact\n\n\
         def artifactSha256 : String := \"{artifact_hash}\"\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSemanticLink\n",
    )
}

fn generated_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ARTIFACT_PATH)
}

fn render_source_record(out: &mut String, name: &str, artifact: &SourceLinkArtifact) {
    writeln!(
        out,
        "\ndef {name} : SourceArtifact :=\n  \
         {{ scope := {}, schemaVersion := {SCHEMA_VERSION},\n    \
            profileId := \"{}\", sourceIdentity := \"{}\",\n    \
            sourceRowsSha256 := \"{}\", rowCount := {}, columnCount := {},\n    \
            constantValues := phaseConstantValues,\n    \
            beforeSemanticColumns := {}, afterSemanticColumns := {},\n    \
            beforeLocalColumns := {}, afterLocalColumns := {},\n    \
            beforePayloadStartColumn := {}, afterPayloadStartColumn := {},\n    \
            beforeHashConstantStartColumn := {}, afterHashConstantStartColumn := {},\n    \
            beforeHashOutputColumns := {}, afterHashOutputColumns := {},\n    \
            beforePayloadRowStart := {}, beforeHashConstantRowStart := {},\n    \
            afterPayloadRowStart := {}, afterHashConstantRowStart := {},\n    \
            equalityRowStart := {} }}",
        artifact.scope.lean_constructor(),
        artifact.scope.profile_id(),
        artifact.scope.source_identity(),
        artifact.source_rows_sha256,
        artifact.row_count,
        artifact.column_count,
        lean_nat_list(artifact.before_semantic),
        lean_nat_list(artifact.after_semantic),
        lean_nat_list(artifact.before_local),
        lean_nat_list(artifact.after_local),
        artifact.before_payload_start,
        artifact.after_payload_start,
        artifact.before_hash_constant_start,
        artifact.after_hash_constant_start,
        lean_nat_list(artifact.before_hash_output),
        lean_nat_list(artifact.after_hash_output),
        artifact.before_payload_row_start,
        artifact.before_hash_constant_row_start,
        artifact.after_payload_row_start,
        artifact.after_hash_constant_row_start,
        artifact.equality_row_start,
    )
    .unwrap();
}

fn render_source_artifact() -> String {
    let base = build_source_artifact(SourceScope::Base);
    let recursive = build_source_artifact(SourceScope::Recursive);
    assert_eq!(base.source_rows_sha256, BASE_SOURCE_ROWS_SHA256);
    assert_eq!(recursive.source_rows_sha256, RECURSIVE_SOURCE_ROWS_SHA256);
    assert_eq!(base.constant_values, recursive.constant_values);
    let mut payload = String::new();
    writeln!(
        payload,
        "def phaseConstantValues : List Nat := {}",
        lean_nat_list(base.constant_values.iter().map(|&value| value as usize)),
    )
    .unwrap();
    render_source_record(&mut payload, "baseArtifact", &base);
    render_source_record(&mut payload, "recursiveArtifact", &recursive);
    let artifact_hash = sha256_hex(&payload);
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSourceSemanticLinkSchema\n\n\
         /-! Generated compact geometry for the exact base and recursive lifecycle semantic-link source stages.\n\n\
         The Rust generator compares every represented source row with its scope-specific compact recipe.\n\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink.Artifact\n\n\
         def artifactSha256 : String := \"{artifact_hash}\"\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink\n",
    )
}

fn generated_source_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(SOURCE_ARTIFACT_PATH)
}

#[test]
fn lifecycle_semantic_link_recomputes_both_phase_envelopes() {
    let mut built = build_link();
    let builder = &mut built.builder;

    assert!(builder.is_satisfied());
    assert_eq!(builder.row_family_ranges().len(), 2);
    assert_eq!(
        builder.row_family_ranges()[0].name,
        STREAMING_LIFECYCLE_PAYLOAD_DOMAIN_FAMILY,
    );
    assert_eq!(
        builder.row_family_ranges()[1].name,
        STREAMING_LIFECYCLE_SEMANTIC_LINK_FAMILY,
    );

    let changed_column = built.before_local[0];
    let original = builder.witness()[changed_column];
    builder.tamper_witness(changed_column, original + F::ONE);
    assert!(!builder.is_satisfied());
    assert!(builder.first_unsatisfied_row().is_some_and(
        |row| builder.row_family_ranges()[1].row_start <= row && row < builder.row_family_ranges()[1].row_end
    ));
    builder.tamper_witness(changed_column, original);
    assert!(builder.is_satisfied());

    let changed_column = built.after_semantic[0];
    let original = builder.witness()[changed_column];
    builder.tamper_witness(changed_column, original + F::ONE);
    assert!(!builder.is_satisfied());
}

#[test]
fn lifecycle_source_semantic_link_rows_match_scope_recipes() {
    let base_hash = build_source_artifact(SourceScope::Base).source_rows_sha256;
    let recursive_hash = build_source_artifact(SourceScope::Recursive).source_rows_sha256;
    eprintln!(
        "streaming lifecycle source semantic-link rows: base_rows={BASE_SOURCE_ROWS} base_sha256={base_hash} recursive_rows={RECURSIVE_SOURCE_ROWS} recursive_sha256={recursive_hash}",
    );
    assert_eq!(base_hash, BASE_SOURCE_ROWS_SHA256);
    assert_eq!(recursive_hash, RECURSIVE_SOURCE_ROWS_SHA256);
}

#[test]
fn lifecycle_source_semantic_link_artifact_is_current() {
    let path = generated_source_artifact_path();
    let rendered = render_source_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected lifecycle source semantic-link artifact");
        panic!(
            "lifecycle source semantic-link Lean artifact drifted; inspect {}",
            expected.display()
        );
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_lifecycle_source_semantic_link_artifact() {
    std::fs::write(generated_source_artifact_path(), render_source_artifact())
        .expect("write generated lifecycle source semantic-link artifact");
}

#[test]
fn lifecycle_semantic_link_artifact_is_current() {
    let path = generated_artifact_path();
    let rendered = render_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected lifecycle semantic-link artifact");
        panic!(
            "lifecycle semantic-link Lean artifact drifted; inspect {}",
            expected.display()
        );
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_lifecycle_semantic_link_artifact() {
    std::fs::write(generated_artifact_path(), render_artifact())
        .expect("write generated lifecycle semantic-link artifact");
}
