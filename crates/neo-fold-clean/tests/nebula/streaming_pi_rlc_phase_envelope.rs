//! Exact compact artifact gate for the PiRLC carry-phase semantic envelope.
//!
//! The Rust rows are authoritative. This test checks every row in both
//! 662,971-row phase ranges against the compact alias, Boolean, constant-pin,
//! sponge-definition, and renamed Poseidon2 recipes before it emits Lean data.

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
    NebulaFPrimePiRlcFamilyBodySynthesis, NebulaFPrimePiRlcFamilyReplayArmKind, STREAMING_CARRY_PHASE_ENVELOPE_FAMILY,
    STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS, STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY, STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: usize = 1;
const PROFILE_ID: &str = "nebula-f-prime-streaming-pi-rlc-phase-envelope-v1";
const ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.lean";
const DIGEST_FIELDS: usize = 4;
const DOMAIN_FIELDS: usize = 10;
const HASH_CONSTANT_FIELDS: usize = DOMAIN_FIELDS + 1;
const HASH_INPUT_FIELDS: usize = HASH_CONSTANT_FIELDS + DIGEST_FIELDS + STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS;
const ABSORB_ROUNDS: usize = HASH_INPUT_FIELDS / 4;
const HASH_ROUNDS: usize = ABSORB_ROUNDS + 1;
const POSEIDON2_ROWS: usize = 600;
const ABSORB_DEFINITION_ROWS: usize = 4;
const ABSORB_ROUND_ROWS: usize = ABSORB_DEFINITION_ROWS + POSEIDON2_ROWS;
const HASH_TRACE_ROWS: usize = 1 + ABSORB_ROUNDS * ABSORB_ROUND_ROWS + 1 + POSEIDON2_ROWS;
const HASH_TOTAL_ROWS: usize = HASH_CONSTANT_FIELDS + HASH_TRACE_ROWS;
const ALIAS_AND_PAYLOAD_ROWS: usize = DIGEST_FIELDS + STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS + DIGEST_FIELDS;
const PHASE_ROWS: usize = ALIAS_AND_PAYLOAD_ROWS + 2 * HASH_TOTAL_ROWS;
const X_OUT_SEMANTIC_START: usize = 19;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct SparseRow {
    a: Vec<(usize, F)>,
    b: Vec<(usize, F)>,
    c: Vec<(usize, F)>,
}

struct ArmArtifact {
    source_identity: &'static str,
    source_rows_sha256: String,
    body_rows: usize,
    body_columns: usize,
    phase_rows: Range<usize>,
    phase_columns: Range<usize>,
    before_local_source_columns: [usize; DIGEST_FIELDS],
    after_local_source_columns: [usize; DIGEST_FIELDS],
    before_local_alias_columns: [usize; DIGEST_FIELDS],
    after_local_alias_columns: [usize; DIGEST_FIELDS],
    payload_start_column: usize,
    before_hash_constant_start_column: usize,
    after_hash_constant_start_column: usize,
    before_semantic_digest_columns: [usize; DIGEST_FIELDS],
    after_semantic_digest_columns: [usize; DIGEST_FIELDS],
    before_x_out_semantic_columns: [usize; DIGEST_FIELDS],
    after_x_out_semantic_columns: [usize; DIGEST_FIELDS],
    constant_values: Vec<u64>,
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

fn normalized_rows_range(builder: &R1csBuilder, row_range: Range<usize>) -> Vec<SparseRow> {
    assert!(row_range.start <= row_range.end && row_range.end <= builder.rows());
    let (a, b, c) = builder.sparse_triplets();
    let a = normalized_matrix_range(builder.rows(), &row_range, a);
    let b = normalized_matrix_range(builder.rows(), &row_range, b);
    let c = normalized_matrix_range(builder.rows(), &row_range, c);
    (0..row_range.len())
        .map(|row| SparseRow {
            a: a[row].clone(),
            b: b[row].clone(),
            c: c[row].clone(),
        })
        .collect()
}

fn linear_row(output: usize, terms: &[(usize, F)]) -> SparseRow {
    let a = std::iter::once((output, F::ONE)).chain(
        terms
            .iter()
            .map(|&(column, coefficient)| (column, -coefficient)),
    );
    SparseRow {
        a: normalize_terms(a),
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

fn rename_terms(terms: &[(usize, F)], column_map: &impl Fn(usize) -> usize) -> Vec<(usize, F)> {
    normalize_terms(
        terms
            .iter()
            .map(|&(column, coefficient)| (column_map(column), coefficient)),
    )
}

fn rename_row(row: &SparseRow, column_map: &impl Fn(usize) -> usize) -> SparseRow {
    SparseRow {
        a: rename_terms(&row.a, column_map),
        b: rename_terms(&row.b, column_map),
        c: rename_terms(&row.c, column_map),
    }
}

fn poseidon2_template() -> Vec<SparseRow> {
    let mut builder = R1csBuilder::new();
    let inputs = std::array::from_fn(|lane| builder.alloc(F::from_usize(lane + 1)));
    let _ = enforce_poseidon2_permutation(&mut builder, &inputs);
    assert!(builder.is_satisfied());
    assert_eq!(builder.rows(), POSEIDON2_ROWS);
    assert_eq!(builder.cols(), 609);
    normalized_rows_range(&builder, 0..builder.rows())
}

fn assert_row(rows: &[SparseRow], phase_start: usize, global_row: usize, expected: SparseRow, label: &str) {
    assert_eq!(rows[global_row - phase_start], expected, "{label} at row {global_row}");
}

fn assert_poseidon2_call(
    rows: &[SparseRow],
    phase_start: usize,
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
        assert_row(
            rows,
            phase_start,
            call_rows.start + offset,
            expected,
            "Poseidon2 permutation",
        );
    }
}

fn exact_column_range(builder: &R1csBuilder, name: &str) -> Range<usize> {
    let matches = builder
        .column_family_ranges()
        .iter()
        .filter(|range| range.name == name)
        .collect::<Vec<_>>();
    let [range] = matches.as_slice() else {
        panic!("expected one column family {name}")
    };
    range.column_start..range.column_end
}

fn exact_phase_rows(builder: &R1csBuilder) -> Range<usize> {
    let matches = builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == STREAMING_CARRY_PHASE_ENVELOPE_FAMILY)
        .collect::<Vec<_>>();
    let [range] = matches.as_slice() else {
        panic!("expected one carry phase-envelope row family")
    };
    range.row_start..range.row_end
}

fn validate_hash(
    builder: &R1csBuilder,
    rows: &[SparseRow],
    phase_start: usize,
    hash: &Poseidon2HashAudit,
    constant_start: usize,
    local_columns: [usize; DIGEST_FIELDS],
    payload_columns: &[usize],
    output_columns: [usize; DIGEST_FIELDS],
    template: &[SparseRow],
) -> Vec<u64> {
    let expected_inputs = (constant_start..constant_start + HASH_CONSTANT_FIELDS)
        .chain(local_columns)
        .chain(payload_columns.iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(expected_inputs.len(), HASH_INPUT_FIELDS);
    assert_eq!(hash.input_cols, expected_inputs);
    assert_eq!(hash.zero_col, constant_start + HASH_CONSTANT_FIELDS);
    assert_eq!(hash.zero_row, hash.row_start);
    assert_eq!(hash.rounds.len(), HASH_ROUNDS);
    assert_eq!(hash.output_cols, output_columns);
    assert_eq!(hash.row_end - hash.row_start, HASH_TRACE_ROWS);

    let constant_row_start = hash.row_start - HASH_CONSTANT_FIELDS;
    let constant_values = (0..HASH_CONSTANT_FIELDS)
        .map(|offset| {
            let column = constant_start + offset;
            let value = builder.witness()[column].as_canonical_u64();
            assert_row(
                rows,
                phase_start,
                constant_row_start + offset,
                linear_row(column, &[(0, F::from_u64(value))]),
                "phase preimage constant",
            );
            value
        })
        .collect::<Vec<_>>();

    assert_row(
        rows,
        phase_start,
        hash.zero_row,
        linear_row(hash.zero_col, &[]),
        "hash zero",
    );

    let mut prior_outputs = [hash.zero_col; 8];
    for round_index in 0..ABSORB_ROUNDS {
        let round = &hash.rounds[round_index];
        let Poseidon2HashRoundAuditKind::Absorb { chunk_cols } = &round.kind else {
            panic!("data round must absorb")
        };
        let chunk = &hash.input_cols[4 * round_index..4 * round_index + 4];
        assert_eq!(chunk_cols, chunk);
        assert_eq!(round.state_before_cols, prior_outputs);

        let definition_row_start = hash.row_start + 1 + round_index * ABSORB_ROUND_ROWS;
        let absorb_column_start = hash.zero_col + 1 + round_index * ABSORB_ROUND_ROWS;
        let expected_inputs: [usize; 8] = std::array::from_fn(|lane| {
            if lane < 4 {
                absorb_column_start + lane
            } else {
                prior_outputs[lane]
            }
        });
        assert_eq!(round.permutation_input_cols, expected_inputs);
        assert_eq!(
            round.defining_rows,
            (definition_row_start..definition_row_start + 4).collect::<Vec<_>>(),
        );
        for lane in 0..4 {
            assert_row(
                rows,
                phase_start,
                definition_row_start + lane,
                linear_row(
                    expected_inputs[lane],
                    &[(prior_outputs[lane], F::ONE), (chunk[lane], F::ONE)],
                ),
                "hash absorb definition",
            );
        }
        let first_allocated = absorb_column_start + 4;
        let call_rows = definition_row_start + 4..definition_row_start + ABSORB_ROUND_ROWS;
        assert_poseidon2_call(rows, phase_start, call_rows, expected_inputs, first_allocated, template);
        let expected_outputs = std::array::from_fn(|lane| first_allocated + 592 + lane);
        assert_eq!(round.permutation_output_cols, expected_outputs);
        prior_outputs = expected_outputs;
    }

    let pad = &hash.rounds[ABSORB_ROUNDS];
    assert_eq!(pad.kind, Poseidon2HashRoundAuditKind::Pad);
    assert_eq!(pad.state_before_cols, prior_outputs);
    let pad_row = hash.row_start + 1 + ABSORB_ROUNDS * ABSORB_ROUND_ROWS;
    let pad_column = hash.zero_col + 1 + ABSORB_ROUNDS * ABSORB_ROUND_ROWS;
    let pad_inputs = std::array::from_fn(|lane| if lane == 0 { pad_column } else { prior_outputs[lane] });
    assert_eq!(pad.permutation_input_cols, pad_inputs);
    assert_eq!(pad.defining_rows, vec![pad_row]);
    assert_row(
        rows,
        phase_start,
        pad_row,
        linear_row(pad_column, &[(prior_outputs[0], F::ONE), (0, F::ONE)]),
        "hash padding definition",
    );
    let pad_first_allocated = pad_column + 1;
    assert_poseidon2_call(
        rows,
        phase_start,
        pad_row + 1..pad_row + 1 + POSEIDON2_ROWS,
        pad_inputs,
        pad_first_allocated,
        template,
    );
    let final_outputs = std::array::from_fn(|lane| pad_first_allocated + 592 + lane);
    assert_eq!(pad.permutation_output_cols, final_outputs);
    assert_eq!(hash.output_cols, final_outputs[..DIGEST_FIELDS]);
    assert_eq!(hash.row_end, pad_row + 1 + POSEIDON2_ROWS);
    constant_values
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

fn build_arm(kind: NebulaFPrimePiRlcFamilyReplayArmKind, template: &[SparseRow]) -> ArmArtifact {
    let synthesis = NebulaFPrimePiRlcFamilyBodySynthesis::production(kind);
    let builder = synthesis.builder_for_artifact();
    let phase_rows = exact_phase_rows(builder);
    assert_eq!(phase_rows.len(), PHASE_ROWS);
    let rows = normalized_rows_range(builder, phase_rows.clone());

    let before_alias_range = exact_column_range(builder, STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY);
    let before_payload_range = exact_column_range(builder, STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY);
    let after_alias_range = exact_column_range(builder, STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY);
    let after_payload_range = exact_column_range(builder, STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY);
    assert_eq!(before_alias_range.len(), DIGEST_FIELDS);
    assert_eq!(before_payload_range.len(), STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS);
    assert_eq!(after_alias_range.len(), DIGEST_FIELDS);
    assert_eq!(before_payload_range, after_payload_range);
    assert_eq!(before_alias_range.end, before_payload_range.start);
    assert_eq!(before_payload_range.end, after_alias_range.start);

    let phase_columns = before_alias_range.start..before_alias_range.start + PHASE_ROWS;
    let before_alias = std::array::from_fn(|offset| before_alias_range.start + offset);
    let after_alias = std::array::from_fn(|offset| after_alias_range.start + offset);
    let payload_columns = (before_payload_range.start..before_payload_range.end).collect::<Vec<_>>();
    assert_eq!(payload_columns, synthesis.phase_delayed_payload_columns());
    let before_source = synthesis.before_phase_local_state_source_columns();
    let after_source = synthesis.after_phase_local_state_source_columns();
    assert_eq!(before_alias, synthesis.before_phase_local_state_columns());
    assert_eq!(after_alias, synthesis.after_phase_local_state_columns());

    for lane in 0..DIGEST_FIELDS {
        assert_row(
            &rows,
            phase_rows.start,
            phase_rows.start + lane,
            linear_row(before_alias[lane], &[(before_source[lane], F::ONE)]),
            "before local alias",
        );
    }
    for (offset, &column) in payload_columns.iter().enumerate() {
        assert_row(
            &rows,
            phase_rows.start,
            phase_rows.start + DIGEST_FIELDS + offset,
            bit_row(column),
            "delayed payload bit",
        );
    }
    let after_alias_row_start = phase_rows.start + DIGEST_FIELDS + STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS;
    for lane in 0..DIGEST_FIELDS {
        assert_row(
            &rows,
            phase_rows.start,
            after_alias_row_start + lane,
            linear_row(after_alias[lane], &[(after_source[lane], F::ONE)]),
            "after local alias",
        );
    }

    let before_output = synthesis.before_phase_semantic_digest_columns();
    let after_output = synthesis.after_phase_semantic_digest_columns();
    let hash_audits = builder.poseidon2_hash_audits();
    let before_hash = hash_audits
        .iter()
        .find(|audit| audit.output_cols == before_output)
        .expect("before phase hash audit");
    let after_hash = hash_audits
        .iter()
        .find(|audit| audit.output_cols == after_output)
        .expect("after phase hash audit");
    let before_constant_start = phase_columns.start + ALIAS_AND_PAYLOAD_ROWS;
    let after_constant_start = before_constant_start + HASH_TOTAL_ROWS;
    assert_eq!(
        before_hash.row_start,
        phase_rows.start + ALIAS_AND_PAYLOAD_ROWS + HASH_CONSTANT_FIELDS
    );
    assert_eq!(after_hash.row_start, before_hash.row_start + HASH_TOTAL_ROWS);
    let before_constants = validate_hash(
        builder,
        &rows,
        phase_rows.start,
        before_hash,
        before_constant_start,
        before_alias,
        &payload_columns,
        before_output,
        template,
    );
    let after_constants = validate_hash(
        builder,
        &rows,
        phase_rows.start,
        after_hash,
        after_constant_start,
        after_alias,
        &payload_columns,
        after_output,
        template,
    );
    assert_eq!(before_constants, after_constants);
    assert_eq!(before_constants.len(), HASH_CONSTANT_FIELDS);
    assert_eq!(phase_rows.end, after_hash.row_end);
    assert_eq!(phase_columns.end, after_constant_start + HASH_TOTAL_ROWS);

    let before_x_out_semantic = synthesis.before_x_out_preimage_columns()
        [X_OUT_SEMANTIC_START..X_OUT_SEMANTIC_START + DIGEST_FIELDS]
        .try_into()
        .expect("four before XOut semantic columns");
    let after_x_out_semantic = synthesis.after_x_out_preimage_columns()
        [X_OUT_SEMANTIC_START..X_OUT_SEMANTIC_START + DIGEST_FIELDS]
        .try_into()
        .expect("four after XOut semantic columns");
    assert_eq!(before_x_out_semantic, before_output);
    assert_eq!(after_x_out_semantic, after_output);

    let (source_identity, body_rows, body_columns) = match kind {
        NebulaFPrimePiRlcFamilyReplayArmKind::Even => {
            ("rust:pi-rlc-family-even/body-v3", synthesis.rows(), synthesis.columns())
        }
        NebulaFPrimePiRlcFamilyReplayArmKind::Odd => {
            ("rust:pi-rlc-family-odd/body-v3", synthesis.rows(), synthesis.columns())
        }
    };
    ArmArtifact {
        source_identity,
        source_rows_sha256: source_rows_sha256(&rows),
        body_rows,
        body_columns,
        phase_rows,
        phase_columns,
        before_local_source_columns: before_source,
        after_local_source_columns: after_source,
        before_local_alias_columns: before_alias,
        after_local_alias_columns: after_alias,
        payload_start_column: before_payload_range.start,
        before_hash_constant_start_column: before_constant_start,
        after_hash_constant_start_column: after_constant_start,
        before_semantic_digest_columns: before_output,
        after_semantic_digest_columns: after_output,
        before_x_out_semantic_columns: before_x_out_semantic,
        after_x_out_semantic_columns: after_x_out_semantic,
        constant_values: before_constants,
    }
}

fn render_arm(arm: &ArmArtifact) -> String {
    format!(
        "{{ sourceIdentity := \"{}\", sourceRowsSha256 := \"{}\",\n    \
         bodyRows := {}, bodyColumns := {},\n    \
         phaseRowStart := {}, phaseRowEnd := {}, phaseColumnStart := {}, phaseColumnEnd := {},\n    \
         beforeLocalSourceColumns := {}, afterLocalSourceColumns := {},\n    \
         beforeLocalAliasColumns := {}, afterLocalAliasColumns := {},\n    \
         payloadStartColumn := {}, beforeHashConstantStartColumn := {},\n    \
         afterHashConstantStartColumn := {},\n    \
         beforeSemanticDigestColumns := {}, afterSemanticDigestColumns := {},\n    \
         beforeXOutSemanticColumns := {}, afterXOutSemanticColumns := {} }}",
        arm.source_identity,
        arm.source_rows_sha256,
        arm.body_rows,
        arm.body_columns,
        arm.phase_rows.start,
        arm.phase_rows.end,
        arm.phase_columns.start,
        arm.phase_columns.end,
        lean_nat_list(arm.before_local_source_columns),
        lean_nat_list(arm.after_local_source_columns),
        lean_nat_list(arm.before_local_alias_columns),
        lean_nat_list(arm.after_local_alias_columns),
        arm.payload_start_column,
        arm.before_hash_constant_start_column,
        arm.after_hash_constant_start_column,
        lean_nat_list(arm.before_semantic_digest_columns),
        lean_nat_list(arm.after_semantic_digest_columns),
        lean_nat_list(arm.before_x_out_semantic_columns),
        lean_nat_list(arm.after_x_out_semantic_columns),
    )
}

fn render_artifact() -> String {
    assert_eq!(STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS, 2_169);
    assert_eq!(HASH_INPUT_FIELDS, 2_184);
    assert_eq!(ABSORB_ROUNDS, 546);
    assert_eq!(HASH_TRACE_ROWS, 330_386);
    assert_eq!(HASH_TOTAL_ROWS, 330_397);
    assert_eq!(PHASE_ROWS, 662_971);
    let template = poseidon2_template();
    let even = build_arm(NebulaFPrimePiRlcFamilyReplayArmKind::Even, &template);
    let odd = build_arm(NebulaFPrimePiRlcFamilyReplayArmKind::Odd, &template);
    assert_eq!(even.constant_values, odd.constant_values);

    let mut payload = String::new();
    writeln!(
        payload,
        "def phaseConstantValues : List Nat := {}",
        lean_nat_list(even.constant_values.iter().map(|&value| value as usize)),
    )
    .unwrap();
    writeln!(payload, "\ndef evenArm : RawArm :=\n  {}", render_arm(&even)).unwrap();
    writeln!(payload, "\ndef oddArm : RawArm :=\n  {}", render_arm(&odd)).unwrap();
    writeln!(
        payload,
        "\ndef rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := {SCHEMA_VERSION}, profileId := \"{PROFILE_ID}\",\n    \
            constantValues := phaseConstantValues, even := evenArm, odd := oddArm }}",
    )
    .unwrap();
    let artifact_hash = sha256_hex(&payload);
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelopeSchema\n\n\
         /-! Generated file: compact exact geometry for both Rust-emitted PiRLC\n\
         carry-phase semantic envelopes. The Rust generator exhaustively checks\n\
         every represented row before it emits this data.\n\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact\n\n\
         def artifactSha256 : String := \"{artifact_hash}\"\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope\n",
    )
}

fn generated_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ARTIFACT_PATH)
}

#[test]
fn production_pi_rlc_phase_envelope_artifact_is_current() {
    let path = generated_artifact_path();
    let rendered = render_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected PiRLC phase-envelope artifact");
        panic!(
            "PiRLC phase-envelope Lean artifact drifted; inspect {}",
            expected.display()
        );
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_production_pi_rlc_phase_envelope_artifact() {
    std::fs::write(generated_artifact_path(), render_artifact())
        .expect("write generated PiRLC phase-envelope artifact");
}
