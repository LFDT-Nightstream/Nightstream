//! Exact artifact checks for the public-binding suffix of one PiRLC family.

#[path = "../gadgets/lean_artifact_support.rs"]
#[allow(dead_code)]
mod lean_artifact_support;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Range;
use std::path::{Path, PathBuf};

use lean_artifact_support::{lean_nat_list, sha256_hex};
use neo_fold_clean::engine::r1cs_circuit::builder::{Poseidon2HashAudit, Poseidon2HashRoundAuditKind};
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::{enforce_poseidon2_permutation, R1csBuilder, Var};
use neo_fold_clean::frontends::nebula::f_prime::{
    production_pi_rlc_family_body_compiler_audit, NebulaFPrimePiRlcFamilyBodySynthesis,
    NebulaFPrimePiRlcFamilyReplayArmKind, NebulaFPrimeStreamingProgramAudit, PI_RLC_FAMILY_BODY_EVEN_COLUMNS,
    PI_RLC_FAMILY_BODY_EVEN_ROWS, PI_RLC_FAMILY_BODY_EVEN_SOURCE_ROWS, PI_RLC_FAMILY_BODY_ODD_COLUMNS,
    PI_RLC_FAMILY_BODY_ODD_ROWS, PI_RLC_FAMILY_BODY_ODD_SOURCE_ROWS,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const SCHEMA_VERSION: usize = 4;
const PROFILE_ID: &str = "nebula-f-prime-streaming-pi-rlc-family-public-v4";
const FAMILY_STATE_FIELDS: usize = 1_045;
const SHARED_PUBLIC_WORDS: usize = 10;
const PUBLIC_BITS_PER_WORD: usize = 64;
const SUFFIX_POSEIDON2_CALLS: usize = 544;
const PHASE_ENVELOPE_POSEIDON2_CALLS: usize = 1_094;
const PHASE_ENVELOPE_ROWS: usize = 662_971;
const DIGEST_PIN_COUNT: usize = 13;
const POSEIDON2_ROWS: usize = 600;
const LOW_NORM_ROWS: usize = 491_046;
const LOW_NORM_COLUMNS: usize = 8_858_862;
const LOW_NORM_PUBLIC_COLUMNS: usize = 648;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct SparseRow {
    a: Vec<(usize, F)>,
    b: Vec<(usize, F)>,
    c: Vec<(usize, F)>,
}

#[derive(Clone, Debug)]
struct CanonicalCall {
    rows: Range<usize>,
    field: usize,
    bits: [usize; PUBLIC_BITS_PER_WORD],
    high_flag: usize,
    inverse: usize,
}

#[derive(Clone, Debug)]
struct Poseidon2Call {
    rows: Range<usize>,
    inputs: [usize; 8],
    first_allocated: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OwnerKind {
    Canonical,
    Poseidon2,
    Glue,
    PhaseEnvelope,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Owner {
    rows: Range<usize>,
    kind: OwnerKind,
    index: usize,
}

#[derive(Clone, Debug)]
struct GlueRow {
    index: usize,
    row: SparseRow,
}

struct ArmArtifact {
    source_row_count: usize,
    row_count: usize,
    column_count: usize,
    public_column_count: usize,
    replay_poseidon2_call_count: usize,
    before_family_cursor_column: usize,
    after_family_cursor_column: usize,
    before_state_columns: Vec<usize>,
    after_state_columns: Vec<usize>,
    after_x_out_preimage_columns: Vec<usize>,
    before_x_out_preimage_columns: Vec<usize>,
    after_x_out_digest_columns: Vec<usize>,
    before_x_out_digest_columns: Vec<usize>,
    after_x_out_hash: Poseidon2HashAudit,
    before_x_out_hash: Poseidon2HashAudit,
    public_word_call_indices: Vec<usize>,
    after_digest_pin_columns: Vec<usize>,
    before_digest_pin_columns: Vec<usize>,
    phase_envelope_rows: Range<usize>,
    canonical_calls: Vec<CanonicalCall>,
    poseidon2_calls: Vec<Poseidon2Call>,
    glue_rows: Vec<GlueRow>,
    owners: Vec<Owner>,
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
    assert!(row_range.start <= row_range.end && row_range.end <= total_rows);
    let mut raw = vec![Vec::new(); row_range.len()];
    for &(row, column, coefficient) in trips {
        assert!(row < total_rows, "sparse triplet row is in range");
        if row_range.contains(&row) {
            raw[row - row_range.start].push((column, coefficient));
        }
    }
    raw.into_iter().map(normalize_terms).collect()
}

fn normalized_rows_range(builder: &R1csBuilder, row_range: Range<usize>) -> Vec<SparseRow> {
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

fn canonical_template() -> Vec<SparseRow> {
    let mut builder = R1csBuilder::new();
    let field = builder.alloc(F::from_u64(37));
    let bits = decompose_var_to_u64_bits(&mut builder, field);
    assert_eq!(field.col(), 1);
    assert_eq!(bits.map(Var::col), std::array::from_fn(|index| index + 2));
    assert_eq!(builder.rows(), 69);
    assert_eq!(builder.cols(), 68);
    normalized_rows_range(&builder, 0..builder.rows())
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

fn assert_renamed_slice(
    actual: &[SparseRow],
    actual_start: usize,
    rows: Range<usize>,
    template: &[SparseRow],
    column_map: impl Fn(usize) -> usize,
    label: &str,
) {
    assert_eq!(rows.len(), template.len(), "{label} row count");
    assert!(actual_start <= rows.start && rows.end <= actual_start + actual.len());
    for (offset, expected) in template.iter().enumerate() {
        let expected = rename_row(expected, &column_map);
        assert_eq!(
            actual[rows.start - actual_start + offset],
            expected,
            "{label} row {offset}",
        );
    }
}

fn mark_owner(
    slots: &mut [Option<(OwnerKind, usize)>],
    suffix_start: usize,
    rows: Range<usize>,
    kind: OwnerKind,
    index: usize,
) {
    assert!(suffix_start <= rows.start && rows.end <= suffix_start + slots.len());
    assert!(rows.start < rows.end);
    for slot in &mut slots[rows.start - suffix_start..rows.end - suffix_start] {
        assert!(slot.replace((kind, index)).is_none(), "artifact leaf rows overlap");
    }
}

fn body_shape(kind: NebulaFPrimePiRlcFamilyReplayArmKind) -> (usize, usize, usize) {
    match kind {
        NebulaFPrimePiRlcFamilyReplayArmKind::Even => (
            PI_RLC_FAMILY_BODY_EVEN_SOURCE_ROWS,
            PI_RLC_FAMILY_BODY_EVEN_ROWS,
            PI_RLC_FAMILY_BODY_EVEN_COLUMNS,
        ),
        NebulaFPrimePiRlcFamilyReplayArmKind::Odd => (
            PI_RLC_FAMILY_BODY_ODD_SOURCE_ROWS,
            PI_RLC_FAMILY_BODY_ODD_ROWS,
            PI_RLC_FAMILY_BODY_ODD_COLUMNS,
        ),
    }
}

fn poseidon2_call_outputs(call: &Poseidon2Call) -> [usize; 8] {
    std::array::from_fn(|lane| call.first_allocated + 592 + lane)
}

fn exact_x_out_hash(
    audits: &[Poseidon2HashAudit],
    input_columns: &[usize],
    output_columns: &[usize],
    poseidon2_calls: &[Poseidon2Call],
) -> (Poseidon2HashAudit, usize) {
    let matches = audits
        .iter()
        .filter(|audit| audit.input_cols == input_columns && audit.output_cols.as_slice() == output_columns)
        .cloned()
        .collect::<Vec<_>>();
    let [audit] = matches.as_slice() else {
        panic!("one exact XOut hash audit")
    };
    assert_eq!(audit.input_cols.len(), 32);
    assert_eq!(audit.rounds.len(), 9);
    assert_eq!(audit.row_start, audit.zero_row);
    let call_start = poseidon2_calls
        .iter()
        .position(|call| {
            call.inputs == audit.rounds[0].permutation_input_cols
                && poseidon2_call_outputs(call) == audit.rounds[0].permutation_output_cols
        })
        .expect("XOut first permutation call");
    let mut previous = [audit.zero_col; 8];
    for (round_index, round) in audit.rounds.iter().enumerate() {
        assert_eq!(round.state_before_cols, previous);
        match &round.kind {
            Poseidon2HashRoundAuditKind::Absorb { chunk_cols } => {
                assert!(round_index < 8);
                assert_eq!(chunk_cols, &input_columns[round_index * 4..(round_index + 1) * 4]);
                assert_eq!(round.defining_rows.len(), 4);
            }
            Poseidon2HashRoundAuditKind::Pad => {
                assert_eq!(round_index, 8);
                assert_eq!(round.defining_rows.len(), 1);
            }
        }
        let call = &poseidon2_calls[call_start + round_index];
        assert_eq!(call.inputs, round.permutation_input_cols);
        assert_eq!(poseidon2_call_outputs(call), round.permutation_output_cols);
        previous = round.permutation_output_cols;
    }
    assert_eq!(&previous[..4], output_columns);
    assert_eq!(audit.row_end, poseidon2_calls[call_start + 8].rows.end);
    (audit.clone(), call_start)
}

fn build_arm(
    kind: NebulaFPrimePiRlcFamilyReplayArmKind,
    canonical_template: &[SparseRow],
    poseidon2_template: &[SparseRow],
) -> ArmArtifact {
    let synthesis = NebulaFPrimePiRlcFamilyBodySynthesis::production(kind);
    let builder = synthesis.builder_for_artifact();
    let (source_row_count, row_count, column_count) = body_shape(kind);
    assert_eq!(builder.rows(), row_count);
    assert_eq!(builder.cols(), column_count);
    assert_eq!(synthesis.public_columns(), 641);
    assert!(synthesis.is_satisfied());

    let rows = normalized_rows_range(builder, source_row_count..row_count);
    let mut owner_slots = vec![None; row_count - source_row_count];
    let phase_stages = builder
        .row_family_ranges()
        .iter()
        .filter(|stage| stage.name == "fprime.streaming.phase.carry.semantic_envelope")
        .collect::<Vec<_>>();
    let [phase_stage] = phase_stages.as_slice() else {
        panic!("one exact PiRLC phase-envelope stage")
    };
    let phase_envelope_rows = phase_stage.row_start..phase_stage.row_end;
    assert_eq!(phase_envelope_rows.len(), PHASE_ENVELOPE_ROWS);
    mark_owner(
        &mut owner_slots,
        source_row_count,
        phase_envelope_rows.clone(),
        OwnerKind::PhaseEnvelope,
        0,
    );

    let canonical_calls = builder
        .encoding_trace()
        .canonical_u64_decompositions()
        .iter()
        .enumerate()
        .map(|(index, trace)| {
            let bits = trace.bits.map(Var::col);
            assert_eq!(bits, std::array::from_fn(|offset| bits[0] + offset));
            let call = CanonicalCall {
                rows: trace.source_rows.clone(),
                field: trace.field.col(),
                bits,
                high_flag: trace.high_is_max.col(),
                inverse: trace.inverse.col(),
            };
            assert_eq!(call.high_flag, call.bits[0] + 64);
            assert_eq!(call.inverse, call.bits[0] + 65);
            assert_renamed_slice(
                &rows,
                source_row_count,
                call.rows.clone(),
                canonical_template,
                |column| match column {
                    0 => 0,
                    1 => call.field,
                    2..=65 => call.bits[column - 2],
                    66 => call.high_flag,
                    67 => call.inverse,
                    _ => panic!("canonical template column out of range"),
                },
                "canonical-u64",
            );
            mark_owner(
                &mut owner_slots,
                source_row_count,
                call.rows.clone(),
                OwnerKind::Canonical,
                index,
            );
            call
        })
        .collect::<Vec<_>>();
    assert_eq!(canonical_calls.len(), 11);

    let all_poseidon2_calls = builder.poseidon2_permutation_audits();
    let replay_poseidon2_call_count = kind.poseidon2_calls();
    assert_eq!(
        all_poseidon2_calls.len(),
        replay_poseidon2_call_count + PHASE_ENVELOPE_POSEIDON2_CALLS + SUFFIX_POSEIDON2_CALLS,
    );
    let poseidon2_calls = all_poseidon2_calls
        .into_iter()
        .skip(replay_poseidon2_call_count)
        .filter(|trace| {
            let rows = trace.row_start..trace.row_end;
            assert!(
                rows.end <= phase_envelope_rows.start
                    || phase_envelope_rows.end <= rows.start
                    || (phase_envelope_rows.start <= rows.start && rows.end <= phase_envelope_rows.end),
                "a Poseidon2 call cannot cross the phase-envelope stage boundary",
            );
            !(phase_envelope_rows.start <= rows.start && rows.end <= phase_envelope_rows.end)
        })
        .enumerate()
        .map(|(index, trace)| {
            assert_eq!(trace.allocated_col_count, POSEIDON2_ROWS);
            assert_eq!(
                trace.output_cols,
                std::array::from_fn(|lane| trace.first_allocated_col + 592 + lane),
            );
            let call = Poseidon2Call {
                rows: trace.row_start..trace.row_end,
                inputs: trace.input_cols,
                first_allocated: trace.first_allocated_col,
            };
            assert_renamed_slice(
                &rows,
                source_row_count,
                call.rows.clone(),
                poseidon2_template,
                |column| match column {
                    0 => 0,
                    1..=8 => call.inputs[column - 1],
                    _ => call.first_allocated + column - 9,
                },
                "Poseidon2",
            );
            mark_owner(
                &mut owner_slots,
                source_row_count,
                call.rows.clone(),
                OwnerKind::Poseidon2,
                index,
            );
            call
        })
        .collect::<Vec<_>>();
    assert_eq!(poseidon2_calls.len(), SUFFIX_POSEIDON2_CALLS);

    let after_x_out_preimage_columns = synthesis.after_x_out_preimage_columns().to_vec();
    let before_x_out_preimage_columns = synthesis.before_x_out_preimage_columns().to_vec();
    let after_x_out_digest_columns = synthesis.after_x_out_digest_columns().to_vec();
    let before_x_out_digest_columns = synthesis.before_x_out_digest_columns().to_vec();
    let hash_audits = builder.poseidon2_hash_audits();
    let (after_x_out_hash, after_call_start) = exact_x_out_hash(
        &hash_audits,
        &after_x_out_preimage_columns,
        &after_x_out_digest_columns,
        &poseidon2_calls,
    );
    let (before_x_out_hash, before_call_start) = exact_x_out_hash(
        &hash_audits,
        &before_x_out_preimage_columns,
        &before_x_out_digest_columns,
        &poseidon2_calls,
    );
    assert_eq!(after_call_start, SUFFIX_POSEIDON2_CALLS - 18);
    assert_eq!(before_call_start, SUFFIX_POSEIDON2_CALLS - 9);

    let public_output_columns = (0..synthesis.public_columns() - 1)
        .map(|index| {
            synthesis
                .public_output_column(index)
                .expect("public output column")
        })
        .collect::<Vec<_>>();
    assert_eq!(public_output_columns.len(), SHARED_PUBLIC_WORDS * PUBLIC_BITS_PER_WORD);
    let public_word_call_indices = public_output_columns
        .chunks_exact(PUBLIC_BITS_PER_WORD)
        .map(|word| {
            canonical_calls
                .iter()
                .position(|call| call.bits.as_slice() == word)
                .expect("every public word uses one canonical decomposition")
        })
        .collect::<Vec<_>>();
    assert_eq!(public_word_call_indices, vec![3, 4, 5, 6, 7, 8, 9, 10, 0, 1]);

    let before_state_columns = synthesis.before_state_field_columns().to_vec();
    let after_state_columns = synthesis.after_state_field_columns().to_vec();
    assert_eq!(before_state_columns.len(), FAMILY_STATE_FIELDS);
    assert_eq!(after_state_columns.len(), FAMILY_STATE_FIELDS);
    assert_eq!(
        before_state_columns.last().copied(),
        Some(synthesis.before_family_cursor_column()),
    );
    assert_eq!(
        after_state_columns.last().copied(),
        Some(synthesis.after_family_cursor_column()),
    );

    let mut glue_rows = Vec::new();
    for row in source_row_count..row_count {
        if owner_slots[row - source_row_count].is_none() {
            let index = glue_rows.len();
            glue_rows.push(GlueRow {
                index: row,
                row: rows[row - source_row_count].clone(),
            });
            owner_slots[row - source_row_count] = Some((OwnerKind::Glue, index));
        }
    }

    let mut owners = Vec::new();
    let mut cursor = source_row_count;
    while cursor < row_count {
        let (kind, index) = owner_slots[cursor - source_row_count].expect("every suffix row has one owner");
        let owned_rows = match kind {
            OwnerKind::Canonical => canonical_calls[index].rows.clone(),
            OwnerKind::Poseidon2 => poseidon2_calls[index].rows.clone(),
            OwnerKind::Glue => cursor..cursor + 1,
            OwnerKind::PhaseEnvelope => phase_envelope_rows.clone(),
        };
        assert_eq!(owned_rows.start, cursor, "owner range starts at the row cursor");
        assert!(
            owner_slots[owned_rows.start - source_row_count..owned_rows.end - source_row_count]
                .iter()
                .all(|slot| *slot == Some((kind, index))),
            "one owner covers its complete range",
        );
        owners.push(Owner {
            rows: owned_rows.clone(),
            kind,
            index,
        });
        cursor = owned_rows.end;
    }

    ArmArtifact {
        source_row_count,
        row_count,
        column_count,
        public_column_count: synthesis.public_columns(),
        replay_poseidon2_call_count,
        before_family_cursor_column: synthesis.before_family_cursor_column(),
        after_family_cursor_column: synthesis.after_family_cursor_column(),
        before_state_columns,
        after_state_columns,
        after_x_out_preimage_columns,
        before_x_out_preimage_columns,
        after_x_out_digest_columns,
        before_x_out_digest_columns,
        after_x_out_hash,
        before_x_out_hash,
        public_word_call_indices,
        after_digest_pin_columns: synthesis.after_digest_pin_columns().to_vec(),
        before_digest_pin_columns: synthesis.before_digest_pin_columns().to_vec(),
        phase_envelope_rows,
        canonical_calls,
        poseidon2_calls,
        glue_rows,
        owners,
    }
}

fn lean_terms(terms: &[(usize, F)]) -> String {
    let values = terms
        .iter()
        .map(|(column, coefficient)| format!("({column}, {})", coefficient.as_canonical_u64()))
        .collect::<Vec<_>>();
    format!("[{}]", values.join(", "))
}

fn lean_row(row: &SparseRow) -> String {
    format!(
        "⟨{}, {}, {}⟩",
        lean_terms(&row.a),
        lean_terms(&row.b),
        lean_terms(&row.c),
    )
}

fn grouped_list(items: Vec<String>, per_line: usize) -> String {
    if items.is_empty() {
        return "[]".to_string();
    }
    let lines = items
        .chunks(per_line)
        .map(|chunk| format!("    {}", chunk.join(", ")))
        .collect::<Vec<_>>();
    format!("[\n{}\n  ]", lines.join(",\n"))
}

fn render_hash(audit: &Poseidon2HashAudit, permutation_call_start: usize) -> String {
    let rounds = grouped_list(
        audit
            .rounds
            .iter()
            .map(|round| {
                let (kind, chunk_columns) = match &round.kind {
                    Poseidon2HashRoundAuditKind::Absorb { chunk_cols } => (".absorb", chunk_cols.as_slice()),
                    Poseidon2HashRoundAuditKind::Pad => (".pad", &[][..]),
                };
                format!(
                    "{{ kind := {kind}, chunkColumns := {}, stateBeforeColumns := {}, permutationInputColumns := {}, definingRows := {}, permutationOutputColumns := {} }}",
                    lean_nat_list(chunk_columns.iter().copied()),
                    lean_nat_list(round.state_before_cols),
                    lean_nat_list(round.permutation_input_cols),
                    lean_nat_list(round.defining_rows.iter().copied()),
                    lean_nat_list(round.permutation_output_cols),
                )
            })
            .collect(),
        1,
    );
    format!(
        "{{ rowStart := {}, rowEnd := {}, inputColumns := {}, zeroColumn := {}, zeroRow := {}, permutationCallStart := {permutation_call_start}, outputColumns := {}, rounds := {rounds} }}",
        audit.row_start,
        audit.row_end,
        lean_nat_list(audit.input_cols.iter().copied()),
        audit.zero_col,
        audit.zero_row,
        lean_nat_list(audit.output_cols),
    )
}

fn render_arm(arm: &ArmArtifact) -> String {
    let canonical_calls = grouped_list(
        arm.canonical_calls
            .iter()
            .map(|call| {
                format!(
                    "{{ rowStart := {}, rowEnd := {}, fieldColumn := {}, bitBase := {}, \
                     highFlagColumn := {}, inverseColumn := {} }}",
                    call.rows.start, call.rows.end, call.field, call.bits[0], call.high_flag, call.inverse,
                )
            })
            .collect(),
        2,
    );
    let poseidon2_calls = grouped_list(
        arm.poseidon2_calls
            .iter()
            .map(|call| {
                format!(
                    "{{ rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }}",
                    call.rows.start,
                    call.rows.end,
                    lean_nat_list(call.inputs),
                    call.first_allocated,
                )
            })
            .collect(),
        2,
    );
    let glue_rows = grouped_list(
        arm.glue_rows
            .iter()
            .map(|glue| format!("{{ index := {}, row := {} }}", glue.index, lean_row(&glue.row)))
            .collect(),
        2,
    );
    let owners = grouped_list(
        arm.owners
            .iter()
            .map(|owner| {
                let kind = match owner.kind {
                    OwnerKind::Canonical => ".canonical",
                    OwnerKind::Poseidon2 => ".poseidon2",
                    OwnerKind::Glue => ".glue",
                    OwnerKind::PhaseEnvelope => ".phaseEnvelope",
                };
                format!(
                    "{{ rowStart := {}, rowEnd := {}, kind := {kind}, index := {} }}",
                    owner.rows.start, owner.rows.end, owner.index,
                )
            })
            .collect(),
        4,
    );
    format!(
        "{{ sourceRowCount := {}, rowCount := {}, columnCount := {}, publicColumnCount := {},\n    \
         replayPoseidon2CallCount := {}, publicPoseidon2CallCount := {},\n    \
         phaseEnvelopeRowStart := {}, phaseEnvelopeRowEnd := {},\n    \
         beforeFamilyCursorColumn := {}, afterFamilyCursorColumn := {},\n    \
         beforeStateColumns := {},\n    \
         afterStateColumns := {},\n    \
         afterXOutPreimageColumns := {},\n    \
         beforeXOutPreimageColumns := {},\n    \
         afterXOutDigestColumns := {}, beforeXOutDigestColumns := {},\n    \
         afterXOutHash := {},\n    \
         beforeXOutHash := {},\n    \
         publicWordCallIndices := {},\n    \
         afterDigestPinColumns := {}, beforeDigestPinColumns := {},\n    \
         canonicalCalls := {canonical_calls},\n    \
         poseidon2Calls := {poseidon2_calls},\n    \
         glueRows := {glue_rows},\n    \
         owners := {owners} }}",
        arm.source_row_count,
        arm.row_count,
        arm.column_count,
        arm.public_column_count,
        arm.replay_poseidon2_call_count,
        SUFFIX_POSEIDON2_CALLS,
        arm.phase_envelope_rows.start,
        arm.phase_envelope_rows.end,
        arm.before_family_cursor_column,
        arm.after_family_cursor_column,
        lean_nat_list(arm.before_state_columns.iter().copied()),
        lean_nat_list(arm.after_state_columns.iter().copied()),
        lean_nat_list(arm.after_x_out_preimage_columns.iter().copied()),
        lean_nat_list(arm.before_x_out_preimage_columns.iter().copied()),
        lean_nat_list(arm.after_x_out_digest_columns.iter().copied()),
        lean_nat_list(arm.before_x_out_digest_columns.iter().copied()),
        render_hash(&arm.after_x_out_hash, SUFFIX_POSEIDON2_CALLS - 18),
        render_hash(&arm.before_x_out_hash, SUFFIX_POSEIDON2_CALLS - 9),
        lean_nat_list(arm.public_word_call_indices.iter().copied()),
        lean_nat_list(arm.after_digest_pin_columns.iter().copied()),
        lean_nat_list(arm.before_digest_pin_columns.iter().copied()),
    )
}

fn render_artifact() -> String {
    let canonical = canonical_template();
    let poseidon2 = poseidon2_template();
    let even = build_arm(NebulaFPrimePiRlcFamilyReplayArmKind::Even, &canonical, &poseidon2);
    let odd = build_arm(NebulaFPrimePiRlcFamilyReplayArmKind::Odd, &canonical, &poseidon2);
    assert_eq!(even.after_digest_pin_columns.len(), DIGEST_PIN_COUNT);
    assert_eq!(even.before_digest_pin_columns.len(), DIGEST_PIN_COUNT);
    assert_eq!(odd.after_digest_pin_columns.len(), DIGEST_PIN_COUNT);
    assert_eq!(odd.before_digest_pin_columns.len(), DIGEST_PIN_COUNT);
    for arm in [&even, &odd] {
        assert_eq!(arm.after_x_out_preimage_columns.len(), 32);
        assert_eq!(arm.before_x_out_preimage_columns.len(), 32);
        assert_eq!(arm.after_x_out_digest_columns.len(), 4);
        assert_eq!(arm.before_x_out_digest_columns.len(), 4);
    }

    let compiler = production_pi_rlc_family_body_compiler_audit().expect("production PiRLC family compiler audit");
    let public_layout = compiler.layout();
    assert_eq!(public_layout.logical_public_input_len(), even.public_column_count);
    assert_eq!(public_layout.logical_public_input_len(), odd.public_column_count);
    assert_eq!(public_layout.public_input_len(), LOW_NORM_PUBLIC_COLUMNS);
    assert_eq!(
        public_layout.public_padding_columns(),
        (public_layout.logical_public_input_len()..public_layout.public_input_len()).collect::<Vec<_>>(),
    );

    let first_family_program_cursor =
        NebulaFPrimeStreamingProgramAudit::production().first_pi_rlc_family_program_cursor();
    let mut payload = String::new();
    writeln!(payload, "def evenArm : RawArm :=\n  {}", render_arm(&even)).unwrap();
    writeln!(payload, "\ndef oddArm : RawArm :=\n  {}", render_arm(&odd)).unwrap();
    writeln!(
        payload,
        "\ndef rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := {SCHEMA_VERSION}, profileId := \"{PROFILE_ID}\",\n    \
            familyStateFields := {FAMILY_STATE_FIELDS}, sharedPublicWords := {SHARED_PUBLIC_WORDS},\n    \
            publicBitsPerWord := {PUBLIC_BITS_PER_WORD},\n    \
            firstFamilyProgramCursor := {first_family_program_cursor},\n    \
            lowNormRows := {LOW_NORM_ROWS}, lowNormColumns := {LOW_NORM_COLUMNS},\n    \
            lowNormPublicColumns := {LOW_NORM_PUBLIC_COLUMNS},\n    \
            publicDecoder := {{ constantOneColumn := 0, sourceFieldStart := 1, sourceFieldEnd := {}, paddingStart := {}, paddingEnd := {} }},\n    \
            even := evenArm, odd := oddArm }}",
        public_layout.logical_public_input_len(),
        public_layout.logical_public_input_len(),
        public_layout.public_input_len(),
    )
    .unwrap();

    let hash = sha256_hex(&payload);
    let rendered = format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicSchema\n\n\
         /-! Generated file: exact Rust metadata for the full-state public suffix\n\
         of both production PiRLC family body shapes.\n\n\
         Owns: local state columns, complete x_out preimages and outputs,\n\
         derived cursor words, canonical-u64 calls, Poseidon2 calls, glue\n\
         rows, and exact suffix row ownership.\n\n\
         Does not own: the PiRLC source prefix, lifecycle composition, or\n\
         Poseidon2 collision resistance.\n\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         set_option maxRecDepth 524288\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact\n\n\
         def artifactSha256 : String := \"{hash}\"\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic\n",
    );
    assert!(
        rendered.lines().count() < 1_500,
        "generated Lean artifact must stay below 1,500 lines",
    );
    rendered
}

fn generated_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyPublic.lean",
    )
}

#[test]
fn production_pi_rlc_family_public_artifact_is_current() {
    let path = generated_artifact_path();
    let rendered = render_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!("PiRLC family public Lean artifact drifted; inspect {}", path.display());
    }
}
