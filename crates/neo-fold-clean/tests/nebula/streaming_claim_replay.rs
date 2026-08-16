//! Constraint and size checks for the bounded-width claim-replay arms.

#[path = "../gadgets/lean_artifact_support.rs"]
#[allow(dead_code)]
mod lean_artifact_support;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Range;
use std::path::Path;

use lean_artifact_support::{lean_nat_list, sha256_hex};
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::{enforce_poseidon2_permutation, R1csBuilder, Var};
use neo_fold_clean::frontends::nebula::f_prime::{
    build_production_claim_coordinate_overlay_low_norm_r1cs, build_production_claim_replay_base_low_norm_r1cs,
    claim_replay_shape_audit_for_chunk_fields, production_claim_coordinate_overlay_kind_map,
    production_claim_coordinate_overlay_links, production_claim_coordinate_overlay_shape_audit,
    production_claim_replay_base_shape_audit, production_claim_replay_shape_audit,
    production_claim_running_metadata_field_map, production_claim_statement_fresh_field_map,
    NebulaFPrimeClaimCoordinateOverlaySynthesis, NebulaFPrimeClaimReplayArmKind, NebulaFPrimeClaimReplaySynthesis,
};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const SCHEMA_VERSION: usize = 4;
const PROFILE_ID: &str = "nebula-f-prime-streaming-claim-replay-v4";
const FRAME_FIELDS: usize = 88_023;
const CHUNK_FIELDS: usize = 1_024;
const FINAL_CHUNK_FIELDS: usize = 983;
const FULL_CHUNKS: usize = 85;
const TRANSITION_STATE_WORDS: usize = 472;
const STATE_DIGEST_WORDS: usize = 8;
const SHARED_PUBLIC_WORDS: usize = 10;
const PUBLIC_BITS_PER_WORD: usize = 64;
const PI_CCS_STATEMENT_FRESH_FIELDS: usize = 25_648;
const PI_CCS_RUNNING_METADATA_FIELDS: usize = 61_992;
const COORDINATE_DIGITS: usize = 41;
const COORDINATE_OPENING_COLUMNS: usize = 122;
const COORDINATE_OPENING_ROWS: usize = 124;
const COORDINATE_OUTPUTS: usize = 108;

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
    bits: [usize; 64],
    high_flag: usize,
    inverse: usize,
}

#[derive(Clone, Debug)]
struct Poseidon2Call {
    rows: Range<usize>,
    inputs: [usize; 8],
    first_allocated: usize,
}

#[derive(Clone, Debug)]
struct CoordinateCall {
    map_kind: CoordinateMapKind,
    rows: Range<usize>,
    chunk_index: usize,
    chunk_base: usize,
    zero_digit_start: usize,
    active_digit_base: usize,
    d_column: usize,
    kappa_column: usize,
    output_base: usize,
    seeded_row_start: usize,
    chunk_size: usize,
    seeds_by_output: Vec<Vec<[u8; 32]>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoordinateMapKind {
    StatementFresh,
    RunningMetadata,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OwnerKind {
    Canonical,
    Poseidon2,
    Coordinate,
    Glue,
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
    row_count: usize,
    column_count: usize,
    public_column_count: usize,
    active_fields: usize,
    replay_poseidon2_call_count: usize,
    state_digest_poseidon2_call_count: usize,
    state_word_columns: Vec<usize>,
    public_word_call_indices: Vec<usize>,
    after_digest_pin_columns: Vec<usize>,
    before_digest_pin_columns: Vec<usize>,
    canonical_calls: Vec<CanonicalCall>,
    poseidon2_calls: Vec<Poseidon2Call>,
    coordinate_calls: Vec<CoordinateCall>,
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

fn normalized_matrix(rows: usize, trips: &[(usize, usize, F)]) -> Vec<Vec<(usize, F)>> {
    let mut raw = vec![Vec::new(); rows];
    for &(row, column, coefficient) in trips {
        assert!(row < rows, "sparse triplet row is in range");
        raw[row].push((column, coefficient));
    }
    raw.into_iter().map(normalize_terms).collect()
}

fn normalized_rows(builder: &R1csBuilder) -> Vec<SparseRow> {
    let (a, b, c) = builder.sparse_triplets();
    let a = normalized_matrix(builder.rows(), a);
    let b = normalized_matrix(builder.rows(), b);
    let c = normalized_matrix(builder.rows(), c);
    (0..builder.rows())
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
    assert_eq!(field.col(), 1);
    let bits = decompose_var_to_u64_bits(&mut builder, field);
    assert_eq!(bits.map(Var::col), std::array::from_fn(|index| index + 2));
    assert_eq!(builder.rows(), 69);
    assert_eq!(builder.cols(), 68);
    normalized_rows(&builder)
}

fn poseidon2_template() -> Vec<SparseRow> {
    let mut builder = R1csBuilder::new();
    let inputs = std::array::from_fn(|lane| builder.alloc(F::from_u64((lane + 1) as u64)));
    assert_eq!(inputs.map(Var::col), std::array::from_fn(|index| index + 1));
    let _ = enforce_poseidon2_permutation(&mut builder, &inputs);
    assert!(builder.is_satisfied());
    assert_eq!(builder.rows(), 600);
    assert_eq!(builder.cols(), 609);
    normalized_rows(&builder)
}

fn assert_renamed_slice(
    actual: &[SparseRow],
    rows: Range<usize>,
    template: &[SparseRow],
    column_map: impl Fn(usize) -> usize,
    label: &str,
) {
    assert_eq!(rows.len(), template.len(), "{label} row count");
    for (offset, expected) in template.iter().enumerate() {
        let expected = rename_row(expected, &column_map);
        assert_eq!(actual[rows.start + offset], expected, "{label} row {offset}");
    }
}

fn mark_owner(slots: &mut [Option<(OwnerKind, usize)>], rows: Range<usize>, kind: OwnerKind, index: usize) {
    assert!(rows.start < rows.end && rows.end <= slots.len());
    for slot in &mut slots[rows] {
        assert!(slot.replace((kind, index)).is_none(), "artifact leaf rows overlap");
    }
}

fn build_arm(
    synthesis: &NebulaFPrimeClaimReplaySynthesis,
    active_fields: usize,
    canonical_template: &[SparseRow],
    poseidon2_template: &[SparseRow],
) -> ArmArtifact {
    let builder = synthesis.builder_for_artifact();
    let rows = normalized_rows(builder);
    let mut owner_slots = vec![None; builder.rows()];

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
            mark_owner(&mut owner_slots, call.rows.clone(), OwnerKind::Canonical, index);
            call
        })
        .collect::<Vec<_>>();
    assert_eq!(canonical_calls.len(), 2 + STATE_DIGEST_WORDS);

    let poseidon2_calls = builder
        .poseidon2_permutation_audits()
        .into_iter()
        .enumerate()
        .map(|(index, trace)| {
            assert_eq!(trace.allocated_col_count, 600);
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
                call.rows.clone(),
                poseidon2_template,
                |column| match column {
                    0 => 0,
                    1..=8 => call.inputs[column - 1],
                    _ => call.first_allocated + column - 9,
                },
                "Poseidon2",
            );
            mark_owner(&mut owner_slots, call.rows.clone(), OwnerKind::Poseidon2, index);
            call
        })
        .collect::<Vec<_>>();
    let replay_poseidon2_call_count = active_fields / 4;
    assert!(poseidon2_calls.len() > replay_poseidon2_call_count);
    let state_digest_poseidon2_call_count = poseidon2_calls.len() - replay_poseidon2_call_count;

    let coordinate_calls = builder
        .seeded_phi81_a_blocks()
        .iter()
        .enumerate()
        .map(|(index, block)| {
            let (map_kind, active, total_fields, output_columns) = match block.word_starts().len() {
                PI_CCS_STATEMENT_FRESH_FIELDS => (
                    CoordinateMapKind::StatementFresh,
                    synthesis.statement_fresh_fields(),
                    PI_CCS_STATEMENT_FRESH_FIELDS,
                    (0..COORDINATE_OUTPUTS)
                        .map(|coordinate| {
                            synthesis
                                .partial_statement_fresh_commitment_column(coordinate)
                                .expect("partial statement-and-fresh commitment output")
                        })
                        .collect::<Vec<_>>(),
                ),
                PI_CCS_RUNNING_METADATA_FIELDS => (
                    CoordinateMapKind::RunningMetadata,
                    synthesis.running_metadata_fields(),
                    PI_CCS_RUNNING_METADATA_FIELDS,
                    (0..COORDINATE_OUTPUTS)
                        .map(|coordinate| {
                            synthesis
                                .partial_running_metadata_commitment_column(coordinate)
                                .expect("partial running-metadata commitment output")
                        })
                        .collect::<Vec<_>>(),
                ),
                total => panic!("unexpected PiCCS coordinate-map width {total}"),
            };
            assert!(!active.is_empty(), "seeded coordinate block requires active fields");
            assert_eq!(block.word_width(), COORDINATE_DIGITS);
            assert_eq!(block.kappa(), 2);
            assert_eq!(block.message_cols(), (total_fields * COORDINATE_DIGITS).div_ceil(D));
            assert_eq!(block.row_end() - block.row_start(), COORDINATE_OUTPUTS);

            let active_digit_base = block.word_starts()[active[0].0];
            for (rank, &(field, offset)) in active.iter().enumerate() {
                assert_eq!(
                    block.word_starts()[field],
                    active_digit_base + rank * COORDINATE_OPENING_COLUMNS,
                    "active coordinate words follow the exact field order",
                );
                assert_eq!(
                    synthesis.chunk_column(offset),
                    Some(synthesis.chunk_column(0).expect("chunk base") + offset),
                );
            }
            let active_fields = active
                .iter()
                .map(|&(field, _)| field)
                .collect::<std::collections::BTreeSet<_>>();
            let zero_digit_start = (0..total_fields)
                .find(|field| !active_fields.contains(field))
                .map(|field| block.word_starts()[field])
                .expect("coordinate phase leaves inactive fields");
            for field in 0..total_fields {
                if !active_fields.contains(&field) {
                    assert_eq!(block.word_starts()[field], zero_digit_start);
                }
            }

            let output_base = output_columns[0];
            assert_eq!(
                output_columns,
                (output_base..output_base + COORDINATE_OUTPUTS).collect::<Vec<_>>()
            );
            let source_rows = COORDINATE_DIGITS + active.len() * COORDINATE_OPENING_ROWS + 2;
            let row_start = block
                .row_start()
                .checked_sub(source_rows)
                .expect("coordinate source rows precede seeded rows");
            let call = CoordinateCall {
                map_kind,
                rows: row_start..block.row_end(),
                chunk_index: synthesis.chunk_index(),
                chunk_base: synthesis.chunk_column(0).expect("chunk base"),
                zero_digit_start,
                active_digit_base,
                d_column: output_base - 2,
                kappa_column: output_base - 1,
                output_base,
                seeded_row_start: block.row_start(),
                chunk_size: block.chunk_size(),
                seeds_by_output: block.chunk_seeds_by_row().to_vec(),
            };
            assert_eq!(call.rows.len(), source_rows + COORDINATE_OUTPUTS);
            mark_owner(&mut owner_slots, call.rows.clone(), OwnerKind::Coordinate, index);
            call
        })
        .collect::<Vec<_>>();
    assert_eq!(
        coordinate_calls.is_empty(),
        synthesis.statement_fresh_fields().is_empty() && synthesis.running_metadata_fields().is_empty()
    );

    let state_word_columns = (0..TRANSITION_STATE_WORDS)
        .map(|index| {
            synthesis
                .state_word_column(index)
                .expect("state word column")
        })
        .collect::<Vec<_>>();

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
    assert_eq!(
        public_word_call_indices,
        vec![2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
        "shared public words use the exact digest-then-cursor order"
    );

    let mut glue_rows = Vec::new();
    for row in 0..rows.len() {
        if owner_slots[row].is_none() {
            let index = glue_rows.len();
            glue_rows.push(GlueRow {
                index: row,
                row: rows[row].clone(),
            });
            owner_slots[row] = Some((OwnerKind::Glue, index));
        }
    }

    let mut owners = Vec::new();
    let mut cursor = 0;
    while cursor < rows.len() {
        let (kind, index) = owner_slots[cursor].expect("every source row has one owner");
        let owned_rows = match kind {
            OwnerKind::Canonical => canonical_calls[index].rows.clone(),
            OwnerKind::Poseidon2 => poseidon2_calls[index].rows.clone(),
            OwnerKind::Coordinate => coordinate_calls[index].rows.clone(),
            OwnerKind::Glue => cursor..cursor + 1,
        };
        assert_eq!(owned_rows.start, cursor, "owner range starts at the row cursor");
        assert!(
            owner_slots[owned_rows.clone()]
                .iter()
                .all(|slot| *slot == Some((kind, index))),
            "one owner covers its complete range"
        );
        owners.push(Owner {
            rows: owned_rows.clone(),
            kind,
            index,
        });
        cursor = owned_rows.end;
    }

    ArmArtifact {
        row_count: builder.rows(),
        column_count: builder.cols(),
        public_column_count: synthesis.public_columns(),
        active_fields,
        replay_poseidon2_call_count,
        state_digest_poseidon2_call_count,
        state_word_columns,
        public_word_call_indices,
        after_digest_pin_columns: synthesis.after_digest_pin_columns().to_vec(),
        before_digest_pin_columns: synthesis.before_digest_pin_columns().to_vec(),
        canonical_calls,
        poseidon2_calls,
        coordinate_calls,
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
        lean_terms(&row.c)
    )
}

fn lean_seed_rows(rows: &[Vec<[u8; 32]>]) -> String {
    let rows = rows
        .iter()
        .map(|chunks| {
            let chunks = chunks
                .iter()
                .map(|seed| {
                    format!(
                        "[{}]",
                        seed.iter()
                            .map(u8::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
                .collect::<Vec<_>>();
            format!("[{}]", chunks.join(", "))
        })
        .collect::<Vec<_>>();
    format!("[{}]", rows.join(", "))
}

fn lean_seed_schedule(call: &CoordinateCall) -> String {
    format!(
        "{{ chunkSize := {}, seedsByOutput := {}, rejectionFuel := 16 }}",
        call.chunk_size,
        lean_seed_rows(&call.seeds_by_output),
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
    let coordinate_calls = grouped_list(
        arm.coordinate_calls
            .iter()
            .map(|call| {
                let (map_kind, schedule) = match call.map_kind {
                    CoordinateMapKind::StatementFresh => (".statementFresh", "statementFreshSchedule"),
                    CoordinateMapKind::RunningMetadata => (".runningMetadata", "runningMetadataSchedule"),
                };
                format!(
                    "{{ mapKind := {map_kind}, rowStart := {}, rowEnd := {}, chunkIndex := {}, chunkBase := {}, \
                     zeroDigitStart := {}, activeDigitBase := {}, dColumn := {}, kappaColumn := {}, \
                     outputBase := {}, seededRowStart := {}, chunkSize := {schedule}.chunkSize, \
                     seedsByOutput := {schedule}.seedsByOutput }}",
                    call.rows.start,
                    call.rows.end,
                    call.chunk_index,
                    call.chunk_base,
                    call.zero_digit_start,
                    call.active_digit_base,
                    call.d_column,
                    call.kappa_column,
                    call.output_base,
                    call.seeded_row_start,
                )
            })
            .collect(),
        1,
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
                    OwnerKind::Coordinate => ".coordinate",
                    OwnerKind::Glue => ".glue",
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
        "{{ rowCount := {}, columnCount := {}, publicColumnCount := {}, activeFields := {},\n    \
         replayPoseidon2CallCount := {}, stateDigestPoseidon2CallCount := {},\n    \
         stateWordColumns := {},\n    \
         publicWordCallIndices := {},\n    \
         afterDigestPinColumns := {}, beforeDigestPinColumns := {},\n    \
         canonicalCalls := {canonical_calls},\n    \
         poseidon2Calls := {poseidon2_calls},\n    \
         coordinateCalls := {coordinate_calls},\n    \
         glueRows := {glue_rows},\n    \
         owners := {owners} }}",
        arm.row_count,
        arm.column_count,
        arm.public_column_count,
        arm.active_fields,
        arm.replay_poseidon2_call_count,
        arm.state_digest_poseidon2_call_count,
        lean_nat_list(arm.state_word_columns.iter().copied()),
        lean_nat_list(arm.public_word_call_indices.iter().copied()),
        lean_nat_list(arm.after_digest_pin_columns.iter().copied()),
        lean_nat_list(arm.before_digest_pin_columns.iter().copied()),
    )
}

fn render_artifact() -> String {
    let full = NebulaFPrimeClaimReplaySynthesis::production_full(0).expect("first full claim chunk");
    let final_chunk = NebulaFPrimeClaimReplaySynthesis::production_final();
    let canonical = canonical_template();
    let poseidon2 = poseidon2_template();
    let full_arm = build_arm(&full, CHUNK_FIELDS, &canonical, &poseidon2);
    let final_arm = build_arm(&final_chunk, FINAL_CHUNK_FIELDS, &canonical, &poseidon2);
    let shape = production_claim_replay_shape_audit().expect("claim-replay shape audit");

    let full_statement_fresh = full_arm
        .coordinate_calls
        .iter()
        .find(|call| call.map_kind == CoordinateMapKind::StatementFresh)
        .expect("full arm statement-and-fresh schedule");
    let full_running_metadata = full_arm
        .coordinate_calls
        .iter()
        .find(|call| call.map_kind == CoordinateMapKind::RunningMetadata)
        .expect("full arm running-metadata schedule");
    let final_statement_fresh = final_arm
        .coordinate_calls
        .iter()
        .find(|call| call.map_kind == CoordinateMapKind::StatementFresh)
        .expect("final arm statement-and-fresh schedule");
    assert_eq!(
        full_statement_fresh.chunk_size, final_statement_fresh.chunk_size,
        "both Rust statement-and-fresh blocks use one chunk size",
    );
    assert_eq!(
        full_statement_fresh.seeds_by_output, final_statement_fresh.seeds_by_output,
        "both Rust statement-and-fresh blocks use one exact seed schedule",
    );

    assert_eq!(full_arm.glue_rows.len(), 486);
    assert_eq!(final_arm.glue_rows.len(), 323);

    let mut payload = String::new();
    writeln!(
        payload,
        "def statementFreshSchedule : Nightstream.Implementation.R1CS.SeededPhi81.SeedSchedule :=\n  {}",
        lean_seed_schedule(full_statement_fresh),
    )
    .unwrap();
    writeln!(
        payload,
        "\ndef runningMetadataSchedule : Nightstream.Implementation.R1CS.SeededPhi81.SeedSchedule :=\n  {}",
        lean_seed_schedule(full_running_metadata),
    )
    .unwrap();
    writeln!(payload, "def fullArm : RawArm :=\n  {}", render_arm(&full_arm)).unwrap();
    writeln!(payload, "\ndef finalArm : RawArm :=\n  {}", render_arm(&final_arm)).unwrap();
    writeln!(
        payload,
        "\ndef rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := {SCHEMA_VERSION}, profileId := \"{PROFILE_ID}\",\n    \
            frameFields := {FRAME_FIELDS}, chunkFields := {CHUNK_FIELDS},\n    \
            finalChunkFields := {FINAL_CHUNK_FIELDS}, fullChunks := {FULL_CHUNKS},\n    \
            transitionStateWords := {TRANSITION_STATE_WORDS}, stateDigestWords := {STATE_DIGEST_WORDS},\n    \
            sharedPublicWords := {SHARED_PUBLIC_WORDS}, publicBitsPerWord := {PUBLIC_BITS_PER_WORD},\n    \
            sharedPrivateFields := {}, lowNormRows := {}, lowNormColumns := {},\n    \
            lowNormPublicColumns := {}, lowNormTotalCoordinates := {},\n    \
            lowNormArity := {}, lowNormDegree := {},\n    \
            lowNormSharedPrivateCoordinates := {},\n    \
            lowNormFullBranchCoordinates := {}, lowNormFinalBranchCoordinates := {},\n    \
            lowNormFullPoseidon2Coordinates := {}, lowNormFinalPoseidon2Coordinates := {},\n    \
            full := fullArm, finalChunk := finalArm }}",
        shape.shared_private_fields,
        shape.low_norm_rows,
        shape.low_norm_columns,
        shape.low_norm_public_columns,
        shape.low_norm_total_coordinates,
        shape.low_norm_arity,
        shape.low_norm_degree,
        shape.low_norm_shared_private_coordinates,
        shape.low_norm_full_branch_coordinates,
        shape.low_norm_final_branch_coordinates,
        shape.low_norm_full_poseidon2_coordinates,
        shape.low_norm_final_poseidon2_coordinates,
    )
    .unwrap();

    let hash = sha256_hex(&payload);
    let rendered = format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplaySchema\n\n\
         /-! Generated file: deterministic Rust artifact for the two production\n\
         streaming claim-replay arms.\n\n\
         Owns: the exact full and final-chunk arm geometry emitted by Rust.\n\n\
         Does not own: source compiler semantics, verifier authority, or a\n\
         complete recursive-relation refinement theorem.\n\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         set_option maxRecDepth 524288\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact\n\n\
         def artifactSha256 : String := \"{hash}\"\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay\n"
    );
    assert!(
        rendered.lines().count() < 1_500,
        "generated Lean artifact must stay below 1,500 lines"
    );
    rendered
}

fn generated_artifact_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingClaimReplay.lean",
    )
}

#[test]
fn production_claim_replay_lean_artifact_is_current() {
    let path = generated_artifact_path();
    let rendered = render_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected claim-replay artifact");
        panic!("claim-replay Lean artifact drifted; inspect {}", expected.display());
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_production_claim_replay_lean_artifact() {
    std::fs::write(generated_artifact_path(), render_artifact()).expect("write generated claim-replay artifact");
}

#[test]
fn production_claim_replay_arms_are_satisfied_and_fully_constrained() {
    let full = NebulaFPrimeClaimReplaySynthesis::production_full(0).expect("first full claim chunk");
    let full_width = NebulaFPrimeClaimReplaySynthesis::production_full(61).expect("full-width coordinate chunk");
    let final_chunk = NebulaFPrimeClaimReplaySynthesis::production_final();

    assert_eq!(full.kind(), NebulaFPrimeClaimReplayArmKind::Full);
    assert_eq!(final_chunk.kind(), NebulaFPrimeClaimReplayArmKind::Final);
    assert!(full.is_satisfied(), "full arm: {:?}", full.first_unsatisfied_row());
    assert!(
        full_width.is_satisfied(),
        "full-width coordinate arm: {:?}",
        full_width.first_unsatisfied_row()
    );
    assert!(
        final_chunk.is_satisfied(),
        "final arm: {:?}",
        final_chunk.first_unsatisfied_row()
    );
    assert_eq!(full.poseidon2_permutations(), 378);
    assert_eq!(full_width.poseidon2_permutations(), 378);
    assert_eq!(final_chunk.poseidon2_permutations(), 367);
    assert_eq!(full.public_columns(), 641);
    assert_eq!(final_chunk.public_columns(), 641);
    assert!(
        full.unconstrained_columns().is_empty(),
        "full arm has unused witness columns"
    );
    assert!(
        full_width.unconstrained_columns().is_empty(),
        "full-width coordinate arm has unused witness columns"
    );
    assert!(
        final_chunk.unconstrained_columns().is_empty(),
        "final arm has unused witness columns"
    );
}

#[test]
fn claim_chunks_use_the_exact_piccs_coordinate_partitions() {
    let statement_fresh = production_claim_statement_fresh_field_map();
    let running_metadata = production_claim_running_metadata_field_map();
    assert_eq!(statement_fresh.len(), 86);
    assert_eq!(running_metadata.len(), 86);
    assert_eq!(
        statement_fresh[0],
        (0..52)
            .map(|field| (field, 383 + field))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        statement_fresh[60],
        (52..89)
            .map(|field| (field, 987 + field - 52))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        statement_fresh[61],
        (89..1_113)
            .map(|field| (field, field - 89))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        statement_fresh[81],
        (20_569..21_593)
            .map(|field| (field, field - 20_569))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        statement_fresh[85],
        (24_665..25_648)
            .map(|field| (field, field - 24_665))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        running_metadata[0],
        (0..589)
            .map(|field| (field, 435 + field))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        running_metadata[60],
        (61_005..61_992)
            .map(|field| (field, field - 61_005))
            .collect::<Vec<_>>()
    );

    let statement_fresh_active_chunks = statement_fresh
        .iter()
        .enumerate()
        .filter_map(|(chunk, fields)| (!fields.is_empty()).then_some(chunk))
        .collect::<Vec<_>>();
    assert_eq!(
        statement_fresh_active_chunks,
        std::iter::once(0).chain(60..=85).collect::<Vec<_>>()
    );
    let running_metadata_active_chunks = running_metadata
        .iter()
        .enumerate()
        .filter_map(|(chunk, fields)| (!fields.is_empty()).then_some(chunk))
        .collect::<Vec<_>>();
    assert_eq!(running_metadata_active_chunks, (0..=60).collect::<Vec<_>>());
    assert_eq!(
        statement_fresh
            .iter()
            .flatten()
            .map(|&(field, _)| field)
            .collect::<Vec<_>>(),
        (0..25_648).collect::<Vec<_>>()
    );
    assert_eq!(
        running_metadata
            .iter()
            .flatten()
            .map(|&(field, _)| field)
            .collect::<Vec<_>>(),
        (0..61_992).collect::<Vec<_>>()
    );
}

#[test]
fn claim_replay_rejects_tampered_coordinate_commitments() {
    let mut selected = NebulaFPrimeClaimReplaySynthesis::production_full(0).expect("selected point chunk");
    assert_eq!(selected.statement_fresh_fields().len(), 52);
    assert_eq!(selected.running_metadata_fields().len(), 589);
    let partial = selected
        .partial_statement_fresh_commitment_column(0)
        .expect("selected chunk partial commitment");
    let changed = selected
        .witness_value(partial)
        .expect("partial commitment value")
        + F::ONE;
    selected.tamper_witness_for_test(partial, changed);
    assert!(!selected.is_satisfied(), "changed partial commitment must fail");

    let mut selected = NebulaFPrimeClaimReplaySynthesis::production_full(0).expect("selected point chunk");
    let before = selected
        .before_statement_fresh_commitment_column(0)
        .expect("before coordinate accumulator");
    selected.tamper_witness_for_test(before, F::ONE);
    assert!(
        !selected.is_satisfied(),
        "chunk zero must start from the zero commitment"
    );

    let mut selected = NebulaFPrimeClaimReplaySynthesis::production_full(0).expect("selected point chunk");
    let after = selected
        .after_statement_fresh_commitment_column(0)
        .expect("after coordinate accumulator");
    let changed = selected
        .witness_value(after)
        .expect("after accumulator value")
        + F::ONE;
    selected.tamper_witness_for_test(after, changed);
    assert!(!selected.is_satisfied(), "changed coordinate update must fail");

    let mut running_only = NebulaFPrimeClaimReplaySynthesis::production_full(1).expect("running-only claim chunk");
    assert!(running_only.statement_fresh_fields().is_empty());
    assert!(!running_only.running_metadata_fields().is_empty());
    assert!(running_only
        .partial_statement_fresh_commitment_column(0)
        .is_none());
    let after = running_only
        .after_statement_fresh_commitment_column(0)
        .expect("carried statement-and-fresh accumulator");
    let changed = running_only
        .witness_value(after)
        .expect("carried accumulator value")
        + F::ONE;
    running_only.tamper_witness_for_test(after, changed);
    assert!(
        !running_only.is_satisfied(),
        "a map without local fields must carry its commitment unchanged"
    );

    let mut running = NebulaFPrimeClaimReplaySynthesis::production_full(1).expect("running metadata chunk");
    let partial = running
        .partial_running_metadata_commitment_column(0)
        .expect("running-metadata partial commitment");
    let changed = running
        .witness_value(partial)
        .expect("running partial value")
        + F::ONE;
    running.tamper_witness_for_test(partial, changed);
    assert!(!running.is_satisfied(), "changed running-metadata commitment must fail");
}

#[test]
fn claim_replay_rejects_tampered_chunk_and_declared_output() {
    let mut full = NebulaFPrimeClaimReplaySynthesis::production_full(0).expect("first full claim chunk");
    let chunk = full.chunk_column(17).expect("chunk field column");
    let changed = full.witness_value(chunk).expect("chunk field value") + F::ONE;
    full.tamper_witness_for_test(chunk, changed);
    assert!(!full.is_satisfied(), "changed chunk field must fail");

    let mut full = NebulaFPrimeClaimReplaySynthesis::production_full(0).expect("first full claim chunk");
    let digest_bit = full
        .public_output_column(0)
        .expect("after-state digest bit");
    let changed = F::ONE - full.witness_value(digest_bit).expect("digest bit value");
    full.tamper_witness_for_test(digest_bit, changed);
    assert!(!full.is_satisfied(), "changed public state digest bit must fail");

    let mut final_chunk = NebulaFPrimeClaimReplaySynthesis::production_final();
    let output = final_chunk
        .after_runtime_column(0)
        .expect("declared output lane");
    let changed = final_chunk.witness_value(output).expect("output value") + F::ONE;
    final_chunk.tamper_witness_for_test(output, changed);
    assert!(!final_chunk.is_satisfied(), "changed final state must fail");
}

fn decode_public_word(synthesis: &NebulaFPrimeClaimReplaySynthesis, word: usize) -> u64 {
    (0..PUBLIC_BITS_PER_WORD).fold(0u64, |value, bit| {
        let index = word * PUBLIC_BITS_PER_WORD + bit;
        let column = synthesis
            .public_output_column(index)
            .expect("public bit column");
        let bit_value = synthesis
            .witness_value(column)
            .expect("public bit value")
            .as_canonical_u64();
        assert!(bit_value <= 1, "public output is a bit");
        value | (bit_value << bit)
    })
}

#[test]
fn claim_replay_public_words_use_digest_then_cursor_layout() {
    let full = NebulaFPrimeClaimReplaySynthesis::production_full(0).expect("first full claim chunk");
    let final_chunk = NebulaFPrimeClaimReplaySynthesis::production_final();

    assert_eq!(decode_public_word(&full, 8), 83);
    assert_eq!(decode_public_word(&full, 9), 84);
    assert_eq!(decode_public_word(&final_chunk, 8), 168);
    assert_eq!(decode_public_word(&final_chunk, 9), 169);
}

#[test]
fn production_claim_replay_shape_is_exact_and_bounded() {
    let audit = production_claim_replay_shape_audit().expect("claim-replay shape audit");
    eprintln!("{audit:#?}");
    assert_eq!(audit.full.poseidon2_permutations, 378);
    assert_eq!(audit.final_chunk.poseidon2_permutations, 367);
    assert_eq!(audit.full.public_columns, 641);
    assert_eq!(audit.final_chunk.public_columns, 641);
    assert_eq!(audit.low_norm_rows, 167_491);
    assert_eq!(audit.low_norm_columns, 808_110);
    assert_eq!(audit.low_norm_public_columns, 648);
    assert_eq!(audit.low_norm_total_coordinates, 808_068);
    assert_eq!(audit.low_norm_shared_private_coordinates, 476);
    assert_eq!(audit.low_norm_full_branch_coordinates, 796_380);
    assert_eq!(audit.low_norm_final_branch_coordinates, 786_634);
    assert_eq!(audit.low_norm_full_poseidon2_coordinates, 748_196);
    assert_eq!(audit.low_norm_final_poseidon2_coordinates, 726_438);
    assert!(
        audit.low_norm_rows < 1 << 20,
        "one claim-replay step must stay below 2^20 rows"
    );
    assert!(
        audit.low_norm_columns < 1 << 20,
        "one claim-replay step must stay below 2^20 columns"
    );
}

#[test]
fn claim_replay_candidate_shapes_are_monotone() {
    let candidates = [64, 128, 256, 512, 1_024];
    let audits = candidates.map(|chunk_fields| {
        claim_replay_shape_audit_for_chunk_fields(chunk_fields).expect("valid rate-aligned candidate")
    });
    let exact_shapes = audits.map(|audit| {
        (
            audit.chunk_fields,
            audit.final_chunk_fields,
            audit.full_chunks,
            audit.low_norm_rows,
            audit.low_norm_columns,
            audit.low_norm_total_coordinates,
        )
    });
    assert_eq!(
        exact_shapes,
        [
            (64, 23, 1_375, 27_604, 286_848, 286_828),
            (128, 87, 687, 34_274, 319_950, 319_948),
            (256, 215, 343, 47_668, 386_208, 386_188),
            (512, 471, 171, 82_497, 523_692, 523_652),
            (1_024, 983, 85, 167_491, 808_110, 808_068),
        ]
    );

    for (chunk_fields, audit) in candidates.into_iter().zip(audits) {
        assert_eq!(audit.chunk_fields, chunk_fields);
        assert_eq!(audit.full_chunks * chunk_fields + audit.final_chunk_fields, 88_023);
        assert_eq!(audit.full.poseidon2_permutations, chunk_fields / 4 + 122);
        assert_eq!(
            audit.final_chunk.poseidon2_permutations,
            audit.final_chunk_fields / 4 + 122
        );
        assert!(audit.low_norm_columns < 1 << 20);
        eprintln!(
            "chunk={chunk_fields} steps={} rows={} columns={} coordinates={}",
            audit.full_chunks + 1,
            audit.low_norm_rows,
            audit.low_norm_columns,
            audit.low_norm_total_coordinates,
        );
    }

    for pair in audits.windows(2) {
        assert!(pair[0].low_norm_rows < pair[1].low_norm_rows);
        assert!(pair[0].low_norm_columns < pair[1].low_norm_columns);
    }
}

#[test]
fn claim_coordinate_overlay_uses_exact_schedule_kinds_and_private_links() {
    let kinds = production_claim_coordinate_overlay_kind_map();
    assert_eq!(kinds.len(), 400);
    assert_eq!(kinds[82], 0);
    assert_eq!(kinds[83], 1);
    assert_eq!(kinds[84], 2);
    assert_eq!(kinds[143], 61);
    assert_eq!(kinds[164], 82);
    assert_eq!(kinds[165], 83);
    assert_eq!(kinds[168], 86);
    assert_eq!(kinds[169], 0);

    let links = production_claim_coordinate_overlay_links();
    assert_eq!(links.len(), 86);
    assert_eq!(links[0].overlay_kind, 1);
    assert_eq!(links[0].fields.len(), 432 + 641);
    assert_eq!(links[1].overlay_kind, 2);
    assert_eq!(links[1].fields.len(), 432 + 1_024);
    assert_eq!(links[60].overlay_kind, 61);
    assert_eq!(links[60].fields.len(), 432 + 1_024);
    assert_eq!(links[85].overlay_kind, 86);
    assert_eq!(links[85].fields.len(), 432 + 983);
}

#[test]
fn claim_coordinate_overlay_arms_are_satisfied_and_fully_constrained() {
    for kind in 0..87 {
        let synthesis =
            NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(kind).expect("production overlay kind");
        assert!(synthesis.is_satisfied(), "overlay kind {kind}");
        assert!(
            synthesis.unconstrained_columns().is_empty(),
            "overlay kind {kind} has unused columns"
        );
    }

    for (label, kind, offset) in [
        ("prior point", 1, 383),
        ("running commitment", 1, 435),
        ("running public input", 54, 595),
        ("running evaluation", 61, 987),
        ("fresh commitment", 82, 651),
        ("fresh public input", 86, 443),
    ] {
        let mut active = NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(kind)
            .expect("selected claim metadata overlay");
        let column = active
            .chunk_columns()
            .iter()
            .find_map(|&(candidate, column)| (candidate == offset).then_some(column))
            .expect("exact metadata frame offset");
        let changed = active.witness_value(column).expect("overlay chunk value") + F::ONE;
        active.tamper_witness_for_test(column, changed);
        assert!(!active.is_satisfied(), "changed {label} field must fail");
    }
}

#[test]
fn claim_coordinate_overlay_selective_union_is_bounded() {
    let audit = production_claim_coordinate_overlay_shape_audit().expect("coordinate overlay shape");
    eprintln!("{audit:#?}");
    assert_eq!(audit.kinds, 87);
    assert_eq!(audit.active_kinds, 86);
    assert_eq!(audit.active_fields, 87_640);
    assert_eq!(audit.source_rows, 10_899_441);
    assert_eq!(audit.source_columns, 10_830_247);
    assert_eq!(audit.low_norm_rows, 5_404_913);
    assert_eq!(audit.low_norm_columns, 72_576);
    assert_eq!(audit.low_norm_public_columns, 1);
    assert_eq!(audit.low_norm_total_coordinates, 72_570);
    assert!(audit.low_norm_rows < 1 << 24);
    assert!(audit.low_norm_columns < 1 << 24);

    let relation =
        build_production_claim_coordinate_overlay_low_norm_r1cs().expect("build coordinate overlay relation");
    assert_eq!(relation.selector_cols().len(), 87);
    assert_eq!(relation.public_input_len(), 1);
    assert_eq!(relation.structure().n, audit.low_norm_rows);
    assert_eq!(relation.structure().m, audit.low_norm_columns);
}

#[test]
fn claim_replay_base_stores_poseidon_body_without_coordinate_overlay() {
    let full_zero = NebulaFPrimeClaimReplaySynthesis::production_base_full(0).expect("first base full arm");
    let full_active = NebulaFPrimeClaimReplaySynthesis::production_base_full(61).expect("full-width base arm");
    let final_chunk = NebulaFPrimeClaimReplaySynthesis::production_base_final();
    assert!(full_zero.is_satisfied());
    assert!(full_active.is_satisfied());
    assert!(final_chunk.is_satisfied());
    assert_eq!(full_zero.rows(), full_active.rows());
    assert_eq!(full_zero.columns(), full_active.columns());
    assert!(full_zero
        .partial_statement_fresh_commitment_column(0)
        .is_none());
    assert!(full_zero
        .partial_running_metadata_commitment_column(0)
        .is_none());
    assert!(full_active
        .partial_statement_fresh_commitment_column(0)
        .is_none());
    assert!(full_active
        .partial_running_metadata_commitment_column(0)
        .is_none());
    assert!(final_chunk
        .partial_statement_fresh_commitment_column(0)
        .is_none());
    assert!(final_chunk
        .partial_running_metadata_commitment_column(0)
        .is_none());
    assert!(full_zero.unconstrained_columns().is_empty());
    assert!(full_active.unconstrained_columns().is_empty());
    assert!(final_chunk.unconstrained_columns().is_empty());

    let audit = production_claim_replay_base_shape_audit().expect("claim replay base shape");
    eprintln!("{audit:#?}");
    assert_eq!(audit.full_rows, 227_544);
    assert_eq!(audit.full_columns, 228_987);
    assert_eq!(audit.final_rows, 220_997);
    assert_eq!(audit.final_columns, 222_387);
    assert_eq!(audit.low_norm_rows, 66_757);
    assert_eq!(audit.low_norm_columns, 783_648);
    assert_eq!(audit.low_norm_public_columns, 648);
    assert_eq!(audit.low_norm_total_coordinates, 783_628);
    assert!(audit.low_norm_rows < 1 << 24);
    assert!(audit.low_norm_columns < 1 << 24);
    let relation = build_production_claim_replay_base_low_norm_r1cs().expect("build claim replay base relation");
    assert_eq!(relation.selector_cols().len(), 2);
    assert_eq!(relation.structure().n, audit.low_norm_rows);
    assert_eq!(relation.structure().m, audit.low_norm_columns);
}
