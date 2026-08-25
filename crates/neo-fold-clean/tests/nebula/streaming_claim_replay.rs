//! Constraint and size checks for the bounded-width claim-replay arms.

#[path = "streaming_claim_replay/coordinate_overlay_artifact.rs"]
mod coordinate_overlay_artifact;

#[path = "../gadgets/lean_artifact_support.rs"]
#[allow(dead_code)]
mod lean_artifact_support;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Range;
use std::path::Path;

use lean_artifact_support::{lean_nat_list, sha256_hex};
use neo_ccs::CcsMatrix;
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::{enforce_poseidon2_permutation, R1csBuilder, Var};
use neo_fold_clean::frontends::nebula::f_prime::{
    build_production_claim_coordinate_overlay_low_norm_r1cs, build_production_claim_replay_base_low_norm_r1cs,
    claim_replay_shape_audit_for_chunk_fields, production_claim_coordinate_overlay_kind_map,
    production_claim_coordinate_overlay_link_runs, production_claim_coordinate_overlay_links,
    production_claim_coordinate_overlay_shape_audit, production_claim_replay_base_shape_audit,
    production_claim_replay_base_source_arms, production_claim_replay_shape_audit,
    production_claim_running_commitment_field_map, production_claim_running_public_field_map,
    production_claim_statement_fresh_field_map, NebulaFPrimeClaimCoordinateOverlaySynthesis,
    NebulaFPrimeClaimReplayArmKind, NebulaFPrimeClaimReplaySynthesis,
};
use neo_fold_clean::frontends::r1cs_f_prime::{SelectiveCompilerAudit, SparseR1cs};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: usize = 5;
const PROFILE_ID: &str = "nebula-f-prime-streaming-claim-replay-goldilocks-b2-k16-v6";
const FRAME_FIELDS: usize = 99_903;
const CHUNK_FIELDS: usize = 1_024;
const FINAL_CHUNK_FIELDS: usize = 575;
const FULL_CHUNKS: usize = 97;
const TRANSITION_STATE_WORDS: usize = 688;
const STATE_DIGEST_WORDS: usize = 8;
const SHARED_PUBLIC_WORDS: usize = 10;
const PUBLIC_BITS_PER_WORD: usize = 64;
const PI_CCS_STATEMENT_FRESH_FIELDS: usize = 28_672;
const PI_CCS_RUNNING_COMMITMENT_FIELDS: usize = 62_208;
const PI_CCS_RUNNING_PUBLIC_FIELDS: usize = 8_640;
const COORDINATE_DIGITS: usize = 41;
const COORDINATE_OPENING_COLUMNS: usize = 122;
const COORDINATE_OPENING_ROWS: usize = 124;
const COORDINATE_OUTPUTS: usize = 108;
const SOURCE_HASH_SCHEMA: &str = "nightstream-normalized-sparse-r1cs-csc-v1";
const BASE_SOURCE_IDENTITY: &str = "rust:nightstream/streaming-claim-replay-base/source-rows/v1";
const COMPLETE_SOURCE_IDENTITY: &str = "rust:nightstream/streaming-claim-replay-complete/source-rows/v6";
const FINAL_LINK_IDENTITY: &str = "rust:nightstream/streaming-selective-ccs/claim-replay-base-coordinate-links/v1";

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
    RunningCommitments,
    RunningPublic,
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

#[derive(Clone, Debug)]
struct ReceiptRange {
    start: usize,
    stop: usize,
}

#[derive(Clone, Debug)]
struct ReceiptStage {
    path: String,
    rows: ReceiptRange,
    columns: ReceiptRange,
}

#[derive(Clone, Debug)]
struct ReceiptNamedRange {
    name: String,
    range: ReceiptRange,
}

#[derive(Clone, Debug)]
struct ReceiptState {
    before_statement_fresh: usize,
    after_statement_fresh: usize,
    before_running_commitments: usize,
    after_running_commitments: usize,
    before_running_public: usize,
    after_running_public: usize,
}

#[derive(Clone, Debug)]
struct BaseReceipt {
    label: &'static str,
    source_sha256: String,
    source_rows: usize,
    source_columns: usize,
    public_columns: usize,
    public_word_bindings: Vec<(ReceiptRange, ReceiptRange)>,
    physical_stages: Vec<ReceiptStage>,
    complete_stages: Vec<ReceiptStage>,
    row_families: Vec<ReceiptNamedRange>,
    column_families: Vec<ReceiptNamedRange>,
    source_state: ReceiptState,
    normalized_state: ReceiptState,
    source_chunk: ReceiptRange,
    normalized_chunk: ReceiptRange,
    replay_initial_capacity: (ReceiptRange, ReceiptRange),
    replay_poseidon2: (ReceiptRange, ReceiptRange),
    phase_kind: usize,
    chunk_scope: ReceiptRange,
    replay_poseidon2_calls: usize,
    compiler_source_runs: usize,
    compiler_mapping_sha256: String,
    final_rows: usize,
    final_columns: usize,
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

fn source_rows_sha256(source: &SparseR1cs) -> String {
    let mut hasher = Sha256::new();
    hasher.update(SOURCE_HASH_SCHEMA.as_bytes());
    hasher.update([0]);
    hasher.update((source.n as u64).to_le_bytes());
    hasher.update((source.m as u64).to_le_bytes());
    hasher.update((source.m_in as u64).to_le_bytes());
    for (matrix_index, matrix) in [&source.a, &source.b, &source.c].into_iter().enumerate() {
        assert!(matrix.seeded_phi81_blocks().is_empty());
        assert!(matrix.geometric_runs().is_empty());
        let csc = matrix
            .sparse_component()
            .expect("claim-replay source uses canonical CSC matrices");
        assert!(csc.is_canonical());
        hasher.update([matrix_index as u8]);
        hasher.update((csc.nrows as u64).to_le_bytes());
        hasher.update((csc.ncols as u64).to_le_bytes());
        hasher.update((csc.col_ptr.len() as u64).to_le_bytes());
        for &pointer in &csc.col_ptr {
            hasher.update(pointer.to_le_bytes());
        }
        hasher.update((csc.row_idx.len() as u64).to_le_bytes());
        for (&row, coefficient) in csc.row_idx.iter().zip(&csc.vals) {
            hasher.update(row.to_le_bytes());
            hasher.update(coefficient.as_canonical_u64().to_le_bytes());
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn matrix_stage_terms(matrix: &CcsMatrix<F>, rows: Range<usize>) -> Vec<(usize, usize, u64)> {
    assert!(matrix
        .seeded_phi81_blocks()
        .iter()
        .all(|block| block.row_end() <= rows.start || rows.end <= block.row_start()));
    assert!(matrix
        .geometric_runs()
        .iter()
        .all(|run| !rows.contains(&run.row())));
    let csc = matrix
        .sparse_component()
        .expect("claim-replay source uses canonical CSC matrices");
    let mut terms = Vec::new();
    for column in 0..csc.ncols {
        let start = csc.col_ptr[column] as usize;
        let stop = csc.col_ptr[column + 1] as usize;
        for index in start..stop {
            let row = csc.row_idx[index] as usize;
            if rows.contains(&row) {
                terms.push((row - rows.start, column, csc.vals[index].as_canonical_u64()));
            }
        }
    }
    terms.sort_unstable();
    terms
}

fn receipt_stage(stage: &neo_fold_clean::engine::r1cs_circuit::PhysicalStageRange) -> ReceiptStage {
    ReceiptStage {
        path: stage.path().to_owned(),
        rows: ReceiptRange {
            start: stage.row_start(),
            stop: stage.row_end(),
        },
        columns: ReceiptRange {
            start: stage.column_start(),
            stop: stage.column_end(),
        },
    }
}

fn assert_stage_partition(source: &SparseR1cs) {
    let mut row_cursor = 0;
    let mut column_cursor = source.m_in;
    for stage in source.physical_stage_ranges() {
        assert_eq!(
            stage.row_start(),
            row_cursor,
            "physical stages must cover every source row once"
        );
        assert_eq!(
            stage.column_start(),
            column_cursor,
            "physical stages must cover every normalized private column once"
        );
        assert!(stage.row_start() <= stage.row_end());
        assert!(stage.column_start() <= stage.column_end());
        row_cursor = stage.row_end();
        column_cursor = stage.column_end();
    }
    assert_eq!(row_cursor, source.n, "physical stage rows have no remainder");
    assert_eq!(column_cursor, source.m, "physical stage columns have no remainder");
}

fn assert_exact_stage_transport(base: &SparseR1cs, complete: &SparseR1cs) -> Vec<ReceiptStage> {
    assert_stage_partition(base);
    assert_stage_partition(complete);
    assert_eq!(base.m_in, complete.m_in);
    let mut seen = BTreeMap::<&str, usize>::new();
    let mut matched = Vec::with_capacity(base.physical_stage_ranges().len());
    for base_stage in base.physical_stage_ranges() {
        let occurrence = seen.entry(base_stage.path()).or_default();
        let complete_stage = complete
            .physical_stage_ranges()
            .iter()
            .filter(|stage| stage.path() == base_stage.path())
            .nth(*occurrence)
            .unwrap_or_else(|| {
                panic!(
                    "complete arm lacks retained stage {} occurrence {}",
                    base_stage.path(),
                    occurrence
                )
            });
        *occurrence += 1;
        assert!(base_stage.rows().len() <= complete_stage.rows().len());
        assert!(base_stage.columns().len() <= complete_stage.columns().len());
        let complete_rows = complete_stage.row_start()..complete_stage.row_start() + base_stage.rows().len();
        let complete_columns =
            complete_stage.column_start()..complete_stage.column_start() + base_stage.columns().len();
        for (base_matrix, complete_matrix) in [(&base.a, &complete.a), (&base.b, &complete.b), (&base.c, &complete.c)] {
            let mut base_terms = matrix_stage_terms(base_matrix, base_stage.rows());
            for (_, column, _) in &mut base_terms {
                if base_stage.columns().contains(column) {
                    *column = complete_stage.column_start() + (*column - base_stage.column_start());
                }
            }
            base_terms.sort_unstable();
            let complete_terms = matrix_stage_terms(complete_matrix, complete_rows.clone());
            assert_eq!(
                base_terms,
                complete_terms,
                "retained stage {} must be an exact source-row transport",
                base_stage.path()
            );
        }
        matched.push(ReceiptStage {
            path: complete_stage.path().to_owned(),
            rows: ReceiptRange {
                start: complete_rows.start,
                stop: complete_rows.end,
            },
            columns: ReceiptRange {
                start: complete_columns.start,
                stop: complete_columns.end,
            },
        });
    }
    matched
}

fn compiler_mapping_sha256(audit: &SelectiveCompilerAudit) -> (usize, String) {
    let mut hasher = Sha256::new();
    hasher.update(b"nightstream-selective-source-row-mapping-v1\0");
    let mut count = 0;
    for (arm, mapping) in audit.rows().arms().iter().enumerate() {
        let mut source_cursor = 0;
        for run in mapping.source_runs() {
            let rows = run.source_rows();
            assert_eq!(rows.start, source_cursor, "source runs must be adjacent");
            source_cursor = rows.end;
            count += 1;
            hasher.update((arm as u64).to_le_bytes());
            hasher.update((rows.start as u64).to_le_bytes());
            hasher.update((rows.end as u64).to_le_bytes());
            hasher.update(format!("{:?}", run.disposition()).as_bytes());
            hasher.update((run.stage_occurrence().unwrap_or(usize::MAX) as u64).to_le_bytes());
            hasher.update((run.emitted_start().unwrap_or(usize::MAX) as u64).to_le_bytes());
        }
    }
    let hash = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    (count, hash)
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
                PI_CCS_RUNNING_COMMITMENT_FIELDS => (
                    CoordinateMapKind::RunningCommitments,
                    synthesis.running_commitment_fields(),
                    PI_CCS_RUNNING_COMMITMENT_FIELDS,
                    (0..COORDINATE_OUTPUTS)
                        .map(|coordinate| {
                            synthesis
                                .partial_running_commitments_binding_column(coordinate)
                                .expect("partial running-commitments binding output")
                        })
                        .collect::<Vec<_>>(),
                ),
                PI_CCS_RUNNING_PUBLIC_FIELDS => (
                    CoordinateMapKind::RunningPublic,
                    synthesis.running_public_fields(),
                    PI_CCS_RUNNING_PUBLIC_FIELDS,
                    (0..COORDINATE_OUTPUTS)
                        .map(|coordinate| {
                            synthesis
                                .partial_running_public_binding_column(coordinate)
                                .expect("partial running-public binding output")
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
    let has_coordinate_outputs = synthesis
        .partial_statement_fresh_commitment_column(0)
        .or_else(|| synthesis.partial_running_commitments_binding_column(0))
        .or_else(|| synthesis.partial_running_public_binding_column(0))
        .is_some();
    assert_eq!(!coordinate_calls.is_empty(), has_coordinate_outputs);

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
                    CoordinateMapKind::RunningCommitments => (".runningCommitments", "runningCommitmentsSchedule"),
                    CoordinateMapKind::RunningPublic => (".runningPublic", "runningPublicSchedule"),
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
        4,
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
    let running_public = NebulaFPrimeClaimReplaySynthesis::production_full(61).expect("running-public claim chunk");
    let final_chunk = NebulaFPrimeClaimReplaySynthesis::production_final();
    let canonical = canonical_template();
    let poseidon2 = poseidon2_template();
    let full_arm = build_arm(&full, CHUNK_FIELDS, &canonical, &poseidon2);
    let running_public_arm = build_arm(&running_public, CHUNK_FIELDS, &canonical, &poseidon2);
    let final_arm = build_arm(&final_chunk, FINAL_CHUNK_FIELDS, &canonical, &poseidon2);
    let shape = production_claim_replay_shape_audit().expect("claim-replay shape audit");

    let full_statement_fresh = full_arm
        .coordinate_calls
        .iter()
        .find(|call| call.map_kind == CoordinateMapKind::StatementFresh)
        .expect("full arm statement-and-fresh schedule");
    let full_running_commitments = full_arm
        .coordinate_calls
        .iter()
        .find(|call| call.map_kind == CoordinateMapKind::RunningCommitments)
        .expect("full arm running-commitments schedule");
    let running_public = running_public_arm
        .coordinate_calls
        .iter()
        .find(|call| call.map_kind == CoordinateMapKind::RunningPublic)
        .expect("running-public schedule");
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

    let mut payload = String::new();
    writeln!(
        payload,
        "def statementFreshSchedule : Nightstream.Implementation.R1CS.SeededPhi81.SeedSchedule :=\n  {}",
        lean_seed_schedule(full_statement_fresh),
    )
    .unwrap();
    writeln!(
        payload,
        "\ndef runningCommitmentsSchedule : Nightstream.Implementation.R1CS.SeededPhi81.SeedSchedule :=\n  {}",
        lean_seed_schedule(full_running_commitments),
    )
    .unwrap();
    writeln!(
        payload,
        "\ndef runningPublicSchedule : Nightstream.Implementation.R1CS.SeededPhi81.SeedSchedule :=\n  {}",
        lean_seed_schedule(running_public),
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

fn public_word_bindings(synthesis: &NebulaFPrimeClaimReplaySynthesis) -> Vec<(ReceiptRange, ReceiptRange)> {
    (0..SHARED_PUBLIC_WORDS)
        .map(|word| {
            let source = (0..PUBLIC_BITS_PER_WORD)
                .map(|bit| {
                    synthesis
                        .public_output_column(word * PUBLIC_BITS_PER_WORD + bit)
                        .expect("complete public bit layout")
                })
                .collect::<Vec<_>>();
            assert_eq!(
                source,
                (source[0]..source[0] + PUBLIC_BITS_PER_WORD).collect::<Vec<_>>(),
                "each public word must occupy one exact source range"
            );
            for (bit, &source_column) in source.iter().enumerate() {
                assert_eq!(
                    synthesis.normalized_field_column_for_artifact(source_column),
                    Some(1 + word * PUBLIC_BITS_PER_WORD + bit),
                    "public word normalization must match the canonical public order"
                );
            }
            (
                ReceiptRange {
                    start: source[0],
                    stop: source[0] + PUBLIC_BITS_PER_WORD,
                },
                ReceiptRange {
                    start: 1 + word * PUBLIC_BITS_PER_WORD,
                    stop: 1 + (word + 1) * PUBLIC_BITS_PER_WORD,
                },
            )
        })
        .collect()
}

fn receipt_normalized_column(source: usize, public_columns: usize, bindings: &[(ReceiptRange, ReceiptRange)]) -> usize {
    if source == 0 {
        return 0;
    }
    if let Some((source_range, normalized_range)) = bindings
        .iter()
        .find(|(source_range, _)| source_range.start <= source && source < source_range.stop)
    {
        return normalized_range.start + source - source_range.start;
    }
    let moved_before = bindings
        .iter()
        .map(|(source_range, _)| {
            (source.saturating_sub(source_range.start)).min(source_range.stop - source_range.start)
        })
        .sum::<usize>();
    public_columns + source - 1 - moved_before
}

fn assert_receipt_normalization(
    synthesis: &NebulaFPrimeClaimReplaySynthesis,
    bindings: &[(ReceiptRange, ReceiptRange)],
) {
    for source in 0..synthesis.columns() {
        assert_eq!(
            synthesis.normalized_field_column_for_artifact(source),
            Some(receipt_normalized_column(source, synthesis.public_columns(), bindings,)),
            "compact receipt must map source column {source} exactly",
        );
    }
}

fn source_column_for_normalized(synthesis: &NebulaFPrimeClaimReplaySynthesis, normalized: usize) -> usize {
    let mut sources = (0..synthesis.columns())
        .filter(|&source| synthesis.normalized_field_column_for_artifact(source) == Some(normalized));
    let source = sources
        .next()
        .expect("normalized column must have one source column");
    assert!(
        sources.next().is_none(),
        "normalized column must have exactly one source column"
    );
    source
}

fn contiguous_receipt_range(columns: &[usize], label: &str) -> ReceiptRange {
    let start = *columns.first().expect("receipt range must not be empty");
    assert_eq!(
        columns,
        (start..start + columns.len()).collect::<Vec<_>>(),
        "{label} must occupy one exact contiguous range"
    );
    ReceiptRange {
        start,
        stop: start + columns.len(),
    }
}

fn replay_initial_capacity_binding(synthesis: &NebulaFPrimeClaimReplaySynthesis) -> (ReceiptRange, ReceiptRange) {
    let normalized_columns = (4..8)
        .map(|lane| {
            synthesis
                .normalized_before_runtime_column(lane)
                .expect("initial replay capacity column")
        })
        .collect::<Vec<_>>();
    let source_columns = normalized_columns
        .iter()
        .map(|&normalized| source_column_for_normalized(synthesis, normalized))
        .collect::<Vec<_>>();
    let source = contiguous_receipt_range(&source_columns, "initial replay capacity source columns");
    let normalized = contiguous_receipt_range(&normalized_columns, "initial replay capacity normalized columns");
    for offset in 0..source_columns.len() {
        assert_eq!(
            synthesis.normalized_field_column_for_artifact(source.start + offset),
            Some(normalized.start + offset),
            "initial replay capacity binding must be exact"
        );
    }
    (source, normalized)
}

fn replay_poseidon2_binding(
    synthesis: &NebulaFPrimeClaimReplaySynthesis,
    source: &SparseR1cs,
) -> (ReceiptRange, ReceiptRange) {
    let mut stages = source
        .physical_stage_ranges()
        .iter()
        .filter(|stage| stage.path() == "nebula.streaming.claim_replay.poseidon2");
    let stage = stages
        .next()
        .expect("production base must have one claim-replay Poseidon2 stage");
    assert!(
        stages.next().is_none(),
        "production base must have exactly one claim-replay Poseidon2 stage"
    );
    let normalized = ReceiptRange {
        start: stage.column_start(),
        stop: stage.column_end(),
    };
    assert!(
        normalized.start < normalized.stop,
        "claim-replay Poseidon2 stage must allocate columns"
    );
    let source_start = source_column_for_normalized(synthesis, normalized.start);
    let source = ReceiptRange {
        start: source_start,
        stop: source_start + normalized.stop - normalized.start,
    };
    assert!(
        source.stop <= synthesis.columns(),
        "claim-replay Poseidon2 source range must be in bounds"
    );
    for offset in 0..source.stop - source.start {
        assert_eq!(
            synthesis.normalized_field_column_for_artifact(source.start + offset),
            Some(normalized.start + offset),
            "claim-replay Poseidon2 binding must be exact"
        );
    }
    (source, normalized)
}

fn receipt_selected_column(
    source: usize,
    public_columns: usize,
    bindings: &[(ReceiptRange, ReceiptRange)],
    source_state: &ReceiptState,
    normalized_state: &ReceiptState,
    source_chunk: &ReceiptRange,
    normalized_chunk: &ReceiptRange,
    replay_initial_capacity: &(ReceiptRange, ReceiptRange),
    replay_poseidon2: &(ReceiptRange, ReceiptRange),
) -> usize {
    for (source_start, normalized_start) in [
        (
            source_state.before_statement_fresh,
            normalized_state.before_statement_fresh,
        ),
        (
            source_state.after_statement_fresh,
            normalized_state.after_statement_fresh,
        ),
        (
            source_state.before_running_commitments,
            normalized_state.before_running_commitments,
        ),
        (
            source_state.after_running_commitments,
            normalized_state.after_running_commitments,
        ),
        (
            source_state.before_running_public,
            normalized_state.before_running_public,
        ),
        (source_state.after_running_public, normalized_state.after_running_public),
    ] {
        if source_start <= source && source < source_start + COORDINATE_OUTPUTS {
            return normalized_start + source - source_start;
        }
    }
    if source_chunk.start <= source && source < source_chunk.stop {
        return normalized_chunk.start + source - source_chunk.start;
    }
    for (source_range, normalized_range) in [replay_initial_capacity, replay_poseidon2] {
        if source_range.start <= source && source < source_range.stop {
            return normalized_range.start + source - source_range.start;
        }
    }
    receipt_normalized_column(source, public_columns, bindings)
}

#[allow(clippy::too_many_arguments)]
fn assert_receipt_selected_normalization(
    synthesis: &NebulaFPrimeClaimReplaySynthesis,
    bindings: &[(ReceiptRange, ReceiptRange)],
    source_state: &ReceiptState,
    normalized_state: &ReceiptState,
    source_chunk: &ReceiptRange,
    normalized_chunk: &ReceiptRange,
    replay_initial_capacity: &(ReceiptRange, ReceiptRange),
    replay_poseidon2: &(ReceiptRange, ReceiptRange),
) {
    for source in 0..synthesis.columns() {
        assert_eq!(
            synthesis.normalized_field_column_for_artifact(source),
            Some(receipt_selected_column(
                source,
                synthesis.public_columns(),
                bindings,
                source_state,
                normalized_state,
                source_chunk,
                normalized_chunk,
                replay_initial_capacity,
                replay_poseidon2,
            )),
            "selected receipt must map source column {source} exactly",
        );
    }
}

fn source_state(synthesis: &NebulaFPrimeClaimReplaySynthesis) -> ReceiptState {
    ReceiptState {
        before_statement_fresh: synthesis
            .before_statement_fresh_commitment_column(0)
            .expect("before statement-and-fresh state base"),
        after_statement_fresh: synthesis
            .after_statement_fresh_commitment_column(0)
            .expect("after statement-and-fresh state base"),
        before_running_commitments: synthesis
            .before_running_commitments_binding_column(0)
            .expect("before running-commitments state base"),
        after_running_commitments: synthesis
            .after_running_commitments_binding_column(0)
            .expect("after running-commitments state base"),
        before_running_public: synthesis
            .before_running_public_binding_column(0)
            .expect("before running-public state base"),
        after_running_public: synthesis
            .after_running_public_binding_column(0)
            .expect("after running-public state base"),
    }
}

fn normalized_state(synthesis: &NebulaFPrimeClaimReplaySynthesis) -> ReceiptState {
    ReceiptState {
        before_statement_fresh: synthesis
            .normalized_before_statement_fresh_commitment_column(0)
            .expect("normalized before statement-and-fresh state base"),
        after_statement_fresh: synthesis
            .normalized_after_statement_fresh_commitment_column(0)
            .expect("normalized after statement-and-fresh state base"),
        before_running_commitments: synthesis
            .normalized_before_running_commitments_binding_column(0)
            .expect("normalized before running-commitments state base"),
        after_running_commitments: synthesis
            .normalized_after_running_commitments_binding_column(0)
            .expect("normalized after running-commitments state base"),
        before_running_public: synthesis
            .normalized_before_running_public_binding_column(0)
            .expect("normalized before running-public state base"),
        after_running_public: synthesis
            .normalized_after_running_public_binding_column(0)
            .expect("normalized after running-public state base"),
    }
}

fn named_row_ranges(source: &SparseR1cs) -> Vec<ReceiptNamedRange> {
    source
        .row_family_ranges()
        .iter()
        .map(|family| ReceiptNamedRange {
            name: family.name.to_owned(),
            range: ReceiptRange {
                start: family.row_start,
                stop: family.row_end,
            },
        })
        .collect()
}

fn named_column_ranges(source: &SparseR1cs) -> Vec<ReceiptNamedRange> {
    source
        .column_family_ranges()
        .iter()
        .map(|family| ReceiptNamedRange {
            name: family.name.to_owned(),
            range: ReceiptRange {
                start: family.column_start,
                stop: family.column_end,
            },
        })
        .collect()
}

fn build_base_receipts() -> (BaseReceipt, BaseReceipt, Vec<ReceiptRange>, Vec<usize>) {
    let full_synthesis =
        NebulaFPrimeClaimReplaySynthesis::production_base_full(0).expect("canonical base full claim chunk");
    let final_synthesis = NebulaFPrimeClaimReplaySynthesis::production_base_final();
    let full_assignment = full_synthesis
        .normalized_field_assignment_for_artifact()
        .expect("normalize full base assignment");
    let final_assignment = final_synthesis
        .normalized_field_assignment_for_artifact()
        .expect("normalize final base assignment");
    let full_public_words = public_word_bindings(&full_synthesis);
    let final_public_words = public_word_bindings(&final_synthesis);
    assert_receipt_normalization(&full_synthesis, &full_public_words);
    assert_receipt_normalization(&final_synthesis, &final_public_words);
    let full_source_chunk = ReceiptRange {
        start: full_synthesis.chunk_column(0).expect("full chunk start"),
        stop: full_synthesis
            .chunk_column(CHUNK_FIELDS - 1)
            .expect("full chunk end")
            + 1,
    };
    let final_source_chunk = ReceiptRange {
        start: final_synthesis.chunk_column(0).expect("final chunk start"),
        stop: final_synthesis
            .chunk_column(FINAL_CHUNK_FIELDS - 1)
            .expect("final chunk end")
            + 1,
    };
    let full_normalized_chunk = ReceiptRange {
        start: full_synthesis
            .normalized_chunk_column(0)
            .expect("normalized full chunk start"),
        stop: full_synthesis
            .normalized_chunk_column(CHUNK_FIELDS - 1)
            .expect("normalized full chunk end")
            + 1,
    };
    let final_normalized_chunk = ReceiptRange {
        start: final_synthesis
            .normalized_chunk_column(0)
            .expect("normalized final chunk start"),
        stop: final_synthesis
            .normalized_chunk_column(FINAL_CHUNK_FIELDS - 1)
            .expect("normalized final chunk end")
            + 1,
    };
    let full_source_state = source_state(&full_synthesis);
    let full_normalized_state = normalized_state(&full_synthesis);
    let final_source_state = source_state(&final_synthesis);
    let final_normalized_state = normalized_state(&final_synthesis);
    let (sources, shared) = production_claim_replay_base_source_arms().expect("exact base source arms");
    assert_eq!(sources.len(), 2);
    assert_eq!(shared, 692);
    sources[0]
        .is_satisfied_by(&full_assignment)
        .expect("full normalized assignment satisfies the exact source");
    sources[1]
        .is_satisfied_by(&final_assignment)
        .expect("final normalized assignment satisfies the exact source");
    let full_replay_initial_capacity = replay_initial_capacity_binding(&full_synthesis);
    let final_replay_initial_capacity = replay_initial_capacity_binding(&final_synthesis);
    let full_replay_poseidon2 = replay_poseidon2_binding(&full_synthesis, &sources[0]);
    let final_replay_poseidon2 = replay_poseidon2_binding(&final_synthesis, &sources[1]);
    assert_receipt_selected_normalization(
        &full_synthesis,
        &full_public_words,
        &full_source_state,
        &full_normalized_state,
        &full_source_chunk,
        &full_normalized_chunk,
        &full_replay_initial_capacity,
        &full_replay_poseidon2,
    );
    assert_receipt_selected_normalization(
        &final_synthesis,
        &final_public_words,
        &final_source_state,
        &final_normalized_state,
        &final_source_chunk,
        &final_normalized_chunk,
        &final_replay_initial_capacity,
        &final_replay_poseidon2,
    );

    let complete_full = NebulaFPrimeClaimReplaySynthesis::production_full(0)
        .expect("complete full claim chunk")
        .into_lowered_for_artifact()
        .expect("lower complete full claim chunk")
        .into_parts()
        .0;
    let complete_final = NebulaFPrimeClaimReplaySynthesis::production_final()
        .into_lowered_for_artifact()
        .expect("lower complete final claim chunk")
        .into_parts()
        .0;
    let full_complete_stages = assert_exact_stage_transport(&sources[0], &complete_full);
    let final_complete_stages = assert_exact_stage_transport(&sources[1], &complete_final);

    let relation = build_production_claim_replay_base_low_norm_r1cs().expect("compile exact base source arms");
    let audit = relation
        .selective_compiler_audit()
        .expect("base selective compiler keeps its exact source-row audit");
    let (compiler_source_runs, compiler_mapping_sha256) = compiler_mapping_sha256(audit);
    assert_eq!(
        audit.source_arm_physical_stages()[0],
        sources[0].physical_stage_ranges()
    );
    assert_eq!(
        audit.source_arm_physical_stages()[1],
        sources[1].physical_stage_ranges()
    );

    let receipt = |label,
                   source: &SparseR1cs,
                   public_word_bindings,
                   complete_stages,
                   source_state,
                   normalized_state,
                   source_chunk,
                   normalized_chunk,
                   replay_initial_capacity,
                   replay_poseidon2,
                   phase_kind,
                   chunk_scope,
                   replay_poseidon2_calls| BaseReceipt {
        label,
        source_sha256: source_rows_sha256(source),
        source_rows: source.n,
        source_columns: source.m,
        public_columns: source.m_in,
        public_word_bindings,
        physical_stages: source
            .physical_stage_ranges()
            .iter()
            .map(receipt_stage)
            .collect(),
        complete_stages,
        row_families: named_row_ranges(source),
        column_families: named_column_ranges(source),
        source_state,
        normalized_state,
        source_chunk,
        normalized_chunk,
        replay_initial_capacity,
        replay_poseidon2,
        phase_kind,
        chunk_scope,
        replay_poseidon2_calls,
        compiler_source_runs,
        compiler_mapping_sha256: compiler_mapping_sha256.clone(),
        final_rows: relation.structure().n,
        final_columns: relation.structure().m,
    };
    let full = receipt(
        "full",
        &sources[0],
        full_public_words,
        full_complete_stages,
        full_source_state,
        full_normalized_state,
        full_source_chunk,
        full_normalized_chunk,
        full_replay_initial_capacity,
        full_replay_poseidon2,
        3,
        ReceiptRange {
            start: 0,
            stop: FULL_CHUNKS,
        },
        CHUNK_FIELDS / 4,
    );
    let final_chunk = receipt(
        "final",
        &sources[1],
        final_public_words,
        final_complete_stages,
        final_source_state,
        final_normalized_state,
        final_source_chunk,
        final_normalized_chunk,
        final_replay_initial_capacity,
        final_replay_poseidon2,
        4,
        ReceiptRange {
            start: FULL_CHUNKS,
            stop: FULL_CHUNKS + 1,
        },
        FINAL_CHUNK_FIELDS / 4,
    );

    let links = production_claim_coordinate_overlay_links();
    let mut cursor = 0;
    let mut link_rows = Vec::with_capacity(links.len());
    let mut link_counts = Vec::with_capacity(links.len());
    for contract in &links {
        let start = cursor;
        cursor += contract.fields.len();
        link_rows.push(ReceiptRange { start, stop: cursor });
        link_counts.push(contract.fields.len());
    }
    assert_eq!(links.len(), 98);
    assert_eq!(link_rows.first().map(|range| range.start), Some(0));
    assert!(link_rows
        .windows(2)
        .all(|pair| pair[0].stop == pair[1].start));
    assert_eq!(link_rows.last().map(|range| range.stop), Some(cursor));
    (full, final_chunk, link_rows, link_counts)
}

fn lean_receipt_range(range: &ReceiptRange) -> String {
    format!("{{ start := {}, stop := {} }}", range.start, range.stop)
}

fn lean_receipt_stages(stages: &[ReceiptStage]) -> String {
    format!(
        "[{}]",
        stages
            .iter()
            .map(|stage| format!(
                "{{ path := {:?}, sourceRows := {}, normalizedPrivateColumns := {} }}",
                stage.path,
                lean_receipt_range(&stage.rows),
                lean_receipt_range(&stage.columns)
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_receipt_named_ranges(ranges: &[ReceiptNamedRange]) -> String {
    format!(
        "[{}]",
        ranges
            .iter()
            .map(|range| format!(
                "{{ name := {:?}, range := {} }}",
                range.name,
                lean_receipt_range(&range.range)
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_public_word_binding(binding: &(ReceiptRange, ReceiptRange)) -> String {
    format!(
        "{{ source := {}, normalized := {} }}",
        lean_receipt_range(&binding.0),
        lean_receipt_range(&binding.1)
    )
}

fn lean_public_word_bindings(bindings: &[(ReceiptRange, ReceiptRange)]) -> String {
    format!(
        "[{}]",
        bindings
            .iter()
            .map(lean_public_word_binding)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_receipt_state(state: &ReceiptState) -> String {
    format!(
        "{{ beforeStatementFresh := {}, afterStatementFresh := {}, beforeRunningCommitments := {}, \
         afterRunningCommitments := {}, beforeRunningPublic := {}, afterRunningPublic := {} }}",
        state.before_statement_fresh,
        state.after_statement_fresh,
        state.before_running_commitments,
        state.after_running_commitments,
        state.before_running_public,
        state.after_running_public,
    )
}

fn render_base_receipt(receipt: &BaseReceipt) -> String {
    format!(
        "{{ armScope := {:?}, sourceSha256 := {:?}, sourceRows := {}, sourceColumns := {}, publicColumns := {},\n    \
         publicWordBindings := {}, normalizedPrivateStart := 641,\n    \
         physicalStages := {}, completePhysicalStages := {},\n    \
         rowFamilies := {}, columnFamilies := {},\n    \
         sourceState := {}, normalizedState := {},\n    \
         sourceChunk := {}, normalizedChunk := {},\n    \
         replayInitialCapacity := {}, replayPoseidon2 := {},\n    \
         phaseKind := {}, chunkScope := {},\n    \
         replayPoseidon2Calls := {}, compilerSourceRuns := {}, compilerMappingSha256 := {:?},\n    \
         finalRows := {}, finalColumns := {} }}",
        receipt.label,
        receipt.source_sha256,
        receipt.source_rows,
        receipt.source_columns,
        receipt.public_columns,
        lean_public_word_bindings(&receipt.public_word_bindings),
        lean_receipt_stages(&receipt.physical_stages),
        lean_receipt_stages(&receipt.complete_stages),
        lean_receipt_named_ranges(&receipt.row_families),
        lean_receipt_named_ranges(&receipt.column_families),
        lean_receipt_state(&receipt.source_state),
        lean_receipt_state(&receipt.normalized_state),
        lean_receipt_range(&receipt.source_chunk),
        lean_receipt_range(&receipt.normalized_chunk),
        lean_public_word_binding(&receipt.replay_initial_capacity),
        lean_public_word_binding(&receipt.replay_poseidon2),
        receipt.phase_kind,
        lean_receipt_range(&receipt.chunk_scope),
        receipt.replay_poseidon2_calls,
        receipt.compiler_source_runs,
        receipt.compiler_mapping_sha256,
        receipt.final_rows,
        receipt.final_columns,
    )
}

fn render_base_artifact() -> String {
    let (full, final_chunk, link_rows, link_counts) = build_base_receipts();
    let mut payload = String::new();
    writeln!(payload, "def full : RawArm :=\n  {}", render_base_receipt(&full)).unwrap();
    writeln!(
        payload,
        "\ndef finalChunk : RawArm :=\n  {}",
        render_base_receipt(&final_chunk)
    )
    .unwrap();
    writeln!(
        payload,
        "\ndef linkRowRanges : List Range :=\n  [{}]",
        link_rows
            .iter()
            .map(lean_receipt_range)
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(
        payload,
        "\ndef linkFieldCounts : List Nat :=\n  {}",
        lean_nat_list(link_counts)
    )
    .unwrap();
    let hash = sha256_hex(&payload);
    let rendered = format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPriorStateReplaySourceSchema\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayCoordinateOverlaySchema\n\n\
         /-! GENERATED FILE. DO NOT EDIT. Compact Rust-checked transport receipt\n\
         for the deferred-overlay production claim-replay base arms. -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayBase\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay.Artifact\n\n\
         abbrev Range := Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact.Range\n\
         abbrev PhysicalStage := Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact.PhysicalStage\n\
         abbrev NamedRange := Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact.NamedRange\n\n\
         structure PublicWordBinding where\n  source : Range\n  normalized : Range\n\
         deriving DecidableEq, Repr\n\n\
         structure RawArm where\n  armScope : String\n  sourceSha256 : String\n  sourceRows : Nat\n  sourceColumns : Nat\n  publicColumns : Nat\n  publicWordBindings : List PublicWordBinding\n  normalizedPrivateStart : Nat\n  physicalStages : List PhysicalStage\n  completePhysicalStages : List PhysicalStage\n  rowFamilies : List NamedRange\n  columnFamilies : List NamedRange\n  sourceState : StateBases\n  normalizedState : StateBases\n  sourceChunk : Range\n  normalizedChunk : Range\n  replayInitialCapacity : PublicWordBinding\n  replayPoseidon2 : PublicWordBinding\n  phaseKind : Nat\n  chunkScope : Range\n  replayPoseidon2Calls : Nat\n  compilerSourceRuns : Nat\n  compilerMappingSha256 : String\n  finalRows : Nat\n  finalColumns : Nat\n\
         deriving DecidableEq, Repr\n\n\
         def artifactSha256 : String := \"{hash}\"\n\
         def schemaVersion : Nat := 2\n\
         def profileId : String := \"{PROFILE_ID}\"\n\
         def sourceHashSchema : String := \"{SOURCE_HASH_SCHEMA}\"\n\
         def sourceArtifactIdentity : String := \"{BASE_SOURCE_IDENTITY}\"\n\
         def completeSourceArtifactIdentity : String := \"{COMPLETE_SOURCE_IDENTITY}\"\n\
         def finalLinkArtifactIdentity : String := \"{FINAL_LINK_IDENTITY}\"\n\n\
         {payload}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayBase\n"
    );
    assert!(
        rendered.lines().count() < 1_500,
        "generated Lean base receipt must stay below 1,500 lines"
    );
    rendered
}

fn generated_base_artifact_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingClaimReplayBase.lean",
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
fn production_claim_replay_base_lean_artifact_is_current() {
    let path = generated_base_artifact_path();
    let rendered = render_base_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected claim-replay base artifact");
        panic!(
            "claim-replay base Lean artifact drifted; inspect {}",
            expected.display()
        );
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_production_claim_replay_base_lean_artifact() {
    std::fs::write(generated_base_artifact_path(), render_base_artifact())
        .expect("write generated claim-replay base artifact");
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
    assert_eq!(full.poseidon2_permutations(), 432);
    assert_eq!(full_width.poseidon2_permutations(), 432);
    assert_eq!(final_chunk.poseidon2_permutations(), 319);
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
    let running_commitments = production_claim_running_commitment_field_map();
    let running_public = production_claim_running_public_field_map();
    assert_eq!(statement_fresh.len(), 98);
    assert_eq!(running_commitments.len(), 98);
    assert_eq!(running_public.len(), 98);
    assert_eq!(
        statement_fresh[0],
        (0..52)
            .map(|field| (field, 383 + field))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        statement_fresh[69],
        (52..449)
            .map(|field| (field, 627 + field - 52))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        statement_fresh[70],
        (449..1_473)
            .map(|field| (field, field - 449))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        statement_fresh[93],
        (24_001..25_025)
            .map(|field| (field, field - 24_001))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        statement_fresh[97],
        (28_097..28_672)
            .map(|field| (field, field - 28_097))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        running_commitments[0],
        (0..589)
            .map(|field| (field, 435 + field))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        running_commitments[61],
        (62_029..62_208)
            .map(|field| (field, field - 62_029))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        running_public[61],
        (0..845)
            .map(|field| (field, 179 + field))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        running_public[69],
        (8_013..8_640)
            .map(|field| (field, field - 8_013))
            .collect::<Vec<_>>()
    );

    let statement_fresh_active_chunks = statement_fresh
        .iter()
        .enumerate()
        .filter_map(|(chunk, fields)| (!fields.is_empty()).then_some(chunk))
        .collect::<Vec<_>>();
    assert_eq!(
        statement_fresh_active_chunks,
        std::iter::once(0).chain(69..=97).collect::<Vec<_>>()
    );
    let running_commitment_active_chunks = running_commitments
        .iter()
        .enumerate()
        .filter_map(|(chunk, fields)| (!fields.is_empty()).then_some(chunk))
        .collect::<Vec<_>>();
    assert_eq!(running_commitment_active_chunks, (0..=61).collect::<Vec<_>>());
    let running_public_active_chunks = running_public
        .iter()
        .enumerate()
        .filter_map(|(chunk, fields)| (!fields.is_empty()).then_some(chunk))
        .collect::<Vec<_>>();
    assert_eq!(running_public_active_chunks, (61..=69).collect::<Vec<_>>());
    assert_eq!(
        statement_fresh
            .iter()
            .flatten()
            .map(|&(field, _)| field)
            .collect::<Vec<_>>(),
        (0..28_672).collect::<Vec<_>>()
    );
    assert_eq!(
        running_commitments
            .iter()
            .flatten()
            .map(|&(field, _)| field)
            .collect::<Vec<_>>(),
        (0..62_208).collect::<Vec<_>>()
    );
    assert_eq!(
        running_public
            .iter()
            .flatten()
            .map(|&(field, _)| field)
            .collect::<Vec<_>>(),
        (0..8_640).collect::<Vec<_>>()
    );
}

#[test]
fn claim_replay_rejects_tampered_coordinate_commitments() {
    let mut selected = NebulaFPrimeClaimReplaySynthesis::production_full(0).expect("selected point chunk");
    assert_eq!(selected.statement_fresh_fields().len(), 52);
    assert_eq!(selected.running_commitment_fields().len(), 589);
    assert!(selected.running_public_fields().is_empty());
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
    assert!(!running_only.running_commitment_fields().is_empty());
    assert!(running_only.running_public_fields().is_empty());
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

    let mut running = NebulaFPrimeClaimReplaySynthesis::production_full(1).expect("running commitment chunk");
    let partial = running
        .partial_running_commitments_binding_column(0)
        .expect("running-commitments partial binding");
    let changed = running
        .witness_value(partial)
        .expect("running partial value")
        + F::ONE;
    running.tamper_witness_for_test(partial, changed);
    assert!(!running.is_satisfied(), "changed running-commitments binding must fail");

    let mut running_public = NebulaFPrimeClaimReplaySynthesis::production_full(61).expect("running-public chunk");
    let partial = running_public
        .partial_running_public_binding_column(0)
        .expect("running-public partial binding");
    let changed = running_public
        .witness_value(partial)
        .expect("running-public partial value")
        + F::ONE;
    running_public.tamper_witness_for_test(partial, changed);
    assert!(
        !running_public.is_satisfied(),
        "changed running-public binding must fail"
    );
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

    assert_eq!(decode_public_word(&full, 8), 95);
    assert_eq!(decode_public_word(&full, 9), 96);
    assert_eq!(decode_public_word(&final_chunk, 8), 192);
    assert_eq!(decode_public_word(&final_chunk, 9), 193);
}

#[test]
fn production_claim_replay_shape_is_exact_and_bounded() {
    let audit = production_claim_replay_shape_audit().expect("claim-replay shape audit");
    eprintln!("{audit:#?}");
    assert_eq!(audit.full.poseidon2_permutations, 432);
    assert_eq!(audit.final_chunk.poseidon2_permutations, 319);
    assert_eq!(audit.full.public_columns, 641);
    assert_eq!(audit.final_chunk.public_columns, 641);
    assert_eq!(audit.low_norm_rows, 118_213);
    assert_eq!(audit.low_norm_columns, 1_608_012);
    assert_eq!(audit.low_norm_public_columns, 648);
    assert_eq!(audit.low_norm_total_coordinates, 1_608_006);
    assert_eq!(audit.low_norm_shared_private_coordinates, 692);
    assert_eq!(audit.low_norm_full_branch_coordinates, 1_578_966);
    assert_eq!(audit.low_norm_final_branch_coordinates, 1_160_758);
    assert_eq!(audit.low_norm_full_poseidon2_coordinates, 1_523_744);
    assert_eq!(audit.low_norm_final_poseidon2_coordinates, 1_125_306);
    assert!(
        audit.low_norm_rows <= 1 << 24,
        "one claim-replay step must stay within the joint domain"
    );
    assert!(
        audit.low_norm_columns <= 1 << 24,
        "one claim-replay step must stay within the joint domain"
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
            (64, 63, 1_560, 39_206, 709_506, 709_504),
            (128, 63, 780, 40_618, 768_582, 768_544),
            (256, 63, 390, 43_334, 886_626, 886_624),
            (512, 63, 195, 54_229, 1_125_468, 1_125_446),
            (1_024, 575, 97, 118_213, 1_608_012, 1_608_006),
        ]
    );

    for (chunk_fields, audit) in candidates.into_iter().zip(audits) {
        assert_eq!(audit.chunk_fields, chunk_fields);
        assert_eq!(audit.full_chunks * chunk_fields + audit.final_chunk_fields, 99_903);
        assert_eq!(audit.full.poseidon2_permutations, chunk_fields / 4 + 176);
        assert_eq!(
            audit.final_chunk.poseidon2_permutations,
            audit.final_chunk_fields / 4 + 176
        );
        assert!(audit.low_norm_columns <= 1 << 24);
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
    assert_eq!(kinds.len(), 436);
    assert_eq!(kinds[94], 0);
    assert_eq!(kinds[95], 1);
    assert_eq!(kinds[96], 2);
    assert_eq!(kinds[155], 61);
    assert_eq!(kinds[176], 82);
    assert_eq!(kinds[177], 83);
    assert_eq!(kinds[192], 98);
    assert_eq!(kinds[193], 0);

    let links = production_claim_coordinate_overlay_links();
    assert_eq!(links.len(), 98);
    assert_eq!(links[0].overlay_kind, 1);
    assert_eq!(links[0].fields.len(), 648 + 641);
    assert_eq!(links[1].overlay_kind, 2);
    assert_eq!(links[1].fields.len(), 648 + 1_024);
    assert_eq!(links[69].overlay_kind, 70);
    assert_eq!(links[69].fields.len(), 648 + 1_024);
    assert_eq!(links[97].overlay_kind, 98);
    assert_eq!(links[97].fields.len(), 648 + 575);
}

#[test]
fn claim_coordinate_overlay_arms_are_satisfied_and_fully_constrained() {
    for kind in 0..99 {
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
        ("running public input", 62, 179),
        ("running evaluation", 70, 627),
        ("fresh commitment", 94, 243),
        ("fresh public input", 98, 35),
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
    assert_eq!(audit.kinds, 99);
    assert_eq!(audit.active_kinds, 98);
    assert_eq!(audit.active_fields, 99_520);
    assert_eq!(audit.source_rows, 12_387_808);
    assert_eq!(audit.source_columns, 12_319_814);
    assert_eq!(audit.low_norm_rows, 4_095_518);
    assert_eq!(audit.low_norm_columns, 84_834);
    assert_eq!(audit.low_norm_public_columns, 1);
    assert_eq!(audit.low_norm_total_coordinates, 84_786);
    assert!(audit.low_norm_rows <= 1 << 24);
    assert!(audit.low_norm_columns <= 1 << 24);

    let relation =
        build_production_claim_coordinate_overlay_low_norm_r1cs().expect("build coordinate overlay relation");
    assert_eq!(relation.selector_cols().len(), 99);
    assert_eq!(relation.public_input_len(), 1);
    assert_eq!(relation.structure().n, audit.low_norm_rows);
    assert_eq!(relation.structure().m, audit.low_norm_columns);
}

#[test]
fn production_claim_replay_base_sources_assignments_and_links_are_exact() {
    let (sources, shared) = production_claim_replay_base_source_arms().expect("canonical base source arms");
    assert_eq!(sources.len(), 2);
    assert_eq!(shared, 692);

    for chunk in 0..FULL_CHUNKS {
        let lowered = NebulaFPrimeClaimReplaySynthesis::production_base_full(chunk)
            .expect("production base full chunk")
            .into_lowered_for_artifact()
            .expect("lower production base full chunk");
        let (shape, assignment) = lowered.into_parts();
        assert_eq!(
            shape, sources[0],
            "full chunk {chunk} must use the canonical source matrix"
        );
        sources[0]
            .is_satisfied_by(&assignment)
            .unwrap_or_else(|error| panic!("full chunk {chunk} assignment must satisfy the canonical source: {error}"));
    }
    let final_lowered = NebulaFPrimeClaimReplaySynthesis::production_base_final()
        .into_lowered_for_artifact()
        .expect("lower production base final chunk");
    let (final_shape, final_assignment) = final_lowered.into_parts();
    assert_eq!(final_shape, sources[1]);
    sources[1]
        .is_satisfied_by(&final_assignment)
        .expect("final assignment must satisfy the canonical final source");

    let links = production_claim_coordinate_overlay_links();
    let runs = production_claim_coordinate_overlay_link_runs();
    assert_eq!(links.len(), runs.len());
    for (chunk, (contract, run)) in links.iter().zip(&runs).enumerate() {
        let base = if chunk + 1 == FULL_CHUNKS + 1 {
            NebulaFPrimeClaimReplaySynthesis::production_base_final()
        } else {
            NebulaFPrimeClaimReplaySynthesis::production_base_full(chunk).expect("linked base full chunk")
        };
        let overlay =
            NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(chunk + 1).expect("linked coordinate overlay");
        assert_eq!(contract.overlay_kind, chunk + 1);
        assert_eq!(contract.phase_kind, if chunk == FULL_CHUNKS { 4 } else { 3 });
        assert_eq!(run.overlay_kind(), contract.overlay_kind);
        assert_eq!(run.phase_kind(), contract.phase_kind);
        assert_eq!(run.chunk_index(), chunk);
        assert_eq!(contract.fields.len(), 6 * COORDINATE_OUTPUTS + run.active_field_count());

        for coordinate in 0..COORDINATE_OUTPUTS {
            let links = &contract.fields[6 * coordinate..6 * coordinate + 6];
            assert_eq!(
                links[0].phase_field,
                base.normalized_before_statement_fresh_commitment_column(coordinate)
                    .expect("base before statement-and-fresh field")
            );
            assert_eq!(
                links[0].overlay_field,
                overlay
                    .before_statement_fresh_column(coordinate)
                    .expect("overlay before statement-and-fresh field")
            );
            assert_eq!(
                links[1].phase_field,
                base.normalized_after_statement_fresh_commitment_column(coordinate)
                    .expect("base after statement-and-fresh field")
            );
            assert_eq!(
                links[1].overlay_field,
                overlay
                    .after_statement_fresh_column(coordinate)
                    .expect("overlay after statement-and-fresh field")
            );
            assert_eq!(
                links[2].phase_field,
                base.normalized_before_running_commitments_binding_column(coordinate)
                    .expect("base before running-commitments field")
            );
            assert_eq!(
                links[2].overlay_field,
                overlay
                    .before_running_commitments_column(coordinate)
                    .expect("overlay before running-commitments field")
            );
            assert_eq!(
                links[3].phase_field,
                base.normalized_after_running_commitments_binding_column(coordinate)
                    .expect("base after running-commitments field")
            );
            assert_eq!(
                links[3].overlay_field,
                overlay
                    .after_running_commitments_column(coordinate)
                    .expect("overlay after running-commitments field")
            );
            assert_eq!(
                links[4].phase_field,
                base.normalized_before_running_public_binding_column(coordinate)
                    .expect("base before running-public field")
            );
            assert_eq!(
                links[4].overlay_field,
                overlay
                    .before_running_public_column(coordinate)
                    .expect("overlay before running-public field")
            );
            assert_eq!(
                links[5].phase_field,
                base.normalized_after_running_public_binding_column(coordinate)
                    .expect("base after running-public field")
            );
            assert_eq!(
                links[5].overlay_field,
                overlay
                    .after_running_public_column(coordinate)
                    .expect("overlay after running-public field")
            );
        }
        let active_links = &contract.fields[6 * COORDINATE_OUTPUTS..];
        assert_eq!(active_links.len(), overlay.chunk_columns().len());
        for (link, &(offset, overlay_field)) in active_links.iter().zip(overlay.chunk_columns()) {
            assert_eq!(
                link.phase_field,
                base.normalized_chunk_column(offset)
                    .expect("base active chunk field")
            );
            assert_eq!(link.overlay_field, overlay_field);
        }
    }
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
        .partial_running_commitments_binding_column(0)
        .is_none());
    assert!(full_zero.partial_running_public_binding_column(0).is_none());
    assert!(full_active
        .partial_statement_fresh_commitment_column(0)
        .is_none());
    assert!(full_active
        .partial_running_commitments_binding_column(0)
        .is_none());
    assert!(full_active
        .partial_running_public_binding_column(0)
        .is_none());
    assert!(final_chunk
        .partial_statement_fresh_commitment_column(0)
        .is_none());
    assert!(final_chunk
        .partial_running_commitments_binding_column(0)
        .is_none());
    assert!(final_chunk
        .partial_running_public_binding_column(0)
        .is_none());
    assert!(full_zero.unconstrained_columns().is_empty());
    assert!(full_active.unconstrained_columns().is_empty());
    assert!(final_chunk.unconstrained_columns().is_empty());

    let audit = production_claim_replay_base_shape_audit().expect("claim replay base shape");
    eprintln!("{audit:#?}");
    assert_eq!(audit.full_rows, 259_944);
    assert_eq!(audit.full_columns, 261_603);
    assert_eq!(audit.final_rows, 192_605);
    assert_eq!(audit.final_columns, 193_803);
    assert_eq!(audit.low_norm_rows, 67_255);
    assert_eq!(audit.low_norm_columns, 1_595_106);
    assert_eq!(audit.low_norm_public_columns, 648);
    assert_eq!(audit.low_norm_total_coordinates, 1_595_104);
    assert!(audit.low_norm_rows <= 1 << 24);
    assert!(audit.low_norm_columns <= 1 << 24);
    let relation = build_production_claim_replay_base_low_norm_r1cs().expect("build claim replay base relation");
    assert_eq!(relation.selector_cols().len(), 2);
    assert_eq!(relation.structure().n, audit.low_norm_rows);
    assert_eq!(relation.structure().m, audit.low_norm_columns);
}
