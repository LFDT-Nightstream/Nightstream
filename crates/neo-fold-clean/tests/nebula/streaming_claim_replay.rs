//! Constraint and size checks for the bounded-width claim-replay arms.

#[path = "../gadgets/lean_artifact_support.rs"]
#[allow(dead_code)]
mod lean_artifact_support;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Range;
use std::path::Path;

use lean_artifact_support::{lean_nat_list, sha256_hex, SCHEMA_VERSION};
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::{enforce_poseidon2_permutation, R1csBuilder, Var};
use neo_fold_clean::frontends::nebula::f_prime::{
    claim_replay_shape_audit_for_chunk_fields, production_claim_replay_shape_audit, NebulaFPrimeClaimReplayArmKind,
    NebulaFPrimeClaimReplaySynthesis,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const PROFILE_ID: &str = "nebula-f-prime-streaming-claim-replay-v1";
const FRAME_FIELDS: usize = 88_023;
const CHUNK_FIELDS: usize = 1_024;
const FINAL_CHUNK_FIELDS: usize = 983;
const FULL_CHUNKS: usize = 85;
const TRANSITION_PUBLIC_WORDS: usize = 40;
const PUBLIC_BITS_PER_WORD: usize = 64;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OwnerKind {
    Canonical,
    Poseidon2,
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

fn normalized_matrix(rows: usize, trips: &[(usize, usize, F)]) -> Vec<Vec<(usize, F)>> {
    let mut raw = vec![Vec::new(); rows];
    for &(row, column, coefficient) in trips {
        assert!(row < rows, "sparse triplet row is in range");
        raw[row].push((column, coefficient));
    }
    raw.into_iter().map(normalize_terms).collect()
}

fn normalized_rows(builder: &R1csBuilder) -> Vec<SparseRow> {
    assert!(
        builder.seeded_phi81_a_blocks().is_empty(),
        "claim replay must not hide rows in seeded Phi81 blocks"
    );
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
    assert_eq!(canonical_calls.len(), TRANSITION_PUBLIC_WORDS);

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
    assert_eq!(poseidon2_calls.len(), active_fields / 4);

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
        lean_terms(&row.c)
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
    let glue_rows = grouped_list(
        arm.glue_rows
            .iter()
            .map(|glue| format!("{{ index := {}, row := {} }}", glue.index, lean_row(&glue.row)))
            .collect(),
        1,
    );
    let owners = grouped_list(
        arm.owners
            .iter()
            .map(|owner| {
                let kind = match owner.kind {
                    OwnerKind::Canonical => ".canonical",
                    OwnerKind::Poseidon2 => ".poseidon2",
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
         canonicalCalls := {canonical_calls},\n    \
         poseidon2Calls := {poseidon2_calls},\n    \
         glueRows := {glue_rows},\n    \
         owners := {owners} }}",
        arm.row_count, arm.column_count, arm.public_column_count, arm.active_fields,
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

    assert_eq!(full_arm.glue_rows.len(), 24);
    assert_eq!(final_arm.glue_rows.len(), 77);

    let mut payload = String::new();
    writeln!(payload, "def fullArm : RawArm :=\n  {}", render_arm(&full_arm)).unwrap();
    writeln!(payload, "\ndef finalArm : RawArm :=\n  {}", render_arm(&final_arm)).unwrap();
    writeln!(
        payload,
        "\ndef rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := {SCHEMA_VERSION}, profileId := \"{PROFILE_ID}\",\n    \
            frameFields := {FRAME_FIELDS}, chunkFields := {CHUNK_FIELDS},\n    \
            finalChunkFields := {FINAL_CHUNK_FIELDS}, fullChunks := {FULL_CHUNKS},\n    \
            transitionPublicWords := {TRANSITION_PUBLIC_WORDS}, publicBitsPerWord := {PUBLIC_BITS_PER_WORD},\n    \
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
         /-! Generated file: deterministic Rust artifact for both production\n\
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
         theorem rawArtifact_valid : rawArtifact.Valid := by native_decide\n\n\
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
fn production_claim_replay_arms_are_satisfied_and_fully_constrained() {
    let full = NebulaFPrimeClaimReplaySynthesis::production_full(0).expect("first full claim chunk");
    let final_chunk = NebulaFPrimeClaimReplaySynthesis::production_final();

    assert_eq!(full.kind(), NebulaFPrimeClaimReplayArmKind::Full);
    assert_eq!(final_chunk.kind(), NebulaFPrimeClaimReplayArmKind::Final);
    assert!(full.is_satisfied(), "full arm: {:?}", full.first_unsatisfied_row());
    assert!(
        final_chunk.is_satisfied(),
        "final arm: {:?}",
        final_chunk.first_unsatisfied_row()
    );
    assert_eq!(full.poseidon2_permutations(), 256);
    assert_eq!(final_chunk.poseidon2_permutations(), 245);
    assert_eq!(full.public_columns(), 2_561);
    assert_eq!(final_chunk.public_columns(), 2_561);
    assert!(
        full.unconstrained_columns().is_empty(),
        "full arm has unused witness columns"
    );
    assert!(
        final_chunk.unconstrained_columns().is_empty(),
        "final arm has unused witness columns"
    );
}

#[test]
fn claim_replay_rejects_tampered_chunk_and_declared_output() {
    let mut full = NebulaFPrimeClaimReplaySynthesis::production_full(0).expect("first full claim chunk");
    let chunk = full.chunk_column(17).expect("chunk field column");
    let changed = full.witness_value(chunk).expect("chunk field value") + F::ONE;
    full.tamper_witness_for_test(chunk, changed);
    assert!(!full.is_satisfied(), "changed chunk field must fail");

    let mut final_chunk = NebulaFPrimeClaimReplaySynthesis::production_final();
    let output = final_chunk
        .after_runtime_column(0)
        .expect("declared output lane");
    let changed = final_chunk.witness_value(output).expect("output value") + F::ONE;
    final_chunk.tamper_witness_for_test(output, changed);
    assert!(!final_chunk.is_satisfied(), "changed final state must fail");
}

#[test]
fn production_claim_replay_shape_is_exact_and_bounded() {
    let audit = production_claim_replay_shape_audit().expect("claim-replay shape audit");
    assert_eq!(audit.full.poseidon2_permutations, 256);
    assert_eq!(audit.final_chunk.poseidon2_permutations, 245);
    assert_eq!(audit.full.public_columns, 2_561);
    assert_eq!(audit.final_chunk.public_columns, 2_561);
    assert_eq!(audit.low_norm_rows, 51_338);
    assert_eq!(audit.low_norm_columns, 536_112);
    assert_eq!(audit.low_norm_public_columns, 2_592);
    assert_eq!(audit.low_norm_total_coordinates, 536_086);
    assert_eq!(audit.low_norm_shared_private_coordinates, 1_103);
    assert_eq!(audit.low_norm_full_branch_coordinates, 507_311);
    assert_eq!(audit.low_norm_final_branch_coordinates, 484_610);
    assert_eq!(audit.low_norm_full_poseidon2_coordinates, 506_368);
    assert_eq!(audit.low_norm_final_poseidon2_coordinates, 484_610);
    assert!(
        audit.low_norm_rows < 1 << 20,
        "one claim-replay step must stay below 2^20 rows"
    );
    assert!(
        audit.low_norm_columns < 1 << 20,
        "one claim-replay step must stay below 2^20 columns"
    );
    eprintln!("{audit:#?}");
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
            (64, 23, 1_375, 10_058, 39_312, 39_286),
            (128, 87, 687, 12_792, 72_414, 72_406),
            (256, 215, 343, 18_314, 138_672, 138_646),
            (512, 471, 171, 29_304, 271_134, 271_126),
            (1_024, 983, 85, 51_338, 536_112, 536_086),
        ]
    );

    for (chunk_fields, audit) in candidates.into_iter().zip(audits) {
        assert_eq!(audit.chunk_fields, chunk_fields);
        assert_eq!(audit.full_chunks * chunk_fields + audit.final_chunk_fields, 88_023);
        assert_eq!(audit.full.poseidon2_permutations, chunk_fields / 4);
        assert_eq!(audit.final_chunk.poseidon2_permutations, audit.final_chunk_fields / 4);
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
