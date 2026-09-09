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
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
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
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const FULL_CHUNKS: usize = 97;
const PUBLIC_BITS_PER_WORD: usize = 64;
const PI_CCS_STATEMENT_FRESH_FIELDS: usize = 28_672;
const PI_CCS_RUNNING_COMMITMENT_FIELDS: usize = 62_208;
const PI_CCS_RUNNING_PUBLIC_FIELDS: usize = 8_640;
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
