use std::collections::BTreeSet;

use neo_fold_prototype::core::opening::{OpeningDomain, OpeningSource};
use neo_fold_prototype::core::witness_layout::commit_cols_for_full_width;
use neo_fold_prototype::rv32im::kernel::{
    build_rv32im_opening_bundle_from_accepted_artifact, rv32im_exact_stage_pp_seed, rv32im_simple_kernel_pp_seed,
    verify_rv32im_opening_bundle_from_accepted_artifact, Rv32imOpeningBundle,
};
use neo_fold_prototype::rv32im::{
    build_rv32im_accepted_proof_artifact, parity_source_cases, prove_rv32im_public_proof,
    stage1::stage1_row_word_width,
    stage2::{ram_event_word_width, register_read_word_width, register_write_word_width, twist_link_word_width},
    stage3::continuity_event_word_width,
    Rv32imProofInput,
};
use neo_math::ring::D;
use neo_math::K;
use p3_field::PrimeCharacteristicRing;

fn source_case(name: &str) -> neo_fold_prototype::rv32im::Rv32imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn proof_input(name: &str) -> Rv32imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv32imProofInput { source, max_steps }
}

fn build_test_bundle() -> Rv32imOpeningBundle {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv32im_public_proof(&input).expect("prove rv32im public proof");
    let artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    build_rv32im_opening_bundle_from_accepted_artifact(&artifact).expect("build opening bundle")
}

/// Family tag constants matching AjtaiFamilyKind::tag().
const TAG_ROOT_MAIN_LANE_COLUMNS: u32 = 0;
const TAG_STAGE1_ROWS: u32 = 1;
const TAG_STAGE2_REGISTER_READS: u32 = 2;
const TAG_STAGE2_REGISTER_WRITES: u32 = 3;
const TAG_STAGE2_RAM_EVENTS: u32 = 4;
const TAG_STAGE2_TWIST_LINKS: u32 = 5;
const TAG_STAGE3_CONTINUITY: u32 = 6;
const TAG_KERNEL_BINDINGS: u32 = 7;
const TAG_KERNEL_PREPARED_STEPS: u32 = 8;
const TAG_ROOT_MAIN_LANE_PUBLIC_STEPS: u32 = 9;
const TAG_ROOT_MAIN_LANE_COMMITTED_ROWS: u32 = 10;

const IN_SCOPE_FAMILY_TAGS: [u32; 6] = [
    TAG_STAGE1_ROWS,
    TAG_STAGE2_REGISTER_READS,
    TAG_STAGE2_REGISTER_WRITES,
    TAG_STAGE2_RAM_EVENTS,
    TAG_STAGE2_TWIST_LINKS,
    TAG_STAGE3_CONTINUITY,
];

const EXCLUDED_FAMILIES: [(u32, &str); 5] = [
    (TAG_ROOT_MAIN_LANE_COLUMNS, "RootMainLaneColumns"),
    (TAG_KERNEL_BINDINGS, "KernelBindings"),
    (TAG_KERNEL_PREPARED_STEPS, "KernelPreparedSteps"),
    (TAG_ROOT_MAIN_LANE_PUBLIC_STEPS, "RootMainLanePublicSteps"),
    (TAG_ROOT_MAIN_LANE_COMMITTED_ROWS, "RootMainLaneCommittedRows"),
];

#[test]
fn rv32im_opening_bundle_round_trip_from_accepted_artifact() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv32im_public_proof(&input).expect("prove rv32im public proof");
    let artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let bundle = build_rv32im_opening_bundle_from_accepted_artifact(&artifact).expect("build opening bundle");

    assert!(!bundle.claims.is_empty());
    assert_ne!(bundle.digest, [0; 32]);
    assert_ne!(bundle.compact_proof.unification.claimed_sum, K::ZERO);

    verify_rv32im_opening_bundle_from_accepted_artifact(&artifact, &bundle).expect("verify opening bundle");
}

#[test]
fn rv32im_opening_bundle_rejects_tampered_summary_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv32im_public_proof(&input).expect("prove rv32im public proof");
    let artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let mut bundle = build_rv32im_opening_bundle_from_accepted_artifact(&artifact).expect("build opening bundle");
    bundle.compact_proof.unification.claimed_sum += K::ONE;

    let err =
        verify_rv32im_opening_bundle_from_accepted_artifact(&artifact, &bundle).expect_err("tampered bundle must fail");
    assert!(format!("{err}").contains("opening"));
}

// ---------------------------------------------------------------------------
// Checkpoint 1: Regression tests pinning the current carried opening surface.
//
// These tests lock the exact shape of the current Rv32imOpeningBundle so that
// any future convergence work starts from a measured, verified baseline.
// ---------------------------------------------------------------------------

#[test]
fn opening_surface_contains_only_stage123_families() {
    let bundle = build_test_bundle();

    for (i, claim) in bundle.claims.iter().enumerate() {
        assert_eq!(
            claim.column_ids.len(),
            1,
            "claim {i}: expected exactly one family tag in column_ids, got {}",
            claim.column_ids.len()
        );
        let tag = claim.column_ids[0];
        assert!(
            IN_SCOPE_FAMILY_TAGS.contains(&tag),
            "claim {i}: family tag {tag} is outside the six in-scope stage families {:?}",
            IN_SCOPE_FAMILY_TAGS
        );
    }
}

#[test]
fn opening_surface_excludes_root_and_kernel_families() {
    let bundle = build_test_bundle();
    let family_tags: Vec<u32> = bundle.claims.iter().map(|c| c.column_ids[0]).collect();

    for (tag, name) in EXCLUDED_FAMILIES {
        assert!(
            !family_tags.contains(&tag),
            "{name} (tag {tag}) must not appear in the current RV32IM opening bundle"
        );
    }
}

#[test]
fn opening_surface_stage1_always_produces_four_claims() {
    let bundle = build_test_bundle();

    let stage1_count = bundle
        .claims
        .iter()
        .filter(|c| c.column_ids[0] == TAG_STAGE1_ROWS)
        .count();
    assert_eq!(
        stage1_count, 4,
        "Stage1Rows must produce exactly 4 claims (first, effect, commit, last), got {stage1_count}"
    );
}

#[test]
fn opening_surface_points_are_synthetic_one_dimensional() {
    let bundle = build_test_bundle();

    for (i, claim) in bundle.claims.iter().enumerate() {
        assert_eq!(
            claim.point.len(),
            1,
            "claim {i}: expected 1-dimensional synthetic point (logical_index anchor), got {} coordinates",
            claim.point.len()
        );
    }
}

#[test]
fn opening_surface_all_claims_are_rv32im_kernel_source() {
    let bundle = build_test_bundle();

    for (i, claim) in bundle.claims.iter().enumerate() {
        assert_eq!(
            claim.source,
            OpeningSource::Rv32imKernel,
            "claim {i}: expected Rv32imKernel source"
        );
    }
}

#[test]
fn opening_surface_stage2_families_use_mem_domain() {
    let bundle = build_test_bundle();
    let stage2_tags = [
        TAG_STAGE2_REGISTER_READS,
        TAG_STAGE2_REGISTER_WRITES,
        TAG_STAGE2_RAM_EVENTS,
        TAG_STAGE2_TWIST_LINKS,
    ];

    for (i, claim) in bundle.claims.iter().enumerate() {
        let tag = claim.column_ids[0];
        if stage2_tags.contains(&tag) {
            assert_eq!(
                claim.domain,
                OpeningDomain::Mem,
                "claim {i}: Stage2 family (tag {tag}) must use Mem domain"
            );
        } else {
            assert_eq!(
                claim.domain,
                OpeningDomain::Cpu,
                "claim {i}: non-Stage2 family (tag {tag}) must use Cpu domain"
            );
        }
    }
}

#[test]
fn opening_surface_ordinals_are_stage_labels_only() {
    let bundle = build_test_bundle();

    // Valid ordinals for stage-only bundle: 0..=13 (Stage1First..Stage3LastContinuity).
    // Tags 14-17 are KernelFirstBinding, KernelLastBinding, KernelFirstPreparedStep,
    // KernelLastPreparedStep — none should appear.
    for (i, claim) in bundle.claims.iter().enumerate() {
        assert!(
            claim.ordinal <= 13,
            "claim {i}: ordinal {} exceeds stage label range (0..=13), \
             indicating a kernel binding/prepared-step label leaked into the bundle",
            claim.ordinal
        );
    }
}

#[test]
fn opening_surface_stage1_ordinals_are_0_through_3() {
    let bundle = build_test_bundle();

    let stage1_ordinals: Vec<u64> = bundle
        .claims
        .iter()
        .filter(|c| c.column_ids[0] == TAG_STAGE1_ROWS)
        .map(|c| c.ordinal)
        .collect();

    // Stage1First=0, Stage1Effect=1, Stage1Commit=2, Stage1Last=3
    assert_eq!(stage1_ordinals, vec![0, 1, 2, 3]);
}

// ---------------------------------------------------------------------------
// Checkpoint 2: Measure real commitment contexts across in-scope families.
//
// Pins the exact word widths, Ajtai commitment column counts, and the number
// of distinct (d, m) commitment contexts for the six in-scope stage families.
// ---------------------------------------------------------------------------

/// Per-family word encoding widths. Each u64 word becomes 4 Goldilocks field elements
/// (16-bit limb decomposition with LIMB_COUNT = 64 / 16 = 4).
const LIMB_COUNT: usize = 4;

fn stage1_rows_word_width() -> usize {
    stage1_row_word_width()
}

fn stage2_register_reads_word_width() -> usize {
    register_read_word_width()
}

fn stage2_register_writes_word_width() -> usize {
    register_write_word_width()
}

fn stage2_ram_events_word_width() -> usize {
    ram_event_word_width()
}

fn stage2_twist_links_word_width() -> usize {
    twist_link_word_width()
}

fn stage3_continuity_word_width() -> usize {
    continuity_event_word_width()
}

/// Compute the Ajtai commitment column count for a given word width.
/// Pipeline: word_count * LIMB_COUNT -> logical_width; +1 -> full_width; div_ceil(D) -> m.
fn commitment_cols_for_word_width(word_width: usize) -> usize {
    let logical_width = word_width * LIMB_COUNT;
    let full_width = logical_width + 1;
    commit_cols_for_full_width(full_width)
}

#[test]
fn commitment_context_word_widths_match_frozen_v1_map() {
    // These match the payload widths documented in the convergence design doc.
    assert_eq!(stage1_rows_word_width(), 23, "Stage1Rows: 23 u64 words");
    assert_eq!(
        stage2_register_reads_word_width(),
        5,
        "Stage2RegisterReads: 5 u64 words"
    );
    assert_eq!(
        stage2_register_writes_word_width(),
        5,
        "Stage2RegisterWrites: 5 u64 words"
    );
    assert_eq!(stage2_ram_events_word_width(), 6, "Stage2RamEvents: 6 u64 words");
    assert_eq!(stage2_twist_links_word_width(), 6, "Stage2TwistLinks: 6 u64 words");
    assert_eq!(stage3_continuity_word_width(), 6, "Stage3Continuity: 6 u64 words");
}

#[test]
fn commitment_context_ajtai_column_counts() {
    // Stage1Rows: 23 words * 4 = 92 field elems + 1 = 93 full_width -> ceil(93/54) = 2
    assert_eq!(commitment_cols_for_word_width(stage1_rows_word_width()), 2);

    // Stage2RegisterReads: 5 * 4 = 20 + 1 = 21 -> ceil(21/54) = 1
    assert_eq!(commitment_cols_for_word_width(stage2_register_reads_word_width()), 1);

    // Stage2RegisterWrites: 5 * 4 = 20 + 1 = 21 -> ceil(21/54) = 1
    assert_eq!(commitment_cols_for_word_width(stage2_register_writes_word_width()), 1);

    // Stage2RamEvents: 6 * 4 = 24 + 1 = 25 -> ceil(25/54) = 1
    assert_eq!(commitment_cols_for_word_width(stage2_ram_events_word_width()), 1);

    // Stage2TwistLinks: 6 * 4 = 24 + 1 = 25 -> ceil(25/54) = 1
    assert_eq!(commitment_cols_for_word_width(stage2_twist_links_word_width()), 1);

    // Stage3Continuity: 6 * 4 = 24 + 1 = 25 -> ceil(25/54) = 1
    assert_eq!(commitment_cols_for_word_width(stage3_continuity_word_width()), 1);
}

#[test]
fn commitment_context_exactly_two_distinct_contexts() {
    let all_word_widths = [
        ("Stage1Rows", stage1_rows_word_width()),
        ("Stage2RegisterReads", stage2_register_reads_word_width()),
        ("Stage2RegisterWrites", stage2_register_writes_word_width()),
        ("Stage2RamEvents", stage2_ram_events_word_width()),
        ("Stage2TwistLinks", stage2_twist_links_word_width()),
        ("Stage3Continuity", stage3_continuity_word_width()),
    ];

    // Collect distinct (d, m) pairs.
    let contexts: BTreeSet<(usize, usize)> = all_word_widths
        .iter()
        .map(|(_, w)| (D, commitment_cols_for_word_width(*w)))
        .collect();

    assert_eq!(
        contexts.len(),
        2,
        "expected exactly 2 distinct (d, m) commitment contexts, got {}: {:?}",
        contexts.len(),
        contexts
    );

    // Context 1: (d=54, m=2) — Stage1Rows only
    assert!(contexts.contains(&(54, 2)), "missing Stage1Rows context (54, 2)");

    // Context 2: (d=54, m=1) — all 5 other families
    assert!(contexts.contains(&(54, 1)), "missing Stage2/Stage3 context (54, 1)");
}

#[test]
fn commitment_context_stage1_is_isolated() {
    // Stage1Rows has m=2, all others have m=1. This means Stage1Rows claims
    // CANNOT be accumulated with Stage2/Stage3 claims via s_lincomb in Phase 2.
    let stage1_m = commitment_cols_for_word_width(stage1_rows_word_width());
    let other_m_values: Vec<usize> = [
        stage2_register_reads_word_width(),
        stage2_register_writes_word_width(),
        stage2_ram_events_word_width(),
        stage2_twist_links_word_width(),
        stage3_continuity_word_width(),
    ]
    .iter()
    .map(|w| commitment_cols_for_word_width(*w))
    .collect();

    for m in &other_m_values {
        assert_ne!(
            *m, stage1_m,
            "Stage1Rows (m={stage1_m}) must be in a different commitment context than Stage2/Stage3 (m={m})"
        );
    }
}

#[test]
fn commitment_context_ring_dimension_is_54() {
    assert_eq!(
        D, 54,
        "ring dimension D must be 54 for all Nightstream Ajtai commitments"
    );
}

#[test]
fn commitment_context_exact_stage_and_simple_kernel_share_seed() {
    assert_eq!(
        rv32im_exact_stage_pp_seed(),
        rv32im_simple_kernel_pp_seed(),
        "exact-stage and simple-kernel Ajtai contexts must share the same PP seed in v1"
    );
}
