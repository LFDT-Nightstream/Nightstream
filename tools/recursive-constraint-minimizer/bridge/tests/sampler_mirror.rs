//! Production-scale Rust/Lean seeded-Phi81 conformance gate (campaign bar 3).
//!
//! `mirror_block` transcribes the Lean sampler semantics. The committed
//! conformance fixtures pin the Lean sampler and this mirror to the same
//! exact data per code-path class; this gate then replays every seeded block
//! of the frozen campaign profile through the mirror and requires
//! term-for-term equality with the production `for_each_term` expansion.

use neo_ccs::SeededPhi81LinearBlock;
use neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeBranch;
use neo_math::F;
use nightstream_constraint_exporter::mirror_block;
use p3_field::PrimeField64;

const REJECTION_FUEL: usize = 4;

fn expanded_rows(block: &SeededPhi81LinearBlock) -> Vec<Vec<(usize, u64)>> {
    let mut rows = vec![Vec::new(); block.row_end() - block.row_start()];
    block.for_each_term::<F, _>(|row, column, coefficient| {
        rows[row - block.row_start()].push((column, coefficient.as_canonical_u64()));
    });
    rows
}

fn assert_mirror_matches(label: &str, block: &SeededPhi81LinearBlock) -> usize {
    assert!(
        !block.has_superneo_transformed_columns(),
        "{label}: the mirror only replays original seeded columns"
    );
    let replay = mirror_block(block, REJECTION_FUEL)
        .unwrap_or_else(|| panic!("{label}: mirror replay exhausted its rejection fuel"));
    let expanded = expanded_rows(block);
    assert_eq!(replay.rows.len(), expanded.len(), "{label}: row count");
    for (index, (mirrored, production)) in replay.rows.iter().zip(&expanded).enumerate() {
        assert_eq!(
            mirrored, production,
            "{label}: row {index} diverged between the Lean-mirror and production expansions"
        );
    }
    replay.rejected_words
}

fn class_seed(tag: u8, chunk: u8) -> [u8; 32] {
    let mut seed = [tag; 32];
    seed[31] = chunk;
    seed
}

fn rejection_seed() -> [u8; 32] {
    let mut seed = [0xC3; 32];
    seed[..8].copy_from_slice(&79_842_272u64.to_le_bytes());
    seed
}

#[test]
fn mirror_matches_the_committed_conformance_fixture_classes() {
    let original = SeededPhi81LinearBlock::new_with_word_width(
        0,
        vec![1, 3],
        2,
        1,
        1,
        1,
        vec![vec![core::array::from_fn(|index| index as u8)]],
    )
    .expect("committed SeededPhi81Artifact fixture block");
    assert_eq!(assert_mirror_matches("original-width-2", &original), 0);

    let multi_chunk = SeededPhi81LinearBlock::new_with_word_width(
        0,
        vec![1, 45, 90, 140],
        41,
        1,
        4,
        3,
        vec![vec![class_seed(0xC1, 0), class_seed(0xC1, 1)]],
    )
    .expect("committed MultiChunk conformance block");
    assert_eq!(assert_mirror_matches("multi-chunk", &multi_chunk), 0);

    let two_outputs = SeededPhi81LinearBlock::new_with_word_width(
        0,
        vec![1, 45],
        41,
        2,
        2,
        1,
        vec![
            vec![class_seed(0xC2, 0), class_seed(0xC2, 1)],
            vec![class_seed(0xC2, 2), class_seed(0xC2, 3)],
        ],
    )
    .expect("committed TwoOutputs conformance block");
    assert_eq!(assert_mirror_matches("two-outputs", &two_outputs), 0);

    let rejection = SeededPhi81LinearBlock::new_with_word_width(0, vec![1], 41, 1, 1, 1, vec![vec![rejection_seed()]])
        .expect("committed Rejection conformance block");
    assert_eq!(
        assert_mirror_matches("rejection", &rejection),
        1,
        "the rejection class must consume exactly one replacement draw"
    );
}

#[test]
fn mirror_matches_every_frozen_profile_seeded_block() {
    let audit = nightstream_constraint_exporter::campaign_profile_audit().expect("discover campaign source arms");

    let mut blocks = 0usize;
    let mut rejected_words = 0usize;
    for branch in [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::Recursive] {
        let arm = audit.arm(branch);
        for (matrix_name, matrix) in [("a", &arm.a), ("b", &arm.b), ("c", &arm.c)] {
            for block in matrix.seeded_phi81_blocks() {
                if branch == NebulaFPrimeBranch::Base {
                    panic!("the frozen base arm is expected to carry no seeded blocks");
                }
                let label = format!("{branch:?}.{matrix_name}.row{}", block.row_start());
                rejected_words += assert_mirror_matches(&label, block);
                blocks += 1;
            }
        }
    }
    assert_eq!(blocks, 36, "the frozen profile must expose exactly 36 seeded blocks");
    eprintln!("frozen-profile seeded blocks: {blocks}, replacement draws: {rejected_words}");
}
