//! Production-class SeededPhi81 Rust/Lean coefficient conformance fixtures.
//!
//! The committed `SeededPhi81Artifact` pins the sampler on a width-2 single
//! chunk. The frozen campaign profile's 36 production blocks use width-41
//! words, kappa 1 and 2, multi-chunk schedules, and (rarely) the rejection
//! replacement path. Each class here is small enough to materialize and pins
//! `SeededPhi81LinearBlock::for_each_term` against the compact Lean compiler.

use neo_ccs::SeededPhi81LinearBlock;
use neo_math::{D, F};
use p3_field::PrimeField64;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/SeededPhi81/Generated/SeededPhi81ConformanceArtifact.lean";

const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;
const WORD_WIDTH: usize = 41;

/// Found by `search_rejection_seed` (2026-08-15): the first 54-word draw of
/// this seed contains one `u64 >= p`, so the accepted vector consumes one
/// replacement word from the stream tail.
const REJECTION_SEED: [u8; 32] = {
    let mut seed = [0xC3; 32];
    let counter: u64 = 79_842_272;
    let bytes = counter.to_le_bytes();
    let mut index = 0;
    while index < 8 {
        seed[index] = bytes[index];
        index += 1;
    }
    seed
};

fn class_seed(tag: u8, chunk: u8) -> [u8; 32] {
    let mut seed = [tag; 32];
    seed[31] = chunk;
    seed
}

struct ClassFixture {
    label: &'static str,
    output_start: usize,
    block: SeededPhi81LinearBlock,
}

fn class_fixtures() -> Vec<ClassFixture> {
    let multi_chunk = SeededPhi81LinearBlock::new_with_word_width(
        0,
        vec![1, 45, 90, 140],
        WORD_WIDTH,
        1,
        4,
        3,
        vec![vec![class_seed(0xC1, 0), class_seed(0xC1, 1)]],
    )
    .expect("valid width-41 multi-chunk fixture");
    let two_outputs = SeededPhi81LinearBlock::new_with_word_width(
        0,
        vec![1, 45],
        WORD_WIDTH,
        2,
        2,
        1,
        vec![
            vec![class_seed(0xC2, 0), class_seed(0xC2, 1)],
            vec![class_seed(0xC2, 2), class_seed(0xC2, 3)],
        ],
    )
    .expect("valid width-41 kappa-2 fixture");
    let rejection =
        SeededPhi81LinearBlock::new_with_word_width(0, vec![1], WORD_WIDTH, 1, 1, 1, vec![vec![REJECTION_SEED]])
            .expect("valid width-41 rejection fixture");
    vec![
        ClassFixture {
            label: "MultiChunk",
            output_start: 200,
            block: multi_chunk,
        },
        ClassFixture {
            label: "TwoOutputs",
            output_start: 300,
            block: two_outputs,
        },
        ClassFixture {
            label: "Rejection",
            output_start: 100,
            block: rejection,
        },
    ]
}

fn rejected_words_in_first_draw(seed: [u8; 32]) -> usize {
    let mut rng = ChaCha8Rng::from_seed(seed);
    let mut bytes = [0u8; D * 8];
    rng.fill_bytes(&mut bytes);
    (0..D)
        .filter(|index| {
            let start = index * 8;
            let word = u64::from_le_bytes(bytes[start..start + 8].try_into().expect("eight bytes"));
            word >= GOLDILOCKS_P
        })
        .count()
}

fn lean_nat_list(values: impl IntoIterator<Item = usize>) -> String {
    let values = values
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    format!("[{}]", values.join(", "))
}

fn lean_seed_schedule(values: &[Vec<[u8; 32]>]) -> String {
    let rows = values
        .iter()
        .map(|row| {
            let chunks = row
                .iter()
                .map(|seed| lean_nat_list(seed.iter().copied().map(usize::from)))
                .collect::<Vec<_>>();
            format!("[{}]", chunks.join(", "))
        })
        .collect::<Vec<_>>();
    format!("[{}]", rows.join(", "))
}

fn emit_class(out: &mut String, fixture: &ClassFixture) {
    let block = &fixture.block;
    let rows_len = D * block.kappa();
    let mut rows = vec![Vec::<(usize, u64)>::new(); rows_len];
    block.for_each_term::<F, _>(|row, column, coefficient| {
        rows[row].push((column, coefficient.as_canonical_u64()));
    });

    let label = fixture.label;
    out.push_str(&format!("def block{label} : SeededPhi81.Block :=\n"));
    out.push_str("  { rowStart := 0\n");
    out.push_str(&format!(
        "    wordStarts := {}\n",
        lean_nat_list(block.word_starts().iter().copied())
    ));
    out.push_str(&format!("    wordWidth := {}\n", block.word_width()));
    out.push_str(&format!("    kappa := {}\n", block.kappa()));
    out.push_str(&format!("    messageCols := {}\n", block.message_cols()));
    out.push_str(&format!(
        "    outputColumns := {}\n",
        lean_nat_list(fixture.output_start..fixture.output_start + rows_len)
    ));
    out.push_str("    superneoTransformedColumns := false\n");
    out.push_str("    schedule :=\n");
    out.push_str(&format!("      {{ chunkSize := {}\n", block.chunk_size()));
    out.push_str(&format!(
        "        seedsByOutput := {}\n",
        lean_seed_schedule(block.chunk_seeds_by_row())
    ));
    out.push_str("        rejectionFuel := 4 } }\n\n");
    out.push_str(&format!("def expectedRows{label} : List Row :=\n  ["));
    for (row, terms) in rows.iter().enumerate() {
        if row != 0 {
            out.push_str(",\n   ");
        }
        let terms = terms
            .iter()
            .map(|(column, coefficient)| format!("({column}, {coefficient})"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "\u{27e8}[{terms}], [(0, 1)], [({}, 1)]\u{27e9}",
            fixture.output_start + row
        ));
    }
    out.push_str("]\n\n");
}

fn emit_lean() -> String {
    let mut out = String::new();
    out.push_str("import Nightstream.Implementation.R1CS.Core.SeededPhi81\n\n");
    out.push_str("/-!\nGENERATED FILE - do not edit by hand.\n\n");
    out.push_str("Production-class SeededPhi81 conformance fixtures. Rows come from\n");
    out.push_str("`SeededPhi81LinearBlock::for_each_term` over `rand_chacha` streams:\n");
    out.push_str("width-41 words with an uneven two-chunk schedule (MultiChunk),\n");
    out.push_str("kappa 2 with per-output chunk seeds (TwoOutputs), and a seed whose\n");
    out.push_str("first 54-word draw rejects one word (Rejection).\n-/\n\n");
    out.push_str("namespace Nightstream.Implementation.R1CS.SeededPhi81ConformanceArtifact\n\n");
    out.push_str("set_option maxRecDepth 65536\n\n");
    for fixture in class_fixtures() {
        emit_class(&mut out, &fixture);
    }
    out.push_str("end Nightstream.Implementation.R1CS.SeededPhi81ConformanceArtifact\n");
    out
}

#[test]
fn class_fixtures_cover_the_production_paths() {
    let fixtures = class_fixtures();
    assert_eq!(fixtures[0].block.chunk_seeds_by_row()[0].len(), 2);
    assert_eq!(
        fixtures[0].block.message_cols() % fixtures[0].block.chunk_size(),
        1,
        "multi-chunk class must cross an uneven chunk boundary"
    );
    assert_eq!(fixtures[1].block.kappa(), 2);
    assert_eq!(
        rejected_words_in_first_draw(REJECTION_SEED),
        1,
        "rejection class must exercise exactly one replacement draw"
    );
    for fixture in &fixtures {
        assert_eq!(fixture.block.word_width(), WORD_WIDTH);
        assert!(!fixture.block.has_superneo_transformed_columns());
    }
}

#[test]
fn lean_artifact_matches_committed_file() {
    let emitted = emit_lean();
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != emitted {
        let expected_path = format!("{path}.expected");
        std::fs::write(&expected_path, emitted).expect("write expected SeededPhi81 conformance artifact");
        panic!(
            "generated SeededPhi81 conformance Lean artifact drifted. Wrote {expected_path}; inspect and promote it"
        );
    }
}

#[test]
#[ignore = "one-shot search for a rejection-exercising seed; run with --ignored --nocapture"]
fn search_rejection_seed() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let found = std::sync::Arc::new(AtomicBool::new(false));
    let shard_count = 16u64;
    let handles = (0..shard_count)
        .map(|shard| {
            let found = found.clone();
            std::thread::spawn(move || {
                let mut counter = shard;
                while !found.load(Ordering::Relaxed) {
                    let mut seed = [0xC3u8; 32];
                    seed[..8].copy_from_slice(&counter.to_le_bytes());
                    let rejected = rejected_words_in_first_draw(seed);
                    if rejected > 0 {
                        found.store(true, Ordering::Relaxed);
                        return Some((counter, rejected));
                    }
                    counter += shard_count;
                }
                None
            })
        })
        .collect::<Vec<_>>();
    for handle in handles {
        if let Ok(Some((counter, rejected))) = handle.join() {
            eprintln!("rejection seed counter={counter} rejected_words={rejected}");
        }
    }
}
