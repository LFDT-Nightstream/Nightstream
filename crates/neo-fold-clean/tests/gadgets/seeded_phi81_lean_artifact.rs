//! Compact SeededPhi81 Rust/Lean coefficient-source conformance fixture.
//!
//! The production matrix keeps these `A` rows implicit.  This fixture is
//! intentionally tiny enough to materialize: it pins the ChaCha8 word stream,
//! Phi81 rotations, input-word mapping, sparse zero elision, and output-column
//! equations against the compact Lean compiler.

use neo_ajtai::seeded_pp_chunk_seeds;
use neo_ccs::SeededPhi81LinearBlock;
use neo_math::{D, F};
use p3_field::PrimeField64;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/SeededPhi81/Generated/SeededPhi81Artifact.lean";

const WORD_STARTS: [usize; 2] = [1, 3];
const WORD_WIDTH: usize = 2;
const OUTPUT_START: usize = 10;
const HIGH_WORD_START: u128 = 100_000;
const SETUP_ROWS: usize = 2;
const SETUP_MESSAGE_COLS: usize = (1 << 15) + 1;

fn seed() -> [u8; 32] {
    core::array::from_fn(|index| index as u8)
}

fn block() -> SeededPhi81LinearBlock {
    SeededPhi81LinearBlock::new_with_word_width(0, WORD_STARTS.to_vec(), WORD_WIDTH, 1, 1, 1, vec![vec![seed()]])
        .expect("valid compact SeededPhi81 fixture")
}

fn first_words() -> Vec<u32> {
    let mut rng = ChaCha8Rng::from_seed(seed());
    (0..64).map(|_| rng.next_u32()).collect()
}

fn high_words() -> Vec<u32> {
    let mut rng = ChaCha8Rng::from_seed(seed());
    rng.set_word_pos(HIGH_WORD_START);
    (0..64).map(|_| rng.next_u32()).collect()
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

fn emit_lean() -> String {
    let block = block();
    let (setup_chunk_size, setup_chunk_seeds) = seeded_pp_chunk_seeds(seed(), SETUP_ROWS, SETUP_MESSAGE_COLS);
    let mut rows = vec![Vec::<(usize, u64)>::new(); D];
    block.for_each_term::<F, _>(|row, column, coefficient| {
        rows[row].push((column, coefficient.as_canonical_u64()));
    });

    let mut out = String::new();
    out.push_str("import Nightstream.Implementation.R1CS.Core.SeededPhi81\n");
    out.push_str("import Nightstream.Implementation.R1CS.Core.SeededAjtai\n\n");
    out.push_str("/-!\nGENERATED FILE - do not edit by hand.\n\n");
    out.push_str("Small exact Rust fixture for the compact SeededPhi81 compiler.\n");
    out.push_str("The 64 stream words come directly from `rand_chacha::ChaCha8Rng`;\n");
    out.push_str("the rows come from `SeededPhi81LinearBlock::for_each_term`.\n-/\n\n");
    out.push_str("namespace Nightstream.Implementation.R1CS.SeededPhi81Artifact\n\n");
    out.push_str(&format!(
        "def seed : List Nat := {}\n\n",
        lean_nat_list(seed().into_iter().map(usize::from))
    ));
    out.push_str(&format!(
        "def expectedWords : List Nat :=\n  {}\n\n",
        lean_nat_list(first_words().into_iter().map(|word| word as usize))
    ));
    out.push_str(&format!("def highWordStart : Nat := {HIGH_WORD_START}\n\n"));
    out.push_str(&format!(
        "def expectedHighWords : List Nat :=\n  {}\n\n",
        lean_nat_list(high_words().into_iter().map(|word| word as usize))
    ));
    out.push_str(&format!("def setupRows : Nat := {SETUP_ROWS}\n\n"));
    out.push_str(&format!("def setupMessageCols : Nat := {SETUP_MESSAGE_COLS}\n\n"));
    out.push_str(&format!("def expectedSetupChunkSize : Nat := {setup_chunk_size}\n\n"));
    out.push_str(&format!(
        "def expectedSetupSeedsByOutput : List (List (List Nat)) :=\n  {}\n\n",
        lean_seed_schedule(&setup_chunk_seeds)
    ));
    out.push_str("def block : SeededPhi81.Block :=\n");
    out.push_str("  { rowStart := 0\n");
    out.push_str(&format!("    wordStarts := {}\n", lean_nat_list(WORD_STARTS)));
    out.push_str(&format!("    wordWidth := {WORD_WIDTH}\n"));
    out.push_str("    kappa := 1\n");
    out.push_str("    messageCols := 1\n");
    out.push_str(&format!(
        "    outputColumns := {}\n",
        lean_nat_list(OUTPUT_START..OUTPUT_START + D)
    ));
    out.push_str("    superneoTransformedColumns := false\n");
    out.push_str("    schedule :=\n");
    out.push_str("      { chunkSize := 1\n");
    out.push_str("        seedsByOutput := [[seed]]\n");
    out.push_str("        rejectionFuel := 4 } }\n\n");
    out.push_str("def expectedRows : List Row :=\n  [");
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
            OUTPUT_START + row
        ));
    }
    out.push_str("]\n\nend Nightstream.Implementation.R1CS.SeededPhi81Artifact\n");
    out
}

#[test]
fn compact_fixture_has_the_expected_geometry() {
    let block = block();
    assert_eq!(block.row_end() - block.row_start(), D);
    assert_eq!(block.message_cols(), 1);
    assert_eq!(block.chunk_size(), 1);
    assert_eq!(block.word_width(), WORD_WIDTH);
    assert_eq!(block.word_starts(), WORD_STARTS);
}

#[test]
fn lean_artifact_matches_committed_file() {
    let emitted = emit_lean();
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != emitted {
        let expected_path = format!("{path}.expected");
        std::fs::write(&expected_path, emitted).expect("write expected SeededPhi81 artifact");
        panic!("generated SeededPhi81 Lean artifact drifted. Wrote {expected_path}; inspect and promote it");
    }
}
